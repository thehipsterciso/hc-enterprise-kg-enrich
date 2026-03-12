"""ConfidenceAgent — assigns T1–T4 confidence tiers to enrichment proposals.

The ConfidenceAgent sits between the ReasoningAgent and CoherenceAgent.
It evaluates the quality of evidence supporting a proposal and assigns:

  T1 (Verified Fact)       0.94–1.00  — grounded in multiple high-quality sources
  T2 (Strong Inference)    0.80–0.93  — multiple corroborating sources or single authoritative
  T3 (Reasoned Inference)  0.65–0.79  — limited/indirect evidence, graph-structure reasoning
  T4 (Working Hypothesis)  0.50–0.64  — LLM judgment with minimal external grounding

This is a deterministic, rule-based agent (no LLM call) to keep it fast, testable,
and explainable. The scoring rubric is transparent and auditable.

Evidence signals evaluated:
  - Number of search sources cited
  - Source relevance scores
  - Search query specificity (entity name referenced)
  - Proposal specificity (non-generic field values)
  - Graph context corroboration (relationships already exist vs. net-new)
  - Reasoning text quality signals (hedging language → lower tier)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from hckg_enrich.agents.base import AbstractEnrichmentAgent, AgentMessage, AgentRole
from hckg_enrich.provenance.record import ConfidenceTier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hedging language patterns — presence lowers confidence
# ---------------------------------------------------------------------------

HEDGING_PATTERNS = re.compile(
    r"\b(might|may|could|possibly|probably|perhaps|unclear|uncertain|"
    r"appears to|seems to|likely|assume|speculate|hypothesize|infer)\b",
    re.IGNORECASE,
)

# Words that signal strong evidential grounding
GROUNDED_PATTERNS = re.compile(
    r"\b(confirms|confirmed|according to|per|documented|recorded|stated|"
    r"disclosed|reported|regulatory filing|annual report|10-K|press release)\b",
    re.IGNORECASE,
)

# Overly generic attribute values that add little value
GENERIC_VALUES = frozenset(
    {
        "n/a", "unknown", "tbd", "to be determined", "varies", "see documentation",
        "standard", "default", "typical", "common", "general", "various",
        "", "null", "none",
    }
)


# ---------------------------------------------------------------------------
# Evidence scoring helpers
# ---------------------------------------------------------------------------


@dataclass
class EvidenceSignals:
    source_count: int = 0
    avg_relevance: float = 0.0
    entity_name_in_queries: bool = False
    grounding_phrases: int = 0
    hedging_phrases: int = 0
    specific_attributes: int = 0
    generic_attributes: int = 0
    graph_corroborated_relationships: int = 0
    net_new_relationships: int = 0


def _extract_signals(
    payload: dict[str, Any],
    entity_name: str,
) -> EvidenceSignals:
    signals = EvidenceSignals()

    # Search source quality
    search_sources: list[dict[str, Any]] = list(payload.get("search_sources", []))
    signals.source_count = len(search_sources)
    if search_sources:
        scores = [float(s.get("score", 0.5)) for s in search_sources]
        signals.avg_relevance = sum(scores) / len(scores)

    # Query specificity
    search_queries: list[str] = list(payload.get("search_queries", []))
    entity_lower = entity_name.lower()
    signals.entity_name_in_queries = any(
        entity_lower in q.lower() for q in search_queries
    )

    # Reasoning text signals
    reasoning: str = str(payload.get("proposal", {}).get("reasoning", ""))
    signals.hedging_phrases = len(HEDGING_PATTERNS.findall(reasoning))
    signals.grounding_phrases = len(GROUNDED_PATTERNS.findall(reasoning))

    # Attribute specificity
    proposed_attrs: dict[str, Any] = payload.get("proposal", {}).get("proposed_attributes", {})
    for val in proposed_attrs.values():
        str_val = str(val).strip().lower()
        if str_val in GENERIC_VALUES or len(str_val) < 3:
            signals.generic_attributes += 1
        else:
            signals.specific_attributes += 1

    # Relationship corroboration — check if proposed relationships match existing context
    proposed_rels: list[dict[str, Any]] = payload.get("proposal", {}).get(
        "proposed_relationships", []
    )
    existing_rels: list[dict[str, Any]] = list(payload.get("existing_relationships", []))
    existing_targets = {str(r.get("target", "")) for r in existing_rels}
    for rel in proposed_rels:
        target = str(rel.get("target_name", ""))
        if target and target in existing_targets:
            signals.graph_corroborated_relationships += 1
        else:
            signals.net_new_relationships += 1

    return signals


def _score_to_tier(score: float) -> ConfidenceTier:
    if score >= 0.94:
        return ConfidenceTier.T1
    elif score >= 0.80:
        return ConfidenceTier.T2
    elif score >= 0.65:
        return ConfidenceTier.T3
    return ConfidenceTier.T4


def _compute_confidence(signals: EvidenceSignals) -> tuple[float, str]:
    """Compute a 0.0–1.0 confidence score from evidence signals.

    Returns (score, rationale).
    """
    score = 0.50  # Baseline: T4 (LLM judgment alone)
    rationale_parts: list[str] = ["baseline=0.50 (LLM-only)"]

    # Source count contribution (max +0.20)
    if signals.source_count >= 3:
        score += 0.20
        rationale_parts.append(f"+0.20 ({signals.source_count} sources)")
    elif signals.source_count == 2:
        score += 0.12
        rationale_parts.append("+0.12 (2 sources)")
    elif signals.source_count == 1:
        score += 0.06
        rationale_parts.append("+0.06 (1 source)")

    # Source quality contribution (max +0.08)
    if signals.avg_relevance >= 0.8:
        score += 0.08
        rationale_parts.append(f"+0.08 (avg relevance={signals.avg_relevance:.2f})")
    elif signals.avg_relevance >= 0.6:
        score += 0.04
        rationale_parts.append(f"+0.04 (avg relevance={signals.avg_relevance:.2f})")

    # Entity name in search queries (max +0.05)
    if signals.entity_name_in_queries:
        score += 0.05
        rationale_parts.append("+0.05 (entity-specific queries)")

    # Grounding language (max +0.06)
    if signals.grounding_phrases >= 2:
        score += 0.06
        rationale_parts.append(f"+0.06 ({signals.grounding_phrases} grounding phrases)")
    elif signals.grounding_phrases == 1:
        score += 0.03
        rationale_parts.append("+0.03 (1 grounding phrase)")

    # Specific attributes (max +0.05)
    if signals.specific_attributes >= 3:
        score += 0.05
        rationale_parts.append(f"+0.05 ({signals.specific_attributes} specific attributes)")
    elif signals.specific_attributes >= 1:
        score += 0.02
        rationale_parts.append(f"+0.02 ({signals.specific_attributes} specific attributes)")

    # Hedging penalty (max -0.10)
    if signals.hedging_phrases >= 3:
        score -= 0.10
        rationale_parts.append(f"-0.10 ({signals.hedging_phrases} hedging phrases)")
    elif signals.hedging_phrases >= 1:
        score -= 0.04
        rationale_parts.append(f"-0.04 ({signals.hedging_phrases} hedging phrases)")

    # Generic attribute penalty
    if signals.generic_attributes >= 2:
        score -= 0.05
        rationale_parts.append(f"-0.05 ({signals.generic_attributes} generic values)")

    # Graph corroboration bonus
    if signals.graph_corroborated_relationships >= 1:
        score += 0.03
        rationale_parts.append(
            f"+0.03 ({signals.graph_corroborated_relationships} graph-corroborated rels)"
        )

    score = round(max(0.50, min(1.0, score)), 4)
    return score, " | ".join(rationale_parts)


# ---------------------------------------------------------------------------
# ConfidenceAgent
# ---------------------------------------------------------------------------


class ConfidenceAgent(AbstractEnrichmentAgent):
    """Evaluates evidence quality and assigns a T1–T4 confidence tier.

    Reads from payload:
        proposal:               dict — EnrichmentProposal output
        search_sources:         list[dict] — sources from SearchAgent
        search_queries:         list[str]  — queries used
        existing_relationships: list[dict] — graph edges for corroboration
        entity_name:            str

    Writes to payload:
        confidence_tier:        str  — "T1"|"T2"|"T3"|"T4"
        confidence_score:       float
        confidence_rationale:   str
        evidence_signals:       dict — raw signal breakdown
    """

    role = AgentRole.REASONING  # Runs after ReasoningAgent, before CoherenceAgent

    async def run(self, message: AgentMessage) -> AgentMessage:
        payload = dict(message.payload)
        entity_name = str(payload.get("entity_name", ""))
        proposal = payload.get("proposal", {})

        # If there's no proposal (search or reasoning failed), assign T4
        if not proposal:
            payload["confidence_tier"] = ConfidenceTier.T4.value
            payload["confidence_score"] = 0.50
            payload["confidence_rationale"] = "No proposal generated — defaulting to T4"
            payload["evidence_signals"] = {}
            return AgentMessage(
                sender=self.role,
                recipient=AgentRole.COHERENCE,
                correlation_id=message.correlation_id,
                payload=payload,
            )

        signals = _extract_signals(payload, entity_name)
        score, rationale = _compute_confidence(signals)
        tier = _score_to_tier(score)

        logger.debug(
            "ConfidenceAgent: entity=%s tier=%s score=%.3f | %s",
            entity_name,
            tier.value,
            score,
            rationale,
        )

        payload["confidence_tier"] = tier.value
        payload["confidence_score"] = score
        payload["confidence_rationale"] = rationale
        payload["evidence_signals"] = {
            "source_count": signals.source_count,
            "avg_relevance": signals.avg_relevance,
            "entity_name_in_queries": signals.entity_name_in_queries,
            "grounding_phrases": signals.grounding_phrases,
            "hedging_phrases": signals.hedging_phrases,
            "specific_attributes": signals.specific_attributes,
            "generic_attributes": signals.generic_attributes,
            "graph_corroborated_relationships": signals.graph_corroborated_relationships,
            "net_new_relationships": signals.net_new_relationships,
        }

        return AgentMessage(
            sender=self.role,
            recipient=AgentRole.COHERENCE,
            correlation_id=message.correlation_id,
            payload=payload,
        )
