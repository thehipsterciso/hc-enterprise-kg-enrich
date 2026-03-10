"""PrioritizationAgent — ranks entities by enrichment value before pipeline runs.

The prioritization agent sits before the main 5-agent pipeline and produces
an ordered entity list so that highest-value enrichments run first. This matters
for rate-limited runs (limit=N) and for partial failures: if the run aborts,
the most valuable entities have already been enriched.

Scoring model (additive, max ~1.0):
  - Entity type priority weight  (0.00–0.30)  — critical types score higher
  - Missing field ratio          (0.00–0.25)  — more gaps = higher value
  - Graph connectivity           (0.00–0.25)  — central entities propagate enrichment
  - Staleness                    (0.00–0.20)  — never-enriched > recently-enriched

The agent is synchronous (no LLM call) to keep it fast and avoid burning tokens
on prioritization. It uses only structural graph signals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hckg_enrich.agents.base import AgentRole, AgentMessage, AbstractEnrichmentAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity type priority weights — aligned with hc-enterprise-kg L00-L11 layers
# ---------------------------------------------------------------------------

ENTITY_TYPE_WEIGHTS: dict[str, float] = {
    # Critical infrastructure / highest enrichment ROI
    "system": 0.30,
    "data_asset": 0.28,
    "integration": 0.27,
    "vendor": 0.25,
    "contract": 0.24,
    # Governance / compliance
    "control": 0.22,
    "regulation": 0.20,
    "risk": 0.20,
    "policy": 0.18,
    # People / org structure
    "person": 0.20,
    "role": 0.18,
    "department": 0.18,
    "organizational_unit": 0.17,
    # Secondary domains
    "threat": 0.15,
    "vulnerability": 0.15,
    "incident": 0.15,
    "data_flow": 0.14,
    "data_domain": 0.13,
    "business_capability": 0.12,
    "initiative": 0.10,
    # Default for unlisted types
    "_default": 0.10,
}

# Core attributes that should be present on any entity
EXPECTED_CORE_FIELDS: frozenset[str] = frozenset(
    {"name", "description", "entity_type"}
)

# Fields that carry high enrichment value when populated
HIGH_VALUE_FIELDS: frozenset[str] = frozenset(
    {
        "description", "owner", "responsible_team", "criticality",
        "data_classification", "risk_tier", "tech_stack", "vendor_name",
        "budget", "headcount", "framework", "status",
    }
)


# ---------------------------------------------------------------------------
# Score components
# ---------------------------------------------------------------------------


@dataclass
class EntityPriorityScore:
    entity_id: str
    entity_name: str
    entity_type: str
    type_weight: float
    missing_field_score: float
    connectivity_score: float
    staleness_score: float
    total_score: float
    reasons: list[str]


def _type_weight(entity_type: str) -> float:
    return ENTITY_TYPE_WEIGHTS.get(entity_type, ENTITY_TYPE_WEIGHTS["_default"])


def _missing_field_score(entity: dict[str, Any]) -> tuple[float, list[str]]:
    """Score based on how many high-value fields are empty."""
    reasons: list[str] = []
    populated = {k for k, v in entity.items() if v not in (None, "", [], {})}
    missing = HIGH_VALUE_FIELDS - populated
    ratio = len(missing) / len(HIGH_VALUE_FIELDS) if HIGH_VALUE_FIELDS else 0.0
    if ratio > 0.6:
        reasons.append(f"missing {len(missing)}/{len(HIGH_VALUE_FIELDS)} key fields")
    return round(ratio * 0.25, 4), reasons


def _connectivity_score(
    entity_id: str,
    relationships: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    """Score based on how many relationships touch this entity.

    More connected entities are more impactful — enriching them propagates
    context to many neighbors.
    """
    degree = sum(
        1
        for r in relationships
        if str(r.get("source")) == entity_id or str(r.get("target")) == entity_id
    )
    # Sigmoid-style: cap at 20 relationships = full score
    capped = min(degree, 20)
    score = round((capped / 20) * 0.25, 4)
    reasons: list[str] = []
    if degree >= 10:
        reasons.append(f"high connectivity ({degree} relationships)")
    return score, reasons


def _staleness_score(entity: dict[str, Any]) -> tuple[float, list[str]]:
    """Score based on enrichment history in provenance field."""
    provenance = entity.get("provenance", {})
    reasons: list[str] = []
    if not provenance:
        # Never enriched — maximum staleness value
        reasons.append("never enriched")
        return 0.20, reasons
    enriched_at = provenance.get("enriched_at")
    if not enriched_at:
        reasons.append("no enrichment timestamp")
        return 0.15, reasons
    # Partial credit for entities enriched previously
    return 0.05, reasons


# ---------------------------------------------------------------------------
# PrioritizationAgent
# ---------------------------------------------------------------------------


class PrioritizationAgent(AbstractEnrichmentAgent):
    """Ranks entities by enrichment value and returns an ordered list.

    This agent does NOT sit in the main per-entity pipeline chain.
    It is called once per run by the EnrichmentController before the
    main pipeline loop begins.

    Input payload:
        entities: list[dict]         — full entity list from the graph
        relationships: list[dict]    — full relationship list for connectivity
        entity_type_filter: str|None — optional type filter
        limit: int|None              — cap on output list

    Output payload:
        prioritized_entities: list[dict]   — entities ordered high→low priority
        priority_scores: list[dict]        — score breakdown per entity
        total_candidates: int
    """

    role = AgentRole.CONTEXT  # Logical pre-pipeline role (not a chain member)

    async def run(self, message: AgentMessage) -> AgentMessage:
        payload = dict(message.payload)
        entities: list[dict[str, Any]] = list(payload.get("entities", []))
        relationships: list[dict[str, Any]] = list(payload.get("relationships", []))
        entity_type_filter: str | None = payload.get("entity_type_filter")
        limit: int | None = payload.get("limit")

        if entity_type_filter:
            entities = [e for e in entities if e.get("entity_type") == entity_type_filter]

        scores: list[EntityPriorityScore] = []
        for entity in entities:
            eid = str(entity.get("id", ""))
            etype = str(entity.get("entity_type", "unknown"))
            ename = str(entity.get("name", eid))

            tw = _type_weight(etype)
            mfs, mf_reasons = _missing_field_score(entity)
            cs, c_reasons = _connectivity_score(eid, relationships)
            ss, s_reasons = _staleness_score(entity)
            total = round(tw + mfs + cs + ss, 4)

            all_reasons = [f"type={etype} (w={tw})"] + mf_reasons + c_reasons + s_reasons

            scores.append(
                EntityPriorityScore(
                    entity_id=eid,
                    entity_name=ename,
                    entity_type=etype,
                    type_weight=tw,
                    missing_field_score=mfs,
                    connectivity_score=cs,
                    staleness_score=ss,
                    total_score=total,
                    reasons=all_reasons,
                )
            )

        scores.sort(key=lambda s: s.total_score, reverse=True)
        if limit:
            scores = scores[:limit]

        entity_map = {str(e.get("id", "")): e for e in entities}
        prioritized = [entity_map[s.entity_id] for s in scores if s.entity_id in entity_map]

        logger.info(
            "PrioritizationAgent: %d entities ranked, top entity=%s (score=%.3f)",
            len(scores),
            scores[0].entity_name if scores else "none",
            scores[0].total_score if scores else 0.0,
        )

        payload["prioritized_entities"] = prioritized
        payload["priority_scores"] = [
            {
                "entity_id": s.entity_id,
                "entity_name": s.entity_name,
                "entity_type": s.entity_type,
                "total_score": s.total_score,
                "breakdown": {
                    "type_weight": s.type_weight,
                    "missing_field_score": s.missing_field_score,
                    "connectivity_score": s.connectivity_score,
                    "staleness_score": s.staleness_score,
                },
                "reasons": s.reasons,
            }
            for s in scores
        ]
        payload["total_candidates"] = len(scores)

        return AgentMessage(
            sender=AgentRole.CONTEXT,
            recipient=AgentRole.CONTEXT,
            correlation_id=message.correlation_id,
            payload=payload,
        )
