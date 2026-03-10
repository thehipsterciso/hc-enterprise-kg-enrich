"""EntityDeduplicationContract — detects proposed relationships to likely-duplicate entities.

Rule-based. Uses token-level fuzzy matching to detect when a proposed relationship
references an entity name that strongly resembles an existing entity. This catches
common enrichment hallucinations: the LLM proposes "Microsoft Azure" when "Azure Cloud"
already exists — two nodes that should be one.

Severity: WARNING. Deduplication requires human judgment to confirm canonical identity.
This contract flags candidates but does not block — the enrichment may still be valid.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from hckg_enrich.guard.contract import ContractResult, ContractSeverity, QualityContract

logger = logging.getLogger(__name__)

# Similarity threshold above which two names are considered likely duplicates
DEFAULT_SIMILARITY_THRESHOLD = 0.82


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity. Fast and dependency-free."""
    tokens_a = set(_normalize(a).split())
    tokens_b = set(_normalize(b).split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _find_duplicate_candidates(
    proposed_name: str,
    existing_names: list[str],
    threshold: float,
) -> list[tuple[str, float]]:
    """Return (existing_name, score) pairs exceeding threshold."""
    candidates = []
    for existing in existing_names:
        score = _token_jaccard(proposed_name, existing)
        if score >= threshold:
            candidates.append((existing, round(score, 4)))
    return sorted(candidates, key=lambda x: x[1], reverse=True)


class EntityDeduplicationContract(QualityContract):
    """Warns when proposed relationships reference entities that likely already exist.

    Rule-based fuzzy matching — no LLM call.
    """

    id = "entity-deduplication-001"
    severity = ContractSeverity.WARNING
    description = "Proposed entity references should not duplicate existing graph entities"

    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> None:
        self._threshold = similarity_threshold

    async def evaluate(
        self,
        entity_id: str,
        proposed_enrichments: dict[str, Any],
        graph_context: str,
    ) -> ContractResult:
        proposed_rels: list[dict[str, Any]] = list(
            proposed_enrichments.get("proposed_relationships", [])
        )
        existing_entities: list[dict[str, Any]] = list(
            proposed_enrichments.get("existing_entities", [])
        )

        if not proposed_rels or not existing_entities:
            return ContractResult(
                contract_id=self.id,
                passed=True,
                severity=self.severity,
                message="No proposed relationships or no existing entities to check against",
                entity_id=entity_id,
            )

        existing_names = [str(e.get("name", "")) for e in existing_entities if e.get("name")]
        duplicates: list[str] = []

        for rel in proposed_rels:
            target_name = str(rel.get("target_name", ""))
            if not target_name:
                continue
            candidates = _find_duplicate_candidates(target_name, existing_names, self._threshold)
            for existing_name, score in candidates:
                msg = f"'{target_name}' → '{existing_name}' (similarity={score:.2f})"
                duplicates.append(msg)
                logger.debug("EntityDeduplicationContract: %s", msg)

        if duplicates:
            return ContractResult(
                contract_id=self.id,
                passed=False,
                severity=self.severity,
                message=(
                    f"{len(duplicates)} likely duplicate reference(s) detected: "
                    + "; ".join(duplicates[:3])
                    + ("..." if len(duplicates) > 3 else "")
                ),
                entity_id=entity_id,
            )

        return ContractResult(
            contract_id=self.id,
            passed=True,
            severity=self.severity,
            message=f"No duplicate entity references detected (threshold={self._threshold})",
            entity_id=entity_id,
        )
