"""CircularDependencyContract — detects circular dependency chains in proposed enrichments.

Rule-based (no LLM). Uses DFS to detect cycles in the dependency graph formed by
proposed `depends_on` / `connects_to` relationships combined with existing relationships.

Circular dependencies in knowledge graphs represent modeling errors — they cannot exist
in directed dependency chains (e.g., System A depends_on System B depends_on System A).
This is an ERROR-severity contract because circular deps cause downstream analysis failures
(blast radius, shortest path, topological sort).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from hckg_enrich.guard.contract import ContractResult, ContractSeverity, QualityContract

logger = logging.getLogger(__name__)

# Relationship types that form directed dependency edges
DEPENDENCY_RELATIONSHIP_TYPES = frozenset(
    {
        "depends_on",
        "connects_to",
        "integrates_with",
        "feeds_data_to",
        "flows_to",
        "runs_on",
        "hosted_on",
    }
)


def _build_adjacency(
    existing_rels: list[dict[str, Any]],
    proposed_rels: list[dict[str, Any]],
    focal_entity_id: str,
    focal_entity_name: str,
) -> dict[str, set[str]]:
    """Build a directed adjacency map for dependency relationships."""
    adj: dict[str, set[str]] = defaultdict(set)

    for rel in existing_rels:
        rel_type = str(rel.get("relationship_type", rel.get("type", "")))
        if rel_type in DEPENDENCY_RELATIONSHIP_TYPES:
            src = str(rel.get("source", rel.get("source_id", "")))
            tgt = str(rel.get("target", rel.get("target_id", "")))
            if src and tgt:
                adj[src].add(tgt)

    for rel in proposed_rels:
        rel_type = str(rel.get("relationship_type", ""))
        if rel_type in DEPENDENCY_RELATIONSHIP_TYPES:
            target_name = str(rel.get("target_name", ""))
            if target_name:
                adj[focal_entity_id].add(target_name)

    return dict(adj)


def _has_cycle(adj: dict[str, set[str]]) -> tuple[bool, list[str]]:
    """DFS cycle detection. Returns (has_cycle, cycle_path)."""
    visited: set[str] = set()
    in_stack: set[str] = set()
    cycle_path: list[str] = []

    def dfs(node: str, path: list[str]) -> bool:
        visited.add(node)
        in_stack.add(node)
        path.append(node)
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                if dfs(neighbor, path):
                    return True
            elif neighbor in in_stack:
                # Found cycle — capture path
                cycle_start = path.index(neighbor) if neighbor in path else len(path) - 1
                cycle_path.extend(path[cycle_start:] + [neighbor])
                return True
        path.pop()
        in_stack.discard(node)
        return False

    for node in list(adj.keys()):
        if node not in visited:
            if dfs(node, []):
                return True, cycle_path

    return False, []


class CircularDependencyContract(QualityContract):
    """Rejects enrichments that would introduce circular dependency chains.

    Rule-based DFS — no LLM call. Fast and deterministic.
    """

    id = "circular-dependency-001"
    severity = ContractSeverity.ERROR
    description = "Proposed dependency relationships must not form cycles in the graph"

    async def evaluate(
        self,
        entity_id: str,
        proposed_enrichments: dict[str, Any],
        graph_context: str,
    ) -> ContractResult:
        existing_rels: list[dict[str, Any]] = list(
            proposed_enrichments.get("existing_relationships", [])
        )
        proposed_rels: list[dict[str, Any]] = list(
            proposed_enrichments.get("proposed_relationships", [])
        )
        entity_name = str(proposed_enrichments.get("entity_name", entity_id))

        # Filter to only dependency-type proposed relationships
        dep_proposed = [
            r for r in proposed_rels
            if str(r.get("relationship_type", "")) in DEPENDENCY_RELATIONSHIP_TYPES
        ]

        if not dep_proposed:
            return ContractResult(
                contract_id=self.id,
                passed=True,
                severity=self.severity,
                message="No dependency relationships proposed — no cycle risk",
                entity_id=entity_id,
            )

        adj = _build_adjacency(existing_rels, dep_proposed, entity_id, entity_name)
        has_cycle, cycle_path = _has_cycle(adj)

        if has_cycle:
            cycle_str = " → ".join(cycle_path)
            logger.warning(
                "CircularDependencyContract: cycle detected for entity %s: %s",
                entity_id,
                cycle_str,
            )
            return ContractResult(
                contract_id=self.id,
                passed=False,
                severity=self.severity,
                message=f"Circular dependency detected: {cycle_str}",
                entity_id=entity_id,
            )

        return ContractResult(
            contract_id=self.id,
            passed=True,
            severity=self.severity,
            message=f"No circular dependencies found in {len(dep_proposed)} proposed rels",
            entity_id=entity_id,
        )
