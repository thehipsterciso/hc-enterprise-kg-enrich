"""RelationshipTypeSemanticsContract — validates relationship type domain/range constraints.

Hybrid rule-based + LLM. First applies a fast schema lookup to catch obvious
domain/range violations (rule-based), then calls the LLM for ambiguous cases.

The hc-enterprise-kg platform defines 52 relationship types with specific
domain/range constraints. Violations produce structurally incoherent graphs that
break analysis operations (blast radius, dependency mapping, attack path).

Examples of violations:
  - works_in(System → Department)          — wrong: works_in is Person→Department
  - depends_on(Person → System)            — wrong domain: depends_on is System→System
  - governs(Risk → Control)               — wrong: governs is Control→Risk or Policy→*
  - exploits(Vendor → Vulnerability)       — wrong: exploits is ThreatActor→Vulnerability

Severity: ERROR. Schema violations corrupt the graph structure and invalidate queries.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from hckg_enrich.guard.contract import ContractResult, ContractSeverity, QualityContract
from hckg_enrich.providers.base import LLMProvider, Message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Relationship schema — (source_entity_types, target_entity_types)
# None = "any entity type"
# ---------------------------------------------------------------------------

RELATIONSHIP_SCHEMA: dict[str, tuple[frozenset[str] | None, frozenset[str] | None]] = {
    "works_in": (
        frozenset({"person"}),
        frozenset({"department", "organizational_unit"}),
    ),
    "manages": (
        frozenset({"person"}),
        frozenset({"person", "department", "organizational_unit"}),
    ),
    "reports_to": (
        frozenset({"person"}),
        frozenset({"person"}),
    ),
    "has_role": (
        frozenset({"person"}),
        frozenset({"role"}),
    ),
    "depends_on": (
        frozenset({"system", "integration", "data_flow"}),
        frozenset({"system", "integration", "vendor"}),
    ),
    "exploits": (
        frozenset({"threat_actor", "threat"}),
        frozenset({"vulnerability", "system"}),
    ),
    "governs": (
        frozenset({"control", "policy", "regulation"}),
        frozenset({"system", "data_asset", "risk", "person", "department"}),
    ),
    "supplied_by": (
        frozenset({"system", "data_asset", "contract"}),
        frozenset({"vendor"}),
    ),
    "contracts_with": (
        frozenset({"department", "organizational_unit"}),
        frozenset({"vendor"}),
    ),
    "stores": (
        frozenset({"system"}),
        frozenset({"data_asset"}),
    ),
    "classified_as": (
        frozenset({"data_asset"}),
        None,  # any target
    ),
    "flows_to": (
        frozenset({"data_flow", "data_asset"}),
        frozenset({"system", "data_asset", "data_domain"}),
    ),
    "mitigates": (
        frozenset({"control"}),
        frozenset({"risk", "threat", "vulnerability"}),
    ),
    "subject_to": (
        frozenset({"system", "department", "data_asset", "organizational_unit"}),
        frozenset({"regulation", "policy", "control"}),
    ),
}

# Relationship types that require LLM evaluation (not in schema or schema has None entries)
AMBIGUOUS_TYPES: frozenset[str] = frozenset(
    {
        "supports", "impacts", "responsible_for", "provides_service",
        "located_at", "member_of", "enables", "realized_by",
    }
)

SYSTEM = """You are a knowledge graph schema validator for the hc-enterprise-kg platform.

The platform uses specific domain/range rules for relationship types. Given a proposed
relationship, determine if the source and target entity types are semantically valid.

Key rules:
- works_in: only Person → Department/OrgUnit
- depends_on: only System/Integration → System/Integration/Vendor
- exploits: only ThreatActor/Threat → Vulnerability/System
- governs: only Control/Policy/Regulation → System/DataAsset/Risk/Person/Department
- supplied_by: only System/DataAsset/Contract → Vendor
- reports_to: only Person → Person
- has_role: only Person → Role
- stores: only System → DataAsset

Respond ONLY with valid JSON:
{"passes": true, "reason": "brief explanation"}
or
{"passes": false, "reason": "specific schema violation"}
"""


def _check_schema(
    rel_type: str,
    source_type: str,
    target_type: str,
) -> tuple[bool | None, str]:
    """Fast schema check. Returns (passes, reason) or (None, "") if unknown/ambiguous."""
    if rel_type not in RELATIONSHIP_SCHEMA:
        return None, ""  # Defer to LLM

    allowed_sources, allowed_targets = RELATIONSHIP_SCHEMA[rel_type]

    if allowed_sources is not None and source_type not in allowed_sources:
        return (
            False,
            f"'{rel_type}' cannot have source type '{source_type}' "
            f"(allowed: {sorted(allowed_sources)})",
        )

    if allowed_targets is not None and target_type not in allowed_targets:
        return (
            False,
            f"'{rel_type}' cannot have target type '{target_type}' "
            f"(allowed: {sorted(allowed_targets)})",
        )

    return True, f"Schema validates {source_type} —[{rel_type}]→ {target_type}"


class RelationshipTypeSemanticsContract(QualityContract):
    """Validates relationship domain/range constraints.

    Rule-based for known types, LLM-based for ambiguous types.
    """

    id = "relationship-semantics-001"
    severity = ContractSeverity.ERROR
    description = "Relationship types must respect their domain and range entity type constraints"

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def evaluate(
        self,
        entity_id: str,
        proposed_enrichments: dict[str, Any],
        graph_context: str,
    ) -> ContractResult:
        proposed_rels: list[dict[str, Any]] = list(
            proposed_enrichments.get("proposed_relationships", [])
        )
        source_type = str(proposed_enrichments.get("entity_type", ""))

        if not proposed_rels:
            return ContractResult(
                contract_id=self.id,
                passed=True,
                severity=self.severity,
                message="No relationships proposed",
                entity_id=entity_id,
            )

        violations: list[str] = []
        needs_llm: list[dict[str, Any]] = []

        for rel in proposed_rels:
            rel_type = str(rel.get("relationship_type", ""))
            target_type = str(rel.get("target_type", "")).lower().replace(" ", "_")

            if not rel_type:
                continue

            passes, reason = _check_schema(rel_type, source_type, target_type)
            if passes is False:
                violations.append(reason)
            elif passes is None:
                needs_llm.append(rel)

        # LLM evaluation for ambiguous types
        if needs_llm:
            prompt = (
                f"Graph context:\n{graph_context}\n\n"
                f"Source entity type: {source_type}\n"
                f"Proposed relationships requiring semantic validation:\n"
                f"{json.dumps(needs_llm, indent=2)}\n\n"
                "Do any of these relationships violate knowledge graph schema conventions?"
            )
            raw = await self._llm.complete(
                [Message(role="user", content=prompt)], system=SYSTEM
            )
            try:
                text = raw.strip()
                if text.startswith("```"):
                    text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()
                data = json.loads(text)
                llm_passes = bool(data.get("passes", False))
                llm_reason = str(data.get("reason", ""))
                if not llm_passes:
                    violations.append(f"LLM: {llm_reason}")
            except Exception:
                violations.append(
                    "Could not parse LLM schema validation response — failing closed"
                )

        if violations:
            return ContractResult(
                contract_id=self.id,
                passed=False,
                severity=self.severity,
                message=f"{len(violations)} schema violation(s): " + "; ".join(violations),
                entity_id=entity_id,
            )

        return ContractResult(
            contract_id=self.id,
            passed=True,
            severity=self.severity,
            message=f"All {len(proposed_rels)} relationships pass schema validation",
            entity_id=entity_id,
        )
