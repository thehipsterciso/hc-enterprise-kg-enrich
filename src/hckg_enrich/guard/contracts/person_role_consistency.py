"""PersonRoleConsistencyContract — validates person-role alignment semantics.

LLM-based. Verifies that proposed person-role assignments are consistent with
enterprise organizational conventions:

- A CFO-level person should not be assigned a junior analyst role
- A person in the Engineering department should not be assigned a purely
  Finance-domain role (Controller, Treasurer) without context
- Executive-level persons should not have operational individual-contributor roles
  as their primary role
- Role seniority must be semantically consistent with the person's described seniority

Severity: WARNING. Role mismatches may be intentional (cross-functional, interim),
but they should be flagged for review.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from hckg_enrich.guard.contract import ContractResult, ContractSeverity, QualityContract
from hckg_enrich.providers.base import LLMProvider, Message

logger = logging.getLogger(__name__)

SYSTEM = """You are an enterprise HR and organizational design validator.

Given a proposed knowledge graph enrichment for a person entity, determine if the
proposed role assignments are consistent with the person's described seniority,
department, and organizational context.

ROLE CONSISTENCY VIOLATIONS to detect:
- A named C-suite executive (CEO, CFO, CTO, CISO, etc.) assigned a junior or
  individual contributor role as their primary role
- A person described as being in Engineering assigned a pure Finance domain role
  (Controller, Treasurer, FP&A Analyst) with no explanation
- A person in Finance assigned a technical engineering role (DevOps Engineer,
  Software Developer) with no context
- Extreme role-level mismatches (VP-level person assigned an intern-level role)
- A person assigned roles in 3+ unrelated functional domains simultaneously

Note: interim assignments, dual-hatted executives, and cross-functional roles
are legitimate. Flag only clear semantic inconsistencies.

Respond ONLY with valid JSON:
{"passes": true, "reason": "brief explanation"}
or
{"passes": false, "reason": "specific inconsistency found"}
"""


class PersonRoleConsistencyContract(QualityContract):
    """Validates that person-role assignments follow organizational seniority conventions."""

    id = "person-role-consistency-001"
    severity = ContractSeverity.WARNING
    description = (
        "Person role assignments must be semantically consistent with seniority and department"
    )

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def evaluate(
        self,
        entity_id: str,
        proposed_enrichments: dict[str, Any],
        graph_context: str,
    ) -> ContractResult:
        entity_type = str(proposed_enrichments.get("entity_type", ""))

        # Only evaluate person entities — skip others
        if entity_type != "person":
            return ContractResult(
                contract_id=self.id,
                passed=True,
                severity=self.severity,
                message=f"Not a person entity (type={entity_type}) — skipped",
                entity_id=entity_id,
            )

        prompt = (
            f"Graph context:\n{graph_context}\n\n"
            f"Proposed enrichments:\n{json.dumps(proposed_enrichments, indent=2)}\n\n"
            "Are the proposed role assignments semantically consistent with this person's "
            "organizational context and seniority level?"
        )

        raw = await self._llm.complete([Message(role="user", content=prompt)], system=SYSTEM)
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()
            data = json.loads(text)
            passes = bool(data.get("passes", False))
            reason = str(data.get("reason", ""))
        except Exception:
            passes = False
            reason = "Could not parse LLM response — failing closed (GG-006 security policy)"

        if not passes:
            logger.info(
                "PersonRoleConsistencyContract WARNING entity=%s: %s", entity_id, reason
            )

        return ContractResult(
            contract_id=self.id,
            passed=passes,
            severity=self.severity,
            message=reason,
            entity_id=entity_id,
        )
