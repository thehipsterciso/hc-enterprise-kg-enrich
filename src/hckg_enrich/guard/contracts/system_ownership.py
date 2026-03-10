"""SystemOwnershipContract — validates system ownership semantics via LLM."""
from __future__ import annotations

import json
from typing import Any

from hckg_enrich.guard.contract import ContractResult, ContractSeverity, QualityContract
from hckg_enrich.providers.base import LLMProvider, Message

SYSTEM = """You are an enterprise IT governance validator.
Given a proposed knowledge graph enrichment, determine if the system ownership
relationships are plausible for a typical enterprise organization.

Common violations:
- An ERP system (SAP, Oracle) owned by the CEO directly
- A CRM system owned by the Finance department
- Infrastructure systems owned by business units with no IT function
- HR systems owned by the Sales department

Respond ONLY with JSON: {"passes": true, "reason": "explanation"}
"""


class SystemOwnershipContract(QualityContract):
    id = "system-ownership-001"
    severity = ContractSeverity.ERROR
    description = "Systems must be owned by semantically appropriate business units"

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def evaluate(
        self,
        entity_id: str,
        proposed_enrichments: dict[str, Any],
        graph_context: str,
    ) -> ContractResult:
        prompt = (
            f"Graph context:\n{graph_context}\n\n"
            f"Proposed enrichments:\n{json.dumps(proposed_enrichments, indent=2)}\n\n"
            "Do any proposed system ownership relationships violate enterprise IT governance?"
        )
        raw = await self._llm.complete([Message(role="user", content=prompt)], system=SYSTEM)
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()
            data = json.loads(text)
            passes = bool(data.get("passes", True))
            reason = str(data.get("reason", ""))
        except Exception:
            passes = False
            reason = "Could not parse LLM response — failing closed (GG-006 security policy)"

        return ContractResult(
            contract_id=self.id,
            passed=passes,
            severity=self.severity,
            message=reason,
            entity_id=entity_id,
        )
