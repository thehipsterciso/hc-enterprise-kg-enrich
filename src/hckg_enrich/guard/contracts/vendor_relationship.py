"""VendorRelationshipContract — validates vendor relationship semantics via LLM."""
from __future__ import annotations

import json
from typing import Any

from hckg_enrich.guard.contract import ContractResult, ContractSeverity, QualityContract
from hckg_enrich.providers.base import LLMProvider, Message

SYSTEM = """You are an enterprise vendor governance validator.
Given a proposed knowledge graph enrichment, determine if vendor relationships
are plausible for a typical enterprise organization.

Common violations:
- A cloud infrastructure vendor managed by the Marketing department
- A payroll vendor managed by the IT department with no HR involvement
- A security vendor managed by the Finance department

Respond ONLY with JSON: {"passes": true, "reason": "explanation"}
"""


class VendorRelationshipContract(QualityContract):
    id = "vendor-relationship-001"
    severity = ContractSeverity.WARNING
    description = "Vendor relationships should follow procurement and domain governance norms"

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
            "Do any proposed vendor relationships violate enterprise governance conventions?"
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
            passes = True
            reason = "Could not parse LLM response — defaulting to pass"

        return ContractResult(
            contract_id=self.id,
            passed=passes,
            severity=self.severity,
            message=reason,
            entity_id=entity_id,
        )
