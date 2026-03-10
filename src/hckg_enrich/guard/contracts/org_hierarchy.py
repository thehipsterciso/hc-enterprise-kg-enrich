"""OrgHierarchyContract — validates org reporting semantics via LLM."""
from __future__ import annotations

import json
from typing import Any

from hckg_enrich.guard.contract import ContractResult, ContractSeverity, QualityContract
from hckg_enrich.providers.base import LLMProvider, Message

SYSTEM = """You are a strict organizational structure validator.
You will be given a proposed enrichment for a knowledge graph entity
and the current graph context. Your job is to determine if the proposed
enrichment violates basic organizational hierarchy conventions.

Common violations:
- A Finance function reporting to an HR function
- A Sales function reporting to an Engineering function
- A Legal function reporting to a Marketing function
- Any functional unit reporting to a unit with completely unrelated domain

Respond ONLY with JSON: {"passes": true, "reason": "explanation"}
"""


class OrgHierarchyContract(QualityContract):
    id = "org-hierarchy-001"
    severity = ContractSeverity.ERROR
    description = "Functional units must report to semantically appropriate parent functions"

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
            "Do any of these proposed enrichments violate org hierarchy conventions?"
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
