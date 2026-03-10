"""DataAssetOwnershipContract — validates that data assets have appropriate stewards.

LLM-based. Verifies that proposed data asset ownership relationships follow
enterprise data governance conventions:

- PII/sensitive data assets must be owned by a department with data governance mandate
  (Legal, Compliance, IT, Data, Privacy — NOT Sales, Marketing, Engineering directly)
- Financial data assets must have Finance or Accounting department involvement
- Customer data must involve Customer Success, CRM, or a business unit with direct
  customer accountability — not backend infrastructure teams

This contract is ERROR severity. Incorrectly attributed data ownership is a
compliance and governance risk in healthcare, financial services, and regulated industries.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from hckg_enrich.guard.contract import ContractResult, ContractSeverity, QualityContract
from hckg_enrich.providers.base import LLMProvider, Message

logger = logging.getLogger(__name__)

SYSTEM = """You are an enterprise data governance validator specializing in data asset stewardship.

Given a proposed knowledge graph enrichment for a data asset entity, determine if the
proposed ownership and stewardship relationships are appropriate for an enterprise.

DATA GOVERNANCE VIOLATIONS to detect:
- PII or sensitive data assets (customer records, HR data, health data) owned directly
  by Engineering or DevOps without a Data/Privacy governance layer
- Financial data assets (revenue, P&L, payroll) owned by Marketing or Sales departments
- Customer data assets owned by backend infrastructure teams with no business accountability
- Regulated data (HIPAA, PCI-DSS, GDPR) without any compliance or legal governance link
- Data assets with no owner proposed when the entity type implies one is required

Respond ONLY with valid JSON:
{"passes": true, "reason": "brief explanation"}
or
{"passes": false, "reason": "specific violation found"}
"""


class DataAssetOwnershipContract(QualityContract):
    """Validates data asset ownership and stewardship governance semantics."""

    id = "data-asset-ownership-001"
    severity = ContractSeverity.ERROR
    description = "Data assets must be owned by semantically appropriate stewards"

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def evaluate(
        self,
        entity_id: str,
        proposed_enrichments: dict[str, Any],
        graph_context: str,
    ) -> ContractResult:
        entity_type = str(proposed_enrichments.get("entity_type", ""))

        # Only evaluate data asset entities — skip others for efficiency
        if entity_type not in ("data_asset", "data_domain", "data_flow"):
            return ContractResult(
                contract_id=self.id,
                passed=True,
                severity=self.severity,
                message=f"Not a data asset entity (type={entity_type}) — skipped",
                entity_id=entity_id,
            )

        prompt = (
            f"Graph context:\n{graph_context}\n\n"
            f"Entity type: {entity_type}\n"
            f"Proposed enrichments:\n{json.dumps(proposed_enrichments, indent=2)}\n\n"
            "Do any proposed ownership or stewardship relationships violate enterprise "
            "data governance conventions?"
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
            logger.warning(
                "DataAssetOwnershipContract BLOCKED entity=%s: %s", entity_id, reason
            )

        return ContractResult(
            contract_id=self.id,
            passed=passes,
            severity=self.severity,
            message=reason,
            entity_id=entity_id,
        )
