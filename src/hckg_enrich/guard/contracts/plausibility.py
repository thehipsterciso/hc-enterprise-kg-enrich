"""PlausibilityContract — domain bounds validation for proposed enrichments.

Rejects numeric field values that fall outside empirically grounded ranges
for each entity type.  Rule-based, no LLM call.  Fast and deterministic.

Absorbed from KARMA's AdversarialValidator.PLAUSIBILITY_BOUNDS (hc-enterprise-kg).
"""

from __future__ import annotations

import logging
from typing import Any

from hckg_enrich.guard.contract import ContractResult, ContractSeverity, QualityContract

logger = logging.getLogger(__name__)

# Empirical plausibility bounds per entity_type → field → (min, max)
# Sources: Gartner, Hackett Group, BLS, NIST, McKinsey (per hc-enterprise-kg ADRs)
PLAUSIBILITY_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "person": {
        "annual_compensation": (15_000, 15_000_000),
        "years_experience": (0, 60),
        "direct_reports_count": (0, 500),
    },
    "system": {
        "annual_cost": (0, 500_000_000),
        "availability_target": (0, 100),
        "uptime_sla": (0, 100),
    },
    "risk": {
        "probability": (0.0, 1.0),
        "impact_score": (0, 100),
        "risk_score": (0, 100),
        "cvss_score": (0.0, 10.0),
    },
    "vulnerability": {
        "cvss_score": (0.0, 10.0),
        "severity_score": (0.0, 10.0),
    },
    "department": {
        "head_count": (0, 100_000),
        "headcount": (0, 100_000),
        "budget": (0, 50_000_000_000),
        "annual_budget": (0, 50_000_000_000),
    },
    "vendor": {
        "annual_spend": (0, 10_000_000_000),
        "risk_score": (0, 100),
        "sla_uptime": (0, 100),
    },
    "control": {
        "effectiveness_score": (0, 100),
        "coverage_percentage": (0, 100),
    },
    "data_asset": {
        "size_gb": (0, 10_000_000),
        "retention_days": (0, 36_500),  # 100 years max
    },
    "integration": {
        "latency_ms": (0, 600_000),     # 10 min
        "throughput_rps": (0, 10_000_000),
        "sla_uptime": (0, 100),
    },
}


class PlausibilityContract(QualityContract):
    """Reject proposed attribute values that fall outside domain plausibility bounds.

    CONTRACT_ID: GG-PLAUS-001
    SEVERITY: WARNING (out-of-bounds values are suspicious but not always wrong)

    Checks every key in ``proposed_enrichments["proposed_attributes"]``
    against PLAUSIBILITY_BOUNDS for the entity's type.  Numeric values
    outside the range generate a WARNING violation.
    """

    CONTRACT_ID = "GG-PLAUS-001"
    DESCRIPTION = (
        "Domain bounds check: numeric field values must fall within empirically "
        "grounded plausibility ranges for the entity type."
    )
    DEFAULT_SEVERITY = ContractSeverity.WARNING

    async def evaluate(
        self,
        entity_id: str,
        proposed_enrichments: dict[str, Any],
        graph_context: dict[str, Any],
    ) -> ContractResult:
        entities: list[dict[str, Any]] = graph_context.get("entities", [])
        entity = next((e for e in entities if e.get("id") == entity_id), None)

        if entity is None:
            return ContractResult(
                contract_id=self.CONTRACT_ID,
                passed=True,
                message="Entity not found in graph context — skipping plausibility check",
                severity=self.DEFAULT_SEVERITY,
            )

        entity_type = str(entity.get("entity_type", "")).lower()
        bounds = PLAUSIBILITY_BOUNDS.get(entity_type, {})

        if not bounds:
            return ContractResult(
                contract_id=self.CONTRACT_ID,
                passed=True,
                message=f"No plausibility bounds defined for entity_type={entity_type!r}",
                severity=self.DEFAULT_SEVERITY,
            )

        proposed_attrs: dict[str, Any] = proposed_enrichments.get("proposed_attributes", {})
        violations: list[str] = []

        for field_name, value in proposed_attrs.items():
            if field_name not in bounds:
                continue
            if not isinstance(value, (int, float)):
                continue
            low, high = bounds[field_name]
            if value < low or value > high:
                violations.append(
                    f"{field_name}={value!r} outside plausible range [{low}, {high}]"
                    f" for {entity_type}"
                )

        if violations:
            msg = f"Plausibility violations ({len(violations)}): " + "; ".join(violations)
            logger.warning("GG-PLAUS-001 [%s]: %s", entity_id, msg)
            return ContractResult(
                contract_id=self.CONTRACT_ID,
                passed=False,
                message=msg,
                severity=self.DEFAULT_SEVERITY,
                details={"violations": violations, "entity_type": entity_type},
            )

        return ContractResult(
            contract_id=self.CONTRACT_ID,
            passed=True,
            message=f"All numeric fields within plausibility bounds for {entity_type}",
            severity=self.DEFAULT_SEVERITY,
        )
