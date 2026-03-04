"""Guardian: runs a set of QualityContracts at a pipeline checkpoint."""
from __future__ import annotations

from typing import Any

from hckg_enrich.guard.contract import ContractResult, QualityContract
from hckg_enrich.guard.reports import QualityValidationReport


class EnrichmentGuardian:
    """Runs quality contracts and aggregates results into a report."""

    def __init__(self, contracts: list[QualityContract]) -> None:
        self._contracts = contracts

    async def validate(
        self,
        entity_id: str,
        proposed_enrichments: dict[str, Any],
        graph_context: str,
    ) -> QualityValidationReport:
        results: list[ContractResult] = []
        for contract in self._contracts:
            result = await contract.evaluate(entity_id, proposed_enrichments, graph_context)
            results.append(result)
        return QualityValidationReport(entity_id=entity_id, results=results)
