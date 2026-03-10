"""Guardian: runs a set of QualityContracts at a pipeline checkpoint.

Contracts execute in PARALLEL via asyncio.gather (ADR-009). This eliminates
sequential latency when multiple LLM-based contracts are registered. Each contract
is fully independent and carries its own result — parallel execution is safe.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from hckg_enrich.guard.contract import ContractResult, QualityContract
from hckg_enrich.guard.reports import QualityValidationReport

logger = logging.getLogger(__name__)


class EnrichmentGuardian:
    """Runs quality contracts in parallel and aggregates results into a report."""

    def __init__(self, contracts: list[QualityContract]) -> None:
        self._contracts = contracts

    async def validate(
        self,
        entity_id: str,
        proposed_enrichments: dict[str, Any],
        graph_context: str,
    ) -> QualityValidationReport:
        """Evaluate all contracts concurrently via asyncio.gather.

        A contract that raises an unhandled exception is treated as a FAIL
        (fail-closed policy, GG-006). The error is logged but does not abort
        other contracts.
        """
        if not self._contracts:
            return QualityValidationReport(entity_id=entity_id, results=[])

        async def _safe_evaluate(contract: QualityContract) -> ContractResult:
            try:
                return await contract.evaluate(entity_id, proposed_enrichments, graph_context)
            except Exception as exc:
                logger.error(
                    "Contract %s raised unhandled exception for entity %s: %s",
                    contract.id,
                    entity_id,
                    exc,
                )
                return ContractResult(
                    contract_id=contract.id,
                    passed=False,
                    severity=contract.severity,
                    message=f"Contract raised exception — failing closed: {exc}",
                    entity_id=entity_id,
                )

        results: list[ContractResult] = list(
            await asyncio.gather(*[_safe_evaluate(c) for c in self._contracts])
        )
        return QualityValidationReport(entity_id=entity_id, results=results)
