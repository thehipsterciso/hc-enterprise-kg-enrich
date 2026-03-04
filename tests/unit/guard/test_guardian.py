"""Tests for EnrichmentGuardian and QualityValidationReport."""
from __future__ import annotations

from typing import Any

import pytest

from hckg_enrich.guard.contract import ContractResult, ContractSeverity, QualityContract
from hckg_enrich.guard.guardian import EnrichmentGuardian


class AlwaysPassContract(QualityContract):
    id = "always-pass"
    severity = ContractSeverity.ERROR

    async def evaluate(
        self, entity_id: str, proposed_enrichments: dict[str, Any], graph_context: str
    ) -> ContractResult:
        return ContractResult(self.id, passed=True, severity=self.severity, entity_id=entity_id)


class AlwaysFailContract(QualityContract):
    id = "always-fail"
    severity = ContractSeverity.ERROR

    async def evaluate(
        self, entity_id: str, proposed_enrichments: dict[str, Any], graph_context: str
    ) -> ContractResult:
        return ContractResult(
            self.id, passed=False, severity=self.severity,
            message="Always fails", entity_id=entity_id
        )


class AlwaysWarnContract(QualityContract):
    id = "always-warn"
    severity = ContractSeverity.WARNING

    async def evaluate(
        self, entity_id: str, proposed_enrichments: dict[str, Any], graph_context: str
    ) -> ContractResult:
        return ContractResult(
            self.id, passed=False, severity=self.severity,
            message="Warning", entity_id=entity_id
        )


@pytest.mark.asyncio
async def test_guardian_passes_when_all_pass() -> None:
    guardian = EnrichmentGuardian([AlwaysPassContract()])
    report = await guardian.validate("e1", {}, "ctx")
    assert report.passed
    assert report.blocking_failures == []
    assert report.warnings == []


@pytest.mark.asyncio
async def test_guardian_fails_on_error_contract() -> None:
    guardian = EnrichmentGuardian([AlwaysPassContract(), AlwaysFailContract()])
    report = await guardian.validate("e1", {}, "ctx")
    assert not report.passed
    assert len(report.blocking_failures) == 1
    assert report.blocking_failures[0].contract_id == "always-fail"


@pytest.mark.asyncio
async def test_guardian_passes_with_only_warnings() -> None:
    guardian = EnrichmentGuardian([AlwaysWarnContract()])
    report = await guardian.validate("e1", {}, "ctx")
    assert report.passed  # warnings don't block
    assert len(report.warnings) == 1


@pytest.mark.asyncio
async def test_guardian_summary_contains_entity_id() -> None:
    guardian = EnrichmentGuardian([AlwaysPassContract()])
    report = await guardian.validate("entity-xyz", {}, "ctx")
    assert "entity-xyz" in report.summary()


@pytest.mark.asyncio
async def test_guardian_runs_all_contracts() -> None:
    guardian = EnrichmentGuardian([AlwaysPassContract(), AlwaysWarnContract()])
    report = await guardian.validate("e1", {}, "ctx")
    assert len(report.results) == 2
