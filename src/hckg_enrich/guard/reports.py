"""QualityValidationReport — queryable audit record of contract evaluations."""
from __future__ import annotations

from hckg_enrich.guard.contract import ContractResult, ContractSeverity


class QualityValidationReport:
    def __init__(self, entity_id: str, results: list[ContractResult]) -> None:
        self.entity_id = entity_id
        self.results = results

    @property
    def passed(self) -> bool:
        return all(
            r.passed or r.severity == ContractSeverity.WARNING
            for r in self.results
        )

    @property
    def blocking_failures(self) -> list[ContractResult]:
        return [r for r in self.results if not r.passed and r.severity == ContractSeverity.ERROR]

    @property
    def warnings(self) -> list[ContractResult]:
        return [r for r in self.results if not r.passed and r.severity == ContractSeverity.WARNING]

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        parts = [f"Entity {self.entity_id}: {passed}/{total} contracts passed"]
        if self.blocking_failures:
            parts.append(f"{len(self.blocking_failures)} blocking failures")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
        return ", ".join(parts)
