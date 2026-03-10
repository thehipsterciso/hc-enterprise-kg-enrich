"""QualityContract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any


class ContractSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ContractResult:
    def __init__(
        self,
        contract_id: str,
        passed: bool,
        severity: ContractSeverity,
        message: str = "",
        entity_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.contract_id = contract_id
        self.passed = passed
        self.severity = severity
        self.message = message
        self.entity_id = entity_id
        self.details = details

    def __repr__(self) -> str:
        status = "PASS" if self.passed else f"FAIL({self.severity})"
        return f"ContractResult({self.contract_id}, {status}, entity={self.entity_id})"


class QualityContract(ABC):
    """Base for all GraphGuard quality contracts."""

    id: str
    severity: ContractSeverity = ContractSeverity.ERROR
    description: str = ""

    @abstractmethod
    async def evaluate(
        self,
        entity_id: str,
        proposed_enrichments: dict[str, Any],
        graph_context: str,
    ) -> ContractResult:
        """Evaluate the contract against proposed enrichments."""
        ...
