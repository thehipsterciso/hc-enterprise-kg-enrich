"""Tests for GraphGuard quality contracts."""
from __future__ import annotations

import pytest

from hckg_enrich.guard.contract import ContractSeverity
from hckg_enrich.guard.contracts.org_hierarchy import OrgHierarchyContract
from hckg_enrich.guard.contracts.system_ownership import SystemOwnershipContract
from hckg_enrich.guard.contracts.vendor_relationship import VendorRelationshipContract


@pytest.mark.asyncio
async def test_org_hierarchy_passes_on_valid(mock_llm: object) -> None:
    mock_llm.complete.return_value = '{"passes": true, "reason": "Valid org structure"}'  # type: ignore[union-attr]
    contract = OrgHierarchyContract(llm=mock_llm)  # type: ignore[arg-type]
    result = await contract.evaluate("dept-001", {"parent": "CFO"}, "Finance dept context")
    assert result.passed
    assert result.contract_id == "org-hierarchy-001"
    assert result.severity == ContractSeverity.ERROR


@pytest.mark.asyncio
async def test_org_hierarchy_fails_on_invalid(mock_llm: object) -> None:
    mock_llm.complete.return_value = '{"passes": false, "reason": "Finance cannot report to HR"}'  # type: ignore[union-attr]
    contract = OrgHierarchyContract(llm=mock_llm)  # type: ignore[arg-type]
    result = await contract.evaluate("dept-001", {"parent": "HR"}, "Finance dept context")
    assert not result.passed
    assert result.severity == ContractSeverity.ERROR
    assert "HR" in result.message


@pytest.mark.asyncio
async def test_system_ownership_passes(mock_llm: object) -> None:
    mock_llm.complete.return_value = '{"passes": true, "reason": "ERP owned by IT is valid"}'  # type: ignore[union-attr]
    contract = SystemOwnershipContract(llm=mock_llm)  # type: ignore[arg-type]
    result = await contract.evaluate("sys-001", {"owner": "IT"}, "SAP ERP context")
    assert result.passed
    assert result.contract_id == "system-ownership-001"


@pytest.mark.asyncio
async def test_vendor_relationship_is_warning_severity(mock_llm: object) -> None:
    # noqa: E501
    mock_llm.complete.return_value = (  # type: ignore[union-attr]
        '{"passes": false, "reason": "Marketing should not manage cloud infra"}'
    )
    contract = VendorRelationshipContract(llm=mock_llm)  # type: ignore[arg-type]
    result = await contract.evaluate("v-001", {}, "Cloud vendor context")
    assert not result.passed
    assert result.severity == ContractSeverity.WARNING  # WARNING, not ERROR


@pytest.mark.asyncio
async def test_contract_handles_bad_llm_response(mock_llm: object) -> None:
    mock_llm.complete.return_value = "not json at all"  # type: ignore[union-attr]
    contract = OrgHierarchyContract(llm=mock_llm)  # type: ignore[arg-type]
    result = await contract.evaluate("dept-001", {}, "context")
    assert result.passed  # defaults to pass on parse failure


@pytest.mark.asyncio
async def test_contract_handles_markdown_wrapped_json(mock_llm: object) -> None:
    mock_llm.complete.return_value = '```json\n{"passes": true, "reason": "ok"}\n```'  # type: ignore[union-attr]
    contract = OrgHierarchyContract(llm=mock_llm)  # type: ignore[arg-type]
    result = await contract.evaluate("dept-001", {}, "context")
    assert result.passed
