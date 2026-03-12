"""Tests for PlausibilityContract (GG-PLAUS-001)."""
from __future__ import annotations

import pytest

from hckg_enrich.guard.contract import ContractSeverity
from hckg_enrich.guard.contracts.plausibility import PLAUSIBILITY_BOUNDS, PlausibilityContract


@pytest.fixture
def contract():
    return PlausibilityContract()


def _ctx(entity_id: str, entity_type: str) -> dict:
    return {
        "entities": [{"id": entity_id, "entity_type": entity_type, "name": "Test"}]
    }


# ---------------------------------------------------------------------------
# Contract metadata
# ---------------------------------------------------------------------------

def test_contract_id(contract):
    assert contract.CONTRACT_ID == "GG-PLAUS-001"


def test_severity_is_warning(contract):
    assert contract.DEFAULT_SEVERITY == ContractSeverity.WARNING


# ---------------------------------------------------------------------------
# Entity not in context → skip gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_entity_passes(contract):
    result = await contract.evaluate(
        entity_id="unknown-id",
        proposed_enrichments={"proposed_attributes": {"annual_compensation": 9999}},
        graph_context={"entities": []},
    )
    assert result.passed
    assert "not found" in result.message


# ---------------------------------------------------------------------------
# No bounds defined for entity_type → pass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_entity_type_passes(contract):
    result = await contract.evaluate(
        entity_id="ent-001",
        proposed_enrichments={"proposed_attributes": {"some_field": 42}},
        graph_context=_ctx("ent-001", "unknown_type"),
    )
    assert result.passed
    assert "No plausibility bounds" in result.message


# ---------------------------------------------------------------------------
# Person — within bounds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_person_valid_compensation_passes(contract):
    result = await contract.evaluate(
        entity_id="p-001",
        proposed_enrichments={"proposed_attributes": {"annual_compensation": 150_000}},
        graph_context=_ctx("p-001", "person"),
    )
    assert result.passed


@pytest.mark.asyncio
async def test_person_valid_experience_passes(contract):
    result = await contract.evaluate(
        entity_id="p-001",
        proposed_enrichments={"proposed_attributes": {"years_experience": 15}},
        graph_context=_ctx("p-001", "person"),
    )
    assert result.passed


# ---------------------------------------------------------------------------
# Person — out of bounds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_person_negative_experience_fails(contract):
    result = await contract.evaluate(
        entity_id="p-001",
        proposed_enrichments={"proposed_attributes": {"years_experience": -1}},
        graph_context=_ctx("p-001", "person"),
    )
    assert not result.passed
    assert result.severity == ContractSeverity.WARNING
    assert "years_experience" in result.message


@pytest.mark.asyncio
async def test_person_excessive_compensation_fails(contract):
    result = await contract.evaluate(
        entity_id="p-001",
        proposed_enrichments={"proposed_attributes": {"annual_compensation": 20_000_000}},
        graph_context=_ctx("p-001", "person"),
    )
    assert not result.passed
    assert "annual_compensation" in result.message


@pytest.mark.asyncio
async def test_person_too_many_direct_reports_fails(contract):
    result = await contract.evaluate(
        entity_id="p-001",
        proposed_enrichments={"proposed_attributes": {"direct_reports_count": 501}},
        graph_context=_ctx("p-001", "person"),
    )
    assert not result.passed


# ---------------------------------------------------------------------------
# Risk — boundary conditions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_risk_probability_at_zero_passes(contract):
    result = await contract.evaluate(
        entity_id="r-001",
        proposed_enrichments={"proposed_attributes": {"probability": 0.0}},
        graph_context=_ctx("r-001", "risk"),
    )
    assert result.passed


@pytest.mark.asyncio
async def test_risk_probability_at_one_passes(contract):
    result = await contract.evaluate(
        entity_id="r-001",
        proposed_enrichments={"proposed_attributes": {"probability": 1.0}},
        graph_context=_ctx("r-001", "risk"),
    )
    assert result.passed


@pytest.mark.asyncio
async def test_risk_probability_exceeds_one_fails(contract):
    result = await contract.evaluate(
        entity_id="r-001",
        proposed_enrichments={"proposed_attributes": {"probability": 1.5}},
        graph_context=_ctx("r-001", "risk"),
    )
    assert not result.passed


@pytest.mark.asyncio
async def test_risk_cvss_score_too_high_fails(contract):
    result = await contract.evaluate(
        entity_id="r-001",
        proposed_enrichments={"proposed_attributes": {"cvss_score": 10.1}},
        graph_context=_ctx("r-001", "risk"),
    )
    assert not result.passed


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_system_sla_over_100_fails(contract):
    result = await contract.evaluate(
        entity_id="sys-001",
        proposed_enrichments={"proposed_attributes": {"uptime_sla": 101}},
        graph_context=_ctx("sys-001", "system"),
    )
    assert not result.passed
    assert "uptime_sla" in result.message


@pytest.mark.asyncio
async def test_system_valid_cost_passes(contract):
    result = await contract.evaluate(
        entity_id="sys-001",
        proposed_enrichments={"proposed_attributes": {"annual_cost": 1_000_000}},
        graph_context=_ctx("sys-001", "system"),
    )
    assert result.passed


# ---------------------------------------------------------------------------
# Non-numeric fields are skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_numeric_field_is_ignored(contract):
    result = await contract.evaluate(
        entity_id="p-001",
        proposed_enrichments={
            "proposed_attributes": {"annual_compensation": "high", "years_experience": "many"}
        },
        graph_context=_ctx("p-001", "person"),
    )
    assert result.passed


# ---------------------------------------------------------------------------
# Fields not in bounds dict are ignored
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_field_is_ignored(contract):
    result = await contract.evaluate(
        entity_id="p-001",
        proposed_enrichments={"proposed_attributes": {"favorite_color": 99999}},
        graph_context=_ctx("p-001", "person"),
    )
    assert result.passed


# ---------------------------------------------------------------------------
# Multiple violations reported together
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_violations_all_reported(contract):
    result = await contract.evaluate(
        entity_id="p-001",
        proposed_enrichments={
            "proposed_attributes": {
                "annual_compensation": -1,
                "years_experience": 200,
                "direct_reports_count": 600,
            }
        },
        graph_context=_ctx("p-001", "person"),
    )
    assert not result.passed
    assert result.details is not None
    assert len(result.details["violations"]) == 3


# ---------------------------------------------------------------------------
# Vendor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vendor_valid_spend_passes(contract):
    result = await contract.evaluate(
        entity_id="v-001",
        proposed_enrichments={"proposed_attributes": {"annual_spend": 5_000_000}},
        graph_context=_ctx("v-001", "vendor"),
    )
    assert result.passed


@pytest.mark.asyncio
async def test_vendor_negative_spend_fails(contract):
    result = await contract.evaluate(
        entity_id="v-001",
        proposed_enrichments={"proposed_attributes": {"annual_spend": -100}},
        graph_context=_ctx("v-001", "vendor"),
    )
    assert not result.passed


# ---------------------------------------------------------------------------
# PLAUSIBILITY_BOUNDS coverage sanity check
# ---------------------------------------------------------------------------

def test_all_expected_entity_types_in_bounds():
    expected = {
        "person", "system", "risk", "vulnerability", "department",
        "vendor", "control", "data_asset", "integration",
    }
    assert expected.issubset(set(PLAUSIBILITY_BOUNDS.keys()))
