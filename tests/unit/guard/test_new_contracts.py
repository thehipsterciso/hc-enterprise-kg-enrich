"""Tests for the 5 new GraphGuard contracts and the parallel guardian upgrade."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from hckg_enrich.guard.contracts.circular_dependency import CircularDependencyContract
from hckg_enrich.guard.contracts.data_asset_ownership import DataAssetOwnershipContract
from hckg_enrich.guard.contracts.entity_deduplication import EntityDeduplicationContract
from hckg_enrich.guard.contracts.person_role_consistency import PersonRoleConsistencyContract
from hckg_enrich.guard.contracts.relationship_semantics import RelationshipTypeSemanticsContract
from hckg_enrich.guard.guardian import EnrichmentGuardian
from hckg_enrich.guard.contract import ContractSeverity


# ---------------------------------------------------------------------------
# CircularDependencyContract
# ---------------------------------------------------------------------------

@pytest.fixture
def circ_contract():
    return CircularDependencyContract()


@pytest.mark.asyncio
async def test_circular_no_dep_rels_passes(circ_contract):
    # No dependency-type relationships → no cycle risk
    result = await circ_contract.evaluate(
        entity_id="sys-001",
        proposed_enrichments={
            "entity_name": "ERP",
            "proposed_relationships": [
                {"relationship_type": "works_in", "target_name": "Finance"}
            ],
            "existing_relationships": [],
        },
        graph_context="",
    )
    assert result.passed


@pytest.mark.asyncio
async def test_circular_simple_cycle_blocked(circ_contract):
    result = await circ_contract.evaluate(
        entity_id="sys-A",
        proposed_enrichments={
            "entity_name": "System A",
            "proposed_relationships": [
                {"relationship_type": "depends_on", "target_name": "sys-B"}
            ],
            "existing_relationships": [
                # B already depends on A → cycle
                {"relationship_type": "depends_on", "source": "sys-B", "target": "sys-A"}
            ],
        },
        graph_context="",
    )
    assert not result.passed
    assert result.severity == ContractSeverity.ERROR
    assert "circular" in result.message.lower()


@pytest.mark.asyncio
async def test_circular_no_cycle_passes(circ_contract):
    result = await circ_contract.evaluate(
        entity_id="sys-A",
        proposed_enrichments={
            "entity_name": "System A",
            "proposed_relationships": [
                {"relationship_type": "depends_on", "target_name": "sys-C"}
            ],
            "existing_relationships": [
                # B depends on A — but A doesn't depend on C through B
                {"relationship_type": "depends_on", "source": "sys-B", "target": "sys-A"}
            ],
        },
        graph_context="",
    )
    assert result.passed


# ---------------------------------------------------------------------------
# EntityDeduplicationContract
# ---------------------------------------------------------------------------

@pytest.fixture
def dedup_contract():
    return EntityDeduplicationContract(similarity_threshold=0.7)


@pytest.mark.asyncio
async def test_dedup_no_similar_entities_passes(dedup_contract):
    result = await dedup_contract.evaluate(
        entity_id="ent-001",
        proposed_enrichments={
            "proposed_relationships": [
                {"relationship_type": "supplied_by", "target_name": "Accenture Inc"}
            ],
            "existing_entities": [{"name": "IBM Watson"}, {"name": "Salesforce CRM"}],
        },
        graph_context="",
    )
    assert result.passed


@pytest.mark.asyncio
async def test_dedup_near_duplicate_warns(dedup_contract):
    # "Microsoft Azure Cloud Platform" vs "Microsoft Azure Cloud"
    # tokens: {microsoft, azure, cloud, platform} ∩ {microsoft, azure, cloud} = 3/4 = 0.75 >= 0.7
    result = await dedup_contract.evaluate(
        entity_id="ent-001",
        proposed_enrichments={
            "proposed_relationships": [
                {"relationship_type": "supplied_by", "target_name": "Microsoft Azure Cloud Platform"}
            ],
            "existing_entities": [
                {"name": "Microsoft Azure Cloud"},
                {"name": "AWS Cloud"},
            ],
        },
        graph_context="",
    )
    assert not result.passed
    assert result.severity == ContractSeverity.WARNING


@pytest.mark.asyncio
async def test_dedup_empty_proposed_rels_passes(dedup_contract):
    result = await dedup_contract.evaluate(
        entity_id="ent-001",
        proposed_enrichments={
            "proposed_relationships": [],
            "existing_entities": [{"name": "Azure"}],
        },
        graph_context="",
    )
    assert result.passed


# ---------------------------------------------------------------------------
# DataAssetOwnershipContract (LLM-based)
# ---------------------------------------------------------------------------

def _make_llm(response: str = '{"passes": true, "reason": "Valid stewardship"}') -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=response)
    return llm


@pytest.mark.asyncio
async def test_data_asset_skips_non_data_entity():
    contract = DataAssetOwnershipContract(llm=_make_llm())
    result = await contract.evaluate(
        entity_id="dept-001",
        proposed_enrichments={"entity_type": "department", "proposed_relationships": []},
        graph_context="",
    )
    assert result.passed
    assert "skipped" in result.message


@pytest.mark.asyncio
async def test_data_asset_passes_on_valid_ownership():
    contract = DataAssetOwnershipContract(llm=_make_llm('{"passes": true, "reason": "OK"}'))
    result = await contract.evaluate(
        entity_id="da-001",
        proposed_enrichments={"entity_type": "data_asset", "proposed_relationships": []},
        graph_context="",
    )
    assert result.passed


@pytest.mark.asyncio
async def test_data_asset_blocked_on_violation():
    contract = DataAssetOwnershipContract(
        llm=_make_llm('{"passes": false, "reason": "PII data owned by Sales — governance violation"}')
    )
    result = await contract.evaluate(
        entity_id="da-001",
        proposed_enrichments={"entity_type": "data_asset", "proposed_relationships": []},
        graph_context="",
    )
    assert not result.passed
    assert result.severity == ContractSeverity.ERROR


@pytest.mark.asyncio
async def test_data_asset_fail_closed_on_parse_error():
    contract = DataAssetOwnershipContract(llm=_make_llm("not valid json"))
    result = await contract.evaluate(
        entity_id="da-001",
        proposed_enrichments={"entity_type": "data_asset", "proposed_relationships": []},
        graph_context="",
    )
    # Must fail closed — NOT pass
    assert not result.passed
    assert "GG-006" in result.message


# ---------------------------------------------------------------------------
# RelationshipTypeSemanticsContract (hybrid rule-based + LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rel_semantics_no_rels_passes():
    contract = RelationshipTypeSemanticsContract(llm=_make_llm())
    result = await contract.evaluate(
        entity_id="ent-001",
        proposed_enrichments={
            "entity_type": "system",
            "proposed_relationships": [],
        },
        graph_context="",
    )
    assert result.passed


@pytest.mark.asyncio
async def test_rel_semantics_valid_schema_passes():
    contract = RelationshipTypeSemanticsContract(llm=_make_llm())
    result = await contract.evaluate(
        entity_id="person-001",
        proposed_enrichments={
            "entity_type": "person",
            "proposed_relationships": [
                {"relationship_type": "works_in", "target_name": "Finance", "target_type": "department"}
            ],
        },
        graph_context="",
    )
    assert result.passed


@pytest.mark.asyncio
async def test_rel_semantics_schema_violation_blocked():
    contract = RelationshipTypeSemanticsContract(llm=_make_llm())
    result = await contract.evaluate(
        entity_id="sys-001",
        proposed_enrichments={
            "entity_type": "system",
            "proposed_relationships": [
                # works_in requires Person source, not System
                {"relationship_type": "works_in", "target_name": "Finance", "target_type": "department"}
            ],
        },
        graph_context="",
    )
    assert not result.passed
    assert result.severity == ContractSeverity.ERROR


@pytest.mark.asyncio
async def test_rel_semantics_fail_closed_on_llm_parse_error():
    contract = RelationshipTypeSemanticsContract(
        llm=_make_llm("malformed json {{{")
    )
    # ambiguous type that defers to LLM
    result = await contract.evaluate(
        entity_id="ent-001",
        proposed_enrichments={
            "entity_type": "system",
            "proposed_relationships": [
                {"relationship_type": "supports", "target_name": "X", "target_type": "system"}
            ],
        },
        graph_context="",
    )
    # Should fail closed
    assert not result.passed


# ---------------------------------------------------------------------------
# Parallel guardian upgrade
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guardian_runs_contracts_in_parallel():
    """Verify parallel execution: all contracts fire concurrently."""
    call_times: list[float] = []
    import time

    async def slow_contract_fn(entity_id, proposed, ctx):
        start = time.monotonic()
        await asyncio.sleep(0.05)  # simulate 50ms LLM call
        from hckg_enrich.guard.contract import ContractResult, ContractSeverity
        call_times.append(time.monotonic() - start)
        return ContractResult(
            contract_id="slow-contract",
            passed=True,
            severity=ContractSeverity.ERROR,
            entity_id=entity_id,
        )

    # Build 3 mock contracts
    from hckg_enrich.guard.contract import QualityContract
    class SlowContract(QualityContract):
        id = "slow-contract"
        severity = ContractSeverity.ERROR
        async def evaluate(self, entity_id, proposed, ctx):
            return await slow_contract_fn(entity_id, proposed, ctx)

    contracts = [SlowContract() for _ in range(3)]
    guardian = EnrichmentGuardian(contracts=contracts)

    start = time.monotonic()
    report = await guardian.validate("ent-001", {}, "")
    elapsed = time.monotonic() - start

    # 3 × 50ms sequential = 150ms; parallel should be ~50ms
    assert elapsed < 0.12, f"Guardian too slow ({elapsed:.3f}s) — may not be parallel"
    assert report.passed


@pytest.mark.asyncio
async def test_guardian_fail_closed_on_contract_exception():
    """A contract that throws should result in a FAIL, not a pass."""
    from hckg_enrich.guard.contract import QualityContract, ContractSeverity

    class ExplodingContract(QualityContract):
        id = "exploding-contract"
        severity = ContractSeverity.ERROR
        async def evaluate(self, entity_id, proposed, ctx):
            raise RuntimeError("boom")

    guardian = EnrichmentGuardian(contracts=[ExplodingContract()])
    report = await guardian.validate("ent-001", {}, "")
    assert not report.passed
    assert any("boom" in r.message for r in report.blocking_failures)
