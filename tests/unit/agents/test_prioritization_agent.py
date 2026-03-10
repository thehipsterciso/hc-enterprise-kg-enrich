"""Tests for PrioritizationAgent."""
from __future__ import annotations

import pytest

from hckg_enrich.agents.base import AgentMessage, AgentRole
from hckg_enrich.agents.prioritization_agent import (
    PrioritizationAgent,
    _missing_field_score,
    _staleness_score,
    _connectivity_score,
    _type_weight,
    HIGH_VALUE_FIELDS,
)


@pytest.fixture
def agent() -> PrioritizationAgent:
    return PrioritizationAgent()


@pytest.fixture
def sample_entities():
    return [
        {
            "id": "sys-001",
            "entity_type": "system",
            "name": "SAP ERP",
            "description": "ERP system",
        },
        {
            "id": "dept-001",
            "entity_type": "department",
            "name": "Finance",
            "description": "Finance dept",
        },
        {
            "id": "vendor-001",
            "entity_type": "vendor",
            "name": "AWS",
            "description": "Cloud provider",
            "risk_tier": "high",
        },
    ]


@pytest.mark.asyncio
async def test_prioritization_returns_ordered_list(agent, sample_entities):
    msg = AgentMessage(
        sender=AgentRole.CONTEXT,
        recipient=AgentRole.CONTEXT,
        payload={"entities": sample_entities, "relationships": []},
    )
    result = await agent.run(msg)
    prioritized = result.payload["prioritized_entities"]
    assert len(prioritized) == 3
    # system should score higher than department (higher type weight)
    names = [e["name"] for e in prioritized]
    assert names.index("SAP ERP") < names.index("Finance")


@pytest.mark.asyncio
async def test_prioritization_respects_limit(agent, sample_entities):
    msg = AgentMessage(
        sender=AgentRole.CONTEXT,
        recipient=AgentRole.CONTEXT,
        payload={"entities": sample_entities, "relationships": [], "limit": 2},
    )
    result = await agent.run(msg)
    assert len(result.payload["prioritized_entities"]) == 2
    assert result.payload["total_candidates"] == 2


@pytest.mark.asyncio
async def test_prioritization_respects_entity_type_filter(agent, sample_entities):
    msg = AgentMessage(
        sender=AgentRole.CONTEXT,
        recipient=AgentRole.CONTEXT,
        payload={
            "entities": sample_entities,
            "relationships": [],
            "entity_type_filter": "system",
        },
    )
    result = await agent.run(msg)
    prioritized = result.payload["prioritized_entities"]
    assert len(prioritized) == 1
    assert prioritized[0]["entity_type"] == "system"


@pytest.mark.asyncio
async def test_prioritization_includes_scores(agent, sample_entities):
    msg = AgentMessage(
        sender=AgentRole.CONTEXT,
        recipient=AgentRole.CONTEXT,
        payload={"entities": sample_entities, "relationships": []},
    )
    result = await agent.run(msg)
    scores = result.payload["priority_scores"]
    assert len(scores) == 3
    for s in scores:
        assert "total_score" in s
        assert "breakdown" in s
        assert s["total_score"] > 0


def test_type_weight_known_types():
    assert _type_weight("system") == 0.30
    assert _type_weight("vendor") == 0.25
    assert _type_weight("department") == 0.18


def test_type_weight_unknown_type():
    assert _type_weight("unknown_type_xyz") == 0.10


def test_missing_field_score_empty_entity():
    entity = {"id": "x", "entity_type": "system", "name": "X"}
    score, reasons = _missing_field_score(entity)
    # Most high-value fields are missing
    assert score > 0.15
    assert len(reasons) > 0


def test_missing_field_score_fully_populated_entity():
    entity = {k: "value" for k in HIGH_VALUE_FIELDS}
    entity.update({"id": "x", "entity_type": "system", "name": "X"})
    score, _ = _missing_field_score(entity)
    # Almost nothing missing — score should be close to 0
    assert score < 0.05


def test_connectivity_score_no_rels():
    score, reasons = _connectivity_score("ent-001", [])
    assert score == 0.0
    assert reasons == []


def test_connectivity_score_highly_connected():
    rels = [{"source": "ent-001", "target": f"ent-{i:03d}"} for i in range(15)]
    score, reasons = _connectivity_score("ent-001", rels)
    assert score > 0.15
    assert len(reasons) > 0


def test_staleness_score_never_enriched():
    entity = {"id": "x"}
    score, reasons = _staleness_score(entity)
    assert score == 0.20
    assert "never enriched" in reasons[0]


def test_staleness_score_previously_enriched():
    entity = {"id": "x", "provenance": {"enriched_at": "2025-01-01T00:00:00"}}
    score, _ = _staleness_score(entity)
    assert score < 0.20
