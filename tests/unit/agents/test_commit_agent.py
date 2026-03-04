"""Tests for CommitAgent relationship proposal commit (issue #5)."""
from __future__ import annotations

import pytest

from hckg_enrich.agents.base import AgentMessage, AgentRole
from hckg_enrich.agents.commit_agent import CommitAgent


@pytest.fixture()
def graph_with_two_entities() -> dict:
    return {
        "entities": [
            {"id": "e1", "entity_type": "system", "name": "SAP ERP"},
            {"id": "e2", "entity_type": "department", "name": "Finance"},
        ],
        "relationships": [],
    }


def _passed_report() -> dict:
    return {"passed": True, "blocking_failures": [], "warnings": []}


def _failed_report() -> dict:
    return {
        "passed": False,
        "blocking_failures": [{"contract": "x", "message": "bad"}],
        "warnings": [],
    }


@pytest.mark.asyncio
async def test_commit_adds_relationship_proposal(graph_with_two_entities):
    agent = CommitAgent(graph_with_two_entities)
    msg = AgentMessage(
        sender=AgentRole.COHERENCE,
        recipient=AgentRole.COMMIT,
        payload={
            "entity_id": "e1",
            "proposal": {
                "proposed_attributes": {},
                "proposed_relationships": [
                    {
                        "relationship_type": "owned_by",
                        "target_name": "Finance",
                        "target_type": "department",
                        "rationale": "ERP systems are owned by Finance",
                    }
                ],
            },
            "validation_report": _passed_report(),
        },
    )
    result_msg = await agent.run(msg)
    result = result_msg.payload["commit_result"]

    assert result["applied"] is True
    assert result["relationships_added"] == 1
    assert len(graph_with_two_entities["relationships"]) == 1
    rel = graph_with_two_entities["relationships"][0]
    assert rel["relationship_type"] == "owned_by"
    assert rel["source_id"] == "e1"
    assert rel["target_id"] == "e2"


@pytest.mark.asyncio
async def test_commit_skips_duplicate_relationship(graph_with_two_entities):
    # pre-populate a relationship
    graph_with_two_entities["relationships"].append({
        "id": "existing",
        "relationship_type": "owned_by",
        "source_id": "e1",
        "target_id": "e2",
    })

    agent = CommitAgent(graph_with_two_entities)
    msg = AgentMessage(
        sender=AgentRole.COHERENCE,
        recipient=AgentRole.COMMIT,
        payload={
            "entity_id": "e1",
            "proposal": {
                "proposed_attributes": {},
                "proposed_relationships": [
                    {
                        "relationship_type": "owned_by",
                        "target_name": "Finance",
                        "target_type": "department",
                        "rationale": "duplicate",
                    }
                ],
            },
            "validation_report": _passed_report(),
        },
    )
    await agent.run(msg)
    assert len(graph_with_two_entities["relationships"]) == 1  # no new rel added


@pytest.mark.asyncio
async def test_commit_skips_relationships_when_blocked(graph_with_two_entities):
    agent = CommitAgent(graph_with_two_entities)
    msg = AgentMessage(
        sender=AgentRole.COHERENCE,
        recipient=AgentRole.COMMIT,
        payload={
            "entity_id": "e1",
            "proposal": {
                "proposed_attributes": {},
                "proposed_relationships": [
                    {
                        "relationship_type": "owned_by",
                        "target_name": "Finance",
                        "target_type": "department",
                        "rationale": "should not be committed",
                    }
                ],
            },
            "validation_report": _failed_report(),
        },
    )
    result_msg = await agent.run(msg)
    result = result_msg.payload["commit_result"]
    assert result["applied"] is False
    assert result["reason"] == "Blocked by GraphGuard"
    assert len(graph_with_two_entities["relationships"]) == 0


@pytest.mark.asyncio
async def test_commit_skips_relationship_when_target_not_found(graph_with_two_entities):
    agent = CommitAgent(graph_with_two_entities)
    msg = AgentMessage(
        sender=AgentRole.COHERENCE,
        recipient=AgentRole.COMMIT,
        payload={
            "entity_id": "e1",
            "proposal": {
                "proposed_attributes": {},
                "proposed_relationships": [
                    {
                        "relationship_type": "owned_by",
                        "target_name": "Nonexistent Department",
                        "target_type": "department",
                        "rationale": "target missing",
                    }
                ],
            },
            "validation_report": _passed_report(),
        },
    )
    result_msg = await agent.run(msg)
    result = result_msg.payload["commit_result"]
    assert result["applied"] is True
    assert result["relationships_added"] == 0
    assert len(graph_with_two_entities["relationships"]) == 0
