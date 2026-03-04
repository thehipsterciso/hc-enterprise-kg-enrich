"""Tests for ContextAgent."""
from __future__ import annotations

from typing import Any

import pytest

from hckg_enrich.agents.base import AgentMessage, AgentRole
from hckg_enrich.agents.context_agent import ContextAgent
from hckg_enrich.context.retriever import KGContextRetriever


@pytest.mark.asyncio
async def test_context_agent_returns_graph_context(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    agent = ContextAgent(retriever)
    msg = AgentMessage(
        sender=AgentRole.CONTEXT,
        recipient=AgentRole.CONTEXT,
        payload={"entity_id": "dept-finance-001"},
    )
    result = await agent.run(msg)
    assert "graph_context" in result.payload
    assert "Finance" in result.payload["graph_context"]
    assert result.payload["entity_type"] == "department"
    assert result.payload["entity_name"] == "Finance"


@pytest.mark.asyncio
async def test_context_agent_routes_to_search(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    agent = ContextAgent(retriever)
    msg = AgentMessage(
        sender=AgentRole.CONTEXT,
        recipient=AgentRole.CONTEXT,
        payload={"entity_id": "dept-finance-001"},
    )
    result = await agent.run(msg)
    assert result.sender == AgentRole.CONTEXT
    assert result.recipient == AgentRole.SEARCH


@pytest.mark.asyncio
async def test_context_agent_preserves_correlation_id(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    agent = ContextAgent(retriever)
    msg = AgentMessage(
        sender=AgentRole.CONTEXT,
        recipient=AgentRole.CONTEXT,
        payload={"entity_id": "dept-finance-001"},
    )
    result = await agent.run(msg)
    assert result.correlation_id == msg.correlation_id
