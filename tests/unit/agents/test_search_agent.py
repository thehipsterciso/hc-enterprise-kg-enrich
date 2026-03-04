"""Tests for SearchAgent."""
from __future__ import annotations

import pytest

from hckg_enrich.agents.base import AgentMessage, AgentRole
from hckg_enrich.agents.search_agent import SearchAgent

_BASE_PAYLOAD = {
    "entity_id": "e1",
    "entity_name": "Finance",
    "entity_type": "department",
    "graph_context": "ctx",
}


@pytest.mark.asyncio
async def test_search_agent_with_no_provider() -> None:
    agent = SearchAgent(search=None)
    msg = AgentMessage(
        sender=AgentRole.CONTEXT,
        recipient=AgentRole.SEARCH,
        payload=dict(_BASE_PAYLOAD),
    )
    result = await agent.run(msg)
    assert result.payload["search_context"] == ""
    assert result.recipient == AgentRole.REASONING


@pytest.mark.asyncio
async def test_search_agent_with_provider(mock_search: object) -> None:
    agent = SearchAgent(search=mock_search)  # type: ignore[arg-type]
    msg = AgentMessage(
        sender=AgentRole.CONTEXT,
        recipient=AgentRole.SEARCH,
        payload=dict(_BASE_PAYLOAD),
    )
    result = await agent.run(msg)
    assert len(result.payload["search_context"]) > 0
    assert "ERP" in result.payload["search_context"]


@pytest.mark.asyncio
async def test_search_agent_routes_to_reasoning(mock_search: object) -> None:
    agent = SearchAgent(search=mock_search)  # type: ignore[arg-type]
    msg = AgentMessage(
        sender=AgentRole.CONTEXT,
        recipient=AgentRole.SEARCH,
        payload=dict(_BASE_PAYLOAD),
    )
    result = await agent.run(msg)
    assert result.recipient == AgentRole.REASONING
