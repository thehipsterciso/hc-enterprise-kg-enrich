"""Tests for SearchAgent — including URL propagation regression (v0.6.0 bug fix)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hckg_enrich.agents.base import AgentMessage, AgentRole
from hckg_enrich.agents.search_agent import SearchAgent
from hckg_enrich.providers.base import SearchResult

_BASE_PAYLOAD = {
    "entity_id": "e1",
    "entity_name": "Finance",
    "entity_type": "department",
    "graph_context": "ctx",
}


def _make_search(results: list | None = None) -> AsyncMock:
    if results is None:
        results = [
            SearchResult(
                title="ERP Ownership Best Practices",
                url="https://example.com/erp-ownership",
                snippet="ERP systems are typically owned by Finance or IT departments.",
                score=0.9,
            )
        ]
    mock = AsyncMock()
    mock.search = AsyncMock(return_value=results)
    return mock


def _msg(extra: dict | None = None) -> AgentMessage:
    payload = dict(_BASE_PAYLOAD)
    if extra:
        payload.update(extra)
    return AgentMessage(
        sender=AgentRole.CONTEXT,
        recipient=AgentRole.SEARCH,
        payload=payload,
    )


class TestNoProvider:
    @pytest.mark.asyncio
    async def test_empty_context_when_no_provider(self):
        agent = SearchAgent(search=None)
        result = await agent.run(_msg())
        assert result.payload["search_context"] == ""

    @pytest.mark.asyncio
    async def test_routes_to_reasoning_when_no_provider(self):
        agent = SearchAgent(search=None)
        result = await agent.run(_msg())
        assert result.recipient == AgentRole.REASONING

    @pytest.mark.asyncio
    async def test_search_sources_empty_when_no_provider(self):
        agent = SearchAgent(search=None)
        result = await agent.run(_msg())
        assert result.payload["search_sources"] == []

    @pytest.mark.asyncio
    async def test_search_queries_empty_when_no_provider(self):
        agent = SearchAgent(search=None)
        result = await agent.run(_msg())
        assert result.payload["search_queries"] == []


class TestURLPropagation:
    @pytest.mark.asyncio
    async def test_search_sources_populated_with_url(self):
        agent = SearchAgent(search=_make_search())
        result = await agent.run(_msg())
        sources = result.payload["search_sources"]
        assert len(sources) > 0
        assert sources[0]["url"] == "https://example.com/erp-ownership"

    @pytest.mark.asyncio
    async def test_search_sources_contain_all_fields(self):
        agent = SearchAgent(search=_make_search())
        result = await agent.run(_msg())
        source = result.payload["search_sources"][0]
        for key in ("url", "title", "snippet", "score", "query"):
            assert key in source

    @pytest.mark.asyncio
    async def test_search_context_includes_url(self):
        agent = SearchAgent(search=_make_search())
        result = await agent.run(_msg())
        ctx = result.payload["search_context"]
        assert "https://example.com/erp-ownership" in ctx

    @pytest.mark.asyncio
    async def test_search_queries_populated(self):
        agent = SearchAgent(search=_make_search())
        result = await agent.run(_msg())
        queries = result.payload["search_queries"]
        assert isinstance(queries, list)
        assert len(queries) >= 1

    @pytest.mark.asyncio
    async def test_multiple_sources_all_have_urls(self):
        results = [
            SearchResult(title="A", url="https://a.com", snippet="aa", score=0.9),
            SearchResult(title="B", url="https://b.com", snippet="bb", score=0.8),
        ]
        agent = SearchAgent(search=_make_search(results))
        result = await agent.run(_msg())
        urls = [s["url"] for s in result.payload["search_sources"]]
        assert "https://a.com" in urls
        assert "https://b.com" in urls

    @pytest.mark.asyncio
    async def test_source_query_matches_issued_query(self):
        agent = SearchAgent(search=_make_search())
        result = await agent.run(_msg())
        queries = result.payload["search_queries"]
        for source in result.payload["search_sources"]:
            assert source["query"] in queries


class TestAdaptiveQueries:
    @pytest.mark.asyncio
    async def test_generates_at_least_one_query(self):
        agent = SearchAgent(search=_make_search())
        result = await agent.run(_msg())
        assert len(result.payload["search_queries"]) >= 1

    @pytest.mark.asyncio
    async def test_max_four_queries(self):
        agent = SearchAgent(search=_make_search())
        payload = {**_BASE_PAYLOAD, "entity": {}}
        result = await agent.run(AgentMessage(
            sender=AgentRole.CONTEXT,
            recipient=AgentRole.SEARCH,
            payload=payload,
        ))
        assert len(result.payload["search_queries"]) <= 4

    @pytest.mark.asyncio
    async def test_org_context_used_when_org_profile_provided(self):
        payload = {
            **_BASE_PAYLOAD,
            "org_profile": {"org_name": "Acme Corp", "industry": "financial services"},
        }
        agent = SearchAgent(search=_make_search())
        result = await agent.run(AgentMessage(
            sender=AgentRole.CONTEXT,
            recipient=AgentRole.SEARCH,
            payload=payload,
        ))
        queries = result.payload["search_queries"]
        combined = " ".join(queries).lower()
        assert "acme" in combined or "financial" in combined

    @pytest.mark.asyncio
    async def test_fewer_queries_when_fields_populated(self):
        agent = SearchAgent(search=_make_search())
        full_entity = {
            "criticality": "HIGH", "owner": "CIO", "tech_stack": "AWS",
            "data_classification": "Confidential", "risk_tier": "T1",
            "framework": "NIST", "vendor_name": "SAP", "budget": "1M",
            "headcount": "50", "responsible_team": "IT", "status": "Active",
        }
        payload = {**_BASE_PAYLOAD, "entity": full_entity}
        result = await agent.run(AgentMessage(
            sender=AgentRole.CONTEXT,
            recipient=AgentRole.SEARCH,
            payload=payload,
        ))
        assert len(result.payload["search_queries"]) <= 2


class TestResilience:
    @pytest.mark.asyncio
    async def test_routes_to_reasoning(self):
        agent = SearchAgent(search=_make_search())
        result = await agent.run(_msg())
        assert result.recipient == AgentRole.REASONING

    @pytest.mark.asyncio
    async def test_search_failure_does_not_raise(self):
        failing = AsyncMock()
        failing.search = AsyncMock(side_effect=RuntimeError("network error"))
        agent = SearchAgent(search=failing)
        result = await agent.run(_msg())
        assert result.payload["search_sources"] == []
        assert result.payload["search_context"] == ""

    @pytest.mark.asyncio
    async def test_partial_failure_returns_successful_results(self):
        call_count = 0

        async def side_effect(query, n=3):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("timeout")
            return [SearchResult(title="OK", url="https://ok.com", snippet="ok", score=0.8)]

        mock = AsyncMock()
        mock.search = AsyncMock(side_effect=side_effect)
        agent = SearchAgent(search=mock)
        payload = {**_BASE_PAYLOAD, "entity": {}}
        result = await agent.run(AgentMessage(
            sender=AgentRole.CONTEXT,
            recipient=AgentRole.SEARCH,
            payload=payload,
        ))
        assert len(result.payload["search_sources"]) >= 1
