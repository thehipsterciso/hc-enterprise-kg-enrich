"""Tests for EntityDiscoveryAgent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel

import pytest

from hckg_enrich.agents.discovery_agent import EntityDiscoveryAgent
from hckg_enrich.org.profile import OrgProfile
from hckg_enrich.providers.base import SearchResult
from hckg_enrich.scoring.gap_analysis import GapReport, GapItem


def _gap_report(entity_types: list[str], entity_ids: list[str] | None = None) -> GapReport:
    return GapReport(
        gaps=[],
        entity_ids_to_enrich=entity_ids or [],
        entity_types_to_create=entity_types,
    )


def _org_profile(name="Acme Corp", industry="technology") -> OrgProfile:
    return OrgProfile(org_name=name, industry=industry)


def _mock_llm(entity_names: list[str] | None = None) -> MagicMock:
    llm = MagicMock()

    class _Discovered(BaseModel):
        name: str
        description: str = "Test entity"

    class _Result(BaseModel):
        entities: list[_Discovered] = []

    names = entity_names or ["Entity A", "Entity B"]
    result = _Result(entities=[_Discovered(name=n) for n in names])
    llm.complete_structured = AsyncMock(return_value=result)
    return llm


def _mock_search(urls: list[str] | None = None) -> AsyncMock:
    mock = AsyncMock()
    results = [
        SearchResult(
            title=f"Result {i}",
            url=(urls or ["https://example.com"])[min(i, len(urls or ["https://example.com"])-1)],
            snippet="Test snippet",
            score=0.9,
        )
        for i in range(len(urls or ["https://example.com"]))
    ]
    mock.search = AsyncMock(return_value=results)
    return mock


class TestEntityDiscovery:
    @pytest.mark.asyncio
    async def test_discovered_entities_added_to_graph(self):
        graph = {"entities": [], "relationships": []}
        agent = EntityDiscoveryAgent(
            graph=graph,
            llm=_mock_llm(["Alpha System", "Beta System"]),
            search=_mock_search(["https://a.com"]),
        )
        gap = _gap_report(["system"])
        added = await agent.discover(gap, _org_profile())
        assert len(added) == 2
        assert len(graph["entities"]) == 2

    @pytest.mark.asyncio
    async def test_discovered_entity_has_correct_type(self):
        graph = {"entities": [], "relationships": []}
        agent = EntityDiscoveryAgent(
            graph=graph,
            llm=_mock_llm(["Some Risk"]),
            search=_mock_search(["https://risk.com"]),
        )
        await agent.discover(_gap_report(["risk"]), _org_profile())
        entity = graph["entities"][0]
        assert entity["entity_type"] == "risk"

    @pytest.mark.asyncio
    async def test_discovered_entity_has_uuid_id(self):
        import re
        graph = {"entities": [], "relationships": []}
        agent = EntityDiscoveryAgent(
            graph=graph,
            llm=_mock_llm(["Test Entity"]),
            search=_mock_search(["https://x.com"]),
        )
        await agent.discover(_gap_report(["vendor"]), _org_profile())
        eid = graph["entities"][0]["id"]
        assert re.match(r"[0-9a-f-]{36}", eid)

    @pytest.mark.asyncio
    async def test_discovery_provenance_contains_source_urls(self):
        graph = {"entities": [], "relationships": []}
        agent = EntityDiscoveryAgent(
            graph=graph,
            llm=_mock_llm(["Some Control"]),
            search=_mock_search(["https://nist.gov/control"]),
        )
        await agent.discover(_gap_report(["control"]), _org_profile())
        prov = graph["entities"][0]["provenance"]
        assert "source_urls" in prov
        assert "https://nist.gov/control" in prov["source_urls"]

    @pytest.mark.asyncio
    async def test_discovery_provenance_has_discovery_method(self):
        graph = {"entities": [], "relationships": []}
        agent = EntityDiscoveryAgent(
            graph=graph,
            llm=_mock_llm(["Test"]),
            search=_mock_search(["https://x.com"]),
        )
        await agent.discover(_gap_report(["control"]), _org_profile())
        prov = graph["entities"][0]["provenance"]
        assert prov["discovery_method"] == "entity_discovery_agent"

    @pytest.mark.asyncio
    async def test_discovery_confidence_is_t3(self):
        graph = {"entities": [], "relationships": []}
        agent = EntityDiscoveryAgent(
            graph=graph,
            llm=_mock_llm(["Test"]),
            search=_mock_search(["https://x.com"]),
        )
        await agent.discover(_gap_report(["control"]), _org_profile())
        prov = graph["entities"][0]["provenance"]
        assert prov["confidence_tier"] == "T3"

    @pytest.mark.asyncio
    async def test_duplicate_names_not_added(self):
        existing = {"id": "existing-001", "entity_type": "system", "name": "Alpha System"}
        graph = {"entities": [existing], "relationships": []}
        agent = EntityDiscoveryAgent(
            graph=graph,
            llm=_mock_llm(["Alpha System", "Beta System"]),
            search=_mock_search(["https://x.com"]),
        )
        await agent.discover(_gap_report(["system"]), _org_profile())
        names = [e["name"] for e in graph["entities"]]
        assert names.count("Alpha System") == 1
        assert "Beta System" in names

    @pytest.mark.asyncio
    async def test_no_search_provider_returns_empty(self):
        graph = {"entities": [], "relationships": []}
        agent = EntityDiscoveryAgent(graph=graph, llm=_mock_llm(), search=None)
        added = await agent.discover(_gap_report(["control"]), _org_profile())
        assert added == []
        assert graph["entities"] == []

    @pytest.mark.asyncio
    async def test_multiple_entity_types_all_discovered(self):
        from pydantic import BaseModel
        graph = {"entities": [], "relationships": []}

        # Return different entity names per entity type to avoid dedup collision
        class _D(BaseModel):
            name: str
            description: str = "Test entity"

        class _R(BaseModel):
            entities: list[_D] = []

        call_counter = {"n": 0}
        type_names = [["Access Control Policy"], ["Credit Risk Framework"]]

        async def _side_effect(messages, schema, system=""):
            idx = call_counter["n"] % 2
            call_counter["n"] += 1
            return _R(entities=[_D(name=n) for n in type_names[idx]])

        llm = MagicMock()
        llm.complete_structured = AsyncMock(side_effect=_side_effect)

        agent = EntityDiscoveryAgent(
            graph=graph,
            llm=llm,
            search=_mock_search(["https://x.com"]),
        )
        await agent.discover(_gap_report(["control", "risk"]), _org_profile())
        types = {e["entity_type"] for e in graph["entities"]}
        assert "control" in types
        assert "risk" in types

    @pytest.mark.asyncio
    async def test_run_id_stored_in_provenance(self):
        graph = {"entities": [], "relationships": []}
        agent = EntityDiscoveryAgent(
            graph=graph,
            llm=_mock_llm(["Test"]),
            search=_mock_search(["https://x.com"]),
        )
        await agent.discover(_gap_report(["vendor"]), _org_profile(), run_id="run-123")
        assert graph["entities"][0]["provenance"]["run_id"] == "run-123"
