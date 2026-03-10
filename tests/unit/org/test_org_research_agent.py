"""Tests for OrgResearchAgent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from hckg_enrich.org.profile import OrgProfile
from hckg_enrich.org.research_agent import OrgResearchAgent
from hckg_enrich.providers.base import SearchResult


def _make_llm(fields: dict | None = None) -> MagicMock:
    llm = MagicMock()
    if fields is None:
        fields = {
            "org_name": "Apple Inc",
            "industry": "technology",
            "sector": "Information Technology",
            "headcount_tier": "enterprise",
            "regulatory_regime": ["SOX"],
            "industry_frameworks": ["ISO 27001"],
        }

    class _Extracted(BaseModel):
        org_name: str = fields.get("org_name", "")
        industry: str = fields.get("industry", "")
        sector: str = fields.get("sector", "")
        country: str = fields.get("country", "US")
        headcount_tier: str = fields.get("headcount_tier", "")
        revenue_tier: str = fields.get("revenue_tier", "")
        key_roles: list[str] = fields.get("key_roles", [])
        subsidiaries: list[str] = fields.get("subsidiaries", [])
        regulatory_regime: list[str] = fields.get("regulatory_regime", [])
        industry_frameworks: list[str] = fields.get("industry_frameworks", [])
        tech_profile: dict[str, str] = {}

    extracted = _Extracted()
    llm.complete_structured = AsyncMock(return_value=extracted)
    return llm


def _make_search(results: list[SearchResult] | None = None) -> AsyncMock:
    if results is None:
        results = [
            SearchResult(
                title="Apple Inc Profile",
                url="https://apple.com/about",
                snippet="Apple is a technology company.",
                score=0.9,
            )
        ]
    mock = AsyncMock()
    mock.search = AsyncMock(return_value=results)
    return mock


class TestNoSearch:
    @pytest.mark.asyncio
    async def test_returns_minimal_profile_without_search(self):
        agent = OrgResearchAgent(llm=_make_llm(), search=None)
        profile = await agent.research(ticker="AAPL", org_name="Apple")
        assert isinstance(profile, OrgProfile)
        assert profile.research_confidence == 0.0

    @pytest.mark.asyncio
    async def test_ticker_preserved_without_search(self):
        agent = OrgResearchAgent(llm=_make_llm(), search=None)
        profile = await agent.research(ticker="AAPL")
        assert profile.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_industry_hint_used_when_no_search(self):
        agent = OrgResearchAgent(llm=_make_llm(), search=None)
        profile = await agent.research(org_name="Acme", industry="manufacturing")
        assert profile.industry == "manufacturing"


class TestWithSearch:
    @pytest.mark.asyncio
    async def test_sources_populated_from_search_results(self):
        agent = OrgResearchAgent(llm=_make_llm(), search=_make_search())
        profile = await agent.research(ticker="AAPL")
        assert len(profile.sources) > 0

    @pytest.mark.asyncio
    async def test_source_urls_are_real_urls(self):
        agent = OrgResearchAgent(llm=_make_llm(), search=_make_search())
        profile = await agent.research(ticker="AAPL")
        for source in profile.sources:
            assert source["url"].startswith("http")

    @pytest.mark.asyncio
    async def test_llm_extracted_fields_used(self):
        agent = OrgResearchAgent(llm=_make_llm(), search=_make_search())
        profile = await agent.research(ticker="AAPL")
        assert profile.industry == "technology"
        assert profile.headcount_tier == "enterprise"
        assert "SOX" in profile.regulatory_regime

    @pytest.mark.asyncio
    async def test_research_confidence_nonzero_with_results(self):
        agent = OrgResearchAgent(llm=_make_llm(), search=_make_search())
        profile = await agent.research(ticker="AAPL")
        assert profile.research_confidence > 0.0

    @pytest.mark.asyncio
    async def test_search_failure_returns_minimal_profile(self):
        failing = AsyncMock()
        failing.search = AsyncMock(side_effect=RuntimeError("network error"))
        agent = OrgResearchAgent(llm=_make_llm(), search=failing)
        profile = await agent.research(ticker="AAPL", org_name="Apple")
        assert isinstance(profile, OrgProfile)
        assert profile.research_confidence == 0.0

    @pytest.mark.asyncio
    async def test_four_queries_issued(self):
        search = _make_search()
        agent = OrgResearchAgent(llm=_make_llm(), search=search)
        await agent.research(ticker="AAPL")
        assert search.search.call_count == 4

    @pytest.mark.asyncio
    async def test_ticker_preserved_in_profile(self):
        agent = OrgResearchAgent(llm=_make_llm(), search=_make_search())
        profile = await agent.research(ticker="MSFT")
        assert profile.ticker == "MSFT"
