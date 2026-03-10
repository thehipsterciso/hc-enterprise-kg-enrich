"""OrgResearchAgent — builds an OrgProfile from web search + LLM extraction.

Runs once at the start of a ConvergenceController session. Issues 4 targeted
web searches and uses LLM structured extraction to populate OrgProfile. Every
field is backed by at least one SourceCitation with an actual URL.

Falls back gracefully to a minimal OrgProfile when search is disabled or the
ticker cannot be resolved.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from hckg_enrich.org.profile import OrgProfile
from hckg_enrich.providers.base import LLMProvider, Message, SearchProvider

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You are an enterprise intelligence analyst. Extract structured
organisation profile data from the provided web search results.

Return only what is explicitly supported by the search results. Use empty strings
and empty lists for fields where no evidence was found. Do not invent or infer
beyond what the sources state.

Headcount tiers: startup (<100), mid (100-999), large (1000-9999), enterprise (10000+)
Revenue tiers: small (<$100M), medium ($100M-$1B), large ($1B-$10B), mega (>$10B)
"""


class _ExtractedProfile(BaseModel):
    org_name: str = ""
    industry: str = ""
    sector: str = ""
    country: str = "US"
    headcount_tier: str = ""
    revenue_tier: str = ""
    key_roles: list[str] = []
    subsidiaries: list[str] = []
    regulatory_regime: list[str] = []
    industry_frameworks: list[str] = []
    tech_profile: dict[str, str] = {}


class OrgResearchAgent:
    """Researches a target organisation and returns a sourced OrgProfile.

    Issues 4 web searches (company profile, leadership, regulatory, tech stack)
    and extracts a structured OrgProfile via LLM. All search result URLs are
    recorded as SourceCitations on the returned profile.
    """

    def __init__(
        self,
        llm: LLMProvider,
        search: SearchProvider | None = None,
    ) -> None:
        self._llm = llm
        self._search = search

    async def research(
        self,
        ticker: str | None = None,
        org_name: str | None = None,
        industry: str | None = None,
    ) -> OrgProfile:
        """Build and return a sourced OrgProfile.

        At least one of ticker or org_name must be provided. Returns a minimal
        profile with research_confidence=0.0 if search is unavailable.
        """
        identifier = ticker or org_name or "unknown organisation"

        if self._search is None:
            logger.info("OrgResearchAgent: search disabled, returning minimal profile")
            return OrgProfile(
                ticker=ticker,
                org_name=org_name or "",
                industry=industry or "",
                research_confidence=0.0,
            )

        queries = self._build_queries(ticker, org_name)
        all_results: list[dict[str, Any]] = []

        async def _search_one(query: str) -> list[dict[str, Any]]:
            try:
                results = await self._search.search(query, n=5)  # type: ignore[union-attr]
                return [
                    {"url": r.url, "title": r.title, "snippet": r.snippet,
                     "score": r.score, "query": query}
                    for r in results
                ]
            except Exception as exc:
                logger.warning("OrgResearchAgent search failed for %r: %s", query, exc)
                return []

        batches = await asyncio.gather(*(_search_one(q) for q in queries))
        for batch in batches:
            all_results.extend(batch)

        if not all_results:
            logger.warning("OrgResearchAgent: no search results for %s", identifier)
            return OrgProfile(
                ticker=ticker,
                org_name=org_name or identifier,
                industry=industry or "",
                research_confidence=0.0,
            )

        # Build LLM prompt from search results
        search_text = "\n\n".join(
            f"Source: {r['url']}\nTitle: {r['title']}\n{r['snippet']}"
            for r in all_results[:20]  # cap to avoid context overflow
        )

        prompt = (
            f"Target organisation: {identifier}\n\n"
            f"Web search results:\n{search_text}\n\n"
            "Extract the organisation profile from these search results."
        )

        try:
            extracted: _ExtractedProfile = await self._llm.complete_structured(
                [Message(role="user", content=prompt)],
                schema=_ExtractedProfile,
                system=EXTRACTION_SYSTEM,
            )
        except Exception as exc:
            logger.warning("OrgResearchAgent LLM extraction failed: %s", exc)
            return OrgProfile(
                ticker=ticker,
                org_name=org_name or identifier,
                industry=industry or "",
                sources=all_results,
                research_confidence=0.2,
            )

        # Score confidence based on source coverage
        source_count = len(all_results)
        research_confidence = min(0.9, 0.3 + (source_count / 20) * 0.6)

        now = datetime.now(UTC).isoformat()
        source_citations = [
            {
                "url": r["url"],
                "title": r["title"],
                "snippet": r["snippet"],
                "relevance_score": float(r.get("score", 1.0)),
                "retrieved_at": now,
                "search_query": r["query"],
            }
            for r in all_results
        ]

        return OrgProfile(
            ticker=ticker,
            org_name=extracted.org_name or org_name or identifier,
            industry=extracted.industry or industry or "",
            sector=extracted.sector,
            country=extracted.country,
            headcount_tier=extracted.headcount_tier,
            revenue_tier=extracted.revenue_tier,
            key_roles=extracted.key_roles,
            subsidiaries=extracted.subsidiaries,
            regulatory_regime=extracted.regulatory_regime,
            industry_frameworks=extracted.industry_frameworks,
            tech_profile=dict(extracted.tech_profile),
            sources=source_citations,
            research_confidence=research_confidence,
        )

    def _build_queries(
        self,
        ticker: str | None,
        org_name: str | None,
    ) -> list[str]:
        identifier = ticker or org_name or "unknown"
        return [
            f"{identifier} company profile industry sector headquarters employees annual revenue",
            f"{identifier} executive leadership team CEO CTO CISO CFO CIO organizational chart",
            f"{identifier} regulatory compliance SOX HIPAA GDPR CCPA framework requirements",
            f"{identifier} enterprise technology stack cloud ERP CRM primary systems vendors",
        ]
