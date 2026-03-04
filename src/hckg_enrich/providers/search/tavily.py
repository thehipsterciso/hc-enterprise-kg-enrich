"""Tavily web search implementation of SearchProvider."""
from __future__ import annotations

import os

from hckg_enrich.providers.base import SearchResult


class TavilyProvider:
    """SearchProvider backed by Tavily."""

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from tavily import AsyncTavilyClient  # noqa: PGH003
        except ImportError as e:
            raise ImportError(
                "tavily-python is required for TavilyProvider. "
                "Install with: pip install hc-enterprise-kg-enrich[tavily]"
            ) from e
        self._client = AsyncTavilyClient(
            api_key=api_key or os.environ["TAVILY_API_KEY"]
        )

    async def search(self, query: str, n: int = 5) -> list[SearchResult]:
        response = await self._client.search(query, max_results=n)
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                score=r.get("score", 1.0),
            )
            for r in response.get("results", [])
        ]
