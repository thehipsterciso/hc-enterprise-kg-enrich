"""Tests for TavilyProvider (mocked — no real API calls)."""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_fake_tavily(async_client: AsyncMock) -> types.ModuleType:
    mod = types.ModuleType("tavily")
    mod.AsyncTavilyClient = MagicMock(return_value=async_client)  # type: ignore[attr-defined]
    return mod


@pytest.fixture()
def mock_tavily_client() -> AsyncMock:
    client = AsyncMock()
    client.search = AsyncMock(
        return_value={
            "results": [
                {
                    "title": "Result One",
                    "url": "https://example.com/1",
                    "content": "Some snippet",
                    "score": 0.9,
                },
                {
                    "title": "Result Two",
                    "url": "https://example.com/2",
                    "content": "Another snippet",
                    "score": 0.8,
                },
            ]
        }
    )
    return client


@pytest.fixture()
def provider(mock_tavily_client: AsyncMock):
    fake_tavily = _make_fake_tavily(mock_tavily_client)
    with patch.dict(sys.modules, {"tavily": fake_tavily}):
        import importlib

        import hckg_enrich.providers.search.tavily as mod
        importlib.reload(mod)
        prov = mod.TavilyProvider(api_key="test-key")
    sys.modules.pop("hckg_enrich.providers.search.tavily", None)
    return prov, mock_tavily_client


@pytest.mark.asyncio
async def test_search_returns_results(provider):
    prov, client = provider
    results = await prov.search("enterprise ERP ownership", n=2)
    assert len(results) == 2
    assert results[0].title == "Result One"
    assert results[0].url == "https://example.com/1"
    assert results[0].snippet == "Some snippet"
    assert results[0].score == 0.9
    assert results[1].title == "Result Two"


@pytest.mark.asyncio
async def test_import_error_without_tavily():
    with patch.dict(sys.modules, {"tavily": None}):  # type: ignore[dict-item]
        import importlib

        import hckg_enrich.providers.search.tavily as mod
        importlib.reload(mod)
        with pytest.raises(ImportError, match="tavily-python"):
            mod.TavilyProvider(api_key="x")
    sys.modules.pop("hckg_enrich.providers.search.tavily", None)
