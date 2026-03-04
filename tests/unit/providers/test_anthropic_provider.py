"""Tests for AnthropicProvider (mocked — no real API calls)."""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hckg_enrich.providers.base import Message


class _Schema(BaseModel):
    value: str


def _make_fake_anthropic(client: AsyncMock) -> types.ModuleType:
    mod = types.ModuleType("anthropic")
    mod.AsyncAnthropic = MagicMock(return_value=client)  # type: ignore[attr-defined]
    return mod


@pytest.fixture()
def mock_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def provider(mock_client: AsyncMock):
    fake_anthropic = _make_fake_anthropic(mock_client)
    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        import importlib

        import hckg_enrich.providers.anthropic as mod
        importlib.reload(mod)
        prov = mod.AnthropicProvider(api_key="test-key")
    sys.modules.pop("hckg_enrich.providers.anthropic", None)
    return prov, mock_client


@pytest.mark.asyncio
async def test_complete_returns_text(provider):
    prov, client = provider
    block = MagicMock()
    block.text = "hello from claude"
    client.messages.create = AsyncMock(
        return_value=MagicMock(content=[block])
    )
    result = await prov.complete([Message(role="user", content="ping")])
    assert result == "hello from claude"


@pytest.mark.asyncio
async def test_complete_structured_parses_response(provider):
    prov, client = provider
    block = MagicMock()
    block.text = '{"value": "structured"}'
    client.messages.create = AsyncMock(
        return_value=MagicMock(content=[block])
    )
    result = await prov.complete_structured(
        [Message(role="user", content="give me json")],
        schema=_Schema,
    )
    assert isinstance(result, _Schema)
    assert result.value == "structured"


@pytest.mark.asyncio
async def test_complete_raises_on_empty(provider):
    prov, client = provider
    block = MagicMock(spec=[])  # no .text attribute
    client.messages.create = AsyncMock(
        return_value=MagicMock(content=[block])
    )
    with pytest.raises(ValueError, match="Unexpected content block type"):
        await prov.complete([Message(role="user", content="ping")])
