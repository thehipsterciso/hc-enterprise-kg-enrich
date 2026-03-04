"""Tests for OpenAIProvider (mocked — no real API calls)."""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hckg_enrich.providers.base import Message


class _Schema(BaseModel):
    value: str


def _make_fake_openai(client: AsyncMock) -> types.ModuleType:
    mod = types.ModuleType("openai")
    mod.AsyncOpenAI = MagicMock(return_value=client)  # type: ignore[attr-defined]
    return mod


@pytest.fixture()
def mock_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def provider(mock_client: AsyncMock):
    fake_openai = _make_fake_openai(mock_client)
    with patch.dict(sys.modules, {"openai": fake_openai}):
        import importlib

        import hckg_enrich.providers.openai_provider as mod
        importlib.reload(mod)
        prov = mod.OpenAIProvider(api_key="test-key")
    sys.modules.pop("hckg_enrich.providers.openai_provider", None)
    return prov, mock_client


@pytest.mark.asyncio
async def test_complete_returns_text(provider):
    prov, client = provider
    choice = MagicMock()
    choice.message.content = "hello world"
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[choice])
    )
    result = await prov.complete([Message(role="user", content="ping")])
    assert result == "hello world"


@pytest.mark.asyncio
async def test_complete_structured_parses_json(provider):
    prov, client = provider
    choice = MagicMock()
    choice.message.content = '{"value": "parsed"}'
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[choice])
    )
    result = await prov.complete_structured(
        [Message(role="user", content="give me json")],
        schema=_Schema,
    )
    assert isinstance(result, _Schema)
    assert result.value == "parsed"


@pytest.mark.asyncio
async def test_complete_raises_on_empty_content(provider):
    prov, client = provider
    choice = MagicMock()
    choice.message.content = None
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[choice])
    )
    with pytest.raises(ValueError, match="empty content"):
        await prov.complete([Message(role="user", content="ping")])


def test_import_error_without_openai():
    with patch.dict(sys.modules, {"openai": None}):  # type: ignore[dict-item]
        import importlib

        import hckg_enrich.providers.openai_provider as mod
        importlib.reload(mod)
        with pytest.raises(ImportError, match="openai"):
            mod.OpenAIProvider(api_key="x")
    sys.modules.pop("hckg_enrich.providers.openai_provider", None)
