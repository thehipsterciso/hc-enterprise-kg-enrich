"""Tests for OpenAIEmbeddingProvider (mocked — no real API calls)."""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_fake_openai(client: AsyncMock) -> types.ModuleType:
    mod = types.ModuleType("openai")
    mod.AsyncOpenAI = MagicMock(return_value=client)  # type: ignore[attr-defined]
    return mod


def _make_embedding_response(vectors: list[list[float]]) -> MagicMock:
    """Build a mock that mimics openai EmbeddingsResponse."""
    items = []
    for vec in vectors:
        item = MagicMock()
        item.embedding = vec
        items.append(item)
    response = MagicMock()
    response.data = items
    return response


@pytest.fixture()
def mock_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def provider(mock_client: AsyncMock):
    fake_openai = _make_fake_openai(mock_client)
    with patch.dict(sys.modules, {"openai": fake_openai}):
        import importlib

        import hckg_enrich.providers.embedding as mod
        importlib.reload(mod)
        prov = mod.OpenAIEmbeddingProvider(api_key="test-key", batch_size=3)
    sys.modules.pop("hckg_enrich.providers.embedding", None)
    return prov, mock_client


@pytest.mark.asyncio
async def test_embed_returns_vectors(provider):
    prov, client = provider
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    client.embeddings.create = AsyncMock(
        return_value=_make_embedding_response(vectors)
    )
    result = await prov.embed(["text one", "text two"])
    assert result == vectors
    client.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_batches_large_inputs(provider):
    prov, client = provider
    # batch_size=3, 7 texts → 3 batches: [3, 3, 1]
    texts = [f"text {i}" for i in range(7)]
    batch_vectors = {
        0: [[float(i)] for i in range(3)],
        1: [[float(i + 3)] for i in range(3)],
        2: [[6.0]],
    }
    call_count = 0

    async def fake_create(model, input):  # noqa: A002
        nonlocal call_count
        resp = _make_embedding_response(batch_vectors[call_count])
        call_count += 1
        return resp

    client.embeddings.create = fake_create
    result = await prov.embed(texts)
    assert len(result) == 7
    assert call_count == 3
