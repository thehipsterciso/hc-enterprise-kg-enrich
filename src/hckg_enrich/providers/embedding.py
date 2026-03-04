"""Embedding provider protocol and OpenAI implementation."""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for each text."""
        ...


class OpenAIEmbeddingProvider:
    """EmbeddingProvider backed by OpenAI text-embedding models."""

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        batch_size: int = 100,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # noqa: PGH003
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIEmbeddingProvider. "
                "Install with: pip install hc-enterprise-kg-enrich[openai]"
            ) from e
        self._client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            results.extend(item.embedding for item in response.data)
        return results
