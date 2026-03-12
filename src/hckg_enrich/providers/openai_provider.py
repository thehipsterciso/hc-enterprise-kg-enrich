"""OpenAI implementation of LLMProvider."""
from __future__ import annotations

import json
import os
from typing import Any, cast

from pydantic import BaseModel

from hckg_enrich.providers.base import Message


class OpenAIProvider:
    """LLMProvider backed by OpenAI (GPT-4o / o3)."""

    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # noqa: PGH003
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIProvider. "
                "Install with: pip install hc-enterprise-kg-enrich[openai]"
            ) from e
        self._client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, messages: list[Message], system: str = "") -> str:
        oai_messages: list[dict[str, Any]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend({"role": m.role, "content": m.content} for m in messages)

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=cast(Any, oai_messages),
        )
        raw = response.choices[0].message.content
        if raw is None:
            raise ValueError("OpenAI returned empty content")
        return str(raw)

    async def complete_structured(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        system: str = "",
    ) -> BaseModel:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        structured_system = (
            f"{system}\n\nRespond with valid JSON matching this schema:\n{schema_json}"
            if system
            else f"Respond with valid JSON matching this schema:\n{schema_json}"
        )
        oai_messages: list[dict[str, Any]] = [
            {"role": "system", "content": structured_system}
        ]
        oai_messages.extend({"role": m.role, "content": m.content} for m in messages)

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=cast(Any, oai_messages),
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned empty content")
        return schema.model_validate_json(content)
