"""Anthropic Claude implementation of LLMProvider."""
from __future__ import annotations

import json
import os
from typing import Any

import anthropic
from pydantic import BaseModel

from hckg_enrich.providers.base import Message


class AnthropicProvider:
    """LLMProvider backed by Anthropic Claude."""

    DEFAULT_MODEL = "claude-opus-4-5"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, messages: list[Message], system: str = "") -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            kwargs["system"] = system
        response = await self._client.messages.create(**kwargs)
        block = response.content[0]
        if not hasattr(block, "text"):
            raise ValueError(f"Unexpected content block type: {type(block)}")
        return str(block.text)

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
        raw = await self.complete(messages, system=structured_system)
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()
        return schema.model_validate_json(text)
