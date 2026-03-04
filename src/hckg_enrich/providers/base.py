"""Provider protocol definitions for LLM and search providers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    pass


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    score: float = 1.0


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM completion providers."""

    async def complete(self, messages: list[Message], system: str = "") -> str:
        """Return a text completion for the given messages."""
        ...

    async def complete_structured(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        system: str = "",
    ) -> BaseModel:
        """Return a structured (Pydantic) completion."""
        ...


@runtime_checkable
class SearchProvider(Protocol):
    """Protocol for web search providers."""

    async def search(self, query: str, n: int = 5) -> list[SearchResult]:
        """Return up to n search results for the query."""
        ...
