# ADR-002: Pluggable LLM Provider

**Status:** Accepted
**Date:** 2026-03-04

## Decision

LLM access is abstracted behind a `LLMProvider` protocol. The default implementation uses Anthropic Claude. Alternative providers (OpenAI, local models) can be plugged in without modifying enrichment logic.

## Interface

```python
class LLMProvider(Protocol):
    async def complete(self, messages: list[Message], system: str = "") -> str: ...
    async def complete_structured(self, messages: list[Message], schema: type[BaseModel], system: str = "") -> BaseModel: ...
```

## Rationale

Enrichment logic must be decoupled from LLM vendor. Structured output (Pydantic models) is a first-class requirement — the reasoning agent produces typed proposals, not free text. Using Python's `Protocol` (structural subtyping) means providers need not inherit from a base class.
