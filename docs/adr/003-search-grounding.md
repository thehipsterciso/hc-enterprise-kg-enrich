# ADR-003: Web Search for Domain Grounding

**Status:** Accepted
**Date:** 2026-03-04

## Decision

A `SearchProvider` abstraction allows the enrichment pipeline to query the web for domain context. Default implementation uses Tavily. The search layer is optional — enrichment degrades gracefully without it.

## Rationale

Static rules cannot capture the full range of org hierarchy conventions, system ownership patterns, and vendor relationships. Web search provides dynamic grounding:

- "Who typically owns Workday in a financial services firm?"
- "What business unit is SAP S/4HANA usually governed by?"
- "Industry standard: does a CISO report to CIO or CEO?"

This grounds LLM reasoning in current industry practice rather than training data alone.

## Graceful degradation

If no `SearchProvider` is configured, `SearchAgent` returns an empty search context and the pipeline continues. This means enrichment is possible without a search API key, just less well-grounded.
