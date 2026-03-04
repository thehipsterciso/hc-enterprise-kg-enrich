# ADR-001: hc-enterprise-kg-enrich as a Separate Package

**Status:** Accepted
**Date:** 2026-03-04

## Decision

Enrichment is implemented as a separate package (`hc-enterprise-kg-enrich`) rather than a module inside `hc-enterprise-kg`.

## Rationale

The core engine solves graph mechanics (storage, schema, querying). Enrichment solves domain semantics — a fundamentally different, harder problem with different failure modes, evaluation criteria, and dependency profiles.

| Concern | Core engine | Enrichment |
|---|---|---|
| Problem | Graph mechanics | Domain semantics |
| Failure mode | Schema violations | Plausible-but-wrong data |
| Evaluation | Binary correctness | Semantic coherence |
| Deps | networkx, pydantic | LLMs, search APIs, embeddings |
| Velocity | Stable | Rapid iteration |

The enrichment package depends on the engine's public API. The engine has no knowledge of enrichment.

## Consequences

- `hc-enterprise-kg` remains stable and minimal
- `hc-enterprise-kg-enrich` iterates rapidly on domain semantics
- The coupling surface is the engine's public `graph.json` format — a stable, documented contract
