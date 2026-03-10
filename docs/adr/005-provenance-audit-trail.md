# ADR-005: Provenance and Audit Trail Architecture

**Status:** Accepted
**Date:** 2026-03-10
**Deciders:** Pipeline engineering
**Relates to:** ADR-004 (GraphGuard Contracts), ADR-006 (Observability)

---

## Context

The v0.2.0 enrichment pipeline committed changes to the graph with minimal provenance:
a hardcoded `enriched_by: "hckg-enrich/reasoning-agent"` string and an `enriched_at` timestamp.
This made it impossible to:

- Correlate all enrichments from a single batch run
- Reconstruct what the graph looked like before enrichment
- Audit which LLM model and provider made each decision
- Determine confidence levels of committed data
- Trace which web search sources grounded each proposal
- Support rollback operations

As the platform targets regulated enterprise environments (financial services, healthcare),
the absence of structured provenance is a compliance blocker.

---

## Decision

Implement a three-layer provenance system:

### Layer 1: EnrichmentRun (Session-Level)

`EnrichmentRun` captures the full execution session:

```python
@dataclass
class EnrichmentRun:
    run_id: str          # UUID, unique per session
    started_at: str      # ISO timestamp
    completed_at: str    # ISO timestamp (on completion)
    graph_path: str      # Graph file processed
    llm_provider: str    # "anthropic" | "openai"
    llm_model: str       # e.g. "claude-opus-4-5"
    pipeline_version: str
    # Stats populated by complete()
    total_entities: int
    enriched_count: int
    blocked_count: int
    skipped_count: int
    error_count: int
    relationships_added: int
    config: dict         # Arbitrary config snapshot
```

The `run_id` is injected into every `AgentMessage.payload` at the controller level,
threading session identity through the entire pipeline.

### Layer 2: Entity-Level Provenance (Inline)

Every enriched entity's `provenance` dict is upgraded:

```json
{
  "enriched_at": "2026-03-10T12:00:00+00:00",
  "enriched_by": "hckg-enrich/v0.3.0",
  "run_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "llm_model": "claude-opus-4-5",
  "confidence_tier": "T2",
  "confidence_score": 0.87,
  "source_count": 3
}
```

### Layer 3: AuditLog (Append-Only JSONL)

`AuditLog` appends `AuditEvent` records to a JSONL file for every pipeline event:
entity enrichments, guard blocks, guard warnings, run start/complete, rollbacks.

```
audit/enrichment-2026-03-10.jsonl
```

Each event is a single JSON line, queryable by entity, run, event type, or confidence tier.
Designed for log aggregation pipelines (Splunk, Elastic, CloudWatch).

---

## T1–T4 Confidence Tiers

Every committed enrichment carries an explicit confidence tier (from `ConfidenceAgent`):

| Tier | Score Range | Basis |
|------|-------------|-------|
| T1   | 0.94–1.00   | Multiple high-quality sources + grounding language |
| T2   | 0.80–0.93   | Multiple corroborating sources |
| T3   | 0.65–0.79   | Single source or indirect evidence |
| T4   | 0.50–0.64   | LLM judgment with minimal external grounding |

The tier system is aligned with the hc-enterprise-kg confidence tier taxonomy
used throughout the broader platform.

---

## Consequences

**Positive:**
- Full lineage for every graph change: who (which model), when, with what evidence
- Rollback possible: EntityDiff captures before/after state
- Compliance-ready: JSONL audit trail meets GDPR Article 5 accountability requirements
- Session correlation: any enrichment can be traced back to its run_id
- Trust scores: downstream consumers can filter by confidence tier

**Negative:**
- JSONL file grows unboundedly; operators need log rotation policy
- AuditLog adds a file I/O operation per entity (mitigated by buffered writes)
- `audit_log_path` is optional — if not provided, no durable audit trail exists

**Rejected alternatives:**
- **SQLite backend:** Adds a dependency; JSONL is simpler, portable, and streaming-friendly
- **In-memory only:** No durability; useless for post-run analysis
- **Postgres:** Operational overhead for what is essentially an append-only event stream

---

## Re-evaluation triggers

- Audit volume exceeds 100,000 events/run → consider batched async writes
- Rollback requirements become formal → implement ProvenanceRecord with full EntityDiff persistence
- Compliance audit of T4 data in production → enforce minimum T3 for committed changes
