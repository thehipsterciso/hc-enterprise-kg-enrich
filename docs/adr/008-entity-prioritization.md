# ADR-008: Entity Prioritization Before Pipeline Execution

**Status:** Accepted
**Date:** 2026-03-10
**Deciders:** Pipeline engineering
**Relates to:** ADR-006 (Observability), ADR-007 (Confidence Tiers)

---

## Context

The v0.2.0 pipeline processed entities in the order they appeared in the graph JSON file.
This order is arbitrary — it reflects insertion order during graph construction, not
enrichment value.

For runs against large graphs (1,000+ entities) with rate-limited LLM APIs and token budgets,
the order entities are processed determines which enrichments happen if the run is interrupted.

Processing a `customer` entity (low enrichment ROI) before a `system` entity (high enrichment ROI
— dependency mapping, blast radius, attack path analysis) means an interrupted run produces
lower value than a prioritized run would.

---

## Decision

Introduce `PrioritizationAgent` as a run-level pre-processing step that executes once
before the per-entity pipeline loop.

The controller calls `PrioritizationAgent.run()` once, passing the full entity list.
It returns the list ordered by enrichment priority score.

### Scoring Model (Additive, Max ~1.00)

| Component | Max | Signal |
|-----------|-----|--------|
| Entity type weight | 0.30 | Critical types (system, data_asset) score highest |
| Missing field ratio | 0.25 | More high-value fields missing = higher urgency |
| Connectivity | 0.25 | More graph edges = higher blast radius of enrichment |
| Staleness | 0.20 | Never-enriched > recently-enriched |

#### Entity Type Priority Weights

| Type | Weight | Rationale |
|------|--------|-----------|
| system | 0.30 | Critical for blast radius, dependency, attack path |
| data_asset | 0.28 | Governance, compliance, lineage analysis |
| integration | 0.27 | Dependency topology |
| vendor | 0.25 | Third-party risk |
| person | 0.20 | Key person dependency analysis |
| department | 0.18 | Org structure |
| initiative | 0.10 | Lower analytical leverage |

#### High-Value Fields Tracked

```python
{"description", "owner", "responsible_team", "criticality",
 "data_classification", "risk_tier", "tech_stack", "vendor_name",
 "budget", "headcount", "framework", "status"}
```

### Connectivity Scoring

Entities with more relationships propagate enrichment context to more neighbors.
Score = `min(degree, 20) / 20 × 0.25`. Capped at 20 relationships to prevent
hub entities from dominating the score entirely.

---

## Agent Design Rationale

`PrioritizationAgent` is:
- **Synchronous** (despite being `async`) — no LLM call, no I/O
- **Sub-millisecond** for graphs up to 10,000 entities
- **Purely structural** — uses only graph signals, no domain knowledge
- **Traceable** — `priority_scores` in the payload shows every score breakdown

It is NOT part of the per-entity pipeline chain. It runs once at the controller level
before `asyncio.create_task()` launches per-entity workers.

---

## Consequences

**Positive:**
- High-ROI entities enriched first — partial runs deliver maximum value
- Score breakdown is logged and auditable (`priority_scores` in payload)
- Type weights are configurable by subclassing or monkeypatching `ENTITY_TYPE_WEIGHTS`
- Staleness tracking incentivizes enrichment of new entities before revisiting old ones

**Negative:**
- Priority is computed before any enrichment occurs — a system with many relationships
  but all fields populated will still score high (connectivity dominates)
- Scoring is additive and uncalibrated against actual enrichment outcomes
- Does not account for inter-entity dependencies (enriching A first may make B easier to enrich)

**Rejected alternatives:**
- **LLM-based prioritization:** LLM cannot see the full graph in one call; no ROI
- **Random ordering:** Status quo; lowest value for interrupted runs
- **Domain-specific heuristics:** Too rigid; generic structural signals generalize better

---

## Re-evaluation triggers

- User feedback shows certain entity types consistently produce low-quality enrichments
  → adjust type weights accordingly
- Graph connectivity analysis shows hub entities should be deprioritized
  (to enrich leaves first and propagate context upward) → invert connectivity score
- Add `last_enriched_at` staleness decay → entities enrich in time-decay order
