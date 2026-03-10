# ADR-013: Convergence Loop

**Status:** Accepted  
**Date:** 2026-03-10  
**Deciders:** hc-enterprise-kg-enrich maintainers

## Context

The previous pipeline was a single-pass system. It enriched up to `--limit N` entities and stopped. There was no concept of "the KG is now comprehensive enough". Users had to manually re-run the pipeline multiple times and had no signal for when to stop.

## Decision

Introduce `ConvergenceController` — a new orchestration layer that wraps `EnrichmentController` in an iterative loop that runs until the KG meets a configurable completeness threshold.

### Convergence Algorithm

```
org_profile = OrgResearchAgent.research(ticker, org_name)   # once

for iteration in 1..max_iterations:
    report = KGCompletenessScorer.score(graph, org_profile)

    if report.overall_score >= target_coverage:
        STOP — threshold met

    if iteration > 1 and (score - prev_score) < delta_threshold:
        STOP — plateau, no material progress

    gap_report = GapAnalysisAgent.analyze(report, org_profile)

    if gap_report.entity_types_to_create:
        EntityDiscoveryAgent.discover(gap_report, org_profile)

    EnrichmentController.enrich_all(
        entity_ids=gap_report.entity_ids_to_enrich,
        org_profile=org_profile,
    )

    prev_score = report.overall_score

return ConvergenceResult(iterations, org_profile, iteration_reports, final_report)
```

### Stopping Conditions (in priority order)

1. **Threshold met** — `overall_score >= target_coverage` (default 0.80)
2. **Plateau** — improvement < `delta_threshold` (default 0.01 = 1%) for one pass
3. **Safety guard** — `max_iterations` reached (default 10)

### ConvergenceResult

```python
@dataclass
class ConvergenceResult:
    iterations: int
    converged: bool               # True if threshold met (not plateau or max_iter)
    org_profile: OrgProfile
    iteration_reports: list[CompletenessReport]
    final_report: CompletenessReport
    total_entities_enriched: int
    total_entities_discovered: int
    total_relationships_added: int
```

### CLI Integration

```bash
hckg-enrich run \
  --graph graph.json \
  --out enriched.json \
  --ticker AAPL \
  --target-coverage 0.80 \
  --max-iterations 10 \
  --artifacts-dir ./artifacts \
  --audit-log ./audit/run.jsonl
```

`ConvergenceController` is only activated when `--ticker` or `--org-name` is provided. Without these flags, the existing `EnrichmentController` single-pass behaviour is unchanged (backwards-compatible).

### Per-Iteration Audit Logging

Each iteration emits `CONVERGENCE_ITERATION_COMPLETE` audit events recording:
- Iteration number
- Score before and after
- Entities enriched in this pass
- Entities discovered in this pass
- Gap items addressed

### Cost Model

Each iteration costs approximately:
- 1 × GapAnalysisAgent LLM call (once per iteration)
- 1 × EntityDiscoveryAgent call (only when new entity types needed)
- N × entity enrichment LLM calls (N = prioritised entities for this pass)

A typical 10,000-entity graph converging to 80% in 4 iterations costs ~4,000 LLM calls total — equivalent to running the old `enrich_all` 4 times manually.

## Consequences

**Positive:**
- Fully autonomous enrichment: give it a ticker, it runs until done
- Plateau detection prevents wasted compute when the graph cannot improve further
- `ConvergenceResult` provides complete visibility into every iteration

**Negative:**
- Non-deterministic total cost — depends on initial graph quality and convergence speed
- `max_iterations=10` is a safe default but may need tuning for very sparse graphs

**Neutral:**
- Existing `EnrichmentController` is unchanged; no breaking changes for single-pass users
