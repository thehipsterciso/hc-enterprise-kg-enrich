# ADR-006: Observability-First Pipeline Design

**Status:** Accepted
**Date:** 2026-03-10
**Deciders:** Platform engineering
**Relates to:** ADR-005 (Provenance), ADR-009 (Parallel Contracts)

---

## Context

The v0.2.0 pipeline had no metrics, no tracing, and no structured logging beyond
`logger.info(f"Committed enrichment for {entity_id}")`. This made it impossible to:

- Measure per-agent latency (which stage is the bottleneck?)
- Track LLM/search API call volumes and failure rates
- Monitor confidence tier distribution across a run
- Correlate a slow enrichment run with a specific entity or agent
- Export pipeline telemetry to existing enterprise observability stacks

For an enterprise pipeline running against graphs with thousands of entities,
observability is a first-class operational requirement.

---

## Decision

Build a zero-dependency observability layer with two components:

### Component 1: EnrichmentMetrics

Thread-safe, in-process metrics registry with Prometheus-compatible export.
No external dependencies (no `prometheus_client`, no `opentelemetry-sdk`).

Key metric families:

| Metric | Type | Labels |
|--------|------|--------|
| `hckg_enrich_entities_total` | Counter | — |
| `hckg_enrich_entities_enriched_total` | Counter | — |
| `hckg_enrich_entities_blocked_total` | Counter | — |
| `hckg_enrich_entities_errored_total` | Counter | — |
| `hckg_enrich_agent_duration_seconds` | Histogram | `agent` |
| `hckg_enrich_pipeline_duration_seconds` | Histogram | — |
| `hckg_enrich_llm_calls_total` | Counter | `provider`, `model`, `status` |
| `hckg_enrich_guard_evaluations_total` | Counter | `contract`, `result` |
| `hckg_enrich_confidence_tier_total` | Counter | `tier` |
| `hckg_enrich_active_pipelines` | Gauge | — |

Export: `metrics.to_prometheus()` → Prometheus text format; `metrics.to_dict()` → JSON.

### Component 2: EnrichmentTracer

OpenTelemetry-compatible span model with zero external dependencies.
No OTLP exporter, no Jaeger client — just structured span data that can be
exported as OTLP JSON for ingestion by any collector.

Each enrichment run creates a trace. Each agent stage creates a child span.
Spans record: start/end time, status (OK/ERROR), attributes, and events.

```python
async with tracer.span("reasoning_agent", trace_id=run_id, entity_id=eid) as ctx:
    ctx.set_attribute("llm.model", model)
    result = await self._reasoning_agent.run(msg)
    ctx.add_event("proposal_generated", {"fields": len(attrs)})
```

Export: `tracer.to_otlp_json(trace_id)` → OTLP ResourceSpans JSON.

---

## Zero-Dependency Rationale

Adding `opentelemetry-sdk` or `prometheus_client` to core dependencies would:
- Add ~15MB of transitive dependencies
- Require operators to configure exporters before any value is delivered
- Create version conflicts in environments with existing OTel installations

The internal implementations are minimal and produce standards-compatible output.
Operators who want native OTel instrumentation can wrap the tracer in a thin adapter.

---

## Controller Integration

`EnrichmentController` creates one `EnrichmentMetrics` and one `EnrichmentTracer`
instance per controller lifecycle. They are exposed as `.metrics` and `.tracer` properties.

The controller wraps each agent stage in a tracer span and records duration via metrics:

```python
async with self._tracer.span("context_agent", trace_id=run_id) as ctx:
    t = time.monotonic()
    msg = await self._context_agent.run(msg)
    self._metrics.record_agent_duration("context", time.monotonic() - t)
```

---

## Consequences

**Positive:**
- Zero new production dependencies
- Prometheus-native output integrates with existing enterprise monitoring stacks
- OTLP-compatible traces work with Jaeger, Zipkin, AWS X-Ray, Datadog
- Agent-level latency is now measurable and actionable
- Confidence tier distribution visible per run

**Negative:**
- Internal implementations are not full OTel SDK replacements; some advanced features
  (baggage, exemplars, sampling) are not supported
- Metrics are in-process only — no push/pull endpoint by default; operators must
  integrate the export call into their own HTTP handler

**Rejected alternatives:**
- **opentelemetry-sdk as core dependency:** Too heavy; creates version conflicts
- **Logging-only approach:** Not queryable; no time-series analysis possible
- **Prometheus-client library:** Adds dependency; global singleton antipattern

---

## Re-evaluation triggers

- Enterprise deployments require OTel Collector push → wrap tracer with OTLP HTTP exporter
- Prometheus scrape endpoint needed → add optional Flask/FastAPI metrics route
- Sampling required for high-volume runs → implement head-based sampling in the tracer
