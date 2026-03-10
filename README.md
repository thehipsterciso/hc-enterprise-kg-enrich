# hc-enterprise-kg-enrich

AI-powered enrichment pipeline for [`hc-enterprise-kg`](https://github.com/thehipsterciso/hc-enterprise-kg).

## Why

The core engine stores and queries enterprise knowledge graphs. Enrichment is a separate, harder problem: given a partial graph, intelligently infer missing entities, relationships, and attributes using:

1. **KG context** — retrieve the relevant subgraph to ground decisions in what's already known
2. **Web search** — ground domain semantics in industry conventions (who typically owns ERP? how does a financial services org structure its data function?)
3. **LLM reasoning** — synthesise KG context + web search into coherent, semantically correct enrichments
4. **Confidence scoring** — deterministic T1–T4 tier assignment based on evidence quality
5. **GraphGuard validation** — 9 semantic contracts ensure enrichments don't violate structural coherence rules
6. **Full provenance** — every enrichment committed with structured audit trail and run tracking

## Architecture

### 7-Agent Pipeline (v0.5.0)

```
EnrichmentController
├── PrioritizationAgent → Ranks entities by enrichment value (once per batch)
├── ContextAgent        → KG subgraph retrieval (graph traversal + embeddings)
├── SearchAgent         → Web search for domain grounding (Tavily)
├── ReasoningAgent      → LLM: entity + context + search → typed proposals
├── ConfidenceAgent     → Deterministic T1–T4 tier assignment from evidence signals
├── CoherenceAgent      → GraphGuard: 9 semantic contracts run in parallel
└── CommitAgent         → Apply validated enrichments with full provenance
```

### GraphGuard Contracts (9)

| Contract | Severity | Mechanism | Validates |
|---|---|---|---|
| `OrgHierarchyContract` | ERROR | LLM | Org reporting line coherence |
| `SystemOwnershipContract` | ERROR | LLM | System-to-department ownership |
| `VendorRelationshipContract` | ERROR | LLM | Vendor relationship plausibility |
| `CircularDependencyContract` | ERROR | Rule-based (DFS) | Cycles in dependency chains |
| `EntityDeduplicationContract` | WARNING | Rule-based (Jaccard) | Near-duplicate entity proposals |
| `DataAssetOwnershipContract` | ERROR | LLM | PII/regulated data ownership |
| `PersonRoleConsistencyContract` | WARNING | LLM | C-suite vs IC role alignment |
| `RelationshipTypeSemanticsContract` | ERROR | Schema + LLM hybrid | Valid domain/range for rel types |
| `PlausibilityContract` | WARNING | Rule-based (bounds) | Numeric field values within empirical ranges |

All contracts run in parallel via `asyncio.gather`. All fail-closed (GG-006 security policy).

### Confidence Tiers

| Tier | Label | Score Range | Basis |
|---|---|---|---|
| T1 | Verified Fact | ≥ 0.94 | Multiple corroborating sources, grounded language |
| T2 | Strong Inference | ≥ 0.80 | High source count, specific attributes |
| T3 | Reasoned Inference | ≥ 0.65 | Limited sources, baseline specificity |
| T4 | Working Hypothesis | < 0.65 | Single source, hedging language, generic values |

### Provenance & Audit Trail

Every enrichment writes structured provenance to the entity:

```json
{
  "provenance": {
    "enriched_by": "hckg-enrich/v0.5.0",
    "enriched_at": "2026-03-10T12:34:56.789Z",
    "run_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "llm_model": "claude-opus-4-5",
    "confidence_tier": "T2",
    "confidence_score": 0.82,
    "source_count": 3
  }
}
```

The audit log captures every pipeline event as append-only JSONL:

```bash
hckg-enrich run --graph graph.json --out enriched.json --audit-log audit/run.jsonl
```

### Observability

Zero-dependency Prometheus-compatible metrics and OpenTelemetry-compatible spans:

```bash
hckg-enrich run --graph graph.json --out enriched.json --metrics metrics.txt
```

Metrics include: entity results by status, agent duration histograms, LLM calls by provider/model, confidence tier distribution, active pipeline gauge.

### File Safety & KGAdapter

The `hckg_enrich.io` module provides crash-safe graph persistence and a bridge to `hc-enterprise-kg`:

```python
from hckg_enrich.io import atomic_write_json, GraphFileLock, KGAdapter

# Atomic write with backup rotation (crash-safe)
atomic_write_json(Path("enriched.json"), graph_dict)

# Manual exclusive lock
with GraphFileLock(Path("graph.json"), exclusive=True):
    # safe to write

# Bridge to hc-enterprise-kg KnowledgeGraph facade
from graph.knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph()
adapter = KGAdapter(kg)
graph_dict = adapter.to_dict()            # export for enrichment
adapter.apply_enrichments(graph_dict)     # write results back
```

## Install

```bash
pip install hc-enterprise-kg-enrich            # core (Anthropic Claude)
pip install hc-enterprise-kg-enrich[tavily]    # + Tavily web search
pip install hc-enterprise-kg-enrich[full]      # everything
```

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export TAVILY_API_KEY=tvly-...          # optional

# Enrich a graph
hckg-enrich run --graph graph.json --out enriched.json

# With audit log and metrics
hckg-enrich run \
  --graph graph.json \
  --out enriched.json \
  --audit-log audit/run.jsonl \
  --metrics metrics.txt

# Enrich only departments, max 10, with 8 concurrent pipelines
hckg-enrich run \
  --graph graph.json \
  --out enriched.json \
  --entity-type department \
  --limit 10 \
  --concurrency 8

# Generate a synthetic enterprise graph
hckg-enrich demo --out demo.json --size medium --industry "healthcare"
```

## CLI Reference

### `run` — Enrich a graph

| Flag | Default | Description |
|---|---|---|
| `--graph` | required | Path to input `graph.json` |
| `--out` | required | Output path for enriched graph |
| `--entity-type` | all types | Only enrich entities of this type |
| `--limit` | unlimited | Max entities to enrich |
| `--concurrency` | 5 | Parallel entity pipelines |
| `--no-search` | false | Disable Tavily web search |
| `--audit-log` | none | Write JSONL audit log (e.g. `audit/run.jsonl`) |
| `--metrics` | none | Write Prometheus metrics text after run |

### `demo` — Generate synthetic graph

| Flag | Default | Description |
|---|---|---|
| `--out` | required | Output path for generated graph |
| `--size` | medium | `small`, `medium`, or `large` |
| `--industry` | financial services | Industry vertical |
| `--no-search` | false | Disable web search grounding |

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `TAVILY_API_KEY` | If using Tavily | Tavily search API key |

## Python API

```python
from hckg_enrich.pipeline.controller import EnrichmentController
from hckg_enrich.providers.anthropic import AnthropicProvider

llm = AnthropicProvider()

controller = EnrichmentController(
    graph=graph,
    llm=llm,
    concurrency=5,
    audit_log_path="audit/run.jsonl",
)

# Enrich all entities — returns EnrichmentRun
run = await controller.enrich_all()
print(f"Run ID: {run.run_id}")
print(f"Enriched: {run.enriched_count}, Blocked: {run.blocked_count}")

# Enrich with streaming progress
async for event in controller.enrich_all_streaming():
    if event.type == "entity_done":
        print(f"Done: {event.entity_id} ({event.completed}/{event.total})")

# Access metrics after run
print(controller.metrics.to_prometheus())

# Add custom GraphGuard contracts
from hckg_enrich.guard.contracts.circular_dependency import CircularDependencyContract
controller = EnrichmentController(
    graph=graph,
    llm=llm,
    extra_contracts=[CircularDependencyContract()],
)
```

## Architecture Decision Records

| ADR | Decision |
|---|---|
| [001](docs/adr/001-provider-abstraction.md) | Provider abstraction layer |
| [002](docs/adr/002-graphguard-fail-closed.md) | GraphGuard fail-closed policy |
| [003](docs/adr/003-streaming-pipeline.md) | Streaming pipeline architecture |
| [004](docs/adr/004-graphguard-contracts.md) | GraphGuard contract model |
| [005](docs/adr/005-provenance-audit-trail.md) | Provenance & audit trail |
| [006](docs/adr/006-observability-first.md) | Zero-dependency observability |
| [007](docs/adr/007-confidence-tiers.md) | T1–T4 confidence tier system |
| [008](docs/adr/008-entity-prioritization.md) | Entity prioritization scoring |
| [009](docs/adr/009-parallel-contract-execution.md) | Parallel contract execution + GG-006 |

## Development

```bash
poetry install --extras full
poetry run pytest
poetry run ruff check src/ tests/
poetry run mypy src/
```

Tests: 182+ across unit (agents, guard contracts, io, observability, provenance, providers) and integration (full 7-agent pipeline).

## Breaking changes

### v0.5.0

- Added `PlausibilityContract` (GG-PLAUS-001, WARNING) — 9th GraphGuard contract. Rule-based numeric bounds validation. No breaking API change; existing `EnrichmentController` picks it up automatically via the default contract set in `CoherenceAgent`.
- `atomic_write_json()` now writes graph output files with exclusive locking and backup rotation. Output files gain a `.1`/`.2`/`.3` backup on each overwrite.
- CLI `run` and `demo` now emit a stderr warning if `schema_version` in the input graph is not `"1.x"`.
- `KGAdapter` added to `hckg_enrich.io` for integration with `hc-enterprise-kg >= 0.32.0`.

### v0.4.0

- `EnrichmentController.enrich_all()` now returns `EnrichmentRun` (was `EnrichmentStats`). Field names: `enriched_count`, `blocked_count`, `skipped_count`, `error_count`, `relationships_added` (was `enriched`, `blocked`, `skipped`, `errors`).
- GraphGuard contracts now fail-closed on LLM parse errors (GG-006). Previously failed-open.
- Entity provenance `enriched_by` is now `"hckg-enrich/v0.4.0"` (was `"hckg-enrich/reasoning-agent"`).
