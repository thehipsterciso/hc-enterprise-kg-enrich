# ADR-011: KG Completeness Scoring

**Status:** Accepted  
**Date:** 2026-03-10  
**Deciders:** hc-enterprise-kg-enrich maintainers

## Context

The previous pipeline had no definition of "done". It stopped when `--limit N` entities were processed. Users had no way to know whether their KG was comprehensive or how far it was from industry standards.

An enterprise knowledge graph is typically considered production-ready when it covers all relevant entity type layers, relationships between entities are dense and semantically grounded, individual entities have their key fields populated, and all enrichments have traceable sources.

## Decision

Introduce `KGCompletenessScorer` — a pure-Python (no LLM), deterministic scoring module that evaluates a graph against a 5-dimension completeness model.

### Scoring Dimensions

| # | Dimension | Weight | Benchmark |
|---|-----------|--------|-----------|
| 1 | Layer coverage | 30% | 12 expected entity type layers; score = layers_present / 12 |
| 2 | Field population | 25% | Avg % of HIGH_VALUE_FIELDS non-empty per entity |
| 3 | Relationship density | 20% | edges_per_entity; enterprise benchmark ≥ 2.0 |
| 4 | Provenance quality | 15% | % of entities with ≥ 1 URL-backed source (source_count > 0) |
| 5 | Confidence quality | 10% | % of enriched entities at T1 or T2 tier |

**Provenance quality is a first-class scoring dimension.** A KG populated entirely by T4 LLM guesses with no source URLs cannot achieve a high completeness score, regardless of how many fields are filled.

### The 12 Expected Entity Type Layers

```
person, department, role, system, vendor, risk, control,
data_asset, initiative, location, network, jurisdiction
```

These represent the full entity taxonomy of hc-enterprise-kg's generation model. A missing layer is a gap.

### Industry Adjustment via OrgProfile

When an `OrgProfile` is provided:
- `industry="financial services"` → `control`, `risk`, `jurisdiction` layers are **required** (weight boosted)
- `regulatory_regime=["HIPAA"]` → `data_asset` layer required with data_classification populated
- `headcount_tier="enterprise"` → minimum entity count benchmarks applied per layer

### CompletenessReport

```python
@dataclass
class CompletenessReport:
    overall_score: float           # Weighted composite 0.0–1.0
    layer_coverage: float
    field_population_rate: float
    relationship_density: float
    provenance_quality: float      # % entities with URL-backed sources
    confidence_quality: float
    missing_layers: list[str]
    underpopulated_layers: list[str]
    entities_without_sources: list[str]  # IDs — need re-enrichment with search enabled
    passes_threshold: bool
    total_entities: int
    total_relationships: int
    scored_at: str
```

### Default Convergence Threshold

`target_coverage = 0.80` (80%). This is achievable with 2–4 enrichment passes on a well-seeded graph. The user can override with `--target-coverage`.

## Consequences

**Positive:**
- Objective, reproducible definition of "done" for KG enrichment
- Provenance quality scoring creates a forcing function for URL-backed enrichments
- Industry-adjusted benchmarks produce more relevant gap analysis

**Negative:**
- 12-layer benchmark may not apply to all org types (e.g., a startup may never have `jurisdiction` entities)
- Scoring is deterministic but the benchmark weights are opinionated — they may need tuning per use case

**Neutral:**
- Scorer runs in < 100ms on graphs up to 100k entities (pure Python, no I/O)
