# ADR-012: Entity Discovery

**Status:** Accepted  
**Date:** 2026-03-10  
**Deciders:** hc-enterprise-kg-enrich maintainers

## Context

The previous pipeline could only enrich **existing** entities — it filled fields on nodes already in the graph. If a KG was missing entire entity type layers (e.g., no `control` entities, no `location` entities), the pipeline had no way to address this gap.

Real enterprise KGs are incomplete not just at the field level but at the structural level: missing roles, missing systems, missing vendors, missing risks. The `GapAnalysisAgent` can identify these gaps, but without a mechanism to create new nodes, the gap cannot be closed by enrichment alone.

## Decision

Introduce `EntityDiscoveryAgent` — a new pipeline agent that creates sparse entity stubs for missing or underpopulated entity type layers.

### How It Works

1. `GapAnalysisAgent` produces a `GapReport` with `entity_types_to_create` (list of entity types that are missing or underpopulated)
2. `EntityDiscoveryAgent` receives this list and the `OrgProfile`
3. For each entity type to create:
   - Issues 2 targeted web searches: `"{org_name} {entity_type} list examples enterprise"`
   - LLM extracts a list of realistic named instances from search results
   - Creates sparse entity stubs with identity fields only (name, type, description)
4. Each discovered entity is stamped with full **discovery provenance**:

```python
{
  "id": str(uuid4()),
  "entity_type": entity_type,
  "name": discovered_name,
  "description": discovered_description,
  "provenance": {
    "discovered_at": now,
    "discovered_by": "hckg-enrich/v0.6.0",
    "discovery_method": "entity_discovery_agent",
    "run_id": run_id,
    "source_urls": [...],   # actual URLs from search
    "source_count": N,
    "confidence_tier": "T3",  # discovered, not yet deeply enriched
  }
}
```

5. Discovered entities are added to `graph["entities"]` as stubs, making them candidates for the next enrichment pass in the convergence loop

### CommitAgent Extension

`CommitAgent` gains an entity creation path alongside its existing update path. When `payload["new_entities"]` is present, each entity in the list is appended to the graph with a `ENTITY_CREATED` audit event.

### Trust Model

Discovered entities start at **T3 confidence** ("Reasoned Inference — limited evidence"). After a full enrichment pass with web search grounding, they can be promoted to T2 or T1. The `provenance.discovery_method` field permanently records how an entity entered the graph.

## Consequences

**Positive:**
- Closes structural gaps, not just field-level gaps
- Every discovered entity has source URLs — users can verify the entity's existence
- T3 starting confidence is honest about the evidence state

**Negative:**
- LLM may hallucinate entity names; mitigated by requiring URL-backed search results and starting at T3
- Discovery can overpopulate the graph if benchmarks are set too aggressively; `GapReport` limits to entities where coverage is genuinely below threshold

**Neutral:**
- Discovery only runs when `GapReport.entity_types_to_create` is non-empty — no impact on existing single-pass enrichment
