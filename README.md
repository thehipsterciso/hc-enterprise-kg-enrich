# hc-enterprise-kg-enrich

AI-powered enrichment pipeline for [`hc-enterprise-kg`](https://github.com/thehipsterciso/hc-enterprise-kg).

## Why

The core engine stores and queries enterprise knowledge graphs. Enrichment is a separate, harder problem: given a partial graph, intelligently infer missing entities, relationships, and attributes using:

1. **KG context** — retrieve the relevant subgraph to ground decisions in what's already known
2. **Web search** — ground domain semantics in industry conventions (who typically owns ERP? how does a financial services org structure its data function?)
3. **LLM reasoning** — synthesise KG context + web search into coherent, semantically correct enrichments
4. **GraphGuard validation** — ensure proposed enrichments don't violate org hierarchy, system ownership, or vendor relationship coherence contracts

## Architecture

```
EnrichmentController
├── ContextAgent        → KG subgraph retrieval (graph traversal, v0.2 → RAG)
├── SearchAgent         → Web search for domain grounding (Tavily)
├── ReasoningAgent      → LLM: entity + context + search → typed proposals
├── CoherenceAgent      → GraphGuard semantic contract validation
└── CommitAgent         → Apply validated enrichments to graph
```

See `docs/adr/` for architectural decision records.

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

hckg-enrich run --graph graph.json --out enriched.json
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `TAVILY_API_KEY` | If using Tavily | Tavily search API key |

## Development

```bash
poetry install --extras full
poetry run pytest
poetry run ruff check src/ tests/
poetry run mypy src/
```

## Roadmap

See [v0.2.0 milestone](https://github.com/thehipsterciso/hc-enterprise-kg-enrich/milestone/1) for next priorities:

- AI-driven synthetic digital twin generator (replaces `hckg demo`)
- Embedding-based KG context retrieval (RAG over graph)
- Streaming pipeline with progress reporting
- OpenAI provider
- Relationship proposal commit (currently captured, not written)
