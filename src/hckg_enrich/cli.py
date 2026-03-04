"""CLI entrypoint: hckg-enrich."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="hckg-enrich",
        description="AI-powered enrichment for hc-enterprise-kg knowledge graphs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Enrich a graph.json")
    run_p.add_argument("--graph", required=True, help="Path to input graph.json")
    run_p.add_argument("--out", required=True, help="Output path for enriched graph.json")
    run_p.add_argument("--entity-type", dest="entity_type",
                       help="Only enrich entities of this type")
    run_p.add_argument("--limit", type=int, help="Max entities to enrich")
    run_p.add_argument("--concurrency", type=int, default=5)
    run_p.add_argument("--no-search", dest="no_search", action="store_true",
                       help="Disable web search grounding")

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(_run(args))


async def _run(args: Any) -> None:
    from hckg_enrich.pipeline.controller import EnrichmentController
    from hckg_enrich.providers.anthropic import AnthropicProvider

    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found", file=sys.stderr)
        sys.exit(1)

    graph: dict[str, Any] = json.loads(graph_path.read_text())
    llm = AnthropicProvider()
    search = None

    if not getattr(args, "no_search", False):
        try:
            from hckg_enrich.providers.search.tavily import TavilyProvider
            search = TavilyProvider()
            print("Search grounding: Tavily ✓")
        except (ImportError, KeyError):
            print("Search grounding: disabled "
                  "(install hc-enterprise-kg-enrich[tavily] and set TAVILY_API_KEY)")

    controller = EnrichmentController(
        graph=graph,
        llm=llm,
        search=search,
        concurrency=getattr(args, "concurrency", 5),
    )

    n_entities = len(graph.get("entities", []))
    print(f"Enriching {n_entities} entities...")
    stats = await controller.enrich_all(
        entity_type=getattr(args, "entity_type", None),
        limit=getattr(args, "limit", None),
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(graph, indent=2) + "\n")

    print("\nDone:")
    print(f"  Enriched:  {stats.enriched}")
    print(f"  Blocked:   {stats.blocked} (GraphGuard)")
    print(f"  Skipped:   {stats.skipped}")
    print(f"  Errors:    {stats.errors}")
    print(f"  Output:    {out_path}")


if __name__ == "__main__":
    main()
