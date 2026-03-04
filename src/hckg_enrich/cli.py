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

    # --- run ---
    run_p = sub.add_parser("run", help="Enrich a graph.json")
    run_p.add_argument("--graph", required=True, help="Path to input graph.json")
    run_p.add_argument("--out", required=True, help="Output path for enriched graph.json")
    run_p.add_argument("--entity-type", dest="entity_type",
                       help="Only enrich entities of this type")
    run_p.add_argument("--limit", type=int, help="Max entities to enrich")
    run_p.add_argument("--concurrency", type=int, default=5)
    run_p.add_argument("--no-search", dest="no_search", action="store_true",
                       help="Disable web search grounding")

    # --- demo ---
    demo_p = sub.add_parser("demo", help="Generate a synthetic enterprise digital twin")
    demo_p.add_argument("--out", required=True, help="Output path for generated graph.json")
    demo_p.add_argument("--size", choices=["small", "medium", "large"], default="medium",
                        help="Size profile (default: medium)")
    demo_p.add_argument(
        "--industry",
        default="financial services",
        help="Industry vertical (default: financial services)",
    )
    demo_p.add_argument("--no-search", dest="no_search", action="store_true",
                        help="Disable web search grounding")

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(_run(args))
    elif args.command == "demo":
        asyncio.run(_demo(args))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_llm_and_search(
    no_search: bool,
) -> tuple[Any, Any]:
    from hckg_enrich.providers.anthropic import AnthropicProvider
    llm = AnthropicProvider()
    search = None
    if not no_search:
        try:
            from hckg_enrich.providers.search.tavily import TavilyProvider
            search = TavilyProvider()
            print("Search grounding: Tavily ✓")
        except (ImportError, KeyError):
            print(
                "Search grounding: disabled "
                "(install hc-enterprise-kg-enrich[tavily] and set TAVILY_API_KEY)"
            )
    return llm, search


def _try_rich_progress(total: int) -> Any:
    """Return a rich Progress context manager if rich is installed, else None."""
    try:
        from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ), total
    except ImportError:
        return None, total


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

async def _run(args: Any) -> None:
    from hckg_enrich.pipeline.controller import EnrichmentController, ProgressEvent

    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found", file=sys.stderr)
        sys.exit(1)

    graph: dict[str, Any] = json.loads(graph_path.read_text())
    llm, search = _make_llm_and_search(getattr(args, "no_search", False))

    controller = EnrichmentController(
        graph=graph,
        llm=llm,
        search=search,
        concurrency=getattr(args, "concurrency", 5),
    )

    progress_ctx, _ = _try_rich_progress(len(graph.get("entities", [])))
    stats = None

    if progress_ctx is not None:
        from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as prog:
            task_id = prog.add_task(
                "Enriching", total=len(graph.get("entities", []))
            )
            async for event in controller.enrich_all_streaming(
                entity_type=getattr(args, "entity_type", None),
                limit=getattr(args, "limit", None),
            ):
                if isinstance(event, ProgressEvent) and event.type == "entity_done":
                    prog.advance(task_id)
                elif isinstance(event, ProgressEvent) and event.type == "completed":
                    stats = event.stats
    else:
        n_entities = len(graph.get("entities", []))
        print(f"Enriching {n_entities} entities...")
        stats = await controller.enrich_all(
            entity_type=getattr(args, "entity_type", None),
            limit=getattr(args, "limit", None),
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(graph, indent=2) + "\n")

    if stats:
        print("\nDone:")
        print(f"  Enriched:              {stats.enriched}")
        print(f"  Relationships added:   {stats.relationships_added}")
        print(f"  Blocked (GraphGuard):  {stats.blocked}")
        print(f"  Skipped:               {stats.skipped}")
        print(f"  Errors:                {stats.errors}")
    print(f"  Output: {out_path}")


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

async def _demo(args: Any) -> None:
    from hckg_enrich.synthetic.twin_generator import TwinGenerator

    llm, search = _make_llm_and_search(getattr(args, "no_search", False))

    print(
        f"Generating synthetic {args.industry} enterprise "
        f"({args.size} size profile)..."
    )
    generator = TwinGenerator(
        llm=llm,
        search=search,
        size=args.size,
        industry=args.industry,
    )
    graph = await generator.generate()

    out_path = Path(args.out)
    out_path.write_text(json.dumps(graph, indent=2) + "\n")

    entities = graph.get("entities", [])
    rels = graph.get("relationships", [])
    print(f"\nGenerated: {len(entities)} entities, {len(rels)} relationships")
    print(f"Output:    {out_path}")


if __name__ == "__main__":
    main()
