"""MCP tool definitions for hc-enterprise-kg-enrich.

All read-only tools carry  ``readOnlyHint=True``.
All write/destructive tools carry ``destructiveHint=True``.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from mcp.types import ToolAnnotations

from hckg_enrich.mcp_server.state import (
    NoGraphError,
    load_graph,
    persist_graph,
    require_graph,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Annotation singletons
# ---------------------------------------------------------------------------

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)
_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=True)
_SAFE_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_tools(mcp: FastMCP) -> None:  # noqa: C901
    """Register all MCP tools on the given FastMCP instance."""

    # -----------------------------------------------------------------------
    # Graph management
    # -----------------------------------------------------------------------

    @mcp.tool(annotations=_SAFE_WRITE)
    def load_graph_tool(path: str) -> dict[str, Any]:
        """Load a graph.json file into the enrichment server.

        Must be called before any enrichment or inspection tools.

        Args:
            path: Absolute or relative path to a graph.json file
                  (hc-enterprise-kg format: entities + relationships keys).

        Returns:
            Summary: entity_count, relationship_count, entity_types loaded,
            and the resolved file path.
        """
        try:
            return load_graph(path)
        except FileNotFoundError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Failed to load graph: {exc}"}

    @mcp.tool(annotations=_READ_ONLY)
    def get_statistics() -> dict[str, Any]:
        """Return statistics about the currently loaded graph.

        Returns:
            entity_count, relationship_count, entity_types breakdown,
            and relationship_types breakdown.
        """
        try:
            graph = require_graph()
        except NoGraphError as exc:
            return {"error": str(exc)}

        entities: list[dict[str, Any]] = graph.get("entities", [])
        rels: list[dict[str, Any]] = graph.get("relationships", [])

        type_counts: dict[str, int] = {}
        for e in entities:
            t = e.get("entity_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        rel_counts: dict[str, int] = {}
        for r in rels:
            rt = r.get("relationship_type", "unknown")
            rel_counts[rt] = rel_counts.get(rt, 0) + 1

        return {
            "entity_count": len(entities),
            "relationship_count": len(rels),
            "entity_types": type_counts,
            "relationship_types": rel_counts,
        }

    @mcp.tool(annotations=_READ_ONLY)
    def list_entities(entity_type: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """List entities in the loaded graph, optionally filtered by type.

        Args:
            entity_type: Entity type filter (e.g. "system", "department",
                "person", "vendor"). Empty string returns all types.
            limit: Maximum number of results (default 50).

        Returns:
            List of compact entity dicts: id, name, entity_type, description.
        """
        try:
            graph = require_graph()
        except NoGraphError as exc:
            return [{"error": str(exc)}]

        entities: list[dict[str, Any]] = graph.get("entities", [])
        if entity_type:
            entities = [e for e in entities if e.get("entity_type") == entity_type]

        return [
            {k: v for k, v in e.items() if k in {"id", "name", "entity_type", "description"}}
            for e in entities[:limit]
        ]

    @mcp.tool(annotations=_READ_ONLY)
    def get_entity(entity_id: str) -> dict[str, Any]:
        """Return full details for a single entity.

        Args:
            entity_id: UUID of the entity.

        Returns:
            Full entity dict, or an error if not found.
        """
        try:
            graph = require_graph()
        except NoGraphError as exc:
            return {"error": str(exc)}

        entities: list[dict[str, Any]] = graph.get("entities", [])
        for e in entities:
            if e.get("id") == entity_id:
                return e
        return {"error": f"Entity '{entity_id}' not found."}

    @mcp.tool(annotations=_READ_ONLY)
    def get_entity_relationships(
        entity_id: str,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Return relationships connected to an entity.

        Args:
            entity_id: UUID of the entity.
            direction: "in", "out", or "both" (default "both").

        Returns:
            List of relationship dicts matching the direction filter.
        """
        try:
            graph = require_graph()
        except NoGraphError as exc:
            return [{"error": str(exc)}]

        rels: list[dict[str, Any]] = graph.get("relationships", [])
        result = []
        for r in rels:
            is_source = r.get("source_id") == entity_id
            is_target = r.get("target_id") == entity_id
            if direction == "out" and is_source:
                result.append(r)
            elif direction == "in" and is_target:
                result.append(r)
            elif direction == "both" and (is_source or is_target):
                result.append(r)
        return result

    # -----------------------------------------------------------------------
    # Enrichment
    # -----------------------------------------------------------------------

    @mcp.tool(annotations=_WRITE)
    def enrich_entity(
        entity_id: str,
        no_search: bool = False,
    ) -> dict[str, Any]:
        """Run the full enrichment pipeline on a single entity.

        Calls the five-agent pipeline (Context → Search → Reasoning →
        Coherence → Commit) for the specified entity and applies any
        validated attribute and relationship changes to the in-memory graph.

        Args:
            entity_id: UUID of the entity to enrich.
            no_search: If True, skip web search grounding (faster, less accurate).

        Returns:
            Enrichment result: enriched (bool), relationships_added, attributes_updated,
            blocked (bool), error (str if failed).
        """
        try:
            graph = require_graph()
        except NoGraphError as exc:
            return {"error": str(exc)}

        entities: list[dict[str, Any]] = graph.get("entities", [])
        entity = next((e for e in entities if e.get("id") == entity_id), None)
        if entity is None:
            return {"error": f"Entity '{entity_id}' not found."}

        return asyncio.get_event_loop().run_until_complete(
            _enrich_entity_async(graph, entity_id, no_search)
        )

    @mcp.tool(annotations=_WRITE)
    def enrich_all(
        entity_type: str = "",
        limit: int = 0,
        concurrency: int = 3,
        no_search: bool = False,
    ) -> dict[str, Any]:
        """Run the enrichment pipeline on all (or filtered) entities.

        Args:
            entity_type: If set, only enrich entities of this type.
            limit: Maximum number of entities to enrich (0 = all).
            concurrency: Number of parallel enrichment workers (default 3).
            no_search: If True, skip web search grounding.

        Returns:
            Stats: total_entities, enriched, relationships_added, blocked,
            skipped, errors.
        """
        try:
            graph = require_graph()
        except NoGraphError as exc:
            return {"error": str(exc)}

        return asyncio.get_event_loop().run_until_complete(
            _enrich_all_async(
                graph,
                entity_type=entity_type or None,
                limit=limit or None,
                concurrency=concurrency,
                no_search=no_search,
            )
        )

    @mcp.tool(annotations=_WRITE)
    def generate_twin(
        industry: str = "financial services",
        size: str = "medium",
        no_search: bool = False,
    ) -> dict[str, Any]:
        """Generate a synthetic enterprise digital twin using LLM reasoning.

        Produces a realistic enterprise knowledge graph with departments,
        systems, vendors, data assets, and people — ready for enrichment
        or as a test fixture.

        Args:
            industry: Industry vertical (e.g. "financial services",
                "healthcare", "manufacturing"). Default: "financial services".
            size: Organisation size profile — "small", "medium", or "large".
                Default: "medium".
            no_search: If True, skip web search grounding for the generation.

        Returns:
            Generated graph summary: entity_count, relationship_count,
            company_name, industry, size.
        """
        global _generated_graph  # noqa: PLW0603
        return asyncio.get_event_loop().run_until_complete(
            _generate_twin_async(industry, size, no_search)
        )

    @mcp.tool(annotations=_SAFE_WRITE)
    def save_graph(out_path: str = "") -> dict[str, Any]:
        """Persist the in-memory graph to disk.

        Args:
            out_path: Destination file path. If empty, overwrites the
                originally loaded file.

        Returns:
            saved_to (str) — absolute path of the written file.
        """
        try:
            dest = persist_graph(out_path or None)
            return {"saved_to": str(dest)}
        except NoGraphError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Failed to save: {exc}"}


# ---------------------------------------------------------------------------
# Async helpers (run inside event loop from sync tool handlers)
# ---------------------------------------------------------------------------

async def _enrich_entity_async(
    graph: dict[str, Any],
    entity_id: str,
    no_search: bool,
) -> dict[str, Any]:
    from hckg_enrich.pipeline.controller import EnrichmentController
    from hckg_enrich.providers.anthropic import AnthropicProvider

    # Build a single-entity view; all relationships stay so context is intact.
    entity = next(
        (e for e in graph.get("entities", []) if e.get("id") == entity_id), None
    )
    if entity is None:
        return {"error": f"Entity '{entity_id}' not found."}

    entity_type: str = entity.get("entity_type", "")
    llm = AnthropicProvider()
    search = _make_search(no_search)
    ctrl = EnrichmentController(graph=graph, llm=llm, search=search, concurrency=1)
    # Use entity_type + limit=1 if there is exactly 1 entity of that type,
    # otherwise fall back to limit=1 without type filter and rely on ordering.
    same_type = [e for e in graph.get("entities", []) if e.get("entity_type") == entity_type]
    if len(same_type) == 1:
        stats = await ctrl.enrich_all(entity_type=entity_type, limit=1)
    else:
        # Move target entity to front so limit=1 processes it.
        others = [e for e in graph["entities"] if e.get("id") != entity_id]
        graph["entities"] = [entity] + others
        stats = await ctrl.enrich_all(limit=1)
        graph["entities"] = others + [entity]  # restore order
    return {
        "enriched": stats.enriched > 0,
        "relationships_added": stats.relationships_added,
        "blocked": stats.blocked > 0,
        "errors": stats.errors,
    }


async def _enrich_all_async(
    graph: dict[str, Any],
    entity_type: str | None,
    limit: int | None,
    concurrency: int,
    no_search: bool,
) -> dict[str, Any]:
    from hckg_enrich.pipeline.controller import EnrichmentController
    from hckg_enrich.providers.anthropic import AnthropicProvider

    llm = AnthropicProvider()
    search = _make_search(no_search)
    ctrl = EnrichmentController(
        graph=graph, llm=llm, search=search, concurrency=concurrency
    )
    stats = await ctrl.enrich_all(entity_type=entity_type, limit=limit)
    return {
        "total_entities": stats.total_entities,
        "enriched": stats.enriched,
        "relationships_added": stats.relationships_added,
        "blocked": stats.blocked,
        "skipped": stats.skipped,
        "errors": stats.errors,
    }


async def _generate_twin_async(
    industry: str,
    size: str,
    no_search: bool,
) -> dict[str, Any]:
    from hckg_enrich.mcp_server import state
    from hckg_enrich.providers.anthropic import AnthropicProvider
    from hckg_enrich.synthetic.twin_generator import TwinGenerator

    llm = AnthropicProvider()
    search = _make_search(no_search)
    gen = TwinGenerator(llm=llm, search=search, industry=industry, size=size)
    graph = await gen.generate()

    # Load into server state so subsequent tools work
    state._graph = graph
    state._graph_path = None

    entities = graph.get("entities", [])
    rels = graph.get("relationships", [])
    meta = graph.get("metadata", {})

    return {
        "company_name": meta.get("company_name", ""),
        "industry": meta.get("industry", industry),
        "size": meta.get("size", size),
        "entity_count": len(entities),
        "relationship_count": len(rels),
    }


def _make_search(no_search: bool) -> Any:
    if no_search:
        return None
    try:
        from hckg_enrich.providers.search.tavily import TavilyProvider
        return TavilyProvider()
    except (ImportError, KeyError):
        return None
