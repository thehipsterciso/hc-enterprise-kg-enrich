"""MCP server for hc-enterprise-kg-enrich — Claude Desktop integration.

Exposes the AI enrichment pipeline as MCP tools so Claude can:
  • Inspect and enrich knowledge-graph entities via LLM reasoning
  • Run web-search-grounded enrichment at scale
  • Generate synthetic enterprise digital twins

Entry point:
    hckg-enrich-mcp          (registered by pyproject.toml script)
    python -m hckg_enrich.mcp_server.server

Environment:
    HCKG_DEFAULT_GRAPH   — absolute path to graph.json; auto-loaded on startup
    ANTHROPIC_API_KEY    — required for LLM enrichment
    TAVILY_API_KEY       — optional; enables web search grounding
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from hckg_enrich.mcp_server.state import auto_load_default_graph
from hckg_enrich.mcp_server.tools import register_tools

mcp = FastMCP(
    "hc-enterprise-kg-enrich",
    instructions=(
        "AI-powered knowledge-graph enrichment. "
        "Call load_graph_tool first, then use enrich_entity or enrich_all "
        "to enrich entities with LLM reasoning and web search grounding. "
        "Use generate_twin to create a synthetic enterprise graph for testing."
    ),
)

register_tools(mcp)


def main() -> None:
    """Run the MCP server over stdio transport (used by Claude Desktop)."""
    auto_load_default_graph()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
