"""Shared mutable server state: loaded graph, controller, provider instances."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_graph: dict[str, Any] | None = None
_graph_path: Path | None = None


class NoGraphError(RuntimeError):
    """Raised when an operation requires a loaded graph but none is loaded."""


def load_graph(path: str) -> dict[str, Any]:
    """Load *path* into the server state. Returns summary statistics."""
    global _graph, _graph_path

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Graph file not found: {p}")

    raw: dict[str, Any] = json.loads(p.read_text())
    _graph = raw
    _graph_path = p

    entities: list[Any] = raw.get("entities", [])
    rels: list[Any] = raw.get("relationships", [])

    type_counts: dict[str, int] = {}
    for e in entities:
        t = e.get("entity_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "loaded": str(p),
        "entity_count": len(entities),
        "relationship_count": len(rels),
        "entity_types": type_counts,
    }


def require_graph() -> dict[str, Any]:
    if _graph is None:
        raise NoGraphError("No graph loaded. Call load_graph first.")
    return _graph


def require_graph_path() -> Path:
    if _graph_path is None:
        raise NoGraphError("No graph loaded. Call load_graph first.")
    return _graph_path


def persist_graph(out_path: str | None = None) -> Path:
    """Write the in-memory graph to *out_path* (or the original load path)."""
    graph = require_graph()
    dest = Path(out_path).expanduser().resolve() if out_path else require_graph_path()
    dest.write_text(json.dumps(graph, indent=2) + "\n")
    return dest


def auto_load_default_graph() -> None:
    """Load HCKG_DEFAULT_GRAPH env var on startup if set."""
    env = os.environ.get("HCKG_DEFAULT_GRAPH")
    if env:
        try:
            load_graph(env)
        except Exception:
            pass  # startup — don't crash if file missing yet
