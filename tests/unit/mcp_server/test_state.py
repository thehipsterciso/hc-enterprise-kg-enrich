"""Tests for mcp_server.state — graph loading and persistence."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import hckg_enrich.mcp_server.state as state_mod


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module-level singletons before each test."""
    state_mod._graph = None
    state_mod._graph_path = None
    yield
    state_mod._graph = None
    state_mod._graph_path = None


def _write_graph(path: Path, entities: list, rels: list | None = None) -> None:
    data = {"entities": entities, "relationships": rels or []}
    path.write_text(json.dumps(data))


def test_load_graph_returns_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "graph.json"
        _write_graph(p, [{"id": "e1", "entity_type": "system", "name": "SAP"}])
        result = state_mod.load_graph(str(p))
        assert result["entity_count"] == 1
        assert result["relationship_count"] == 0
        assert "system" in result["entity_types"]


def test_load_graph_sets_module_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "graph.json"
        _write_graph(p, [{"id": "e1", "entity_type": "system", "name": "SAP"}])
        state_mod.load_graph(str(p))
        assert state_mod._graph is not None
        assert state_mod._graph_path == p.resolve()


def test_load_graph_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        state_mod.load_graph("/nonexistent/graph.json")


def test_require_graph_raises_when_no_graph():
    from hckg_enrich.mcp_server.state import NoGraphError
    with pytest.raises(NoGraphError):
        state_mod.require_graph()


def test_require_graph_returns_loaded_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "graph.json"
        _write_graph(p, [{"id": "e1", "entity_type": "system", "name": "SAP"}])
        state_mod.load_graph(str(p))
        g = state_mod.require_graph()
        assert len(g["entities"]) == 1


def test_persist_graph_writes_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "graph.json"
        _write_graph(p, [])
        state_mod.load_graph(str(p))
        # Mutate in-memory graph
        state_mod._graph["entities"].append({"id": "e99", "name": "New"})  # type: ignore[index]
        dest = state_mod.persist_graph()
        reloaded = json.loads(dest.read_text())
        assert len(reloaded["entities"]) == 1


def test_persist_graph_to_custom_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "graph.json"
        out = Path(tmpdir) / "out.json"
        _write_graph(p, [])
        state_mod.load_graph(str(p))
        state_mod.persist_graph(str(out))
        assert out.exists()


def test_auto_load_default_graph_with_env(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "graph.json"
        _write_graph(p, [{"id": "e1", "entity_type": "system", "name": "SAP"}])
        monkeypatch.setenv("HCKG_DEFAULT_GRAPH", str(p))
        state_mod.auto_load_default_graph()
        assert state_mod._graph is not None


def test_auto_load_default_graph_missing_env_is_noop(monkeypatch):
    monkeypatch.delenv("HCKG_DEFAULT_GRAPH", raising=False)
    state_mod.auto_load_default_graph()
    assert state_mod._graph is None


def test_auto_load_default_graph_bad_path_is_noop(monkeypatch):
    monkeypatch.setenv("HCKG_DEFAULT_GRAPH", "/definitely/does/not/exist.json")
    state_mod.auto_load_default_graph()  # must not raise
    assert state_mod._graph is None
