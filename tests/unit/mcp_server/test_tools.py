"""Tests for mcp_server.tools — registered tool behaviour (no real LLM calls)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import hckg_enrich.mcp_server.state as state_mod
from hckg_enrich.mcp_server.tools import (
    _READ_ONLY,
    _SAFE_WRITE,
    _WRITE,
    register_tools,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeMCP:
    """Minimal stand-in for FastMCP — captures registered tools."""

    def __init__(self) -> None:
        self._tools: dict[str, object] = {}

    def tool(self, *, annotations=None, name=None):
        def decorator(fn):
            self._tools[name or fn.__name__] = fn
            return fn
        return decorator


@pytest.fixture(autouse=True)
def reset_state():
    state_mod._graph = None
    state_mod._graph_path = None
    yield
    state_mod._graph = None
    state_mod._graph_path = None


@pytest.fixture()
def fake_mcp() -> _FakeMCP:
    mcp = _FakeMCP()
    register_tools(mcp)  # type: ignore[arg-type]
    return mcp


def _write_graph(path: Path, entities=None, rels=None) -> None:
    data = {"entities": entities or [], "relationships": rels or []}
    path.write_text(json.dumps(data))


def _load_test_graph(entities=None, rels=None):
    state_mod._graph = {"entities": entities or [], "relationships": rels or []}
    state_mod._graph_path = None


# ---------------------------------------------------------------------------
# load_graph_tool
# ---------------------------------------------------------------------------

def test_load_graph_tool_success(fake_mcp: _FakeMCP):
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "graph.json"
        _write_graph(p, [{"id": "e1", "entity_type": "system", "name": "SAP"}])
        result = fake_mcp._tools["load_graph_tool"](str(p))
        assert result["entity_count"] == 1


def test_load_graph_tool_missing_file(fake_mcp: _FakeMCP):
    result = fake_mcp._tools["load_graph_tool"]("/no/such/file.json")
    assert "error" in result


# ---------------------------------------------------------------------------
# get_statistics
# ---------------------------------------------------------------------------

def test_get_statistics_no_graph(fake_mcp: _FakeMCP):
    result = fake_mcp._tools["get_statistics"]()
    assert "error" in result


def test_get_statistics_returns_counts(fake_mcp: _FakeMCP):
    _load_test_graph(
        entities=[
            {"id": "e1", "entity_type": "system", "name": "SAP"},
            {"id": "e2", "entity_type": "department", "name": "Finance"},
        ],
        rels=[{"id": "r1", "relationship_type": "owns", "source_id": "e2", "target_id": "e1"}],
    )
    result = fake_mcp._tools["get_statistics"]()
    assert result["entity_count"] == 2
    assert result["relationship_count"] == 1
    assert result["entity_types"]["system"] == 1
    assert result["entity_types"]["department"] == 1


# ---------------------------------------------------------------------------
# list_entities
# ---------------------------------------------------------------------------

def test_list_entities_no_graph(fake_mcp: _FakeMCP):
    result = fake_mcp._tools["list_entities"]()
    assert any("error" in r for r in result)


def test_list_entities_all(fake_mcp: _FakeMCP):
    _load_test_graph(
        entities=[
            {"id": "e1", "entity_type": "system", "name": "SAP"},
            {"id": "e2", "entity_type": "department", "name": "Finance"},
        ]
    )
    result = fake_mcp._tools["list_entities"]()
    assert len(result) == 2


def test_list_entities_filtered_by_type(fake_mcp: _FakeMCP):
    _load_test_graph(
        entities=[
            {"id": "e1", "entity_type": "system", "name": "SAP"},
            {"id": "e2", "entity_type": "department", "name": "Finance"},
        ]
    )
    result = fake_mcp._tools["list_entities"](entity_type="system")
    assert len(result) == 1
    assert result[0]["name"] == "SAP"


def test_list_entities_respects_limit(fake_mcp: _FakeMCP):
    _load_test_graph(
        entities=[{"id": f"e{i}", "entity_type": "system", "name": f"S{i}"} for i in range(10)]
    )
    result = fake_mcp._tools["list_entities"](limit=3)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# get_entity
# ---------------------------------------------------------------------------

def test_get_entity_found(fake_mcp: _FakeMCP):
    _load_test_graph(
        entities=[{"id": "e1", "entity_type": "system", "name": "SAP", "description": "ERP"}]
    )
    result = fake_mcp._tools["get_entity"]("e1")
    assert result["name"] == "SAP"


def test_get_entity_not_found(fake_mcp: _FakeMCP):
    _load_test_graph(entities=[])
    result = fake_mcp._tools["get_entity"]("missing-id")
    assert "error" in result


def test_get_entity_no_graph(fake_mcp: _FakeMCP):
    result = fake_mcp._tools["get_entity"]("e1")
    assert "error" in result


# ---------------------------------------------------------------------------
# get_entity_relationships
# ---------------------------------------------------------------------------

def test_get_entity_relationships_both_directions(fake_mcp: _FakeMCP):
    _load_test_graph(
        entities=[
            {"id": "e1", "entity_type": "system", "name": "SAP"},
            {"id": "e2", "entity_type": "department", "name": "Finance"},
        ],
        rels=[
            {"id": "r1", "relationship_type": "owns", "source_id": "e2", "target_id": "e1"},
            {"id": "r2", "relationship_type": "uses", "source_id": "e1", "target_id": "e2"},
        ],
    )
    result = fake_mcp._tools["get_entity_relationships"]("e1", direction="both")
    assert len(result) == 2


def test_get_entity_relationships_out_only(fake_mcp: _FakeMCP):
    _load_test_graph(
        entities=[
            {"id": "e1", "entity_type": "system", "name": "SAP"},
            {"id": "e2", "entity_type": "department", "name": "Finance"},
        ],
        rels=[
            {"id": "r1", "relationship_type": "owns", "source_id": "e2", "target_id": "e1"},
            {"id": "r2", "relationship_type": "uses", "source_id": "e1", "target_id": "e2"},
        ],
    )
    result = fake_mcp._tools["get_entity_relationships"]("e1", direction="out")
    assert len(result) == 1
    assert result[0]["relationship_type"] == "uses"


def test_get_entity_relationships_no_graph(fake_mcp: _FakeMCP):
    result = fake_mcp._tools["get_entity_relationships"]("e1")
    assert any("error" in r for r in result)


# ---------------------------------------------------------------------------
# save_graph
# ---------------------------------------------------------------------------

def test_save_graph_no_graph(fake_mcp: _FakeMCP):
    result = fake_mcp._tools["save_graph"]()
    assert "error" in result


def test_save_graph_writes_to_custom_path(fake_mcp: _FakeMCP):
    _load_test_graph(entities=[{"id": "e1", "entity_type": "system", "name": "SAP"}])
    with tempfile.TemporaryDirectory() as tmpdir:
        out = str(Path(tmpdir) / "out.json")
        result = fake_mcp._tools["save_graph"](out)
        assert "saved_to" in result
        assert Path(result["saved_to"]).exists()


# ---------------------------------------------------------------------------
# Annotation constants
# ---------------------------------------------------------------------------

def test_read_only_annotation():
    assert _READ_ONLY.readOnlyHint is True
    assert _READ_ONLY.destructiveHint is False


def test_write_annotation():
    assert _WRITE.readOnlyHint is False
    assert _WRITE.destructiveHint is True


def test_safe_write_annotation():
    assert _SAFE_WRITE.readOnlyHint is False
    assert _SAFE_WRITE.destructiveHint is False
