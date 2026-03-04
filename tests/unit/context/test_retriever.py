"""Tests for KGContextRetriever."""
from __future__ import annotations

from typing import Any

import pytest

from hckg_enrich.context.retriever import KGContextRetriever


def test_get_context_returns_focal_entity(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    ctx = retriever.get_context("dept-finance-001")
    assert ctx.focal_entity.entity_id == "dept-finance-001"
    assert ctx.focal_entity.name == "Finance"
    assert ctx.focal_entity.entity_type == "department"


def test_get_context_includes_neighbors(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    ctx = retriever.get_context("person-cfo-001")
    neighbor_ids = [n.entity_id for n in ctx.neighbors]
    assert "dept-finance-001" in neighbor_ids


def test_get_context_includes_relationships(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    ctx = retriever.get_context("person-cfo-001")
    rel_types = [r.relationship_type for r in ctx.relationships]
    assert "works_in" in rel_types


def test_get_context_unknown_entity_raises(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    with pytest.raises(KeyError):
        retriever.get_context("nonexistent-id")


def test_get_context_similar_entities(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    ctx = retriever.get_context("dept-finance-001")
    similar_names = [s.name for s in ctx.similar_entities]
    assert "Human Resources" in similar_names


def test_get_context_no_self_in_similar(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    ctx = retriever.get_context("dept-finance-001")
    similar_ids = [s.entity_id for s in ctx.similar_entities]
    assert "dept-finance-001" not in similar_ids
