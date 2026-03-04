"""Tests for ContextBuilder."""
from __future__ import annotations

from typing import Any

from hckg_enrich.context.builder import ContextBuilder
from hckg_enrich.context.retriever import KGContextRetriever


def test_builder_produces_string(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    ctx = retriever.get_context("dept-finance-001")
    builder = ContextBuilder()
    result = builder.build(ctx)
    assert "Finance" in result
    assert "department" in result


def test_builder_includes_relationships(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    ctx = retriever.get_context("person-cfo-001")
    builder = ContextBuilder()
    result = builder.build(ctx)
    assert "works_in" in result


def test_builder_includes_similar_entities(sample_graph: dict[str, Any]) -> None:
    retriever = KGContextRetriever(sample_graph)
    ctx = retriever.get_context("dept-finance-001")
    builder = ContextBuilder()
    result = builder.build(ctx)
    assert "Human Resources" in result
