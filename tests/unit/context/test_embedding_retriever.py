"""Tests for EmbeddingContextRetriever."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hckg_enrich.context.embedding_retriever import EmbeddingContextRetriever, _cosine_similarity


@pytest.fixture()
def small_graph() -> dict:
    return {
        "entities": [
            {"id": "e1", "entity_type": "system", "name": "SAP ERP",
             "description": "Enterprise resource planning"},
            {"id": "e2", "entity_type": "system", "name": "Workday",
             "description": "HR management system"},
            {"id": "e3", "entity_type": "department", "name": "Finance",
             "description": "Financial operations"},
        ],
        "relationships": [
            {"id": "r1", "relationship_type": "owned_by",
             "source_id": "e1", "target_id": "e3"},
        ],
    }


@pytest.fixture()
def mock_ep():
    ep = AsyncMock()
    # return distinct unit vectors for e1, e2, e3
    ep.embed = AsyncMock(return_value=[
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [0.0, 1.0, 0.0],
    ])
    return ep


def test_cosine_similarity_identical():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


@pytest.mark.asyncio
async def test_build_index_stores_embeddings(small_graph, mock_ep):
    retriever = EmbeddingContextRetriever(small_graph, mock_ep)
    await retriever.build_index()
    assert len(retriever._index) == 3


@pytest.mark.asyncio
async def test_get_context_uses_embedding_similarity(small_graph, mock_ep):
    retriever = EmbeddingContextRetriever(small_graph, mock_ep, top_k=2)
    await retriever.build_index()
    ctx = retriever.get_context("e1")

    assert ctx.focal_entity.entity_id == "e1"
    # e2 (cosine ~0.8) should rank higher than e3 (cosine 0.0)
    assert ctx.similar_entities[0].entity_id == "e2"


def test_get_context_falls_back_without_index(small_graph, mock_ep):
    retriever = EmbeddingContextRetriever(small_graph, mock_ep)
    # no build_index — uses same-type fallback
    ctx = retriever.get_context("e1")
    assert ctx.focal_entity.entity_id == "e1"
    # fallback: similar = same entity_type (system)
    similar_ids = {e.entity_id for e in ctx.similar_entities}
    assert "e2" in similar_ids
    assert "e3" not in similar_ids


def test_get_context_includes_structural_relationships(small_graph, mock_ep):
    retriever = EmbeddingContextRetriever(small_graph, mock_ep)
    ctx = retriever.get_context("e1")
    assert len(ctx.relationships) == 1
    assert ctx.relationships[0].relationship_type == "owned_by"


def test_get_context_raises_for_missing_entity(small_graph, mock_ep):
    retriever = EmbeddingContextRetriever(small_graph, mock_ep)
    with pytest.raises(KeyError):
        retriever.get_context("does-not-exist")
