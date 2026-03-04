"""Embedding-based KG context retrieval (RAG over graph)."""
from __future__ import annotations

import math
from typing import Any

from hckg_enrich.context.retriever import (
    EntitySummary,
    GraphContext,
    KGContextRetriever,
    RelationshipSummary,
)
from hckg_enrich.providers.embedding import EmbeddingProvider


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _entity_text(entity: dict[str, Any]) -> str:
    """Produce a short text representation of an entity for embedding."""
    parts = [
        entity.get("name", ""),
        entity.get("entity_type", ""),
        entity.get("description", ""),
    ]
    return " | ".join(p for p in parts if p)


class EmbeddingContextRetriever:
    """Retrieves semantically similar entities using embedding cosine similarity.

    Falls back to traversal-based retrieval (KGContextRetriever) for the
    structural neighbor/relationship portion; embedding similarity replaces the
    ``similar_entities`` slot with semantically nearest neighbours instead of
    same-type neighbours.

    Call ``await retriever.build_index()`` once before use to pre-compute
    embeddings.  If the index has not been built, the retriever falls back to
    traversal-based similar-entity selection transparently.
    """

    def __init__(
        self,
        graph: dict[str, Any],
        embedding_provider: EmbeddingProvider,
        top_k: int = 10,
    ) -> None:
        self._entities: dict[str, dict[str, Any]] = {
            e["id"]: e for e in graph.get("entities", [])
        }
        self._relationships: list[dict[str, Any]] = graph.get("relationships", [])
        self._ep = embedding_provider
        self._top_k = top_k
        self._index: dict[str, list[float]] = {}  # entity_id → embedding
        self._fallback = KGContextRetriever(graph)

    async def build_index(self) -> None:
        """Pre-compute embeddings for all entities and cache them."""
        ids = list(self._entities.keys())
        texts = [_entity_text(self._entities[eid]) for eid in ids]
        vectors = await self._ep.embed(texts)
        self._index = dict(zip(ids, vectors, strict=False))

    def get_context(self, entity_id: str, depth: int = 1) -> GraphContext:
        """Return GraphContext; uses embedding similarity if index is built."""
        entity = self._entities.get(entity_id)
        if entity is None:
            raise KeyError(f"Entity {entity_id!r} not found in graph")

        focal = self._to_summary(entity)

        # structural portion: direct relationships + immediate neighbors
        rels = [
            r for r in self._relationships
            if r.get("source_id") == entity_id or r.get("target_id") == entity_id
        ]
        neighbor_ids: set[str] = set()
        for r in rels:
            other = r["target_id"] if r.get("source_id") == entity_id else r["source_id"]
            neighbor_ids.add(other)

        neighbors = [
            self._to_summary(self._entities[nid])
            for nid in neighbor_ids
            if nid in self._entities
        ]
        rel_summaries = [
            RelationshipSummary(
                relationship_type=r["relationship_type"],
                source_id=r["source_id"],
                source_name=self._entities.get(r["source_id"], {}).get("name", r["source_id"]),
                target_id=r["target_id"],
                target_name=self._entities.get(r["target_id"], {}).get("name", r["target_id"]),
            )
            for r in rels
        ]

        # similar entities: embedding if index built, else same-type fallback
        if self._index and entity_id in self._index:
            query_vec = self._index[entity_id]
            scored = [
                (eid, _cosine_similarity(query_vec, vec))
                for eid, vec in self._index.items()
                if eid != entity_id
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            similar = [
                self._to_summary(self._entities[eid])
                for eid, _ in scored[: self._top_k]
                if eid in self._entities
            ]
        else:
            similar = [
                self._to_summary(e)
                for eid, e in self._entities.items()
                if eid != entity_id and e.get("entity_type") == entity.get("entity_type")
            ][: self._top_k]

        return GraphContext(
            focal_entity=focal,
            neighbors=neighbors,
            relationships=rel_summaries,
            similar_entities=similar,
        )

    def _to_summary(self, entity: dict[str, Any]) -> EntitySummary:
        return EntitySummary(
            entity_id=entity.get("id", ""),
            name=entity.get("name", ""),
            entity_type=entity.get("entity_type", ""),
            attributes={
                k: v for k, v in entity.items()
                if k not in {"id", "name", "entity_type"}
            },
        )
