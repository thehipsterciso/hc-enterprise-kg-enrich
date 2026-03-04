"""Retrieve relevant KG subgraph context for enrichment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EntitySummary:
    entity_id: str
    name: str
    entity_type: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipSummary:
    relationship_type: str
    source_id: str
    source_name: str
    target_id: str
    target_name: str


@dataclass
class GraphContext:
    focal_entity: EntitySummary
    neighbors: list[EntitySummary] = field(default_factory=list)
    relationships: list[RelationshipSummary] = field(default_factory=list)
    similar_entities: list[EntitySummary] = field(default_factory=list)


class KGContextRetriever:
    """Retrieves focused subgraph context for a given entity.

    Works with any dict-based graph representation loaded from graph.json
    without hard-coding engine internals.
    """

    def __init__(self, graph: dict[str, Any]) -> None:
        self._entities: dict[str, dict[str, Any]] = {
            e["id"]: e for e in graph.get("entities", [])
        }
        self._relationships: list[dict[str, Any]] = graph.get("relationships", [])

    def get_context(self, entity_id: str, depth: int = 1) -> GraphContext:  # noqa: ARG002
        entity = self._entities.get(entity_id)
        if entity is None:
            raise KeyError(f"Entity {entity_id!r} not found in graph")

        focal = self._to_summary(entity)

        rels = [
            r for r in self._relationships
            if r.get("source_id") == entity_id or r.get("target_id") == entity_id
        ]

        neighbor_ids: set[str] = set()
        for r in rels:
            if r.get("source_id") == entity_id:
                neighbor_ids.add(r["target_id"])
            else:
                neighbor_ids.add(r["source_id"])

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

        similar = [
            self._to_summary(e)
            for eid, e in self._entities.items()
            if eid != entity_id and e.get("entity_type") == entity.get("entity_type")
        ][:10]

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
