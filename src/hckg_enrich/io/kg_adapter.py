"""KnowledgeGraph adapter for hc-enterprise-kg-enrich.

Bridges the hc-enterprise-kg ``KnowledgeGraph`` facade to the dict-based
graph format that the enrichment pipeline operates on.  The adapter is an
optional component — hc-enterprise-kg need not be installed for the enrich
pipeline to function with a raw ``graph.json``.

Usage (with hc-enterprise-kg installed)::

    from graph.knowledge_graph import KnowledgeGraph
    from ingest.json_ingestor import JSONIngestor
    from hckg_enrich.io.kg_adapter import KGAdapter

    kg = KnowledgeGraph()
    result = JSONIngestor().ingest("graph.json")
    kg.add_entities_bulk(result.entities)
    kg.add_relationships_bulk(result.relationships)

    adapter = KGAdapter(kg)
    graph_dict = adapter.to_dict()   # pass to EnrichmentController

    # After enrichment, write results back through the facade:
    adapter.apply_enrichments(graph_dict)

Usage (without hc-enterprise-kg — pass raw dict directly)::

    graph_dict = json.loads(Path("graph.json").read_text())
    controller = EnrichmentController(graph=graph_dict, ...)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Only needed for type checking — not a hard dependency
    pass


class KGAdapter:
    """Adapter from hc-enterprise-kg ``KnowledgeGraph`` to enrich's dict format.

    Parameters
    ----------
    kg:
        A ``KnowledgeGraph`` instance from ``hc-enterprise-kg``.
        Must expose ``get_entities()``, ``get_relationships()``,
        ``update_entity()``, and ``add_relationship()`` or equivalent.
    """

    def __init__(self, kg: Any) -> None:
        self._kg = kg

    def to_dict(self) -> dict[str, Any]:
        """Serialize the KnowledgeGraph to the enrich pipeline's dict format.

        Returns a ``{"entities": [...], "relationships": [...]}`` dict
        that can be passed directly to ``EnrichmentController``.

        The entities and relationships are serialized using Pydantic's
        ``model_dump()`` so all fields (including extra/provenance fields)
        are preserved.
        """
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        # Try the engine export path first (most complete serialization)
        try:
            raw = self._kg.engine.export_dict()
            return {
                "entities": raw.get("entities", []),
                "relationships": raw.get("relationships", []),
            }
        except AttributeError:
            pass

        # Fallback: iterate entity/relationship collections
        try:
            for entity in self._kg.get_entities():
                if hasattr(entity, "model_dump"):
                    entities.append(entity.model_dump())
                elif hasattr(entity, "__dict__"):
                    entities.append(dict(entity.__dict__))
                else:
                    entities.append(entity)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("KGAdapter.to_dict: could not read entities: %s", exc)

        try:
            for rel in self._kg.get_relationships():
                if hasattr(rel, "model_dump"):
                    relationships.append(rel.model_dump())
                elif hasattr(rel, "__dict__"):
                    relationships.append(dict(rel.__dict__))
                else:
                    relationships.append(rel)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("KGAdapter.to_dict: could not read relationships: %s", exc)

        return {"entities": entities, "relationships": relationships}

    def apply_enrichments(self, enriched_graph: dict[str, Any]) -> int:
        """Write enriched entity data back through the KnowledgeGraph facade.

        Iterates the enriched graph dict and calls ``kg.update_entity()``
        for each entity that carries a ``provenance.enriched_by`` field
        (i.e. entities that were actually touched by the pipeline).

        New relationships added by the pipeline are committed via
        ``kg.add_relationship()`` if the relationship ID does not already
        exist in the graph.

        Parameters
        ----------
        enriched_graph:
            The graph dict after enrichment (as returned by the pipeline).

        Returns
        -------
        int
            Number of entities updated in the KG facade.
        """
        updated = 0

        # Update enriched entities
        for entity_dict in enriched_graph.get("entities", []):
            provenance = entity_dict.get("provenance", {})
            if not provenance.get("enriched_by"):
                continue  # untouched by enrich pipeline

            entity_id = entity_dict.get("id")
            if not entity_id:
                continue

            try:
                # Extract only non-identity, non-structural fields to update
                skip = {"id", "entity_type", "name", "created_at"}
                updates = {k: v for k, v in entity_dict.items() if k not in skip}
                self._kg.update_entity(entity_id, **updates)
                updated += 1
            except Exception as exc:
                logger.warning(
                    "KGAdapter.apply_enrichments: failed to update entity %s: %s",
                    entity_id,
                    exc,
                )

        # Add new relationships (those with an enriched_by provenance)
        existing_ids: set[str] = set()
        try:
            existing_ids = {
                r.id if hasattr(r, "id") else r.get("id", "")
                for r in self._kg.get_relationships()
            }
        except Exception:
            pass

        for rel_dict in enriched_graph.get("relationships", []):
            rel_prov = rel_dict.get("provenance", {})
            if not rel_prov.get("enriched_by"):
                continue  # pre-existing relationship

            rel_id = rel_dict.get("id", "")
            if rel_id in existing_ids:
                continue

            try:
                self._kg.add_relationship(
                    source_id=rel_dict["source_id"],
                    target_id=rel_dict["target_id"],
                    relationship_type=rel_dict["relationship_type"],
                    weight=rel_dict.get("weight", 1.0),
                    confidence=rel_dict.get("confidence", 1.0),
                    properties=rel_dict.get("provenance", {}),
                )
            except Exception as exc:
                logger.warning(
                    "KGAdapter.apply_enrichments: failed to add relationship %s: %s",
                    rel_id,
                    exc,
                )

        logger.info("KGAdapter: updated %d entities in KnowledgeGraph facade", updated)
        return updated
