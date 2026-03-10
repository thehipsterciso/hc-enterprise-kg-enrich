"""Tests for KGAdapter — bridge between KnowledgeGraph facade and enrich dict format."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hckg_enrich.io.kg_adapter import KGAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(entity_id: str, entity_type: str = "department", name: str = "Finance") -> dict:
    return {"id": entity_id, "entity_type": entity_type, "name": name}


def _enriched_entity(entity_id: str) -> dict:
    return {
        "id": entity_id,
        "entity_type": "department",
        "name": "Finance",
        "provenance": {"enriched_by": "hckg-enrich/v0.5.0", "confidence_score": 0.82},
    }


def _enriched_rel(rel_id: str, source: str, target: str) -> dict:
    return {
        "id": rel_id,
        "source_id": source,
        "target_id": target,
        "relationship_type": "works_in",
        "weight": 1.0,
        "confidence": 0.9,
        "provenance": {"enriched_by": "hckg-enrich/v0.5.0"},
    }


# ---------------------------------------------------------------------------
# KGAdapter.to_dict — engine.export_dict() path
# ---------------------------------------------------------------------------

class TestToDictViaEngine:
    def test_uses_engine_export_dict_when_available(self):
        kg = MagicMock()
        kg.engine.export_dict.return_value = {
            "entities": [_entity("dept-001")],
            "relationships": [],
        }
        adapter = KGAdapter(kg)
        result = adapter.to_dict()
        assert result["entities"] == [_entity("dept-001")]
        assert result["relationships"] == []
        kg.engine.export_dict.assert_called_once()

    def test_engine_export_returns_both_keys(self):
        kg = MagicMock()
        kg.engine.export_dict.return_value = {
            "entities": [_entity("e1"), _entity("e2")],
            "relationships": [{"id": "r1", "source_id": "e1", "target_id": "e2"}],
        }
        result = KGAdapter(kg).to_dict()
        assert len(result["entities"]) == 2
        assert len(result["relationships"]) == 1


# ---------------------------------------------------------------------------
# KGAdapter.to_dict — fallback iteration path
# ---------------------------------------------------------------------------

def _fallback_kg():
    """Return a MagicMock KG whose engine.export_dict() raises AttributeError (triggers fallback)."""
    kg = MagicMock()
    kg.engine.export_dict.side_effect = AttributeError("no export_dict")
    return kg


class TestToDictFallback:
    def test_fallback_when_engine_missing(self):
        kg = _fallback_kg()
        entity = MagicMock()
        entity.model_dump.return_value = _entity("dept-001")
        rel = MagicMock()
        rel.model_dump.return_value = {"id": "r1", "source_id": "dept-001", "target_id": "dept-002"}
        kg.get_entities.return_value = [entity]
        kg.get_relationships.return_value = [rel]
        result = KGAdapter(kg).to_dict()
        assert result["entities"] == [_entity("dept-001")]
        assert len(result["relationships"]) == 1

    def test_fallback_uses_dict_when_no_model_dump(self):
        kg = _fallback_kg()
        entity = MagicMock(spec=["__dict__"])
        entity.__dict__ = {"id": "dept-001", "name": "Finance"}
        rel = MagicMock(spec=["__dict__"])
        rel.__dict__ = {"id": "r1"}
        kg.get_entities.return_value = [entity]
        kg.get_relationships.return_value = [rel]
        result = KGAdapter(kg).to_dict()
        assert result["entities"][0]["id"] == "dept-001"

    def test_fallback_handles_entity_read_error(self):
        kg = _fallback_kg()
        kg.get_entities.side_effect = RuntimeError("DB unavailable")
        kg.get_relationships.return_value = []
        result = KGAdapter(kg).to_dict()
        assert result["entities"] == []
        assert result["relationships"] == []

    def test_fallback_handles_relationship_read_error(self):
        kg = _fallback_kg()
        entity = MagicMock()
        entity.model_dump.return_value = _entity("e1")
        kg.get_entities.return_value = [entity]
        kg.get_relationships.side_effect = RuntimeError("timeout")
        result = KGAdapter(kg).to_dict()
        assert len(result["entities"]) == 1
        assert result["relationships"] == []


# ---------------------------------------------------------------------------
# KGAdapter.apply_enrichments — entity updates
# ---------------------------------------------------------------------------

class TestApplyEnrichmentsEntities:
    def test_updates_enriched_entities(self):
        kg = MagicMock()
        kg.get_relationships.return_value = []
        adapter = KGAdapter(kg)
        enriched = {"entities": [_enriched_entity("dept-001")], "relationships": []}
        count = adapter.apply_enrichments(enriched)
        assert count == 1
        kg.update_entity.assert_called_once()
        call_kwargs = kg.update_entity.call_args
        assert call_kwargs[0][0] == "dept-001"

    def test_skips_unenriched_entities(self):
        kg = MagicMock()
        kg.get_relationships.return_value = []
        adapter = KGAdapter(kg)
        plain = {"entities": [_entity("dept-001")], "relationships": []}
        count = adapter.apply_enrichments(plain)
        assert count == 0
        kg.update_entity.assert_not_called()

    def test_skips_entity_without_id(self):
        kg = MagicMock()
        kg.get_relationships.return_value = []
        adapter = KGAdapter(kg)
        entity = {"provenance": {"enriched_by": "hckg-enrich/v0.5.0"}}  # no id
        count = adapter.apply_enrichments({"entities": [entity], "relationships": []})
        assert count == 0

    def test_does_not_update_identity_fields(self):
        kg = MagicMock()
        kg.get_relationships.return_value = []
        adapter = KGAdapter(kg)
        entity = _enriched_entity("dept-001")
        adapter.apply_enrichments({"entities": [entity], "relationships": []})
        _, kwargs = kg.update_entity.call_args
        for forbidden in ("id", "entity_type", "name", "created_at"):
            assert forbidden not in kwargs

    def test_handles_update_entity_exception(self):
        kg = MagicMock()
        kg.get_relationships.return_value = []
        kg.update_entity.side_effect = RuntimeError("write failed")
        adapter = KGAdapter(kg)
        # Should not raise; just log warning
        count = adapter.apply_enrichments(
            {"entities": [_enriched_entity("dept-001")], "relationships": []}
        )
        assert count == 0

    def test_returns_count_of_updated_entities(self):
        kg = MagicMock()
        kg.get_relationships.return_value = []
        adapter = KGAdapter(kg)
        enriched = {
            "entities": [_enriched_entity("e1"), _enriched_entity("e2"), _entity("e3")],
            "relationships": [],
        }
        count = adapter.apply_enrichments(enriched)
        assert count == 2


# ---------------------------------------------------------------------------
# KGAdapter.apply_enrichments — relationship adds
# ---------------------------------------------------------------------------

class TestApplyEnrichmentsRelationships:
    def test_adds_new_enriched_relationship(self):
        kg = MagicMock()
        kg.get_relationships.return_value = []
        adapter = KGAdapter(kg)
        rel = _enriched_rel("r-new", "e1", "e2")
        adapter.apply_enrichments({"entities": [], "relationships": [rel]})
        kg.add_relationship.assert_called_once()

    def test_skips_pre_existing_relationship(self):
        existing = MagicMock()
        existing.id = "r-old"
        kg = MagicMock()
        kg.get_relationships.return_value = [existing]
        adapter = KGAdapter(kg)
        rel = _enriched_rel("r-old", "e1", "e2")  # same id as existing
        adapter.apply_enrichments({"entities": [], "relationships": [rel]})
        kg.add_relationship.assert_not_called()

    def test_skips_unenriched_relationship(self):
        kg = MagicMock()
        kg.get_relationships.return_value = []
        adapter = KGAdapter(kg)
        rel = {
            "id": "r-old",
            "source_id": "e1",
            "target_id": "e2",
            "relationship_type": "works_in",
        }
        adapter.apply_enrichments({"entities": [], "relationships": [rel]})
        kg.add_relationship.assert_not_called()

    def test_handles_add_relationship_exception(self):
        kg = MagicMock()
        kg.get_relationships.return_value = []
        kg.add_relationship.side_effect = RuntimeError("constraint violation")
        adapter = KGAdapter(kg)
        rel = _enriched_rel("r-new", "e1", "e2")
        # Should not raise
        adapter.apply_enrichments({"entities": [], "relationships": [rel]})

    def test_handles_get_relationships_exception_in_apply(self):
        kg = MagicMock()
        kg.get_relationships.side_effect = RuntimeError("connection lost")
        adapter = KGAdapter(kg)
        rel = _enriched_rel("r-new", "e1", "e2")
        # existing_ids defaults to empty set; rel should still be added
        adapter.apply_enrichments({"entities": [], "relationships": [rel]})
        kg.add_relationship.assert_called_once()

    def test_passes_correct_fields_to_add_relationship(self):
        kg = MagicMock()
        kg.get_relationships.return_value = []
        adapter = KGAdapter(kg)
        rel = _enriched_rel("r-new", "src-001", "tgt-001")
        adapter.apply_enrichments({"entities": [], "relationships": [rel]})
        kg.add_relationship.assert_called_once_with(
            source_id="src-001",
            target_id="tgt-001",
            relationship_type="works_in",
            weight=1.0,
            confidence=0.9,
            properties=rel["provenance"],
        )
