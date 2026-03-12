"""Tests for EnrichmentRun."""
from __future__ import annotations

from hckg_enrich.provenance.run import EnrichmentRun


def test_run_has_unique_id():
    r1 = EnrichmentRun()
    r2 = EnrichmentRun()
    assert r1.run_id != r2.run_id


def test_run_defaults():
    run = EnrichmentRun()
    assert run.llm_provider == "anthropic"
    assert run.pipeline_version == "0.3.0"
    assert run.completed_at is None
    assert run.enriched_count == 0


def test_run_complete_populates_stats():
    run = EnrichmentRun()
    run.complete(
        total=10, enriched=7, blocked=1, skipped=1, errors=1, relationships_added=5
    )
    assert run.total_entities == 10
    assert run.enriched_count == 7
    assert run.blocked_count == 1
    assert run.skipped_count == 1
    assert run.error_count == 1
    assert run.relationships_added == 5
    assert run.completed_at is not None


def test_run_to_dict_roundtrip():
    run = EnrichmentRun(graph_path="/tmp/graph.json", concurrency=3)
    run.complete(total=5, enriched=4, blocked=0, skipped=1, errors=0, relationships_added=2)
    d = run.to_dict()
    assert d["graph_path"] == "/tmp/graph.json"
    assert d["enriched_count"] == 4
    restored = EnrichmentRun.from_dict(d)
    assert restored.run_id == run.run_id
    assert restored.graph_path == "/tmp/graph.json"
    assert restored.enriched_count == 4


def test_run_started_at_is_iso_format():
    run = EnrichmentRun()
    # Should parse without error
    from datetime import datetime
    dt = datetime.fromisoformat(run.started_at)
    assert dt is not None


def test_run_config_field_accepts_arbitrary_dict():
    run = EnrichmentRun(config={"custom_key": "custom_value", "nested": {"a": 1}})
    d = run.to_dict()
    assert d["config"]["custom_key"] == "custom_value"
