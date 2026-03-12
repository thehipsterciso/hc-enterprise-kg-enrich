"""Tests for AuditLog and AuditEvent."""
from __future__ import annotations

import json
from pathlib import Path

from hckg_enrich.provenance.audit_log import AuditEvent, AuditEventType, AuditLog


def _make_event(
    event_type: AuditEventType = AuditEventType.ENTITY_ENRICHED,
    run_id: str = "run-001",
    entity_id: str = "ent-001",
    confidence_tier: str = "T2",
) -> AuditEvent:
    return AuditEvent(
        event_type=event_type,
        run_id=run_id,
        entity_id=entity_id,
        entity_name="Test Entity",
        entity_type="system",
        agent_role="commit",
        pipeline_version="0.3.0",
        llm_model="claude-opus-4-5",
        attribute_changes=["set description='A system'"],
        relationships_added=[],
        confidence_tier=confidence_tier,
        guard_contracts_passed=["org-hierarchy-001"],
        guard_warnings=[],
        guard_blocking_failures=[],
        reasoning="LLM proposed description",
        search_source_count=2,
    )


def test_event_has_unique_id():
    e1 = _make_event()
    e2 = _make_event()
    assert e1.event_id != e2.event_id


def test_event_to_from_jsonl_roundtrip():
    event = _make_event()
    line = event.to_jsonl()
    # Must be valid JSON
    data = json.loads(line)
    assert data["event_type"] == AuditEventType.ENTITY_ENRICHED.value
    restored = AuditEvent.from_jsonl(line)
    assert restored.event_id == event.event_id
    assert restored.entity_id == event.entity_id
    assert restored.confidence_tier == event.confidence_tier


def test_audit_log_appends_and_reads(tmp_path: Path):
    log_path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path=log_path)
    events = [
        _make_event(entity_id="ent-001", run_id="run-A"),
        _make_event(event_type=AuditEventType.GUARD_BLOCKED, entity_id="ent-002", run_id="run-A"),
        _make_event(entity_id="ent-003", run_id="run-B"),
    ]
    for e in events:
        log.append(e)
    assert Path(log_path).stat().st_size > 0
    all_lines = Path(log_path).read_text().strip().split("\n")
    assert len(all_lines) == 3


def test_audit_log_query_by_entity(tmp_path: Path):
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    log.append(_make_event(entity_id="ent-001"))
    log.append(_make_event(entity_id="ent-002"))
    log.append(_make_event(entity_id="ent-001"))
    results = log.query_by_entity("ent-001")
    assert len(results) == 2
    assert all(e.entity_id == "ent-001" for e in results)


def test_audit_log_query_by_run(tmp_path: Path):
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    log.append(_make_event(run_id="run-X"))
    log.append(_make_event(run_id="run-Y"))
    log.append(_make_event(run_id="run-X"))
    results = log.query_by_run("run-X")
    assert len(results) == 2


def test_audit_log_query_blocked(tmp_path: Path):
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    log.append(_make_event(event_type=AuditEventType.ENTITY_ENRICHED))
    log.append(_make_event(event_type=AuditEventType.GUARD_BLOCKED))
    log.append(_make_event(event_type=AuditEventType.GUARD_BLOCKED))
    blocked = log.query_blocked()
    assert len(blocked) == 2


def test_audit_log_get_stats(tmp_path: Path):
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    log.append(_make_event(event_type=AuditEventType.ENTITY_ENRICHED, confidence_tier="T1"))
    log.append(_make_event(event_type=AuditEventType.ENTITY_ENRICHED, confidence_tier="T2"))
    log.append(_make_event(event_type=AuditEventType.GUARD_BLOCKED))
    stats = log.get_stats()
    assert stats["total_events"] == 3
    assert stats["event_types"][AuditEventType.ENTITY_ENRICHED.value] == 2
    assert stats["event_types"][AuditEventType.GUARD_BLOCKED.value] == 1


def test_audit_log_append_batch(tmp_path: Path):
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    events = [_make_event(entity_id=f"ent-{i:03d}") for i in range(5)]
    log.append_batch(events)
    all_events = log.query_by_run("run-001")
    assert len(all_events) == 5


def test_audit_log_export_jsonl(tmp_path: Path):
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    log.append(_make_event())
    log.append(_make_event())
    export_path = str(tmp_path / "export.jsonl")
    count = log.export_jsonl(export_path)
    assert count == 2
    assert Path(export_path).exists()
