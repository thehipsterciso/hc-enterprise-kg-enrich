"""AuditLog — append-only JSONL audit trail for enrichment operations.

Every enrichment commit writes an immutable audit event. Events are never
modified or deleted — rollbacks create new compensating events that reference
the original. This is the compliance and forensics backbone of the pipeline.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class AuditEventType(StrEnum):
    ENTITY_ENRICHED = "entity_enriched"
    RELATIONSHIP_ADDED = "relationship_added"
    FIELD_UPDATED = "field_updated"
    GUARD_BLOCKED = "guard_blocked"
    GUARD_WARNING = "guard_warning"
    PIPELINE_ERROR = "pipeline_error"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    ROLLBACK_INITIATED = "rollback_initiated"
    ROLLBACK_COMPLETED = "rollback_completed"


@dataclass
class AuditEvent:
    """Single immutable audit log entry.

    Written once, never modified. Rollbacks produce new ROLLBACK_* events
    that reference the original event_id.
    """

    event_id: str = field(default_factory=lambda: __import__("uuid").uuid4().__str__())
    event_type: AuditEventType = AuditEventType.ENTITY_ENRICHED
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Links
    run_id: str = ""
    entity_id: str = ""
    entity_name: str = ""
    entity_type: str = ""

    # Actor
    agent_role: str = ""
    pipeline_version: str = "0.3.0"
    llm_model: str = ""

    # Change summary
    attribute_changes: list[str] = field(default_factory=list)
    relationships_added: int = 0

    # Quality
    confidence_tier: str = "T4"
    guard_contracts_passed: list[str] = field(default_factory=list)
    guard_warnings: list[str] = field(default_factory=list)
    guard_blocking_failures: list[str] = field(default_factory=list)

    # Reasoning
    reasoning: str = ""
    search_source_count: int = 0

    # Error info
    error_message: str = ""

    # Rollback linkage
    references_event_id: str = ""

    # Free-form metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Serialize to a single JSON line (no trailing newline)."""
        return json.dumps(
            {
                "event_id": self.event_id,
                "event_type": str(self.event_type),
                "timestamp": self.timestamp,
                "run_id": self.run_id,
                "entity_id": self.entity_id,
                "entity_name": self.entity_name,
                "entity_type": self.entity_type,
                "agent_role": self.agent_role,
                "pipeline_version": self.pipeline_version,
                "llm_model": self.llm_model,
                "attribute_changes": self.attribute_changes,
                "relationships_added": self.relationships_added,
                "confidence_tier": self.confidence_tier,
                "guard_contracts_passed": self.guard_contracts_passed,
                "guard_warnings": self.guard_warnings,
                "guard_blocking_failures": self.guard_blocking_failures,
                "reasoning": self.reasoning,
                "search_source_count": self.search_source_count,
                "error_message": self.error_message,
                "references_event_id": self.references_event_id,
                "metadata": self.metadata,
            },
            separators=(",", ":"),
            default=str,
        )

    @classmethod
    def from_jsonl(cls, line: str) -> AuditEvent:
        """Deserialize from a JSON line."""
        data = json.loads(line.strip())
        event = cls()
        for k, v in data.items():
            if hasattr(event, k):
                if k == "event_type":
                    try:
                        setattr(event, k, AuditEventType(v))
                    except ValueError:
                        setattr(event, k, v)
                else:
                    setattr(event, k, v)
        return event


class AuditLog:
    """Append-only JSONL audit log with thread-safe writes and queryable reads.

    Storage: one line per event in a .jsonl file.
    Writes: serialised under a threading.Lock() for concurrent pipeline safety.
    Reads: scan-and-filter (no index); acceptable for audit use — not hot path.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        # Ensure the file and its directory exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not os.path.exists(path):
            open(path, "w").close()

    @property
    def path(self) -> str:
        return self._path

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def append(self, event: AuditEvent) -> None:
        """Append a single event (thread-safe)."""
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(event.to_jsonl() + "\n")

    def append_batch(self, events: list[AuditEvent]) -> None:
        """Append multiple events in a single lock acquisition."""
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                for evt in events:
                    f.write(evt.to_jsonl() + "\n")

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def _all_events(self) -> list[AuditEvent]:
        events: list[AuditEvent] = []
        if not os.path.exists(self._path):
            return events
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(AuditEvent.from_jsonl(line))
                except Exception:
                    pass  # Skip malformed lines — never break audit reads
        return events

    def query_by_entity(self, entity_id: str) -> list[AuditEvent]:
        return [e for e in self._all_events() if e.entity_id == entity_id]

    def query_by_run(self, run_id: str) -> list[AuditEvent]:
        return [e for e in self._all_events() if e.run_id == run_id]

    def query_by_type(self, event_type: AuditEventType) -> list[AuditEvent]:
        return [e for e in self._all_events() if e.event_type == event_type]

    def query_by_confidence_tier(self, tier: str) -> list[AuditEvent]:
        return [e for e in self._all_events() if e.confidence_tier == tier]

    def query_blocked(self) -> list[AuditEvent]:
        return [e for e in self._all_events() if e.event_type == AuditEventType.GUARD_BLOCKED]

    def get_stats(self) -> dict[str, Any]:
        events = self._all_events()
        type_counts: dict[str, int] = {}
        tier_counts: dict[str, int] = {}
        run_ids: set[str] = set()
        entity_ids: set[str] = set()

        for e in events:
            type_counts[str(e.event_type)] = type_counts.get(str(e.event_type), 0) + 1
            tier_counts[e.confidence_tier] = tier_counts.get(e.confidence_tier, 0) + 1
            if e.run_id:
                run_ids.add(e.run_id)
            if e.entity_id:
                entity_ids.add(e.entity_id)

        return {
            "total_events": len(events),
            "event_types": type_counts,
            "confidence_distribution": tier_counts,
            "unique_runs": len(run_ids),
            "unique_entities_enriched": len(entity_ids),
        }

    def export_jsonl(self, output_path: str) -> int:
        """Export entire log to an external JSONL file. Returns event count."""
        events = self._all_events()
        with open(output_path, "w", encoding="utf-8") as f:
            for e in events:
                f.write(e.to_jsonl() + "\n")
        return len(events)
