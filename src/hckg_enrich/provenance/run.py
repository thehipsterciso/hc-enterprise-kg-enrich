"""EnrichmentRun — tracks a full pipeline execution session with config snapshot."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class EnrichmentRun:
    """Represents a single pipeline execution session.

    A run groups all enrichment activity that happened together: same graph,
    same model, same config, same wall-clock window.  Every AuditEvent and
    ProvenanceRecord references back to a run_id so the full lineage of any
    graph change is traceable.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None

    graph_path: str = ""
    entity_type_filter: str | None = None
    limit: int | None = None
    concurrency: int = 5

    # Model provenance
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-4-5"
    search_provider: str | None = None
    pipeline_version: str = "0.3.0"

    # Results (populated at completion)
    total_entities: int = 0
    enriched_count: int = 0
    blocked_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    relationships_added: int = 0

    # Arbitrary config snapshot for reproducibility
    config: dict[str, Any] = field(default_factory=dict)

    def complete(
        self,
        *,
        total: int,
        enriched: int,
        blocked: int,
        skipped: int,
        errors: int,
        relationships_added: int,
    ) -> None:
        """Finalise the run with outcome stats."""
        self.completed_at = datetime.now(UTC).isoformat()
        self.total_entities = total
        self.enriched_count = enriched
        self.blocked_count = blocked
        self.skipped_count = skipped
        self.error_count = errors
        self.relationships_added = relationships_added

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "graph_path": self.graph_path,
            "entity_type_filter": self.entity_type_filter,
            "limit": self.limit,
            "concurrency": self.concurrency,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "search_provider": self.search_provider,
            "pipeline_version": self.pipeline_version,
            "total_entities": self.total_entities,
            "enriched_count": self.enriched_count,
            "blocked_count": self.blocked_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "relationships_added": self.relationships_added,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnrichmentRun:
        run = cls()
        for k, v in data.items():
            if hasattr(run, k):
                setattr(run, k, v)
        return run
