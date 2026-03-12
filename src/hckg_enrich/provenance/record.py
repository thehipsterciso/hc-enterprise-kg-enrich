"""ProvenanceRecord, EntityDiff, SourceCitation — per-change provenance models."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ConfidenceTier(StrEnum):
    """Maps enrichment evidence quality to the T1-T4 tier system.

    T1 — Verified Fact: attribute confirmed from existing graph or T1 source
    T2 — Strong Inference: multiple corroborating sources (search + graph)
    T3 — Reasoned Inference: single source or indirect evidence
    T4 — Working Hypothesis: LLM judgment with limited direct evidence
    """

    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"

    @property
    def label(self) -> str:
        labels = {
            "T1": "Verified Fact",
            "T2": "Strong Inference",
            "T3": "Reasoned Inference",
            "T4": "Working Hypothesis",
        }
        return labels[self.value]

    @property
    def confidence_range(self) -> tuple[float, float]:
        ranges = {
            "T1": (0.94, 1.00),
            "T2": (0.80, 0.93),
            "T3": (0.65, 0.79),
            "T4": (0.50, 0.64),
        }
        return ranges[self.value]


@dataclass
class SourceCitation:
    """Records which external source(s) contributed to an enrichment decision."""

    url: str
    title: str
    snippet: str
    relevance_score: float = 1.0
    retrieved_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    search_query: str = ""
    artifact_id: str | None = None  # Set when document stored as local artifact

    def to_dict(self) -> dict[str, Any]:
        d = {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "relevance_score": self.relevance_score,
            "retrieved_at": self.retrieved_at,
            "search_query": self.search_query,
        }
        if self.artifact_id is not None:
            d["artifact_id"] = self.artifact_id
        return d


@dataclass
class EntityDiff:
    """Before/after snapshot of an entity at commit time."""

    entity_id: str
    before: dict[str, Any]
    after: dict[str, Any]

    @property
    def added_fields(self) -> dict[str, Any]:
        return {k: v for k, v in self.after.items() if k not in self.before}

    @property
    def changed_fields(self) -> dict[str, tuple[Any, Any]]:
        return {
            k: (self.before[k], v)
            for k, v in self.after.items()
            if k in self.before and self.before[k] != v
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "before": self.before,
            "after": self.after,
            "added_fields": self.added_fields,
            "changed_fields": {
                k: {"before": b, "after": a} for k, (b, a) in self.changed_fields.items()
            },
        }


@dataclass
class ProvenanceRecord:
    """Full provenance record for a single entity enrichment commit.

    Written to the audit log by CommitAgent after a successful enrichment.
    Captures the complete lineage: who decided what, why, from what evidence,
    with what confidence, in which run.
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    entity_id: str = ""
    entity_name: str = ""
    entity_type: str = ""

    committed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    pipeline_version: str = "0.6.0"
    llm_model: str = ""

    # What changed
    attribute_changes: list[str] = field(default_factory=list)
    relationships_added: list[dict[str, Any]] = field(default_factory=list)
    entity_diff: EntityDiff | None = None

    # Why — from ReasoningAgent
    reasoning: str = ""
    confidence_tier: ConfidenceTier = ConfidenceTier.T4
    confidence_score: float = 0.5

    # Evidence — from SearchAgent
    sources: list[SourceCitation] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    graph_context_used: bool = True

    # Guard outcome
    guard_contracts_passed: list[str] = field(default_factory=list)
    guard_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "run_id": self.run_id,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "committed_at": self.committed_at,
            "pipeline_version": self.pipeline_version,
            "llm_model": self.llm_model,
            "attribute_changes": self.attribute_changes,
            "relationships_added": self.relationships_added,
            "entity_diff": self.entity_diff.to_dict() if self.entity_diff else None,
            "reasoning": self.reasoning,
            "confidence_tier": self.confidence_tier.value,
            "confidence_tier_label": self.confidence_tier.label,
            "confidence_score": self.confidence_score,
            "sources": [s.to_dict() for s in self.sources],
            "search_queries": self.search_queries,
            "graph_context_used": self.graph_context_used,
            "guard_contracts_passed": self.guard_contracts_passed,
            "guard_warnings": self.guard_warnings,
        }
