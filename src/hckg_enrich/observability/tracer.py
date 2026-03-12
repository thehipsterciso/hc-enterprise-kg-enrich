"""Distributed tracing for the enrichment pipeline.

OpenTelemetry-compatible span model with zero external dependencies.
Spans can be exported to OTLP-compatible JSON for ingestion by Jaeger,
Zipkin, or any OpenTelemetry collector.

Usage:
    tracer = EnrichmentTracer(service_name="hckg-enrich")
    async with tracer.span("reasoning_agent", run_id=run_id, entity_id=eid) as span:
        span.set_attribute("llm.model", model)
        result = await do_work()
        span.add_event("proposal_generated", {"fields_count": 3})
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SpanEvent:
    """A timestamped event within a span (OTel-compatible)."""

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """A single unit of work in the enrichment pipeline.

    Mirrors the OpenTelemetry Span data model:
    - trace_id: Groups all spans for one enrichment run
    - span_id:  Unique ID for this span
    - parent_span_id: Parent span (None for root)
    - name: Agent stage or operation name
    - start_time / end_time: Wall-clock timestamps (seconds since epoch)
    - attributes: Key-value metadata (OTel Attributes)
    - events: Timestamped events within the span
    - status: "UNSET" | "OK" | "ERROR"
    - status_message: Error message on failure
    """

    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    status: str = "UNSET"
    status_message: str = ""

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        self.attributes.update(attrs)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))

    def set_ok(self) -> None:
        self.status = "OK"

    def set_error(self, message: str) -> None:
        self.status = "ERROR"
        self.status_message = message

    def end(self) -> None:
        if self.end_time is None:
            self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
        }

    def to_otlp_dict(self) -> dict[str, Any]:
        """OpenTelemetry OTLP-compatible representation."""
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "name": self.name,
            "startTimeUnixNano": int(self.start_time * 1e9),
            "endTimeUnixNano": int((self.end_time or time.time()) * 1e9),
            "status": {
                "code": {"UNSET": 0, "OK": 1, "ERROR": 2}.get(self.status, 0),
                "message": self.status_message,
            },
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in self.attributes.items()
            ],
            "events": [
                {
                    "timeUnixNano": int(e.timestamp * 1e9),
                    "name": e.name,
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}}
                        for k, v in e.attributes.items()
                    ],
                }
                for e in self.events
            ],
        }


# ---------------------------------------------------------------------------
# SpanContext — async context manager for automatic span lifecycle
# ---------------------------------------------------------------------------


class SpanContext:
    """Wraps a Span for use as an async context manager.

        async with tracer.span("reasoning", run_id=run_id) as ctx:
            ctx.set_attribute("model", "claude-opus-4-5")
    """

    def __init__(self, span: Span, tracer: EnrichmentTracer) -> None:
        self._span = span
        self._tracer = tracer

    @property
    def span(self) -> Span:
        return self._span

    # Delegate common Span methods for convenience
    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        self._span.set_attributes(attrs)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._span.add_event(name, attributes)

    async def __aenter__(self) -> SpanContext:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self._span.set_error(str(exc_val) if exc_val else repr(exc_type))
        else:
            self._span.set_ok()
        self._span.end()
        self._tracer._complete_span(self._span)


# ---------------------------------------------------------------------------
# EnrichmentTracer
# ---------------------------------------------------------------------------


class EnrichmentTracer:
    """Manages distributed traces for the enrichment pipeline.

    Each EnrichmentRun creates a trace_id. Each agent stage creates a child span.
    Completed spans are stored in-memory and can be exported as OTLP JSON.

    Thread-safe: uses asyncio.Lock for span storage.
    """

    def __init__(self, service_name: str = "hckg-enrich") -> None:
        self.service_name = service_name
        self._completed_spans: list[Span] = []
        self._lock = asyncio.Lock()

    def new_trace_id(self) -> str:
        """Generate a new W3C-compatible trace ID (32 hex chars)."""
        return uuid.uuid4().hex + uuid.uuid4().hex[:0]  # 32 chars

    def start_span(
        self,
        name: str,
        trace_id: str,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Create a new span (not auto-managed — caller must call span.end())."""
        span = Span(
            name=name,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        if attributes:
            span.set_attributes(attributes)
        return span

    @asynccontextmanager
    async def span(
        self,
        name: str,
        trace_id: str,
        parent_span_id: str | None = None,
        run_id: str | None = None,
        entity_id: str | None = None,
        agent: str | None = None,
    ) -> AsyncIterator[SpanContext]:
        """Async context manager that auto-ends and records the span.

        Example:
            async with tracer.span("context_agent", trace_id=tid, entity_id=eid) as ctx:
                ctx.set_attribute("neighbors_found", 5)
        """
        span = self.start_span(name=name, trace_id=trace_id, parent_span_id=parent_span_id)
        # Standard enrichment attributes
        if run_id:
            span.set_attribute("enrichment.run_id", run_id)
        if entity_id:
            span.set_attribute("enrichment.entity_id", entity_id)
        if agent:
            span.set_attribute("enrichment.agent", agent)
        span.set_attribute("service.name", self.service_name)

        ctx = SpanContext(span=span, tracer=self)
        try:
            yield ctx
        except Exception as e:
            span.set_error(str(e))
            span.end()
            self._complete_span(span)
            raise
        else:
            span.set_ok()
            span.end()
            self._complete_span(span)

    def _complete_span(self, span: Span) -> None:
        """Called by SpanContext on exit. Stores span for later export."""
        self._completed_spans.append(span)

    def get_trace(self, trace_id: str) -> list[Span]:
        """Return all spans for a given trace_id, ordered by start_time."""
        return sorted(
            [s for s in self._completed_spans if s.trace_id == trace_id],
            key=lambda s: s.start_time,
        )

    def get_all_spans(self) -> list[Span]:
        return list(self._completed_spans)

    def clear(self) -> None:
        """Discard all recorded spans. Use after export."""
        self._completed_spans.clear()

    def to_otlp_json(self, trace_id: str | None = None) -> str:
        """Export spans as OTLP JSON (ResourceSpans envelope).

        Compatible with OpenTelemetry collectors, Jaeger, and Zipkin.
        """
        spans = self.get_trace(trace_id) if trace_id else self._completed_spans
        resource_spans = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}}
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "hckg_enrich.pipeline"},
                            "spans": [s.to_otlp_dict() for s in spans],
                        }
                    ],
                }
            ]
        }
        return json.dumps(resource_spans, indent=2)

    def to_dict(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        """Export spans as a list of dicts."""
        spans = self.get_trace(trace_id) if trace_id else self._completed_spans
        return [s.to_dict() for s in spans]

    def summary(self, trace_id: str) -> dict[str, Any]:
        """High-level summary of a trace for logging/debugging."""
        spans = self.get_trace(trace_id)
        if not spans:
            return {"trace_id": trace_id, "spans": 0}
        root = min(spans, key=lambda s: s.start_time)
        last = max(spans, key=lambda s: s.end_time or s.start_time)
        total_ms = ((last.end_time or time.time()) - root.start_time) * 1000
        error_spans = [s.name for s in spans if s.status == "ERROR"]
        return {
            "trace_id": trace_id,
            "spans": len(spans),
            "total_duration_ms": round(total_ms, 2),
            "agents": [s.name for s in spans],
            "errors": error_spans,
            "ok": len(error_spans) == 0,
        }
