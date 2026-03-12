"""Tests for EnrichmentTracer."""
from __future__ import annotations

import json

import pytest

from hckg_enrich.observability.tracer import EnrichmentTracer, Span


@pytest.fixture
def tracer() -> EnrichmentTracer:
    return EnrichmentTracer(service_name="test-service")


def test_span_defaults():
    span = Span(name="test", trace_id="trace-abc")
    assert span.status == "UNSET"
    assert span.end_time is None
    assert span.events == []


def test_span_set_ok():
    span = Span(name="test", trace_id="trace-abc")
    span.set_ok()
    assert span.status == "OK"


def test_span_set_error():
    span = Span(name="test", trace_id="trace-abc")
    span.set_error("something went wrong")
    assert span.status == "ERROR"
    assert span.status_message == "something went wrong"


def test_span_duration_ms():
    span = Span(name="test", trace_id="trace-abc")
    span.end()
    assert span.duration_ms is not None
    assert span.duration_ms >= 0.0


def test_span_add_event():
    span = Span(name="test", trace_id="trace-abc")
    span.add_event("proposal_generated", {"fields": 3})
    assert len(span.events) == 1
    assert span.events[0].name == "proposal_generated"
    assert span.events[0].attributes["fields"] == 3


def test_span_to_dict():
    span = Span(name="reasoning_agent", trace_id="trace-xyz")
    span.set_attribute("llm.model", "claude-opus-4-5")
    span.set_ok()
    span.end()
    d = span.to_dict()
    assert d["name"] == "reasoning_agent"
    assert d["status"] == "OK"
    assert d["attributes"]["llm.model"] == "claude-opus-4-5"
    assert d["duration_ms"] is not None


def test_span_to_otlp_dict():
    span = Span(name="context_agent", trace_id="abcdef1234567890abcdef1234567890")
    span.end()
    otlp = span.to_otlp_dict()
    assert "traceId" in otlp
    assert "spanId" in otlp
    assert "startTimeUnixNano" in otlp
    assert "status" in otlp


@pytest.mark.asyncio
async def test_tracer_span_context_manager(tracer: EnrichmentTracer):
    trace_id = tracer.new_trace_id()
    async with tracer.span("test_agent", trace_id=trace_id, entity_id="ent-001") as ctx:
        ctx.set_attribute("custom.key", "custom_value")
        ctx.add_event("test_event")
    spans = tracer.get_trace(trace_id)
    assert len(spans) == 1
    assert spans[0].name == "test_agent"
    assert spans[0].status == "OK"
    assert spans[0].attributes["custom.key"] == "custom_value"


@pytest.mark.asyncio
async def test_tracer_span_records_error_on_exception(tracer: EnrichmentTracer):
    trace_id = tracer.new_trace_id()
    with pytest.raises(ValueError):
        async with tracer.span("failing_agent", trace_id=trace_id):
            raise ValueError("test error")
    spans = tracer.get_trace(trace_id)
    assert len(spans) == 1
    assert spans[0].status == "ERROR"
    assert "test error" in spans[0].status_message


@pytest.mark.asyncio
async def test_tracer_multiple_spans_same_trace(tracer: EnrichmentTracer):
    trace_id = tracer.new_trace_id()
    async with tracer.span("context_agent", trace_id=trace_id):
        pass
    async with tracer.span("search_agent", trace_id=trace_id):
        pass
    async with tracer.span("reasoning_agent", trace_id=trace_id):
        pass
    spans = tracer.get_trace(trace_id)
    assert len(spans) == 3
    names = [s.name for s in spans]
    assert "context_agent" in names
    assert "reasoning_agent" in names


@pytest.mark.asyncio
async def test_tracer_to_otlp_json(tracer: EnrichmentTracer):
    trace_id = tracer.new_trace_id()
    async with tracer.span("test", trace_id=trace_id):
        pass
    otlp_json = tracer.to_otlp_json(trace_id=trace_id)
    data = json.loads(otlp_json)
    assert "resourceSpans" in data
    spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1


@pytest.mark.asyncio
async def test_tracer_summary(tracer: EnrichmentTracer):
    trace_id = tracer.new_trace_id()
    async with tracer.span("agent_a", trace_id=trace_id):
        pass
    async with tracer.span("agent_b", trace_id=trace_id):
        pass
    summary = tracer.summary(trace_id)
    assert summary["spans"] == 2
    assert summary["ok"] is True
    assert "agent_a" in summary["agents"]


def test_tracer_clear(tracer: EnrichmentTracer):
    span = Span(name="test", trace_id="t1")
    tracer._complete_span(span)
    assert len(tracer.get_all_spans()) == 1
    tracer.clear()
    assert len(tracer.get_all_spans()) == 0
