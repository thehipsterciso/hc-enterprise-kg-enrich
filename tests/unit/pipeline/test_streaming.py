"""Tests for streaming pipeline (ProgressEvent, enrich_all_streaming)."""
from __future__ import annotations

import pytest

from hckg_enrich.pipeline.controller import EnrichmentController


@pytest.fixture()
def minimal_graph():
    return {
        "entities": [
            {"id": "e1", "entity_type": "system", "name": "SAP"},
            {"id": "e2", "entity_type": "department", "name": "Finance"},
        ],
        "relationships": [],
    }


@pytest.mark.asyncio
async def test_streaming_emits_started_event(minimal_graph, mock_llm):
    ctrl = EnrichmentController(minimal_graph, mock_llm, concurrency=1)
    events = []
    async for event in ctrl.enrich_all_streaming():
        events.append(event)

    started = [e for e in events if e.type == "started"]
    assert len(started) == 1
    assert started[0].total == 2


@pytest.mark.asyncio
async def test_streaming_emits_completed_event(minimal_graph, mock_llm):
    ctrl = EnrichmentController(minimal_graph, mock_llm, concurrency=1)
    events = []
    async for event in ctrl.enrich_all_streaming():
        events.append(event)

    completed = [e for e in events if e.type == "completed"]
    assert len(completed) == 1
    assert completed[0].stats is not None


@pytest.mark.asyncio
async def test_streaming_emits_entity_done_per_entity(minimal_graph, mock_llm):
    ctrl = EnrichmentController(minimal_graph, mock_llm, concurrency=2)
    events = []
    async for event in ctrl.enrich_all_streaming():
        events.append(event)

    done_events = [e for e in events if e.type == "entity_done"]
    assert len(done_events) == 2
    ids = {e.entity_id for e in done_events}
    assert ids == {"e1", "e2"}


@pytest.mark.asyncio
async def test_enrich_all_returns_stats(minimal_graph, mock_llm):
    ctrl = EnrichmentController(minimal_graph, mock_llm, concurrency=1)
    stats = await ctrl.enrich_all()
    assert stats.total_entities == 2


@pytest.mark.asyncio
async def test_streaming_entity_type_filter(minimal_graph, mock_llm):
    ctrl = EnrichmentController(minimal_graph, mock_llm, concurrency=1)
    events = []
    async for event in ctrl.enrich_all_streaming(entity_type="system"):
        events.append(event)

    started = next(e for e in events if e.type == "started")
    assert started.total == 1


@pytest.mark.asyncio
async def test_streaming_limit(minimal_graph, mock_llm):
    ctrl = EnrichmentController(minimal_graph, mock_llm, concurrency=1)
    events = []
    async for event in ctrl.enrich_all_streaming(limit=1):
        events.append(event)

    done = [e for e in events if e.type == "entity_done"]
    assert len(done) == 1
