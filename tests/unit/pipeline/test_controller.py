"""Tests for EnrichmentController."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from hckg_enrich.agents.reasoning_agent import EnrichmentProposal
from hckg_enrich.pipeline.controller import EnrichmentController


def _passing_proposal() -> EnrichmentProposal:
    return EnrichmentProposal(
        proposed_attributes={"fiscal_year": "Q4"},
        proposed_relationships=[],
        reasoning="Test",
    )


def _empty_proposal() -> EnrichmentProposal:
    return EnrichmentProposal(proposed_attributes={}, proposed_relationships=[], reasoning="")


@pytest.mark.asyncio
async def test_controller_enrich_all_counts_total(sample_graph: dict[str, Any]) -> None:
    mock_llm = AsyncMock()
    mock_llm.complete_structured = AsyncMock(return_value=_passing_proposal())
    mock_llm.complete = AsyncMock(return_value='{"passes": true, "reason": "ok"}')

    controller = EnrichmentController(graph=sample_graph, llm=mock_llm, search=None)
    run = await controller.enrich_all()
    assert run.total_entities == len(sample_graph["entities"])
    assert run.error_count == 0


@pytest.mark.asyncio
async def test_controller_filters_by_entity_type(sample_graph: dict[str, Any]) -> None:
    mock_llm = AsyncMock()
    mock_llm.complete_structured = AsyncMock(return_value=_empty_proposal())
    mock_llm.complete = AsyncMock(return_value='{"passes": true, "reason": "ok"}')

    controller = EnrichmentController(graph=sample_graph, llm=mock_llm, search=None)
    run = await controller.enrich_all(entity_type="department")
    assert run.total_entities == 2  # Finance + HR


@pytest.mark.asyncio
async def test_controller_respects_limit(sample_graph: dict[str, Any]) -> None:
    mock_llm = AsyncMock()
    mock_llm.complete_structured = AsyncMock(return_value=_empty_proposal())
    mock_llm.complete = AsyncMock(return_value='{"passes": true, "reason": "ok"}')

    controller = EnrichmentController(graph=sample_graph, llm=mock_llm, search=None)
    run = await controller.enrich_all(limit=1)
    assert run.total_entities == 1


@pytest.mark.asyncio
async def test_controller_blocked_by_guard(sample_graph: dict[str, Any]) -> None:
    mock_llm = AsyncMock()
    mock_llm.complete_structured = AsyncMock(return_value=_empty_proposal())
    mock_llm.complete = AsyncMock(return_value='{"passes": false, "reason": "Invalid"}')

    controller = EnrichmentController(graph=sample_graph, llm=mock_llm, search=None)
    run = await controller.enrich_all()
    assert run.blocked_count > 0
    assert run.enriched_count == 0
