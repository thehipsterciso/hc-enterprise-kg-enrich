"""Integration tests: full 5-agent pipeline with mocked LLM and search."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from hckg_enrich.agents.reasoning_agent import EnrichmentProposal
from hckg_enrich.pipeline.controller import EnrichmentController


@pytest.mark.asyncio
async def test_full_pipeline_applies_attribute(sample_graph: dict[str, Any]) -> None:
    """End-to-end: a new attribute should be written to the entity."""
    mock_llm = AsyncMock()
    mock_llm.complete_structured = AsyncMock(
        return_value=EnrichmentProposal(
            proposed_attributes={"fiscal_year": "October-September"},
            proposed_relationships=[],
            reasoning="Finance dept US fiscal year",
        )
    )
    mock_llm.complete = AsyncMock(return_value='{"passes": true, "reason": "Valid"}')

    controller = EnrichmentController(graph=sample_graph, llm=mock_llm, search=None)
    result = await controller.enrich_entity("dept-finance-001")

    assert result["applied"] is True
    finance = next(e for e in sample_graph["entities"] if e["id"] == "dept-finance-001")
    assert finance.get("fiscal_year") == "October-September"
    assert "enriched_at" in finance.get("provenance", {})
    assert "hckg-enrich" in finance["provenance"]["enriched_by"]


@pytest.mark.asyncio
async def test_full_pipeline_does_not_overwrite_existing(sample_graph: dict[str, Any]) -> None:
    """Existing non-empty attributes must not be overwritten."""
    sample_graph["entities"][0]["description"] = "Existing description"

    mock_llm = AsyncMock()
    mock_llm.complete_structured = AsyncMock(
        return_value=EnrichmentProposal(
            proposed_attributes={"description": "Should be ignored"},
            proposed_relationships=[],
            reasoning="Test",
        )
    )
    mock_llm.complete = AsyncMock(return_value='{"passes": true, "reason": "Valid"}')

    controller = EnrichmentController(graph=sample_graph, llm=mock_llm, search=None)
    await controller.enrich_entity("dept-finance-001")

    finance = next(e for e in sample_graph["entities"] if e["id"] == "dept-finance-001")
    assert finance["description"] == "Existing description"


@pytest.mark.asyncio
async def test_full_pipeline_blocked_by_org_hierarchy(sample_graph: dict[str, Any]) -> None:
    """Finance→HR relationship proposal must be blocked by GraphGuard."""
    call_count: list[int] = [0]

    async def llm_complete(messages: object, system: str = "") -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return '{"passes": false, "reason": "Finance cannot report to HR"}'
        return '{"passes": true, "reason": "ok"}'

    mock_llm = AsyncMock()
    mock_llm.complete_structured = AsyncMock(
        return_value=EnrichmentProposal(
            proposed_attributes={},
            proposed_relationships=[
                {"relationship_type": "reports_to", "target_name": "Human Resources",
                 "target_type": "department", "rationale": "bad enrichment"}
            ],
            reasoning="Bad enrichment",
        )
    )
    mock_llm.complete = AsyncMock(side_effect=llm_complete)

    controller = EnrichmentController(graph=sample_graph, llm=mock_llm, search=None)
    result = await controller.enrich_entity("dept-finance-001")

    assert result["applied"] is False
    assert result.get("reason") == "Blocked by GraphGuard"


@pytest.mark.asyncio
async def test_full_pipeline_enrich_all_with_mix(sample_graph: dict[str, Any]) -> None:
    """enrich_all should accumulate stats correctly across multiple entities."""
    async def llm_complete(messages: object, system: str = "") -> str:
        return '{"passes": true, "reason": "ok"}'

    mock_llm = AsyncMock()
    mock_llm.complete_structured = AsyncMock(
        return_value=EnrichmentProposal(
            proposed_attributes={"tag": "enriched"},
            proposed_relationships=[],
            reasoning="Test",
        )
    )
    mock_llm.complete = AsyncMock(side_effect=llm_complete)

    controller = EnrichmentController(graph=sample_graph, llm=mock_llm, search=None)
    run = await controller.enrich_all()

    assert run.total_entities == 4
    assert run.error_count == 0
    assert run.enriched_count + run.blocked_count + run.skipped_count == 4
