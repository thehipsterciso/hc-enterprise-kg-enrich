"""Tests for GapAnalysisAgent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from hckg_enrich.org.profile import OrgProfile
from hckg_enrich.scoring.completeness import CompletenessReport
from hckg_enrich.scoring.gap_analysis import (
    FRAMEWORK_URLS,
    GapAnalysisAgent,
    GapItem,
    GapReport,
)


def _mock_llm(gaps=None, entity_types_to_create=None, entity_ids=None, gain=0.2):
    """Return a mock LLM that yields a _GapReportSchema-like object."""
    llm = MagicMock()

    class _GapSchema(BaseModel):
        priority: int
        gap_type: str
        entity_type: str = ""
        description: str
        recommended_action: str
        industry_basis: str
        framework_name: str = ""

    class _ReportSchema(BaseModel):
        gaps: list[_GapSchema] = []
        entity_ids_to_enrich: list[str] = []
        entity_types_to_create: list[str] = []
        estimated_coverage_gain: float = gain

    result = _ReportSchema(
        gaps=gaps or [
            _GapSchema(
                priority=1,
                gap_type="missing_layer",
                entity_type="control",
                description="No control entities found",
                recommended_action="Create control entities",
                industry_basis="NIST SP 800-53 requires access control",
                framework_name="NIST SP 800-53",
            )
        ],
        entity_types_to_create=entity_types_to_create or ["control"],
        entity_ids_to_enrich=entity_ids or ["e1", "e2"],
        estimated_coverage_gain=gain,
    )
    llm.complete_structured = AsyncMock(return_value=result)
    return llm


def _report(missing_layers=None, prov_quality=1.0, density=1.0, overall=0.4):
    return CompletenessReport(
        overall_score=overall,
        layer_coverage=0.5,
        field_population_rate=0.5,
        relationship_density=density,
        provenance_quality=prov_quality,
        confidence_quality=0.5,
        missing_layers=missing_layers or ["control", "risk"],
        required_layers_missing=missing_layers or ["control", "risk"],
        entities_without_sources=[],
        passes_threshold=False,
        total_entities=10,
        total_relationships=5,
    )


class TestGapReportStructure:
    @pytest.mark.asyncio
    async def test_returns_gap_report(self):
        agent = GapAnalysisAgent(llm=_mock_llm())
        report = await agent.analyze(_report())
        assert isinstance(report, GapReport)

    @pytest.mark.asyncio
    async def test_gaps_are_gap_items(self):
        agent = GapAnalysisAgent(llm=_mock_llm())
        report = await agent.analyze(_report())
        assert all(isinstance(g, GapItem) for g in report.gaps)

    @pytest.mark.asyncio
    async def test_entity_types_to_create_populated(self):
        agent = GapAnalysisAgent(llm=_mock_llm(entity_types_to_create=["control", "risk"]))
        report = await agent.analyze(_report())
        assert "control" in report.entity_types_to_create

    @pytest.mark.asyncio
    async def test_entity_ids_to_enrich_populated(self):
        agent = GapAnalysisAgent(llm=_mock_llm(entity_ids=["abc", "def"]))
        report = await agent.analyze(_report())
        assert "abc" in report.entity_ids_to_enrich

    @pytest.mark.asyncio
    async def test_estimated_coverage_gain_is_float(self):
        agent = GapAnalysisAgent(llm=_mock_llm(gain=0.15))
        report = await agent.analyze(_report())
        assert isinstance(report.estimated_coverage_gain, float)
        assert report.estimated_coverage_gain > 0.0


class TestFrameworkCitations:
    @pytest.mark.asyncio
    async def test_framework_url_populated_for_known_framework(self):
        agent = GapAnalysisAgent(llm=_mock_llm())
        report = await agent.analyze(_report())
        items_with_framework = [g for g in report.gaps if g.framework_url]
        assert len(items_with_framework) > 0

    @pytest.mark.asyncio
    async def test_framework_url_is_valid_url(self):
        agent = GapAnalysisAgent(llm=_mock_llm())
        report = await agent.analyze(_report())
        for gap in report.gaps:
            if gap.framework_url:
                assert gap.framework_url.startswith("http")

    @pytest.mark.asyncio
    async def test_gap_sources_contain_framework_citations(self):
        agent = GapAnalysisAgent(llm=_mock_llm())
        report = await agent.analyze(_report())
        assert len(report.gap_sources) > 0
        for src in report.gap_sources:
            assert "url" in src
            assert src["url"].startswith("http")

    def test_framework_urls_dict_has_canonical_urls(self):
        assert "NIST SP 800-53" in FRAMEWORK_URLS
        assert "ISO 27001" in FRAMEWORK_URLS
        assert "CIS Controls v8" in FRAMEWORK_URLS
        for url in FRAMEWORK_URLS.values():
            assert url.startswith("http")


class TestLLMFailureFallback:
    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        llm = MagicMock()
        llm.complete_structured = AsyncMock(side_effect=RuntimeError("LLM down"))
        agent = GapAnalysisAgent(llm=llm)
        # Should not raise — falls back to deterministic gaps
        report = await agent.analyze(_report(missing_layers=["control"]))
        assert isinstance(report, GapReport)

    @pytest.mark.asyncio
    async def test_fallback_includes_missing_layers(self):
        llm = MagicMock()
        llm.complete_structured = AsyncMock(side_effect=RuntimeError("LLM down"))
        agent = GapAnalysisAgent(llm=llm)
        report = await agent.analyze(_report(missing_layers=["control", "risk"]))
        types = [g.entity_type for g in report.gaps]
        assert "control" in types or "risk" in types

    @pytest.mark.asyncio
    async def test_fallback_includes_low_provenance_gap(self):
        llm = MagicMock()
        llm.complete_structured = AsyncMock(side_effect=RuntimeError("LLM down"))
        agent = GapAnalysisAgent(llm=llm)
        low_prov_report = _report(prov_quality=0.2)
        result = await agent.analyze(low_prov_report)
        gap_types = [g.gap_type for g in result.gaps]
        assert "no_provenance" in gap_types

    @pytest.mark.asyncio
    async def test_fallback_includes_low_density_gap(self):
        llm = MagicMock()
        llm.complete_structured = AsyncMock(side_effect=RuntimeError("LLM down"))
        agent = GapAnalysisAgent(llm=llm)
        low_density_report = _report(density=0.1)
        result = await agent.analyze(low_density_report)
        gap_types = [g.gap_type for g in result.gaps]
        assert "low_density" in gap_types
