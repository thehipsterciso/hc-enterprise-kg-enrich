"""Tests for ConvergenceController — iterative enrichment loop."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hckg_enrich.org.profile import OrgProfile
from hckg_enrich.pipeline.convergence import ConvergenceController, ConvergenceResult
from hckg_enrich.provenance.run import EnrichmentRun
from hckg_enrich.scoring.completeness import CompletenessReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(enriched: int = 2, rels: int = 1) -> EnrichmentRun:
    run = EnrichmentRun(graph_path="test.json")
    run.complete(
        total=3, enriched=enriched, blocked=0, skipped=0, errors=0, relationships_added=rels
    )
    return run


def _make_report(score: float, passes: bool = False) -> CompletenessReport:
    return CompletenessReport(
        overall_score=score,
        layer_coverage=score,
        field_population_rate=score,
        relationship_density=1.0,
        provenance_quality=score,
        confidence_quality=score,
        missing_layers=[],
        underpopulated_layers=[],
        entities_without_sources=[],
        passes_threshold=passes,
        total_entities=5,
        total_relationships=3,
        scored_at="2026-01-01T00:00:00+00:00",
    )


def _make_org_profile(name: str = "ACME Corp") -> OrgProfile:
    return OrgProfile(org_name=name, industry="technology")


def _make_gap_report(entity_ids=None, entity_types=None):
    from hckg_enrich.scoring.gap_analysis import GapReport
    return GapReport(
        gaps=[],
        entity_ids_to_enrich=entity_ids or [],
        entity_types_to_create=entity_types or [],
        estimated_coverage_gain=0.05,
        gap_sources=[],
    )


def _make_llm():
    llm = MagicMock()
    llm.complete_structured = AsyncMock()
    return llm


# ---------------------------------------------------------------------------
# ConvergenceResult
# ---------------------------------------------------------------------------


class TestConvergenceResult:
    def test_to_dict_basic(self):
        result = ConvergenceResult(
            iterations=3,
            converged=True,
            stop_reason="threshold_met",
            org_profile=_make_org_profile(),
            total_entities_enriched=10,
            total_entities_discovered=5,
            total_relationships_added=8,
            duration_seconds=12.5,
        )
        d = result.to_dict()
        assert d["iterations"] == 3
        assert d["converged"] is True
        assert d["stop_reason"] == "threshold_met"
        assert d["total_entities_enriched"] == 10
        assert d["total_entities_discovered"] == 5
        assert d["total_relationships_added"] == 8
        assert d["duration_seconds"] == 12.5

    def test_to_dict_no_org_profile(self):
        result = ConvergenceResult(
            iterations=1,
            converged=False,
            stop_reason="max_iterations",
            org_profile=None,
        )
        d = result.to_dict()
        assert d["org_profile"] is None

    def test_to_dict_with_final_report(self):
        report = _make_report(score=0.75)
        result = ConvergenceResult(
            iterations=5,
            converged=False,
            stop_reason="plateau",
            org_profile=None,
            final_report=report,
        )
        d = result.to_dict()
        assert d["final_score"] == pytest.approx(0.75)

    def test_to_dict_no_final_report(self):
        result = ConvergenceResult(
            iterations=1,
            converged=False,
            stop_reason="max_iterations",
            org_profile=None,
            final_report=None,
        )
        d = result.to_dict()
        assert d["final_score"] == 0.0

    def test_duration_rounded(self):
        result = ConvergenceResult(
            iterations=1,
            converged=True,
            stop_reason="threshold_met",
            org_profile=None,
            duration_seconds=3.141592653589793,
        )
        d = result.to_dict()
        assert d["duration_seconds"] == 3.14


# ---------------------------------------------------------------------------
# ConvergenceController — threshold_met on first iteration
# ---------------------------------------------------------------------------


class TestConvergenceThresholdMet:
    @pytest.mark.asyncio
    async def test_stops_when_threshold_met_iteration_1(self):
        graph = {"entities": [{"id": "e1", "name": "Corp"}], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile()
        # Score already passes threshold
        passing_report = _make_report(score=0.85, passes=True)

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent"),
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent"),
            patch("hckg_enrich.pipeline.convergence.EnrichmentController"),
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            MockScorer.return_value.score = MagicMock(return_value=passing_report)

            controller = ConvergenceController(
                graph=graph,
                llm=llm,
                target_coverage=0.80,
                max_iterations=10,
            )
            result = await controller.enrich_until_complete()

        assert result.converged is True
        assert result.stop_reason == "threshold_met"
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_org_profile_on_result(self):
        graph = {"entities": [], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile("TestCorp")
        passing_report = _make_report(score=0.90, passes=True)

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent"),
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent"),
            patch("hckg_enrich.pipeline.convergence.EnrichmentController"),
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            MockScorer.return_value.score = MagicMock(return_value=passing_report)

            controller = ConvergenceController(graph=graph, llm=llm)
            result = await controller.enrich_until_complete()

        assert result.org_profile is org_profile
        assert result.org_profile.org_name == "TestCorp"


# ---------------------------------------------------------------------------
# ConvergenceController — plateau detection
# ---------------------------------------------------------------------------


class TestConvergencePlateau:
    @pytest.mark.asyncio
    async def test_plateau_detected_on_second_iteration(self):
        graph = {"entities": [{"id": "e1", "name": "A"}], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile()
        # Iteration 1: score 0.50, Iteration 2: same score → plateau
        low_report = _make_report(score=0.50, passes=False)

        gap_report = _make_gap_report()
        mock_run = _make_run()

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent") as MockGap,
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent") as MockDisc,
            patch("hckg_enrich.pipeline.convergence.EnrichmentController") as MockInner,
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            # Score always returns same value
            MockScorer.return_value.score = MagicMock(return_value=low_report)
            MockGap.return_value.analyze = AsyncMock(return_value=gap_report)
            MockDisc.return_value.discover = AsyncMock(return_value=[])
            MockInner.return_value.enrich_all = AsyncMock(return_value=mock_run)

            controller = ConvergenceController(
                graph=graph,
                llm=llm,
                target_coverage=0.80,
                max_iterations=5,
                delta_threshold=0.01,
            )
            result = await controller.enrich_until_complete()

        assert result.stop_reason == "plateau"
        assert result.converged is False
        # Plateau triggers on iteration 2
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_no_plateau_when_score_improves(self):
        graph = {"entities": [{"id": "e1", "name": "A"}], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile()
        gap_report = _make_gap_report()
        mock_run = _make_run()

        # Scores improve each iteration: 0.50, 0.60, 0.70, 0.85 (passes) + 1 final report call
        scores = [0.50, 0.60, 0.70, 0.85, 0.85]
        reports = [_make_report(score=s, passes=(s >= 0.80)) for s in scores]
        score_iter = iter(reports)

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent") as MockGap,
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent") as MockDisc,
            patch("hckg_enrich.pipeline.convergence.EnrichmentController") as MockInner,
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            MockScorer.return_value.score = MagicMock(side_effect=lambda *a, **kw: next(score_iter))
            MockGap.return_value.analyze = AsyncMock(return_value=gap_report)
            MockDisc.return_value.discover = AsyncMock(return_value=[])
            MockInner.return_value.enrich_all = AsyncMock(return_value=mock_run)

            controller = ConvergenceController(
                graph=graph,
                llm=llm,
                target_coverage=0.80,
                max_iterations=10,
            )
            result = await controller.enrich_until_complete()

        assert result.stop_reason == "threshold_met"
        assert result.converged is True


# ---------------------------------------------------------------------------
# ConvergenceController — max_iterations
# ---------------------------------------------------------------------------


class TestConvergenceMaxIterations:
    @pytest.mark.asyncio
    async def test_stops_at_max_iterations(self):
        graph = {"entities": [{"id": "e1", "name": "A"}], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile()
        # Score slowly improves but never hits threshold
        gap_report = _make_gap_report()
        mock_run = _make_run()

        call_count = 0

        def _score(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            score = 0.40 + call_count * 0.02  # slowly improves
            return _make_report(score=score, passes=False)

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent") as MockGap,
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent") as MockDisc,
            patch("hckg_enrich.pipeline.convergence.EnrichmentController") as MockInner,
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            MockScorer.return_value.score = MagicMock(side_effect=_score)
            MockGap.return_value.analyze = AsyncMock(return_value=gap_report)
            MockDisc.return_value.discover = AsyncMock(return_value=[])
            MockInner.return_value.enrich_all = AsyncMock(return_value=mock_run)

            controller = ConvergenceController(
                graph=graph,
                llm=llm,
                target_coverage=0.90,
                max_iterations=3,
            )
            result = await controller.enrich_until_complete()

        assert result.stop_reason == "max_iterations"
        assert result.converged is False
        assert result.iterations == 3


# ---------------------------------------------------------------------------
# ConvergenceController — entity discovery
# ---------------------------------------------------------------------------


class TestConvergenceEntityDiscovery:
    @pytest.mark.asyncio
    async def test_discovery_called_when_types_to_create(self):
        graph = {"entities": [{"id": "e1", "name": "A"}], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile()
        # First iteration: no pass, needs discovery; second: passes
        gap_report_with_types = _make_gap_report(entity_types=["risk", "control"])
        mock_run = _make_run()

        new_entities = [
            {"id": "new-1", "entity_type": "risk", "name": "Credit Risk"},
            {"id": "new-2", "entity_type": "control", "name": "Access Control"},
        ]

        # 2 iterations + 1 final score call = 3 total
        score_iter = iter([
            _make_report(score=0.50, passes=False),
            _make_report(score=0.85, passes=True),
            _make_report(score=0.85, passes=True),  # final report call
        ])

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent") as MockGap,
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent") as MockDisc,
            patch("hckg_enrich.pipeline.convergence.EnrichmentController") as MockInner,
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            MockScorer.return_value.score = MagicMock(side_effect=lambda *a, **kw: next(score_iter))
            MockGap.return_value.analyze = AsyncMock(return_value=gap_report_with_types)
            MockDisc.return_value.discover = AsyncMock(return_value=new_entities)
            MockInner.return_value.enrich_all = AsyncMock(return_value=mock_run)

            controller = ConvergenceController(
                graph=graph,
                llm=llm,
                target_coverage=0.80,
                max_iterations=5,
            )
            result = await controller.enrich_until_complete()

        # Discovery was called
        MockDisc.return_value.discover.assert_called_once()
        assert result.total_entities_discovered == 2

    @pytest.mark.asyncio
    async def test_discovery_not_called_when_no_types_needed(self):
        graph = {"entities": [{"id": "e1", "name": "A"}], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile()
        gap_report_no_types = _make_gap_report(entity_types=[])
        mock_run = _make_run()

        # 2 iterations + 1 final score call = 3 total
        score_iter = iter([
            _make_report(score=0.50, passes=False),
            _make_report(score=0.85, passes=True),
            _make_report(score=0.85, passes=True),  # final report call
        ])

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent") as MockGap,
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent") as MockDisc,
            patch("hckg_enrich.pipeline.convergence.EnrichmentController") as MockInner,
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            MockScorer.return_value.score = MagicMock(side_effect=lambda *a, **kw: next(score_iter))
            MockGap.return_value.analyze = AsyncMock(return_value=gap_report_no_types)
            MockDisc.return_value.discover = AsyncMock(return_value=[])
            MockInner.return_value.enrich_all = AsyncMock(return_value=mock_run)

            controller = ConvergenceController(
                graph=graph,
                llm=llm,
                target_coverage=0.80,
                max_iterations=5,
            )
            result = await controller.enrich_until_complete()

        # Discovery was NOT called when no entity types needed
        MockDisc.return_value.discover.assert_not_called()
        assert result.total_entities_discovered == 0


# ---------------------------------------------------------------------------
# ConvergenceController — totals accumulation
# ---------------------------------------------------------------------------


class TestConvergenceTotals:
    @pytest.mark.asyncio
    async def test_enrichment_totals_accumulate_across_iterations(self):
        graph = {"entities": [{"id": "e1", "name": "A"}], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile()
        gap_report = _make_gap_report()

        # 3 iterations, each enriching 2 entities with 1 relationship
        run1 = _make_run(enriched=2, rels=1)
        run2 = _make_run(enriched=3, rels=2)
        run3 = _make_run(enriched=1, rels=0)

        score_iter = iter([
            _make_report(score=0.50, passes=False),
            _make_report(score=0.62, passes=False),
            _make_report(score=0.75, passes=False),
            # final score call after loop
            _make_report(score=0.78, passes=False),
        ])
        run_iter = iter([run1, run2, run3])

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent") as MockGap,
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent") as MockDisc,
            patch("hckg_enrich.pipeline.convergence.EnrichmentController") as MockInner,
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            MockScorer.return_value.score = MagicMock(side_effect=lambda *a, **kw: next(score_iter))
            MockGap.return_value.analyze = AsyncMock(return_value=gap_report)
            MockDisc.return_value.discover = AsyncMock(return_value=[])
            MockInner.return_value.enrich_all = AsyncMock(side_effect=lambda **kw: next(run_iter))

            controller = ConvergenceController(
                graph=graph,
                llm=llm,
                target_coverage=0.90,
                max_iterations=3,
            )
            result = await controller.enrich_until_complete()

        # Total enriched = 2 + 3 + 1 = 6
        assert result.total_entities_enriched == 6
        # Total relationships = 1 + 2 + 0 = 3
        assert result.total_relationships_added == 3
        assert result.stop_reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_iteration_reports_collected(self):
        graph = {"entities": [{"id": "e1", "name": "A"}], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile()
        gap_report = _make_gap_report()
        mock_run = _make_run()

        scores = [0.50, 0.65, 0.80]
        reports = [_make_report(score=s, passes=(s >= 0.80)) for s in scores]
        final_report = _make_report(score=0.80, passes=True)
        # score called: 3 times in loop + 1 final call = 4
        score_iter = iter(reports + [final_report])

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent") as MockGap,
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent") as MockDisc,
            patch("hckg_enrich.pipeline.convergence.EnrichmentController") as MockInner,
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            MockScorer.return_value.score = MagicMock(side_effect=lambda *a, **kw: next(score_iter))
            MockGap.return_value.analyze = AsyncMock(return_value=gap_report)
            MockDisc.return_value.discover = AsyncMock(return_value=[])
            MockInner.return_value.enrich_all = AsyncMock(return_value=mock_run)

            controller = ConvergenceController(
                graph=graph,
                llm=llm,
                target_coverage=0.80,
                max_iterations=10,
            )
            result = await controller.enrich_until_complete()

        # Three iterations scored before threshold met on iteration 3
        assert len(result.iteration_reports) == 3
        assert result.iteration_reports[0].overall_score == pytest.approx(0.50)
        assert result.iteration_reports[1].overall_score == pytest.approx(0.65)
        assert result.iteration_reports[2].overall_score == pytest.approx(0.80)

    @pytest.mark.asyncio
    async def test_final_report_always_computed(self):
        graph = {"entities": [], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile()
        passing_report = _make_report(score=0.85, passes=True)

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent"),
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent"),
            patch("hckg_enrich.pipeline.convergence.EnrichmentController"),
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            # First call in loop passes, second call is final_report
            MockScorer.return_value.score = MagicMock(return_value=passing_report)

            controller = ConvergenceController(graph=graph, llm=llm)
            result = await controller.enrich_until_complete()

        assert result.final_report is not None
        assert result.final_report.overall_score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_duration_seconds_positive(self):
        graph = {"entities": [], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile()
        passing_report = _make_report(score=0.90, passes=True)

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent"),
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent"),
            patch("hckg_enrich.pipeline.convergence.EnrichmentController"),
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            MockScorer.return_value.score = MagicMock(return_value=passing_report)

            controller = ConvergenceController(graph=graph, llm=llm)
            result = await controller.enrich_until_complete()

        assert result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# ConvergenceController — org_profile threading
# ---------------------------------------------------------------------------


class TestConvergenceOrgProfileThreading:
    @pytest.mark.asyncio
    async def test_org_profile_dict_passed_to_inner_controller(self):
        graph = {"entities": [{"id": "e1", "name": "A"}], "relationships": []}
        llm = _make_llm()

        org_profile = _make_org_profile("MegaCorp")
        gap_report = _make_gap_report()
        mock_run = _make_run()

        # 2 iterations + 1 final score call = 3 total
        score_iter = iter([
            _make_report(score=0.50, passes=False),
            _make_report(score=0.85, passes=True),
            _make_report(score=0.85, passes=True),  # final report call
        ])

        captured_org_profile = {}

        async def _enrich_all(**kwargs):
            captured_org_profile.update(kwargs.get("org_profile", {}) or {})
            return mock_run

        with (
            patch("hckg_enrich.pipeline.convergence.OrgResearchAgent") as MockOrgAgent,
            patch("hckg_enrich.pipeline.convergence.KGCompletenessScorer") as MockScorer,
            patch("hckg_enrich.pipeline.convergence.GapAnalysisAgent") as MockGap,
            patch("hckg_enrich.pipeline.convergence.EntityDiscoveryAgent") as MockDisc,
            patch("hckg_enrich.pipeline.convergence.EnrichmentController") as MockInner,
        ):
            MockOrgAgent.return_value.research = AsyncMock(return_value=org_profile)
            MockScorer.return_value.score = MagicMock(side_effect=lambda *a, **kw: next(score_iter))
            MockGap.return_value.analyze = AsyncMock(return_value=gap_report)
            MockDisc.return_value.discover = AsyncMock(return_value=[])
            MockInner.return_value.enrich_all = AsyncMock(side_effect=_enrich_all)

            controller = ConvergenceController(
                graph=graph,
                llm=llm,
                target_coverage=0.80,
                max_iterations=5,
            )
            await controller.enrich_until_complete()

        # org_profile dict should have been passed through
        assert "org_name" in captured_org_profile
        assert captured_org_profile["org_name"] == "MegaCorp"
