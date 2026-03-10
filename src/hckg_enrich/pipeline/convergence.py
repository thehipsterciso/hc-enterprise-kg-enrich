"""ConvergenceController — iterative enrichment until KG meets completeness threshold.

Wraps EnrichmentController in a convergence loop:
  1. Research the target organisation once (OrgResearchAgent)
  2. Score KG completeness (KGCompletenessScorer)
  3. If below threshold: run GapAnalysis → EntityDiscovery → EnrichmentPass
  4. Repeat until threshold met, plateau detected, or max_iterations reached

Usage:
    controller = ConvergenceController(
        graph=graph,
        llm=llm,
        search=search,
        ticker="AAPL",
        target_coverage=0.80,
        max_iterations=10,
    )
    result = await controller.enrich_until_complete()
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hckg_enrich.agents.discovery_agent import EntityDiscoveryAgent
from hckg_enrich.org.profile import OrgProfile
from hckg_enrich.org.research_agent import OrgResearchAgent
from hckg_enrich.pipeline.controller import EnrichmentController
from hckg_enrich.providers.base import LLMProvider, SearchProvider
from hckg_enrich.scoring.completeness import CompletenessReport, KGCompletenessScorer
from hckg_enrich.scoring.gap_analysis import GapAnalysisAgent, GapReport

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceResult:
    """Complete record of a convergence run."""

    iterations: int
    converged: bool                             # True = threshold met
    stop_reason: str                            # "threshold_met" | "plateau" | "max_iterations"
    org_profile: OrgProfile | None
    iteration_reports: list[CompletenessReport] = field(default_factory=list)
    final_report: CompletenessReport | None = None
    total_entities_enriched: int = 0
    total_entities_discovered: int = 0
    total_relationships_added: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "org_profile": self.org_profile.to_dict() if self.org_profile else None,
            "final_score": self.final_report.overall_score if self.final_report else 0.0,
            "total_entities_enriched": self.total_entities_enriched,
            "total_entities_discovered": self.total_entities_discovered,
            "total_relationships_added": self.total_relationships_added,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class ConvergenceController:
    """Orchestrates iterative KG enrichment until completeness threshold is met.

    Activated by the CLI when --ticker or --org-name is provided. The existing
    EnrichmentController single-pass behaviour is unchanged for users who do not
    use convergence flags.
    """

    def __init__(
        self,
        graph: dict[str, Any],
        llm: LLMProvider,
        search: SearchProvider | None = None,
        ticker: str | None = None,
        org_name: str | None = None,
        industry: str | None = None,
        target_coverage: float = 0.80,
        max_iterations: int = 10,
        delta_threshold: float = 0.01,
        concurrency: int = 5,
        audit_log_path: str | None = None,
        artifacts_dir: str | None = None,
        llm_model: str = "claude-opus-4-6",
        llm_provider: str = "anthropic",
        search_provider: str | None = None,
        extra_contracts: list[Any] | None = None,
    ) -> None:
        self._graph = graph
        self._ticker = ticker
        self._org_name = org_name
        self._industry = industry
        self._target_coverage = target_coverage
        self._max_iterations = max_iterations
        self._delta_threshold = delta_threshold

        self._org_agent = OrgResearchAgent(llm=llm, search=search)
        self._scorer = KGCompletenessScorer()
        self._gap_agent = GapAnalysisAgent(llm=llm)
        self._discovery_agent = EntityDiscoveryAgent(graph=graph, llm=llm, search=search)
        self._inner = EnrichmentController(
            graph=graph,
            llm=llm,
            search=search,
            concurrency=concurrency,
            audit_log_path=audit_log_path,
            llm_model=llm_model,
            llm_provider=llm_provider,
            search_provider=search_provider,
            extra_contracts=extra_contracts,
        )

    async def enrich_until_complete(self, graph_path: str = "") -> ConvergenceResult:
        """Run iterative enrichment until convergence or max_iterations.

        Returns a ConvergenceResult with complete session metadata.
        """
        start = time.monotonic()
        logger.info(
            "ConvergenceController starting: ticker=%s org=%s target=%.2f max_iter=%d",
            self._ticker, self._org_name, self._target_coverage, self._max_iterations,
        )

        # Step 1: Research organisation once
        org_profile = await self._org_agent.research(
            ticker=self._ticker,
            org_name=self._org_name,
            industry=self._industry,
        )
        logger.info(
            "OrgProfile built: %s (confidence=%.2f, sources=%d)",
            org_profile.org_name or self._ticker,
            org_profile.research_confidence,
            len(org_profile.sources),
        )

        iteration_reports: list[CompletenessReport] = []
        prev_score = 0.0
        total_enriched = 0
        total_discovered = 0
        total_rels = 0
        stop_reason = "max_iterations"
        converged = False
        org_profile_dict = org_profile.to_dict()

        for i in range(1, self._max_iterations + 1):
            logger.info("=== Convergence iteration %d / %d ===", i, self._max_iterations)

            # Step 2: Score current state
            report = self._scorer.score(
                self._graph,
                org_profile=org_profile,
                threshold=self._target_coverage,
            )
            iteration_reports.append(report)

            logger.info(
                "Iteration %d score: %.3f (layer=%.2f field=%.2f density=%.2f prov=%.2f conf=%.2f)",
                i, report.overall_score,
                report.layer_coverage, report.field_population_rate,
                report.relationship_density, report.provenance_quality,
                report.confidence_quality,
            )

            # Step 3: Check convergence conditions
            if report.passes_threshold:
                stop_reason = "threshold_met"
                converged = True
                logger.info("Convergence threshold %.2f met at iteration %d", self._target_coverage, i)
                break

            if i > 1 and (report.overall_score - prev_score) < self._delta_threshold:
                stop_reason = "plateau"
                logger.info(
                    "Plateau detected at iteration %d (delta=%.4f < %.4f)",
                    i, report.overall_score - prev_score, self._delta_threshold,
                )
                break

            # Step 4: Gap analysis
            gap_report: GapReport = await self._gap_agent.analyze(report, org_profile)
            logger.info(
                "Gap analysis: %d gaps, %d entity types to create, %d entities to enrich",
                len(gap_report.gaps),
                len(gap_report.entity_types_to_create),
                len(gap_report.entity_ids_to_enrich),
            )

            # Step 5: Entity discovery for missing layers
            if gap_report.entity_types_to_create:
                new_entities = await self._discovery_agent.discover(
                    gap_report=gap_report,
                    org_profile=org_profile,
                    run_id=f"convergence-iter-{i}",
                )
                total_discovered += len(new_entities)
                logger.info("Discovered %d new entities in iteration %d", len(new_entities), i)

            # Step 6: Enrichment pass on prioritised entities
            target_ids = gap_report.entity_ids_to_enrich or None
            run = await self._inner.enrich_all(
                graph_path=graph_path,
                entity_ids=target_ids,
                org_profile=org_profile_dict,
            )
            total_enriched += run.enriched_count
            total_rels += run.relationships_added

            prev_score = report.overall_score

        # Final score
        final_report = self._scorer.score(
            self._graph, org_profile=org_profile, threshold=self._target_coverage
        )

        result = ConvergenceResult(
            iterations=i,
            converged=converged,
            stop_reason=stop_reason,
            org_profile=org_profile,
            iteration_reports=iteration_reports,
            final_report=final_report,
            total_entities_enriched=total_enriched,
            total_entities_discovered=total_discovered,
            total_relationships_added=total_rels,
            duration_seconds=time.monotonic() - start,
        )

        logger.info(
            "Convergence complete: %s in %d iterations, final_score=%.3f, "
            "enriched=%d, discovered=%d, rels=%d, duration=%.1fs",
            stop_reason, i, final_report.overall_score,
            total_enriched, total_discovered, total_rels,
            result.duration_seconds,
        )

        return result
