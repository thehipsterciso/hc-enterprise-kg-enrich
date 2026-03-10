"""GapAnalysisAgent — LLM-powered gap analysis with framework citations.

Consumes a CompletenessReport + OrgProfile and produces a GapReport that
identifies what is missing and why, with citations to industry frameworks.
Each GapItem includes a framework_url so users can verify the recommendation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hckg_enrich.providers.base import LLMProvider, Message
from hckg_enrich.scoring.completeness import CompletenessReport

if TYPE_CHECKING:
    from hckg_enrich.org.profile import OrgProfile

logger = logging.getLogger(__name__)

# Hardcoded framework reference URLs (canonical, stable)
FRAMEWORK_URLS: dict[str, str] = {
    "NIST SP 800-53": "https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final",
    "NIST CSF": "https://www.nist.gov/cyberframework",
    "ISO 27001": "https://www.iso.org/standard/27001",
    "CIS Controls v8": "https://www.cisecurity.org/controls/v8",
    "CMMI Level 3": "https://cmmiinstitute.com/learning/appraisals/levels",
    "SOX": "https://www.sec.gov/divisions/corpfin/guidance/soxact2002.htm",
    "HIPAA": "https://www.hhs.gov/hipaa/for-professionals/index.html",
    "GDPR": "https://gdpr.eu/",
    "PCI-DSS": "https://www.pcisecuritystandards.org/",
    "COBIT": "https://www.isaca.org/resources/cobit",
}

GAP_ANALYSIS_SYSTEM = """You are an enterprise knowledge graph completeness analyst.
Given a completeness report and organisation profile, identify the most impactful gaps
and recommend specific actions to address them.

For each gap, cite the industry framework or standard that motivates the recommendation.
Focus on gaps that will most improve the overall completeness score.

Be specific: name entity types, field names, and relationship types that are missing.
Prioritise: rank gaps 1 (highest impact) to N.
"""


class _GapItemSchema(BaseModel):
    priority: int
    gap_type: str          # "missing_layer" | "low_density" | "no_provenance" | "missing_relationships" | "low_field_population"
    entity_type: str = ""
    description: str
    recommended_action: str
    industry_basis: str    # citation text
    framework_name: str = ""  # key into FRAMEWORK_URLS


class _GapReportSchema(BaseModel):
    gaps: list[_GapItemSchema] = []
    entity_ids_to_enrich: list[str] = []
    entity_types_to_create: list[str] = []
    estimated_coverage_gain: float = 0.0


@dataclass
class GapItem:
    priority: int
    gap_type: str
    entity_type: str | None
    description: str
    recommended_action: str
    industry_basis: str
    framework_url: str | None
    priority_entity_ids: list[str] = field(default_factory=list)


@dataclass
class GapReport:
    gaps: list[GapItem] = field(default_factory=list)
    entity_ids_to_enrich: list[str] = field(default_factory=list)
    entity_types_to_create: list[str] = field(default_factory=list)
    estimated_coverage_gain: float = 0.0
    gap_sources: list[dict[str, Any]] = field(default_factory=list)  # framework citations


class GapAnalysisAgent:
    """Analyses KG completeness gaps and produces a prioritised remediation plan.

    Uses LLM reasoning grounded in industry frameworks. Appends canonical
    framework URLs to each GapItem so users can verify the rationale.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def analyze(
        self,
        report: CompletenessReport,
        org_profile: OrgProfile | None = None,
    ) -> GapReport:
        """Produce a GapReport from a CompletenessReport and optional OrgProfile."""

        # Build deterministic gaps from the report first (no LLM needed for obvious ones)
        deterministic_gaps = self._deterministic_gaps(report)

        # Build LLM prompt
        prompt = self._build_prompt(report, org_profile, deterministic_gaps)

        try:
            schema_result: _GapReportSchema = await self._llm.complete_structured(
                [Message(role="user", content=prompt)],
                schema=_GapReportSchema,
                system=GAP_ANALYSIS_SYSTEM,
            )
        except Exception as exc:
            logger.warning("GapAnalysisAgent LLM call failed: %s", exc)
            # Fall back to deterministic gaps only
            schema_result = _GapReportSchema(
                gaps=[
                    _GapItemSchema(
                        priority=i + 1,
                        gap_type=g["gap_type"],
                        entity_type=g.get("entity_type", ""),
                        description=g["description"],
                        recommended_action=g["recommended_action"],
                        industry_basis=g.get("industry_basis", "KG completeness best practice"),
                        framework_name="NIST CSF",
                    )
                    for i, g in enumerate(deterministic_gaps)
                ],
                entity_types_to_create=list(report.missing_layers),
                estimated_coverage_gain=round(
                    (1.0 - report.overall_score) * 0.4, 3
                ),
            )

        gap_items = [
            GapItem(
                priority=g.priority,
                gap_type=g.gap_type,
                entity_type=g.entity_type or None,
                description=g.description,
                recommended_action=g.recommended_action,
                industry_basis=g.industry_basis,
                framework_url=FRAMEWORK_URLS.get(g.framework_name),
            )
            for g in schema_result.gaps
        ]

        # Build framework citation records for auditability
        now = datetime.now(UTC).isoformat()
        cited_frameworks = {g.framework_name for g in schema_result.gaps if g.framework_name}
        gap_sources = [
            {
                "url": FRAMEWORK_URLS[fw],
                "title": fw,
                "snippet": f"Industry standard cited in gap analysis for {fw}",
                "relevance_score": 1.0,
                "retrieved_at": now,
                "search_query": "gap analysis framework reference",
            }
            for fw in cited_frameworks
            if fw in FRAMEWORK_URLS
        ]

        return GapReport(
            gaps=sorted(gap_items, key=lambda g: g.priority),
            entity_ids_to_enrich=list(schema_result.entity_ids_to_enrich),
            entity_types_to_create=list(schema_result.entity_types_to_create),
            estimated_coverage_gain=float(schema_result.estimated_coverage_gain),
            gap_sources=gap_sources,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _deterministic_gaps(self, report: CompletenessReport) -> list[dict[str, Any]]:
        """Build obvious gaps from the report without LLM."""
        gaps: list[dict[str, Any]] = []

        for layer in report.missing_layers:
            gaps.append({
                "gap_type": "missing_layer",
                "entity_type": layer,
                "description": f"No {layer} entities found in the graph",
                "recommended_action": f"Create {layer} entities via EntityDiscoveryAgent",
                "industry_basis": "Enterprise KG completeness requires all 12 entity type layers",
            })

        if report.provenance_quality < 0.5:
            gaps.append({
                "gap_type": "no_provenance",
                "description": f"{len(report.entities_without_sources)} entities lack URL-backed sources",
                "recommended_action": "Re-enrich these entities with search enabled to capture source URLs",
                "industry_basis": "Data trustworthiness requires traceable source citations",
            })

        if report.relationship_density < 0.5:
            gaps.append({
                "gap_type": "low_density",
                "description": f"Relationship density below enterprise benchmark (score={report.relationship_density:.2f})",
                "recommended_action": "Enrich entities with focus on relationship proposals",
                "industry_basis": "Enterprise KGs require ≥2 relationships per entity for structural completeness",
            })

        return gaps

    def _build_prompt(
        self,
        report: CompletenessReport,
        org_profile: OrgProfile | None,
        deterministic_gaps: list[dict[str, Any]],
    ) -> str:
        parts = ["## KG Completeness Report"]
        parts.append(f"Overall score: {report.overall_score:.2%} (threshold: {report.threshold:.2%})")
        parts.append(f"Layer coverage: {report.layer_coverage:.2%}")
        parts.append(f"Field population: {report.field_population_rate:.2%}")
        parts.append(f"Relationship density: {report.relationship_density:.2%}")
        parts.append(f"Provenance quality: {report.provenance_quality:.2%}")
        parts.append(f"Confidence quality: {report.confidence_quality:.2%}")

        if report.missing_layers:
            parts.append(f"\nMissing layers: {', '.join(report.missing_layers)}")
        if report.required_layers_missing:
            parts.append(f"Required missing (industry): {', '.join(report.required_layers_missing)}")
        if report.underpopulated_layers:
            parts.append(f"Underpopulated: {', '.join(report.underpopulated_layers)}")

        parts.append(f"\nTotal entities: {report.total_entities}")
        parts.append(f"Total relationships: {report.total_relationships}")
        parts.append(f"Entities without sources: {len(report.entities_without_sources)}")

        if org_profile:
            parts.append(f"\n## Organisation Profile\n{org_profile.context_string()}")

        if deterministic_gaps:
            parts.append("\n## Known Gaps (deterministic)")
            for g in deterministic_gaps:
                parts.append(f"- [{g['gap_type']}] {g['description']}")

        parts.append(
            "\n\nAnalyse these gaps and produce a prioritised remediation plan. "
            "For entity_ids_to_enrich, return the IDs of entities that most need enrichment "
            f"(max 50). For entity_types_to_create, return entity types from: "
            f"{', '.join(sorted(report.missing_layers))}."
        )

        # Append entity ID samples for entities_without_sources
        if report.entities_without_sources:
            sample = report.entities_without_sources[:20]
            parts.append(f"\nSample entity IDs lacking sources: {sample}")

        return "\n".join(parts)
