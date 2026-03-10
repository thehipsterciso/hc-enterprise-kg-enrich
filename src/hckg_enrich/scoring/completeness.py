"""KGCompletenessScorer — deterministic 5-dimension KG quality scoring.

Pure Python, no LLM calls. Runs in < 100ms on graphs up to 100k entities.
Produces a CompletenessReport used by GapAnalysisAgent and ConvergenceController
to determine whether the graph meets industry standards and what to fix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hckg_enrich.org.profile import OrgProfile

# The 12 canonical entity type layers of an enterprise KG
EXPECTED_LAYERS = frozenset({
    "person", "department", "role", "system", "vendor", "risk",
    "control", "data_asset", "initiative", "location", "network", "jurisdiction",
})

# Fields that indicate a well-populated entity
HIGH_VALUE_FIELDS = frozenset({
    "description", "owner", "responsible_team", "criticality",
    "data_classification", "risk_tier", "tech_stack", "vendor_name",
    "budget", "headcount", "framework", "status",
})

# Scoring dimension weights (must sum to 1.0)
_W_LAYER = 0.30
_W_FIELD = 0.25
_W_DENSITY = 0.20
_W_PROVENANCE = 0.15
_W_CONFIDENCE = 0.10

# Industry-specific required layers
_INDUSTRY_REQUIRED: dict[str, set[str]] = {
    "financial services": {"control", "risk", "jurisdiction"},
    "banking": {"control", "risk", "jurisdiction"},
    "healthcare": {"control", "risk", "data_asset"},
    "insurance": {"control", "risk"},
    "government": {"control", "risk", "jurisdiction"},
}

_REGULATORY_REQUIRED: dict[str, set[str]] = {
    "SOX": {"control"},
    "HIPAA": {"data_asset", "control"},
    "GDPR": {"data_asset", "jurisdiction"},
    "CCPA": {"data_asset"},
    "PCI-DSS": {"control", "system"},
    "FedRAMP": {"control", "system"},
    "CMMC": {"control", "system"},
}


@dataclass
class CompletenessReport:
    """Snapshot of KG completeness across 5 scoring dimensions."""

    overall_score: float = 0.0          # Weighted composite 0.0–1.0
    layer_coverage: float = 0.0         # % of expected entity type layers present
    field_population_rate: float = 0.0  # Avg % of HIGH_VALUE_FIELDS populated
    relationship_density: float = 0.0   # Normalised edges-per-entity score
    provenance_quality: float = 0.0     # % entities with ≥1 URL-backed source
    confidence_quality: float = 0.0     # % enriched entities at T1 or T2

    missing_layers: list[str] = field(default_factory=list)
    underpopulated_layers: list[str] = field(default_factory=list)
    entities_without_sources: list[str] = field(default_factory=list)  # entity IDs
    required_layers_missing: list[str] = field(default_factory=list)   # industry-required

    passes_threshold: bool = False
    threshold: float = 0.80

    total_entities: int = 0
    total_relationships: int = 0
    layers_present: list[str] = field(default_factory=list)
    scored_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class KGCompletenessScorer:
    """Scores a graph dict against a 5-dimension completeness model.

    Instantiate once; call score() as many times as needed (stateless).
    """

    def score(
        self,
        graph: dict[str, Any],
        org_profile: OrgProfile | None = None,
        threshold: float = 0.80,
    ) -> CompletenessReport:
        """Score the graph and return a CompletenessReport."""
        entities: list[dict[str, Any]] = list(graph.get("entities", []))
        relationships: list[dict[str, Any]] = list(graph.get("relationships", []))

        report = CompletenessReport(
            total_entities=len(entities),
            total_relationships=len(relationships),
            threshold=threshold,
        )

        if not entities:
            report.passes_threshold = False
            return report

        # Determine additional required layers from org/regulatory context.
        # The base EXPECTED_LAYERS are already captured by layer_score below.
        # The penalty only applies to layers *additionally* required by org_profile.
        additional_required: set[str] = set()
        if org_profile:
            ind = (org_profile.industry or "").lower()
            for key, layers in _INDUSTRY_REQUIRED.items():
                if key in ind:
                    additional_required.update(layers)
            for regime in org_profile.regulatory_regime:
                additional_required.update(_REGULATORY_REQUIRED.get(regime, set()))

        # -- Dimension 1: Layer coverage --
        present_types: set[str] = {
            str(e.get("entity_type", "")).lower() for e in entities
        }
        present_layers = EXPECTED_LAYERS & present_types
        missing_layers = EXPECTED_LAYERS - present_types
        # Only penalise layers that are specifically mandated by org/regulatory context
        required_missing = additional_required - present_types

        layer_score = len(present_layers) / len(EXPECTED_LAYERS)
        report.layers_present = sorted(present_layers)
        report.missing_layers = sorted(missing_layers)
        report.required_layers_missing = sorted(required_missing)

        # Penalty for missing org-mandated layers
        if required_missing:
            layer_score = max(0.0, layer_score - 0.1 * len(required_missing))

        # -- Underpopulated layers --
        type_counts: dict[str, int] = {}
        for e in entities:
            t = str(e.get("entity_type", "")).lower()
            type_counts[t] = type_counts.get(t, 0) + 1

        # Minimum entity count per layer relative to total graph size
        min_count = max(1, len(entities) // 100)
        report.underpopulated_layers = sorted(
            t for t in present_layers if type_counts.get(t, 0) < min_count
        )

        # -- Dimension 2: Field population --
        field_scores: list[float] = []
        for entity in entities:
            populated = sum(1 for f in HIGH_VALUE_FIELDS if entity.get(f))
            field_scores.append(populated / len(HIGH_VALUE_FIELDS))
        field_pop_rate = sum(field_scores) / len(field_scores) if field_scores else 0.0

        # -- Dimension 3: Relationship density --
        # Enterprise benchmark: ≥2.0 edges per entity
        edges_per_entity = len(relationships) / len(entities) if entities else 0.0
        density_score = min(1.0, edges_per_entity / 2.0)

        # -- Dimension 4: Provenance quality --
        no_sources: list[str] = []
        for entity in entities:
            prov = entity.get("provenance", {})
            has_url = (
                prov.get("source_count", 0) > 0
                or bool(prov.get("source_urls"))
                or bool(prov.get("discovery_method"))
            )
            if not has_url:
                eid = str(entity.get("id", ""))
                if eid:
                    no_sources.append(eid)
        provenance_score = 1.0 - (len(no_sources) / len(entities)) if entities else 0.0
        report.entities_without_sources = no_sources

        # -- Dimension 5: Confidence quality --
        enriched = [e for e in entities if e.get("provenance")]
        t1t2 = sum(
            1 for e in enriched
            if str(e.get("provenance", {}).get("confidence_tier", "T4")) in ("T1", "T2")
        )
        confidence_score = (t1t2 / len(enriched)) if enriched else 0.0

        # -- Weighted composite --
        overall = (
            _W_LAYER * layer_score
            + _W_FIELD * field_pop_rate
            + _W_DENSITY * density_score
            + _W_PROVENANCE * provenance_score
            + _W_CONFIDENCE * confidence_score
        )

        report.layer_coverage = round(layer_score, 4)
        report.field_population_rate = round(field_pop_rate, 4)
        report.relationship_density = round(density_score, 4)
        report.provenance_quality = round(provenance_score, 4)
        report.confidence_quality = round(confidence_score, 4)
        report.overall_score = round(overall, 4)
        report.passes_threshold = overall >= threshold

        return report
