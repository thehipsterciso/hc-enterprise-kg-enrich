"""KG completeness scoring and gap analysis."""
from hckg_enrich.scoring.completeness import CompletenessReport, KGCompletenessScorer
from hckg_enrich.scoring.gap_analysis import GapAnalysisAgent, GapItem, GapReport

__all__ = [
    "KGCompletenessScorer", "CompletenessReport",
    "GapAnalysisAgent", "GapItem", "GapReport",
]
