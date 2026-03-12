"""Tests for KGCompletenessScorer — pure Python, no mocks needed."""
from __future__ import annotations

from hckg_enrich.org.profile import OrgProfile
from hckg_enrich.scoring.completeness import (
    EXPECTED_LAYERS,
    KGCompletenessScorer,
)


def _entity(eid: str, etype: str, **fields) -> dict:
    return {"id": eid, "entity_type": etype, "name": f"Test {eid}", **fields}


def _rel(src: str, tgt: str, rtype: str = "works_in") -> dict:
    return {"id": f"r-{src}-{tgt}", "relationship_type": rtype,
            "source_id": src, "target_id": tgt}


def _graph(entities: list[dict], relationships: list[dict] | None = None) -> dict:
    return {"entities": entities, "relationships": relationships or []}


class TestEmptyGraph:
    def test_empty_graph_score_zero(self):
        scorer = KGCompletenessScorer()
        report = scorer.score(_graph([]))
        assert report.overall_score == 0.0
        assert not report.passes_threshold

    def test_empty_graph_totals_zero(self):
        scorer = KGCompletenessScorer()
        report = scorer.score(_graph([]))
        assert report.total_entities == 0
        assert report.total_relationships == 0


class TestLayerCoverage:
    def test_single_layer_partial_coverage(self):
        scorer = KGCompletenessScorer()
        g = _graph([_entity("p1", "person")])
        report = scorer.score(g)
        expected = 1 / len(EXPECTED_LAYERS)
        assert abs(report.layer_coverage - expected) < 0.01

    def test_all_12_layers_full_coverage(self):
        scorer = KGCompletenessScorer()
        entities = [_entity(f"e{i}", layer) for i, layer in enumerate(EXPECTED_LAYERS)]
        g = _graph(entities)
        report = scorer.score(g)
        assert report.layer_coverage == 1.0

    def test_missing_layers_reported(self):
        scorer = KGCompletenessScorer()
        g = _graph([_entity("p1", "person"), _entity("s1", "system")])
        report = scorer.score(g)
        assert "department" in report.missing_layers
        assert "vendor" in report.missing_layers

    def test_present_layers_not_in_missing(self):
        scorer = KGCompletenessScorer()
        g = _graph([_entity("p1", "person")])
        report = scorer.score(g)
        assert "person" not in report.missing_layers

    def test_layers_present_list_accurate(self):
        scorer = KGCompletenessScorer()
        g = _graph([_entity("p1", "person"), _entity("d1", "department")])
        report = scorer.score(g)
        assert "person" in report.layers_present
        assert "department" in report.layers_present


class TestFieldPopulation:
    def test_no_high_value_fields_low_score(self):
        scorer = KGCompletenessScorer()
        g = _graph([_entity("p1", "person")])
        report = scorer.score(g)
        assert report.field_population_rate == 0.0

    def test_all_high_value_fields_high_score(self):
        scorer = KGCompletenessScorer()
        entity = _entity(
            "p1", "person",
            description="Chief Executive Officer",
            owner="Board",
            responsible_team="Executive",
            criticality="HIGH",
            data_classification="Confidential",
            risk_tier="T1",
            tech_stack="N/A",
            vendor_name="N/A",
            budget="$5M",
            headcount="1",
            framework="NIST",
            status="Active",
        )
        g = _graph([entity])
        report = scorer.score(g)
        assert report.field_population_rate == 1.0

    def test_partial_field_population(self):
        scorer = KGCompletenessScorer()
        entity = _entity("p1", "person", description="CEO", owner="Board")
        g = _graph([entity])
        report = scorer.score(g)
        assert 0.0 < report.field_population_rate < 1.0


class TestRelationshipDensity:
    def test_no_relationships_zero_density(self):
        scorer = KGCompletenessScorer()
        g = _graph([_entity("p1", "person"), _entity("d1", "department")])
        report = scorer.score(g)
        assert report.relationship_density == 0.0

    def test_two_edges_per_entity_full_density(self):
        scorer = KGCompletenessScorer()
        entities = [_entity("p1", "person"), _entity("p2", "person")]
        rels = [
            _rel("p1", "p2"),
            _rel("p2", "p1"),
            _rel("p1", "p2", "manages"),
            _rel("p2", "p1", "reports_to"),
        ]
        g = _graph(entities, rels)
        report = scorer.score(g)
        assert report.relationship_density == 1.0


class TestProvenanceQuality:
    def test_no_provenance_zero_quality(self):
        scorer = KGCompletenessScorer()
        g = _graph([_entity("p1", "person")])
        report = scorer.score(g)
        assert report.provenance_quality == 0.0

    def test_entity_with_source_urls_counted(self):
        scorer = KGCompletenessScorer()
        entity = _entity("p1", "person")
        entity["provenance"] = {"source_urls": ["https://example.com"], "source_count": 1}
        g = _graph([entity])
        report = scorer.score(g)
        assert report.provenance_quality == 1.0

    def test_entity_with_discovery_method_counted(self):
        scorer = KGCompletenessScorer()
        entity = _entity("p1", "person")
        entity["provenance"] = {"discovery_method": "entity_discovery_agent"}
        g = _graph([entity])
        report = scorer.score(g)
        assert report.provenance_quality == 1.0

    def test_entities_without_sources_listed(self):
        scorer = KGCompletenessScorer()
        e1 = _entity("p1", "person")
        e2 = _entity("p2", "person")
        e2["provenance"] = {"source_urls": ["https://x.com"], "source_count": 1}
        g = _graph([e1, e2])
        report = scorer.score(g)
        assert "p1" in report.entities_without_sources
        assert "p2" not in report.entities_without_sources

    def test_mixed_provenance_partial_quality(self):
        scorer = KGCompletenessScorer()
        e1 = _entity("p1", "person")
        e1["provenance"] = {"source_urls": ["https://x.com"], "source_count": 1}
        e2 = _entity("p2", "person")  # no provenance
        g = _graph([e1, e2])
        report = scorer.score(g)
        assert report.provenance_quality == 0.5


class TestConfidenceQuality:
    def test_no_enriched_entities_zero_quality(self):
        scorer = KGCompletenessScorer()
        g = _graph([_entity("p1", "person")])
        report = scorer.score(g)
        assert report.confidence_quality == 0.0

    def test_t1_t2_entities_high_quality(self):
        scorer = KGCompletenessScorer()
        e1 = _entity("p1", "person")
        e1["provenance"] = {"confidence_tier": "T1"}
        e2 = _entity("p2", "person")
        e2["provenance"] = {"confidence_tier": "T2"}
        g = _graph([e1, e2])
        report = scorer.score(g)
        assert report.confidence_quality == 1.0

    def test_t4_entities_low_quality(self):
        scorer = KGCompletenessScorer()
        e = _entity("p1", "person")
        e["provenance"] = {"confidence_tier": "T4"}
        g = _graph([e])
        report = scorer.score(g)
        assert report.confidence_quality == 0.0


class TestThreshold:
    def test_passes_threshold_when_score_above(self):
        scorer = KGCompletenessScorer()
        # Full provenance, all high-value fields, dense rels, all layers
        entities = [
            {**_entity(f"e{i}", layer),
             "description": "x", "owner": "y", "responsible_team": "z",
             "criticality": "HIGH", "data_classification": "C",
             "risk_tier": "T1", "tech_stack": "AWS", "vendor_name": "AWS",
             "budget": "$1M", "headcount": "10", "framework": "NIST", "status": "Active",
             "provenance": {"source_urls": ["https://x.com"], "source_count": 1,
                            "confidence_tier": "T1"}}
            for i, layer in enumerate(EXPECTED_LAYERS)
        ]
        rels = [_rel(f"e{i}", f"e{j}") for i in range(min(5, len(entities)))
                for j in range(min(5, len(entities))) if i != j]
        g = _graph(entities, rels)
        report = scorer.score(g, threshold=0.5)
        assert report.passes_threshold

    def test_fails_threshold_when_score_below(self):
        scorer = KGCompletenessScorer()
        g = _graph([_entity("p1", "person")])
        report = scorer.score(g, threshold=0.8)
        assert not report.passes_threshold

    def test_threshold_stored_in_report(self):
        scorer = KGCompletenessScorer()
        report = scorer.score(_graph([_entity("p1", "person")]), threshold=0.7)
        assert report.threshold == 0.7


class TestIndustryAdjustment:
    def test_financial_services_requires_control_risk_jurisdiction(self):
        scorer = KGCompletenessScorer()
        org = OrgProfile(industry="financial services")
        g = _graph([_entity("p1", "person")])
        report = scorer.score(g, org_profile=org)
        # control, risk, jurisdiction should be in required_layers_missing
        for layer in ("control", "risk", "jurisdiction"):
            assert layer in report.required_layers_missing

    def test_sox_requires_control(self):
        scorer = KGCompletenessScorer()
        org = OrgProfile(regulatory_regime=["SOX"])
        g = _graph([_entity("p1", "person")])
        report = scorer.score(g, org_profile=org)
        assert "control" in report.required_layers_missing

    def test_hipaa_requires_data_asset_and_control(self):
        scorer = KGCompletenessScorer()
        org = OrgProfile(regulatory_regime=["HIPAA"])
        g = _graph([_entity("p1", "person")])
        report = scorer.score(g, org_profile=org)
        assert "data_asset" in report.required_layers_missing
        assert "control" in report.required_layers_missing

    def test_no_org_profile_uses_base_expected_layers(self):
        scorer = KGCompletenessScorer()
        g = _graph([_entity("p1", "person")])
        report = scorer.score(g, org_profile=None)
        assert report.required_layers_missing == []
