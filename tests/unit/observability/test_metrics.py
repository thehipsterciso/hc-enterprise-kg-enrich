"""Tests for EnrichmentMetrics."""
from __future__ import annotations

import threading

from hckg_enrich.observability.metrics import (
    Counter,
    EnrichmentMetrics,
    Gauge,
    Histogram,
    get_metrics,
    reset_metrics,
)


def test_counter_increments():
    c = Counter(name="test_counter", help="test")
    assert c.value == 0.0
    c.inc()
    assert c.value == 1.0
    c.inc(5.0)
    assert c.value == 6.0


def test_gauge_set_inc_dec():
    g = Gauge(name="test_gauge", help="test")
    g.set(10.0)
    assert g.value == 10.0
    g.inc(3.0)
    assert g.value == 13.0
    g.dec(5.0)
    assert g.value == 8.0


def test_histogram_observe_and_snapshot():
    h = Histogram(name="test_hist", help="test", buckets=(0.1, 0.5, 1.0, 5.0))
    h.observe(0.05)
    h.observe(0.3)
    h.observe(2.0)
    snap = h.snapshot()
    assert snap["count"] == 3
    assert abs(snap["sum"] - 2.35) < 0.001
    assert snap["buckets"][0.1] == 1  # only 0.05 fits under 0.1
    assert snap["buckets"][0.5] == 2  # 0.05 + 0.3
    assert snap["buckets"][5.0] == 3  # all three


def test_metrics_record_entity_result():
    m = EnrichmentMetrics()
    m.record_entity_result("enriched")
    m.record_entity_result("enriched")
    m.record_entity_result("blocked")
    m.record_entity_result("error")
    m.record_entity_result("skipped")
    assert m.entities_total.value == 5.0
    assert m.entities_enriched.value == 2.0
    assert m.entities_blocked.value == 1.0
    assert m.entities_errored.value == 1.0
    assert m.entities_skipped.value == 1.0


def test_metrics_record_agent_duration():
    m = EnrichmentMetrics()
    m.record_agent_duration("reasoning", 1.5)
    m.record_agent_duration("reasoning", 2.5)
    m.record_agent_duration("context", 0.1)
    hists = {h.labels.get("agent"): h for h in m.agent_duration_seconds.all_samples()}
    assert hists["reasoning"].count == 2
    assert hists["context"].count == 1


def test_metrics_record_confidence_tier():
    m = EnrichmentMetrics()
    m.record_confidence_tier("T2")
    m.record_confidence_tier("T2")
    m.record_confidence_tier("T3")
    counters = {c.labels.get("tier"): c for c in m.confidence_tier_total.all_samples()}
    assert counters["T2"].value == 2.0
    assert counters["T3"].value == 1.0


def test_metrics_record_guard_evaluation():
    m = EnrichmentMetrics()
    m.record_guard_evaluation("org-hierarchy-001", "passed")
    m.record_guard_evaluation("org-hierarchy-001", "blocked")
    counters = {
        (c.labels.get("contract"), c.labels.get("result")): c
        for c in m.guard_evaluations_total.all_samples()
    }
    assert counters[("org-hierarchy-001", "passed")].value == 1.0
    assert counters[("org-hierarchy-001", "blocked")].value == 1.0


def test_metrics_to_dict_structure():
    m = EnrichmentMetrics()
    m.record_entity_result("enriched")
    d = m.to_dict()
    assert "entities" in d
    assert d["entities"]["enriched"] == 1.0
    assert "snapshot_at" in d


def test_metrics_to_prometheus_output():
    m = EnrichmentMetrics()
    m.record_entity_result("enriched")
    output = m.to_prometheus()
    assert "hckg_enrich_entities_enriched_total" in output
    assert "# HELP" in output
    assert "# TYPE" in output


def test_metrics_thread_safety():
    m = EnrichmentMetrics()
    errors: list[Exception] = []

    def worker() -> None:
        for _ in range(100):
            try:
                m.record_entity_result("enriched")
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert m.entities_total.value == 1000.0


def test_get_metrics_singleton():
    reset_metrics()
    m1 = get_metrics()
    m2 = get_metrics()
    assert m1 is m2
    reset_metrics()
