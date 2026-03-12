"""Observability metrics for the enrichment pipeline.

Zero external dependencies. Prometheus-compatible text exposition format.
Thread-safe via threading.Lock. OpenTelemetry-compatible labels.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Primitive metric types
# ---------------------------------------------------------------------------


@dataclass
class Counter:
    """Monotonically increasing counter."""

    name: str
    help: str
    labels: dict[str, str] = field(default_factory=dict)
    _value: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        return self._value


@dataclass
class Gauge:
    """Current value that can go up or down."""

    name: str
    help: str
    labels: dict[str, str] = field(default_factory=dict)
    _value: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        return self._value


@dataclass
class Histogram:
    """Distribution of values with configurable buckets."""

    name: str
    help: str
    buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    labels: dict[str, str] = field(default_factory=dict)
    _sum: float = field(default=0.0, init=False)
    _count: int = field(default=0, init=False)
    _buckets: dict[float, int] = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._buckets = {b: 0 for b in self.buckets}

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._count += 1
            for b in self.buckets:
                if value <= b:
                    self._buckets[b] += 1

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "count": self._count,
                "sum": self._sum,
                "mean": self.mean,
                "buckets": dict(self._buckets),
            }


# ---------------------------------------------------------------------------
# LabeledMetric — family of metrics keyed by label values
# ---------------------------------------------------------------------------


class LabeledCounterFamily:
    """Counter family keyed by label tuple."""

    def __init__(self, name: str, help: str, label_names: list[str]) -> None:
        self.name = name
        self.help = help
        self.label_names = label_names
        self._counters: dict[tuple[str, ...], Counter] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs: str) -> Counter:
        key = tuple(kwargs.get(n, "") for n in self.label_names)
        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter(
                    name=self.name,
                    help=self.help,
                    labels=dict(zip(self.label_names, key, strict=False)),
                )
        return self._counters[key]

    def all_samples(self) -> list[Counter]:
        with self._lock:
            return list(self._counters.values())


class LabeledHistogramFamily:
    """Histogram family keyed by label tuple."""

    def __init__(
        self,
        name: str,
        help: str,
        label_names: list[str],
        buckets: tuple[float, ...] = (0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
    ) -> None:
        self.name = name
        self.help = help
        self.label_names = label_names
        self.buckets = buckets
        self._histograms: dict[tuple[str, ...], Histogram] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs: str) -> Histogram:
        key = tuple(kwargs.get(n, "") for n in self.label_names)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram(
                    name=self.name,
                    help=self.help,
                    buckets=self.buckets,
                    labels=dict(zip(self.label_names, key, strict=False)),
                )
        return self._histograms[key]

    def all_samples(self) -> list[Histogram]:
        with self._lock:
            return list(self._histograms.values())


# ---------------------------------------------------------------------------
# EnrichmentMetrics — the canonical pipeline metrics registry
# ---------------------------------------------------------------------------


class EnrichmentMetrics:
    """Thread-safe metrics registry for the enrichment pipeline.

    Tracks entities processed, agent durations, LLM/search call volumes,
    GraphGuard evaluation results, and confidence tier distribution.

    Exports Prometheus-compatible text format and JSON snapshot.
    """

    def __init__(self) -> None:
        # Entity-level counters
        self.entities_total = Counter(
            "hckg_enrich_entities_total",
            "Total entities processed by the pipeline",
        )
        self.entities_enriched = Counter(
            "hckg_enrich_entities_enriched_total",
            "Entities successfully enriched and committed",
        )
        self.entities_blocked = Counter(
            "hckg_enrich_entities_blocked_total",
            "Entities blocked by GraphGuard contracts",
        )
        self.entities_skipped = Counter(
            "hckg_enrich_entities_skipped_total",
            "Entities skipped (no changes proposed)",
        )
        self.entities_errored = Counter(
            "hckg_enrich_entities_errored_total",
            "Entities that encountered pipeline errors",
        )
        self.relationships_added = Counter(
            "hckg_enrich_relationships_added_total",
            "New relationships committed to the graph",
        )

        # Agent duration histograms
        self.agent_duration_seconds = LabeledHistogramFamily(
            "hckg_enrich_agent_duration_seconds",
            "Time spent in each agent stage",
            label_names=["agent"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )

        # Pipeline-level duration
        self.pipeline_duration_seconds = Histogram(
            "hckg_enrich_pipeline_duration_seconds",
            "End-to-end pipeline duration per entity",
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
        )
        self.run_duration_seconds = Histogram(
            "hckg_enrich_run_duration_seconds",
            "Total enrichment run duration",
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
        )

        # LLM and search call counters
        self.llm_calls_total = LabeledCounterFamily(
            "hckg_enrich_llm_calls_total",
            "Total LLM API calls",
            label_names=["provider", "model", "status"],
        )
        self.search_calls_total = LabeledCounterFamily(
            "hckg_enrich_search_calls_total",
            "Total web search API calls",
            label_names=["provider", "status"],
        )
        self.llm_tokens_total = LabeledCounterFamily(
            "hckg_enrich_llm_tokens_total",
            "Total LLM tokens (input+output)",
            label_names=["provider", "model", "token_type"],
        )

        # GraphGuard contract evaluations
        self.guard_evaluations_total = LabeledCounterFamily(
            "hckg_enrich_guard_evaluations_total",
            "Total GraphGuard contract evaluations",
            label_names=["contract", "result"],  # result: passed|blocked|warned|error
        )

        # Confidence tier distribution
        self.confidence_tier_total = LabeledCounterFamily(
            "hckg_enrich_confidence_tier_total",
            "Distribution of confidence tiers assigned",
            label_names=["tier"],  # T1, T2, T3, T4
        )

        # Active concurrency gauge
        self.active_pipelines = Gauge(
            "hckg_enrich_active_pipelines",
            "Number of entity pipelines currently executing",
        )

        self._created_at: float = time.time()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def record_entity_result(self, status: str) -> None:
        """status: 'enriched' | 'blocked' | 'skipped' | 'error'"""
        self.entities_total.inc()
        if status == "enriched":
            self.entities_enriched.inc()
        elif status == "blocked":
            self.entities_blocked.inc()
        elif status == "error":
            self.entities_errored.inc()
        else:
            self.entities_skipped.inc()

    def record_agent_duration(self, agent: str, duration: float) -> None:
        self.agent_duration_seconds.labels(agent=agent).observe(duration)

    def record_llm_call(self, provider: str, model: str, status: str = "success") -> None:
        self.llm_calls_total.labels(provider=provider, model=model, status=status).inc()

    def record_search_call(self, provider: str, status: str = "success") -> None:
        self.search_calls_total.labels(provider=provider, status=status).inc()

    def record_guard_evaluation(self, contract: str, result: str) -> None:
        self.guard_evaluations_total.labels(contract=contract, result=result).inc()

    def record_confidence_tier(self, tier: str) -> None:
        self.confidence_tier_total.labels(tier=tier).inc()

    # ------------------------------------------------------------------
    # Export formats
    # ------------------------------------------------------------------

    def to_prometheus(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines: list[str] = []

        def _counter(c: Counter) -> str:
            label_str = ",".join(f'{k}="{v}"' for k, v in c.labels.items())
            suffix = f"{{{label_str}}}" if label_str else ""
            return f"{c.name}{suffix} {c.value}"

        def _histogram(h: Histogram, extra_labels: dict[str, str] | None = None) -> list[str]:
            base_labels = {**(extra_labels or {}), **h.labels}
            label_str = ",".join(f'{k}="{v}"' for k, v in base_labels.items())
            prefix = f"{{{label_str}}}" if label_str else ""
            out = []
            snap = h.snapshot()
            for le, cnt in snap["buckets"].items():
                le_str = '+Inf' if le == float("inf") else str(le)
                sep = "," if label_str else ""
                suffix = ("," + label_str) if label_str else ""
                out.append(
                    f'{h.name}_bucket{{{sep}le="{le_str}"{suffix}}}{cnt}'
                )
            out.append(f"{h.name}_sum{prefix} {snap['sum']}")
            out.append(f"{h.name}_count{prefix} {snap['count']}")
            return out

        simple_counters = [
            self.entities_total, self.entities_enriched, self.entities_blocked,
            self.entities_skipped, self.entities_errored, self.relationships_added,
        ]
        for c in simple_counters:
            lines.append(f"# HELP {c.name} {c.help}")
            lines.append(f"# TYPE {c.name} counter")
            lines.append(_counter(c))

        lines.append(f"# HELP {self.active_pipelines.name} {self.active_pipelines.help}")
        lines.append(f"# TYPE {self.active_pipelines.name} gauge")
        lines.append(f"{self.active_pipelines.name} {self.active_pipelines.value}")

        for hist in [self.pipeline_duration_seconds, self.run_duration_seconds]:
            lines.append(f"# HELP {hist.name} {hist.help}")
            lines.append(f"# TYPE {hist.name} histogram")
            lines.extend(_histogram(hist))

        for family in [
            self.agent_duration_seconds,
        ]:
            for h in family.all_samples():
                lines.extend(_histogram(h))

        for family in [
            self.llm_calls_total, self.search_calls_total, self.llm_tokens_total,
            self.guard_evaluations_total, self.confidence_tier_total,
        ]:
            for c in family.all_samples():
                lines.append(_counter(c))

        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Snapshot all metrics as a JSON-serializable dict."""
        return {
            "created_at": self._created_at,
            "snapshot_at": time.time(),
            "entities": {
                "total": self.entities_total.value,
                "enriched": self.entities_enriched.value,
                "blocked": self.entities_blocked.value,
                "skipped": self.entities_skipped.value,
                "errored": self.entities_errored.value,
                "relationships_added": self.relationships_added.value,
            },
            "active_pipelines": self.active_pipelines.value,
            "pipeline_duration_seconds": self.pipeline_duration_seconds.snapshot(),
            "run_duration_seconds": self.run_duration_seconds.snapshot(),
            "agent_durations": {
                str(h.labels): h.snapshot()
                for h in self.agent_duration_seconds.all_samples()
            },
            "llm_calls": {
                str(c.labels): c.value
                for c in self.llm_calls_total.all_samples()
            },
            "search_calls": {
                str(c.labels): c.value
                for c in self.search_calls_total.all_samples()
            },
            "guard_evaluations": {
                str(c.labels): c.value
                for c in self.guard_evaluations_total.all_samples()
            },
            "confidence_tiers": {
                str(c.labels): c.value
                for c in self.confidence_tier_total.all_samples()
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton (opt-in — controller can also instantiate its own)
# ---------------------------------------------------------------------------

_default_metrics: EnrichmentMetrics | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> EnrichmentMetrics:
    """Return the module-level default metrics registry (created on first call)."""
    global _default_metrics
    with _metrics_lock:
        if _default_metrics is None:
            _default_metrics = EnrichmentMetrics()
    return _default_metrics


def reset_metrics() -> None:
    """Reset the module-level registry. Primarily for testing."""
    global _default_metrics
    with _metrics_lock:
        _default_metrics = None
