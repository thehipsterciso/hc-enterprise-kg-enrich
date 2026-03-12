"""EnrichmentController — orchestrates the 7-agent enrichment pipeline.

Pipeline stages (v0.3.0):
  1. PrioritizationAgent — ranks entities by enrichment value (run-level, once)
  2. ContextAgent        — KG subgraph retrieval
  3. SearchAgent         — web search grounding (optional)
  4. ReasoningAgent      — LLM-powered proposal generation
  5. ConfidenceAgent     — evidence-based T1–T4 tier assignment
  6. CoherenceAgent      — GraphGuard semantic contract validation
  7. CommitAgent         — applies enrichments with full provenance

Enterprise additions:
  - EnrichmentRun tracks the full session (config, model, stats)
  - AuditLog captures every enrichment event as JSONL
  - EnrichmentMetrics exposes Prometheus-compatible pipeline stats
  - EnrichmentTracer records OpenTelemetry-compatible spans per entity
  - run_id is threaded through every AgentMessage payload
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from hckg_enrich.agents.base import AgentMessage, AgentRole
from hckg_enrich.agents.coherence_agent import CoherenceAgent
from hckg_enrich.agents.commit_agent import CommitAgent
from hckg_enrich.agents.confidence_agent import ConfidenceAgent
from hckg_enrich.agents.context_agent import ContextAgent
from hckg_enrich.agents.prioritization_agent import PrioritizationAgent
from hckg_enrich.agents.reasoning_agent import ReasoningAgent
from hckg_enrich.agents.search_agent import SearchAgent
from hckg_enrich.context.retriever import KGContextRetriever
from hckg_enrich.guard.contracts.org_hierarchy import OrgHierarchyContract
from hckg_enrich.guard.contracts.system_ownership import SystemOwnershipContract
from hckg_enrich.guard.contracts.vendor_relationship import VendorRelationshipContract
from hckg_enrich.guard.guardian import EnrichmentGuardian
from hckg_enrich.observability.metrics import EnrichmentMetrics
from hckg_enrich.observability.tracer import EnrichmentTracer
from hckg_enrich.provenance.audit_log import AuditEvent, AuditEventType, AuditLog
from hckg_enrich.provenance.run import EnrichmentRun
from hckg_enrich.providers.base import LLMProvider, SearchProvider

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.3.0"


@dataclass
class EnrichmentStats:
    total_entities: int = 0
    enriched: int = 0
    skipped: int = 0
    blocked: int = 0
    errors: int = 0
    relationships_added: int = 0
    changes: list[str] = field(default_factory=list)


@dataclass
class ProgressEvent:
    """Emitted by enrich_all_streaming() to report pipeline progress."""

    type: str  # "started" | "entity_started" | "entity_done" | "completed" | "error"
    entity_id: str | None = None
    total: int | None = None
    completed: int | None = None
    result: dict[str, Any] | None = None
    stats: EnrichmentStats | None = None


class EnrichmentController:
    """Orchestrates the full 7-agent enrichment pipeline for a knowledge graph.

    Instantiate once per graph file. The controller creates a new EnrichmentRun
    for each call to enrich_all / enrich_all_streaming, which groups all activity
    from that execution session under a single run_id.
    """

    def __init__(
        self,
        graph: dict[str, Any],
        llm: LLMProvider,
        search: SearchProvider | None = None,
        concurrency: int = 5,
        audit_log_path: str | None = None,
        llm_model: str = "claude-opus-4-5",
        llm_provider: str = "anthropic",
        search_provider: str | None = None,
        extra_contracts: list[Any] | None = None,
    ) -> None:
        self._graph = graph
        self._concurrency = concurrency
        self._llm_model = llm_model
        self._org_profile: dict | None = None  # set per-run by enrich_all_streaming
        self._llm_provider = llm_provider
        self._search_provider = search_provider

        # Audit log (JSONL backend)
        self._audit_log: AuditLog | None = None
        if audit_log_path:
            self._audit_log = AuditLog(path=audit_log_path)

        # Observability
        self._metrics = EnrichmentMetrics()
        self._tracer = EnrichmentTracer(service_name="hckg-enrich")

        # Build contracts list (3 core + any extras)
        contracts = [
            OrgHierarchyContract(llm),
            SystemOwnershipContract(llm),
            VendorRelationshipContract(llm),
        ]
        if extra_contracts:
            contracts.extend(extra_contracts)

        retriever = KGContextRetriever(graph)
        guardian = EnrichmentGuardian(contracts=contracts)

        self._prioritization_agent = PrioritizationAgent()
        self._context_agent = ContextAgent(retriever)
        self._search_agent = SearchAgent(search)
        self._reasoning_agent = ReasoningAgent(llm)
        self._confidence_agent = ConfidenceAgent()
        self._coherence_agent = CoherenceAgent(guardian)
        self._commit_agent = CommitAgent(graph, audit_log=self._audit_log)

    @property
    def metrics(self) -> EnrichmentMetrics:
        return self._metrics

    @property
    def tracer(self) -> EnrichmentTracer:
        return self._tracer

    async def enrich_entity(
        self,
        entity_id: str,
        run_id: str = "",
    ) -> dict[str, Any]:
        """Run the full 7-agent pipeline for a single entity."""
        start = time.monotonic()
        self._metrics.active_pipelines.inc()

        org_profile = getattr(self, "_org_profile", None)
        msg = AgentMessage(
            sender=AgentRole.CONTEXT,
            recipient=AgentRole.CONTEXT,
            payload={
                "entity_id": entity_id,
                "run_id": run_id,
                "llm_model": self._llm_model,
                "llm_provider": self._llm_provider,
                **({"org_profile": org_profile} if org_profile else {}),
            },
        )

        try:
            async with self._tracer.span(
                "entity_pipeline",
                trace_id=run_id or entity_id,
                entity_id=entity_id,
                run_id=run_id,
            ) as trace_ctx:
                trace_ctx.set_attribute("entity_id", entity_id)

                # Stage 1: Context retrieval
                async with self._tracer.span(
                    "context_agent", trace_id=run_id or entity_id
                ) as ctx:
                    t = time.monotonic()
                    msg = await self._context_agent.run(msg)
                    self._metrics.record_agent_duration("context", time.monotonic() - t)

                # Stage 2: Web search
                async with self._tracer.span(
                    "search_agent", trace_id=run_id or entity_id
                ) as ctx:
                    t = time.monotonic()
                    msg = await self._search_agent.run(msg)
                    self._metrics.record_agent_duration("search", time.monotonic() - t)

                # Stage 3: LLM reasoning
                async with self._tracer.span(
                    "reasoning_agent", trace_id=run_id or entity_id
                ) as ctx:
                    t = time.monotonic()
                    msg = await self._reasoning_agent.run(msg)
                    ctx.set_attribute("llm_model", self._llm_model)
                    self._metrics.record_agent_duration("reasoning", time.monotonic() - t)
                    self._metrics.record_llm_call(
                        provider=self._llm_provider, model=self._llm_model
                    )

                # Stage 4: Confidence tier assignment
                async with self._tracer.span(
                    "confidence_agent", trace_id=run_id or entity_id
                ):
                    t = time.monotonic()
                    msg = await self._confidence_agent.run(msg)
                    self._metrics.record_agent_duration("confidence", time.monotonic() - t)

                # Stage 5: GraphGuard coherence
                async with self._tracer.span(
                    "coherence_agent", trace_id=run_id or entity_id
                ):
                    t = time.monotonic()
                    msg = await self._coherence_agent.run(msg)
                    self._metrics.record_agent_duration("coherence", time.monotonic() - t)

                # Stage 6: Commit
                async with self._tracer.span(
                    "commit_agent", trace_id=run_id or entity_id
                ):
                    t = time.monotonic()
                    msg = await self._commit_agent.run(msg)
                    self._metrics.record_agent_duration("commit", time.monotonic() - t)

            result = dict(msg.payload.get("commit_result", {}))
            duration = time.monotonic() - start
            self._metrics.pipeline_duration_seconds.observe(duration)

            # Record outcome in metrics
            tier = str(result.get("confidence_tier", "T4"))
            if result.get("error"):
                self._metrics.record_entity_result("error")
            elif result.get("applied"):
                self._metrics.record_entity_result("enriched")
                self._metrics.relationships_added.inc(result.get("relationships_added", 0))
                self._metrics.record_confidence_tier(tier)
            elif result.get("reason") == "Blocked by GraphGuard":
                self._metrics.record_entity_result("blocked")
            else:
                self._metrics.record_entity_result("skipped")

            return result

        except Exception as e:
            logger.error("Pipeline failed for entity %s: %s", entity_id, e)
            self._metrics.record_entity_result("error")
            return {"entity_id": entity_id, "applied": False, "error": str(e)}
        finally:
            self._metrics.active_pipelines.dec()

    async def enrich_all(
        self,
        entity_type: str | None = None,
        limit: int | None = None,
        graph_path: str = "",
        entity_ids: list[str] | None = None,
        org_profile: dict | None = None,
    ) -> EnrichmentRun:
        """Enrich all (or filtered) entities with bounded concurrency.

        Args:
            entity_ids: When provided, only enrich these specific entity IDs
                (used by ConvergenceController to target gap-analysis priorities).
            org_profile: When provided, thread OrgProfile dict through every
                agent message payload for organisational grounding.

        Returns a completed EnrichmentRun with full session statistics.
        """
        run = EnrichmentRun(
            graph_path=graph_path,
            entity_type_filter=entity_type,
            limit=limit,
            concurrency=self._concurrency,
            llm_provider=self._llm_provider,
            llm_model=self._llm_model,
            search_provider=self._search_provider,
        )
        stats = EnrichmentStats()
        async for event in self.enrich_all_streaming(
            entity_type=entity_type, limit=limit, run=run,
            entity_ids=entity_ids, org_profile=org_profile,
        ):
            if event.type == "completed" and event.stats:
                stats = event.stats

        run.complete(
            total=stats.total_entities,
            enriched=stats.enriched,
            blocked=stats.blocked,
            skipped=stats.skipped,
            errors=stats.errors,
            relationships_added=stats.relationships_added,
        )

        # Emit RUN_COMPLETED audit event
        if self._audit_log:
            self._audit_log.append(
                AuditEvent(
                    event_type=AuditEventType.RUN_COMPLETED,
                    run_id=run.run_id,
                    entity_id="",
                    entity_name="",
                    entity_type="",
                    agent_role="controller",
                    pipeline_version=PIPELINE_VERSION,
                    llm_model=self._llm_model,
                    attribute_changes=[],
                    relationships_added=[],
                    confidence_tier="",
                    guard_contracts_passed=[],
                    guard_warnings=[],
                    guard_blocking_failures=[],
                    reasoning="",
                    search_source_count=0,
                    metadata=run.to_dict(),
                )
            )

        return run

    async def enrich_all_streaming(
        self,
        entity_type: str | None = None,
        limit: int | None = None,
        run: EnrichmentRun | None = None,
        entity_ids: list[str] | None = None,
        org_profile: dict | None = None,
    ) -> AsyncIterator[ProgressEvent]:
        """Enrich entities and yield ProgressEvents as each completes.

        Yields:
            ProgressEvent(type="started", total=N)
            ProgressEvent(type="entity_started", entity_id=..., completed=i, total=N)
            ProgressEvent(type="entity_done",    entity_id=..., completed=i, total=N, result=...)
            ProgressEvent(type="completed", stats=EnrichmentStats)
        """
        # Store org_profile for threading through entity pipelines
        self._org_profile = org_profile

        # Prioritize entities before the main loop
        entities: list[dict[str, Any]] = list(self._graph.get("entities", []))
        relationships: list[dict[str, Any]] = list(self._graph.get("relationships", []))

        # Filter to specific entity IDs when provided by ConvergenceController
        if entity_ids is not None:
            id_set = set(entity_ids)
            entities = [e for e in entities if str(e.get("id", "")) in id_set]
            # Preserve the priority ordering from entity_ids
            entity_order = {eid: i for i, eid in enumerate(entity_ids)}
            entities.sort(key=lambda e: entity_order.get(str(e.get("id", "")), 9999))

        priority_msg = AgentMessage(
            sender=AgentRole.CONTEXT,
            recipient=AgentRole.CONTEXT,
            payload={
                "entities": entities,
                "relationships": relationships,
                "entity_type_filter": entity_type,
                "limit": limit,
            },
        )
        priority_msg = await self._prioritization_agent.run(priority_msg)
        entities = list(priority_msg.payload.get("prioritized_entities", entities))

        total = len(entities)
        run_id = run.run_id if run else ""
        stats = EnrichmentStats(total_entities=total)

        # Emit RUN_STARTED audit event
        if self._audit_log and run_id:
            self._audit_log.append(
                AuditEvent(
                    event_type=AuditEventType.RUN_STARTED,
                    run_id=run_id,
                    entity_id="",
                    entity_name="",
                    entity_type="",
                    agent_role="controller",
                    pipeline_version=PIPELINE_VERSION,
                    llm_model=self._llm_model,
                    attribute_changes=[],
                    relationships_added=[],
                    confidence_tier="",
                    guard_contracts_passed=[],
                    guard_warnings=[],
                    guard_blocking_failures=[],
                    reasoning="",
                    search_source_count=0,
                    metadata={"entity_count": total, "entity_type_filter": entity_type},
                )
            )

        run_start = time.monotonic()
        yield ProgressEvent(type="started", total=total)

        sem = asyncio.Semaphore(self._concurrency)
        queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()

        async def enrich_one(entity: dict[str, Any]) -> None:
            eid = str(entity["id"])
            async with sem:
                await queue.put(ProgressEvent(type="entity_started", entity_id=eid))
                result = await self.enrich_entity(eid, run_id=run_id)
                await queue.put(
                    ProgressEvent(type="entity_done", entity_id=eid, result=result)
                )

        tasks = [asyncio.create_task(enrich_one(e)) for e in entities]
        done_count = 0

        while done_count < total:
            event = await queue.get()
            if event.type == "entity_done":
                done_count += 1
                result = event.result or {}
                event.completed = done_count
                event.total = total
                if result.get("error"):
                    stats.errors += 1
                elif result.get("applied"):
                    stats.enriched += 1
                    stats.relationships_added += result.get("relationships_added", 0)
                    stats.changes.extend(result.get("changes", []))
                elif result.get("reason") == "Blocked by GraphGuard":
                    stats.blocked += 1
                else:
                    stats.skipped += 1
            yield event

        await asyncio.gather(*tasks)
        self._metrics.run_duration_seconds.observe(time.monotonic() - run_start)
        yield ProgressEvent(type="completed", stats=stats)
