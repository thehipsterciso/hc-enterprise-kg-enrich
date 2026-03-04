"""EnrichmentController — orchestrates the 5-agent enrichment pipeline."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from hckg_enrich.agents.base import AgentMessage, AgentRole
from hckg_enrich.agents.coherence_agent import CoherenceAgent
from hckg_enrich.agents.commit_agent import CommitAgent
from hckg_enrich.agents.context_agent import ContextAgent
from hckg_enrich.agents.reasoning_agent import ReasoningAgent
from hckg_enrich.agents.search_agent import SearchAgent
from hckg_enrich.context.retriever import KGContextRetriever
from hckg_enrich.guard.contracts.org_hierarchy import OrgHierarchyContract
from hckg_enrich.guard.contracts.system_ownership import SystemOwnershipContract
from hckg_enrich.guard.contracts.vendor_relationship import VendorRelationshipContract
from hckg_enrich.guard.guardian import EnrichmentGuardian
from hckg_enrich.providers.base import LLMProvider, SearchProvider

logger = logging.getLogger(__name__)


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
    """Orchestrates the full 5-agent enrichment pipeline for a knowledge graph."""

    def __init__(
        self,
        graph: dict[str, Any],
        llm: LLMProvider,
        search: SearchProvider | None = None,
        concurrency: int = 5,
    ) -> None:
        self._graph = graph
        self._concurrency = concurrency

        retriever = KGContextRetriever(graph)
        guardian = EnrichmentGuardian(
            contracts=[
                OrgHierarchyContract(llm),
                SystemOwnershipContract(llm),
                VendorRelationshipContract(llm),
            ]
        )

        self._context_agent = ContextAgent(retriever)
        self._search_agent = SearchAgent(search)
        self._reasoning_agent = ReasoningAgent(llm)
        self._coherence_agent = CoherenceAgent(guardian)
        self._commit_agent = CommitAgent(graph)

    async def enrich_entity(self, entity_id: str) -> dict[str, Any]:
        """Run the full pipeline for a single entity."""
        msg = AgentMessage(
            sender=AgentRole.CONTEXT,
            recipient=AgentRole.CONTEXT,
            payload={"entity_id": entity_id},
        )
        try:
            msg = await self._context_agent.run(msg)
            msg = await self._search_agent.run(msg)
            msg = await self._reasoning_agent.run(msg)
            msg = await self._coherence_agent.run(msg)
            msg = await self._commit_agent.run(msg)
            return dict(msg.payload.get("commit_result", {}))
        except Exception as e:
            logger.error(f"Pipeline failed for entity {entity_id}: {e}")
            return {"entity_id": entity_id, "applied": False, "error": str(e)}

    async def enrich_all(
        self,
        entity_type: str | None = None,
        limit: int | None = None,
    ) -> EnrichmentStats:
        """Enrich all (or filtered) entities with bounded concurrency."""
        stats = EnrichmentStats()
        async for event in self.enrich_all_streaming(entity_type=entity_type, limit=limit):
            if event.type == "completed" and event.stats:
                stats = event.stats
        return stats

    async def enrich_all_streaming(
        self,
        entity_type: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[ProgressEvent]:
        """Enrich entities and yield ProgressEvents as each completes.

        Yields:
            ProgressEvent(type="started", total=N)
            ProgressEvent(type="entity_started", entity_id=..., completed=i, total=N)
            ProgressEvent(type="entity_done",    entity_id=..., completed=i, total=N, result=...)
            ProgressEvent(type="completed", stats=EnrichmentStats)
        """
        entities: list[dict[str, Any]] = list(self._graph.get("entities", []))
        if entity_type:
            entities = [e for e in entities if e.get("entity_type") == entity_type]
        if limit:
            entities = entities[:limit]

        total = len(entities)
        stats = EnrichmentStats(total_entities=total)
        yield ProgressEvent(type="started", total=total)

        sem = asyncio.Semaphore(self._concurrency)
        queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()

        async def enrich_one(entity: dict[str, Any]) -> None:
            eid = str(entity["id"])
            async with sem:
                await queue.put(ProgressEvent(type="entity_started", entity_id=eid))
                result = await self.enrich_entity(eid)
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
        yield ProgressEvent(type="completed", stats=stats)
