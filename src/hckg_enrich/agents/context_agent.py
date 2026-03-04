"""ContextAgent — retrieves KG subgraph for the entity being enriched."""
from __future__ import annotations

from hckg_enrich.agents.base import AbstractEnrichmentAgent, AgentMessage, AgentRole
from hckg_enrich.context.builder import ContextBuilder
from hckg_enrich.context.retriever import KGContextRetriever


class ContextAgent(AbstractEnrichmentAgent):
    role = AgentRole.CONTEXT

    def __init__(self, retriever: KGContextRetriever) -> None:
        self._retriever = retriever
        self._builder = ContextBuilder()

    async def run(self, message: AgentMessage) -> AgentMessage:
        entity_id: str = message.payload["entity_id"]
        ctx = self._retriever.get_context(entity_id)
        context_str = self._builder.build(ctx)
        return AgentMessage(
            sender=self.role,
            recipient=AgentRole.SEARCH,
            correlation_id=message.correlation_id,
            payload={
                "entity_id": entity_id,
                "entity_name": ctx.focal_entity.name,
                "entity_type": ctx.focal_entity.entity_type,
                "graph_context": context_str,
            },
        )
