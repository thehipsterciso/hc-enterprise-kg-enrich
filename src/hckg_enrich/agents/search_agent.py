"""SearchAgent — grounds enrichment in web search results."""
from __future__ import annotations

import logging

from hckg_enrich.agents.base import AbstractEnrichmentAgent, AgentMessage, AgentRole
from hckg_enrich.providers.base import SearchProvider

logger = logging.getLogger(__name__)


class SearchAgent(AbstractEnrichmentAgent):
    role = AgentRole.SEARCH

    def __init__(self, search: SearchProvider | None = None) -> None:
        self._search = search

    async def run(self, message: AgentMessage) -> AgentMessage:
        payload = dict(message.payload)

        if self._search is None:
            payload["search_context"] = ""
            return AgentMessage(
                sender=self.role,
                recipient=AgentRole.REASONING,
                correlation_id=message.correlation_id,
                payload=payload,
            )

        entity_name = str(payload.get("entity_name", ""))
        entity_type = str(payload.get("entity_type", ""))
        queries = [
            f"{entity_type} {entity_name} enterprise organizational structure",
            f"{entity_name} industry ownership reporting structure",
        ]

        snippets: list[str] = []
        for query in queries:
            try:
                results = await self._search.search(query, n=3)
                for r in results:
                    snippets.append(f"[{r.title}] {r.snippet}")
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")

        payload["search_context"] = "\n".join(snippets)
        return AgentMessage(
            sender=self.role,
            recipient=AgentRole.REASONING,
            correlation_id=message.correlation_id,
            payload=payload,
        )
