"""SearchAgent — grounds enrichment in web search results with full URL provenance."""
from __future__ import annotations

import logging
from typing import Any

from hckg_enrich.agents.base import AbstractEnrichmentAgent, AgentMessage, AgentRole
from hckg_enrich.providers.base import SearchProvider

logger = logging.getLogger(__name__)

# Fields that drive adaptive query generation when empty
HIGH_VALUE_FIELDS = {
    "description", "owner", "responsible_team", "criticality",
    "data_classification", "risk_tier", "tech_stack", "vendor_name",
    "budget", "headcount", "framework", "status",
}

# Per-field query templates
_FIELD_QUERIES: dict[str, str] = {
    "criticality": "{entity_type} {entity_name} criticality tier classification enterprise",
    "owner": "{entity_type} {entity_name} ownership business owner responsible {org_context}",
    "tech_stack": "{entity_name} technology stack components architecture {org_context}",
    "data_classification": "{entity_name} data classification sensitivity PII confidential",
    "risk_tier": "{entity_type} {entity_name} risk tier classification enterprise risk",
    "framework": "{entity_name} compliance framework standard certification",
    "vendor_name": "{entity_name} vendor supplier manufacturer provider",
    "budget": "{entity_type} {entity_name} budget cost allocation enterprise",
    "headcount": "{entity_type} {entity_name} headcount team size employees",
    "responsible_team": "{entity_type} {entity_name} responsible team department governance",
}


class SearchAgent(AbstractEnrichmentAgent):
    """Issues adaptive web searches and propagates full source URLs into payload.

    v0.6.0: Fixed URL propagation — URLs are no longer discarded. Payload now
    contains:
      - search_sources: list[dict] with url, title, snippet, score, query
      - search_queries: list[str] of queries issued
      - search_context: formatted text for LLM consumption (includes URLs)
    """

    role = AgentRole.SEARCH

    def __init__(self, search: SearchProvider | None = None) -> None:
        self._search = search

    async def run(self, message: AgentMessage) -> AgentMessage:
        payload = dict(message.payload)

        if self._search is None:
            payload.setdefault("search_sources", [])
            payload.setdefault("search_queries", [])
            payload["search_context"] = ""
            return AgentMessage(
                sender=self.role,
                recipient=AgentRole.REASONING,
                correlation_id=message.correlation_id,
                payload=payload,
            )

        entity_name = str(payload.get("entity_name", ""))
        entity_type = str(payload.get("entity_type", ""))
        entity = dict(payload.get("entity", {}))
        org_profile = dict(payload.get("org_profile", {}))

        org_name = str(org_profile.get("org_name", ""))
        industry = str(org_profile.get("industry", ""))
        org_context = f"{org_name} {industry}".strip() or "enterprise"

        queries = self._build_queries(entity_name, entity_type, entity, org_context)

        search_sources: list[dict[str, Any]] = []
        for query in queries:
            try:
                results = await self._search.search(query, n=3)
                for r in results:
                    search_sources.append({
                        "url": r.url,
                        "title": r.title,
                        "snippet": r.snippet,
                        "score": r.score,
                        "query": query,
                    })
            except Exception as e:
                logger.warning("Search failed for query %r: %s", query, e)

        # Build LLM-readable context that includes URLs for transparency
        search_context = "\n".join(
            f"[{s['title']}] ({s['url']})\n{s['snippet']}"
            for s in search_sources
        )

        payload["search_sources"] = search_sources
        payload["search_queries"] = queries
        payload["search_context"] = search_context

        if not search_sources:
            logger.error(
                "SearchAgent: all queries failed for entity %s — "
                "enrichment will proceed with no source grounding",
                payload.get("entity_id", "unknown"),
            )
            payload["search_skipped"] = True
        else:
            payload["search_skipped"] = False

        return AgentMessage(
            sender=self.role,
            recipient=AgentRole.REASONING,
            correlation_id=message.correlation_id,
            payload=payload,
        )

    def _build_queries(
        self,
        entity_name: str,
        entity_type: str,
        entity: dict[str, Any],
        org_context: str,
    ) -> list[str]:
        """Build up to 4 adaptive queries based on entity's empty high-value fields."""
        queries: list[str] = []

        # Always include a broad context query
        queries.append(
            f"{entity_type} {entity_name} enterprise {org_context} organizational structure"
        )

        # Add field-specific queries for empty high-value fields (up to 3 more)
        empty_fields = [f for f in HIGH_VALUE_FIELDS if not entity.get(f)]
        for field in empty_fields:
            if field in _FIELD_QUERIES and len(queries) < 4:
                q = _FIELD_QUERIES[field].format(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    org_context=org_context,
                )
                queries.append(q)

        # Fallback: include industry/ownership query if only 1 query built
        if len(queries) < 2:
            queries.append(f"{entity_name} industry ownership reporting structure")

        return queries[:4]
