"""EntityDiscoveryAgent — creates missing entity stubs with sourced provenance.

When GapAnalysisAgent identifies entity type layers that are absent from the
graph, EntityDiscoveryAgent searches for real-world instances and creates
sparse entity stubs. Each stub is stamped with full discovery provenance
(source URLs, discovery method, T3 confidence) so users can audit how new
entities entered the graph.

Discovered stubs are added to graph["entities"] and become candidates for the
next enrichment pass in the convergence loop.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hckg_enrich.agents.base import AgentRole
from hckg_enrich.providers.base import LLMProvider, Message, SearchProvider

if TYPE_CHECKING:
    from hckg_enrich.org.profile import OrgProfile
    from hckg_enrich.scoring.gap_analysis import GapReport

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.6.0"

DISCOVERY_SYSTEM = """You are an enterprise knowledge graph entity discovery specialist.

Given web search results about an organisation, extract a list of specific, named
entities of the requested entity type that genuinely exist within this organisation.

Rules:
- Only include entities explicitly mentioned in the search results
- Use the exact name as it appears in the source material
- Include a brief, factual description from the sources
- Do NOT invent entities not mentioned in the search results
- Return between 3 and 15 entities per type
"""


class _DiscoveredEntity(BaseModel):
    name: str
    description: str = ""


class _DiscoveryResult(BaseModel):
    entities: list[_DiscoveredEntity] = []


class EntityDiscoveryAgent:
    """Discovers and creates entity stubs for missing entity type layers.

    Each discovered entity includes full provenance:
    - source_urls: actual URLs from search results
    - source_count: number of sources
    - discovery_method: "entity_discovery_agent"
    - confidence_tier: "T3" (reasoned inference — discovered but not yet enriched)
    """

    def __init__(
        self,
        graph: dict[str, Any],
        llm: LLMProvider,
        search: SearchProvider | None = None,
    ) -> None:
        self._graph = graph
        self._llm = llm
        self._search = search

    async def discover(
        self,
        gap_report: GapReport,
        org_profile: OrgProfile | None = None,
        run_id: str = "",
    ) -> list[dict[str, Any]]:
        """Create entity stubs for each entity type in gap_report.entity_types_to_create.

        Returns the list of newly created entity dicts (also appended to graph).
        """
        newly_created: list[dict[str, Any]] = []

        for entity_type in gap_report.entity_types_to_create:
            try:
                stubs = await self._discover_type(entity_type, org_profile, run_id)
                newly_created.extend(stubs)
            except Exception as exc:
                logger.warning(
                    "EntityDiscoveryAgent failed for type %s: %s", entity_type, exc
                )

        # Append to graph, avoiding duplicate names
        existing_names_lower = {
            str(e.get("name", "")).lower()
            for e in self._graph.get("entities", [])
        }
        added: list[dict[str, Any]] = []
        for stub in newly_created:
            if stub["name"].lower() not in existing_names_lower:
                self._graph.setdefault("entities", []).append(stub)
                existing_names_lower.add(stub["name"].lower())
                added.append(stub)
            else:
                logger.debug("Skipping duplicate entity name: %s", stub["name"])

        logger.info(
            "EntityDiscoveryAgent: created %d new entities across %d types",
            len(added),
            len(gap_report.entity_types_to_create),
        )
        return added

    async def _discover_type(
        self,
        entity_type: str,
        org_profile: OrgProfile | None,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """Discover named instances of entity_type for the organisation."""
        org_name = org_profile.org_name if org_profile else "the organisation"
        industry = org_profile.industry if org_profile else ""

        queries = [
            f"{org_name} {entity_type} list examples enterprise",
            f"{org_name} {industry} {entity_type} instances real",
        ]

        search_results: list[dict[str, Any]] = []
        if self._search:
            for query in queries:
                try:
                    results = await self._search.search(query, n=5)
                    for r in results:
                        search_results.append({
                            "url": r.url,
                            "title": r.title,
                            "snippet": r.snippet,
                            "score": r.score,
                            "query": query,
                        })
                except Exception as exc:
                    logger.warning("Discovery search failed for %r: %s", query, exc)

        if not search_results:
            logger.warning(
                "No search results for entity type %s; skipping discovery", entity_type
            )
            return []

        # Build extraction prompt
        search_text = "\n\n".join(
            f"Source: {r['url']}\n{r['title']}: {r['snippet']}"
            for r in search_results[:12]
        )
        prompt = (
            f"Organisation: {org_name}\n"
            f"Entity type to discover: {entity_type}\n\n"
            f"Search results:\n{search_text}\n\n"
            f"Extract named {entity_type} entities that exist within {org_name}."
        )

        try:
            result: _DiscoveryResult = await self._llm.complete_structured(
                [Message(role="user", content=prompt)],
                schema=_DiscoveryResult,
                system=DISCOVERY_SYSTEM,
            )
        except Exception as exc:
            logger.warning("Discovery LLM extraction failed for %s: %s", entity_type, exc)
            return []

        now = datetime.now(UTC).isoformat()
        source_urls = [r["url"] for r in search_results if r.get("url")]

        stubs: list[dict[str, Any]] = []
        for entity in result.entities:
            if not entity.name.strip():
                continue
            stub = {
                "id": str(uuid.uuid4()),
                "entity_type": entity_type,
                "name": entity.name.strip(),
                "description": entity.description.strip(),
                "provenance": {
                    "discovered_at": now,
                    "discovered_by": f"hckg-enrich/v{PIPELINE_VERSION}",
                    "discovery_method": "entity_discovery_agent",
                    "run_id": run_id,
                    "source_urls": source_urls,
                    "source_count": len(source_urls),
                    "confidence_tier": "T3",
                    "confidence_score": 0.65,
                },
            }
            stubs.append(stub)

        logger.info(
            "EntityDiscoveryAgent: discovered %d %s entities from %d sources",
            len(stubs), entity_type, len(search_results),
        )
        return stubs
