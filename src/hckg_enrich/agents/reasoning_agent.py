"""ReasoningAgent — LLM-powered enrichment proposal generation."""
from __future__ import annotations

from pydantic import BaseModel

from hckg_enrich.agents.base import AbstractEnrichmentAgent, AgentMessage, AgentRole
from hckg_enrich.providers.base import LLMProvider, Message

_BASE_SYSTEM = """You are an expert enterprise knowledge graph enrichment agent.

Given:
1. A focal entity and its existing graph context
2. Web search results grounding industry/domain conventions (including source URLs)
3. The entity type and name
4. Organisational context (when provided)

Your task: propose specific, coherent enrichments for this entity.
Enrichments must be semantically correct for the entity's domain and consistent
with the organisational context provided.

CRITICAL rules:
- Infer from the org structure already present in the graph
- Prefer specific, grounded proposals over generic ones
- When source URLs are provided in search context, ground your reasoning in them
- Do NOT invent information not supported by the context or search results

Return JSON with this exact structure:
{
  "proposed_attributes": {"field": "value"},
  "proposed_relationships": [
    {"relationship_type": "...", "target_name": "...", "target_type": "...", "rationale": "..."}
  ],
  "reasoning": "brief explanation of enrichment decisions"
}
"""


def _build_system(org_profile: dict | None) -> str:
    """Inject org grounding into the system prompt when available."""
    if not org_profile:
        return _BASE_SYSTEM

    org_name = org_profile.get("org_name", "")
    ticker = org_profile.get("ticker", "")
    industry = org_profile.get("industry", "")
    sector = org_profile.get("sector", "")
    headcount_tier = org_profile.get("headcount_tier", "")
    regulatory = org_profile.get("regulatory_regime", [])
    frameworks = org_profile.get("industry_frameworks", [])
    key_roles = org_profile.get("key_roles", [])

    org_parts: list[str] = []
    if org_name:
        ticker_str = f" ({ticker})" if ticker else ""
        org_parts.append(f"Organisation: {org_name}{ticker_str}")
    if industry:
        org_parts.append(f"Industry: {industry}")
    if sector:
        org_parts.append(f"Sector: {sector}")
    if headcount_tier:
        org_parts.append(f"Scale: {headcount_tier}")
    if regulatory:
        org_parts.append(f"Regulatory regime: {', '.join(regulatory)}")
    if frameworks:
        org_parts.append(f"Industry frameworks: {', '.join(frameworks)}")
    if key_roles:
        org_parts.append(f"Key roles: {', '.join(key_roles)}")

    if not org_parts:
        return _BASE_SYSTEM

    org_context = "\n".join(org_parts)
    return (
        _BASE_SYSTEM
        + f"\n\n## Organisational Context\n{org_context}\n"
        "Use this context to ground all enrichment decisions in the actual "
        "structure and norms of this specific organisation."
    )


class EnrichmentProposal(BaseModel):
    proposed_attributes: dict[str, str] = {}
    proposed_relationships: list[dict[str, str]] = []
    reasoning: str = ""


class ReasoningAgent(AbstractEnrichmentAgent):
    role = AgentRole.REASONING

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def run(self, message: AgentMessage) -> AgentMessage:
        payload = dict(message.payload)
        entity_name = str(payload.get("entity_name", ""))
        entity_type = str(payload.get("entity_type", ""))
        graph_context = str(payload.get("graph_context", ""))
        search_context = str(payload.get("search_context", ""))
        org_profile = payload.get("org_profile")

        prompt_parts = [
            f"Entity: {entity_name} (type: {entity_type})",
            "",
            "## Current Graph Context",
            graph_context,
        ]
        if search_context:
            prompt_parts += ["", "## Web Search Grounding (with source URLs)", search_context]
        prompt_parts += ["", "Propose enrichments for this entity."]

        system = _build_system(org_profile)

        proposal = await self._llm.complete_structured(
            [Message(role="user", content="\n".join(prompt_parts))],
            schema=EnrichmentProposal,
            system=system,
        )

        payload["proposal"] = proposal.model_dump()
        return AgentMessage(
            sender=self.role,
            recipient=AgentRole.COHERENCE,
            correlation_id=message.correlation_id,
            payload=payload,
        )
