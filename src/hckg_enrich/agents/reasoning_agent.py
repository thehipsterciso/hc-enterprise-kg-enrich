"""ReasoningAgent — LLM-powered enrichment proposal generation."""
from __future__ import annotations

from pydantic import BaseModel

from hckg_enrich.agents.base import AbstractEnrichmentAgent, AgentMessage, AgentRole
from hckg_enrich.providers.base import LLMProvider, Message

SYSTEM = """You are an expert enterprise knowledge graph enrichment agent.

Given:
1. A focal entity and its existing graph context
2. Web search results grounding industry/domain conventions
3. The entity type and name

Your task: propose specific, coherent enrichments for this entity.
Enrichments must be semantically correct for the entity's domain and consistent
with established enterprise organizational conventions.

CRITICAL rules:
- Finance entities MUST relate to financial functions/leadership (CFO domain)
- HR entities MUST relate to people/workforce functions (CHRO domain)
- ERP/Finance systems belong to IT or Finance governance, NEVER directly to CEO
- Infer from the org structure already present in the graph
- Prefer specific, grounded proposals over generic ones

Return JSON with this exact structure:
{
  "proposed_attributes": {"field": "value"},
  "proposed_relationships": [
    {"relationship_type": "...", "target_name": "...", "target_type": "...", "rationale": "..."}
  ],
  "reasoning": "brief explanation of enrichment decisions"
}
"""


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

        prompt_parts = [
            f"Entity: {entity_name} (type: {entity_type})",
            "",
            "## Current Graph Context",
            graph_context,
        ]
        if search_context:
            prompt_parts += ["", "## Web Search Grounding", search_context]
        prompt_parts += ["", "Propose enrichments for this entity."]

        proposal = await self._llm.complete_structured(
            [Message(role="user", content="\n".join(prompt_parts))],
            schema=EnrichmentProposal,
            system=SYSTEM,
        )

        payload["proposal"] = proposal.model_dump()
        return AgentMessage(
            sender=self.role,
            recipient=AgentRole.COHERENCE,
            correlation_id=message.correlation_id,
            payload=payload,
        )
