"""CoherenceAgent — runs GraphGuard contracts against proposed enrichments."""
from __future__ import annotations

import logging

from hckg_enrich.agents.base import AbstractEnrichmentAgent, AgentMessage, AgentRole
from hckg_enrich.guard.guardian import EnrichmentGuardian

logger = logging.getLogger(__name__)


class CoherenceAgent(AbstractEnrichmentAgent):
    role = AgentRole.COHERENCE

    def __init__(self, guardian: EnrichmentGuardian) -> None:
        self._guardian = guardian

    async def run(self, message: AgentMessage) -> AgentMessage:
        payload = dict(message.payload)
        entity_id = str(payload.get("entity_id", ""))
        proposal = dict(payload.get("proposal", {}))
        graph_context = str(payload.get("graph_context", ""))

        report = await self._guardian.validate(
            entity_id=entity_id,
            proposed_enrichments=proposal,
            graph_context=graph_context,
        )

        payload["validation_report"] = {
            "passed": report.passed,
            "blocking_failures": [
                {"contract": r.contract_id, "message": r.message}
                for r in report.blocking_failures
            ],
            "warnings": [
                {"contract": r.contract_id, "message": r.message}
                for r in report.warnings
            ],
        }

        if not report.passed:
            logger.warning(
                f"Entity {entity_id} failed GraphGuard: "
                + "; ".join(f.message for f in report.blocking_failures)
            )

        return AgentMessage(
            sender=self.role,
            recipient=AgentRole.COMMIT,
            correlation_id=message.correlation_id,
            payload=payload,
        )
