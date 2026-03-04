"""CommitAgent — applies validated enrichments to the graph dict."""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from hckg_enrich.agents.base import AbstractEnrichmentAgent, AgentMessage, AgentRole

logger = logging.getLogger(__name__)


class CommitAgent(AbstractEnrichmentAgent):
    role = AgentRole.COMMIT

    def __init__(self, graph: dict[str, Any]) -> None:
        self._entities: dict[str, dict[str, Any]] = {
            e["id"]: e for e in graph.get("entities", [])
        }

    async def run(self, message: AgentMessage) -> AgentMessage:
        payload = dict(message.payload)
        entity_id = str(payload.get("entity_id", ""))
        proposal = dict(payload.get("proposal", {}))
        report = dict(payload.get("validation_report", {}))

        result: dict[str, Any] = {"entity_id": entity_id, "applied": False, "changes": []}

        if not report.get("passed", True):
            result["reason"] = "Blocked by GraphGuard"
            payload["commit_result"] = result
            return AgentMessage(
                sender=self.role,
                recipient=AgentRole.CONTEXT,
                correlation_id=message.correlation_id,
                payload=payload,
            )

        entity = self._entities.get(entity_id)
        if entity is None:
            result["reason"] = f"Entity {entity_id} not found"
            payload["commit_result"] = result
            return AgentMessage(
                sender=self.role,
                recipient=AgentRole.CONTEXT,
                correlation_id=message.correlation_id,
                payload=payload,
            )

        changes: list[str] = []
        for field_name, value in proposal.get("proposed_attributes", {}).items():
            if not entity.get(field_name):
                entity[field_name] = value
                changes.append(f"set {field_name}={value!r}")

        entity.setdefault("provenance", {})
        entity["provenance"]["enriched_at"] = datetime.now(UTC).isoformat()
        entity["provenance"]["enriched_by"] = "hckg-enrich/reasoning-agent"

        result["applied"] = True
        result["changes"] = changes
        payload["commit_result"] = result

        logger.info(f"Committed enrichment for {entity_id}: {changes}")

        return AgentMessage(
            sender=self.role,
            recipient=AgentRole.CONTEXT,
            correlation_id=message.correlation_id,
            payload=payload,
        )
