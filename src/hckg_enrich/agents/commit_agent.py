"""CommitAgent — applies validated enrichments to the graph dict."""
from __future__ import annotations

import logging
import uuid
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
        self._relationships: list[dict[str, Any]] = graph.setdefault("relationships", [])

    async def run(self, message: AgentMessage) -> AgentMessage:
        payload = dict(message.payload)
        entity_id = str(payload.get("entity_id", ""))
        proposal = dict(payload.get("proposal", {}))
        report = dict(payload.get("validation_report", {}))

        result: dict[str, Any] = {
            "entity_id": entity_id,
            "applied": False,
            "changes": [],
            "relationships_added": 0,
        }

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

        now = datetime.now(UTC).isoformat()
        changes: list[str] = []

        # --- commit attribute enrichments ---
        for field_name, value in proposal.get("proposed_attributes", {}).items():
            if not entity.get(field_name):
                entity[field_name] = value
                changes.append(f"set {field_name}={value!r}")

        entity.setdefault("provenance", {})
        entity["provenance"]["enriched_at"] = now
        entity["provenance"]["enriched_by"] = "hckg-enrich/reasoning-agent"

        # --- commit relationship proposals ---
        rel_count = 0
        for rel in proposal.get("proposed_relationships", []):
            target_name = str(rel.get("target_name", ""))
            rel_type = str(rel.get("relationship_type", ""))
            if not rel_type or not target_name:
                continue

            # resolve target by name (best-effort)
            target_entity = next(
                (e for e in self._entities.values() if e.get("name") == target_name),
                None,
            )
            if target_entity is None:
                logger.debug(
                    f"Skipping proposed relationship {rel_type} → {target_name!r}: "
                    "target not found in graph"
                )
                continue

            # avoid duplicate relationships
            already_exists = any(
                r.get("source_id") == entity_id
                and r.get("target_id") == target_entity["id"]
                and r.get("relationship_type") == rel_type
                for r in self._relationships
            )
            if already_exists:
                continue

            self._relationships.append(
                {
                    "id": str(uuid.uuid4()),
                    "relationship_type": rel_type,
                    "source_id": entity_id,
                    "target_id": target_entity["id"],
                    "provenance": {
                        "enriched_at": now,
                        "enriched_by": "hckg-enrich/reasoning-agent",
                        "rationale": rel.get("rationale", ""),
                    },
                }
            )
            rel_count += 1
            changes.append(f"add relationship {rel_type} → {target_name!r}")

        result["applied"] = True
        result["changes"] = changes
        result["relationships_added"] = rel_count
        payload["commit_result"] = result

        logger.info(f"Committed enrichment for {entity_id}: {changes}")

        return AgentMessage(
            sender=self.role,
            recipient=AgentRole.CONTEXT,
            correlation_id=message.correlation_id,
            payload=payload,
        )
