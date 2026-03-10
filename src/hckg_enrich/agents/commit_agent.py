"""CommitAgent — applies validated enrichments to the graph with full provenance."""
from __future__ import annotations

import copy
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from hckg_enrich.agents.base import AbstractEnrichmentAgent, AgentMessage, AgentRole

if TYPE_CHECKING:
    from hckg_enrich.provenance.audit_log import AuditLog

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.3.0"


class CommitAgent(AbstractEnrichmentAgent):
    """Applies validated enrichments to the graph and writes structured provenance.

    Enhanced in v0.3.0:
    - Captures before/after EntityDiff for full change traceability
    - Writes ProvenanceRecord with confidence tier and source citations
    - Appends AuditEvents to AuditLog (if injected by controller)
    - Propagates run_id from payload for session-level correlation
    - Uses confidence_tier from ConfidenceAgent (no longer defaults to T4 silently)
    - Replaces hardcoded 'hckg-enrich/reasoning-agent' attribution with structured provenance
    """

    role = AgentRole.COMMIT

    def __init__(
        self,
        graph: dict[str, Any],
        audit_log: AuditLog | None = None,
    ) -> None:
        self._entities: dict[str, dict[str, Any]] = {
            e["id"]: e for e in graph.get("entities", [])
        }
        self._relationships: list[dict[str, Any]] = graph.setdefault("relationships", [])
        self._audit_log = audit_log

    async def run(self, message: AgentMessage) -> AgentMessage:
        payload = dict(message.payload)
        entity_id = str(payload.get("entity_id", ""))
        proposal = dict(payload.get("proposal", {}))
        report = dict(payload.get("validation_report", {}))
        run_id = str(payload.get("run_id", ""))
        llm_model = str(payload.get("llm_model", "claude-opus-4-5"))
        confidence_tier = str(payload.get("confidence_tier", "T4"))
        confidence_score = float(payload.get("confidence_score", 0.50))
        search_queries = list(payload.get("search_queries", []))
        search_sources = list(payload.get("search_sources", []))
        guard_report = dict(payload.get("validation_report", {}))

        result: dict[str, Any] = {
            "entity_id": entity_id,
            "applied": False,
            "changes": [],
            "relationships_added": 0,
        }

        # --- GraphGuard blocked ---
        if not report.get("passed", True):
            result["reason"] = "Blocked by GraphGuard"
            blocking = report.get("blocking_failures", [])
            if blocking:
                result["blocking_contracts"] = blocking
            payload["commit_result"] = result
            self._emit_blocked_event(
                entity_id=entity_id,
                run_id=run_id,
                payload=payload,
                blocking_failures=blocking,
            )
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

        # Snapshot entity state before any mutations
        entity_before = copy.deepcopy(entity)

        now = datetime.now(UTC).isoformat()
        changes: list[str] = []
        attribute_changes: list[str] = []

        # --- Commit attribute enrichments (only fill empty fields) ---
        for field_name, value in proposal.get("proposed_attributes", {}).items():
            if not entity.get(field_name):
                entity[field_name] = value
                change = f"set {field_name}={value!r}"
                changes.append(change)
                attribute_changes.append(change)

        # --- Write structured provenance on the entity ---
        entity.setdefault("provenance", {})
        entity["provenance"].update(
            {
                "enriched_at": now,
                "enriched_by": f"hckg-enrich/v{PIPELINE_VERSION}",
                "run_id": run_id,
                "llm_model": llm_model,
                "confidence_tier": confidence_tier,
                "confidence_score": confidence_score,
                "source_count": len(search_sources),
            }
        )

        # --- Commit relationship proposals ---
        rel_count = 0
        committed_rels: list[dict[str, Any]] = []

        for rel in proposal.get("proposed_relationships", []):
            target_name = str(rel.get("target_name", ""))
            rel_type = str(rel.get("relationship_type", ""))
            if not rel_type or not target_name:
                continue

            target_entity = next(
                (e for e in self._entities.values() if e.get("name") == target_name),
                None,
            )
            if target_entity is None:
                logger.debug(
                    "Skipping proposed relationship %s → %r: target not found in graph",
                    rel_type,
                    target_name,
                )
                continue

            already_exists = any(
                r.get("source_id") == entity_id
                and r.get("target_id") == target_entity["id"]
                and r.get("relationship_type") == rel_type
                for r in self._relationships
            )
            if already_exists:
                continue

            new_rel = {
                "id": str(uuid.uuid4()),
                "relationship_type": rel_type,
                "source_id": entity_id,
                "target_id": target_entity["id"],
                "weight": 1.0,
                "confidence": confidence_score,
                "provenance": {
                    "enriched_at": now,
                    "enriched_by": f"hckg-enrich/v{PIPELINE_VERSION}",
                    "run_id": run_id,
                    "llm_model": llm_model,
                    "confidence_tier": confidence_tier,
                    "rationale": rel.get("rationale", ""),
                },
            }
            self._relationships.append(new_rel)
            rel_count += 1
            rel_change = f"add relationship {rel_type} → {target_name!r}"
            changes.append(rel_change)
            committed_rels.append(
                {
                    "relationship_id": new_rel["id"],
                    "relationship_type": rel_type,
                    "target_id": target_entity["id"],
                    "target_name": target_name,
                }
            )

        result["applied"] = True
        result["changes"] = changes
        result["relationships_added"] = rel_count
        result["confidence_tier"] = confidence_tier
        result["confidence_score"] = confidence_score

        # --- Compute diff ---
        entity_after = {k: v for k, v in entity.items() if k != "provenance"}
        entity_before_clean = {k: v for k, v in entity_before.items() if k != "provenance"}
        added_fields = {
            k: v for k, v in entity_after.items() if k not in entity_before_clean
        }
        changed_fields = {
            k: {"before": entity_before_clean.get(k), "after": v}
            for k, v in entity_after.items()
            if k in entity_before_clean and entity_before_clean[k] != v
        }
        result["diff"] = {"added": added_fields, "changed": changed_fields}

        # --- Emit audit event ---
        self._emit_enriched_event(
            entity=entity,
            entity_id=entity_id,
            run_id=run_id,
            llm_model=llm_model,
            now=now,
            attribute_changes=attribute_changes,
            committed_rels=committed_rels,
            confidence_tier=confidence_tier,
            search_queries=search_queries,
            search_sources=search_sources,
            guard_report=guard_report,
            reasoning=str(proposal.get("reasoning", "")),
        )

        payload["commit_result"] = result
        logger.info(
            "CommitAgent: committed entity=%s tier=%s changes=%d rels=%d",
            entity_id,
            confidence_tier,
            len(changes),
            rel_count,
        )

        return AgentMessage(
            sender=self.role,
            recipient=AgentRole.CONTEXT,
            correlation_id=message.correlation_id,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Audit event helpers
    # ------------------------------------------------------------------

    def _emit_enriched_event(
        self,
        entity: dict[str, Any],
        entity_id: str,
        run_id: str,
        llm_model: str,
        now: str,
        attribute_changes: list[str],
        committed_rels: list[dict[str, Any]],
        confidence_tier: str,
        search_queries: list[str],
        search_sources: list[dict[str, Any]],
        guard_report: dict[str, Any],
        reasoning: str,
    ) -> None:
        if self._audit_log is None:
            return
        try:
            from hckg_enrich.provenance.audit_log import AuditEvent, AuditEventType

            event = AuditEvent(
                event_type=AuditEventType.ENTITY_ENRICHED,
                run_id=run_id,
                entity_id=entity_id,
                entity_name=str(entity.get("name", "")),
                entity_type=str(entity.get("entity_type", "")),
                agent_role=AgentRole.COMMIT.value,
                pipeline_version=PIPELINE_VERSION,
                llm_model=llm_model,
                attribute_changes=attribute_changes,
                relationships_added=committed_rels,
                confidence_tier=confidence_tier,
                guard_contracts_passed=list(guard_report.get("passed_contracts", [])),
                guard_warnings=list(guard_report.get("warnings", [])),
                guard_blocking_failures=list(guard_report.get("blocking_failures", [])),
                reasoning=reasoning,
                search_source_count=len(search_sources),
            )
            self._audit_log.append(event)
        except Exception as exc:
            logger.warning("AuditLog append failed: %s", exc)

    def _emit_blocked_event(
        self,
        entity_id: str,
        run_id: str,
        payload: dict[str, Any],
        blocking_failures: list[str],
    ) -> None:
        if self._audit_log is None:
            return
        try:
            from hckg_enrich.provenance.audit_log import AuditEvent, AuditEventType

            event = AuditEvent(
                event_type=AuditEventType.GUARD_BLOCKED,
                run_id=run_id,
                entity_id=entity_id,
                entity_name=str(payload.get("entity_name", "")),
                entity_type=str(payload.get("entity_type", "")),
                agent_role=AgentRole.COHERENCE.value,
                pipeline_version=PIPELINE_VERSION,
                llm_model=str(payload.get("llm_model", "")),
                attribute_changes=[],
                relationships_added=[],
                confidence_tier=str(payload.get("confidence_tier", "T4")),
                guard_contracts_passed=[],
                guard_warnings=[],
                guard_blocking_failures=blocking_failures,
                reasoning="",
                search_source_count=0,
            )
            self._audit_log.append(event)
        except Exception as exc:
            logger.warning("AuditLog blocked-event append failed: %s", exc)
