"""Abstract base for all enrichment pipeline agents."""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class AgentRole(StrEnum):
    CONTEXT = "context"
    SEARCH = "search"
    REASONING = "reasoning"
    CONFIDENCE = "confidence"
    COHERENCE = "coherence"
    COMMIT = "commit"
    PRIORITIZATION = "prioritization"


@dataclass
class AgentMessage:
    sender: AgentRole
    recipient: AgentRole
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = field(default_factory=dict)


class AbstractEnrichmentAgent(ABC):
    role: AgentRole

    @abstractmethod
    async def run(self, message: AgentMessage) -> AgentMessage:
        """Process incoming message and return outgoing message."""
        ...
