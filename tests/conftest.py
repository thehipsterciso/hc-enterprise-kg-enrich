"""Shared pytest fixtures."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hckg_enrich.providers.base import SearchResult


@pytest.fixture
def sample_graph() -> dict[str, Any]:
    return {
        "entities": [
            {
                "id": "dept-finance-001",
                "entity_type": "department",
                "name": "Finance",
                "description": "Corporate Finance function",
            },
            {
                "id": "dept-hr-001",
                "entity_type": "department",
                "name": "Human Resources",
                "description": "HR function",
            },
            {
                "id": "sys-erp-001",
                "entity_type": "system",
                "name": "SAP S/4HANA",
                "description": "ERP system",
            },
            {
                "id": "person-cfo-001",
                "entity_type": "person",
                "name": "Jane Smith",
                "description": "Chief Financial Officer",
            },
        ],
        "relationships": [
            {
                "id": "rel-001",
                "relationship_type": "works_in",
                "source_id": "person-cfo-001",
                "target_id": "dept-finance-001",
            }
        ],
    }


def _make_proposal_mock() -> MagicMock:
    """Return a non-async mock whose .model_dump() returns a valid EnrichmentProposal dict."""
    proposal = MagicMock()
    proposal.model_dump.return_value = {
        "proposed_attributes": {},
        "proposed_relationships": [],
        "reasoning": "Mock reasoning — no real LLM call",
    }
    return proposal


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    # complete() is called by guard contracts — return valid JSON string
    llm.complete = AsyncMock(return_value='{"passes": true, "reason": "Valid structure"}')
    # complete_structured() is called by ReasoningAgent; must return something with .model_dump()
    llm.complete_structured = AsyncMock(return_value=_make_proposal_mock())
    return llm


@pytest.fixture
def mock_search() -> AsyncMock:
    search = AsyncMock()
    search.search = AsyncMock(
        return_value=[
            SearchResult(
                title="ERP Ownership Best Practices",
                url="https://example.com",
                snippet="ERP systems are typically owned by Finance or IT departments.",
            )
        ]
    )
    return search
