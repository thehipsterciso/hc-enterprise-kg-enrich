"""Shared pytest fixtures."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

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


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value='{"passes": true, "reason": "Valid structure"}')
    llm.complete_structured = AsyncMock()
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
