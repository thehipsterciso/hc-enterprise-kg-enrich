"""Tests for TwinGenerator."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hckg_enrich.synthetic.twin_generator import OrgDesign, TwinGenerator, _rel


def _make_design(**overrides) -> OrgDesign:
    base = {
        "company_name": "Acme Corp",
        "industry": "financial services",
        "departments": [
            {
                "name": "Finance",
                "function": "Financial operations",
                "leader_title": "CFO",
                "leader_name": "Jane Smith",
                "headcount_range": "50-100",
            },
            {
                "name": "Information Technology",
                "function": "Technology governance",
                "leader_title": "CTO",
                "leader_name": "Bob Lee",
                "headcount_range": "100-200",
            },
        ],
        "systems": [
            {
                "name": "SAP ERP",
                "category": "erp",
                "owner_department": "Finance",
                "vendor": "SAP SE",
            },
        ],
        "vendors": [
            {
                "name": "SAP SE",
                "category": "software",
                "primary_contact": "sales@sap.com",
            }
        ],
        "data_assets": [
            {
                "name": "GL Ledger",
                "classification": "confidential",
                "owner_department": "Finance",
                "format": "structured",
            }
        ],
    }
    base.update(overrides)
    return OrgDesign(**base)


@pytest.fixture()
def mock_llm_with_design():
    design = _make_design()
    llm = AsyncMock()
    llm.complete_structured = AsyncMock(return_value=design)
    return llm


@pytest.mark.asyncio
async def test_generate_returns_graph_structure(mock_llm_with_design):
    gen = TwinGenerator(llm=mock_llm_with_design, industry="financial services", size="medium")
    graph = await gen.generate()

    assert "entities" in graph
    assert "relationships" in graph
    assert "metadata" in graph
    assert graph["metadata"]["company_name"] == "Acme Corp"


@pytest.mark.asyncio
async def test_generate_creates_department_entities(mock_llm_with_design):
    gen = TwinGenerator(llm=mock_llm_with_design)
    graph = await gen.generate()

    entity_types = {e["entity_type"] for e in graph["entities"]}
    assert "department" in entity_types


@pytest.mark.asyncio
async def test_generate_creates_person_leaders(mock_llm_with_design):
    gen = TwinGenerator(llm=mock_llm_with_design)
    graph = await gen.generate()

    persons = [e for e in graph["entities"] if e["entity_type"] == "person"]
    assert len(persons) == 2  # CFO + CTO
    person_names = {p["name"] for p in persons}
    assert "Jane Smith" in person_names
    assert "Bob Lee" in person_names


@pytest.mark.asyncio
async def test_generate_creates_system_owned_by_department(mock_llm_with_design):
    gen = TwinGenerator(llm=mock_llm_with_design)
    graph = await gen.generate()

    system = next(e for e in graph["entities"] if e.get("name") == "SAP ERP")
    finance = next(e for e in graph["entities"] if e.get("name") == "Finance")

    owns_rels = [
        r for r in graph["relationships"]
        if r["relationship_type"] == "owns"
        and r["source_id"] == finance["id"]
        and r["target_id"] == system["id"]
    ]
    assert len(owns_rels) == 1


@pytest.mark.asyncio
async def test_generate_links_system_to_vendor(mock_llm_with_design):
    gen = TwinGenerator(llm=mock_llm_with_design)
    graph = await gen.generate()

    system = next(e for e in graph["entities"] if e.get("name") == "SAP ERP")
    vendor = next(e for e in graph["entities"] if e.get("name") == "SAP SE")

    supplied_rels = [
        r for r in graph["relationships"]
        if r["relationship_type"] == "supplied_by"
        and r["source_id"] == system["id"]
        and r["target_id"] == vendor["id"]
    ]
    assert len(supplied_rels) == 1


@pytest.mark.asyncio
async def test_generate_uses_search_grounding():
    design = _make_design()
    llm = AsyncMock()
    llm.complete_structured = AsyncMock(return_value=design)
    search = AsyncMock()
    search.search = AsyncMock(return_value=[])

    gen = TwinGenerator(llm=llm, search=search, industry="financial services")
    await gen.generate()
    search.search.assert_called_once()


def test_rel_helper_produces_valid_dict():
    r = _rel("owns", "src-id", "tgt-id")
    assert r["relationship_type"] == "owns"
    assert r["source_id"] == "src-id"
    assert r["target_id"] == "tgt-id"
    assert "id" in r
