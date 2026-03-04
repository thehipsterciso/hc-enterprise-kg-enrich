"""AI-driven synthetic enterprise digital twin generator."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel

from hckg_enrich.providers.base import LLMProvider, Message, SearchProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM prompt + response schemas
# ---------------------------------------------------------------------------

_ORG_DESIGN_SYSTEM = """You are an enterprise architecture expert.
Design a realistic synthetic enterprise org structure for the requested industry and size.
Return valid JSON only — no prose, no markdown fences.
"""

_ORG_DESIGN_PROMPT = """\
Design a synthetic {industry} enterprise with {size} size profile.

Return JSON with this exact structure:
{{
  "company_name": "...",
  "industry": "...",
  "departments": [
    {{
      "name": "...",
      "function": "...",
      "leader_title": "...",
      "leader_name": "...",
      "headcount_range": "..."
    }}
  ],
  "systems": [
    {{
      "name": "...",
      "category": "erp|crm|hrms|bi|security|infrastructure|data_platform|other",
      "owner_department": "...",
      "vendor": "..."
    }}
  ],
  "vendors": [
    {{
      "name": "...",
      "category": "cloud|software|consulting|hardware|data",
      "primary_contact": "..."
    }}
  ],
  "data_assets": [
    {{
      "name": "...",
      "classification": "confidential|internal|public",
      "owner_department": "...",
      "format": "..."
    }}
  ]
}}

Rules:
- Finance/Accounting systems must be owned by Finance or IT — never by HR or CEO directly
- HR systems must be owned by HR or IT
- ERP systems are owned by Finance or IT governance
- Each department has exactly one named leader
- Include 4-8 departments, 6-12 systems, 4-8 vendors, 4-8 data assets
{search_context}
"""


class OrgDesign(BaseModel):
    company_name: str
    industry: str
    departments: list[dict[str, str]]
    systems: list[dict[str, str]]
    vendors: list[dict[str, str]]
    data_assets: list[dict[str, str]]


# ---------------------------------------------------------------------------
# Size profiles
# ---------------------------------------------------------------------------

_SIZE_PROFILES = {
    "small": "500-1000 employees, single-region operations",
    "medium": "2000-5000 employees, multi-region, mid-market",
    "large": "10000+ employees, global enterprise, Fortune 500 scale",
}


class TwinGenerator:
    """Generates a semantically correct synthetic enterprise knowledge graph.

    Uses LLM reasoning (+ optional web search) to design an org structure,
    then builds a complete graph.json from the design.  The result passes
    GraphGuard validation because the LLM is constrained by domain rules.
    """

    def __init__(
        self,
        llm: LLMProvider,
        search: SearchProvider | None = None,
        size: str = "medium",
        industry: str = "financial services",
    ) -> None:
        self._llm = llm
        self._search = search
        self._size = size
        self._industry = industry

    async def generate(self) -> dict[str, Any]:
        """Generate a complete synthetic enterprise graph.

        Returns:
            graph dict compatible with graph.json (entities + relationships lists).
        """
        logger.info(f"Designing org structure: {self._industry} / {self._size}")
        design = await self._design_org()

        logger.info(
            f"Building graph for {design.company_name}: "
            f"{len(design.departments)} departments, "
            f"{len(design.systems)} systems, "
            f"{len(design.vendors)} vendors, "
            f"{len(design.data_assets)} data assets"
        )

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        dept_by_name: dict[str, str] = {}
        system_by_name: dict[str, str] = {}
        vendor_by_name: dict[str, str] = {}

        # --- departments + leaders ---
        for dept in design.departments:
            dept_id = str(uuid.uuid4())
            entities.append({
                "id": dept_id,
                "entity_type": "department",
                "name": dept.get("name", "Unknown Department"),
                "function": dept.get("function", ""),
                "headcount_range": dept.get("headcount_range", ""),
            })
            dept_by_name[dept.get("name", "")] = dept_id

            leader_name = dept.get("leader_name", "")
            leader_title = dept.get("leader_title", "")
            if leader_name:
                person_id = str(uuid.uuid4())
                entities.append({
                    "id": person_id,
                    "entity_type": "person",
                    "name": leader_name,
                    "title": leader_title,
                    "department": dept.get("name", ""),
                })
                relationships.append(_rel("leads", person_id, dept_id))
                relationships.append(_rel("member_of", person_id, dept_id))

        # --- systems ---
        for sys_def in design.systems:
            sys_id = str(uuid.uuid4())
            entities.append({
                "id": sys_id,
                "entity_type": "system",
                "name": sys_def.get("name", "Unknown System"),
                "category": sys_def.get("category", "other"),
                "vendor": sys_def.get("vendor", ""),
            })
            system_by_name[sys_def.get("name", "")] = sys_id

            owner_dept = sys_def.get("owner_department", "")
            if owner_dept and owner_dept in dept_by_name:
                relationships.append(_rel("owns", dept_by_name[owner_dept], sys_id))

        # --- vendors ---
        for ven_def in design.vendors:
            ven_id = str(uuid.uuid4())
            entities.append({
                "id": ven_id,
                "entity_type": "vendor",
                "name": ven_def.get("name", "Unknown Vendor"),
                "category": ven_def.get("category", "software"),
                "primary_contact": ven_def.get("primary_contact", ""),
            })
            vendor_by_name[ven_def.get("name", "")] = ven_id

        # link systems → vendors by name match
        for sys_def in design.systems:
            sys_vendor = sys_def.get("vendor", "")
            sys_name = sys_def.get("name", "")
            if sys_vendor in vendor_by_name and sys_name in system_by_name:
                relationships.append(
                    _rel("supplied_by", system_by_name[sys_name], vendor_by_name[sys_vendor])
                )

        # --- data assets ---
        for da_def in design.data_assets:
            da_id = str(uuid.uuid4())
            entities.append({
                "id": da_id,
                "entity_type": "data_asset",
                "name": da_def.get("name", "Unknown Asset"),
                "classification": da_def.get("classification", "internal"),
                "format": da_def.get("format", ""),
            })
            owner_dept = da_def.get("owner_department", "")
            if owner_dept and owner_dept in dept_by_name:
                relationships.append(_rel("owns", dept_by_name[owner_dept], da_id))

        graph: dict[str, Any] = {
            "metadata": {
                "generated_by": "hckg-enrich/twin-generator",
                "company_name": design.company_name,
                "industry": design.industry,
                "size": self._size,
            },
            "entities": entities,
            "relationships": relationships,
        }

        logger.info(
            f"Generated graph: {len(entities)} entities, {len(relationships)} relationships"
        )
        return graph

    async def _design_org(self) -> OrgDesign:
        size_desc = _SIZE_PROFILES.get(self._size, self._size)
        search_context = ""
        if self._search:
            try:
                results = await self._search.search(
                    f"{self._industry} enterprise org structure departments systems"
                )
                search_context = "\n## Industry context from web\n" + "\n".join(
                    f"- {r.title}: {r.snippet}" for r in results[:3]
                )
            except Exception:
                pass

        prompt = _ORG_DESIGN_PROMPT.format(
            industry=self._industry,
            size=size_desc,
            search_context=search_context,
        )
        result = await self._llm.complete_structured(
            [Message(role="user", content=prompt)],
            schema=OrgDesign,
            system=_ORG_DESIGN_SYSTEM,
        )
        if not isinstance(result, OrgDesign):
            raise TypeError(f"Expected OrgDesign, got {type(result)}")
        return result


def _rel(rel_type: str, source_id: str, target_id: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "relationship_type": rel_type,
        "source_id": source_id,
        "target_id": target_id,
    }
