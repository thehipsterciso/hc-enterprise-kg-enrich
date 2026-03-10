"""OrgProfile — structured organisational identity for enrichment grounding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrgProfile:
    """Structured identity of the target organisation being enriched.

    Every field is backed by source URLs stored in the ``sources`` list.
    The profile is built once per convergence session by OrgResearchAgent and
    threaded through all enrichment pipeline calls via AgentMessage payload.
    """

    ticker: str | None = None           # Stock exchange ticker (AAPL, MSFT, …)
    org_name: str = ""                  # Full legal name
    industry: str = ""                  # e.g. "technology", "financial services"
    sector: str = ""                    # e.g. "Information Technology"
    country: str = "US"                 # Primary domicile
    headcount_tier: str = ""            # "startup" | "mid" | "large" | "enterprise"
    revenue_tier: str = ""              # "small" | "medium" | "large" | "mega"
    key_roles: list[str] = field(default_factory=list)          # ["CISO", "CTO", …]
    subsidiaries: list[str] = field(default_factory=list)
    regulatory_regime: list[str] = field(default_factory=list)  # ["SOX", "HIPAA", …]
    industry_frameworks: list[str] = field(default_factory=list) # ["NIST CSF", "ISO 27001"]
    tech_profile: dict[str, Any] = field(default_factory=dict)
    # Provenance — every field grounded in a real source URL
    sources: list[dict[str, Any]] = field(default_factory=list)  # list of SourceCitation dicts
    research_confidence: float = 0.0    # 0.0–1.0 (0.0 = no research performed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "org_name": self.org_name,
            "industry": self.industry,
            "sector": self.sector,
            "country": self.country,
            "headcount_tier": self.headcount_tier,
            "revenue_tier": self.revenue_tier,
            "key_roles": self.key_roles,
            "subsidiaries": self.subsidiaries,
            "regulatory_regime": self.regulatory_regime,
            "industry_frameworks": self.industry_frameworks,
            "tech_profile": self.tech_profile,
            "sources": self.sources,
            "research_confidence": self.research_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrgProfile:
        return cls(
            ticker=data.get("ticker"),
            org_name=data.get("org_name", ""),
            industry=data.get("industry", ""),
            sector=data.get("sector", ""),
            country=data.get("country", "US"),
            headcount_tier=data.get("headcount_tier", ""),
            revenue_tier=data.get("revenue_tier", ""),
            key_roles=list(data.get("key_roles", [])),
            subsidiaries=list(data.get("subsidiaries", [])),
            regulatory_regime=list(data.get("regulatory_regime", [])),
            industry_frameworks=list(data.get("industry_frameworks", [])),
            tech_profile=dict(data.get("tech_profile", {})),
            sources=list(data.get("sources", [])),
            research_confidence=float(data.get("research_confidence", 0.0)),
        )

    def context_string(self) -> str:
        """Return a compact, LLM-readable summary for injection into prompts."""
        parts = []
        if self.org_name:
            ticker_part = f" ({self.ticker})" if self.ticker else ""
            parts.append(f"Organisation: {self.org_name}{ticker_part}")
        if self.industry:
            parts.append(f"Industry: {self.industry}")
        if self.sector:
            parts.append(f"Sector: {self.sector}")
        if self.headcount_tier:
            parts.append(f"Scale: {self.headcount_tier}")
        if self.regulatory_regime:
            parts.append(f"Regulatory regime: {', '.join(self.regulatory_regime)}")
        if self.industry_frameworks:
            parts.append(f"Industry frameworks: {', '.join(self.industry_frameworks)}")
        if self.key_roles:
            parts.append(f"Key roles: {', '.join(self.key_roles)}")
        if self.tech_profile:
            summary = ", ".join(f"{k}: {v}" for k, v in list(self.tech_profile.items())[:4])
            parts.append(f"Tech profile: {summary}")
        return "\n".join(parts)
