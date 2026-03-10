# ADR-010: Organizational Grounding via Stock Ticker

**Status:** Accepted  
**Date:** 2026-03-10  
**Deciders:** hc-enterprise-kg-enrich maintainers

## Context

The enrichment pipeline previously had no knowledge of which organisation it was enriching. The `ReasoningAgent` system prompt contained hardcoded domain rules (`Finance MUST relate to CFO domain`, etc.) that applied universally regardless of the actual target company. This produced generic enrichments that were not grounded in the real-world structure of the target organisation.

An enterprise KG for Apple Inc has fundamentally different structural expectations than one for a regional bank. Enriching without organisational context means the LLM must guess at industry norms, regulatory obligations, leadership archetypes, and technology patterns — producing lower-confidence, less-defensible enrichments.

## Decision

Introduce **organisational grounding**: a new `--ticker SYMBOL` (or `--org-name NAME`) CLI parameter that anchors the entire enrichment session to a real organisation.

### OrgProfile

A new `OrgProfile` dataclass captures the organisation's identity and is threaded through every pipeline call:

```python
@dataclass
class OrgProfile:
    ticker: str | None          # Stock exchange ticker (AAPL, MSFT, JPM, …)
    org_name: str               # Full legal name
    industry: str               # e.g., "technology", "financial services"
    sector: str                 # e.g., "Information Technology"
    country: str                # Primary domicile (default "US")
    headcount_tier: str         # "startup" | "mid" | "large" | "enterprise"
    revenue_tier: str           # "small" | "medium" | "large" | "mega"
    key_roles: list[str]        # Typical C-suite roles for this org type
    subsidiaries: list[str]
    regulatory_regime: list[str]   # ["SOX", "HIPAA", "GDPR", …]
    industry_frameworks: list[str] # ["NIST CSF", "ISO 27001", "CIS v8"]
    tech_profile: dict
    sources: list[SourceCitation]  # EVERY field grounded in a real URL
    research_confidence: float     # 0.0–1.0 mapped to T1–T4 scale
```

### OrgResearchAgent

A new `OrgResearchAgent` runs **once** at the start of a `ConvergenceController` session. It issues 4 targeted web searches and uses LLM extraction to populate the `OrgProfile`. Every field in the profile is backed by at least one `SourceCitation` with an actual URL.

Graceful fallback: if search is disabled or the ticker is unresolvable, a minimal `OrgProfile(org_name=name, industry=industry)` is returned — enrichment continues without deep grounding but records this in `research_confidence=0.0`.

### Usage in ReasoningAgent

The `OrgProfile` is passed as `org_profile` in the `AgentMessage` payload. `ReasoningAgent` injects it into its system prompt:

> "You are enriching a knowledge graph for {org_name} ({ticker}), a {industry} company. Regulatory regime: {regulatory_regime}. Industry frameworks: {industry_frameworks}. Key expected roles: {key_roles}."

All hardcoded domain rules are removed.

## Consequences

**Positive:**
- Enrichments are grounded in real-world org characteristics, not generic heuristics
- `OrgProfile.sources` provides full auditability of the grounding research
- Regulatory regime awareness improves gap analysis (SOX → require Controls layer)
- Adaptive search queries use org context for higher-relevance results

**Negative:**
- One additional LLM + search call per convergence session (amortised cost)
- Ticker resolution may fail for private companies → graceful fallback required

**Neutral:**
- Backwards-compatible: `--ticker` is optional; existing `run` command without it uses `EnrichmentController` unchanged
