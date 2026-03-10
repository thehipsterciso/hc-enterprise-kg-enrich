# ADR-004: GraphGuard Semantic Quality Contracts

**Status:** Accepted
**Date:** 2026-03-04

## Decision

GraphGuard is implemented as first-class semantic contracts using LLM evaluation, not static schema validators.

## Contracts (v0.1.0)

1. **OrgHierarchyContract** (ERROR) — functional units report to appropriate parent functions
2. **SystemOwnershipContract** (ERROR) — systems owned by appropriate business units
3. **VendorRelationshipContract** (WARNING) — vendor relationships follow governance norms

## Key difference from static rules

Static rules like "Finance cannot report to HR" fail for edge cases and require constant maintenance. LLM-based contracts evaluate semantic plausibility in context, using the existing graph structure as grounding. This makes contracts adaptive to org-specific structures.

## Severity model

- `ERROR` — blocks the enrichment from being applied
- `WARNING` — logged but does not block; enrichment is applied

## Relation to hc-enterprise-kg

`hc-enterprise-kg` previously contained a `guard/` scaffolding module (removed in v0.32.0). `hc-enterprise-kg-enrich` is the canonical home for GraphGuard. The enrichment pipeline and all quality contracts live here exclusively.
