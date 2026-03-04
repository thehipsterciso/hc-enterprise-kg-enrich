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

## Relation to hc-enterprise-kg guard module

The `guard/` module in `hc-enterprise-kg` contains scaffolding that was merged but never wired up. This package reimplements GraphGuard from scratch with LLM-based evaluation as a first-class design principle, not an afterthought.
