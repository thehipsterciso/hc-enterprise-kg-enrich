# ADR-009: Parallel GraphGuard Contract Execution and Fail-Closed Policy

**Status:** Accepted
**Date:** 2026-03-10
**Deciders:** Platform engineering, security
**Relates to:** ADR-004 (GraphGuard Contracts), ADR-006 (Observability)

---

## Context

### Problem 1: Sequential contract execution

The v0.2.0 `EnrichmentGuardian.validate()` ran contracts in a `for` loop:

```python
for contract in self._contracts:
    result = await contract.evaluate(...)
    results.append(result)
```

Each contract makes a separate LLM API call (~500ms–2s per call). With 3 contracts,
validation added 1.5–6 seconds per entity in serial. This is the dominant latency
contributor in the enrichment pipeline.

The contracts are completely independent — they each evaluate the same entity and
proposal, share no state, and produce independent results. Serial execution provides
zero benefit.

### Problem 2: Fail-open antipattern (GG-006)

All three original contracts had identical exception handling:

```python
except Exception:
    passes = True  # SECURITY ANTIPATTERN
    reason = "Could not parse LLM response — defaulting to pass"
```

A network timeout, malformed LLM response, or LLM refusal silently allowed any
enrichment through. In a security context, the GraphGuard layer exists precisely
to catch invalid enrichments. Failing open under uncertainty defeats its purpose.

---

## Decision

### Decision 1: Parallel execution via asyncio.gather

Replace the serial `for` loop with `asyncio.gather`:

```python
results: list[ContractResult] = list(
    await asyncio.gather(*[_safe_evaluate(c) for c in self._contracts])
)
```

This collapses N sequential LLM calls into a single parallel wait. For 3 contracts
averaging 1 second each, validation drops from ~3s to ~1s.

### Decision 2: Fail-closed on all exception paths (GG-006)

Change all contract exception handlers from `passes = True` to `passes = False`:

```python
except Exception:
    passes = False
    reason = "Could not parse LLM response — failing closed (GG-006 security policy)"
```

This applies to:
- `OrgHierarchyContract`
- `SystemOwnershipContract`
- `VendorRelationshipContract`
- All 5 new contracts (`DataAssetOwnership`, `PersonRoleConsistency`,
  `RelationshipTypeSemantics`, `CircularDependency`, `EntityDeduplication`)

The `EnrichmentGuardian._safe_evaluate()` wrapper adds a second defense layer:
any unhandled exception from a contract (not just parse failures) is caught,
logged, and converted to a `ContractResult(passed=False)`.

### Decision 3: New contract portfolio

Add 5 new contracts to address coverage gaps:

| Contract | Severity | Mechanism | Gap Addressed |
|----------|----------|-----------|---------------|
| `CircularDependencyContract` | ERROR | DFS, rule-based | Cycles in depends_on/connects_to chains |
| `EntityDeduplicationContract` | WARNING | Token Jaccard, rule-based | Proposed refs to near-duplicate entities |
| `DataAssetOwnershipContract` | ERROR | LLM | PII/regulated data owned by wrong dept |
| `PersonRoleConsistencyContract` | WARNING | LLM | C-suite persons assigned IC-level roles |
| `RelationshipTypeSemanticsContract` | ERROR | Schema + LLM hybrid | Wrong domain/range for relationship types |

The two rule-based contracts (`CircularDependency`, `EntityDeduplication`) add
zero LLM latency and are always fast regardless of provider status.

---

## Latency Analysis

### Before (3 contracts, serial)
```
Context → Search → Reasoning → [OrgHierarchy LLM] → [SystemOwnership LLM] → [VendorRel LLM] → Commit
                                └────────────────── ~3–6s sequential ──────────────────┘
```

### After (3 LLM contracts + 2 rule-based, parallel)
```
Context → Search → Reasoning → [OrgHierarchy LLM] ─┐
                               [SystemOwnership LLM]─┤ asyncio.gather → Commit
                               [VendorRel LLM]       ─┤
                               [CircularDep (sync)]  ─┘
                               └──────────────────── ~1–2s parallel ──────────────────┘
```

Estimated improvement: 2–4 seconds per entity on a 3-contract setup.

---

## Consequences

**Positive:**
- Validation latency reduced by ~60–70% for 3 LLM contracts
- Fail-closed eliminates silent enrichment bypass on LLM failures
- New contracts cover 5 previously undetected failure modes
- `_safe_evaluate` wrapper means a crashing contract cannot abort validation of others
- Rule-based contracts add zero latency overhead

**Negative:**
- Parallel LLM calls increase API concurrency — may hit provider rate limits faster
  on large graphs (mitigated by the controller's entity-level `asyncio.Semaphore`)
- Fail-closed increases the blocked_count metric; operators may see more blocks
  initially as LLM providers return non-JSON occasionally
- `RelationshipTypeSemanticsContract` schema is hardcoded; additions require code changes

**Rejected alternatives:**
- **Thread-based parallelism:** asyncio is native to the async pipeline; no benefit from threads
- **Fail-open with telemetry:** Still a security antipattern; observability does not fix trust
- **Single LLM call for all contracts:** Reduces parallelism benefit; harder to add/remove contracts independently

---

## Re-evaluation triggers

- Provider rate limit errors increase on runs with >5 contracts → add per-provider concurrency
  limiter in `_safe_evaluate`
- New relationship types added to hc-enterprise-kg → update `RELATIONSHIP_SCHEMA` in
  `RelationshipTypeSemanticsContract`
- False-positive block rate exceeds 5% → audit fail-closed behavior and adjust parsing
  robustness (not revert to fail-open)
