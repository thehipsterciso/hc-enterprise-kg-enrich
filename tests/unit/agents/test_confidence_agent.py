"""Tests for ConfidenceAgent."""
from __future__ import annotations

import pytest

from hckg_enrich.agents.base import AgentMessage, AgentRole
from hckg_enrich.agents.confidence_agent import (
    ConfidenceAgent,
    EvidenceSignals,
    _compute_confidence,
    _score_to_tier,
)
from hckg_enrich.provenance.record import ConfidenceTier


@pytest.fixture
def agent() -> ConfidenceAgent:
    return ConfidenceAgent()


def _make_payload(
    sources=None,
    queries=None,
    reasoning="",
    proposed_attrs=None,
    proposed_rels=None,
) -> dict:
    return {
        "entity_name": "SAP ERP",
        "entity_type": "system",
        "proposal": {
            "proposed_attributes": proposed_attrs or {},
            "proposed_relationships": proposed_rels or [],
            "reasoning": reasoning,
        },
        "search_sources": sources or [],
        "search_queries": queries or [],
        "existing_relationships": [],
    }


@pytest.mark.asyncio
async def test_no_proposal_returns_t4(agent):
    msg = AgentMessage(
        sender=AgentRole.REASONING,
        recipient=AgentRole.COHERENCE,
        payload={"entity_name": "X", "entity_type": "system"},
    )
    result = await agent.run(msg)
    assert result.payload["confidence_tier"] == ConfidenceTier.T4.value
    assert result.payload["confidence_score"] == 0.50


@pytest.mark.asyncio
async def test_high_quality_sources_raises_tier(agent):
    payload = _make_payload(
        sources=[
            {"score": 0.9, "url": "https://a.com", "title": "SAP ERP official", "snippet": ""},
            {"score": 0.85, "url": "https://b.com", "title": "SAP docs", "snippet": ""},
            {"score": 0.88, "url": "https://c.com", "title": "SAP integration", "snippet": ""},
        ],
        queries=["SAP ERP ownership enterprise"],
        reasoning="According to official SAP documentation, the system is confirmed in Finance.",
        proposed_attrs={"owner": "CFO", "criticality": "high", "description": "ERP system"},
    )
    msg = AgentMessage(
        sender=AgentRole.REASONING,
        recipient=AgentRole.COHERENCE,
        payload=payload,
    )
    result = await agent.run(msg)
    tier = result.payload["confidence_tier"]
    score = result.payload["confidence_score"]
    assert score >= 0.80  # T2 or better
    assert tier in (ConfidenceTier.T1.value, ConfidenceTier.T2.value)


@pytest.mark.asyncio
async def test_heavy_hedging_lowers_score(agent):
    payload = _make_payload(
        sources=[{"score": 0.6, "url": "https://x.com", "title": "x", "snippet": ""}],
        reasoning="Perhaps this might possibly be correct. It could potentially work. "
                  "We might assume this is uncertain.",
        proposed_attrs={"status": "unknown"},
    )
    msg = AgentMessage(
        sender=AgentRole.REASONING,
        recipient=AgentRole.COHERENCE,
        payload=payload,
    )
    result = await agent.run(msg)
    assert result.payload["confidence_score"] < 0.65


@pytest.mark.asyncio
async def test_evidence_signals_in_payload(agent):
    payload = _make_payload(
        sources=[{"score": 0.8, "url": "u", "title": "t", "snippet": "s"}],
        queries=["SAP ERP ownership"],
        reasoning="Confirmed",
        proposed_attrs={"owner": "IT"},
    )
    msg = AgentMessage(
        sender=AgentRole.REASONING,
        recipient=AgentRole.COHERENCE,
        payload=payload,
    )
    result = await agent.run(msg)
    signals = result.payload["evidence_signals"]
    assert signals["source_count"] == 1
    assert "entity_name_in_queries" in signals


def test_score_to_tier_boundaries():
    assert _score_to_tier(0.95) == ConfidenceTier.T1
    assert _score_to_tier(0.85) == ConfidenceTier.T2
    assert _score_to_tier(0.70) == ConfidenceTier.T3
    assert _score_to_tier(0.55) == ConfidenceTier.T4
    assert _score_to_tier(0.50) == ConfidenceTier.T4


def test_compute_confidence_no_evidence():
    signals = EvidenceSignals()
    score, rationale = _compute_confidence(signals)
    assert score == 0.50
    assert "baseline" in rationale


def test_compute_confidence_three_good_sources():
    signals = EvidenceSignals(
        source_count=3,
        avg_relevance=0.85,
        entity_name_in_queries=True,
        grounding_phrases=2,
        specific_attributes=3,
    )
    score, _ = _compute_confidence(signals)
    # 0.50 + 0.20 + 0.08 + 0.05 + 0.06 + 0.05 = 0.94
    assert score >= 0.90


def test_compute_confidence_capped_at_1():
    signals = EvidenceSignals(
        source_count=3,
        avg_relevance=0.95,
        entity_name_in_queries=True,
        grounding_phrases=5,
        specific_attributes=5,
    )
    score, _ = _compute_confidence(signals)
    assert score <= 1.0


def test_compute_confidence_floor_at_0_50():
    signals = EvidenceSignals(
        hedging_phrases=10,
        generic_attributes=5,
    )
    score, _ = _compute_confidence(signals)
    assert score >= 0.50
