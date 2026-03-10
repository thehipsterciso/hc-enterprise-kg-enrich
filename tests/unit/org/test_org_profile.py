"""Tests for OrgProfile dataclass."""
from __future__ import annotations

import pytest

from hckg_enrich.org.profile import OrgProfile


class TestOrgProfileDefaults:
    def test_empty_init(self):
        p = OrgProfile()
        assert p.ticker is None
        assert p.org_name == ""
        assert p.industry == ""
        assert p.country == "US"
        assert p.research_confidence == 0.0
        assert p.sources == []
        assert p.key_roles == []

    def test_ticker_stored(self):
        p = OrgProfile(ticker="AAPL", org_name="Apple Inc")
        assert p.ticker == "AAPL"
        assert p.org_name == "Apple Inc"

    def test_regulatory_regime_list(self):
        p = OrgProfile(regulatory_regime=["SOX", "HIPAA"])
        assert "SOX" in p.regulatory_regime
        assert "HIPAA" in p.regulatory_regime

    def test_sources_are_dicts(self):
        source = {"url": "https://example.com", "title": "Test", "snippet": "..."}
        p = OrgProfile(sources=[source])
        assert len(p.sources) == 1
        assert p.sources[0]["url"] == "https://example.com"


class TestOrgProfileRoundTrip:
    def test_to_dict_contains_all_fields(self):
        p = OrgProfile(
            ticker="MSFT",
            org_name="Microsoft Corporation",
            industry="technology",
            regulatory_regime=["SOX"],
            research_confidence=0.85,
        )
        d = p.to_dict()
        assert d["ticker"] == "MSFT"
        assert d["org_name"] == "Microsoft Corporation"
        assert d["industry"] == "technology"
        assert d["regulatory_regime"] == ["SOX"]
        assert d["research_confidence"] == 0.85

    def test_from_dict_roundtrip(self):
        original = OrgProfile(
            ticker="JPM",
            org_name="JPMorgan Chase",
            industry="financial services",
            regulatory_regime=["SOX", "GDPR"],
            research_confidence=0.7,
        )
        restored = OrgProfile.from_dict(original.to_dict())
        assert restored.ticker == "JPM"
        assert restored.org_name == "JPMorgan Chase"
        assert restored.industry == "financial services"
        assert restored.regulatory_regime == ["SOX", "GDPR"]
        assert restored.research_confidence == 0.7

    def test_from_dict_defaults_on_missing_keys(self):
        p = OrgProfile.from_dict({"org_name": "Acme"})
        assert p.org_name == "Acme"
        assert p.ticker is None
        assert p.country == "US"
        assert p.research_confidence == 0.0


class TestContextString:
    def test_empty_profile_returns_empty(self):
        assert OrgProfile().context_string() == ""

    def test_basic_profile_context(self):
        p = OrgProfile(
            org_name="Apple Inc",
            ticker="AAPL",
            industry="technology",
            headcount_tier="enterprise",
        )
        ctx = p.context_string()
        assert "Apple Inc" in ctx
        assert "AAPL" in ctx
        assert "technology" in ctx
        assert "enterprise" in ctx

    def test_regulatory_in_context(self):
        p = OrgProfile(
            org_name="BigBank",
            regulatory_regime=["SOX", "GDPR"],
        )
        ctx = p.context_string()
        assert "SOX" in ctx
        assert "GDPR" in ctx

    def test_frameworks_in_context(self):
        p = OrgProfile(
            org_name="GovCo",
            industry_frameworks=["NIST CSF", "FedRAMP"],
        )
        ctx = p.context_string()
        assert "NIST CSF" in ctx

    def test_no_ticker_in_context_when_none(self):
        p = OrgProfile(org_name="Private Co")
        ctx = p.context_string()
        assert "Private Co" in ctx
        assert "(None)" not in ctx
