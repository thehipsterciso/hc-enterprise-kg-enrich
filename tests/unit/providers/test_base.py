"""Tests for provider protocol definitions."""
from __future__ import annotations

from hckg_enrich.providers.base import Message, SearchResult


def test_message_creation() -> None:
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_search_result_defaults() -> None:
    r = SearchResult(title="T", url="http://x.com", snippet="s")
    assert r.score == 1.0


def test_search_result_with_score() -> None:
    r = SearchResult(title="T", url="http://x.com", snippet="s", score=0.85)
    assert r.score == 0.85
