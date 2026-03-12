"""Tests for ArtifactStore — document artifact storage."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hckg_enrich.provenance.artifact_store import (
    ArtifactStore,
    EnrichmentArtifact,
    _ext_for_content_type,
)

# ---------------------------------------------------------------------------
# EnrichmentArtifact
# ---------------------------------------------------------------------------


class TestEnrichmentArtifact:
    def test_to_dict_all_fields(self):
        artifact = EnrichmentArtifact(
            artifact_id="abc-123",
            url="https://example.com/doc.pdf",
            title="Annual Report",
            content_type="pdf",
            local_path="/tmp/artifacts/run1/abc-123.pdf",
            file_size_bytes=4096,
            retrieved_at="2026-01-01T00:00:00+00:00",
            run_id="run1",
            entity_id="ent-1",
            summary="Summary of the doc",
        )
        d = artifact.to_dict()
        assert d["artifact_id"] == "abc-123"
        assert d["url"] == "https://example.com/doc.pdf"
        assert d["title"] == "Annual Report"
        assert d["content_type"] == "pdf"
        assert d["local_path"] == "/tmp/artifacts/run1/abc-123.pdf"
        assert d["file_size_bytes"] == 4096
        assert d["run_id"] == "run1"
        assert d["entity_id"] == "ent-1"
        assert d["summary"] == "Summary of the doc"

    def test_to_dict_empty_summary(self):
        artifact = EnrichmentArtifact(
            artifact_id="x",
            url="https://example.com",
            title="",
            content_type="html",
            local_path="/tmp/x.html",
            file_size_bytes=100,
            retrieved_at="2026-01-01T00:00:00+00:00",
            run_id="run1",
            entity_id="ent-1",
        )
        d = artifact.to_dict()
        assert d["summary"] == ""


# ---------------------------------------------------------------------------
# _ext_for_content_type helper
# ---------------------------------------------------------------------------


class TestExtForContentType:
    def test_pdf(self):
        assert _ext_for_content_type("application/pdf") == ".pdf"

    def test_json(self):
        assert _ext_for_content_type("application/json") == ".json"

    def test_text(self):
        assert _ext_for_content_type("text/plain") == ".txt"

    def test_html_default(self):
        assert _ext_for_content_type("text/html") == ".html"

    def test_unknown_defaults_to_html(self):
        assert _ext_for_content_type("application/octet-stream") == ".html"


# ---------------------------------------------------------------------------
# ArtifactStore construction
# ---------------------------------------------------------------------------


class TestArtifactStoreInit:
    def test_creates_run_directory(self, tmp_path):
        _store = ArtifactStore(base_dir=tmp_path, run_id="run-001")
        assert (tmp_path / "run-001").is_dir()

    def test_starts_empty(self, tmp_path):
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")
        assert store.get("nonexistent") is None
        assert store.list_for_entity("ent-1") == []

    def test_to_dict_empty(self, tmp_path):
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")
        d = store.to_dict()
        assert d["run_id"] == "run-001"
        assert d["artifact_count"] == 0
        assert d["artifacts"] == {}


# ---------------------------------------------------------------------------
# ArtifactStore deterministic IDs
# ---------------------------------------------------------------------------


class TestArtifactIDDeterminism:
    def test_same_url_same_id(self, tmp_path):
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")
        url = "https://example.com/doc.pdf"
        id1 = store._artifact_id_for(url)
        id2 = store._artifact_id_for(url)
        assert id1 == id2

    def test_different_urls_different_ids(self, tmp_path):
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")
        id1 = store._artifact_id_for("https://example.com/a")
        id2 = store._artifact_id_for("https://example.com/b")
        assert id1 != id2

    def test_id_is_valid_uuid_like(self, tmp_path):
        import uuid
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")
        artifact_id = store._artifact_id_for("https://example.com")
        # Should be parseable as UUID
        parsed = uuid.UUID(artifact_id)
        assert str(parsed) == artifact_id


# ---------------------------------------------------------------------------
# ArtifactStore.fetch_and_store — no httpx
# ---------------------------------------------------------------------------


class TestFetchAndStoreNoHttpx:
    @pytest.mark.asyncio
    async def test_returns_none_when_httpx_not_installed(self, tmp_path):
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")
        with patch.dict("sys.modules", {"httpx": None}):
            result = await store.fetch_and_store(
                url="https://example.com/doc.html",
                entity_id="ent-1",
            )
        assert result is None


# ---------------------------------------------------------------------------
# ArtifactStore.fetch_and_store — mocked httpx
# ---------------------------------------------------------------------------


def _make_mock_response(content: bytes, content_type: str = "text/html", status: int = 200):
    resp = MagicMock()
    resp.content = content
    resp.headers = {"content-type": content_type}
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


def _mock_httpx(response: MagicMock):
    """Build a mock httpx module that returns the given response."""
    mock_httpx_module = MagicMock()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=response)
    mock_httpx_module.AsyncClient.return_value = mock_client
    return mock_httpx_module, mock_client


class TestFetchAndStore:
    @pytest.mark.asyncio
    async def test_stores_html_artifact(self, tmp_path):
        content = b"<html><body>Hello world</body></html>"
        mock_response = _make_mock_response(content, "text/html")
        mock_httpx_module, _ = _mock_httpx(mock_response)
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")

        with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
            artifact = await store.fetch_and_store(
                url="https://example.com/page.html",
                entity_id="ent-1",
                title="Test Page",
            )

        assert artifact is not None
        assert artifact.url == "https://example.com/page.html"
        assert artifact.entity_id == "ent-1"
        assert artifact.title == "Test Page"
        assert artifact.content_type == "html"
        assert artifact.file_size_bytes == len(content)
        assert artifact.run_id == "run-001"
        assert artifact.artifact_id != ""

    @pytest.mark.asyncio
    async def test_file_written_to_disk(self, tmp_path):
        content = b"PDF content"
        mock_response = _make_mock_response(content, "application/pdf")
        mock_httpx_module, _ = _mock_httpx(mock_response)
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")

        with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
            artifact = await store.fetch_and_store(
                url="https://example.com/report.pdf",
                entity_id="ent-1",
            )

        assert artifact is not None
        assert Path(artifact.local_path).exists()
        assert Path(artifact.local_path).read_bytes() == content

    @pytest.mark.asyncio
    async def test_index_json_written(self, tmp_path):
        content = b"<html>Hello</html>"
        mock_response = _make_mock_response(content, "text/html")
        mock_httpx_module, _ = _mock_httpx(mock_response)
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")

        with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
            await store.fetch_and_store(url="https://example.com/page", entity_id="ent-1")

        index_path = tmp_path / "run-001" / "index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text())
        assert index["artifact_count"] == 1
        assert index["run_id"] == "run-001"

    @pytest.mark.asyncio
    async def test_skips_unsupported_content_type(self, tmp_path):
        content = b"binary data"
        mock_response = _make_mock_response(content, "application/octet-stream")
        mock_httpx_module, _ = _mock_httpx(mock_response)
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")

        with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
            result = await store.fetch_and_store(
                url="https://example.com/binary.bin", entity_id="ent-1"
            )

        assert result is None
        assert store.to_dict()["artifact_count"] == 0

    @pytest.mark.asyncio
    async def test_skips_oversized_content(self, tmp_path):
        content = b"x" * (11 * 1024 * 1024)  # 11 MB — over the 10 MB limit
        mock_response = _make_mock_response(content, "text/html")
        mock_httpx_module, _ = _mock_httpx(mock_response)
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")

        with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
            result = await store.fetch_and_store(
                url="https://example.com/huge.html", entity_id="ent-1"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_network_error(self, tmp_path):
        mock_httpx_module = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_httpx_module.AsyncClient.return_value = mock_client

        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")

        with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
            result = await store.fetch_and_store(
                url="https://example.com/unreachable", entity_id="ent-1"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_deduplicates_same_url(self, tmp_path):
        content = b"<html>page</html>"
        mock_response = _make_mock_response(content, "text/html")
        mock_httpx_module, mock_client = _mock_httpx(mock_response)
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")

        url = "https://example.com/page"
        with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
            a1 = await store.fetch_and_store(url=url, entity_id="ent-1")
            a2 = await store.fetch_and_store(url=url, entity_id="ent-1")

        # Second call returns cached artifact — no second HTTP request
        assert a1 is not None
        assert a2 is not None
        assert a1.artifact_id == a2.artifact_id
        assert store.to_dict()["artifact_count"] == 1

    @pytest.mark.asyncio
    async def test_uses_url_as_title_when_empty(self, tmp_path):
        content = b"<html>page</html>"
        mock_response = _make_mock_response(content, "text/html")
        mock_httpx_module, _ = _mock_httpx(mock_response)
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")

        url = "https://example.com/page"
        with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
            artifact = await store.fetch_and_store(url=url, entity_id="ent-1", title="")

        assert artifact is not None
        assert artifact.title == url


# ---------------------------------------------------------------------------
# ArtifactStore retrieval
# ---------------------------------------------------------------------------


class TestArtifactStoreRetrieval:
    @pytest.mark.asyncio
    async def test_get_returns_artifact(self, tmp_path):
        content = b"<html>page</html>"
        mock_response = _make_mock_response(content, "text/html")
        mock_httpx_module, _ = _mock_httpx(mock_response)
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")

        with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
            artifact = await store.fetch_and_store(url="https://example.com/p", entity_id="ent-1")

        assert artifact is not None
        retrieved = store.get(artifact.artifact_id)
        assert retrieved is artifact

    @pytest.mark.asyncio
    async def test_list_for_entity(self, tmp_path):
        content = b"<html>page</html>"
        mock_httpx_module = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=_make_mock_response(content, "text/html"))
        mock_httpx_module.AsyncClient.return_value = mock_client
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")

        with patch.dict("sys.modules", {"httpx": mock_httpx_module}):
            _a1 = await store.fetch_and_store(url="https://example.com/a", entity_id="ent-1")
            _a2 = await store.fetch_and_store(url="https://example.com/b", entity_id="ent-1")
            _a3 = await store.fetch_and_store(url="https://example.com/c", entity_id="ent-2")

        ent1_artifacts = store.list_for_entity("ent-1")
        assert len(ent1_artifacts) == 2

        ent2_artifacts = store.list_for_entity("ent-2")
        assert len(ent2_artifacts) == 1

        empty = store.list_for_entity("ent-unknown")
        assert empty == []

    def test_to_dict_includes_all_artifacts(self, tmp_path):
        store = ArtifactStore(base_dir=tmp_path, run_id="run-001")
        # Inject directly to test to_dict without HTTP
        from hckg_enrich.provenance.artifact_store import EnrichmentArtifact
        artifact = EnrichmentArtifact(
            artifact_id="abc-123",
            url="https://example.com",
            title="Test",
            content_type="html",
            local_path="/tmp/abc-123.html",
            file_size_bytes=100,
            retrieved_at="2026-01-01T00:00:00+00:00",
            run_id="run-001",
            entity_id="ent-1",
        )
        store._artifacts["abc-123"] = artifact

        d = store.to_dict()
        assert d["artifact_count"] == 1
        assert "abc-123" in d["artifacts"]
