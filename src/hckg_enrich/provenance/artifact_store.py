"""ArtifactStore — stores downloaded source documents as local artifacts.

When the enrichment pipeline retrieves a URL pointing to a structured document
(SEC filing, NIST publication, annual report, PDF), ArtifactStore downloads and
stores it locally so users can inspect the source material that backed an
enrichment decision.

Usage is optional: enabled by --artifacts-dir CLI flag. When disabled, source
URLs are still captured in entity provenance but documents are not downloaded.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Content types we'll attempt to fetch and store
_FETCHABLE_CONTENT_TYPES = {"text/html", "application/pdf", "application/json", "text/plain"}

# Maximum artifact file size (10 MB)
_MAX_ARTIFACT_BYTES = 10 * 1024 * 1024


@dataclass
class EnrichmentArtifact:
    """Metadata record for a stored source document."""

    artifact_id: str
    url: str
    title: str
    content_type: str       # "pdf" | "html" | "json" | "text"
    local_path: str         # Absolute path to stored file
    file_size_bytes: int
    retrieved_at: str
    run_id: str
    entity_id: str          # Entity this artifact was retrieved for
    summary: str = ""       # Brief description of the document

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "url": self.url,
            "title": self.title,
            "content_type": self.content_type,
            "local_path": self.local_path,
            "file_size_bytes": self.file_size_bytes,
            "retrieved_at": self.retrieved_at,
            "run_id": self.run_id,
            "entity_id": self.entity_id,
            "summary": self.summary,
        }


class ArtifactStore:
    """Stores downloaded source documents with deterministic artifact IDs.

    Artifacts are stored under: {base_dir}/{run_id}/{artifact_id}.{ext}

    An index file at {base_dir}/{run_id}/index.json maps artifact_id to
    EnrichmentArtifact metadata. The index is written after each store().
    """

    def __init__(self, base_dir: Path, run_id: str) -> None:
        self._base_dir = base_dir
        self._run_id = run_id
        self._run_dir = base_dir / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts: dict[str, EnrichmentArtifact] = {}
        self._by_entity: dict[str, list[str]] = {}  # entity_id → [artifact_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_and_store(
        self,
        url: str,
        entity_id: str,
        title: str = "",
    ) -> EnrichmentArtifact | None:
        """Download URL content, store locally, return artifact reference.

        Returns None if the URL is not fetchable, too large, or an error occurs.
        Never raises — errors are logged and None is returned.
        """
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not installed; artifact store disabled")
            return None

        # Deterministic artifact_id from URL (same URL in same run → same file)
        artifact_id = self._artifact_id_for(url)
        if artifact_id in self._artifacts:
            return self._artifacts[artifact_id]

        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(url, headers={"User-Agent": "hckg-enrich/0.6.0"})
                response.raise_for_status()

                content_header = response.headers.get("content-type", "text/html").split(";")[0].lower()
                if content_header not in _FETCHABLE_CONTENT_TYPES:
                    logger.debug("Skipping artifact for unsupported content-type %s: %s", content_header, url)
                    return None

                content = response.content
                if len(content) > _MAX_ARTIFACT_BYTES:
                    logger.debug("Skipping artifact — too large (%d bytes): %s", len(content), url)
                    return None

        except Exception as exc:
            logger.warning("Failed to fetch artifact from %s: %s", url, exc)
            return None

        ext = _ext_for_content_type(content_header)
        local_path = self._run_dir / f"{artifact_id}{ext}"

        try:
            local_path.write_bytes(content)
        except OSError as exc:
            logger.warning("Failed to write artifact to %s: %s", local_path, exc)
            return None

        artifact = EnrichmentArtifact(
            artifact_id=artifact_id,
            url=url,
            title=title or url,
            content_type=ext.lstrip("."),
            local_path=str(local_path),
            file_size_bytes=len(content),
            retrieved_at=datetime.now(UTC).isoformat(),
            run_id=self._run_id,
            entity_id=entity_id,
        )

        self._artifacts[artifact_id] = artifact
        self._by_entity.setdefault(entity_id, []).append(artifact_id)
        self._write_index()

        logger.info("Stored artifact %s (%d bytes) for entity %s", artifact_id, len(content), entity_id)
        return artifact

    def get(self, artifact_id: str) -> EnrichmentArtifact | None:
        return self._artifacts.get(artifact_id)

    def list_for_entity(self, entity_id: str) -> list[EnrichmentArtifact]:
        return [
            self._artifacts[aid]
            for aid in self._by_entity.get(entity_id, [])
            if aid in self._artifacts
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self._run_id,
            "artifact_count": len(self._artifacts),
            "artifacts": {aid: a.to_dict() for aid, a in self._artifacts.items()},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _artifact_id_for(self, url: str) -> str:
        """Deterministic UUID-like ID from URL hash."""
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return str(uuid.UUID(digest))

    def _write_index(self) -> None:
        import json
        index_path = self._run_dir / "index.json"
        try:
            index_path.write_text(json.dumps(self.to_dict(), indent=2))
        except OSError as exc:
            logger.warning("Failed to write artifact index: %s", exc)


def _ext_for_content_type(content_type: str) -> str:
    return {
        "application/pdf": ".pdf",
        "application/json": ".json",
        "text/plain": ".txt",
    }.get(content_type, ".html")
