"""Artifact storage backends."""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from agentic_workflows.artifacts.generator import Artifact, ArtifactType


class ArtifactStorage(ABC):
    """Abstract base class for artifact storage."""

    @abstractmethod
    def save(self, artifact: Artifact) -> str:
        """Save an artifact.

        Args:
            artifact: Artifact to save.

        Returns:
            Storage path/key.
        """
        pass

    @abstractmethod
    def load(self, artifact_id: str) -> Artifact | None:
        """Load an artifact.

        Args:
            artifact_id: Artifact ID.

        Returns:
            Artifact or None if not found.
        """
        pass

    @abstractmethod
    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact.

        Args:
            artifact_id: Artifact ID.

        Returns:
            True if deleted.
        """
        pass

    @abstractmethod
    def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists.

        Args:
            artifact_id: Artifact ID.

        Returns:
            True if exists.
        """
        pass

    @abstractmethod
    def list_artifacts(
        self,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """List artifact IDs.

        Args:
            artifact_type: Filter by type.
            tags: Filter by tags (any match).
            agent_id: Filter by agent.
            limit: Maximum results.

        Returns:
            List of artifact IDs.
        """
        pass

    def iterate(
        self,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[Artifact]:
        """Iterate over artifacts.

        Args:
            artifact_type: Filter by type.
            tags: Filter by tags.

        Yields:
            Artifacts.
        """
        for artifact_id in self.list_artifacts(artifact_type=artifact_type, tags=tags):
            artifact = self.load(artifact_id)
            if artifact:
                yield artifact


class MemoryStorage(ArtifactStorage):
    """In-memory artifact storage.

    Good for development and testing.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize memory storage.

        Args:
            max_size: Maximum number of artifacts.
        """
        self.max_size = max_size
        self._artifacts: dict[str, Artifact] = {}
        self._lock = threading.Lock()

    def save(self, artifact: Artifact) -> str:
        """Save artifact to memory."""
        with self._lock:
            # Enforce size limit
            if len(self._artifacts) >= self.max_size:
                # Remove oldest
                oldest = min(
                    self._artifacts.values(),
                    key=lambda a: a.metadata.created_at,
                )
                del self._artifacts[oldest.id]

            self._artifacts[artifact.id] = artifact
            return artifact.id

    def load(self, artifact_id: str) -> Artifact | None:
        """Load artifact from memory."""
        return self._artifacts.get(artifact_id)

    def delete(self, artifact_id: str) -> bool:
        """Delete artifact from memory."""
        with self._lock:
            if artifact_id in self._artifacts:
                del self._artifacts[artifact_id]
                return True
            return False

    def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists."""
        return artifact_id in self._artifacts

    def list_artifacts(
        self,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """List artifact IDs."""
        results = []

        for artifact_id, artifact in self._artifacts.items():
            if artifact_type and artifact.artifact_type != artifact_type:
                continue
            if agent_id and artifact.metadata.agent_id != agent_id:
                continue
            if tags and not any(t in artifact.metadata.tags for t in tags):
                continue

            results.append(artifact_id)

            if limit and len(results) >= limit:
                break

        return results

    def clear(self) -> int:
        """Clear all artifacts.

        Returns:
            Number cleared.
        """
        with self._lock:
            count = len(self._artifacts)
            self._artifacts.clear()
            return count

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            total_size = sum(a.size_bytes for a in self._artifacts.values())
            by_type = {}
            for a in self._artifacts.values():
                t = a.artifact_type.value
                by_type[t] = by_type.get(t, 0) + 1

            return {
                "count": len(self._artifacts),
                "max_size": self.max_size,
                "total_bytes": total_size,
                "by_type": by_type,
            }


class FileStorage(ArtifactStorage):
    """File-based artifact storage.

    Stores artifacts as JSON files with optional content files.
    """

    def __init__(
        self,
        base_path: str | Path,
        separate_content: bool = True,
    ):
        """Initialize file storage.

        Args:
            base_path: Base directory for storage.
            separate_content: Store content in separate files.
        """
        self.base_path = Path(base_path)
        self.separate_content = separate_content

        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "metadata").mkdir(exist_ok=True)
        if separate_content:
            (self.base_path / "content").mkdir(exist_ok=True)

        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load artifact index."""
        index_path = self.base_path / "index.json"
        if index_path.exists():
            try:
                self._index = json.loads(index_path.read_text())
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        """Save artifact index."""
        index_path = self.base_path / "index.json"
        index_path.write_text(json.dumps(self._index, indent=2))

    def _metadata_path(self, artifact_id: str) -> Path:
        """Get metadata file path."""
        return self.base_path / "metadata" / f"{artifact_id}.json"

    def _content_path(self, artifact_id: str, extension: str = "") -> Path:
        """Get content file path."""
        return self.base_path / "content" / f"{artifact_id}{extension}"

    def save(self, artifact: Artifact) -> str:
        """Save artifact to file."""
        # Determine content extension
        ext_map = {
            ArtifactType.CODE: ".txt",
            ArtifactType.TEXT: ".txt",
            ArtifactType.JSON: ".json",
            ArtifactType.MARKDOWN: ".md",
            ArtifactType.HTML: ".html",
            ArtifactType.CSV: ".csv",
            ArtifactType.LOG: ".log",
            ArtifactType.REPORT: ".md",
            ArtifactType.CONFIG: ".json",
            ArtifactType.DIFF: ".diff",
            ArtifactType.BINARY: ".bin",
            ArtifactType.IMAGE: ".png",
        }
        extension = ext_map.get(artifact.artifact_type, ".bin")

        if self.separate_content:
            # Save metadata
            metadata = artifact.to_dict()
            metadata.pop("content", None)
            self._metadata_path(artifact.id).write_text(
                json.dumps(metadata, indent=2)
            )

            # Save content
            content_path = self._content_path(artifact.id, extension)
            if isinstance(artifact.content, bytes):
                content_path.write_bytes(artifact.content)
            elif isinstance(artifact.content, str):
                content_path.write_text(artifact.content)
            else:
                content_path.write_text(json.dumps(artifact.content, indent=2))
        else:
            # Save everything in one file
            self._metadata_path(artifact.id).write_text(
                json.dumps(artifact.to_dict(), indent=2)
            )

        # Update index
        self._index[artifact.id] = {
            "name": artifact.name,
            "type": artifact.artifact_type.value,
            "created_at": artifact.metadata.created_at,
            "agent_id": artifact.metadata.agent_id,
            "tags": artifact.metadata.tags,
        }
        self._save_index()

        return artifact.id

    def load(self, artifact_id: str) -> Artifact | None:
        """Load artifact from file."""
        metadata_path = self._metadata_path(artifact_id)
        if not metadata_path.exists():
            return None

        try:
            data = json.loads(metadata_path.read_text())

            if self.separate_content:
                # Find content file
                content_dir = self.base_path / "content"
                content_files = list(content_dir.glob(f"{artifact_id}.*"))

                if content_files:
                    content_path = content_files[0]
                    artifact_type = ArtifactType(data.get("artifact_type", "text"))

                    if artifact_type in (ArtifactType.BINARY, ArtifactType.IMAGE):
                        data["content"] = content_path.read_bytes()
                    elif artifact_type == ArtifactType.JSON:
                        data["content"] = json.loads(content_path.read_text())
                    else:
                        data["content"] = content_path.read_text()
                else:
                    data["content"] = ""

            return Artifact.from_dict(data)

        except Exception:
            return None

    def delete(self, artifact_id: str) -> bool:
        """Delete artifact files."""
        deleted = False

        # Delete metadata
        metadata_path = self._metadata_path(artifact_id)
        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True

        # Delete content
        if self.separate_content:
            content_dir = self.base_path / "content"
            for f in content_dir.glob(f"{artifact_id}.*"):
                f.unlink()
                deleted = True

        # Update index
        if artifact_id in self._index:
            del self._index[artifact_id]
            self._save_index()

        return deleted

    def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists."""
        return artifact_id in self._index

    def list_artifacts(
        self,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """List artifact IDs."""
        results = []

        for artifact_id, info in self._index.items():
            if artifact_type and info.get("type") != artifact_type.value:
                continue
            if agent_id and info.get("agent_id") != agent_id:
                continue
            if tags:
                artifact_tags = info.get("tags", [])
                if not any(t in artifact_tags for t in tags):
                    continue

            results.append(artifact_id)

            if limit and len(results) >= limit:
                break

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        total_size = 0

        # Count metadata files
        metadata_dir = self.base_path / "metadata"
        for f in metadata_dir.glob("*.json"):
            total_size += f.stat().st_size

        # Count content files
        if self.separate_content:
            content_dir = self.base_path / "content"
            for f in content_dir.iterdir():
                total_size += f.stat().st_size

        by_type = {}
        for info in self._index.values():
            t = info.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "count": len(self._index),
            "total_bytes": total_size,
            "by_type": by_type,
            "path": str(self.base_path),
        }

    def cleanup_orphans(self) -> int:
        """Remove orphaned content files.

        Returns:
            Number of files removed.
        """
        removed = 0

        if not self.separate_content:
            return 0

        content_dir = self.base_path / "content"
        for f in content_dir.iterdir():
            artifact_id = f.stem
            if artifact_id not in self._index:
                f.unlink()
                removed += 1

        return removed
