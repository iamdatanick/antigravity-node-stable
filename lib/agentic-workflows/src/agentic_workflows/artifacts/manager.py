"""Artifact manager for high-level artifact operations."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentic_workflows.artifacts.generator import (
    Artifact,
    ArtifactGenerator,
    ArtifactType,
)
from agentic_workflows.artifacts.storage import ArtifactStorage, MemoryStorage


@dataclass
class ArtifactRef:
    """Reference to an artifact."""

    artifact_id: str
    name: str
    artifact_type: ArtifactType
    created_at: float
    content_hash: str

    def __str__(self) -> str:
        return f"ArtifactRef({self.name}, id={self.artifact_id})"


@dataclass
class ArtifactCollection:
    """A named collection of artifacts."""

    name: str
    description: str = ""
    artifact_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactManager:
    """High-level manager for artifact operations.

    Features:
    - Unified interface for generation and storage
    - Collections for grouping related artifacts
    - Versioning support
    - Event callbacks
    - Search and filtering
    """

    def __init__(
        self,
        storage: ArtifactStorage | None = None,
        agent_id: str = "",
        session_id: str = "",
        on_artifact_created: Callable[[Artifact], None] | None = None,
        on_artifact_deleted: Callable[[str], None] | None = None,
    ):
        """Initialize artifact manager.

        Args:
            storage: Storage backend (default: MemoryStorage).
            agent_id: Default agent ID.
            session_id: Default session ID.
            on_artifact_created: Callback when artifact is created.
            on_artifact_deleted: Callback when artifact is deleted.
        """
        self.storage = storage or MemoryStorage()
        self.generator = ArtifactGenerator(
            agent_id=agent_id,
            session_id=session_id,
        )
        self.on_artifact_created = on_artifact_created
        self.on_artifact_deleted = on_artifact_deleted

        self._collections: dict[str, ArtifactCollection] = {}
        self._versions: dict[str, list[str]] = {}  # name -> [artifact_ids]
        self._lock = threading.Lock()

    def create(
        self,
        name: str,
        artifact_type: ArtifactType,
        content: Any,
        collection: str | None = None,
        **metadata_kwargs,
    ) -> Artifact:
        """Create and store an artifact.

        Args:
            name: Artifact name.
            artifact_type: Type of artifact.
            content: Artifact content.
            collection: Optional collection to add to.
            **metadata_kwargs: Additional metadata.

        Returns:
            Created artifact.
        """
        artifact = self.generator.create(name, artifact_type, content, **metadata_kwargs)

        # Check for versioning
        with self._lock:
            if name in self._versions:
                # This is a new version
                versions = self._versions[name]
                artifact.metadata.version = len(versions) + 1
                if versions:
                    artifact.metadata.parent_id = versions[-1]
                versions.append(artifact.id)
            else:
                self._versions[name] = [artifact.id]

        # Store
        self.storage.save(artifact)

        # Add to collection
        if collection:
            self.add_to_collection(collection, artifact.id)

        # Callback
        if self.on_artifact_created:
            self.on_artifact_created(artifact)

        return artifact

    def create_code(
        self,
        name: str,
        content: str,
        language: str = "",
        **kwargs,
    ) -> Artifact:
        """Create a code artifact."""
        artifact = self.generator.code(name, content, language, **kwargs)
        self.storage.save(artifact)

        with self._lock:
            if name not in self._versions:
                self._versions[name] = []
            self._versions[name].append(artifact.id)

        if self.on_artifact_created:
            self.on_artifact_created(artifact)

        return artifact

    def create_report(
        self,
        name: str,
        title: str,
        sections: list[tuple[str, str]],
        **kwargs,
    ) -> Artifact:
        """Create a report artifact."""
        artifact = self.generator.report(name, title, sections, **kwargs)
        self.storage.save(artifact)

        with self._lock:
            if name not in self._versions:
                self._versions[name] = []
            self._versions[name].append(artifact.id)

        if self.on_artifact_created:
            self.on_artifact_created(artifact)

        return artifact

    def create_diff(
        self,
        name: str,
        old_content: str,
        new_content: str,
        **kwargs,
    ) -> Artifact:
        """Create a diff artifact."""
        artifact = self.generator.diff(name, old_content, new_content, **kwargs)
        self.storage.save(artifact)

        if self.on_artifact_created:
            self.on_artifact_created(artifact)

        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        """Get an artifact by ID."""
        return self.storage.load(artifact_id)

    def get_ref(self, artifact_id: str) -> ArtifactRef | None:
        """Get an artifact reference (lightweight)."""
        artifact = self.storage.load(artifact_id)
        if artifact:
            return ArtifactRef(
                artifact_id=artifact.id,
                name=artifact.name,
                artifact_type=artifact.artifact_type,
                created_at=artifact.metadata.created_at,
                content_hash=artifact.content_hash,
            )
        return None

    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        # Remove from versions
        with self._lock:
            for name, versions in self._versions.items():
                if artifact_id in versions:
                    versions.remove(artifact_id)
                    break

            # Remove from collections
            for collection in self._collections.values():
                if artifact_id in collection.artifact_ids:
                    collection.artifact_ids.remove(artifact_id)

        result = self.storage.delete(artifact_id)

        if result and self.on_artifact_deleted:
            self.on_artifact_deleted(artifact_id)

        return result

    def get_versions(self, name: str) -> list[Artifact]:
        """Get all versions of an artifact by name.

        Args:
            name: Artifact name.

        Returns:
            List of artifact versions (oldest first).
        """
        with self._lock:
            version_ids = self._versions.get(name, [])

        return [a for a in (self.storage.load(aid) for aid in version_ids) if a is not None]

    def get_latest_version(self, name: str) -> Artifact | None:
        """Get latest version of an artifact.

        Args:
            name: Artifact name.

        Returns:
            Latest artifact or None.
        """
        with self._lock:
            version_ids = self._versions.get(name, [])
            if not version_ids:
                return None
            return self.storage.load(version_ids[-1])

    def create_collection(
        self,
        name: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactCollection:
        """Create an artifact collection.

        Args:
            name: Collection name.
            description: Collection description.
            metadata: Additional metadata.

        Returns:
            Created collection.
        """
        collection = ArtifactCollection(
            name=name,
            description=description,
            metadata=metadata or {},
        )

        with self._lock:
            self._collections[name] = collection

        return collection

    def add_to_collection(self, collection_name: str, artifact_id: str) -> bool:
        """Add artifact to collection.

        Args:
            collection_name: Collection name.
            artifact_id: Artifact to add.

        Returns:
            True if added.
        """
        with self._lock:
            if collection_name not in self._collections:
                self._collections[collection_name] = ArtifactCollection(name=collection_name)

            collection = self._collections[collection_name]
            if artifact_id not in collection.artifact_ids:
                collection.artifact_ids.append(artifact_id)
                return True

        return False

    def get_collection(self, name: str) -> list[Artifact]:
        """Get all artifacts in a collection.

        Args:
            name: Collection name.

        Returns:
            List of artifacts.
        """
        with self._lock:
            collection = self._collections.get(name)
            if not collection:
                return []
            artifact_ids = list(collection.artifact_ids)

        return [a for a in (self.storage.load(aid) for aid in artifact_ids) if a is not None]

    def search(
        self,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        name_contains: str | None = None,
        limit: int = 100,
    ) -> list[Artifact]:
        """Search for artifacts.

        Args:
            artifact_type: Filter by type.
            tags: Filter by tags.
            agent_id: Filter by agent.
            name_contains: Filter by name substring.
            limit: Maximum results.

        Returns:
            Matching artifacts.
        """
        artifact_ids = self.storage.list_artifacts(
            artifact_type=artifact_type,
            tags=tags,
            agent_id=agent_id,
            limit=limit * 2 if name_contains else limit,
        )

        results = []
        for aid in artifact_ids:
            artifact = self.storage.load(aid)
            if artifact:
                if name_contains and name_contains.lower() not in artifact.name.lower():
                    continue
                results.append(artifact)
                if len(results) >= limit:
                    break

        return results

    def export_collection(
        self,
        collection_name: str,
        include_content: bool = True,
    ) -> dict[str, Any]:
        """Export a collection to dictionary.

        Args:
            collection_name: Collection to export.
            include_content: Include artifact content.

        Returns:
            Exported collection data.
        """
        with self._lock:
            collection = self._collections.get(collection_name)
            if not collection:
                return {}

        artifacts = self.get_collection(collection_name)

        return {
            "name": collection.name,
            "description": collection.description,
            "created_at": collection.created_at,
            "metadata": collection.metadata,
            "artifacts": [
                a.to_dict()
                if include_content
                else {
                    "id": a.id,
                    "name": a.name,
                    "type": a.artifact_type.value,
                    "created_at": a.metadata.created_at,
                }
                for a in artifacts
            ],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        storage_stats = self.storage.get_stats() if hasattr(self.storage, "get_stats") else {}

        with self._lock:
            collection_count = len(self._collections)
            versioned_count = len(self._versions)
            total_versions = sum(len(v) for v in self._versions.values())

        return {
            "storage": storage_stats,
            "collections": collection_count,
            "versioned_artifacts": versioned_count,
            "total_versions": total_versions,
        }

    def clear(self) -> int:
        """Clear all artifacts.

        Returns:
            Number cleared.
        """
        with self._lock:
            self._collections.clear()
            self._versions.clear()

        if hasattr(self.storage, "clear"):
            return self.storage.clear()

        # Manual clear
        count = 0
        for aid in self.storage.list_artifacts():
            if self.storage.delete(aid):
                count += 1
        return count
