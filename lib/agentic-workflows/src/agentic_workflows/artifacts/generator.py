"""Artifact generation utilities."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class ArtifactType(Enum):
    """Types of artifacts."""

    CODE = "code"
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    IMAGE = "image"
    BINARY = "binary"
    LOG = "log"
    REPORT = "report"
    CONFIG = "config"
    DIFF = "diff"


@dataclass
class ArtifactMetadata:
    """Metadata for an artifact."""

    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    agent_id: str = ""
    task_id: str = ""
    session_id: str = ""
    version: int = 1
    parent_id: str | None = None
    tags: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "created_at": self.created_at,
            "created_by": self.created_by,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "version": self.version,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMetadata:
        """Create from dictionary."""
        return cls(
            created_at=data.get("created_at", time.time()),
            created_by=data.get("created_by", ""),
            agent_id=data.get("agent_id", ""),
            task_id=data.get("task_id", ""),
            session_id=data.get("session_id", ""),
            version=data.get("version", 1),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            custom=data.get("custom", {}),
        )


@dataclass
class Artifact:
    """An artifact produced by an agent."""

    id: str
    name: str
    artifact_type: ArtifactType
    content: Any
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)
    mime_type: str = ""
    encoding: str = "utf-8"
    content_hash: str = ""

    def __post_init__(self):
        """Compute content hash if not provided."""
        if not self.content_hash and self.content is not None:
            self.content_hash = self._compute_hash()

        if not self.mime_type:
            self.mime_type = self._infer_mime_type()

    def _compute_hash(self) -> str:
        """Compute hash of content."""
        if isinstance(self.content, bytes):
            data = self.content
        elif isinstance(self.content, str):
            data = self.content.encode(self.encoding)
        else:
            data = json.dumps(self.content, sort_keys=True).encode()

        return hashlib.sha256(data).hexdigest()[:16]

    def _infer_mime_type(self) -> str:
        """Infer MIME type from artifact type."""
        mime_types = {
            ArtifactType.CODE: "text/plain",
            ArtifactType.TEXT: "text/plain",
            ArtifactType.JSON: "application/json",
            ArtifactType.MARKDOWN: "text/markdown",
            ArtifactType.HTML: "text/html",
            ArtifactType.CSV: "text/csv",
            ArtifactType.IMAGE: "image/png",
            ArtifactType.BINARY: "application/octet-stream",
            ArtifactType.LOG: "text/plain",
            ArtifactType.REPORT: "text/markdown",
            ArtifactType.CONFIG: "application/json",
            ArtifactType.DIFF: "text/x-diff",
        }
        return mime_types.get(self.artifact_type, "application/octet-stream")

    @property
    def size_bytes(self) -> int:
        """Get content size in bytes."""
        if isinstance(self.content, bytes):
            return len(self.content)
        elif isinstance(self.content, str):
            return len(self.content.encode(self.encoding))
        else:
            return len(json.dumps(self.content).encode())

    @property
    def is_text(self) -> bool:
        """Check if artifact is text-based."""
        return self.artifact_type in (
            ArtifactType.CODE,
            ArtifactType.TEXT,
            ArtifactType.MARKDOWN,
            ArtifactType.HTML,
            ArtifactType.CSV,
            ArtifactType.LOG,
            ArtifactType.REPORT,
            ArtifactType.DIFF,
        )

    def get_text(self) -> str:
        """Get content as text."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, bytes):
            return self.content.decode(self.encoding)
        else:
            return json.dumps(self.content, indent=2)

    def get_bytes(self) -> bytes:
        """Get content as bytes."""
        if isinstance(self.content, bytes):
            return self.content
        elif isinstance(self.content, str):
            return self.content.encode(self.encoding)
        else:
            return json.dumps(self.content).encode()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        content = self.content
        if isinstance(content, bytes):
            import base64
            content = base64.b64encode(content).decode()

        return {
            "id": self.id,
            "name": self.name,
            "artifact_type": self.artifact_type.value,
            "content": content,
            "metadata": self.metadata.to_dict(),
            "mime_type": self.mime_type,
            "encoding": self.encoding,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        """Create from dictionary."""
        content = data["content"]
        artifact_type = ArtifactType(data["artifact_type"])

        # Decode binary content
        if artifact_type == ArtifactType.BINARY or artifact_type == ArtifactType.IMAGE:
            import base64
            if isinstance(content, str):
                content = base64.b64decode(content)

        return cls(
            id=data["id"],
            name=data["name"],
            artifact_type=artifact_type,
            content=content,
            metadata=ArtifactMetadata.from_dict(data.get("metadata", {})),
            mime_type=data.get("mime_type", ""),
            encoding=data.get("encoding", "utf-8"),
            content_hash=data.get("content_hash", ""),
        )


class ArtifactGenerator:
    """Generator for creating artifacts.

    Provides factory methods for creating different types of artifacts
    with proper metadata and validation.
    """

    def __init__(
        self,
        agent_id: str = "",
        session_id: str = "",
        default_tags: list[str] | None = None,
    ):
        """Initialize generator.

        Args:
            agent_id: Default agent ID for artifacts.
            session_id: Default session ID.
            default_tags: Default tags to apply.
        """
        self.agent_id = agent_id
        self.session_id = session_id
        self.default_tags = default_tags or []
        self._counter = 0

    def _generate_id(self) -> str:
        """Generate unique artifact ID."""
        self._counter += 1
        return f"art-{uuid.uuid4().hex[:8]}-{self._counter}"

    def _create_metadata(
        self,
        created_by: str = "",
        task_id: str = "",
        tags: list[str] | None = None,
        parent_id: str | None = None,
        custom: dict[str, Any] | None = None,
    ) -> ArtifactMetadata:
        """Create metadata with defaults."""
        all_tags = list(self.default_tags)
        if tags:
            all_tags.extend(tags)

        return ArtifactMetadata(
            created_by=created_by,
            agent_id=self.agent_id,
            task_id=task_id,
            session_id=self.session_id,
            tags=all_tags,
            parent_id=parent_id,
            custom=custom or {},
        )

    def create(
        self,
        name: str,
        artifact_type: ArtifactType,
        content: Any,
        **metadata_kwargs,
    ) -> Artifact:
        """Create an artifact.

        Args:
            name: Artifact name.
            artifact_type: Type of artifact.
            content: Artifact content.
            **metadata_kwargs: Additional metadata options.

        Returns:
            Created artifact.
        """
        return Artifact(
            id=self._generate_id(),
            name=name,
            artifact_type=artifact_type,
            content=content,
            metadata=self._create_metadata(**metadata_kwargs),
        )

    def code(
        self,
        name: str,
        content: str,
        language: str = "",
        **kwargs,
    ) -> Artifact:
        """Create a code artifact.

        Args:
            name: File name.
            content: Code content.
            language: Programming language.
            **kwargs: Additional metadata.

        Returns:
            Code artifact.
        """
        custom = kwargs.pop("custom", {})
        if language:
            custom["language"] = language

        artifact = self.create(name, ArtifactType.CODE, content, custom=custom, **kwargs)

        # Infer MIME type from language
        mime_types = {
            "python": "text/x-python",
            "javascript": "text/javascript",
            "typescript": "text/typescript",
            "rust": "text/x-rust",
            "go": "text/x-go",
            "java": "text/x-java",
            "c": "text/x-c",
            "cpp": "text/x-c++",
            "html": "text/html",
            "css": "text/css",
            "sql": "text/x-sql",
            "shell": "text/x-shellscript",
            "yaml": "text/yaml",
            "json": "application/json",
        }
        if language.lower() in mime_types:
            artifact.mime_type = mime_types[language.lower()]

        return artifact

    def text(self, name: str, content: str, **kwargs) -> Artifact:
        """Create a text artifact."""
        return self.create(name, ArtifactType.TEXT, content, **kwargs)

    def json_artifact(self, name: str, content: Any, **kwargs) -> Artifact:
        """Create a JSON artifact."""
        return self.create(name, ArtifactType.JSON, content, **kwargs)

    def markdown(self, name: str, content: str, **kwargs) -> Artifact:
        """Create a markdown artifact."""
        return self.create(name, ArtifactType.MARKDOWN, content, **kwargs)

    def html(self, name: str, content: str, **kwargs) -> Artifact:
        """Create an HTML artifact."""
        return self.create(name, ArtifactType.HTML, content, **kwargs)

    def csv(self, name: str, content: str, **kwargs) -> Artifact:
        """Create a CSV artifact."""
        return self.create(name, ArtifactType.CSV, content, **kwargs)

    def log(self, name: str, content: str, **kwargs) -> Artifact:
        """Create a log artifact."""
        return self.create(name, ArtifactType.LOG, content, **kwargs)

    def report(
        self,
        name: str,
        title: str,
        sections: list[tuple[str, str]],
        **kwargs,
    ) -> Artifact:
        """Create a report artifact.

        Args:
            name: Report name.
            title: Report title.
            sections: List of (heading, content) tuples.
            **kwargs: Additional metadata.

        Returns:
            Report artifact.
        """
        lines = [f"# {title}", ""]

        for heading, content in sections:
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(content)
            lines.append("")

        return self.create(name, ArtifactType.REPORT, "\n".join(lines), **kwargs)

    def diff(
        self,
        name: str,
        old_content: str,
        new_content: str,
        context_lines: int = 3,
        **kwargs,
    ) -> Artifact:
        """Create a diff artifact.

        Args:
            name: Diff name.
            old_content: Original content.
            new_content: New content.
            context_lines: Lines of context.
            **kwargs: Additional metadata.

        Returns:
            Diff artifact.
        """
        import difflib

        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old",
            tofile="new",
            n=context_lines,
        )

        return self.create(name, ArtifactType.DIFF, "".join(diff), **kwargs)

    def binary(self, name: str, content: bytes, **kwargs) -> Artifact:
        """Create a binary artifact."""
        return self.create(name, ArtifactType.BINARY, content, **kwargs)

    def from_file(self, path: str | Path, **kwargs) -> Artifact:
        """Create artifact from file.

        Args:
            path: Path to file.
            **kwargs: Additional metadata.

        Returns:
            Artifact from file contents.
        """
        path = Path(path)
        name = path.name

        # Infer type from extension
        ext_types = {
            ".py": (ArtifactType.CODE, "python"),
            ".js": (ArtifactType.CODE, "javascript"),
            ".ts": (ArtifactType.CODE, "typescript"),
            ".go": (ArtifactType.CODE, "go"),
            ".rs": (ArtifactType.CODE, "rust"),
            ".java": (ArtifactType.CODE, "java"),
            ".c": (ArtifactType.CODE, "c"),
            ".cpp": (ArtifactType.CODE, "cpp"),
            ".h": (ArtifactType.CODE, "c"),
            ".json": (ArtifactType.JSON, None),
            ".md": (ArtifactType.MARKDOWN, None),
            ".html": (ArtifactType.HTML, None),
            ".htm": (ArtifactType.HTML, None),
            ".css": (ArtifactType.CODE, "css"),
            ".csv": (ArtifactType.CSV, None),
            ".txt": (ArtifactType.TEXT, None),
            ".log": (ArtifactType.LOG, None),
            ".yaml": (ArtifactType.CONFIG, None),
            ".yml": (ArtifactType.CONFIG, None),
            ".toml": (ArtifactType.CONFIG, None),
            ".png": (ArtifactType.IMAGE, None),
            ".jpg": (ArtifactType.IMAGE, None),
            ".jpeg": (ArtifactType.IMAGE, None),
            ".gif": (ArtifactType.IMAGE, None),
        }

        suffix = path.suffix.lower()
        artifact_type, language = ext_types.get(suffix, (ArtifactType.BINARY, None))

        # Read content
        if artifact_type in (ArtifactType.BINARY, ArtifactType.IMAGE):
            content = path.read_bytes()
        elif artifact_type == ArtifactType.JSON:
            content = json.loads(path.read_text())
        else:
            content = path.read_text()

        # Add file info to custom metadata
        custom = kwargs.pop("custom", {})
        custom["source_path"] = str(path)
        if language:
            custom["language"] = language

        return self.create(name, artifact_type, content, custom=custom, **kwargs)
