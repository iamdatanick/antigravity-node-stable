"""Provenance tracking for context origin and lineage."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(Enum):
    """Types of data sources."""

    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"
    TOOL = "tool"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    FILE = "file"
    DERIVED = "derived"


@dataclass
class Source:
    """A data source."""

    source_type: SourceType
    identifier: str  # e.g., "code-reviewer", "Read", "api.example.com"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_identifier(self) -> str:
        """Get full source identifier."""
        return f"{self.source_type.value}:{self.identifier}"


@dataclass
class ProvenanceEntry:
    """A single entry in the provenance chain."""

    source: Source
    timestamp: float
    action: str  # e.g., "created", "transformed", "validated"
    description: str = ""
    input_hash: str | None = None
    output_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Provenance:
    """Complete provenance record for a piece of data."""

    id: str
    original_source: Source
    created_at: float = field(default_factory=time.time)
    chain: list[ProvenanceEntry] = field(default_factory=list)
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def current_source(self) -> Source:
        """Get most recent source."""
        if self.chain:
            return self.chain[-1].source
        return self.original_source

    @property
    def hop_count(self) -> int:
        """Get number of transformations/hops."""
        return len(self.chain)

    @property
    def sources(self) -> list[Source]:
        """Get all sources in order."""
        sources = [self.original_source]
        for entry in self.chain:
            sources.append(entry.source)
        return sources

    def add_entry(
        self,
        source: Source,
        action: str,
        description: str = "",
        input_hash: str | None = None,
        output_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an entry to the provenance chain.

        Args:
            source: Source of this action.
            action: Action performed.
            description: Description of action.
            input_hash: Hash of input data.
            output_hash: Hash of output data.
            metadata: Additional metadata.
        """
        entry = ProvenanceEntry(
            source=source,
            timestamp=time.time(),
            action=action,
            description=description,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata=metadata or {},
        )
        self.chain.append(entry)
        if output_hash:
            self.content_hash = output_hash

    def verify_chain(self) -> tuple[bool, str]:
        """Verify integrity of the provenance chain.

        Returns:
            Tuple of (is_valid, error_message).
        """
        prev_output_hash = None

        for i, entry in enumerate(self.chain):
            # Check hash continuity
            if prev_output_hash and entry.input_hash:
                if prev_output_hash != entry.input_hash:
                    return (
                        False,
                        f"Hash mismatch at entry {i}: expected {prev_output_hash}, got {entry.input_hash}",
                    )

            # Check timestamp ordering
            if i > 0 and entry.timestamp < self.chain[i - 1].timestamp:
                return False, f"Timestamp out of order at entry {i}"

            prev_output_hash = entry.output_hash

        return True, ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "original_source": {
                "source_type": self.original_source.source_type.value,
                "identifier": self.original_source.identifier,
                "metadata": self.original_source.metadata,
            },
            "created_at": self.created_at,
            "chain": [
                {
                    "source": {
                        "source_type": e.source.source_type.value,
                        "identifier": e.source.identifier,
                        "metadata": e.source.metadata,
                    },
                    "timestamp": e.timestamp,
                    "action": e.action,
                    "description": e.description,
                    "input_hash": e.input_hash,
                    "output_hash": e.output_hash,
                    "metadata": e.metadata,
                }
                for e in self.chain
            ],
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        """Deserialize from dictionary."""
        original_source = Source(
            source_type=SourceType(data["original_source"]["source_type"]),
            identifier=data["original_source"]["identifier"],
            metadata=data["original_source"].get("metadata", {}),
        )

        prov = cls(
            id=data["id"],
            original_source=original_source,
            created_at=data.get("created_at", time.time()),
            content_hash=data.get("content_hash"),
            metadata=data.get("metadata", {}),
        )

        for entry_data in data.get("chain", []):
            source = Source(
                source_type=SourceType(entry_data["source"]["source_type"]),
                identifier=entry_data["source"]["identifier"],
                metadata=entry_data["source"].get("metadata", {}),
            )
            entry = ProvenanceEntry(
                source=source,
                timestamp=entry_data["timestamp"],
                action=entry_data["action"],
                description=entry_data.get("description", ""),
                input_hash=entry_data.get("input_hash"),
                output_hash=entry_data.get("output_hash"),
                metadata=entry_data.get("metadata", {}),
            )
            prov.chain.append(entry)

        return prov


class ProvenanceChain:
    """Manager for tracking provenance across multiple data items."""

    def __init__(self):
        """Initialize provenance chain manager."""
        self._records: dict[str, Provenance] = {}
        self._hash_index: dict[str, str] = {}  # hash -> provenance id

    def create(
        self,
        data_id: str,
        source: Source,
        content: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Provenance:
        """Create provenance record for new data.

        Args:
            data_id: Unique data identifier.
            source: Original data source.
            content: Data content (for hashing).
            metadata: Additional metadata.

        Returns:
            Created provenance record.
        """
        content_hash = None
        if content is not None:
            content_hash = self._compute_hash(content)

        prov = Provenance(
            id=data_id,
            original_source=source,
            content_hash=content_hash,
            metadata=metadata or {},
        )

        self._records[data_id] = prov
        if content_hash:
            self._hash_index[content_hash] = data_id

        return prov

    def record_transformation(
        self,
        data_id: str,
        source: Source,
        action: str,
        input_content: Any = None,
        output_content: Any = None,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Provenance | None:
        """Record a transformation of data.

        Args:
            data_id: Data identifier.
            source: Source performing transformation.
            action: Action name.
            input_content: Input data (for hashing).
            output_content: Output data (for hashing).
            description: Action description.
            metadata: Additional metadata.

        Returns:
            Updated provenance record, or None if not found.
        """
        prov = self._records.get(data_id)
        if prov is None:
            return None

        input_hash = self._compute_hash(input_content) if input_content else None
        output_hash = self._compute_hash(output_content) if output_content else None

        prov.add_entry(
            source=source,
            action=action,
            description=description,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata=metadata,
        )

        if output_hash:
            self._hash_index[output_hash] = data_id

        return prov

    def get(self, data_id: str) -> Provenance | None:
        """Get provenance record by ID."""
        return self._records.get(data_id)

    def get_by_hash(self, content_hash: str) -> Provenance | None:
        """Get provenance record by content hash."""
        data_id = self._hash_index.get(content_hash)
        if data_id:
            return self._records.get(data_id)
        return None

    def find_by_source(
        self, source_type: SourceType, identifier: str | None = None
    ) -> list[Provenance]:
        """Find all provenance records from a source.

        Args:
            source_type: Type of source.
            identifier: Optional identifier to match.

        Returns:
            Matching provenance records.
        """
        results = []
        for prov in self._records.values():
            for source in prov.sources:
                if source.source_type == source_type:
                    if identifier is None or source.identifier == identifier:
                        results.append(prov)
                        break
        return results

    def verify_all(self) -> dict[str, tuple[bool, str]]:
        """Verify all provenance chains.

        Returns:
            Dict of data_id to (is_valid, error_message).
        """
        return {data_id: prov.verify_chain() for data_id, prov in self._records.items()}

    def _compute_hash(self, content: Any) -> str:
        """Compute hash of content."""
        if isinstance(content, bytes):
            data = content
        elif isinstance(content, str):
            data = content.encode()
        else:
            data = str(content).encode()

        return hashlib.sha256(data).hexdigest()[:16]

    def get_lineage(self, data_id: str) -> list[dict[str, Any]]:
        """Get human-readable lineage for data.

        Args:
            data_id: Data identifier.

        Returns:
            List of lineage entries.
        """
        prov = self._records.get(data_id)
        if prov is None:
            return []

        lineage = [
            {
                "step": 0,
                "source": prov.original_source.full_identifier,
                "action": "created",
                "timestamp": prov.created_at,
            }
        ]

        for i, entry in enumerate(prov.chain):
            lineage.append(
                {
                    "step": i + 1,
                    "source": entry.source.full_identifier,
                    "action": entry.action,
                    "description": entry.description,
                    "timestamp": entry.timestamp,
                }
            )

        return lineage

    def export(self) -> dict[str, Any]:
        """Export all provenance records."""
        return {data_id: prov.to_dict() for data_id, prov in self._records.items()}

    def import_records(self, data: dict[str, Any]) -> int:
        """Import provenance records.

        Args:
            data: Exported provenance data.

        Returns:
            Number of records imported.
        """
        count = 0
        for data_id, prov_data in data.items():
            prov = Provenance.from_dict(prov_data)
            self._records[data_id] = prov
            if prov.content_hash:
                self._hash_index[prov.content_hash] = data_id
            count += 1
        return count

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()
        self._hash_index.clear()
