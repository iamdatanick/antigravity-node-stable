"""
Persistence Layer for Context and Learning Data.

Connects Scratchpad, LearningContextGraph, and ContextGraph to storage backends
for automatic save/load across sessions.

Usage:
    from agentic_workflows.core.persistence import ContextPersistence
    from agentic_workflows.storage import RedisStorage

    # Create persistence layer
    storage = RedisStorage(...)
    persistence = ContextPersistence(storage)

    # Save context
    await persistence.save_scratchpad("session-123", scratchpad)
    await persistence.save_learning_graph("agent-001", learning_graph)
    await persistence.save_context_graph("workflow-456", context_graph)

    # Load context
    scratchpad = await persistence.load_scratchpad("session-123")
    learning_graph = await persistence.load_learning_graph("agent-001")
    context_graph = await persistence.load_context_graph("workflow-456")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_workflows.storage.base import StorageBackend

from agentic_workflows.context.graph import ContextGraph

from .context_graph import LearningContextGraph
from .scratchpad import Scratchpad


@dataclass
class PersistenceConfig:
    """Configuration for context persistence."""

    # Key prefixes
    scratchpad_prefix: str = "scratchpad:"
    learning_graph_prefix: str = "learning:"
    context_graph_prefix: str = "context:"

    # TTLs (None = no expiration)
    scratchpad_ttl: int | None = 86400  # 24 hours
    learning_graph_ttl: int | None = None  # Persist indefinitely
    context_graph_ttl: int | None = 604800  # 7 days

    # Auto-save interval (seconds)
    auto_save_interval: float = 300.0  # 5 minutes

    # Compression for large graphs
    compress_threshold: int = 10000  # Compress if > 10KB


class ContextPersistence:
    """Persistence layer for context and learning data.

    Features:
    - Save/load Scratchpad working memory
    - Save/load LearningContextGraph insights
    - Save/load ContextGraph DAG
    - Auto-save support
    - Metadata tracking
    """

    def __init__(
        self,
        storage: StorageBackend,
        config: PersistenceConfig | None = None,
    ):
        """Initialize persistence layer.

        Args:
            storage: Storage backend (Redis, Postgres, etc.)
            config: Persistence configuration
        """
        self.storage = storage
        self.config = config or PersistenceConfig()
        self._last_save: dict[str, float] = {}

    # ===== Scratchpad Persistence =====

    async def save_scratchpad(
        self,
        session_id: str,
        scratchpad: Scratchpad,
        ttl: int | None = None,
    ) -> bool:
        """Save scratchpad to storage.

        Args:
            session_id: Session identifier
            scratchpad: Scratchpad to save
            ttl: Override TTL (None = use config)

        Returns:
            True if saved successfully
        """
        key = f"{self.config.scratchpad_prefix}{session_id}"
        data = {
            "type": "scratchpad",
            "session_id": session_id,
            "saved_at": time.time(),
            "summary": scratchpad.get_summary(),
            "data": scratchpad.export(),
        }

        ttl = ttl if ttl is not None else self.config.scratchpad_ttl
        await self.storage.set(key, json.dumps(data), ttl=ttl)
        self._last_save[key] = time.time()
        return True

    async def load_scratchpad(
        self,
        session_id: str,
        max_entries: int = 100,
    ) -> Scratchpad | None:
        """Load scratchpad from storage.

        Args:
            session_id: Session identifier
            max_entries: Max entries for new scratchpad

        Returns:
            Loaded scratchpad or None if not found
        """
        key = f"{self.config.scratchpad_prefix}{session_id}"
        raw = await self.storage.get(key)

        if raw is None:
            return None

        data = json.loads(raw)
        scratchpad = Scratchpad(max_entries=max_entries)
        scratchpad.import_data(data["data"])
        return scratchpad

    async def delete_scratchpad(self, session_id: str) -> bool:
        """Delete scratchpad from storage."""
        key = f"{self.config.scratchpad_prefix}{session_id}"
        return await self.storage.delete(key)

    # ===== Learning Context Graph Persistence =====

    async def save_learning_graph(
        self,
        agent_id: str,
        graph: LearningContextGraph,
        ttl: int | None = None,
    ) -> bool:
        """Save learning context graph to storage.

        Args:
            agent_id: Agent identifier
            graph: Learning graph to save
            ttl: Override TTL (None = use config)

        Returns:
            True if saved successfully
        """
        key = f"{self.config.learning_graph_prefix}{agent_id}"
        data = {
            "type": "learning_graph",
            "agent_id": agent_id,
            "saved_at": time.time(),
            "statistics": graph.get_statistics(),
            "data": graph.export(),
        }

        ttl = ttl if ttl is not None else self.config.learning_graph_ttl
        await self.storage.set(key, json.dumps(data), ttl=ttl)
        self._last_save[key] = time.time()
        return True

    async def load_learning_graph(
        self,
        agent_id: str,
        max_nodes: int = 10000,
    ) -> LearningContextGraph | None:
        """Load learning context graph from storage.

        Args:
            agent_id: Agent identifier
            max_nodes: Max nodes for graph

        Returns:
            Loaded graph or None if not found
        """
        key = f"{self.config.learning_graph_prefix}{agent_id}"
        raw = await self.storage.get(key)

        if raw is None:
            return None

        data = json.loads(raw)
        graph = LearningContextGraph(max_nodes=max_nodes)
        graph.import_data(data["data"])
        return graph

    async def delete_learning_graph(self, agent_id: str) -> bool:
        """Delete learning graph from storage."""
        key = f"{self.config.learning_graph_prefix}{agent_id}"
        return await self.storage.delete(key)

    # ===== Context Graph Persistence =====

    async def save_context_graph(
        self,
        workflow_id: str,
        graph: ContextGraph,
        ttl: int | None = None,
    ) -> bool:
        """Save context graph to storage.

        Args:
            workflow_id: Workflow identifier
            graph: Context graph to save
            ttl: Override TTL (None = use config)

        Returns:
            True if saved successfully
        """
        key = f"{self.config.context_graph_prefix}{workflow_id}"
        data = {
            "type": "context_graph",
            "workflow_id": workflow_id,
            "saved_at": time.time(),
            "stats": graph.get_stats(),
            "data": graph.to_dict(),
        }

        ttl = ttl if ttl is not None else self.config.context_graph_ttl
        await self.storage.set(key, json.dumps(data), ttl=ttl)
        self._last_save[key] = time.time()
        return True

    async def load_context_graph(
        self,
        workflow_id: str,
    ) -> ContextGraph | None:
        """Load context graph from storage.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Loaded graph or None if not found
        """
        key = f"{self.config.context_graph_prefix}{workflow_id}"
        raw = await self.storage.get(key)

        if raw is None:
            return None

        data = json.loads(raw)
        graph = ContextGraph.from_dict(data["data"])
        return graph

    async def delete_context_graph(self, workflow_id: str) -> bool:
        """Delete context graph from storage."""
        key = f"{self.config.context_graph_prefix}{workflow_id}"
        return await self.storage.delete(key)

    # ===== Bulk Operations =====

    async def list_scratchpads(self) -> list[str]:
        """List all saved scratchpad session IDs."""
        pattern = f"{self.config.scratchpad_prefix}*"
        keys = await self.storage.keys(pattern)
        prefix_len = len(self.config.scratchpad_prefix)
        return [k[prefix_len:] for k in keys]

    async def list_learning_graphs(self) -> list[str]:
        """List all saved learning graph agent IDs."""
        pattern = f"{self.config.learning_graph_prefix}*"
        keys = await self.storage.keys(pattern)
        prefix_len = len(self.config.learning_graph_prefix)
        return [k[prefix_len:] for k in keys]

    async def list_context_graphs(self) -> list[str]:
        """List all saved context graph workflow IDs."""
        pattern = f"{self.config.context_graph_prefix}*"
        keys = await self.storage.keys(pattern)
        prefix_len = len(self.config.context_graph_prefix)
        return [k[prefix_len:] for k in keys]

    async def get_metadata(self, key_type: str, identifier: str) -> dict | None:
        """Get metadata about a saved item without loading full data.

        Args:
            key_type: "scratchpad", "learning", or "context"
            identifier: The session/agent/workflow ID

        Returns:
            Metadata dict or None
        """
        prefix_map = {
            "scratchpad": self.config.scratchpad_prefix,
            "learning": self.config.learning_graph_prefix,
            "context": self.config.context_graph_prefix,
        }

        prefix = prefix_map.get(key_type)
        if not prefix:
            return None

        key = f"{prefix}{identifier}"
        raw = await self.storage.get(key)

        if raw is None:
            return None

        data = json.loads(raw)
        # Return metadata without the full data payload
        return {
            "type": data.get("type"),
            "identifier": identifier,
            "saved_at": data.get("saved_at"),
            "summary": data.get("summary") or data.get("statistics") or data.get("stats"),
        }

    # ===== Auto-Save Support =====

    def should_auto_save(self, key: str) -> bool:
        """Check if auto-save is needed for a key."""
        last = self._last_save.get(key, 0)
        return time.time() - last >= self.config.auto_save_interval

    async def auto_save_scratchpad(
        self,
        session_id: str,
        scratchpad: Scratchpad,
    ) -> bool:
        """Auto-save scratchpad if interval has passed."""
        key = f"{self.config.scratchpad_prefix}{session_id}"
        if self.should_auto_save(key):
            return await self.save_scratchpad(session_id, scratchpad)
        return False

    async def auto_save_learning_graph(
        self,
        agent_id: str,
        graph: LearningContextGraph,
    ) -> bool:
        """Auto-save learning graph if interval has passed."""
        key = f"{self.config.learning_graph_prefix}{agent_id}"
        if self.should_auto_save(key):
            return await self.save_learning_graph(agent_id, graph)
        return False

    async def auto_save_context_graph(
        self,
        workflow_id: str,
        graph: ContextGraph,
    ) -> bool:
        """Auto-save context graph if interval has passed."""
        key = f"{self.config.context_graph_prefix}{workflow_id}"
        if self.should_auto_save(key):
            return await self.save_context_graph(workflow_id, graph)
        return False


class FileContextPersistence:
    """Simple file-based persistence for local development.

    Saves context data as JSON files in a directory.
    """

    def __init__(self, base_dir: str = "./.agentic_context"):
        """Initialize file persistence.

        Args:
            base_dir: Directory for storing context files
        """
        import os

        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(f"{base_dir}/scratchpads", exist_ok=True)
        os.makedirs(f"{base_dir}/learning", exist_ok=True)
        os.makedirs(f"{base_dir}/context", exist_ok=True)

    def _safe_filename(self, name: str) -> str:
        """Convert name to safe filename."""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

    def save_scratchpad(self, session_id: str, scratchpad: Scratchpad) -> str:
        """Save scratchpad to file."""
        filename = f"{self.base_dir}/scratchpads/{self._safe_filename(session_id)}.json"
        data = {
            "type": "scratchpad",
            "session_id": session_id,
            "saved_at": time.time(),
            "summary": scratchpad.get_summary(),
            "data": scratchpad.export(),
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        return filename

    def load_scratchpad(self, session_id: str) -> Scratchpad | None:
        """Load scratchpad from file."""
        import os

        filename = f"{self.base_dir}/scratchpads/{self._safe_filename(session_id)}.json"
        if not os.path.exists(filename):
            return None

        with open(filename) as f:
            data = json.load(f)

        scratchpad = Scratchpad()
        scratchpad.import_data(data["data"])
        return scratchpad

    def save_learning_graph(self, agent_id: str, graph: LearningContextGraph) -> str:
        """Save learning graph to file."""
        filename = f"{self.base_dir}/learning/{self._safe_filename(agent_id)}.json"
        data = {
            "type": "learning_graph",
            "agent_id": agent_id,
            "saved_at": time.time(),
            "statistics": graph.get_statistics(),
            "data": graph.export(),
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        return filename

    def load_learning_graph(self, agent_id: str) -> LearningContextGraph | None:
        """Load learning graph from file."""
        import os

        filename = f"{self.base_dir}/learning/{self._safe_filename(agent_id)}.json"
        if not os.path.exists(filename):
            return None

        with open(filename) as f:
            data = json.load(f)

        graph = LearningContextGraph()
        graph.import_data(data["data"])
        return graph

    def save_context_graph(self, workflow_id: str, graph: ContextGraph) -> str:
        """Save context graph to file."""
        filename = f"{self.base_dir}/context/{self._safe_filename(workflow_id)}.json"
        data = {
            "type": "context_graph",
            "workflow_id": workflow_id,
            "saved_at": time.time(),
            "stats": graph.get_stats(),
            "data": graph.to_dict(),
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        return filename

    def load_context_graph(self, workflow_id: str) -> ContextGraph | None:
        """Load context graph from file."""
        import os

        filename = f"{self.base_dir}/context/{self._safe_filename(workflow_id)}.json"
        if not os.path.exists(filename):
            return None

        with open(filename) as f:
            data = json.load(f)

        return ContextGraph.from_dict(data["data"])

    def list_all(self) -> dict[str, list[str]]:
        """List all saved context data."""
        import os

        result = {
            "scratchpads": [],
            "learning_graphs": [],
            "context_graphs": [],
        }

        for f in os.listdir(f"{self.base_dir}/scratchpads"):
            if f.endswith(".json"):
                result["scratchpads"].append(f[:-5])

        for f in os.listdir(f"{self.base_dir}/learning"):
            if f.endswith(".json"):
                result["learning_graphs"].append(f[:-5])

        for f in os.listdir(f"{self.base_dir}/context"):
            if f.endswith(".json"):
                result["context_graphs"].append(f[:-5])

        return result
