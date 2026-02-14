"""Context graph for DAG-based context management.

Enhanced with reducer pattern for state management (inspired by LangGraph).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


# =============================================================================
# Reducers for State Management
# =============================================================================


def add_reducer(current: list, new: Any) -> list:
    """Append new item(s) to list. Default reducer for conversation history."""
    if current is None:
        current = []
    if isinstance(new, list):
        return current + new
    return current + [new]


def replace_reducer(current: Any, new: Any) -> Any:
    """Replace current with new value. Default for most state fields."""
    return new


def merge_reducer(current: dict, new: dict) -> dict:
    """Merge dictionaries. Useful for metadata and artifacts."""
    if current is None:
        current = {}
    if new is None:
        return current
    return {**current, **new}


def max_reducer(current: float | None, new: float) -> float:
    """Keep maximum value. Useful for priority, trust scores."""
    if current is None:
        return new
    return max(current, new)


def min_reducer(current: float | None, new: float) -> float:
    """Keep minimum value. Useful for timestamps."""
    if current is None:
        return new
    return min(current, new)


def sum_reducer(current: float | None, new: float) -> float:
    """Sum values. Useful for counters."""
    if current is None:
        return new
    return current + new


@dataclass
class ReducerConfig:
    """Configuration for reducers on context graph state."""

    # Field name -> reducer function
    reducers: dict[str, Callable[[Any, Any], Any]] = field(default_factory=dict)

    def __post_init__(self):
        # Set default reducers
        if "conversation_history" not in self.reducers:
            self.reducers["conversation_history"] = add_reducer
        if "metadata" not in self.reducers:
            self.reducers["metadata"] = merge_reducer
        if "artifacts" not in self.reducers:
            self.reducers["artifacts"] = merge_reducer
        if "trust_score" not in self.reducers:
            self.reducers["trust_score"] = max_reducer

    def apply(self, field: str, current: Any, new: Any) -> Any:
        """Apply reducer for a field."""
        reducer = self.reducers.get(field, replace_reducer)
        return reducer(current, new)


class NodeType(Enum):
    """Types of context nodes."""

    USER_INPUT = "user_input"
    SYSTEM_PROMPT = "system_prompt"
    AGENT_OUTPUT = "agent_output"
    TOOL_RESULT = "tool_result"
    EXTERNAL_DATA = "external_data"
    DERIVED = "derived"
    CHECKPOINT = "checkpoint"


@dataclass
class ContextNode:
    """A node in the context graph."""

    id: str
    node_type: NodeType
    content: Any
    source: str  # Agent or tool that created this
    metadata: dict[str, Any] = field(default_factory=dict)

    # Graph relationships
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)

    # Trust and provenance
    trust_score: float = 1.0
    created_at: float = field(default_factory=time.time)

    # Optional fields
    expires_at: float | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """Check if node has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get node age in seconds."""
        return time.time() - self.created_at


class ContextGraph:
    """DAG-based context management.

    Maintains a directed acyclic graph of context nodes, where each
    node represents a piece of information with provenance tracking.

    Features:
    - DAG structure with parent/child relationships
    - Trust score propagation
    - Expiration and cleanup
    - Subgraph extraction
    - Serialization
    - Reducer-based state management (v5.1.0)
    - Checkpoint and rollback support
    """

    def __init__(
        self,
        max_nodes: int = 1000,
        default_ttl_seconds: float | None = None,
        reducer_config: ReducerConfig | None = None,
    ):
        """Initialize context graph.

        Args:
            max_nodes: Maximum nodes before cleanup.
            default_ttl_seconds: Default time-to-live for nodes.
            reducer_config: Configuration for state reducers.
        """
        self.max_nodes = max_nodes
        self.default_ttl_seconds = default_ttl_seconds
        self.reducer_config = reducer_config or ReducerConfig()

        self._nodes: dict[str, ContextNode] = {}
        self._roots: set[str] = set()  # Nodes with no parents

        # Managed state with reducers
        self._state: dict[str, Any] = {}

        # Checkpoint support
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._checkpoint_counter = 0

    def add_node(
        self,
        node_type: NodeType,
        content: Any,
        source: str,
        parents: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        trust_score: float = 1.0,
        ttl_seconds: float | None = None,
        tags: list[str] | None = None,
        node_id: str | None = None,
    ) -> ContextNode:
        """Add a node to the graph.

        Args:
            node_type: Type of the node.
            content: Node content.
            source: Source agent/tool.
            parents: Parent node IDs.
            metadata: Additional metadata.
            trust_score: Initial trust score.
            ttl_seconds: Time-to-live (None = use default).
            tags: Node tags for filtering.
            node_id: Custom node ID (auto-generated if None).

        Returns:
            Created node.

        Raises:
            ValueError: If parent doesn't exist or would create cycle.
        """
        # Generate ID
        if node_id is None:
            node_id = str(uuid.uuid4())[:12]

        # Validate parents
        parent_ids = parents or []
        for pid in parent_ids:
            if pid not in self._nodes:
                raise ValueError(f"Parent node '{pid}' not found")

        # Calculate expiration
        expires_at = None
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        if ttl:
            expires_at = time.time() + ttl

        # Create node
        node = ContextNode(
            id=node_id,
            node_type=node_type,
            content=content,
            source=source,
            parents=parent_ids,
            metadata=metadata or {},
            trust_score=trust_score,
            expires_at=expires_at,
            tags=tags or [],
        )

        # Add to graph
        self._nodes[node_id] = node

        # Update relationships
        if not parent_ids:
            self._roots.add(node_id)
        else:
            for pid in parent_ids:
                self._nodes[pid].children.append(node_id)

        # Cleanup if needed
        if len(self._nodes) > self.max_nodes:
            self._cleanup()

        return node

    def get_node(self, node_id: str) -> ContextNode | None:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and update relationships.

        Args:
            node_id: Node to remove.

        Returns:
            True if removed, False if not found.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return False

        # Remove from parents' children lists
        for pid in node.parents:
            parent = self._nodes.get(pid)
            if parent and node_id in parent.children:
                parent.children.remove(node_id)

        # Update children to point to this node's parents
        for cid in node.children:
            child = self._nodes.get(cid)
            if child:
                child.parents.remove(node_id)
                child.parents.extend(node.parents)
                if not child.parents:
                    self._roots.add(cid)

        # Remove node
        del self._nodes[node_id]
        self._roots.discard(node_id)

        return True

    def get_ancestors(self, node_id: str, max_depth: int = -1) -> list[ContextNode]:
        """Get all ancestors of a node.

        Args:
            node_id: Starting node.
            max_depth: Maximum depth (-1 = unlimited).

        Returns:
            List of ancestor nodes.
        """
        visited: set[str] = set()
        ancestors: list[ContextNode] = []

        def traverse(nid: str, depth: int):
            if nid in visited:
                return
            if max_depth >= 0 and depth > max_depth:
                return

            visited.add(nid)
            node = self._nodes.get(nid)
            if node:
                ancestors.append(node)
                for pid in node.parents:
                    traverse(pid, depth + 1)

        node = self._nodes.get(node_id)
        if node:
            for pid in node.parents:
                traverse(pid, 1)

        return ancestors

    def get_descendants(self, node_id: str, max_depth: int = -1) -> list[ContextNode]:
        """Get all descendants of a node.

        Args:
            node_id: Starting node.
            max_depth: Maximum depth (-1 = unlimited).

        Returns:
            List of descendant nodes.
        """
        visited: set[str] = set()
        descendants: list[ContextNode] = []

        def traverse(nid: str, depth: int):
            if nid in visited:
                return
            if max_depth >= 0 and depth > max_depth:
                return

            visited.add(nid)
            node = self._nodes.get(nid)
            if node:
                descendants.append(node)
                for cid in node.children:
                    traverse(cid, depth + 1)

        node = self._nodes.get(node_id)
        if node:
            for cid in node.children:
                traverse(cid, 1)

        return descendants

    def get_path(self, from_id: str, to_id: str) -> list[ContextNode] | None:
        """Find path between two nodes.

        Args:
            from_id: Starting node.
            to_id: Target node.

        Returns:
            Path as list of nodes, or None if no path.
        """
        if from_id not in self._nodes or to_id not in self._nodes:
            return None

        # BFS to find shortest path
        from collections import deque

        queue = deque([(from_id, [from_id])])
        visited = {from_id}

        while queue:
            current, path = queue.popleft()

            if current == to_id:
                return [self._nodes[nid] for nid in path]

            node = self._nodes.get(current)
            if node:
                for child_id in node.children:
                    if child_id not in visited:
                        visited.add(child_id)
                        queue.append((child_id, path + [child_id]))

        return None

    def get_subgraph(
        self,
        root_ids: list[str],
        max_depth: int = -1,
    ) -> list[ContextNode]:
        """Extract subgraph starting from roots.

        Args:
            root_ids: Starting node IDs.
            max_depth: Maximum depth.

        Returns:
            List of nodes in subgraph.
        """
        nodes: list[ContextNode] = []
        visited: set[str] = set()

        for root_id in root_ids:
            node = self._nodes.get(root_id)
            if node and root_id not in visited:
                visited.add(root_id)
                nodes.append(node)
                nodes.extend(self.get_descendants(root_id, max_depth))

        return nodes

    def filter_nodes(
        self,
        node_type: NodeType | None = None,
        source: str | None = None,
        tags: list[str] | None = None,
        min_trust: float | None = None,
        include_expired: bool = False,
    ) -> list[ContextNode]:
        """Filter nodes by criteria.

        Args:
            node_type: Filter by type.
            source: Filter by source.
            tags: Filter by tags (any match).
            min_trust: Minimum trust score.
            include_expired: Include expired nodes.

        Returns:
            Matching nodes.
        """
        results: list[ContextNode] = []

        for node in self._nodes.values():
            if not include_expired and node.is_expired:
                continue
            if node_type and node.node_type != node_type:
                continue
            if source and node.source != source:
                continue
            if min_trust and node.trust_score < min_trust:
                continue
            if tags and not any(t in node.tags for t in tags):
                continue

            results.append(node)

        return results

    def iterate(
        self,
        order: str = "creation",
        reverse: bool = False,
    ) -> Iterator[ContextNode]:
        """Iterate over nodes.

        Args:
            order: "creation", "trust", or "topological"
            reverse: Reverse order.

        Yields:
            Context nodes.
        """
        nodes = list(self._nodes.values())

        if order == "creation":
            nodes.sort(key=lambda n: n.created_at, reverse=reverse)
        elif order == "trust":
            nodes.sort(key=lambda n: n.trust_score, reverse=not reverse)
        elif order == "topological":
            nodes = self._topological_sort()
            if reverse:
                nodes = list(reversed(nodes))

        for node in nodes:
            yield node

    def _topological_sort(self) -> list[ContextNode]:
        """Topological sort of nodes."""
        in_degree = {nid: len(n.parents) for nid, n in self._nodes.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: list[ContextNode] = []

        while queue:
            nid = queue.pop(0)
            node = self._nodes.get(nid)
            if node:
                result.append(node)
                for child_id in node.children:
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0:
                        queue.append(child_id)

        return result

    def _cleanup(self) -> int:
        """Remove expired and old nodes.

        Returns:
            Number of nodes removed.
        """
        # Remove expired
        expired = [nid for nid, n in self._nodes.items() if n.is_expired]
        for nid in expired:
            self.remove_node(nid)

        # If still over limit, remove oldest non-root nodes
        if len(self._nodes) > self.max_nodes:
            non_roots = [
                (nid, n.created_at) for nid, n in self._nodes.items() if nid not in self._roots
            ]
            non_roots.sort(key=lambda x: x[1])

            to_remove = len(self._nodes) - self.max_nodes
            for nid, _ in non_roots[:to_remove]:
                self.remove_node(nid)

        return len(expired)

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        type_counts = {}
        for node in self._nodes.values():
            type_counts[node.node_type.value] = type_counts.get(node.node_type.value, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "root_nodes": len(self._roots),
            "type_counts": type_counts,
            "avg_trust": (
                sum(n.trust_score for n in self._nodes.values()) / len(self._nodes)
                if self._nodes
                else 0
            ),
            "expired_count": sum(1 for n in self._nodes.values() if n.is_expired),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": {
                nid: {
                    "id": n.id,
                    "node_type": n.node_type.value,
                    "content": n.content,
                    "source": n.source,
                    "parents": n.parents,
                    "children": n.children,
                    "trust_score": n.trust_score,
                    "created_at": n.created_at,
                    "expires_at": n.expires_at,
                    "tags": n.tags,
                    "metadata": n.metadata,
                }
                for nid, n in self._nodes.items()
            },
            "roots": list(self._roots),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextGraph:
        """Deserialize graph from dictionary."""
        graph = cls()

        # Create nodes
        for nid, node_data in data["nodes"].items():
            node = ContextNode(
                id=node_data["id"],
                node_type=NodeType(node_data["node_type"]),
                content=node_data["content"],
                source=node_data["source"],
                parents=node_data["parents"],
                children=node_data["children"],
                trust_score=node_data["trust_score"],
                created_at=node_data["created_at"],
                expires_at=node_data["expires_at"],
                tags=node_data["tags"],
                metadata=node_data["metadata"],
            )
            graph._nodes[nid] = node

        graph._roots = set(data["roots"])
        return graph

    def clear(self) -> None:
        """Clear all nodes."""
        self._nodes.clear()
        self._roots.clear()
        self._state.clear()

    # =========================================================================
    # Reducer-Based State Management
    # =========================================================================

    def update_state(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Update managed state using reducers.

        Args:
            updates: Dictionary of field updates.

        Returns:
            Updated state dictionary.

        Example:
            >>> graph.update_state({
            ...     "conversation_history": {"role": "user", "content": "Hello"},
            ...     "metadata": {"last_update": time.time()},
            ... })
        """
        for field, new_value in updates.items():
            current_value = self._state.get(field)
            self._state[field] = self.reducer_config.apply(field, current_value, new_value)

        return self._state.copy()

    def get_state(self, field: str | None = None) -> Any:
        """Get current state or specific field.

        Args:
            field: Optional specific field to get.

        Returns:
            Full state dict or specific field value.
        """
        if field is None:
            return self._state.copy()
        return self._state.get(field)

    def set_reducer(self, field: str, reducer: Callable[[Any, Any], Any]) -> None:
        """Set a custom reducer for a field.

        Args:
            field: Field name.
            reducer: Reducer function (current, new) -> result.
        """
        self.reducer_config.reducers[field] = reducer

    # =========================================================================
    # Checkpoint and Rollback Support
    # =========================================================================

    def create_checkpoint(self, name: str | None = None) -> str:
        """Create a checkpoint of current state.

        Args:
            name: Optional checkpoint name.

        Returns:
            Checkpoint ID.
        """
        if name is None:
            self._checkpoint_counter += 1
            name = f"checkpoint_{self._checkpoint_counter}"

        checkpoint_data = {
            "timestamp": time.time(),
            "state": {k: self._deep_copy(v) for k, v in self._state.items()},
            "node_ids": list(self._nodes.keys()),
            "roots": list(self._roots),
        }

        self._checkpoints[name] = checkpoint_data
        return name

    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback to a previous checkpoint.

        Args:
            checkpoint_id: Checkpoint to rollback to.

        Returns:
            True if successful.
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            return False

        # Restore state
        self._state = {k: self._deep_copy(v) for k, v in checkpoint["state"].items()}

        # Remove nodes added after checkpoint
        current_ids = set(self._nodes.keys())
        checkpoint_ids = set(checkpoint["node_ids"])
        to_remove = current_ids - checkpoint_ids

        for node_id in to_remove:
            self.remove_node(node_id)

        self._roots = set(checkpoint["roots"])
        return True

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints.

        Returns:
            List of checkpoint metadata.
        """
        return [
            {
                "id": cid,
                "timestamp": data["timestamp"],
                "node_count": len(data["node_ids"]),
                "state_fields": list(data["state"].keys()),
            }
            for cid, data in self._checkpoints.items()
        ]

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint to delete.

        Returns:
            True if deleted.
        """
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
            return True
        return False

    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy an object for checkpointing."""
        import copy

        try:
            return copy.deepcopy(obj)
        except Exception:
            # Fallback for non-copyable objects
            return obj

    # =========================================================================
    # Conversation History Management (uses add_reducer)
    # =========================================================================

    def add_message(
        self,
        role: str,
        content: str,
        agent_id: str = "",
        **metadata,
    ) -> None:
        """Add a message to conversation history using add_reducer.

        Args:
            role: Message role (user, assistant, system, tool).
            content: Message content.
            agent_id: ID of the agent that sent/received.
            **metadata: Additional metadata.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "agent_id": agent_id,
            **metadata,
        }
        self.update_state({"conversation_history": message})

    def get_conversation_history(self, limit: int = 0) -> list[dict[str, Any]]:
        """Get conversation history.

        Args:
            limit: Maximum messages to return (0 = all).

        Returns:
            List of messages.
        """
        history = self._state.get("conversation_history", [])
        if limit > 0:
            return history[-limit:]
        return history
