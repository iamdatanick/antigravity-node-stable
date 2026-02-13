"""Learning Context Graph - Logs interactions and learns over time.

Maintains a graph of all context, decisions, and outcomes to enable
the agent to learn from experience and make better decisions.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EdgeType(Enum):
    """Types of edges in the context graph."""

    DERIVED_FROM = "derived_from"  # Node was derived from another
    CAUSED = "caused"  # Action caused outcome
    SIMILAR_TO = "similar_to"  # Semantic similarity
    CONTRADICTS = "contradicts"  # Conflicting information
    SUPPORTS = "supports"  # Supporting evidence
    SUPERSEDES = "supersedes"  # New info replaces old
    REFERENCES = "references"  # References another node
    FOLLOWED_BY = "followed_by"  # Temporal sequence


class NodeType(Enum):
    """Types of nodes in the context graph."""

    TASK = "task"  # User task/request
    THOUGHT = "thought"  # Agent reasoning
    DECISION = "decision"  # Decision made
    ACTION = "action"  # Action taken
    OUTCOME = "outcome"  # Result of action
    OBSERVATION = "observation"  # Observed fact
    INSIGHT = "insight"  # Learned insight
    PATTERN = "pattern"  # Detected pattern
    ERROR = "error"  # Error/failure
    CORRECTION = "correction"  # Correction of error


@dataclass
class ContextNode:
    """A node in the context graph."""

    id: str
    node_type: NodeType
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Embeddings for similarity search
    embedding: list[float] | None = None

    # Quality/relevance scores
    importance: float = 0.5
    confidence: float = 0.5
    usefulness: float = 0.5  # Updated based on outcomes

    # Learning-related
    access_count: int = 0
    last_accessed: float = 0.0
    decay_rate: float = 0.01  # How fast importance decays

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "importance": self.importance,
            "confidence": self.confidence,
            "usefulness": self.usefulness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ContextNode:
        return cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 0.5),
            confidence=data.get("confidence", 0.5),
            usefulness=data.get("usefulness", 0.5),
        )


@dataclass
class Edge:
    """An edge connecting two nodes."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class LearningInsight:
    """An insight learned from experience."""

    pattern: str
    evidence_count: int
    confidence: float
    first_seen: float
    last_seen: float
    applications: int = 0  # Times this insight was used
    success_rate: float = 0.5  # How often using this insight led to success
    insight_type: str = "pattern"  # Type of insight

    @property
    def content(self) -> str:
        """Return pattern as content for compatibility."""
        return self.pattern

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "evidence_count": self.evidence_count,
            "confidence": self.confidence,
            "applications": self.applications,
            "success_rate": self.success_rate,
            "insight_type": self.insight_type,
        }


class LearningContextGraph:
    """A graph that learns from interactions.

    Features:
    - Stores all context, thoughts, decisions, and outcomes
    - Tracks causal relationships (what caused what)
    - Detects patterns across interactions
    - Learns which approaches work best
    - Provides relevant context for new tasks
    - Supports similarity-based retrieval
    """

    def __init__(
        self,
        max_nodes: int = 10000,
        decay_interval: float = 3600.0,  # 1 hour
        embedding_fn: callable | None = None,
    ):
        """Initialize the learning context graph.

        Args:
            max_nodes: Maximum nodes to retain
            decay_interval: How often to decay importance
            embedding_fn: Function to generate embeddings
        """
        self.max_nodes = max_nodes
        self.decay_interval = decay_interval
        self.embedding_fn = embedding_fn

        # Core storage
        self._nodes: dict[str, ContextNode] = {}
        self._edges: list[Edge] = []

        # Indexes for efficient lookup
        self._by_type: dict[NodeType, set[str]] = defaultdict(set)
        self._outgoing: dict[str, list[Edge]] = defaultdict(list)
        self._incoming: dict[str, list[Edge]] = defaultdict(list)

        # Learning storage
        self._insights: dict[str, LearningInsight] = {}
        self._patterns: dict[str, list[str]] = defaultdict(list)  # pattern -> node_ids

        # Task-outcome tracking for learning
        self._task_outcomes: dict[str, dict] = {}  # task_id -> outcome data

        # Temporal tracking
        self._last_decay = time.time()

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        timestamp = str(time.time())
        return hashlib.sha256(f"{content}{timestamp}".encode()).hexdigest()[:16]

    def add_node(
        self,
        content: str,
        node_type: NodeType,
        metadata: dict | None = None,
        importance: float = 0.5,
        confidence: float = 0.5,
        node_id: str | None = None,
    ) -> ContextNode:
        """Add a node to the graph.

        Args:
            content: Node content
            node_type: Type of node
            metadata: Additional metadata
            importance: Initial importance score
            confidence: Confidence in this information
            node_id: Optional specific ID

        Returns:
            Created node
        """
        node_id = node_id or self._generate_id(content)

        node = ContextNode(
            id=node_id,
            node_type=node_type,
            content=content,
            metadata=metadata or {},
            importance=importance,
            confidence=confidence,
        )

        # Generate embedding if function available
        if self.embedding_fn:
            try:
                node.embedding = self.embedding_fn(content)
            except Exception:
                pass

        self._nodes[node_id] = node
        self._by_type[node_type].add(node_id)

        # Prune if needed
        if len(self._nodes) > self.max_nodes:
            self._prune_least_important()

        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> Edge | None:
        """Add an edge between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            weight: Edge weight
            metadata: Additional metadata

        Returns:
            Created edge or None if nodes don't exist
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )

        self._edges.append(edge)
        self._outgoing[source_id].append(edge)
        self._incoming[target_id].append(edge)

        return edge

    def get_node(self, node_id: str) -> ContextNode | None:
        """Get a node by ID."""
        node = self._nodes.get(node_id)
        if node:
            node.access_count += 1
            node.last_accessed = time.time()
        return node

    def get_nodes_by_type(self, node_type: NodeType) -> list[ContextNode]:
        """Get all nodes of a specific type."""
        return [self._nodes[nid] for nid in self._by_type[node_type] if nid in self._nodes]

    def get_related(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None = None,
        direction: str = "both",  # "outgoing", "incoming", "both"
    ) -> list[tuple[ContextNode, Edge]]:
        """Get nodes related to a given node.

        Args:
            node_id: Node to find relations for
            edge_types: Filter by edge types
            direction: Direction of edges to follow

        Returns:
            List of (node, edge) tuples
        """
        results = []

        if direction in ("outgoing", "both"):
            for edge in self._outgoing.get(node_id, []):
                if edge_types and edge.edge_type not in edge_types:
                    continue
                target = self._nodes.get(edge.target_id)
                if target:
                    results.append((target, edge))

        if direction in ("incoming", "both"):
            for edge in self._incoming.get(node_id, []):
                if edge_types and edge.edge_type not in edge_types:
                    continue
                source = self._nodes.get(edge.source_id)
                if source:
                    results.append((source, edge))

        return results

    def find_similar(
        self,
        query: str | list[float],
        top_k: int = 5,
        node_types: list[NodeType] | None = None,
    ) -> list[tuple[ContextNode, float]]:
        """Find nodes similar to query.

        Args:
            query: Query string or embedding
            top_k: Number of results
            node_types: Filter by node types

        Returns:
            List of (node, similarity) tuples
        """
        if isinstance(query, str) and self.embedding_fn:
            query_embedding = self.embedding_fn(query)
        elif isinstance(query, list):
            query_embedding = query
        else:
            return []

        results = []
        for node in self._nodes.values():
            if node_types and node.node_type not in node_types:
                continue
            if node.embedding:
                similarity = self._cosine_similarity(query_embedding, node.embedding)
                results.append((node, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    # ===== Learning Methods =====

    def record_task_start(
        self,
        task_id: str,
        task_content: str,
        approach: str,
        context: dict | None = None,
    ) -> ContextNode:
        """Record the start of a task for learning.

        Args:
            task_id: Unique task identifier
            task_content: Task description
            approach: Planned approach
            context: Additional context

        Returns:
            Task node
        """
        task_node = self.add_node(
            content=task_content,
            node_type=NodeType.TASK,
            metadata={"approach": approach, **(context or {})},
            node_id=task_id,
        )

        self._task_outcomes[task_id] = {
            "start_time": time.time(),
            "approach": approach,
            "steps": [],
        }

        return task_node

    def record_step(
        self,
        task_id: str,
        step_type: NodeType,
        content: str,
        metadata: dict | None = None,
    ) -> ContextNode | None:
        """Record a step in task execution.

        Args:
            task_id: Task being executed
            step_type: Type of step
            content: Step content
            metadata: Additional metadata

        Returns:
            Step node
        """
        if task_id not in self._task_outcomes:
            return None

        step_node = self.add_node(
            content=content,
            node_type=step_type,
            metadata=metadata,
        )

        # Link to task
        self.add_edge(task_id, step_node.id, EdgeType.CAUSED)

        # Link to previous step
        steps = self._task_outcomes[task_id]["steps"]
        if steps:
            self.add_edge(steps[-1], step_node.id, EdgeType.FOLLOWED_BY)

        steps.append(step_node.id)

        return step_node

    def record_task_outcome(
        self,
        task_id: str,
        success: bool,
        outcome_content: str,
        quality_score: float = 0.5,
        feedback: str | None = None,
    ) -> None:
        """Record task outcome for learning.

        Args:
            task_id: Task identifier
            success: Whether task succeeded
            outcome_content: Description of outcome
            quality_score: Quality of result (0-1)
            feedback: Optional feedback
        """
        if task_id not in self._task_outcomes:
            return

        task_data = self._task_outcomes[task_id]

        # Create outcome node
        outcome_node = self.add_node(
            content=outcome_content,
            node_type=NodeType.OUTCOME,
            metadata={
                "success": success,
                "quality_score": quality_score,
                "feedback": feedback,
            },
            importance=0.8 if success else 0.6,
        )

        self.add_edge(task_id, outcome_node.id, EdgeType.CAUSED)

        # Learn from outcome
        self._learn_from_outcome(
            task_id=task_id,
            approach=task_data["approach"],
            steps=task_data["steps"],
            success=success,
            quality_score=quality_score,
        )

        # Update node usefulness based on outcome
        for step_id in task_data["steps"]:
            if step_id in self._nodes:
                node = self._nodes[step_id]
                # Increase usefulness for successful tasks
                delta = 0.1 if success else -0.05
                node.usefulness = max(0, min(1, node.usefulness + delta))

    def _learn_from_outcome(
        self,
        task_id: str,
        approach: str,
        steps: list[str],
        success: bool,
        quality_score: float,
    ) -> None:
        """Extract learnings from task outcome."""
        # Create insight based on approach effectiveness
        insight_key = f"approach:{approach}"

        if insight_key not in self._insights:
            self._insights[insight_key] = LearningInsight(
                pattern=f"Approach '{approach}' for tasks",
                evidence_count=0,
                confidence=0.5,
                first_seen=time.time(),
                last_seen=time.time(),
            )

        insight = self._insights[insight_key]
        insight.evidence_count += 1
        insight.last_seen = time.time()
        insight.applications += 1

        # Update success rate with exponential moving average
        alpha = 0.1
        outcome_score = quality_score if success else 0.0
        insight.success_rate = (1 - alpha) * insight.success_rate + alpha * outcome_score

        # Update confidence based on evidence count
        insight.confidence = min(0.95, 0.5 + (insight.evidence_count * 0.05))

        # Store pattern for retrieval
        self._patterns[approach].append(task_id)

    def get_relevant_insights(
        self,
        task: str,
        top_k: int = 5,
    ) -> list[LearningInsight]:
        """Get insights relevant to a task.

        Args:
            task: Task description
            top_k: Number of insights to return

        Returns:
            Most relevant insights sorted by usefulness
        """
        insights = list(self._insights.values())

        # Score insights by relevance and proven success
        scored = []
        for insight in insights:
            score = (
                insight.success_rate
                * insight.confidence
                * (insight.applications / max(1, insight.evidence_count))
            )
            scored.append((insight, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [insight for insight, _ in scored[:top_k]]

    def get_successful_patterns(
        self,
        task_type: str | None = None,
        min_success_rate: float = 0.7,
    ) -> list[dict]:
        """Get patterns that have proven successful.

        Args:
            task_type: Filter by task type
            min_success_rate: Minimum success rate

        Returns:
            List of successful patterns with stats
        """
        results = []
        for key, insight in self._insights.items():
            if insight.success_rate >= min_success_rate and insight.evidence_count >= 3:
                results.append(
                    {
                        "pattern": insight.pattern,
                        "success_rate": insight.success_rate,
                        "confidence": insight.confidence,
                        "applications": insight.applications,
                    }
                )

        results.sort(key=lambda x: x["success_rate"] * x["confidence"], reverse=True)
        return results

    def _prune_least_important(self) -> None:
        """Remove least important nodes to stay under limit."""
        # Score nodes
        scored = []
        for node in self._nodes.values():
            # Combine factors: importance, recency, access count, usefulness
            recency = 1.0 / (1.0 + time.time() - node.last_accessed)
            score = (
                node.importance * 0.3
                + recency * 0.2
                + min(1.0, node.access_count / 10) * 0.2
                + node.usefulness * 0.3
            )
            scored.append((node.id, score))

        # Sort by score
        scored.sort(key=lambda x: x[1])

        # Remove bottom 10%
        to_remove = int(len(scored) * 0.1)
        for node_id, _ in scored[:to_remove]:
            self._remove_node(node_id)

    def _remove_node(self, node_id: str) -> None:
        """Remove a node and its edges."""
        if node_id in self._nodes:
            node = self._nodes.pop(node_id)
            self._by_type[node.node_type].discard(node_id)

            # Remove edges
            self._edges = [
                e for e in self._edges if e.source_id != node_id and e.target_id != node_id
            ]
            self._outgoing.pop(node_id, None)
            self._incoming.pop(node_id, None)

    def decay_importance(self) -> None:
        """Apply time-based decay to node importance."""
        now = time.time()
        if now - self._last_decay < self.decay_interval:
            return

        for node in self._nodes.values():
            age_hours = (now - node.timestamp) / 3600
            decay = node.decay_rate * age_hours
            node.importance = max(0.1, node.importance - decay)

        self._last_decay = now

    def get_context_summary(self, max_tokens: int = 2000) -> str:
        """Get a summary of relevant context for the current state.

        Args:
            max_tokens: Approximate token limit

        Returns:
            Context summary string
        """
        # Get most important recent nodes
        recent = sorted(
            self._nodes.values(),
            key=lambda n: n.importance * (1.0 / (1.0 + time.time() - n.timestamp)),
            reverse=True,
        )[:20]

        # Get successful patterns
        patterns = self.get_successful_patterns(min_success_rate=0.6)[:5]

        lines = ["# Relevant Context\n"]

        if patterns:
            lines.append("## Proven Patterns:")
            for p in patterns:
                lines.append(f"- {p['pattern']} (success: {p['success_rate']:.0%})")
            lines.append("")

        lines.append("## Recent Context:")
        for node in recent[:10]:
            lines.append(f"- [{node.node_type.value}] {node.content[:100]}...")

        return "\n".join(lines)

    def export(self) -> dict:
        """Export graph to serializable format."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type.value,
                    "weight": e.weight,
                }
                for e in self._edges
            ],
            "insights": [i.to_dict() for i in self._insights.values()],
        }

    def import_data(self, data: dict) -> None:
        """Import graph from serialized format."""
        for node_data in data.get("nodes", []):
            node = ContextNode.from_dict(node_data)
            self._nodes[node.id] = node
            self._by_type[node.node_type].add(node.id)

        for edge_data in data.get("edges", []):
            self.add_edge(
                edge_data["source"],
                edge_data["target"],
                EdgeType(edge_data["type"]),
                edge_data.get("weight", 1.0),
            )

    def get_statistics(self) -> dict:
        """Get statistics about the context graph.

        Returns:
            Dictionary with graph statistics
        """
        node_counts = {nt.value: len(ids) for nt, ids in self._by_type.items() if ids}

        insight_stats = {
            "total": len(self._insights),
            "high_confidence": len([i for i in self._insights.values() if i.confidence > 0.7]),
            "proven_patterns": len([i for i in self._insights.values() if i.success_rate > 0.7]),
        }

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "nodes_by_type": node_counts,
            "insights": insight_stats,
            "active_tasks": len(self._task_outcomes),
        }
