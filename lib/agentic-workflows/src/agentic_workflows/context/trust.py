"""Trust scoring with decay based on provenance."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentic_workflows.context.graph import ContextGraph, ContextNode, NodeType
from agentic_workflows.context.provenance import Provenance, SourceType


class TrustPolicy(Enum):
    """Trust propagation policies."""

    MINIMUM = "minimum"  # Use minimum parent trust
    AVERAGE = "average"  # Use average parent trust
    MAXIMUM = "maximum"  # Use maximum parent trust
    WEIGHTED = "weighted"  # Use weighted average


@dataclass
class TrustDecayConfig:
    """Configuration for trust decay."""

    # Decay factor per hop (0.0 = no decay, 1.0 = full decay)
    hop_decay_factor: float = 0.1

    # Time-based decay half-life in seconds (None = no time decay)
    time_decay_half_life: float | None = 3600.0

    # Minimum trust score
    min_trust: float = 0.1

    # Maximum trust score
    max_trust: float = 1.0

    # Trust penalties by source type
    source_penalties: dict[SourceType, float] = field(default_factory=dict)

    # Trust bonuses for verification
    verification_bonus: float = 0.1


@dataclass
class TrustScore:
    """Detailed trust score breakdown."""

    final_score: float
    base_score: float
    hop_decay: float
    time_decay: float
    source_adjustment: float
    verification_bonus: float
    factors: list[str] = field(default_factory=list)

    @property
    def is_trusted(self) -> bool:
        """Check if score is above threshold."""
        return self.final_score >= 0.5


class TrustCalculator:
    """Calculate and propagate trust scores.

    Trust scoring considers:
    - Base trust from original source
    - Decay based on number of hops from origin
    - Time-based decay
    - Source type penalties
    - Verification bonuses
    """

    # Default source trust levels
    DEFAULT_SOURCE_TRUST: dict[SourceType, float] = {
        SourceType.USER: 0.9,
        SourceType.SYSTEM: 1.0,
        SourceType.AGENT: 0.8,
        SourceType.TOOL: 0.85,
        SourceType.EXTERNAL_API: 0.6,
        SourceType.DATABASE: 0.7,
        SourceType.FILE: 0.75,
        SourceType.DERIVED: 0.7,
    }

    def __init__(
        self,
        config: TrustDecayConfig | None = None,
        source_trust: dict[SourceType, float] | None = None,
        custom_scorer: Callable[[ContextNode], float] | None = None,
    ):
        """Initialize trust calculator.

        Args:
            config: Trust decay configuration.
            source_trust: Custom source trust levels.
            custom_scorer: Custom scoring function.
        """
        self.config = config or TrustDecayConfig()
        self.source_trust = {**self.DEFAULT_SOURCE_TRUST}
        if source_trust:
            self.source_trust.update(source_trust)
        self.custom_scorer = custom_scorer

    def calculate(
        self,
        node: ContextNode,
        provenance: Provenance | None = None,
        verified: bool = False,
    ) -> TrustScore:
        """Calculate trust score for a context node.

        Args:
            node: Context node to score.
            provenance: Optional provenance record.
            verified: Whether content has been verified.

        Returns:
            Detailed trust score.
        """
        factors: list[str] = []

        # Base score from node type
        base_score = self._get_base_score(node)
        factors.append(f"Base score: {base_score:.2f}")

        # Hop decay
        hop_count = provenance.hop_count if provenance else 0
        hop_decay = self._calculate_hop_decay(hop_count)
        if hop_count > 0:
            factors.append(f"Hop decay ({hop_count} hops): -{(1 - hop_decay):.2f}")

        # Time decay
        time_decay = self._calculate_time_decay(node.created_at)
        if time_decay < 1.0:
            factors.append(f"Time decay: -{(1 - time_decay):.2f}")

        # Source adjustment
        source_adjustment = self._get_source_adjustment(provenance)
        if source_adjustment != 0:
            factors.append(f"Source adjustment: {source_adjustment:+.2f}")

        # Verification bonus
        verification_bonus = 0.0
        if verified:
            verification_bonus = self.config.verification_bonus
            factors.append(f"Verification bonus: +{verification_bonus:.2f}")

        # Custom scorer
        custom_adjustment = 0.0
        if self.custom_scorer:
            try:
                custom_adjustment = self.custom_scorer(node) - base_score
                if custom_adjustment != 0:
                    factors.append(f"Custom adjustment: {custom_adjustment:+.2f}")
            except Exception:
                pass

        # Calculate final score
        score = base_score
        score *= hop_decay
        score *= time_decay
        score += source_adjustment
        score += verification_bonus
        score += custom_adjustment

        # Clamp to bounds
        final_score = max(self.config.min_trust, min(self.config.max_trust, score))

        return TrustScore(
            final_score=final_score,
            base_score=base_score,
            hop_decay=hop_decay,
            time_decay=time_decay,
            source_adjustment=source_adjustment,
            verification_bonus=verification_bonus,
            factors=factors,
        )

    def _get_base_score(self, node: ContextNode) -> float:
        """Get base score from node type."""
        type_scores = {
            NodeType.USER_INPUT: 0.9,
            NodeType.SYSTEM_PROMPT: 1.0,
            NodeType.AGENT_OUTPUT: 0.8,
            NodeType.TOOL_RESULT: 0.85,
            NodeType.EXTERNAL_DATA: 0.6,
            NodeType.DERIVED: 0.7,
            NodeType.CHECKPOINT: 0.9,
        }
        return type_scores.get(node.node_type, 0.7)

    def _calculate_hop_decay(self, hop_count: int) -> float:
        """Calculate decay factor based on hops."""
        if hop_count == 0:
            return 1.0

        decay = 1.0 - (self.config.hop_decay_factor * hop_count)
        return max(self.config.min_trust, decay)

    def _calculate_time_decay(self, created_at: float) -> float:
        """Calculate decay factor based on age."""
        if self.config.time_decay_half_life is None:
            return 1.0

        age_seconds = time.time() - created_at
        if age_seconds <= 0:
            return 1.0

        # Exponential decay
        decay = math.pow(0.5, age_seconds / self.config.time_decay_half_life)
        return max(self.config.min_trust, decay)

    def _get_source_adjustment(self, provenance: Provenance | None) -> float:
        """Get trust adjustment based on source type."""
        if provenance is None:
            return 0.0

        adjustment = 0.0
        for source in provenance.sources:
            if source.source_type in self.config.source_penalties:
                adjustment -= self.config.source_penalties[source.source_type]

        return adjustment

    def propagate_trust(
        self,
        graph: ContextGraph,
        policy: TrustPolicy = TrustPolicy.MINIMUM,
    ) -> dict[str, float]:
        """Propagate trust scores through graph.

        Args:
            graph: Context graph.
            policy: Trust propagation policy.

        Returns:
            Dict of node_id to updated trust score.
        """
        scores: dict[str, float] = {}

        # Process in topological order
        for node in graph.iterate(order="topological"):
            if not node.parents:
                # Root node - use node's trust score
                scores[node.id] = node.trust_score
            else:
                # Calculate from parents
                parent_scores = [
                    scores.get(pid, graph.get_node(pid).trust_score if graph.get_node(pid) else 0.5)
                    for pid in node.parents
                ]

                if policy == TrustPolicy.MINIMUM:
                    inherited = min(parent_scores)
                elif policy == TrustPolicy.MAXIMUM:
                    inherited = max(parent_scores)
                elif policy == TrustPolicy.AVERAGE:
                    inherited = sum(parent_scores) / len(parent_scores)
                else:  # WEIGHTED
                    # Weight by parent creation time (newer = higher weight)
                    weights = []
                    for pid in node.parents:
                        parent = graph.get_node(pid)
                        if parent:
                            weights.append(parent.created_at)
                        else:
                            weights.append(0)
                    total_weight = sum(weights)
                    if total_weight > 0:
                        inherited = (
                            sum(s * w for s, w in zip(parent_scores, weights)) / total_weight
                        )
                    else:
                        inherited = sum(parent_scores) / len(parent_scores)

                # Apply hop decay
                inherited *= 1 - self.config.hop_decay_factor
                scores[node.id] = max(self.config.min_trust, inherited)

        return scores

    def update_graph_trust(
        self,
        graph: ContextGraph,
        policy: TrustPolicy = TrustPolicy.MINIMUM,
    ) -> None:
        """Update trust scores in graph.

        Args:
            graph: Context graph to update.
            policy: Trust propagation policy.
        """
        scores = self.propagate_trust(graph, policy)
        for node_id, score in scores.items():
            node = graph.get_node(node_id)
            if node:
                node.trust_score = score

    def explain_score(self, score: TrustScore) -> str:
        """Generate human-readable explanation of trust score.

        Args:
            score: Trust score to explain.

        Returns:
            Explanation string.
        """
        lines = [
            f"Trust Score: {score.final_score:.2f} ({'Trusted' if score.is_trusted else 'Untrusted'})",
            "",
            "Calculation:",
        ]
        lines.extend(f"  - {factor}" for factor in score.factors)

        return "\n".join(lines)

    def get_trust_summary(self, nodes: list[ContextNode]) -> dict[str, Any]:
        """Get trust summary for multiple nodes.

        Args:
            nodes: Nodes to summarize.

        Returns:
            Summary statistics.
        """
        if not nodes:
            return {"count": 0}

        scores = [n.trust_score for n in nodes]
        return {
            "count": len(nodes),
            "min": min(scores),
            "max": max(scores),
            "average": sum(scores) / len(scores),
            "trusted_count": sum(1 for s in scores if s >= 0.5),
            "untrusted_count": sum(1 for s in scores if s < 0.5),
        }
