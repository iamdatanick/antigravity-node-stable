"""Extended trust scoring for multi-agent interactions.

Builds on the base trust module to provide:
- Agent-to-agent trust scoring
- Reputation tracking over time
- Behavioral analysis
- Trust delegation and transitive trust
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .trust import TrustCalculator, TrustScore


class InteractionType(Enum):
    """Types of agent interactions."""

    TASK_DELEGATION = "task_delegation"
    INFORMATION_SHARING = "information_sharing"
    TOOL_INVOCATION = "tool_invocation"
    VERIFICATION = "verification"
    ERROR_REPORT = "error_report"
    FEEDBACK = "feedback"


class InteractionOutcome(Enum):
    """Outcomes of agent interactions."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    MALICIOUS = "malicious"


@dataclass
class InteractionRecord:
    """Record of an agent interaction."""

    agent_id: str
    peer_agent_id: str
    interaction_type: InteractionType
    outcome: InteractionOutcome
    timestamp: float = field(default_factory=time.time)
    duration: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    # Trust-relevant metrics
    data_quality: float = 0.8  # Quality of data/results (0-1)
    response_time: float = 1.0  # Normalized response time (1 = expected)
    policy_compliance: float = 1.0  # Compliance with policies (0-1)


@dataclass
class AgentReputation:
    """Reputation profile for an agent."""

    agent_id: str
    created_at: float = field(default_factory=time.time)

    # Core metrics
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    malicious_interactions: int = 0

    # Running averages
    avg_data_quality: float = 0.8
    avg_response_time: float = 1.0
    avg_policy_compliance: float = 1.0

    # Trust scores by interaction type
    type_scores: dict[InteractionType, float] = field(default_factory=dict)

    # Recent interaction history (for decay calculations)
    recent_interactions: list[InteractionRecord] = field(default_factory=list)

    # Maximum history to keep
    max_history: int = 100

    def add_interaction(self, record: InteractionRecord) -> None:
        """Add an interaction record to reputation."""
        self.total_interactions += 1

        if record.outcome == InteractionOutcome.SUCCESS:
            self.successful_interactions += 1
        elif record.outcome in (InteractionOutcome.FAILURE, InteractionOutcome.TIMEOUT):
            self.failed_interactions += 1
        elif record.outcome == InteractionOutcome.MALICIOUS:
            self.malicious_interactions += 1

        # Update running averages
        n = self.total_interactions
        self.avg_data_quality = ((self.avg_data_quality * (n - 1)) + record.data_quality) / n
        self.avg_response_time = ((self.avg_response_time * (n - 1)) + record.response_time) / n
        self.avg_policy_compliance = (
            (self.avg_policy_compliance * (n - 1)) + record.policy_compliance
        ) / n

        # Update type-specific scores
        if record.interaction_type not in self.type_scores:
            self.type_scores[record.interaction_type] = 0.5

        # Adjust type score based on outcome
        current = self.type_scores[record.interaction_type]
        if record.outcome == InteractionOutcome.SUCCESS:
            self.type_scores[record.interaction_type] = min(1.0, current + 0.05)
        elif record.outcome == InteractionOutcome.MALICIOUS:
            self.type_scores[record.interaction_type] = max(0.0, current - 0.3)
        elif record.outcome in (InteractionOutcome.FAILURE, InteractionOutcome.TIMEOUT):
            self.type_scores[record.interaction_type] = max(0.0, current - 0.1)

        # Maintain recent history
        self.recent_interactions.append(record)
        if len(self.recent_interactions) > self.max_history:
            self.recent_interactions = self.recent_interactions[-self.max_history :]

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_interactions == 0:
            return 0.5
        return self.successful_interactions / self.total_interactions

    @property
    def malicious_rate(self) -> float:
        """Calculate malicious interaction rate."""
        if self.total_interactions == 0:
            return 0.0
        return self.malicious_interactions / self.total_interactions


@dataclass
class AgentTrustConfig:
    """Configuration for agent trust calculations."""

    # Weight factors for trust calculation
    success_rate_weight: float = 0.3
    data_quality_weight: float = 0.2
    response_time_weight: float = 0.1
    policy_compliance_weight: float = 0.2
    history_weight: float = 0.2

    # Decay settings
    time_decay_days: float = 30.0  # Half-life in days
    interaction_decay_factor: float = 0.95  # Per-interaction decay

    # Thresholds
    min_interactions_for_trust: int = 5
    malicious_threshold: float = 0.1  # Block if malicious rate exceeds this
    trust_threshold: float = 0.5

    # Transitive trust settings
    max_trust_hops: int = 3
    transitive_decay: float = 0.7  # Trust decay per hop


class AgentTrustCalculator:
    """Calculate trust scores for agent interactions.

    Extends the base trust calculator with:
    - Agent reputation tracking
    - Interaction history analysis
    - Transitive trust computation
    - Trust delegation management
    """

    def __init__(
        self,
        config: AgentTrustConfig | None = None,
        base_calculator: TrustCalculator | None = None,
    ):
        """Initialize agent trust calculator.

        Args:
            config: Agent trust configuration.
            base_calculator: Base trust calculator for node-level trust.
        """
        self.config = config or AgentTrustConfig()
        self.base_calculator = base_calculator or TrustCalculator()

        # Agent reputations
        self._reputations: dict[str, AgentReputation] = {}

        # Trust delegations: delegator -> set of delegates
        self._delegations: dict[str, set[str]] = {}

        # Trust graph for transitive calculations
        self._trust_graph: dict[str, dict[str, float]] = {}

    def get_or_create_reputation(self, agent_id: str) -> AgentReputation:
        """Get or create reputation for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            Agent reputation record.
        """
        if agent_id not in self._reputations:
            self._reputations[agent_id] = AgentReputation(agent_id=agent_id)
        return self._reputations[agent_id]

    def record_interaction(
        self,
        from_agent: str,
        to_agent: str,
        interaction_type: InteractionType,
        outcome: InteractionOutcome,
        **kwargs,
    ) -> InteractionRecord:
        """Record an interaction between agents.

        Args:
            from_agent: Initiating agent.
            to_agent: Receiving agent.
            interaction_type: Type of interaction.
            outcome: Interaction outcome.
            **kwargs: Additional interaction details.

        Returns:
            The recorded interaction.
        """
        record = InteractionRecord(
            agent_id=from_agent,
            peer_agent_id=to_agent,
            interaction_type=interaction_type,
            outcome=outcome,
            **kwargs,
        )

        # Update target agent's reputation
        target_rep = self.get_or_create_reputation(to_agent)
        target_rep.add_interaction(record)

        # Update trust graph
        if from_agent not in self._trust_graph:
            self._trust_graph[from_agent] = {}

        current_trust = self._trust_graph[from_agent].get(to_agent, 0.5)
        new_trust = self._calculate_interaction_trust_delta(record, current_trust)
        self._trust_graph[from_agent][to_agent] = new_trust

        return record

    def _calculate_interaction_trust_delta(
        self,
        record: InteractionRecord,
        current_trust: float,
    ) -> float:
        """Calculate trust change from an interaction.

        Args:
            record: Interaction record.
            current_trust: Current trust score.

        Returns:
            Updated trust score.
        """
        # Base delta based on outcome
        outcome_deltas = {
            InteractionOutcome.SUCCESS: 0.05,
            InteractionOutcome.PARTIAL_SUCCESS: 0.02,
            InteractionOutcome.FAILURE: -0.1,
            InteractionOutcome.TIMEOUT: -0.05,
            InteractionOutcome.REJECTED: -0.02,
            InteractionOutcome.MALICIOUS: -0.4,
        }

        delta = outcome_deltas.get(record.outcome, 0)

        # Adjust by quality metrics
        delta *= 0.5 + 0.5 * record.data_quality
        delta *= record.policy_compliance

        # Apply change
        new_trust = current_trust + delta
        return max(0.0, min(1.0, new_trust))

    def calculate_agent_trust(
        self,
        agent_id: str,
        from_perspective: str | None = None,
    ) -> TrustScore:
        """Calculate trust score for an agent.

        Args:
            agent_id: Agent to evaluate.
            from_perspective: Optional evaluating agent's perspective.

        Returns:
            Detailed trust score.
        """
        rep = self.get_or_create_reputation(agent_id)
        factors: list[str] = []

        # Check for blocking conditions
        if rep.malicious_rate > self.config.malicious_threshold:
            return TrustScore(
                final_score=0.0,
                base_score=0.0,
                hop_decay=1.0,
                time_decay=1.0,
                source_adjustment=0.0,
                verification_bonus=0.0,
                factors=["BLOCKED: Malicious activity rate exceeded threshold"],
            )

        # Calculate component scores
        scores: dict[str, float] = {}

        # Success rate component
        if rep.total_interactions >= self.config.min_interactions_for_trust:
            scores["success_rate"] = rep.success_rate
            factors.append(f"Success rate: {rep.success_rate:.2f}")
        else:
            scores["success_rate"] = 0.5  # Neutral for new agents
            factors.append(
                f"Insufficient history ({rep.total_interactions}/{self.config.min_interactions_for_trust})"
            )

        # Data quality component
        scores["data_quality"] = rep.avg_data_quality
        factors.append(f"Avg data quality: {rep.avg_data_quality:.2f}")

        # Response time component (normalize: faster is better)
        response_score = 1.0 / max(0.1, rep.avg_response_time)
        response_score = min(1.0, response_score)
        scores["response_time"] = response_score
        factors.append(f"Response time score: {response_score:.2f}")

        # Policy compliance component
        scores["policy_compliance"] = rep.avg_policy_compliance
        factors.append(f"Policy compliance: {rep.avg_policy_compliance:.2f}")

        # History score (recency-weighted)
        history_score = self._calculate_history_score(rep)
        scores["history"] = history_score
        factors.append(f"History score: {history_score:.2f}")

        # Weighted combination
        base_score = (
            scores["success_rate"] * self.config.success_rate_weight
            + scores["data_quality"] * self.config.data_quality_weight
            + scores["response_time"] * self.config.response_time_weight
            + scores["policy_compliance"] * self.config.policy_compliance_weight
            + scores["history"] * self.config.history_weight
        )

        # Apply time decay
        time_decay = self._calculate_time_decay(rep)
        factors.append(f"Time decay: {time_decay:.2f}")

        # Consider perspective-specific trust
        perspective_adjustment = 0.0
        if from_perspective and from_perspective in self._trust_graph:
            direct_trust = self._trust_graph[from_perspective].get(agent_id)
            if direct_trust is not None:
                # Blend direct experience with reputation
                perspective_adjustment = (direct_trust - base_score) * 0.3
                factors.append(f"Direct experience adjustment: {perspective_adjustment:+.2f}")

        final_score = base_score * time_decay + perspective_adjustment
        final_score = max(0.0, min(1.0, final_score))

        return TrustScore(
            final_score=final_score,
            base_score=base_score,
            hop_decay=1.0,  # N/A for agent trust
            time_decay=time_decay,
            source_adjustment=perspective_adjustment,
            verification_bonus=0.0,
            factors=factors,
        )

    def _calculate_history_score(self, rep: AgentReputation) -> float:
        """Calculate score based on recent interaction history.

        Args:
            rep: Agent reputation.

        Returns:
            History-based score.
        """
        if not rep.recent_interactions:
            return 0.5

        # Weight recent interactions more heavily
        weights = []
        scores = []

        for i, record in enumerate(rep.recent_interactions):
            # Exponential decay for older interactions
            age_factor = self.config.interaction_decay_factor ** (
                len(rep.recent_interactions) - i - 1
            )

            outcome_scores = {
                InteractionOutcome.SUCCESS: 1.0,
                InteractionOutcome.PARTIAL_SUCCESS: 0.7,
                InteractionOutcome.FAILURE: 0.2,
                InteractionOutcome.TIMEOUT: 0.3,
                InteractionOutcome.REJECTED: 0.4,
                InteractionOutcome.MALICIOUS: 0.0,
            }

            weights.append(age_factor)
            scores.append(outcome_scores.get(record.outcome, 0.5))

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5

        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    def _calculate_time_decay(self, rep: AgentReputation) -> float:
        """Calculate time-based trust decay.

        Args:
            rep: Agent reputation.

        Returns:
            Time decay factor.
        """
        if not rep.recent_interactions:
            return 0.8  # Penalty for no recent activity

        # Find most recent interaction
        most_recent = max(r.timestamp for r in rep.recent_interactions)
        age_days = (time.time() - most_recent) / 86400

        if age_days <= 0:
            return 1.0

        # Exponential decay
        decay = math.pow(0.5, age_days / self.config.time_decay_days)
        return max(0.3, decay)  # Floor at 0.3

    def delegate_trust(
        self,
        delegator: str,
        delegate: str,
        scope: list[InteractionType] | None = None,
    ) -> None:
        """Allow one agent to delegate trust to another.

        Args:
            delegator: Agent granting delegation.
            delegate: Agent receiving delegation.
            scope: Optional interaction types for delegation.
        """
        if delegator not in self._delegations:
            self._delegations[delegator] = set()
        self._delegations[delegator].add(delegate)

    def revoke_delegation(self, delegator: str, delegate: str) -> None:
        """Revoke a trust delegation.

        Args:
            delegator: Agent revoking delegation.
            delegate: Agent losing delegation.
        """
        if delegator in self._delegations:
            self._delegations[delegator].discard(delegate)

    def calculate_transitive_trust(
        self,
        from_agent: str,
        to_agent: str,
    ) -> float:
        """Calculate transitive trust between agents.

        Uses shortest path with trust decay per hop.

        Args:
            from_agent: Source agent.
            to_agent: Target agent.

        Returns:
            Transitive trust score.
        """
        if from_agent == to_agent:
            return 1.0

        # BFS with trust decay
        visited = {from_agent}
        queue = [(from_agent, 1.0, 0)]  # (agent, trust, hops)
        max_trust = 0.0

        while queue:
            current, current_trust, hops = queue.pop(0)

            if hops >= self.config.max_trust_hops:
                continue

            # Check direct connections
            if current in self._trust_graph:
                for neighbor, direct_trust in self._trust_graph[current].items():
                    if neighbor == to_agent:
                        path_trust = current_trust * direct_trust * self.config.transitive_decay
                        max_trust = max(max_trust, path_trust)
                    elif neighbor not in visited:
                        visited.add(neighbor)
                        neighbor_trust = current_trust * direct_trust * self.config.transitive_decay
                        queue.append((neighbor, neighbor_trust, hops + 1))

            # Check delegations
            if current in self._delegations:
                for delegate in self._delegations[current]:
                    if delegate == to_agent:
                        # Delegation grants trust
                        path_trust = current_trust * 0.8 * self.config.transitive_decay
                        max_trust = max(max_trust, path_trust)
                    elif delegate not in visited:
                        visited.add(delegate)
                        queue.append((delegate, current_trust * 0.8, hops + 1))

        return max_trust

    def should_trust(
        self,
        agent_id: str,
        from_perspective: str | None = None,
        interaction_type: InteractionType | None = None,
    ) -> tuple[bool, str]:
        """Determine if an agent should be trusted.

        Args:
            agent_id: Agent to evaluate.
            from_perspective: Evaluating agent.
            interaction_type: Optional specific interaction type.

        Returns:
            Tuple of (should_trust, reason).
        """
        rep = self.get_or_create_reputation(agent_id)

        # Check for blocking conditions
        if rep.malicious_rate > self.config.malicious_threshold:
            return False, f"Malicious activity rate ({rep.malicious_rate:.1%}) exceeds threshold"

        # Calculate trust score
        score = self.calculate_agent_trust(agent_id, from_perspective)

        # Check interaction-type specific trust
        if interaction_type and interaction_type in rep.type_scores:
            type_trust = rep.type_scores[interaction_type]
            if type_trust < self.config.trust_threshold:
                return (
                    False,
                    f"Low trust for {interaction_type.value} interactions ({type_trust:.2f})",
                )

        if score.final_score >= self.config.trust_threshold:
            return True, f"Trust score {score.final_score:.2f} meets threshold"
        else:
            return False, f"Trust score {score.final_score:.2f} below threshold"

    def get_trust_report(self, agent_id: str) -> dict[str, Any]:
        """Generate a detailed trust report for an agent.

        Args:
            agent_id: Agent to report on.

        Returns:
            Detailed trust report.
        """
        rep = self.get_or_create_reputation(agent_id)
        score = self.calculate_agent_trust(agent_id)

        return {
            "agent_id": agent_id,
            "trust_score": score.final_score,
            "is_trusted": score.is_trusted,
            "factors": score.factors,
            "reputation": {
                "total_interactions": rep.total_interactions,
                "success_rate": rep.success_rate,
                "malicious_rate": rep.malicious_rate,
                "avg_data_quality": rep.avg_data_quality,
                "avg_response_time": rep.avg_response_time,
                "avg_policy_compliance": rep.avg_policy_compliance,
            },
            "type_scores": {t.value: s for t, s in rep.type_scores.items()},
            "delegations": list(self._delegations.get(agent_id, set())),
            "recent_interaction_count": len(rep.recent_interactions),
        }
