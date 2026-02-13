"""Metrics collection for token counting and cost tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Model(Enum):
    """Supported models with pricing."""

    # Anthropic models (per million tokens)
    OPUS = "claude-opus-4"
    SONNET = "claude-sonnet-4"
    HAIKU = "claude-haiku-3.5"

    # Pricing: (input_per_million, output_per_million)
    @property
    def pricing(self) -> tuple[float, float]:
        """Get pricing per million tokens (input, output)."""
        prices = {
            Model.OPUS: (15.0, 75.0),
            Model.SONNET: (3.0, 15.0),
            Model.HAIKU: (0.25, 1.25),
        }
        return prices.get(self, (0.0, 0.0))


@dataclass
class TokenUsage:
    """Token usage for a single call."""

    input_tokens: int
    output_tokens: int
    model: Model
    timestamp: float = field(default_factory=time.time)
    agent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Get total tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Calculate cost in USD."""
        input_price, output_price = self.model.pricing
        input_cost = (self.input_tokens / 1_000_000) * input_price
        output_cost = (self.output_tokens / 1_000_000) * output_price
        return input_cost + output_cost


@dataclass
class CostSummary:
    """Summary of costs."""

    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    calls_count: int
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_agent: dict[str, dict[str, Any]] = field(default_factory=dict)
    time_range_seconds: float = 0.0


class MetricsCollector:
    """Collects and aggregates metrics for agent workflows.

    Features:
    - Token counting per model
    - Cost calculation
    - Per-agent tracking
    - Time-series data
    - Budget enforcement
    """

    def __init__(
        self,
        budget_usd: float | None = None,
        budget_tokens: int | None = None,
        on_budget_warning: Callable[[str, float], None] | None = None,
        warning_threshold: float = 0.8,
    ):
        """Initialize metrics collector.

        Args:
            budget_usd: Maximum budget in USD.
            budget_tokens: Maximum token budget.
            on_budget_warning: Callback when approaching budget.
            warning_threshold: Threshold for warning (0.0-1.0).
        """
        self.budget_usd = budget_usd
        self.budget_tokens = budget_tokens
        self.on_budget_warning = on_budget_warning
        self.warning_threshold = warning_threshold

        self._usage: list[TokenUsage] = []
        self._lock = threading.Lock()
        self._warning_sent = False

    def record(
        self,
        agent_id: str,
        model: Model,
        input_tokens: int,
        output_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> TokenUsage:
        """Record token usage.

        Args:
            agent_id: Agent identifier.
            model: Model used.
            input_tokens: Input token count.
            output_tokens: Output token count.
            metadata: Additional metadata.

        Returns:
            Created usage record.
        """
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            agent_id=agent_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._usage.append(usage)

        # Check budget
        self._check_budget()

        return usage

    def _check_budget(self) -> None:
        """Check budget and send warning if needed."""
        if self._warning_sent:
            return

        if self.budget_usd:
            current_cost = self.get_total_cost()
            if current_cost >= self.budget_usd * self.warning_threshold:
                self._warning_sent = True
                if self.on_budget_warning:
                    self.on_budget_warning(
                        f"Cost budget warning: ${current_cost:.4f} / ${self.budget_usd:.4f}",
                        current_cost / self.budget_usd,
                    )

        if self.budget_tokens:
            total_tokens = self.get_total_tokens()
            if total_tokens >= self.budget_tokens * self.warning_threshold:
                self._warning_sent = True
                if self.on_budget_warning:
                    self.on_budget_warning(
                        f"Token budget warning: {total_tokens:,} / {self.budget_tokens:,}",
                        total_tokens / self.budget_tokens,
                    )

    def get_total_cost(self) -> float:
        """Get total cost in USD."""
        with self._lock:
            return sum(u.cost_usd for u in self._usage)

    def get_total_tokens(self) -> int:
        """Get total token count."""
        with self._lock:
            return sum(u.total_tokens for u in self._usage)

    def get_summary(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> CostSummary:
        """Get usage summary.

        Args:
            start_time: Filter start timestamp.
            end_time: Filter end timestamp.

        Returns:
            Cost summary.
        """
        with self._lock:
            filtered = self._usage

            if start_time:
                filtered = [u for u in filtered if u.timestamp >= start_time]
            if end_time:
                filtered = [u for u in filtered if u.timestamp <= end_time]

            if not filtered:
                return CostSummary(
                    total_cost_usd=0.0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    calls_count=0,
                )

            # Aggregate by model
            by_model: dict[str, dict[str, Any]] = {}
            for u in filtered:
                model_name = u.model.value
                if model_name not in by_model:
                    by_model[model_name] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "calls": 0,
                    }
                by_model[model_name]["input_tokens"] += u.input_tokens
                by_model[model_name]["output_tokens"] += u.output_tokens
                by_model[model_name]["cost_usd"] += u.cost_usd
                by_model[model_name]["calls"] += 1

            # Aggregate by agent
            by_agent: dict[str, dict[str, Any]] = {}
            for u in filtered:
                if u.agent_id:
                    if u.agent_id not in by_agent:
                        by_agent[u.agent_id] = {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cost_usd": 0.0,
                            "calls": 0,
                        }
                    by_agent[u.agent_id]["input_tokens"] += u.input_tokens
                    by_agent[u.agent_id]["output_tokens"] += u.output_tokens
                    by_agent[u.agent_id]["cost_usd"] += u.cost_usd
                    by_agent[u.agent_id]["calls"] += 1

            # Time range
            time_range = 0.0
            if filtered:
                time_range = filtered[-1].timestamp - filtered[0].timestamp

            return CostSummary(
                total_cost_usd=sum(u.cost_usd for u in filtered),
                total_input_tokens=sum(u.input_tokens for u in filtered),
                total_output_tokens=sum(u.output_tokens for u in filtered),
                calls_count=len(filtered),
                by_model=by_model,
                by_agent=by_agent,
                time_range_seconds=time_range,
            )

    def get_agent_usage(self, agent_id: str) -> dict[str, Any]:
        """Get usage for a specific agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            Agent usage statistics.
        """
        with self._lock:
            agent_usage = [u for u in self._usage if u.agent_id == agent_id]

            if not agent_usage:
                return {
                    "agent_id": agent_id,
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }

            return {
                "agent_id": agent_id,
                "calls": len(agent_usage),
                "input_tokens": sum(u.input_tokens for u in agent_usage),
                "output_tokens": sum(u.output_tokens for u in agent_usage),
                "cost_usd": sum(u.cost_usd for u in agent_usage),
            }

    def get_remaining_budget(self) -> dict[str, Any]:
        """Get remaining budget information."""
        current_cost = self.get_total_cost()
        current_tokens = self.get_total_tokens()

        return {
            "cost_usd": {
                "used": current_cost,
                "budget": self.budget_usd,
                "remaining": (self.budget_usd - current_cost) if self.budget_usd else None,
                "percent_used": (current_cost / self.budget_usd * 100) if self.budget_usd else None,
            },
            "tokens": {
                "used": current_tokens,
                "budget": self.budget_tokens,
                "remaining": (self.budget_tokens - current_tokens) if self.budget_tokens else None,
                "percent_used": (current_tokens / self.budget_tokens * 100) if self.budget_tokens else None,
            },
        }

    def is_budget_exceeded(self) -> tuple[bool, str]:
        """Check if budget is exceeded.

        Returns:
            Tuple of (exceeded, reason).
        """
        if self.budget_usd:
            current_cost = self.get_total_cost()
            if current_cost >= self.budget_usd:
                return True, f"Cost budget exceeded: ${current_cost:.4f} >= ${self.budget_usd:.4f}"

        if self.budget_tokens:
            total_tokens = self.get_total_tokens()
            if total_tokens >= self.budget_tokens:
                return True, f"Token budget exceeded: {total_tokens:,} >= {self.budget_tokens:,}"

        return False, ""

    def format_summary(self) -> str:
        """Get formatted summary string."""
        summary = self.get_summary()

        lines = [
            "=== Usage Summary ===",
            f"Total Cost: ${summary.total_cost_usd:.4f}",
            f"Total Tokens: {summary.total_input_tokens + summary.total_output_tokens:,}",
            f"  Input: {summary.total_input_tokens:,}",
            f"  Output: {summary.total_output_tokens:,}",
            f"API Calls: {summary.calls_count}",
        ]

        if summary.by_model:
            lines.append("")
            lines.append("By Model:")
            for model, stats in summary.by_model.items():
                lines.append(f"  {model}:")
                lines.append(f"    Calls: {stats['calls']}")
                lines.append(f"    Cost: ${stats['cost_usd']:.4f}")

        if summary.by_agent:
            lines.append("")
            lines.append("By Agent:")
            for agent, stats in summary.by_agent.items():
                lines.append(f"  {agent}:")
                lines.append(f"    Calls: {stats['calls']}")
                lines.append(f"    Cost: ${stats['cost_usd']:.4f}")

        budget = self.get_remaining_budget()
        if budget["cost_usd"]["budget"]:
            lines.append("")
            lines.append(f"Budget: ${budget['cost_usd']['used']:.4f} / ${budget['cost_usd']['budget']:.4f} ({budget['cost_usd']['percent_used']:.1f}%)")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._usage.clear()
            self._warning_sent = False

    def export(self) -> list[dict[str, Any]]:
        """Export all usage records."""
        with self._lock:
            return [
                {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "model": u.model.value,
                    "timestamp": u.timestamp,
                    "agent_id": u.agent_id,
                    "cost_usd": u.cost_usd,
                    "metadata": u.metadata,
                }
                for u in self._usage
            ]
