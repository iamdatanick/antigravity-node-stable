"""Kill switch for emergency agent termination."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class KillSwitchTriggered(Exception):
    """Raised when kill switch is activated."""

    def __init__(self, reason: str, agent_id: str | None = None):
        self.reason = reason
        self.agent_id = agent_id
        super().__init__(f"Kill switch triggered: {reason}")


class TriggerReason(Enum):
    """Reasons for kill switch activation."""

    MANUAL = "manual"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIME_EXCEEDED = "time_exceeded"
    ERROR_THRESHOLD = "error_threshold"
    SAFETY_VIOLATION = "safety_violation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_SIGNAL = "external_signal"


@dataclass
class KillSwitchConfig:
    """Kill switch configuration."""

    # Budget limits
    max_cost_usd: float | None = None
    max_tokens: int | None = None

    # Time limits
    max_runtime_seconds: float | None = None

    # Error limits
    max_consecutive_errors: int = 5
    max_total_errors: int = 20

    # Check interval
    check_interval_seconds: float = 1.0


@dataclass
class KillSwitchState:
    """Current state of the kill switch."""

    is_active: bool = False
    trigger_reason: TriggerReason | None = None
    trigger_message: str = ""
    triggered_at: float | None = None

    # Tracked metrics
    current_cost_usd: float = 0.0
    current_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    consecutive_errors: int = 0
    total_errors: int = 0

    @property
    def runtime_seconds(self) -> float:
        """Get current runtime in seconds."""
        return time.time() - self.start_time


class KillSwitch:
    """Emergency stop mechanism for agent workflows.

    Monitors agent activity and can terminate execution when:
    - Budget (cost/tokens) is exceeded
    - Runtime exceeds limit
    - Too many errors occur
    - Safety violations are detected
    - Manual trigger is activated
    """

    def __init__(
        self,
        config: KillSwitchConfig | None = None,
        on_trigger: Callable[[TriggerReason, str], None] | None = None,
    ):
        """Initialize kill switch.

        Args:
            config: Kill switch configuration.
            on_trigger: Callback when kill switch is triggered.
        """
        self.config = config or KillSwitchConfig()
        self.on_trigger = on_trigger

        self._state = KillSwitchState()
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._async_event: asyncio.Event | None = None

        # Per-agent states for multi-agent scenarios
        self._agent_states: dict[str, KillSwitchState] = {}

    @property
    def is_active(self) -> bool:
        """Check if kill switch is active."""
        with self._lock:
            return self._state.is_active

    @property
    def state(self) -> KillSwitchState:
        """Get current state (copy)."""
        with self._lock:
            return KillSwitchState(
                is_active=self._state.is_active,
                trigger_reason=self._state.trigger_reason,
                trigger_message=self._state.trigger_message,
                triggered_at=self._state.triggered_at,
                current_cost_usd=self._state.current_cost_usd,
                current_tokens=self._state.current_tokens,
                start_time=self._state.start_time,
                consecutive_errors=self._state.consecutive_errors,
                total_errors=self._state.total_errors,
            )

    def trigger(
        self,
        reason: TriggerReason,
        message: str,
        agent_id: str | None = None,
    ) -> None:
        """Trigger the kill switch.

        Args:
            reason: Reason for triggering.
            message: Human-readable message.
            agent_id: Optional agent ID for agent-specific trigger.
        """
        with self._lock:
            if agent_id:
                if agent_id not in self._agent_states:
                    self._agent_states[agent_id] = KillSwitchState()
                state = self._agent_states[agent_id]
            else:
                state = self._state

            state.is_active = True
            state.trigger_reason = reason
            state.trigger_message = message
            state.triggered_at = time.time()

        # Signal waiting threads
        self._event.set()
        if self._async_event:
            self._async_event.set()

        # Call callback
        if self.on_trigger:
            self.on_trigger(reason, message)

    def reset(self, agent_id: str | None = None) -> None:
        """Reset the kill switch.

        Args:
            agent_id: Optional agent ID to reset (None = global).
        """
        with self._lock:
            if agent_id:
                if agent_id in self._agent_states:
                    self._agent_states[agent_id] = KillSwitchState()
            else:
                self._state = KillSwitchState()
                self._event.clear()
                if self._async_event:
                    self._async_event.clear()

    def check(self, agent_id: str | None = None) -> None:
        """Check if kill switch is active and raise if so.

        Args:
            agent_id: Optional agent ID to check.

        Raises:
            KillSwitchTriggered: If kill switch is active.
        """
        with self._lock:
            # Check global state
            if self._state.is_active:
                raise KillSwitchTriggered(
                    self._state.trigger_message,
                    agent_id,
                )

            # Check agent-specific state
            if agent_id and agent_id in self._agent_states:
                state = self._agent_states[agent_id]
                if state.is_active:
                    raise KillSwitchTriggered(
                        state.trigger_message,
                        agent_id,
                    )

    def record_cost(self, cost_usd: float, agent_id: str | None = None) -> None:
        """Record cost and check budget limit.

        Args:
            cost_usd: Cost in USD to add.
            agent_id: Optional agent ID.
        """
        with self._lock:
            state = self._get_state(agent_id)
            state.current_cost_usd += cost_usd

            if self.config.max_cost_usd and state.current_cost_usd >= self.config.max_cost_usd:
                self._trigger_internal(
                    TriggerReason.BUDGET_EXCEEDED,
                    f"Cost limit exceeded: ${state.current_cost_usd:.4f} >= ${self.config.max_cost_usd:.4f}",
                    agent_id,
                )

    def record_tokens(self, tokens: int, agent_id: str | None = None) -> None:
        """Record token usage and check limit.

        Args:
            tokens: Tokens to add.
            agent_id: Optional agent ID.
        """
        with self._lock:
            state = self._get_state(agent_id)
            state.current_tokens += tokens

            if self.config.max_tokens and state.current_tokens >= self.config.max_tokens:
                self._trigger_internal(
                    TriggerReason.BUDGET_EXCEEDED,
                    f"Token limit exceeded: {state.current_tokens:,} >= {self.config.max_tokens:,}",
                    agent_id,
                )

    def record_error(self, agent_id: str | None = None) -> None:
        """Record an error and check thresholds.

        Args:
            agent_id: Optional agent ID.
        """
        with self._lock:
            state = self._get_state(agent_id)
            state.consecutive_errors += 1
            state.total_errors += 1

            if state.consecutive_errors >= self.config.max_consecutive_errors:
                self._trigger_internal(
                    TriggerReason.ERROR_THRESHOLD,
                    f"Consecutive error limit: {state.consecutive_errors} errors",
                    agent_id,
                )
            elif state.total_errors >= self.config.max_total_errors:
                self._trigger_internal(
                    TriggerReason.ERROR_THRESHOLD,
                    f"Total error limit: {state.total_errors} errors",
                    agent_id,
                )

    def record_success(self, agent_id: str | None = None) -> None:
        """Record a successful operation (resets consecutive error count).

        Args:
            agent_id: Optional agent ID.
        """
        with self._lock:
            state = self._get_state(agent_id)
            state.consecutive_errors = 0

    def check_runtime(self, agent_id: str | None = None) -> None:
        """Check if runtime limit is exceeded.

        Args:
            agent_id: Optional agent ID.
        """
        if not self.config.max_runtime_seconds:
            return

        with self._lock:
            state = self._get_state(agent_id)
            runtime = state.runtime_seconds

            if runtime >= self.config.max_runtime_seconds:
                self._trigger_internal(
                    TriggerReason.TIME_EXCEEDED,
                    f"Runtime limit exceeded: {runtime:.1f}s >= {self.config.max_runtime_seconds:.1f}s",
                    agent_id,
                )

    def _get_state(self, agent_id: str | None) -> KillSwitchState:
        """Get state for agent or global."""
        if agent_id:
            if agent_id not in self._agent_states:
                self._agent_states[agent_id] = KillSwitchState()
            return self._agent_states[agent_id]
        return self._state

    def _trigger_internal(
        self,
        reason: TriggerReason,
        message: str,
        agent_id: str | None,
    ) -> None:
        """Internal trigger (must hold lock)."""
        state = self._get_state(agent_id)

        if not state.is_active:
            state.is_active = True
            state.trigger_reason = reason
            state.trigger_message = message
            state.triggered_at = time.time()

            # Also set global state for non-agent triggers
            if not agent_id:
                self._event.set()
                if self._async_event:
                    self._async_event.set()

            # Call callback (release lock first)
            if self.on_trigger:
                # Schedule callback outside lock
                threading.Thread(
                    target=self.on_trigger,
                    args=(reason, message),
                    daemon=True,
                ).start()

    def wait_for_trigger(self, timeout: float | None = None) -> bool:
        """Wait for kill switch to be triggered.

        Args:
            timeout: Maximum time to wait (None = forever).

        Returns:
            True if triggered, False if timeout.
        """
        return self._event.wait(timeout)

    async def wait_for_trigger_async(self, timeout: float | None = None) -> bool:
        """Async wait for kill switch to be triggered.

        Args:
            timeout: Maximum time to wait (None = forever).

        Returns:
            True if triggered, False if timeout.
        """
        if self._async_event is None:
            self._async_event = asyncio.Event()

        try:
            await asyncio.wait_for(self._async_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def get_remaining_budget(self, agent_id: str | None = None) -> dict[str, float | int | None]:
        """Get remaining budget.

        Args:
            agent_id: Optional agent ID.

        Returns:
            Dict with remaining cost_usd, tokens, and runtime_seconds.
        """
        with self._lock:
            state = self._get_state(agent_id)

            remaining_cost = None
            if self.config.max_cost_usd:
                remaining_cost = max(0, self.config.max_cost_usd - state.current_cost_usd)

            remaining_tokens = None
            if self.config.max_tokens:
                remaining_tokens = max(0, self.config.max_tokens - state.current_tokens)

            remaining_time = None
            if self.config.max_runtime_seconds:
                remaining_time = max(0, self.config.max_runtime_seconds - state.runtime_seconds)

            return {
                "cost_usd": remaining_cost,
                "tokens": remaining_tokens,
                "runtime_seconds": remaining_time,
            }

    def get_usage_summary(self, agent_id: str | None = None) -> str:
        """Get human-readable usage summary.

        Args:
            agent_id: Optional agent ID.

        Returns:
            Summary string.
        """
        with self._lock:
            state = self._get_state(agent_id)

            lines = []

            if self.config.max_cost_usd:
                pct = (state.current_cost_usd / self.config.max_cost_usd) * 100
                lines.append(f"Cost: ${state.current_cost_usd:.4f} / ${self.config.max_cost_usd:.4f} ({pct:.1f}%)")
            else:
                lines.append(f"Cost: ${state.current_cost_usd:.4f}")

            if self.config.max_tokens:
                pct = (state.current_tokens / self.config.max_tokens) * 100
                lines.append(f"Tokens: {state.current_tokens:,} / {self.config.max_tokens:,} ({pct:.1f}%)")
            else:
                lines.append(f"Tokens: {state.current_tokens:,}")

            runtime = state.runtime_seconds
            if self.config.max_runtime_seconds:
                pct = (runtime / self.config.max_runtime_seconds) * 100
                lines.append(f"Runtime: {runtime:.1f}s / {self.config.max_runtime_seconds:.1f}s ({pct:.1f}%)")
            else:
                lines.append(f"Runtime: {runtime:.1f}s")

            lines.append(f"Errors: {state.total_errors} total, {state.consecutive_errors} consecutive")

            if state.is_active:
                lines.append(f"\nKILL SWITCH ACTIVE: {state.trigger_message}")

            return "\n".join(lines)
