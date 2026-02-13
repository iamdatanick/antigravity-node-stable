"""Recovery orchestration for self-healing workflows."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class RecoveryStrategy(Enum):
    """Recovery strategies."""

    RETRY = "retry"                # Simple retry
    FALLBACK = "fallback"          # Use fallback function
    ROLLBACK = "rollback"          # Revert to previous state
    SKIP = "skip"                  # Skip and continue
    ESCALATE = "escalate"          # Escalate to human
    COMPENSATE = "compensate"      # Run compensation logic
    RESTART = "restart"            # Restart from beginning


class RecoveryStatus(Enum):
    """Recovery attempt status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RECOVERED = "recovered"
    FAILED = "failed"
    ESCALATED = "escalated"


@dataclass
class RecoveryConfig:
    """Recovery configuration."""

    # Strategies to try in order
    strategies: list[RecoveryStrategy] = field(default_factory=lambda: [
        RecoveryStrategy.RETRY,
        RecoveryStrategy.FALLBACK,
        RecoveryStrategy.ESCALATE,
    ])

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff: float = 2.0

    # Fallback chain
    fallbacks: list[Callable[[], Any]] = field(default_factory=list)

    # Escalation settings
    escalation_handler: Callable[[Exception, dict], Any] | None = None

    # Compensation logic
    compensator: Callable[[dict], None] | None = None

    # Timeout
    timeout_seconds: float | None = 60.0


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""

    status: RecoveryStatus
    strategy_used: RecoveryStrategy | None = None
    attempts: int = 0
    result: Any = None
    error: str | None = None
    duration_seconds: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAttempt:
    """A single recovery attempt."""

    strategy: RecoveryStrategy
    attempt_number: int
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    success: bool = False
    error: str | None = None


class RecoveryOrchestrator:
    """Orchestrates recovery from failures.

    Features:
    - Multiple recovery strategies
    - Automatic strategy selection
    - Rollback support
    - Human escalation
    - Recovery history
    """

    def __init__(
        self,
        config: RecoveryConfig | None = None,
        on_recovery_start: Callable[[str, RecoveryStrategy], None] | None = None,
        on_recovery_complete: Callable[[RecoveryResult], None] | None = None,
    ):
        """Initialize recovery orchestrator.

        Args:
            config: Recovery configuration.
            on_recovery_start: Callback when recovery starts.
            on_recovery_complete: Callback when recovery completes.
        """
        self.config = config or RecoveryConfig()
        self.on_recovery_start = on_recovery_start
        self.on_recovery_complete = on_recovery_complete

        self._history: list[RecoveryResult] = []
        self._state_snapshots: dict[str, Any] = {}
        self._lock = threading.Lock()

    def save_state(self, key: str, state: Any) -> None:
        """Save state snapshot for potential rollback.

        Args:
            key: State identifier.
            state: State to save.
        """
        with self._lock:
            self._state_snapshots[key] = {
                "state": state,
                "timestamp": time.time(),
            }

    def get_state(self, key: str) -> Any | None:
        """Get saved state.

        Args:
            key: State identifier.

        Returns:
            Saved state or None.
        """
        with self._lock:
            snapshot = self._state_snapshots.get(key)
            return snapshot["state"] if snapshot else None

    def recover(
        self,
        operation: Callable[[], T],
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> RecoveryResult:
        """Attempt recovery from an error.

        Args:
            operation: Original operation to retry.
            error: The error that occurred.
            context: Additional context.

        Returns:
            Recovery result.
        """
        start_time = time.time()
        context = context or {}
        context["original_error"] = str(error)
        context["error_type"] = type(error).__name__

        result = RecoveryResult(status=RecoveryStatus.PENDING, context=context)
        attempts: list[RecoveryAttempt] = []

        for strategy in self.config.strategies:
            if self.on_recovery_start:
                self.on_recovery_start(context.get("operation_id", "unknown"), strategy)

            attempt = RecoveryAttempt(
                strategy=strategy,
                attempt_number=len(attempts) + 1,
            )
            attempts.append(attempt)

            try:
                if strategy == RecoveryStrategy.RETRY:
                    recovered_result = self._try_retry(operation)
                elif strategy == RecoveryStrategy.FALLBACK:
                    recovered_result = self._try_fallback()
                elif strategy == RecoveryStrategy.ROLLBACK:
                    recovered_result = self._try_rollback(context)
                elif strategy == RecoveryStrategy.SKIP:
                    recovered_result = self._try_skip(context)
                elif strategy == RecoveryStrategy.ESCALATE:
                    recovered_result = self._try_escalate(error, context)
                elif strategy == RecoveryStrategy.COMPENSATE:
                    recovered_result = self._try_compensate(context)
                elif strategy == RecoveryStrategy.RESTART:
                    recovered_result = self._try_restart(operation)
                else:
                    continue

                # Success!
                attempt.success = True
                attempt.completed_at = time.time()

                result.status = RecoveryStatus.RECOVERED
                result.strategy_used = strategy
                result.result = recovered_result
                result.attempts = len(attempts)
                result.duration_seconds = time.time() - start_time

                with self._lock:
                    self._history.append(result)

                if self.on_recovery_complete:
                    self.on_recovery_complete(result)

                return result

            except Exception as e:
                attempt.error = str(e)
                attempt.completed_at = time.time()
                continue

        # All strategies failed
        result.status = RecoveryStatus.FAILED
        result.attempts = len(attempts)
        result.error = f"All recovery strategies failed. Original error: {error}"
        result.duration_seconds = time.time() - start_time

        with self._lock:
            self._history.append(result)

        if self.on_recovery_complete:
            self.on_recovery_complete(result)

        return result

    async def recover_async(
        self,
        operation: Callable[[], Any],
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> RecoveryResult:
        """Async version of recover.

        Args:
            operation: Original operation.
            error: The error that occurred.
            context: Additional context.

        Returns:
            Recovery result.
        """
        start_time = time.time()
        context = context or {}
        context["original_error"] = str(error)
        context["error_type"] = type(error).__name__

        result = RecoveryResult(status=RecoveryStatus.PENDING, context=context)
        attempts: list[RecoveryAttempt] = []

        for strategy in self.config.strategies:
            if self.on_recovery_start:
                self.on_recovery_start(context.get("operation_id", "unknown"), strategy)

            attempt = RecoveryAttempt(
                strategy=strategy,
                attempt_number=len(attempts) + 1,
            )
            attempts.append(attempt)

            try:
                if strategy == RecoveryStrategy.RETRY:
                    recovered_result = await self._try_retry_async(operation)
                elif strategy == RecoveryStrategy.FALLBACK:
                    recovered_result = await self._try_fallback_async()
                elif strategy == RecoveryStrategy.ESCALATE:
                    recovered_result = await self._try_escalate_async(error, context)
                else:
                    # Fall back to sync for other strategies
                    loop = asyncio.get_event_loop()
                    if strategy == RecoveryStrategy.ROLLBACK:
                        recovered_result = await loop.run_in_executor(
                            None, self._try_rollback, context
                        )
                    elif strategy == RecoveryStrategy.SKIP:
                        recovered_result = await loop.run_in_executor(
                            None, self._try_skip, context
                        )
                    elif strategy == RecoveryStrategy.COMPENSATE:
                        recovered_result = await loop.run_in_executor(
                            None, self._try_compensate, context
                        )
                    elif strategy == RecoveryStrategy.RESTART:
                        recovered_result = await loop.run_in_executor(
                            None, self._try_restart, operation
                        )
                    else:
                        continue

                attempt.success = True
                attempt.completed_at = time.time()

                result.status = RecoveryStatus.RECOVERED
                result.strategy_used = strategy
                result.result = recovered_result
                result.attempts = len(attempts)
                result.duration_seconds = time.time() - start_time

                with self._lock:
                    self._history.append(result)

                if self.on_recovery_complete:
                    self.on_recovery_complete(result)

                return result

            except Exception as e:
                attempt.error = str(e)
                attempt.completed_at = time.time()
                continue

        result.status = RecoveryStatus.FAILED
        result.attempts = len(attempts)
        result.error = f"All recovery strategies failed. Original error: {error}"
        result.duration_seconds = time.time() - start_time

        with self._lock:
            self._history.append(result)

        if self.on_recovery_complete:
            self.on_recovery_complete(result)

        return result

    def _try_retry(self, operation: Callable[[], T]) -> T:
        """Attempt retry strategy."""
        last_error = None
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries):
            try:
                return operation()
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(delay)
                    delay *= self.config.retry_backoff

        raise last_error or Exception("Retry failed")

    async def _try_retry_async(self, operation: Callable[[], Any]) -> Any:
        """Async retry strategy."""
        last_error = None
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= self.config.retry_backoff

        raise last_error or Exception("Retry failed")

    def _try_fallback(self) -> Any:
        """Attempt fallback strategy."""
        if not self.config.fallbacks:
            raise Exception("No fallbacks configured")

        for fallback in self.config.fallbacks:
            try:
                return fallback()
            except Exception:
                continue

        raise Exception("All fallbacks failed")

    async def _try_fallback_async(self) -> Any:
        """Async fallback strategy."""
        if not self.config.fallbacks:
            raise Exception("No fallbacks configured")

        for fallback in self.config.fallbacks:
            try:
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback()
                else:
                    return fallback()
            except Exception:
                continue

        raise Exception("All fallbacks failed")

    def _try_rollback(self, context: dict[str, Any]) -> Any:
        """Attempt rollback strategy."""
        state_key = context.get("state_key")
        if not state_key:
            raise Exception("No state_key in context for rollback")

        state = self.get_state(state_key)
        if state is None:
            raise Exception(f"No saved state for key: {state_key}")

        return state

    def _try_skip(self, context: dict[str, Any]) -> Any:
        """Skip strategy - return default value."""
        return context.get("default_value")

    def _try_escalate(self, error: Exception, context: dict[str, Any]) -> Any:
        """Escalate to human."""
        if self.config.escalation_handler is None:
            raise Exception("No escalation handler configured")

        return self.config.escalation_handler(error, context)

    async def _try_escalate_async(self, error: Exception, context: dict[str, Any]) -> Any:
        """Async escalation."""
        if self.config.escalation_handler is None:
            raise Exception("No escalation handler configured")

        if asyncio.iscoroutinefunction(self.config.escalation_handler):
            return await self.config.escalation_handler(error, context)
        else:
            return self.config.escalation_handler(error, context)

    def _try_compensate(self, context: dict[str, Any]) -> Any:
        """Run compensation logic."""
        if self.config.compensator is None:
            raise Exception("No compensator configured")

        self.config.compensator(context)
        return None

    def _try_restart(self, operation: Callable[[], T]) -> T:
        """Restart from beginning."""
        return operation()

    def get_history(
        self,
        status: RecoveryStatus | None = None,
        strategy: RecoveryStrategy | None = None,
        limit: int | None = None,
    ) -> list[RecoveryResult]:
        """Get recovery history.

        Args:
            status: Filter by status.
            strategy: Filter by strategy used.
            limit: Maximum results.

        Returns:
            Matching recovery results.
        """
        with self._lock:
            results = list(self._history)

        if status:
            results = [r for r in results if r.status == status]
        if strategy:
            results = [r for r in results if r.strategy_used == strategy]

        results = sorted(results, key=lambda r: r.duration_seconds, reverse=True)

        if limit:
            results = results[:limit]

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get recovery statistics."""
        with self._lock:
            total = len(self._history)
            if total == 0:
                return {"total": 0, "success_rate": 0}

            recovered = sum(1 for r in self._history if r.status == RecoveryStatus.RECOVERED)
            failed = sum(1 for r in self._history if r.status == RecoveryStatus.FAILED)

            by_strategy = {}
            for r in self._history:
                if r.strategy_used:
                    key = r.strategy_used.value
                    by_strategy[key] = by_strategy.get(key, 0) + 1

            avg_attempts = sum(r.attempts for r in self._history) / total
            avg_duration = sum(r.duration_seconds for r in self._history) / total

            return {
                "total": total,
                "recovered": recovered,
                "failed": failed,
                "success_rate": recovered / total,
                "by_strategy": by_strategy,
                "avg_attempts": avg_attempts,
                "avg_duration_seconds": avg_duration,
            }

    def clear_history(self) -> None:
        """Clear recovery history."""
        with self._lock:
            self._history.clear()

    def clear_snapshots(self) -> None:
        """Clear state snapshots."""
        with self._lock:
            self._state_snapshots.clear()
