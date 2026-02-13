"""Parallel execution for independent tasks."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TaskState(Enum):
    """Parallel task state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ParallelTask(Generic[T]):
    """A task for parallel execution."""

    id: str
    callable: Callable[[], T]
    timeout_seconds: float | None = None

    # Runtime state
    state: TaskState = TaskState.PENDING
    result: T | None = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Get task duration."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at


@dataclass
class ParallelResult:
    """Result of parallel execution."""

    success: bool
    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    completed_count: int = 0
    failed_count: int = 0
    total_duration_seconds: float = 0.0

    @property
    def all_succeeded(self) -> bool:
        """Check if all tasks succeeded."""
        return self.failed_count == 0


class ParallelExecutor:
    """Execute multiple independent tasks in parallel.

    Features:
    - Configurable concurrency
    - Per-task timeout
    - Partial failure handling
    - Progress tracking
    - Both sync and async execution
    """

    def __init__(
        self,
        max_workers: int = 10,
        fail_fast: bool = False,
        default_timeout: float | None = None,
    ):
        """Initialize executor.

        Args:
            max_workers: Maximum concurrent workers.
            fail_fast: Stop all tasks on first failure.
            default_timeout: Default task timeout.
        """
        self.max_workers = max_workers
        self.fail_fast = fail_fast
        self.default_timeout = default_timeout

        self._tasks: dict[str, ParallelTask] = {}
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

    def add_task(
        self,
        task_id: str,
        callable: Callable[[], Any],
        timeout_seconds: float | None = None,
    ) -> ParallelExecutor:
        """Add a task for parallel execution.

        Args:
            task_id: Unique task identifier.
            callable: Function to execute.
            timeout_seconds: Task timeout.

        Returns:
            Self for chaining.
        """
        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task '{task_id}' already exists")

            self._tasks[task_id] = ParallelTask(
                id=task_id,
                callable=callable,
                timeout_seconds=timeout_seconds or self.default_timeout,
            )

        return self

    def run(self) -> ParallelResult:
        """Run all tasks in parallel (synchronous).

        Returns:
            Parallel execution result.
        """
        start_time = time.time()
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        self._cancel_event.clear()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task: dict[concurrent.futures.Future, ParallelTask] = {}

            for task in self._tasks.values():
                if self._cancel_event.is_set():
                    break

                task.state = TaskState.RUNNING
                task.started_at = time.time()

                future = executor.submit(self._execute_task, task)
                future_to_task[future] = task

            # Wait for completion
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    result = future.result()
                    task.result = result
                    task.state = TaskState.COMPLETED
                    task.completed_at = time.time()
                    results[task.id] = result

                except concurrent.futures.TimeoutError:
                    task.state = TaskState.TIMEOUT
                    task.error = f"Timeout after {task.timeout_seconds}s"
                    task.completed_at = time.time()
                    errors[task.id] = task.error

                    if self.fail_fast:
                        self._cancel_event.set()

                except Exception as e:
                    task.state = TaskState.FAILED
                    task.error = str(e)
                    task.completed_at = time.time()
                    errors[task.id] = task.error

                    if self.fail_fast:
                        self._cancel_event.set()

        # Count results
        completed = sum(1 for t in self._tasks.values() if t.state == TaskState.COMPLETED)
        failed = sum(
            1 for t in self._tasks.values() if t.state in (TaskState.FAILED, TaskState.TIMEOUT)
        )

        return ParallelResult(
            success=failed == 0,
            results=results,
            errors=errors,
            completed_count=completed,
            failed_count=failed,
            total_duration_seconds=time.time() - start_time,
        )

    def _execute_task(self, task: ParallelTask) -> Any:
        """Execute a single task with timeout."""
        if self._cancel_event.is_set():
            task.state = TaskState.CANCELLED
            raise concurrent.futures.CancelledError("Cancelled due to fail_fast")

        return task.callable()

    async def run_async(self) -> ParallelResult:
        """Run all tasks in parallel (asynchronous).

        Returns:
            Parallel execution result.
        """
        start_time = time.time()
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}

        # Create async tasks
        async_tasks: dict[asyncio.Task, ParallelTask] = {}
        semaphore = asyncio.Semaphore(self.max_workers)

        for task in self._tasks.values():
            task.state = TaskState.RUNNING
            task.started_at = time.time()

            async_task = asyncio.create_task(self._execute_task_async(task, semaphore))
            async_tasks[async_task] = task

        # Wait for completion
        if self.fail_fast:
            # Stop on first failure
            done, pending = await asyncio.wait(
                async_tasks.keys(),
                return_when=asyncio.FIRST_EXCEPTION,
            )

            # Cancel pending tasks
            for pending_task in pending:
                pending_task.cancel()
                task = async_tasks[pending_task]
                task.state = TaskState.CANCELLED

        else:
            # Wait for all
            done, _ = await asyncio.wait(async_tasks.keys())

        # Process results
        for async_task in done:
            task = async_tasks[async_task]
            task.completed_at = time.time()

            try:
                result = async_task.result()
                task.result = result
                task.state = TaskState.COMPLETED
                results[task.id] = result

            except asyncio.TimeoutError:
                task.state = TaskState.TIMEOUT
                task.error = f"Timeout after {task.timeout_seconds}s"
                errors[task.id] = task.error

            except asyncio.CancelledError:
                task.state = TaskState.CANCELLED
                task.error = "Cancelled"
                errors[task.id] = task.error

            except Exception as e:
                task.state = TaskState.FAILED
                task.error = str(e)
                errors[task.id] = task.error

        completed = sum(1 for t in self._tasks.values() if t.state == TaskState.COMPLETED)
        failed = sum(
            1
            for t in self._tasks.values()
            if t.state in (TaskState.FAILED, TaskState.TIMEOUT, TaskState.CANCELLED)
        )

        return ParallelResult(
            success=failed == 0,
            results=results,
            errors=errors,
            completed_count=completed,
            failed_count=failed,
            total_duration_seconds=time.time() - start_time,
        )

    async def _execute_task_async(
        self,
        task: ParallelTask,
        semaphore: asyncio.Semaphore,
    ) -> Any:
        """Execute a single task with semaphore and timeout."""
        async with semaphore:
            if asyncio.iscoroutinefunction(task.callable):
                coro = task.callable()
            else:
                loop = asyncio.get_event_loop()
                coro = loop.run_in_executor(None, task.callable)

            if task.timeout_seconds:
                return await asyncio.wait_for(coro, timeout=task.timeout_seconds)
            else:
                return await coro

    def get_progress(self) -> dict[str, Any]:
        """Get execution progress."""
        with self._lock:
            total = len(self._tasks)
            completed = sum(1 for t in self._tasks.values() if t.state == TaskState.COMPLETED)
            failed = sum(1 for t in self._tasks.values() if t.state == TaskState.FAILED)
            running = sum(1 for t in self._tasks.values() if t.state == TaskState.RUNNING)

            return {
                "total": total,
                "completed": completed,
                "failed": failed,
                "running": running,
                "progress_percent": (completed / total * 100) if total > 0 else 0,
            }

    def reset(self) -> None:
        """Reset executor, clearing all tasks."""
        with self._lock:
            self._tasks.clear()
            self._cancel_event.clear()


async def gather_with_concurrency(
    limit: int,
    *coros,
    return_exceptions: bool = False,
) -> list[Any]:
    """Run coroutines with concurrency limit.

    Args:
        limit: Maximum concurrent coroutines.
        *coros: Coroutines to run.
        return_exceptions: If True, return exceptions instead of raising.

    Returns:
        List of results in order.
    """
    semaphore = asyncio.Semaphore(limit)

    async def limited_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *[limited_coro(c) for c in coros],
        return_exceptions=return_exceptions,
    )


def map_parallel(
    func: Callable[[T], Any],
    items: list[T],
    max_workers: int = 10,
) -> list[Any]:
    """Map function over items in parallel.

    Args:
        func: Function to apply.
        items: Items to process.
        max_workers: Maximum concurrent workers.

    Returns:
        Results in same order as items.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))


async def map_parallel_async(
    func: Callable[[T], Any],
    items: list[T],
    max_concurrent: int = 10,
) -> list[Any]:
    """Async map function over items in parallel.

    Args:
        func: Function to apply (sync or async).
        items: Items to process.
        max_concurrent: Maximum concurrent executions.

    Returns:
        Results in same order as items.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_item(item: T) -> Any:
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, item)

    return await asyncio.gather(*[process_item(item) for item in items])
