"""Supervisor for coordinating multiple agents."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """A task to be executed by an agent."""

    id: str
    agent_type: str
    payload: dict[str, Any]
    priority: int = 0
    timeout_seconds: float | None = None
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime fields
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
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
class TaskResult:
    """Result of task execution."""

    task_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    duration_seconds: float | None = None


class Supervisor:
    """Coordinates multiple agent tasks with dependency management.

    Features:
    - Task queue with priority ordering
    - Dependency resolution
    - Concurrency control
    - Timeout handling
    - Progress tracking
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        default_timeout: float | None = 300.0,
        executor: Callable[[str, dict[str, Any]], Any] | None = None,
    ):
        """Initialize supervisor.

        Args:
            max_concurrent: Maximum concurrent tasks.
            default_timeout: Default task timeout in seconds.
            executor: Function to execute tasks (agent_type, payload) -> result.
        """
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.executor = executor or self._default_executor

        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._running_count = 0

    def add_task(
        self,
        task_id: str,
        agent_type: str,
        payload: dict[str, Any],
        priority: int = 0,
        timeout_seconds: float | None = None,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Add a task to the supervisor.

        Args:
            task_id: Unique task identifier.
            agent_type: Type of agent to execute task.
            payload: Task payload/parameters.
            priority: Task priority (higher = more important).
            timeout_seconds: Task timeout (None = use default).
            dependencies: List of task IDs that must complete first.
            metadata: Additional task metadata.

        Returns:
            Created task.
        """
        task = Task(
            id=task_id,
            agent_type=agent_type,
            payload=payload,
            priority=priority,
            timeout_seconds=timeout_seconds or self.default_timeout,
            dependencies=dependencies or [],
            metadata=metadata or {},
        )

        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task '{task_id}' already exists")
            self._tasks[task_id] = task

        return task

    def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.

        Args:
            task_id: Task to cancel.

        Returns:
            True if cancelled, False if not cancellable.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                return True
            return False

    def _default_executor(self, agent_type: str, payload: dict[str, Any]) -> Any:
        """Default executor (placeholder)."""
        raise NotImplementedError(
            "No executor provided. Set executor in constructor or override this method."
        )

    def _can_run_task(self, task: Task) -> bool:
        """Check if task can run (dependencies satisfied)."""
        if task.status != TaskStatus.PENDING:
            return False

        for dep_id in task.dependencies:
            dep_task = self._tasks.get(dep_id)
            if dep_task is None:
                # Missing dependency
                task.status = TaskStatus.FAILED
                task.error = f"Missing dependency: {dep_id}"
                return False
            if dep_task.status == TaskStatus.FAILED:
                task.status = TaskStatus.FAILED
                task.error = f"Dependency failed: {dep_id}"
                return False
            if dep_task.status != TaskStatus.COMPLETED:
                return False

        return True

    def _get_next_tasks(self) -> list[Task]:
        """Get next tasks to run (sorted by priority)."""
        runnable = []

        with self._lock:
            available_slots = self.max_concurrent - self._running_count

            for task in self._tasks.values():
                if available_slots <= 0:
                    break
                if self._can_run_task(task):
                    runnable.append(task)
                    available_slots -= 1

        # Sort by priority (descending)
        runnable.sort(key=lambda t: -t.priority)
        return runnable[: self.max_concurrent - self._running_count]

    async def run_task(self, task: Task) -> TaskResult:
        """Run a single task.

        Args:
            task: Task to run.

        Returns:
            Task result.
        """
        with self._lock:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            self._running_count += 1

        try:
            # Run with timeout
            if asyncio.iscoroutinefunction(self.executor):
                if task.timeout_seconds:
                    result = await asyncio.wait_for(
                        self.executor(task.agent_type, task.payload),
                        timeout=task.timeout_seconds,
                    )
                else:
                    result = await self.executor(task.agent_type, task.payload)
            else:
                # Sync executor - run in thread pool
                loop = asyncio.get_event_loop()
                if task.timeout_seconds:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, self.executor, task.agent_type, task.payload),
                        timeout=task.timeout_seconds,
                    )
                else:
                    result = await loop.run_in_executor(
                        None, self.executor, task.agent_type, task.payload
                    )

            task.status = TaskStatus.COMPLETED
            task.result = result

        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = f"Task timed out after {task.timeout_seconds}s"

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)

        finally:
            task.completed_at = time.time()
            with self._lock:
                self._running_count -= 1

        return TaskResult(
            task_id=task.id,
            status=task.status,
            result=task.result,
            error=task.error,
            duration_seconds=task.duration_seconds,
        )

    async def run_all(self) -> dict[str, TaskResult]:
        """Run all tasks respecting dependencies and concurrency.

        Returns:
            Dict of task_id to TaskResult.
        """
        results: dict[str, TaskResult] = {}
        running_tasks: dict[str, asyncio.Task] = {}

        while True:
            # Start new tasks
            next_tasks = self._get_next_tasks()
            for task in next_tasks:
                async_task = asyncio.create_task(self.run_task(task))
                running_tasks[task.id] = async_task

            # Check if done
            if not running_tasks:
                # No running tasks - check if all complete
                all_done = all(
                    t.status
                    in (
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                        TaskStatus.TIMEOUT,
                    )
                    for t in self._tasks.values()
                )
                if all_done:
                    break

                # Wait a bit and retry (might have pending tasks with deps)
                pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
                if not pending:
                    break

                await asyncio.sleep(0.1)
                continue

            # Wait for any task to complete
            done, _ = await asyncio.wait(
                running_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Process completed tasks
            for async_task in done:
                result = async_task.result()
                results[result.task_id] = result
                del running_tasks[result.task_id]

        return results

    def run_all_sync(self) -> dict[str, TaskResult]:
        """Synchronous version of run_all."""
        return asyncio.run(self.run_all())

    def get_progress(self) -> dict[str, Any]:
        """Get execution progress."""
        with self._lock:
            total = len(self._tasks)
            completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
            running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
            pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)

            return {
                "total": total,
                "completed": completed,
                "failed": failed,
                "running": running,
                "pending": pending,
                "progress_percent": (completed / total * 100) if total > 0 else 0,
            }

    def get_all_results(self) -> dict[str, TaskResult]:
        """Get results for all completed tasks."""
        results = {}
        for task in self._tasks.values():
            if task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT,
            ):
                results[task.id] = TaskResult(
                    task_id=task.id,
                    status=task.status,
                    result=task.result,
                    error=task.error,
                    duration_seconds=task.duration_seconds,
                )
        return results

    def reset(self) -> None:
        """Reset supervisor, clearing all tasks."""
        with self._lock:
            self._tasks.clear()
            self._running_count = 0


def create_task_id() -> str:
    """Generate unique task ID."""
    return str(uuid.uuid4())[:8]
