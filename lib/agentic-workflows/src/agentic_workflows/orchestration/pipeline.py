"""Pipeline pattern for sequential processing."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class StageStatus(Enum):
    """Pipeline stage status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage(Generic[T, U]):
    """A stage in the pipeline."""

    name: str
    processor: Callable[[T], U]
    timeout_seconds: float | None = None
    skip_on_error: bool = False
    validator: Callable[[U], bool] | None = None

    # Runtime state
    status: StageStatus = StageStatus.PENDING
    input_value: Any = None
    output_value: Any = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Get stage duration."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    success: bool
    final_output: Any = None
    stages_completed: int = 0
    stages_failed: int = 0
    total_duration_seconds: float = 0.0
    stage_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class Pipeline:
    """Sequential processing pipeline.

    Processes data through a series of stages, where each stage's
    output becomes the next stage's input.

    Features:
    - Type-safe stage chaining
    - Per-stage timeout
    - Error handling and skip options
    - Output validation
    - Progress tracking
    """

    def __init__(
        self,
        name: str = "pipeline",
        on_stage_complete: Callable[[str, Any], None] | None = None,
    ):
        """Initialize pipeline.

        Args:
            name: Pipeline name.
            on_stage_complete: Callback for stage completion.
        """
        self.name = name
        self.on_stage_complete = on_stage_complete
        self._stages: list[PipelineStage] = []

    def add_stage(
        self,
        name: str,
        processor: Callable[[Any], Any],
        timeout_seconds: float | None = None,
        skip_on_error: bool = False,
        validator: Callable[[Any], bool] | None = None,
    ) -> Pipeline:
        """Add a stage to the pipeline.

        Args:
            name: Stage name.
            processor: Function to process input.
            timeout_seconds: Stage timeout.
            skip_on_error: Whether to skip (not fail) on error.
            validator: Optional output validator.

        Returns:
            Self for chaining.
        """
        stage = PipelineStage(
            name=name,
            processor=processor,
            timeout_seconds=timeout_seconds,
            skip_on_error=skip_on_error,
            validator=validator,
        )
        self._stages.append(stage)
        return self

    def run(self, initial_input: Any) -> PipelineResult:
        """Run the pipeline synchronously.

        Args:
            initial_input: Input for the first stage.

        Returns:
            Pipeline result.
        """
        start_time = time.time()
        current_value = initial_input
        stages_completed = 0
        stages_failed = 0
        stage_results = []

        for stage in self._stages:
            stage.status = StageStatus.RUNNING
            stage.started_at = time.time()
            stage.input_value = current_value

            try:
                # Execute processor
                if stage.timeout_seconds:
                    # For sync, we can't easily timeout, so just run
                    # Consider using concurrent.futures for true timeout
                    output = stage.processor(current_value)
                else:
                    output = stage.processor(current_value)

                # Validate output
                if stage.validator and not stage.validator(output):
                    raise ValueError(f"Validation failed for stage '{stage.name}'")

                stage.output_value = output
                stage.status = StageStatus.COMPLETED
                stage.completed_at = time.time()
                stages_completed += 1
                current_value = output

                if self.on_stage_complete:
                    self.on_stage_complete(stage.name, output)

            except Exception as e:
                stage.error = str(e)
                stage.completed_at = time.time()

                if stage.skip_on_error:
                    stage.status = StageStatus.SKIPPED
                    # Keep current value for next stage
                else:
                    stage.status = StageStatus.FAILED
                    stages_failed += 1

                    stage_results.append(
                        {
                            "name": stage.name,
                            "status": stage.status.value,
                            "duration": stage.duration_seconds,
                            "error": stage.error,
                        }
                    )

                    return PipelineResult(
                        success=False,
                        final_output=None,
                        stages_completed=stages_completed,
                        stages_failed=stages_failed,
                        total_duration_seconds=time.time() - start_time,
                        stage_results=stage_results,
                        error=f"Stage '{stage.name}' failed: {e}",
                    )

            stage_results.append(
                {
                    "name": stage.name,
                    "status": stage.status.value,
                    "duration": stage.duration_seconds,
                    "error": stage.error,
                }
            )

        return PipelineResult(
            success=True,
            final_output=current_value,
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            total_duration_seconds=time.time() - start_time,
            stage_results=stage_results,
        )

    async def run_async(self, initial_input: Any) -> PipelineResult:
        """Run the pipeline asynchronously.

        Args:
            initial_input: Input for the first stage.

        Returns:
            Pipeline result.
        """
        start_time = time.time()
        current_value = initial_input
        stages_completed = 0
        stages_failed = 0
        stage_results = []

        for stage in self._stages:
            stage.status = StageStatus.RUNNING
            stage.started_at = time.time()
            stage.input_value = current_value

            try:
                # Execute processor
                if asyncio.iscoroutinefunction(stage.processor):
                    coro = stage.processor(current_value)
                else:
                    # Run sync processor in executor
                    loop = asyncio.get_event_loop()
                    coro = loop.run_in_executor(None, stage.processor, current_value)

                if stage.timeout_seconds:
                    output = await asyncio.wait_for(coro, timeout=stage.timeout_seconds)
                else:
                    output = await coro

                # Validate
                if stage.validator:
                    if asyncio.iscoroutinefunction(stage.validator):
                        valid = await stage.validator(output)
                    else:
                        valid = stage.validator(output)
                    if not valid:
                        raise ValueError(f"Validation failed for stage '{stage.name}'")

                stage.output_value = output
                stage.status = StageStatus.COMPLETED
                stage.completed_at = time.time()
                stages_completed += 1
                current_value = output

                if self.on_stage_complete:
                    self.on_stage_complete(stage.name, output)

            except asyncio.TimeoutError:
                stage.error = f"Timeout after {stage.timeout_seconds}s"
                stage.completed_at = time.time()

                if stage.skip_on_error:
                    stage.status = StageStatus.SKIPPED
                else:
                    stage.status = StageStatus.FAILED
                    stages_failed += 1

                    stage_results.append(
                        {
                            "name": stage.name,
                            "status": stage.status.value,
                            "duration": stage.duration_seconds,
                            "error": stage.error,
                        }
                    )

                    return PipelineResult(
                        success=False,
                        final_output=None,
                        stages_completed=stages_completed,
                        stages_failed=stages_failed,
                        total_duration_seconds=time.time() - start_time,
                        stage_results=stage_results,
                        error=f"Stage '{stage.name}' timed out",
                    )

            except Exception as e:
                stage.error = str(e)
                stage.completed_at = time.time()

                if stage.skip_on_error:
                    stage.status = StageStatus.SKIPPED
                else:
                    stage.status = StageStatus.FAILED
                    stages_failed += 1

                    stage_results.append(
                        {
                            "name": stage.name,
                            "status": stage.status.value,
                            "duration": stage.duration_seconds,
                            "error": stage.error,
                        }
                    )

                    return PipelineResult(
                        success=False,
                        final_output=None,
                        stages_completed=stages_completed,
                        stages_failed=stages_failed,
                        total_duration_seconds=time.time() - start_time,
                        stage_results=stage_results,
                        error=f"Stage '{stage.name}' failed: {e}",
                    )

            stage_results.append(
                {
                    "name": stage.name,
                    "status": stage.status.value,
                    "duration": stage.duration_seconds,
                    "error": stage.error,
                }
            )

        return PipelineResult(
            success=True,
            final_output=current_value,
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            total_duration_seconds=time.time() - start_time,
            stage_results=stage_results,
        )

    def get_progress(self) -> dict[str, Any]:
        """Get pipeline progress."""
        total = len(self._stages)
        completed = sum(1 for s in self._stages if s.status == StageStatus.COMPLETED)
        failed = sum(1 for s in self._stages if s.status == StageStatus.FAILED)
        running = sum(1 for s in self._stages if s.status == StageStatus.RUNNING)

        current_stage = None
        for stage in self._stages:
            if stage.status == StageStatus.RUNNING:
                current_stage = stage.name
                break

        return {
            "total_stages": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "current_stage": current_stage,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
        }

    def reset(self) -> None:
        """Reset all stages to pending."""
        for stage in self._stages:
            stage.status = StageStatus.PENDING
            stage.input_value = None
            stage.output_value = None
            stage.error = None
            stage.started_at = None
            stage.completed_at = None


class PipelineBuilder:
    """Fluent builder for pipelines."""

    def __init__(self, name: str = "pipeline"):
        """Initialize builder."""
        self._pipeline = Pipeline(name)

    def add(
        self,
        name: str,
        processor: Callable[[Any], Any],
        **kwargs,
    ) -> PipelineBuilder:
        """Add a stage."""
        self._pipeline.add_stage(name, processor, **kwargs)
        return self

    def on_complete(
        self,
        callback: Callable[[str, Any], None],
    ) -> PipelineBuilder:
        """Set stage completion callback."""
        self._pipeline.on_stage_complete = callback
        return self

    def build(self) -> Pipeline:
        """Build the pipeline."""
        return self._pipeline
