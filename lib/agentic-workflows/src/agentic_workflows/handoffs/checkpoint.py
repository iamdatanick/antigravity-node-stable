"""Checkpoint system for human-in-the-loop approval."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class CheckpointStatus(Enum):
    """Checkpoint status."""

    WAITING = "waiting"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class CheckpointType(Enum):
    """Types of checkpoints."""

    APPROVAL = "approval"      # Simple yes/no
    REVIEW = "review"          # Review with feedback
    MODIFICATION = "modification"  # Allow changes
    CONFIRMATION = "confirmation"  # Confirm understanding


@dataclass
class Checkpoint:
    """A checkpoint requiring human interaction."""

    id: str
    checkpoint_type: CheckpointType
    title: str
    description: str
    agent_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: CheckpointStatus = CheckpointStatus.WAITING
    priority: int = 0

    # Timing
    created_at: float = field(default_factory=time.time)
    timeout_seconds: float | None = None
    responded_at: float | None = None

    # Response
    response: Any = None
    feedback: str = ""
    modified_payload: dict[str, Any] | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        """Check if checkpoint is resolved."""
        return self.status not in (CheckpointStatus.WAITING,)

    @property
    def wait_time_seconds(self) -> float:
        """Get time spent waiting."""
        if self.responded_at:
            return self.responded_at - self.created_at
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        """Check if checkpoint has expired."""
        if self.timeout_seconds is None:
            return False
        return time.time() - self.created_at > self.timeout_seconds


@dataclass
class CheckpointResponse:
    """Response to a checkpoint."""

    approved: bool
    feedback: str = ""
    modifications: dict[str, Any] | None = None
    respondent: str = "human"


class CheckpointManager:
    """Manages checkpoints for human-in-the-loop workflows.

    Features:
    - Multiple checkpoint types
    - Async/sync waiting
    - Timeout handling
    - Response history
    - Persistence support
    """

    def __init__(
        self,
        on_checkpoint_created: Callable[[Checkpoint], None] | None = None,
        on_checkpoint_resolved: Callable[[Checkpoint], None] | None = None,
        default_timeout: float | None = None,
        auto_approve: bool = False,
        persistence_path: Path | None = None,
    ):
        """Initialize checkpoint manager.

        Args:
            on_checkpoint_created: Callback when checkpoint created.
            on_checkpoint_resolved: Callback when checkpoint resolved.
            default_timeout: Default timeout in seconds.
            auto_approve: Auto-approve all checkpoints (for testing).
            persistence_path: Path for checkpoint persistence.
        """
        self.on_checkpoint_created = on_checkpoint_created
        self.on_checkpoint_resolved = on_checkpoint_resolved
        self.default_timeout = default_timeout
        self.auto_approve = auto_approve
        self.persistence_path = persistence_path

        self._checkpoints: dict[str, Checkpoint] = {}
        self._events: dict[str, threading.Event] = {}
        self._async_events: dict[str, asyncio.Event] = {}
        self._lock = threading.Lock()

        # Load persisted checkpoints
        if persistence_path:
            self._load_checkpoints()

    def create_checkpoint(
        self,
        checkpoint_type: CheckpointType,
        title: str,
        description: str,
        agent_id: str,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
        priority: int = 0,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a new checkpoint.

        Args:
            checkpoint_type: Type of checkpoint.
            title: Short title.
            description: Detailed description.
            agent_id: Agent creating checkpoint.
            payload: Data for review.
            timeout_seconds: Timeout (None = use default).
            priority: Priority level.
            tags: Tags for filtering.
            metadata: Additional metadata.

        Returns:
            Created checkpoint.
        """
        checkpoint_id = str(uuid.uuid4())[:12]

        checkpoint = Checkpoint(
            id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            title=title,
            description=description,
            agent_id=agent_id,
            payload=payload or {},
            timeout_seconds=timeout_seconds or self.default_timeout,
            priority=priority,
            tags=tags or [],
            metadata=metadata or {},
        )

        with self._lock:
            self._checkpoints[checkpoint_id] = checkpoint
            self._events[checkpoint_id] = threading.Event()

        # Auto-approve if configured
        if self.auto_approve:
            self.respond(checkpoint_id, CheckpointResponse(approved=True, feedback="Auto-approved"))

        if self.on_checkpoint_created:
            self.on_checkpoint_created(checkpoint)

        # Persist
        self._persist_checkpoint(checkpoint)

        return checkpoint

    def request_approval(
        self,
        agent_id: str,
        title: str,
        description: str,
        payload: dict[str, Any] | None = None,
        **kwargs,
    ) -> Checkpoint:
        """Create an approval checkpoint.

        Args:
            agent_id: Agent requesting approval.
            title: What needs approval.
            description: Details.
            payload: Data to review.
            **kwargs: Additional checkpoint options.

        Returns:
            Checkpoint instance.
        """
        return self.create_checkpoint(
            CheckpointType.APPROVAL,
            title,
            description,
            agent_id,
            payload,
            **kwargs,
        )

    def request_review(
        self,
        agent_id: str,
        title: str,
        content: str,
        payload: dict[str, Any] | None = None,
        **kwargs,
    ) -> Checkpoint:
        """Create a review checkpoint.

        Args:
            agent_id: Agent requesting review.
            title: What needs review.
            content: Content to review.
            payload: Additional data.
            **kwargs: Additional checkpoint options.

        Returns:
            Checkpoint instance.
        """
        payload = payload or {}
        payload["review_content"] = content
        return self.create_checkpoint(
            CheckpointType.REVIEW,
            title,
            content[:200] + "..." if len(content) > 200 else content,
            agent_id,
            payload,
            **kwargs,
        )

    def request_confirmation(
        self,
        agent_id: str,
        message: str,
        **kwargs,
    ) -> Checkpoint:
        """Create a confirmation checkpoint.

        Args:
            agent_id: Agent requesting confirmation.
            message: Message to confirm.
            **kwargs: Additional checkpoint options.

        Returns:
            Checkpoint instance.
        """
        return self.create_checkpoint(
            CheckpointType.CONFIRMATION,
            "Confirmation Required",
            message,
            agent_id,
            **kwargs,
        )

    def respond(
        self,
        checkpoint_id: str,
        response: CheckpointResponse,
    ) -> bool:
        """Respond to a checkpoint.

        Args:
            checkpoint_id: Checkpoint to respond to.
            response: Response details.

        Returns:
            True if responded successfully.
        """
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if checkpoint is None or checkpoint.is_resolved:
                return False

            checkpoint.responded_at = time.time()
            checkpoint.feedback = response.feedback

            if response.approved:
                if response.modifications:
                    checkpoint.status = CheckpointStatus.MODIFIED
                    checkpoint.modified_payload = response.modifications
                else:
                    checkpoint.status = CheckpointStatus.APPROVED
            else:
                checkpoint.status = CheckpointStatus.REJECTED

            checkpoint.response = response

            # Signal waiting threads
            event = self._events.get(checkpoint_id)
            if event:
                event.set()

            # Signal async waiters
            if checkpoint_id in self._async_events:
                self._async_events[checkpoint_id].set()

        if self.on_checkpoint_resolved:
            self.on_checkpoint_resolved(checkpoint)

        # Persist
        self._persist_checkpoint(checkpoint)

        return True

    def approve(self, checkpoint_id: str, feedback: str = "") -> bool:
        """Approve a checkpoint."""
        return self.respond(checkpoint_id, CheckpointResponse(approved=True, feedback=feedback))

    def reject(self, checkpoint_id: str, feedback: str = "") -> bool:
        """Reject a checkpoint."""
        return self.respond(checkpoint_id, CheckpointResponse(approved=False, feedback=feedback))

    def modify(
        self,
        checkpoint_id: str,
        modifications: dict[str, Any],
        feedback: str = "",
    ) -> bool:
        """Approve with modifications."""
        return self.respond(
            checkpoint_id,
            CheckpointResponse(approved=True, feedback=feedback, modifications=modifications),
        )

    def wait(
        self,
        checkpoint_id: str,
        timeout: float | None = None,
    ) -> Checkpoint | None:
        """Wait for checkpoint resolution (synchronous).

        Args:
            checkpoint_id: Checkpoint to wait for.
            timeout: Timeout in seconds.

        Returns:
            Resolved checkpoint or None if timeout.
        """
        event = self._events.get(checkpoint_id)
        if event is None:
            return None

        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint and checkpoint.is_resolved:
            return checkpoint

        # Determine timeout
        effective_timeout = timeout
        if effective_timeout is None and checkpoint:
            effective_timeout = checkpoint.timeout_seconds

        # Wait
        if event.wait(timeout=effective_timeout):
            return self._checkpoints.get(checkpoint_id)
        else:
            # Timeout
            self._handle_timeout(checkpoint_id)
            return self._checkpoints.get(checkpoint_id)

    async def wait_async(
        self,
        checkpoint_id: str,
        timeout: float | None = None,
    ) -> Checkpoint | None:
        """Wait for checkpoint resolution (asynchronous).

        Args:
            checkpoint_id: Checkpoint to wait for.
            timeout: Timeout in seconds.

        Returns:
            Resolved checkpoint or None if timeout.
        """
        # Create async event if needed
        if checkpoint_id not in self._async_events:
            self._async_events[checkpoint_id] = asyncio.Event()

        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint and checkpoint.is_resolved:
            return checkpoint

        # Determine timeout
        effective_timeout = timeout
        if effective_timeout is None and checkpoint:
            effective_timeout = checkpoint.timeout_seconds

        # Wait
        try:
            if effective_timeout:
                await asyncio.wait_for(
                    self._async_events[checkpoint_id].wait(),
                    timeout=effective_timeout,
                )
            else:
                await self._async_events[checkpoint_id].wait()
            return self._checkpoints.get(checkpoint_id)
        except asyncio.TimeoutError:
            self._handle_timeout(checkpoint_id)
            return self._checkpoints.get(checkpoint_id)

    def _handle_timeout(self, checkpoint_id: str) -> None:
        """Handle checkpoint timeout."""
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if checkpoint and not checkpoint.is_resolved:
                checkpoint.status = CheckpointStatus.TIMEOUT
                checkpoint.responded_at = time.time()

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Get checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    def get_pending_checkpoints(
        self,
        agent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> list[Checkpoint]:
        """Get pending checkpoints.

        Args:
            agent_id: Filter by agent.
            tags: Filter by tags.

        Returns:
            List of pending checkpoints.
        """
        checkpoints = [
            cp for cp in self._checkpoints.values()
            if cp.status == CheckpointStatus.WAITING
        ]

        if agent_id:
            checkpoints = [cp for cp in checkpoints if cp.agent_id == agent_id]

        if tags:
            checkpoints = [
                cp for cp in checkpoints
                if any(t in cp.tags for t in tags)
            ]

        return sorted(checkpoints, key=lambda cp: (-cp.priority, cp.created_at))

    def skip_checkpoint(self, checkpoint_id: str) -> bool:
        """Skip a checkpoint without approval.

        Args:
            checkpoint_id: Checkpoint to skip.

        Returns:
            True if skipped.
        """
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if checkpoint and not checkpoint.is_resolved:
                checkpoint.status = CheckpointStatus.SKIPPED
                checkpoint.responded_at = time.time()

                event = self._events.get(checkpoint_id)
                if event:
                    event.set()

                return True
            return False

    def _persist_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Persist checkpoint to file."""
        if self.persistence_path is None:
            return

        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            filepath = self.persistence_path / f"{checkpoint.id}.json"

            data = {
                "id": checkpoint.id,
                "checkpoint_type": checkpoint.checkpoint_type.value,
                "title": checkpoint.title,
                "description": checkpoint.description,
                "agent_id": checkpoint.agent_id,
                "payload": checkpoint.payload,
                "status": checkpoint.status.value,
                "priority": checkpoint.priority,
                "created_at": checkpoint.created_at,
                "timeout_seconds": checkpoint.timeout_seconds,
                "responded_at": checkpoint.responded_at,
                "feedback": checkpoint.feedback,
                "modified_payload": checkpoint.modified_payload,
                "metadata": checkpoint.metadata,
                "tags": checkpoint.tags,
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Fail silently for persistence

    def _load_checkpoints(self) -> None:
        """Load persisted checkpoints."""
        if self.persistence_path is None or not self.persistence_path.exists():
            return

        try:
            for filepath in self.persistence_path.glob("*.json"):
                with open(filepath) as f:
                    data = json.load(f)

                checkpoint = Checkpoint(
                    id=data["id"],
                    checkpoint_type=CheckpointType(data["checkpoint_type"]),
                    title=data["title"],
                    description=data["description"],
                    agent_id=data["agent_id"],
                    payload=data.get("payload", {}),
                    status=CheckpointStatus(data["status"]),
                    priority=data.get("priority", 0),
                    created_at=data.get("created_at", time.time()),
                    timeout_seconds=data.get("timeout_seconds"),
                    responded_at=data.get("responded_at"),
                    feedback=data.get("feedback", ""),
                    modified_payload=data.get("modified_payload"),
                    metadata=data.get("metadata", {}),
                    tags=data.get("tags", []),
                )

                self._checkpoints[checkpoint.id] = checkpoint
                self._events[checkpoint.id] = threading.Event()
                if checkpoint.is_resolved:
                    self._events[checkpoint.id].set()
        except Exception:
            pass

    def get_stats(self) -> dict[str, Any]:
        """Get checkpoint statistics."""
        with self._lock:
            total = len(self._checkpoints)
            by_status = {}
            by_type = {}

            for cp in self._checkpoints.values():
                by_status[cp.status.value] = by_status.get(cp.status.value, 0) + 1
                by_type[cp.checkpoint_type.value] = by_type.get(cp.checkpoint_type.value, 0) + 1

            resolved = [cp for cp in self._checkpoints.values() if cp.responded_at]
            avg_wait = (
                sum(cp.wait_time_seconds for cp in resolved) / len(resolved)
                if resolved else 0
            )

            return {
                "total": total,
                "by_status": by_status,
                "by_type": by_type,
                "avg_wait_seconds": avg_wait,
                "pending": by_status.get("waiting", 0),
            }

    def clear(self) -> None:
        """Clear all checkpoints."""
        with self._lock:
            self._checkpoints.clear()
            self._events.clear()
            self._async_events.clear()
