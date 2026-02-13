"""Handoff manager for agent-to-agent and agent-to-human transfers.

Enhanced with best practices from research:
- Schema versioning for payload compatibility
- Trace IDs for distributed tracing
- Conversation history preservation
- Validate-Repair-Escalate (VRE) pattern for failures
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Schema version for HandoffContext - increment on breaking changes
HANDOFF_SCHEMA_VERSION = "1.1.0"


class HandoffType(Enum):
    """Types of handoffs."""

    DELEGATION = "delegation"      # Assign subtask to specialist
    COLLABORATION = "collaboration"  # Joint work on task
    SUCCESSION = "succession"      # Complete transfer of ownership
    ESCALATION = "escalation"      # Escalate to human or higher authority


class HandoffStatus(Enum):
    """Handoff status."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REPAIRING = "repairing"  # VRE pattern: attempting repair


class HandoffFailureType(Enum):
    """Types of handoff failures for VRE pattern."""

    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    TARGET_UNAVAILABLE = "target_unavailable"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    SCHEMA_MISMATCH = "schema_mismatch"
    UNKNOWN = "unknown"


@dataclass
class ConversationMessage:
    """A message in the conversation history."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    agent_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HandoffContext:
    """Context passed during handoff.

    Enhanced with schema versioning and trace ID for distributed tracing.
    """

    task_description: str
    background: str = ""
    constraints: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Schema versioning for compatibility (v1.1.0)
    schema_version: str = HANDOFF_SCHEMA_VERSION

    # Distributed tracing support
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: str = ""

    # Conversation history for context preservation
    conversation_history: list[ConversationMessage] = field(default_factory=list)

    # State checksum for validation
    state_checksum: str = ""

    def add_message(self, role: str, content: str, agent_id: str = "", **metadata) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(ConversationMessage(
            role=role,
            content=content,
            agent_id=agent_id,
            metadata=metadata,
        ))

    def get_recent_messages(self, limit: int = 10) -> list[ConversationMessage]:
        """Get the most recent messages."""
        return self.conversation_history[-limit:] if self.conversation_history else []

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the handoff context.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not self.task_description:
            errors.append("task_description is required")

        if self.schema_version != HANDOFF_SCHEMA_VERSION:
            errors.append(f"schema_version mismatch: {self.schema_version} != {HANDOFF_SCHEMA_VERSION}")

        if not self.trace_id:
            errors.append("trace_id is required for distributed tracing")

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "task_description": self.task_description,
            "background": self.background,
            "constraints": self.constraints.copy(),
            "artifacts": self.artifacts.copy(),
            "metadata": self.metadata.copy(),
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "conversation_history": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "agent_id": msg.agent_id,
                    "metadata": msg.metadata,
                }
                for msg in self.conversation_history
            ],
            "state_checksum": self.state_checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HandoffContext":
        """Deserialize context from dictionary."""
        history = [
            ConversationMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp", time.time()),
                agent_id=msg.get("agent_id", ""),
                metadata=msg.get("metadata", {}),
            )
            for msg in data.get("conversation_history", [])
        ]

        return cls(
            task_description=data["task_description"],
            background=data.get("background", ""),
            constraints=data.get("constraints", []),
            artifacts=data.get("artifacts", {}),
            metadata=data.get("metadata", {}),
            schema_version=data.get("schema_version", HANDOFF_SCHEMA_VERSION),
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            parent_span_id=data.get("parent_span_id", ""),
            conversation_history=history,
            state_checksum=data.get("state_checksum", ""),
        )


@dataclass
class Handoff:
    """A handoff between agents or to human."""

    id: str
    handoff_type: HandoffType
    source_agent: str
    target_agent: str  # Can be "human" for escalation
    context: HandoffContext
    status: HandoffStatus = HandoffStatus.PENDING
    priority: int = 0

    # Timing
    created_at: float = field(default_factory=time.time)
    accepted_at: float | None = None
    completed_at: float | None = None

    # Results
    result: Any = None
    error: str | None = None
    feedback: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Get handoff duration."""
        if self.completed_at and self.accepted_at:
            return self.completed_at - self.accepted_at
        return None

    @property
    def wait_time_seconds(self) -> float | None:
        """Get time waiting to be accepted."""
        if self.accepted_at:
            return self.accepted_at - self.created_at
        return None

    @property
    def is_to_human(self) -> bool:
        """Check if this is a human handoff."""
        return self.target_agent.lower() in ("human", "user", "operator")


class HandoffManager:
    """Manages handoffs between agents and humans.

    Features:
    - Agent-to-agent handoffs
    - Human-in-the-loop escalation
    - Handoff lifecycle management
    - Result aggregation
    - Timeout handling
    """

    def __init__(
        self,
        on_handoff_created: Callable[[Handoff], None] | None = None,
        on_handoff_completed: Callable[[Handoff], None] | None = None,
        human_handler: Callable[[Handoff], Any] | None = None,
        default_timeout: float = 300.0,
    ):
        """Initialize handoff manager.

        Args:
            on_handoff_created: Callback when handoff is created.
            on_handoff_completed: Callback when handoff completes.
            human_handler: Handler for human escalations.
            default_timeout: Default handoff timeout.
        """
        self.on_handoff_created = on_handoff_created
        self.on_handoff_completed = on_handoff_completed
        self.human_handler = human_handler
        self.default_timeout = default_timeout

        self._handoffs: dict[str, Handoff] = {}
        self._agent_queues: dict[str, list[str]] = {}  # agent -> [handoff_ids]
        self._lock = threading.Lock()

    def create_handoff(
        self,
        handoff_type: HandoffType,
        source_agent: str,
        target_agent: str,
        context: HandoffContext,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Handoff:
        """Create a new handoff.

        Args:
            handoff_type: Type of handoff.
            source_agent: Source agent ID.
            target_agent: Target agent ID or "human".
            context: Handoff context.
            priority: Priority (higher = more important).
            metadata: Additional metadata.

        Returns:
            Created handoff.
        """
        handoff_id = str(uuid.uuid4())[:12]

        handoff = Handoff(
            id=handoff_id,
            handoff_type=handoff_type,
            source_agent=source_agent,
            target_agent=target_agent,
            context=context,
            priority=priority,
            metadata=metadata or {},
        )

        with self._lock:
            self._handoffs[handoff_id] = handoff

            # Add to target's queue
            if target_agent not in self._agent_queues:
                self._agent_queues[target_agent] = []
            self._agent_queues[target_agent].append(handoff_id)

        if self.on_handoff_created:
            self.on_handoff_created(handoff)

        return handoff

    def delegate(
        self,
        source_agent: str,
        target_agent: str,
        task: str,
        background: str = "",
        artifacts: dict[str, Any] | None = None,
    ) -> Handoff:
        """Create a delegation handoff.

        Args:
            source_agent: Agent delegating.
            target_agent: Agent receiving task.
            task: Task description.
            background: Background context.
            artifacts: Relevant artifacts.

        Returns:
            Handoff instance.
        """
        context = HandoffContext(
            task_description=task,
            background=background,
            artifacts=artifacts or {},
        )
        return self.create_handoff(
            HandoffType.DELEGATION,
            source_agent,
            target_agent,
            context,
        )

    def collaborate(
        self,
        source_agent: str,
        target_agent: str,
        task: str,
        shared_context: dict[str, Any] | None = None,
    ) -> Handoff:
        """Create a collaboration handoff.

        Args:
            source_agent: Initiating agent.
            target_agent: Collaborating agent.
            task: Joint task.
            shared_context: Shared artifacts/state.

        Returns:
            Handoff instance.
        """
        context = HandoffContext(
            task_description=task,
            artifacts=shared_context or {},
            metadata={"collaboration_mode": True},
        )
        return self.create_handoff(
            HandoffType.COLLABORATION,
            source_agent,
            target_agent,
            context,
        )

    def escalate(
        self,
        source_agent: str,
        reason: str,
        context_summary: str,
        artifacts: dict[str, Any] | None = None,
        priority: int = 1,
    ) -> Handoff:
        """Escalate to human.

        Args:
            source_agent: Agent escalating.
            reason: Reason for escalation.
            context_summary: Summary of context.
            artifacts: Relevant artifacts.
            priority: Escalation priority.

        Returns:
            Handoff instance.
        """
        context = HandoffContext(
            task_description=reason,
            background=context_summary,
            artifacts=artifacts or {},
            metadata={"escalation_reason": reason},
        )
        return self.create_handoff(
            HandoffType.ESCALATION,
            source_agent,
            "human",
            context,
            priority=priority,
        )

    def accept_handoff(self, handoff_id: str) -> bool:
        """Accept a handoff.

        Args:
            handoff_id: Handoff to accept.

        Returns:
            True if accepted.
        """
        with self._lock:
            handoff = self._handoffs.get(handoff_id)
            if handoff and handoff.status == HandoffStatus.PENDING:
                handoff.status = HandoffStatus.ACCEPTED
                handoff.accepted_at = time.time()
                return True
            return False

    def reject_handoff(self, handoff_id: str, reason: str = "") -> bool:
        """Reject a handoff.

        Args:
            handoff_id: Handoff to reject.
            reason: Rejection reason.

        Returns:
            True if rejected.
        """
        with self._lock:
            handoff = self._handoffs.get(handoff_id)
            if handoff and handoff.status == HandoffStatus.PENDING:
                handoff.status = HandoffStatus.REJECTED
                handoff.error = reason
                handoff.completed_at = time.time()
                return True
            return False

    def complete_handoff(
        self,
        handoff_id: str,
        result: Any = None,
        feedback: str = "",
    ) -> bool:
        """Complete a handoff.

        Args:
            handoff_id: Handoff to complete.
            result: Result of the work.
            feedback: Feedback on the handoff.

        Returns:
            True if completed.
        """
        with self._lock:
            handoff = self._handoffs.get(handoff_id)
            if handoff and handoff.status in (HandoffStatus.ACCEPTED, HandoffStatus.IN_PROGRESS):
                handoff.status = HandoffStatus.COMPLETED
                handoff.result = result
                handoff.feedback = feedback
                handoff.completed_at = time.time()

                if self.on_handoff_completed:
                    self.on_handoff_completed(handoff)

                return True
            return False

    def fail_handoff(self, handoff_id: str, error: str) -> bool:
        """Mark handoff as failed.

        Args:
            handoff_id: Handoff that failed.
            error: Error message.

        Returns:
            True if marked failed.
        """
        with self._lock:
            handoff = self._handoffs.get(handoff_id)
            if handoff and handoff.status in (HandoffStatus.PENDING, HandoffStatus.ACCEPTED, HandoffStatus.IN_PROGRESS):
                handoff.status = HandoffStatus.FAILED
                handoff.error = error
                handoff.completed_at = time.time()

                if self.on_handoff_completed:
                    self.on_handoff_completed(handoff)

                return True
            return False

    def get_handoff(self, handoff_id: str) -> Handoff | None:
        """Get handoff by ID."""
        return self._handoffs.get(handoff_id)

    def get_pending_handoffs(self, agent_id: str) -> list[Handoff]:
        """Get pending handoffs for an agent.

        Args:
            agent_id: Agent ID.

        Returns:
            List of pending handoffs, sorted by priority.
        """
        with self._lock:
            handoff_ids = self._agent_queues.get(agent_id, [])
            handoffs = [
                self._handoffs[hid]
                for hid in handoff_ids
                if hid in self._handoffs and self._handoffs[hid].status == HandoffStatus.PENDING
            ]
            return sorted(handoffs, key=lambda h: -h.priority)

    def get_human_escalations(self) -> list[Handoff]:
        """Get all human escalations.

        Returns:
            List of human escalations.
        """
        return [
            h for h in self._handoffs.values()
            if h.is_to_human and h.status in (HandoffStatus.PENDING, HandoffStatus.ACCEPTED)
        ]

    def process_human_escalation(
        self,
        handoff_id: str,
        response: Any = None,
        approved: bool = True,
        feedback: str = "",
    ) -> bool:
        """Process a human escalation response.

        Args:
            handoff_id: Escalation ID.
            response: Human response.
            approved: Whether approved.
            feedback: Additional feedback.

        Returns:
            True if processed.
        """
        handoff = self.get_handoff(handoff_id)
        if handoff is None or not handoff.is_to_human:
            return False

        self.accept_handoff(handoff_id)

        if approved:
            return self.complete_handoff(handoff_id, response, feedback)
        else:
            return self.reject_handoff(handoff_id, feedback)

    def get_handoffs_by_source(self, source_agent: str) -> list[Handoff]:
        """Get all handoffs from a source agent."""
        return [h for h in self._handoffs.values() if h.source_agent == source_agent]

    def get_handoffs_by_target(self, target_agent: str) -> list[Handoff]:
        """Get all handoffs to a target agent."""
        return [h for h in self._handoffs.values() if h.target_agent == target_agent]

    def get_active_handoffs(self) -> list[Handoff]:
        """Get all active (non-completed) handoffs."""
        active_statuses = {
            HandoffStatus.PENDING,
            HandoffStatus.ACCEPTED,
            HandoffStatus.IN_PROGRESS,
        }
        return [h for h in self._handoffs.values() if h.status in active_statuses]

    def check_timeouts(self) -> list[Handoff]:
        """Check for timed out handoffs.

        Returns:
            List of handoffs that timed out.
        """
        timed_out = []
        now = time.time()

        with self._lock:
            for handoff in self._handoffs.values():
                if handoff.status in (HandoffStatus.PENDING, HandoffStatus.ACCEPTED):
                    if now - handoff.created_at > self.default_timeout:
                        handoff.status = HandoffStatus.FAILED
                        handoff.error = "Timeout"
                        handoff.completed_at = now
                        timed_out.append(handoff)

        return timed_out

    def get_stats(self) -> dict[str, Any]:
        """Get handoff statistics."""
        with self._lock:
            total = len(self._handoffs)
            by_status = {}
            by_type = {}

            for h in self._handoffs.values():
                by_status[h.status.value] = by_status.get(h.status.value, 0) + 1
                by_type[h.handoff_type.value] = by_type.get(h.handoff_type.value, 0) + 1

            completed = [h for h in self._handoffs.values() if h.duration_seconds]
            avg_duration = (
                sum(h.duration_seconds for h in completed) / len(completed)
                if completed else 0
            )

            return {
                "total": total,
                "by_status": by_status,
                "by_type": by_type,
                "avg_duration_seconds": avg_duration,
                "human_escalations": sum(1 for h in self._handoffs.values() if h.is_to_human),
            }

    def clear(self) -> None:
        """Clear all handoffs."""
        with self._lock:
            self._handoffs.clear()
            self._agent_queues.clear()

    # =========================================================================
    # Validate-Repair-Escalate (VRE) Pattern
    # =========================================================================

    def validate_handoff(self, handoff_id: str) -> tuple[bool, list[str]]:
        """Validate a handoff before processing.

        Part of VRE pattern - Step 1: Validate.

        Args:
            handoff_id: Handoff to validate.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        handoff = self.get_handoff(handoff_id)
        if handoff is None:
            return False, [f"Handoff {handoff_id} not found"]

        errors = []

        # Validate context
        context_valid, context_errors = handoff.context.validate()
        if not context_valid:
            errors.extend(context_errors)

        # Validate target agent availability
        if handoff.target_agent not in self._agent_queues:
            if not handoff.is_to_human:
                errors.append(f"Target agent {handoff.target_agent} not registered")

        # Validate schema version compatibility
        if handoff.context.schema_version != HANDOFF_SCHEMA_VERSION:
            errors.append(
                f"Schema version mismatch: {handoff.context.schema_version} "
                f"(expected {HANDOFF_SCHEMA_VERSION})"
            )

        return len(errors) == 0, errors

    def repair_handoff(
        self,
        handoff_id: str,
        failure_type: HandoffFailureType,
        repair_strategy: str = "auto",
    ) -> tuple[bool, str]:
        """Attempt to repair a failed handoff.

        Part of VRE pattern - Step 2: Repair.

        Repair strategies:
        - "auto": Automatic repair based on failure type
        - "retry": Simply retry the handoff
        - "reroute": Route to alternative target
        - "simplify": Simplify the context and retry

        Args:
            handoff_id: Handoff to repair.
            failure_type: Type of failure encountered.
            repair_strategy: Strategy to use for repair.

        Returns:
            Tuple of (success, message).
        """
        handoff = self.get_handoff(handoff_id)
        if handoff is None:
            return False, f"Handoff {handoff_id} not found"

        logger.info(f"Attempting repair of handoff {handoff_id} with strategy: {repair_strategy}")

        with self._lock:
            handoff.status = HandoffStatus.REPAIRING
            handoff.metadata["repair_attempts"] = handoff.metadata.get("repair_attempts", 0) + 1
            handoff.metadata["last_failure_type"] = failure_type.value

        # Max repair attempts
        if handoff.metadata.get("repair_attempts", 0) > 3:
            return False, "Max repair attempts exceeded"

        if repair_strategy == "auto":
            # Select strategy based on failure type
            if failure_type == HandoffFailureType.TIMEOUT:
                repair_strategy = "retry"
            elif failure_type == HandoffFailureType.TARGET_UNAVAILABLE:
                repair_strategy = "reroute"
            elif failure_type == HandoffFailureType.SCHEMA_MISMATCH:
                repair_strategy = "upgrade_schema"
            elif failure_type == HandoffFailureType.VALIDATION_ERROR:
                repair_strategy = "simplify"
            else:
                repair_strategy = "retry"

        if repair_strategy == "retry":
            # Reset to pending and extend timeout
            with self._lock:
                handoff.status = HandoffStatus.PENDING
                handoff.created_at = time.time()  # Reset timer
                handoff.error = None
            return True, "Handoff reset for retry"

        elif repair_strategy == "reroute":
            # Find alternative target
            alternative = self._find_alternative_target(handoff)
            if alternative:
                with self._lock:
                    handoff.metadata["original_target"] = handoff.target_agent
                    handoff.target_agent = alternative
                    handoff.status = HandoffStatus.PENDING
                    handoff.created_at = time.time()
                    # Update queue
                    if alternative not in self._agent_queues:
                        self._agent_queues[alternative] = []
                    self._agent_queues[alternative].append(handoff_id)
                return True, f"Rerouted to {alternative}"
            return False, "No alternative target available"

        elif repair_strategy == "upgrade_schema":
            # Migrate context to current schema
            with self._lock:
                handoff.context.schema_version = HANDOFF_SCHEMA_VERSION
                if not handoff.context.trace_id:
                    handoff.context.trace_id = str(uuid.uuid4())
                handoff.status = HandoffStatus.PENDING
            return True, "Schema upgraded to current version"

        elif repair_strategy == "simplify":
            # Simplify context by removing optional data
            with self._lock:
                # Keep only essential history
                if len(handoff.context.conversation_history) > 5:
                    handoff.context.conversation_history = handoff.context.get_recent_messages(5)
                # Clear non-essential metadata
                handoff.context.metadata = {
                    k: v for k, v in handoff.context.metadata.items()
                    if k in ("priority", "trace_id", "critical")
                }
                handoff.status = HandoffStatus.PENDING
            return True, "Context simplified"

        return False, f"Unknown repair strategy: {repair_strategy}"

    def _find_alternative_target(self, handoff: Handoff) -> str | None:
        """Find an alternative target agent for rerouting.

        Args:
            handoff: The handoff to reroute.

        Returns:
            Alternative agent ID or None.
        """
        original = handoff.target_agent
        original_base = original.rstrip("0123456789")  # e.g., "builder" from "builder1"

        # Look for similar agents
        for agent_id in self._agent_queues.keys():
            if agent_id != original and agent_id.startswith(original_base):
                # Check queue size (prefer less busy agents)
                queue_size = len(self._agent_queues.get(agent_id, []))
                if queue_size < 10:  # Arbitrary threshold
                    return agent_id

        # Check for fallback in metadata
        fallback = handoff.metadata.get("fallback_agent")
        if fallback and fallback in self._agent_queues:
            return fallback

        return None

    def escalate_failed_handoff(
        self,
        handoff_id: str,
        failure_type: HandoffFailureType,
        repair_attempted: bool = True,
    ) -> Handoff:
        """Escalate a handoff that couldn't be repaired.

        Part of VRE pattern - Step 3: Escalate.

        Creates a new escalation handoff to human/supervisor when
        automated repair fails.

        Args:
            handoff_id: Original handoff that failed.
            failure_type: Type of failure.
            repair_attempted: Whether repair was attempted.

        Returns:
            New escalation handoff.
        """
        original = self.get_handoff(handoff_id)
        if original is None:
            raise ValueError(f"Handoff {handoff_id} not found")

        # Mark original as failed if not already
        with self._lock:
            if original.status != HandoffStatus.FAILED:
                original.status = HandoffStatus.FAILED
                original.error = f"Escalated: {failure_type.value}"
                original.completed_at = time.time()

        # Create escalation context
        escalation_context = HandoffContext(
            task_description=f"Handoff failed: {original.context.task_description}",
            background=f"""
Original handoff {handoff_id} failed and requires human intervention.

Failure Type: {failure_type.value}
Repair Attempted: {repair_attempted}
Source Agent: {original.source_agent}
Target Agent: {original.target_agent}

Original Background:
{original.context.background}
            """.strip(),
            constraints=original.context.constraints,
            artifacts={
                "original_handoff_id": handoff_id,
                "failure_type": failure_type.value,
                "repair_attempts": original.metadata.get("repair_attempts", 0),
                **original.context.artifacts,
            },
            metadata={
                "escalated_from": handoff_id,
                "original_trace_id": original.context.trace_id,
            },
            trace_id=original.context.trace_id,  # Preserve trace
            parent_span_id=handoff_id,
            conversation_history=original.context.conversation_history,
        )

        # Create escalation
        escalation = self.create_handoff(
            handoff_type=HandoffType.ESCALATION,
            source_agent=original.target_agent,
            target_agent="human",
            context=escalation_context,
            priority=original.priority + 1,  # Higher priority
            metadata={"original_handoff_id": handoff_id},
        )

        logger.warning(
            f"Handoff {handoff_id} escalated to human as {escalation.id} "
            f"due to {failure_type.value}"
        )

        return escalation

    def process_with_vre(
        self,
        handoff_id: str,
        processor: Callable[[Handoff], Any],
        max_repair_attempts: int = 3,
    ) -> tuple[bool, Any]:
        """Process a handoff using the Validate-Repair-Escalate pattern.

        This is the main entry point for VRE-based handoff processing.

        Args:
            handoff_id: Handoff to process.
            processor: Function to process the handoff.
            max_repair_attempts: Maximum repair attempts before escalation.

        Returns:
            Tuple of (success, result_or_escalation).
        """
        handoff = self.get_handoff(handoff_id)
        if handoff is None:
            return False, None

        # Step 1: Validate
        is_valid, errors = self.validate_handoff(handoff_id)
        if not is_valid:
            logger.warning(f"Handoff {handoff_id} validation failed: {errors}")

            # Attempt repair for validation errors
            repaired, msg = self.repair_handoff(
                handoff_id,
                HandoffFailureType.VALIDATION_ERROR,
            )
            if not repaired:
                escalation = self.escalate_failed_handoff(
                    handoff_id,
                    HandoffFailureType.VALIDATION_ERROR,
                )
                return False, escalation

        # Step 2: Process with repair on failure
        attempts = 0
        last_error = None

        while attempts < max_repair_attempts:
            try:
                self.accept_handoff(handoff_id)
                result = processor(handoff)
                self.complete_handoff(handoff_id, result)
                return True, result

            except TimeoutError as e:
                last_error = e
                failure_type = HandoffFailureType.TIMEOUT
            except ConnectionError as e:
                last_error = e
                failure_type = HandoffFailureType.TARGET_UNAVAILABLE
            except Exception as e:
                last_error = e
                failure_type = HandoffFailureType.UNKNOWN

            logger.warning(
                f"Handoff {handoff_id} failed (attempt {attempts + 1}): {last_error}"
            )

            # Attempt repair
            repaired, msg = self.repair_handoff(handoff_id, failure_type)
            if not repaired:
                break

            attempts += 1

        # Step 3: Escalate if all repairs failed
        escalation = self.escalate_failed_handoff(
            handoff_id,
            failure_type,
            repair_attempted=True,
        )
        return False, escalation
