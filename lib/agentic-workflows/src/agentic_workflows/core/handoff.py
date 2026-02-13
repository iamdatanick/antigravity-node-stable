"""
Agent Handoff Protocol

Manages transitions between agents including delegation, collaboration,
succession, and human escalation with trust transfer and accountability.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class HandoffType(Enum):
    """Types of agent handoff"""

    DELEGATION = "delegation"  # Parent-child, wait for result
    COLLABORATION = "collaboration"  # Peer-to-peer, shared context
    SUCCESSION = "succession"  # Complete transfer of responsibility
    ESCALATION = "escalation"  # To human or higher authority


class HandoffStatus(Enum):
    """Status of a handoff"""

    PENDING = "pending"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class TrustTransfer:
    """Trust transfer semantics for handoff"""

    inherit_permissions: bool = False  # Receiving agent inherits sender's permissions
    trust_level: float = 0.7  # Trust level for transferred context
    verify_context: bool = True  # Receiving agent should verify context
    can_delegate_further: bool = False  # Can receiving agent delegate to others
    max_delegation_depth: int = 3  # Maximum delegation chain depth


@dataclass
class Accountability:
    """Accountability assignment for handoff"""

    outcome_owner: str  # Agent responsible for outcome
    decision_maker: str  # Agent that made the decision
    approver: str | None  # Human who approved (if applicable)
    audit_required: bool = True  # Whether audit trail is required


@dataclass
class HandoffContext:
    """Context transferred during handoff"""

    task_description: str
    task_state: dict[str, Any]
    relevant_context: list[dict[str, Any]]
    trace_context: dict[str, str]  # trace_id, span_id for continuity
    scratchpad: str | None = None
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HandoffRequest:
    """A request to hand off work to another agent"""

    handoff_id: str
    handoff_type: HandoffType
    from_agent: str
    to_agent: str
    context: HandoffContext
    trust_transfer: TrustTransfer
    accountability: Accountability
    reason: str
    created_at: datetime
    timeout_seconds: int = 300
    status: HandoffStatus = HandoffStatus.PENDING

    # Response fields (filled after acceptance)
    accepted_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any | None = None
    error: str | None = None


@dataclass
class HandoffEvent:
    """Event in the handoff lifecycle"""

    timestamp: datetime
    event_type: str
    handoff_id: str
    from_agent: str
    to_agent: str
    details: dict[str, Any]


class HandoffManager:
    """
    Manages agent-to-agent handoffs with full tracking and accountability.

    Features:
    - Multiple handoff types (delegation, collaboration, succession, escalation)
    - Trust transfer semantics
    - Trace continuity
    - Accountability tracking
    - Human escalation support
    """

    def __init__(self):
        self.active_handoffs: dict[str, HandoffRequest] = {}
        self.completed_handoffs: list[HandoffRequest] = []
        self.events: list[HandoffEvent] = []
        self.handlers: dict[str, Callable] = {}
        self.human_approval_callback: Callable | None = None

    def create_handoff(
        self,
        handoff_type: HandoffType,
        from_agent: str,
        to_agent: str,
        task_description: str,
        task_state: dict[str, Any],
        trace_context: dict[str, str],
        reason: str,
        relevant_context: list[dict[str, Any]] = None,
        trust_transfer: TrustTransfer = None,
        timeout_seconds: int = 300,
    ) -> HandoffRequest:
        """Create a new handoff request"""

        handoff_id = f"handoff-{uuid4().hex[:12]}"

        context = HandoffContext(
            task_description=task_description,
            task_state=task_state,
            relevant_context=relevant_context or [],
            trace_context=trace_context,
        )

        # Default trust transfer
        if trust_transfer is None:
            trust_transfer = TrustTransfer()

        # Set accountability
        if handoff_type == HandoffType.DELEGATION:
            accountability = Accountability(
                outcome_owner=from_agent,  # Delegator owns outcome
                decision_maker=from_agent,
            )
        elif handoff_type == HandoffType.SUCCESSION:
            accountability = Accountability(
                outcome_owner=to_agent,  # Successor owns outcome
                decision_maker=from_agent,
            )
        elif handoff_type == HandoffType.ESCALATION:
            accountability = Accountability(
                outcome_owner=to_agent,  # Escalation target owns outcome
                decision_maker=from_agent,
                audit_required=True,
            )
        else:  # COLLABORATION
            accountability = Accountability(outcome_owner="shared", decision_maker=from_agent)

        handoff = HandoffRequest(
            handoff_id=handoff_id,
            handoff_type=handoff_type,
            from_agent=from_agent,
            to_agent=to_agent,
            context=context,
            trust_transfer=trust_transfer,
            accountability=accountability,
            reason=reason,
            created_at=datetime.utcnow(),
            timeout_seconds=timeout_seconds,
        )

        self.active_handoffs[handoff_id] = handoff
        self._emit_event("handoff_created", handoff)

        return handoff

    def accept_handoff(self, handoff_id: str) -> bool:
        """Accept a handoff request"""
        handoff = self.active_handoffs.get(handoff_id)
        if not handoff or handoff.status != HandoffStatus.PENDING:
            return False

        handoff.status = HandoffStatus.ACCEPTED
        handoff.accepted_at = datetime.utcnow()

        self._emit_event("handoff_accepted", handoff)
        return True

    def start_handoff(self, handoff_id: str) -> bool:
        """Mark handoff as in progress"""
        handoff = self.active_handoffs.get(handoff_id)
        if not handoff or handoff.status != HandoffStatus.ACCEPTED:
            return False

        handoff.status = HandoffStatus.IN_PROGRESS
        self._emit_event("handoff_started", handoff)
        return True

    def complete_handoff(self, handoff_id: str, result: Any, success: bool = True) -> bool:
        """Complete a handoff with result"""
        handoff = self.active_handoffs.get(handoff_id)
        if not handoff:
            return False

        handoff.completed_at = datetime.utcnow()
        handoff.result = result

        if success:
            handoff.status = HandoffStatus.COMPLETED
        else:
            handoff.status = HandoffStatus.FAILED
            handoff.error = str(result) if not success else None

        # Move to completed
        del self.active_handoffs[handoff_id]
        self.completed_handoffs.append(handoff)

        self._emit_event("handoff_completed", handoff, {"success": success})
        return True

    def reject_handoff(self, handoff_id: str, reason: str) -> bool:
        """Reject a handoff request"""
        handoff = self.active_handoffs.get(handoff_id)
        if not handoff:
            return False

        handoff.status = HandoffStatus.REJECTED
        handoff.error = reason
        handoff.completed_at = datetime.utcnow()

        del self.active_handoffs[handoff_id]
        self.completed_handoffs.append(handoff)

        self._emit_event("handoff_rejected", handoff, {"reason": reason})
        return True

    def escalate_to_human(
        self,
        from_agent: str,
        task_description: str,
        context: dict[str, Any],
        trace_context: dict[str, str],
        reason: str,
        urgency: str = "normal",
    ) -> HandoffRequest:
        """Create an escalation to human review"""

        handoff = self.create_handoff(
            handoff_type=HandoffType.ESCALATION,
            from_agent=from_agent,
            to_agent="human",
            task_description=task_description,
            task_state=context,
            trace_context=trace_context,
            reason=reason,
            trust_transfer=TrustTransfer(
                inherit_permissions=False, verify_context=True, can_delegate_further=True
            ),
        )

        # Add urgency metadata
        handoff.context.metadata["urgency"] = urgency
        handoff.context.metadata["escalation_type"] = "human_review"

        # Trigger callback if registered
        if self.human_approval_callback:
            self.human_approval_callback(handoff)

        return handoff

    def delegate(
        self,
        from_agent: str,
        to_agent: str,
        task_description: str,
        task_state: dict[str, Any],
        trace_context: dict[str, str],
        reason: str = "specialized expertise required",
    ) -> HandoffRequest:
        """Create a delegation handoff (parent-child)"""

        return self.create_handoff(
            handoff_type=HandoffType.DELEGATION,
            from_agent=from_agent,
            to_agent=to_agent,
            task_description=task_description,
            task_state=task_state,
            trace_context=trace_context,
            reason=reason,
            trust_transfer=TrustTransfer(
                inherit_permissions=True,
                trust_level=0.8,
                can_delegate_further=True,
                max_delegation_depth=3,
            ),
        )

    def collaborate(
        self,
        from_agent: str,
        to_agent: str,
        shared_context: dict[str, Any],
        trace_context: dict[str, str],
        collaboration_goal: str,
    ) -> HandoffRequest:
        """Create a collaboration handoff (peer-to-peer)"""

        return self.create_handoff(
            handoff_type=HandoffType.COLLABORATION,
            from_agent=from_agent,
            to_agent=to_agent,
            task_description=collaboration_goal,
            task_state=shared_context,
            trace_context=trace_context,
            reason="peer collaboration",
            trust_transfer=TrustTransfer(
                inherit_permissions=False, trust_level=0.7, can_delegate_further=False
            ),
        )

    def succeed(
        self,
        from_agent: str,
        to_agent: str,
        complete_state: dict[str, Any],
        trace_context: dict[str, str],
        reason: str = "handoff of full responsibility",
    ) -> HandoffRequest:
        """Create a succession handoff (complete transfer)"""

        return self.create_handoff(
            handoff_type=HandoffType.SUCCESSION,
            from_agent=from_agent,
            to_agent=to_agent,
            task_description="Full task succession",
            task_state=complete_state,
            trace_context=trace_context,
            reason=reason,
            trust_transfer=TrustTransfer(
                inherit_permissions=True, trust_level=0.9, can_delegate_further=True
            ),
        )

    def get_handoff(self, handoff_id: str) -> HandoffRequest | None:
        """Get handoff by ID"""
        return self.active_handoffs.get(handoff_id) or next(
            (h for h in self.completed_handoffs if h.handoff_id == handoff_id), None
        )

    def get_active_for_agent(self, agent_id: str) -> list[HandoffRequest]:
        """Get active handoffs involving an agent"""
        return [
            h
            for h in self.active_handoffs.values()
            if h.from_agent == agent_id or h.to_agent == agent_id
        ]

    def get_trace_context(self, handoff_id: str) -> dict[str, str] | None:
        """Get trace context for continuity"""
        handoff = self.get_handoff(handoff_id)
        if handoff:
            return handoff.context.trace_context
        return None

    def set_human_approval_callback(self, callback: Callable):
        """Set callback for human escalations"""
        self.human_approval_callback = callback

    def get_audit_trail(self, handoff_id: str = None) -> list[HandoffEvent]:
        """Get audit trail of events"""
        if handoff_id:
            return [e for e in self.events if e.handoff_id == handoff_id]
        return self.events

    def get_accountability_chain(self, handoff_id: str) -> list[dict]:
        """Get the chain of accountability for a handoff"""
        handoff = self.get_handoff(handoff_id)
        if not handoff:
            return []

        chain = [
            {
                "agent": handoff.from_agent,
                "role": "initiator",
                "timestamp": handoff.created_at.isoformat(),
            }
        ]

        if handoff.accountability.approver:
            chain.append(
                {
                    "agent": handoff.accountability.approver,
                    "role": "approver",
                    "timestamp": handoff.accepted_at.isoformat() if handoff.accepted_at else None,
                }
            )

        chain.append(
            {
                "agent": handoff.to_agent,
                "role": "executor" if handoff.to_agent != "human" else "human_reviewer",
                "timestamp": handoff.completed_at.isoformat() if handoff.completed_at else None,
            }
        )

        return chain

    def _emit_event(self, event_type: str, handoff: HandoffRequest, details: dict = None):
        """Emit a handoff event"""
        event = HandoffEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            handoff_id=handoff.handoff_id,
            from_agent=handoff.from_agent,
            to_agent=handoff.to_agent,
            details=details or {},
        )
        self.events.append(event)


# Convenience functions


def create_delegation(
    manager: HandoffManager,
    from_agent: str,
    to_agent: str,
    task: str,
    context: dict,
    trace_id: str,
    span_id: str,
) -> str:
    """Quick delegation helper"""
    handoff = manager.delegate(
        from_agent=from_agent,
        to_agent=to_agent,
        task_description=task,
        task_state=context,
        trace_context={"trace_id": trace_id, "span_id": span_id},
    )
    return handoff.handoff_id


def escalate_to_human(
    manager: HandoffManager, agent_id: str, reason: str, context: dict, trace_id: str
) -> str:
    """Quick human escalation helper"""
    handoff = manager.escalate_to_human(
        from_agent=agent_id,
        task_description=f"Human review required: {reason}",
        context=context,
        trace_context={"trace_id": trace_id, "span_id": f"span-{uuid4().hex[:8]}"},
        reason=reason,
        urgency="high",
    )
    return handoff.handoff_id
