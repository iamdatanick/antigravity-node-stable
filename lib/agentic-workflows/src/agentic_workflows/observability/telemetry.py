"""
Agent Telemetry and Observability

Provides structured logging, metrics, tracing, and cost tracking
for agentic workflows.
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4


class EventType(Enum):
    """Types of observable events"""
    # Agent lifecycle
    AGENT_START = "agent.start"
    AGENT_COMPLETE = "agent.complete"
    AGENT_ERROR = "agent.error"

    # Tool usage
    TOOL_START = "tool.start"
    TOOL_COMPLETE = "tool.complete"
    TOOL_ERROR = "tool.error"

    # Decision tracking
    DECISION_MADE = "decision.made"
    DELEGATION = "delegation"
    HANDOFF = "handoff"

    # Context operations
    CONTEXT_READ = "context.read"
    CONTEXT_WRITE = "context.write"

    # Security events
    ACCESS_DENIED = "security.access_denied"
    APPROVAL_REQUIRED = "security.approval_required"
    INJECTION_DETECTED = "security.injection_detected"

    # Cost events
    TOKENS_USED = "cost.tokens"
    BUDGET_WARNING = "cost.warning"
    BUDGET_EXCEEDED = "cost.exceeded"

    # Checkpoints
    CHECKPOINT = "checkpoint"
    HUMAN_REVIEW = "human_review"


@dataclass
class TelemetryEvent:
    """A single telemetry event"""
    event_type: EventType
    timestamp: datetime
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    agent_id: str
    data: Dict[str, Any]
    duration_ms: Optional[float] = None
    status: str = "ok"  # ok, error, warning


@dataclass
class Span:
    """A traced operation span"""
    span_id: str
    parent_span_id: Optional[str]
    trace_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[TelemetryEvent] = field(default_factory=list)
    status: str = "running"

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return None


class TelemetryCollector:
    """
    Collects and manages telemetry data for agent workflows.

    Features:
    - Distributed tracing with trace/span IDs
    - Structured event logging
    - Cost tracking
    - Export to various formats
    """

    def __init__(self, agent_id: str, db_path: str = None):
        self.agent_id = agent_id
        self.trace_id = f"trace-{uuid4().hex[:16]}"
        self.events: List[TelemetryEvent] = []
        self.spans: Dict[str, Span] = {}
        self.current_span_id: Optional[str] = None
        self.observers: List[Callable[[TelemetryEvent], None]] = []

        # Cost tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        self.token_prices = {
            "input": 0.003 / 1000,   # $3 per 1M input tokens
            "output": 0.015 / 1000,  # $15 per 1M output tokens
        }

        # Optional SQLite persistence
        self.db_path = db_path
        if db_path:
            self._init_db()

    def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any] = None,
        status: str = "ok"
    ) -> TelemetryEvent:
        """Emit a telemetry event"""
        span_id = self.current_span_id or f"span-{uuid4().hex[:12]}"

        event = TelemetryEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=self._get_parent_span_id(span_id),
            agent_id=self.agent_id,
            data=data or {},
            status=status
        )

        self.events.append(event)

        # Persist if DB enabled
        if self.db_path:
            self._persist_event(event)

        # Notify observers
        for observer in self.observers:
            try:
                observer(event)
            except Exception:
                pass

        return event

    @contextmanager
    def span(self, name: str, attributes: Dict[str, Any] = None):
        """Create a traced span for an operation"""
        span_id = f"span-{uuid4().hex[:12]}"
        parent_span_id = self.current_span_id

        span = Span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            trace_id=self.trace_id,
            name=name,
            start_time=datetime.utcnow(),
            attributes=attributes or {}
        )

        self.spans[span_id] = span
        prev_span_id = self.current_span_id
        self.current_span_id = span_id

        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            span.end_time = datetime.utcnow()
            self.current_span_id = prev_span_id

            # Persist if DB enabled
            if self.db_path:
                self._persist_span(span)

    def record_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "unknown"
    ):
        """Record token usage and calculate cost"""
        self.total_tokens += input_tokens + output_tokens

        input_cost = input_tokens * self.token_prices["input"]
        output_cost = output_tokens * self.token_prices["output"]
        total_cost = input_cost + output_cost
        self.total_cost += total_cost

        self.emit(EventType.TOKENS_USED, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "cost": total_cost,
            "cumulative_cost": self.total_cost
        })

    def check_budget(self, budget: float) -> bool:
        """Check if within budget and emit warnings"""
        if self.total_cost > budget:
            self.emit(EventType.BUDGET_EXCEEDED, {
                "budget": budget,
                "current_cost": self.total_cost
            }, status="error")
            return False

        if self.total_cost > budget * 0.8:
            self.emit(EventType.BUDGET_WARNING, {
                "budget": budget,
                "current_cost": self.total_cost,
                "percentage": self.total_cost / budget * 100
            }, status="warning")

        return True

    def add_observer(self, callback: Callable[[TelemetryEvent], None]):
        """Add an observer for real-time event monitoring"""
        self.observers.append(callback)

    def get_trace(self) -> Dict:
        """Get the complete trace data"""
        return {
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "spans": [self._span_to_dict(s) for s in self.spans.values()],
            "events": [self._event_to_dict(e) for e in self.events],
            "summary": self.get_summary()
        }

    def get_summary(self) -> Dict:
        """Get a summary of the telemetry data"""
        error_events = [e for e in self.events if e.status == "error"]
        tool_events = [e for e in self.events if e.event_type in [EventType.TOOL_START, EventType.TOOL_COMPLETE]]

        return {
            "total_events": len(self.events),
            "total_spans": len(self.spans),
            "error_count": len(error_events),
            "tool_calls": len([e for e in tool_events if e.event_type == EventType.TOOL_START]),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "duration_ms": self._calculate_total_duration()
        }

    def export_json(self, filepath: str):
        """Export trace to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_trace(), f, indent=2, default=str)

    def export_jsonl(self, filepath: str):
        """Export events to JSON Lines format"""
        with open(filepath, 'w') as f:
            for event in self.events:
                f.write(json.dumps(self._event_to_dict(event), default=str) + '\n')

    def _get_parent_span_id(self, span_id: str) -> Optional[str]:
        """Get parent span ID if exists"""
        span = self.spans.get(span_id)
        if span:
            return span.parent_span_id
        return None

    def _calculate_total_duration(self) -> Optional[float]:
        """Calculate total trace duration"""
        if not self.events:
            return None
        first = min(e.timestamp for e in self.events)
        last = max(e.timestamp for e in self.events)
        return (last - first).total_seconds() * 1000

    def _event_to_dict(self, event: TelemetryEvent) -> Dict:
        """Convert event to dictionary"""
        return {
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "trace_id": event.trace_id,
            "span_id": event.span_id,
            "parent_span_id": event.parent_span_id,
            "agent_id": event.agent_id,
            "data": event.data,
            "duration_ms": event.duration_ms,
            "status": event.status
        }

    def _span_to_dict(self, span: Span) -> Dict:
        """Convert span to dictionary"""
        return {
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "trace_id": span.trace_id,
            "name": span.name,
            "start_time": span.start_time.isoformat(),
            "end_time": span.end_time.isoformat() if span.end_time else None,
            "duration_ms": span.duration_ms,
            "attributes": span.attributes,
            "status": span.status
        }

    def _init_db(self):
        """Initialize SQLite database for persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                timestamp TEXT,
                trace_id TEXT,
                span_id TEXT,
                parent_span_id TEXT,
                agent_id TEXT,
                data TEXT,
                status TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                parent_span_id TEXT,
                trace_id TEXT,
                name TEXT,
                start_time TEXT,
                end_time TEXT,
                attributes TEXT,
                status TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cost_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                agent_id TEXT,
                trace_id TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trace_id ON events(trace_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_id ON events(agent_id)')

        conn.commit()
        conn.close()

    def _persist_event(self, event: TelemetryEvent):
        """Persist event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO events (event_type, timestamp, trace_id, span_id, parent_span_id, agent_id, data, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_type.value,
            event.timestamp.isoformat(),
            event.trace_id,
            event.span_id,
            event.parent_span_id,
            event.agent_id,
            json.dumps(event.data),
            event.status
        ))

        conn.commit()
        conn.close()

    def _persist_span(self, span: Span):
        """Persist span to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO spans (span_id, parent_span_id, trace_id, name, start_time, end_time, attributes, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            span.span_id,
            span.parent_span_id,
            span.trace_id,
            span.name,
            span.start_time.isoformat(),
            span.end_time.isoformat() if span.end_time else None,
            json.dumps(span.attributes),
            span.status
        ))

        conn.commit()
        conn.close()


class CostTracker:
    """
    Dedicated cost tracking for agent workflows.

    Tracks token usage across models and provides budget enforcement.
    """

    # Model pricing (as of late 2024/2025, prices in USD per token)
    MODEL_PRICING = {
        "claude-3-opus": {"input": 15.0 / 1_000_000, "output": 75.0 / 1_000_000},
        "claude-3-sonnet": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
        "claude-3-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
        "claude-opus-4": {"input": 15.0 / 1_000_000, "output": 75.0 / 1_000_000},
        "claude-sonnet-4": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
        "gpt-4-turbo": {"input": 10.0 / 1_000_000, "output": 30.0 / 1_000_000},
        "gpt-4o": {"input": 5.0 / 1_000_000, "output": 15.0 / 1_000_000},
        "default": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    }

    def __init__(self, budget: float = 100.0, db_path: str = None):
        self.budget = budget
        self.usage: List[Dict] = []
        self.total_cost = 0.0
        self.db_path = db_path

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent_id: str = None,
        metadata: Dict = None
    ) -> Dict:
        """Record token usage and return cost info"""
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["default"])

        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        total_cost = input_cost + output_cost

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "agent_id": agent_id,
            "metadata": metadata or {}
        }

        self.usage.append(record)
        self.total_cost += total_cost

        return {
            "cost": total_cost,
            "cumulative_cost": self.total_cost,
            "budget_remaining": self.budget - self.total_cost,
            "budget_used_pct": (self.total_cost / self.budget) * 100
        }

    def check_budget(self) -> Tuple[bool, str]:
        """Check if within budget"""
        if self.total_cost > self.budget:
            return False, f"Budget exceeded: ${self.total_cost:.4f} / ${self.budget:.2f}"

        if self.total_cost > self.budget * 0.9:
            return True, f"Warning: 90%+ budget used: ${self.total_cost:.4f} / ${self.budget:.2f}"

        if self.total_cost > self.budget * 0.75:
            return True, f"Notice: 75%+ budget used: ${self.total_cost:.4f} / ${self.budget:.2f}"

        return True, f"Budget OK: ${self.total_cost:.4f} / ${self.budget:.2f}"

    def estimate_cost(self, model: str, estimated_tokens: int) -> float:
        """Estimate cost for a planned operation"""
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["default"])
        # Assume 70/30 split input/output for estimation
        input_tokens = int(estimated_tokens * 0.7)
        output_tokens = int(estimated_tokens * 0.3)

        return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])

    def get_report(self) -> Dict:
        """Get a cost report"""
        by_model = {}
        by_agent = {}

        for record in self.usage:
            model = record["model"]
            agent = record.get("agent_id", "unknown")

            if model not in by_model:
                by_model[model] = {"tokens": 0, "cost": 0.0}
            by_model[model]["tokens"] += record["input_tokens"] + record["output_tokens"]
            by_model[model]["cost"] += record["total_cost"]

            if agent not in by_agent:
                by_agent[agent] = {"tokens": 0, "cost": 0.0}
            by_agent[agent]["tokens"] += record["input_tokens"] + record["output_tokens"]
            by_agent[agent]["cost"] += record["total_cost"]

        return {
            "budget": self.budget,
            "total_cost": self.total_cost,
            "remaining": self.budget - self.total_cost,
            "usage_count": len(self.usage),
            "by_model": by_model,
            "by_agent": by_agent
        }


# Import for type hints
from typing import Tuple
