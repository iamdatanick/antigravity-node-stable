"""Tracing for OpenAI-style agent runs.

This module provides tracing capabilities for agent execution,
integrating with agentic_workflows observability module.

Features:
- Automatic tracing of agent runs
- Span hierarchy for tools and handoffs
- Export to multiple formats
- Integration with OpenTelemetry

Reference: https://github.com/openai/openai-agents-python
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TraceEventType(Enum):
    """Types of trace events."""

    RUN_START = "run_start"
    RUN_END = "run_end"
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    HANDOFF_START = "handoff_start"
    HANDOFF_END = "handoff_end"
    GUARDRAIL_CHECK = "guardrail_check"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class TraceEvent:
    """A single event in a trace.

    Attributes:
        event_type: Type of event.
        timestamp: When event occurred.
        data: Event-specific data.
        span_id: Associated span.
        parent_span_id: Parent span if nested.
    """

    event_type: TraceEventType
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    span_id: str = ""
    parent_span_id: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "duration_ms": self.duration_ms,
        }


@dataclass
class TraceSpan:
    """A span representing a traced operation.

    Spans form a hierarchy representing the execution structure.
    """

    span_id: str
    name: str
    trace_id: str
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: str = "in_progress"  # "in_progress", "success", "error"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def add_event(
        self,
        event_type: TraceEventType,
        data: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Add an event to the span."""
        event = TraceEvent(
            event_type=event_type,
            data=data or {},
            span_id=self.span_id,
            parent_span_id=self.parent_id,
        )
        self.events.append(event)
        return event

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def end(self, status: str = "success", error: str | None = None) -> None:
        """End the span."""
        self.end_time = time.time()
        self.status = status
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_id": self.span_id,
            "name": self.name,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "error": self.error,
        }


@dataclass
class AgentTrace:
    """Complete trace of an agent run.

    Contains all spans and events from a single agent execution.
    """

    trace_id: str
    agent_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    spans: list[TraceSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Aggregated metrics
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_handoffs: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def duration_ms(self) -> float | None:
        """Get total trace duration."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def root_span(self) -> TraceSpan | None:
        """Get the root span."""
        for span in self.spans:
            if span.parent_id is None:
                return span
        return None

    def add_span(self, span: TraceSpan) -> None:
        """Add a span to the trace."""
        self.spans.append(span)

    def get_span(self, span_id: str) -> TraceSpan | None:
        """Get span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_spans_by_name(self, name: str) -> list[TraceSpan]:
        """Get all spans with a given name."""
        return [s for s in self.spans if s.name == name]

    def end(self) -> None:
        """End the trace."""
        self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
            "metrics": {
                "total_llm_calls": self.total_llm_calls,
                "total_tool_calls": self.total_tool_calls,
                "total_handoffs": self.total_handoffs,
                "total_tokens": self.total_tokens,
                "total_cost_usd": self.total_cost_usd,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class AgentTracer:
    """Tracer for agent executions.

    Provides tracing capabilities with integration to agentic_workflows
    observability module.

    Example:
        tracer = AgentTracer()

        with tracer.trace("assistant") as trace:
            with trace.span("llm_call"):
                # LLM call
                pass
            with trace.span("tool_call"):
                # Tool execution
                pass

        print(trace.to_json())
    """

    def __init__(
        self,
        service_name: str = "openai-agents",
        enable_observability: bool = True,
        on_trace_complete: Callable[[AgentTrace], None] | None = None,
    ):
        """Initialize tracer.

        Args:
            service_name: Service name for traces.
            enable_observability: Integrate with observability module.
            on_trace_complete: Callback when trace completes.
        """
        self.service_name = service_name
        self.enable_observability = enable_observability
        self.on_trace_complete = on_trace_complete

        # Storage
        self._traces: dict[str, AgentTrace] = {}
        self._active_trace: AgentTrace | None = None
        self._active_span: TraceSpan | None = None

        # Try to import observability module
        self._obs_tracer = None
        if enable_observability:
            try:
                from agentic_workflows.observability.tracing import get_tracer

                self._obs_tracer = get_tracer()
            except ImportError:
                logger.debug("Observability module not available")

    @contextmanager
    def trace(
        self,
        agent_name: str,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[AgentTrace]:
        """Context manager for tracing an agent run.

        Args:
            agent_name: Name of the agent.
            trace_id: Optional trace ID.
            metadata: Additional metadata.

        Yields:
            AgentTrace instance.
        """
        trace_id = trace_id or str(uuid.uuid4())[:16]

        trace = AgentTrace(
            trace_id=trace_id,
            agent_name=agent_name,
            metadata=metadata or {},
        )

        self._traces[trace_id] = trace
        self._active_trace = trace

        # Create root span
        root_span = TraceSpan(
            span_id=str(uuid.uuid4())[:8],
            name=f"agent_run:{agent_name}",
            trace_id=trace_id,
        )
        trace.add_span(root_span)
        self._active_span = root_span

        # Start observability span if available
        obs_span = None
        if self._obs_tracer:
            obs_span = self._obs_tracer.start_span(
                f"agent_run:{agent_name}",
                attributes={"agent.name": agent_name, **trace.metadata},
            )

        try:
            yield trace
            root_span.end(status="success")
        except Exception as e:
            root_span.end(status="error", error=str(e))
            raise
        finally:
            trace.end()
            self._active_trace = None
            self._active_span = None

            if obs_span and self._obs_tracer:
                self._obs_tracer.end_span(obs_span)

            if self.on_trace_complete:
                self.on_trace_complete(trace)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        """Context manager for a trace span.

        Args:
            name: Span name.
            attributes: Span attributes.

        Yields:
            TraceSpan instance.
        """
        if self._active_trace is None:
            raise RuntimeError("No active trace. Use tracer.trace() first.")

        parent_id = self._active_span.span_id if self._active_span else None

        span = TraceSpan(
            span_id=str(uuid.uuid4())[:8],
            name=name,
            trace_id=self._active_trace.trace_id,
            parent_id=parent_id,
            attributes=attributes or {},
        )

        self._active_trace.add_span(span)
        prev_span = self._active_span
        self._active_span = span

        # Start observability span
        obs_span = None
        if self._obs_tracer:
            obs_span = self._obs_tracer.start_span(name, attributes=attributes)

        try:
            yield span
            span.end(status="success")
        except Exception as e:
            span.end(status="error", error=str(e))
            raise
        finally:
            self._active_span = prev_span

            if obs_span and self._obs_tracer:
                from agentic_workflows.observability.tracing import SpanStatus

                status = SpanStatus.OK if span.status == "success" else SpanStatus.ERROR
                self._obs_tracer.end_span(obs_span, status=status, error=span.error)

    def add_event(
        self,
        event_type: TraceEventType,
        data: dict[str, Any] | None = None,
    ) -> TraceEvent | None:
        """Add an event to the current span.

        Args:
            event_type: Event type.
            data: Event data.

        Returns:
            Created event or None if no active span.
        """
        if self._active_span is None:
            return None

        return self._active_span.add_event(event_type, data)

    def record_llm_call(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        duration_ms: float = 0.0,
    ) -> None:
        """Record an LLM call.

        Args:
            model: Model used.
            input_tokens: Input token count.
            output_tokens: Output token count.
            cost_usd: Cost in USD.
            duration_ms: Call duration.
        """
        if self._active_trace:
            self._active_trace.total_llm_calls += 1
            self._active_trace.total_tokens += input_tokens + output_tokens
            self._active_trace.total_cost_usd += cost_usd

        self.add_event(
            TraceEventType.LLM_CALL_END,
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "duration_ms": duration_ms,
            },
        )

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a tool call.

        Args:
            tool_name: Tool name.
            arguments: Tool arguments.
            result: Tool result.
            error: Error if failed.
            duration_ms: Execution duration.
        """
        if self._active_trace:
            self._active_trace.total_tool_calls += 1

        self.add_event(
            TraceEventType.TOOL_CALL_END,
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": str(result)[:1000] if result else None,
                "error": error,
                "duration_ms": duration_ms,
            },
        )

    def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
        reason: str = "",
    ) -> None:
        """Record an agent handoff.

        Args:
            from_agent: Source agent.
            to_agent: Target agent.
            reason: Handoff reason.
        """
        if self._active_trace:
            self._active_trace.total_handoffs += 1

        self.add_event(
            TraceEventType.HANDOFF_END,
            {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "reason": reason,
            },
        )

    def get_trace(self, trace_id: str) -> AgentTrace | None:
        """Get a trace by ID.

        Args:
            trace_id: Trace ID.

        Returns:
            AgentTrace or None.
        """
        return self._traces.get(trace_id)

    def get_all_traces(self) -> list[AgentTrace]:
        """Get all stored traces."""
        return list(self._traces.values())

    def clear(self) -> None:
        """Clear all stored traces."""
        self._traces.clear()


# Global tracer instance
_default_tracer: AgentTracer | None = None


def get_tracer() -> AgentTracer:
    """Get the default tracer instance."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = AgentTracer()
    return _default_tracer


def set_tracer(tracer: AgentTracer) -> None:
    """Set the default tracer instance."""
    global _default_tracer
    _default_tracer = tracer


# ============================================================================
# Trace Export Functions
# ============================================================================


def export_trace_to_json(trace: AgentTrace) -> str:
    """Export trace to JSON format.

    Args:
        trace: Trace to export.

    Returns:
        JSON string.
    """
    return trace.to_json()


def export_trace_to_otlp(trace: AgentTrace) -> dict[str, Any]:
    """Export trace to OpenTelemetry Protocol format.

    Args:
        trace: Trace to export.

    Returns:
        OTLP-compatible dict.
    """
    spans = []
    for span in trace.spans:
        otlp_span = {
            "traceId": trace.trace_id,
            "spanId": span.span_id,
            "parentSpanId": span.parent_id,
            "name": span.name,
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": int(span.start_time * 1e9),
            "endTimeUnixNano": int((span.end_time or time.time()) * 1e9),
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}} for k, v in span.attributes.items()
            ],
            "status": {
                "code": "STATUS_CODE_OK" if span.status == "success" else "STATUS_CODE_ERROR",
                "message": span.error or "",
            },
            "events": [
                {
                    "name": e.event_type.value,
                    "timeUnixNano": int(e.timestamp * 1e9),
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}} for k, v in e.data.items()
                    ],
                }
                for e in span.events
            ],
        }
        spans.append(otlp_span)

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "openai-agents"}},
                        {"key": "agent.name", "value": {"stringValue": trace.agent_name}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "openai-agents"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def export_trace_to_chrome(trace: AgentTrace) -> list[dict[str, Any]]:
    """Export trace to Chrome Trace Event format.

    Useful for viewing in chrome://tracing.

    Args:
        trace: Trace to export.

    Returns:
        List of Chrome trace events.
    """
    events = []
    pid = 1

    for span in trace.spans:
        # Begin event
        events.append(
            {
                "name": span.name,
                "cat": "agent",
                "ph": "B",  # Begin
                "ts": span.start_time * 1e6,  # Microseconds
                "pid": pid,
                "tid": hash(span.span_id) % 10,
                "args": span.attributes,
            }
        )

        # End event
        end_time = span.end_time or time.time()
        events.append(
            {
                "name": span.name,
                "cat": "agent",
                "ph": "E",  # End
                "ts": end_time * 1e6,
                "pid": pid,
                "tid": hash(span.span_id) % 10,
            }
        )

        # Add span events as instant events
        for event in span.events:
            events.append(
                {
                    "name": event.event_type.value,
                    "cat": "event",
                    "ph": "i",  # Instant
                    "ts": event.timestamp * 1e6,
                    "pid": pid,
                    "tid": hash(span.span_id) % 10,
                    "s": "t",  # Thread scope
                    "args": event.data,
                }
            )

    return events


# ============================================================================
# Tracing Decorators
# ============================================================================


def traced(name: str | None = None):
    """Decorator to trace a function.

    Args:
        name: Span name (defaults to function name).

    Returns:
        Decorated function.

    Example:
        @traced("llm_call")
        async def call_llm(messages):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                if tracer._active_trace:
                    with tracer.span(span_name):
                        return await func(*args, **kwargs)
                else:
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                if tracer._active_trace:
                    with tracer.span(span_name):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator
