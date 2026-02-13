"""Distributed tracing for agent workflows."""

from __future__ import annotations

import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class SpanStatus(Enum):
    """Span status."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """An event within a span."""

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A trace span representing an operation."""

    trace_id: str
    span_id: str
    name: str
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def is_root(self) -> bool:
        """Check if this is a root span."""
        return self.parent_id is None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            attributes=attributes or {},
        ))

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_status(self, status: SpanStatus, error: str | None = None) -> None:
        """Set span status."""
        self.status = status
        if error:
            self.error = error

    def end(self, status: SpanStatus = SpanStatus.OK) -> None:
        """End the span."""
        self.end_time = time.time()
        if self.status == SpanStatus.UNSET:
            self.status = status


@dataclass
class TraceContext:
    """Context for trace propagation."""

    trace_id: str
    span_id: str
    baggage: dict[str, str] = field(default_factory=dict)

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP headers for propagation."""
        return {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-01",
            "baggage": ",".join(f"{k}={v}" for k, v in self.baggage.items()),
        }

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> TraceContext | None:
        """Parse from HTTP headers."""
        traceparent = headers.get("traceparent", "")
        if not traceparent:
            return None

        parts = traceparent.split("-")
        if len(parts) < 3:
            return None

        baggage = {}
        baggage_str = headers.get("baggage", "")
        if baggage_str:
            for item in baggage_str.split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    baggage[k.strip()] = v.strip()

        return cls(
            trace_id=parts[1],
            span_id=parts[2],
            baggage=baggage,
        )


# Context variable for current span
_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)


class AgentTracer:
    """Tracer for agent workflows.

    Features:
    - Distributed tracing
    - OpenTelemetry integration (optional)
    - Span hierarchy
    - Context propagation
    - In-memory span storage
    """

    def __init__(
        self,
        service_name: str = "agentic-workflows",
        enable_otel: bool = False,
        on_span_end: Callable[[Span], None] | None = None,
    ):
        """Initialize tracer.

        Args:
            service_name: Service name for traces.
            enable_otel: Enable OpenTelemetry integration.
            on_span_end: Callback when span ends.
        """
        self.service_name = service_name
        self.on_span_end = on_span_end

        self._spans: dict[str, Span] = {}
        self._traces: dict[str, list[str]] = {}  # trace_id -> [span_ids]
        self._lock = threading.Lock()

        # OpenTelemetry setup
        self._otel_tracer = None
        if enable_otel and OTEL_AVAILABLE:
            self._setup_otel()

    def _setup_otel(self) -> None:
        """Set up OpenTelemetry."""
        provider = TracerProvider()
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self._otel_tracer = trace.get_tracer(self.service_name)

    def _generate_id(self, length: int = 16) -> str:
        """Generate a random ID."""
        return uuid.uuid4().hex[:length]

    def start_span(
        self,
        name: str,
        parent: Span | TraceContext | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span.

        Args:
            name: Span name.
            parent: Parent span or trace context.
            attributes: Initial attributes.

        Returns:
            New span.
        """
        # Determine trace and parent
        if parent is None:
            parent = _current_span.get()

        if isinstance(parent, Span):
            trace_id = parent.trace_id
            parent_id = parent.span_id
        elif isinstance(parent, TraceContext):
            trace_id = parent.trace_id
            parent_id = parent.span_id
        else:
            trace_id = self._generate_id(32)
            parent_id = None

        span_id = self._generate_id(16)

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            name=name,
            parent_id=parent_id,
            attributes=attributes or {},
        )

        # Store span
        with self._lock:
            self._spans[span_id] = span
            if trace_id not in self._traces:
                self._traces[trace_id] = []
            self._traces[trace_id].append(span_id)

        # Set as current
        _current_span.set(span)

        # OpenTelemetry
        if self._otel_tracer:
            self._start_otel_span(span)

        return span

    def _start_otel_span(self, span: Span) -> None:
        """Start corresponding OpenTelemetry span."""
        if not self._otel_tracer:
            return

        otel_span = self._otel_tracer.start_span(span.name)
        for key, value in span.attributes.items():
            otel_span.set_attribute(key, value)

    def end_span(
        self,
        span: Span | None = None,
        status: SpanStatus = SpanStatus.OK,
        error: str | None = None,
    ) -> None:
        """End a span.

        Args:
            span: Span to end (or current span).
            status: Final status.
            error: Error message if failed.
        """
        if span is None:
            span = _current_span.get()

        if span is None:
            return

        span.end_time = time.time()
        span.status = status
        if error:
            span.error = error

        # Callback
        if self.on_span_end:
            self.on_span_end(span)

        # Restore parent as current
        if span.parent_id:
            parent = self._spans.get(span.parent_id)
            _current_span.set(parent)
        else:
            _current_span.set(None)

    def get_current_span(self) -> Span | None:
        """Get current active span."""
        return _current_span.get()

    def get_current_context(self) -> TraceContext | None:
        """Get current trace context for propagation."""
        span = _current_span.get()
        if span is None:
            return None

        return TraceContext(
            trace_id=span.trace_id,
            span_id=span.span_id,
        )

    def get_span(self, span_id: str) -> Span | None:
        """Get span by ID."""
        return self._spans.get(span_id)

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans in a trace."""
        with self._lock:
            span_ids = self._traces.get(trace_id, [])
            return [self._spans[sid] for sid in span_ids if sid in self._spans]

    def get_trace_tree(self, trace_id: str) -> dict[str, Any]:
        """Get trace as a tree structure."""
        spans = self.get_trace(trace_id)
        if not spans:
            return {}

        # Build tree
        span_map = {s.span_id: s for s in spans}
        children: dict[str | None, list[Span]] = {None: []}

        for span in spans:
            if span.parent_id not in children:
                children[span.parent_id] = []
            children[span.parent_id].append(span)

        def build_node(span: Span) -> dict[str, Any]:
            return {
                "span_id": span.span_id,
                "name": span.name,
                "duration_ms": span.duration_ms,
                "status": span.status.value,
                "error": span.error,
                "attributes": span.attributes,
                "events": [
                    {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                    for e in span.events
                ],
                "children": [
                    build_node(child)
                    for child in children.get(span.span_id, [])
                ],
            }

        # Find root spans
        roots = children.get(None, [])
        return {
            "trace_id": trace_id,
            "spans_count": len(spans),
            "roots": [build_node(root) for root in roots],
        }

    def iterate_spans(
        self,
        trace_id: str | None = None,
        status: SpanStatus | None = None,
    ) -> Iterator[Span]:
        """Iterate over spans with optional filtering.

        Args:
            trace_id: Filter by trace ID.
            status: Filter by status.

        Yields:
            Matching spans.
        """
        with self._lock:
            for span in self._spans.values():
                if trace_id and span.trace_id != trace_id:
                    continue
                if status and span.status != status:
                    continue
                yield span

    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ):
        """Context manager for spans.

        Usage:
            with tracer.span("operation") as span:
                span.set_attribute("key", "value")
                # do work
        """
        return SpanContextManager(self, name, attributes)

    def get_stats(self) -> dict[str, Any]:
        """Get tracer statistics."""
        with self._lock:
            total_spans = len(self._spans)
            completed = sum(1 for s in self._spans.values() if s.end_time)
            errors = sum(1 for s in self._spans.values() if s.status == SpanStatus.ERROR)

            durations = [
                s.duration_ms for s in self._spans.values()
                if s.duration_ms is not None
            ]

            return {
                "total_traces": len(self._traces),
                "total_spans": total_spans,
                "completed_spans": completed,
                "error_spans": errors,
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0,
            }

    def clear(self) -> None:
        """Clear all traces and spans."""
        with self._lock:
            self._spans.clear()
            self._traces.clear()


class SpanContextManager:
    """Context manager for spans."""

    def __init__(
        self,
        tracer: AgentTracer,
        name: str,
        attributes: dict[str, Any] | None = None,
    ):
        self.tracer = tracer
        self.name = name
        self.attributes = attributes
        self.span: Span | None = None

    def __enter__(self) -> Span:
        self.span = self.tracer.start_span(self.name, attributes=self.attributes)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            self.tracer.end_span(
                self.span,
                status=SpanStatus.ERROR,
                error=str(exc_val) if exc_val else None,
            )
        else:
            self.tracer.end_span(self.span, status=SpanStatus.OK)


# Global tracer instance
_default_tracer: AgentTracer | None = None


def get_tracer() -> AgentTracer:
    """Get or create default tracer."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = AgentTracer()
    return _default_tracer


def set_tracer(tracer: AgentTracer) -> None:
    """Set default tracer."""
    global _default_tracer
    _default_tracer = tracer
