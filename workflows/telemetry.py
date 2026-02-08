"""OpenTelemetry initialization for Antigravity Node v13.0."""

import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger("antigravity.telemetry")

OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "antigravity-node")


def init_telemetry():
    """Initialize OpenTelemetry tracing."""
    resource = Resource.create({"service.name": SERVICE_NAME, "service.version": "13.0.0"})
    provider = TracerProvider(resource=resource)

    if OTEL_ENDPOINT:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            otlp_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP exporter configured: {OTEL_ENDPOINT}")
        except Exception as e:
            logger.warning(f"Failed to configure OTLP exporter: {e}. Using console exporter.")
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        logger.info("No OTEL_EXPORTER_OTLP_ENDPOINT set. Tracing to console.")
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str = "antigravity"):
    """Get a tracer instance."""
    return trace.get_tracer(name)
