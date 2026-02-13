"""OpenTelemetry initialization for Antigravity Node v14.1."""

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger("antigravity.telemetry")

OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "antigravity-node")
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "14.1.0")


def init_telemetry():
    """Initialize OpenTelemetry tracing."""
    resource = Resource.create({"service.name": SERVICE_NAME, "service.version": SERVICE_VERSION})
    provider = TracerProvider(resource=resource)

    if OTEL_ENDPOINT:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            otlp_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP exporter configured: {OTEL_ENDPOINT}")
        except Exception as e:
            logger.warning(f"Failed to configure OTLP exporter: {e}. Using console exporter.", exc_info=True)
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        logger.info("No OTEL_EXPORTER_OTLP_ENDPOINT set. Tracing to console.")
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str = "antigravity"):
    """Get a tracer instance."""
    return trace.get_tracer(name)
