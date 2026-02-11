from __future__ import annotations

"""OpenLineage integration for Antigravity Node.

Emits lineage events to Marquez for data provenance tracking.
All lineage calls are non-blocking: failures are logged and never
propagate to the caller.

In v14.1 Phoenix, OpenLineage/Marquez is optional. If the openlineage
package is not installed, all functions become safe no-ops.

Environment variables:
    OPENLINEAGE_URL       -- Marquez API base (default: http://marquez:5000)
    OPENLINEAGE_NAMESPACE -- Namespace for jobs/datasets (default: antigravity)
"""

import logging
import os
import uuid
from datetime import UTC, datetime

try:
    from openlineage.client import OpenLineageClient
    from openlineage.client.facet import ErrorMessageRunFacet
    from openlineage.client.run import (
        InputDataset,
        Job,
        OutputDataset,
        Run,
        RunEvent,
        RunState,
    )
    from openlineage.client.transport import HttpConfig, HttpTransport
    _HAS_OPENLINEAGE = True
except ImportError:
    _HAS_OPENLINEAGE = False

from opentelemetry import trace

logger = logging.getLogger("antigravity.lineage")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENLINEAGE_URL = os.environ.get("OPENLINEAGE_URL", "http://marquez:5000")
OPENLINEAGE_NAMESPACE = os.environ.get("OPENLINEAGE_NAMESPACE", "antigravity")
PRODUCER = "https://github.com/iamdatanick/Antigravity-Node"

# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------
tracer = trace.get_tracer("antigravity.lineage")

# ---------------------------------------------------------------------------
# Client (lazy singleton)
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """Return a lazily-initialised OpenLineage HTTP client."""
    global _client
    if _client is None:
        http_config = HttpConfig(
            url=OPENLINEAGE_URL,
            endpoint="api/v1/lineage",
            timeout=5.0,
        )
        transport = HttpTransport(http_config)
        _client = OpenLineageClient(transport=transport)
    return _client


def _now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Core: emit_run_event
# ---------------------------------------------------------------------------
async def emit_run_event(
    job_name: str,
    event_type: RunState,
    run_id: str | None = None,
    inputs: list[dict[str, str]] | None = None,
    outputs: list[dict[str, str]] | None = None,
    error_message: str | None = None,
) -> str | None:
    """Emit a RunEvent to Marquez.

    Returns the run_id on success, None on failure or if openlineage
    is not installed.
    """
    if not _HAS_OPENLINEAGE:
        logger.debug("OpenLineage not installed, skipping lineage for %s", job_name)
        return run_id or str(uuid.uuid4())

    with tracer.start_as_current_span("lineage.emit_run_event") as span:
        span.set_attribute("lineage.job_name", job_name)
        span.set_attribute("lineage.event_type", event_type.value)

        try:
            rid = run_id or str(uuid.uuid4())
            # OpenLineage Run requires a valid UUID.  If the caller passes an
            # arbitrary string (e.g. a session_id or task_id) we derive a
            # deterministic UUID-5 so the same logical ID always maps to the
            # same UUID.
            try:
                uuid.UUID(rid)
            except ValueError:
                rid = str(uuid.uuid5(uuid.NAMESPACE_URL, rid))
            span.set_attribute("lineage.run_id", rid)

            # Build run facets
            run_facets: dict = {}
            if event_type == RunState.FAIL and error_message:
                run_facets["errorMessage"] = ErrorMessageRunFacet(
                    message=error_message,
                    programmingLanguage="python",
                )

            run = Run(runId=rid, facets=run_facets)
            job = Job(namespace=OPENLINEAGE_NAMESPACE, name=job_name)

            input_datasets = [
                InputDataset(
                    namespace=ds.get("namespace", OPENLINEAGE_NAMESPACE),
                    name=ds["name"],
                )
                for ds in (inputs or [])
            ]

            output_datasets = [
                OutputDataset(
                    namespace=ds.get("namespace", OPENLINEAGE_NAMESPACE),
                    name=ds["name"],
                )
                for ds in (outputs or [])
            ]

            event = RunEvent(
                eventType=event_type,
                eventTime=_now_iso(),
                run=run,
                job=job,
                producer=PRODUCER,
                inputs=input_datasets or None,
                outputs=output_datasets or None,
            )

            client = _get_client()
            client.emit(event)

            logger.debug(
                "Lineage event emitted: job=%s type=%s run_id=%s",
                job_name,
                event_type.value,
                rid,
            )
            span.set_attribute("lineage.success", True)
            return rid

        except Exception as exc:
            # Lineage failures must NEVER block workflows
            logger.warning("Lineage emission failed for job=%s: %s", job_name, exc)
            span.set_attribute("lineage.success", False)
            span.set_attribute("lineage.error", str(exc))
            return None


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
async def start_job(
    job_name: str,
    inputs: list[dict[str, str]] | None = None,
) -> str | None:
    """Emit a START event and return the generated run_id."""
    return await emit_run_event(
        job_name=job_name,
        event_type=RunState.START,
        inputs=inputs,
    )


async def complete_job(
    job_name: str,
    run_id: str,
    outputs: list[dict[str, str]] | None = None,
) -> str | None:
    """Emit a COMPLETE event for an existing run."""
    return await emit_run_event(
        job_name=job_name,
        event_type=RunState.COMPLETE,
        run_id=run_id,
        outputs=outputs,
    )


async def fail_job(
    job_name: str,
    run_id: str,
    error_message: str,
) -> str | None:
    """Emit a FAIL event for an existing run."""
    return await emit_run_event(
        job_name=job_name,
        event_type=RunState.FAIL,
        run_id=run_id,
        error_message=error_message,
    )
