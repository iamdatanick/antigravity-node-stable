"""OpenLineage integration for Antigravity Node v13.0.

Emits lineage events to Marquez for data provenance tracking.
All lineage calls are non-blocking: failures are logged and never
propagate to the caller.

Environment variables:
    OPENLINEAGE_URL       -- Marquez API base (default: http://marquez:5000)
    OPENLINEAGE_NAMESPACE -- Namespace for jobs/datasets (default: antigravity)
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

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
_client: Optional[OpenLineageClient] = None


def _get_client() -> OpenLineageClient:
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
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Core: emit_run_event
# ---------------------------------------------------------------------------
async def emit_run_event(
    job_name: str,
    event_type: RunState,
    run_id: Optional[str] = None,
    inputs: Optional[List[Dict[str, str]]] = None,
    outputs: Optional[List[Dict[str, str]]] = None,
    error_message: Optional[str] = None,
) -> Optional[str]:
    """Emit a RunEvent to Marquez.

    Parameters
    ----------
    job_name : str
        Logical job name (e.g. ``a2a.task``, ``a2a.inference``).
    event_type : RunState
        One of START, COMPLETE, FAIL, ABORT, RUNNING, OTHER.
    run_id : str, optional
        UUID for the run.  Auto-generated if omitted.
    inputs : list of dict, optional
        Input datasets as ``[{"namespace": ..., "name": ...}]``.
    outputs : list of dict, optional
        Output datasets as ``[{"namespace": ..., "name": ...}]``.
    error_message : str, optional
        Error description (only used for FAIL events).

    Returns
    -------
    str or None
        The ``run_id`` on success, ``None`` on failure.
    """
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
            run_facets: Dict = {}
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
                job_name, event_type.value, rid,
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
    inputs: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """Emit a START event and return the generated run_id."""
    return await emit_run_event(
        job_name=job_name,
        event_type=RunState.START,
        inputs=inputs,
    )


async def complete_job(
    job_name: str,
    run_id: str,
    outputs: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
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
) -> Optional[str]:
    """Emit a FAIL event for an existing run."""
    return await emit_run_event(
        job_name=job_name,
        event_type=RunState.FAIL,
        run_id=run_id,
        error_message=error_message,
    )
