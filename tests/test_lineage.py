"""Tests for OpenLineage integration (workflows/lineage.py)."""

import os
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest

# Ensure workflows package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def lineage_env(monkeypatch):
    """Set OpenLineage env vars for all tests."""
    monkeypatch.setenv("OPENLINEAGE_URL", "http://localhost:5000")
    monkeypatch.setenv("OPENLINEAGE_NAMESPACE", "test")


@pytest.fixture(autouse=True)
def reset_client():
    """Reset the module-level singleton client between tests."""
    import workflows.lineage as mod
    mod._client = None
    yield
    mod._client = None


def _mock_client():
    """Create a MagicMock that behaves like OpenLineageClient."""
    client = MagicMock()
    client.emit = MagicMock()
    return client


# ---------------------------------------------------------------------------
# Unit tests: emit_run_event
# ---------------------------------------------------------------------------

class TestEmitRunEvent:
    """Tests for emit_run_event()."""

    @pytest.mark.asyncio
    async def test_emit_start_event(self):
        """Emitting a START event returns a run_id and calls client.emit."""
        mock_client = _mock_client()

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            run_id = await emit_run_event("test.job", RunState.START)

        assert run_id is not None
        # Verify it's a valid UUID
        uuid.UUID(run_id)
        mock_client.emit.assert_called_once()

        # Verify the RunEvent structure
        emitted_event = mock_client.emit.call_args[0][0]
        assert emitted_event.eventType == RunState.START
        assert emitted_event.job.name == "test.job"
        assert emitted_event.job.namespace == "test"
        assert emitted_event.run.runId == run_id

    @pytest.mark.asyncio
    async def test_emit_complete_event_with_outputs(self):
        """Emitting a COMPLETE event with output datasets."""
        mock_client = _mock_client()
        run_id = str(uuid.uuid4())

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            result = await emit_run_event(
                "test.job",
                RunState.COMPLETE,
                run_id=run_id,
                outputs=[{"name": "output_table"}],
            )

        assert result == run_id
        emitted_event = mock_client.emit.call_args[0][0]
        assert emitted_event.eventType == RunState.COMPLETE
        assert emitted_event.run.runId == run_id
        assert len(emitted_event.outputs) == 1
        assert emitted_event.outputs[0].name == "output_table"

    @pytest.mark.asyncio
    async def test_emit_fail_event_with_error(self):
        """Emitting a FAIL event includes error facet."""
        mock_client = _mock_client()
        run_id = str(uuid.uuid4())

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            result = await emit_run_event(
                "test.job",
                RunState.FAIL,
                run_id=run_id,
                error_message="Something went wrong",
            )

        assert result == run_id
        emitted_event = mock_client.emit.call_args[0][0]
        assert emitted_event.eventType == RunState.FAIL
        assert "errorMessage" in emitted_event.run.facets
        assert emitted_event.run.facets["errorMessage"].message == "Something went wrong"
        assert emitted_event.run.facets["errorMessage"].programmingLanguage == "python"

    @pytest.mark.asyncio
    async def test_emit_with_inputs(self):
        """Emitting an event with input datasets."""
        mock_client = _mock_client()

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            result = await emit_run_event(
                "test.job",
                RunState.START,
                inputs=[
                    {"name": "source_table"},
                    {"namespace": "external", "name": "api_feed"},
                ],
            )

        assert result is not None
        emitted_event = mock_client.emit.call_args[0][0]
        assert len(emitted_event.inputs) == 2
        assert emitted_event.inputs[0].name == "source_table"
        assert emitted_event.inputs[0].namespace == "test"
        assert emitted_event.inputs[1].name == "api_feed"
        assert emitted_event.inputs[1].namespace == "external"

    @pytest.mark.asyncio
    async def test_emit_uses_custom_uuid_run_id(self):
        """When a valid UUID run_id is provided, it is used as-is."""
        mock_client = _mock_client()
        custom_id = str(uuid.uuid4())

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            result = await emit_run_event("test.job", RunState.START, run_id=custom_id)

        assert result == custom_id
        emitted_event = mock_client.emit.call_args[0][0]
        assert emitted_event.run.runId == custom_id

    @pytest.mark.asyncio
    async def test_emit_converts_non_uuid_run_id(self):
        """When a non-UUID run_id is provided, it is converted to UUID-5."""
        mock_client = _mock_client()
        non_uuid_id = "my-custom-task-id"
        expected_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, non_uuid_id))

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            result = await emit_run_event("test.job", RunState.START, run_id=non_uuid_id)

        assert result == expected_uuid
        emitted_event = mock_client.emit.call_args[0][0]
        assert emitted_event.run.runId == expected_uuid

    @pytest.mark.asyncio
    async def test_emit_event_has_producer(self):
        """RunEvent includes producer URI."""
        mock_client = _mock_client()

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            await emit_run_event("test.job", RunState.START)

        emitted_event = mock_client.emit.call_args[0][0]
        assert "github.com" in emitted_event.producer


# ---------------------------------------------------------------------------
# Unit tests: failure resilience
# ---------------------------------------------------------------------------

class TestLineageFailureResilience:
    """Lineage failures must never propagate to callers."""

    @pytest.mark.asyncio
    async def test_emit_swallows_client_error(self):
        """If client.emit raises, return None instead of propagating."""
        mock_client = _mock_client()
        mock_client.emit.side_effect = ConnectionError("Marquez is down")

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            result = await emit_run_event("test.job", RunState.START)

        assert result is None

    @pytest.mark.asyncio
    async def test_emit_swallows_transport_error(self):
        """If _get_client raises, return None instead of propagating."""
        with patch("workflows.lineage._get_client", side_effect=RuntimeError("bad config")):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            result = await emit_run_event("test.job", RunState.COMPLETE)

        assert result is None

    @pytest.mark.asyncio
    async def test_start_job_swallows_error(self):
        """start_job returns None on failure without raising."""
        with patch("workflows.lineage._get_client", side_effect=Exception("boom")):
            from workflows.lineage import start_job

            result = await start_job("test.job")

        assert result is None

    @pytest.mark.asyncio
    async def test_complete_job_swallows_error(self):
        """complete_job returns None on failure without raising."""
        with patch("workflows.lineage._get_client", side_effect=Exception("boom")):
            from workflows.lineage import complete_job

            result = await complete_job("test.job", "run-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_fail_job_swallows_error(self):
        """fail_job returns None on failure without raising."""
        with patch("workflows.lineage._get_client", side_effect=Exception("boom")):
            from workflows.lineage import fail_job

            result = await fail_job("test.job", "run-123", "error msg")

        assert result is None


# ---------------------------------------------------------------------------
# Unit tests: convenience helpers
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tests for start_job, complete_job, fail_job helpers."""

    @pytest.mark.asyncio
    async def test_start_job_emits_start(self):
        """start_job emits a START event and returns a run_id."""
        mock_client = _mock_client()

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import start_job
            from openlineage.client.run import RunState

            run_id = await start_job("pipeline.ingest")

        assert run_id is not None
        emitted_event = mock_client.emit.call_args[0][0]
        assert emitted_event.eventType == RunState.START
        assert emitted_event.job.name == "pipeline.ingest"

    @pytest.mark.asyncio
    async def test_start_job_with_inputs(self):
        """start_job forwards input datasets."""
        mock_client = _mock_client()

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import start_job

            run_id = await start_job(
                "pipeline.ingest",
                inputs=[{"name": "raw_data"}],
            )

        assert run_id is not None
        emitted_event = mock_client.emit.call_args[0][0]
        assert len(emitted_event.inputs) == 1
        assert emitted_event.inputs[0].name == "raw_data"

    @pytest.mark.asyncio
    async def test_complete_job_emits_complete(self):
        """complete_job emits a COMPLETE event with the provided run_id."""
        mock_client = _mock_client()
        run_id = str(uuid.uuid4())

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import complete_job
            from openlineage.client.run import RunState

            result = await complete_job(
                "pipeline.ingest",
                run_id,
                outputs=[{"name": "processed_table"}],
            )

        assert result == run_id
        emitted_event = mock_client.emit.call_args[0][0]
        assert emitted_event.eventType == RunState.COMPLETE
        assert emitted_event.run.runId == run_id
        assert len(emitted_event.outputs) == 1

    @pytest.mark.asyncio
    async def test_fail_job_emits_fail(self):
        """fail_job emits a FAIL event with error message."""
        mock_client = _mock_client()
        run_id = str(uuid.uuid4())

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import fail_job
            from openlineage.client.run import RunState

            result = await fail_job("pipeline.ingest", run_id, "OOMKilled")

        assert result == run_id
        emitted_event = mock_client.emit.call_args[0][0]
        assert emitted_event.eventType == RunState.FAIL
        assert emitted_event.run.runId == run_id
        assert "errorMessage" in emitted_event.run.facets
        assert emitted_event.run.facets["errorMessage"].message == "OOMKilled"


# ---------------------------------------------------------------------------
# Unit tests: RunEvent structure correctness
# ---------------------------------------------------------------------------

class TestRunEventStructure:
    """Verify the emitted RunEvent has correct OpenLineage structure."""

    @pytest.mark.asyncio
    async def test_event_has_correct_namespace(self):
        """Job namespace matches OPENLINEAGE_NAMESPACE env var."""
        mock_client = _mock_client()

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            await emit_run_event("test.job", RunState.START)

        emitted_event = mock_client.emit.call_args[0][0]
        assert emitted_event.job.namespace == "test"

    @pytest.mark.asyncio
    async def test_event_has_iso_timestamp(self):
        """eventTime is a valid ISO-8601 string."""
        mock_client = _mock_client()

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            await emit_run_event("test.job", RunState.START)

        from datetime import datetime

        emitted_event = mock_client.emit.call_args[0][0]
        # Should parse without error
        datetime.fromisoformat(emitted_event.eventTime)

    @pytest.mark.asyncio
    async def test_event_has_schema_url(self):
        """RunEvent includes OpenLineage schemaURL."""
        mock_client = _mock_client()

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            await emit_run_event("test.job", RunState.START)

        emitted_event = mock_client.emit.call_args[0][0]
        assert "openlineage.io" in emitted_event.schemaURL

    @pytest.mark.asyncio
    async def test_complete_event_without_outputs(self):
        """COMPLETE event with no outputs has None/empty outputs."""
        mock_client = _mock_client()
        run_id = str(uuid.uuid4())

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            await emit_run_event("test.job", RunState.COMPLETE, run_id=run_id)

        emitted_event = mock_client.emit.call_args[0][0]
        assert emitted_event.outputs is None

    @pytest.mark.asyncio
    async def test_fail_event_without_error_message(self):
        """FAIL event without error_message has no errorMessage facet."""
        mock_client = _mock_client()
        run_id = str(uuid.uuid4())

        with patch("workflows.lineage._get_client", return_value=mock_client):
            from workflows.lineage import emit_run_event
            from openlineage.client.run import RunState

            await emit_run_event("test.job", RunState.FAIL, run_id=run_id)

        emitted_event = mock_client.emit.call_args[0][0]
        assert "errorMessage" not in emitted_event.run.facets
