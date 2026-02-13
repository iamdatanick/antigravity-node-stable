"""Intel SuperBuilder gRPC middleware — maps Intel GUI requests to Goose agents."""

import asyncio
import logging
import os
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from workflows import superbuilder_pb2, superbuilder_pb2_grpc

logger = logging.getLogger("antigravity.grpc")

GRPC_PORT = int(os.environ.get("GRPC_PORT", "8081"))


class SuperBuilderServicer(superbuilder_pb2_grpc.SuperBuilderServicer):
    """gRPC service for Intel SuperBuilder integration."""

    def ExecuteWorkflow(self, request, context):  # noqa: N802
        """Maps Intel GUI requests to Goose agent execution."""
        tenant_id = request.tenant_id or dict(context.invocation_metadata()).get("x-tenant-id", "default")
        logger.info(f"gRPC ExecuteWorkflow: tenant={tenant_id}")

        workflow_name = request.workflow_name or "default"
        params = dict(request.parameters)

        # Run async workflow submission in sync context
        # Use asyncio.run() which handles event loop creation/cleanup automatically
        from workflows.workflow_defs import submit_workflow

        try:
            run_id = asyncio.run(submit_workflow(workflow_name, params))
        except RuntimeError as e:
            # If there's already a running event loop (shouldn't happen in ThreadPoolExecutor)
            # fall back to creating a new loop
            logger.warning(f"asyncio.run() failed, using new event loop: {e}", exc_info=True)
            loop = asyncio.new_event_loop()
            try:
                run_id = loop.run_until_complete(submit_workflow(workflow_name, params))
            finally:
                loop.close()

        logger.info(f"gRPC workflow submitted: {run_id}")
        return superbuilder_pb2.WorkflowResponse(status="submitted", run_id=run_id, agent_id=run_id)

    def GetWorkflowStatus(self, request, context):  # noqa: N802
        """Get the status of a submitted workflow."""
        run_id = request.run_id
        logger.info(f"gRPC GetWorkflowStatus: run_id={run_id}")

        from workflows.workflow_defs import get_workflow_status

        try:
            status = asyncio.run(get_workflow_status(run_id))
        except RuntimeError as e:
            # If there's already a running event loop (shouldn't happen in ThreadPoolExecutor)
            # fall back to creating a new loop
            logger.warning(f"asyncio.run() failed, using new event loop: {e}", exc_info=True)
            loop = asyncio.new_event_loop()
            try:
                status = loop.run_until_complete(get_workflow_status(run_id))
            finally:
                loop.close()

        phase = status.get("phase", "Unknown")
        message = f"Workflow {run_id} is in {phase} phase"
        logger.info(f"gRPC workflow status: {run_id} -> {phase}")
        return superbuilder_pb2.StatusResponse(run_id=run_id, phase=phase, message=message)


def serve_grpc():
    """Start the gRPC server on the configured port."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Health check service
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    # Register SuperBuilder servicer
    superbuilder_pb2_grpc.add_SuperBuilderServicer_to_server(SuperBuilderServicer(), server)
    logger.info("SuperBuilder servicer registered successfully")

    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    logger.info(f"gRPC server started on port {GRPC_PORT} (health check active)")
    server.wait_for_termination()
