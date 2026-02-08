"""Intel SuperBuilder gRPC middleware — maps Intel GUI requests to Goose agents."""

import asyncio
import logging
import os
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

logger = logging.getLogger("antigravity.grpc")

GRPC_PORT = int(os.environ.get("GRPC_PORT", "8081"))


class SuperBuilderServicer:
    """gRPC service for Intel SuperBuilder integration.

    NOTE: Proto-generated servicer registration is pending.
    Once .proto files are compiled, uncomment the registration in serve_grpc()
    and inherit from the generated servicer base class.
    """

    def ExecuteWorkflow(self, request, context):  # noqa: N802
        """Maps Intel GUI requests to Goose agent execution."""
        tenant_id = dict(context.invocation_metadata()).get("x-tenant-id", "default")
        logger.info(f"gRPC ExecuteWorkflow: tenant={tenant_id}")

        workflow_name = getattr(request, "workflow_name", "default")
        params = {}

        # Run async workflow submission in sync context
        # Use asyncio.run() which handles event loop creation/cleanup automatically
        from workflows.workflow_defs import submit_workflow

        try:
            run_id = asyncio.run(submit_workflow(workflow_name, params))
        except RuntimeError as e:
            # If there's already a running event loop (shouldn't happen in ThreadPoolExecutor)
            # fall back to creating a new loop
            logger.warning(f"asyncio.run() failed, using new event loop: {e}")
            loop = asyncio.new_event_loop()
            try:
                run_id = loop.run_until_complete(submit_workflow(workflow_name, params))
            finally:
                loop.close()

        logger.info(f"gRPC workflow submitted: {run_id}")
        # TODO: Return proto-generated response once stubs are available
        # return superbuilder_pb2.WorkflowResponse(status="submitted", agent_id=run_id)
        # Until proto stubs exist, abort with UNIMPLEMENTED instead of returning None,
        # which would cause a serialization error for a unary RPC.
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            f"ExecuteWorkflow response type not yet implemented (workflow submitted with run_id={run_id})",
        )


def serve_grpc():
    """Start the gRPC server on the configured port."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Health check service
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    # TODO: Register SuperBuilder servicer once proto stubs are compiled:
    # from workflows import superbuilder_pb2_grpc
    # superbuilder_pb2_grpc.add_SuperBuilderServicer_to_server(
    #     SuperBuilderServicer(), server
    # )
    logger.warning(
        "SuperBuilder servicer not registered — proto stubs not yet compiled. gRPC health check is available."
    )

    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    logger.info(f"gRPC server started on port {GRPC_PORT} (health check active)")
    server.wait_for_termination()
