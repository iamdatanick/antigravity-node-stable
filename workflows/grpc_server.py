"""Intel SuperBuilder gRPC middleware — maps Intel GUI requests to Goose agents."""

import os
import logging
import asyncio
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

    def ExecuteWorkflow(self, request, context):
        """Maps Intel GUI requests to Goose agent execution."""
        tenant_id = dict(context.invocation_metadata()).get("x-tenant-id", "default")
        logger.info(f"gRPC ExecuteWorkflow: tenant={tenant_id}")

        workflow_name = getattr(request, "workflow_name", "default")
        params = {}

        # Run async workflow submission in sync context
        loop = asyncio.new_event_loop()
        try:
            from workflows.workflow_defs import submit_workflow
            run_id = loop.run_until_complete(submit_workflow(workflow_name, params))
        finally:
            loop.close()

        logger.info(f"gRPC workflow submitted: {run_id}")
        # TODO: Return proto-generated response once stubs are available
        # return superbuilder_pb2.WorkflowResponse(status="submitted", agent_id=run_id)
        context.set_code(grpc.StatusCode.OK)
        context.set_details(f"Workflow submitted: {run_id}")
        return None  # Placeholder until proto stubs are generated


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
    logger.warning("SuperBuilder servicer not registered — proto stubs not yet compiled. "
                    "gRPC health check is available.")

    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    logger.info(f"gRPC server started on port {GRPC_PORT} (health check active)")
    server.wait_for_termination()
