"""Intel SuperBuilder gRPC middleware — maps Intel GUI requests to Goose agents."""

import os
import logging
from concurrent import futures

import grpc

logger = logging.getLogger("antigravity.grpc")

GRPC_PORT = int(os.environ.get("GRPC_PORT", "8081"))


# Placeholder imports — these are generated from proto compilation
# from workflows import superbuilder_pb2, superbuilder_pb2_grpc


class SuperBuilderServicer:
    """gRPC service for Intel SuperBuilder integration."""

    async def ExecuteWorkflow(self, request, context):
        """Maps Intel GUI requests to Goose agent execution."""
        tenant_id = dict(context.invocation_metadata()).get("x-tenant-id", "default")
        logger.info(f"gRPC ExecuteWorkflow: tenant={tenant_id}")

        # Extract from request
        workflow_name = getattr(request, "workflow_name", "default")
        params = {}

        # Dispatch to workflow engine
        from workflows.workflow_defs import submit_workflow
        run_id = await submit_workflow(workflow_name, params)

        logger.info(f"gRPC workflow submitted: {run_id}")
        # Return response (proto-generated class)
        # return superbuilder_pb2.WorkflowResponse(status="submitted", agent_id=run_id)
        return {"status": "submitted", "agent_id": run_id}


def serve_grpc():
    """Start the gRPC server on port 8081."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Register servicer
    # superbuilder_pb2_grpc.add_SuperBuilderServicer_to_server(
    #     SuperBuilderServicer(), server
    # )

    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    logger.info(f"gRPC server started on port {GRPC_PORT}")
    server.wait_for_termination()
