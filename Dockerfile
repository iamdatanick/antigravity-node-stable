FROM python:3.11-slim AS base
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev curl python3-dev git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install vendored agentic-workflows library
COPY lib/agentic-workflows /app/lib/agentic-workflows
RUN pip install --no-cache-dir /app/lib/agentic-workflows

COPY . .

# gRPC proto compilation
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. workflows/protos/superbuilder.proto || true

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

EXPOSE 8080 8081
CMD ["python", "-m", "workflows.main"]
