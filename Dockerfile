
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libffi-dev curl python3-dev && rm -rf /var/lib/apt/lists/*
COPY deployment/cloud-test/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. workflows/superbuilder.proto || true
HEALTHCHECK --interval=10s --timeout=5s --retries=3 CMD ls /app/workflows/a2a_server.py || exit 1
EXPOSE 8080 8081
CMD ["python", "-m", "workflows.a2a_server"]
