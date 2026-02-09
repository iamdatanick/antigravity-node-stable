FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    bzip2 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Goose binary (release is a tar.bz2 archive)
ARG GOOSE_VERSION=1.20.1
RUN curl -fsSL "https://github.com/block/goose/releases/download/v${GOOSE_VERSION}/goose-x86_64-unknown-linux-gnu.tar.bz2" \
    | tar xj -C /tmp/ && mv /tmp/goose /usr/local/bin/goose && chmod +x /usr/local/bin/goose

RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy protos and compile gRPC stubs
COPY workflows/protos /app/workflows/protos
RUN python -m grpc_tools.protoc \
    -I/app/workflows/protos \
    --python_out=/app/workflows \
    --grpc_python_out=/app/workflows \
    /app/workflows/protos/*.proto

COPY workflows /app/workflows
COPY well-known /app/well-known
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 8080 8081
CMD ["python", "-u", "/app/workflows/main.py"]
