group "default" {
  targets = ["orchestrator", "mcp-starrocks", "mcp-filesystem", "trace-viewer", "master-ui", "budget-proxy", "goose-ui"]
}

target "orchestrator" {
  context    = "."
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/antigravity-brain:v13.0", "docker.io/centillionai/antigravity-brain:latest"]
  platforms  = ["linux/amd64", "linux/arm64"]
  cache-from = ["type=registry,ref=docker.io/centillionai/antigravity-brain:cache"]
  cache-to   = ["type=registry,ref=docker.io/centillionai/antigravity-brain:cache,mode=max"]
}

target "mcp-starrocks" {
  context    = "./src/mcp-starrocks"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/mcp-starrocks:v13.0", "docker.io/centillionai/mcp-starrocks:latest"]
  platforms  = ["linux/amd64"]
}

target "mcp-filesystem" {
  context    = "./src/mcp-filesystem"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/mcp-filesystem:v13.0", "docker.io/centillionai/mcp-filesystem:latest"]
  platforms  = ["linux/amd64"]
}

target "trace-viewer" {
  context    = "./src/trace-viewer"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/trace-viewer:v13.0", "docker.io/centillionai/trace-viewer:latest"]
  platforms  = ["linux/amd64"]
}

target "master-ui" {
  context    = "./src/master-ui"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/antigravity-ui:v13.0", "docker.io/centillionai/antigravity-ui:latest"]
  platforms  = ["linux/amd64"]
}

target "budget-proxy" {
  context    = "./src/budget-proxy"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/budget-proxy:v13.0", "docker.io/centillionai/budget-proxy:latest"]
  platforms  = ["linux/amd64"]
}

target "goose-ui" {
  context    = "./src/goose-ui"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/goose-ui:v13.0", "docker.io/centillionai/goose-ui:latest"]
  platforms  = ["linux/amd64"]
}
