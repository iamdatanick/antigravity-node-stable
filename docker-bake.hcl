group "default" {
  targets = ["orchestrator", "mcp-starrocks", "trace-viewer", "master-ui", "budget-proxy"]
}

target "orchestrator" {
  context    = "."
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/antigravity-brain:v13.0"]
  platforms  = ["linux/amd64", "linux/arm64"]
  cache-from = ["type=registry,ref=docker.io/centillionai/antigravity-brain:cache"]
  cache-to   = ["type=registry,ref=docker.io/centillionai/antigravity-brain:cache,mode=max"]
}

target "mcp-starrocks" {
  context    = "./src/mcp-starrocks"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/mcp-starrocks:v13.0"]
  platforms  = ["linux/amd64"]
}

target "trace-viewer" {
  context    = "./src/trace-viewer"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/trace-viewer:v13.0"]
  platforms  = ["linux/amd64"]
}

target "master-ui" {
  context    = "./src/master-ui"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/antigravity-ui:v13.0"]
  platforms  = ["linux/amd64"]
}

target "budget-proxy" {
  context    = "./src/budget-proxy"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/centillionai/budget-proxy:v13.0"]
  platforms  = ["linux/amd64"]
}
