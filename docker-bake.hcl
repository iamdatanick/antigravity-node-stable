group "default" {
  targets = ["orchestrator", "mcp-starrocks", "trace-viewer", "master-ui"]
}

target "orchestrator" {
  context    = "."
  dockerfile = "Dockerfile"
  tags       = ["docker.io/antigravity/antigravity-brain:v13.0"]
  platforms  = ["linux/amd64", "linux/arm64"]
  cache-from = ["type=registry,ref=docker.io/antigravity/antigravity-brain:cache"]
  cache-to   = ["type=registry,ref=docker.io/antigravity/antigravity-brain:cache,mode=max"]
}

target "mcp-starrocks" {
  context    = "./src/mcp-starrocks"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/antigravity/mcp-starrocks:v13.0"]
  platforms  = ["linux/amd64"]
}

target "trace-viewer" {
  context    = "./src/trace-viewer"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/antigravity/trace-viewer:v13.0"]
  platforms  = ["linux/amd64"]
}

target "master-ui" {
  context    = "./src/master-ui"
  dockerfile = "Dockerfile"
  tags       = ["docker.io/antigravity/antigravity-ui:v13.0"]
  platforms  = ["linux/amd64"]
}
