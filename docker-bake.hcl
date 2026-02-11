group "default" {
  targets = ["orchestrator", "master-ui", "budget-proxy"]
}

target "orchestrator" {
  context    = "."
  dockerfile = "src/orchestrator/Dockerfile.cloud"
  tags       = [
    "us-central1-docker.pkg.dev/agentic1111/antigravity/orchestrator:v14.1",
    "us-central1-docker.pkg.dev/agentic1111/antigravity/orchestrator:latest"
  ]
  platforms  = ["linux/amd64"]
}

target "master-ui" {
  context    = "."
  dockerfile = "src/master-ui/Dockerfile"
  tags       = [
    "us-central1-docker.pkg.dev/agentic1111/antigravity/ui:v14.1",
    "us-central1-docker.pkg.dev/agentic1111/antigravity/ui:latest"
  ]
  platforms  = ["linux/amd64"]
}

target "budget-proxy" {
  context    = "."
  dockerfile = "src/budget-proxy/Dockerfile"
  tags       = [
    "us-central1-docker.pkg.dev/agentic1111/antigravity/budget-proxy:v14.1",
    "us-central1-docker.pkg.dev/agentic1111/antigravity/budget-proxy:latest"
  ]
  platforms  = ["linux/amd64"]
}
