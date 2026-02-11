# pull-models.ps1 — Pull default LLM model into Ollama container
# Idempotent: safe to run multiple times

$ErrorActionPreference = "Stop"
$Container = "antigravity-v13-llm"
$Model = "qwen3:8b"

Write-Host "Pulling $Model into Ollama container ($Container)..." -ForegroundColor Cyan

# Check container is running
$state = docker inspect -f '{{.State.Running}}' $Container 2>$null
if ($state -ne "true") {
    Write-Host "ERROR: Container $Container is not running. Start with: docker compose up -d ollama" -ForegroundColor Red
    exit 1
}

# Pull model (idempotent — Ollama skips if already present)
docker exec $Container ollama pull $Model

if ($LASTEXITCODE -eq 0) {
    Write-Host "Model $Model ready." -ForegroundColor Green
    # Verify
    docker exec $Container ollama list
} else {
    Write-Host "ERROR: Failed to pull $Model" -ForegroundColor Red
    exit 1
}
