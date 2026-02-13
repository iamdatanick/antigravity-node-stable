#Requires -Version 5.1
<#
.SYNOPSIS
    Antigravity Node v13 — One-Command Install
.DESCRIPTION
    Sets up the full Antigravity Node stack from scratch.
    Prerequisites: Docker Desktop (running), Git.
.EXAMPLE
    .\install.ps1
    .\install.ps1 -SkipPull
    .\install.ps1 -WithTunnel
#>
param(
    [switch]$SkipPull,
    [switch]$WithTunnel,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$COMPOSE_FILE = Join-Path $PSScriptRoot "docker-compose.yml"
$ENV_EXAMPLE  = Join-Path $PSScriptRoot ".env.example"
$ENV_FILE     = Join-Path $PSScriptRoot ".env"

function Write-Step($num, $msg) { Write-Host "`n[$num] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)        { Write-Host "  [+] $msg" -ForegroundColor Green }
function Write-Warn($msg)      { Write-Host "  [!] $msg" -ForegroundColor Yellow }
function Write-Fail($msg)      { Write-Host "  [X] $msg" -ForegroundColor Red }

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

Write-Host "=" * 60
Write-Host "  Antigravity Node v13 — Installer"
Write-Host "=" * 60

# --- 1. Prerequisites ---
Write-Step 1 "Checking prerequisites..."

# Docker
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Docker not running" }
    Write-Ok "Docker Desktop: running"
} catch {
    Write-Fail "Docker Desktop is not running. Start it and try again."
    exit 1
}

# Git
try {
    $null = git --version 2>&1
    Write-Ok "Git: installed"
} catch {
    Write-Fail "Git is not installed. Install Git and try again."
    exit 1
}

# --- 2. Environment ---
Write-Step 2 "Setting up environment..."

if (-not (Test-Path $ENV_FILE)) {
    if (Test-Path $ENV_EXAMPLE) {
        Copy-Item $ENV_EXAMPLE $ENV_FILE
        Write-Ok "Created .env from .env.example"
        Write-Warn "Edit .env with your API keys before production use."
    } else {
        Write-Fail "No .env.example found. Cannot continue."
        exit 1
    }
} else {
    Write-Ok ".env already exists"
}

# --- 3. Data Directories ---
Write-Step 3 "Creating data directories..."

$dirs = @("data/etcd", "models", "config")
foreach ($d in $dirs) {
    $fullPath = Join-Path $PSScriptRoot $d
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Ok "Created $d/"
    }
}

# Ensure empty model config for OVMS if missing
$modelConfig = Join-Path $PSScriptRoot "models/model_config.json"
if (-not (Test-Path $modelConfig)) {
    '{"model_config_list":[]}' | Set-Content $modelConfig -Encoding UTF8
    Write-Ok "Created models/model_config.json (empty)"
}

# --- 4. Pull Images ---
if (-not $SkipPull) {
    Write-Step 4 "Pulling Docker images (this may take a few minutes)..."
    docker compose -f $COMPOSE_FILE pull 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "All images pulled"
    } else {
        Write-Warn "Some images failed to pull (build-from-source services). Continuing."
    }
} else {
    Write-Step 4 "Skipping image pull (--SkipPull)"
}

# --- 5. Start L0 Infrastructure First ---
Write-Step 5 "Starting L0 infrastructure (postgres, etcd, seaweedfs, openbao)..."

$l0Services = @("postgres", "etcd", "seaweedfs", "openbao")
docker compose -f $COMPOSE_FILE up -d @l0Services 2>&1 | Out-Null

Write-Host "  Waiting for L0 services to become healthy..." -NoNewline
$maxWait = 120
$elapsed = 0
while ($elapsed -lt $maxWait) {
    Start-Sleep -Seconds 5
    $elapsed += 5
    Write-Host "." -NoNewline

    $healthy = $true
    foreach ($svc in $l0Services) {
        $state = docker inspect --format='{{.State.Health.Status}}' "antigravity-v13-$( switch($svc) { 'postgres' {'db'} 'etcd' {'etcd'} 'seaweedfs' {'s3'} 'openbao' {'secrets'} } )" 2>&1
        if ($state -ne "healthy") { $healthy = $false; break }
    }
    if ($healthy) { break }
}
Write-Host ""

if ($healthy) {
    Write-Ok "L0 infrastructure healthy"
} else {
    Write-Warn "L0 not fully healthy after ${maxWait}s — continuing anyway"
}

# --- 6. Start All Services ---
Write-Step 6 "Starting all services..."

docker compose -f $COMPOSE_FILE up -d --build 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Ok "All services starting"
} else {
    Write-Warn "Some services may have failed to start. Check: docker compose ps"
}

# --- 7. Health Verification ---
Write-Step 7 "Verifying health (retrying up to 5 times)..."

$healthOk = $false
for ($i = 1; $i -le 5; $i++) {
    Start-Sleep -Seconds 10
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:8080/health" -TimeoutSec 5 -ErrorAction Stop
        if ($resp.status -eq "healthy") {
            $healthOk = $true
            Write-Ok "Orchestrator health: HEALTHY"
            break
        } else {
            Write-Warn "Attempt $i/5: status=$($resp.status)"
        }
    } catch {
        Write-Warn "Attempt $i/5: orchestrator not ready yet"
    }
}

if (-not $healthOk) {
    Write-Warn "Orchestrator not fully healthy yet. It may still be starting."
    Write-Host "  Check manually: curl http://localhost:8080/health"
}

# --- 8. Ollama Model ---
Write-Step 8 "Checking Ollama model..."

try {
    $models = docker exec antigravity-v13-llm ollama list 2>&1
    if ($models -match "tinyllama") {
        Write-Ok "tinyllama model already loaded"
    } else {
        Write-Host "  Pulling tinyllama (this takes ~2 minutes)..."
        docker exec antigravity-v13-llm ollama pull tinyllama 2>&1 | Out-Null
        Write-Ok "tinyllama model pulled"
    }
} catch {
    Write-Warn "Could not check Ollama models. Pull manually: docker exec antigravity-v13-llm ollama pull tinyllama"
}

# --- 9. Cloudflare Tunnel (optional) ---
if ($WithTunnel) {
    Write-Step 9 "Starting Cloudflare tunnel..."

    # Check for tunnel token in .env
    $envContent = Get-Content $ENV_FILE -Raw
    if ($envContent -match "CLOUDFLARE_TUNNEL_TOKEN=\S+") {
        docker compose -f $COMPOSE_FILE --profile tunnel up -d cloudflare-tunnel 2>&1 | Out-Null
        Write-Ok "Cloudflare tunnel started"
    } else {
        Write-Warn "No CLOUDFLARE_TUNNEL_TOKEN in .env. Set it first:"
        Write-Host "  1. Go to https://one.dash.cloudflare.com → Networks → Tunnels"
        Write-Host "  2. Create a tunnel and copy the token"
        Write-Host "  3. Add CLOUDFLARE_TUNNEL_TOKEN=<token> to .env"
        Write-Host "  4. Run: docker compose --profile tunnel up -d cloudflare-tunnel"
    }
} else {
    Write-Step 9 "Cloudflare tunnel: skipped (use -WithTunnel to enable)"
}

# --- 10. Summary ---
Write-Host "`n" + ("=" * 60)
Write-Host "  Antigravity Node — Installation Complete"
Write-Host ("=" * 60)
Write-Host ""
Write-Host "  Service Map:" -ForegroundColor Cyan
Write-Host "    Master UI:      http://localhost:1055"
Write-Host "    Orchestrator:   http://localhost:8080/health"
Write-Host "    Budget Proxy:   http://localhost:4055/health"
Write-Host "    Keycloak:       http://localhost:8082"
Write-Host "    Perses:         http://localhost:3055"
Write-Host "    Marquez:        http://localhost:5000"
Write-Host "    OpenSearch:     http://localhost:9200"
Write-Host "    SeaweedFS:      http://localhost:9333"
Write-Host "    OVMS:           http://localhost:8500"
Write-Host "    Ollama:         http://localhost:11434"
Write-Host ""
Write-Host "  Useful Commands:" -ForegroundColor Cyan
Write-Host "    docker compose ps                    # Container status"
Write-Host "    docker compose logs orchestrator     # Orchestrator logs"
Write-Host "    curl http://localhost:8080/health    # Health check"
Write-Host ""

$containerCount = (docker compose -f $COMPOSE_FILE ps --format json 2>$null | ConvertFrom-Json).Count
Write-Host "  Containers running: $containerCount" -ForegroundColor Green
Write-Host ("=" * 60)
