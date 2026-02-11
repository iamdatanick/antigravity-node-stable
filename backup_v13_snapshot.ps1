# backup_v13_snapshot.ps1 — Antigravity Node v13.0 Full Snapshot Backup
# Creates restorative backup of all Docker volumes + source code
# Read-only operation — no data is deleted or modified

$ErrorActionPreference = "Stop"

# --- Configuration ---
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupDir = Join-Path $ProjectDir "backups\v13_snapshot_$Timestamp"

$Volumes = @(
    "antigravity-v13_postgres-data",
    "antigravity-v13_seaweedfs-data",
    "antigravity-v13_milvus-data",
    "antigravity-v13_valkey-data",
    "antigravity-v13_openbao-data",
    "antigravity-v13_perses-data",
    "antigravity-v13_opensearch-data"
)

# --- Create backup directory ---
Write-Host "`n=== ANTIGRAVITY NODE v13.0 SNAPSHOT BACKUP ===" -ForegroundColor Cyan
Write-Host "Timestamp : $Timestamp"
Write-Host "Backup dir: $BackupDir`n"

New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

# --- Step 1: Stop services for data consistency ---
Write-Host "[1/4] Stopping services for consistent snapshot..." -ForegroundColor Yellow
Push-Location $ProjectDir
docker compose stop
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: docker compose stop returned non-zero. Continuing anyway." -ForegroundColor Red
}
Pop-Location

# --- Step 2: Backup each Docker volume ---
Write-Host "`n[2/4] Backing up Docker volumes..." -ForegroundColor Yellow

$VolBackupDir = $BackupDir.Replace('\', '/')

foreach ($vol in $Volumes) {
    $tarName = "$vol.tar.gz"
    Write-Host "  Backing up $vol -> $tarName"

    docker run --rm `
        -v "${vol}:/data:ro" `
        -v "${VolBackupDir}:/backup" `
        alpine tar czf "/backup/$tarName" -C /data .

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to backup $vol" -ForegroundColor Red
    } else {
        $size = (Get-Item (Join-Path $BackupDir $tarName)).Length
        $sizeMB = [math]::Round($size / 1MB, 2)
        Write-Host "  OK: $sizeMB MB" -ForegroundColor Green
    }
}

# --- Step 3: Backup source code ---
Write-Host "`n[3/4] Archiving source code..." -ForegroundColor Yellow

# Use tar via docker to create a consistent archive (excludes .git, node_modules, backups)
$ProjectDirDocker = $ProjectDir.Replace('\', '/')
docker run --rm `
    -v "${ProjectDirDocker}:/source:ro" `
    -v "${VolBackupDir}:/backup" `
    alpine sh -c "cd /source && tar czf /backup/source_code.tar.gz --exclude=.git --exclude=node_modules --exclude=backups --exclude=__pycache__ --exclude='*.pyc' ."

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Source code archive failed" -ForegroundColor Red
} else {
    $size = (Get-Item (Join-Path $BackupDir "source_code.tar.gz")).Length
    $sizeMB = [math]::Round($size / 1MB, 2)
    Write-Host "  OK: source_code.tar.gz ($sizeMB MB)" -ForegroundColor Green
}

# --- Step 4: Restart services ---
Write-Host "`n[4/4] Restarting services..." -ForegroundColor Yellow
Push-Location $ProjectDir
docker compose start
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: docker compose start returned non-zero. Check manually." -ForegroundColor Red
} else {
    Write-Host "Services restarted." -ForegroundColor Green
}
Pop-Location

# --- Summary ---
Write-Host "`n=== BACKUP COMPLETE ===" -ForegroundColor Cyan
Write-Host "Location: $BackupDir`n"
Write-Host "Contents:" -ForegroundColor Yellow
Get-ChildItem $BackupDir | Format-Table Name, @{N="Size (MB)";E={[math]::Round($_.Length / 1MB, 2)}} -AutoSize
