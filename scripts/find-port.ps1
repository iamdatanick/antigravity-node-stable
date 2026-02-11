$connections = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue
foreach ($c in $connections) {
    $proc = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
    Write-Output "$($proc.Id) $($proc.ProcessName)"
}
