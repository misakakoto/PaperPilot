$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $projectRoot ".paperpilot\launcher\server.json"

if (-not (Test-Path -LiteralPath $statePath)) {
    Write-Host "PaperPilot launcher has no recorded background service." -ForegroundColor Yellow
    exit 0
}

$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$processId = [int]$state.process_id
$serverProcess = Get-Process -Id $processId -ErrorAction SilentlyContinue
if (-not $serverProcess) {
    Remove-Item -LiteralPath $statePath -Force
    Write-Host "Removed a stale PaperPilot launcher record." -ForegroundColor Yellow
    exit 0
}

$expectedPython = [System.IO.Path]::GetFullPath([string]$state.python_path)
$actualPython = [System.IO.Path]::GetFullPath([string]$serverProcess.Path)
$expectedSource = [System.IO.Path]::GetFullPath([string]$state.source_root).TrimEnd("\")
try {
    $bootstrap = Invoke-RestMethod -Uri "$($state.url)/api/bootstrap" -TimeoutSec 2
}
catch {
    throw "Refusing to stop PID $processId because its PaperPilot identity endpoint is unavailable."
}
$reportedSource = [System.IO.Path]::GetFullPath([string]$bootstrap.source_root).TrimEnd("\")
$identityMatches = (
    [int]$bootstrap.process_id -eq $processId -and
    [string]$bootstrap.launcher_id -eq [string]$state.launcher_id -and
    $reportedSource -eq $expectedSource
)
if ($actualPython -ne $expectedPython -or -not $identityMatches) {
    throw "Refusing to stop PID $processId because it is not the recorded PaperPilot Web process."
}

Stop-Process -Id $processId -Force
Remove-Item -LiteralPath $statePath -Force
Write-Host "PaperPilot stopped." -ForegroundColor Green
