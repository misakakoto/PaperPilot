param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if ($env:PAPERPILOT_NO_BROWSER -in @("1", "true", "yes")) {
    $NoBrowser = $true
}

function Resolve-AgtPython {
    foreach ($configured in @($env:PAPERPILOT_PYTHON, $env:PAPERPILOT_AGT_PYTHON)) {
        if ($configured) {
            if (-not (Test-Path -LiteralPath $configured)) { throw "Configured Python does not exist." }
            return (Resolve-Path -LiteralPath $configured).Path
        }
    }
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) { return $venvPython }
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { return $pythonCommand.Source }
    throw "Python 3.11+ was not found. Activate your environment or set PAPERPILOT_PYTHON."
}

function Test-LocalPortAvailable([int]$Port) {
    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $Port
    )
    try {
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        $listener.Stop()
    }
}

function Get-PaperPilotSource([string]$BaseUrl) {
    try {
        $value = Invoke-RestMethod -Uri "$BaseUrl/api/bootstrap" -TimeoutSec 2
        if ($value.name -eq "PaperPilot") {
            return [string]$value.source_root
        }
    }
    catch {
        return $null
    }
    return $null
}

function Get-PaperPilotBrowserUrl([string]$BaseUrl) {
    # Old browsers may retain the previous root HTML despite the new server's
    # no-store policy. Version the entry URL without clearing user history.
    $info = Invoke-RestMethod -Uri "$BaseUrl/api/bootstrap" -TimeoutSec 3
    if (-not $info.capabilities.workbench -or $info.build.restart_required) {
        Write-Warning "The running service is older than this checkout. Finish or stop active work, then restart PaperPilot."
    }
    $version = [string]$info.build.loaded.source_digest
    if (-not $version) { $version = "legacy" }
    return "$BaseUrl/?ui=$([Uri]::EscapeDataString($version))"
}

$agtPython = Resolve-AgtPython
$sourceRoot = Join-Path $projectRoot "src"
$separator = [System.IO.Path]::PathSeparator
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$sourceRoot$separator$($env:PYTHONPATH)"
}
else {
    $env:PYTHONPATH = $sourceRoot
}

# cmd.exe may expose the process variable as PATH while Windows PowerShell 5.1
# merges a registry variable named Path. Start-Process treats those as duplicate
# dictionary keys, so normalize it before either opening a browser or spawning.
$processPathValue = $env:Path
Remove-Item Env:Path -ErrorAction SilentlyContinue
$env:Path = $processPathValue

$pythonConfigCode = "from dotenv import load_dotenv; import os; load_dotenv(override=False); print(os.getenv('PAPERPILOT_WEB_HOST', '127.0.0.1')); print(os.getenv('PAPERPILOT_WEB_PORT', '8014'))"
$webConfig = & $agtPython -c $pythonConfigCode
if ($LASTEXITCODE -ne 0 -or $webConfig.Count -lt 2) {
    throw "PaperPilot Web configuration could not be loaded."
}

$bindAddress = [string]$webConfig[-2]
$configuredPort = [int]$webConfig[-1]
$browserAddress = if ($bindAddress -in @("0.0.0.0", "::")) { "127.0.0.1" } else { $bindAddress }
$port = $configuredPort
$baseUrl = "http://${browserAddress}:$port"
$existingSource = Get-PaperPilotSource $baseUrl

if ($existingSource) {
    $existingPath = [System.IO.Path]::GetFullPath($existingSource).TrimEnd("\")
    $currentPath = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd("\")
    if ($existingPath -eq $currentPath) {
        Write-Host "PaperPilot is already running at $baseUrl" -ForegroundColor Green
        if (-not $NoBrowser) {
            Start-Process (Get-PaperPilotBrowserUrl $baseUrl)
        }
        exit 0
    }
}

if (-not (Test-LocalPortAvailable $port)) {
    $freePort = (($configuredPort + 1)..($configuredPort + 20) |
        Where-Object { Test-LocalPortAvailable $_ } |
        Select-Object -First 1)
    if (-not $freePort) {
        throw "Ports $configuredPort through $($configuredPort + 20) are unavailable."
    }
    $port = [int]$freePort
    $env:PAPERPILOT_WEB_PORT = [string]$port
    $baseUrl = "http://${browserAddress}:$port"
    Write-Host "Port $configuredPort is occupied; using port $port instead." -ForegroundColor Yellow
}

$loadedSource = @(& $agtPython -c "import pathlib, paperpilot_agentic; print(pathlib.Path(paperpilot_agentic.__file__).resolve().parents[2])")
if ($LASTEXITCODE -ne 0) {
    throw "paperpilot_agentic could not be imported from this project."
}
$loadedPath = [System.IO.Path]::GetFullPath([string]$loadedSource[-1]).TrimEnd("\")
$expectedPath = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd("\")
if ($loadedPath -ne $expectedPath) {
    throw "PaperPilot was loaded from the wrong source tree: $loadedPath"
}

$launcherRoot = Join-Path $projectRoot ".paperpilot\launcher"
New-Item -ItemType Directory -Path $launcherRoot -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$launcherId = [guid]::NewGuid().ToString("N")
$env:PAPERPILOT_LAUNCHER_ID = $launcherId
$stdoutLog = Join-Path $launcherRoot "web-$timestamp.stdout.log"
$stderrLog = Join-Path $launcherRoot "web-$timestamp.stderr.log"
$statePath = Join-Path $launcherRoot "server.json"

Write-Host "Starting PaperPilot at $baseUrl" -ForegroundColor Cyan
$serverProcess = Start-Process `
    -FilePath $agtPython `
    -ArgumentList @("-m", "paperpilot_agentic.web_app") `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

$state = [ordered]@{
    process_id = $serverProcess.Id
    launcher_id = $launcherId
    url = $baseUrl
    source_root = $projectRoot
    python_path = $agtPython
    started_at = (Get-Date).ToString("o")
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
}
$state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8

$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if ($serverProcess.HasExited) {
        break
    }
    try {
        $health = Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 1
        $bootstrap = Invoke-RestMethod -Uri "$baseUrl/api/bootstrap" -TimeoutSec 1
        $reportedSource = [System.IO.Path]::GetFullPath([string]$bootstrap.source_root).TrimEnd("\")
        if ($health.status -eq "ok" -and $reportedSource -eq $expectedPath) {
            $ready = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}

if (-not $ready) {
    if (-not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
    $details = @()
    if (Test-Path -LiteralPath $stderrLog) {
        $details = Get-Content -LiteralPath $stderrLog -Tail 20 -ErrorAction SilentlyContinue
    }
    throw "PaperPilot did not become healthy. Log: $stderrLog`n$($details -join [Environment]::NewLine)"
}

Write-Host "PaperPilot is ready at $baseUrl" -ForegroundColor Green
Write-Host "Use 停止PaperPilot.cmd to stop the background service." -ForegroundColor DarkGray
if (-not $NoBrowser) {
    Start-Process (Get-PaperPilotBrowserUrl $baseUrl)
}
exit 0
