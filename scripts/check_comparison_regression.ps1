param(
    [string]$Python = (Get-Command python -ErrorAction Stop).Source
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "agt Python was not found: $Python" }
$env:PYTHONPATH = (Join-Path $projectRoot "src") + ";" + $projectRoot
$env:PYTHONIOENCODING = "utf-8"

# Environment-only probes. No model requests, source changes or cleanup bypass.
$probe = @'
from pathlib import Path
import sqlite3
import tempfile
with tempfile.TemporaryDirectory(prefix="paperpilot-preflight-") as folder:
    Path(folder).resolve(strict=True)
print("SQLite and strict temporary-path checks passed")
'@
& $Python -X utf8 -c $probe
if ($LASTEXITCODE -ne 0) { throw "Environment preflight failed; tests were not started." }

$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$client = [System.Net.Sockets.TcpClient]::new()
try {
    $listener.Start()
    $port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    $connect = $client.ConnectAsync([System.Net.IPAddress]::Loopback, $port)
    if (-not $connect.Wait(2000)) { throw "Loopback preflight timed out; tests were not started." }
    $connect.GetAwaiter().GetResult()
}
finally {
    $client.Dispose()
    $listener.Stop()
}

$runName = "comparison-repair-full-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [guid]::NewGuid().ToString("N")
$report = Join-Path $projectRoot ("outputs\" + $runName + ".xml")
if (Test-Path -LiteralPath $report) { throw "Refusing to overwrite a previous report." }
$tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$testTemp = [System.IO.Path]::GetFullPath((Join-Path $tempRoot ($runName + "-tmp")))
# pytest removes an existing --basetemp recursively. Never hand it an old path,
# including the shared pytest-of-user directory from a different execution user.
if (-not $testTemp.StartsWith($tempRoot.TrimEnd("\") + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Test directory escaped the intended temporary root."
}
if (Test-Path -LiteralPath $testTemp) { throw "Refusing to reuse a previous test directory." }
Write-Host "Running the complete offline suite. Report: $report"
Write-Host "Isolated test directory: $testTemp"
& $Python -X utf8 -m pytest -q -p no:cacheprovider --tb=short ("--basetemp=" + $testTemp) ("--junitxml=" + $report)
exit $LASTEXITCODE
