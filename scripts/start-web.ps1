param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"
# Share the launcher that pins PYTHONPATH and verifies the active checkout.
& (Join-Path $PSScriptRoot 'launch.ps1') -NoBrowser:$NoBrowser
exit $LASTEXITCODE
