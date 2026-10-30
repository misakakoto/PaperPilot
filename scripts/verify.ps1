$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& conda run -n agt python -m ruff check src tests scripts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& conda run -n agt python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& conda run -n agt python -m paperpilot_agentic.cli tools
exit $LASTEXITCODE
