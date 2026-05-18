# Start Arabic STT (kills anything on GRADIO_SERVER_PORT first).
# Usage:
#   .\run.ps1              - watchdog reload (default)
#   .\run.ps1 -NoReload    - single run, no file watcher
#   .\run.ps1 -GradioReload - gradio in-process reload instead of dev_watch.py

param(
    [switch]$NoReload,
    [switch]$GradioReload
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$port = if ($env:GRADIO_SERVER_PORT) { [int]$env:GRADIO_SERVER_PORT } else { 7860 }
Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$gradio = Join-Path $PSScriptRoot ".venv\Scripts\gradio.exe"

if (-not (Test-Path $python)) {
    Write-Error "Missing .venv - run: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
}

if ($NoReload) {
    & $python app.py
} elseif ($GradioReload) {
    Write-Host "Gradio reload mode (in-process watch)" -ForegroundColor Cyan
    & $gradio app.py
} else {
    Write-Host "Watchdog mode (restarts on .py changes)" -ForegroundColor Cyan
    & $python dev_watch.py
}
