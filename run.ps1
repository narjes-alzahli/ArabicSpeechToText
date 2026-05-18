# Start Arabic STT (kills anything on GRADIO_SERVER_PORT first).
# Usage:
#   .\run.ps1          — normal run
#   .\run.ps1 -Reload  — auto-restart when app.py / transcribe.py change

param([switch]$Reload)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$port = if ($env:GRADIO_SERVER_PORT) { [int]$env:GRADIO_SERVER_PORT } else { 7860 }
Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$gradio = Join-Path $PSScriptRoot ".venv\Scripts\gradio.exe"

if (-not (Test-Path $python)) {
    Write-Error "Missing .venv — run: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
}

if ($Reload) {
    Write-Host "Reload mode: watching app.py and transcribe.py" -ForegroundColor Cyan
    & $gradio app.py
} else {
    & $python app.py
}
