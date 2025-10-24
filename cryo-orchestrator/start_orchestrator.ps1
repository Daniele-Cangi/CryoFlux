$env:JOULE_AGENT_URL = "http://127.0.0.1:8787"
Set-Location C:\Users\dacan\OneDrive\Desktop\Cryo\cryo-orchestrator
if (Test-Path .venv\Scripts\Activate.ps1) {
    & .\.venv\Scripts\Activate.ps1
}
Write-Host "[Launcher] Starting CryoFlux Orchestrator..." -ForegroundColor Green
python -u cryo.py
