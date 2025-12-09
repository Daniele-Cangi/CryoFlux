# CryoFlux v0.1 - Development Run Script (PowerShell)
# Run from repo root: .\scripts\dev_run.ps1

$ErrorActionPreference = "Stop"

Write-Host "[dev_run] CryoFlux v0.1 Development Runner" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the repo root
if (-not (Test-Path "joule-agent-rs")) {
    Write-Host "[ERROR] Please run this script from the CryoFlux repo root." -ForegroundColor Red
    Write-Host "Usage: .\scripts\dev_run.ps1" -ForegroundColor Yellow
    exit 1
}

# Check if setup was run
if (-not (Test-Path "joule-agent-rs/target/release/joule-agent-rs.exe")) {
    Write-Host "[ERROR] JouleAgent not built. Please run .\scripts\dev_setup.ps1 first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "cryo-orchestrator/.venv")) {
    Write-Host "[ERROR] Python venv not found. Please run .\scripts\dev_setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Generate default config.toml if missing
if (-not (Test-Path "config.toml")) {
    Write-Host "[INFO] config.toml not found, generating default configuration..." -ForegroundColor Yellow
    $defaultConfig = @"
# CryoFlux v0.1 Configuration (auto-generated)

[joule_agent]
hz = 2.0
cpu_tdp_w = 65.0
smoothing_alpha = 0.2
idle_learn_w = 5.0
bind_addr = "127.0.0.1:8787"

[orchestrator]
agent_url = "http://127.0.0.1:8787"
receipts_db = "./state/receipts.db"
seed = 42

[orchestrator.model]
encoder_model = "sentence-transformers/all-MiniLM-L6-v2"
clf_base = "distilbert-base-uncased"
lora_rank = 8

[orchestrator.data]
incoming_dir = "./data/incoming"
holdout_csv = "./data/holdout.csv"
embeddings_cache = "./state/embeddings"

[orchestrator.storage]
capsules_dir = "./state/capsules"
base_dir = "./state/base_model"
candidates_dir = "./state/candidates"

[orchestrator.energy]
min_joule_to_run = 1.0
task_index_est_joules = 20.0
task_lora_est_joules = 120.0

[orchestrator.merge]
lora_accept_min_delta = 0.003
merge_every_n_capsules = 1
"@
    $defaultConfig | Out-File -FilePath "config.toml" -Encoding UTF8
    Write-Host "[INFO] Created config.toml with default values" -ForegroundColor Green
}

# Parse bind address from config.toml
$bindAddr = "127.0.0.1:8787"
if (Test-Path "config.toml") {
    $configContent = Get-Content "config.toml" -Raw
    if ($configContent -match 'bind_addr\s*=\s*"([^"]+)"') {
        $bindAddr = $matches[1]
    }
}

$agentUrl = "http://$bindAddr"
Write-Host "[1/3] Starting JouleAgent on $bindAddr..." -ForegroundColor Green

# Start JouleAgent in background
$jouleAgentExe = "joule-agent-rs\target\release\joule-agent-rs.exe"
$agentProcess = Start-Process -FilePath $jouleAgentExe -WorkingDirectory "joule-agent-rs" -PassThru -WindowStyle Hidden

if (-not $agentProcess) {
    Write-Host "[ERROR] Failed to start JouleAgent" -ForegroundColor Red
    exit 1
}

Write-Host "  JouleAgent started (PID: $($agentProcess.Id))" -ForegroundColor Gray

# Wait for JouleAgent to be ready
Write-Host "[2/3] Waiting for JouleAgent to be ready..." -ForegroundColor Green
$maxAttempts = 20
$attempt = 0
$agentReady = $false

while ($attempt -lt $maxAttempts) {
    Start-Sleep -Milliseconds 500
    try {
        $response = Invoke-WebRequest -Uri "$agentUrl/v1/sample" -TimeoutSec 1 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $agentReady = $true
            break
        }
    } catch {
        # Connection failed, retry
    }
    $attempt++
    Write-Host "  Attempt $attempt/$maxAttempts..." -ForegroundColor Gray
}

if (-not $agentReady) {
    Write-Host "[ERROR] JouleAgent did not become ready after $maxAttempts attempts" -ForegroundColor Red
    Write-Host "[INFO] Stopping JouleAgent (PID: $($agentProcess.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $agentProcess.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "  JouleAgent is ready!" -ForegroundColor Gray
Write-Host ""

# Start Orchestrator in foreground
Write-Host "[3/3] Starting Orchestrator..." -ForegroundColor Green
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "CryoFlux v0.1 is running!" -ForegroundColor Cyan
Write-Host "JouleAgent: $agentUrl" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

Push-Location cryo-orchestrator

# Cleanup function
$cleanup = {
    Write-Host ""
    Write-Host "[INFO] Shutting down..." -ForegroundColor Yellow
    Write-Host "[INFO] Stopping JouleAgent (PID: $($agentProcess.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $agentProcess.Id -Force -ErrorAction SilentlyContinue
    Write-Host "[INFO] Shutdown complete" -ForegroundColor Green
    Pop-Location
    exit 0
}

# Register cleanup on Ctrl+C
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action $cleanup | Out-Null

try {
    # Activate venv and run orchestrator
    $venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        & $venvPython -u cryo.py
    } else {
        Write-Host "[ERROR] Python venv not found at expected location" -ForegroundColor Red
        & $cleanup
    }
} catch {
    Write-Host "[ERROR] Orchestrator crashed: $_" -ForegroundColor Red
} finally {
    & $cleanup
}
