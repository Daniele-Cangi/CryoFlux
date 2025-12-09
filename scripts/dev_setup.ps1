# CryoFlux v0.1 - Development Setup Script (PowerShell)
# Run from repo root: .\scripts\dev_setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "[dev_setup] CryoFlux v0.1 Development Setup" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the repo root
if (-not (Test-Path "joule-agent-rs")) {
    Write-Host "[ERROR] Please run this script from the CryoFlux repo root." -ForegroundColor Red
    Write-Host "Usage: .\scripts\dev_setup.ps1" -ForegroundColor Yellow
    exit 1
}

# 1. Check Rust and Cargo
Write-Host "[1/5] Checking Rust toolchain..." -ForegroundColor Green
try {
    $rustVersion = cargo --version 2>&1
    Write-Host "  Found: $rustVersion" -ForegroundColor Gray
} catch {
    Write-Host "[ERROR] Cargo not found. Please install Rust from https://rustup.rs/" -ForegroundColor Red
    exit 1
}

# 2. Build JouleAgent
Write-Host "[2/5] Building JouleAgent (Rust)..." -ForegroundColor Green
Push-Location joule-agent-rs
try {
    cargo build --release
    if ($LASTEXITCODE -ne 0) {
        throw "Cargo build failed"
    }
    Write-Host "  JouleAgent built successfully" -ForegroundColor Gray
} catch {
    Write-Host "[ERROR] Failed to build JouleAgent: $_" -ForegroundColor Red
    Pop-Location
    exit 1
}
Pop-Location

# 3. Check Python
Write-Host "[3/5] Checking Python..." -ForegroundColor Green
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Gray
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.10+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# 4. Create Python venv
Write-Host "[4/5] Setting up Python virtual environment..." -ForegroundColor Green
Push-Location cryo-orchestrator
if (Test-Path ".venv") {
    Write-Host "  Virtual environment already exists, skipping creation" -ForegroundColor Gray
} else {
    python -m venv .venv
    Write-Host "  Created .venv/" -ForegroundColor Gray
}

# Activate venv and install dependencies
Write-Host "  Installing Python dependencies..." -ForegroundColor Gray
$venvActivate = Join-Path (Get-Location) ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install Python dependencies" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Write-Host "  Dependencies installed successfully" -ForegroundColor Gray
} else {
    Write-Host "[ERROR] Could not activate virtual environment" -ForegroundColor Red
    Pop-Location
    exit 1
}
Pop-Location

# 5. Create data directories and placeholder files
Write-Host "[5/5] Setting up data directories..." -ForegroundColor Green

# Create directories
$dirs = @("data", "data/incoming", "state", "state/capsules", "state/base_model", "state/candidates", "state/embeddings")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  Created $dir/" -ForegroundColor Gray
    }
}

# Create placeholder holdout.csv if missing
if (-not (Test-Path "data/holdout.csv")) {
    Write-Host "  Creating placeholder data/holdout.csv..." -ForegroundColor Gray
    $holdoutContent = @"
"text",label
"This product exceeded my expectations",1
"Absolutely terrible, waste of money",0
"Great value for the price",1
"Poor quality, broke after one use",0
"Highly recommend this to everyone",1
"Disappointed with the service",0
"Works perfectly as described",1
"Not worth the cost at all",0
"Amazing experience, will buy again",1
"Defective item, requesting refund",0
"Best purchase I've made this year",1
"Cheap materials, feels flimsy",0
"Exceeded all my expectations",1
"Horrible customer support",0
"Exactly what I was looking for",1
"Complete waste of time and money",0
"@

    $holdoutContent | Out-File -FilePath "data/holdout.csv" -Encoding UTF8
    Write-Host "  Created data/holdout.csv with 16 samples" -ForegroundColor Gray
}

# Create placeholder news.txt if missing
if (-not (Test-Path "data/incoming/news.txt")) {
    Write-Host "  Creating placeholder data/incoming/news.txt..." -ForegroundColor Gray
    $newsContent = @"
Markets rally on strong economic data
Tech sector faces regulatory challenges
Energy demand surges amid supply concerns
Global supply chains show signs of recovery
Central bank announces new monetary policy
Renewable energy investment reaches record high
Consumer confidence index shows improvement
Inflation pressures continue to build
Manufacturing output exceeds expectations
Trade negotiations yield positive results
"@
    $newsContent | Out-File -FilePath "data/incoming/news.txt" -Encoding UTF8
    Write-Host "  Created data/incoming/news.txt with placeholder data" -ForegroundColor Gray
}

# Create config.toml if missing
if (-not (Test-Path "config.toml")) {
    Write-Host "[WARN] config.toml not found. Run dev_run.ps1 to generate default config." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[dev_setup] Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run: .\scripts\dev_run.ps1" -ForegroundColor White
Write-Host "  2. Monitor logs for JouleAgent + Orchestrator activity" -ForegroundColor White
Write-Host ""
