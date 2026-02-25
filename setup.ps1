# MiXL — Intelligent AI DJ System
# PowerShell Setup & Launch Script
# Run as: .\setup.ps1

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "MiXL Setup"

function Write-Header {
    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Cyan
    Write-Host "   __  __ _ __  ____  __" -ForegroundColor Cyan
    Write-Host "  |  \/  | |\ \/ /  \/  |" -ForegroundColor Cyan
    Write-Host "  | |\/| | | |>  <| |\/| |" -ForegroundColor Cyan
    Write-Host "  |_|  |_|_|/_/\_\_|  |_|" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   Intelligent AI DJ System — Windows Setup" -ForegroundColor White
    Write-Host "  =====================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Step, [string]$Msg)
    Write-Host "  [$Step] " -ForegroundColor Yellow -NoNewline
    Write-Host $Msg -ForegroundColor White
}

function Write-OK { Write-Host "  ✓ $args" -ForegroundColor Green }
function Write-Fail { Write-Host "  ✗ $args" -ForegroundColor Red }
function Write-Info { Write-Host "  → $args" -ForegroundColor Cyan }

Write-Header

# ── Check Python ─────────────────────────────────────────────────────────
Write-Step "1/5" "Checking Python..."
try {
    $pyVersion = python --version 2>&1
    Write-OK "Found: $pyVersion"
} catch {
    Write-Fail "Python not found!"
    Write-Host ""
    Write-Host "  Download Python 3.10+ from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/" -ForegroundColor White
    Write-Host ""
    Write-Host "  IMPORTANT: Check 'Add Python to PATH' during install" -ForegroundColor Red
    Read-Host "  Press Enter to exit"
    exit 1
}

# ── Check pip ────────────────────────────────────────────────────────────
Write-Step "2/5" "Checking pip..."
try {
    $pipVer = pip --version 2>&1
    Write-OK "pip ready"
} catch {
    Write-Info "Installing pip..."
    python -m ensurepip --upgrade
    Write-OK "pip installed"
}

# ── Create venv ──────────────────────────────────────────────────────────
Write-Step "3/5" "Setting up virtual environment..."
if (Test-Path "venv") {
    Write-OK "Virtual environment already exists"
} else {
    python -m venv venv
    Write-OK "Virtual environment created"
}

# ── Install deps ─────────────────────────────────────────────────────────
Write-Step "4/5" "Installing dependencies..."
Write-Info "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

Write-Info "Upgrading pip..."
pip install --upgrade pip --quiet

Write-Info "Installing flask, numpy, scipy..."
pip install flask numpy scipy --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Fail "Dependency installation failed"
    Write-Host "  Check your internet connection and try again" -ForegroundColor Yellow
    Read-Host "  Press Enter to exit"
    exit 1
}
Write-OK "All dependencies installed"

# ── Create folders ────────────────────────────────────────────────────────
Write-Step "5/5" "Creating required directories..."
@("static\uploads", "static\output") | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
        Write-OK "Created: $_"
    } else {
        Write-OK "Exists: $_"
    }
}

# ── Done ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  =====================================================" -ForegroundColor Green
Write-Host "   Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "   To start MiXL:" -ForegroundColor White
Write-Host "   CMD:        double-click start.bat" -ForegroundColor Cyan
Write-Host "   PowerShell: .\start.ps1" -ForegroundColor Cyan
Write-Host "  =====================================================" -ForegroundColor Green
Write-Host ""
Read-Host "  Press Enter to exit"