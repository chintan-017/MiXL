@echo off
setlocal EnableDelayedExpansion
title MiXL — Intelligent AI DJ Setup
color 0B

echo.
echo  =====================================================
echo   __  __ _ __  ____  __
echo  ^|  \/  ^| ^|\ \/ /  \/  ^|
echo  ^| ^|\/^| ^| ^| ^|>  ^<^| ^|\/^| ^|
echo  ^|_^|  ^|_^|_^|/_/\_\_^|  ^|_^|
echo.
echo   Intelligent AI DJ System — Windows Setup
echo  =====================================================
echo.

:: ── Check Python ──────────────────────────────────────────────────────────
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Python not found.
    echo.
    echo  Please install Python 3.10 or higher from:
    echo  https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: During installation, check the box:
    echo  "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Python %PYVER% found.

:: ── Check pip ─────────────────────────────────────────────────────────────
echo.
echo [2/5] Checking pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  pip not found. Installing...
    python -m ensurepip --upgrade
)
echo  pip ready.

:: ── Create virtual environment ────────────────────────────────────────────
echo.
echo [3/5] Creating virtual environment...
if exist venv (
    echo  Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo  Virtual environment created.
)

:: ── Activate and install deps ─────────────────────────────────────────────
echo.
echo [4/5] Installing dependencies (flask, numpy, scipy)...
call venv\Scripts\activate.bat

pip install --upgrade pip --quiet
pip install flask numpy scipy --quiet

if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Failed to install dependencies.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)
echo  All dependencies installed successfully.

:: ── Create required folders ───────────────────────────────────────────────
echo.
echo [5/5] Creating required folders...
if not exist static\uploads mkdir static\uploads
if not exist static\output  mkdir static\output
echo  Folders ready.

:: ── Done ─────────────────────────────────────────────────────────────────
echo.
echo  =====================================================
echo   Setup complete!
echo.
echo   To start MiXL, run:  start.bat
echo   Or double-click:     start.bat
echo  =====================================================
echo.
pause