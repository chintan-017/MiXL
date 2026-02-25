@echo off
setlocal EnableDelayedExpansion
title MiXL — Intelligent AI DJ
color 0B

echo.
echo  =====================================================
echo   MiXL ^| Intelligent AI DJ System
echo   Starting server...
echo  =====================================================
echo.

:: ── Check virtual environment ─────────────────────────────────────────────
if not exist venv (
    echo  Virtual environment not found.
    echo  Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

:: ── Check required folders ────────────────────────────────────────────────
if not exist static\uploads mkdir static\uploads
if not exist static\output  mkdir static\output

:: ── Activate venv ─────────────────────────────────────────────────────────
call venv\Scripts\activate.bat

:: ── Open browser after 2 seconds ──────────────────────────────────────────
echo  Opening browser in 2 seconds...
start "" /b cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5000"

:: ── Launch Flask server ───────────────────────────────────────────────────
echo  Server running at: http://localhost:5000
echo  Press Ctrl+C to stop.
echo.
python app.py

echo.
echo  MiXL server stopped.
pause