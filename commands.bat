@echo off
:: MiXL — Quick Command Reference
:: Run this file to see all available commands and shortcuts
color 0B
title MiXL — Command Reference

echo.
echo  =====================================================
echo   MiXL - Intelligent AI DJ - Command Reference
echo  =====================================================
echo.
echo  QUICK START (run these in order):
echo  ──────────────────────────────────────────────────
echo.
echo   1. setup.bat          First-time setup (installs everything)
echo   2. start.bat          Launch MiXL (opens browser automatically)
echo.
echo  MANUAL COMMANDS (Command Prompt):
echo  ──────────────────────────────────────────────────
echo.
echo   Activate environment:    venv\Scripts\activate
echo   Start server:            python app.py
echo   Deactivate environment:  deactivate
echo   Update dependencies:     pip install -r requirements.txt --upgrade
echo   Reset everything:        rmdir /s /q venv ^&^& setup.bat
echo.
echo  MANUAL COMMANDS (PowerShell):
echo  ──────────────────────────────────────────────────
echo.
echo   Setup:                   .\setup.ps1
echo   Start:                   .\start.ps1
echo   Activate environment:    .\venv\Scripts\Activate.ps1
echo   Fix execution policy:    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
echo.
echo  ACCESS:
echo  ──────────────────────────────────────────────────
echo.
echo   Browser URL:             http://localhost:5000
echo   Alternative port:        http://localhost:5001  (if 5000 is busy)
echo.
echo  OUTPUT FILES:
echo  ──────────────────────────────────────────────────
echo.
echo   Uploaded tracks:         static\uploads\
echo   Generated mixes:         static\output\
echo.
echo  CONVERT MP3 TO WAV (VLC):
echo  ──────────────────────────────────────────────────
echo.
echo   VLC: Media - Convert/Save - Add file - Profile: WAV - Start
echo.
echo  =====================================================
echo.
pause