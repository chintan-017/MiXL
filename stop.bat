@echo off
title MiXL — Stop Server
color 0C

echo.
echo  Stopping MiXL server...
echo.

:: Kill any Python process running app.py on port 5000
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
    echo  Found process on port 5000 (PID: %%p) - terminating...
    taskkill /F /PID %%p >nul 2>&1
)

:: Also try killing by name if flask is running
taskkill /F /IM python.exe /FI "WINDOWTITLE eq MiXL*" >nul 2>&1

echo  MiXL server stopped.
echo.
pause