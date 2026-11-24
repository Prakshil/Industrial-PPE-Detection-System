@echo off
title Industrial PPE Monitor Dashboard
cd /d "%~dp0"
echo ===================================================
echo   Starting Industrial PPE Safety Dashboard (Edge)
echo   Dashboard URL: http://localhost:8000/
echo ===================================================
python main.py
pause
