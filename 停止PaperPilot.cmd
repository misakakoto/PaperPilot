@echo off
chcp 65001 >nul
title Stop PaperPilot
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop.ps1"
if errorlevel 1 (
    echo.
    echo PaperPilot could not be stopped safely. Review the error above.
    pause
)
