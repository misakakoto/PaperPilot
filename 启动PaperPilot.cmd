@echo off
chcp 65001 >nul
title PaperPilot Agent Launcher
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1"
if errorlevel 1 (
    echo.
    echo PaperPilot failed to start. Review the error above and .paperpilot\launcher logs.
    pause
)
