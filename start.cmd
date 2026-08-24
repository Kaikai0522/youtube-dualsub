@echo off
rem Double-click this file to run youtube_dualsub. Everything else is start.ps1.
rem -ExecutionPolicy Bypass is what lets a .ps1 run without changing a machine-
rem wide policy; without it Windows opens the script in Notepad instead.
title youtube_dualsub
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo Startup failed with exit code %RC%. The messages above say why.
    echo.
    pause
)
