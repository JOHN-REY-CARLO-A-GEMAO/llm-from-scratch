@echo off
rem Double-click / cmd launcher for run.ps1 (bypasses PowerShell execution policy).
rem Any arguments are forwarded, e.g.:  run.bat -Prompt "Once upon a time" -N 150
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*

rem If started by double-click, keep the window open so the output stays visible.
echo %cmdcmdline% | find /i "%~f0" >nul && pause
