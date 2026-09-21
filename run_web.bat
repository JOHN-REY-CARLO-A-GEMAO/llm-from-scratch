@echo off
rem One-click Windows launcher: chat with the fine-tuned model in your browser.
rem Double-click this file, or forward any argument:
rem     run_web.bat -Port 8080
rem It simply calls run_web.ps1 with the execution policy bypassed, so no admin
rem rights and no PowerShell configuration are needed. The page opens at
rem http://127.0.0.1:7860 (also reachable on your LAN via 0.0.0.0:7860).
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_web.ps1" %*

rem If started by double-click, keep the window open so the log stays visible.
echo %cmdcmdline% | find /i "%~f0" >nul && pause
