@echo off
rem One-click Windows launcher: generate text from the pre-trained model on the CPU.
rem Double-click this file, or forward any argument:
rem     run.bat -Prompt "Once upon a time" -N 250
rem It simply calls run.ps1 with the execution policy bypassed, so no admin
rem rights and no PowerShell configuration are needed.
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*

rem If started by double-click, keep the window open so the text stays visible.
echo %cmdcmdline% | find /i "%~f0" >nul && pause
