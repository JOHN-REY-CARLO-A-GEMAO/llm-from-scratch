<#
.SYNOPSIS
  One command: chat with the fine-tuned model in your browser.
  No training, no GPU - the web UI wraps out/model_chat.pt, committed to this repo.

.DESCRIPTION
  Starts web_ui.py on http://127.0.0.1:7860 (also bound on 0.0.0.0:7860, so other
  devices on your LAN can reach it) and opens your default browser. Tokens stream
  into the page as the model generates them; temperature and top-p are sliders in
  the UI, and the Clear Chat button wipes the conversation.

  Uses the gradio frontend if gradio happens to be installed, otherwise the
  built-in zero-dependency interface - either way there is nothing extra to
  install beyond PyTorch.

  First-time setup, once per machine:
      python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

.EXAMPLE
  .\run_web.ps1                                 # http://127.0.0.1:7860 + browser opens
  .\run_web.ps1 -Port 8080                      # different port
  .\run_web.ps1 -NoBrowser                      # serve only, do not open a tab
  .\run_web.ps1 -Python "C:\Python312\python.exe"   # pick a specific interpreter

.NOTES
  If script execution is blocked by policy, use run_web.bat (forwards every
  argument) or:
      powershell -NoProfile -ExecutionPolicy Bypass -File run_web.ps1
#>
param(
    [int]   $Port     = 7860,
    [string]$BindHost    = "0.0.0.0",
    [string]$Ckpt     = "out/model_chat.pt",
    [string]$Python   = "python",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# Windows consoles default to a legacy code page; the model emits curly quotes
# and em dashes, which would otherwise come out as mojibake in the server log.
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# --- 1. find an interpreter -------------------------------------------------
$exe = $null
foreach ($cand in @($Python, "python", "python3", "py")) {
    if ($cand -and (Get-Command $cand -ErrorAction SilentlyContinue)) { $exe = $cand; break }
}
if (-not $exe) {
    Write-Host "[web] No Python found on PATH (tried: $Python, python, python3, py)." -ForegroundColor Red
    Write-Host "      Install Python 3.10+ from https://python.org (tick 'Add python.exe to PATH'),"
    Write-Host "      then re-run this script, or pass the path:  .\run_web.ps1 -Python C:\Python312\python.exe"
    exit 1
}

# --- 2. is PyTorch installed? ----------------------------------------------
& $exe -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[web] PyTorch is not importable by '$exe'. Install the CPU build (once):" -ForegroundColor Yellow
    Write-Host "      $exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
    exit 1
}

# --- 3. is the chat checkpoint there? --------------------------------------
if (-not (Test-Path -LiteralPath $Ckpt)) {
    Write-Host "[web] Checkpoint '$Ckpt' not found." -ForegroundColor Yellow
    Write-Host "      It is committed to this repository (~8 MB) - fetch it with:  git pull"
    exit 1
}
if (-not (Test-Path -LiteralPath "out/model_chat_config.json")) {
    Write-Host "[web] out/model_chat_config.json not found - it pairs with the checkpoint:  git pull" -ForegroundColor Yellow
    exit 1
}

# --- 4. serve ---------------------------------------------------------------
$webArgs = @("web_ui.py", "--host", $BindHost, "--port", $Port, "--ckpt", $Ckpt)
if ($NoBrowser) { $webArgs += "--no-browser" }

Write-Host "[web] starting the chat UI with $exe  (Ctrl+C stops it)" -ForegroundColor Cyan
& $exe @webArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "[web] the server exited with code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}
