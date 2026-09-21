<#
.SYNOPSIS
  One command: generate text from the pre-trained model on the CPU.
  No training, no GPU, no dataset - just the checkpoint committed to this repo.

.DESCRIPTION
  `out/model_best.pt` (1.9 M params, ~8 MB, Llama-3 style) ships with the repository, so this
  script never trains anything: it checks your Python + PyTorch, then samples
  from the checkpoint. That is the whole workflow - training happened once, on
  another machine, and its result is committed here (README section "Generate
  text (CPU, no training)").

  First-time setup, once per machine:
      python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

.EXAMPLE
  .\run.ps1                                        # default prompt, 150 tokens, temp 0.8
  .\run.ps1 -Prompt "Once upon a time" -N 250
  .\run.ps1 -Interactive                           # prompts without reloading weights
  .\run.ps1 -Chat                                  # instruction-tuned multi-turn chatbot
  .\run.ps1 -Seed 4294967295                       # reproducible output
  .\run.ps1 -Python "C:\Python312\python.exe"      # pick a specific interpreter

.NOTES
  If script execution is blocked by policy, use run.bat (forwards every
  argument) or:
      powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1
#>
param(
    [string]$Prompt = "Once upon a time, there was a little girl named Lily",
    [int]   $N      = 150,        # tokens to generate
    [double]$Temp   = 0.8,
    [int]   $TopK   = 40,
    [double]$TopP   = 0.9,
    [long]  $Seed   = -1,         # -1 = random (the seed used is printed)
    [string]$Ckpt   = "out/model_best.pt",
    [string]$Python = "python",
    [double]$RepeatPenalty = 1.15,
    [switch]$Interactive,
    [switch]$Chat
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# Windows consoles default to a legacy code page; the model emits curly quotes
# and em dashes, which would otherwise come out as mojibake or raise
# UnicodeEncodeError. Set the encoding before Python starts.
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# --- 1. find an interpreter -------------------------------------------------
$exe = $null
foreach ($cand in @($Python, "python", "python3", "py")) {
    if ($cand -and (Get-Command $cand -ErrorAction SilentlyContinue)) { $exe = $cand; break }
}
if (-not $exe) {
    Write-Host "[run] No Python found on PATH (tried: $Python, python, python3, py)." -ForegroundColor Red
    Write-Host "      Install Python 3.10+ from https://python.org (tick 'Add python.exe to PATH'),"
    Write-Host "      then re-run this script, or pass the path:  .\run.ps1 -Python C:\Python312\python.exe"
    exit 1
}

# --- 2. is PyTorch installed? ----------------------------------------------
& $exe -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[run] PyTorch is not importable by '$exe'. Install the CPU build (once):" -ForegroundColor Yellow
    Write-Host "      $exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
    exit 1
}

# --- 3. is the checkpoint there? -------------------------------------------
# -Chat selects the SFT checkpoint unless the caller explicitly chose -Ckpt.
if ($Chat -and -not $PSBoundParameters.ContainsKey("Ckpt")) { $Ckpt = "out/model_chat.pt" }
if (-not (Test-Path -LiteralPath $Ckpt)) {
    Write-Host "[run] Checkpoint '$Ckpt' not found." -ForegroundColor Yellow
    Write-Host "      It is committed to this repository (~8 MB) - fetch it with:  git pull"
    Write-Host "      (git pull must have completed; the file lives at out/model_best.pt)"
    exit 1
}

# --- 4. generate ------------------------------------------------------------
Write-Host "[run] generating with $exe  (Ctrl+C stops it)" -ForegroundColor Cyan
$genArgs = @("generate.py", "--ckpt", $Ckpt, "--prompt", $Prompt, "--n", $N,
             "--temp", $Temp, "--top-k", $TopK, "--top-p", $TopP, "--repeat-penalty", $RepeatPenalty)
if ($Interactive) { $genArgs += "--interactive" }
if ($Chat) { $genArgs += "--chat" }
if ($Seed -ge 0) { $genArgs += @("--seed", $Seed) }

& $exe @genArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "[run] generation failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[run] done. Change the prompt with:  .\run.ps1 -Prompt `"your text`" -N 250" -ForegroundColor DarkGray
