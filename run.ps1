<#
.SYNOPSIS
  One-command entry point: train the tiny CPU model if needed, then generate text.

.DESCRIPTION
  First run trains the small "out_fast" model (30 iters, d_model 96, 2 layers -
  under a minute on a modern laptop CPU), then samples from it. Later runs reuse
  the existing checkpoint and just generate.

  Works on Windows PowerShell 5.1 and PowerShell 7. If script execution is
  blocked by policy, use:  powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1
  or just double-click run.bat.

.EXAMPLE
  .\run.ps1                                          # default prompt, 200 tokens
  .\run.ps1 -Prompt "Once upon a time" -N 150
  .\run.ps1 -Eval                                    # also print eval ppl
  .\run.ps1 -Retrain                                 # wipe out_fast, train fresh
#>
param(
    [string]$Prompt  = "It was a dark and stormy night",
    [int]$N          = 200,      # tokens to generate
    [double]$Temp    = 0.8,
    [int]$TopK       = 50,
    [string]$OutDir  = "out_fast",
    [string]$Python  = "python", # or a full path to python.exe
    [switch]$Eval,               # also run 05_evaluate.py (val ppl)
    [switch]$Retrain             # delete $OutDir and train from scratch
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$ckpt = Join-Path $OutDir "model_best.pt"
$cfg  = Join-Path $OutDir "config.txt"

# --- sanity checks: python and torch present? --------------------------------
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "python executable '$Python' not found on PATH. Edit the -Python parameter or install Python."
}
& $Python -c "import torch"
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch import failed. Install the CPU build with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
}

# --- get a checkpoint (train on first use) -----------------------------------
if ($Retrain -and (Test-Path $OutDir)) {
    Write-Host "[run] -Retrain given: removing $OutDir" -ForegroundColor Yellow
    Remove-Item -Recurse -Force -LiteralPath $OutDir
}

if (-not (Test-Path $ckpt)) {
    Write-Host "[run] no checkpoint in $OutDir - training the tiny model (about a minute on CPU) ..." -ForegroundColor Cyan
    & $Python 04_train.py --iters 30 --batch-size 32 --grad-accum 1 `
        --d-model 96 --n-layer 2 --n-head 3 --out-dir $OutDir
    if ($LASTEXITCODE -ne 0) { throw "training failed (exit code $LASTEXITCODE)" }
}
else {
    Write-Host "[run] reusing checkpoint $ckpt" -ForegroundColor DarkGray
}

if ($Eval) {
    & $Python 05_evaluate.py --ckpt $ckpt
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed (exit code $LASTEXITCODE)" }
}

# --- generate -----------------------------------------------------------------
Write-Host ""
Write-Host "[run] generating $N tokens (temp $Temp, top-k $TopK) ..." -ForegroundColor Cyan
& $Python sample.py --ckpt $ckpt --config $cfg --prompt $Prompt --n $N --temp $Temp --top-k $TopK
if ($LASTEXITCODE -ne 0) { throw "generation failed (exit code $LASTEXITCODE)" }
