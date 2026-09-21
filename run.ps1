<#
.SYNOPSIS
  One command: generate text from the pre-trained model on the CPU.
  No training, no GPU, no dataset - just the checkpoint committed to this repo.

.DESCRIPTION
  `out/model_15m_chat.pt` (~15 M params, ~60 MB fp32 / ~15 MB INT8, Llama-3 style)
  ships with the repository, so this script never trains anything: it checks your
  Python + PyTorch, auto-picks the INT8-quantized checkpoint when available, then
  samples. That is the whole workflow - training happened once, on another machine,
  and its result is committed here (README section "Generate text (CPU, no training)").

  In -Chat mode the model plans inside a <think> ... </think> block on questions
  that need several steps (the terminal prints the trace dimmed and the answer
  normally), and arithmetic, word counts and date questions are answered by
  tools.py exactly instead of being guessed at. -Doc FILE indexes a local
  .txt/.md file in memory and answers from it (BM25 over chunks, no vector
  database, no network).

  First-time setup, once per machine:
      python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

.EXAMPLE
  .\run.ps1                                        # default prompt, 150 tokens, temp 0.8
  .\run.ps1 -Prompt "Once upon a time" -N 250
  .\run.ps1 -Interactive                           # prompts without reloading weights
  .\run.ps1 -Chat                                  # instruction-tuned multi-turn chatbot
  .\run.ps1 -Chat -Persona pirate                   # ...as a salty sea dog (friendly|pirate|math|storyteller|reasoner|none)
  .\run.ps1 -Chat -SystemPrompt "You are a robot."  # ...with your own system prompt
  .\run.ps1 -Chat -Doc .\notes.md                   # ...answering from your own document (local RAG)
  .\run.ps1 -Chat -NoTools                          # ...without the calculator/python hook
  .\run.ps1 -Seed 4294967295                       # reproducible output
  .\run.ps1 -Python "C:\Python312\python.exe"      # pick a specific interpreter
  .\run.ps1 -Int8                                  # force INT8 quantized checkpoint

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
    [string]$Ckpt   = "",          # empty = auto-select (15m -> 1.9m, int8 preferred)
    [string]$Python = "python",
    [double]$RepeatPenalty = 1.15,
    [string]$Persona = "",        # chat system prompt preset (friendly|pirate|math|storyteller|none)
    [string]$SystemPrompt = "",   # chat system prompt, free text (wins over -Persona)
    [switch]$Interactive,
    [switch]$Chat,
    [switch]$NoTools,             # disable the calculator hook (math goes to the model)
    [string]$Doc = "",            # chat: answer from this .txt/.md file (local retrieval)
    [switch]$NoRag,               # chat: disable document retrieval entirely
    [switch]$Int8
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

# --- 3. pick a checkpoint: 15M preferred, INT8 preferred -------------------
if ($Ckpt -eq "") {
    $suffix = $(if ($Int8) { "_int8.pt" } else { ".pt" })
    $base = $(if ($Chat) { "out/model_15m_chat" } else { "out/model_15m_best" })
    $fallback = $(if ($Chat) { "out/model_chat" } else { "out/model_best" })
    # Auto-prefer INT8 if it exists and -Int8 wasn't forced off
    if (-not $Int8) {
        foreach ($c in @("out/model_15m_chat_int8.pt", "out/model_chat_int8.pt",
                        "out/model_15m_best_int8.pt", "out/model_best_int8.pt")) {
            if ((-not $Chat -and ($c -like "*best*")) -or ($Chat -and ($c -like "*chat*"))) {
                if (Test-Path -LiteralPath $c) { $Ckpt = $c; break }
            }
        }
    }
    if ($Ckpt -eq "") {
        foreach ($c in @($base + $suffix, $fallback + $suffix,
                         $base + ".pt", $fallback + ".pt")) {
            if (Test-Path -LiteralPath $c) { $Ckpt = $c; break }
        }
    }
}
if (-not (Test-Path -LiteralPath $Ckpt)) {
    Write-Host "[run] Checkpoint '$Ckpt' not found." -ForegroundColor Yellow
    Write-Host "      Run git pull to fetch the committed checkpoint (~15 MB for the 15M model)."
    exit 1
}

# --- 4. generate ------------------------------------------------------------
Write-Host "[run] generating with $exe from $Ckpt (Ctrl+C stops it)" -ForegroundColor Cyan
$genArgs = @("generate.py", "--ckpt", $Ckpt, "--prompt", $Prompt, "--n", $N,
             "--temp", $Temp, "--top-k", $TopK, "--top-p", $TopP, "--repeat-penalty", $RepeatPenalty)
if ($Int8) { $genArgs += "--quantized" }
if ($Interactive) { $genArgs += "--interactive" }
if ($Chat) { $genArgs += "--chat" }
if ($Persona) { $genArgs += @("--persona", $Persona) }
if ($SystemPrompt) { $genArgs += @("--system", $SystemPrompt) }
if ($NoTools) { $genArgs += "--no-tools" }
if ($Doc) { $genArgs += @("--doc", $Doc) }
if ($NoRag) { $genArgs += "--no-rag" }
if ($Seed -ge 0) { $genArgs += @("--seed", $Seed) }

& $exe @genArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "[run] generation failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[run] done. Change the prompt with:  .\run.ps1 -Prompt `"your text`" -N 250" -ForegroundColor DarkGray
