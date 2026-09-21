# LLM from scratch — data → tokenizer → transformer → training → eval → generation

A complete, working, small-scale language model built without `transformers`,
`datasets`, `sentencepiece` or `tiktoken`. Read **`GUIDE.md`** for the full
walkthrough with design decisions; this file is the how-to-run reference.

Every artefact in this repo was produced by running these commands on a
2-core / 4 GB CPU box. Model: **1.93 M parameters**, Llama-3 style architecture
(RMSNorm, RoPE, SwiGLU, `scaled_dot_product_attention`), 4096-token byte-level
BPE vocab, 256-token context, trained on **20 MB of TinyStories** (~4.3 M tokens).
Numbers for the shipped checkpoint are in **`out/MODEL_CARD.md`**.

**The trained checkpoint is committed** (`out/model_best.pt`, 7.7 MB), so you do
not have to train anything to use it — a fresh clone can generate text with
PyTorch alone (2.0+). See *Generate text (CPU, no training)* below.

## Generate text (CPU, no training)

**Once per machine** — the CPU build of PyTorch (~200 MB, no CUDA):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Then, on a fresh clone** — pull the committed weights and generate:

```bash
git pull          # brings out/model_best.pt (7.7 MB) + out/config.json
.\run.ps1         # Windows PowerShell: TinyStories prompt, 150 tokens, temp 0.8
```

On Windows you can also just double-click **`run.bat`**, or pass arguments:

```powershell
.\run.ps1                                     # "Once upon a time, there was a little girl named Lily"
.\run.ps1 -Prompt "Once upon a time, Lily found a little kitten" -N 150 -Temp 0.7
.\run.ps1 -Interactive                        # keep weights loaded between prompts
.\run.ps1 -Seed 4294967295                    # reproducible sample (64-bit seeds supported)
.\run.ps1 -TopK 40 -TopP 0.9 -RepeatPenalty 1.15
.\run.ps1 -Python C:\Python312\python.exe     # pick a specific interpreter
```

`run.ps1` never trains: it checks Python + PyTorch, checks the checkpoint is
present, then calls `generate.py`. `run.bat` is the same thing for cmd/double-click
(execution-policy bypass included, forwards every argument). Under the hood:

```bash
python generate.py --prompt "Once upon a time, Lily found a little kitten" --n 150 --temp 0.7 --seed 1337
```

### Instruction-tuned chat

The repository also ships a 250-step supervised fine-tune of the TinyStories
checkpoint. Start a multi-turn conversation with:

```powershell
.\run.ps1 -Chat
```

or directly:

```bash
python generate.py --chat
```

Chat mode selects `out/model_chat.pt` and `tokenizer/chat`, wraps every turn as
`<|im_start|>user\n...<|im_end|>` / `<|im_start|>assistant\n...`, retains
complete turns in the 256-token context, and stops as soon as `<|im_end|>` is
sampled. Type `/clear` to discard history or `/exit` to leave. The base
continuation mode and `out/model_best.pt` remain unchanged.

The committed `data/sft/chat_sft.jsonl` contains 6,000 deterministic synthetic
instruction-response pairs (greetings, children's animal questions, everyday
explanations, kindness, arithmetic, and tiny stories). Each row includes the
prompt, response, and exact serialized `text`. Rebuild and fine-tune it with:

```bash
python 07_make_sft.py                         # writes 6,000 JSONL rows
python 08_sft.py --iters 250 --lr 1e-4        # cosine decay; writes model_chat.pt
```

`08_sft.py` appends the two chat markers without changing any pretrained token
ID, expands the tied embedding by two rows, and masks user text, role markers,
prior turns, and padding with label `-100`. Thus cross-entropy is computed only
on the current assistant response and its end marker. It also trains on
synthetic two-turn arrangements so retained history works at inference. The
shipped run reached validation loss 0.367 (perplexity 1.44) on 256 held-out
examples; see `out/chat_log.jsonl` and `out/chat_run_config.json`.

Text streams to the terminal as tokens are generated (including with `--quiet`,
which prints only prompt + continuation). Byte-level UTF-8 fragments are buffered
until a complete character is ready, so Unicode doesn't break during streaming.

For a persistent prompt session:

```bash
python generate.py --interactive --top-k 40 --top-p 0.9 --repeat-penalty 1.15
```

The model and tokenizer load **once**. Each prompt is an independent continuation,
not a chat with accumulated history; `--prompt` is ignored in interactive mode.
Blank input is skipped. Type `/exit` or `/quit` (also `exit` / `quit`), or use EOF
(Ctrl+D on Unix, Ctrl+Z then Enter on Windows) or Ctrl+C to exit cleanly.
Ctrl+C during generation also exits the session.

Sampling defaults: **top-k 40**, **top-p 0.9**, **repetition penalty 1.15**.
Use `--top-k 0`, `--top-p 1`, or `--repeat-penalty 1` to disable each filter;
`--temp 0` selects greedy decoding (still applying the repetition penalty).
The penalty lowers the score of each distinct token already in the prompt/output.
`--repetition-penalty` is an alias for `--repeat-penalty`.
A fixed `--seed` is reused for each prompt, making repeated prompts reproducible;
without it each prompt gets a new seed, printed unless `--quiet` is set.
Generation stops at the trained context limit. Overlong prompts use their last
255 tokens with the shipped 256-token model; no weights are updated.

`generate.py` is the *inference-only* CLI: its only third-party dependency is
torch (no numpy, no matplotlib; the tokenizer falls back to stdlib `re`), it runs
on the CPU, uses the KV cache, and reads the architecture from `out/config.json`
next to the weights. Note that generation needs **torch ≥ 2.0** (the model calls
`F.scaled_dot_product_attention`; older versions have no such function).
Numbers and sample output for the shipped checkpoint: **`out/MODEL_CARD.md`**.
`sample.py` is the richer research CLI (cached-vs-uncached comparison, extra
knobs) for machines that also have the training environment.

## Layout — files are numbered in pipeline order

```
00_tinystories.py      stream + clean TinyStories (2.2 GB raw) into ~20 MB of stories
01_tokenizer.py        train a byte-level BPE from scratch  (+ self-tests)
02_preprocess.py       pack the token stream into uint16 blocks, story-level train/val split
03_model.py            Llama-3 style decoder-only transformer (+ invariant tests)
04_train.py            training loop: AdamW, cosine LR, grad clip, fp32, eval, checkpoints
05_evaluate.py         loss / ppl / top-1 / bits-per-byte + unigram & bigram baselines + curves
06_generate.py         temperature, top-k, top-p, repeat penalty, KV cache
07_make_sft.py         deterministic 6,000-example children's assistant dataset
08_sft.py              response-only-loss SFT from model_best.pt (250 steps)
generate.py            continuation + multi-turn chat CLI (CPU, torch-only deps)
run.ps1 / run.bat      one-click Windows launchers for generate.py
sample.py              research generation CLI
_stage.py              import shim (module names can't start with a digit)
data/tinystories/      the cleaned corpus + provenance (committed)
data/packed/           tokenised uint16 blocks the trainer reads (regenerated, git-ignored)
out/                   the trained checkpoint + its config, metrics and model card
tests/                 property-based tests for corpus cleaning, tokenizer, model, sampler and CLI
```

## Reproduce from scratch

```bash
pip install torch numpy regex            # + matplotlib for the loss curve

# 0. corpus: stream ~32 MB of the official TinyStories *.txt (2.2 GB on disk, so
#    range-read and stop early) and keep 20 MB of clean, deduplicated stories
#    (~10 s, no full download). If the Hub is unreachable, --source github streams
#    a verified 5 MB sample fixture, and --source local reads a file you supply.
python 00_tinystories.py --source hf-http --out data/tinystories --max-bytes $((32<<20)) --max-chars 20000000

# 1. tokenizer: 4096-token byte-level BPE over 4 M characters of it  (~25 s)
python 01_tokenizer.py --corpus data/tinystories/corpus.txt --out tokenizer --vocab 4096 --special '<|doc_end|>' --train-chars 4000000

# 2. dataset: pack the token stream into 256+1-wide uint16 blocks  (~6 s)
python 02_preprocess.py --corpus data/tinystories/corpus.txt --tokenizer tokenizer --out data/packed --block-size 256 --mode stream

# 3. sanity-check the architecture's invariants  (seconds)
python 03_model.py

# 4. train the shipped checkpoint: 1400 steps x 8,192 tokens = 11.5 M tokens,
#    ~33 min on 2 CPU cores at ~8,600 tok/s (fp32; see --autocast below)
python 04_train.py --data-dir data/packed --out-dir out --iters 1400 --warmup 100 \
    --batch-size 32 --d-model 160 --n-layer 4 --n-head 5 --lr 1e-3 --block-size 256 \
    --dropout 0.0 --eval-every 100 --eval-iters 4 --autocast none --seed 1337

# 5. evaluate (val ppl, top-1, bits/byte, baselines, loss curve)
python 05_evaluate.py --ckpt out/model_best.pt --data-dir data/packed --autocast none

# 6. generate
python generate.py --prompt "Once upon a time, Lily found a little kitten" --n 150 --temp 0.7

# all tests
bash tests/run_all.sh
```

Steps are ordered by dependency: raw text → corpus → tokenizer → dataset.
`04_train.py` refuses to write into an `out/` that already holds a `model_best*.pt`
unless you pass `--resume` — that guard exists because it once overwrote a
trained model.

`--autocast none` matters on CPU: bf16 is emulated on CPUs without
AVX512-BF16/AMX and measured **~2.2x SLOWER per step** than fp32 here
(0.95 s/step fp32 vs 2.6 s/step bf16 for this model). On a GPU, use bf16.
The books in `data/raw/` are still usable with
`python 02_preprocess.py --mode docs` (the pre-TinyStories pipeline).

## Outputs you should see

| artefact | meaning |
|---|---|
| `data/tinystories/corpus.txt` | 25,147 clean stories, 20 MB, `<|doc_end|>` separated — **committed** |
| `data/tinystories/stats.json` | provenance: URL, sha256, story/char counts, what was rejected |
| `tokenizer/meta.json` | `{"vocab_size": 4096, "n_merges": 3839}` |
| `data/packed/{train,val}.npy` | uint16 blocks of width 256+1, story-level val split (regenerated by step 2, git-ignored) |
| `out/model_best.pt` + `config.json`/`config.txt` | weights **and** the config needed to read them — **committed** |
| `out/eval.json` | metrics and baselines — **committed** |
| `out/log.jsonl` | per-step loss/lr/gnorm + eval rows — plot with 05 — **committed** |
| `out/MODEL_CARD.md` | provenance, metrics, honest limits of the shipped checkpoint |
| `out/opt.pt`, `out/model_*.pt` | optimizer state + per-eval snapshots (git-ignored, regenerable) |

## The four numbers that tell you whether the pipeline is correct

1. `ln(4096) = 8.32` — loss at step 0. Measured: **8.3584**. Much higher ⇒ head/init is broken.
2. `4.49 chars/token` — tokenizer compression on TinyStories. ~1.0 ⇒ merges aren't being
   applied to the vocabulary (this was a real bug here; see GUIDE.md §3).
3. Final val loss **2.5826** (ppl **13.23**), top-1 **45.26%**, **0.830** bits/byte on the
   held-out stories (the trainer's own smaller eval window reports 2.5663 / 13.02) — see
   `out/MODEL_CARD.md` for the full table.
4. Val loss must *beat the bigram Markov baseline* — **52.84** ppl on this split, the floor
   for "has learned more than one token of context". The shipped run crosses it before step
   100 and ends **75% below it in perplexity / 35% below it in cross-entropy** (unigram
   sits at 613.28 ppl).

## Scaling knobs

`--d-model --n-layer --n-head --block-size --vocab --iters --batch-size
--grad-accum --lr --autocast --rope-base`. Rough cost model: step time ∝
`batch × block × (params_non_embed + vocab × d_model)`.
