# LLM from scratch — data → tokenizer → transformer → training → eval → generation

A complete, working, small-scale language model built without `transformers`,
`datasets`, `sentencepiece` or `tiktoken`. Read **`GUIDE.md`** for the full
walkthrough with design decisions; this file is the how-to-run reference.

Every artefact in this repo was produced by running these commands on a
2-core / 4 GB CPU box. Model: **1.9 M parameters**, 4096 vocab, 256-token
context, trained on **1.44 MB** of public-domain English prose.

**The trained checkpoint is committed** (`out/model_best.pt`, 7.6 MB), so you do
not have to train anything to use it — a fresh clone can generate text with
PyTorch alone. See *Generate text (CPU, no training)* below.

## Generate text (CPU, no training)

**Once per machine** — the CPU build of PyTorch (~200 MB, no CUDA):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Then, on a fresh clone** — pull the committed weights and generate:

```bash
git pull          # brings out/model_best.pt (7.6 MB) + out/config.json
.\run.ps1         # Windows PowerShell: default prompt, 150 tokens, temp 0.8
```

On Windows you can also just double-click **`run.bat`**, or pass arguments:

```powershell
.\run.ps1                                   # "It was a dark and stormy night", 150 tokens, temp 0.8
.\run.ps1 -Prompt "Once upon a time" -N 250 -Temp 0.9
.\run.ps1 -Seed 1337                        # reproducible sample
.\run.ps1 -Python C:\Python312\python.exe   # pick a specific interpreter
```

`run.ps1` never trains: it checks Python + PyTorch, checks the checkpoint is
present, then calls `generate.py`. `run.bat` is the same thing for cmd/double-click
(execution-policy bypass included, forwards every argument). Under the hood:

```bash
python generate.py --prompt "It was a dark and stormy night" --n 150 --temp 0.8 --seed 1337
```

`generate.py` is the *inference-only* CLI: its only third-party dependency is
torch (no numpy, no matplotlib; the tokenizer falls back to stdlib `re`), it runs
on the CPU, uses the KV cache, and reads the architecture from `out/config.json`
next to the weights. Numbers and sample output for the shipped checkpoint:
**`out/MODEL_CARD.md`**. `sample.py` is the richer research CLI (cached-vs-uncached
comparison, extra knobs) for machines that also have the training environment.

## Layout — files are numbered in pipeline order

```
data/raw/              3 Project Gutenberg books (input)
01_tokenizer.py        train a byte-level BPE from scratch  (+ self-tests)
02_preprocess.py       clean corpus, split by document, pack blocks, memmap
03_model.py            RoPE + RMSNorm + SwiGLU decoder-only transformer (+ invariant tests)
04_train.py            training loop: AdamW, cosine LR, grad clip, bf16, eval, checkpoints
05_evaluate.py         loss / ppl / top-1 / bits-per-byte + unigram & bigram baselines + curves
06_generate.py         temperature, top-k, top-p, repeat penalty, KV cache
generate.py            inference-only CLI (CPU, torch-only deps)   <- what your laptop runs
run.ps1 / run.bat      one-click Windows launchers for generate.py
sample.py              research generation CLI
_stage.py              import shim (module names can't start with a digit)
out/                   the trained checkpoint + its config, metrics and model card
tests/                 property-based tests for the tokenizer and the sampler
```

## Reproduce from scratch

```bash
pip install torch numpy regex            # + matplotlib for the loss curve

# 1. corpus: clean data/raw/*.txt -> data/corpus.txt   (see GUIDE.md §2 for the download)
python -c "import sys; sys.path.insert(0,\".\"); from importlib import import_module as I; I(\"02_preprocess\").write_corpus(\"data/raw\",\"data/corpus.txt\")"

# 2. tokenizer  (~30 s)
python 01_tokenizer.py --corpus data/corpus.txt --out tokenizer --vocab 4096 --train-chars 400000

# 3. dataset (needs the tokenizer from step 2)
python 02_preprocess.py --corpus data/corpus.txt --tokenizer tokenizer --out data/shuffled --block-size 256

# 3. sanity-check the architecture's invariants  (seconds)
python 03_model.py

# 4. train the shipped checkpoint.  400 steps x 8,192 tokens = 3.28 M tokens, ~8 min
#    on 2 CPU cores at ~7,300 tok/s. On a GPU: batch 64+, 4k context, 1000s of steps,
#    drop --autocast none (see below).
#    `--autocast none` matters on CPU: bf16 is emulated on CPUs without
#    AVX512-BF16/AMX and measured ~5x SLOWER per step than fp32 here.
python 04_train.py --iters 400 --warmup 40 --batch-size 32 --grad-accum 1 \
    --d-model 160 --n-layer 4 --n-head 5 --lr 6e-4 --block-size 256 \
    --dropout 0.0 --eval-every 50 --autocast none --seed 1337 --out-dir out

# 5. evaluate (val ppl, top-1, bits/byte, baselines, loss curve)
python 05_evaluate.py --ckpt out/model_best.pt --autocast none --plot out/curves.png

# 6. generate
python generate.py --prompt "It was a dark and stormy night" --n 150 --temp 0.8

# all tests
bash tests/run_all.sh
```

Steps are ordered by dependency: corpus file → tokenizer → dataset. `04_train.py`
refuses to write into an `out/` that already holds a `model_best*.pt` unless you
pass `--resume` — that guard exists because it once overwrote a trained model.

## Outputs you should see

| artefact | meaning |
|---|---|
| `tokenizer/meta.json` | `{"vocab_size": 4096, "n_merges": 3839}` |
| `data/shuffled/{train,val}.npy` | uint16 blocks of width 256+1 |
| `out/model_best.pt` + `config.txt`/`config.json` | weights **and** the config needed to read them — **committed** |
| `out/eval.json` | metrics and baselines — **committed** |
| `out/log.jsonl` | per-step loss/lr/gnorm + eval rows — plot with 05 — **committed** |
| `out/MODEL_CARD.md` | provenance, metrics, honest limits of the shipped checkpoint |
| `out/opt.pt`, `out/model_*.pt` | optimizer state + per-eval snapshots (git-ignored, regenerable) |

## The four numbers that tell you whether the pipeline is correct

1. `ln(4096) = 8.32` — loss at step 0. Measured: **8.307**. Much higher ⇒ head/init is broken.
2. `3.79` chars/token — tokenizer compression. ~1.0 ⇒ merges aren't being
   applied to the vocabulary (this was a real bug here; see GUIDE.md §3).
3. Final val loss **4.4949** (ppl **89.56**), top-1 24.7%, 1.712 bits/byte on the
   held-out documents.
4. Val loss must *beat the bigram Markov baseline* (**ppl 178.89** on this split) —
   it is the floor for "has learned more than one token of context". This run crosses
   it between iter 100 (213.9) and iter 150 (140.0) and ends **49.9% lower in
   perplexity** (13.3% lower cross-entropy) than the bigram chain.

## Scaling knobs

`--d-model --n-layer --n-head --block-size --vocab --iters --batch-size
--grad-accum --lr --autocast`. Rough cost model: step time ∝
`batch × block × (params_non_embed + vocab × d_model)`.
