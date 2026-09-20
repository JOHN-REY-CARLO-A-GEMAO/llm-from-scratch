# LLM from scratch — data → tokenizer → transformer → training → eval → generation

A complete, working, small-scale language model built without `transformers`,
`datasets`, `sentencepiece` or `tiktoken`. Read **`GUIDE.md`** for the full
walkthrough with design decisions; this file is the how-to-run reference.

Every artefact in this repo was produced by running these commands on a
2-core / 2 GB CPU box. Model: **1.9 M parameters**, 4096 vocab, 256-token
context, trained on **1.44 MB** of public-domain English prose.

## Layout — files are numbered in pipeline order

```
data/raw/              3 Project Gutenberg books (input)
01_tokenizer.py        train a byte-level BPE from scratch  (+ self-tests)
02_preprocess.py       clean corpus, split by document, pack blocks, memmap
03_model.py            RoPE + RMSNorm + SwiGLU decoder-only transformer (+ invariant tests)
04_train.py            training loop: AdamW, cosine LR, grad clip, bf16, eval, checkpoints
05_evaluate.py         loss / ppl / top-1 / bits-per-byte + unigram & bigram baselines + curves
06_generate.py         temperature, top-k, top-p, repeat penalty, KV cache
sample.py              generation CLI
_stage.py              import shim (module names can't start with a digit)
tests/                 property-based tests for the tokenizer and the sampler
```

## Reproduce from scratch

```bash
pip install torch numpy regex            # + matplotlib for the loss curve

# 1. corpus: clean data/raw/*.txt -> data/corpus.txt   (see GUIDE.md §2 for the download)
python -c "import sys; sys.path.insert(0,"."); from importlib import import_module as I; I("02_preprocess").write_corpus("data/raw","data/corpus.txt")"

# 2. tokenizer  (~30 s)
python 01_tokenizer.py --corpus data/corpus.txt --out tokenizer --vocab 4096 --train-chars 400000

# 3. dataset (needs the tokenizer from step 2)
python 02_preprocess.py --corpus data/corpus.txt --tokenizer tokenizer --out data/shuffled --block-size 256

# 3. sanity-check the architecture's invariants  (seconds)
python 03_model.py

# 4. train.
#    2 GB RAM box -> batch 8 + grad_accum 4 (effective 32), minutes-to-hours.
#    a real GPU  -> batch 64+, 4k context, 1000s of steps; same script.
#    NB on CPU, grad_accum costs ~4x throughput (see GUIDE.md 6) - prefer the
#    biggest single batch that does not swap.
python 04_train.py --iters 150 --warmup 30 --batch-size 8 --grad-accum 4 \
    --eval-every 75 --d-model 160 --n-layer 4 --n-head 5 --lr 6e-4 --out-dir out

# 5. evaluate (val ppl, top-1, bits/byte, baselines, loss curve)
python 05_evaluate.py --ckpt out/model_best.pt --plot out/curves.png

# 6. generate
python sample.py --prompt "It was a dark and stormy night" --n 200 --temp 0.8 --compare

# all tests
bash tests/run_all.sh
```

Steps are ordered by dependency: corpus file → tokenizer → dataset.

## One-command run (Windows)

`run.ps1` trains the small **`out_fast`** model on first use (30 iters,
d_model 96 / 2 layers / 3 heads — under a minute on a laptop CPU) and then
generates; later runs reuse the checkpoint and just sample.

```powershell
.\run.ps1                                  # default prompt, 200 tokens
.\run.ps1 -Prompt "Once upon a time" -N 150
.\run.ps1 -Eval                            # also print val ppl
.\run.ps1 -Retrain                         # wipe out_fast and train fresh
```

If the execution policy blocks `.ps1` scripts, or you prefer double-click /
cmd: **`run.bat`** (forwards all arguments). Checkpoints stay under `out_fast/`,
which is git-ignored.

## Outputs you should see

| artefact | meaning |
|---|---|
| `tokenizer/meta.json` | `{"vocab_size": 4096, "n_merges": 3839}` |
| `data/shuffled/{train,val}.npy` | uint16 blocks of width 256+1 |
| `out/model_best.pt` + `config.txt` | weights **and** the config needed to read them |
| `out/log.jsonl` | per-step loss/lr/gnorm + eval rows — plot with 05 |
| `out/eval.json` | metrics and baselines |

## The four numbers that tell you whether the pipeline is correct

1. `ln(4096) = 8.32` — loss at step 0. Much higher ⇒ head/init is broken.
2. `3.81` chars/token — tokenizer compression. ~1.0 ⇒ merges aren't being
   applied to the vocabulary (this was a real bug here; see GUIDE.md §3).
3. Final val loss ~2.5–3.5 (ppl ~12–33) on this corpus.
4. Val loss must *beat the bigram Markov baseline* (**ppl 179** on this split). Until it does,
   the model has not learned context, whatever the curve looks like. At iter 100 this run is
   still at 408 — see GUIDE.md §9 for the honest, mid-training numbers.

## Scaling knobs

`--d-model --n-layer --n-head --block-size --vocab --iters --batch-size
--grad-accum --lr`. Rough cost model: step time ∝ `batch × block × (params_non_embed + vocab × d_model)`.
