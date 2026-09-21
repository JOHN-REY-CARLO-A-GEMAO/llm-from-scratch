# Model card — `out/model_best.pt`

A 1.9 M-parameter decoder-only transformer, trained from scratch (no `transformers`,
no pretrained weights) on 1.44 MB of public-domain English prose. It is committed to
this repository so that **inference needs no training**: clone, `pip install torch`,
run `run.ps1`.

## What it is

| | |
|---|---|
| parameters | 1,897,780 (35% of them the tied token embedding) |
| file | `out/model_best.pt`, 7.6 MB, fp32 `state_dict` |
| architecture | RoPE + RMSNorm + SwiGLU, pre-norm, tied embeddings (`03_model.py`) |
| config | `d_model 160`, 4 layers, 5 heads (head dim 32), context 256, vocab 4096, dropout 0 |
| tokenizer | byte-level BPE, 4096 tokens, 3,839 merges + `<|doc_end|>` (`tokenizer/`) |
| corpus | 3 Project Gutenberg books — *Alice in Wonderland*, *Pride and Prejudice*, *The Adventures of Sherlock Holmes* |
| training | 400 steps × 8,192 tokens = 3.28 M tokens (~11 passes over the 0.30 M-token train split) |

## The exact command that produced it

Run on a 2-core / 4 GB CPU container, fp32, ≈8 minutes wall clock, ~7,300 tok/s:

```bash
python 04_train.py --iters 400 --warmup 40 --batch-size 32 --grad-accum 1 \
    --d-model 160 --n-layer 4 --n-head 5 --lr 6e-4 --block-size 256 \
    --dropout 0.0 --eval-every 50 --autocast none --seed 1337 --out-dir out

python 05_evaluate.py --ckpt out/model_best.pt --autocast none
```

`--autocast none` is deliberate: this CPU has AVX-512 but no AVX512-BF16/AMX, and
PyTorch emulates bf16 there — measured 5× *slower* per step than fp32 (8.84 ms vs
1.84 ms on a 1600×1600 matmul). On hardware with native bf16, drop the flag.

## Metrics (all 60 validation blocks, fp32, held-out documents)

| metric | this model | unigram | bigram Markov | uniform |
|---|---|---|---|---|
| val loss (nats/token) | **4.4949** | 6.6016 | 5.1868 | 8.3178 |
| perplexity | **89.56** | 736.25 | 178.89 | 4,096 |
| top-1 next-token acc | **24.73%** | — | — | — |
| bits per byte | **1.712** | — | — | — |

It beats the bigram Markov chain — the floor a model must clear before it has
demonstrably learned more than one token of context — by **50% in perplexity**
(13.3% in cross-entropy). Validation perplexity during training:

```
iter  50   578.05
iter 100   213.88     <- still worse than the bigram chain
iter 150   139.99     <- crosses the bigram floor here
iter 200   113.36
iter 250   101.11
iter 300    94.42
iter 350    90.83
iter 400    89.56     <- out/model_best.pt
```

## Sample (prompt "It was a dark and stormy night", temp 0.8, 150 tokens, seed 1337)

```
It was a dark and stormy night at the cars of his taced at the lock that his a man might be
only all an good, and the tramoning fug.”

“I was so much of some unppeconer.”

“And it did you be a few few minutes,’ said he. “How is sofgs for the stlient. You think
which I have been in the pusing?’

“If I cannot be in the clue or in his hand?” He cried. “My bay to look. The rooms which may
be as much from Miss Holmes, and I can ask that for
```

## What it does not do — read this before judging the output

* **It is not an assistant.** No instruction tuning, no RLHF: it continues text, it
  does not answer questions or follow commands.
* **It knows no facts.** 1.9 M parameters and 3.3 M training tokens cannot encode
  world knowledge. Expect Victorian paragraph layout, dialogue punctuation and
  plausible-looking function words; expect garbled content words (`tramoning`, `fug`).
* **Out-of-distribution prompts degrade fast.** The corpus is three 19th-century
  novels; modern text or code gets you gibberish with a Victorian accent.
* The corpus is public domain (Project Gutenberg), so the checkpoint is free to
  share and reuse.

## Running it

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # once
.\run.ps1                                                                        # 150 tokens
.\run.ps1 -Prompt "Once upon a time" -N 250 -Temp 0.9 -Seed 7
```

`run.bat` is the double-click equivalent; `generate.py` is the CLI both call.
If you would rather not carry a 7.6 MB binary in git, delete it and retrain —
the command above reproduces it in ~8 minutes on two CPU cores.
