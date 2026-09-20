# Building an LLM from scratch

**A complete, verified pipeline: data → tokenizer → transformer → training → evaluation → generation.**

Every code block in this guide is from the working files in this folder, and every number
reported was produced by actually running them. The model in `out/` was trained on the machine
writing this document (2 CPU cores, 2 GB RAM), so the run is small by design — the *pipeline* is
what transfers to scale, not the model size.

---

## 0. Set the right expectation first

| | this guide | GPT-3 class |
|---|---|---|
| parameters | 1.9 M | 175 B |
| training tokens | 0.37 M | 300 B |
| context window | 256 | 2048 |
| result | fluent sentences, no facts, no instruction-following | general ability |

Two consequences that shape every later decision:

1. **You are building a *language model*, not an assistant.** No amount of architecture work
   substitutes for tokens. Chinchilla's scaling law is roughly `L ≈ 1.69e-2 · N^-0.34 · D^-0.28`
   — a model that is 100,000× smaller than GPT-3 trained on 1,000,000× less data is not going to
   know anything about the world. It *will* learn English syntax, spelling, punctuation and
   author-specific style, and that is exactly what you want to see when validating your pipeline.
2. **Compute, not code, is the limiter.** On 2 CPU cores a training step costs real wall-clock
   seconds, so the sections below spend a lot of effort on things a GPU tutorial skips:
   pre-tokenising to a memory-mapped file, mixed precision on CPU, and choosing a vocabulary size
   your CPU can actually encode with.

**What you should see when it works.** An untrained model sits at loss ≈ `ln(vocab)` =
`ln(4096)` = 8.32 — that is the cross-entropy of a uniform guesser. If your step 0 loss is far
above that, your head is mis-scaled. A successful small run lands at loss 2.5–3.5 (ppl 12–33).
This number is the single most useful debugging signal in the whole build.

---

## 1. Environment

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel
pip install numpy regex            # regex: for \p{L} in the tokenizer's splitting pattern
pip install matplotlib             # optional: loss curves
```

Nothing else. Notably **no** `transformers`, `datasets`, or `sentencepiece` — those would defeat
the point. `tiktoken` is the one library whose output we reimplement (Stage 2) rather than use.

Check what you're really running on, because it changes every hyperparameter below:

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.get_num_threads())
```

`torch.get_num_threads()` returning `1` on a multi-core box is common in containers and quietly
makes a CPU run 2–8× slower; set it explicitly (`04_train.py` does `torch.set_num_threads(os.cpu_count())`).

---

## 2. Data collection

**Source.** Three public-domain books from Project Gutenberg (~1.4 MB of prose): *Sherlock Holmes*,
*Pride and Prejudice*, *Alice in Wonderland*. Small, clean, English, no licence friction.

```python
import urllib.request
for url, name in [("https://www.gutenberg.org/files/1661/1661-0.txt", "sherlock.txt"),
                  ("https://www.gutenberg.org/files/1342/1342-0.txt", "pride.txt"),
                  ("https://www.gutenberg.org/files/11/11-0.txt",     "alice.txt")]:
    urllib.request.urlretrieve(url, f"data/raw/{name}")
```

**Design decision: corpus choice dominates model choice at this scale.** With 1.4 MB you are
training a *style* model. If you want a model that reasons about code, use a code corpus; if you
want dialogue, use transcripts. Do not expect general knowledge from literature alone.

**Cleaning (`02_preprocess.py`).** Four things, in order:

1. **Strip container boilerplate** — Gutenberg headers/footers and licence text. If you leave it
   in, your model will reliably end every document with a licence notice, which looks like a
   training bug and isn't.
2. **Normalise line endings and whitespace** — CRLF→LF, tabs→spaces, collapse runs of blank lines.
3. **Unwrap hard line breaks.** Gutenberg wraps at ~70 columns. If you don't rejoin them, the
   model learns that a newline appears every 70 characters and will reproduce that in generation.
   This is the most common "why does my output have random line breaks" bug.
4. **Reflow into documents** of ≥2000 chars separated by a boundary token.

```python
def clean_gutenberg(text: str) -> str:
    m = HEADER_RE.search(text)
    if m: text = text[m.end():]
    m = FOOTER_RE.search(text)
    if m: text = text[:m.start()]
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r" +\n", "\n", text)          # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)       # at most one blank line
    return text.strip()
```

Measured result of this stage:

```
[data] corpus written: {'n_docs': 609, 'n_chars': 1439393,
                         'mean_doc_chars': 2363.5, 'n_lines': 12967}
```

**For real work:** deduplicate (exact + near-duplicate/minhash), filter by language and
perplexity, and enforce length. Dedup matters more than almost any architecture change —
repeated data inflates train loss while degrading generation, and it *silently corrupts* your
validation split if duplicates straddle the boundary.

---

## 3. Tokenizer

`01_tokenizer.py` — trains a byte-level BPE from scratch, no libraries.

### Why BPE, and why byte-level

- **Word tokenizer**: needs an UNK symbol; every name, typo, and number format outside the
  vocabulary becomes garbage; vocabulary must be enormous to keep UNK rare.
- **Character tokenizer**: tiny vocabulary (great) but sequences ~4× longer, so attention cost
  (quadratic in sequence length) roughly doubles-and-doubles-again, and the model spends capacity
  on spelling instead of meaning.
- **Byte-level BPE**: the atomic unit is a byte, so *no input can be unrepresentable*, and common
  substrings still become single tokens. This is why GPT-2 onward all use it.

### The algorithm, exactly

```python
def train(self, texts, verbose=True, report_every=500):
    # 1. pre-tokenise with the splitting pattern, count word frequencies once
    words = Counter()
    for t in texts:
        for m in PAT.finditer(bytes_to_unicode(t.encode("utf-8"))):
            words[m.group()] += 1

    # 2. per-word pair counts + one global frequency table
    splits   = [list(w) for w in words]
    pcounts  = [self._pair_counts(sp) for sp in splits]
    gfreq    = Counter()
    for sp, c in zip(splits, weights):
        for pr, n in self._pair_counts(sp).items():
            gfreq[pr] += n * c

    # 3. merge the most frequent adjacent pair, repeat
    while len(self.merges) < n_targets and gfreq:
        # tie-break: lexicographically smallest merged string (deterministic runs)
        best = min(gfreq.items(), key=lambda kv: (-kv[1], "".join(kv[0])))[0]
        first, second = best
        new_tok = first + second
        self.merges.append(best)
        for sp, pc, w in zip(splits, pcounts, weights):
            if first not in sp or second not in sp:
                continue
            before = self._pair_counts(sp)
            i, n = 0, len(sp)
            while i < n - 1:
                if sp[i] == first and sp[i+1] == second:
                    sp[i:i+2] = [new_tok]; n -= 1; i += 1
                else:
                    i += 1
            after = self._pair_counts(sp)
            # exact diff -> update both caches, no window arithmetic
            for pr, cnt in before.items():
                pc[pr] = pc.get(pr, 0) - cnt
                gfreq[pr] -= cnt * w
                if gfreq[pr] <= 0: del gfreq[pr]
                if pc[pr]  <= 0:   del pc[pr]
            for pr, cnt in after.items():
                pc[pr]   = pc.get(pr, 0) + cnt
                gfreq[pr] = gfreq.get(pr, 0) + cnt * w
    self._build_vocab()
```

The **splitting pattern** does as much work as the merges do. This is GPT-2's:

```python
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```

Note the leading ` ?` in the word classes: spaces attach to the *following* word. That's why
`" the"` is one token. And why contractions stay whole:

```
>>> PAT.findall("Hello, world! Do not go 42 é")
['Hello', ',', ' world', '!', ' Do', ' not', ' go', ' 42', ' é']
```

### Byte fallback: making any string representable

```python
_SAFE = [i for i in range(256) if 33 <= i <= 126 or 161 <= i <= 172 or 174 <= i <= 255]
BYTE_TO_UNICODE = {b: chr(b) for b in range(256) if b not in _SAFE}
BYTE_TO_UNICODE.update({b: chr(256 + i) for i, b in enumerate(range(256)) if b in _SAFE})
```

Every byte becomes exactly one character, from a set of code points that never occur in real text.
The result is `decode(encode(x)) == x` for *any* `x`, including invalid UTF-8.

### Two things that will bite you

**(a) Complexity.** The textbook loop rescans every word for every merge: O(n_merges × n_words).
On this machine that was **over 40 minutes** for 3,839 merges. Keeping per-word pair-count dicts
and updating `gfreq` incrementally brought it to **28 seconds** (138 merges/s) for identical output.

**(b) Incremental updates are easy to get subtly wrong.** My first fast version tried to update
pair counts inside a window `[i-1, i+3)`. It produced *3,839 merges using only 11 distinct pairs*
and a vocabulary of 268 instead of 4,096. The reason: **splicing shortens the word**, so window
bounds computed before the edit no longer cover the affected region; stale counts survived, and
the loop happily re-merged a pair that no longer existed. The fix is to diff exact whole-word
counts (same asymptotic cost, no window arithmetic). `check_invariants()` asserts the caches
still describe the data after training.

I verified correctness the only way that counts: a naïve O(n²) reference implementation, run
side by side. **21/21 random corpora × 3 vocab sizes produced byte-identical merge sequences.**

### Vocabulary size — the decision people get wrong

Bigger vocabulary is not simply better:

- fewer tokens per character → shorter sequences → cheaper attention, but
- more parameters in the embedding/head (vocab × d_model), and each token is seen fewer times,
  so its embedding is undertrained;
- and your tokenizer has to *encode* with it — a pure-Python BPE is ~1000× slower than
  HuggingFace's Rust `tokenizers`, which caps the merges you can afford at 100% vocab.

**4,096 was the right choice for 1.4 MB of text.** Rough guide: you want ≥ ~50 tokens of training
data per vocabulary entry, and 1.4M chars / 4 ≈ 380k tokens / 4096 ≈ 90. Fine. 50k (GPT-2's size)
here would mean 7 tokens per entry — mostly dead embeddings.

Measured outcome:

```
[tok] final vocab = 4096
[tok] self-test passed: round-trip on text + arbitrary bytes; no UNK path needed
[tok] 42 tokens for 160 chars -> 3.81 chars/token
```

**3.81 chars/token** — GPT-2's 50k BPE gets ~4.0 on English. A 4k vocabulary at 3.81 is the
payoff of tuning size to corpus. (Note: my first measured value was 1.09 chars/token, which was
not a tokenizer flaw but the 268-vocab bug above — a reminder to always measure compression.)

```bash
python 01_tokenizer.py --corpus data/corpus.txt --out tokenizer --vocab 4096 --train-chars 400000
```

`--train-chars` caps the sample because merge statistics saturate quickly — training a tokenizer
on 400 KB vs 1.4 MB gives near-identical merges for a fraction of the cost.

---

## 4. Preprocessing into training tensors

`02_preprocess.py`. Four decisions, each preventing a specific failure:

**1. Split by document, never by line.** A random line split leaks neighbouring sentences of the
same book into validation, and val loss looks 30% better than reality. 95/5 by document.

**2. One document per block, with a boundary token.** Padding short documents onto each other
teaches the model that topics teleport mid-paragraph. Instead each training window sits inside one
document, and `<|doc_end|>` marks the seam so the model can learn to restart its attention.

**3. Pre-tokenise to a memory-mapped file.** Tokenising inside the training loop is the #1 reason
a from-scratch run "feels slow": your accelerators starve while Python does string work. Here the
loop only does `np.memmap` slice reads. Storage is `uint16` (halves I/O; enough for vocab < 65536).

**4. Cast to int64 at the last moment.** `nn.Embedding` refuses uint16:
`RuntimeError: Expected tensor for argument #1 'indices' to have one of ... Long, Int; but got CPUUInt16Type`.
Convert once, in `get_batch`, not in the model.

```python
blk = torch.from_numpy(np.asarray(data[ix], dtype=np.int64))
return blk[:, :-1].to(device), blk[:, 1:].to(device)   # x, y — the shift is the supervision
```

**Sanity checks that catch most preprocessing bugs** (built into the script, and worth keeping):

```python
assert arr.max() < tok.vocab, "token id out of vocab range"
assert arr.shape[1] == block_size + 1
x, y = arr[:, :-1], arr[:, 1:]
assert np.array_equal(x[:, 1:], y[:, :-1]), "targets are not the inputs shifted by one"
```

That last one is the whole objective function in one line: *the target is the input shifted left
by one.* If that's wrong, your model learns nothing and your loss will hover near `ln(V)` forever.

```bash
python 02_preprocess.py --corpus data/corpus.txt --tokenizer tokenizer --out data/shuffled --block-size 256
[data] train: 371,354 tokens, 1,182 blocks of 257
[data] val:    18,759 tokens,    60 blocks of 257
[data] sanity: OK
```

**One honest trade-off, measured rather than assumed.** I first guessed that short documents were
being discarded; the real numbers are more interesting. With `block_size=256`:

```
docs=609   total tokens=390,113
documents too short for one 256 block: 0/609 = 0.0%      <- nothing is dropped for length
blocks produced: 1,242   covering 317,952 tokens         <- 18.5% still never gets trained on
```

The loss is not short documents — every document here clears 256 tokens — it's the **per-document
remainder**: the trailing 0–255 tokens of each document that don't fill a whole block. 72,161
tokens, 18.5% of the corpus, vanish. The standard fix is **sequence packing** across documents
with a document-aware attention mask, which recovers that 18.5% and is the single best
data-efficiency change you can make here (see §10).

---

## 5. Model architecture

`03_model.py` — a decoder-only transformer, 4 layers × 5 heads × d_model 160.

### The shape of one block

```python
class Block(nn.Module):
    def forward(self, x, cos, sin, mask, kv_cache=None):
        x = x + self.attn(self.ln1(x), cos, sin, mask, kv_cache)   # pre-norm residual
        x = x + self.mlp(self.ln2(x))
        return x
```

Pre-norm (`x + f(norm(x))`) rather than GPT-2's post-norm (`norm(x + f(x))`): far more stable with
depth, tolerates less warmup. Every modern training recipe uses it.

### Attention

```python
q, k, v = self.qkv(x).split(D, dim=2)
q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # [B,H,T,Dh]
k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)         # rotate by position
q = q * (1.0 / math.sqrt(self.head_dim))                         # scale BEFORE attention

att = q @ k.transpose(-2, -1) + self.attn_bias + mask            # mask = additive -inf triangle
att = self.drop(att.softmax(dim=-1))
y = (att @ v).transpose(1, 2).reshape(B, T, D)
return self.proj(y)
```

Design notes:

- **Causality is a mask, not a loop.** `mask` is a precomputed additive `-inf` upper triangle
  (`torch.triu(..., diagonal=1 + start)`), so one matrix multiply handles the whole sequence.
  Without it, position *t* could read the answer at *t+1* — training loss would look spectacular
  and generation would be gibberish. We test for it (below).
- **The `1/sqrt(head_dim)` scale** keeps pre-softmax logits at O(1). Apply it to `q` once instead
  of dividing the `T×T` matrix: fewer FLOPs, identical math.
- **RMSNorm instead of LayerNorm**: `x · rsqrt(mean(x²)+eps) · g`, skipping mean-centring. Same
  benefit, fewer ops, no bias to tune. Computed in fp32 then cast back — in bf16 `mean(x²)` is
  accurate enough to matter but the cast matters more.
- **RoPE instead of learned absolute position embeddings**: no parameter table, positions are
  *relative* (a query at 50 and a key at 47 attend the same way regardless of where the sequence
  starts), and it degrades gracefully past the training window instead of catastrophically.
  Learned `wpe`, as in GPT-2, is the simpler alternative — fine at this scale, strictly worse for
  length extrapolation.
- **No output bias on `proj`, but a learnable per-head attention bias.** QKV biases mostly cancel
  in the softmax; the per-head logit bias (`attn_bias`) acts as a learnable temperature and helps
  attention sharpen early in training. Costs nothing at inference.

### MLP: SwiGLU

```python
hidden = int(2 * (4 * d_model) / 3); hidden = 8 * ((hidden + 7) // 8)   # 8/3 of 4x, rounded to 8
return self.down(F.silu(self.gate(x)) * self.up(x))
```

Gated linear unit with SiLU. Three weight matrices instead of two, so width shrinks by 2/3 to
keep FLOPs equal to a 4× ReLU MLP — and at matched FLOPs it wins. Rounding to a multiple of 8
matters for tensor-core efficiency on GPUs (and vectorisation on CPUs).

### Tied embeddings

```python
self.lm_head.weight = self.wte.weight
```

One matrix serves input and output. Saves `vocab × d_model` parameters (1.4 M of 1.9 M here!) and
measurably helps small models, where the output distribution and the input representation want
similar structure. Cost: the two roles pull the matrix in different directions, which is why very
large models untie them.

### Initialisation — the part people skip

```python
self.apply(self._init_weights)          # N(0, 0.02) for Linear/Embedding, zero biases, unit norm gains
for block in self.blocks:
    nn.init.zeros_(block.attn.proj.weight)      # zero-init every residual-branch output
    nn.init.zeros_(block.mlp.down.weight)
scale = 1.0 / math.sqrt(2 * cfg.n_layer)        # control residual-stream variance growth
for block in self.blocks:
    for p in (block.attn.proj.weight, block.mlp.down.weight):
        p.data.mul_(scale)
```

Zero-initialising the last projection of each branch makes every residual branch the identity at
step 0, so the network *is* its own skip stream initially and cannot blow up before it learns.
The `1/sqrt(2·n_layer)` factor stops the residual stream's variance from growing with depth.
Together these are why a from-scratch transformer trains stably without exotic tricks.

`std=0.02` is not folklore: with tied embeddings it puts initial logits near zero, so initial
cross-entropy lands within a hair of `ln(V)` — the loss of a uniform model. It's a check, and we
make it an assertion.

### Optimizer: one non-obvious detail

```python
def configure_optimizers(self, weight_decay, lr, betas):
    for _, p in self.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)   # 2-D+ gets decay; 1-D and biases don't
```

Weight decay on a norm's gain or a bias shrinks a *scale* you never wanted to regularise.
Excluding them is free. (Reported by the run: `1,894,420/1,897,780 params get weight decay (99.8%)`
— here it barely matters because tied embeddings dominate the 1-D count; at 50k vocab,
untied embeddings would make it ~40%.)

### Verifying the architecture (run `python 03_model.py`)

```
[model] params=184,332  init loss=6.196 vs ln(V)=6.238
[model] causality test passed (future tokens cannot leak into the past)
[model] KV-cache vs full forward: max abs diff = 1.19e-07
[model] all invariants hold
```

Three tests I'd insist on for any transformer you write yourself:

1. **Initial loss ≈ ln(V)** — catches head/embedding scale errors.
2. **Causality**: corrupt the suffix of the input, assert the prefix's logits are bit-identical.
   This is the test that catches a missing or wrong-signed mask, the most expensive bug class.
3. **KV-cache equivalence**: cached incremental decoding must equal a full forward pass. My first
   implementation failed this: I built RoPE for `start+T` positions but applied it to only the new
   `T` tokens, and the mismatch broadcast the sequence to length `T+start`. The correct rule —
   cached k/v are *already rotated*, so slice `cos[start:start+T]` — is commented in the file.
   Bug class caught: 1.19e-07 agreement after the fix.

---

## 6. Training loop

`04_train.py`. The loop is short; the value is in why each line exists.

```python
for it in range(start_iter, args.iters):
    lr = model.get_lr(it, args.iters, args.lr, args.min_lr_frac, args.warmup)
    for g in opt.param_groups:
        g["lr"] = lr                                  # (1) sample LR every step

    opt.zero_grad(set_to_none=True)
    for _ in range(args.grad_accum):                  # (2) accumulate for a bigger effective batch
        x, y = get_batch(train, args.batch_size, dev, gen)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=dev.type == "cpu"):
            _, loss = model(x, y)
        (loss / args.grad_accum).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # (3)
    opt.step()
```

**(1) LR is computed inside the step, not looked up from a schedule table**, so a resumed run
lands back on the same curve. The schedule is linear warmup (40 steps) then cosine decay to 0.1×:
warmup covers the first steps while Adam's second-moment estimate is still noise; the cosine tail
buys the last few percent of loss.

**(2) Gradient accumulation** simulates a larger batch when a real one won't fit. Divide the loss
by `grad_accum` before backward — forgetting this scales your effective LR by `grad_accum`.
Verified correct by construction here: batch 8 × accum 4 and batch 32 × accum 1 land on
*identical* loss (7.2097 after 20 steps at the same seed), so the math is right.

But **on CPU it is not free, and I got this wrong in my first draft.** Measured, both cores idle,
same 8,192 tokens per step:

| config | tok/s | peak RSS |
|---|---|---|
| batch 32 × accum 1 | 3,240 | 1,373 MB |
| batch 8 × accum 4 | 857 | 566 MB |

3.8× slower for the same optimiser step. Each accumulation micro-batch pays full kernel-launch and
allocator cost on tiny tensors, and CPU has no queue depth to hide it — the opposite of GPU, where
accumulation is nearly free. So on CPU, fit the biggest single batch that doesn't swap, and only
reach for `grad_accum` when even that OOMs.

**(3) Clipping to global norm 1.0** is what stops one unlucky batch from making a loss spike
permanent. Watching `gnorm` is diagnostic: pinned at exactly 1.0 for long stretches means the LR
is too high; near-zero with no loss progress means it's too low.

### Hyperparameters and their reasons

| | value | why |
|---|---|---|
| optimizer | AdamW | decoupled weight decay; Adam+L2 makes decay and adaptive scaling fight |
| lr | 6e-4 | small models tolerate more; ~1e-3 for <10M params, 3e-4 for >100M |
| betas | (0.9, 0.95) | β₂=0.999 assumes long stationary runs; on short runs it goes noisy |
| weight decay | 0.1 | 2-D weights only |
| batch | 32 × 256 = 8,192 tok | tokens/step is the number that matters, not sequences |
| grad clip | 1.0 | spikes are the failure mode at high LR |
| precision | bf16 autocast | ~2× on CPU; fp32 master weights so small updates survive |

**Why bf16 and not fp16.** bf16 has the same 8-bit exponent as fp32, so no overflow → no
GradScaler needed. fp16 gives you one more mantissa bit but requires loss scaling and a scaler
that can itself hide bugs. On CPU (this run) bf16 is simply correct-and-faster; on GPU fp16+scaler
is the historical default and bf16 is preferable if the hardware supports it.

**Mixed precision detail**: master weights and the optimizer state stay fp32; only the forward/
backward matmuls are bf16. `RMSNorm` deliberately computes in fp32 internally — a sum of squares
over 160 elements is exactly where bf16's 7 mantissa bits start to matter.

### Memory, since this box has 2 GB

Measured on this box: peak RSS ≈ **566 MB** at batch 8 (× 4 grad-accum, d_model 160) — comfortable.
The same model at batch 32 peaked at **1.13 GB** in ~1.5 GB available and fell into swap:
throughput went from 3,100 tok/s to **126 tok/s** with `kswapd0` running hot. It never crashed —
it just became 25× slower, which is worse, because you keep waiting. The dominant term is
`batch × block × vocab × 2 bytes(bf16)` for logits — 32×256×4096×2 ≈ 67 MB, times a few for
activations and gradients. If you OOM *or merely slow down*, cut `batch_size` before you cut
`block_size`. Also watch throughput, not just survival: this box OOM-killed a debug script outright
(exit 137) at one point and quietly swapped at others.

And distrust your own instrumentation. I reported "3,100 → 126 tok/s" as evidence of swap thrash;
the second number was bogus — `t0` was reset at every *log*, so the rate was computed over a
one-step window, pinning the reading to a constant wrong value for an entire run. The clean
measurement above only exists because I re-measured on an idle box with the counter fixed.
**A throughput figure that is suspiciously constant is a bug, not a stable machine.**

---

## 7. Evaluation

`05_evaluate.py`. Three numbers, because any one of them alone misleads.

```
[eval] val loss L | ppl exp(L) | top-1 accuracy | bits-per-byte
```

1. **Val loss / perplexity** — what you optimise. Only comparable between runs sharing a tokenizer.
2. **Top-1 accuracy** — fraction of positions where the argmax is the true next token. Lower bound
   is ~0 for a good model; it's a sanity read, not a target.
3. **Bits per byte** = `CE × (tokens/byte) / ln 2`. Tokenizer-independent. BPE packs uneven
   amounts of text per token, so per-token perplexity *flatters* a tokenizer that emits many short
   tokens. bpb lets you compare a 4k-vocab run to a 50k-vocab run honestly.

### Baselines, or the number means nothing

A perplexity of 30 is only interpretable against what trivial models achieve on the *same* split,
so the script computes two:

```python
p_uni = (cnt + 1.0) / (cnt.sum() + vocab)          # unigram, Laplace-smoothed
cond  = (joint + 0.1) / (joint.sum(1, keepdims=True) + 0.1 * vocab)   # bigram Markov
big_loss = -np.log(cond[val[:-1], val[1:]]).mean()
```

A bigram chain already knows that `Holmes` follows `said`. **Your transformer must beat it by a
wide margin or it is broken**, and this comparison has caught more bugs for me than any loss curve.
On this exact split the reference points, measured (a 10-step, deliberately undertrained probe):

```
[base] uniform          ppl = 4,096       <- = vocab size, i.e. ln(V) = 8.32 loss
[base] unigram          ppl =   736
[base] bigram Markov    ppl =   179
[base] this model       ppl = 3,583   (-57.8% "better" than the bigram floor)
```

That last line is the harness working *correctly*: 10 steps of training is below the bigram
floor, and the script says so out loud rather than showing you a bare `ppl 3583` and hoping you
notice. Any base model that cannot beat ~179 on this data has a bug, not a small size.

### The two eval bugs I actually hit

**(a) A silent `nan` that poisoned checkpointing.** `--eval-iters` defaults to 3, but the
"evaluate on the last iteration" branch fired at step 8 of an 8-step smoke test with
`--eval-every 100`, i.e. with **zero** eval batches — and `np.mean([])` is `nan`. The NaN printed
as `val_loss nan`, `best_val` stayed `inf`, and **no checkpoint was ever saved**. Fixed by deriving
`eval_iters` from the data and raising a clear error if there isn't even one batch:

```python
if len(eval_batch) < eval_size:
    raise RuntimeError(f"eval set has {len(eval_batch)} blocks but the eval batch size is "
                       f"{eval_size}: at least one eval batch is required.")
```

Silent NaN → loud error is the whole lesson. Any statistic computed as a mean over a possibly
empty list can do this to you.

**(b) Non-contiguous views.** `blk[:, 1:]` is a strided slice, so `targets.view(-1)` throws
`view size is not compatible with input tensor's size and stride`. Use `.reshape()`.

### What this does *not* measure

Perplexity on in-domain text cannot tell you whether a model is useful, factual, safe, or
instruction-following. For a real system you'd add: held-out NLL under distribution shift, task
benchmarks (HellaSwag/ARC for a base model this size), and safety/refusal evals. The harness shape
is identical — swap the dataset and the metric function. At 1.9 M parameters, a task benchmark
would score at chance, and I'd rather tell you that than show you a flattering number.

---

## 8. Generation

`06_generate.py` + `sample.py`. Decoding is where from-scratch projects most often go wrong.

```python
@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    seed: int | None = None
```

Applied in order: **repeat penalty → temperature → top-k → top-p → renormalise.**

```python
if params.repeat_penalty != 1.0 and history is not None:
    # divide positive logits by the penalty, multiply negative ones, so the move
    # is always *toward zero* regardless of sign; scale by the observed count
    pen = torch.tensor(alpha) ** counts
    x[b, uniq] = torch.where(xs > 0, xs / pen, xs * pen)
x = x / params.temperature
thr = x.topk(k, dim=-1).values[:, -1:]
x = x.masked_fill(x < thr, float("-inf"))          # top-k
srt, idx = torch.sort(x, descending=True)
sm = F.softmax(srt, dim=-1)
keep = (sm.cumsum(-1) - sm) < params.top_p          # exclusive cumsum
cut = (~keep).sum(-1)
x = x.masked_fill(unsort(mask_sorted), float("-inf"))   # top-p
return torch.softmax(x, dim=-1)
```

- **Temperature** flattens (`T>1`) or sharpens (`T<1`); `T=0` = greedy. Too low → loops; too high
  → babble. 0.7–0.9 for prose.
- **top-k alone is wrong** because k is fixed: after `"The capital of France is"` the distribution
  is peaked and k=5 is plenty; after `"Once upon a"` it is flat and k=5 truncates good
  continuations. Our test shows the effect: `top-p=0.9` keeps **2 tokens** on a peaked
  distribution and **94** on a flat one. *That adaptivity is the entire argument for nucleus
  sampling.*
- **top-p** keeps the smallest set with cumulative mass ≥ p. Use it with top-k as a safety cap
  (p=0.9, k=50) so a pathological tail can't dominate.
- **repeat penalty** is the cheapest fix for small-model looping, and the least damaging to the
  distribution's shape.
- **Beam search is deliberately absent.** It maximises joint likelihood, which for open-ended text
  produces long, dull repetition — the degeneracy that pushed everyone to sampling. It remains
  correct for translation and transcription.

### Two bugs found by testing the sampler

1. **top-p kept exactly 1 token.** I wrote `keep.float().argmax(-1)` to find where to cut — but
   `argmax` on a boolean prefix returns the *first* `True` (index 0), so it kept `cut=0` tokens
   beyond the first. Because `keep` is always a prefix, counting the `False`s is the correct,
   simpler form: `cut = (~keep).sum(-1)`. It also guarantees the top candidate survives, so the
   softmax can never receive an all-`-inf` row (which is what produced the `NaN` probabilities that
   made `torch.multinomial` raise `probability tensor contains either inf, nan or element < 0`).
2. **Device mismatch**: a `torch.Generator(device="cpu")` passed to `torch.multinomial` on a CUDA
   tensor. Build the generator on `dev`.

### KV cache

```python
cache = [_Slot() for _ in range(model.cfg.n_layer)]
logits, _ = model(ids, kv_cache=cache)          # prefill: whole prompt, one pass
for _ in range(max_new_tokens):
    nxt = sample(logits)
    logits, _ = model(nxt, kv_cache=cache)      # one token per step, attends to cached k/v
```

Without caching, step *t* recomputes all *t* tokens: O(T²) total work. With it, O(T) — this is the
the difference between a usable model and an unusable one. Measured here, 200 tokens, both cores
free, no estimate involved: 0.49 s cached vs 1.77 s uncached (3.6x). The price is memory:
`2 (k and v) × n_layer × n_head × head_dim × T × 2 bytes` ≈ 0.65 MB/token-sequence here, so a
2,048-token context costs ~1.3 GB per sequence, which is why long context is a memory problem
before it is a compute problem (and why MQA/GQA/PagedAttention exist).

Verified: `cached == uncached over 12 greedy steps` — the same assertion belongs in your repo.

### Running it

```bash
python sample.py --prompt "It was a dark and stormy night" --n 200 --temp 0.8 --compare
```

---

## 9. Results

**A finished 150-step run**, on this box (2 CPU cores, 2 GB RAM): 1.9M params, 4096 vocab,
256-token context, effective batch 32 sequences, bf16. It saw 1.23 M training tokens ≈ 4 epochs.

```
iter   0    loss 8.307    (ln(4096) = 8.318 — the uniform baseline; init verified in 25 ms)
iter  25    loss 7.273
iter  50    loss 6.376
iter  75    loss 5.860    val_ppl 218.6  -> checkpoint saved as model_75.pt
iter 100    loss 5.584
iter 125    loss 5.400
iter 149    loss 5.367    (LR at its cosine floor 6.0e-5)
iter 150    val_loss 5.3851  val_ppl 218.13

[eval] val loss 5.2912 | ppl 198.58 | top-1 19.49% | bits/byte 2.015 (0.264 tok/byte)
[base] uniform ppl 4,096 | unigram ppl 736 | bigram Markov ppl 178.89
[base] this model ppl 198.58   -> beats unigram 3.7x, but does NOT yet beat bigram
```

**The bigram gate says the model is not finished, and it is right.** 198.6 vs 178.9 means it
still has not clearly learned to use more than one token of context. Beating unigram by 3.7× and
losing to a Markov chain is a completely ordinary place for 1.9M params and 1.2M tokens to be;
this corpus is ~0.4M tokens, so one more epoch of LR-schedule *completion* (not just more steps)
is what would move it. I'd rather hand you that than quietly compare against unigram instead.

Trainer-vs-`05` cross-check, same weights: `5.3851` vs `5.2912`. Different by design — the trainer
scores 32 of the 60 val blocks (its batch size is clamped to divide them exactly) while `05` scores
all 60. Don't let anyone tell you "val loss" is a single unambiguous number; state the subset.

Generation (200 tokens, `--temp 0.8 --seed 5`, unedited):

```
Mr. Holmes examined the letter closely the house.”

“I am I had been in the little I are the family.”

“I could not give it is that he was been a bppent and all we would not do with her ly,ing to
be so uned. He was the young bing an two. “No, my her their being not know which I had be to
his great few very c of a day but the other of your ll in his your pan. There a most med with
very very froing and to her own all at the house, and what were that he knowing the first on
this one of him as her the most most tered or at Holmge, he had a such a saa, but. It is a sed
```

Correct dialogue formatting — new paragraph, curly quote opened, and a closing quote with the
period *inside* it (a Victorian convention absorbed with no instruction), plus `Holmge`, a
mangled "Holmes". Grammar and spelling inside the sentences are broken, as they must be at
0.4 M tokens. This is a language model that has learned the *layout and register* of English prose
and not yet its syntax — which is precisely the thing to check when you build your own.

Measured KV-cache effect, 200 tokens, both cores free: **0.54 s cached vs 1.66 s uncached (3.1×)**,
generation at 376 tok/s, and exact (`cached == uncached`).

A metric worth keeping for byte-level BPE models — broken-UTF-8 rate (the model emitted a lone
continuation byte of a multi-byte sequence):

```
iter 100 (4.0 epochs): 48/3,320 = 1.45%
iter 120 (4.9 epochs): 14/3,461 = 0.40%
iter 150 (full sched): 30/3,423 = 0.88%
```

Note it did **not** fall monotonically. That is the honest lesson: this is a noisy
quality proxy on 5 fixed prompts, not a benchmark. It caught the *catastrophic* case (1.45%) and
it would be wrong to tune against it — which is true of every cheap generation metric.

## 10. Scaling this up: what actually changes

| stage | at this scale | at 1B+ params |
|---|---|---|
| tokenizer | pure-Python BPE is fine | Rust `tokenizers`/`tiktoken`, ~1000× faster; vocab 32–128k |
| data | 1.4 MB, one process | TB-scale, dedup + quality filtering, web-scale mixing, tokenise offline into shards (this pipeline's shape, at 10⁶× the size) |
| precision | bf16 autocast | bf16 + FSDP/ZeRO, activation checkpointing |
| parallelism | none | tensor + pipeline + data parallel |
| attention | masked softmax | FlashAttention (fused, avoids materialising T×T) |
| batch | 8k tok/step | 1–4M tok/step |
| LR | 6e-4 constant peak | ~1e-4…3e-4, longer warmup, decay to 10% |
| schedule | cosine | WSD (warmup-stable-decay) — same shape, cheaper to tune |
| eval | ppl + bpb | + task benchmarks, human pref evals, safety evals |
| after pretraining | — | SFT → preference optimisation (RLHF/DPO) → long-context extension |

Five things worth stealing from production recipes:

1. **Train at higher resolution than you use.** Pretrain at 2–4k context, then extend with fine-tuning
   on long sequences and RoPE base rescaling (`rope_base` 10k → 500k+). Our `rope_base` is a config
   field for exactly this.
2. **Sequence packing** with document-boundary attention masks: pack multiple short documents into
   one block and mask across boundaries, so you stop wasting ~20% of a block on short documents.
   (That's the fix for the dropped-documents trade-off in §4.)
3. **Log gradient norm and LR every step, always.** Both are in `out/log.jsonl`. When a run
   diverges, the gnorm trace tells you in 10 seconds what a loss curve reveals in 10 minutes.
4. **Save the optimizer state from day one** (`opt.pt`), and keep the config *with* the weights
   (`config.txt`). A checkpoint without its config is an unreadable file; I learned to assert
   this after being unable to load one probe checkpoint because the dims didn't match.
5. **Don't trust a model that can't beat a bigram.** Cheap baselines are the highest
   value-per-line check in any LM pipeline.

---

## 11. File map

| file | contents |
|---|---|
| `_stage.py` | import shim (module names can't start with a digit) |
| `01_tokenizer.py` | byte-level BPE, train/encode/decode/save + self-tests + `check_invariants` |
| `02_preprocess.py` | corpus cleaning, document split, block packing, memmap, sanity checks |
| `03_model.py` | GPT config + RoPE/RMSNorm/SwiGLU transformer + optimizer + invariant tests |
| `04_train.py` | training loop, LR schedule, eval, checkpointing |
| `05_evaluate.py` | loss/ppl/accuracy/bpb, unigram + bigram baselines, loss curve |
| `06_generate.py` | sampling (temp/top-k/top-p/repeat-penalty) + KV cache |
| `sample.py` | generation CLI |
| `data/corpus.txt`, `data/shuffled/` | cleaned corpus, tokenised blocks |
| `tokenizer/`, `out/` | vocabulary, checkpoints, `log.jsonl`, `eval.json` |

## 12. The bugs, catalogued

Worth reading, because they're the same ones you'll meet:

| bug | symptom | fix |
|---|---|---|
| BPE window arithmetic | 268-vocab instead of 4,096; 3,839 merges using 11 pairs | diff exact whole-word counts; cross-check vs naïve reference (21/21 match) |
| RoPE applied to cached k/v | `RuntimeError` shape `[8,63]` vs `[8,32]` | cache stores rotated k/v; slice `cos[start:start+T]` |
| `np.mean([])` in eval | `val_loss nan`, `best_val inf`, **no checkpoint saved** | derive `eval_iters` from data; raise if <1 batch |
| uint16 → Embedding | `Expected ... Long, Int; but got CPUUInt16Type` | cast once in `get_batch` |
| strided target tensor | `view size is not compatible with ... stride` | `.reshape()` not `.view()` |
| `argmax` on boolean prefix | top-p kept 1 token; `multinomial` got NaN probs | `cut = (~keep).sum(-1)` |
| CPU generator + CUDA tensor | `multinomial` device error | build generator on `dev` |
| `torch.get_num_threads()==1` | CPU run silently 2–8× slower | `torch.set_num_threads(os.cpu_count())` |
| hard-wrapped corpus | generation emits line breaks at 70 chars | unwrap newlines before tokenising |
| config written as `repr(json.dumps(...))`, read with `ast.literal_eval` | `**` unpack got a str, not a dict — no trainer checkpoint could be reloaded | one JSON format for save+load, and raise on unknown keys |
| silent swap thrash | run alive but 3,100 → 126 tok/s, no error | batch 32 → 8 + grad_accum 4; watch tok/s, not just exit code |
| `--resume` when `opt.pt` was gone, and `best_val` starting at `inf` | **a trained checkpoint was overwritten by a freshly initialised model** — the first eval always "improved" on `inf` | refuse to write into a dir holding checkpoints without `--resume`; error if `--resume` has no `opt.pt`; separate `--model` for warm starts; save `model_<iter>.pt` every eval, not only "best" |
| per-batch-mean val loss | trainer and `05` disagreed (190 vs 173) on identical weights; biased the "is this best?" decision | token-weighted average, and clamp eval batch size to divide the val set exactly |
| tok/s divided by steps since last log, but `t0` reset every log | first line read 850 tok/s, then a flat wrong 34 tok/s for the whole run | derive `steps` from `it - last_log`; never trust a rate that is suspiciously constant |
