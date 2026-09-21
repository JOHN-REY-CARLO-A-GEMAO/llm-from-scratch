# Building an LLM from scratch

**A complete, verified pipeline: data → tokenizer → transformer → training → evaluation → generation.**

Every code block in this guide is from the working files in this folder, and every number
reported was produced by actually running them. The model in `out/` was trained on a 2-core CPU
box in about 33 minutes (1,400 steps, fp32, ~8,600 tok/s), so the run is small by design — the
*pipeline* is what transfers to scale, not the model size. That checkpoint is committed, which is
why `run.ps1` can generate text on a laptop without training anything; §9 has its numbers and
`out/MODEL_CARD.md` has its limits. The architecture is Llama-3 style (RMSNorm, RoPE, SwiGLU,
`scaled_dot_product_attention`); §5 explains each choice, including what the earlier GPT-2-style
version of this file did differently and why it changed.

---

## 0. Set the right expectation first

| | this guide | GPT-3 class |
|---|---|---|
| parameters | 1.93 M | 175 B |
| training tokens | 11.5 M | 300 B |
| context window | 256 | 2048 |
| result | coherent little stories, no facts, no instruction-following | general ability |

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

**Source: TinyStories** (Eldan & Li, 2023, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)) —
~2.1 M synthetically written children's stories in deliberately simple English, distributed as a
single 2.2 GB `.txt` on the HuggingFace Hub. `00_tinystories.py` **streams** it (HTTP range reads,
bye-bye 2.2 GB download), keeps ~20 MB of clean stories, and writes `data/tinystories/corpus.txt`.

```bash
python 00_tinystories.py --source hf-http --out data/tinystories --max-bytes $((32<<20)) --max-chars 20000000
```

Why this corpus and not, say, Shakespeare: **corpus choice dominates model choice at this scale.**
A 2 M-parameter model cannot learn general knowledge from 11 M tokens, but TinyStories was *designed*
so that small models can learn to write coherent English from it — the vocabulary is ~1.5 k words,
sentences are short, every story has a beginning, a middle and an end. It makes the difference
between "the pipeline works" and "fluent sentences, no structure".

Cleaning is the whole job here, and it is not optional. This dump contains:

1. **Mojibake** — `â€œ` for a curly quote (a UTF-8 → cp1252 → UTF-8 round trip). 72,430 characters
   of it in the 20 MB slice we keep. Repairing the exact byte sequences is safe; guessing is not.
2. **Lost story separators** — where `<|endoftext|>` was eaten, two stories are glued into one
   sentence: `"...she was happy.Once upon a time, ..."`. A story opener printed immediately after a
   sentence end is the tell, and that is where the splitter cuts.
3. **Truncated stories, unicode noise, Prompt-GPT artifacts** — filtered by the length,
   sentence-structure, non-ASCII-budget and letter-ratio checks in `clean_story()`.
4. **Duplicates.** The dump is not uniformly unique; a naive slice-and-shuffle can hand you the
   same story thousands of times. `build_corpus()` deduplicates by content hash (blake2b) and
   reports what it dropped.

The unit of the corpus is a *story*, not a paragraph. Getting this wrong is subtle: splitting on
blank lines yields 136-character fragments (median) because dialogue and the moral of a story are
separate paragraphs; 84% of them then fail the length filter. `split_stories()` therefore cuts at
sentence boundaries in front of an opener and re-groups the pieces into 300–2000-character stories,
which keeps 25,147 stories (1.6% rejected).

```
[data] raw text 61,976,042 chars from local:/tmp/ts_train.txt
[data] wrote data/tinystories/corpus.txt (20.0 MB, 25,147 stories, mean 795 chars)
[data] rejected 400 of 25,547 raw chunks
```

(The classic Gutenberg path still exists — three public-domain books in `data/raw/`, cleaned by
`clean_gutenberg()` and packed with `python 02_preprocess.py --mode docs`. It is a fine *style*
corpus, and it is what the earlier 400-step checkpoint in this repo's history was trained on, but
`--mode stream` on TinyStories is what the shipped model uses.)

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

**4,096 was the right choice for 20 MB of text.** Rough guide: you want ≥ ~50 tokens of training
data per vocabulary entry, and 20M chars / 4.5 ≈ 4.4M tokens / 4096 ≈ 1,080. Comfortable. 50k
(GPT-2's size) would still be defensible — TinyStories' ~1.5k-word vocabulary compresses well — but
it would put 8 M of a 2 M-parameter model's budget into the embedding matrix.

Measured outcome:

```
[tok] 3,839/3,839 merges  182.4/s  eta 0s
[tok] final vocab = 4096
[tok] self-test passed: round-trip on text + arbitrary bytes; no UNK path needed
[tok] 34 tokens for 160 chars -> 4.71 chars/token
[tok] compression on the corpus: 4.49 chars/token
```

**4.49 chars/token** — GPT-2's 50k BPE gets ~4.0 on general English, so a 4k vocabulary *tuned to
this corpus* beats it on simplicity of text. (Note: an earlier run of this pipeline measured 1.09
chars/token, which was not a tokenizer flaw but a vocabulary that ignored its merges — a reminder
to always measure compression.)

```bash
python 01_tokenizer.py --corpus data/tinystories/corpus.txt --out tokenizer --vocab 4096 --train-chars 4000000
```

`--train-chars` caps the sample because merge statistics saturate quickly: 4 M characters (55 s here)
gives essentially the same 3,839 merges as the full 20 MB.

**Encoding speed is a separate problem from training speed.** The textbook encoder replays all 3,839
merges against every word — correct, and unusable at corpus scale. `encode(fast=True)` (the default)
instead merges the *lowest-ranked adjacent pair* until none is left, which is the same function
computed in O(len²) per word, plus a cache of word → pieces (~10⁴ distinct words in a 20 MB corpus,
10⁶ occurrences). `encode_reference()` keeps the slow version, and
`tests/test_tokenizer_encoding.py` asserts both produce identical ids on 46 real stories — a fast
path that disagrees with the reference is how a corpus silently stops matching its vocabulary.

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
python 02_preprocess.py --corpus data/tinystories/corpus.txt --tokenizer tokenizer --out data/packed --mode stream
[data] train: 23,890 stories, 4,144,892 tokens, 16,190 blocks of 256 -> data/packed/train.npy
[data] val:    1,257 stories,   216,121 tokens,    844 blocks of 256 -> data/packed/val.npy
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
q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # [B,H,T,Dh]
k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   # rotate q,k by position; never v

# one fused kernel for scale + causal mask + softmax + matmul:
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
return self.o_proj(y.transpose(1, 2).reshape(B, T, D))
```

That is four lines where the GPT-2 version of this file had nine, and it is measurably faster:
SDPA fuses the causal mask into the kernel (nothing of size `T×T` is materialised) and lets
PyTorch pick a flash-attention-style CPU/CUDA implementation. Measured on this box, the whole
model: **0.950 s/step vs 1.162 s/step** for the hand-written path at the same shapes.

The cost is that the mask is no longer a single object you can pass around, and the three cases
have to be handled explicitly — this is the one place where the modern code is easier to get
*wrong* than the old one:

| case | mask |
|---|---|
| no cache (training, prefill) | `is_causal=True` |
| one new token, warm cache | none: `is_causal=False` and *no* mask — the single query must see every cached key |
| T > 1 against a warm cache (chunked prefill) | explicit bool mask `j <= i + past` ("bottom-right" causal) |

`is_causal=True` with `q_len=1` keeps only the *first* key, which would silently turn decoding
into nonsense — `tests/test_architecture.py` and the chunked-prefill invariant in `03_model.py`
pin all three cases.

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
- **No biases anywhere in q/k/v/o.** q/k/v biases either cancel in the softmax or shift q
  uniformly. The earlier version of this file kept a learnable per-head logit bias as a temperature;
  it is gone, because Llama-3 has none and the depth-scaled init below makes it unnecessary.

### MLP: SwiGLU

```python
hidden = 64 * ((int(8 * d_model / 3) + 63) // 64)      # 8/3 * d_model, up to a multiple of 64
return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
```

(The names are Llama-3's; at `d_model=160` this is `hidden=448`, i.e. 1.4 M of the model's
1.9 M parameters live here. Rounding to a multiple of 64 matters for tensor-core efficiency on
GPUs and vectorisation on CPUs.)

Gated linear unit with SiLU. Three weight matrices instead of two, so width shrinks by 2/3 to
keep FLOPs equal to a 4× ReLU MLP — and at matched FLOPs it wins. Rounding to a multiple of 8
matters for tensor-core efficiency on GPUs (and vectorisation on CPUs).

### Tied embeddings

```python
self.lm_head.weight = self.wte.weight
```

One matrix serves input and output. Saves `vocab × d_model` parameters (655 k of 1.93 M here) and
measurably helps small models, where the output distribution and the input representation want
similar structure. Cost: the two roles pull the matrix in different directions, which is why very
large models untie them.

### Initialisation — the part people skip

```python
self.apply(self._init_weights)          # N(0, 0.02) for Linear/Embedding, zero biases, unit norm gains
scale = 1.0 / math.sqrt(2 * cfg.n_layer)
for block in self.blocks:
    nn.init.normal_(block.attn.o_proj.weight, std=0.02 * scale)     # depth-scaled
    nn.init.normal_(block.mlp.down_proj.weight, std=0.02 * scale)
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
[model] params=192,960  init loss=6.259 vs ln(V)=6.238
[model] causality test passed (future tokens cannot leak into the past)
[model] RoPE relative-position invariance: max abs diff = 7.15e-07
[model] SDPA vs explicit attention: max abs diff = 3.26e-09
[model] KV-cache vs full forward (1 token): max abs diff = 1.79e-07
[model] chunked prefill vs full forward: max abs diff = 2.38e-07
[model] all invariants hold
```

Six tests I'd insist on for any transformer you write yourself — the last three only exist because
the architecture is Llama-style:

1. **Initial loss ≈ ln(V)** — catches head/embedding scale errors.
2. **Causality**: corrupt the suffix of the input, assert the prefix's logits are bit-identical.
   This is the test that catches a missing or wrong-signed mask, the most expensive bug class.
3. **KV-cache equivalence**: cached incremental decoding must equal a full forward pass. My first
   implementation failed this: I built RoPE for `start+T` positions but applied it to only the new
   `T` tokens, and the mismatch broadcast the sequence to length `T+start`. The correct rule —
   cached k/v are *already rotated*, so slice `cos[start:start+T]` — is commented in the file.
   Bug class caught: 1.79e-07 agreement after the fix. The **chunked-prefill** case (feed
   `[0:20)` then `[20:32)` and compare with one 32-token pass) is the same test one level deeper:
   it exercises the explicit bottom-right causal mask that `is_causal=True` cannot express.
4. **RoPE is relative**: rotate q and k by the same offset and the attention logits must not move.
   If someone "simplifies" the rotation into an absolute embedding, this fails immediately.
5. **SDPA == explicit attention**: same projections, same rotation, then a hand-written
   `softmax(qk^T/√d + mask)`. 3e-09 agreement is what tells you the fused kernel is doing what you
   think, including where you *didn't* pass a mask.

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
| precision | bf16 autocast (`--autocast none` = fp32) | ~2× on CPU *with* bf16 hardware; fp32 master weights so small updates survive |

**Why bf16 and not fp16.** bf16 has the same 8-bit exponent as fp32, so no overflow → no
GradScaler needed. fp16 gives you one more mantissa bit but requires loss scaling and a scaler
that can itself hide bugs; on GPU fp16+scaler is the historical default and bf16 is preferable if
the hardware supports it.

**The catch, measured on this box:** "bf16 is faster on CPU" is only true if the CPU has
AVX512-BF16 (or AMX). Without it PyTorch emulates bf16 with fp32 ops, and the emulation costs more
than it saves: the same 1600×1600 matmul took **1.84 ms/step in fp32 vs 8.84 ms/step under bf16
autocast** — ~5× *slower*. `--autocast` is a flag precisely so the precision follows the hardware
instead of being baked in: `bf16` (the default) on a bf16-capable box, `none` on anything older.
Whichever you pick, keep it for evaluation too (`05_evaluate.py --autocast …`) or the val loss you
report will not be the one the trainer selected checkpoints with.

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
  distribution and **158** on a flat one. *That adaptivity is the entire argument for nucleus
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

**The shipped checkpoint** — 1.93 M parameters, Llama-3 style, 4096-token byte-level BPE, 256-token
context, trained on 20 MB of clean TinyStories for **1,400 steps × 8,192 tokens = 11.5 M tokens**
(≈2.8 epochs over the 4.14 M-token train split) on 2 CPU cores in **33 minutes** at fp32
**~8,600 tok/s**:

```
[run] device=cpu  threads=2  precision=fp32
[run] params 1,926,560 (34% in the embedding)  flops/token 3,197,760
[opt] 1,925,120/1,926,560 params get weight decay (99.9%)
[run] train blocks 16,190  val blocks 844

[train] iter     0/1400  loss 8.3584  lr 1.00e-05  gnorm 1.35  7,333 tok/s   # ln(4096) = 8.318
[train] iter   100/1400  loss 4.9005  lr 1.00e-03  gnorm 0.55  7,118 tok/s
[train] iter   400/1400  loss 3.1569  lr 8.87e-04  gnorm 0.80  7,719 tok/s
[train] iter   800/1400  loss 2.7591  lr 4.96e-04  gnorm 0.98  7,955 tok/s
[train] iter  1200/1400  loss 2.3761  lr 1.52e-04  gnorm 0.96  8,197 tok/s
[train] iter  1399/1400  loss 2.4299  lr 1.00e-04  gnorm 1.02  8,771 tok/s
[  eval] iter  1400       val_loss 2.5663  val_ppl 13.02            <- out/model_best.pt

[  eval] iter   100  val_loss 4.8810  val_ppl 131.76     <- crosses the bigram floor almost at once
[  eval] iter   400  val_loss 3.2065  val_ppl  24.69
[  eval] iter   800  val_loss 2.7640  val_ppl  15.86
[  eval] iter  1200  val_loss 2.5975  val_ppl  13.43
[  eval] iter  1400  val_loss 2.5663  val_ppl  13.02
```

Full metrics from `05_evaluate.py` (`out/eval.json`), plus the baselines that give them meaning:

```
[eval] val loss 2.5826 | ppl 13.23 | top-1 45.26% | bits/byte 0.830 (0.223 tok/byte)
[base] uniform          ppl = 4,096
[base] unigram          ppl = 613.28
[base] bigram Markov    ppl = 52.84
[base] this model       ppl = 13.23   (75.0% lower ppl, 34.9% lower cross-entropy, vs the bigram floor)
```

(The trainer's own eval uses a fixed 32,768-token window and reports 2.5663 / ppl 13.02 —
the same weights, a smaller sample of the same held-out stories; `05_evaluate.py` scores
512 of the 844 val blocks, i.e. 131,072 tokens, and is the number to quote.)

Read those four numbers in order, because that is the whole point of the section:

1. **Uniform** (4,096) is where we started; the init test pinned it at 8.3584 vs `ln(4096)=8.318`.
2. **Unigram** (613.28) is "sample tokens by their frequency". Any model that cannot beat this is
   broken, not untrained.
3. **Bigram Markov** (52.84) is the real gate: one token of context, no network. It is crossed
   before iter 100 and ends **4× below it** (`ppl` 13.23 vs 52.84).
4. **This model** (13.23) adds a 256-token window of context (and 1.9 M parameters of
   representation) on top of that single-token count. It is trained on 11.5 M tokens,
   i.e. ~2.8 epochs.

An honest caveat: TinyStories is a *simple* distribution. These perplexities are not comparable to
a Wikipedia-trained model's, and the generation below is coherent but semantically shallow (a
"kitten" can become a "bunny" two sentences later). What the run does demonstrate is everything the
pipeline is supposed to: the loss starts at `ln(V)`, beats the Markov floor, keeps improving until
the LR schedule ends, and the checkpoint produces grammatical, story-shaped English.

For comparison with the repo's earlier life: the same 1.9 M parameters trained on 1.44 MB of
Project Gutenberg prose (400 steps, 3.28 M tokens) reached val loss 4.4949 / ppl 89.56 / top-1
24.7% / 1.712 bits-per-byte. **Same architecture family, same compute order — 7× the data, a
cleaner corpus and 3.5× the steps is the difference between "fluent but lost" and "a story".**
That is the argument of §0 in one line.

## 10. Scaling this up: what actually changes

| stage | at this scale | at 1B+ params |
|---|---|---|
| tokenizer | pure-Python BPE is fine | Rust `tokenizers`/`tiktoken`, ~1000× faster; vocab 32–128k |
| data | 20 MB, one process | TB-scale, dedup + quality filtering, web-scale mixing, tokenise offline into shards (this pipeline's shape, at 10⁶× the size) |
| precision | bf16 autocast (fp32 if the CPU can't do bf16 natively) | bf16 + FSDP/ZeRO, activation checkpointing |
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

## 11. System prompts, personas, and tool use

Four features sit between the sampler and the user, and all four exist because of
the same observation: **a chat model is a text completer, and some of the text it
is asked to produce is not text at all.** Tone is best steered with prompt text;
arithmetic is not — so tone is a prompt and arithmetic is a program. Reasoning
is text, so it is *trained* — as a marked-up block the program can find, split
and render. And "answer from this document" is retrieval plus a fixed template,
not something to hope the weights learned.

### 11.1 The system turn

The SFT corpus in `07_make_sft.py` teaches one framing:

```
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
```

A system prompt is one more turn, in front of the first user turn:

```
<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n…
```

`personas.py` owns that string. `generate.py --chat` and `web_ui.py` both call
`personas.resolve()` and `generate.serialize_chat()`, so the terminal REPL and
the browser cannot drift: identical tone steering, identical token boundaries,
identical trimming behaviour. `tests/test_web_ui.py` asserts that
`ChatEngine.serialize` *is* `serialize_chat`.

Five presets ship (Friendly Assistant — the default, Pirate, Math Tutor,
Storyteller, Reasoner), plus free text and an explicit "none". Four details are
deliberate:

* **Honesty about the effect.** Tone presets are conditioning *text*, not an
  instruction channel the model was trained to obey: at 7 M parameters expect a
  change of tone and vocabulary, not of capability. Two presets are different,
  because the corpus trains the behaviour they ask for: `math` attaches
  `personas.CALC_HINT` ("write `<<calc: 5 + 5>>` and the exact result is
  substituted") and `reasoner` attaches `COT_HINT` (plan inside
  `<think> … </think>` first). 300 corpus rows and a pinned curriculum teach the
  marker, 2,296 teach the trace, and ~590 more repeat ordinary rows with each
  preset's real text in front of them, so the *default* path (Friendly Assistant,
  no thinking hint) is a prompt the model has actually seen. `personas.resolve("none")` is still a
  first-class option — it reproduces exactly the framing most of the corpus used.
* **Marker stripping.** `personas.sanitize()` deletes `<|im_start|>`,
  `<|im_end|>`, `<|doc_end|>` and `<|endoftext|>` from anything a human typed —
  in the system prompt, in the message, and in echoed history. Without it, the
  "system prompt" box is an injection surface: type
  `<|im_end|><|im_start|>user\n…` and you are writing the framing yourself.
  A chat template must be assembled by the program, never by the text.
* **A cost, in tokens.** The shipped context is 256 tokens. `build_prompt()`
  accounts for the system block in its budget: history turns are dropped first,
  and only if the block still does not fit is the block itself shortened by
  `personas.shrink()` (binary search on a character prefix, `cost()` supplied by
  the caller). Silently letting prefill truncate from the left would drop the
  system *and* the question — a bug, not a degradation.

### 11.2 The calculator

Ask a 1.9 M-parameter story model `5 + 5` and it answers in the register of the
question with a number of its own invention. Two hooks in `tools.py` remove that
failure mode without a bigger model:

| where | trigger | what happens |
|---|---|---|
| before sampling | the message *is* an expression (`5 + 5`, `12 * 4 = ?`) | answered locally: `5 + 5 = 10`, `0` model tokens, labelled `🧮 computed locally` |
| after sampling | the message *contains* a math question (`what is 12 * 4?`) | the model answers, then the exact line `🧮 12 * 4 = 48` is appended if the reply lacks the right value |
| during sampling | the model writes `<<calc: 5 + 5>>` | the marker is replaced by `10` mid-stream, including when it is split across tokens |

Two of those three are about the *model's* output, so the substitution has to
survive tokenization: `<<calc: 5 + 5>>` can arrive as `"<<ca"`, `"lc: 5 + 5>"`,
`">"`. `ToolTextFilter` holds back a tail that could still become a marker and
releases it as soon as the call completes or is proven impossible — the same
technique as a stop-string buffer in a streaming API. The first version of it
searched for the *last* `<<` and so emitted the first `<` of a split marker on
its own; the fix is to walk back over the whole run of `<` before asking whether
what follows could still become `<<calc:`. `tests/test_tools.py` feeds every
single split of the marker, plus one character at a time, and asserts that no
half-marker is ever visible.

Evaluation itself is where the security argument lives. `eval("5+5")` is a
one-liner and an RCE: the string is Python, and Python reaches the filesystem.
`tools.evaluate()` parses with `ast.parse(..., mode="eval")` and walks a
whitelist — numeric literals, `+ - * / // % **`, parentheses, unary sign — and
raises `CalcError` on anything else (names, calls, attributes, subscripts,
comprehensions, comparisons). Nothing is "sandboxed"; everything else is simply
not implemented. Work is bounded too: ≤200 characters, ≤64 AST nodes, nesting
≤12, |exponent| ≤512, so `9**9**9` is rejected instead of allocating the universe.

Arithmetic is exact, because float is a bad default for a calculator:
`fractions.Fraction` throughout, seeded from the *decimal literal text*
(`Fraction(repr(0.1))` is `1/10`, not the binary expansion), and formatted once
at the end — integers stay integers (`10`, not `10.0`), `0.1 + 0.2` prints `0.3`,
and a non-terminating result is rounded to 12 significant digits instead of
claiming seventeen digits of truth.

Detection is deliberately conservative: a bare expression, or a trigger word
("what is", "calculate", `=`) plus an evaluable number run. Prose that merely
contains digits — "call me at 5 - 3pm", "2024-01-02" — stays with the model,
because a false positive here replaces a good answer with arithmetic. Word
operators (`12 times 4`, `20% of 80`) and typographic operators (`×`, `÷`, `−`)
are normalised first; `--no-tools` turns the whole hook off, which is the honest
way to see what the model does without it.

### 11.3 Chain-of-thought

The cheapest way to get a small model to answer a multi-step question better is
to make it write the steps down first, because every step becomes context for the
next token. The catch is that "write the steps down" has to be *legible to the
program*, or the user reads a paragraph of working-out before the answer they
wanted. So the trace is a marked block:

```text
<think>
1. Identify the core question: why does it rain.
2. Check the key facts: water warms, rises into the sky, and cools into tiny drops.
3. Put the facts together and answer in one or two short sentences.
</think>
Water warms, rises into the sky, and cools into tiny drops in clouds. …
```

Four decisions are worth defending:

* **The markers are plain text, not added tokens.** The chat tokenizer's only
  added tokens stay `<|im_start|>`/`<|im_end|>` (4096 → 4098). Adding
  `<think>`/`</think>` as *tokens* would mean a second vocabulary expansion, a
  second embedding row per marker, and a tokenizer artifact that no longer
  matches the base one for no real gain — byte-level BPE already encodes each
  marker in a handful of bytes, and a model that has to *spell* the marker is
  exactly as predictable as one that samples a dedicated token.
* **Traces are selective.** Only questions that need more than one step get one
  (why-questions, word problems, comparisons, document questions); greetings,
  one-fact answers and `5 + 5` do not. A corpus where every reply opens with
  `<think>` teaches a prefix, not a decision. `08_sft.py --cot-pins 96` repeats
  three reasoned rows and one direct row through training for the same reason —
  a 7 M model drifts back to "answer immediately" without the extra weight, and
  the pinned *direct* row is what stops the trace from becoming unconditional.
* **Parsing is one function, used twice.** `personas.split_think()` is the
  definition (`reasoning`, `answer`, `closed`), and
  `personas.ThinkStreamFilter` is the streaming view of it — implemented as the
  **delta of two monotone streams** rather than a second state machine, so the
  browser, the terminal and the tests cannot disagree about the same reply. The
  browser needs its own copy anyway (it is JavaScript), so the parity is a unit
  test: `tests/test_web_ui.py` extracts `splitThink`/`stripThink` from the real
  page and runs both implementations over the same replies in node.
* **`closed` is reported, not guessed.** A reply that was cut off by the context
  limit mid-trace has no `</think>`; the honest state is `closed: false`, which
  the UI renders as "🧠 Thinking…" and the meta line as "(unclosed)". Text before
  the opening marker is kept as part of the answer instead of being discarded:
  a model that writes "Let me work this out. `<think>…`" has said something, and
  hiding it would hide a formatting failure rather than show it.

The measured result (greedy, `--cot-check 3`, recorded in
`out/chat_run_config.json`) is in the README: the format is reliable, the
*arithmetic inside* the trace is not — which is why the corpus also teaches the
model to put the arithmetic in a `<<calc: …>>` marker (300 rows), so the numbers
in the trace and the number in the answer come from `tools.py` rather than from
sampling.

### 11.4 Document Q&A without a vector database

`rag.py` is ~450 lines of stdlib Python and no dependencies: no embeddings
package, no vector store, no network call. For a chat UI that answers questions
about *one* file at a time, that is the right size of tool, and BM25 over word
unigrams **and bigrams** is a strong lexical baseline for exactly this shape of
problem:

* **Chunking on structure.** 900 characters with 150 of overlap, cutting on a
  paragraph break first, then a sentence end, then a space — a chunk that starts
  mid-sentence costs attention for nothing. Overlap exists so a fact that
  straddles a boundary is still inside one chunk.
* **Unigrams + bigrams.** Bigrams ("Cora River", "silver fish") carry most of the
  meaning in short passages and cost one dictionary lookup; a unigram-only index
  ranks a chunk that shares "river" as highly as one that shares the phrase.
* **Relevance as a fraction, not a raw score.** BM25 scores are not comparable
  across documents — in a two-chunk note every IDF is small, so a fixed
  `score > 0.5` threshold silently retrieves *nothing* while a 200-chunk book
  retrieves everything. The score is therefore normalised by the best match the
  query could achieve in *this* index, and the floor (`MIN_RELEVANCE = 0.30`)
  means "30% of the achievable match" in both. The number comes from a probe, not
  from taste: on a one-chunk note a relevant question scores 1.00 while off-topic
  ones land at 0.18–0.21 ("why does it rain?" against a note about a creature:
  0.182), so 0.30 separates them with room to spare. On a 232-chunk book the same
  floor only removes the bottom of the distribution — an off-topic query can still
  reach 1.00 there, which is why the sources fold and the grounding warning exist
  and this floor is not sold as a correctness guarantee.
* **The floor matters more than the ranking.** If nothing clears it, no context
  is injected at all. Padding a 256-token prompt with an unrelated passage and
  asking "answer ONLY from this context" is how a RAG demo produces confident
  nonsense; the page says "no relevant passage" instead.
* **A trigram fallback for typos.** When BM25 scores nothing at all (every query
  term unknown), character-trigram cosine similarity is tried — so "the blue
  wale" still finds the whale passage. It only runs when BM25 found nothing:
  mixing the two would let a fuzzy match outrank a real keyword hit.
* **The block is budgeted, and the context goes first.** The injection is the
  exact template the feature is specified with:

  ```text
  Context from uploaded document:
  [1] <passage>
  [2] <passage>

  Answer the user's question based ONLY on this context.
  ```

  `context_block()` takes a token budget and a `cost()` callable (the caller owns
  the tokenizer) and trims the passage to the window around the matched terms —
  binary search on the text, same idea as `personas.shrink()`. The document block
  precedes any persona text because the persona is the part that may be dropped:
  a truncated instruction to obey the document is worth more than a truncated
  description of a pirate.
* **Grounding is measured.** `rag.grounding(reply, context)` is the share of the
  reply's content words that appear in the retrieved context, and the UI warns
  when a reply shares almost nothing with the document. It is a heuristic — a
  well-phrased paraphrase can score low — so it is presented as a warning next to
  the sources, never as a verdict, and short replies are not judged at all.

The part that makes this a *feature* rather than plumbing is the data: 2,027 of the
6,000 SFT rows carry a real `rag.context_block()` system turn. The entities in
those passages are invented (`zorble`, `quibble`, the Cora River), the sentences
name their subject and carry the keyword the question asks with
(`lives`/`eats`/`makes`), and they are shuffled inside the passage next to a
second creature's sentences — so the answer is never simply the first line, and
position or shape alone cannot pick it out. 184 rows ask about a fact the passage
does not contain and answer `The document does not say.` — the honest answer,
trained.

Two versions of those rows were lessons in how a 7 M model takes a shortcut.
The first rephrased the passage, made refusals 35% of the document rows, and
pinned answers and refusals equally hard; the model answered *"The document does
not say."* to **every** document question, including three about a passage that
plainly contained the answer. The second copied the passage's own sentence, but
paired twelve fixed names with twelve fixed facts — and the model learned the
table. Asked "Where does the marlow live?" with a passage that said *Low Fen*, it
replied with the fact it had memorized from training, and for an entity it had
never seen it said the document does not say. Both failures were caught by
`--doc-check`, which is the whole argument for having a check instead of a claim.

The fix is in the data: the values are paired with names **at random** across
rows, so there is no table to memorize and copying is the only way to be right.
`--doc-pins 64` then repeats the two *longest* answering rows (a short pinned row
would be memorized outright and would teach that a document answer is a sentence
fragment) and one refusal at half that weight. `--doc-check 3` checks it with two
held-out subjects and two series of sentence shapes, one of which no training row
uses — and each series also asks a question the passage *cannot* answer, because
"always copy something" and "never copy anything" fail in opposite directions and
each passes the other's test.

That third version still did not read, and the third diagnosis is the one worth
keeping: replaying a **training** row's own prefix recovers its recorded answer
exactly, so the model had memorized the passages the checks could not tell apart
from reading. A fourth run trained on the document family *alone* — nothing else
in the corpus, 400 steps, validation loss 0.0015, i.e. a near-perfect fit of the
training rows — still answered "The document does not say." to a held-out
passage. So in-context copying did not emerge at this scale, and no amount of
re-balancing the mixture was going to fix it: 7 M parameters, six layers and a
256-token window will store 2,000 passages before they will learn to *look one
up*. The feature still ships, because retrieval, injection, the exact template,
the sources card and the grounding warning are all exact and tested; what ships
with a caveat is the last step, and the caveat is printed by `--doc-check`
rather than left for the user to discover.

## 12. File map

| file | contents |
|---|---|
| `_stage.py` | import shim (module names can't start with a digit) |
| `00_tinystories.py` | stream TinyStories (~32 MB), clean it, write ~20 MB corpus + provenance |
| `01_tokenizer.py` | byte-level BPE, train/encode/decode/save + self-tests + `check_invariants` |
| `02_preprocess.py` | corpus cleaning, document split, block packing, memmap, sanity checks |
| `03_model.py` | Llama-3 style transformer (RMSNorm/RoPE/SwiGLU/SDPA) + optimizer + invariant tests |
| `04_train.py` | training loop, LR schedule, eval, checkpointing |
| `05_evaluate.py` | loss/ppl/accuracy/bpb, unigram + bigram baselines, loss curve |
| `06_generate.py` | sampling (temp/top-k/top-p/repeat-penalty) + KV cache |
| `sample.py` | generation CLI (research: every knob, cached-vs-uncached timing) |
| `generate.py` | inference-only CLI: CPU, torch-only deps, no training code paths; `--chat` REPL with system prompts, tools, `--doc` |
| `web_ui.py` | browser chat UI (gradio or stdlib): streaming, thinking folds, tool cards, upload, latency monitor |
| `personas.py` | system-prompt presets, `<think>` parsing + stream filter, marker sanitising, prompt shrinking (§11.1, §11.3) |
| `rag.py` | in-memory retrieval: chunking, BM25 + trigram fallback, budgeted context block, grounding (§11.4) |
| `tools.py` | calculator + whitelisted python interpreter, question detection, stream-safe markers (§11.2) |
| `07_make_sft.py` | the synthetic corpus: chat, CoT traces, document-grounded rows |
| `08_sft.py` | response-only SFT + CoT/document curricula and post-training checks |
| `run.ps1` / `run.bat` | one-click Windows launchers for `generate.py` |
| `run_web.ps1` / `run_web.bat` | one-click Windows launchers for `web_ui.py` |
| `data/tinystories/` | cleaned TinyStories corpus (20 MB, committed) + `stats.json` provenance |
| `data/packed/` | tokenised uint16 blocks, story-level train/val split (committed) |
| `data/raw/` | the three Gutenberg books, for `--mode docs` |
| `tokenizer/`, `out/` | vocabulary, the committed checkpoint + `config.*`, `log.jsonl`, `eval.json`, `MODEL_CARD.md` |

## 13. The bugs, catalogued

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
| document rows trained the model on *rephrased* answers, with refusals 35% of the family and pinned as hard as the answers | the model answered "The document does not say." to **every** document question, including three about a passage that plainly held the answer | answer with the passage's *own sentence*, refusals down to ~9% of the family (184/2,027), and pin two answering rows at `--doc-pins` but the refusal at half of it |
| twelve fixed names paired with twelve fixed facts | the trained model answered document questions from *memory*: given a passage saying the marlow lives in Low Fen it replied with the fact it had memorized, and it refused every subject it had not seen | pair names and values **at random** across rows, shuffle the sentences next to a second creature's, and check two held-out subjects — one with the corpus's sentence shapes, one with shapes no training row contains |
| a training row longer than the 256-token window | `encode_example` front-trims it, silently deleting part of the prompt or of the injected document, so the row teaches the opposite of what it was written for — and no error is raised | the corpus maker counts chat tokens with the real tokenizer and drops any row that does not fit (a char-count proxy is wrong: an 815-char row is 256 tokens, a 636-char one is 257); the SFT suite re-asserts the invariant |
| `--doc-check`'s reply budget was set to half the passage budget | document questions that opened a `<think>` block were cut off mid-trace, so the check measured its own token cap instead of the model | a named `DOC_CHECK_REPLY_TOKENS = 64` for the reply, separate from the passage budget |
| a stale `_int8.pt` next to a fresh fp32 checkpoint | both frontends auto-prefer the INT8 file when it exists, so `run.ps1 -Chat` answered with the *previous* fine-tune until it was re-quantized | quantize in the same step as training, and say so in the reproduce steps |
