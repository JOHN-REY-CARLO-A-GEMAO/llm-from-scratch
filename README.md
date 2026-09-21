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

The repository also ships a supervised fine-tune of the 15 M-parameter
TinyStories checkpoint, trained with chain-of-thought traces (600 steps,
validation loss 0.106, perplexity 1.11 on 256 held-out rows). Start a
multi-turn conversation with:

```powershell
.\run.ps1 -Chat
```

or directly:

```bash
python generate.py --chat
```

Chat mode selects `out/model_15m_chat.pt` (falling back to `out/model_chat.pt`)
and `tokenizer/chat`, wraps every turn as `<|im_start|>user\n...<|im_end|>` /
`<|im_start|>assistant\n...`, retains complete turns in the 256-token context,
and stops as soon as `<|im_end|>` is sampled. Type `/clear` to discard history
or `/exit` to leave. The base continuation mode and `out/model_best.pt` remain
unchanged.

Three things are layered on top of the plain text turn, and each one has a
deterministic path that does not depend on the model being clever:

| layer | what the model does | what the program does instead |
|---|---|---|
| **chain-of-thought** | plans inside `<think> … </think>` before complex answers | splits the block out and renders it as a fold; reports `closed`/`reasoning_chars` |
| **tools** | may write `<<calc: …>>` / `<<tool: …>>` | evaluates it locally and substitutes the exact value mid-stream |
| **document Q&A** | *may* answer from a passage in its system turn — the measured weak spot, see below | retrieves the passage with `rag.py`, injects it in the exact template, shows the sources, and flags a reply that shares almost nothing with it |

#### Chain-of-thought

`07_make_sft.py` writes a reasoning trace for the questions that need more than
one step — why-questions, word problems, comparisons, document questions — and a
direct answer for the ones that do not (greetings, one-fact answers, `5 + 5`),
so the model learns *when* to reason instead of prefixing every reply with a
template:

```text
<|im_start|>assistant
<think>
1. Identify the core question: why does it rain.
2. Check the key facts: water warms, rises into the sky, and cools into tiny drops.
3. Put the facts together and answer in one or two short sentences.
</think>
Water warms, rises into the sky, and cools into tiny drops in clouds. When the drops grow heavy, they fall as rain.
```

2,296 of the 6,000 rows carry a trace over five different step wordings, and
`08_sft.py --cot-pins 96` repeats three of them far beyond their natural
frequency: a 7 M-parameter model drifts back to its pretrained "answer
immediately" habit without that. The traces are only a *format*; the arithmetic
inside them is exact only where the model used a `<<calc: …>>` marker (below).

Honest limits, measured on the shipped checkpoint by
`08_sft.py --cot-check 3` (greedy, temperature 0) and recorded in
`out/chat_run_config.json`:

* the trace opens and closes correctly on complex questions, and the answer
  follows it;
* the *numbers inside* a multi-step word problem can still be wrong — a 7 M
  model is a poor calculator, which is exactly why the corpus teaches it to put
  the arithmetic in `<<calc: …>>`;
* a question the corpus never taught gets the trace habit, not a true plan.

#### Python tool (arithmetic, strings, dates — not just math)

`tools.py` is a whitelisted interpreter, not a sandbox and never `eval`/`exec`:
the AST is walked and anything outside the list is refused. It answers four
kinds of question exactly:

```bash
python generate.py --chat
You> 5 + 5
[calc] '5 + 5' -> 5 + 5 = 10 (computed locally; the model was not asked)
You> how many words are in "the quick brown fox jumps over the lazy dog"?
[tool] word_count('the quick brown fox jumps over the lazy dog') -> 9 (computed locally)
You> what is the date today?
[tool] today() -> 2026-09-21 (computed locally)
You> what day of the week is 2026-12-25?
[tool] weekday_name('2026-12-25') -> 'Friday' (computed locally)
```

Supported: exact rational arithmetic (`0.1 + 0.2` is `0.3`), string operations
(`word_count`, `char_count`, `reverse`, `upper`/`lower`, `split`, `join`,
slicing, `replace`), date/time (`today`, `now`, `date_of`, `days_between`,
`days_until`, `weekday_name`, `days_in_month`, `strftime`), and short multi-line
programs (`minutes = 60\nhours = 2.5\nminutes * hours` → `150`). Refused:
`import`, attribute escapes (`().__class__`), comprehensions, `lambda`, `while`,
`open`, f-strings, and anything that builds more than a bounded amount of text
or numbers. Numeric results stay exact (`Fraction`), so the reply can be checked
character for character.

A question the detector recognises is answered without sampling anything
(`n_tokens: 0`, `stop_reason: "tool"`) — the value is in the message meta as
`🧮` or `🐍`, so a computed number is never mistaken for a sampled one. In prose
("what is 12 * 4?") the model still answers, and the exact line is appended if
the model's own number disagrees.

#### Document Q&A (local RAG, zero dependencies)

`rag.py` is an in-memory retrieval engine with **no vector database, no
embeddings package, no network**: BM25 over word unigrams and bigrams, with a
character-trigram cosine fallback for typos, chunking on paragraph and sentence
boundaries (900 characters, 150 of overlap). Ask about the uploaded file and the
`top-k` passages are injected into the system turn, in exactly this shape:

```text
<|im_start|>system
Context from uploaded document:
[1] The zorble lives in the warm shallows of the Cora River. …

Answer the user's question based ONLY on this context.<|im_end|>
```

Retrieval happens before sampling, and the block is fitted to the model's
256-token window: a passage that does not fit is trimmed to its most relevant
window rather than the persona text surviving instead. If nothing clears the
relevance floor, **no context is injected at all** — the page says so instead of
padding the prompt with an unrelated passage. The reply is compared back against
the retrieved text (fraction of content words shared) and the UI warns when a
reply shares almost nothing with the document, because "grounded" is a
measurement here, not a promise.

2,027 of the 6,000 SFT rows carry a real `rag.context_block` system turn whose
facts are invented (`zorble`, `quibble`, the Cora River), so answering means
reading the passage rather than remembering it: the answer is the passage's *own
sentence*, copied, and the sentences are shuffled inside the passage next to a
second creature's. The names and facts are paired **at random** across rows —
the first version of this family fixed 12 names to 12 facts, and the trained
model answered document questions from memory (asked where the `marlow` lives
with a passage that said *Low Fen*, it replied with the fact it had memorized
from training). A mapping that cannot be memorized leaves copying as the only way
to be right. 184 of the rows ask about a fact the passage does not contain and
are trained to answer `The document does not say.` — the honest answer, trained
rather than hoped for. A further 500 rows carry the *persona* blocks the UI
actually sends, so a system turn the model has never seen next to a question
cannot derail it.

`08_sft.py --doc-check 3` re-tests the copy task after training with two
**held-out** subjects that appear in no training row, in two series: one with the
sentence shapes the corpus uses, one with shapes no training row ever shows. A
model that parrots a template passes the first and fails the second; a model that
reads the passage passes both. Each series also asks a question the passage cannot
answer, because "always copy something" and "never copy anything" fail in
opposite directions and each looks fine on the other's test. All of it is
recorded in `out/chat_run_config.json`.

**The measured result is a partial pass, and the reason is scoped.** Asked about
the passage at the top of this section (which says the marlow lives in the tall
grass of the Plain of Tesh), the shipped checkpoint answers *"The cobb lives in
the still water of the Mere."* — the **format** is perfect (a trace, then one
sentence about one creature) and the **content** is recall, not reading. The
sources card shows the passage, the meta line reports `grounding` below the
threshold, and the page adds the ⚠ warning, so the wrong answer is at least
labelled as one. The
checkpoint reproduces a *training* passage exactly (replaying a row's own prefix
recovers its recorded answer), and a fourth run trained on the document family
**alone** — 400 steps to a validation loss of 0.0015 — still answered
"The document does not say." to a held-out passage. In-context copying did not
emerge at 7 M parameters; the model stores passages far more readily than it
looks one up. So document Q&A is honest about which half works: the retrieval,
the exact injected template, the sources card and the grounding warning are exact
and tested, and the passage is always on screen next to the reply. Reading it is
the part this checkpoint gets wrong, and the page says so instead of pretending.

```bash
python generate.py --chat --doc notes.md     # preload a document in the terminal
python web_ui.py --doc notes.md              # ...or in the browser
```

#### System prompts (personas)

A system turn is prepended as `<|im_start|>system\n...<|im_end|>`:

```bash
python generate.py --chat --persona pirate                       # preset
python generate.py --chat --system "You are a terse robot."      # free text
python generate.py --chat --persona none                         # no system turn
```

Presets are `friendly` (default), `pirate`, `math`, `storyteller` and `reasoner`;
the same
list is the dropdown in the web UI, and `personas.py` is the single source of
truth for both. Inside the REPL, `/persona` lists them, `/persona pirate`
switches, `/system TEXT` sets free text and `/system` shows what is in force.
A system turn steers tone, and (for `reasoner`, or `math` with the calculator
hint) adds an instruction the model was trained to follow — `personas.COT_HINT`
asks for a `<think>` block on questions that need more than one step. The
checkpoint was not trained to *obey* arbitrary prose in a system turn, so read
most of it as tone steering; `--persona none` reproduces the framing most of the
corpus used. Role markers typed into a prompt or a message
(`<|im_start|>`, `<|im_end|>`, …) are stripped, so the turn structure is always
assembled by the program.

#### Calculator (the arithmetic hook, in detail)

`5 + 5` is not a language question, so it is not asked of the model. An explicit
arithmetic question is evaluated with `tools.py` (a whitelisted `ast` walk —
never `eval`) and answered exactly:

```bash
python generate.py --chat
You> 5 + 5
[calc] '5 + 5' -> 5 + 5 = 10 (computed locally; the model was not asked)
Assistant> 5 + 5 = 10
You> what is 12 * 4?
Assistant> 12 plus 4 is 20

🧮 12 * 4 = 48
```

Three paths, one rule — the exact value always appears: a bare expression is
answered locally; a math question inside prose gets the model's reply *plus* the
computed line; and if the model writes `<<calc: 5 + 5>>`, the marker is replaced
by `10` while it streams. Arithmetic is exact rational (`0.1 + 0.2` is `0.3`,
integers stay integers), and `--no-tools` disables the hook entirely.

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

### Browser chat UI

The same chat checkpoint is also wrapped in a dark-mode web interface —
tokens stream into the page as they are generated, **Temperature** (0.1–1.5)
and **Top-P** (0.1–1.0) are sliders, a **System Prompt** dropdown (🤖 Friendly
Assistant, 🏴‍☠️ Pirate, 🧮 Math Tutor, 🏰 Storyteller, 🧠 Reasoner, ✏️ Custom…,
∅ None) steers the persona, **📄 Upload Document (.txt / .md)** loads a file to
answer from, and **Clear Chat** wipes the history:

```powershell
.\run_web.ps1       # Windows: serve + open the browser  (double-click run_web.bat also works)
```

or directly:

```bash
python web_ui.py    # http://127.0.0.1:7860  (also bound on 0.0.0.0:7860 for your LAN)
```

It loads `out/model_15m_chat.pt` (or the INT8 variant, auto-detected) +
`out/model_15m_chat_config.json`, uses a **gradio** `ChatInterface` when gradio is
installed (`pip install gradio`), and falls back to a built-in zero-dependency
`http.server` page when it is not — nothing extra to install beyond PyTorch.
Endpoints: `GET /` (the page), `POST /api/chat` (newline-delimited JSON event
stream), `POST /api/stop`, `GET /api/info`, `GET /api/docs`, `POST /api/upload`,
`POST /api/docs/clear`. Multi-turn history, the `<|im_start|>` framing, the
system turn, the tools and context trimming behave exactly like the terminal chat
above — the browser posts `{message, history, temperature, top_p, system,
persona}` and the server re-serialises the whole prompt with the same code the
REPL uses. `--backend stdlib|gradio|auto`, `--no-browser`, `--port`, `--host`,
`--persona NAME`, `--system TEXT`, `--no-tools`, `--doc FILE` and `--no-rag`
adjust the launch (`--persona` / `--system` set which persona the page starts on).

Four things the terminal cannot show:

* **Thinking folds** — a `<think>` block renders as a collapsible
  DeepSeek-style card above the answer ("🧠 Thinking… · 412 chars" while it
  streams, then "🧠 Reasoning · 412 chars"), with a **Copy answer** button that
  copies the answer *without* the trace. The block is parsed by the same rules as
  `personas.split_think`, and `tests/test_web_ui.py` runs both implementations
  over the same replies to keep them identical.
* **Tool cards** — a `🐍 Python Tool Execution` card shows the exact source and
  its result (`word_count('one two three')` → `3`), and a computed answer is
  labelled `🧮` in the message meta, so a computed number is never mistaken for a
  sampled one.
* **📄 Upload Document (.txt / .md)** — the browser reads the file and POSTs its
  text (no multipart parser in the stdlib, nothing written to disk); the server
  chunks and indexes it in memory and shows `name · N chunks · N chars` in the
  header, with a ✕ to unload it. Every reply then carries a **sources** fold
  listing the retrieved passages with their relevance scores, plus the share of
  the reply that came from the document.
* A **token counter / latency monitor** in the header (live tokens, tok/s and
  elapsed time while streaming; then time-to-first-token, prompt size, stop
  reason and seed for the finished turn, plus session totals), **Copy / Copy
  response / Copy code** buttons on every message, and the persona picker.

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
07_make_sft.py         deterministic 6,000-example dataset: chat + CoT traces + document context
08_sft.py              response-only-loss SFT; CoT and document-grounding curricula + checks
generate.py            continuation + multi-turn chat CLI (CPU, torch-only deps, --doc/--persona)
web_ui.py              browser chat UI: streaming, thinking folds, tool cards, upload, latency monitor
personas.py            system-prompt presets, <think> split/stream filter, marker sanitising
rag.py                 zero-dependency retrieval: chunking, BM25 + trigram, context block, grounding
tools.py               calculator + python interpreter (strings/dates), question detection, markers
run.ps1 / run.bat      one-click Windows launchers for generate.py
run_web.ps1 / .bat     one-click Windows launchers for the browser chat UI
sample.py              research generation CLI
_stage.py              import shim (module names can't start with a digit)
data/tinystories/      the cleaned corpus + provenance (committed)
data/packed/           tokenised uint16 blocks the trainer reads (regenerated, git-ignored)
out/                   the trained checkpoint + its config, metrics and model card
tests/                 property-based tests for corpus cleaning, tokenizer, model, sampler, CLI,
                       tools/personas (incl. CoT parsing), rag.py, and the web UI end-to-end
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

# 7. instruction-tune on the 15 M checkpoint (chat + chain-of-thought + document
#    grounding; ~8 min on 2 CPU cores). --out names the shipped chat checkpoint.
python 07_make_sft.py --out data/sft/chat_sft.jsonl
python 08_sft.py --data data/sft/chat_sft.jsonl \
    --model out/model_15m_best.pt --config out/model_15m_best_config.json \
    --out out/model_15m_chat.pt --iters 900 --batch-size 16 --lr 1e-4 --warmup 20 \
    --cot-pins 96 --cot-check 3 --doc-pins 64 --doc-check 3 --seed 2026
# INT8 is what the frontends load *by default* when the file is present, so a
# stale `_int8.pt` next to a fresh fp32 checkpoint is a silent wrong answer:
python quantize.py --in out/model_15m_chat.pt \
    --config out/model_15m_chat_config.json --out out/model_15m_chat_int8.pt

# 8. chat with it (terminal or browser); --doc preloads a document for local Q&A
python generate.py --chat --persona pirate
python web_ui.py --doc README.md

# all tests (12 suites: 11 unittest files + 03_model.py's self-test)
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
