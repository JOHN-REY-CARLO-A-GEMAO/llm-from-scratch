# Model card — `out/model_best.pt`

A 1.93 M-parameter Llama-3 style language model trained from scratch on TinyStories,
on 2 CPU cores, in 33 minutes. This card records what it is, what it scores, what it
writes, and what it cannot do.

## What it is

| | |
|---|---|
| architecture | decoder-only transformer, **Llama-3 style**: RMSNorm, RoPE, SwiGLU, `F.scaled_dot_product_attention`, pre-norm blocks, depth-scaled init, tied embeddings |
| parameters | **1,926,560** (34% in the 4096×160 token embedding; 1,271,200 non-embedding) |
| layers / heads / width | 4 / 5 / 160 (head dim 32, SwiGLU hidden 448) |
| context window | 256 tokens (RoPE base 10,000 — extrapolates beyond, degrades gracefully) |
| tokenizer | byte-level BPE trained from scratch **on this corpus**: 4096 tokens, 3,839 merges, 4.49 chars/token, one special token `<|doc_end|>` |
| training data | **20.0 MB of TinyStories** (Eldan & Li 2023, arXiv:2305.07759) → 25,147 stories → 4.14 M train tokens + 216 k held-out tokens |
| training | 1,400 steps × 8,192 tokens = **11.5 M tokens** (~2.8 epochs), AdamW (β 0.9/0.95, wd 0.1), cosine LR 1e-3 → 1e-4, 100-step warmup, grad-clip 1.0, fp32, dropout 0 |
| hardware / wall clock | 2 CPU cores (Xeon @2.6 GHz, AVX-512), **33 minutes**, ~8,600 tok/s |
| artefacts | `out/model_best.pt` (7.7 MB state_dict), `out/config.json` + `out/config.txt` (identical), `out/eval.json`, `out/log.jsonl` |

## Training curve (held-out stories)

```
iter     0  loss 8.3584        ln(4096) = 8.318: the uniform baseline, so init is correct
iter   100  val_loss 4.8810   val_ppl 131.76     <- already below the bigram Markov floor
iter   400  val_loss 3.2065   val_ppl  24.69
iter   800  val_loss 2.7640   val_ppl  15.86
iter  1200  val_loss 2.5975   val_ppl  13.43
iter  1400  val_loss 2.5663   val_ppl  13.02     <- out/model_best.pt (fixed 32,768-token eval window)
```

## Metrics

`python 05_evaluate.py --ckpt out/model_best.pt --data-dir data/packed --autocast none`
(bigger window: 512 of the 844 held-out blocks = 131,072 tokens — hence the slightly
higher loss than the trainer's own 32,768-token eval):

| metric | value |
|---|---|
| val loss (cross-entropy, nats/token) | **2.5826** |
| val perplexity | **13.23** |
| top-1 next-token accuracy | **45.26%** |
| bits per byte (tokenizer-independent) | **0.830** |
| tokens per byte | 0.223 |

Baselines on the same split — the reason a perplexity number alone is not a result:

| predictor | perplexity |
|---|---|
| uniform over 4096 tokens | 4,096 |
| unigram (frequency only, zero context) | 613.28 |
| bigram Markov chain (1 token of context) | 52.84 |
| **this model** (256 tokens of context) | **13.23** |

The model is **75.0% lower in perplexity** and **34.9% lower in cross-entropy**
than the bigram floor. The gap between those two percentages is not a rounding
artefact: perplexity is exponential in cross-entropy, so quoting only the first
inflates the win.

## Sample generation

```bash
python generate.py --prompt "Once upon a time, Lily found a little kitten" --n 150 --temp 0.7 --seed 2024
```

**seed 2024** — a typical draw. 150 new tokens; the blank line is the `<|doc_end|>`
separator, i.e. the model finished a story and started another one, exactly as the corpus
asked it to:

```
Once upon a time, Lily found a little kitten on the ground. She picked it up and showed it to her mommy. Her mommy said, "Oh no, Lily. We need to take care of it." As they were walking, Lily's mommy saw some other birds in the garden. It was so pretty that they forgot about it. Lily didn't want to share her juice with her mommy. But her mommy told her that it's important to listen to others, even if things happen at first, we have to be careful. So, Lily just put out her favorite toy and went home with a big smile on her face.

One day, a boy named Tim went for a walk. He liked to play near the trees. He saw many animals, but he wanted to play with it. Tim started to play with his ball. He ran as fast as he could. But when he got there, something unexpected happened.
```

**seed 1337** — the identical command with the training seed. Still grammatical and
story-shaped, and it also ends a story and starts a new one, but the kitten becomes an apple
in the second sentence and the moral loops back on itself. Both samples are honest draws from
the same distribution; quoting only the flattering one would be cherry-picking.

```
Once upon a time, Lily found a little kitten. It was a big, red apple that could open it. She picked it up and showed it to her mommy. "Mommy, can we find the apple?" she asked. "Sure, but be careful!" Her mommy replied. After they finished eating, Lily's mommy told her that it's okay to take care of it, but it's important to listen to your friends and not take things from me." Lily smiled and said, "It's okay. We should ask for help when you don't need to stay safe." The next day, Lily went home with her mommy. She felt sad because she loved playing outside with her friends. From that day on, Lily learned that sometimes it's better to share and make sure what you look, even if we want to play with it."

Once upon a time, there was a little girl named Lily. She had a pretty dress
```

## How to reproduce

```bash
python 00_tinystories.py --source hf-http --max-bytes $((32<<20)) --max-chars 20000000
python 01_tokenizer.py  --corpus data/tinystories/corpus.txt --out tokenizer --vocab 4096 --train-chars 4000000
python 02_preprocess.py --corpus data/tinystories/corpus.txt --tokenizer tokenizer --out data/packed --mode stream
python 04_train.py --data-dir data/packed --out-dir out --iters 1400 --warmup 100 --batch-size 32 \
    --d-model 160 --n-layer 4 --n-head 5 --lr 1e-3 --block-size 256 --eval-every 100 --autocast none --seed 1337
```

Deterministic on the same machine: fixed seed, CPU, dropout 0 (this run was reproduced
byte-for-byte after an accidental `rm`).

## The four numbers that say the pipeline is correct

1. **Step-0 loss ≈ ln(4096) = 8.318** — measured 8.3584. Far above it means the head or
   the init is mis-scaled.
2. **4.49 chars/token** — the tokenizer actually applies its merges. ~1.0 means it does not.
3. **Val loss below the bigram Markov floor (52.84 ppl)** — crossed before step 100.
4. **Loss still improving when the cosine schedule ends** — it did, monotonically:
   131.76 → 47.38 → 31.32 → 24.69 → 21.05 → 18.67 → 16.95 → 15.86 → 14.97 → 14.24 → 13.78
   → 13.43 → 13.19 → 13.02 (ppl). More tokens/data would still help, which is the point.

## Honest limits

- **Semantics drift.** A 1.93 M-parameter model has no world knowledge; in the seed-1337
  sample above a kitten becomes an apple (seed 2024 keeps the kitten). Grammar, punctuation, dialogue and story *shape* are
  learned well; noun-level meaning is not reliable beyond a few sentences.
- **Distribution is TinyStories alone.** Simple children's English, ~1.5 k-word
  vocabulary, no facts, no dates, no reasoning, no instruction-following. Converting a
  prompt into an unrelated book/scene vocabulary will get prose that reads fluently and
  says nothing true.
- **256-token window.** Longer continuations need either a longer block size at training
  time or a larger `rope_base`; generation stops at the trained limit by design.
- **It is a base model.** There was no instruction tuning, no RLHF, no chat template;
  `--interactive` is a prompt loop over a raw LM, not a conversation.
- **Where the numbers come from.** Validation is 5% of the *stories*, held out before
  tokenisation, so no val token is in any training block. The 20 MB corpus was itself
  deduplicated by content hash — duplicated data inflates apparent quality, and the
  first candidate copy of the dataset found in the wild was 98% duplicates.

## The chat fine-tune (`out/model_15m_chat.pt`)

`08_sft.py` continues this recipe on the 15 M-parameter checkpoint with a
response-only loss (user text, role markers, prior turns and padding are `-100`),
and the corpus carries three things the base model never saw: `<think> … </think>`
reasoning traces on questions that need several steps, `<<calc: …>>` /
`<<tool: …>>` markers whose arithmetic is substituted by `tools.py` at inference
time, and `rag.context_block`-formatted document passages in a system turn.
Measured on this machine, and reproducible from the command recorded in
`out/chat_run_config.json`:

| check | what it measures | result |
|---|---|---|
| validation loss | held-out rows, assistant tokens only | see `best_val_loss` in `out/chat_run_config.json` |
| `--cot-check 3` | greedy reply opens *and closes* a `<think>` block on complex questions | see `cot_check` |
| `--doc-check 3` | questions about **two** subjects that appear nowhere in training, answered from an injected passage — one series in the corpus's sentence shapes, one in shapes no training row uses, plus a question the passage cannot answer | see `doc_check` |

Honest limits, in the same register as the base model's:

- **The trace is a format, not a proof.** The steps are numbered, ordered and
  parsed by the UI, but the arithmetic *inside* a trace is only exact where the
  model used a `<<calc: …>>` marker — a 7 M-parameter model is a poor calculator,
  and the corpus deliberately teaches it to hand the arithmetic to a tool.
- **Document reading is the weak spot, and it is measured.** `--doc-check 3`
  hands the model a passage about a subject that appears *nowhere* in training
  and asks about it; the honest result on this checkpoint is in
  `out/chat_run_config.json`, and it is a partial pass at best. The three
  measurements behind that number:
  *(1)* the model reproduces a **training** passage exactly (replaying a row's
  own prefix recovers its recorded answer), so what looks like reading can be
  recall;
  *(2)* the same model, trained on the document family **alone** for 400 steps to
  a validation loss of 0.0015, still could not copy a held-out passage — it
  answered "The document does not say." So in-context copying did not emerge at
  this scale; it is a capacity/data-scale limit, not a mixture accident. The
  shipped run scores 2/8 on `--doc-check` (both honest refusals, none of the six
  held-out facts), and a live session shows the split exactly: asked where the
  marlow lives in a document that says *the tall grass of the Plain of Tesh*, it
  answers "The cobb lives in the still water of the Mere." — right shape, wrong
  source, `grounding` 0.19;
  *(3)* the corpus was therefore rebuilt to make recall useless — names and facts
  are paired at random across rows, sentences are shuffled next to a second
  creature's, and the check uses two held-out subjects and one series of sentence
  shapes no training row contains.
  The practical consequence for the UI: the retrieved passage is always shown
  next to the reply (sources card) and a reply that shares almost nothing with it
  is flagged. Document Q&A is useful because the *retrieval* is exact and the
  passage is on screen; the 7 M-parameter model reading it is the part that fails.
- **Grounding is measured, not promised.** The share of the reply's content words
  that appear in the retrieved passage is reported (`grounding`), and the page
  warns when a reply shares almost nothing with the document. A well-phrased
  paraphrase can score low; a memorised answer can score high.
- **No system-prompt obedience.** The corpus trains two system blocks (the
  document context and the reasoner/math hints); arbitrary prose in that slot
  steers tone and vocabulary, not capability.
- **Selection effects.** `--cot-pins`/`--doc-pins` deliberately overweight a
  handful of examples. That is a curriculum, not the natural distribution, and
  the checks above are the price of it: results are reported whether they pass
  or fail.
- **The documents are invented, and the check knows it.** The document rows teach
  a *skill* — find the sentence, copy it, or say the document does not say — on
  invented creatures and places, and their facts are paired with names at random
  across rows so there is no table to memorize. The held-out values in
  `--doc-check 3` are the same kind of invented fact: they measure whether the
  model reads the passage it was handed, not whether it knows the world.
