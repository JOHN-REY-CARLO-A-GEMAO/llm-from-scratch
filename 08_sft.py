"""Supervised fine-tuning of the TinyStories checkpoint for chat.

Only response tokens (including ``<|im_end|>``) contribute to cross entropy.
User text, role markers, and right padding use label -100. Two chat markers are
appended to the existing tokenizer, preserving every pretrained token id.

## Chain-of-thought

`07_make_sft.py` writes reasoning traces (`<think> … </think>`, see
`personas.wrap_think`) for the questions that need more than one step, and a
direct answer for the ones that do not. Three things make that stick in a 7 M
parameter model:

 * **Volume.** Roughly a third of the corpus carries a trace, over several
   phrasings of the same three moves, so the block is a habit rather than a
   memorised string.
 * **A pinned curriculum.** `--cot-pins` repeats a handful of reasoned examples
   (one explanation, one word problem, one comparison) until they are a
   meaningful share of the batch: a small model drifts back to its pretrained
   "answer immediately" prior without them.
 * **A check, not a hope.** `--cot-check` generates a reply to a few complex
   questions with the reasoner persona after training and reports whether a
   *closed* `<think>` block came out. The result is written to
   `chat_run_config.json` (`cot_check`), because "the model reasons now" is a
   claim that has to be reproducible.

## Document grounding

Some corpus rows carry a *system* turn holding a
`rag.context_block`-formatted passage, and the answer is only in that passage
(`07_make_sft.py`). `encode_example` therefore takes an optional `system` text
and serializes the system turn in front of the user turn, exactly as
`generate.serialize_chat` does at inference time. `--doc-check` then asks a
question about an entity that appears **nowhere** in training, with the passage
that answers it injected: a reply that reuses the passage's words is a
fine-tune that reads its context, and one that invents a habitat is a
fine-tune that does not. Both are recorded (`doc_check`), not asserted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from _stage import load
import personas
import rag                                    # stdlib-only, torch-free

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IGNORE = -100


def make_chat_tokenizer(base_dir: Path, output_dir: Path):
    tok_mod = load("01_tokenizer")
    base = tok_mod.ByteLevelBPE.load(base_dir)
    if base.added_tokens and base.added_tokens != [IM_START, IM_END]:
        raise ValueError(f"unexpected existing added tokens: {base.added_tokens}")
    base.added_tokens = [IM_START, IM_END]
    base._build_vocab()
    base.save(output_dir)
    return base


def encode_example(tok, prompt: str, response: str, block_size: int, system: str | None = None):
    """One supervised turn. `system` is the document/persona block in front of it.

    Nothing about the system turn is supervised: it is context, like the user
    turn, and only the assistant tokens (and its `<|im_end|>`) carry a label.
    """
    specials = [IM_START, IM_END]
    head = f"{IM_START}system\n{system}{IM_END}\n" if system else ""
    prefix = head + f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
    full = prefix + response + IM_END
    prefix_ids = tok.encode_with_specials(prefix, specials)
    ids = tok.encode_with_specials(full, specials)
    if len(ids) > block_size:
        # Responses are intentionally short. If a custom row is long, retain its
        # end and as much user context as fits rather than training over context.
        trim = len(ids) - block_size
        ids = ids[trim:]
        response_at = max(0, len(prefix_ids) - trim)
    else:
        response_at = len(prefix_ids)
    x, y = ids[:-1], ids[1:]
    # y[j] is the token at ids[j+1]. The first response token is ids[response_at].
    labels = [IGNORE if j + 1 < response_at else target for j, target in enumerate(y)]
    if not any(v != IGNORE for v in labels):
        raise ValueError("example has no assistant target tokens")
    return x, labels


def encode_conditioned(tok, previous: dict, current: dict, block_size: int):
    """Train a response after one completed turn, as seen by chat_repl.

    The previous turn is context-only. The current response and its end marker
    are supervised, while both users, role markers, and prior answer are masked.
    """
    specials = [IM_START, IM_END]
    prior = (f"{IM_START}user\n{previous['prompt']}{IM_END}\n"
             f"{IM_START}assistant\n{previous['response']}{IM_END}\n")
    prefix = prior + f"{IM_START}user\n{current['prompt']}{IM_END}\n{IM_START}assistant\n"
    full = prefix + current["response"] + IM_END
    prefix_ids = tok.encode_with_specials(prefix, specials)
    ids = tok.encode_with_specials(full, specials)
    if len(ids) > block_size:
        trim = len(ids) - block_size
        ids = ids[trim:]
        response_at = max(0, len(prefix_ids) - trim)
    else:
        response_at = len(prefix_ids)
    labels = [IGNORE if j + 1 < response_at else target
              for j, target in enumerate(ids[1:])]
    return ids[:-1], labels


def collate(rows, indices, pad_id: int, device):
    selected = [rows[i] for i in indices]
    width = max(len(x) for x, _ in selected)
    x = torch.full((len(selected), width), pad_id, dtype=torch.long, device=device)
    y = torch.full((len(selected), width), IGNORE, dtype=torch.long, device=device)
    for i, (ids, labels) in enumerate(selected):
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
    return x, y


def masked_loss(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE)


@torch.no_grad()
def evaluate(model, rows, indices, batch_size, pad_id, device, amp):
    model.eval()
    total, tokens = 0.0, 0
    for start in range(0, len(indices), batch_size):
        x, y = collate(rows, indices[start:start + batch_size], pad_id, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp and device.type in {"cpu", "cuda"}):
            loss = masked_loss(model, x, y)
        n = int((y != IGNORE).sum())
        total += loss.item() * n
        tokens += n
    model.train()
    return total / tokens


#: A held-out entity for `--doc-check`: it appears in no training row, so an
#: answer that names its habitat got it from the injected passage.
#: How many new tokens the grounding check may sample. 96 is enough for a short
#: answer plus a trace, and it leaves the passage most of the 256-token window.
DOC_CHECK_TOKENS = 96
#: Tokens the check allows for the reply (the frontends allow more; this is a cap,
#: not the model's limit).
DOC_CHECK_REPLY_TOKENS = 64

#: Two held-out subjects, neither of which appears in any training row. The first
#: is checked with the *same* sentence shapes the corpus uses, the second with
#: shapes (`DOC_ALT_*` in 07_make_sft.py) no training row ever shows. A model
#: that parrots the shape it learned passes the first and fails the second; a
#: model that reads the passage passes both, which is the difference between a
#: document feature and a template.
#: `(series, entity, (home, food, sound), sentence templates, questions)`.
DOC_CHECK_SERIES = (
    ("trained shapes", "wobbin",
     ("the salt flats of Grim", "amber seeds", "a low wooden knock"),
     ("The {name} lives in {value}.", "The {name} eats {value}.",
      "The {name} makes {value}."),
     ("Where does the {name} live?", "What does the {name} eat?",
      "What sound does the {name} make?")),
    ("unseen shapes", "gorlin",
     ("the cold spring of Havn", "bitter cloudberries", "a single ringing note"),
     ("You can find the {name} in {value}.", "The {name} feeds on {value}.",
      "The {name} calls out with {value}."),
     ("Where can I find the {name}?", "What is the food of the {name}?",
      "What noise does the {name} make?")),
)
#: A question the passage cannot answer, per series: the trained reply is the
#: honest one. It is checked too, because "always copy something" and "never copy
#: anything" fail in opposite directions and each looks fine on the other's test.
DOC_CHECK_MISSING = ("What colour is the {name}?", "What colour is the {name}?")


@torch.no_grad()
def check_grounding(model, tok, cfg, device, args) -> dict | None:
    """Ask about held-out subjects with their passages injected as the system turn.

    The reply is graded against the passage: a hit is the passage's own words
    coming back out. This is the cheapest honest test of "does the checkpoint use
    the retrieved context, or does it answer from its prior" — and the first
    version of the corpus failed it, which is why the *values* are held out here
    and the second series changes the sentence shapes as well.
    """
    gen_mod = load("06_generate")
    import rag                                    # stdlib-only, torch-free

    cost = lambda text: len(tok.encode_with_specials(text, [IM_START, IM_END]))   # noqa: E731
    end_id = tok.token_to_id[IM_END]
    params = gen_mod.SamplingParams(temperature=0.0, top_k=1, top_p=1.0,
                                    repeat_penalty=1.0, seed=0)
    was_training = model.training
    model.eval()

    def ask(passage: str, question: str) -> tuple[str, int]:
        index = rag.DocumentIndex.from_text(passage, name="held-out.txt")
        system, hits = index.context_block(question,
                                           cfg.block_size - DOC_CHECK_TOKENS, cost, top_k=1)
        text = (f"{IM_START}system\n{system}{IM_END}\n" if system else "") + \
               f"{IM_START}user\n{question}{IM_END}\n{IM_START}assistant\n"
        ids = tok.encode_with_specials(text, [IM_START, IM_END])
        # 64 tokens, not half the passage budget: a document question may open a
        # `\u003cthink\u003e` block first, and a check that cuts the reply off mid-trace
        # measures its own budget rather than the model. 96 + 64 still leaves the
        # 256-token window room to spare.
        out, _, _ = gen_mod.generate(model, ids, DOC_CHECK_REPLY_TOKENS, params, use_cache=True,
                                     eos_id=end_id, device=device, block_size=cfg.block_size)
        reply = tok.decode(out[len(ids):]).strip()
        for marker in personas.MARKERS:          # control markers are not prose
            reply = reply.replace(marker, "")
        return reply, len(hits)

    series_rows: list[dict] = []
    for (label, name, values, templates, questions), missing in zip(
            DOC_CHECK_SERIES, DOC_CHECK_MISSING):
        passage = " ".join(template.format(name=name, value=value)
                           for template, value in zip(templates, values))
        rows: list[dict] = []
        for question, expected in zip(questions, values):
            asked = question.format(name=name)
            reply, retrieved = ask(passage, asked)
            thought = personas.split_think(reply)
            answer = thought.answer or thought.reasoning
            graded = rag.grounding(answer, passage)
            copied = bool(graded is not None and graded >= 0.5)
            rows.append({"question": asked, "reply": reply, "expected": expected,
                         "retrieved": retrieved, "grounding": graded, "copied": copied})
            print(f"[sft] {'OK ' if copied else 'NO '}[{label}] {asked!r} -> {reply[:110]!r} "
                  f"(expected {expected!r}, grounding={graded})")
        asked = missing.format(name=name)
        reply, _ = ask(passage, asked)
        refused = "does not say" in reply.lower()
        rows.append({"question": asked, "reply": reply, "expected": "(refusal)",
                     "retrieved": 1, "grounding": None, "copied": refused})
        print(f"[sft] {'OK ' if refused else 'NO '}[{label}] {asked!r} -> {reply[:110]!r} "
              f"(expected the honest refusal)")
        series_rows.append({"series": label, "entity": name, "passage": passage, "rows": rows,
                            "copied": sum(1 for row in rows if row["copied"]),
                            "asked": len(rows)})
    if was_training:
        model.train()
    return {"series": series_rows,
            "copied": sum(entry["copied"] for entry in series_rows),
            "asked": sum(entry["asked"] for entry in series_rows)}


COT_CHECK_PROMPTS = (
    "Why does it rain?",
    "Sam has 12 marbles, gets 4 more, and gives 6 away. How many marbles does Sam have?",
    "Is 7 + 5 bigger than 45?",
)


@torch.no_grad()
def check_reasoning(model, tok, cfg, device, args) -> list[dict]:
    """Greedily answer a few *complex* questions and report the `<think>` status.

    Greedy on purpose: sampling a reasoning block and calling it a format check
    would be a coin toss. The persona hint is attached (that is what the browser
    sends for the reasoner preset) but the questions are the ones the corpus
    taught, so this measures the fine-tune, not the system prompt.
    """
    gen_mod = load("06_generate")
    label, system = personas.resolve("reasoner")
    specials = [IM_START, IM_END]
    end_id = tok.token_to_id[IM_END]
    was_training = model.training
    model.eval()
    report: list[dict] = []
    for prompt in COT_CHECK_PROMPTS[:max(0, args.cot_check)]:
        text = (f"{IM_START}system\n{system}{IM_END}\n"
                if system else "") + f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
        ids = tok.encode_with_specials(text, specials)
        ids = ids[-(cfg.block_size - 1):]
        params = gen_mod.SamplingParams(temperature=0.0, top_k=1, top_p=1.0,
                                        repeat_penalty=1.0, seed=0)
        out, _, _ = gen_mod.generate(model, ids, 160, params, use_cache=True,
                                     eos_id=end_id, device=device,
                                     block_size=cfg.block_size)
        reply = tok.decode(out[len(ids):]).strip()
        for marker in personas.MARKERS:          # control markers are not prose
            reply = reply.replace(marker, "")
        thought = personas.split_think(reply)
        report.append({"prompt": prompt, "reply": reply, "system": label,
                       "reasoning": thought.reasoning, "answer": thought.answer,
                       "closed": thought.closed and thought.has_reasoning,
                       "reasoning_chars": len(thought.reasoning)})
    if was_training:
        model.train()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Instruction-tune TinyStories as a small chatbot")
    ap.add_argument("--data", default="data/sft/chat_sft.jsonl")
    ap.add_argument("--model", default="out/model_best.pt")
    ap.add_argument("--config", default="out/config.json")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--chat-tokenizer", default="tokenizer/chat")
    ap.add_argument("--out", default="out/model_chat.pt")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--val-examples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--autocast", choices=["none", "bf16"], default="none")
    ap.add_argument("--cot-pins", type=int, default=96,
                    help="repeats per pinned chain-of-thought example (0 disables)")
    ap.add_argument("--cot-check", type=int, default=3,
                    help="questions to sample greedily after training to verify the "
                         "<think> format (0 disables)")
    ap.add_argument("--doc-check", type=int, default=3,
                    help="document questions to sample greedily after training, with "
                         "a held-out passage injected (0 disables)")
    ap.add_argument("--doc-pins", type=int, default=64,
                    help="repeats per pinned document-grounding example (0 disables)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.autocast == "bf16"

    tok = make_chat_tokenizer(Path(args.tokenizer), Path(args.chat_tokenizer))
    model_mod = load("03_model")
    old_cfg = model_mod.GPTConfig.load(args.config)
    cfg = replace(old_cfg, vocab_size=tok.vocab)
    model = model_mod.GPT(cfg).to(device)

    pretrained = torch.load(args.model, map_location=device, weights_only=True)
    current = model.state_dict()
    for name, value in pretrained.items():
        if name in {"wte.weight", "lm_head.weight"}:
            current[name][:old_cfg.vocab_size].copy_(value)
        else:
            current[name].copy_(value)
    model.load_state_dict(current)
    print(f"[sft] expanded vocabulary {old_cfg.vocab_size} -> {cfg.vocab_size}; "
          f"loaded {args.model}")

    raw = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line]
    reasoned = sum(1 for row in raw
                   if personas.THINK_OPEN in row.get("response", "")
                   or row.get("reasoning"))
    print(f"[sft] {len(raw):,} examples, {reasoned:,} with a {personas.THINK_OPEN} trace")
    singles = [encode_example(tok, r["prompt"], r["response"], cfg.block_size,
                              r.get("system") or None) for r in raw]
    order = list(range(len(singles)))
    random.Random(args.seed).shuffle(order)
    nval = min(args.val_examples, max(1, len(singles) // 20))
    val_rows = [singles[i] for i in order[:nval]]
    train_raw = [raw[i] for i in order[nval:]]
    train_rows = [singles[i] for i in order[nval:]]
    # Half the optimization pool contains a completed prior turn. This teaches
    # role-boundary recovery and makes retained REPL history useful rather than
    # causing the model to continue answering the oldest user. Pin the requested
    # verification exchange several times as a tiny regression curriculum.
    plain = [r for r in train_raw if not r.get("system")]
    for i in range(0, len(plain) - 1, 2):
        train_rows.append(encode_conditioned(tok, plain[i], plain[i + 1], cfg.block_size))
    hello = {"prompt": "Hello! Who are you?", "response": "Hello! I am a small, friendly assistant. I can answer simple questions and tell little stories."}
    frog = {"prompt": "Can you tell me a story about a frog?", "response": "Once there was a little green frog named Pip. Pip was afraid to leap across a stream. A kind dragonfly said, ‘Try one small jump.’ Pip jumped, landed on a lily pad, and smiled. Soon he crossed the whole stream, one brave hop at a time."}
    train_rows.extend(encode_conditioned(tok, hello, frog, cfg.block_size) for _ in range(256))
    # Chain-of-thought curriculum: the *format* is the lesson, so a few examples
    # are repeated far beyond their natural frequency. Without this a 7 M model
    # regresses to the pretrained habit of answering immediately.
    if args.cot_pins > 0:
        reasoned_rows = [r for r in train_raw if personas.THINK_OPEN in r["response"]]
        greetings = [r for r in train_raw if personas.THINK_OPEN not in r["response"]]
        for i, row in enumerate(reasoned_rows[:3]):
            train_rows.extend(encode_example(tok, row["prompt"], row["response"], cfg.block_size)
                              for _ in range(args.cot_pins))
        # ...and one *direct* example, pinned the same way, so "think" does not
        # become "always think": the trace is a response to the question, not a
        # prefix on every reply.
        if greetings:
            row = greetings[0]
            train_rows.extend(encode_example(tok, row["prompt"], row["response"], cfg.block_size)
                              for _ in range(args.cot_pins // 2))
        print(f"[sft] pinned chain-of-thought examples: "
              f"{sum(1 for r in train_raw if personas.THINK_OPEN in r['response']):,} reasoned rows "
              f"in the pool; {args.cot_pins} repeats each")
    # Document grounding curriculum: the same argument as the CoT pins. One
    # answer-from-the-passage example and one "the document does not say"
    # example, repeated, so the injected block is never wallpaper.
    if args.doc_pins > 0:
        # document rows only: `system` is also used for persona rows, and pinning
        # a persona greeting as "document grounding" would teach the wrong block
        grounded = [r for r in train_raw
                    if (r.get("system") or "").startswith(rag.DOC_HEADER)]
        answers = [r for r in grounded if "does not say" not in r["response"]]
        refusals = [r for r in grounded if "does not say" in r["response"]]
        # Two answering rows get the full weight and the refusal gets half of it:
        # the copy task is the hard one, and a refusal pinned as hard as an answer
        # is how the first version of these rows taught the model to answer
        # "The document does not say." to everything with a document loaded.
        #
        # Which two? The *longest* copies, and the first refusal the file holds.
        # A pinned row is seen ~64 times, so a short one ("It makes a soft
        # fluting song") would be memorized outright and would teach that a
        # document answer is a sentence fragment: sort by length so the pin is a
        # whole sentence whose value cannot be produced without reading it.
        answers = sorted(answers, key=lambda r: (-len(r["response"]), r["prompt"]))
        for row in answers[:2]:
            train_rows.extend(
                encode_example(tok, row["prompt"], row["response"], cfg.block_size,
                               row["system"]) for _ in range(args.doc_pins))
        for row in refusals[:1]:
            train_rows.extend(
                encode_example(tok, row["prompt"], row["response"], cfg.block_size,
                               row["system"]) for _ in range(args.doc_pins // 2))
        print(f"[sft] pinned document-grounding examples: {len(grounded):,} grounded rows "
              f"({len(answers):,} answering, {len(refusals):,} refusing) in the pool; "
              f"{args.doc_pins} repeats per answer, {args.doc_pins // 2} per refusal")
    rows = train_rows + val_rows
    train_idx = list(range(len(train_rows)))
    val_idx = list(range(len(train_rows), len(rows)))
    response_tokens = sum(sum(v != IGNORE for v in y) for _, y in rows)
    print(f"[sft] {len(train_idx):,} train / {len(val_idx):,} val examples; "
          f"{response_tokens:,} supervised assistant tokens; device={device}")

    opt = model.configure_optimizers(args.weight_decay, args.lr, (0.9, 0.95))
    rng = random.Random(args.seed + 1)
    best = float("inf")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_name("chat_log.jsonl")
    history = []
    pad_id = tok.token_to_id[IM_END]
    model.train()
    started = time.time()

    for step in range(args.iters):
        if step < args.warmup:
            lr = args.lr * (step + 1) / max(1, args.warmup)
        else:
            pct = (step - args.warmup) / max(1, args.iters - args.warmup)
            lr = args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * (1 + math.cos(math.pi * pct)) / 2)
        for group in opt.param_groups:
            group["lr"] = lr
        batch = rng.sample(train_idx, args.batch_size)
        x, y = collate(rows, batch, pad_id, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp and device.type in {"cpu", "cuda"}):
            loss = masked_loss(model, x, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 10 == 0 or step == args.iters - 1:
            print(f"[sft] step {step+1:3d}/{args.iters} loss={loss.item():.4f} "
                  f"lr={lr:.2e} gnorm={float(grad_norm):.2f}")
            history.append({"step": step + 1, "loss": loss.item(), "lr": lr,
                            "grad_norm": float(grad_norm)})

        if (step + 1) % args.eval_every == 0 or step == args.iters - 1:
            val = evaluate(model, rows, val_idx, args.batch_size, pad_id, device, amp)
            print(f"[sft] validation step {step+1}: loss={val:.4f} ppl={math.exp(val):.2f}")
            history.append({"step": step + 1, "val_loss": val, "val_ppl": math.exp(val)})
            if val < best:
                best = val
                torch.save(model.state_dict(), out)
                print(f"[sft] new best -> {out}")

    # ---- chain-of-thought check: does the fine-tune actually emit a trace? ----
    cot_report = check_reasoning(model, tok, cfg, device, args) if args.cot_check > 0 else None
    if cot_report:
        for row in cot_report:
            flag = "OK " if row["closed"] else "NO "
            print(f"[sft] {flag}{row['prompt']!r} -> {row['reply'][:120]!r}")
        ok = sum(1 for row in cot_report if row["closed"])
        print(f"[sft] reasoning check: {ok}/{len(cot_report)} replies opened *and* closed "
              f"a {personas.THINK_OPEN} block")

    doc_report = check_grounding(model, tok, cfg, device, args) if args.doc_check > 0 else None
    if doc_report:
        for entry in doc_report["series"]:
            print(f"[sft] document check [{entry['series']}]: "
                  f"{entry['copied']}/{entry['asked']} replies reused the injected passage")
        print(f"[sft] document check: {doc_report['copied']}/{doc_report['asked']} replies "
              f"reused the injected passage")

    config_path = out.with_name(out.stem + "_config.json")
    config_path.write_text(json.dumps(asdict(cfg), indent=2))
    log_path.write_text("\n".join(json.dumps(row) for row in history) + "\n")
    run_path = out.with_name("chat_run_config.json")
    run_path.write_text(json.dumps({**vars(args), "vocab_size": cfg.vocab_size,
                                    "best_val_loss": best, "cot_check": cot_report,
                                    "doc_check": doc_report,
                                    "seconds": time.time() - started}, indent=2))
    print(f"[sft] done; best validation loss={best:.4f}; wrote {out} and {config_path}")


if __name__ == "__main__":
    main()
