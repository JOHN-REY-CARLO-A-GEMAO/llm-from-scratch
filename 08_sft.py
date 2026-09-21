"""Supervised fine-tuning of the TinyStories checkpoint for chat.

Only response tokens (including ``<|im_end|>``) contribute to cross entropy.
User text, role markers, and right padding use label -100. Two chat markers are
appended to the existing tokenizer, preserving every pretrained token id.
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


def encode_example(tok, prompt: str, response: str, block_size: int):
    specials = [IM_START, IM_END]
    prefix = f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
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
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.autocast == "bf16"

    tok = make_chat_tokenizer(Path(args.tokenizer), Path(args.chat_tokenizer))
    tok_mod = load("01_tokenizer")
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
    singles = [encode_example(tok, r["prompt"], r["response"], cfg.block_size) for r in raw]
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
    for i in range(0, len(train_raw) - 1, 2):
        train_rows.append(encode_conditioned(tok, train_raw[i], train_raw[i + 1], cfg.block_size))
    hello = {"prompt": "Hello! Who are you?", "response": "Hello! I am a small, friendly assistant. I can answer simple questions and tell little stories."}
    frog = {"prompt": "Can you tell me a story about a frog?", "response": "Once there was a little green frog named Pip. Pip was afraid to leap across a stream. A kind dragonfly said, ‘Try one small jump.’ Pip jumped, landed on a lily pad, and smiled. Soon he crossed the whole stream, one brave hop at a time."}
    train_rows.extend(encode_conditioned(tok, hello, frog, cfg.block_size) for _ in range(256))
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

    config_path = out.with_name(out.stem + "_config.json")
    config_path.write_text(json.dumps(asdict(cfg), indent=2))
    log_path.write_text("\n".join(json.dumps(row) for row in history) + "\n")
    run_path = out.with_name("chat_run_config.json")
    run_path.write_text(json.dumps({**vars(args), "vocab_size": cfg.vocab_size,
                                    "best_val_loss": best,
                                    "seconds": time.time() - started}, indent=2))
    print(f"[sft] done; best validation loss={best:.4f}; wrote {out} and {config_path}")


if __name__ == "__main__":
    main()
