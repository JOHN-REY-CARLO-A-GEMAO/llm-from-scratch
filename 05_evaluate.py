"""
Stage 5 — Evaluation.

Three numbers, because perplexity alone can mislead:

 1. val loss / perplexity   the thing you optimise; comparable only between runs
                           that share a tokenizer. Cross-entropy of the *next
                           token* over held-out blocks.
 2. top-1 accuracy          fraction of positions where the argmax token is the
                           true next token. Lower-bounded, human-legible.
 3. bits per BYTE           tokenizer-independent. BPE splits tokens unevenly, so
                           ppl-per-token flatters a tokenizer that produces many
                           short tokens. bpb = CE * (tokens/bytes) / ln 2 lets you
                           compare a 4k-vocab run to a 50k-vocab run honestly.
                           (Karpathy's nanoGPT / "Letter" GPT reports this.)

Plus: a qualitative generation sample, because a model can have a mediocre loss
and still produce fluent text (or the reverse).

Note on what this does NOT measure: no LLM is finished after perplexity. For a
real system you would add held-out NLL under distribution shift, a task benchmark
(HellaSwag/ARC for a base model of this size), and safety/refusal evals. Those all
need a bigger model than is buildable in a notebook; the harness below is the same
shape though - swap the dataset and the metric function.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _stage import load as _load
model_mod = _load("03_model")
GPT, GPTConfig = model_mod.GPT, model_mod.GPTConfig


@torch.no_grad()
def metrics(model, data: np.ndarray, batch_size: int, dev, max_blocks: int | None) -> dict:
    """Next-token cross-entropy, teacher-forced, over held-out blocks."""
    model.eval()
    tot_loss, tot_tok, correct, tot_logp = 0.0, 0, 0, 0.0
    for s in range(0, data.shape[0], batch_size):
        if max_blocks and s >= max_blocks:
            break
        blk = torch.from_numpy(np.asarray(data[s:s + batch_size], dtype=np.int64))
        x, y = blk[:, :-1].to(dev), blk[:, 1:].to(dev)
        # Identical autocast to the training loop: computing the "final" val loss in
        # fp32 while training reported bf16 made the two numbers disagree by ~1%,
        # which reads like a corrupted checkpoint. One dtype for both.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=dev.type == "cpu"):
            logits, _ = model(x)
        logits = logits.float()      # softmax/CE in fp32 on top of bf16 logits
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        tot_loss += nll.sum().item()
        tot_tok += nll.numel()
        correct += (logits.argmax(-1) == y).sum().item()
    val_loss = tot_loss / max(tot_tok, 1)
    return {"val_loss": val_loss, "val_ppl": math.exp(val_loss),
            "top1_acc": correct / max(tot_tok, 1), "n_tokens": tot_tok}


def baselines(train: np.ndarray, val: np.ndarray, vocab: int) -> dict:
    """Two trivial predictors. A trained LM that cannot beat these is broken."""
    tr = train.reshape(-1).astype(np.int64)
    va = val.reshape(-1).astype(np.int64)
    # 1. unigram: p(x) from training frequencies, no context at all
    cnt = np.bincount(tr, minlength=vocab).astype(np.float64)
    p_uni = (cnt + 1.0) / (cnt.sum() + vocab)          # +1 (Laplace) so no inf
    uni_loss = -np.log(p_uni[va]).mean()
    # 2. bigram Markov chain: p(x_t | x_{t-1}), add-alpha smoothed. This is the
    #    real floor - it captures "what words follow 'Holmes'" without any net.
    prev, nxt = tr[:-1], tr[1:]
    joint = np.bincount(prev * vocab + nxt, minlength=vocab * vocab).reshape(vocab, vocab).astype(np.float64)
    cond = (joint + 0.1) / (joint.sum(1, keepdims=True) + 0.1 * vocab)
    vprev, vnxt = va[:-1], va[1:]
    big_loss = -np.log(cond[vprev, vnxt]).mean()
    return {"unigram_loss": float(uni_loss), "unigram_ppl": float(math.exp(uni_loss)),
            "bigram_loss": float(big_loss), "bigram_ppl": float(math.exp(min(big_loss, 50)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/model_best.pt")
    ap.add_argument("--data-dir", default="data/shuffled")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-blocks", type=int, default=512)
    ap.add_argument("--plot", default="out/curves.png")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GPTConfig.load(Path(args.ckpt).parent / "config.txt")
    model = GPT(cfg).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    model.eval()

    tok = _load("01_tokenizer").ByteLevelBPE.load(args.tokenizer)
    data = np.load(Path(args.data_dir) / f"{args.split}.npy", mmap_mode="r")
    print(f"[eval] {args.split} blocks={data.shape[0]:,} vocab={cfg.vocab_size}")

    m = metrics(model, data, args.batch_size, dev, args.max_blocks)

    # tokens-per-byte on the corpus itself, to get a tokenizer-free metric
    corpus = Path("data/corpus.txt").read_text(encoding="utf-8")[:200_000]
    tok_per_byte = len(tok.encode(corpus)) / len(corpus.encode("utf-8"))
    bpb = m["val_loss"] * tok_per_byte / math.log(2)
    print(f"[eval] val loss {m['val_loss']:.4f} | ppl {m['val_ppl']:.2f} | "
          f"top-1 {m['top1_acc']*100:.2f}% | bits/byte {bpb:.3f} "
          f"({tok_per_byte:.3f} tok/byte)")

    # the number only means something next to baselines
    train_arr = np.load(Path(args.data_dir) / "train.npy", mmap_mode="r")
    bl = baselines(np.asarray(train_arr), np.asarray(data), cfg.vocab_size)
    print(f"[base] uniform          ppl = {cfg.vocab_size:,}")
    print(f"[base] unigram          ppl = {bl['unigram_ppl']:.2f}")
    print(f"[base] bigram Markov    ppl = {bl['bigram_ppl']:.2f}")
    print(f"[base] this model       ppl = {m['val_ppl']:.2f}   "
          f"({100*(1-m['val_loss']/bl['bigram_loss']):.1f}% better than the bigram floor)")

    out = {**m, "bpb": bpb, "tok_per_byte": tok_per_byte, "baselines": bl}
    Path(args.ckpt).parent.joinpath("eval.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[eval] wrote {Path(args.ckpt).parent/'eval.json'}")

    # ------------------------------------------------------------- loss curve
    log = Path(args.ckpt).parent / "log.jsonl"
    if log.exists():
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        tr = [(r["iter"], r["loss"]) for r in rows if "loss" in r]
        va = [(r["iter"], r["val_loss"]) for r in rows if "val_loss" in r]
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
            ax[0].plot(*zip(*tr), lw=1.2, label="train")
            if va:
                ax[0].plot(*zip(*va), "o-", ms=3, lw=1.2, label="val")
            ax[0].set_xscale("log"); ax[0].set_xlabel("iter"); ax[0].set_ylabel("cross-entropy")
            ax[0].set_title("next-token loss"); ax[0].legend(); ax[0].grid(alpha=.3)
            if va:
                ax[1].plot([x for x, _ in va], [math.exp(y) for _, y in va], "s-", ms=3)
                ax[1].set_yscale("log"); ax[1].set_xlabel("iter"); ax[1].set_ylabel("perplexity")
                ax[1].set_title("val perplexity"); ax[1].grid(alpha=.3)
            Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(); fig.savefig(args.plot, dpi=110)
            print(f"[eval] loss curve -> {args.plot}")
        except ImportError:
            # ASCII fallback so the guide works without matplotlib
            if tr:
                ys = [y for _, y in tr]
                w = 48
                print("[eval] loss curve (ascii):")
                lo, hi = min(ys), max(ys)
                for i in range(0, len(ys), max(1, len(ys) // 12)):
                    n = int((hi - ys[i]) / max(hi - lo, 1e-6) * w)
                    print(f"  {i:>6} {ys[i]:.3f} |" + "#" * n)


if __name__ == "__main__":
    main()
