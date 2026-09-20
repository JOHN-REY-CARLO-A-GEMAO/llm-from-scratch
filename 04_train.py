"""
Stage 4 — Training loop.

Everything here is justified by a specific failure mode it prevents:

 * mixed precision (bf16 autocast)   ~2x faster; fp32 master weights kept, so
   small updates are not lost to rounding. bf16 needs no GradScaler; fp16 (GPU) does.
 * AdamW, not Adam                    decoupled weight decay (Loshchilov & Hutter).
                                      With Adam, L2 penalty and adaptive scaling
                                      fight each other.
 * no decay on 1-D params/biases      a norm's gain is a scale, not a direction;
                                      decaying it shrinks the signal it controls.
 * betas=(0.9, 0.95)                  beta2=0.999 assumes a long, stationary
                                      run. On short runs the second-moment
                                      estimate is stale and training goes noisy.
 * grad clip 1.0 (global norm)        loss spikes from an unlucky batch would
                                      otherwise permanently move the weights.
 * warmup then cosine                 warmup covers the first steps while Adam's
                                      moment estimates are still garbage.
 * sample LR *inside* the step        not a fixed schedule table: this is what
                                      lets a run be resumed and stay on-curve.
 * fixed eval batch index             so val loss is comparable across steps
                                      instead of bouncing with sampling noise.
 * save optimizer state + best val    resume-ability, and "best" by val loss,
                                      never by train loss.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _stage import load as _load
model_mod = _load("03_model")
GPT, GPTConfig = model_mod.GPT, model_mod.GPTConfig


def get_batch(data: np.ndarray, size: int, device, gen: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    ix = gen.integers(0, data.shape[0] - 1, size=(size,))
    # np.asarray copies out of the memmap; int64 because nn.Embedding requires
    # a long tensor (our storage is uint16 to halve I/O - convert once, here).
    blk = torch.from_numpy(np.asarray(data[ix], dtype=np.int64))
    return blk[:, :-1].to(device, non_blocking=True), blk[:, 1:].to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, data, eval_batch, device, iters: int, eval_size: int) -> dict:
    """Val loss on a *fixed* set of block indices -> comparable across steps.

    `iters` is derived from the data, never hard-coded blindly: an empty eval
    set silently produced np.mean([]) == nan, which poisoned every downstream
    "is this the best checkpoint?" comparison. Fail loudly instead.
    """
    if len(eval_batch) < eval_size:
        raise RuntimeError(f"eval set has {len(eval_batch)} blocks but the eval batch "
                           f"size is {eval_size}: at least one eval batch is required. "
                           f"Lower --batch-size or add validation data.")
    if len(eval_batch) % eval_size:
        raise RuntimeError(f"eval batch size {eval_size} does not divide the {len(eval_batch)} "
                           f"eval blocks: the final short batch would be averaged as if it were "
                           f"full. Pass --eval-iters {len(eval_batch) // eval_size} or a batch size "
                           f"that divides evenly.")
    model.eval()
    # TOKEN-weighted, not mean-of-batch-means. Batches are the same size here, but the
    # moment one is short (or block widths differ) an unweighted mean silently biases the
    # number -- and this eval decides which checkpoint we keep, so bias is not free.
    tot_loss, ntok = 0.0, 0
    for k in range(iters):
        ix = eval_batch[k * eval_size:(k + 1) * eval_size]
        blk = torch.from_numpy(np.asarray(data[ix], dtype=np.int64))
        x, y = blk[:, :-1].to(device), blk[:, 1:].to(device)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=device.type == "cpu"):
            _, loss = model(x, y)
        tot_loss += loss.item() * y.numel()
        ntok += y.numel()
    model.train()
    val_loss = tot_loss / max(ntok, 1)
    return {"val_loss": val_loss, "val_ppl": math.exp(val_loss), "val_tokens": ntok}


def write_cfg(out: Path, vocab: int, args) -> None:
    """Config travels with the weights: a checkpoint is useless if you cannot
    reconstruct the exact architecture it was trained with."""
    (out / "config.txt").write_text(json.dumps({
        "vocab_size": vocab, "block_size": args.block_size, "n_layer": args.n_layer,
        "n_head": args.n_head, "d_model": args.d_model, "dropout": args.dropout}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/shuffled")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--resume", action="store_true",
                    help="restore model+optimizer+iteration from opt.pt (continues the schedule)")
    ap.add_argument("--model", default=None,
                    help="warm-start weights from a model_best/last .pt; fresh optimizer and "
                         "schedule. Deliberately separate from --resume: loading weights is not "
                         "resuming a run, and conflating them lost a checkpoint once already.")
    args = ap.parse_args()

    # PyTorch defaults to 1 thread in some containers; pin it explicitly so the
    # log line reports what the run actually uses.
    torch.set_num_threads(os.cpu_count() or 1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] device={dev}  threads={torch.get_num_threads()}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # Guard: a "best" checkpoint from a previous run must not be clobbered by a run
    # that never loaded it. This exact mistake (resuming into a dir whose opt.pt was
    # gone) replaced a trained model with a freshly initialised one, because best_val
    # started at inf and the first eval therefore always "improved".
    if not args.resume and list(out.glob("model_best*.pt")):
        raise RuntimeError(
            f"{out} already holds checkpoint(s) {sorted(q.name for q in out.glob('model_best*.pt'))} "
            f"and --resume was not given, so this run would train from scratch and overwrite them. "
            f"Use a fresh --out-dir, or pass --resume.")
    meta = json.loads((Path(args.data_dir) / "dataset_info.json").read_text())
    vocab = meta["vocab"]

    cfg = GPTConfig(vocab_size=vocab, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, d_model=args.d_model, dropout=args.dropout)
    model = GPT(cfg).to(dev)
    if args.compile:
        model = torch.compile(model)
    nparams = model.num_params()
    nemb = model.wte.weight.numel()
    print(f"[run] params {nparams:,} ({100*nemb/nparams:.0f}% in the embedding)  "
          f"flops/token {model.estimate_flops(args.block_size)/args.block_size:,.0f}")

    if args.model:                      # weights only: no optimizer state, no iteration offset
        model.load_state_dict(torch.load(args.model, map_location=dev, weights_only=True))
        print(f"[run] warm start from {args.model} (fresh optimizer + schedule)")

    opt = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2))

    train = np.load(Path(args.data_dir) / "train.npy", mmap_mode="r")
    val = np.load(Path(args.data_dir) / "val.npy", mmap_mode="r")
    print(f"[run] train blocks {train.shape[0]:,}  val blocks {val.shape[0]:,}")

    # fixed eval indices: same windows at every eval, so the curve is trustworthy
    # Prefer evaluating the WHOLE val set in batches that divide it exactly.
    # With 60 val blocks and batch 32 the old code silently used 32 + a short
    # 28-block batch. Pick the largest batch size <= requested that divides n.
    n = min(args.eval_iters * args.batch_size, val.shape[0])
    bs = args.batch_size
    while bs > 1 and n % bs:
        bs -= 1
    eval_batch = np.random.default_rng(0).permutation(val.shape[0])[:n]
    args.eval_iters = n // bs
    args.batch_size_eval = bs
    if bs != args.batch_size:
        print(f"[run] eval batch size {args.batch_size} -> {bs} (divides {n} val blocks exactly)")

    gen = np.random.default_rng(args.seed)
    start_iter, best, last_log = 0, float("inf"), None
    if args.resume and not (out / "opt.pt").exists():
        raise RuntimeError(f"--resume given but {out/'opt.pt'} is missing: there is no optimizer "
                           f"state to restore, so resuming would silently restart training from "
                           f"iteration 0. To fine-tune from weights only, use --model.")
    if args.resume and (out / "opt.pt").exists():
        ck = torch.load(out / "opt.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"] + 1
        best = ck.get("best_val", float("inf"))
        print(f"[run] resumed at iter {start_iter}, best val {best:.4f}")

    t_log = time.time()          # throughput window: reset only when we log
    hist_path = out / "log.jsonl"
    hist = [json.loads(l) for l in hist_path.read_text().splitlines()] if hist_path.exists() else []
    running, t0 = 0.0, time.time()   # t0: step stopwatch (used for timing the very first log)
    model.train()
    for it in range(start_iter, args.iters):
        lr = model.get_lr(it, args.iters, args.lr, args.min_lr_frac, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        # gradient accumulation: larger effective batch at the cost of memory only
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train, args.batch_size, dev, gen)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=dev.type == "cpu"):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            running += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step_loss = running / args.grad_accum
        running = 0.0
        if it % args.log_every == 0 or it == args.iters - 1:
            # divide by the steps ACTUALLY elapsed, not log_every: otherwise the
            # first line (1 step in) reports a wildly optimistic throughput.
            steps = max(it - (last_log if last_log is not None else start_iter), 1)
            dt = time.time() - t_log
            tps = steps * args.batch_size * args.block_size / max(dt, 1e-9)
            last_log, t_log = it, time.time()
            print(f"[train] iter {it:5d}/{args.iters}  loss {step_loss:.4f}  "
                  f"lr {lr:.2e}  gnorm {grad_norm:.2f}  {tps:,.0f} tok/s")
            hist.append({"iter": it, "loss": step_loss, "lr": lr, "gnorm": float(grad_norm)})
            t0 = time.time()

        if ((it + 1) % args.eval_every == 0) or it == args.iters - 1:
            ev = evaluate(model, val, eval_batch, dev, args.eval_iters, args.batch_size_eval)
            print(f"[  eval] iter {it+1:5d}  val_loss {ev['val_loss']:.4f}  "
                  f"val_ppl {ev['val_ppl']:.2f}")
            hist.append({"iter": it + 1, **ev})
            # Save the current model every eval, best or not. A run whose only artefact
            # is "best" has nothing to show for every step after the best one -- and on a
            # short run "best" is often just an early fluke of the fixed eval window.
            torch.save(model.state_dict(), out / f"model_{it+1}.pt")
            write_cfg(out, vocab, args)
            if ev["val_loss"] < best:
                best = ev["val_loss"]
                torch.save(model.state_dict(), out / "model_best.pt")
                write_cfg(out, vocab, args)
                print(f"[  eval] new best -> saved to {out}/model_best.pt")

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": args.iters - 1,
                "best_val": best}, out / "opt.pt")
    torch.save(model.state_dict(), out / "model_last.pt")
    write_cfg(out, vocab, args)
    hist_path.write_text("\n".join(json.dumps(h) for h in hist))
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"[run] done. best val_loss {best:.4f} (ppl {math.exp(best):.2f})")


if __name__ == "__main__":
    main()
