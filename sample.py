"""CLI for generation: continue a prompt from the command line.

    python sample.py --prompt "To my right there was a small" --n 120 --temp 0.8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stage import load

model_mod = load("03_model")
gen_mod = load("06_generate")
tok_mod = load("01_tokenizer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/model_best.pt")
    ap.add_argument("--config", default="out/config.txt")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--prompt", default="It was a dark and stormy night")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repeat-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--compare", action="store_true", help="print cached vs uncached timing")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = model_mod.GPTConfig.load(args.config)
    model = model_mod.GPT(cfg).to(dev)
    sd = torch.load(args.ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    tok = tok_mod.ByteLevelBPE.load(args.tokenizer)

    prompt_ids = tok.encode(args.prompt)
    params = gen_mod.SamplingParams(temperature=args.temp, top_k=args.top_k,
                                    top_p=args.top_p, repeat_penalty=args.repeat_penalty,
                                    seed=args.seed)
    ids, tps, n = gen_mod.generate(model, prompt_ids, args.n, params,
                                   use_cache=not args.no_cache, device=dev,
                                   block_size=cfg.block_size)
    print(f"\n=== prompt ({len(prompt_ids)} tokens) ===\n{args.prompt}")
    print(f"\n=== continuation ({n - len(prompt_ids)} new tokens, {tps:,.0f} tok/s) ===")
    print(tok.decode(ids))

    if args.compare:
        import time
        for label, use_cache in [("KV-cached", True), ("no cache  ", False)]:
            t0 = time.time()
            gen_mod.generate(model, prompt_ids, args.n, params, use_cache=use_cache,
                             device=dev, block_size=cfg.block_size)
            print(f"[speed] {label}: {(time.time()-t0):.2f}s for {args.n} tokens")


if __name__ == "__main__":
    main()
