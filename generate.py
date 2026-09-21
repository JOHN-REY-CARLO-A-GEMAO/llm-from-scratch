"""
Inference only: load a trained checkpoint and write text on the CPU.

This is the file your laptop runs. Training never happens here — the checkpoint
is committed to the repository, so all you need is PyTorch (CPU build) and this
file. Deliberately **no numpy, no matplotlib, no regex requirement** (the
tokenizer falls back to stdlib `re`), and no GPU code path: what you measure
here is the model, not the harness.

    python generate.py --prompt "It was a dark and stormy night" --n 150 --temp 0.8

Design notes, each of which is a real inference-time bug it avoids:

 * `torch.inference_mode()` + `eval()` + `requires_grad_(False)`: no autograd
   graph is built, dropout is off, and nothing can accidentally update weights.
 * KV-cached decoding (`06_generate.generate(use_cache=True)`): token t costs
   one attention step over the cache instead of recomputing the whole prefix,
   so 150 tokens are ~3x cheaper and grow linearly, not quadratically.
   tests/test_generate.py pins cached == uncached.
 * `weights_only=True` on load: a checkpoint is data. Unpickling arbitrary
   objects from a file you downloaded is exactly how "just run this model"
   becomes remote code execution. The checkpoint here is a plain state_dict.
 * config travels next to the weights (config.json / config.txt). A checkpoint
   whose architecture you have to guess is a checkpoint you cannot load; the
   mix-up shows up as a shape error, so it is checked *before* generating.
 * threads pinned to the physical core count: PyTorch defaults to 1 thread in
   some environments, which quietly makes CPU generation several times slower.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:            # the stage files are importable by name
    sys.path.insert(0, str(HERE))


def die(msg: str) -> None:
    print("\n[gen] ERROR: " + msg + "\n", file=sys.stderr)
    raise SystemExit(1)


def import_torch():
    import warnings
    try:
        with warnings.catch_warnings():
            # A torch install without numpy (the default from the CPU wheel index)
            # prints a scary "Failed to initialize NumPy" UserWarning at import.
            # Nothing on the inference path needs numpy, so it is noise, not news.
            warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
            import torch
    except ImportError:
        die("PyTorch is not installed for this Python interpreter.\n"
            "      Install the CPU build once (no CUDA, ~200 MB):\n"
            "        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
    return torch


def resolve(p: Path) -> Path:
    """Repo-relative paths must work no matter where Python was started from —
    double-clicking run.bat, a shortcut with a different "start in", or
    `python C:\\path\\to\\generate.py`. Only the *file's own* directory is a safe
    fallback, never the caller's cwd."""
    return p if p.is_absolute() or p.exists() else HERE / p


def find_config(ckpt: Path, explicit: str | None) -> Path:
    """The architecture is stored beside the weights; accept JSON or the
    trainer's plain-text variant, and say exactly what to pull if neither is
    there (a missing config looks like a corrupt checkpoint otherwise)."""
    if explicit:
        p = resolve(Path(explicit))
        if not p.exists():
            die(f"--config {p} does not exist.")
        return p
    for name in ("config.json", "config.txt"):
        p = ckpt.parent / name
        if p.exists():
            return p
    die(f"no config.json / config.txt next to {ckpt}.\n"
        f"      The checkpoint and its config are a pair — fetch both, e.g.:  git pull")


def render(tok, ids: list[int]) -> tuple[str, int]:
    """Decode ids for display. `<|doc_end|>` is the document separator the model
    was trained on, not prose: show it as a paragraph break and report how many
    times it fired, rather than leaking a literal marker into the text."""
    special = {tok.token_to_id[s]: s for s in tok.special_tokens}
    if not special:
        return tok.decode(ids), 0
    chunks, run, n_special = [], [], 0
    for i in ids:
        if i in special:
            chunks.append(tok.decode(run))
            chunks.append("\n\n")
            run = []
            n_special += 1
        else:
            run.append(i)
    chunks.append(tok.decode(run))
    return "".join(chunks), n_special


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate text from a trained checkpoint on the CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", default="out/model_best.pt", help="checkpoint (.pt) to load")
    ap.add_argument("--config", default=None, help="default: config.json/config.txt next to --ckpt")
    ap.add_argument("--tokenizer", default="tokenizer", help="directory holding bpe.pkl")
    ap.add_argument("--prompt", default="It was a dark and stormy night",
                    help="text to continue; '' starts from the model's document token")
    ap.add_argument("-n", "--n", "--tokens", dest="n", type=int, default=150,
                    help="number of new tokens to sample")
    ap.add_argument("--temp", "--temperature", dest="temp", type=float, default=0.8,
                    help="0 = greedy; 0.7-0.9 reads as prose for this model size")
    ap.add_argument("--top-k", type=int, default=50, help="0 disables the top-k cap")
    ap.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling; 1.0 disables")
    ap.add_argument("--repeat-penalty", type=float, default=1.05,
                    help="1.0 disables; small models loop without it")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: a random seed, which is printed so the run can be repeated")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    ap.add_argument("--quiet", action="store_true", help="print the generated text only")
    args = ap.parse_args()

    torch = import_torch()
    from _stage import load as _load          # noqa: E402  (needs torch already importable)
    model_mod, gen_mod = _load("03_model"), _load("06_generate")
    tok_mod = _load("01_tokenizer")

    args.threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)             # belt and braces with inference_mode()
    dev = torch.device("cpu")                 # this file is CPU-only on purpose

    ckpt = resolve(Path(args.ckpt))
    if not ckpt.exists():
        die(f"checkpoint {ckpt} not found.\n"
            f"      It is committed to the repo (~8 MB) — run `git pull`, then retry.\n"
            f"      Or point at another one:  python generate.py --ckpt out/model_best.pt")
    cfg_path = find_config(ckpt, args.config)
    tok_dir = resolve(Path(args.tokenizer))
    if not (tok_dir / "bpe.pkl").exists():
        die(f"tokenizer {tok_dir}/bpe.pkl not found — run `git pull`, it is committed too.")

    cfg = model_mod.GPTConfig.load(cfg_path)   # strict: unknown keys are an error, not a guess
    model = model_mod.GPT(cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        die(f"{ckpt} does not match {cfg_path}.\n      {e}\n"
            f"      They are a pair: pull both from the same commit.")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = tok_mod.ByteLevelBPE.load(tok_dir)

    prompt_ids = tok.encode(args.prompt) if args.prompt else []
    if not prompt_ids:                        # empty prompt: start a new document
        if not tok.special_tokens:
            die("empty prompt, and this tokenizer has no special start token.")
        prompt_ids = [tok.token_to_id[tok.special_tokens[0]]]

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    params = gen_mod.SamplingParams(temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
                                    repeat_penalty=args.repeat_penalty, seed=seed)

    if not args.quiet:
        print(f"[gen] torch {torch.__version__}  device cpu  threads {args.threads}")
        print(f"[gen] checkpoint {ckpt}  ({model.num_params():,} params, "
              f"{ckpt.stat().st_size / 1e6:.1f} MB)")
        print(f"[gen] config {cfg_path}  ({cfg.n_layer} layers, {cfg.n_head} heads, "
              f"d_model {cfg.d_model}, context {cfg.block_size}, vocab {cfg.vocab_size})")
        print(f"[gen] tokenizer {tok_dir}  (vocab {tok.vocab:,})")
        if getattr(tok_mod, "_regex", None) is None:
            print("[gen] note: `regex` is not installed, so the stdlib word-splitting pattern is "
                  "used (pip install regex for byte-identical prompt tokenisation).")

    t0 = time.perf_counter()
    ids, tok_per_s, n_total = gen_mod.generate(
        model, prompt_ids, args.n, params, use_cache=True, device=dev, block_size=cfg.block_size)
    elapsed = time.perf_counter() - t0
    text, n_doc_end = render(tok, ids)
    new_tokens = n_total - len(prompt_ids)

    if not args.quiet:
        print(f"\n=== prompt ({len(prompt_ids)} tokens) ===\n{args.prompt}")
        print(f"\n=== continuation ({new_tokens} new tokens, {tok_per_s:,.0f} tok/s, "
              f"{elapsed:.2f} s) ===")
    print(text)
    if not args.quiet:
        if new_tokens < args.n:
            why = (f"hit the {cfg.block_size}-token context window" if not n_doc_end
                   else "that is all this checkpoint can be pushed to")
            print(f"\n[gen] stopped after {new_tokens} of {args.n} tokens: {why}. "
                  f"Shorter --prompt or smaller --n fits more.")
        if n_doc_end:
            print(f"[gen] the model emitted {n_doc_end} document separator token(s) "
                  f"(<|doc_end|>), shown as a blank line.")
        print(f"[gen] seed {seed}  (re-run with --seed {seed} to reproduce this text)")


if __name__ == "__main__":
    if not sys.stdout.isatty():               # redirected output on Windows may be cp1252
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass
    main()
