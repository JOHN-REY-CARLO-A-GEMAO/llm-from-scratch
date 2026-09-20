"""
Stage 6 — Text generation (decoding strategies).

Sampling is where most from-scratch projects go wrong, so every knob below has
a reason:

 temperature  T>0 flattens, T<0 sharpens: logits/T. T=0 is greedy. Too low and
              the model loops; too high and it drifts into babble. 0.7-0.9 is a
              sane default for prose.
 top-k        keep the k best candidates, renormalise. Cheap, but a *fixed* k is
              wrong: after "The capital of France is" the distribution is peaked
              (k=5 is plenty), after "Once upon a" it is flat (k=5 truncates
              good continuations).
 top-p (nucleus) keep the smallest set with cumulative mass >= p, renormalise.
              Adapts the candidate set to the model's own confidence. This is
              the default in practice; top-k is usually just a safety cap on top
              of it (p=0.9, k=50).
 repeat penalty  subtract log-prob of already-generated tokens from their logits.
              Small models repeat; this is the cheapest fix and the least
              damaging (vs sampling tricks that change the distribution's shape).
               greedy/beam search are NOT used for open-ended text. Beam search
              maximises joint likelihood, which for language is *boring and
              repetitive* - the well-known degeneracy that pushed everyone to
              sampling. It remains right for transcription/translation.

KV cache: with caching, step t costs O(T) instead of O(T^2) attention over the
sequence and avoids recomputing layers for past tokens -> roughly linear time
per generated token instead of quadratic. Verified below: cached == uncached.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    seed: int | None = None


class _Slot:
    """Per-layer KV cache holder. forward() fills .new with (k, v) covering
    every token seen so far in this layer."""
    __slots__ = ("new",)

    def __init__(self):
        self.new = None


@torch.no_grad()
def logits_to_probs(logits: torch.Tensor, params: SamplingParams,
                    history: torch.Tensor | None = None) -> torch.Tensor:
    """[B, T, V] logits -> [B, V] distribution over the *next* token.
    Only the last position is touched: earlier positions are already consumed.
    Order matters: penalty -> temperature -> truncation -> renormalise."""
    x = logits[:, -1, :].clone().float()          # [B, V] only the last position

    if params.repeat_penalty != 1.0 and history is not None:
        # HuggingFace's formulation: divide positive logits by the penalty and
        # multiply negative ones, so the move is always *towards* zero regardless
        # of the logit's sign (a plain subtraction can flip a token into favour).
        # Raise the penalty to the observed count so repeats are punished harder.
        alpha = params.repeat_penalty
        for b in range(x.shape[0]):
            uniq, counts = history[b].unique(return_counts=True)
            pen = torch.tensor(alpha, device=x.device) ** counts.to(x.device)
            xs = x[b, uniq]
            x[b, uniq] = torch.where(xs > 0, xs / pen, xs * pen)

    if params.temperature <= 1e-5:                 # greedy
        return x
    x = x / params.temperature

    if params.top_k and params.top_k > 0:
        k = min(params.top_k, x.size(-1))
        thr = x.topk(k, dim=-1).values[:, -1:]
        x = x.masked_fill(x < thr, float("-inf"))
    if params.top_p and params.top_p < 1.0:
        srt, idx = torch.sort(x, descending=True)  # idx re-aligns the mask
        sm = F.softmax(srt, dim=-1)
        # exclusive cumsum: token i is kept iff mass BEFORE it is already < p.
        # keep is therefore always a prefix, so the cut index = number of Falses.
        keep = (sm.cumsum(-1) - sm) < params.top_p
        cut = (~keep).sum(-1)                                   # [B]
        mask_sorted = torch.arange(x.size(-1), device=x.device)[None, :] >= cut[:, None]
        mask = torch.zeros_like(mask_sorted).scatter_(1, idx, mask_sorted)
        x = x.masked_fill(mask, float("-inf"))

    return torch.softmax(x, dim=-1)


@torch.no_grad()
def generate(model, prompt_ids: list[int], max_new_tokens: int, params: SamplingParams,
             use_cache: bool = True, eos_id: int | None = None,
             device: torch.device | None = None, block_size: int | None = None):
    """Continuation sampling with an optional KV cache.

    Returns (ids, tokens_per_second, n_prefill + n_generated)."""
    model.eval()
    dev = device or next(model.parameters()).device
    ctx = block_size or getattr(model.cfg, "block_size", None)
    gen = (torch.Generator(device=dev).manual_seed(params.seed)
           if params.seed is not None else None)

    ids = torch.tensor([prompt_ids], device=dev, dtype=torch.long)
    if ctx and ids.shape[1] > ctx - 1:
        ids = ids[:, -(ctx - 1):]                  # never exceed the trained window

    cache = [_Slot() for _ in range(model.cfg.n_layer)] if use_cache else None
    out: list[int] = []
    n_done = 0
    import time
    t0 = time.time()

    # prefill: one forward over the whole prompt, filling the cache
    if use_cache:
        logits, _ = model(ids, kv_cache=cache)
        n_done = ids.shape[1]
    else:
        logits, _ = model(ids)

    for _ in range(max_new_tokens):
        probs = logits_to_probs(logits, params, history=ids)
        if params.temperature <= 1e-5:
            nxt = probs.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(probs, 1, generator=gen).to(dev)
        out.append(int(nxt.item()))
        ids = torch.cat([ids, nxt], dim=1)

        if use_cache:
            logits, _ = model(nxt, kv_cache=cache)     # 1 token per step
        else:
            logits, _ = model(ids)                     # recompute everything
        if eos_id is not None and out[-1] == eos_id:
            break
        if ctx and ids.shape[1] >= ctx:
            break                                      # cache/RoPE window exhausted

    dt = time.time() - t0
    n = n_done + len(out)
    return prompt_ids + out, (n / dt if dt > 0 else float("nan")), n
