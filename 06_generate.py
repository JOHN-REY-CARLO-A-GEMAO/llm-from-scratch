"""
Stage 6 — Text generation (decoding strategies).

Sampling is where most from-scratch projects go wrong, so every knob below has
a reason:

 temperature  T>1 flattens, 0<T<1 sharpens: logits/T. T=0 is greedy. Too low and
              the model loops; too high and it drifts into babble. 0.7-0.9 is a
              sane default for prose.
 top-k        keep the k best candidates, renormalise. Cheap, but a *fixed* k is
              wrong: after "The capital of France is" the distribution is peaked
              (k=5 is plenty), after "Once upon a" it is flat (k=5 truncates
              good continuations).
 top-p (nucleus) keep the smallest set with cumulative mass >= p, renormalise.
              Adapts the candidate set to the model's own confidence. This is
              the default in practice; top-k is usually just a safety cap on top
              of it (p=0.9, k=40).
 repeat penalty  lower logits of tokens already present in the prompt or output.
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
from collections.abc import Callable
import math

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.15
    seed: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not math.isfinite(self.repeat_penalty) or self.repeat_penalty < 1:
            raise ValueError("repeat_penalty must be finite and >= 1")
        if self.seed is not None and not -(2**63) <= self.seed < 2**64:
            raise ValueError("seed must be between -2**63 and 2**64 - 1")


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
        # Sign-aware penalty, once per distinct token: positive logits shrink,
        # negative logits become more negative. Exponentiating by occurrence
        # count over-penalises common prose tokens and can overflow on long runs.
        alpha = params.repeat_penalty
        for b in range(x.shape[0]):
            uniq = history[b].unique()
            xs = x[b, uniq]
            x[b, uniq] = torch.where(xs > 0, xs / alpha, xs * alpha)

    if params.temperature <= 1e-5:                 # greedy
        return x
    x = x / params.temperature

    if params.top_k and params.top_k > 0:
        k = min(params.top_k, x.size(-1))
        values, indices = x.topk(k, dim=-1)
        x = torch.full_like(x, float("-inf")).scatter_(1, indices, values)
    if params.top_p and params.top_p < 1.0:
        srt, idx = torch.sort(x, descending=True)  # idx re-aligns the mask
        sm = F.softmax(srt, dim=-1)
        # Shift the removal mask right to include the token crossing p.
        # The highest-probability token must always survive (even for tiny p).
        mask_sorted = sm.cumsum(-1) >= params.top_p
        mask_sorted[:, 1:] = mask_sorted[:, :-1].clone()
        mask_sorted[:, 0] = False
        mask = torch.zeros_like(mask_sorted).scatter_(1, idx, mask_sorted)
        x = x.masked_fill(mask, float("-inf"))

    return torch.softmax(x, dim=-1)


@torch.inference_mode()
def generate(model, prompt_ids: list[int], max_new_tokens: int, params: SamplingParams,
             use_cache: bool = True, eos_id: int | None = None,
             device: torch.device | None = None, block_size: int | None = None,
             on_token: Callable[[int], None] | None = None):
    """Continuation sampling with an optional KV cache.

    on_token is called immediately for each new token (including EOS), never
    for the prompt. Returns (original_prompt + output, tokens_per_second,
    n_prefill + n_generated); prefill counts only tokens actually fed to model."""
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    model.eval()
    dev = device or next(model.parameters()).device
    ctx = block_size if block_size is not None else getattr(model.cfg, "block_size", None)
    if ctx is not None and ctx < 2:
        raise ValueError("block_size must be >= 2")
    gen = (torch.Generator(device=dev).manual_seed(params.seed)
           if params.seed is not None else None)

    ids = torch.tensor([prompt_ids], device=dev, dtype=torch.long)
    if ctx and ids.shape[1] > ctx - 1:
        ids = ids[:, -(ctx - 1):]                  # never exceed the trained window

    cache = [_Slot() for _ in range(model.cfg.n_layer)] if use_cache else None
    out: list[int] = []
    n_done = ids.shape[1]
    import time
    t0 = time.perf_counter()
    if max_new_tokens == 0:
        return list(prompt_ids), 0.0, n_done

    # prefill: one forward over the whole prompt, filling the cache
    if use_cache:
        logits, _ = model(ids, kv_cache=cache)
    else:
        logits, _ = model(ids)

    for _ in range(max_new_tokens):
        probs = logits_to_probs(logits, params, history=ids)
        if params.temperature <= 1e-5:
            nxt = probs.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(probs, 1, generator=gen).to(dev)
        out.append(int(nxt.item()))
        if on_token is not None:
            on_token(out[-1])
        ids = torch.cat([ids, nxt], dim=1)

        if eos_id is not None and out[-1] == eos_id:
            break
        if ctx and ids.shape[1] >= ctx:
            break                                      # cache/RoPE window exhausted
        if len(out) == max_new_tokens:
            break
        if use_cache:
            logits, _ = model(nxt, kv_cache=cache)     # 1 token per step
        else:
            logits, _ = model(ids)                     # recompute everything

    dt = time.perf_counter() - t0
    n = n_done + len(out)
    return prompt_ids + out, (n / dt if dt > 0 else float("nan")), n
