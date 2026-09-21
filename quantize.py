"""INT8 weight quantization from scratch.

This is classic symmetric, per-output-channel (per-row) weight quantization:

    scale[c]   = max(|W[c, :]|) / 127.0
    W_int8[c]  = round(W[c] / scale[c])   clamped to [-127, 127]
    W_deq      = W_int8 * scale[c]

Per-channel scales beat per-tensor because different output channels in a linear
layer often live at wildly different magnitudes; forcing them to share one scale
throws away several bits of mantissa on the quiet channels.

The quantized model keeps the original RMSNorm weights and embeddings in fp32
(they are a tiny fraction of parameters and quantising them costs quality for
negligible size savings) and replaces every nn.Linear's weight with an int8
tensor + a per-channel fp32 scale. The forward pass dequantises on the fly, so
from the outside a QuantizedLlama looks and behaves like a regular Llama —
samplers, generate.py and web_ui.py need no changes beyond loading it.

Usage:
    python quantize.py --in out/model_15m_chat.pt --config out/model_15m_chat_config.json \\
                       --out out/model_15m_chat_int8.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from _stage import load  # noqa: E402


# ------------------------------------------------------------------ quant core
def quantize_tensor_per_channel(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantisation, one scale per output channel (per row).

    W: shape [out_features, in_features] for a linear layer.
    Returns (W_int8 char, scales float32 [out_features, 1]).
    """
    assert W.dim() == 2, f"expected a 2-D weight matrix, got {W.shape}"
    W = W.detach().float()
    # per-row max absolute value; eps guards against a zero-initialised row
    amax = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scales = amax / 127.0
    q = torch.round(W / scales).clamp(-127, 127).to(torch.int8)
    return q, scales.squeeze(-1)


def quantize_tensor_per_tensor(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 with a single scalar scale (fallback / reference)."""
    W = W.detach().float()
    amax = W.abs().max().clamp(min=1e-8)
    scale = amax / 127.0
    q = torch.round(W / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize(W_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """W_int8 * scales -> fp32. Handles per-channel (scales 1-D) and
    per-tensor (scales scalar or 0-D)."""
    if scales.dim() == 0 or scales.numel() == 1:
        return W_q.float() * scales.float()
    # per-channel: multiply each row by its scale
    return W_q.float() * scales.float().unsqueeze(-1)


# ------------------------------------------------------------ quantized linear
class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear that stores INT8 weights.

    Bias is kept in fp32 (our model doesn't use biases, but support them for
    drop-in generality). The forward pass dequantises weights to fp32 right
    before the matmul — that is the classic "INT8 weights, fp32 activations"
    dynamic-weight-quantisation scheme used by llama.cpp, bitsandbytes 8-bit
    and PyTorch's own quantize_dynamic. It gives ~4x smaller checkpoint size
    and better memory bandwidth at near-identical outputs.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_q", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, per_channel: bool = True) -> "QuantizedLinear":
        q = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        if per_channel:
            wq, scl = quantize_tensor_per_channel(linear.weight.data)
        else:
            wq, scl = quantize_tensor_per_tensor(linear.weight.data)
        q.weight_q.copy_(wq)
        q.weight_scale.copy_(scl if scl.dim() > 0 else scl.reshape(1).expand(linear.out_features))
        if linear.bias is not None:
            q.bias.copy_(linear.bias.data.float())
        # quantization error reporting (for diagnostics; not used at runtime)
        with torch.no_grad():
            deq = dequantize(q.weight_q, q.weight_scale)
            err = (deq - linear.weight.data.float()).abs().mean().item()
            q.quant_err = err
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantise weights to fp32 for the matmul. On a CPU this is purely
        # a bandwidth win: the int8 tensor is 4x smaller to read, and the
        # multiply-and-accumulate stays in fp32 (int8 matmul kernels are
        # possible but a separate optimisation; this is the classic recipe
        # and keeps the code fully portable).
        w = dequantize(self.weight_q, self.weight_scale).to(x.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, dtype=int8")


# ------------------------------------------------------------ quantized model
def quantize_model(model: nn.Module, per_channel: bool = True) -> tuple[nn.Module, dict]:
    """Replace every nn.Linear in the model with a QuantizedLinear.

    Embeddings and RMSNorm gains are left in fp32 on purpose: together they
    are <<1% of parameters in our architecture (tied 4096*d_model embed is
    the only large one, but quantising the embedding lookup table introduces
    easily-audible quality loss for very little size win because the lookup
    already happens in fp32 on gather).
    """
    stats = {"n_layers": 0, "n_params": 0, "mean_quant_err": 0.0}
    errs = []
    replaced = 0
    # Walk all submodules and swap nn.Linear -> QuantizedLinear in place.
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                ql = QuantizedLinear.from_linear(child, per_channel=per_channel)
                setattr(module, child_name, ql)
                replaced += 1
                errs.append(ql.quant_err)
                stats["n_params"] += child.weight.numel()
                if child.bias is not None:
                    stats["n_params"] += child.bias.numel()
    stats["n_layers"] = replaced
    stats["mean_quant_err"] = float(sum(errs) / max(len(errs), 1))
    stats["max_quant_err"] = float(max(errs)) if errs else 0.0
    return model, stats


def is_quantized(model: nn.Module) -> bool:
    """True iff any submodule is a QuantizedLinear."""
    return any(isinstance(m, QuantizedLinear) for m in model.modules())


# -------------------------------------------------------------- checkpoint IO
QUANT_MAGIC = "quantized_v1"


def save_quantized(model: nn.Module, cfg, path: str | Path, per_channel: bool = True) -> dict:
    """Save a quantized checkpoint.

    The file is a single dict:
        { "magic": "quantized_v1",
          "config": {...cfg...},
          "per_channel": bool,
          "state_dict": { name: tensor (int8 where quantised, else fp32) } }

    size on disk is dominated by int8 weights, so a ~15 M-parameter model
    becomes ~15 MB instead of ~60 MB in fp32.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    payload = {
        "magic": QUANT_MAGIC,
        "per_channel": per_channel,
        "config": cfg if isinstance(cfg, dict) else cfg.__dict__ if hasattr(cfg, "__dict__") else None,
        "state_dict": sd,
    }
    # write as a contiguous pickle-free? torch.save uses zip+pickle; fine for
    # CPU inference. Measure size.
    torch.save(payload, path)
    mb = path.stat().st_size / 1e6
    return {"path": str(path), "mb": mb}


def load_quantized(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, object, dict]:
    """Load a quantized checkpoint, rebuild the architecture, and return
    (model, cfg, metadata). The returned model is in eval() mode with grad
    disabled, ready for generate().
    """
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("magic") != QUANT_MAGIC:
        raise ValueError(f"{path} does not look like a quantized checkpoint (missing magic={QUANT_MAGIC!r})")
    cfg_dict = payload["config"]
    mm = load("03_model")
    cfg = mm.GPTConfig(**cfg_dict)
    model = mm.GPT(cfg).to(device)
    # replace linears BEFORE loading the state dict (so the shapes match)
    quantize_model(model, per_channel=payload.get("per_channel", True))
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        # tolerate nothing
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta = {"magic": payload["magic"], "per_channel": payload["per_channel"]}
    return model, cfg, meta


# ------------------------------------------------------------------- CLI demo
def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize a trained fp32/bf16 checkpoint to INT8 (from scratch).")
    ap.add_argument("--in", dest="ckpt_in", required=True, help="fp32 checkpoint .pt (state_dict)")
    ap.add_argument("--config", required=True, help="config.json / config.txt for the checkpoint")
    ap.add_argument("--out", required=True, help="output path for the INT8 checkpoint")
    ap.add_argument("--per-tensor", action="store_true", help="use a single scale per layer (worse, for ablation)")
    ap.add_argument("--ppl-check", action="store_true", help="re-evaluate a few val batches and report ppl delta")
    args = ap.parse_args()

    mm = load("03_model")
    cfg = mm.GPTConfig.load(args.config)
    model = mm.GPT(cfg)
    sd = torch.load(args.ckpt_in, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    nparams = model.num_params()
    fp_mb = Path(args.ckpt_in).stat().st_size / 1e6
    print(f"[quantize] loaded fp32 model: {nparams:,} params, {fp_mb:.2f} MB on disk")

    t0 = time.time()
    model, stats = quantize_model(model, per_channel=not args.per_tensor)
    dt = time.time() - t0
    print(f"[quantize] replaced {stats['n_layers']} linear layers ({stats['n_params']:,} params) "
          f"in {dt*1000:.1f} ms")
    print(f"[quantize] mean weight reconstruction error: {stats['mean_quant_err']:.3e}  "
          f"(max across layers {stats['max_quant_err']:.3e})")

    info = save_quantized(model, cfg, args.out, per_channel=not args.per_tensor)
    int_mb = info["mb"]
    print(f"[quantize] wrote {info['path']}  ({int_mb:.2f} MB, "
          f"{fp_mb/int_mb:.2f}x smaller)")

    # Quick sanity: a forward pass on random token ids should produce finite logits
    with torch.inference_mode():
        ids = torch.randint(0, cfg.vocab_size, (1, min(32, cfg.block_size)))
        logits, _ = model(ids)
        assert torch.isfinite(logits).all(), "quantized forward produced NaN/Inf!"
    print("[quantize] sanity forward pass OK (finite logits)")


if __name__ == "__main__":
    main()
