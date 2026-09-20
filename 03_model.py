"""
Stage 3 — The model: a decoder-only (GPT-style) transformer.

Annotated design decisions, each of which is a real fork in the road:

 pre-norm residual blocks     x = x + attn(norm(x)) instead of norm(x + attn(x)).
                              Massively more stable at depth, needs less warmup
                              (OpenAI's GPT-2 code uses post-norm; every modern
                              recipe uses pre-norm).
 RMSNorm                      LayerNorm without the mean-centring: same benefit,
                              fewer ops, no bias term to tune.
 RoPE                         Rotary position encoding applied to q/k. No
                              embedding table, position information is relative,
                              and it degrades gracefully past the training
                              window. GPT-2's learned absolute embeddings are
                              the simpler alternative and are strictly worse
                              for length extrapolation.
 SwiGLU                       SiLU(xW_gate) * (xW_up). Better loss/param than
                              ReLU-GEGLU MLPs at matched width; needs 3 matrices,
                              so hidden width shrinks to 8/3*d_ff to keep FLOPs equal.
 tied embeddings              lm_head.weight is the token embedding matrix. Saves
                              a big chunk of parameters (vocab x d_model) and
                              measurably helps at small scale.
 zero-init output projections attn out_proj and both MLP projections start at 0,
                              so every residual branch is the identity at step 0
                              -> the network is exactly its own skip stream at
                              initialisation and cannot blow up before it learns.
 1/sqrt(2*n_layer) residual   Scales the accumulating variance of the residual
                              stream so it stays ~O(1) at any depth.
 no bias in QKV, but ...      Biases on q/k/v are useless (they cancel in the
                              softmax for k; shift q uniformly). We DO keep a
                              learnable attention-logit bias ("verge") because
                              it acts as a per-head temperature and helps
                              sharpness early in training. It costs nothing at
                              inference.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 192
    dropout: float = 0.0          # 0.0 for large corpora; >0 only if overfitting
    rope_base: float = 10_000.0   # raise this (1e5..5e5) to extend context
    rms_norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model must divide by n_head"
        return self.d_model // self.n_head

    def save(self, path: str | Path) -> None:
        """Plain JSON, one canonical format. A checkpoint whose config cannot be
        parsed is a checkpoint you cannot read, so save() and load() must agree."""
        import json
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "GPTConfig":
        import json
        d = json.loads(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__}
        extra = set(d) - known
        if extra:                       # never silently drop a config key
            raise ValueError(f"unknown config keys {extra}; cannot reconstruct faithfully")
        return cls(**d)


# --------------------------------------------------------------------- layers
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x * rsqrt(mean(x^2) + eps) * weight  -- fp32 math for stability
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def build_rope_cache(dim: int, seq_len: int, base: float, device, dtype):
    """Returns (cos, sin) each [seq_len, dim] — dim is the head_dim."""
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)                       # [T, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)           # [T, dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, Dh]. Rotate each 2-D subspace by position-dependent angle."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.attn_bias = nn.Parameter(torch.zeros(cfg.n_head, 1, 1))  # learnable logit scale
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, mask, kv_cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        # [B, T, D] -> [B, H, T, Dh]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q = q * (1.0 / math.sqrt(self.head_dim))          # scale before attn

        if kv_cache is not None:      # incremental decoding: append to this layer's cache
            if kv_cache.new is not None:
                ck, cv = kv_cache.new
                k = torch.cat((ck, k), dim=2)
                v = torch.cat((cv, v), dim=2)
            kv_cache.new = (k, v)

        # [B,H,T,T]: masked softmax, then a weighted sum of values
        att = q @ k.transpose(-2, -1) + self.attn_bias + mask
        att = self.drop(att.softmax(dim=-1))
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # 8/3 factor keeps SwiGLU's FLOPs equal to a 4x ReLU MLP (3 matrices).
        hidden = int(2 * (4 * cfg.d_model) / 3)
        hidden = 8 * ((hidden + 7) // 8)                   # round to multiple of 8
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask, kv_cache=None):
        x = x + self.attn(self.ln1(x), cos, sin, mask, kv_cache)
        x = x + self.mlp(self.ln2(x))
        return x



class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight          # tied embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.apply(self._init_weights)
        # --- zero-init every residual-branch output projection (GPT-2 "zero-init")
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            nn.init.zeros_(block.mlp.down.weight)
        # --- residual-stream variance scaling (GPT-2 used sqrt(2*n_layer))
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            for p in (block.attn.proj.weight, block.mlp.down.weight):
                p.data.mul_(scale)
    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 recipe: N(0, 0.02) everywhere, zero biases, unit norm gains.
        std=0.02 is not arbitrary — with tied embeddings it makes E[logit] ~ 0
        and Var[logit] ~ (0.02*sqrt(d))^2 * 0.02*..., which puts the initial
        cross-entropy within a hair of ln(V), the loss of a uniform model."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------------ masking
    def get_mask(self, T: int, start: int, device, dtype):
        key = (T, start)
        if key not in self._mask_cache:
            # additive -inf upper triangle == "can only look backwards"
            m = torch.full((T, T), float("-inf"), device=device, dtype=torch.float32)
            m = torch.triu(m, diagonal=1 + start)
            self._mask_cache[key] = m.to(dtype)
        return self._mask_cache[key]

    # ---------------------------------------------------------------- forward
    def forward(self, idx, targets=None, kv_cache=None):
        """kv_cache: list of _Slot-like objects (one per layer) or None.
        Each slot holds this layer's (k, v) for tokens already processed."""
        B, T = idx.shape
        start = 0 if kv_cache is None else (0 if kv_cache[0].new is None else kv_cache[0].new[0].shape[2])
        x = self.drop(self.wte(idx))
        # Build RoPE for positions [0, start+T) but SLICE to the new tokens.
        # Cached k/v were already rotated when they were created, so re-rotating
        # them (or feeding them a full-length table) is the classic KV-cache bug.
        cos, sin = build_rope_cache(self.cfg.head_dim, start + T, self.cfg.rope_base,
                                    x.device, torch.float32)
        cos, sin = cos[start:start + T], sin[start:start + T]
        cos, sin = cos[None, None], sin[None, None]
        mask = self.get_mask(T, start, x.device, x.dtype)
        for i, block in enumerate(self.blocks):
            kc = None if kv_cache is None else kv_cache[i]
            x = block(x, cos, sin, mask, kc)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            # no ignore_index: blocks are packed from real documents, never padded,
            # so every position is a valid training target. If you switch to sequence
            # packing with padding, set an ignore index and mask it here.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # ------------------------------------------------------------- param groups
    def configure_optimizers(self, weight_decay, lr, betas):
        """AdamW with no decay on 1-D params (norms) and biases. Also reports
        how much of the parameter budget actually receives decay."""
        decay, nodecay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else nodecay).append(p)
        params = [{"params": decay, "weight_decay": weight_decay},
                  {"params": nodecay, "weight_decay": 0.0}]
        n_decay = sum(p.numel() for p in decay)
        total = n_decay + sum(p.numel() for p in nodecay)
        print(f"[opt] {n_decay:,}/{total:,} params get weight decay "
              f"({100*n_decay/total:.1f}%)")
        return torch.optim.AdamW(params, lr=lr, betas=betas)

    # ----------------------------------------------------------------- helpers
    @torch.no_grad()
    def estimate_flops(self, block_size: int) -> int:
        """Forward-pass FLOPs for one sequence, multiply-adds counted as 2.
        = 2 * (non-embedding params) * T, i.e. the matmul cost through every layer,
        plus the quadratic attention term. 6ND for train (fwd + bwd) is 3x this.
        Excludes the embedding lookup, which is a gather, not a matmul."""
        nparams = self.num_params(exclude_embeddings=True)
        fwd = 2 * nparams * block_size
        attn = 0
        for _ in self.blocks:
            attn += 2 * 2 * block_size * block_size * self.cfg.d_model
        return fwd + attn

    def num_params(self, exclude_embeddings: bool = False) -> int:
        emb = self.wte.weight.numel()
        tot = sum(p.numel() for p in self.parameters())
        return tot - emb if exclude_embeddings else tot

    @torch.no_grad()
    def get_lr(self, it: int, max_iters: int, lr: float, min_lr_frac: float, warmup: int) -> float:
        """Linear warmup -> cosine decay to min_lr_frac * lr. Warmup prevents
        early AdamW second-moment blowups; the cosine tail extracts the last few
        percent of loss for free. pct saturates at 1, so past max_iters the rate
        simply stays at its floor."""
        if it < warmup:
            return lr * (it + 1) / warmup
        pct = min(1.0, (it - warmup) / max(1, max_iters - warmup))
        f = (1 + math.cos(math.pi * pct)) / 2
        return lr * (min_lr_frac + (1 - min_lr_frac) * f)

    def save(self, path: str | Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        import ast
        (path / "config.txt").write_text(repr(asdict(self.cfg)))

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "GPT":
        cfg = GPTConfig.load(Path(path) / "config.txt")
        model = cls(cfg).to(device)
        sd = torch.load(Path(path) / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(sd)
        return model.eval()


# ------------------------------------------------------------------ unit tests
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=512, block_size=32, n_layer=3, n_head=4, d_model=64)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))

    logits, loss = model(x, y)
    assert logits.shape == (2, 32, 512)
    print(f"[model] params={model.num_params():,}  init loss={loss.item():.3f} "
          f"vs ln(V)={math.log(cfg.vocab_size):.3f}")
    # invariant: an untrained model should start ~ at the entropy of the uniform
    # distribution over the vocab. Being far above it means the head is mis-scaled.
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.35, "init loss is wrong"

    # invariant: causality — changing token t must not change logits[:t].
    x2 = x.clone(); x2[:, 20:] = torch.randint(0, cfg.vocab_size, (2, 12))
    l1, _ = model(x); l2, _ = model(x2)
    assert torch.allclose(l1[:, :20], l2[:, :20], atol=1e-5), "attention is not causal!"
    print("[model] causality test passed (future tokens cannot leak into the past)")

    # invariant: KV-cached incremental decoding must equal a full forward pass
    class _Slot:
        __slots__ = ("new",)
        def __init__(self): self.new = None

    with torch.no_grad():
        ref, _ = model(x)                       # full-sequence logits
        ref_last = ref[:, -1]
        # (a) feed the whole prefix, capturing k/v for each layer.
        #     NOTE: forward() *mutates* the slots, so snapshot before reuse.
        slots = [_Slot() for _ in range(cfg.n_layer)]
        _, _ = model(x[:, :-1], kv_cache=slots)
        snap = [sl.new for sl in slots]
        # (b) feed ONLY the final token against a fresh copy of that cache
        for sl, c in zip(slots, snap):
            sl.new = (c[0].clone(), c[1].clone()) if c is not None else None
        got, _ = model(x[:, -1:], kv_cache=slots)
        err = (got[:, -1] - ref_last).abs().max().item()
    print(f"[model] KV-cache vs full forward: max abs diff = {err:.2e}")
    assert err < 2e-4, f"KV cache mismatch: {err}"
    print("[model] all invariants hold")
