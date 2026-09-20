"""Property tests for the sampler + cached generation equivalence."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import sys, math, torch
from importlib import import_module
mm = import_module("03_model"); gg = import_module("06_generate")
GPT, GPTConfig = mm.GPT, mm.GPTConfig
SamplingParams, generate, _Slot = gg.SamplingParams, gg.generate, gg._Slot

torch.manual_seed(0)
cfg = GPTConfig(vocab_size=256, block_size=48, n_layer=2, n_head=4, d_model=64)
model = GPT(cfg).eval()
prompt = [int(i) for i in torch.randint(1, 200, (7,))]

# 1. cached == uncached, token for token, under greedy decoding
params = SamplingParams(temperature=0.0, top_k=0, top_p=1.0, repeat_penalty=1.0)
a, _, _ = generate(model, prompt, 12, params, use_cache=True)
b, _, _ = generate(model, prompt, 12, params, use_cache=False)
assert a == b, f"cached != uncached\n{a}\n{b}"
print(f"[gen] cached == uncached over 12 greedy steps: OK  {a[-5:]}")

# 2. determinism: same seed -> same text
p2 = SamplingParams(temperature=0.9, seed=42)
c, _, _ = generate(model, prompt, 16, p2, use_cache=True)
d, _, _ = generate(model, prompt, 16, p2, use_cache=True)
assert c == d, "seeded sampling is not deterministic"
e, _, _ = generate(model, prompt, 16, SamplingParams(temperature=0.9, seed=7), use_cache=True)
print(f"[gen] seeded sampling reproducible: OK (different seed differs: {c != e})")

# 3. top-k really caps support at k
V = cfg.vocab_size
logits = torch.randn(1, 1, V) * 3
probs = gg.logits_to_probs(logits, SamplingParams(temperature=1.0, top_k=10, top_p=1.0, repeat_penalty=1.0))
nz = int((probs[0] > 0).sum().item())
assert (probs[0] >= 0).all(), "negative probability"
assert nz == 10, f"top-k=10 left {nz} live tokens"
assert abs(probs[0].sum().item() - 1) < 1e-5, "probs do not renormalise to 1"
mass = probs[0].sum().item()
assert abs(mass - 1.0) < 1e-5, f"top-k probs sum to {mass}"
print(f"[gen] top-k=10 -> {nz} live tokens, mass renormalised to {mass:.5f}: OK")

# 4. temperature limits
# T=0.01, not 1e-9: dividing by 1e-9 overflows the logits to +-inf, after which
# two near-tied tokens both survive at 0.5 each (inf-inf is undefined, not 0).
# The numerically honest statement is "lower T -> lower entropy".
lo = gg.logits_to_probs(logits, SamplingParams(temperature=0.01, top_k=0, top_p=1.0, repeat_penalty=1.0))
hi = gg.logits_to_probs(logits, SamplingParams(temperature=50.0, top_k=0, top_p=1.0, repeat_penalty=1.0))
# clamp: 0*log(0) -> 0, the convention entropy needs (without it, log(0)=-inf)
H = lambda p: -(p.clamp(min=1e-12) * p.clamp(min=1e-12).log()).sum()
assert H(lo[0]) < H(hi[0]), "temperature did not flatten the distribution"
assert 0.0 < H(lo[0]) < 0.05, f"low-T entropy {H(lo[0])} should be near 0, never negative"
assert H(hi[0]) > H(lo[0]), "temperature did not flatten the distribution"
print(f"[gen] entropy  T->0: {H(lo[0]):.4f} bits  vs  T=50: {H(hi[0]):.4f} bits: OK")

# 5. top-p keeps >= p of the mass and is adaptive
big = torch.full((1, 1, V), -20.0); big[0, 0, 5] = 20.0; big[0, 0, 9] = 19.0   # very peaked
flat = torch.randn(1, 1, V)                                                      # very flat
for name, lg in [("peaked", big), ("flat", flat)]:
    pr = gg.logits_to_probs(lg, SamplingParams(temperature=1.0, top_k=0, top_p=0.9, repeat_penalty=1.0))
    n = int((pr[0] > 1e-7).sum().item()); m = pr[0].sum().item()
    print(f"[gen] top-p=0.9 on {name}: {n} tokens kept, mass {m:.3f}")
    assert 0.999 < m < 1.001, f"top-p mass {m}"
    assert n >= 1, "top-p must always keep at least the top token"
peaked_n = int((gg.logits_to_probs(big, SamplingParams(top_k=0, top_p=0.9, temperature=1.0, repeat_penalty=1.0))[0] > 1e-7).sum())
flat_n   = int((gg.logits_to_probs(flat, SamplingParams(top_k=0, top_p=0.9, temperature=1.0, repeat_penalty=1.0))[0] > 1e-7).sum())
assert peaked_n < flat_n, "top-p is not adaptive"
print(f"[gen] adaptivity confirmed: peaked keeps {peaked_n} vs flat keeps {flat_n}")
print("[gen] all sampler tests passed")
