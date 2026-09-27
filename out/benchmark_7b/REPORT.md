# 7B architecture / training benchmark attempt

Date: 2026-09-27 (Arena sandbox)

## Architecture

Preset: `GPTConfig.preset("7b")`; `d_model=4096`, `n_layer=32`, `n_head=32`,
`head_dim=128`, `d_ff=11008`, tied embeddings, BPE vocabulary 4096, context 128,
gradient checkpointing enabled.

Parameter calculation (no linear biases):

- tied token embedding/output head: `4096 * 4096 = 16,777,216`
- per block attention: `4 * 4096^2 = 67,108,864`
- per block SwiGLU: `3 * 4096 * 11008 = 135,266,304`
- per-block RMSNorm gains: `2 * 4096 = 8,192`; final RMSNorm: `4096`
- total: `16,777,216 + 32 * (67,108,864 + 135,266,304 + 8,192) + 4,096`
  = **6,493,048,832 parameters**.

This is below 6.7B because this repository ties input/output embeddings and uses
its 4096-token BPE vocabulary. The familiar ~6.7B count assumes an untied head
and a much larger vocabulary. Count is architectural, not a model allocation.

## Benchmark attempt

Attempted: `python 04_train.py --arch 7b --iters 10 --batch-size 1 --grad-accum 16 --block-size 128 ...`

**No training steps ran.** The command terminated at startup because NumPy is not
installed (`ModuleNotFoundError: No module named 'numpy'`). A separate runtime
probe also found PyTorch unavailable (`ModuleNotFoundError: No module named
'torch'`). Consequently optimizer/memory behavior, peak training RAM, latency,
tokens/second, checkpoint saving, and quantization could not be measured. No
training stability claim can honestly be made.

Available host memory at probe time: `/proc/meminfo` reported MemTotal **4,034,452
kB (3.85 GiB)** and MemAvailable **3,763,828 kB (3.59 GiB)**. cgroup
`memory.max` reported `max`; these are host figures and not peak process usage.
A full FP32 parameter tensor alone would require about 25.97 GB decimal (24.2
GiB), before gradients, optimizer state, activations, or checkpointing, so a
real training run is not feasible in this environment even if dependencies
were installed.

`bash tests/run_all.sh` was attempted: the corpus and tokenizer sections passed,
then the model section stopped at import because PyTorch is unavailable. Therefore
the new architecture test and remaining model-dependent tests did not execute.
Quantization was not attempted because there is no valid 7B checkpoint. Existing quantization and web
UI load architecture dimensions from serialized `GPTConfig`; the explicit
`d_ff` config field therefore propagates without size-specific UI/quantizer
branches.
