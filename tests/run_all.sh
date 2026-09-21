#!/usr/bin/env bash
# Runs the model invariants, tokenizer/sampler properties, and inference regressions.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== 00 corpus: <|endoftext|>/glued-story splitting, cleaner rules, dedupe =="; python3 tests/test_tinystories.py | tail -3
echo "== 01 tokenizer: fast BPE training vs naive reference, round-trips =="; python3 tests/test_tokenizer_fast_vs_naive.py | tail -2
echo "== 01 tokenizer: fast encoder vs reference encoder ==";               python3 tests/test_tokenizer_encoding.py
echo "== 03 model: init loss, causality, RoPE relative, SDPA, KV cache ==";  python3 03_model.py
echo "== 03 architecture: RMSNorm/SwiGLU/SDPA/config invariants ==";         python3 tests/test_architecture.py 2>&1 | tail -3
echo "== 06 generation: sampler properties, cached==uncached ==";           python3 tests/test_generate.py | tail -7
echo "== CPU inference: streaming, sampling, REPL, CLI ==";                 python3 tests/test_inference.py 2>&1 | tail -3
echo "== SFT: chat tokens, response masking, dataset/checkpoint ==";          python3 tests/test_sft.py 2>&1 | tail -3
echo "== web UI: stream decoder, chat engine, HTTP server end-to-end ==";     python3 tests/test_web_ui.py 2>&1 | tail -3
echo "ALL TESTS PASSED"
