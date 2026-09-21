#!/usr/bin/env bash
# Runs the model invariants, tokenizer/sampler properties, and inference regressions.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== 01 tokenizer: fast BPE vs naive reference, round-trips =="; python3 tests/test_tokenizer_fast_vs_naive.py | tail -2
echo "== 03 model: init loss, causality, KV-cache equivalence ==";      python3 03_model.py
echo "== 06 generation: sampler properties, cached==uncached ==";       python3 tests/test_generate.py | tail -7
echo "== CPU inference: streaming, sampling, REPL, CLI =="; python3 tests/test_inference.py
echo "ALL TESTS PASSED"
