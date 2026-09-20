#!/usr/bin/env bash
# Runs every test in the project: three standalone stage tests + two property suites.
set -e
cd "$(dirname "$0")/.."
echo "== 01 tokenizer: fast BPE vs naive reference, round-trips =="; python3 tests/test_tokenizer_fast_vs_naive.py | tail -2
echo "== 03 model: init loss, causality, KV-cache equivalence ==";      python3 03_model.py
echo "== 06 generation: sampler properties, cached==uncached ==";       python3 tests/test_generate.py | tail -7
echo "ALL TESTS PASSED"
