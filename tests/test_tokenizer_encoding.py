"""The fast BPE encoder must be *identical* to the textbook reference encoder.

The fast path (lowest-rank merge, per-word cache) exists only for speed; if it
ever disagrees with the reference, the training corpus silently stops matching
the vocabulary it was trained with. This test is the contract between them.
"""
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _stage import load                                            # noqa: E402

m = load("01_tokenizer")
tok = m.ByteLevelBPE.load(ROOT / "tokenizer")
corpus_path = ROOT / "data" / "tinystories" / "corpus.txt"
corpus = corpus_path.read_text(encoding="utf-8") if corpus_path.exists() else \
    (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")

random.seed(11)
stories = [s for s in corpus.split("<|doc_end|>") if s.strip()]
samples = random.sample(stories, min(40, len(stories)))
samples += ["", " ", "\n\n", "Once upon a time", "‘’“”—…😀 Ångström 42%", "a" * 300]

bad = 0
for s in samples:
    fast, ref = tok.encode(s), tok.encode_reference(s)
    if fast != ref:
        bad += 1
        print("MISMATCH on:", repr(s[:120]))
        for i, (a, b) in enumerate(zip(fast, ref)):
            if a != b:
                print(f"   first divergence at token {i}: fast={a} ref={b}")
                break
assert not bad, f"{bad} of {len(samples)} samples disagree"
print(f"[tok] fast encoder == reference encoder on {len(samples)} samples "
      f"({len(tok.merges):,} merges)")

# round-trip through both paths, including the special token handling
for s in samples[:10]:
    assert tok.decode(tok.encode(s)) == s
assert tok.decode(tok.encode_with_specials("hi<|doc_end|>there", [m.__dict__.get("DOC_END", "<|doc_end|>")])) \
    == "hi<|doc_end|>there"
print("[tok] round-trip holds through the fast path and encode_with_specials")

# compression sanity: byte-level BPE on English prose must beat 2 chars/token
ids = tok.encode(corpus[:200_000])
ratio = 200_000 / len(ids)
print(f"[tok] compression on the corpus: {ratio:.2f} chars/token")
assert ratio > 2.5, f"compression collapsed to {ratio:.2f} chars/token"
print("[tok] encoder tests passed")
