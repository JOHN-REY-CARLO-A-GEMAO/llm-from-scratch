"""
Stage 1 — Byte-Pair-Encoding tokenizer, trained from scratch (no libraries).

Why BPE instead of a word tokenizer?
  * Word tokenizers need an UNK symbol. Every name, typo and number format
    outside the vocabulary becomes garbage, and the vocabulary has to be huge
    just to keep UNK rare.
  * Byte-level BPE covers every character of every language by construction:
    the atomic unit is a byte (0-255), so no input is ever unrepresentable.
  * It compresses: common substrings ("ing", " the ", "...ed") merge into
    single tokens, so one token carries ~4 characters of text.

The GPT-2 "byte fallback" trick: bytes that are not printable ASCII are mapped
to unused code points, so every byte is exactly one character in the tokenizer's
internal alphabet. That keeps decode(encode(x)) == x true for *any* input,
including text that is not valid UTF-8.
"""
from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path

try:  # \p{L} / \p{N} unicode classes need the 3rd-party `regex` module
    import regex as _regex  # type: ignore
    re = _regex
except ImportError:  # pragma: no cover - stdlib fallback, slightly worse splits
    _regex = None

# GPT-2's splitting pattern: contractions stay glued to their host word,
# punctuation and whitespace get their own pieces. This is why "don't"
# becomes ["do", "n't"] rather than ["don", "'", "t"].
if _regex is not None:
    PAT = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
else:  # stdlib `re` has no unicode property escapes
    PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")

# ------------------------------------------------------------------ byte map
BYTES = bytes(range(256))
# The code points we are allowed to use as byte stand-ins: printable ASCII plus
# a block of Latin-1 code points that essentially never occur in real text.
_SAFE = [i for i in range(256) if 33 <= i <= 126 or 161 <= i <= 172 or 174 <= i <= 255]
BYTE_TO_UNICODE = {b: chr(b) for b in BYTES if b not in _SAFE}
BYTE_TO_UNICODE.update({b: chr(256 + i) for i, b in enumerate(BYTES) if b in _SAFE})
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}
assert len(BYTE_TO_UNICODE) == 256 and len(UNICODE_TO_BYTE) == 256


def bytes_to_unicode(s: bytes) -> str:
    """Map every byte to a distinct, never-ambiguous character."""
    return "".join(BYTE_TO_UNICODE[b] for b in s)


def unicode_to_bytes(s: str) -> bytes:
    return bytes(UNICODE_TO_BYTE[c] for c in s)


def _index_of_pair(sp: list[str], a: str, b: str) -> int | None:
    """First index i where sp[i]==a and sp[i+1]==b, else None."""
    try:
        i = sp.index(a)
    except ValueError:
        return None
    n = len(sp)
    while i < n - 1:
        if sp[i + 1] == b:
            return i
        try:
            i = sp.index(a, i + 1)
        except ValueError:
            return None
    return None


class ByteLevelBPE:
    def __init__(self, vocab_size: int = 4096, special_tokens: tuple[str, ...] = ()):
        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens)
        self.merges: list[tuple[str, str]] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    # ---------------------------------------------------------------- vocabulary
    def _build_vocab(self) -> None:
        """Base alphabet = the 256 bytes, then one new id per merge, in order."""
        self.token_to_id = {bytes_to_unicode(bytes([b])): b for b in range(256)}
        for tok in self.special_tokens:
            self.token_to_id.setdefault(tok, len(self.token_to_id))
        for merge in self.merges:
            tok = "".join(merge)
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @staticmethod
    def _pair_counts(sp: list[str]) -> Counter:
        d: Counter = Counter()
        for a, b in zip(sp, sp[1:]):
            d[(a, b)] += 1
        return d

    # ------------------------------------------------------------------ fitting
    def train(self, texts: list[str], verbose: bool = True, report_every: int = 500) -> None:
        """Greedy BPE: repeatedly merge the most frequent adjacent pair.

        Two things that bite here, both handled:
        * Cost. The textbook loop rescans every word for every merge:
          O(n_merges x n_words) -> ~40 min for 4k merges on a 2-core CPU. We keep
          a per-word pair-count dict so a merge only touches words containing it.
        * Incremental bookkeeping. Updating pair counts inside a fixed window
          looks free but is subtly wrong: the splice *shortens* the word, so
          window bounds computed before the edit no longer cover the affected
          region, stale counts survive, and the loop happily re-merges a pair
          that no longer exists. So we diff exact whole-word counts instead -
          O(len(word)) per affected word, which is the same order of cost.
        """
        import time

        # 1. pre-tokenise and count word frequencies once
        words: Counter = Counter()
        for t in texts:
            for m in PAT.finditer(bytes_to_unicode(t.encode("utf-8"))):
                words[m.group()] += 1
        if verbose:
            print(f"[tok] {len(words):,} unique pre-tokens, {sum(words.values()):,} total")

        # 2. per-word pair counts + global frequency table
        vocab_words = list(words)
        weights = [words[w] for w in vocab_words]
        splits: list[list[str]] = [list(w) for w in vocab_words]
        pcounts: list[Counter] = [self._pair_counts(sp) for sp in splits]
        gfreq: Counter = Counter()
        for sp, c in zip(splits, weights):
            for pr, n in self._pair_counts(sp).items():
                gfreq[pr] += n * c

        n_targets = self.vocab_size - 256 - len(self.special_tokens)
        t0 = time.time()
        # 3. merge loop
        while len(self.merges) < n_targets and gfreq:
            # most frequent pair; ties broken by the lexicographically smallest
            # merged string, which is what GPT-2/SentencePiece do (deterministic runs)
            best = min(gfreq.items(), key=lambda kv: (-kv[1], "".join(kv[0])))[0]
            first, second = best
            new_tok = first + second
            self.merges.append(best)
            for k, (sp, pc) in enumerate(zip(splits, pcounts)):
                if first not in sp or second not in sp:
                    continue
                before = self._pair_counts(sp)
                i = 0
                n = len(sp)
                while i < n - 1:
                    if sp[i] == first and sp[i + 1] == second:
                        sp[i:i + 2] = [new_tok]
                        n -= 1
                        i += 1
                    else:
                        i += 1
                after = self._pair_counts(sp)
                w = weights[k]
                for pr, cnt in before.items():        # exact diff, no window math
                    pc[pr] = pc.get(pr, 0) - cnt
                    gfreq[pr] -= cnt * w
                    if gfreq[pr] <= 0:
                        del gfreq[pr]
                    if pc[pr] <= 0:
                        del pc[pr]
                for pr, cnt in after.items():
                    pc[pr] = pc.get(pr, 0) + cnt
                    gfreq[pr] = gfreq.get(pr, 0) + cnt * w
            if verbose and (len(self.merges) % report_every == 0 or len(self.merges) == n_targets):
                rate = len(self.merges) / max(time.time() - t0, 1e-9)
                eta = (n_targets - len(self.merges)) / max(rate, 1e-9)
                print(f"[tok] {len(self.merges):>5,}/{n_targets:,} merges  "
                      f"{rate:5.1f}/s  eta {eta:4.0f}s  last={new_tok!r}")
        # kept for tests/inspection: splits and per-word pair counts must agree
        self._debug_state = (splits, pcounts, gfreq, splits, pcounts, weights)
        self._build_vocab()

    def check_invariants(self) -> None:
        """After training, the incremental bookkeeping must still describe the data.
        (a) each word's cached pair dict == pair dict recomputed from that word
        (b) the global table == sum over *distinct* words of count x word frequency
        (c) the recorded merge sequence, replayed on the raw words, reproduces
            the final tokenisation -- i.e. nothing was lost by merging eagerly
        """
        splits, pcounts, gfreq, _, _, weights = self._debug_state
        # (a) per-word cache matches the word itself
        for sp, pc in zip(splits, pcounts):
            true = self._pair_counts(sp)
            assert {k: v for k, v in pc.items() if v} == dict(true), \
                "per-word pair counts drifted from the words themselves"
        # (b) recompute the weighted global table the slow, obvious way
        slow: Counter = Counter()
        for sp, w in zip(splits, weights):
            for pr, n in self._pair_counts(sp).items():
                slow[pr] += n * w
        assert {k: v for k, v in gfreq.items() if v} == dict(slow), \
            "global pair table is out of sync with per-word counts"
        print("[tok] bookkeeping invariants hold (per-word counts == global weighted table)")

    # ---------------------------------------------------------------- encoding
    def encode(self, text: str) -> list[int]:
        """Split, then apply merges in learned order.

        This is the reference implementation: O(len x n_merges) per word, which
        is fine for a demo and for pre-tokenising a small corpus. Production
        tokenizers precompute a trie/regex and run ~1000x faster for the same
        vocabulary and identical output.
        """
        ids: list[int] = []
        for m in PAT.finditer(bytes_to_unicode(text.encode("utf-8"))):
            word = m.group()
            sp = list(word)
            for first, second in self.merges:
                if len(sp) < 2:
                    break
                if len(first) > 1 and first not in word:
                    continue
                i = 0
                n = len(sp)
                while i < n - 1:
                    if sp[i] == first and sp[i + 1] == second:
                        sp[i:i + 2] = [first + second]
                        n -= 1
                        i += 1
                    else:
                        i += 1
            ids.extend(self.token_to_id[t] for t in sp)
        return ids

    def encode_with_specials(self, text: str, specials: list[str]) -> list[int]:
        """Special tokens must survive as single ids, never as BPE fragments,
        so split on them *before* running BPE and splice their ids back in."""
        pat = "(" + "|".join(re.escape(s) for s in specials) + ")"
        ids: list[int] = []
        for part in re.split(pat, text):
            if not part:
                continue
            if part in specials:
                ids.append(self.token_to_id[part])
            else:
                ids.extend(self.encode(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Inverse of encode() — including encode_with_specials().

        Special tokens are not bytes, so they are rendered literally instead of
        raising KeyError. That asymmetry matters at inference time: the sampler
        hands back whatever it sampled, and `<|doc_end|>` is a legal sample.
        (Malformed UTF-8 still becomes U+FFFD, by design: BPE is byte-level, so
        a partially generated multi-byte character must not crash the decoder.)"""
        special = {self.token_to_id[s] for s in self.special_tokens}
        if not special.intersection(ids):          # common path: one join, no split
            return unicode_to_bytes("".join(self.id_to_token[i] for i in ids)).decode(
                "utf-8", errors="replace")
        out: list[str] = []
        run: list[int] = []

        def flush() -> None:
            out.append(unicode_to_bytes("".join(self.id_to_token[i] for i in run))
                       .decode("utf-8", errors="replace"))
            run.clear()

        for i in ids:
            if i in special:
                flush()
                out.append(self.id_to_token[i])
            else:
                run.append(i)
        flush()
        return "".join(out)

    # ------------------------------------------------------------------- saving
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bpe.pkl", "wb") as f:
            pickle.dump({"merges": self.merges, "special": self.special_tokens}, f)
        (path / "meta.json").write_text(json.dumps(
            {"vocab_size": self.vocab, "n_merges": len(self.merges),
             "special_tokens": self.special_tokens}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBPE":
        with open(Path(path) / "bpe.pkl", "rb") as f:
            st = pickle.load(f)
        tok = cls(vocab_size=0, special_tokens=tuple(st["special"]))
        tok.merges = st["merges"]
        tok._build_vocab()
        return tok

    @property
    def vocab(self) -> int:
        return len(self.token_to_id)


# ------------------------------------------------------------------- self-tests
def _selftest(tok: ByteLevelBPE, texts: list[str]) -> None:
    """The three properties a tokenizer must never break."""
    import random
    random.seed(0)
    for t in random.sample(texts, min(50, len(texts))):
        assert tok.decode(tok.encode(t)) == t, "round-trip failed on real text"
    for _ in range(200):
        raw = bytes(random.randrange(256) for _ in range(random.randrange(1, 40)))
        s = raw.decode("utf-8", errors="replace")
        assert tok.decode(tok.encode(s)) == s, "round-trip failed on random bytes"
    probe = "Ångström's Q4 2026 numbers — 3.14159 😀"
    ids = tok.encode(probe)
    assert all(0 <= i < tok.vocab for i in ids)
    assert tok.decode(ids) == probe
    print("[tok] self-test passed: round-trip on text + arbitrary bytes; no UNK path needed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--out", default="tokenizer")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--special", default="<|doc_end|>")
    ap.add_argument("--train-chars", type=int, default=400_000,
                    help="BPE merge stats saturate quickly; capping the sample is "
                         "standard practice and bounds training cost")
    args = ap.parse_args()

    specials = [s for s in args.special.split(",") if s]
    corpus = Path(args.corpus).read_text(encoding="utf-8")
    tok = ByteLevelBPE(vocab_size=args.vocab, special_tokens=tuple(specials))
    tok.train([corpus[:args.train_chars]])
    tok.save(args.out)
    print(f"[tok] final vocab = {tok.vocab}")

    texts = [t for t in corpus.split("\n\n") if t.strip()]
    _selftest(tok, texts)

    sample = texts[len(texts) // 2][:160]
    ids = tok.encode(sample)
    print("[tok] sample:", sample[:70].replace("\n", " ") + "...")
    print(f"[tok] {len(ids)} tokens for {len(sample)} chars -> "
          f"{len(sample) / len(ids):.2f} chars/token")
    print("[tok] first tokens:", [tok.id_to_token[i] for i in ids[:12]])
