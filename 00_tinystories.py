"""
Stage 0 — Acquire and clean the TinyStories corpus.

TinyStories (Eldan & Li, 2023, arXiv:2305.07759) is ~2.1 M synthetically written
children's stories in deliberately simple English. It exists because a small
model *can* learn to write coherent English from it: the vocabulary is ~1.5 k
words, sentences are short, and every story has a beginning, a middle and an
end. That is exactly the property this project needs — the architecture is
2 M parameters, so the corpus, not the model, decides whether the output reads
like English.

Why this stage is a separate file
---------------------------------
The raw dataset is a 2.2 GB single .txt of concatenated stories. Two things
have to happen before it is training data, and neither belongs in the
tokenizer or the trainer:

 1. **Stream, don't download.** We only want ~20 MB (about 5 M tokens), so the
    file is fetched with HTTP range requests and the connection is closed once
    enough bytes have arrived. A 2.2 GB download for a 20 MB corpus is a
    mistake you make once.
 2. **Clean.** The public dump is *mostly* clean but not entirely: this copy
    carries mojibake (`â€œ` for a curly quote — a UTF-8 -> cp1252 -> UTF-8 round
    trip), some stories whose `<|endoftext|>` separator was eaten so two
    stories are glued into one sentence ("...she was happy.Once upon a time,
    ..."), truncated stories, and non-English/unicode noise. All of it is
    cheap to detect and expensive to ignore: garbage tokens are learned as if
    they were English.

In the official file each story ends with the GPT-2 separator `<|endoftext|>`. In
some redistributions (and in the copy this project was built from) that separator
was decoded away, leaving stories concatenated with only a sentence break between
them ("...she was happy.Once upon a time, ..."). `split_stories` therefore cuts on
`<|endoftext|>` *first*, and then still re-cuts at story openers inside a
document, so a corpus with no separators at all is cleaned just as well as the
official one.

Sources (`--source`), in the order a normal machine should try them:
  hf-http   stream the official TinyStories text from the HuggingFace Hub
  github    a 5 MB sample fixture from a public GitHub repo, for machines that
            cannot reach the Hub; enough to build a small corpus end-to-end
  local     a file already on disk (what this sandbox used, and the only source
            that works with no network at all)

Everything downstream (`01_tokenizer.py`, `02_preprocess.py`) consumes
`data/tinystories/corpus.txt`: stories separated by the `<|doc_end|>` marker,
plus `data/tinystories/stats.json` recording provenance (URL, sha256, counts).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

DOC_END = "<|doc_end|>"
SEPARATOR = f"\n\n{DOC_END}\n\n"

# The story separator used by the official TinyStories *.txt files (it is the
# GPT-2 end-of-text token, written literally into the text file).
EOT = "<|endoftext|>"

# --------------------------------------------------------------- provenance
HF_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
          "TinyStoriesV2-GPT4-train.txt")
# Fallback for machines that cannot reach the HuggingFace Hub: a 5 MB slice of
# the same corpus, committed to a public GitHub repository (the test fixture of
# a TinyStories training project). It is a *sample*, not the whole dataset: it
# cleans to ~4-5 MB of stories, not 20 MB. Verified offline against the copy of
# this file in tests/ (see README: --source github).
GITHUB_URL = ("https://raw.githubusercontent.com/kwu0737-commits/"
              "tinystories-transformer-from-scratch/main/tests/fixtures/"
              "tinystories_sample_5M.txt")

# Mojibake seen in this dump: UTF-8 bytes that were decoded as cp1252 and then
# re-encoded. Repairing them is exact (they are not real text), unlike guessing.
MOJIBAKE = {
    "â€œ": "“", "â€\x9d": "”", "â€": "”", "â€™": "’", "â€˜": "‘",
    "â€“": "–", "â€”": "—", "â€¦": "…", "Â ": " ", "Â": "",
    "Ã©": "é", "Ã¨": "è", "Ã ": "à",
}

# A new story almost always announces itself. Used to re-split stories whose
# blank-line separator was lost, hence only at a *sentence* boundary.
OPENER = (r"(?:Once upon a time|Once there|Once,|One day|One sunny day|One morning|"
          r"One night|There (?:was|were|once|is|lived)|Long ago|Many years ago|"
          r"A long time ago|It was a (?:sunny|rainy|snowy|windy|dark|bright|cold|warm|"
          r"beautiful|hot|stormy)|In a (?:small|big|little|tiny|quiet|faraway|far-off)|"
          r"Deep in|Far away)")
BOUNDARY_RE = re.compile(rf'(?<=[.!?"”])[ \t]*(?={OPENER}\b)')
BLANK_LINE_RE = re.compile(r"\n\s*\n")

# Characters allowed to survive in a story (everything else non-ASCII counts
# against the story's "clean" ratio).
ALLOWED_PUNCT = set("“”‘’—–…’‘·")

SENTENCE_END_RE = re.compile(r'[.!?"”]\s*$')
JUNK_RE = re.compile(r"https?://|<\|[a-z_]+\|>|GPT|\bTODO\b|\\x[0-9a-f]{2}")


# ------------------------------------------------------------------ cleaning
def repair_mojibake(text: str) -> str:
    for bad, good in MOJIBAKE.items():
        text = text.replace(bad, good)
    return text


def normalise(text: str) -> str:
    """Whitespace/unicode normalisation shared by every code path."""
    text = repair_mojibake(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ ]{2,}", " ", text)          # collapse runs of spaces
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_stories(text: str, *, min_chars: int = 300, max_chars: int = 2000) -> list[str]:
    """`text` -> list of story strings.

    Three facts about this dump drive the algorithm:

    * The official file separates stories with `<|endoftext|>`. That is a story
      boundary, always, and it is checked first.
    * Where the separator is missing, a story still almost always *opens* with a
      formula ("Once upon a time", "One day", "There was a little girl"), so a
      sentence boundary in front of an opener is a story boundary too - that is
      the "...she was happy.Once upon a time, ..." failure mode.
    * A paragraph break is *not* a story boundary: dialogue and the moral of a
      story are regularly their own paragraph. Treating every blank line as a
      story would shatter stories into 136-character fragments (median).

    So: split on `<|endoftext|>`, then cut each document at opener boundaries,
    then greedily re-group the pieces into stories of at least `min_chars` (a
    mid-story "One day" then does not split a story in half) and at most
    `max_chars` (a run of stories with no opener at all still gets cut
    somewhere sane). Regrouping never crosses an `<|endoftext|>`.
    """
    stories: list[str] = []
    for doc in text.split(EOT):
        if not doc.strip():
            continue                       # "...<|endoftext|>\n<|endoftext|>..." padding
        pieces: list[str] = []
        for para in BLANK_LINE_RE.split(doc):
            para = para.strip()
            if not para:
                continue
            pieces.extend(p.strip() for p in BOUNDARY_RE.split(para) if p.strip())
        buf: list[str] = []
        size = 0
        for piece in pieces:
            starts_story = bool(re.match(OPENER, piece))
            if buf and ((starts_story and size >= min_chars) or size >= max_chars):
                stories.append(" ".join(buf))
                buf, size = [], 0
            buf.append(piece)
            size += len(piece)
        if buf:
            stories.append(" ".join(buf))
    return stories


def clean_story(story: str, min_chars: int, max_chars: int) -> str | None:
    """Return the story if it is usable English prose, else None.

    The checks are ordered cheapest-first and each one rejects a failure mode
    actually present in the dump: mid-story cuts, glued stories that defeated
    the splitter, truncated endings, CJK/emoji noise, and Prompt-GPT artifacts
    from the generation process itself.
    """
    s = " ".join(story.split())                     # single-line prose
    if not (min_chars <= len(s) <= max_chars):
        return None
    if JUNK_RE.search(s):
        return None
    if not SENTENCE_END_RE.search(s):               # truncated mid-sentence
        return None
    if not s[0].isupper():                          # starts mid-word/mid-sentence
        return None
    # Sentence structure: every ". " must be followed by a capital or a quote.
    for m in re.finditer(r"[.!?] ", s):
        nxt = s[m.end()]
        if not (nxt.isupper() or nxt in '"“'):
            return None
    # Non-ASCII budget: curly quotes and em dashes are fine, a wall of CJK or
    # emoji (both appear in this dump) is not.
    weird = sum(1 for ch in s if ord(ch) > 126 and ch not in ALLOWED_PUNCT)
    if weird > max(2, len(s) * 0.01):
        return None
    # Control characters and lone surrogates never belong in a text corpus.
    if any(unicodedata.category(ch) in {"Cc", "Cs", "Co", "Cn"} for ch in s):
        return None
    # Vocabulary sanity: real stories are mostly lowercase letters/spaces.
    letters = sum(ch.isalpha() for ch in s)
    if letters < 0.6 * len(s):
        return None
    return s


def build_corpus(raw: str, *, min_chars: int = 200, max_chars: int = 2500,
                 max_chars_total: int = 20_000_000, dedupe: bool = True,
                 log_every: int = 20_000) -> tuple[list[str], dict]:
    """Raw dump text -> (stories, stats). Streaming-friendly: one pass."""
    stories: list[str] = []
    seen: set[str] = set()
    total = 0
    n_raw = 0
    for chunk in split_stories(normalise(raw)):
        n_raw += 1
        s = clean_story(chunk, min_chars, max_chars)
        if s is None:
            continue
        if dedupe:
            key = hashlib.blake2b(s.encode(), digest_size=16).digest()
            if key in seen:
                continue
            seen.add(key)
        stories.append(s)
        total += len(s) + len(SEPARATOR)
        if log_every and len(stories) % log_every == 0:
            print(f"[data] {len(stories):,} stories kept, {total/1e6:.1f} M chars", flush=True)
        if total >= max_chars_total:
            break
    stats = {
        "source_stories": n_raw,
        "stories": len(stories),
        "chars": total,
        "mean_story_chars": round(total / max(len(stories), 1), 1),
        "max_chars_total": max_chars_total,
        "rejected": n_raw - len(stories),
    }
    return stories, stats


# ------------------------------------------------------------------ download
def stream_url(url: str, max_bytes: int, block: int = 1 << 21) -> bytes:
    """Read at most `max_bytes` from `url` and return them, decoding leniently.

    Read in blocks and stop early: the official file is 2.2 GB and we want the
    first ~20 MB of it. The tail is trimmed at the last complete story line by
    the caller, so a truncated multi-byte character cannot leak into the
    corpus.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "llm-from-scratch/1.0"})
    buf = bytearray()
    with urllib.request.urlopen(req, timeout=60) as resp:
        while len(buf) < max_bytes:
            data = resp.read(min(block, max_bytes - len(buf)))
            if not data:
                break
            buf.extend(data)
    text = buf.decode("utf-8", errors="ignore")
    if len(text) == len(buf) and not text.endswith((".", "!", "?", '"', "”")):
        # cut back to the last sentence end: cheaper and safer than repairing
        cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if cut > 0:
            text = text[:cut + 1]
    return text


def get_raw(source: str, *, input_path: str | None, max_bytes: int) -> tuple[str, dict]:
    if source == "local":
        if not input_path:
            raise SystemExit("--input is required with --source local")
        p = Path(input_path)
        if not p.exists():
            raise SystemExit(f"--input {p} does not exist")
        raw_bytes = p.read_bytes()
        if len(raw_bytes) > max_bytes:
            raw_bytes = raw_bytes[:max_bytes]
        return raw_bytes.decode("utf-8", errors="ignore"), {"url": "local:" + str(p)}
    url = HF_URL if source == "hf-http" else GITHUB_URL
    try:
        return stream_url(url, max_bytes), {"url": url}
    except Exception as e:                                     # noqa: BLE001
        raise SystemExit(
            f"could not stream {url}: {type(e).__name__}: {e}\n"
            f"      Try --source github (5 MB sample) or --source local --input <file>,\n"
            f"      or download the file yourself and pass it with --source local."
        ) from e


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Stream and clean TinyStories into data/.")
    ap.add_argument("--source", choices=["hf-http", "github", "local"], default="hf-http",
                    help="hf-http: stream from HuggingFace Hub (default); github: 5 MB "
                         "sample fixture (no Hub access needed); local: a file on disk")
    ap.add_argument("--input", default=None, help="file to read with --source local")
    ap.add_argument("--out", default="data/tinystories", help="output directory")
    ap.add_argument("--max-bytes", type=int, default=32 << 20,
                    help="bytes to stream before trimming (default 32 MB)")
    ap.add_argument("--max-chars", type=int, default=20_000_000,
                    help="stop once this many clean characters are collected")
    ap.add_argument("--min-story-chars", type=int, default=200)
    ap.add_argument("--max-story-chars", type=int, default=2500)
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw, prov = get_raw(args.source, input_path=args.input, max_bytes=args.max_bytes)
    print(f"[data] raw text {len(raw):,} chars from {prov['url']}")

    stories, stats = build_corpus(raw, min_chars=args.min_story_chars,
                                  max_chars=args.max_story_chars,
                                  max_chars_total=args.max_chars,
                                  dedupe=not args.no_dedupe)
    if len(stories) < 100:
        raise SystemExit(f"only {len(stories)} clean stories survived; something is wrong")

    corpus = SEPARATOR.join(stories)
    path = out / "corpus.txt"
    path.write_text(corpus, encoding="utf-8")
    stats.update({
        "dataset": "TinyStories (Eldan & Li 2023, arXiv:2305.07759), TinyStoriesV2-GPT4 train split",
        "source": args.source, "source_url": prov["url"], "dataset_url": HF_URL,
        "sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "bytes_on_disk": path.stat().st_size,
        "separator": DOC_END,
        "min_story_chars": args.min_story_chars, "max_story_chars": args.max_story_chars,
        "dedupe": not args.no_dedupe,
    })
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[data] wrote {path} ({path.stat().st_size/1e6:.1f} MB, "
          f"{stats['stories']:,} stories, mean {stats['mean_story_chars']:.0f} chars)")
    print(f"[data] rejected {stats['rejected']:,} of {stats['source_stories']:,} raw chunks")
    print("[data] next: python 01_tokenizer.py --corpus data/tinystories/corpus.txt --out tokenizer")


if __name__ == "__main__":
    sys.exit(main())
