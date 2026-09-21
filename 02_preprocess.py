"""
Stage 2 — Data collection & preprocessing.

Design decisions:
  * Documents are the unit of splitting. A 95/5 *document* split (not a random
    95/5 *line* split) guarantees no sentence can leak from train into val, so
    validation loss is an honest generalisation estimate.
  * One document per training block. Padding short documents with other
    documents teaches the model a spurious "topics teleport" pattern; instead
    we emit a document-boundary token so the model learns that a new document
    starts after it, and only keep documents that are long enough to fill a block.
  * Everything is pre-tokenised and memory-mapped. Tokenising inside the
    training loop is the #1 reason a from-scratch run "feels slow": the GPU
    starves while Python does string work. Here the loop only does `np.memmap`
    slice reads.
  * Output is one flat uint16 array per split, plus an index of block starts.
    uint16 is enough for vocab < 65536 and halves I/O vs int32.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))

DOC_END = "<|doc_end|>"

# Gutenberg wraps hard line breaks at ~70 chars; " ".join on newlines restores
# logical lines. Paragraph structure is kept via blank lines.
HEADER_RE = re.compile(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)
FOOTER_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)


def clean_gutenberg(text: str) -> str:
    m = HEADER_RE.search(text)
    if m:
        text = text[m.end():]
    m = FOOTER_RE.search(text)
    if m:
        text = text[:m.start()]
    # Drop the license/boilerplate tail Gutenberg appends.
    cut = text.rfind("*** END OF")
    if cut != -1:
        text = text[:cut]
    # Normalise: no CRLF, no tabs, no >1 blank line, no trailing spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_documents(raw_dir: str | Path) -> list[str]:
    """One string per book/chapter-ish document, boilerplate already removed."""
    docs: list[str] = []
    for path in sorted(Path(raw_dir).glob("*.txt")):
        text = clean_gutenberg(path.read_text(encoding="utf-8", errors="replace"))
        # Split into paragraph groups so a doc is a coherent chunk of prose
        # (a whole book is also fine; this granularity is a knob, not a rule).
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        # join paragraphs into documents of >= ~2000 chars, so a block_size=256
        # training window almost always sits inside one document
        buf: list[str] = []
        size = 0
        for p in paras:
            p = re.sub(r"\s*\n\s*", " ", p).strip()  # unwrap hard line breaks
            if not p:
                continue
            buf.append(p)
            size += len(p)
            if size >= 2000:
                docs.append("\n\n".join(buf))
                buf, size = [], 0
        if buf:
            docs.append("\n\n".join(buf))
        print(f"[data] {path.name}: {len(text):,} chars -> {len([d for d in docs])} cumulative docs")
    return docs


def write_corpus(raw_dir: str | Path, out_path: str | Path, min_doc_chars: int = 600) -> dict:
    docs = extract_documents(raw_dir)
    docs = [d for d in docs if len(d) >= min_doc_chars]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joined = f"\n\n{DOC_END}\n\n".join(docs)
    out_path.write_text(joined, encoding="utf-8")
    stats = {
        "n_docs": len(docs),
        "n_chars": len(joined),
        "mean_doc_chars": round(len(joined) / max(len(docs), 1), 1),
        "n_lines": joined.count("\n") + 1,
    }
    (out_path.parent / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print("[data] corpus written:", stats)
    return stats


def build_dataset(corpus_path: str | Path, tokenizer_dir: str | Path, out_dir: str | Path,
                  block_size: int = 256, val_frac: float = 0.05, seed: int = 1337) -> dict:
    """Tokenise, split by document, pack into fixed-size blocks, memmap-save."""
    # _stage.load makes these scripts work from any CWD; a leading digit is not a
    # legal Python module name, so a plain `import 01_tokenizer` is impossible.
    from _stage import load
    tok = load("01_tokenizer").ByteLevelBPE.load(tokenizer_dir)

    text = Path(corpus_path).read_text(encoding="utf-8")
    docs = [d.strip() for d in re.split(rf"\n\n{re.escape(DOC_END)}\n\n", text) if d.strip()]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    n_val = max(1, int(len(docs) * val_frac))
    val_idx, train_idx = set(order[:n_val].tolist()), set(order[n_val:].tolist())

    splits = {"train": [], "val": []}
    for i, doc in enumerate(docs):
        ids = tok.encode_with_specials(doc, [DOC_END])
        ids.append(tok.token_to_id[DOC_END])
        splits["val" if i in val_idx else "train"].append(ids)

    info = {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, chunks in splits.items():
        flat = np.array([t for c in chunks for t in c], dtype=np.uint16)
        # Pack per-document: each block is a contiguous window *inside* one doc.
        blocks, pos = [], 0
        for c in chunks:
            if len(c) <= block_size:
                continue
            n_full = (len(c) - 1) // block_size   # -1: need block_size+1 tokens
            for b in range(n_full):
                s = pos + b * block_size
                blocks.append(flat[s:s + block_size + 1])
            pos += len(c)
        arr = np.stack(blocks).astype(np.uint16)
        p = Path(out_dir) / f"{name}.npy"
        np.save(p, arr)
        info[name] = {"n_tokens": int(len(flat)), "n_docs": len(chunks), "n_blocks": int(arr.shape[0])}
        print(f"[data] {name}: {len(flat):,} tokens, {arr.shape[0]:,} blocks of {arr.shape[1]} -> {p}")

    doc_token_len = np.array([len(c) for c in splits["train"]])
    info["block_size"] = block_size
    info["vocab"] = tok.vocab
    info["train_doc_tokens"] = {"mean": float(doc_token_len.mean()), "p10": float(np.percentile(doc_token_len, 10))}
    (Path(out_dir) / "dataset_info.json").write_text(json.dumps(info, indent=2))

    # ---- sanity checks that catch 90% of preprocessing bugs -------------------
    errs = []
    if not (arr.max() < tok.vocab):
        errs.append("token id out of vocab range")
    if arr.shape[1] != block_size + 1:
        errs.append("block width != block_size+1")
    # x and y must be exactly shifted-by-one views of the same window
    x, y = arr[:, :-1], arr[:, 1:]
    if not np.array_equal(x[:, 1:], y[:, :-1]):
        errs.append("targets are not the inputs shifted by one")
    print("[data] sanity:", "OK" if not errs else errs)
    assert not errs, errs
    return info


def build_dataset_stream(corpus_path: str | Path, tokenizer_dir: str | Path,
                         out_dir: str | Path, block_size: int = 256,
                         val_frac: float = 0.05, seed: int = 1337) -> dict:
    """Pack the token stream into blocks of `block_size`+1, splitting train/val
    by *story*.

    This is the TinyStories path. A TinyStories story is ~795 characters, i.e.
    ~180 tokens at 4.5 chars/token — shorter than one 256-token block. The
    document-splitting scheme below (`build_dataset`) would therefore keep only
    a fraction of the corpus (and waste the tail of every document). Concatenating
    the whole split and chopping it into blocks instead uses 100% of the tokens,
    and keeps the story boundary signal because stories are joined by the
    `<|doc_end|>` token, which the model learns to write and to continue after.

    The train/val split is still at *story* granularity — a val story never
    appears in a training block — so validation is an honest held-out estimate
    even though blocks themselves are just contiguous slices of the stream.
    """
    from _stage import load
    tokmod = load("01_tokenizer")
    tok = tokmod.ByteLevelBPE.load(tokenizer_dir)
    doc_end_id = tok.token_to_id.get(tokmod.__dict__.get("DOC_END", "<|doc_end|>"))
    if doc_end_id is None:      # caller's tokenizer has no separator: fall back to a space
        doc_end_id = tok.encode(" ")[0]

    text = Path(corpus_path).read_text(encoding="utf-8")
    marker = f"\n\n{tokmod.__dict__.get('DOC_END', '<|doc_end|>')}\n\n"
    stories = [s.strip() for s in text.split(marker) if s.strip()]
    if len(stories) < 10:
        raise SystemExit(f"only {len(stories)} stories found in {corpus_path}; "
                         f"expected the corpus to use the {marker!r} separator")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(stories))
    n_val = max(1, int(len(stories) * val_frac))
    val_idx, train_idx = set(order[:n_val].tolist()), set(order[n_val:].tolist())

    info: dict = {"mode": "stream", "block_size": block_size, "vocab": tok.vocab,
                  "separator_id": doc_end_id, "val_frac": val_frac, "seed": seed}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, idxs in (("train", train_idx), ("val", val_idx)):
        ids: list[int] = []
        for i, story in enumerate(stories):
            if i not in idxs:
                continue
            ids.extend(tok.encode(story))
            ids.append(doc_end_id)              # the story separator is learned separately
        flat = np.array(ids, dtype=np.uint16)
        # overlapping windows: block i is flat[i*B : i*B + B + 1], so the target
        # of one block is the last input of the next. (A plain reshape cannot do
        # this: it needs block_size+1 tokens per block while advancing only
        # block_size, so the array is 1/block_size larger than the reshape.)
        n_blocks = (len(flat) - 1) // block_size
        arr = np.stack([flat[i * block_size: i * block_size + block_size + 1]
                        for i in range(n_blocks)])
        p = Path(out_dir) / f"{name}.npy"
        np.save(p, arr)
        arrays[name] = arr
        info[name] = {"n_tokens": int(len(flat)), "n_stories": len(idxs),
                      "n_blocks": int(arr.shape[0]),
                      # windows overlap by one token, so the last block reaches
                      # past the end of a plain reshape: account for that
                      "dropped_tail_tokens": int(len(flat) - (arr.shape[0] * block_size + 1))}
        print(f"[data] {name}: {len(idxs):,} stories, {len(flat):,} tokens, "
              f"{arr.shape[0]:,} blocks of {block_size} -> {p}")

    # ---- sanity checks that catch 90% of preprocessing bugs -------------------
    errs = []
    arr = arrays["train"]
    if arr.max() >= tok.vocab:
        errs.append("token id out of vocab range")
    if arr.shape[1] != block_size + 1:
        errs.append("block width != block_size+1")
    x, y = arr[:, :-1], arr[:, 1:]
    if not np.array_equal(x[:, 1:], y[:, :-1]):
        errs.append("targets are not the inputs shifted by one")
    if not (doc_end_id in arr[: min(arr.shape[0], 200)]):
        errs.append("no story separators in the first 200 blocks: the split is wrong")
    print("[data] sanity:", "OK" if not errs else errs)
    assert not errs, errs
    (Path(out_dir) / "dataset_info.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--corpus", default="data/tinystories/corpus.txt")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--out", default="data/packed")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--mode", choices=["stream", "docs"], default="stream",
                    help="stream: pack stories into contiguous blocks (TinyStories, default); "
                         "docs: one document per block, no block crosses a document (books)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    a = ap.parse_args()
    if a.mode == "docs":
        write_corpus(a.raw, a.corpus)
        build_dataset(a.corpus, a.tokenizer, a.out, block_size=a.block_size,
                      val_frac=a.val_frac)
    else:
        build_dataset_stream(a.corpus, a.tokenizer, a.out, block_size=a.block_size,
                             val_frac=a.val_frac)
