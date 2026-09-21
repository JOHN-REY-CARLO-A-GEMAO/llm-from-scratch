"""
Retrieval-augmented generation, from scratch: a document in, the two most
relevant passages out, no vector database and no third-party package.

    index = rag.DocumentIndex.from_text(open("notes.md").read(), name="notes.md")
    block, hits = index.context_block("when was the bridge built?", 180, cost=len)

The retrieval stack is deliberately the boring one:

 * **BM25** (Okapi, k1=1.5, b=0.75) over word unigrams and bigrams with a small
   stop-word list, so "when was the bridge built" ranks the sentence that *has*
   "bridge" above the one that merely shares "the". Bigrams are what make a
   phrase like "blue whale" a single, much rarer term.
 * **Character-trigram cosine** as a fallback when BM25 scores nothing (a typo,
   an inflection, a word the document spells differently). Without it, a query
   with one misspelled keyword retrieves nothing at all — the failure mode users
   actually hit.
 * **Chunks of ~200 tokens** (≈900 characters, cut at paragraph and sentence
   boundaries with a 150-character overlap) — big enough to hold an answer,
   small enough for one to fit the 256-token context of the shipped model.

Why the budget maths is in here rather than in the caller: the model has a
256-token context. A retrieval step that returns three paragraphs is not "more
context", it is a prompt where prefill drops the question. `context_block()`
therefore takes a token budget and a `cost(text) -> tokens` callable, and returns
the longest context that *fits* — trimming each passage around the words that
matched, not just to its first N characters.

Everything is stdlib-only and torch-free: `web_ui.py` imports it, the tests
import it, and nothing here opens a socket, a file or a subprocess.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

#: Target chunk size. ~900 characters is ~200 tokens for English prose (about
#: 4.5 characters per token), which is what the SFT corpus looks like.
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
#: Smaller than this and a "chunk" is a stray heading; larger and one chunk eats
#: the whole 256-token context on its own.
MIN_CHUNK_CHARS = 24
MAX_CHUNK_CHARS = 2400

#: Document limits. There is no index on disk and no vector store: the document
#: lives in memory for the session, so these are the RAM guard rails.
MAX_DOC_CHARS = 2_000_000
MAX_CHUNKS = 4000

#: BM25 parameters (the values from the original paper, still the sane default).
BM25_K1 = 1.5
BM25_B = 0.75
#: Retrieval floor. "Find me the two best chunks, even when none of them are
#: relevant" is how a RAG system injects noise into a 256-token prompt and then
#: blames the model. Below this, the honest answer is *nothing retrieved* — the
#: UI says so, and the model is asked without context instead of with the wrong
#: context. Measured as a *fraction* of the best match this index could produce
#: for the query, so the floor means the same thing in a two-chunk note (small
#: IDFs) and a two-hundred-chunk book — a fixed divisor would reject every
#: passage of a tiny document.
#:
#: 0.30 was chosen from measurements, not taste. On a one-chunk note a question
#: *about* the note's subject scores 1.00 ("where does the zorble live?", "what
#: colour is the zorble?") while an unrelated question that happens to share a
#: bigram scores 0.18–0.21 ("why does it rain?", "who won the world cup") — the
#: floor separates them. In a 232-chunk book the same two groups overlap
#: (0.29–0.93 vs 0.49–1.00), because a large document contains a passage that
#: mentions almost anything; there the *sources* fold and the grounding warning
#: are what keep a weak match honest, and this floor only removes the bottom.
MIN_RELEVANCE = 0.30
TRIGRAM_SIZE = 3

#: Words carrying no retrieval signal. Kept short on purpose: BM25's IDF already
#: discounts whatever is common *in this document*, and a long stop list only
#: removes words a short question may have nothing else to match on.
STOPWORDS = frozenset("""
a an and any are as at be been but by can could did do does doing for from had has
have having he her here hers him his how i if in into is it its itself me my no nor
not of off on once only or other our out over own same she should so some such than
that the their them then there these they this those through to too under until up
very was we were what when where which who whom why will with would you your yours
""".split())

WORD_RE = re.compile(r"[0-9a-z]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")

#: The exact injection format the chat engines use. Kept here (not in web_ui.py)
#: so the terminal, the browser and the tests cannot drift apart.
DOC_HEADER = "Context from uploaded document:"
DOC_INSTRUCTION = "Answer the user's question based ONLY on this context."


class RagError(ValueError):
    """The document or the query cannot be indexed/answered as given."""


# ------------------------------------------------------------------ tokenising
def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric words. No stemming: a wrong stem is worse than
    no stem, and the trigram fallback covers what it would have caught."""
    return WORD_RE.findall(str(text).lower())


def _stem(word: str) -> str:
    """Plural folding only — `words`/`word` are the same term, `running` is not
    `run` and pretending otherwise is how retrieval silently misfires."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def terms(text: str) -> list[str]:
    """Index/query terms: content unigrams plus word bigrams.

    A bigram keeps its stop words (`is the`) because the *phrase* is the signal;
    a unigram that is nothing but a stop word is dropped.
    """
    words = [_stem(word) for word in tokenize(text)]
    out = [word for word in words if word not in STOPWORDS]
    out += [f"{a} {b}" for a, b in zip(words, words[1:])]
    return out


def trigrams(text: str) -> Counter:
    """Character trigrams over normalised text, for the fuzzy fallback."""
    body = " ".join(WORD_RE.findall(str(text).lower()))
    if len(body) < TRIGRAM_SIZE:
        return Counter([body] if body else [])
    return Counter(body[i:i + TRIGRAM_SIZE] for i in range(len(body) - TRIGRAM_SIZE + 1))


def _cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot = sum(left[t] * right[t] for t in shared)
    if not dot:
        return 0.0
    norm = math.sqrt(sum(v * v for v in left.values())) * \
        math.sqrt(sum(v * v for v in right.values()))
    return dot / norm if norm else 0.0


# ------------------------------------------------------------------- chunking
def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split prose into overlapping passages, cut at paragraph/sentence edges.

    Order of preference for every cut: paragraph break, then sentence end, then
    (only when a single sentence is longer than the whole chunk) a word boundary.
    A passage that ends mid-sentence is still readable, but a cut that splits a
    table row or a numbered step is not — hence the order.
    """
    if chunk_chars < MIN_CHUNK_CHARS:
        raise RagError(f"chunk_chars must be at least {MIN_CHUNK_CHARS}")
    overlap = max(0, min(int(overlap), chunk_chars // 2))
    body = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not body:
        return []
    if len(body) <= chunk_chars:
        return [body]

    units: list[str] = []
    for paragraph in PARAGRAPH_RE.split(body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_chars:
            units.append(paragraph)
            continue
        for sentence in SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            while len(sentence) > chunk_chars:           # a runaway "sentence"
                cut = sentence.rfind(" ", 0, chunk_chars)
                cut = cut if cut > MIN_CHUNK_CHARS else chunk_chars
                units.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                units.append(sentence)

    chunks: list[str] = []
    buffer = ""
    for unit in units:
        if buffer and len(buffer) + len(unit) + 2 > chunk_chars:
            chunks.append(buffer)
            tail = _tail(buffer, overlap)
            buffer = f"{tail}\n{unit}".strip() if tail else unit
        else:
            buffer = f"{buffer}\n{unit}".strip() if buffer else unit
    if buffer:
        chunks.append(buffer)
    # Drop a trailing fragment that only repeats the overlap of its predecessor.
    if len(chunks) > 1 and len(chunks[-1]) <= overlap + MIN_CHUNK_CHARS:
        tail = chunks.pop()
        if tail.strip() not in chunks[-1]:
            chunks.append(tail)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS] or [body[:chunk_chars]]


def _tail(text: str, limit: int) -> str:
    """The last ~`limit` characters of `text`, starting on a word boundary.

    `find`, not `rfind`: the first space *after* the cut keeps the overlap the
    requested size. `rfind` would snap to the last space in the text and quietly
    return a five-character overlap.
    """
    if limit <= 0 or len(text) <= limit:
        return text if limit > 0 else ""
    cut = text.find(" ", len(text) - limit)
    return text[cut + 1:] if cut > 0 else text[len(text) - limit:]


def trim_to(text: str, limit: int) -> str:
    """The longest prefix of `text` that fits in `limit` characters.

    Cuts on a word boundary and marks the elision with `…`, so a passage that was
    shortened is *visibly* shortened rather than silently ended mid-thought.
    """
    body = str(text or "")
    if limit <= 0:
        return ""
    if len(body) <= limit:
        return body
    window = body[:max(1, limit - 1)]
    cut = window.rfind(" ")
    if cut > len(window) * 0.6:
        window = window[:cut]
    return window.rstrip(" ,;:") + "…"


def window_around(text: str, needles, limit: int) -> str:
    """A `limit`-character window of `text` centred on the first matching term.

    Trimming a passage to its *first* N characters throws away the sentence that
    matched whenever the match is further down — which is exactly the case
    retrieval just decided was the relevant one.
    """
    body = str(text or "").strip()
    if len(body) <= limit or limit <= 0:
        return body if limit > 0 else ""
    lowered = body.lower()
    position = -1
    for needle in needles or ():
        found = lowered.find(str(needle).lower())
        if found >= 0 and (position < 0 or found < position):
            position = found
    if position < 0:
        return trim_to(body, limit)
    start = max(0, position - limit // 3)
    if start:
        space = body.find(" ", start)
        if space >= 0 and space < position:
            start = space + 1
    piece = body[start:start + limit]
    space = piece.rfind(" ")
    if space > len(piece) * 0.7:
        piece = piece[:space]
    return ("…" if start else "") + piece.strip() + ("…" if start + len(piece) < len(body) else "")


# ---------------------------------------------------------------------- hits
@dataclass(frozen=True)
class Chunk:
    index: int
    text: str

    def preview(self, limit: int = 160) -> str:
        return trim_to(self.text, limit)


@dataclass(frozen=True)
class Hit:
    """One retrieved passage with the score that put it there."""

    chunk: Chunk
    score: float
    relevance: float                     # 0..1, for display only
    matched: tuple = ()

    @property
    def index(self) -> int:
        return self.chunk.index

    @property
    def text(self) -> str:
        return self.chunk.text

    def preview(self, limit: int = 160) -> str:
        return self.chunk.preview(limit)

    def excerpt(self, limit: int = 240) -> str:
        return window_around(self.chunk.text, self.matched, limit)

    def to_json(self, limit: int = 240) -> dict:
        return {"index": self.index, "score": round(self.score, 4),
                "relevance": round(self.relevance, 3),
                "chars": len(self.chunk.text), "excerpt": self.excerpt(limit)}


# ---------------------------------------------------------------------- index
@dataclass
class DocumentIndex:
    """An in-memory BM25 index over the chunks of one document."""

    name: str = "document"
    chunk_chars: int = CHUNK_CHARS
    overlap: int = CHUNK_OVERLAP
    text: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    _freq: list[Counter] = field(default_factory=list, repr=False)
    _trigrams: list[Counter] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _average_length: float = 0.0

    # -- construction
    @classmethod
    def from_text(cls, text: str, name: str = "document",
                  chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP
                  ) -> "DocumentIndex":
        body = str(text or "")
        if len(body) > MAX_DOC_CHARS:
            raise RagError(f"document is larger than {MAX_DOC_CHARS:,} characters")
        pieces = chunk_text(body, chunk_chars, overlap)
        if not pieces:
            raise RagError("document is empty")
        if len(pieces) > MAX_CHUNKS:
            raise RagError(f"document needs more than {MAX_CHUNKS} chunks")
        index = cls(name=str(name or "document")[:120], chunk_chars=chunk_chars,
                    overlap=overlap, text=body)
        index.chunks = [Chunk(i, piece) for i, piece in enumerate(pieces)]
        index._build()
        return index

    def _build(self) -> None:
        self._freq, self._trigrams, self._df = [], [], Counter()
        for chunk in self.chunks:
            frequency = Counter(terms(chunk.text))
            self._freq.append(frequency)
            self._trigrams.append(trigrams(chunk.text))
            for term in frequency:
                self._df[term] += 1
        total = sum(sum(f.values()) for f in self._freq)
        self._average_length = (total / len(self.chunks)) if self.chunks else 0.0

    # -- retrieval
    def _idf(self, term: str) -> float:
        count = self._df.get(term, 0)
        if not count:
            return 0.0
        return math.log(1 + (len(self.chunks) - count + 0.5) / (count + 0.5))

    def search(self, query: str, top_k: int = 4,
               min_relevance: float = MIN_RELEVANCE) -> list[Hit]:
        """The `top_k` best chunks for `query`, best first (possibly none).

        BM25 ranking, with a character-trigram cosine fallback only when BM25
        scores nothing at all — mixing the two would let a fuzzy match outrank a
        real keyword hit. Hits below `min_relevance` are dropped rather than
        padded in: an unrelated passage is worse than no passage.
        """
        if not self.chunks or not str(query or "").strip():
            return []
        top_k = max(1, int(top_k))
        query_terms = terms(query)
        unique = list(dict.fromkeys(query_terms))
        # The best a chunk of *average* length could score for this query: one
        # occurrence of every query term. Normalising against it makes relevance
        # a fraction of the achievable match instead of a raw BM25 number, so
        # the relevance floor means the same thing in a two-chunk note and in a
        # two-hundred-chunk book (a small document has small IDFs, and a fixed
        # divisor then rejects every passage in it).
        ceiling = sum(self._idf(term) for term in unique) or 0.0
        scores: list[float] = []
        for frequency, chunk in zip(self._freq, self.chunks):
            length = sum(frequency.values()) or 1
            score = 0.0
            for term in unique:
                tf = frequency.get(term, 0)
                if not tf:
                    continue
                idf = self._idf(term)
                if idf <= 0:
                    continue
                score += idf * (tf * (BM25_K1 + 1)) / (
                    tf + BM25_K1 * (1 - BM25_B + BM25_B * length / (self._average_length or 1)))
            scores.append(score)
        mode = "bm25"
        if max(scores) <= 0.0:
            query_trigrams = trigrams(query)
            scores = [_cosine(query_trigrams, chunk_trigrams)
                      for chunk_trigrams in self._trigrams]
            mode = "trigram"
        best = max(scores) if scores else 0.0
        if best <= 0.0:
            return []
        matched = tuple({t for t in unique if len(t) > 2})
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        hits: list[Hit] = []
        for position in order[:top_k]:
            score = scores[position]
            if score <= 0.0:
                continue
            relevance = (score if mode == "trigram"
                         else min(1.0, score / ceiling) if ceiling > 0 else 0.0)
            if relevance < min_relevance:
                continue
            hits.append(Hit(self.chunks[position], score, min(1.0, round(relevance, 4)), matched))
        return hits

    def best(self, query: str) -> Hit | None:
        hits = self.search(query, top_k=1)
        return hits[0] if hits else None

    # -- prompt injection
    def context_block(self, question: str, budget_tokens: int,
                      cost, top_k: int = 2
                      ) -> tuple[str, list[Hit]]:
        """The `Context from uploaded document:` system block, sized to fit.

        `cost(text) -> int` counts tokens for the caller's tokenizer, because
        only the caller knows the serialization. The block is built at decreasing
        sizes until it fits the budget; when even the header and the instruction
        do not fit, the context is dropped (`("", hits)`) rather than injected
        over budget — prefill would silently eat the question instead.
        """
        hits = self.search(question, top_k=max(1, int(top_k)))
        if not hits:
            return "", []

        def build(per_chunk: int) -> str:
            parts = [DOC_HEADER]
            for position, hit in enumerate(hits, 1):
                parts.append(f"[{position}] {window_around(hit.text, hit.matched, per_chunk)}")
            parts.append("")
            parts.append(DOC_INSTRUCTION)
            return "\n".join(parts)

        total = sum(len(hit.text) for hit in hits)
        if cost(build(total)) <= budget_tokens:
            return build(total), hits
        low, high, best = 0, total, ""
        while low <= high:
            middle = (low + high) // 2
            candidate = build(max(1, middle // len(hits)))
            if cost(candidate) <= budget_tokens:
                best, low = candidate, middle + 1
            else:
                high = middle - 1
        return (best, hits) if best else ("", hits)

    # -- introspection
    def stats(self) -> dict:
        lengths = [len(chunk.text) for chunk in self.chunks]
        return {"name": self.name, "chars": len(self.text), "chunks": len(self.chunks),
                "terms": len(self._df), "chunk_chars": self.chunk_chars,
                "overlap": self.overlap,
                "average_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0}

    def __len__(self) -> int:
        return len(self.chunks)


# ------------------------------------------------------------------ grounding
def grounding(answer: str, context: str, minimum: int = 3) -> float | None:
    """How much of `answer`'s vocabulary also appears in `context` (0..1).

    A cheap honesty check, not a fact checker: when a 7 M-parameter model ignores
    the injected context and answers from its own priors, almost none of its
    content words appear in the document. `None` means "not enough words to
    judge" (yes/no replies, or a reply of two words) — a warning that fires on
    every short answer is a warning nobody reads.
    """
    words = [_stem(word) for word in tokenize(answer)]
    content = [word for word in words if word not in STOPWORDS and len(word) >= 3]
    if len(content) < minimum:
        return None
    present = {_stem(word) for word in tokenize(context)}
    return sum(1 for word in content if word in present) / len(content)
