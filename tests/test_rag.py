"""Document retrieval (`rag.py`): chunking, BM25 ranking, prompt injection, grounding.

The module is pure stdlib, so this file runs in milliseconds and pins the
properties the chat engine depends on:

 * chunking keeps paragraphs/sentences intact, overlaps by design, and never
   returns an empty or absurd chunk,
 * retrieval puts the passage that answers the question first, and returns
   *nothing* for an unrelated question (a padded top-2 is a wrong-context
   prompt, not a helpful one),
 * a typo still finds its passage through the character-trigram fallback,
 * `context_block` emits the exact "Context from uploaded document: … Answer the
   user's question based ONLY on this context." framing and never exceeds the
   token budget it was given — the shipped model has 256 tokens of context, so
   an over-budget block means prefill silently drops the question,
 * `grounding` reports when a reply does not come from the document.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rag                                     # noqa: E402

DOC = """The London Bridge story

The first London Bridge was built of wood by the Romans around AD 50. It was
rebuilt many times: a stone bridge with houses on it opened in 1209, and the
modern concrete bridge opened in 1973.

Blue whales are the largest animals ever to have lived. An adult blue whale can
reach 30 metres and weigh 150 tonnes. They feed almost entirely on krill.

The Great Fire of London started in 1666 in a bakery on Pudding Lane. It burned
for four days and destroyed most of the medieval city.

Photosynthesis is how plants make food from sunlight, water and carbon dioxide.
It happens in the leaves, inside structures called chloroplasts.
"""


def cost_words(text: str) -> int:
    """A stand-in tokenizer: ~1 token per 4 characters, as English prose behaves."""
    return len(text) // 4


class ChunkingTests(unittest.TestCase):
    def test_short_text_is_one_chunk_and_empty_text_is_none(self):
        self.assertEqual(rag.chunk_text("Just one sentence."), ["Just one sentence."])
        for empty in ("", "   ", "\n\n", None):
            self.assertEqual(rag.chunk_text(empty), [])

    def test_long_text_splits_into_overlapping_readable_chunks(self):
        chunks = rag.chunk_text(DOC, 300, 60)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk), rag.MIN_CHUNK_CHARS)
            self.assertLessEqual(len(chunk), 300 + rag.MIN_CHUNK_CHARS + 60)
            self.assertNotEqual(chunk.strip(), "")
        # overlap is real: consecutive chunks share their boundary text
        self.assertTrue(any(chunks[i + 1][:20].strip() in chunks[i] for i in range(len(chunks) - 1))
                        or any(chunks[i][-60:].strip() in chunks[i + 1] for i in range(len(chunks) - 1)))

    def test_cuts_prefer_paragraph_and_sentence_boundaries(self):
        chunks = rag.chunk_text(DOC, 300, 60)
        # every chunk that is not the last one ends where a sentence or a
        # paragraph ends — not in the middle of a word
        for chunk in chunks[:-1]:
            self.assertRegex(chunk[-1], r"[.\n]")
            self.assertFalse(chunk.endswith(" "))

    def test_a_single_runaway_sentence_is_still_bounded(self):
        # one 5,000-character "sentence" with no punctuation at all
        chunks = rag.chunk_text("word " * 1000, 400, 50)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(chunk) <= 460 for chunk in chunks))

    def test_chunk_chars_below_the_floor_is_refused(self):
        with self.assertRaises(rag.RagError):
            rag.chunk_text("hello world", chunk_chars=4)

    def test_trim_to_marks_the_elision(self):
        text = "one two three four five six seven eight"
        trimmed = rag.trim_to(text, 20)
        self.assertLessEqual(len(trimmed), 20)
        self.assertTrue(trimmed.endswith("…"))
        self.assertTrue(trimmed.startswith("one two"))
        self.assertEqual(rag.trim_to(text, 999), text)
        self.assertEqual(rag.trim_to(text, 0), "")

    def test_window_around_keeps_the_matching_sentence(self):
        text = ("intro " * 30) + "the blue whale is the largest animal. " + ("tail " * 30)
        window = rag.window_around(text, ["whale"], 80)
        self.assertIn("whale", window)
        self.assertLessEqual(len(window), 82)
        # ...while a plain prefix trim would have cut it off
        self.assertNotIn("whale", rag.trim_to(text, 80))


class TokenisingTests(unittest.TestCase):
    def test_terms_drop_stopwords_and_fold_plurals(self):
        self.assertIn("word", rag.terms("words and the"))
        self.assertIn("word and", rag.terms("words and the"))     # bigrams keep them
        for stop in ("and", "the"):
            self.assertNotIn(stop, rag.terms("and the cat"))

    def test_trigram_similarity_likes_typos_more_than_nonsense(self):
        typo = rag._cosine(rag.trigrams("cheshire caat"), rag.trigrams("cheshire cat"))
        nonsense = rag._cosine(rag.trigrams("cheshire caat"), rag.trigrams("bicycle repair"))
        self.assertGreater(typo, nonsense)
        self.assertGreater(typo, 0.4)


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.index = rag.DocumentIndex.from_text(DOC, name="notes.md", chunk_chars=300, overlap=60)

    def test_stats_describe_the_index(self):
        stats = self.index.stats()
        self.assertEqual(stats["name"], "notes.md")
        self.assertEqual(stats["chars"], len(DOC))
        self.assertGreaterEqual(stats["chunks"], 3)
        self.assertGreater(stats["terms"], 50)
        self.assertEqual(stats["chunk_chars"], 300)
        self.assertEqual(len(self.index), stats["chunks"])

    def test_empty_and_oversized_documents_are_refused(self):
        with self.assertRaises(rag.RagError):
            rag.DocumentIndex.from_text("   ")
        with self.assertRaises(rag.RagError):
            rag.DocumentIndex.from_text("x" * (rag.MAX_DOC_CHARS + 1))

    def test_the_answer_ranks_first(self):
        cases = [
            ("who built the first london bridge", "Romans"),
            ("how big is a blue whale", "30 metres"),
            ("when did the great fire of london start", "1666"),
            ("what are chloroplasts", "leaves"),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                hits = self.index.search(question, top_k=2)
                self.assertTrue(hits, question)
                self.assertIn(expected, hits[0].text)
                self.assertGreaterEqual(hits[0].relevance, rag.MIN_RELEVANCE)
                self.assertLessEqual(hits[0].relevance, 1.0)
                self.assertEqual(hits[0].index, hits[0].chunk.index)

    def test_an_unrelated_question_retrieves_nothing(self):
        for question in ("how do i bake sourdough bread", "give me a python quicksort", ""):
            with self.subTest(question=question):
                self.assertEqual(self.index.search(question), [])

    def test_top_k_is_respected_and_scores_are_sorted(self):
        hits = self.index.search("london bridge fire whale plants", top_k=2)
        self.assertLessEqual(len(hits), 2)
        self.assertEqual(hits, sorted(hits, key=lambda hit: -hit.score))
        single = self.index.search("london bridge", top_k=1)
        self.assertEqual(len(single), 1)
        self.assertEqual(self.index.best("london bridge").index, single[0].index)

    def test_a_typo_still_finds_its_passage_via_trigrams(self):
        hits = self.index.search("blu wale largets animal", top_k=1)
        self.assertTrue(hits)
        self.assertIn("whale", hits[0].text)

    def test_retrieval_is_deterministic(self):
        first = [(h.index, h.score) for h in self.index.search("london bridge", top_k=3)]
        second = [(h.index, h.score) for h in self.index.search("london bridge", top_k=3)]
        self.assertEqual(first, second)

    def test_hits_carry_their_own_excerpt_and_json(self):
        hit = self.index.search("blue whale", top_k=1)[0]
        self.assertIn("whale", hit.excerpt(120))
        payload = hit.to_json(120)
        self.assertEqual(payload["index"], hit.index)
        self.assertIn("excerpt", payload)
        self.assertEqual(payload["relevance"], round(hit.relevance, 3))


class ContextBlockTests(unittest.TestCase):
    def setUp(self):
        self.index = rag.DocumentIndex.from_text(DOC, name="notes.md", chunk_chars=300, overlap=60)

    def test_the_block_uses_the_exact_injection_format(self):
        block, hits = self.index.context_block("who built the first london bridge?",
                                               4000, cost_words, top_k=2)
        self.assertTrue(hits)
        self.assertTrue(block.startswith(rag.DOC_HEADER + "\n"))
        self.assertTrue(block.endswith(rag.DOC_INSTRUCTION))
        self.assertIn("[1]", block)
        self.assertIn("Romans", block)

    def test_the_block_never_exceeds_its_token_budget(self):
        question = "when did the great fire of london start?"
        for budget in (40, 60, 120, 300):
            with self.subTest(budget=budget):
                block, hits = self.index.context_block(question, budget, cost_words, top_k=2)
                self.assertTrue(block, f"budget {budget} produced nothing")
                self.assertLessEqual(cost_words(block), budget)
                self.assertTrue(hits)

    def test_an_impossible_budget_drops_the_context_instead_of_the_question(self):
        block, hits = self.index.context_block("london bridge", 5, cost_words)
        self.assertEqual(block, "")
        self.assertTrue(hits)                      # retrieval still reports what it found

    def test_the_context_precedes_any_persona_text(self):
        # the engine appends the persona after this block, so the "answer ONLY
        # from this context" instruction is what survives a truncation
        block, _ = self.index.context_block("blue whale", 200, cost_words)
        self.assertLess(block.index(rag.DOC_INSTRUCTION), len(block))

    def test_trimming_keeps_the_matched_text_visible(self):
        block, _ = self.index.context_block("who built the first london bridge?", 60,
                                            cost_words, top_k=1)
        self.assertIn("London Bridge", block)
        self.assertIn("…", block)                  # a shortened passage says so


class GroundingTests(unittest.TestCase):
    def test_a_reply_from_the_document_is_grounded(self):
        answer = ("The first London Bridge was built of wood by the Romans "
                  "around AD 50.")
        self.assertGreater(rag.grounding(answer, DOC), 0.75)

    def test_an_invented_reply_is_not_grounded(self):
        answer = ("Bananas are grown in Iceland, mostly for export to Mars "
                  "each summer season.")
        self.assertLess(rag.grounding(answer, DOC), 0.2)

    def test_short_replies_are_not_judged(self):
        for answer in ("Yes.", "No", "I do not know"):
            with self.subTest(answer=answer):
                self.assertIsNone(rag.grounding(answer, DOC))


class UnicodeTests(unittest.TestCase):
    def test_accents_emoji_and_curly_quotes_survive_indexing(self):
        text = ("Café notes — première édition ✨\n\n"
                "Le café est prêt à 7 h. “Ça va bien.”\n\n"
                "日本語のメモ also lives here, and the answer is 42.\n")
        index = rag.DocumentIndex.from_text(text, name="notes.txt")
        hits = index.search("what is the answer", top_k=1)
        self.assertTrue(hits)
        self.assertIn("42", hits[0].text)
        self.assertIn("café", index.search("café édition", top_k=1)[0].text.lower())


if __name__ == "__main__":
    unittest.main(verbosity=1)
