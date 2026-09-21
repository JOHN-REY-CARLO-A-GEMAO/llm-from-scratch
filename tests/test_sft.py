"""Contracts for chat tokenization, serialization, masking, and checkpoint shape."""
from pathlib import Path
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from _stage import load


tok_mod = load("01_tokenizer")
sft = load("08_sft")
model_mod = load("03_model")


class SFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tok_mod.ByteLevelBPE.load(ROOT / "tokenizer")
        cls.chat = tok_mod.ByteLevelBPE.load(ROOT / "tokenizer" / "chat")

    def test_added_chat_tokens_preserve_pretrained_ids(self):
        self.assertEqual(self.chat.vocab, self.base.vocab + 2)
        for token, old_id in self.base.token_to_id.items():
            self.assertEqual(self.chat.token_to_id[token], old_id)
        self.assertEqual(self.chat.token_to_id[sft.IM_START], self.base.vocab)
        self.assertEqual(self.chat.token_to_id[sft.IM_END], self.base.vocab + 1)

    def test_only_response_and_end_marker_are_targets(self):
        prompt, response = "What is a frog?", "A frog is an animal."
        x, labels = sft.encode_example(self.chat, prompt, response, 256)
        live = [v for v in labels if v != sft.IGNORE]
        expected = self.chat.encode(response) + [self.chat.token_to_id[sft.IM_END]]
        self.assertEqual(live, expected)
        self.assertGreater(labels.count(sft.IGNORE), 0)
        self.assertEqual(len(x), len(labels))

    def test_the_corpus_carries_traces_and_document_context(self):
        """The CoT/RAG features are *data* features: assert the data, not the hope."""
        import personas
        import rag
        import importlib.util
        path = ROOT / "data" / "sft" / "chat_sft.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(rows), 6000)
        # the persona slice has to survive the subset sampling above; if it thins
        # out, the model meets the UI's system prompt for the first time at
        # inference, which is exactly the failure the persona rows exist to fix
        self.assertGreater(sum(1 for r in rows if r["system"]
                               and not r["system"].startswith("Context")), 500)
        reasoned = [r for r in rows if r["reasoning"]]
        grounded = [r for r in rows if r["grounded"]]
        # every traced row opens *and closes* exactly one block, and the answer
        # follows it — the same rule personas.split_think() enforces at runtime
        for row in reasoned:
            thought = personas.split_think(row["response"])
            self.assertTrue(thought.has_reasoning)
            self.assertTrue(thought.closed, row["prompt"])
            self.assertTrue(thought.answer, row["prompt"])
        self.assertGreater(len(reasoned), 2000)
        # a slice of the corpus carries the *persona* system blocks the UI sends
        self.assertGreater(sum(1 for r in grounded if not r["system"].startswith(rag.DOC_HEADER)),
                           400)
        # ...and the document rows carry the exact injected template
        doc_rows = [r for r in grounded if r["system"].startswith(rag.DOC_HEADER)]
        self.assertGreater(len(doc_rows), 400)
        for row in doc_rows[:50]:
            self.assertIn(rag.DOC_INSTRUCTION, row["system"])
            self.assertIn("Answer the user's question based ONLY on this context.",
                          row["system"])
        # "the document does not say" is trained, not invented at the UI layer
        self.assertGreater(sum(1 for r in doc_rows if "does not say" in r["response"]), 50)
        self.assertGreater(sum(1 for r in doc_rows if "does not say" not in r["response"]), 200)
        # every row must fit the model's window: `encode_example` front-trims an
        # over-long row, and a trimmed row has lost part of its prompt or its
        # injected document (the maker checks the same invariant as it builds,
        # with this tokenizer — this is the assertion that keeps it honest)
        for row in rows:
            length = len(self.chat.encode_with_specials(
                row["text"], ["<|im_start|>", "<|im_end|>"]))
            self.assertLessEqual(length, 256, f"{length} tokens: {row['prompt']!r}")

        # `text` is always the serialization of prompt + response (+ system)
        spec = importlib.util.spec_from_file_location("mk", ROOT / "07_make_sft.py")
        maker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(maker)
        for row in rows[:80] + doc_rows[:20]:
            self.assertEqual(row["text"],
                             maker.format_chat(row["prompt"], row["response"], row["system"]))

    def test_a_numeric_tool_reply_is_supervised_like_any_other(self):
        """`<<calc: 12 + 4>>` is ordinary text for the tokenizer, and the marker's
        substitution happens at inference time — so the loss must cover it."""
        prompt, response = "What is 12 + 4?", "<think>\n1. Add them: <<calc: 12 + 4>>.\n</think>\n12 + 4 = 16."
        x, labels = sft.encode_example(self.chat, prompt, response, 256)
        live = [v for v in labels if v != sft.IGNORE]
        self.assertEqual(live, self.chat.encode(response) + [self.chat.token_to_id[sft.IM_END]])
        marked = self.chat.encode("<<calc: 12 + 4>>")
        self.assertTrue(any(v in marked for v in live))       # markers are ordinary tokens
        for token_id in marked:
            self.assertLess(token_id, self.base.vocab)        # ...no added tokens for them

    def test_dataset_and_checkpoint_are_loadable(self):
        path = ROOT / "data" / "sft" / "chat_sft.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 6000)
        # Row 0 is not pinned to a shape any more: the corpus is an evenly spaced
        # sample of a larger pool, so the first row is whatever that sample
        # starts with. The serialization contract is what matters, and it is
        # checked on the first row of each kind.
        row = next(json.loads(line) for line in lines if not json.loads(line)["system"])
        self.assertEqual(row["text"],
            f"{sft.IM_START}user\n{row['prompt']}{sft.IM_END}\n"
            f"{sft.IM_START}assistant\n{row['response']}{sft.IM_END}")
        system_row = next(json.loads(line) for line in lines if json.loads(line)["system"])
        self.assertTrue(system_row["text"].startswith(f"{sft.IM_START}system\n"))
        cfg = model_mod.GPTConfig.load(ROOT / "out" / "model_chat_config.json")
        state = torch.load(ROOT / "out" / "model_chat.pt", map_location="cpu", weights_only=True)
        self.assertEqual(cfg.vocab_size, self.chat.vocab)
        self.assertEqual(tuple(state["wte.weight"].shape), (cfg.vocab_size, cfg.d_model))


if __name__ == "__main__":
    unittest.main()
