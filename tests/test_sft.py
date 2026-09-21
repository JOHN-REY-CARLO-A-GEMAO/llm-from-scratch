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

    def test_dataset_and_checkpoint_are_loadable(self):
        path = ROOT / "data" / "sft" / "chat_sft.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 6000)
        row = json.loads(lines[0])
        self.assertEqual(row["text"],
            f"{sft.IM_START}user\n{row['prompt']}{sft.IM_END}\n"
            f"{sft.IM_START}assistant\n{row['response']}{sft.IM_END}")
        cfg = model_mod.GPTConfig.load(ROOT / "out" / "model_chat_config.json")
        state = torch.load(ROOT / "out" / "model_chat.pt", map_location="cpu", weights_only=True)
        self.assertEqual(cfg.vocab_size, self.chat.vocab)
        self.assertEqual(tuple(state["wte.weight"].shape), (cfg.vocab_size, cfg.d_model))


if __name__ == "__main__":
    unittest.main()
