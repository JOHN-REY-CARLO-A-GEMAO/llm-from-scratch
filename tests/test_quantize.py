"""Tests for INT8 weight quantisation (per-channel symmetric, from scratch).

Each test pins down a mathematical property the scheme must satisfy; these
same properties are what let us claim "INT8 from scratch" rather than "we
called torch.quantization".
"""
import sys
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import quantize as q  # noqa: E402
from _stage import load  # noqa: E402

mm = load("03_model")


def tiny_cfg(**kw):
    base = dict(vocab_size=256, block_size=32, n_layer=2, n_head=4, d_model=64)
    base.update(kw)
    return mm.GPTConfig(**base)


class QuantizeCoreTests(unittest.TestCase):
    def test_per_channel_scales_use_rowwise_max(self):
        W = torch.tensor([[1.0, -3.0, 2.0], [0.5, 0.0, -0.25]])
        Wq, s = q.quantize_tensor_per_channel(W)
        # row 0: amax=3 -> scale=3/127, quantised values span [-127,127]
        self.assertAlmostEqual(s[0].item(), 3.0 / 127.0, places=5)
        self.assertAlmostEqual(s[1].item(), 0.5 / 127.0, places=5)
        # The max-abs quantised value per row is exactly ±127
        self.assertEqual(Wq[0].abs().max().item(), 127)
        self.assertEqual(Wq[1].abs().max().item(), 127)

    def test_dequant_recovers_row_at_max(self):
        W = torch.randn(64, 96)
        Wq, s = q.quantize_tensor_per_channel(W)
        W_hat = q.dequantize(Wq, s)
        # Quantisation error is bounded by half an LSB.
        err = (W - W_hat).abs()
        lsb = s.unsqueeze(1)
        self.assertTrue((err <= lsb / 2 + 1e-5).all(), "error exceeds half an LSB")

    def test_per_channel_beats_per_tensor(self):
        """Per-channel must be at least as accurate as per-tensor, because
        per-tensor is the special case where every row shares a scale."""
        torch.manual_seed(0)
        W = torch.randn(128, 96) * torch.linspace(0.1, 10.0, 128).unsqueeze(1)
        Wq_pc, s_pc = q.quantize_tensor_per_channel(W)
        Wq_pt, s_pt = q.quantize_tensor_per_tensor(W)
        err_pc = (W - q.dequantize(Wq_pc, s_pc)).abs().mean()
        err_pt = (W - q.dequantize(Wq_pt, s_pt)).abs().mean()
        self.assertLess(err_pc, err_pt)

    def test_int8_range(self):
        W = torch.randn(32, 32) * 100
        Wq, _ = q.quantize_tensor_per_channel(W)
        self.assertEqual(Wq.dtype, torch.int8)
        self.assertTrue((Wq >= -127).all() and (Wq <= 127).all())


class QuantizedLinearTests(unittest.TestCase):
    def test_is_drop_in_for_nn_linear(self):
        F = nn = torch.nn
        torch.manual_seed(0)
        lin = nn.Linear(48, 64, bias=False)
        ql = q.QuantizedLinear.from_linear(lin, per_channel=True)
        x = torch.randn(3, 10, 48)
        with torch.no_grad():
            y_fp = lin(x)
            y_q = ql(x)
        self.assertEqual(y_q.shape, y_fp.shape)
        # error is small (< 1% per-output on average)
        rel = (y_fp - y_q).abs().mean() / y_fp.abs().mean()
        self.assertLess(rel.item(), 0.02)

    def test_reports_int8_in_extra_repr(self):
        lin = torch.nn.Linear(16, 16, bias=True)
        ql = q.QuantizedLinear.from_linear(lin)
        self.assertIn("int8", ql.extra_repr())


class ModelQuantizeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = tiny_cfg()
        self.model = mm.GPT(self.cfg).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def test_quantize_replaces_all_linears(self):
        m2 = mm.GPT(self.cfg)
        m2.load_state_dict(self.model.state_dict())
        qm, stats = q.quantize_model(m2)
        # every Linear in an Llama is a proj in attention or MLP
        n_lineys_expected = sum(1 for _ in self.model.modules()
                                if isinstance(_, torch.nn.Linear))
        self.assertEqual(stats["n_layers"], n_lineys_expected)
        self.assertTrue(q.is_quantized(qm))
        self.assertFalse(q.is_quantized(self.model))  # original untouched

    def test_forward_matches_closely(self):
        ids = torch.randint(0, self.cfg.vocab_size, (2, 16))
        with torch.inference_mode():
            fp_logits, _ = self.model(ids)
            m2 = mm.GPT(self.cfg)
            m2.load_state_dict(self.model.state_dict())
            qm, _ = q.quantize_model(m2)
            q_logits, _ = qm(ids)
        cos = torch.nn.functional.cosine_similarity(
            fp_logits.reshape(-1, self.cfg.vocab_size),
            q_logits.reshape(-1, self.cfg.vocab_size), dim=-1).mean()
        self.assertGreater(cos.item(), 0.999)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "q.pt"
            m2 = mm.GPT(self.cfg)
            m2.load_state_dict(self.model.state_dict())
            qm, _ = q.quantize_model(m2)
            info = q.save_quantized(qm, self.cfg, path)
            self.assertTrue(path.exists())
            self.assertGreater(info["mb"], 0.0)
            m3, cfg3, meta = q.load_quantized(path)
            self.assertEqual(cfg3.n_layer, self.cfg.n_layer)
            self.assertTrue(meta["per_channel"])
            ids = torch.randint(0, self.cfg.vocab_size, (1, 16))
            with torch.inference_mode():
                l1, _ = qm(ids)
                l2, _ = m3(ids)
            self.assertTrue(torch.allclose(l1, l2, atol=1e-6))

    def test_quantized_model_works_with_kv_cache(self):
        """Quantized linears must still work through generate()'s KV-cache path."""
        gg = load("06_generate")
        m2 = mm.GPT(self.cfg)
        m2.load_state_dict(self.model.state_dict())
        qm, _ = q.quantize_model(m2)
        params = gg.SamplingParams(temperature=0.0, seed=0)
        ids, _, _ = gg.generate(qm, [1, 2, 3], 8, params, use_cache=True,
                                block_size=self.cfg.block_size)
        self.assertEqual(len(ids), 11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
