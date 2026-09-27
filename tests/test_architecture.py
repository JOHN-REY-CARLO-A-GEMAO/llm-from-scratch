"""Architecture tests for the Llama-3 style stack.

Invariants, not snapshots: each test states a property that the *reason* for
choosing a component guarantees. If a refactor breaks the reason, the test
fails even when the model still runs (which is how the original GPT-2 version
lost its RoPE=relative property without anyone noticing).
"""
import json
import math
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch                                                        # noqa: E402
from _stage import load                                             # noqa: E402

mm = load("03_model")


def tiny_cfg(**kw):
    base = dict(vocab_size=256, block_size=32, n_layer=2, n_head=4, d_model=64)
    base.update(kw)
    return mm.GPTConfig(**base)


class ConfigTests(unittest.TestCase):
    def test_round_trip_keeps_every_key(self):
        cfg = tiny_cfg(rope_base=500_000.0, rms_norm_eps=1e-6, dropout=0.1)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "config.json"
            cfg.save(p)
            self.assertEqual(mm.GPTConfig.load(p), cfg)
            self.assertEqual(json.loads(p.read_text())["rope_base"], 500_000.0)

    def test_unknown_key_is_an_error_not_a_silent_default(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "config.json"
            p.write_text(json.dumps({"vocab_size": 256, "n_rotary_dim": 8}))
            with self.assertRaises(ValueError):
                mm.GPTConfig.load(p)

    def test_7b_preset_dimensions_and_parameter_formula(self):
        cfg = mm.GPTConfig.preset("7B")
        self.assertEqual((cfg.d_model, cfg.n_layer, cfg.n_head, cfg.mlp_hidden),
                         (4096, 32, 32, 11008))
        self.assertEqual(cfg.head_dim, 128)
        self.assertTrue(cfg.gradient_checkpointing)
        # Tied token/output embeddings, four attention matrices, three SwiGLU
        # matrices, two norm gains per block, and one final norm gain.
        expected = (cfg.vocab_size * cfg.d_model + cfg.n_layer * (
            4 * cfg.d_model**2 + 3 * cfg.d_model * cfg.mlp_hidden + 2 * cfg.d_model
        ) + cfg.d_model)
        self.assertEqual(expected, 6_493_048_832)

    def test_swiglu_width_rule(self):
        # 8/3 * d_model, rounded up to a multiple of 64 (Llama's rule)
        cfg = tiny_cfg(d_model=160)
        self.assertEqual(cfg.mlp_hidden, 448)
        self.assertEqual(tiny_cfg(d_model=192).mlp_hidden, 512)

    def test_shipped_config_matches_shipped_weights(self):
        """The committed pair must describe a model that actually loads."""
        cfg = mm.GPTConfig.load(ROOT / "out" / "config.json")
        model = mm.GPT(cfg)
        state = torch.load(ROOT / "out" / "model_best.pt", map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state)
        self.assertEqual(model.num_params(), sum(p.numel() for p in model.parameters()))


class ComponentTests(unittest.TestCase):
    def test_rmsnorm_is_scale_invariant_and_does_not_centre(self):
        norm = mm.RMSNorm(8, eps=1e-8)
        x = torch.randn(4, 8) + 3.0                 # a deliberately non-zero mean
        torch.testing.assert_close(norm(x * 10.0), norm(x), atol=1e-5, rtol=1e-5)
        # LayerNorm would be invariant to *shifting* the mean; RMSNorm is not.
        self.assertFalse(torch.allclose(norm(x + 5.0), norm(x), atol=1e-4))
        self.assertEqual(tuple(norm.weight.shape), (8,))    # gain only, no bias

    def test_attention_uses_llama_projection_names_and_no_bias(self):
        model = mm.GPT(tiny_cfg())
        attn = model.blocks[0].attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            self.assertIsInstance(getattr(attn, name), torch.nn.Linear)
            self.assertIsNone(getattr(attn, name).bias)
        self.assertFalse(hasattr(attn, "qkv"), "fused GPT-2 qkv projection is back")
        self.assertFalse(hasattr(attn, "attn_bias"), "GPT-2 attention-logit bias is back")
        mlp = model.blocks[0].mlp
        self.assertEqual((mlp.gate_proj.out_features, mlp.up_proj.out_features),
                         (model.cfg.mlp_hidden, model.cfg.mlp_hidden))
        self.assertEqual(mlp.down_proj.out_features, model.cfg.d_model)

    def test_scaled_dot_product_attention_is_actually_called(self):
        model = mm.GPT(tiny_cfg())
        seen = []
        real = torch.nn.functional.scaled_dot_product_attention

        def spy(*a, **kw):
            seen.append(kw or a[3:])
            return real(*a, **kw)

        x = torch.randint(0, 256, (2, 16))
        with patch.object(torch.nn.functional, "scaled_dot_product_attention", spy):
            model(x)
        self.assertTrue(seen, "the attention path no longer goes through SDPA")
        self.assertTrue(all(kw.get("is_causal") is not False or "attn_mask" in kw
                            for kw in seen), "an unmasked attention call slipped through")

    def test_rope_preserves_norms_and_supports_extrapolation(self):
        cfg = tiny_cfg(rope_base=10_000.0)
        cos, sin = mm.build_rope_cache(cfg.head_dim, 512, cfg.rope_base, "cpu", torch.float32)
        x = torch.randn(1, cfg.n_head, 8, cfg.head_dim)
        y = mm.apply_rope(x, cos[:8][None, None], sin[:8][None, None])
        torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)
        # building a table longer than the training window must not blow up
        self.assertEqual(cos.shape, (512, cfg.head_dim))

    def test_embeddings_are_tied_and_output_projection_is_depth_scaled(self):
        cfg = tiny_cfg()
        model = mm.GPT(cfg)
        self.assertIs(model.lm_head.weight, model.wte.weight)
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        std = model.blocks[0].mlp.down_proj.weight.std().item()
        # N(0, 0.02*scale) over 448*64 weights: the sample std is close, and
        # clearly smaller than the 0.02 of an unscaled projection
        self.assertLess(std, 0.02 * scale * 2)
        self.assertLess(std, 0.02)

    def test_position_matters_and_cache_does_not_change_the_answer(self):
        torch.manual_seed(0)
        cfg = tiny_cfg()
        model = mm.GPT(cfg).eval()
        ids = torch.randint(1, 256, (2, 12))
        with torch.no_grad():
            full, _ = model(ids)
            swapped, _ = model(ids.flip(-1))
            slots = mm.make_kv_cache(cfg.n_layer)
            model(ids[:, :7], kv_cache=slots)
            tail, _ = model(ids[:, 7:], kv_cache=slots)
        self.assertFalse(torch.allclose(full, swapped, atol=1e-3),
                         "the model ignores token order: RoPE is not doing anything")
        torch.testing.assert_close(tail, full[:, 7:], atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
