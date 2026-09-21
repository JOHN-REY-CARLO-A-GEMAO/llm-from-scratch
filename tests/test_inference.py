"""Regression tests for streaming, sampling, REPL lifetime and the CPU CLI."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import generate as cli
from _stage import load

gg = load("06_generate")
mm = load("03_model")
tm = load("01_tokenizer")
torch.set_num_threads(2)


class SamplingTests(unittest.TestCase):
    def test_nucleus_keeps_smallest_prefix_in_original_order(self):
        logits = torch.tensor([[[.1, .6, .05, .25]]]).log()
        for p, expected in [(.01, [1]), (.7, [1, 3]), (.9, [0, 1, 3]), (1, [0, 1, 2, 3])]:
            pr = gg.logits_to_probs(logits, gg.SamplingParams(
                temperature=1, top_k=0, top_p=p, repeat_penalty=1))
            self.assertEqual(torch.where(pr[0] > 0)[0].tolist(), expected)
            self.assertAlmostEqual(pr.sum().item(), 1, places=6)

    def test_exact_nucleus_boundary_and_tied_top_k(self):
        logits = torch.zeros(1, 1, 4)
        pr = gg.logits_to_probs(logits, gg.SamplingParams(
            temperature=1, top_k=0, top_p=.5, repeat_penalty=1))
        self.assertEqual((pr > 0).sum().item(), 2)
        pr = gg.logits_to_probs(logits, gg.SamplingParams(
            temperature=1, top_k=1, top_p=1, repeat_penalty=1))
        self.assertEqual((pr > 0).sum().item(), 1)

    def test_top_k_and_top_p_compose(self):
        logits = torch.tensor([[[.6, .25, .1, .05]]]).log()
        pr = gg.logits_to_probs(logits, gg.SamplingParams(
            temperature=1, top_k=2, top_p=.7, repeat_penalty=1))
        self.assertEqual(torch.where(pr[0] > 0)[0].tolist(), [0])
        pr = gg.logits_to_probs(logits, gg.SamplingParams(
            temperature=1, top_k=100, top_p=1, repeat_penalty=1))
        torch.testing.assert_close(pr, logits.exp()[:, -1])

    def test_sign_aware_penalty_once_per_unique_token(self):
        logits = torch.tensor([[[4., -2., 0., 3.]], [[4., -2., 0., 3.]]])
        original = logits.clone()
        result = gg.logits_to_probs(logits, gg.SamplingParams(
            temperature=0, repeat_penalty=2), torch.tensor([[0, 0, 1], [3, 3, 2]]))
        torch.testing.assert_close(result, torch.tensor([[2., -4., 0., 3.], [4., -2., 0., 1.5]]))
        torch.testing.assert_close(logits, original)
        long_history = torch.zeros((1, 2000), dtype=torch.long)
        self.assertTrue(torch.isfinite(gg.logits_to_probs(logits[:1], gg.SamplingParams(), long_history)).all())

    def test_validation(self):
        for kwargs in [dict(top_k=-1), dict(top_p=0), dict(top_p=1.1),
                       dict(top_p=float('nan')), dict(temperature=-1),
                       dict(temperature=float('inf')), dict(repeat_penalty=.9),
                       dict(repeat_penalty=float('nan')), dict(seed=2**64)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gg.SamplingParams(**kwargs)

    def test_stream_matches_buffered_and_stops_without_extra_forward(self):
        torch.manual_seed(3)
        model = mm.GPT(mm.GPTConfig(vocab_size=256, block_size=12, n_layer=1, n_head=2, d_model=16))
        params = gg.SamplingParams(seed=4294967295)
        for cached in (False, True):
            seen = []
            with patch.object(model, 'forward', wraps=model.forward) as forward:
                def callback(token):
                    seen.append(token)
                    self.assertEqual(forward.call_count, len(seen))
                ids, _, total = gg.generate(model, [1, 2], 5, params, use_cache=cached, on_token=callback)
            buffered, _, _ = gg.generate(model, [1, 2], 5, params, use_cache=cached)
            self.assertEqual(ids, buffered)
            self.assertEqual(seen, ids[2:])
            self.assertEqual(total, 7)
            eos_seen = []
            with patch.object(model, 'forward', wraps=model.forward) as forward:
                eos, _, _ = gg.generate(model, [1, 2], 5, params, use_cache=cached,
                                         eos_id=seen[0], on_token=eos_seen.append)
                self.assertEqual(forward.call_count, 1)
            self.assertEqual(eos[2:], eos_seen)
            self.assertEqual(len(eos_seen), 1)
            seen.clear()
            long, _, total = gg.generate(model, [1] * 20, 100, params, use_cache=cached, on_token=seen.append)
            self.assertEqual(len(long), 21)
            self.assertEqual(total, 12)
            self.assertEqual(len(seen), 1)
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            self.assertEqual(gg.generate(model, [1, 2], 0, params)[0], [1, 2])
            forward.assert_not_called()


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tm.ByteLevelBPE.load(ROOT / 'tokenizer')

    def test_incremental_unicode_and_specials_match_render(self):
        tok = self.tok
        raw_ids = lambda b: [tok.token_to_id[tm.BYTE_TO_UNICODE[v]] for v in b]
        special = tok.token_to_id[tok.special_tokens[0]]
        # Split valid multibyte characters, malformed bytes, and incomplete tails.
        ids = raw_ids('Hello “世界” 🙂'.encode()) + [special, special] + raw_ids(b'\xff\xe2')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            stream = cli.TokenStream(tok, tm.unicode_to_bytes)
            for token in ids:
                stream(token)
            stream.finish()
        self.assertEqual(output.getvalue(), cli.render(tok, ids)[0])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(output, 'flush', wraps=output.flush) as flush:
            stream = cli.TokenStream(tok, tm.unicode_to_bytes)
            tokens = raw_ids('€'.encode())
            stream(tokens[0])
            self.assertEqual(output.getvalue(), '')
            stream(tokens[1])
            self.assertEqual(output.getvalue(), '')
            stream(tokens[2])
            self.assertEqual(output.getvalue(), '€')
            self.assertEqual(flush.call_count, 3)
            stream.finish()

    def test_repl_loads_once_and_reuses_weights(self):
        # Real checkpoint and sampler; spy on weight/tokenizer loads across two prompts.
        args = ['generate.py', '--interactive', '--quiet', '--n', '4', '--threads', '2', '--seed', '4294967295']
        output = io.StringIO()
        with patch.object(sys, 'argv', args), contextlib.redirect_stdout(output), \
             patch('builtins.input', side_effect=['', 'Once upon a time', 'Once upon a time', '/exit']), \
             patch.object(torch, 'load', wraps=torch.load) as weights, \
             patch.object(tm.ByteLevelBPE, 'load', wraps=tm.ByteLevelBPE.load) as tokenizer:
            cli.main()
        self.assertEqual(weights.call_count, 1)
        self.assertEqual(tokenizer.call_count, 1)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], lines[1])
        self.assertTrue(lines[0].startswith('Once upon a time'))

    def test_repl_exits_cleanly(self):
        from types import SimpleNamespace
        args = SimpleNamespace(quiet=True)
        for end in [EOFError, KeyboardInterrupt, '/quit', 'exit', 'QUIT']:
            with patch('builtins.input', side_effect=[end]), contextlib.redirect_stdout(io.StringIO()):
                cli.repl(args, None, None, None, None, None, None)
        with patch('builtins.input', return_value='hello'), \
             patch.object(cli, 'run_prompt', side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(io.StringIO()):
            cli.repl(args, None, None, None, None, None, None)

    def test_cli_quiet_empty_prompt_zero_tokens_and_large_seed(self):
        base = [sys.executable, str(ROOT / 'generate.py'), '--threads', '2', '--quiet', '--seed', '4294967295']
        # Launch outside the repo to exercise path resolution too.
        for prompt in ['', 'Hello 🙂']:
            result = subprocess.run(base + ['--prompt', prompt, '--n', '0'], cwd=ROOT.parent,
                                    capture_output=True, text=True, check=True)
            ids = self.tok.encode(prompt) if prompt else [self.tok.token_to_id[self.tok.special_tokens[0]]]
            self.assertEqual(result.stdout, cli.render(self.tok, ids)[0] + '\n')
        result = subprocess.run(base + ['--top-p', '0'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('top_p', result.stderr)

    def test_powershell_launcher_contract(self):
        source = (ROOT / 'run.ps1').read_text()
        self.assertRegex(source, r'\[long\]\s*\$Seed')
        self.assertIn('[switch]$Interactive', source)
        self.assertIn('if ($Interactive) { $genArgs += "--interactive" }', source)
        self.assertIn('"--repeat-penalty", $RepeatPenalty', source)


if __name__ == '__main__':
    unittest.main()
