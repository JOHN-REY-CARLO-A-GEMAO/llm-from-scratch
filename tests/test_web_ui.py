"""Web UI tests: stream decoder, chat serialization, prompt trimming,
engine streaming/cancellation, and the stdlib HTTP server end-to-end.

Loads the real committed checkpoint (out/model_chat.pt, ~8 MB) like every
other test in this repo; a greedy reply of a few tokens costs well under a
second on CPU. gradio coverage is skipped automatically when it is not
installed — the web UI itself falls back the same way.
"""
import json
import threading
import types
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch                                    # noqa: E402
import web_ui                                   # noqa: E402
from _stage import load                         # noqa: E402

mm = load("03_model")
tm = load("01_tokenizer")
torch.set_num_threads(2)


def make_engine(seed=None):
    args = types.SimpleNamespace(temp=0.8, top_p=0.9, top_k=40, repeat_penalty=1.15,
                                 seed=seed, n=24, host="127.0.0.1", port=0,
                                 no_browser=True)
    cfg = mm.GPTConfig.load(ROOT / "out/model_chat_config.json")
    model = mm.GPT(cfg)
    model.load_state_dict(torch.load(ROOT / "out/model_chat.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()
    tok = tm.ByteLevelBPE.load(ROOT / "tokenizer/chat")
    engine = web_ui.ChatEngine(args, torch, model, cfg, tok, tm, load("06_generate"),
                               ROOT / "out/model_chat.pt", ROOT / "out/model_chat_config.json",
                               ROOT / "tokenizer/chat")
    engine.device = torch.device("cpu")
    return engine


class IncrementalTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tm.ByteLevelBPE.load(ROOT / "tokenizer/chat")

    def stream(self, ids):
        dec = web_ui.IncrementalText(self.tok, tm.unicode_to_bytes)
        return "".join(dec.feed(i) for i in ids) + dec.finish()

    def test_multibyte_characters_survive_token_boundaries(self):
        for text in ["héllo wörld", "emoji ✨ test", "“curly quotes” — dash", "日本語"]:
            ids = self.tok.encode(text)
            # a single decoder instance, one byte-fragment token at a time:
            # the worst case for a byte splitter, and exactly how the engine feeds it
            self.assertEqual(self.stream(ids), text)

    def test_chat_markers_are_never_displayed(self):
        a, b = self.tok.token_to_id["<|im_start|>"], self.tok.token_to_id["<|im_end|>"]
        ids = [a] + self.tok.encode("hello") + [b]
        self.assertEqual(self.stream(ids), "hello")


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = make_engine()

    def test_serialize_matches_generate_dot_py_template(self):
        got = self.engine.serialize([("hi", "hello there")], "how are you?")
        self.assertEqual(got,
                         "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
                         "hello there<|im_end|>\n"
                         "<|im_start|>user\nhow are you?<|im_end|>\n<|im_start|>assistant\n")

    def test_build_prompt_accepts_dicts_and_pairs_and_trims_to_context(self):
        long_turn = ("word " * 120).strip()          # ~120 tokens per message
        history = [{"user": long_turn, "assistant": long_turn} for _ in range(3)]
        history += [(long_turn, long_turn)]          # gradio sends (u, a) pairs
        history.append({"user": "ok", "assistant": "ok"})
        history.append({"user": 123})                # malformed: dropped, not fatal
        user, turns, ids, dropped = self.engine.build_prompt("hi", history, 24)
        self.assertEqual(user, "hi")
        self.assertGreaterEqual(dropped, 1)
        self.assertLessEqual(len(ids), self.engine.cfg.block_size - 24)
        self.assertEqual(turns[-1], ("ok", "ok"))
        # short histories keep every turn — both shapes land in the prompt
        _, turns, _, dropped = self.engine.build_prompt(
            "hi", [{"user": "a", "assistant": "b"}, ("c", "d")], 24)
        self.assertEqual((turns, dropped), ([("a", "b"), ("c", "d")], 0))
        # the trimmed prompt still ends exactly at the current assistant turn
        self.assertTrue(self.engine.tok.decode(ids).endswith(
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"))

    def test_stream_chat_streams_then_reports_done(self):
        events = list(self.engine.stream_chat("Hi", [], temperature=0.0, top_p=0.9))
        kinds = [k for k, _ in events]
        self.assertEqual(kinds[-1], "done")
        self.assertGreater(kinds.count("token"), 0)
        tokens = "".join(d for k, d in events if k == "token")
        done = dict(events[-1][1])
        self.assertEqual(done["reply"], tokens.strip())
        self.assertEqual(done["temperature"], 0.0)
        self.assertIn(done["stop_reason"], ("stop", "length", "context", "stopped"))
        self.assertGreater(done["n_tokens"], 0)

    def test_stream_chat_rejects_empty_message(self):
        events = list(self.engine.stream_chat("   ", [], 0.8, 0.9))
        self.assertEqual(events, [("error", "message is empty")])

    def test_inputs_are_clamped(self):
        done = [d for k, d in self.engine.stream_chat("Hi", [], 99.0, -3.0) if k == "done"][0]
        self.assertEqual(done["temperature"], 1.5)
        self.assertEqual(done["top_p"], 0.01)

    def test_seed_makes_greedy_replies_reproducible(self):
        seeded = make_engine(seed=1234)
        r1 = [d["reply"] for k, d in seeded.stream_chat("Hello", [], 0.8, 0.9) if k == "done"][0]
        r2 = [d["reply"] for k, d in seeded.stream_chat("Hello", [], 0.8, 0.9) if k == "done"][0]
        self.assertEqual(r1, r2)

    def test_stop_cancels_between_tokens(self):
        gen = self.engine.stream_chat("Tell me a long story about a dragon.", [],
                                      temperature=1.0, top_p=1.0, max_tokens=255)
        first = next(gen)                            # one token is out the door
        self.assertEqual(first[0], "token")
        stopped = self.engine.stop_active()
        events = [first] + list(gen)
        done = dict(events[-1][1])
        if stopped:                                  # cancel landed before generation ended
            self.assertEqual(done["stop_reason"], "stopped")
        else:                                        # model finished first; still a clean end
            self.assertEqual(done["stop_reason"], "stop")
        self.assertLess(done.get("n_tokens", 0), 255)


class HttpServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine = make_engine()
        args = types.SimpleNamespace(temp=0.8, top_p=0.9, no_browser=True)
        web_ui.Handler.engine = engine
        web_ui.Handler.index_html = (web_ui.INDEX_HTML
                                     .replace("__APP_NAME__", web_ui.APP_NAME)
                                     .replace("__TEMP__", f"{args.temp:g}")
                                     .replace("__TOPP__", f"{args.top_p:g}")).encode("utf-8")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.Handler)
        cls.server.daemon_threads = True
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def ndjson_chat(self, payload):
        req = urllib.request.Request(self.url("/api/chat"), method="POST",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as res:
            self.assertEqual(res.headers["Content-Type"],
                             "application/x-ndjson; charset=utf-8")
            return [json.loads(line) for line in res.read().decode().splitlines() if line]

    def test_index_and_info(self):
        with urllib.request.urlopen(self.url("/"), timeout=10) as res:
            html = res.read().decode()
        self.assertIn(web_ui.APP_NAME, html)
        self.assertIn("/api/chat", html)
        self.assertNotIn("__TEMP__", html)           # every placeholder was filled
        self.assertIn('min="0.1" max="1.5"', html)   # temperature slider range per spec
        self.assertIn('min="0.1" max="1.0"', html)   # top-p slider range per spec
        self.assertIn("Clear Chat", html)
        with urllib.request.urlopen(self.url("/api/info"), timeout=10) as res:
            info = json.loads(res.read())
        self.assertEqual(info["params"], 1_926_880)
        self.assertEqual(info["context"], 256)
        self.assertTrue(info["checkpoint"].replace("\\", "/").endswith("out/model_chat.pt"))
        with urllib.request.urlopen(self.url("/healthz"), timeout=10) as res:
            self.assertEqual(json.loads(res.read()), {"ok": True})

    def test_chat_endpoint_streams_ndjson(self):
        events = self.ndjson_chat({"message": "Hi", "temperature": 0.0, "top_p": 0.9,
                                   "history": []})
        self.assertEqual(events[-1]["type"], "done")
        tokens = "".join(e["text"] for e in events if e["type"] == "token")
        self.assertEqual(events[-1]["reply"], tokens.strip())
        self.assertGreater(events[-1]["n_tokens"], 0)

    def test_chat_endpoint_with_history(self):
        history = [{"user": "Hi", "assistant": "Hello! How can I help?"}]
        events = self.ndjson_chat({"message": "Tell me a story.", "temperature": 0.0,
                                   "top_p": 0.9, "history": history})
        self.assertEqual(events[-1]["history_turns"], 1)

    def test_bad_requests_are_errors_not_crashes(self):
        events = self.ndjson_chat({"message": "", "temperature": 0.8, "top_p": 0.9})
        self.assertEqual(events[-1]["type"], "error")
        req = urllib.request.Request(self.url("/api/chat"), method="POST",
                                     data=b"not-json",
                                     headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 400)
        req = urllib.request.Request(self.url("/nope"), method="POST", data=b"{}")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 404)

    def test_stop_endpoint(self):
        req = urllib.request.Request(self.url("/api/stop"), method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=10) as res:
            self.assertIn(json.loads(res.read())["stopped"], (True, False))


class GradioTests(unittest.TestCase):
    def test_history_conversion(self):
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "dangling"}]      # never paired: dropped
        self.assertEqual(web_ui.gradio_history_to_pairs(msgs), [("hi", "hello")])
        self.assertEqual(web_ui.gradio_history_to_pairs([("a", "b"), ("c", "d")]),
                         [("a", "b"), ("c", "d")])
        self.assertEqual(web_ui.gradio_history_to_pairs(None), [])

    def test_app_builds_when_gradio_installed(self):
        try:
            import gradio as gr                          # noqa: F401
        except ImportError:
            self.skipTest("gradio not installed (the web UI falls back to the stdlib UI)")
        engine = make_engine()
        args = types.SimpleNamespace(temp=0.8, top_p=0.9)
        app = web_ui.build_gradio_app(engine, args)
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main(verbosity=1)
