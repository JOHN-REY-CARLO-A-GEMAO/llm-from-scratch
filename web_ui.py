"""
Browser chat UI for the instruction-tuned checkpoint — no JavaScript build
step, no Node, no GPU.

    python web_ui.py                      # http://127.0.0.1:7860

Two interchangeable frontends, selected with --backend (default: auto):

 * gradio    used when `import gradio` succeeds. A gr.ChatInterface with
             streaming, temperature/top-p additional inputs, and its built-in
             Stop/Clear buttons. Any gradio failure (not installed, version
             mismatch, build error) falls back to the stdlib UI below — the
             chatbot must never be unlaunchable because a pip extra is missing.
 * stdlib    a zero-dependency http.server app: one HTML page (inline CSS/JS,
             dark mode) + a JSON-lines streaming endpoint. Tokens reach the
             browser as they are sampled, via fetch() + ReadableStream.

Both frontends talk to the same ChatEngine, which mirrors generate.py's
--chat REPL exactly: out/model_chat.pt + out/model_chat_config.json, the
tokenizer/chat byte-BPE, <|im_start|>/<|im_end|> turn framing, oldest complete
turns dropped when the 256-token context fills, and generation stops the
moment <|im_end|> is sampled.

Design notes, each of which is a real bug it avoids:

 * Single-model lock: one CPU box cannot run two generations at once without
   thrashing, so requests serialise on a lock; a new message first cancels the
   in-flight one (that's what the Stop button and Clear button do), and a
   disconnected client is detected on the next flush and cancelled too.
 * NDJSON streaming instead of SSE/WebSocket: POST + ReadableStream works in
   every modern browser, needs no framing library, and each JSON line is one
   event ({"type":"token"|"done"|"error"}). No chunked-encoding edge cases.
 * The model never sees raw history JSON: the client sends turns, the server
   re-serialises them with the exact SFT template and tokeniser, so prompt
   formatting cannot drift between terminal and browser.
 * Byte-level UTF-8 fragments are buffered token-by-token (same trick as
   generate.py's TokenStream) so a split multi-byte character never flashes
   replacement glyphs in the browser. <|im_start|>/<|im_end|> are control
   markers and never displayed; <|doc_end|> renders as a paragraph break,
   matching the terminal UI.
 * weights_only=True checkpoint load, strict config parse, threads pinned to
   the core count — inherited from generate.py, where each choice is
   documented. HTML is served from memory with a fixed size limit on request
   bodies, so the stdlib server holds no per-request state at all.
"""
from __future__ import annotations

import argparse
import codecs
import json
import os
import queue
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:            # the stage files are importable by name
    sys.path.insert(0, str(HERE))

APP_NAME = "TinyStories Chat"
MAX_MESSAGE_CHARS = 4000                 # prompt-injection-by-paste is a RAM problem, not a model one
MAX_HISTORY_TURNS = 40                   # complete user/assistant pairs
MAX_BODY_BYTES = 1 << 16                 # HTTP request cap; a chat message is never 64 KB
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def die(msg: str) -> None:
    print("\n[web] ERROR: " + msg + "\n", file=sys.stderr)
    raise SystemExit(1)


def resolve(p: Path) -> Path:
    """Repo-relative paths must work no matter where Python was started from —
    double-clicking run_web.bat or `python C:\\path\\to\\web_ui.py`."""
    return p if p.is_absolute() or p.exists() else HERE / p


class IncrementalText:
    """Turn sampled token ids into display text, token by token.

    Byte-BPE tokens are arbitrary byte fragments; decoding one at a time would
    mangle split multi-byte UTF-8. An incremental decoder buffers the tail of
    a character until the rest arrives. Control markers are not prose:
    <|im_start|>/<|im_end|> vanish, <|doc_end|> becomes a paragraph break
    (exactly what generate.py's TokenStream shows in the terminal).
    """

    def __init__(self, tok, unicode_to_bytes):
        special = set(tok.special_tokens) | set(tok.added_tokens)
        self.hidden = {tok.token_to_id[s] for s in special if s != "<|doc_end|>"}
        self.paragraph = {tok.token_to_id["<|doc_end|>"]} if "<|doc_end|>" in tok.token_to_id else set()
        self.id_to_token = tok.id_to_token
        self.to_bytes = unicode_to_bytes
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, token_id: int) -> str:
        if token_id in self.hidden:
            return ""
        if token_id in self.paragraph:
            tail = self.decoder.decode(b"", final=True)
            self.decoder.reset()
            return tail + "\n\n"
        return self.decoder.decode(self.to_bytes(self.id_to_token[token_id]))

    def finish(self) -> str:
        tail = self.decoder.decode(b"", final=True)
        self.decoder.reset()
        return tail


class _Aborted(Exception):
    """Raised inside the on_token callback to unwind generate() on demand."""


class ChatEngine:
    """One model, one lock, many frontends: serialising access is the whole
    concurrency story for a 1.9 M-parameter CPU model."""

    def __init__(self, args, torch, model, cfg, tok, tok_mod, gen_mod, ckpt_path, cfg_path, tok_dir):
        self.args = args
        self.torch = torch
        self.model = model
        self.cfg = cfg
        self.tok = tok
        self.tok_mod = tok_mod
        self.gen = gen_mod
        self.ckpt_path = ckpt_path
        self.cfg_path = cfg_path
        self.tok_dir = tok_dir
        self.end_id = tok.token_to_id[IM_END]
        self.model_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_cancel: threading.Event | None = None

    # ------------------------------------------------------------ bookkeeping
    def info(self) -> dict:
        return {
            "app": APP_NAME,
            "checkpoint": str(self.ckpt_path),
            "checkpoint_mb": round(self.ckpt_path.stat().st_size / 1e6, 1),
            "config": str(self.cfg_path),
            "tokenizer": str(self.tok_dir),
            "params": self.model.num_params(),
            "vocab": self.tok.vocab,
            "context": self.cfg.block_size,
            "n_layer": self.cfg.n_layer,
            "n_head": self.cfg.n_head,
            "d_model": self.cfg.d_model,
            "torch": self.torch.__version__,
            "defaults": {"temperature": self.args.temp, "top_p": self.args.top_p,
                         "top_k": self.args.top_k, "max_tokens": self.args.n},
        }

    def stop_active(self) -> bool:
        with self._active_lock:
            if self._active_cancel is not None:
                self._active_cancel.set()
                return True
        return False

    # ------------------------------------------------------------- serialize
    def serialize(self, turns: list[tuple[str, str]], user: str) -> str:
        """The exact SFT template from generate.py's chat REPL — divergence
        between terminal and browser formatting would change the model's
        behaviour for the same conversation."""
        done = [f"{IM_START}user\n{u}{IM_END}\n{IM_START}assistant\n{a}{IM_END}"
                for u, a in turns]
        current = f"{IM_START}user\n{user}{IM_END}\n{IM_START}assistant\n"
        return "\n".join(done + [current])

    def build_prompt(self, user: str, history, max_new: int):
        """Validate + clamp inputs from the wire, then drop oldest complete
        turns until the prompt leaves room for an answer."""
        user = str(user).strip()[:MAX_MESSAGE_CHARS]
        if not user:
            raise ValueError("message is empty")
        turns: list[tuple[str, str]] = []
        for h in (history or [])[-MAX_HISTORY_TURNS:]:
            if isinstance(h, dict):           # wire format from the browser
                u, a = h.get("user"), h.get("assistant")
            elif isinstance(h, (list, tuple)) and len(h) == 2:   # gradio pairs
                u, a = h
            else:
                continue                      # a malformed turn is dropped, not fatal
            if not isinstance(u, str) or not isinstance(a, str):
                continue
            u, a = u.strip()[:MAX_MESSAGE_CHARS], a.strip()[:MAX_MESSAGE_CHARS]
            if u and a:
                turns.append((u, a))

        prompt = self.serialize(turns, user)
        ids = self.tok.encode_with_specials(prompt, [IM_START, IM_END])
        reserve = min(max_new, max(32, self.cfg.block_size // 3))
        dropped = 0
        while turns and len(ids) > self.cfg.block_size - reserve:
            turns.pop(0)
            dropped += 1
            prompt = self.serialize(turns, user)
            ids = self.tok.encode_with_specials(prompt, [IM_START, IM_END])
        return user, turns, ids, dropped

    # ------------------------------------------------------------ generation
    def stream_chat(self, message: str, history, temperature: float, top_p: float,
                    max_tokens: int | None = None):
        """Yield ("token", text) / ("done", meta) / ("error", msg) events.

        Generation runs on a worker thread (the caller's thread is busy
        writing events to a socket or generator), so the on_token callback
        only decodes and enqueues. Cancelling sets an event that the callback
        turns into _Aborted, unwinding generate() between two tokens.
        """
        a = self.args
        try:
            temperature = min(max(float(temperature), 0.0), 1.5)
        except (TypeError, ValueError):
            temperature = a.temp
        try:
            top_p = min(max(float(top_p), 0.01), 1.0)
        except (TypeError, ValueError):
            top_p = a.top_p
        try:
            max_new = int(max_tokens) if max_tokens else a.n
        except (TypeError, ValueError):
            max_new = a.n
        max_new = min(max(max_new, 1), self.cfg.block_size - 1)

        try:
            user, turns, prompt_ids, dropped = self.build_prompt(message, history, max_new)
        except ValueError as e:
            yield ("error", str(e))
            return

        cancel = threading.Event()
        with self._active_lock:
            if self._active_cancel is not None:
                self._active_cancel.set()    # latest request wins; the old one ends
            self._active_cancel = cancel

        events: queue.Queue = queue.Queue()
        seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(4), "little")

        def worker() -> None:
            decoder = IncrementalText(self.tok, self.tok_mod.unicode_to_bytes)
            streamed: list[str] = []
            n_emitted = [0]                  # list: closure needs a mutable cell

            def emit(token_id: int) -> None:
                if cancel.is_set():
                    raise _Aborted
                n_emitted[0] += 1
                text = decoder.feed(token_id)
                if text:
                    streamed.append(text)
                    events.put(("token", text))

            try:
                with self.model_lock:        # one generation at a time, CPU-wide
                    if cancel.is_set():
                        raise _Aborted
                    params = self.gen.SamplingParams(temperature=temperature, top_k=a.top_k,
                                                     top_p=top_p, repeat_penalty=a.repeat_penalty,
                                                     seed=seed)
                    t0 = time.perf_counter()
                    ids, _, _ = self.gen.generate(
                        self.model, prompt_ids, max_new, params, use_cache=True,
                        eos_id=self.end_id, device=self.device,
                        block_size=self.cfg.block_size, on_token=emit)
                    elapsed = time.perf_counter() - t0
                    streamed.append(decoder.finish())
                new = ids[len(prompt_ids):]
                stopped_on_eos = bool(new) and new[-1] == self.end_id
                if stopped_on_eos:
                    new = new[:-1]
                reason = ("stop" if stopped_on_eos
                          else "length" if len(new) >= max_new else "context")
                reply = "".join(streamed).strip()
                events.put(("done", {
                    "reply": reply,
                    "n_tokens": len(new),
                    "tok_per_s": round(len(new) / elapsed, 1) if elapsed > 0 else None,
                    "elapsed_s": round(elapsed, 2),
                    "seed": seed,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop_reason": reason,
                    "turns_dropped": dropped,
                    "history_turns": len(turns),
                }))
            except _Aborted:
                streamed.append(decoder.finish())
                events.put(("done", {"reply": "".join(streamed).strip(),
                                     "n_tokens": max(0, n_emitted[0] - 1),
                                     "tok_per_s": None, "elapsed_s": None, "seed": seed,
                                     "temperature": temperature, "top_p": top_p,
                                     "stop_reason": "stopped", "turns_dropped": dropped,
                                     "history_turns": len(turns)}))
            except Exception as e:           # a model error is one bad turn, not a dead server
                events.put(("error", f"{type(e).__name__}: {e}"))
            finally:
                events.put(None)             # sentinel: worker is done either way
                with self._active_lock:
                    if self._active_cancel is cancel:
                        self._active_cancel = None

        threading.Thread(target=worker, daemon=True).start()
        while True:
            ev = events.get()
            if ev is None:
                return
            yield ev
            if ev[0] in ("done", "error"):
                # drain until sentinel so the worker can fully exit on its own
                while events.get() is not None:
                    pass
                return

    device = None                          # set in main after torch loads (CPU on purpose)


# ------------------------------------------------------------------- frontend
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__APP_NAME__ · local LLM</title>
<style>
:root {
  --bg: #0b0e14; --bg-soft: #10141d; --panel: #151a26; --panel-2: #1a2130;
  --border: #232b3d; --text: #e6e9f0; --muted: #8a93a8; --faint: #5b6377;
  --accent: #6d8dff; --accent-2: #4c68d8; --user: #243044; --danger: #ff6b6b;
  --ok: #41d19d;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0; background: radial-gradient(1200px 600px at 70% -10%, #141a2b 0%, var(--bg) 55%);
  color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  display: flex; flex-direction: column;
}
header {
  display: flex; align-items: center; gap: 12px; padding: 12px 18px;
  border-bottom: 1px solid var(--border); background: rgba(13,16,24,.75);
  backdrop-filter: blur(8px); position: sticky; top: 0; z-index: 5;
}
.logo {
  width: 34px; height: 34px; border-radius: 9px; flex: 0 0 auto;
  background: linear-gradient(135deg, var(--accent) 0%, #9d6dff 100%);
  display: grid; place-items: center; font-size: 18px;
}
h1 { font-size: 15px; margin: 0; letter-spacing: .2px; }
.subtitle { font-size: 12px; color: var(--muted); margin-top: 2px; }
.spacer { flex: 1; }
.chip {
  display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted);
  background: var(--panel); border: 1px solid var(--border); border-radius: 999px; padding: 5px 11px;
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--faint); }
.dot.ok { background: var(--ok); box-shadow: 0 0 8px rgba(65,209,157,.7); }
.dot.bad { background: var(--danger); }
button {
  font: inherit; color: var(--text); background: var(--panel-2); border: 1px solid var(--border);
  border-radius: 9px; padding: 8px 14px; cursor: pointer; transition: background .15s, border-color .15s;
}
button:hover { background: #222b40; border-color: #31405e; }
button:disabled { opacity: .5; cursor: default; }
button.primary { background: var(--accent-2); border-color: var(--accent-2); }
button.primary:hover { background: var(--accent); border-color: var(--accent); }
button.danger:hover { border-color: var(--danger); color: #ffb4b4; }
#settings {
  display: flex; flex-wrap: wrap; gap: 18px; padding: 10px 18px;
  border-bottom: 1px solid var(--border); background: rgba(16,20,29,.6);
}
.slider { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--muted); flex: 1; min-width: 220px; }
.slider label { white-space: nowrap; }
.slider output { color: var(--text); font-variant-numeric: tabular-nums; min-width: 34px; text-align: right; }
input[type=range] { flex: 1; accent-color: var(--accent); height: 4px; }
#chat { flex: 1; overflow-y: auto; padding: 20px 0 12px; scroll-behavior: smooth; }
.wrap { max-width: 780px; margin: 0 auto; padding: 0 18px; }
.msg { display: flex; gap: 10px; margin: 14px 0; align-items: flex-start; }
.avatar {
  width: 28px; height: 28px; border-radius: 8px; flex: 0 0 auto; display: grid; place-items: center;
  font-size: 13px; font-weight: 600; margin-top: 2px;
}
.msg.user .avatar { background: #2c3650; color: #aebcdd; }
.msg.assistant .avatar { background: linear-gradient(135deg, var(--accent), #9d6dff); color: #101320; }
.bubble {
  padding: 10px 14px; border-radius: 12px; line-height: 1.55; font-size: 14.5px;
  white-space: pre-wrap; word-wrap: break-word; max-width: 100%;
}
.msg.user .bubble { background: var(--user); border-top-left-radius: 4px; border: 1px solid #2f3c58; }
.msg.assistant { flex-direction: row; }
.msg.assistant .bubblebox { min-width: 0; flex: 1; }
.msg.assistant .bubble { background: var(--panel); border: 1px solid var(--border); border-top-left-radius: 4px; }
.meta { font-size: 11.5px; color: var(--faint); margin-top: 5px; padding-left: 4px; min-height: 14px; }
.bubble.error { border-color: rgba(255,107,107,.5); color: #ffc9c9; background: rgba(255,107,107,.08); }
.caret { display: inline-block; width: 8px; height: 16px; vertical-align: text-bottom;
  background: var(--accent); border-radius: 2px; margin-left: 2px; animation: blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity: 0; } }
#welcome {
  margin: 36px auto 10px; text-align: center; color: var(--muted);
}
#welcome h2 { color: var(--text); font-weight: 600; font-size: 20px; margin-bottom: 8px; }
#welcome p { font-size: 13.5px; max-width: 520px; margin: 0 auto 18px; line-height: 1.6; }
.examples { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.example { font-size: 13px; padding: 8px 13px; border-radius: 999px; }
footer {
  border-top: 1px solid var(--border); background: rgba(13,16,24,.8); backdrop-filter: blur(8px);
  padding: 12px 0 calc(12px + env(safe-area-inset-bottom));
}
#composer { display: flex; gap: 10px; align-items: flex-end; }
#input {
  flex: 1; resize: none; max-height: 160px; min-height: 46px; padding: 12px 14px; font: inherit; font-size: 14.5px;
  color: var(--text); background: var(--panel); border: 1px solid var(--border); border-radius: 12px; outline: none;
}
#input:focus { border-color: var(--accent-2); box-shadow: 0 0 0 3px rgba(76,104,216,.25); }
#send { height: 46px; min-width: 92px; font-weight: 600; }
#send.stop { background: rgba(255,107,107,.12); border-color: rgba(255,107,107,.6); color: #ffb4b4; }
.hint { text-align: center; font-size: 11.5px; color: var(--faint); margin-top: 8px; }
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb { background: #232b3d; border-radius: 6px; border: 2px solid var(--bg); }
::selection { background: rgba(109,141,255,.35); }
@media (max-width: 640px) {
  .subtitle { display: none; }
  #send { min-width: 70px; }
}
</style>
</head>
<body>
<header>
  <div class="logo">🤖</div>
  <div>
    <h1>__APP_NAME__</h1>
    <div class="subtitle" id="subtitle">local LLM · built from scratch</div>
  </div>
  <div class="spacer"></div>
  <span class="chip" title="Model status"><span class="dot" id="statusDot"></span><span id="statusText">loading…</span></span>
  <button id="clearBtn" class="danger">Clear Chat</button>
</header>

<div id="settings">
  <div class="slider">
    <label for="temp">Temperature</label>
    <input type="range" id="temp" min="0.1" max="1.5" step="0.05" value="__TEMP__">
    <output id="tempVal">__TEMP__</output>
  </div>
  <div class="slider">
    <label for="topp">Top-P</label>
    <input type="range" id="topp" min="0.1" max="1.0" step="0.05" value="__TOPP__">
    <output id="toppVal">__TOPP__</output>
  </div>
</div>

<div id="chat"><div class="wrap" id="messages">
  <div id="welcome">
    <h2>Say hello to your from-scratch LLM 👋</h2>
    <p>A ~15 M-parameter transformer (Llama-style), tokenizer and chat fine-tune all
    built and trained locally. INT8 quantized from scratch for ~4x smaller weights.
    It runs entirely on this machine — nothing leaves your network.</p>
    <div class="examples">
      <button class="example">Hello! Who are you?</button>
      <button class="example">Tell me a short story about a cat.</button>
      <button class="example">What is 2 + 3?</button>
      <button class="example">Why is the sky blue?</button>
    </div>
  </div>
</div></div>

<footer><div class="wrap">
  <div id="composer">
    <textarea id="input" rows="1" placeholder="Type a message…" autofocus></textarea>
    <button id="send" class="primary">Send</button>
  </div>
  <div class="hint">Enter to send · Shift+Enter for a new line · responses stream token by token</div>
</div></footer>

<script>
const messagesEl = document.getElementById('messages');
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const clearBtn = document.getElementById('clearBtn');
const tempEl = document.getElementById('temp'), tempVal = document.getElementById('tempVal');
const toppEl = document.getElementById('topp'), toppVal = document.getElementById('toppVal');

let history = [];          // [{user, assistant}, ...] complete turns only — like the terminal REPL
let busy = false;
let controller = null;

function autoresize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
}
inputEl.addEventListener('input', autoresize);
tempEl.addEventListener('input', () => tempVal.textContent = (+tempEl.value).toFixed(2).replace(/0$/, ''));
toppEl.addEventListener('input', () => toppVal.textContent = (+toppEl.value).toFixed(2).replace(/0$/, ''));

function scrollBottom() { chatEl.scrollTop = chatEl.scrollHeight; }

function setBusy(b) {
  busy = b;
  sendBtn.textContent = b ? 'Stop' : 'Send';
  sendBtn.classList.toggle('stop', b);
  sendBtn.classList.toggle('primary', !b);
}

function addMsg(role, text) {
  const msg = document.createElement('div');
  msg.className = 'msg ' + role;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'You' : 'AI';
  if (role === 'user') avatar.style.fontSize = '10.5px';
  const box = document.createElement('div');
  box.className = 'bubblebox';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  const meta = document.createElement('div');
  meta.className = 'meta';
  box.appendChild(bubble);
  box.appendChild(meta);
  msg.appendChild(avatar);
  msg.appendChild(box);
  messagesEl.appendChild(msg);
  scrollBottom();
  return { bubble, meta };
}

async function send(text) {
  text = (text ?? inputEl.value).trim();
  if (!text || busy) return;
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();
  inputEl.value = ''; autoresize();
  addMsg('user', text);
  const { bubble, meta } = addMsg('assistant', '');
  const caret = document.createElement('span');
  caret.className = 'caret';
  bubble.appendChild(caret);
  setBusy(true);
  controller = new AbortController();
  let acc = '';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        temperature: +tempEl.value,
        top_p: +toppEl.value,
        history: history,
      }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) throw new Error('HTTP ' + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\\n')) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'token') {
          acc += ev.text;
          caret.remove();
          bubble.textContent = acc;      // safe: model text becomes a text node, never HTML
          bubble.appendChild(caret);
          scrollBottom();
        } else if (ev.type === 'done') {
          acc = ev.reply || acc;
          bubble.textContent = acc || '(no reply — try a higher temperature)';
          history.push({ user: text, assistant: acc });
          let m = ev.n_tokens + ' tokens';
          if (ev.tok_per_s) m += ' · ' + ev.tok_per_s + ' tok/s';
          m += ' · temp ' + ev.temperature + ' · top-p ' + ev.top_p;
          if (ev.stop_reason === 'stopped') m += ' · stopped';
          else if (ev.stop_reason === 'context') m += ' · context limit';
          if (ev.turns_dropped > 0) m += ' · dropped ' + ev.turns_dropped + ' old turn(s)';
          meta.textContent = m;
        } else if (ev.type === 'error') {
          bubble.classList.add('error');
          bubble.textContent = '⚠ ' + ev.message;
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      bubble.classList.add('error');
      bubble.textContent = '⚠ ' + (err.message || 'connection failed — is the server running?');
    } else if (!acc) {
      bubble.textContent = '';
    }
  } finally {
    caret.remove();
    setBusy(false);
    controller = null;
    inputEl.focus();
  }
}

async function stop() {
  try { await fetch('/api/stop', { method: 'POST' }); } catch (e) { /* best effort */ }
  if (controller) controller.abort();
}

sendBtn.addEventListener('click', () => busy ? stop() : send());
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); busy ? stop() : send(); }
});
clearBtn.addEventListener('click', async () => {
  if (busy) await stop();
  history = [];
  messagesEl.innerHTML = '';
  const w = document.createElement('div');
  w.id = 'welcome';
  w.innerHTML = '<h2>Chat cleared ✨</h2><p>History forgotten — the next message starts a fresh conversation.</p>';
  messagesEl.appendChild(w);
  inputEl.focus();
});
document.querySelectorAll('.example').forEach(btn => {
  btn.addEventListener('click', () => { inputEl.value = btn.textContent; send(); });
});

(async function boot() {
  try {
    const res = await fetch('/api/info');
    const info = await res.json();
    document.getElementById('statusDot').className = 'dot ok';
    document.getElementById('statusText').textContent = 'model ready';
    document.getElementById('subtitle').textContent =
      (info.params / 1e6).toFixed(2) + 'M params · ' + info.n_layer + ' layers · ctx ' +
      info.context + ' · vocab ' + info.vocab + ' · torch ' + info.torch;
    if (info.defaults) {
      tempEl.value = info.defaults.temperature; tempEl.dispatchEvent(new Event('input'));
      toppEl.value = info.defaults.top_p; toppEl.dispatchEvent(new Event('input'));
    }
  } catch (e) {
    document.getElementById('statusDot').className = 'dot bad';
    document.getElementById('statusText').textContent = 'server unreachable';
  }
})();
</script>
</body>
</html>
"""


# ------------------------------------------------------------- stdlib server
class Handler(BaseHTTPRequestHandler):
    """One process, one model: GETs serve the page, POST /api/chat streams
    NDJSON token events, POST /api/stop cancels the in-flight generation."""
    server_version = "LLMScratchWeb/1.0"
    protocol_version = "HTTP/1.1"
    engine: ChatEngine | None = None
    index_html: bytes = b""

    def log_message(self, fmt, *args):     # access log: one short line, stderr
        sys.stderr.write("[web] %s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers
    def _send_bytes(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: dict, status: int = 200) -> None:
        self._send_bytes(json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8", status)

    def _read_body(self) -> bytes:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        return self.rfile.read(n) if n > 0 else b"{}"

    def do_HEAD(self) -> None:             # health-check tools probe with HEAD
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.index_html)))
            self.end_headers()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    # -- routing
    def do_GET(self) -> None:              # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            self._send_bytes(self.index_html, "text/html; charset=utf-8")
        elif self.path == "/healthz":
            self._send_json({"ok": True})
        elif self.path == "/api/info":
            self._send_json(self.engine.info())
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:             # noqa: N802
        if self.path == "/api/stop":
            self._send_json({"stopped": self.engine.stop_active()})
            return
        if self.path != "/api/chat":
            self._send_json({"error": "not found"}, 404)
            return

        try:
            payload = json.loads(self._read_body().decode("utf-8", errors="replace"))
            if not isinstance(payload, dict):
                raise ValueError("JSON object expected")
        except (ValueError, UnicodeDecodeError) as e:
            self._send_json({"type": "error", "message": f"bad request: {e}"}, 400)
            return

        # Streaming body: no Content-Length, close-delimited. Every event is
        # one flushed JSON line, so the browser renders tokens as they land.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        def write(ev: dict) -> None:
            self.wfile.write(json.dumps(ev, ensure_ascii=False).encode("utf-8") + b"\n")
            self.wfile.flush()

        gen = self.engine.stream_chat(
            payload.get("message", ""), payload.get("history"),
            payload.get("temperature", 0.8), payload.get("top_p", 0.9),
            payload.get("max_tokens"))
        try:
            for kind, data in gen:
                write({"type": kind, "text": data} if kind == "token"
                      else dict({"type": kind}, **data) if isinstance(data, dict)
                      else {"type": kind, "message": str(data)})
        except (BrokenPipeError, ConnectionResetError):
            pass                             # client left; .close() cancels the worker
        finally:
            gen.close()
            self.close_connection = True


def run_stdlib(engine: ChatEngine, args) -> None:
    Handler.engine = engine
    html = (INDEX_HTML.replace("__APP_NAME__", APP_NAME)
                      .replace("__TEMP__", f"{args.temp:g}")
                      .replace("__TOPP__", f"{args.top_p:g}"))
    Handler.index_html = html.encode("utf-8")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True           # a stuck client must not block shutdown
    local = f"http://127.0.0.1:{args.port}"
    print(f"\n[web] {APP_NAME} is running (stdlib backend, no gradio needed)")
    print(f"[web]    local:   {local}")
    if args.host == "0.0.0.0":
        try:
            lan = socket.gethostbyname(socket.gethostname())
            if lan and not lan.startswith("127."):
                print(f"[web]    network: http://{lan}:{args.port}   (other devices on your LAN)")
        except OSError:
            pass
    print("[web]    Ctrl+C to stop\n", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(local)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopping…")
    finally:
        server.server_close()


# ------------------------------------------------------------- gradio server
def gradio_history_to_pairs(history):
    """gr 5+/6 sends messages-dicts, gr 4 sends (user, bot) tuples; the engine
    only understands complete (user, assistant) turns."""
    pairs, pending = [], None
    for item in history or []:
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
            if role == "user":
                pending = str(content or "")
            elif role == "assistant" and pending is not None:
                pairs.append((pending, str(content or "")))
                pending = None
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            pairs.append((str(item[0] or ""), str(item[1] or "")))
    return [(u, a) for u, a in pairs if u and a]


def build_gradio_app(engine: ChatEngine, args):
    """Build the gr.ChatInterface without launching (tests import this)."""
    import gradio as gr

    def respond(message, history, temperature, top_p):
        if not str(message).strip():
            yield ""
            return
        acc = ""
        try:
            for kind, data in engine.stream_chat(message, gradio_history_to_pairs(history),
                                                 temperature, top_p):
                if kind == "token":
                    acc += data
                    yield acc            # ChatInterface re-renders on every yield = streaming
                elif kind == "done":
                    yield data.get("reply", acc)
                elif kind == "error":
                    raise gr.Error(data)
        except GeneratorExit:
            engine.stop_active()         # Stop button: cancel between two tokens
            raise

    kw = dict(
        fn=respond,
        title=f"{APP_NAME} · local LLM",
        description=("A ~15 M-parameter Llama-style transformer trained from scratch with "
                     "INT8 weight quantization, running entirely on this machine. Responses stream token by token."),
        examples=[["Hello! Who are you?", args.temp, args.top_p],
                  ["Tell me a short story about a cat.", args.temp, args.top_p],
                  ["What is 2 + 3?", args.temp, args.top_p],
                  ["Why is the sky blue?", args.temp, args.top_p]],
        additional_inputs=[
            gr.Slider(0.1, 1.5, value=args.temp, step=0.05, label="Temperature"),
            gr.Slider(0.1, 1.0, value=args.top_p, step=0.05, label="Top-P"),
        ],
        additional_inputs_accordion="⚙ Sampling settings",
        clear_btn="Clear Chat",
        flagging_mode="never",
        theme=(gr.themes.Soft() if hasattr(gr, "themes") else None),
    )
    # ChatInterface's kwargs keep changing across gradio releases (theme and
    # clear_btn exist in 4/5, are gone in 6); pass only what this install
    # accepts instead of probing versions one kwarg at a time.
    import inspect
    supported = inspect.signature(gr.ChatInterface.__init__).parameters
    return gr.ChatInterface(**{k: v for k, v in kw.items() if k in supported})


def run_gradio(engine: ChatEngine, args) -> None:
    """Only reached when gradio is installed AND --backend allows it. Any
    failure here falls back to the stdlib UI at the call site."""
    import gradio as gr
    app = build_gradio_app(engine, args)
    local = f"http://127.0.0.1:{args.port}"
    print(f"\n[web] {APP_NAME} is running (gradio {gr.__version__} backend)")
    print(f"[web]    local:   {local}   network: http://0.0.0.0:{args.port}")
    print("[web]    Ctrl+C to stop\n", flush=True)
    app.queue().launch(server_name=args.host, server_port=args.port,
                       inbrowser=not args.no_browser, show_error=True)


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        prog="web_ui.py",
        description="Browser chat UI for the fine-tuned checkpoint (gradio if available, stdlib otherwise).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", default=None,
                    help="chat checkpoint (.pt); auto-selects model_15m_chat(_int8).pt when present")
    ap.add_argument("--config", default=None,
                    help="default: <ckpt>_config.json (i.e. out/model_15m_chat_config.json), then config.json/txt")
    ap.add_argument("--tokenizer", default="tokenizer/chat", help="chat tokenizer directory")
    ap.add_argument("--quantized", "--int8", dest="quantized", action="store_true",
                    help="load an INT8-quantized checkpoint (written by quantize.py)")
    ap.add_argument("--host", default="0.0.0.0", help="bind address (0.0.0.0 = localhost + LAN)")
    ap.add_argument("--port", type=int, default=7860, help="web UI port")
    ap.add_argument("-n", "--n", "--max-tokens", dest="n", type=int, default=128,
                    help="max new tokens per reply (256-token context)")
    ap.add_argument("--temp", "--temperature", dest="temp", type=float, default=0.8,
                    help="default sampling temperature (slider: 0.1-1.5)")
    ap.add_argument("--top-p", type=float, default=0.9, help="default nucleus threshold (slider: 0.1-1.0)")
    ap.add_argument("--top-k", type=int, default=40, help="safety cap on top of top-p; 0 disables")
    ap.add_argument("--repeat-penalty", type=float, default=1.15, help="1.0 disables; small models loop without it")
    ap.add_argument("--seed", type=int, default=None, help="default: random seed per reply")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    ap.add_argument("--backend", choices=("auto", "gradio", "stdlib"), default="auto",
                    help="auto: gradio when importable, always with stdlib fallback")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab on start")
    args = ap.parse_args()
    if args.n < 1:
        ap.error("--n must be >= 1")

    # Default checkpoint selection mirrors generate.py: prefer the 15M chat
    # checkpoint (int8 or fp32) if it exists on disk.
    if args.ckpt is None:
        suffix = "_int8.pt" if args.quantized else ".pt"
        for cand in ("out/model_15m_chat", "out/model_chat"):
            p = resolve(Path(cand + suffix))
            if p.exists():
                args.ckpt = cand + suffix
                break
        else:
            args.ckpt = "out/model_chat" + suffix
        # If the int8 variant exists prefer it even without --quantized (auto-detect).
        if not args.quantized:
            for cand in ("out/model_15m_chat_int8.pt", "out/model_chat_int8.pt"):
                p = resolve(Path(cand))
                if p.exists():
                    args.ckpt = cand
                    args.quantized = True
                    break

    # Deferred so `--help` works with no torch installed; same wording as the CLI.
    try:
        import generate as cli
    except Exception as e:
        die(f"generate.py must be importable next to web_ui.py ({e})")
    torch = cli.import_torch()
    from _stage import load as _load        # noqa: E402  (needs torch already importable)
    model_mod, gen_mod = _load("03_model"), _load("06_generate")
    tok_mod = _load("01_tokenizer")
    import quantize                         # noqa: E402
    try:
        gen_mod.SamplingParams(temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
                               repeat_penalty=args.repeat_penalty, seed=args.seed)
    except ValueError as e:
        ap.error(str(e))

    args.threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    ckpt = resolve(Path(args.ckpt))
    if not ckpt.exists():
        die(f"checkpoint {ckpt} not found.\n"
            f"      It is committed to the repo (~15 MB for 15M model, ~8 MB for 1.9M model) "
            f"— run `git pull`, then retry.")
    cfg_path = cli.find_config(ckpt, args.config)   # out/model_chat_config.json by default
    tok_dir = resolve(Path(args.tokenizer))
    if not (tok_dir / "bpe.pkl").exists():
        die(f"tokenizer {tok_dir}/bpe.pkl not found — run `git pull`, it is committed too.")

    # ---- load model: INT8-quantized path OR plain fp32 path ----
    # Auto-detect quantized checkpoints by magic. One torch.load() total.
    is_int8 = args.quantized
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not is_int8:
        is_int8 = isinstance(raw, dict) and raw.get("magic") == "quantized_v1"
    if is_int8:
        if not (isinstance(raw, dict) and raw.get("magic") == "quantized_v1"):
            die(f"{ckpt} was expected to be an INT8 checkpoint but lacks the magic marker.")
        cfg = model_mod.GPTConfig(**raw["config"])
        model = model_mod.GPT(cfg)
        quantize.quantize_model(model, per_channel=raw.get("per_channel", True))
        model.load_state_dict(raw["state_dict"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        cfg = model_mod.GPTConfig.load(cfg_path)        # strict: unknown keys are an error
        model = model_mod.GPT(cfg)
        try:
            model.load_state_dict(raw)
        except RuntimeError as e:
            die(f"{ckpt} does not match {cfg_path}.\n      {e}\n"
                f"      They are a pair: pull both from the same commit.")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    tok = tok_mod.ByteLevelBPE.load(tok_dir)
    for marker in (IM_START, IM_END):
        if marker not in tok.token_to_id:
            die(f"tokenizer {tok_dir} is missing {marker}. Use --tokenizer tokenizer/chat "
                f"(the base tokenizer has no chat markers).")

    engine = ChatEngine(args, torch, model, cfg, tok, tok_mod, gen_mod, ckpt, cfg_path, tok_dir)
    engine.device = torch.device("cpu")             # CPU-only, like generate.py, on purpose
    qtag = "  INT8 quantized" if quantize.is_quantized(model) else ""
    print(f"[web] torch {torch.__version__}  device cpu  threads {args.threads}")
    print(f"[web] checkpoint {ckpt}  ({model.num_params():,} params, "
          f"{ckpt.stat().st_size / 1e6:.1f} MB){qtag}")
    print(f"[web] config {cfg_path}  ({cfg.n_layer} layers, {cfg.n_head} heads, "
          f"d_model {cfg.d_model}, context {cfg.block_size}, vocab {cfg.vocab_size})")
    print(f"[web] tokenizer {tok_dir}  (vocab {tok.vocab})", flush=True)

    use_gradio = args.backend in ("auto", "gradio")
    if use_gradio:
        try:
            run_gradio(engine, args)
            return
        except ImportError:
            if args.backend == "gradio":
                die("gradio is not installed (`pip install gradio`) — or drop --backend gradio "
                    "for the built-in stdlib UI.")
            print("[web] gradio not installed → using the built-in stdlib UI "
                  "(pip install gradio for the gradio frontend)")
        except Exception as e:
            if args.backend == "gradio":
                die(f"gradio failed to start: {type(e).__name__}: {e}")
            print(f"[web] gradio failed ({type(e).__name__}: {e}) → falling back to the stdlib UI")
    run_stdlib(engine, args)


if __name__ == "__main__":
    main()
