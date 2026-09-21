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
tokenizer/chat byte-BPE, <|im_start|>/<|im_end|> turn framing, an optional
<|im_start|>system turn (personas.py), oldest complete turns dropped when the
256-token context fills, and generation stops the moment <|im_end|> is sampled.

What the browser adds on top of the terminal:

 * **Reasoning view.** A reply that plans inside `<think> … </think>` (the
   format `07_make_sft.py` trains) is rendered as a collapsible, DeepSeek-style
   drop-down above the answer, open while it is still being written and folded
   away once the answer starts. The split is `personas.split_think`, the same
   function the terminal uses.
 * **Document Q&A (RAG).** "📄 Upload Document" reads a .txt/.md file in the
   browser, POSTs its text, and `rag.py` indexes it in memory (BM25 over word
   unigrams/bigrams, character-trigram fallback). Each question retrieves the two
   best passages and injects them into the system prompt as
   `Context from uploaded document: … Answer the user's question based ONLY on
   this context.`, trimmed to the token budget that is actually left in the
   256-token window. Retrieved passages are shown with the reply.
 * **System prompt / persona picker.** Four presets (friendly, pirate, math,
   storyteller), a custom text box, and an explicit "None" that sends no system
   turn at all. The resolved text is re-serialised server-side for every
   request, so switching persona mid-conversation cannot leave a stale block in
   the prompt.
 * **Tool hooks** (tools.py). A bare "5 + 5" is answered by the calculator
   *before* the model runs, a `<<calc: …>>`/`<<tool: …>>` the model emits is
   substituted with the exact value mid-stream, and word counts, reversals,
   case changes and date/time questions are computed too ("how many words is
   this?", "what is the date today?"). Python tool runs are shown as
   `🐍 Python Tool Execution` cards with the source and its result; arithmetic
   keeps the 🧮 calculator label. Every path is labelled, so a computed answer is
   never confused with a sampled one.
 * **Token counter / latency monitor** in the header: live tokens, tok/s and
   elapsed time while streaming, then time-to-first-token, prompt size and seed
   for the finished turn — the numbers come from the server, not from guesswork.

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

import generate as cli                   # noqa: E402  (stdlib-only; for serialize_chat)
import personas                          # noqa: E402  (system prompts + <think> protocol)
import rag                               # noqa: E402  (in-memory document retrieval)
import tools                             # noqa: E402  (calculator + python tool)

APP_NAME = "TinyStories Chat"
MAX_MESSAGE_CHARS = 4000                 # prompt-injection-by-paste is a RAM problem, not a model one
MAX_HISTORY_TURNS = 40                   # complete user/assistant pairs
MAX_BODY_BYTES = 1 << 16                 # HTTP request cap; a chat message is never 64 KB
MAX_UPLOAD_BYTES = 2 << 20               # a .txt/.md document is never 2 MB of paste
MAX_DOC_NAME_CHARS = 80
RAG_TOP_K = 2                            # retrieved passages per question
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
        # getattr: tests and embedders build `args` by hand and only set what
        # they need, so every new option must have a working default.
        self.tools_on = not getattr(args, "no_tools", False)
        self.rag_on = not getattr(args, "no_rag", False)
        #: The uploaded document, if any. One at a time on purpose: a chat UI
        #: with three documents and no way to see which passages were used is a
        #: plausible-sounding answer generator, not a document assistant.
        self.docs: "rag.DocumentIndex | None" = None
        self.docs_lock = threading.Lock()
        system = getattr(args, "system", None)
        self.default_persona_value = (system if system is not None
                                      else getattr(args, "persona", None)
                                      if getattr(args, "persona", None) is not None
                                      else personas.DEFAULT_SLUG)

    # ------------------------------------------------------------ bookkeeping
    def info(self) -> dict:
        label, text = personas.resolve(self.default_persona_value, tool_hint=self.tools_on)
        return {
            "app": APP_NAME,
            "persona": label,
            "personas": personas.choices(),
            "system": text,
            "tools": self.tools_on,
            "rag": self.rag_on,
            "doc": self.doc_info(),
            "reasoning": {"open": personas.THINK_OPEN, "close": personas.THINK_CLOSE},
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

    # --------------------------------------------------------------- documents
    def doc_info(self) -> dict | None:
        """Index statistics for the loaded document, or None."""
        with self.docs_lock:
            index = self.docs
            return None if index is None else index.stats()

    def set_document(self, name: str, text: str) -> dict:
        """Index `text` in memory and replace whatever was there before.

        The index is built outside the chat lock (indexing a book takes a moment
        and no model call is in flight), then swapped in atomically: a request
        between the two sees either the old document or the new one, never a
        half-built index.
        """
        clean_name = " ".join(str(name or "").split())[-MAX_DOC_NAME_CHARS:] or "document"
        body = str(text or "").replace("\x00", "").replace("\r\n", "\n")
        if not body.strip():
            raise ValueError("the document is empty")
        index = rag.DocumentIndex.from_text(body, name=clean_name)
        with self.docs_lock:
            self.docs = index
        return index.stats()

    def clear_document(self) -> bool:
        with self.docs_lock:
            had = self.docs is not None
            self.docs = None
        return had

    def _tokens(self, prompt_system: str) -> int:
        """Token count of a system block in this engine's own serialization."""
        text = self.serialize([], "", prompt_system)
        return len(self.tok.encode_with_specials(text, [IM_START, IM_END]))

    def rag_context(self, question: str, system_text: str) -> tuple[str, list]:
        """Inject the retrieved passages into the system prompt.

        Returns `(system, hits)`. The document block goes **first** and the
        persona text after it: `personas.shrink` keeps the longest fitting
        prefix, so if the two together overflow the window, what survives is the
        context and its "answer ONLY from this context" instruction — not a
        persona paragraph the model does not obey anyway.
        """
        with self.docs_lock:
            index = self.docs
        if index is None or not self.rag_on:
            return system_text, []
        reserve = min(getattr(self.args, "n", 128) or 128, max(32, self.cfg.block_size // 3))
        budget = self.cfg.block_size - reserve
        block, hits = index.context_block(question, budget, self._tokens, top_k=RAG_TOP_K)
        if not block:
            return system_text, hits
        if system_text:
            room = budget - self._tokens(block)
            extra = personas.shrink(system_text, self._tokens, room) if room > 0 else ""
            block = f"{block}\n\n{extra}" if extra else block
        return block, hits

    # ------------------------------------------------------------- serialize
    def serialize(self, turns: list[tuple[str, str]], user: str, system: str | None = None) -> str:
        """The exact SFT template from generate.py's chat REPL — divergence
        between terminal and browser formatting would change the model's
        behaviour for the same conversation. `generate.serialize_chat` *is* the
        implementation, so there is exactly one string template in the repo."""
        return cli.serialize_chat(system or "", turns, user)

    def resolve_system(self, system: str | None = None,
                       persona: str | None = None) -> tuple[str, str]:
        """Resolve one request's system prompt to `(label, text)`.

        The page resolves the dropdown itself and sends the literal text, which
        is what makes "edited a preset" work; `persona` is the slug fallback for
        plain API clients. The math tutor's calculator hint is attached here, so
        a browser that only knows preset text still gets it (and a wholly custom
        prompt never does).
        """
        if system is not None and str(system).strip():
            preset = personas.get(persona) if persona else None
            text = personas.sanitize(str(system))
            if preset is not None and preset.hint and self.tools_on and text == preset.system:
                text = f"{text}\n\n{preset.hint}"
            return (preset.label if preset is not None else "Custom"), text
        return personas.resolve(persona if persona is not None else self.default_persona_value,
                                tool_hint=self.tools_on)

    def build_prompt(self, user: str, history, max_new: int, system: str | None = None):
        """Validate + clamp inputs from the wire, then drop oldest complete
        turns until the prompt leaves room for an answer.

        Returns `(user, turns, ids, dropped, system)`. The system block is the
        last thing to give way: it is the only part of the prompt the user did
        not type just now, but leaving it in when it does not fit would make
        prefill truncate from the left and silently discard the *question*.
        """
        user = personas.sanitize(str(user).strip(), MAX_MESSAGE_CHARS)[:MAX_MESSAGE_CHARS]
        if not user:
            raise ValueError("message is empty")
        system = personas.sanitize(system) if system else ""
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
            u, a = (personas.sanitize(u, MAX_MESSAGE_CHARS)[:MAX_MESSAGE_CHARS],
                    personas.sanitize(a, MAX_MESSAGE_CHARS)[:MAX_MESSAGE_CHARS])
            if u and a:
                turns.append((u, a))

        reserve = min(max_new, max(32, self.cfg.block_size // 3))
        budget = self.cfg.block_size - reserve

        def encode(prompt_system: str) -> list[int]:
            return self.tok.encode_with_specials(
                self.serialize(turns, user, prompt_system), [IM_START, IM_END])

        ids = encode(system)
        dropped = 0
        while turns and len(ids) > budget:
            turns.pop(0)
            dropped += 1
            ids = encode(system)
        if system and len(ids) > budget:
            system = personas.shrink(system, lambda text: len(encode(text)), budget)
            ids = encode(system)
        return user, turns, ids, dropped, system

    # ------------------------------------------------------------ generation
    def stream_chat(self, message: str, history, temperature: float, top_p: float,
                    max_tokens: int | None = None, system: str | None = None,
                    persona: str | None = None):
        """Yield ("token", text) / ("tool", info) / ("docs", info) / ("done", meta) / ("error", msg).

        Generation runs on a worker thread (the caller's thread is busy
        writing events to a socket or generator), so the on_token callback
        only decodes and enqueues. Cancelling sets an event that the callback
        turns into _Aborted, unwinding generate() between two tokens.

        Questions the tools can answer exactly never reach the model at all: a
        bare "5 + 5" ("5 + 5 = 10"), "how many words is this?" and "what is the
        date today?" are computed, because the sampler has nothing to add to a
        fact. A question *inside* prose still goes to the model — it may be part
        of a bigger question — and the exact value is appended to the reply.

        When a document is loaded, the two best passages are injected into the
        system prompt and reported in a `docs` event, so the browser can show
        what the answer was allowed to see.
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

        system_label, system_text = self.resolve_system(system, persona)
        try:
            user, turns, prompt_ids, dropped, system_text = self.build_prompt(
                message, history, max_new, system_text)
        except ValueError as e:
            yield ("error", str(e))
            return

        # ---- retrieval: the document goes in the system block, sized to fit ----
        hits: list = []
        doc = self.doc_info()
        if doc is not None and self.rag_on:
            system_text, hits = self.rag_context(user, system_text)
            if hits:
                # re-run the budget check with the augmented system prompt: the
                # context is only useful if the *question* still fits next to it
                try:
                    user, turns, prompt_ids, dropped, system_text = self.build_prompt(
                        user, turns, max_new, system_text)
                except ValueError as e:                      # pragma: no cover
                    yield ("error", str(e))
                    return
                yield ("docs", {"doc": doc, "hits": [h.to_json() for h in hits],
                                "system_chars": len(system_text)})

        # ---- tool hook: anything a tool can answer exactly, without the model ----
        query = tools.find_tool_question(user) if self.tools_on else None
        if query is not None and query.bare:
            started = time.perf_counter()
            event = query.to_event()
            yield ("tool", event)
            yield ("token", query.answer)
            done = {"reply": query.answer,
                    "source": "calculator" if query.kind == "math" else query.source,
                    "n_tokens": 0, "tok_per_s": None,
                    "elapsed_s": round(time.perf_counter() - started, 4),
                    "first_token_s": round(time.perf_counter() - started, 4),
                    "seed": None, "temperature": temperature, "top_p": top_p,
                    "stop_reason": "tool", "turns_dropped": dropped,
                    "history_turns": len(turns), "prompt_tokens": len(prompt_ids),
                    "system": system_text, "persona": system_label,
                    "tool_calls": [], "think": {"reasoning": "", "answer": query.answer,
                                                "closed": True, "has_reasoning": False},
                    "doc": doc, "retrieved": [h.to_json() for h in hits]}
            if query.kind == "math":
                done["tool"] = {"expression": query.expression, "result": query.text,
                                "line": query.answer, "source": "question"}
            yield ("done", done)
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
            tool_filter = tools.ToolTextFilter() if self.tools_on else None
            streamed: list[str] = []
            n_emitted = [0]                  # list: closure needs a mutable cell
            first_token: list[float | None] = [None]
            t0 = time.perf_counter()         # replaced once the lock is held

            def emit(token_id: int) -> None:
                if cancel.is_set():
                    raise _Aborted
                n_emitted[0] += 1
                if first_token[0] is None:
                    first_token[0] = time.perf_counter() - t0
                text = decoder.feed(token_id)
                if tool_filter is not None:
                    text = tool_filter.feed(text)
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
                    tail = decoder.finish()
                    if tool_filter is not None:
                        tail = tool_filter.feed(tail) + tool_filter.finish()
                    streamed.append(tail)
                    # A math question the model answered *without* the right
                    # number is the failure this hook exists to prevent: append
                    # the exact value instead of letting a guess stand alone.
                    if query is not None and query.kind == "math" and not tools.value_in_text(
                            query.result, "".join(streamed)):
                        extra = f"\n\n🧮 {query.answer}"
                        streamed.append(extra)
                        events.put(("token", extra))
                new = ids[len(prompt_ids):]
                stopped_on_eos = bool(new) and new[-1] == self.end_id
                if stopped_on_eos:
                    new = new[:-1]
                reason = ("stop" if stopped_on_eos
                          else "length" if len(new) >= max_new else "context")
                reply = "".join(streamed).strip()
                calls = [c.line for c in tool_filter.calls] if tool_filter is not None else []
                thought = personas.split_think(reply)
                # Which evaluator produced the call, not what type it returned:
                # `word_count('a b')` is a value of 2 and still a python tool run.
                python_calls = [c for c in (tool_filter.calls if tool_filter is not None else [])
                                if c.source != "calc"]
                meta = {
                    "reply": reply,
                    "source": "model",
                    "n_tokens": len(new),
                    "tok_per_s": round(len(new) / elapsed, 1) if elapsed > 0 else None,
                    "elapsed_s": round(elapsed, 2),
                    "first_token_s": (round(first_token[0], 3)
                                     if first_token[0] is not None else None),
                    "prompt_tokens": len(prompt_ids),
                    "seed": seed,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop_reason": reason,
                    "turns_dropped": dropped,
                    "history_turns": len(turns),
                    "system": system_text,
                    "persona": system_label,
                    "tool_calls": calls,
                    "think": thought.to_json(),
                    "doc": doc,
                    "retrieved": [h.to_json() for h in hits],
                    "grounding": (rag.grounding(reply, system_text)
                                  if hits and reply else None),
                }
                if query is not None and query.kind == "math":
                    meta["tool"] = {"expression": query.expression, "result": query.text,
                                    "line": query.answer, "source": "question"}
                if calls:
                    events.put(("tool", {"source": "model", "calls": calls}))
                if python_calls:
                    events.put(("tool", {"source": "model", "kind": "python",
                                         "calls": [c.line for c in python_calls],
                                         "results": [c.to_json() for c in python_calls]}))
                events.put(("done", meta))
            except _Aborted:
                tail = decoder.finish()
                if tool_filter is not None:
                    tail = tool_filter.feed(tail) + tool_filter.finish()
                streamed.append(tail)
                events.put(("done", {"reply": "".join(streamed).strip(),
                                     "source": "model",
                                     "n_tokens": max(0, n_emitted[0] - 1),
                                     "tok_per_s": None, "elapsed_s": None,
                                     "first_token_s": (round(first_token[0], 3)
                                                       if first_token[0] is not None else None),
                                     "prompt_tokens": len(prompt_ids),
                                     "seed": seed, "temperature": temperature, "top_p": top_p,
                                     "stop_reason": "stopped", "turns_dropped": dropped,
                                     "history_turns": len(turns), "system": system_text,
                                     "persona": system_label,
                                     "think": personas.split_think("".join(streamed)).to_json(),
                                     "doc": doc, "retrieved": [h.to_json() for h in hits],
                                     "grounding": None,
                                     "tool_calls": ([c.line for c in tool_filter.calls]
                                                    if tool_filter is not None else [])}))
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
  display: flex; flex-wrap: wrap; gap: 12px 18px; padding: 10px 18px; align-items: flex-end;
  border-bottom: 1px solid var(--border); background: rgba(16,20,29,.6);
}
.slider { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--muted); flex: 1; min-width: 220px; }
.slider label { white-space: nowrap; }
.slider output { color: var(--text); font-variant-numeric: tabular-nums; min-width: 34px; text-align: right; }
input[type=range] { flex: 1; accent-color: var(--accent); height: 4px; }
.ctl { display: flex; flex-direction: column; gap: 5px; font-size: 12.5px; color: var(--muted); }
.ctl.persona { flex: 0 0 auto; }
.ctl.prompt { flex: 1 1 320px; min-width: 220px; }
select, #systemPrompt {
  font: inherit; font-size: 13px; color: var(--text); background: var(--panel);
  border: 1px solid var(--border); border-radius: 9px; padding: 8px 10px; outline: none;
}
select:focus, #systemPrompt:focus { border-color: var(--accent-2); box-shadow: 0 0 0 3px rgba(76,104,216,.22); }
select { cursor: pointer; }
#systemPrompt { resize: none; min-height: 36px; max-height: 120px; line-height: 1.4; }
.chip.hide-sm { display: inline-flex; }
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
.meta { font-size: 11.5px; color: var(--faint); margin-top: 6px; padding-left: 4px; min-height: 14px; }
.bubble.error { border-color: rgba(255,107,107,.5); color: #ffc9c9; background: rgba(255,107,107,.08); }
.actions { display: flex; gap: 6px; margin-top: 6px; padding-left: 4px; }
button.mini {
  font-size: 11.5px; padding: 3px 9px; border-radius: 7px; color: var(--muted);
  background: transparent; border: 1px solid var(--border);
}
button.mini:hover { color: var(--text); background: var(--panel-2); border-color: #31405e; }
button.mini.ok { color: var(--ok); border-color: rgba(65,209,157,.5); }
/* ---- reasoning block: a DeepSeek-style fold above the answer ---- */
.think {
  border: 1px solid #2c3550; border-left: 3px solid #7c6dff; border-radius: 10px;
  background: linear-gradient(180deg, rgba(124,109,255,.10), rgba(124,109,255,.03));
  margin: 0 0 10px; overflow: hidden;
}
.think summary {
  cursor: pointer; user-select: none; list-style: none; padding: 7px 12px;
  font-size: 12.5px; color: #b9b2ff; display: flex; align-items: center; gap: 8px;
}
.think summary::-webkit-details-marker { display: none; }
.think summary::after { content: "▾"; margin-left: auto; color: var(--faint); font-size: 11px; }
.think[open] summary::after { content: "▴"; }
.think summary .spin {
  width: 8px; height: 8px; border-radius: 50%; background: #7c6dff;
  animation: pulse 1.1s ease-in-out infinite;
}
.think[data-live="0"] summary .spin { display: none; }
@keyframes pulse { 0%, 100% { opacity: .35; } 50% { opacity: 1; } }
.think-body {
  padding: 2px 14px 12px; font-size: 13px; line-height: 1.6; color: #b6bed4;
  white-space: pre-wrap; word-wrap: break-word; border-top: 1px solid rgba(124,109,255,.18);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.think.note { border-left-color: var(--danger); background: rgba(255,107,107,.06); }
.think.note summary { color: #ffb4b4; }

/* ---- tool cards: 🐍 Python Tool Execution ---- */
.toolcard {
  margin: 0 0 10px; border: 1px solid #2b3a35; border-left: 3px solid var(--ok);
  border-radius: 10px; background: rgba(65,209,157,.07); overflow: hidden;
}
.toolcard .head {
  padding: 6px 12px; font-size: 12.5px; color: #9fe8c9; border-bottom: 1px solid rgba(65,209,157,.16);
}
.toolcard pre {
  margin: 0; padding: 9px 12px; font-size: 12.5px; line-height: 1.5; overflow-x: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.toolcard pre.out { color: #cfe9df; border-top: 1px dashed rgba(65,209,157,.18); }
.toolcard pre.code { color: #dbe6ff; }
.toolcard .outlabel { color: var(--muted); }

/* ---- retrieval sources ---- */
.sources { margin: 8px 0 0; border: 1px solid var(--border); border-radius: 10px; background: var(--bg-soft); }
.sources summary { cursor: pointer; padding: 7px 12px; font-size: 12.5px; color: var(--muted); list-style: none; }
.sources summary::-webkit-details-marker { display: none; }
.sources .hit { padding: 8px 12px; border-top: 1px solid var(--border); font-size: 12.5px; color: var(--muted); }
.sources .hit b { color: var(--text); font-weight: 600; }
.sources .hit .score { color: var(--ok); font-variant-numeric: tabular-nums; }
.docrow { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.pill.doc { color: #9fe8c9; border-color: rgba(65,209,157,.35); }
#docNote { font-size: 11.5px; color: var(--faint); }
#docNote.error { color: #ffb4b4; }
button.mini.upload { border-style: dashed; }
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
  .chip.hide-sm { display: none; }
  #settings { gap: 10px 12px; }
}
@media (prefers-reduced-motion: reduce) {
  .caret { animation: none; }
  #chat { scroll-behavior: auto; }
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
  <span class="chip" title="Token counter + latency monitor: live tokens, tokens/second and elapsed time for the current reply">
    ⚡ <span id="perf">idle</span></span>
  <span class="chip hide-sm" title="Session totals since this page was loaded: generated tokens and replies">
    Σ <span id="totals">0 tokens · 0 replies</span></span>
  <span class="chip" title="Model status"><span class="dot" id="statusDot"></span><span id="statusText">loading…</span></span>
  <button id="clearBtn" class="danger">Clear Chat</button>
</header>

<div id="settings">
  <div class="ctl persona">
    <label for="persona">System Prompt</label>
    <select id="persona"></select>
  </div>
  <div class="ctl prompt">
    <label for="systemPrompt">Prompt text <span id="sysNote"></span></label>
    <textarea id="systemPrompt" rows="1" spellcheck="false"
      placeholder="You are a helpful assistant…"></textarea>
  </div>
  <div class="ctl doc" id="docCtl">
    <label for="uploadBtn">Document (Local Q&amp;A)</label>
    <div class="docrow">
      <button type="button" id="uploadBtn" class="mini upload">📄 Upload Document (.txt / .md)</button>
      <input type="file" id="fileInput" accept=".txt,.md,.markdown,text/plain,text/markdown" hidden>
      <span class="chip pill doc" id="docChip" style="display:none">
        📄 <b id="docName"></b>&nbsp;<span id="docStats"></span>
        <button type="button" id="docClear" class="mini" title="Remove the document">✕</button>
      </span>
      <span id="docNote"></span>
    </div>
  </div>
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
    built and trained locally with chain-of-thought traces, INT8 quantized from
    scratch for ~4x smaller weights. It runs entirely on this machine — nothing
    leaves your network. Ask a question that needs a few steps and it plans inside
    a <code>&lt;think&gt;</code> block first; upload a .txt/.md file and it answers
    from your document; arithmetic, word counts and dates go to real Python, not to
    the model's guesswork.</p>
    <div class="examples">
      <button class="example">Hello! Who are you?</button>
      <button class="example">Why does it rain?</button>
      <button class="example">Sam has 12 marbles, gets 4 more, and gives 6 away. How many marbles does Sam have?</button>
      <button class="example">What is the date today?</button>
      <button class="example">What is 12 * 4?</button>
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
const personaEl = document.getElementById('persona');
const promptEl = document.getElementById('systemPrompt');
const sysNote = document.getElementById('sysNote');
const perfEl = document.getElementById('perf');
const totalsEl = document.getElementById('totals');

// Presets come from personas.py (server-side, one source of truth for the
// terminal REPL and the browser); the hint text is added server-side.
const PERSONAS = __PERSONAS_JSON__;
const DEFAULTS = { persona: "__DEFAULT_PERSONA__", system: __DEFAULT_SYSTEM_JSON__,
                   tools: __TOOLS__, rag: __RAG__, doc: __DOC_JSON__,
                   maxUploadMB: __MAX_UPLOAD_MB__,
                   thinkOpen: "__THINK_OPEN__", thinkClose: "__THINK_CLOSE__" };

let document_ = DEFAULTS.doc || null;      // {name, chars, chunks, terms, ...} or null

let history = [];          // [{user, assistant}, ...] complete turns only — like the terminal REPL
let busy = false;
let controller = null;
const session = { tokens: 0, replies: 0 };

function autoresize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
  promptEl.style.height = 'auto';
  promptEl.style.height = Math.min(promptEl.scrollHeight, 120) + 'px';
}
inputEl.addEventListener('input', autoresize);
tempEl.addEventListener('input', () => tempVal.textContent = (+tempEl.value).toFixed(2).replace(/0$/, ''));
toppEl.addEventListener('input', () => toppVal.textContent = (+toppEl.value).toFixed(2).replace(/0$/, ''));

// ---------------------------------------------------------------- personas
function preset(slug) { return PERSONAS.find(p => p.slug === slug); }
function textFor(slug, fallback) {
  const p = preset(slug);
  if (!p) return fallback || '';
  return p.system;
}
function selectedSystem() {
  return promptEl.value.trim();
}
function refreshSysNote() {
  const p = preset(personaEl.value);
  const active = selectedSystem();
  let note = '';
  if (!active) note = '· none sent';
  else if (p && p.slug !== 'custom' && p.slug !== 'none') note = '· ' + p.name + ' preset';
  if (DEFAULTS.tools && p && p.slug === 'math') note += ' · calculator hint added';
  sysNote.textContent = note;
}
function selectPersona(slug) {
  personaEl.value = preset(slug) ? slug : 'custom';
  promptEl.value = textFor(personaEl.value, DEFAULTS.system);
  autoresize();
  refreshSysNote();
}
personaEl.addEventListener('change', () => {
  promptEl.value = textFor(personaEl.value, '');
  autoresize();
  refreshSysNote();
  inputEl.focus();
});
promptEl.addEventListener('input', () => {
  const p = preset(personaEl.value);
  // Editing the preset's text makes it a custom prompt — silently sending the
  // preset instead of what the box shows would be a lie.
  if (p && p.slug !== 'custom' && p.slug !== 'none' && promptEl.value.trim() !== p.system) {
    personaEl.value = 'custom';
  }
  refreshSysNote();
});

function scrollBottom() { chatEl.scrollTop = chatEl.scrollHeight; }

function setPerf(text) { perfEl.textContent = text; }
function updateTotals() {
  totalsEl.textContent = session.tokens + ' tokens · ' + session.replies + ' replies';
}

// ----------------------------------------------------- copy / clipboard
async function copyText(text, btn) {
  const label = btn.dataset.label || btn.textContent;
  btn.dataset.label = label;
  let ok = false;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      ok = true;
    }
  } catch (e) { ok = false; }
  if (!ok) {                      // http:// on the LAN is not a secure context
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      ok = document.execCommand('copy');
      ta.remove();
    } catch (e) { ok = false; }
  }
  btn.textContent = ok ? 'Copied ✓' : 'Copy failed';
  btn.classList.toggle('ok', ok);
  setTimeout(() => { btn.textContent = label; btn.classList.remove('ok'); }, 1300);
}

function extractCode(text) {
  const out = [];
  let m;
  const fenced = /```[^\\n]*\\n([\\s\\S]*?)```/g;
  while ((m = fenced.exec(text)) !== null) out.push(m[1].replace(/\\s+$/, ''));
  if (!out.length) {              // one-liner fences: ```code```
    const inline = /```([^`\\n]+?)```/g;
    while ((m = inline.exec(text)) !== null) out.push(m[1].trim());
  }
  return out.join('\\n\\n');
}

function miniButton(label, onClick) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'mini';
  b.textContent = label;
  b.addEventListener('click', onClick);
  return b;
}

function setBusy(b) {
  busy = b;
  sendBtn.textContent = b ? 'Stop' : 'Send';
  sendBtn.classList.toggle('stop', b);
  sendBtn.classList.toggle('primary', !b);
}

// The reasoning split, mirroring personas.split_think: the first <think> block is
// the trace, everything after its closing marker is the answer. Only the *first*
// block counts, and a missing </think> is reported as closed=false rather than
// guessed at — a half-written trace is a stream in progress, not an answer.
function stripThink(text) {
  // mirror of personas.strip_think: remove every block, and an unterminated one
  let out = String(text || '');
  while (out.indexOf(DEFAULTS.thinkOpen) >= 0) {
    const start = out.indexOf(DEFAULTS.thinkOpen);
    const end = out.indexOf(DEFAULTS.thinkClose, start + DEFAULTS.thinkOpen.length);
    if (end < 0) { out = out.slice(0, start); break; }
    out = out.slice(0, start) + out.slice(end + DEFAULTS.thinkClose.length);
  }
  return out.split(DEFAULTS.thinkClose).join('');
}

function splitThink(text) {
  // mirror of personas.split_think — the parity of the two is a unit test
  const raw = String(text || '');
  const open = raw.indexOf(DEFAULTS.thinkOpen);
  if (open < 0) return { reasoning: '', answer: raw.trim(), closed: true, hasReasoning: false };
  const prefix = raw.slice(0, open).trim();      // "Let me work this out." is kept
  const close = raw.indexOf(DEFAULTS.thinkClose, open + DEFAULTS.thinkOpen.length);
  const reasoning = (close < 0 ? raw.slice(open + DEFAULTS.thinkOpen.length)
                               : raw.slice(open + DEFAULTS.thinkOpen.length, close)).trim();
  const hasReasoning = reasoning.length > 0;
  if (close < 0) {
    return { reasoning: reasoning, answer: prefix, closed: false, hasReasoning: hasReasoning };
  }
  const answer = stripThink(raw.slice(close + DEFAULTS.thinkClose.length)).trim();
  return { reasoning: reasoning, closed: true, hasReasoning: hasReasoning,
           answer: (prefix ? prefix + ' ' + answer : answer).trim() };
}

function thinkingCard(thought) {
  const card = document.createElement('details');
  card.className = 'think';
  card.open = !thought.answer && !thought.closed;   // open while it is still thinking
  const summary = document.createElement('summary');
  const spin = document.createElement('span');
  spin.className = 'spin';
  const label = document.createElement('span');
  label.textContent = thought.closed
    ? '🧠 Reasoning · ' + thought.reasoning.length + ' chars'
    : '🧠 Thinking… · ' + thought.reasoning.length + ' chars';
  summary.appendChild(spin);
  summary.appendChild(label);
  const body = document.createElement('div');
  body.className = 'think-body';
  body.textContent = thought.reasoning || '…';
  card.appendChild(summary);
  card.appendChild(body);
  return card;
}

function toolCard(info) {
  const card = document.createElement('div');
  card.className = 'toolcard';
  const head = document.createElement('div');
  head.className = 'head';
  head.textContent = '🐍 Python Tool Execution'
    + (info.kind ? ' · ' + info.kind : '')
    + (info.source === 'model' ? ' (model-requested)' : ' (answered locally, no model tokens)');
  card.appendChild(head);
  const items = info.results || [info];
  for (const item of items) {
    const code = document.createElement('pre');
    code.className = 'code';
    code.textContent = item.code || item.expression || '';
    const out = document.createElement('pre');
    out.className = 'out';
    const tag = document.createElement('span');
    tag.className = 'outlabel';
    tag.textContent = '→ ';
    out.appendChild(tag);
    out.appendChild(document.createTextNode(item.result != null ? String(item.result)
                                                                : String(item.line || '')));
    card.appendChild(code);
    card.appendChild(out);
  }
  return card;
}

function sourcesCard(info) {
  const box = document.createElement('details');
  box.className = 'sources';
  const summary = document.createElement('summary');
  const hits = info.hits || [];
  summary.textContent = '📄 ' + ((info.doc && info.doc.name) || 'document')
    + ' · ' + hits.length + ' passage' + (hits.length === 1 ? '' : 's') + ' retrieved'
    + (info.grounding != null ? ' · ' + Math.round(info.grounding * 100) + '% of the reply from the document' : '');
  box.appendChild(summary);
  for (const hit of hits) {
    const row = document.createElement('div');
    row.className = 'hit';
    const tag = document.createElement('b');
    tag.textContent = '[' + hit.index + '] ';
    const score = document.createElement('span');
    score.className = 'score';
    score.textContent = ' · relevance ' + Number(hit.relevance).toFixed(2) + ' · ';
    row.appendChild(tag);
    row.appendChild(score);
    row.appendChild(document.createTextNode(hit.excerpt || ''));
    box.appendChild(row);
  }
  return box;
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
  const actions = document.createElement('div');
  actions.className = 'actions';
  const meta = document.createElement('div');
  meta.className = 'meta';
  const entry = { bubble, meta, actions, text: text || '', codeBtn: null,
                  answerBtn: null, cards: null, footer: null, docsInfo: null,
                  reasoning: '', answer: '' };
  actions.appendChild(miniButton(role === 'user' ? 'Copy' : 'Copy response',
                                () => copyText(entry.text, actions.firstChild)));
  if (role === 'assistant') {
    const codeBtn = miniButton('Copy code', () => copyText(extractCode(entry.text) || entry.text, codeBtn));
    codeBtn.style.display = 'none';        // only once the reply contains code
    entry.codeBtn = codeBtn;
    actions.appendChild(codeBtn);
    // "just the answer" only makes sense once there is a reasoning block to skip
    const answerBtn = miniButton('Copy answer', () => copyText(entry.answer || entry.text, answerBtn));
    answerBtn.style.display = 'none';
    entry.answerBtn = answerBtn;
    actions.appendChild(answerBtn);
  }
  box.appendChild(bubble);
  box.appendChild(actions);
  box.appendChild(meta);
  msg.appendChild(avatar);
  msg.appendChild(box);
  messagesEl.appendChild(msg);
  scrollBottom();
  return entry;
}

async function send(text) {
  text = (text ?? inputEl.value).trim();
  if (!text || busy) return;
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();
  inputEl.value = ''; autoresize();
  addMsg('user', text);
  const entry = addMsg('assistant', '');
  const { bubble, meta } = entry;
  const caret = document.createElement('span');
  caret.className = 'caret';
  bubble.appendChild(caret);
  setBusy(true);
  controller = new AbortController();
  const started = performance.now();
  let live = 0;
  let firstAt = null;
  setPerf('0 tok · …');
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        temperature: +tempEl.value,
        top_p: +toppEl.value,
        history: history,
        persona: personaEl.value,
        system: selectedSystem(),
      }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) throw new Error('HTTP ' + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    const paint = (streaming) => {
      caret.remove();
      // Everything below is built with textContent: model output and document
      // text become text nodes, never HTML — a document that contains <script>
      // is a document, not an injection.
      const thought = splitThink(entry.text);
      entry.reasoning = thought.reasoning;
      entry.answer = thought.answer;
      bubble.textContent = '';
      if (entry.cards) bubble.appendChild(entry.cards);
      if (thought.hasReasoning) bubble.appendChild(thinkingCard(thought));
      const body = thought.answer || (thought.closed && !thought.hasReasoning ? entry.text : '');
      if (body) bubble.appendChild(document.createTextNode(body));
      if (entry.footer) bubble.appendChild(entry.footer);
      if (entry.answerBtn) entry.answerBtn.style.display = thought.hasReasoning ? '' : 'none';
      bubble.appendChild(caret);
      scrollBottom();
    };
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
          entry.text += ev.text;
          live++;
          if (firstAt === null) firstAt = performance.now();
          const el = (performance.now() - started) / 1000;
          setPerf(live + ' tok · ' + (live / Math.max(el, 0.001)).toFixed(1) + ' tok/s · ' +
                  el.toFixed(1) + 's');
          if (ev.text.indexOf('```') >= 0 && entry.codeBtn) entry.codeBtn.style.display = '';
          paint(true);
        } else if (ev.type === 'docs') {
          // retrieval happened: remember it, show the sources with the answer
          entry.docsInfo = ev;
        } else if (ev.type === 'tool') {
          if (ev.kind && ev.kind !== 'math') {
            // python / date-time tool: a card with the source and its result
            entry.cards = entry.cards || document.createElement('div');
            entry.cards.appendChild(toolCard(ev));
            meta.dataset.tool = '🐍 ' + (ev.answer || ev.line || 'python tool');
          } else {
            const info = (ev.source === 'question' || ev.source === 'datetime')
              ? '🧮 ' + (ev.line || ev.answer) + ' — computed locally, no model tokens'
              : '🧮 calculator: ' + (ev.calls || []).join(', ');
            meta.dataset.tool = info;
          }
        } else if (ev.type === 'done') {
          entry.text = ev.reply || entry.text;
          entry.docsInfo = ev.retrieved && ev.retrieved.length
            ? { doc: ev.doc, hits: ev.retrieved, grounding: ev.grounding }
            : entry.docsInfo;
          if (entry.docsInfo && entry.docsInfo.hits && entry.docsInfo.hits.length) {
            entry.footer = entry.footer || document.createElement('div');
            entry.footer.textContent = '';
            entry.footer.appendChild(sourcesCard(entry.docsInfo));
          }
          paint(false);
          if (!entry.answer && !entry.reasoning) {
            bubble.textContent = entry.text || '(no reply — try a higher temperature)';
            caret.remove();
            bubble.appendChild(caret);
          }
          history.push({ user: text, assistant: entry.text });
          if (entry.codeBtn && extractCode(entry.text)) entry.codeBtn.style.display = '';
          // ---- latency monitor: server numbers, not client-side guesswork ----
          const parts = [];
          if (ev.source === 'calculator' || ev.source === 'python' || ev.source === 'datetime') {
            parts.push('0 model tokens · computed locally');
          } else {
            parts.push(ev.n_tokens + ' tokens');
            if (ev.tok_per_s) parts.push(ev.tok_per_s + ' tok/s');
            if (ev.elapsed_s != null) parts.push(ev.elapsed_s + ' s');
            if (ev.first_token_s != null) parts.push('first token ' + ev.first_token_s + ' s');
          }
          if (ev.prompt_tokens) parts.push('prompt ' + ev.prompt_tokens + ' tok');
          parts.push('temp ' + ev.temperature + ' · top-p ' + ev.top_p);
          if (ev.stop_reason === 'stopped') parts.push('stopped');
          else if (ev.stop_reason === 'context') parts.push('context limit');
          if (ev.turns_dropped > 0) parts.push('dropped ' + ev.turns_dropped + ' old turn(s)');
          const thought = splitThink(entry.text);
          if (thought.hasReasoning) {
            parts.push('🧠 reasoned ' + thought.reasoning.length + ' chars'
                       + (thought.closed ? '' : ' (unclosed)'));
          }
          if (entry.docsInfo && entry.docsInfo.hits) {
            parts.push('📄 ' + entry.docsInfo.hits.length + ' passage(s) from '
                       + ((entry.docsInfo.doc && entry.docsInfo.doc.name) || 'the document'));
          } else if (document_) {
            // A document is loaded but this question cleared no relevance floor,
            // so nothing was injected: say that here rather than leave the reader
            // to wonder why the answer ignored a file that is sitting right there.
            parts.push('📄 nothing relevant in ' + document_.name + ' — no context injected');
          }
          if (ev.grounding != null && ev.retrieved && ev.retrieved.length && ev.grounding < 0.34) {
            parts.push('⚠ the reply shares little wording with the document — check the sources');
          }
          if (meta.dataset.tool) parts.unshift(meta.dataset.tool);
          meta.textContent = parts.join(' · ');
          setPerf((ev.n_tokens || 0) + ' tok · ' + (ev.tok_per_s ? ev.tok_per_s + ' tok/s' : 'tool') +
                  (ev.elapsed_s != null ? ' · ' + ev.elapsed_s + 's' : ''));
          session.tokens += ev.n_tokens || 0;
          session.replies++;
          updateTotals();
        } else if (ev.type === 'error') {
          bubble.classList.add('error');
          bubble.textContent = '⚠ ' + ev.message;
          setPerf('error');
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      bubble.classList.add('error');
      bubble.textContent = '⚠ ' + (err.message || 'connection failed — is the server running?');
      setPerf('disconnected');
    } else {
      if (!entry.text) bubble.textContent = '';
      const el = (performance.now() - started) / 1000;
      setPerf('stopped after ' + live + ' tok · ' + el.toFixed(1) + 's');
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

// ------------------------------------------------- document upload (RAG)
// The browser reads the file and POSTs its text as JSON: no multipart parser in
// the stdlib, no file ever written to disk, and the server indexes it in memory.
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const docChip = document.getElementById('docChip');
const docName = document.getElementById('docName');
const docStats = document.getElementById('docStats');
const docNote = document.getElementById('docNote');

function setDoc(info, note, isError) {
  document_ = info || null;
  if (document_) {
    docName.textContent = document_.name;
    docStats.textContent = document_.chunks + ' chunks · ' + document_.chars.toLocaleString() + ' chars';
    docChip.style.display = '';
  } else {
    docChip.style.display = 'none';
  }
  docNote.textContent = note || '';
  docNote.classList.toggle('error', !!isError);
}

async function uploadFile(file) {
  if (!file) return;
  if (file.size > DEFAULTS.maxUploadMB * 1024 * 1024) {
    setDoc(document_, '⚠ ' + file.name + ' is larger than ' + DEFAULTS.maxUploadMB
                      + ' MB — split it or use the .txt/.md text', true);
    return;
  }
  setDoc(document_, '📄 indexing ' + file.name + '…', false);
  uploadBtn.disabled = true;
  try {
    const text = await file.text();
    const res = await fetch('/api/upload', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: file.name, text: text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
    setDoc(data, '✓ answers now come from ' + data.name + ' only', false);
  } catch (err) {
    setDoc(null, '⚠ ' + (err.message || 'upload failed'), true);
  } finally {
    uploadBtn.disabled = false;
    fileInput.value = '';
  }
}

uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => uploadFile(fileInput.files && fileInput.files[0]));
document.getElementById('docClear').addEventListener('click', async () => {
  try {
    await fetch('/api/docs/clear', { method: 'POST' });
    setDoc(null, 'document removed', false);
  } catch (err) {
    setDoc(document_, '⚠ could not clear the document', true);
  }
});
if (!DEFAULTS.rag) {
  document.getElementById('docCtl').style.display = 'none';
} else if (document_) {
  setDoc(document_, 'preloaded with --doc', false);
}

// Build the persona dropdown, then let /api/info (the server's own idea of the
// default persona) override the value baked into the page.
PERSONAS.forEach(p => personaEl.appendChild(new Option(p.label, p.slug)));
selectPersona(DEFAULTS.persona || 'custom');
updateTotals();

(async function boot() {
  try {
    const res = await fetch('/api/info');
    const info = await res.json();
    document.getElementById('statusDot').className = 'dot ok';
    document.getElementById('statusText').textContent = 'model ready';
    document.getElementById('subtitle').textContent =
      (info.params / 1e6).toFixed(2) + 'M params · ' + info.n_layer + ' layers · ctx ' +
      info.context + ' · vocab ' + info.vocab + ' · torch ' + info.torch;
    if (info.doc) setDoc(info.doc, '', false);
    if (info.defaults) {
      tempEl.value = info.defaults.temperature; tempEl.dispatchEvent(new Event('input'));
      toppEl.value = info.defaults.top_p; toppEl.dispatchEvent(new Event('input'));
    }
    // the server is the authority on the *default* persona (--persona/--system)
    const serverPreset = PERSONAS.find(p => p.label === info.persona || p.name === info.persona);
    if (serverPreset && !DEFAULTS.persona) selectPersona(serverPreset.slug);
    else if (!DEFAULTS.persona && info.system) {
      personaEl.value = 'custom';
      promptEl.value = info.system;
      refreshSysNote();
    }
    setPerf('idle');
  } catch (e) {
    document.getElementById('statusDot').className = 'dot bad';
    document.getElementById('statusText').textContent = 'server unreachable';
    setPerf('offline');
  }
})();
</script>
</body>
</html>
"""


# ------------------------------------------------------------- stdlib server
class Handler(BaseHTTPRequestHandler):
    """One process, one model: GETs serve the page, POST /api/chat streams NDJSON
    token events, POST /api/stop cancels the in-flight generation, POST
    /api/upload indexes a document the browser read (no multipart parsing: the
    page sends the file's text as JSON, which the stdlib can already read)."""
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

    def _read_body(self, limit: int = MAX_BODY_BYTES) -> bytes:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n > limit:
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
        elif self.path == "/api/docs":
            self._send_json({"doc": self.engine.doc_info(), "rag": self.engine.rag_on})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:             # noqa: N802
        if self.path == "/api/stop":
            self._send_json({"stopped": self.engine.stop_active()})
            return
        if self.path == "/api/docs/clear":
            self._send_json({"cleared": self.engine.clear_document()})
            return
        if self.path == "/api/upload":
            self._upload()
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
            payload.get("max_tokens"),
            system=payload.get("system"), persona=payload.get("persona"))
        try:
            for kind, data in gen:
                if isinstance(data, dict):       # done / tool payloads pass through
                    write(dict({"type": kind}, **data))
                elif kind == "token":
                    write({"type": "token", "text": data})
                else:                            # error: one human-readable string
                    write({"type": kind, "message": str(data)})
        except (BrokenPipeError, ConnectionResetError):
            pass                             # client left; .close() cancels the worker
        finally:
            gen.close()
            self.close_connection = True


    # -- document upload
    def _upload(self) -> None:
        """Index a document the browser read and POSTed as JSON.

        `multipart/form-data` is not parsed here on purpose: the stdlib has no
        multipart reader, the page can read the file with `file.text()`, and a
        request body of `{"name": …, "text": …}` is exactly as safe — the text
        never touches the filesystem. The size check is on the *text*, so a 3 MB
        file is refused with a sentence instead of a MemoryError.
        """
        if not self.engine.rag_on:
            self._send_json({"error": "document Q&A is disabled (--no-rag)"}, 403)
            return
        try:
            payload = json.loads(self._read_body(MAX_UPLOAD_BYTES * 3)
                                 .decode("utf-8", errors="replace"))
            if not isinstance(payload, dict):
                raise ValueError("JSON object expected")
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError("'text' must be a string")
            if not text.strip():
                raise ValueError("the document is empty")
            if len(text) > MAX_UPLOAD_BYTES:
                raise ValueError(f"document is larger than {MAX_UPLOAD_BYTES // (1 << 20)} MB")
            stats = self.engine.set_document(str(payload.get("name") or "document"), text)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
            return
        except rag.RagError as e:
            self._send_json({"error": str(e)}, 400)
            return
        self._send_json({"ok": True, **stats})


def render_index(args, engine: ChatEngine | None = None) -> str:
    """Fill the page's placeholders from the CLI defaults.

    Kept as a function (not inline in run_stdlib) so tests can render and assert
    on the real page — and so a missing placeholder is one obvious failure
    instead of a stray `__TOKEN__` in the browser.
    """
    value = (getattr(engine, "default_persona_value", None) if engine is not None
             else (args.system if getattr(args, "system", None) is not None
                   else getattr(args, "persona", None)
                   if getattr(args, "persona", None) is not None else personas.DEFAULT_SLUG))
    preset = personas.get(value)
    if preset is not None:
        slug = preset.slug
    elif not (value or "").strip() or str(value).strip().lower() == personas.NONE_SLUG:
        slug = personas.NONE_SLUG               # "" and "none" both mean: send nothing
    else:
        slug = personas.CUSTOM_SLUG
    _, system_text = personas.resolve(value)
    tools_on = (getattr(engine, "tools_on", None) if engine is not None
                else not getattr(args, "no_tools", False))
    rag_on = (getattr(engine, "rag_on", None) if engine is not None
              else not getattr(args, "no_rag", False))
    document = engine.doc_info() if engine is not None else None
    personas_json = json.dumps(personas.choices(), ensure_ascii=False).replace("<", "\\u003c")
    return (INDEX_HTML
            .replace("__APP_NAME__", APP_NAME)
            .replace("__TEMP__", f"{args.temp:g}")
            .replace("__TOPP__", f"{args.top_p:g}")
            .replace("__PERSONAS_JSON__", personas_json)
            .replace("__DEFAULT_PERSONA__", slug)
            .replace("__DEFAULT_SYSTEM_JSON__", json.dumps(system_text, ensure_ascii=False)
                     .replace("<", "\\u003c"))
            .replace("__DOC_JSON__", json.dumps(document, ensure_ascii=False)
                     .replace("<", "\\u003c"))
            .replace("__THINK_OPEN__", personas.THINK_OPEN)
            .replace("__THINK_CLOSE__", personas.THINK_CLOSE)
            .replace("__RAG__", "true" if rag_on else "false")
            .replace("__MAX_UPLOAD_MB__", f"{MAX_UPLOAD_BYTES // (1 << 20)}")
            .replace("__TOOLS__", "true" if tools_on else "false"))


def run_stdlib(engine: ChatEngine, args) -> None:
    Handler.engine = engine
    Handler.index_html = render_index(args, engine).encode("utf-8")
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
    doc = engine.doc_info()
    print(f"[web] document Q&A: {'on' if engine.rag_on else 'off'} "
          f"({rag.CHUNK_CHARS}-char chunks, in-memory BM25; --no-rag disables)"
          + (f"  loaded: {doc['name']} ({doc['chunks']} chunks)" if doc else ""))
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

    choices = personas.choices()
    label_to_slug = {p["label"]: p["slug"] for p in choices}
    default_label, default_system = personas.resolve(engine.default_persona_value)

    def respond(message, history, temperature, top_p, persona, custom_system):
        if not str(message).strip():
            yield ""
            return
        slug = label_to_slug.get(persona, personas.CUSTOM_SLUG)
        preset = personas.get(slug)
        custom = str(custom_system or "").strip()
        if preset is not None and (not custom or custom == preset.system):
            # dropdown says "Math Tutor" and the box was not edited: send the
            # slug, so the server can attach the calculator hint itself.
            system = None
        elif custom:
            system, slug = custom, personas.CUSTOM_SLUG   # free text wins
        else:
            system, slug = None, personas.NONE_SLUG       # empty box = no system turn
        acc = ""
        try:
            for kind, data in engine.stream_chat(message, gradio_history_to_pairs(history),
                                                 temperature, top_p,
                                                 system=system or None, persona=slug):
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
        # one value per input (message + additional_inputs), so gradio cannot
        # complain about a short row on one release and not another
        examples=[[question, args.temp, args.top_p, default_label, default_system]
                  for question in ("Hello! Who are you?",
                                   "Tell me a short story about a cat.",
                                   "What is 12 * 4?",
                                   "Why is the sky blue?")],
        additional_inputs=[
            gr.Slider(0.1, 1.5, value=args.temp, step=0.05, label="Temperature"),
            gr.Slider(0.1, 1.0, value=args.top_p, step=0.05, label="Top-P"),
            gr.Dropdown(choices=[p["label"] for p in choices], value=default_label,
                        label="System Prompt"),
            gr.Textbox(value=default_system, lines=2, label="Prompt text (editable)",
                       placeholder="Leave empty for no system prompt"),
        ],
        additional_inputs_accordion="⚙ Sampling & System Prompt",
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
    doc = engine.doc_info()
    print(f"[web] document Q&A: {'on' if engine.rag_on else 'off'} "
          f"({rag.CHUNK_CHARS}-char chunks, in-memory BM25; --no-rag disables)"
          + (f"  loaded: {doc['name']} ({doc['chunks']} chunks)" if doc else ""))
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
    ap.add_argument("--persona", default=None, metavar="NAME",
                    help="system prompt the page starts on: friendly (default), pirate, "
                         "math, storyteller, none")
    ap.add_argument("--system", default=None, metavar="TEXT",
                    help="free-text system prompt the page starts on, overriding --persona "
                         "('' for none); the dropdown can still change it per message")
    ap.add_argument("--no-tools", dest="no_tools", action="store_true",
                    help="disable the tool hooks (arithmetic goes back to the model, "
                         "and <<calc: …>> / <<tool: …>> markers are shown as written)")
    ap.add_argument("--doc", default=None, metavar="FILE",
                    help="preload a .txt/.md document for document Q&A (upload or clear "
                         "it from the page); --no-rag disables retrieval entirely")
    ap.add_argument("--no-rag", dest="no_rag", action="store_true",
                    help="disable document upload and retrieval")
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

    # `generate` is already imported at module level (stdlib-only: it is where
    # serialize_chat lives); everything below needs torch, which is deferred so
    # `--help` still works on a machine without it.
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
    print(f"[web] tokenizer {tok_dir}  (vocab {tok.vocab})")
    system_label, system_text = personas.resolve(
        args.system if args.system is not None else args.persona,
        tool_hint=engine.tools_on)
    n_system = len(tok.encode_with_specials(system_text, [IM_START, IM_END])) if system_text else 0
    print(f"[web] system prompt: {system_label}"
          + (f"  ({n_system} tokens)" if n_system else "  (none)"))
    print(f"[web] tool hooks: {'on' if engine.tools_on else 'off'} "
          f"(tools.py: calculator, python tool, date/time; --no-tools disables)", flush=True)
    if args.doc:
        doc_path = resolve(Path(args.doc))
        if not doc_path.exists():
            die(f"--doc {doc_path} does not exist.")
        try:
            stats = engine.set_document(doc_path.name, doc_path.read_text(encoding="utf-8",
                                                                         errors="replace"))
            print(f"[web] document '{stats['name']}': {stats['chunks']} chunks, "
                  f"{stats['chars']:,} chars, {stats['terms']:,} terms", flush=True)
        except (OSError, ValueError, rag.RagError) as e:
            die(f"--doc {doc_path}: {e}")

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
