"""
Inference only: load a trained checkpoint and write text on the CPU.

This is the file your laptop runs. Training never happens here — the checkpoint
is committed to the repository, so all you need is PyTorch (CPU build) and this
file. Deliberately **no numpy, no matplotlib, no regex requirement** (the
tokenizer falls back to stdlib `re`), and no GPU code path: what you measure
here is the model, not the harness.

    python generate.py --prompt "Once upon a time, there was a little girl named Lily" --n 150

The architecture is Llama-3 style (RMSNorm, RoPE, SwiGLU, SDPA — see
`03_model.py`), which matters here in one concrete way: there is no learned
position-embedding table, positions are rotated into q/k. A KV cache therefore
has to rotate each new token by its *absolute* position while leaving the
already-rotated cached keys alone; `03_model.py` does that, and
`tests/test_architecture.py` pins it.

Design notes, each of which is a real inference-time bug it avoids:

 * `torch.inference_mode()` + `eval()` + `requires_grad_(False)`: no autograd
   graph is built, dropout is off, and nothing can accidentally update weights.
 * KV-cached decoding (`06_generate.generate(use_cache=True)`): token t costs
   one attention step over the cache instead of recomputing the whole prefix,
   so 150 tokens are ~3x cheaper and grow linearly, not quadratically.
   tests/test_generate.py pins cached == uncached.
 * `weights_only=True` on load: a checkpoint is data. Unpickling arbitrary
   objects from a file you downloaded is exactly how "just run this model"
   becomes remote code execution. The checkpoint here is a plain state_dict.
 * config travels next to the weights (config.json / config.txt). A checkpoint
   whose architecture you have to guess is a checkpoint you cannot load; the
   mix-up shows up as a shape error, so it is checked *before* generating.
 * threads pinned to the physical core count: PyTorch defaults to 1 thread in
   some environments, which quietly makes CPU generation several times slower.

Chat mode additionally supports a **system prompt** (`--persona`, `--system`,
and `/persona` / `/system` inside the REPL), the **tool hooks** (`tools.py`:
calculator, python tool, date/time; `--no-tools` disables them), a **reasoning
view** (`<think> … </think>` blocks are printed as their own labelled section)
and **document Q&A** (`--doc notes.md`, via `rag.py`). All of it is shared with
`web_ui.py` — `personas.py` owns both halves of the chat protocol, `tools.py`
evaluates, `rag.py` retrieves — so the terminal and the browser send
byte-identical prompts.
"""
from __future__ import annotations

import argparse
import codecs
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:            # the stage files are importable by name
    sys.path.insert(0, str(HERE))

import personas                          # noqa: E402  (stdlib-only, torch-free)
import rag                               # noqa: E402
import tools                             # noqa: E402

IM_START, IM_END = personas.IM_START, personas.IM_END


def die(msg: str) -> None:
    print("\n[gen] ERROR: " + msg + "\n", file=sys.stderr)
    raise SystemExit(1)


def import_torch():
    import warnings
    try:
        with warnings.catch_warnings():
            # A torch install without numpy (the default from the CPU wheel index)
            # prints a scary "Failed to initialize NumPy" UserWarning at import.
            # Nothing on the inference path needs numpy, so it is noise, not news.
            warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
            import torch
    except ImportError:
        die("PyTorch is not installed for this Python interpreter.\n"
            "      Install the CPU build once (no CUDA, ~200 MB):\n"
            "        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
    return torch


def resolve(p: Path) -> Path:
    """Repo-relative paths must work no matter where Python was started from —
    double-clicking run.bat, a shortcut with a different "start in", or
    `python C:\\path\\to\\generate.py`. Only the *file's own* directory is a safe
    fallback, never the caller's cwd."""
    return p if p.is_absolute() or p.exists() else HERE / p


def find_config(ckpt: Path, explicit: str | None) -> Path:
    """The architecture is stored beside the weights; accept JSON or the
    trainer's plain-text variant, and say exactly what to pull if neither is
    there (a missing config looks like a corrupt checkpoint otherwise)."""
    if explicit:
        p = resolve(Path(explicit))
        if not p.exists():
            die(f"--config {p} does not exist.")
        return p
    # Fine-tuned variants can share a directory with the base checkpoint while
    # using an expanded vocabulary, so prefer a checkpoint-specific config.
    for name in (ckpt.stem + "_config.json", "config.json", "config.txt"):
        p = ckpt.parent / name
        if p.exists():
            return p
    die(f"no config.json / config.txt next to {ckpt}.\n"
        f"      The checkpoint and its config are a pair — fetch both, e.g.:  git pull")


def render(tok, ids: list[int]) -> tuple[str, int]:
    """Decode ids for display. `<|doc_end|>` is the document separator the model
    was trained on, not prose: show it as a paragraph break and report how many
    times it fired, rather than leaking a literal marker into the text."""
    special = {tok.token_to_id[s]: s for s in tok.special_tokens}
    if not special:
        return tok.decode(ids), 0
    chunks, run, n_special = [], [], 0
    for i in ids:
        if i in special:
            chunks.append(tok.decode(run))
            chunks.append("\n\n")
            run = []
            n_special += 1
        else:
            run.append(i)
    chunks.append(tok.decode(run))
    return "".join(chunks), n_special


class TokenStream:
    """Render byte-BPE tokens incrementally, buffering only incomplete UTF-8.

    Decoding each token separately would turn split curly quotes/emoji into
    replacement characters. Finalise at document boundaries just like render().

    `tool_filter` (optional) rewrites the text on its way out, which is where a
    model-emitted `<<calc: 5 + 5>>` becomes `10`. Everything written is also
    collected in `written`, so the caller can keep exactly what the user saw as
    the assistant turn for the next round. `display` (optional) takes over
    printing only — the chat REPL uses it to render a `<think>` block as a
    labelled reasoning section while `written` stays the raw reply, markers and
    all, which is what the next turn's prompt needs.
    """

    def __init__(self, tok, unicode_to_bytes, tool_filter=None, display=None):
        self.tok = tok
        self.to_bytes = unicode_to_bytes
        self.special = {tok.token_to_id[s] for s in tok.special_tokens + getattr(tok, "added_tokens", [])}
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.n_special = 0
        self.tool_filter = tool_filter
        self.display = display
        self.written: list[str] = []

    def _emit(self, text: str) -> None:
        self.written.append(text)
        if self.display is not None:
            self.display(text)
            return
        sys.stdout.write(text)
        sys.stdout.flush()

    def _write(self, text: str) -> None:
        if self.tool_filter is None:
            self._emit(text)                 # unchanged path when tools are off
            return
        text = self.tool_filter.feed(text)
        if text:
            self._emit(text)

    def __call__(self, token_id: int) -> None:
        if token_id in self.special:
            text = self.decoder.decode(b"", final=True) + "\n\n"
            self.decoder.reset()
            self.n_special += 1
        else:
            text = self.decoder.decode(self.to_bytes(self.tok.id_to_token[token_id]))
        self._write(text)

    def finish(self) -> None:
        self._write(self.decoder.decode(b"", final=True))
        self.decoder.reset()
        if self.tool_filter is not None:      # release anything the filter held back
            self._emit(self.tool_filter.finish())


def run_prompt(prompt, args, model, cfg, tok, tok_mod, gen_mod, device) -> None:
    prompt_ids = tok.encode(prompt) if prompt else []
    if not prompt_ids:
        if not tok.special_tokens:
            die("empty prompt, and this tokenizer has no special start token.")
        prompt_ids = [tok.token_to_id[tok.special_tokens[0]]]

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    params = gen_mod.SamplingParams(temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
                                    repeat_penalty=args.repeat_penalty, seed=seed)
    if not args.quiet:
        print(f"\n=== prompt ({len(prompt_ids)} tokens) ===\n{prompt}")
        if len(prompt_ids) >= cfg.block_size:
            print(f"[gen] using the last {cfg.block_size - 1} prompt tokens to fit context.")
        print("\n=== continuation ===", flush=True)

    stream = TokenStream(tok, tok_mod.unicode_to_bytes)
    # Preserve --quiet's full-text output, but never print the result twice.
    if args.quiet:
        for token_id in prompt_ids:
            stream(token_id)
    t0 = time.perf_counter()
    try:
        ids, tok_per_s, _ = gen_mod.generate(
            model, prompt_ids, args.n, params, use_cache=True, device=device,
            block_size=cfg.block_size, on_token=stream)
    finally:
        stream.finish()
        print(flush=True)
    elapsed = time.perf_counter() - t0
    # Returned ids include the original prompt, even if prefill was truncated.
    new_tokens = len(ids) - len(prompt_ids)
    if not args.quiet:
        print(f"\n[gen] {new_tokens} new tokens, {tok_per_s:,.0f} tok/s (incl. prefill), "
              f"{elapsed:.2f} s")
        if new_tokens < args.n:
            print(f"[gen] stopped after {new_tokens} of {args.n} tokens: hit the "
                  f"{cfg.block_size}-token context window. Shorten --prompt or --n.")
        if stream.n_special:
            print(f"[gen] the model emitted {stream.n_special} document separator token(s) "
                  f"(<|doc_end|>), shown as a blank line.")
        print(f"[gen] seed {seed}  (re-run with --seed {seed} to reproduce this text)")


def serialize_chat(system: str, turns, user: str) -> str:
    """The exact SFT template, plus an optional leading system turn.

    `system` is inserted as `<|im_start|>system\\n…<|im_end|>` before the first
    user turn. `web_ui.ChatEngine.serialize` calls this same function, so the
    browser and the terminal cannot format a conversation differently.
    """
    parts = [f"{IM_START}system\n{system}{IM_END}"] if system else []
    parts += [f"{IM_START}user\n{u}{IM_END}\n{IM_START}assistant\n{a}{IM_END}"
              for u, a in turns]
    parts.append(f"{IM_START}user\n{user}{IM_END}\n{IM_START}assistant\n")
    return "\n".join(parts)


def initial_system(args) -> tuple[str, str]:
    """Resolve `--persona` / `--system` (the latter wins) to `(label, text)`.

    Defaulting to the friendly persona matches the web UI's default selection;
    `--persona none` or `--system ""` sends no system block at all, which is the
    framing the checkpoint was actually fine-tuned on.
    """
    value = getattr(args, "system", None)
    if value is None:
        value = getattr(args, "persona", None)
    return personas.resolve(value, tool_hint=not getattr(args, "no_tools", False))


class ReasoningPrinter:
    """Print a streamed reply with its `<think>` trace in its own labelled block.

    The split is `personas.ThinkStreamFilter`, the same one the browser's
    `splitThink` mirrors, so the terminal shows the same two parts. No ANSI: a
    Windows console that drops escape codes would print `[2m` in the middle of a
    sentence, and a prefix costs nothing.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.filter = personas.ThinkStreamFilter()
        self.in_reasoning = False
        self.in_answer = False

    def write(self, kind: str, chunk: str) -> None:
        if not chunk:
            return
        if self.enabled and kind == "reasoning" and not self.in_reasoning:
            self.in_reasoning = True
            sys.stdout.write("\n🧠 reasoning> ")
        elif self.enabled and kind == "answer" and not self.in_answer:
            self.in_answer = True
            sys.stdout.write("\nAssistant> " if self.in_reasoning or self.in_answer else "")
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def __call__(self, text: str) -> None:
        for kind, chunk in self.filter.feed(text):
            self.write(kind, chunk)

    def finish(self) -> None:
        for kind, chunk in self.filter.finish():
            self.write(kind, chunk)
        sys.stdout.flush()


def load_document(path, args) -> "object | None":
    """Index `--doc FILE` for chat mode, with the same limits as the web UI."""
    if not path:
        return None
    target = resolve(Path(path))
    if not target.exists():
        die(f"--doc {target} does not exist.")
    try:
        index = rag.DocumentIndex.from_text(target.read_text(encoding="utf-8", errors="replace"),
                                           name=target.name)
    except (OSError, ValueError) as e:
        die(f"--doc {target}: {e}")
    if not args.quiet:
        stats = index.stats()
        print(f"[chat] document '{stats['name']}': {stats['chunks']} chunks, "
              f"{stats['chars']:,} chars, {stats['terms']:,} terms")
    return index


def document_system(index, question: str, system: str, cfg, tok, specials, args) -> tuple[str, list]:
    """`(system prompt with the retrieved context, hits)` — the same block the
    browser builds, sized with the same rule (`reserve = min(n, ctx // 3)`)."""
    if index is None:
        return system, []
    reserve = min(args.n, max(32, cfg.block_size // 3))
    cost = lambda text: len(tok.encode_with_specials(  # noqa: E731
        serialize_chat(text, [], question), specials))
    block, hits = index.context_block(question, cfg.block_size - reserve, cost, top_k=2)
    if not block:
        return system, hits
    return block + (f"\n\n{system}" if system else ""), hits


def chat_repl(args, model, cfg, tok, tok_mod, gen_mod, device) -> None:
    """Conversational REPL using the same serialization as SFT.

    Complete user/assistant turns are retained until the context fills. Older
    turns are removed as units, rather than slicing through a role marker — and
    the system block is the *last* thing to shrink, because it is the only part
    of the prompt the user did not type just now.

    Arithmetic never reaches the model: an explicit math question is answered by
    `tools.py`, and a `<<calc: …>>` marker the model does emit is substituted
    with the exact value while the answer streams.
    """
    try:
        end_id = tok.token_to_id[IM_END]
        tok.token_to_id[IM_START]
    except KeyError:
        die("chat tokenizer is missing <|im_start|>/<|im_end|>. "
            "Run 08_sft.py or pass --tokenizer tokenizer/chat.")
    specials = [IM_START, IM_END]
    tools_on = not getattr(args, "no_tools", False)
    label, system = initial_system(args)
    index = load_document(getattr(args, "doc", None), args)
    history: list[tuple[str, str]] = []
    if not args.quiet:
        print("[chat] Multi-turn chat mode. Type /exit or /quit to leave; /clear resets history.")
        print("[chat] /system <text> sets a system prompt, /persona <name> picks a preset "
              "(/persona lists them).")
        n_system = len(tok.encode_with_specials(system, specials)) if system else 0
        print(f"[chat] system prompt: {label}" + (f"  ({n_system} tokens)" if n_system
                                                  else "  (none)"))

    def encode(prompt_system: str, turns) -> list[int]:
        return tok.encode_with_specials(
            serialize_chat(prompt_system, turns, user), specials)

    while True:
        try:
            user = input("\nYou> " if not args.quiet else "")
            command = user.strip()
            lower = command.lower()
            if lower in {"/exit", "/quit", "exit", "quit"}:
                break
            if lower == "/clear":
                history.clear()
                if not args.quiet:
                    print("[chat] History cleared.")
                continue
            if lower == "/persona" or lower.startswith("/persona "):
                name = command[len("/persona"):].strip()
                if not name:
                    print("[chat] Presets: " + ", ".join(
                        f"{p.slug} ({p.label})" for p in personas.PERSONAS) +
                        f", none.  Current: {label}")
                    continue
                label, system = personas.resolve(name, tool_hint=tools_on)
                if not args.quiet:
                    print(f"[chat] system prompt: {label}")
                continue
            if lower == "/system" or lower.startswith("/system "):
                text = command[len("/system"):].strip()
                if not text:
                    print(f"[chat] system prompt ({label}): "
                          + (system if system else "<none>"))
                    continue
                label, system = personas.resolve(text, tool_hint=tools_on)
                if not args.quiet:
                    print(f"[chat] system prompt: {label}")
                continue
            if not user.strip():
                continue

            # ---- tool hook: anything a tool answers exactly never reaches the
            # model (arithmetic, word counts, reversals, dates). A maths question
            # inside prose is still answered by the model — it may be part of a
            # bigger question — and the exact value is appended afterwards.
            query = tools.find_tool_question(user) if tools_on else None
            if query is not None and query.bare:
                answer = query.answer         # "5 + 5 = 10", computed exactly
                if not args.quiet:
                    tag = "calc" if query.kind == "math" else query.source
                    print(f"[{tag}] {query.expression!r} -> {query.result.text} "
                          f"(computed locally; the model was not asked)")
                    print("Assistant> " + answer, flush=True)
                else:
                    print(answer, flush=True)
                history.append((user, answer))
                continue


            prompt_system, hits = document_system(index, user, system, cfg, tok, specials, args)
            if hits and not args.quiet:
                print(f"[rag] using {len(hits)} passage(s) from {index.name}: "
                      + ", ".join(f"#{hit.index} ({hit.relevance:.2f})" for hit in hits))
            prompt_ids = encode(prompt_system, history)
            # Leave useful room for an answer and discard oldest complete turns.
            reserve = min(args.n, max(32, cfg.block_size // 3))
            while history and len(prompt_ids) > cfg.block_size - reserve:
                history.pop(0)
                prompt_ids = encode(prompt_system, history)
            if prompt_system and len(prompt_ids) > cfg.block_size - reserve:
                # Even an empty history does not fit: shrink the system block
                # rather than let prefill silently drop it (and the question).
                prompt_system = personas.shrink(
                    prompt_system, lambda text: len(encode(text, history)),
                    cfg.block_size - reserve)
                prompt_ids = encode(prompt_system, history)
                if not args.quiet:
                    print(f"[chat] system prompt trimmed to "
                          f"{len(tok.encode_with_specials(prompt_system, specials))} tokens "
                          f"to fit the {cfg.block_size}-token context.")

            seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
            params = gen_mod.SamplingParams(temperature=args.temp, top_k=args.top_k,
                                            top_p=args.top_p,
                                            repeat_penalty=args.repeat_penalty, seed=seed)
            tool_filter = tools.ToolTextFilter() if tools_on else None
            printer = ReasoningPrinter(enabled=True)
            stream = TokenStream(tok, tok_mod.unicode_to_bytes, tool_filter,
                                 display=printer if not args.quiet else None)
            if not args.quiet:
                print("Assistant> ", end="", flush=True)

            # generate() reports EOS to callbacks by contract. Hide the marker
            # from the terminal while retaining it in ids for exact stopping.
            def emit(token_id):
                if token_id != end_id:
                    stream(token_id)

            try:
                # the return value is not needed: the assistant turn is what the
                # (tool-filtered) stream actually printed
                gen_mod.generate(
                    model, prompt_ids, args.n, params, use_cache=True, eos_id=end_id,
                    device=device, block_size=cfg.block_size, on_token=emit)
            finally:
                stream.finish()
                printer.finish()
                print(flush=True)
            if tool_filter is not None and tool_filter.calls:
                if not args.quiet:
                    print("[tool] substituted "
                          + ", ".join(c.line for c in tool_filter.calls) + " into the reply")
            # The assistant turn is what the user saw, tool results included.
            answer = "".join(stream.written).strip()
            if query is not None and not tools.value_in_text(query.result, answer):
                # The model answered a math question with a number of its own
                # invention: the exact value goes with it, every time, so the
                # user is never left with only the guess.
                extra = f"\n\n🧮 {query.answer}"
                answer += extra
                print(extra, flush=True)
            history.append((user, answer))
        except (EOFError, KeyboardInterrupt):
            print()
            break


def repl(args, model, cfg, tok, tok_mod, gen_mod, device) -> None:
    if not args.quiet:
        print("[gen] Interactive mode: each prompt starts a fresh continuation (no chat history).")
        print("[gen] Type /exit or /quit to leave; EOF or Ctrl+C also exits. Blank input is skipped.")
    while True:
        try:
            prompt = input("\nYou> " if not args.quiet else "")
            if prompt.strip().lower() in {"/exit", "/quit", "exit", "quit"}:
                break
            if not prompt.strip():
                continue
            run_prompt(prompt, args, model, cfg, tok, tok_mod, gen_mod, device)
        except (EOFError, KeyboardInterrupt):
            print()
            break


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate text from a trained checkpoint on the CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint (.pt); defaults to model_15m_chat.pt / model_15m_chat_int8.pt in --chat, "
                         "else model_best.pt")
    ap.add_argument("--config", default=None, help="default: checkpoint-specific config, then config.json/config.txt")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer directory; defaults to tokenizer/chat in --chat, else tokenizer")
    ap.add_argument("--quantized", "--int8", dest="quantized", action="store_true",
                    help="load an INT8-quantized checkpoint (written by quantize.py)")
    ap.add_argument("--prompt", default="Once upon a time, there was a little girl named Lily",
                    help="text to continue; '' starts from the model's document token")
    ap.add_argument("-n", "--n", "--tokens", dest="n", type=int, default=150,
                    help="number of new tokens to sample")
    ap.add_argument("--temp", "--temperature", dest="temp", type=float, default=0.8,
                    help="0 = greedy; 0.7-0.9 reads as prose for this model size")
    ap.add_argument("--top-k", type=int, default=40, help="0 disables the top-k cap")
    ap.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling; 1.0 disables")
    ap.add_argument("--repeat-penalty", "--repetition-penalty", type=float, default=1.15,
                    help="1.0 disables; small models loop without it")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: a random seed, which is printed so the run can be repeated")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    ap.add_argument("--quiet", action="store_true", help="print the generated text only")
    ap.add_argument("--doc", default=None, metavar="FILE",
                    help="chat mode: index a .txt/.md file and answer from it (rag.py), "
                         "reporting which passages were used")
    ap.add_argument("--interactive", action="store_true",
                    help="read prompts in a REPL without reloading weights; ignores --prompt")
    ap.add_argument("--chat", action="store_true",
                    help="instruction-tuned multi-turn chat REPL (implies --interactive)")
    ap.add_argument("--persona", default=None, metavar="NAME",
                    help="chat system prompt preset: friendly (default), pirate, math, "
                         "storyteller, none. Also settable with /persona in the REPL")
    ap.add_argument("--system", default=None, metavar="TEXT",
                    help="free-text system prompt, overriding --persona ('' for none). "
                         "/system TEXT changes it mid-conversation")
    ap.add_argument("--no-tools", dest="no_tools", action="store_true",
                    help="disable the calculator: arithmetic questions go to the model "
                         "and <<calc: …>> markers are shown as written")
    args = ap.parse_args()
    if args.chat:
        args.interactive = True
    # Default checkpoint selection: prefer the 15M chat checkpoint if present,
    # then fall back to the original 1.9M model_chat.pt. If --quantized, pick
    # the INT8 variant automatically.
    if args.ckpt is None:
        suffix = "_int8.pt" if args.quantized else ".pt"
        if args.chat:
            for cand in ("out/model_15m_chat", "out/model_chat"):
                p = resolve(Path(cand + suffix))
                if p.exists():
                    args.ckpt = cand + suffix
                    break
            else:
                args.ckpt = "out/model_chat" + suffix
        else:
            for cand in ("out/model_15m_best", "out/model_best"):
                p = resolve(Path(cand + suffix))
                if p.exists():
                    args.ckpt = cand + suffix
                    break
            else:
                args.ckpt = "out/model_best" + suffix
    if args.tokenizer is None:
        args.tokenizer = "tokenizer/chat" if args.chat else "tokenizer"
    if args.n < 0:
        ap.error("--n must be >= 0")

    torch = import_torch()
    from _stage import load as _load          # noqa: E402  (needs torch already importable)
    model_mod, gen_mod = _load("03_model"), _load("06_generate")
    tok_mod = _load("01_tokenizer")
    import quantize                           # noqa: E402
    try:
        gen_mod.SamplingParams(temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
                               repeat_penalty=args.repeat_penalty, seed=args.seed)
    except ValueError as e:
        ap.error(str(e))

    args.threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)             # belt and braces with inference_mode()
    dev = torch.device("cpu")                 # this file is CPU-only on purpose

    ckpt = resolve(Path(args.ckpt))
    if not ckpt.exists():
        die(f"checkpoint {ckpt} not found.\n"
            f"      It is committed to the repo (~15 MB for 15M model, ~8 MB for 1.9M model) "
            f"— run `git pull`, then retry.\n"
            f"      Or point at another one:  python generate.py --ckpt out/model_best.pt")
    cfg_path = find_config(ckpt, args.config)
    tok_dir = resolve(Path(args.tokenizer))
    if not (tok_dir / "bpe.pkl").exists():
        die(f"tokenizer {tok_dir}/bpe.pkl not found — run `git pull`, it is committed too.")

    # ---- load model: INT8-quantized path OR plain fp32 path ----
    # Auto-detect quantized checkpoints by magic key: a quantized file is a dict
    # with "magic"="quantized_v1"; a plain fp32 file is a raw state_dict. We do
    # this by reading the file once and branching, so torch.load is called
    # exactly one time regardless of which path we take.
    is_int8 = args.quantized
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not is_int8:
        is_int8 = isinstance(raw, dict) and raw.get("magic") == "quantized_v1"
    if is_int8:
        if not (isinstance(raw, dict) and raw.get("magic") == "quantized_v1"):
            die(f"{ckpt} was expected to be an INT8 quantized checkpoint but lacks the magic marker.")
        cfg_dict = raw["config"]
        cfg = model_mod.GPTConfig(**cfg_dict)
        model = model_mod.GPT(cfg)
        quantize.quantize_model(model, per_channel=raw.get("per_channel", True))
        model.load_state_dict(raw["state_dict"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        cfg = model_mod.GPTConfig.load(cfg_path)   # strict: unknown keys are an error, not a guess
        model = model_mod.GPT(cfg)
        state = raw
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            die(f"{ckpt} does not match {cfg_path}.\n      {e}\n"
                f"      They are a pair: pull both from the same commit.")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    tok = tok_mod.ByteLevelBPE.load(tok_dir)

    if not args.quiet:
        print(f"[gen] torch {torch.__version__}  device cpu  threads {args.threads}")
        qtag = "  INT8 quantized" if quantize.is_quantized(model) else ""
        print(f"[gen] checkpoint {ckpt}  ({model.num_params():,} params, "
              f"{ckpt.stat().st_size / 1e6:.1f} MB){qtag}")
        print(f"[gen] config {cfg_path}  ({cfg.n_layer} layers, {cfg.n_head} heads, "
              f"d_model {cfg.d_model}, context {cfg.block_size}, vocab {cfg.vocab_size})")
        print(f"[gen] arch   RMSNorm + RoPE(base {getattr(cfg, 'rope_base', 10000.0):g}) + SwiGLU + "
              f"scaled_dot_product_attention, pre-norm, tied embeddings")
        print(f"[gen] tokenizer {tok_dir}  (vocab {tok.vocab:,})")
        if getattr(tok_mod, "_regex", None) is None:
            print("[gen] note: `regex` is not installed, so the stdlib word-splitting pattern is "
                  "used (pip install regex for byte-identical prompt tokenisation).")

    if not args.chat and (args.persona is not None or args.system is not None or args.no_tools):
        print("[gen] note: --persona / --system / --no-tools only apply to chat mode "
              "(--chat).", file=sys.stderr)      # stderr: --quiet output stays clean

    if args.chat:
        chat_repl(args, model, cfg, tok, tok_mod, gen_mod, dev)
    elif args.interactive:
        repl(args, model, cfg, tok, tok_mod, gen_mod, dev)
    else:
        try:
            run_prompt(args.prompt, args, model, cfg, tok, tok_mod, gen_mod, dev)
        except KeyboardInterrupt:
            raise SystemExit(130)


if __name__ == "__main__":
    if not sys.stdout.isatty():               # redirected output on Windows may be cp1252
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass
    main()
