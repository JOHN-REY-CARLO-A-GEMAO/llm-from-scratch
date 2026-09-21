"""
System prompts ("personas") shared by `generate.py` and `web_ui.py`.

A conversation is sent to the model as

    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {message}<|im_end|>
    <|im_start|>assistant

and this module is the single place that builds the *system* half of it, so the
terminal REPL and the browser cannot drift apart.

## What a persona can and cannot do here

`07_make_sft.py` contains no system turn, so the system block is *conditioning
text* the 15 M-parameter checkpoint has never been trained to obey: expect a
change in tone and vocabulary, not a change in ability. The framing is exact and
tested; the obedience is not claimed. `--persona none` removes the block
entirely, which is what the model was actually trained on.

The module also owns the *assistant* half of that framing: the
`<think> … </think>` reasoning block (`split_think`, `ThinkStreamFilter`), so a
reasoning reply is parsed identically in the terminal and in the browser.

## Two deliberate choices

 * **Marker stripping.** `sanitize()` removes `<|im_start|>`, `<|im_end|>`,
   `<|doc_end|>` and `<|endoftext|>` from anything a human typed. Otherwise a
   "system prompt" is a way to close the system turn early and inject a fake
   `<|im_start|>user … <|im_start|>assistant` turn — the one part of a chat
   prompt that must always be assembled by the program, never by the text.
 * **Slugs.** Presets are addressed by a short stable id (`pirate`), so the wire
   format from the browser never depends on display text, emoji, or renaming a
   preset later.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

#: Every marker that must never survive inside human-typed text.
MARKERS = (IM_START, IM_END, "<|doc_end|>", "<|endoftext|>")

#: Hard cap on a system prompt. The shipped model has a 256-token context; a
#: "system prompt" that fills it leaves no room to answer, and the engine has to
#: drop the block entirely once it no longer fits (see `ChatEngine.build_prompt`).
MAX_SYSTEM_CHARS = 1200

#: Appended to the math tutor's prompt when the calculator hook is enabled.
CALC_HINT = ("You have a calculator. To compute something, write <<calc: 5 + 5>> "
             "and the exact result is substituted before the reply is shown.")

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

#: Hard cap on a rendered reasoning block. A 15 M model that never closes
#: `</think>` would otherwise stream its whole context window into one <details>.
MAX_THINK_CHARS = 4000

#: Appended to the reasoner persona (and the math tutor's when CoT is on).
COT_HINT = ("For a question that needs more than one step, work it out first "
            f"inside {THINK_OPEN} … {THINK_CLOSE}: number the steps, then give the "
            "answer. For a simple question, just answer.")

NONE_SLUG = "none"
CUSTOM_SLUG = "custom"


@dataclass(frozen=True)
class Persona:
    slug: str
    name: str
    emoji: str
    system: str
    hint: str = ""

    @property
    def label(self) -> str:
        return f"{self.emoji} {self.name}"

    def to_json(self) -> dict:
        """Wire/UI form: no hints, they are a server-side detail."""
        data = asdict(self)
        data.pop("hint", None)
        data["label"] = self.label
        return data


PERSONAS: tuple[Persona, ...] = (
    Persona("friendly", "Friendly Assistant", "🤖",
            "You are a warm, friendly assistant for children. Answer in short, "
            "clear sentences, stay kind and encouraging, and never invent facts."),
    Persona("pirate", "Pirate", "🏴‍☠️",
            "Talk like a pirate in everything you say. Say ahoy and arr, call the "
            "person matey, and call yourself a salty sea dog. Keep it short and "
            "cheerful."),
    Persona("math", "Math Tutor", "🧮",
            "Be precise and double check every math step before you answer. Show "
            "the calculation, state the exact result, and if you are unsure say so "
            "instead of guessing.",
            hint=CALC_HINT),
    Persona("storyteller", "Storyteller", "🏰",
            "Describe things with magic and wonder. Use vivid images, a little "
            "gentle fantasy and a touch of sparkle, as if telling a bedtime story."),
    Persona("reasoner", "Reasoner", "🧠",
            "Answer carefully. Think the problem through before you reply, check "
            "each fact you use, and say so plainly when you are unsure.",
            hint=COT_HINT),
)

DEFAULT_SLUG = "friendly"                  # the UI's default: Friendly Assistant

_BY_SLUG = {p.slug: p for p in PERSONAS}
_BY_NAME = {p.name.lower(): p for p in PERSONAS}


def sanitize(text: str, limit: int = MAX_SYSTEM_CHARS) -> str:
    """Strip role markers and clamp length; a system prompt is data, not framing.

    Line structure is kept (a custom prompt may be a bulleted list), whitespace
    inside a line is not.
    """
    out = str(text or "")
    for marker in MARKERS:
        out = out.replace(marker, "")
    lines = [" ".join(line.split()) for line in out.splitlines()]
    return "\n".join(line for line in lines if line)[:limit].strip()


def get(slug: str | None) -> Persona | None:
    """Look up a preset by slug ("pirate") or by name ("Pirate", "🏴‍☠️ Pirate").

    Matching is exact: arbitrary text that merely *mentions* a preset ("you are
    a math whiz") stays custom text, it does not silently become that preset.
    """
    if not slug:
        return None
    key = str(slug).strip().lower()
    if key in _BY_SLUG:
        return _BY_SLUG[key]
    key = key.lstrip("🤖🏴‍☠️🧮🏰🧠 ").strip().rstrip(".").strip()
    return _BY_NAME.get(key)


def resolve(value: str | None, *, tool_hint: bool = True) -> tuple[str, str]:
    """Map a slug, a preset name, or free text to `(label, system_prompt)`.

    `None` means "nothing was requested" and selects the default persona (the
    UI's default is Friendly Assistant). An empty string or `"none"` means *no
    system prompt* — the model then sees exactly the framing it was fine-tuned
    on. An unknown non-empty string is treated as a custom system prompt.
    """
    if value is None:
        value = DEFAULT_SLUG
    text = str(value or "").strip()
    if not text or text.lower() == NONE_SLUG:
        return "None", ""
    persona = get(text)
    if persona is not None:
        body = persona.system
        if tool_hint and persona.hint:
            body = f"{body}\n\n{persona.hint}"
        return persona.label, sanitize(body, MAX_SYSTEM_CHARS + 256)
    return "Custom", sanitize(text)


def shrink(system: str, cost: "Callable[[str], int]", budget: int) -> str:
    """Longest prefix of `system` whose encoded form still fits `budget` tokens.

    Called only when the history has already been emptied and the prompt is
    *still* too long: the alternative is prefill silently truncating the prompt
    from the left, which drops the system block *and* the question. `cost(text)`
    is supplied by the caller because only it knows the serialization and the
    tokenizer.
    """
    if not system or cost("") > budget:
        return ""
    low, high, best = 0, len(system), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = system[:middle].rstrip()
        if cost(candidate) <= budget:
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    return best


def default() -> Persona:
    return _BY_SLUG[DEFAULT_SLUG]


# ======================================================= reasoning protocol
# The other half of the chat framing: what the *assistant* is expected to write.
#
# `07_make_sft.py` fine-tunes the checkpoint to plan inside a `<think> … </think>`
# block on questions that need more than one step, and to answer directly on the
# simple ones. The browser (web_ui.py) and the terminal (generate.py) therefore
# both have to recognise that block — in streaming text, where the closing marker
# may not have been sampled yet — and both must split it the *same* way, or the
# same reply renders differently in two places.
#
# The markers are ordinary text, not added tokens: the chat tokenizer's added
# tokens are only `<|im_start|>`/`<|im_end|>`, and adding two more would change
# the vocabulary (4098 -> 4100) and invalidate every trained embedding for a
# string the byte-level BPE already encodes in a handful of tokens.
@dataclass(frozen=True)
class Thinking:
    """A model reply split into its reasoning trace and its answer.

    `closed` is False while `</think>` has not been sampled — the honest state
    for a stream in progress, and for a reply cut off by the context limit.
    """

    reasoning: str
    answer: str
    closed: bool

    @property
    def has_reasoning(self) -> bool:
        return bool(self.reasoning)

    def to_json(self) -> dict:
        data = asdict(self)
        data["has_reasoning"] = self.has_reasoning
        data["reasoning_chars"] = len(self.reasoning)
        data["answer_chars"] = len(self.answer)
        return data


def strip_think(text: str) -> str:
    """Remove every `<think>` block (and any dangling marker) from `text`."""
    out = str(text or "")
    while THINK_OPEN in out:
        start = out.find(THINK_OPEN)
        end = out.find(THINK_CLOSE, start + len(THINK_OPEN))
        if end < 0:
            out = out[:start]
            break
        out = out[:start] + out[end + len(THINK_CLOSE):]
    return out.replace(THINK_CLOSE, "")


def split_think(text: str) -> Thinking:
    """Split a reply into `(reasoning, answer, closed)`.

    Only the *first* block counts: a model that keeps thinking after answering
    is showing a formatting failure, not a second thought, and the answer keeps
    whatever it wrote after the first `</think>`. Without a `<think>` at all the
    whole reply is the answer, which is what a direct reply is.
    """
    raw = str(text or "")
    start = raw.find(THINK_OPEN)
    if start < 0:
        return Thinking("", raw.strip(), True)
    end = raw.find(THINK_CLOSE, start + len(THINK_OPEN))
    # Text before the opening marker is kept in the answer rather than thrown
    # away: a model that writes "Let me work this out. <think>…" has said
    # something, and silently dropping it would hide a formatting slip instead
    # of showing it.
    prefix = raw[:start].strip()
    if end < 0:
        return Thinking(raw[start + len(THINK_OPEN):].strip(), prefix, False)
    reasoning = raw[start + len(THINK_OPEN):end].strip()
    answer = strip_think(raw[end + len(THINK_CLOSE):]).strip()
    return Thinking(reasoning, f"{prefix} {answer}".strip() if prefix else answer, True)


def wrap_think(steps, answer: str) -> str:
    """Serialise a reasoning trace the way the SFT corpus writes it.

    Used by `07_make_sft.py` so the generator, the renderer and the tests cannot
    disagree about where the newlines go: the marker sits on its own line, the
    steps are numbered inside, and the answer starts on the line after
    `</think>`.
    """
    lines = [THINK_OPEN]
    for i, step in enumerate(steps, 1):
        step = " ".join(str(step).split())
        lines.append(f"{i}. {step}" if step else f"{i}.")
    lines.append(THINK_CLOSE)
    return "\n".join(lines) + "\n" + str(answer).strip()


def _trailing_marker(text: str, markers) -> int:
    """Length of the longest tail of `text` that could still become a marker."""
    length = 0
    for marker in markers:
        for size in range(1, min(len(marker), len(text)) + 1):
            if marker.startswith(text[-size:]):
                length = max(length, size)
    return length


class ThinkStreamFilter:
    """Split a *streamed* reply into `("reasoning"|"answer", text)` pieces.

    Implemented as the delta of two monotone streams rather than as a hand-rolled
    state machine: at every step the whole buffer so far is parsed with the same
    rules `split_think` uses, and only the part that has not been emitted yet is
    released. A second implementation of the same grammar is a second place for
    it to be subtly wrong — and the browser, the terminal and the tests would
    then disagree about the same reply.

    Tokens arrive as arbitrary fragments (`"<thi"` + `"nk>"`), so a tail that
    could still become a marker is held back (`_trailing_marker`) and released
    as soon as it is proven not to be one. That is the same trick
    `tools.ToolTextFilter` uses for `<<calc: …>>`.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.reasoning_chars = 0       # emitted so far
        self.answer_chars = 0
        self.used = False              # did a reasoning block appear at all?

    def _streams(self, final: bool) -> tuple[str, str]:
        """`(reasoning, answer)` as far as they are safe to show right now."""
        raw = self.buffer
        start = raw.find(THINK_OPEN)
        if start < 0:
            hold = 0 if final else _trailing_marker(raw, (THINK_OPEN,))
            answer = raw[:len(raw) - hold] if hold else raw
            return "", answer
        prefix = raw[:start]
        end = raw.find(THINK_CLOSE, start + len(THINK_OPEN))
        if end < 0:
            reasoning = raw[start + len(THINK_OPEN):]
            if not final:
                hold = _trailing_marker(reasoning, (THINK_CLOSE,))
                reasoning = reasoning[:len(reasoning) - hold] if hold else reasoning
            return reasoning, prefix
        reasoning = raw[start + len(THINK_OPEN):end]
        answer = prefix + strip_think(raw[end + len(THINK_CLOSE):])
        if not final:
            hold = _trailing_marker(answer, (THINK_OPEN,))
            answer = answer[:len(answer) - hold] if hold else answer
        self.used = True
        return reasoning, answer

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        if chunk:
            self.buffer += chunk
        reasoning, answer = self._streams(final=False)
        out: list[tuple[str, str]] = []
        fresh = reasoning[self.reasoning_chars:]
        if fresh.strip():
            self.used = True
        if fresh:
            self.reasoning_chars = len(reasoning)
            out.append(("reasoning", fresh))
        fresh = answer[self.answer_chars:]
        if fresh:
            self.answer_chars = len(answer)
            out.append(("answer", fresh))
        return out

    def finish(self) -> list[tuple[str, str]]:
        """Flush the held-back tail. An unclosed block stays reasoning: that is
        exactly what the reply contains."""
        reasoning, answer = self._streams(final=True)
        out: list[tuple[str, str]] = []
        fresh = reasoning[self.reasoning_chars:]
        if fresh.strip():
            self.used = True
        if fresh:
            self.reasoning_chars = len(reasoning)
            out.append(("reasoning", fresh))
        fresh = answer[self.answer_chars:]
        if fresh:
            self.answer_chars = len(answer)
            out.append(("answer", fresh))
        return out

    @property
    def reasoning(self) -> bool:
        return self.used


def choices() -> list[dict]:
    """Dropdown data for the web UI: the presets, plus the two non-presets."""
    out = [p.to_json() for p in PERSONAS]
    out.append({"slug": CUSTOM_SLUG, "name": "Custom…", "emoji": "✏️",
                "label": "✏️ Custom…", "system": ""})
    out.append({"slug": NONE_SLUG, "name": "None (no system prompt)", "emoji": "∅",
                "label": "∅ None (no system prompt)", "system": ""})
    return out
