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
    key = key.lstrip("🤖🏴‍☠️🧮🏰 ").strip().rstrip(".").strip()
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


def choices() -> list[dict]:
    """Dropdown data for the web UI: the presets, plus the two non-presets."""
    out = [p.to_json() for p in PERSONAS]
    out.append({"slug": CUSTOM_SLUG, "name": "Custom…", "emoji": "✏️",
                "label": "✏️ Custom…", "system": ""})
    out.append({"slug": NONE_SLUG, "name": "None (no system prompt)", "emoji": "∅",
                "label": "∅ None (no system prompt)", "system": ""})
    return out
