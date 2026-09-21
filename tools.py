"""
Tool hook — a safe calculator, so the chat engine never *guesses* arithmetic.

A 15 M-parameter model trained on children's stories is not a calculator. Ask it
`5 + 5` and it produces a plausible-looking number, which is the worst kind of
wrong answer: confident, well-formed, and false. No amount of sampling tuning
fixes that, because arithmetic is not a language skill — it is the one part of
the reply a program can do exactly:

 * **Detection.** An explicit arithmetic question (`5 + 5`, `what is 12 * 4?`)
   is recognised in the user's message and answered by the calculator *before*
   the model is asked anything (see `ChatEngine` / `generate.py`'s chat REPL).
 * **Tool calls.** If the model writes `<<calc: 5 + 5>>`, the marker is replaced
   by the exact value — including when the marker arrives split across several
   generated tokens (`ToolTextFilter`).
 * **No `eval()`.** `eval("5+5")` happily executes
   `__import__("os").system(...)`, so the safe subset is re-implemented here on
   top of `ast`: numeric literals, `+ - * / // % **`, parentheses, unary sign.
   Everything else (names, calls, attributes, subscripts, comprehensions) is
   *rejected*, not sandboxed. Nothing in this module can touch the filesystem,
   the network or the interpreter state.
 * **Exact arithmetic.** Values are `fractions.Fraction`, seeded from the
   *decimal literal text*, so `0.1 + 0.2` is `0.3` and not `0.30000000000000004`.
   A float appears only when the final result is formatted.
 * **Bounded work.** Expression length, AST node count and exponent size are
   capped, so `9**9**9` is a rejected expression rather than an hour of CPU.

Everything here is stdlib-only and torch-free: `generate.py`, `web_ui.py` and
the tests share it.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from fractions import Fraction

MAX_EXPR_CHARS = 200        # a chat message is not an arithmetic program
MAX_AST_NODES = 64          # ...and neither is an expression
MAX_EXPONENT = 512          # |exponent| cap: 10**512 is fine, 9**9**9 is not
MAX_NESTING = 12


class CalcError(ValueError):
    """The text is not an arithmetic expression this module is willing to run."""


# ------------------------------------------------------------------ evaluation
_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod}


def _number(node: ast.Constant) -> Fraction:
    value = node.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalcError("only numbers are allowed")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CalcError("number is not finite")
        # Fraction(repr(x)) is the *decimal* the user typed (0.1 -> 1/10);
        # Fraction(x) would bake in the binary expansion (36028797018963...).
        return Fraction(repr(value))
    return Fraction(value)


def _eval(node: ast.AST, depth: int = 0) -> Fraction:
    """Walk the whitelisted subset of the AST. Unknown nodes raise CalcError."""
    if depth > MAX_NESTING:
        raise CalcError("expression is nested too deeply")
    if isinstance(node, ast.Expression):
        return _eval(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        return _number(node)
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, depth + 1)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise CalcError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, depth + 1)
        right = _eval(node.right, depth + 1)
        kind = type(node.op)
        if kind in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise CalcError("division by zero")
        if kind is ast.Pow:
            if right.denominator != 1:
                raise CalcError("fractional exponents are not supported")
            exponent = right.numerator
            if abs(exponent) > MAX_EXPONENT:
                raise CalcError(f"exponent larger than {MAX_EXPONENT} is not supported")
            if left == 0 and exponent < 0:
                raise CalcError("division by zero")
            return left ** exponent
        fn = _BIN_OPS.get(kind)
        if fn is None:
            raise CalcError("unsupported operator")
        return fn(left, right)
    raise CalcError(f"unsupported syntax: {type(node).__name__}")


def format_fraction(value: Fraction) -> str:
    """Render the exact rational in the shortest honest decimal form.

    Integers stay integers (`10`, not `10.0`), terminating decimals lose their
    trailing zeros (`0.3`, not `0.300000000000`), and non-terminating ones are
    rounded to 12 significant digits rather than claiming 17 digits of truth.
    """
    if value.denominator == 1:
        return str(value.numerator)
    magnitude = float(value)
    if not math.isfinite(magnitude):             # unreachable in practice
        return f"{value.numerator}/{value.denominator}"
    if magnitude != 0 and (abs(magnitude) >= 1e16 or abs(magnitude) < 1e-6):
        return f"{magnitude:.12g}"
    text = f"{magnitude:.12f}".rstrip("0").rstrip(".")
    return text or "0"


@dataclass(frozen=True)
class Calculation:
    """A successful tool run: the expression as written, and the exact value."""

    expression: str
    value: Fraction

    @property
    def text(self) -> str:
        return format_fraction(self.value)

    @property
    def line(self) -> str:
        return f"{self.expression} = {self.text}"

    def __str__(self) -> str:                    # pragma: no cover - trivia
        return self.line


def evaluate(expression: str) -> Calculation:
    """Evaluate an arithmetic expression exactly. Raises CalcError if unsafe."""
    text = " ".join(str(expression).strip().split())
    if not text:
        raise CalcError("empty expression")
    if len(text) > MAX_EXPR_CHARS:
        raise CalcError(f"expression longer than {MAX_EXPR_CHARS} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as e:
        raise CalcError(f"not a valid expression ({e.__class__.__name__})") from e
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise CalcError("expression is too long")
    return Calculation(text, _eval(tree))


def evaluate_quiet(expression: str) -> Calculation | None:
    """`evaluate`, but None instead of an exception — the streaming path has no
    second chance to decide what to do with a broken expression."""
    try:
        return evaluate(expression)
    except (CalcError, ArithmeticError, OverflowError, MemoryError, RecursionError):
        return None


# ------------------------------------------------------------------- detection
CALL_RE = re.compile(r"<<\s*calc\s*:\s*(?P<expr>[^<>]{1,%d}?)\s*>>" % MAX_EXPR_CHARS,
                     re.IGNORECASE)

# characters an arithmetic expression may consist of (after normalisation)
_EXPR_CHARS = set("0123456789.+-*/%^() \t")
_PUNCT_FIX = str.maketrans({"−": "-", "–": "-", "—": "-", "×": "*", "÷": "/",
                            "⋅": "*", "∗": "*", "／": "/", "％": "%"})
_WORD_OPS = (("multiplied by", "*"), ("divided by", "/"), ("modulo", "%"),
             ("plus", "+"), ("minus", "-"), ("times", "*"), ("mod", "%"))
_PERCENT_OF_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+", re.IGNORECASE)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_TRIGGER_RE = re.compile(r"\b(what\s+is|what's|whats|how\s+much|how\s+many|calculate|"
                         r"compute|evaluate|solve|sum\s+of|total\s+of|tell\s+me|"
                         r"answer)\b|=", re.IGNORECASE)


def normalize(text: str) -> str:
    """Fold typographic operators and number-words into plain ASCII math."""
    out = str(text).translate(_PUNCT_FIX)
    out = _PERCENT_OF_RE.sub(r"(\1/100)*", out)
    out = _THOUSANDS_RE.sub("", out)
    for word, symbol in _WORD_OPS:
        # word operators only count *between operands*: "the model" and
        # "5 times taller" must not turn into arithmetic.
        out = re.sub(r"(?<=\d)\s*%s\s*(?=[\d(])" % re.escape(word), f" {symbol} ", out,
                     flags=re.IGNORECASE)
    return out


def _has_operator(candidate: str) -> bool:
    return bool(re.search(r"[+*/%^]", candidate)) or bool(re.search(r"\d\s*-\s*", candidate))


def _trim(run: str) -> str:
    """Drop the punctuation prose glues to the edges of a number run."""
    text = run.strip()
    while text and (text[-1] in "+-*/%^." or
                    (text[-1] == "(" and text.count("(") > text.count(")"))):
        text = text[:-1].rstrip()
    while text and text[0] in ")/%^*.":
        text = text[1:].lstrip()
    return text


def _balance(text: str) -> str | None:
    """Close unclosed brackets — "what is (2 + 3 * 4" is still a question."""
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None                      # stray ")" : not our expression
    return text + ")" * max(0, depth)


def _runs(text: str):
    """Maximal arithmetic-ish substrings: anything that is not a digit,
    whitespace, operator or bracket separates them."""
    start = None
    for i, ch in enumerate(text):
        if ch in _EXPR_CHARS:
            if start is None:
                start = i
        elif start is not None:
            yield text[start:i]
            start = None
    if start is not None:
        yield text[start:]


@dataclass(frozen=True)
class MathQuery:
    """An arithmetic question found in a user message."""

    expression: str        # normalised, directly evaluable
    bare: bool             # the whole message *was* the expression ("5 + 5")
    calculation: Calculation

    @property
    def text(self) -> str:
        return self.calculation.text

    @property
    def answer(self) -> str:
        """The deterministic reply line, e.g. `5 + 5 = 10`."""
        return self.calculation.line


def _candidates(text: str):
    """Every substring of `text` that could be the arithmetic expression."""
    for run in _runs(text):
        trimmed = _trim(run)
        if not trimmed or not _has_operator(trimmed):
            continue
        for shape in (trimmed, trimmed.rstrip(")")):
            balanced = _balance(shape)
            if balanced and _has_operator(balanced):
                yield balanced


def find_math_question(message: str) -> MathQuery | None:
    """Detect an explicit arithmetic question ("5 + 5", "what is 12 * 4?").

    Two shapes qualify; everything else stays with the model:

    * **bare** — the message is nothing but the expression (`5 + 5`, `12 * 4 = ?`).
    * **asked** — the message contains a trigger ("what is", "calculate", `=`, …)
      *and* an evaluable number run. Without a trigger, prose that merely
      contains digits ("call me at 5 - 3pm") is left alone.
    """
    text = normalize(message)
    if not text.strip() or not re.search(r"\d", text):
        return None

    core = text.strip().rstrip("?=:.! \t").strip()   # "12 * 4 = ?" -> "12 * 4"
    if core and set(core) <= _EXPR_CHARS and _has_operator(core) and not _DATE_RE.match(core):
        for shape in (core, core.rstrip(")")):
            balanced = _balance(_trim(shape))
            if not balanced or not _has_operator(balanced):
                continue
            calc = evaluate_quiet(balanced)
            if calc is not None:
                return MathQuery(balanced, True, calc)

    if not _TRIGGER_RE.search(text):
        return None
    for candidate in _candidates(text):
        calc = evaluate_quiet(candidate)
        if calc is not None:
            return MathQuery(candidate, False, calc)
    return None


# ----------------------------------------------------- model-emitted tool calls
def value_in_text(calculation: Calculation, text: str) -> bool:
    """Is this exact result already stated in `text`?

    Used to decide whether a generated reply needs the computed line appended.
    Matches the full line (`12 * 4 = 48`) or the value as a standalone number
    (so `the answer is 48.` counts) — but not a substring of a longer number, or
    `1` would be "found" in `100` and the injection would be skipped.
    """
    if calculation.line in text:
        return True
    return re.search(r"(?<![\d.])" + re.escape(calculation.text) + r"(?![\d])", text) is not None


def replace_calls(text: str) -> tuple[str, list[Calculation]]:
    """Replace every complete `<<calc: ...>>` in `text` with its exact value.

    A marker whose expression cannot be evaluated is left exactly as written —
    visibly broken is better than silently wrong.
    """
    out: list[str] = []
    calls: list[Calculation] = []
    position = 0
    for match in CALL_RE.finditer(text):
        calc = evaluate_quiet(match.group("expr"))
        if calc is None:
            continue
        out.append(text[position:match.start()])
        out.append(calc.text)
        position = match.end()
        calls.append(calc)
    out.append(text[position:])
    return "".join(out), calls


class ToolTextFilter:
    """Substitute `<<calc: ...>>` in a token stream without ever showing a
    half-finished marker.

    Tokens arrive in arbitrary fragments, so `<<calc: 5 + 5>>` can be split as
    `"<<ca"` + `"lc: 5 + 5>>"`. Emitting each fragment as it arrives would flash
    raw marker text at the user and break the substitution, so a *potential*
    marker at the tail of the buffer is held back (at most one marker's worth of
    text) and released as soon as the call is complete or proven not to be one.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.calls: list[Calculation] = []
        self.failed: list[str] = []

    @staticmethod
    def _could_be_marker(tail: str) -> bool:
        """Could `tail` (starting at a `<` at the very end of the buffer) still
        become a `<<calc: …>>` call?

        One `<` qualifies only while nothing follows it — `<<` is the only real
        opener, and a lone `<` in prose ("3 < 5") must be released on the next
        character rather than held forever. After `<<` the keyword has to remain
        a prefix (`<<c`, `<<calc`, `<<CALC:`), and once the colon is in, the
        expression may still be open; the call only becomes impossible when it
        closes (`>>`) without matching or grows past a legal expression.
        """
        if not tail.startswith("<"):
            return False
        body = tail[1:]
        if not body:
            return True
        if not body.startswith("<"):
            return False
        body = body[1:]
        if ">>" in body:
            return False
        stripped = re.sub(r"\s+", "", body).lower()
        if len(stripped) <= len("calc:"):
            return "calc:".startswith(stripped)
        if not stripped.startswith("calc:"):
            return False
        return len(body) <= MAX_EXPR_CHARS + len("<<calc: >>")

    def _drain(self, final: bool) -> str:
        out: list[str] = []
        while True:
            match = CALL_RE.search(self.buffer)
            if match is None:
                break
            out.append(self.buffer[:match.start()])
            calc = evaluate_quiet(match.group("expr"))
            if calc is None:
                out.append(match.group(0))       # unparseable: show it verbatim
                self.failed.append(match.group("expr").strip())
            else:
                out.append(calc.text)
                self.calls.append(calc)
            self.buffer = self.buffer[match.end():]
        hold = 0
        if not final:
            index = self.buffer.rfind("<")     # a single "<" may still become "<<"
            while index > 0 and self.buffer[index - 1] == "<":
                index -= 1                     # ...so start at the whole "<" run
            if index >= 0 and self._could_be_marker(self.buffer[index:]):
                hold = len(self.buffer) - index
        if hold:
            out.append(self.buffer[:len(self.buffer) - hold])
            self.buffer = self.buffer[len(self.buffer) - hold:]
        else:
            out.append(self.buffer)
            self.buffer = ""
        return "".join(out)

    # -- API
    def feed(self, chunk: str) -> str:
        """Text that is safe to display right now (may be empty)."""
        if chunk:
            self.buffer += chunk
        return self._drain(final=False)

    def finish(self) -> str:
        """Flush everything. A marker cut off by the context limit is still
        evaluated if its expression parses — otherwise it is shown as written."""
        prefix = ""
        partial = re.search(r"<<\s*calc\s*:\s*(?P<expr>[^<>]*)$", self.buffer, re.IGNORECASE)
        if partial:
            calc = evaluate_quiet(partial.group("expr"))
            if calc is not None:
                self.calls.append(calc)
                prefix = self.buffer[:partial.start()] + calc.text
                self.buffer = ""
        return prefix + self._drain(final=True)

    @property
    def used(self) -> bool:
        return bool(self.calls)
