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
import calendar
import datetime
import math
import operator
import re
from dataclasses import dataclass
from datetime import date, datetime as _datetime, timezone
from fractions import Fraction

MAX_EXPR_CHARS = 200        # a chat message is not an arithmetic program
MAX_AST_NODES = 64          # ...and neither is an expression
MAX_EXPONENT = 512          # |exponent| cap: 10**512 is fine, 9**9**9 is not
MAX_NESTING = 12

# The interpreter half of the tool (string / date / multi-line programs) gets
# its own, larger — but still bounded — budget. Every one of these is a real
# denial-of-service if it is missing: `"ab" * 10**9` is a 2 GB string.
MAX_TOOL_CHARS = 1200       # source length for `run_python`
MAX_TOOL_NODES = 240        # AST nodes in a tool program
MAX_RESULT_CHARS = 2000     # widest value we are willing to render
MAX_STRING_CHARS = 4000     # widest string an operation may build
MAX_REPEAT = 5000           # `"x" * n` and `[0] * n` cap
MAX_LIST_ITEMS = 500        # `range()` / comprehension output cap
MAX_PROGRAM_LINES = 24      # statements in one multi-line program
MAX_NUMBER_BITS = 4096      # ~1,234 decimal digits: `factorial(100000)` is 456,574


class CalcError(ValueError):
    """The text is not an expression this module is willing to run."""


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


def format_float(value: float) -> str:
    """Render a float honestly: 12 significant digits, no trailing zeros.

    `math.sqrt(2)` is not exactly 1.4142135623730951, and a line that claims it
    is has the same problem as `0.30000000000000004` has for `0.1 + 0.2` — the
    arithmetic is fine, the printed precision is a lie.
    """
    if not math.isfinite(value):
        return str(value)
    if value == int(value) and abs(value) < 1e16:
        return str(int(value))                   # 3.0 -> "3"
    text = f"{value:.12g}"
    return text


def format_value(value, kind: str = "number") -> str:
    """One display string per supported value type.

    Numbers keep the exact-rational rendering above; strings are quoted (so a
    result of `""` is visibly empty rather than invisible) and lists are bounded
    so a runaway comprehension cannot print a megabyte.
    """
    if kind == "string" and isinstance(value, str):
        text = value if len(value) <= MAX_RESULT_CHARS else value[:MAX_RESULT_CHARS] + "…"
        return repr(text)
    if isinstance(value, Fraction):
        return format_fraction(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_float(value)
    if isinstance(value, _datetime):
        text = value.isoformat(sep=" ")
    elif isinstance(value, date):
        text = value.isoformat()
    elif isinstance(value, (list, tuple)):
        text = repr(value)
    else:                                        # pragma: no cover - defensive
        text = str(value)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "…"
    return text


@dataclass(frozen=True)
class ToolResult:
    """A successful tool run: the source as written, and the value it produced.

    `kind` is `"number"` for arithmetic and one of `"string"`, `"list"`,
    `"boolean"`, `"date"`, `"datetime"` for the rest of the Python subset;
    `source` says *which* evaluator produced it (`"calc"` or `"python"`), because
    `word_count("a b")` is a python-tool result that happens to be a number and
    a UI that decides by value type gets the label wrong. The name `Calculation`
    is kept as an alias: the calculator is the origin of this type and every
    caller of it still works unchanged.
    """

    expression: str
    value: "Fraction | int | float | str | bool | list | date | datetime"
    kind: str = "number"
    source: str = "calc"                 # "calc" = arithmetic, "python" = the tool

    @property
    def text(self) -> str:
        return format_value(self.value, self.kind)

    @property
    def source_line(self) -> str:
        """The last line of the source — what a one-line label should name.

        For `rate = 8.5\\nhours = 7\\nrate * hours` the interesting line is
        `rate * hours`, not the whole program.
        """
        lines = [line.strip() for line in self.expression.strip().splitlines() if line.strip()]
        return lines[-1] if lines else self.expression.strip()

    @property
    def line(self) -> str:
        return f"{self.source_line} = {self.text}"

    def to_json(self) -> dict:
        return {"expression": self.expression, "result": self.text, "kind": self.kind,
                "value": self.text, "line": self.line, "source": self.source,
                "code": self.source_line}

    def __str__(self) -> str:                    # pragma: no cover - trivia
        return self.line


#: Backwards-compatible name: `evaluate()` and the calculator hook speak
#: `Calculation`, which is exactly a numeric `ToolResult`.
Calculation = ToolResult


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


# ============================================================ python tool
# A whitelisted *interpreter*, not a sandbox. `eval()`/`exec()` are never called:
# the AST is walked and anything that is not on the list below is rejected, the
# same policy the calculator uses. What is new is breadth — strings, date/time
# and short multi-line programs — so "how many words is this?" and "what is the
# date today?" have an exact answer instead of a sampled guess.
#
# The language, in full:
#
#   values     int, float, str, bool, list, date, datetime (numbers stay exact
#              `Fraction`s while they remain rational)
#   statements `name = <expr>` lines, ending in one `<expr>` — nothing else: no
#              `import`, no `def`, no loop, no `if`, no `try`, no `with`, no
#              `lambda`, no f-string, no comprehension
#   calls      a fixed table of functions (`word_count`, `reverse`, `now`, …)
#   methods    a fixed table of methods on str/list/date/datetime values
#   indexing   `s[0]`, `s[-1]`, `s[1:4]` with integer expressions
#   literals   numbers, strings, lists, tuples
#
# Everything else — undefined names, an attribute that is not in the method
# table (so `().__class__`, `s.__class__.__mro__` and every other escape hatch),
# comprehensions, `*args`, `**kwargs` — raises CalcError.
#
# `run_python` is the entry point; `evaluate()` above stays arithmetic-only, so
# the calculator's "nothing but numbers" contract is unchanged.

#: The clock. Indirected so tests (and any embedder) can freeze time: everything
#: date-related here reads the time from this one place, never from `datetime`
#: directly. UTC on purpose — "today" has to mean exactly one thing.
CLOCK = lambda: _datetime.now(timezone.utc)          # noqa: E731


class _Missing:
    """Sentinel for an omitted optional parameter (None is a real value)."""

    def __repr__(self) -> str:                   # pragma: no cover - trivia
        return "<missing>"


_MISSING = _Missing()


def _clock() -> _datetime:
    value = CLOCK()
    if not isinstance(value, _datetime):
        raise CalcError("the tool clock did not return a datetime")
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def now() -> _datetime:
    """The current instant (UTC)."""
    return _clock()


def today() -> date:
    """The current date (UTC)."""
    return _clock().date()


def utcnow() -> _datetime:
    """`now()`, spelled the way the stdlib used to spell it."""
    return _clock()


def _to_date(value) -> date:
    """Coerce a `date`/`datetime`/ISO string/`(y, m, d)` tuple to a `date`."""
    if isinstance(value, _datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as e:
            raise CalcError(f"not an ISO date: {value!r}") from e
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return date(int(value[0]), int(value[1]), int(value[2]))
        except (TypeError, ValueError) as e:
            raise CalcError(f"not a date: {value!r}") from e
    raise CalcError("expected a date (use date_of(...) first)")


def date_of(year, month=_MISSING, day=_MISSING) -> date:
    """`date_of(2026, 12, 25)` or `date_of("2026-12-25")`."""
    if month is _MISSING:
        return _to_date(year)
    if day is _MISSING:
        raise CalcError("date_of() needs a year, a month and a day")
    try:
        return date(int(year), int(month), int(day))
    except (TypeError, ValueError) as e:
        raise CalcError(f"not a valid date ({e.__class__.__name__})") from e


def days_between(first, second) -> int:
    """Whole days from `first` to `second` (negative when `second` is earlier)."""
    return (_to_date(second) - _to_date(first)).days


def days_until(target) -> int:
    """Whole days from today to `target` (negative once `target` has passed)."""
    return (_to_date(target) - today()).days


def weekday_name(value) -> str:
    """`weekday_name("2026-12-25")` -> `"Friday"`."""
    return _to_date(value).strftime("%A")


def days_in_month(year, month=_MISSING) -> int:
    """Days in a month; `days_in_month("2026-02-01")` also works."""
    if month is _MISSING:
        value = _to_date(year)
        year, month = value.year, value.month
    try:
        return calendar.monthrange(int(year), int(month))[1]
    except (TypeError, ValueError) as e:
        raise CalcError(f"not a valid month ({e.__class__.__name__})") from e


def word_count(text) -> int:
    """Words in `text`: runs of non-whitespace, i.e. `str.split()` rules."""
    return len(str(text).split())


def char_count(text, ignore_spaces: bool = True) -> int:
    """Characters in `text` (spaces excluded by default, like a word processor)."""
    body = str(text)
    return sum(1 for ch in body if not (ignore_spaces and ch.isspace()))


def reverse(text) -> str:
    """`reverse("hello")` -> `"olleh"`."""
    return str(text)[::-1]


def _bounded(value, what: str = "result"):
    """Refuse an operation that built something absurd, before it is rendered."""
    if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
        raise CalcError(f"{what} would be longer than {MAX_STRING_CHARS} characters")
    if isinstance(value, (list, tuple)) and len(value) > MAX_LIST_ITEMS:
        raise CalcError(f"{what} would hold more than {MAX_LIST_ITEMS} items")
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value.bit_length() > MAX_NUMBER_BITS:
        raise CalcError(f"{what} is a larger number than this tool will compute")
    if isinstance(value, Fraction) and (value.numerator.bit_length() > MAX_NUMBER_BITS
                                        or value.denominator.bit_length() > MAX_NUMBER_BITS):
        raise CalcError(f"{what} is a larger number than this tool will compute")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalcError(f"{what} is not a finite number")
    return value


def _str_arg(value) -> str:
    return value if isinstance(value, str) else str(value)


def _native(arg):
    """A whole `Fraction` becomes an `int` at a call boundary.

    `round(2.5, 6)` and `"7".zfill(3)` need a Python `int`, and `Fraction(6)`
    would raise `TypeError` — while every other value keeps its exact type.
    """
    if isinstance(arg, Fraction) and arg.denominator == 1:
        return int(arg)
    return arg


#: Constants visible to tool source. Kept apart from the functions so `pi` is a
#: value and `reverse` is a call, and neither can be shadowed by an assignment.
_CONSTANTS = {"pi": math.pi, "e": math.e}

#: Functions callable from tool source. Deliberately no `import`, no `open`, no
#: `getattr`, no `eval`: a missing entry here *is* the security policy.
_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len,
    "int": int, "float": float, "str": str, "list": list, "sorted": sorted,
    "word_count": word_count, "char_count": char_count, "reverse": reverse,
    "upper": lambda text: _str_arg(text).upper(),
    "lower": lambda text: _str_arg(text).lower(),
    "title": lambda text: _str_arg(text).title(),
    "capitalize": lambda text: _str_arg(text).capitalize(),
    "strip": lambda text: _str_arg(text).strip(),
    "split": lambda text, sep=_MISSING: (_str_arg(text).split() if sep is _MISSING
                                          else _str_arg(text).split(_str_arg(sep))),
    "join": lambda sep, items: _str_arg(sep).join(_str_arg(item) for item in items),
    "replace": lambda text, old, new: _str_arg(text).replace(_str_arg(old), _str_arg(new)),
    "startswith": lambda text, prefix: _str_arg(text).startswith(_str_arg(prefix)),
    "endswith": lambda text, suffix: _str_arg(text).endswith(_str_arg(suffix)),
    "count_occurrences": lambda text, needle: _str_arg(text).count(_str_arg(needle)),
    "now": now, "today": today, "utcnow": utcnow, "date_of": date_of,
    "days_between": days_between, "days_until": days_until,
    "weekday_name": weekday_name, "days_in_month": days_in_month,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
    "gcd": math.gcd,
}

#: Attribute *calls* allowed on a value, as `name: (min_args, max_args)`. A bare
#: attribute read is refused outright (see `_attribute`), so a method that is not
#: in this table cannot be reached at all.
_METHODS: "dict[type, dict[str, tuple[int, int]]]" = {
    str: {"upper": (0, 0), "lower": (0, 0), "title": (0, 0), "capitalize": (0, 0),
          "swapcase": (0, 0), "strip": (0, 1), "lstrip": (0, 1), "rstrip": (0, 1),
          "split": (0, 1), "splitlines": (0, 0), "replace": (2, 2),
          "count": (1, 1), "find": (1, 1), "startswith": (1, 1), "endswith": (1, 1),
          "isdigit": (0, 0), "isalpha": (0, 0), "isalnum": (0, 0), "isspace": (0, 0),
          "islower": (0, 0), "isupper": (0, 0), "zfill": (1, 1), "center": (1, 2),
          "ljust": (1, 2), "rjust": (1, 2), "join": (1, 1)},
    _datetime: {"strftime": (1, 1), "date": (0, 0), "weekday": (0, 0)},
    date: {"strftime": (1, 1), "weekday": (0, 0)},
    list: {"count": (1, 1), "index": (1, 1), "copy": (0, 0)},
}


def _type_name(value) -> str:
    return type(value).__name__


def _method(value, name: str, args: list):
    """Call one whitelisted method on `value` (arity is checked, not guessed)."""
    if name.startswith("_") or name in ("format", "format_map"):
        raise CalcError("names starting with '_' are not allowed")
    table = None
    for kind, methods in _METHODS.items():
        if isinstance(value, kind) and name in methods:
            table = methods
            break
    if table is None:
        raise CalcError(f"{_type_name(value)} has no allowed method '{name}'")
    low, high = table[name]
    if not low <= len(args) <= high:
        raise CalcError(f"{name}() takes {low} argument(s)" if low == high
                        else f"{name}() takes {low} to {high} arguments")
    if isinstance(value, str) and name == "join":
        return _bounded(value.join(_str_arg(item) for item in args[0]))
    try:
        return _bounded(getattr(value, name)(*[_native(arg) for arg in args]))
    except CalcError:
        raise
    except (TypeError, ValueError, AttributeError, OverflowError,
            ZeroDivisionError) as e:
        raise CalcError(f"{name}() failed ({e.__class__.__name__})") from e


def _constant(node: ast.Constant):
    value = node.value
    if isinstance(value, str):
        return _bounded(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _number(node)                     # exact Fraction, as in `_eval`
    raise CalcError("only numbers, strings and True/False are allowed")


def _index_value(value):
    """A subscript index, as a plain `int` — exact numbers arrive as Fraction."""
    if isinstance(value, bool):
        raise CalcError("index must be a whole number")
    if isinstance(value, Fraction):
        if value.denominator != 1:
            raise CalcError("index must be a whole number")
        return int(value)
    if isinstance(value, int):
        return value
    raise CalcError(f"index must be a whole number, not {_type_name(value)}")


def _subscript(node: ast.Subscript, depth: int, env: dict):
    value = _eval_tool(node.value, depth + 1, env)
    index = _eval_tool(node.slice, depth + 1, env)
    if not isinstance(index, slice):
        index = _index_value(index)
    if isinstance(value, (str, list, tuple)) and isinstance(index, (int, slice)):
        try:
            return _bounded(value[index])
        except IndexError as e:
            raise CalcError("index out of range") from e
    raise CalcError(f"cannot index {_type_name(value)} with {_type_name(index)}")


def _call(node: ast.Call, depth: int, env: dict):
    """A call is either a whitelisted function name or a whitelisted method."""
    if node.keywords:
        raise CalcError("keyword arguments are not allowed")
    if any(isinstance(arg, ast.Starred) for arg in node.args):
        raise CalcError("'*' unpacking is not allowed")
    args = [_native(_eval_tool(arg, depth + 1, env)) for arg in node.args]
    func = node.func
    if isinstance(func, ast.Attribute):
        # Any whitelisted expression may be the receiver (`a.b().c()` is fine):
        # what keeps this safe is `_method`, which resolves the attribute name in
        # a type-keyed table and refuses everything else.
        return _method(_eval_tool(func.value, depth + 1, env), func.attr, args)
    if not isinstance(func, ast.Name):
        raise CalcError("only direct calls are allowed")
    fn = _FUNCTIONS.get(func.id)
    if fn is None:
        raise CalcError(f"unknown function '{func.id}'")
    try:
        return _bounded(fn(*args))
    except CalcError:
        raise
    except (TypeError, ValueError, ArithmeticError, OverflowError,
            MemoryError, RecursionError) as e:
        raise CalcError(f"{func.id}() failed ({e.__class__.__name__})") from e


def _pow(left: Fraction, right: Fraction) -> Fraction:
    """Exact power with a bounded exponent (see `_bounded` for the size cap)."""
    if right.denominator != 1:
        raise CalcError("fractional exponents are not supported")
    exponent = right.numerator
    if abs(exponent) > MAX_EXPONENT:
        raise CalcError(f"exponent larger than {MAX_EXPONENT} is not supported")
    if left == 0 and exponent < 0:
        raise CalcError("division by zero")
    return left ** exponent


def _repeat(text: str, count: Fraction) -> str:
    if count.denominator != 1 or count < 0:
        raise CalcError("a string can only be repeated a whole, non-negative number of times")
    if int(count) > MAX_REPEAT or len(text) * int(count) > MAX_STRING_CHARS:
        raise CalcError(f"repeating a string more than {MAX_REPEAT} times is not allowed")
    return text * int(count)


def _binop(node: ast.BinOp, depth: int, env: dict):
    left = _eval_tool(node.left, depth + 1, env)
    right = _eval_tool(node.right, depth + 1, env)
    kind = type(node.op)
    if kind is ast.Add:
        if isinstance(left, str) and isinstance(right, str):
            return _bounded(left + right)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return _bounded(list(left) + list(right))
    if kind is ast.Mult:
        for text, count in ((left, right), (right, left)):
            if isinstance(text, str) and isinstance(count, (int, Fraction)):
                return _repeat(text, Fraction(count))
            if isinstance(text, (list, tuple)) and isinstance(count, (int, Fraction)):
                if int(count) > MAX_LIST_ITEMS:
                    raise CalcError(f"a list can hold at most {MAX_LIST_ITEMS} items")
                return _bounded(list(text) * int(count))
    if kind in (ast.Div, ast.FloorDiv, ast.Mod) and isinstance(right, Fraction) and right == 0:
        raise CalcError("division by zero")
    if kind is ast.Pow and isinstance(left, Fraction) and isinstance(right, Fraction):
        return _bounded(_pow(left, right))
    if isinstance(left, (int, float, Fraction)) and isinstance(right, (int, float, Fraction)):
        fn = _BIN_OPS.get(kind)
        if fn is None:
            raise CalcError("unsupported operator")
        try:
            return _bounded(fn(left, right))
        except ZeroDivisionError as e:
            raise CalcError("division by zero") from e
    raise CalcError(f"cannot combine {_type_name(left)} and {_type_name(right)}")


def _attribute(node: ast.Attribute, depth: int):
    """Attribute access: method *calls* only (see `_call`), never a value read."""
    raise CalcError(f"'{node.attr}' can only be used as a method call")


def _eval_tool(node: ast.AST, depth: int = 0, env: "dict | None" = None):
    """Walk the whitelisted subset of the AST for a tool program."""
    if depth > MAX_NESTING:
        raise CalcError("expression is nested too deeply")
    env = {} if env is None else env
    if isinstance(node, ast.Expression):
        return _eval_tool(node.body, depth + 1, env)
    if isinstance(node, ast.Constant):
        return _constant(node)
    if isinstance(node, ast.Name):
        if node.id.startswith("_"):
            raise CalcError("names starting with '_' are not allowed")
        if node.id in env:
            return env[node.id]
        if node.id in _FUNCTIONS:
            raise CalcError(f"'{node.id}' is a function — call it, e.g. {node.id}(…)")
        raise CalcError(f"unknown name '{node.id}'")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_tool(node.operand, depth + 1, env)
        if isinstance(node.op, ast.USub) and isinstance(operand, (int, float, Fraction)):
            return -operand
        if isinstance(node.op, ast.UAdd) and isinstance(operand, (int, float, Fraction)):
            return operand
        raise CalcError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        return _binop(node, depth, env)
    if isinstance(node, ast.Call):
        return _call(node, depth, env)
    if isinstance(node, ast.Subscript):
        return _subscript(node, depth, env)
    if isinstance(node, ast.Slice):
        return slice(*[_index_value(_eval_tool(part, depth + 1, env)) if part else None
                       for part in (node.lower, node.upper, node.step)])
    if isinstance(node, ast.List):
        return _bounded([_eval_tool(item, depth + 1, env) for item in node.elts])
    if isinstance(node, ast.Tuple):
        return _bounded(tuple(_eval_tool(item, depth + 1, env) for item in node.elts))
    if isinstance(node, ast.Attribute):
        return _attribute(node, depth + 1)
    raise CalcError(f"unsupported syntax: {type(node).__name__}")


def _kind_of(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Fraction):
        return "number"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, _datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, str):
        return "string"
    return "value"


def run_python(source: str) -> ToolResult:
    """Run a whitelisted Python program and return its value.

        run_python("word_count('the quick brown fox')")     # 7
        run_python("rate = 8.5\\nhours = 7\\nrate * hours")   # 59.5
        run_python("weekday_name('2026-12-25')")            # 'Friday'

    The program is one or more `name = expression` lines followed by a final
    expression (that last value is the result). Anything outside the subset in
    the section comment raises `CalcError` — this is a whitelist, not a sandbox.
    """
    text = str(source).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")
             if line.strip() and not line.lstrip().startswith("#")]
    text = "\n".join(lines).strip()
    if not text:
        raise CalcError("empty program")
    if len(text) > MAX_TOOL_CHARS:
        raise CalcError(f"program longer than {MAX_TOOL_CHARS} characters")
    try:
        tree = ast.parse(text, mode="exec")
    except (SyntaxError, ValueError) as e:
        raise CalcError(f"not a valid expression ({e.__class__.__name__})") from e
    if sum(1 for _ in ast.walk(tree)) > MAX_TOOL_NODES:
        raise CalcError("program is too long")
    if len(tree.body) > MAX_PROGRAM_LINES:
        raise CalcError(f"more than {MAX_PROGRAM_LINES} lines is not a chat expression")

    env: dict = {}
    value = None
    for position, statement in enumerate(tree.body):
        last = position == len(tree.body) - 1
        if isinstance(statement, ast.Expr):
            value = _eval_tool(statement.value, 0, {**_CONSTANTS, **env})
        elif isinstance(statement, ast.Assign) and not last and \
                len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            target = statement.targets[0].id
            if target.startswith("_") or target in _FUNCTIONS or target in _CONSTANTS:
                raise CalcError(f"cannot use the name '{target}'")
            env[target] = _eval_tool(statement.value, 0, {**_CONSTANTS, **env})
        else:
            raise CalcError("only 'name = value' lines and a final expression are allowed")
    if value is None:                            # pragma: no cover - guarded above
        raise CalcError("a program must end with an expression")
    return ToolResult(text, value, _kind_of(value), source="python")


def run_python_quiet(source: str) -> ToolResult | None:
    """`run_python`, but None instead of an exception (the streaming path has no
    second chance to decide what to do with a broken program)."""
    try:
        return run_python(source)
    except (CalcError, ArithmeticError, OverflowError, MemoryError, RecursionError):
        return None


# ------------------------------------------------------------------- detection
CALL_RE = re.compile(r"<<\s*calc\s*:\s*(?P<expr>[^<>]{1,%d}?)\s*>>" % MAX_EXPR_CHARS,
                     re.IGNORECASE)
#: The general tool marker. Newlines are allowed (multi-line programs), `<`/`>`
#: are not: they would be ambiguous with the marker delimiters themselves.
PY_CALL_RE = re.compile(r"<<\s*(?:tool|py|python)\s*:\s*(?P<expr>[^<>]{1,%d}?)\s*>>"
                        % MAX_TOOL_CHARS, re.IGNORECASE | re.DOTALL)
#: Either marker, for scanning a finished string.
ANY_CALL_RE = re.compile(r"<<\s*(?P<kind>calc|tool|py|python)\s*:\s*(?P<expr>[^<>]{1,%d}?)\s*>>"
                         % MAX_TOOL_CHARS, re.IGNORECASE | re.DOTALL)

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
        # "how many days until 2026-12-25" is a calendar question: `2026-12-25`
        # must never be read as `2026 - 12 - 25` (see `find_datetime_question`).
        if _DATE_RE.match(candidate.strip()):
            continue
        calc = evaluate_quiet(candidate)
        if calc is not None:
            return MathQuery(candidate, False, calc)
    return None


# ------------------------------------------------- string / date questions
# The calculator answers arithmetic; these two answer the questions arithmetic
# cannot: "how many words is this?" and "what is the date today?". Both are
# computed, not sampled, and both go through `run_python` so the reply's number
# comes from the same interpreter the `<<tool: …>>` marker uses.

_QUOTED_RE = re.compile(r"[\"'“”‘’`](?P<body>[^\"'“”‘’`]{1,400})[\"'“”‘’`]")
#: `the string hello`, `this text: hi`, `my name Bob` — an *explicit* operand
#: introducer. Without one (and without quotes or a colon) "reverse the
#: argument" stays prose: guessing which words are the operand is how a
#: deterministic tool starts answering questions nobody asked.
_OPERAND_NOUNS = r"(?:strings?|text|words?|sentences?|phrases?|names?|inputs?|lines?|titles?)"
_INTRO_RE = re.compile(rf"^\s*(?:the|this|that|my|following|whole)?\s*{_OPERAND_NOUNS}\s*[:=\-—]?\s*",
                       re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"^\s*(?:the|this|that|my|following|whole)?\s*[:=\-—]\s*",
                            re.IGNORECASE)
_TRAILING_RE = re.compile(r"[\s:;,.!?=\-—]+$")
#: Leading words that belong to the question, not the operand
#: ("how many words **are in the** quick brown fox").
_FILLER_WORDS = frozenset("the this that my a an following in of is are was were "
                          "to for from me please does do it has have count".split())
_EDGE_PUNCT = " '\"“”‘’`:,;.-—"

_WORDS_RE = re.compile(r"\b(?:how\s+many\s+words|number\s+of\s+words|word\s+count|"
                       r"count\s+the\s+words|count\s+words)\b", re.IGNORECASE)
_CHARS_RE = re.compile(r"\b(?:how\s+many\s+(?:characters|letters|chars)|number\s+of\s+"
                       r"(?:characters|letters)|character\s+count|count\s+the\s+"
                       r"(?:characters|letters))\b", re.IGNORECASE)
_REVERSE_RE = re.compile(r"\b(?:reverse|reversed|backwards|backward|flip|mirror)\b",
                         re.IGNORECASE)
_CASE_RE = re.compile(r"\b(?:uppercase|upper\s+case|capitali[sz]e|to\s+upper|"
                      r"lowercase|lower\s+case|to\s+lower)\b", re.IGNORECASE)
_STOP = frozenset({"a", "an", "the", "this", "that", "these", "those", "it", "is",
                   "are", "and", "or", "of", "in", "to", "for", "on", "at", "by",
                   "with", "as", "be", "was", "were", "do", "does", "did", "how",
                   "many", "much", "what", "which", "who", "please", "count", "tell",
                   "me", "my", "i", "you", "your", "there", "here"})


def _clean_tail(tail: str, strip_filler: bool) -> str:
    """Trim punctuation, and (optionally) the question's own leading words.

    Word-by-word, never a regex over the raw text: a pattern that can match `a`
    at the start of a word turns "the argument" into "rgument", which is exactly
    the kind of quiet wrong answer this module exists to prevent.
    """
    text = _TRAILING_RE.sub("", tail.strip()).strip()
    if strip_filler:
        words = text.split()
        while words and words[0].strip(_EDGE_PUNCT).lower() in _FILLER_WORDS:
            words.pop(0)
        text = " ".join(words)
    return text.strip(_EDGE_PUNCT).strip()


def _payload(message: str, match: re.Match | None,
             require_operand: bool = False) -> str:
    """The text a string question is *about*.

    Precedence: a quoted span (unambiguous), then a separator (`reverse: hello`),
    then an explicit introducer (`reverse the string hello`). With
    `require_operand` and none of those, the message is prose — "reverse the
    argument" is a sentence about arguments, not a request to reverse the string
    `argument`, and answering it would be a confident non-sequitur.
    """
    quoted = _QUOTED_RE.findall(message)
    if quoted:
        return max(quoted, key=len).strip()
    if match is None:
        return ""
    tail = message[match.end():]
    stripped = _SEPARATOR_RE.sub("", tail)
    if stripped != tail:
        return _clean_tail(stripped, strip_filler=False)
    intro = _INTRO_RE.match(tail)
    if intro is not None and intro.end() > 0:
        rest = _clean_tail(tail[intro.end():], strip_filler=False)
        if rest:
            return rest
    if require_operand:
        return ""
    return _clean_tail(tail, strip_filler=True)


def _shorten(text: str, limit: int = 80) -> str:
    """A quoted payload, short enough for a one-line answer.

    Named `_shorten` and not `_trim`: that name already belongs to the arithmetic
    side of this module, and two module-level functions with one name means
    whichever is defined last silently wins for both callers.
    """
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _string_query(kind: str, payload: str) -> ToolQuery | None:
    """Build the query for one string intent, or None if the payload is empty."""
    if not payload or not re.search(r"[0-9A-Za-z]", payload):
        return None
    literal = repr(payload)
    if kind == "words":
        count = word_count(payload)
        source = f"word_count({literal})"
        result = ToolResult(source, Fraction(count), "number")
        return ToolQuery("string", source, result,
                         f'"{_shorten(payload)}" has {count} '
                         f"{'word' if count == 1 else 'words'}.")
    if kind == "chars":
        count = char_count(payload)
        source = f"char_count({literal})"
        result = ToolResult(source, Fraction(count), "number")
        return ToolQuery("string", source, result,
                         f'"{_shorten(payload)}" has {count} characters, not counting spaces.')
    if kind == "reverse":
        result = ToolResult(f"reverse({literal})", payload[::-1], "string")
        return ToolQuery("string", result.expression, result,
                         f'"{_shorten(payload)}" reversed is "{_shorten(result.value)}".')
    if kind == "upper":
        result = ToolResult(f"upper({literal})", payload.upper(), "string")
        return ToolQuery("string", result.expression, result,
                         f'"{_shorten(payload)}" in capitals is "{_shorten(result.value)}".')
    if kind == "lower":
        result = ToolResult(f"lower({literal})", payload.lower(), "string")
        return ToolQuery("string", result.expression, result,
                         f'"{_shorten(payload)}" in lower case is "{_shorten(result.value)}".')
    return None                                             # pragma: no cover


def find_string_question(message: str) -> ToolQuery | None:
    """Detect a word/character count, a reversal, or a case change.

    Detection is deliberately narrow and the order fixed: a message that asks for
    two things at once is not a string question, it is prose.
    """
    text = str(message or "")
    if not text.strip():
        return None
    for kind, pattern in (("words", _WORDS_RE), ("chars", _CHARS_RE)):
        match = pattern.search(text)
        if match:
            return _string_query(kind, _payload(text, match))
    match = _REVERSE_RE.search(text)
    if match:
        hit = _string_query("reverse", _payload(text, match, require_operand=True))
        if hit is not None:
            return hit
    match = _CASE_RE.search(text)
    if match:
        kind = "upper" if re.search(r"upper|capital", match.group(0), re.IGNORECASE) else "lower"
        hit = _string_query(kind, _payload(text, match, require_operand=True))
        if hit is not None:
            return hit
    return None


_ISO = r"\d{4}-\d{2}-\d{2}"
_TODAY_RE = re.compile(r"\b(?:today'?s\s+date|what(?:'s|\s+is)?\s+(?:the\s+)?(?:date|day\s+"
                       r"(?:today|now|is\s+it))\b|what(?:'s|\s+is)?\s+the\s+date|"
                       r"(?:what|which)\s+day\s+is\s+(?:it|today)\b|date\s+today|"
                       r"current\s+date|what\s+is\s+today)\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\b(?:what\s+time\s+is\s+it\b|what(?:'s|\s+is)\s+the\s+time\b"
                      r"(?!\s+(?:complexity|zone|difference|limit|it\s+takes))|"
                      r"current\s+time\b|time\s+now\b)", re.IGNORECASE)
_WEEKDAY_RE = re.compile(r"\b(?:what|which)\s+day(?:\s+of\s+the\s+week)?\s+(?:is|was|will)\s+"
                         rf"(?:it\s+on\s+)?(?P<date>{_ISO})\b", re.IGNORECASE)
_DAYS_UNTIL_RE = re.compile(rf"\bhow\s+many\s+days\s+(?P<dir>until|till|to|before|since|ago)\s+"
                            rf"(?P<date>{_ISO})\b", re.IGNORECASE)
_DAYS_BETWEEN_RE = re.compile(rf"\bhow\s+many\s+days\s+(?:between|from)\s+(?P<a>{_ISO})\s+"
                              rf"(?:and|to)\s+(?P<b>{_ISO})\b", re.IGNORECASE)
_MONTH_DAYS_RE = re.compile(rf"\bhow\s+many\s+days\s+(?:are\s+)?(?:in)\s+(?P<date>{_ISO})\b"
                            r"|\bhow\s+many\s+days\s+(?:are\s+)?in\s+(?P<month>January|February|"
                            r"March|April|May|June|July|August|September|October|November|"
                            r"December)\s*(?P<year>\d{4})?\b", re.IGNORECASE)

_MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}


def _date_query(source: str, value, answer: str) -> ToolQuery:
    return ToolQuery("datetime", source, ToolResult(source, value, _kind_of(value)), answer)


def find_datetime_question(message: str) -> ToolQuery | None:
    """Detect a date/time/DST-free calendar question and answer it from the clock.

    A language model cannot know today's date: it is not in its weights, and a
    plausible-looking wrong date is worse than no answer. Everything here is
    computed, and the clock is read once per question so the parts of one answer
    cannot disagree with each other.
    """
    text = str(message or "")
    if not text.strip():
        return None
    moment = _clock()                            # read once: one consistent "now"

    match = _DAYS_BETWEEN_RE.search(text)
    if match:
        first, second = match.group("a"), match.group("b")
        try:
            count = days_between(first, second)
        except CalcError:
            return None
        return _date_query(f"days_between({first!r}, {second!r})", Fraction(count),
                           f"There are {abs(count)} days between {first} and {second}.")
    match = _DAYS_UNTIL_RE.search(text)
    if match:
        target, direction = match.group("date"), match.group("dir").lower()
        try:
            count = days_between(moment.date(), target)
        except CalcError:
            return None
        source = f"days_between({str(moment.date())!r}, {target!r})"
        if direction in ("since", "ago"):
            return _date_query(source, Fraction(count),
                               f"{target} was {abs(count)} days ago."
                               if count <= 0 else
                               f"{target} is in {count} days, not in the past.")
        if count == 0:
            return _date_query(source, Fraction(0), f"{target} is today.")
        if count < 0:
            return _date_query(source, Fraction(count),
                               f"{target} has already passed, {abs(count)} days ago.")
        return _date_query(source, Fraction(count),
                           f"There are {count} days until {target}.")
    match = _WEEKDAY_RE.search(text)
    if match:
        target = match.group("date")
        try:
            name = weekday_name(target)
        except CalcError:
            return None
        return _date_query(f"weekday_name({target!r})", name, f"{target} is a {name}.")
    match = _MONTH_DAYS_RE.search(text)
    if match:
        if match.group("date"):
            target, source = match.group("date"), None
            try:
                value = _to_date(target)
            except CalcError:
                return None
            year, month = value.year, value.month
            source = f"days_in_month({year}, {month})"
        else:
            month = _MONTHS.get(match.group("month").lower())
            year = int(match.group("year") or moment.year)
            if month is None:
                return None
            source = f"days_in_month({year}, {month})"
        count = days_in_month(year, month)
        return _date_query(source, Fraction(count),
                           f"{calendar.month_name[month]} {year} has {count} days.")
    if _TIME_RE.search(text):
        stamp = moment.replace(microsecond=0).strftime("%H:%M")
        return _date_query("now()", moment.replace(microsecond=0),
                           f"The time is {stamp} UTC, on "
                           f"{moment.strftime('%A, %d %B %Y')}.")
    if _TODAY_RE.search(text):
        return _date_query("today()", moment.date(),
                           f"Today is {moment.strftime('%A, %d %B %Y')} (UTC).")
    return None


@dataclass(frozen=True)
class ToolQuery:
    """A question one of the tools can answer exactly.

    `kind` is `"math"`, `"string"` or `"datetime"`; `bare` means the whole
    message was the question (so the model never runs at all).
    """

    kind: str
    expression: str
    result: ToolResult
    answer: str
    bare: bool = True

    @property
    def text(self) -> str:
        return self.result.text

    @property
    def line(self) -> str:
        return self.result.line

    @property
    def source(self) -> str:
        """Wire name for the UI: math keeps its historical `question` source."""
        if self.kind == "math":
            return "question"
        return "datetime" if self.kind == "datetime" else "python"

    def to_event(self) -> dict:
        return {"kind": self.kind, "source": self.source, "expression": self.expression,
                "code": self.expression, "result": self.result.text,
                "line": self.result.line, "answer": self.answer}


def find_math_query(message: str) -> ToolQuery | None:
    """`find_math_question`, in the common `ToolQuery` shape."""
    query = find_math_question(message)
    if query is None:
        return None
    return ToolQuery("math", query.expression, query.calculation, query.answer, query.bare)


def find_tool_question(message: str) -> ToolQuery | None:
    """The one entry point the chat engines call.

    Order matters and is deliberate: calendar questions first (their patterns
    mention a date explicitly and must never be read as arithmetic), then
    arithmetic, then string questions. Each detector is narrow, so the first
    match is the intended one.
    """
    return (find_datetime_question(message)
            or find_math_query(message)
            or find_string_question(message))


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


def _run_marker(keyword: str, source: str) -> ToolResult | None:
    """Evaluate one marker: `calc` is arithmetic-only, `tool`/`py` is the wider
    interpreter. Keeping them apart means `<<calc: …>>` can never suddenly start
    accepting strings, and the arithmetic contract in this module's docstring
    stays true."""
    if keyword.lower() == "calc":
        return evaluate_quiet(source)
    return run_python_quiet(source)


def replace_calls(text: str) -> tuple[str, list[ToolResult]]:
    """Replace every complete `<<calc: …>>` / `<<tool: …>>` marker with its value.

    A marker whose expression cannot be evaluated is left exactly as written —
    visibly broken is better than silently wrong.
    """
    out: list[str] = []
    calls: list[ToolResult] = []
    position = 0
    for match in ANY_CALL_RE.finditer(text):
        result = _run_marker(match.group("kind"), match.group("expr"))
        if result is None:
            continue
        out.append(text[position:match.start()])
        out.append(result.text)
        position = match.end()
        calls.append(result)
    out.append(text[position:])
    return "".join(out), calls


#: Marker keywords, longest first, so `python:` is never mistaken for `py:`.
CALL_KEYWORDS = ("python", "calc", "tool", "py")


class ToolTextFilter:
    """Substitute `<<calc: …>>` / `<<tool: …>>` in a token stream without ever
    showing a half-finished marker.

    Tokens arrive in arbitrary fragments, so `<<calc: 5 + 5>>` can be split as
    `"<<ca"` + `"lc: 5 + 5>>"`. Emitting each fragment as it arrives would flash
    raw marker text at the user and break the substitution, so a *potential*
    marker at the tail of the buffer is held back (at most one marker's worth of
    text) and released as soon as the call is complete or proven not to be one.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.calls: list[ToolResult] = []
        self.failed: list[str] = []

    @staticmethod
    def _could_be_marker(tail: str) -> bool:
        """Could `tail` (starting at a `<` at the very end of the buffer) still
        become a `<<calc: …>>` / `<<tool: …>>` call?

        One `<` qualifies only while nothing follows it — `<<` is the only real
        opener, and a lone `<` in prose ("3 < 5") must be released on the next
        character rather than held forever. After `<<` the keyword has to remain
        a prefix (`<<c`, `<<calc`, `<<py`, `<<python`), and once the colon is in,
        the expression may still be open; the call only becomes impossible when
        it closes (`>>`) without matching or grows past a legal expression.
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
        if ":" not in stripped:
            return any(f"{word}:".startswith(stripped) for word in CALL_KEYWORDS)
        keyword, _, rest = stripped.partition(":")
        if keyword not in CALL_KEYWORDS:
            return False
        return len(rest) + len(keyword) <= MAX_TOOL_CHARS + len("python:")

    def _drain(self, final: bool) -> str:
        out: list[str] = []
        while True:
            match = ANY_CALL_RE.search(self.buffer)
            if match is None:
                break
            out.append(self.buffer[:match.start()])
            result = _run_marker(match.group("kind"), match.group("expr"))
            if result is None:
                out.append(match.group(0))       # unparseable: show it verbatim
                self.failed.append(match.group("expr").strip())
            else:
                out.append(result.text)
                self.calls.append(result)
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
        partial = re.search(r"<<\s*(?P<kind>python|calc|tool|py)\s*:\s*(?P<expr>[^<>]*)$",
                            self.buffer, re.IGNORECASE)
        if partial:
            result = _run_marker(partial.group("kind"), partial.group("expr"))
            if result is not None:
                self.calls.append(result)
                prefix = self.buffer[:partial.start()] + result.text
                self.buffer = ""
        return prefix + self._drain(final=True)

    @property
    def used(self) -> bool:
        return bool(self.calls)
