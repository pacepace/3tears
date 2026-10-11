"""How a number is spelled for a reader — the one rule the analysis surfaces share (arm table, decision surface, reference resolver, chart compiler).

A leaf on purpose: the arm table, the decision-surface table, the reference resolver, the chart
compiler and the MCP renders all state numbers from one analysis, often on one line, and each of
them used to reach for whichever formatter was nearest. Two formatters meant one analysis spelled
a latency ``1.235e+04`` in its value column and ``12,346`` in the interval beside it. Anything that
shows a reader a number imports it from here, and nothing here imports anything of the engine's.
Over the trees listed in ``_SPELLING_TREES`` in ``tests/test_numbers.py``,
that is checked from the source by that test, which names each deliberate fixed spelling it allows — such as a
probability, pass^k, a percentage — and flags every other format spec that chooses digits.

A browser renderer that restates this rule pins itself to the same cases this package's tests read,
``tests/fixtures/number-format-cases.json``.

**The rule, decided once at both ends of the range a value can take:**

- A whole number is written whole, at any magnitude.
- At or above 1000 a value is rounded to the nearest integer and written in full, never in
  scientific notation: four significant figures already reach the units digit there, and an
  exponent on a latency or a token count is a sentence a reader has to decode.
- Below that, four significant figures, written out in full at every magnitude: ``0.00005165``,
  never ``5.165e-05``. Four figures rather than a fixed number of decimals is what keeps a small
  non-zero value from rounding to a zero that reads as "nothing happened"; writing them out rather
  than in ``%g``'s exponent is what keeps a mean and the sem beside it in one spelling
  (``0.0005306 ± 0.00005165``, where ``%g`` wrote ``0.0005306 ± 5.165e-05`` — one magnitude two ways
  on one line). An exponent is a sentence a reader has to decode, at the small end as at the large.
- Below :data:`WRITTEN_FROM` (one millionth) a value is written as the bound it is under —
  ``< 0.000001``, or ``> -0.000001`` for a negative one — rather than as its digits: a p-value of
  3.6e-169 written out is a line of zeros no reader counts, and the bound says what a reader acts on,
  that it is not zero and is smaller than anything the surface distinguishes. The value itself is
  never rounded where it is stored; this is only how it reads.
- No thousands separators. These numbers sit inside intervals and lists (``CI [8420, 16270]``,
  ``a; b``), where a comma inside a number reads as one more item.
- Rounding is half-to-even, which is what Python's float formatting does and what the browser
  implements explicitly.
- An absent or non-finite value is an em dash, never a zero.
"""

from __future__ import annotations

import math

#: What an absent or non-finite number reads as. Never ``0``, which would say something was measured.
ABSENT = "—"

#: The magnitude at and above which a value is written as a rounded integer rather than to four
#: significant figures. Four figures reach the units digit exactly here, so rounding to the integer
#: loses nothing four figures would have kept — and it is where ``%g`` would turn to an exponent.
WHOLE_FROM = 1000.0

#: The magnitude below which a non-zero value is written as the bound it sits under (``< 0.000001``)
#: rather than as its digits. Six places: every figure a surface states that matters to a decision — a
#: spread on a cheap call's cost, a rate's interval end — is above it, and below it the digits are a
#: run of zeros a reader would have to count.
WRITTEN_FROM = 1e-6


def format_number(value: float | None, *, prefix: str = "") -> str:
    """Spell a number for a reader, the same way on every analysis surface.

    A ``prefix`` — a currency symbol — goes on the number itself, after its sign and inside a bound:
    ``$1.23``, ``-$1.23``, ``$1234568``, ``< $0.000001``, ``> -$0.000001``. A caller that joined the
    symbol to the returned text would write ``$< 0.000001``, a symbol on a comparison, and ``$-1.23``;
    one that took the text apart to put it right would be restating the rule from the outside. An
    absent value stays :data:`ABSENT` with no prefix: there is no amount to carry it.

    Args:
        value: The number, or ``None`` when there is none.
        prefix: Text written on the number itself, after any sign and inside a bound. Empty by default,
            which leaves the text exactly as it is without one.

    Returns:
        Its text under the module's rule; :data:`ABSENT` for ``None``, NaN or an infinity.
    """
    if value is None or not math.isfinite(value):
        return ABSENT
    if abs(value) < WRITTEN_FROM and value != 0:
        return f"> -{prefix}{_WRITTEN_FROM_TEXT}" if value < 0 else f"< {prefix}{_WRITTEN_FROM_TEXT}"
    text = _digits(value)
    if text.startswith("-"):
        return f"-{prefix}{text[1:]}"
    return f"{prefix}{text}"


def _digits(value: float) -> str:
    """A finite value at or above :data:`WRITTEN_FROM` in magnitude (or zero) as its digits, sign included."""
    if value == int(value):
        return str(int(value))
    if abs(value) >= WHOLE_FROM:
        return f"{value:.0f}"
    # Where the fourth significant figure falls, read off the value as ROUNDED to four figures, so a value that
    # rounds across a power of ten (9.99996e-5 to 0.0001) is placed by what it became, as `%g` places it.
    exponent = int(f"{value:.3e}".split("e")[1])
    text = f"{value:.{max(0, 3 - exponent)}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


#: :data:`WRITTEN_FROM` as the bound a smaller value reads as.
_WRITTEN_FROM_TEXT = f"{WRITTEN_FROM:.6f}"


def format_signed(value: float | None) -> str:
    """Spell a change, carrying its sign — the direction is half of what a delta says.

    Args:
        value: The change, or ``None`` when there is none.

    Returns:
        :func:`format_number`'s text with a leading ``+`` on a positive value (a positive value below
        :data:`WRITTEN_FROM` reads ``< 0.000001``, which carries its direction already); a negative one carries
        its own ``-``, and zero carries neither.
    """
    text = format_number(value)
    positive = value is not None and math.isfinite(value) and value > 0
    return f"+{text}" if positive and not text.startswith("<") else text


__all__ = ["ABSENT", "WHOLE_FROM", "WRITTEN_FROM", "format_number", "format_signed"]
