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
- Below that, four significant figures, spelled as ``%g`` spells them — including its switch to
  scientific notation below 1e-4, which is what keeps a small non-zero value from rounding to a
  zero that reads as "nothing happened".
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


def format_number(value: float | None) -> str:
    """Spell a number for a reader, the same way on every analysis surface.

    Args:
        value: The number, or ``None`` when there is none.

    Returns:
        Its text under the module's rule; :data:`ABSENT` for ``None``, NaN or an infinity.
    """
    if value is None or not math.isfinite(value):
        return ABSENT
    if value == int(value):
        return str(int(value))
    if abs(value) >= WHOLE_FROM:
        return f"{value:.0f}"
    return f"{value:.4g}"


def format_signed(value: float | None) -> str:
    """Spell a change, carrying its sign — the direction is half of what a delta says.

    Args:
        value: The change, or ``None`` when there is none.

    Returns:
        :func:`format_number`'s text with a leading ``+`` on a positive value; a negative one carries
        its own ``-``, and zero carries neither.
    """
    text = format_number(value)
    return f"+{text}" if value is not None and math.isfinite(value) and value > 0 else text


__all__ = ["ABSENT", "WHOLE_FROM", "format_number", "format_signed"]
