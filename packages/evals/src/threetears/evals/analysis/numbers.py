"""How a number is spelled for a reader — the one rule the analysis surfaces share (arm table, decision surface, reference resolver, chart compiler).

A leaf on purpose: the arm table, the decision-surface table, the reference resolver, the chart
compiler and the MCP renders all state numbers from one analysis, often on one line. Two formatters
would spell one analysis's latency ``1.235e+04`` in its value column and ``12,346`` in the interval
beside it. Anything that
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

    :param value: The number, or ``None`` when there is none.
    :ptype value: float | None
    :param prefix: Text written on the number itself, after any sign and inside a bound. Empty by default,
        which leaves the text exactly as it is without one.
    :ptype prefix: str
    :return: Its text under the module's rule; :data:`ABSENT` for ``None``, NaN or an infinity.
    :rtype: str
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


#: What separates the thousands of a whole amount where the package groups them (``12,000,000``, ``$7,560.00``).
#: The package's one spelling, not a locale setting: prose and standalone amounts group, the tables' intervals and
#: lists (above) do not.
THOUSANDS_SEPARATOR = ","


def group_thousands(text: str) -> str:
    """The integer part of a number's text with its thousands separated by :data:`THOUSANDS_SEPARATOR`.

    :param text: A number as written, optionally signed and optionally with a decimal part (``12000000``,
        ``-7560.00``). Any text that is not digits, after the sign and before a ``.``, comes back unchanged.
    :ptype text: str
    :return: ``12,000,000`` and ``-7,560.00`` for the examples above.
    :rtype: str
    """
    sign, body = (text[:1], text[1:]) if text[:1] in "+-" else ("", text)
    whole, point, fraction = body.partition(".")
    if not whole.isdigit():
        return text
    groups: list[str] = []
    while len(whole) > 3:
        whole, tail = whole[:-3], whole[-3:]
        groups.insert(0, tail)
    return sign + THOUSANDS_SEPARATOR.join([whole, *groups]) + point + fraction


def spell_dollars(amount: float) -> str:
    """An amount of money as the package spells it: ``$0`` for zero, dollars and cents from ten cents up, three
    significant figures below, the whole dollars grouped by thousands, a negative amount's sign before the dollar sign.

    The one money rule a run's summary (:func:`~threetears.evals.analysis.summary.dollars_text`, which adds the word
    for an amount no client priced) and a chart's notes read; why each spelling is chosen is said there.

    :param amount: Dollars, as stored.
    :ptype amount: float
    :return: ``$0``, ``$0.50``, ``$1,234.50``, ``$0.0123``, ``-$1.20``.
    :rtype: str
    """
    if amount == 0:
        return "$0"
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 0.1:
        return f"{sign}${group_thousands(f'{amount:.2f}')}"
    decimals = 2 - math.floor(math.log10(amount))
    return f"{sign}${amount:.{decimals}f}"


#: The amount below which cents keep fewer than three significant figures of a price (``$0.30`` of ``$0.302``).
_THIRD_FIGURE_BELOW = 1.0


def spell_price(amount: float) -> str:
    """An amount of money a reader may multiply, such as a cost per 1,000 replies: :func:`spell_dollars`, with a third
    significant figure from ten cents to a dollar, where cents keep only two.

    Cents are what money is read in, and from a dollar up they carry three significant figures or more; below ten
    cents :func:`spell_dollars` already writes three. Between, cents round away a third of a percent and more: a cost
    of $0.302 per 1,000 calls reads ``$0.30``, and a reader who multiplies it by a declared 4,000,000 calls a month
    reaches $1,200 beside a monthly figure of $1,208. Three significant figures there (``$0.302``) keep the product
    within half a percent of the monthly figure, the rounding of the price's own third figure. A third figure that is
    zero is left off (``$0.30``, not ``$0.300``): cents already say it.

    :param amount: Dollars, as stored.
    :ptype amount: float
    :return: ``$0.302``, ``$0.30``, ``-$0.495``; :func:`spell_dollars`'s spelling outside ten cents to a dollar.
    :rtype: str
    """
    if not 0.1 <= abs(amount) < _THIRD_FIGURE_BELOW:
        return spell_dollars(amount)
    text = f"{abs(amount):.3f}"
    if float(text) >= _THIRD_FIGURE_BELOW:
        return spell_dollars(amount)
    sign = "-" if amount < 0 else ""
    return f"{sign}${text[:-1] if text.endswith('0') else text}"


#: The multiple from which a multiple is written whole: ``21 times``, never ``21.05 times``.
_MULTIPLE_WHOLE_FROM = 10.0


def spell_multiple(ratio: float) -> str:
    """A ratio read as a multiple (one cost as a multiple of another), to the two significant figures a ratio of two
    rounded figures supports, and never past the units digit from ten up.

    From ten up it is whole, its thousands grouped (``21``, ``1,240``): a decimal on ``21.05 times`` claims a
    precision neither figure it divides has. From one to ten it keeps one decimal, a zero one included (``2.4``,
    ``1.0``), so a multiple near one still says it is near one rather than reading ``1``. Below one it keeps two
    significant figures (``0.48``, ``0.048``). The unit word is the caller's.

    :param ratio: The multiple.
    :ptype ratio: float
    :return: Its digits; :data:`ABSENT` for a non-finite ratio.
    :rtype: str
    """
    if not math.isfinite(ratio):
        return ABSENT
    magnitude = abs(ratio)
    if magnitude == 0:
        return "0"
    sign = "-" if ratio < 0 else ""
    if magnitude >= _MULTIPLE_WHOLE_FROM or round(magnitude, 1) >= _MULTIPLE_WHOLE_FROM:
        return sign + group_thousands(f"{magnitude:.0f}")
    if magnitude >= 1 or round(magnitude, 1) >= 1:
        return f"{sign}{magnitude:.1f}"
    decimals = 1 - math.floor(math.log10(magnitude))
    text = f"{magnitude:.{decimals}f}"
    return sign + (text.rstrip("0").rstrip(".") if float(text) < 1 else "1.0")


#: :data:`WRITTEN_FROM` as the bound a smaller value reads as.
_WRITTEN_FROM_TEXT = f"{WRITTEN_FROM:.6f}"


def format_signed(value: float | None) -> str:
    """Spell a change, carrying its sign — the direction is half of what a delta says.

    :param value: The change, or ``None`` when there is none.
    :ptype value: float | None
    :return: :func:`format_number`'s text with a leading ``+`` on a positive value (a positive value below
        :data:`WRITTEN_FROM` reads ``< 0.000001``, which carries its direction already); a negative one carries
        its own ``-``, and zero carries neither.
    :rtype: str
    """
    text = format_number(value)
    positive = value is not None and math.isfinite(value) and value > 0
    return f"+{text}" if positive and not text.startswith("<") else text


__all__ = [
    "ABSENT",
    "WHOLE_FROM",
    "WRITTEN_FROM",
    "format_number",
    "format_signed",
    "spell_dollars",
    "spell_multiple",
    "spell_price",
]
