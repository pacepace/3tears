"""How a number is spelled for a reader — the one rule the analysis surfaces share (arm table, decision surface, reference resolver, chart compiler).

A leaf on purpose: the arm table, the decision-surface table, the reference resolver, the chart
compiler and the MCP renders all state numbers from one analysis, often on one line. Two formatters
would spell one analysis's latency ``1.235e+04`` in its value column and ``12,346`` in the interval
beside it. Anything that
shows a reader a number imports it from here, and nothing here imports anything of the engine's.
It lives in ``kernel``, the package both ``run`` and ``analysis`` may import, so a launch's refusal spells its
figures by the same rule as a report; ``threetears.evals.analysis`` re-exports its public names.
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
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

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


#: What an amount no client could price reads as. Never ``$0``: an unpriced call may have been billed.
UNPRICED = "unpriced"

#: The least amount, in magnitude, that cents state: a non-zero member of a set under it puts the whole set per 1,000.
CENT = 0.01

#: How many times its amount a per-1,000 spelling writes, and the mark it carries: ``$8/1000`` is $0.008.
PER_THOUSAND = 1000
PER_THOUSAND_MARK = "/1000"


@dataclass(frozen=True)
class MoneyForm:
    """One spelling of money for a whole set of amounts: a column, a chart's axis, a figure-key family
    (``cost_per_1k.*``). Every member of the set is written in it, so successive rows never alternate between
    ``$0.008`` and ``$0.11`` (#993). :func:`money_form` chooses it from the set, never from one value.

    - ``per_thousand``: the set reads per 1,000 (``$8/1000`` for $0.008), because some member is too small for cents
      (under :data:`CENT`); then every member does (``$110/1000`` for $0.11 beside it).
    - ``decimals``: the decimals every member is written to, after the per-1,000 factor: cents at least, and as many
      more as give the set's smallest non-zero member two significant figures, so no member reads as a zero it is not.

    An exact zero reads ``$0`` in every form: it is no measurement of a precision. ``None`` reads :data:`UNPRICED`,
    never ``$0``. A non-finite amount reads :data:`ABSENT`. Thousands are grouped (``$1,208.00``) and a negative
    amount's sign goes before the dollar sign.

    :ivar per_thousand: Whether the set reads per 1,000.
    :ivar decimals: The decimals each member is written to.
    """

    per_thousand: bool = False
    decimals: int = 2

    def spell(self, amount: float | None, *, signed: bool = False) -> str:
        """One member of the set, in the set's form.

        :param amount: Dollars, as stored; ``None`` for an amount no client priced.
        :ptype amount: float | None
        :param signed: Whether a positive amount carries its ``+`` (a change); a zero, or an amount that rounds to
            zero in the form, carries none.
        :ptype signed: bool
        :return: ``$1.89``, ``$1.89/1000``, ``-$0.30``, ``+$0.49/1000``, ``$0``; :data:`UNPRICED` for ``None``.
        :rtype: str
        """
        if amount is None:
            return UNPRICED
        if not math.isfinite(amount):
            return ABSENT
        if amount == 0:
            return "$0"
        scaled = abs(amount) * (PER_THOUSAND if self.per_thousand else 1)
        digits = f"{scaled:.{self.decimals}f}"
        zero = float(digits) == 0
        sign = "-" if amount < 0 and not zero else "+" if signed and amount > 0 and not zero else ""
        return f"{sign}${group_thousands(digits)}{PER_THOUSAND_MARK if self.per_thousand else ''}"

    def for_change(self, width: float | None) -> MoneyForm:
        """The form a change between members of the set reads in: the set's own (per 1,000 where the set is), to no
        more decimals than the width of the change's interval supports.

        A change is a difference of two amounts, so it reads in their form; its interval says how much of it is known.
        Two significant figures of the interval's width are what it supports: a change of $0.4924 on an interval
        $0.64 wide reads ``+$0.49``, never ``+$0.492``. Money keeps cents or whole dollars, never one decimal.

        :param width: The width of the change's interval, in dollars as stored; ``None`` where none is stated, which
            leaves the set's own decimals.
        :ptype width: float | None
        :return: The change's form.
        :rtype: MoneyForm
        """
        if width is None or not math.isfinite(width) or width <= 0:
            return self
        scaled = width * (PER_THOUSAND if self.per_thousand else 1)
        supported = max(0, 1 - math.floor(math.log10(scaled)))
        supported = 2 if supported == 1 else supported
        return MoneyForm(per_thousand=self.per_thousand, decimals=min(self.decimals, supported))


def per_thousand_digits(digits: str) -> str:
    """An amount's written digits moved per 1,000 exactly, by its decimal point, never rounded: ``0.008`` to ``8``,
    ``0.0001743`` to ``0.1743``, ``1239`` to ``1239000``; for a column already written that moves into one form
    (#993) and keeps every digit it was written with.

    :param digits: A non-negative amount as written, with no sign, dollar sign or thousands separator.
    :ptype digits: str
    :return: The digits of a thousand times it, without trailing zeros past the point.
    :rtype: str
    """
    return f"{(Decimal(digits) * PER_THOUSAND).normalize():f}"


def money_form(amounts: Iterable[float | None]) -> MoneyForm:
    """The one form a set of amounts is spelled in (:class:`MoneyForm`), chosen from the whole set.

    Per 1,000 where any non-zero member is under a cent (:data:`CENT`) in magnitude; then the decimals that give the
    smallest non-zero member, as the form writes it, two significant figures, and never fewer than cents. ``None`` and
    non-finite members say nothing of the form and are passed over; a set with none left reads in dollars and cents.

    :param amounts: Every amount the set holds, in dollars as stored.
    :ptype amounts: Iterable[float | None]
    :return: The set's form.
    :rtype: MoneyForm
    """
    stated = [abs(amount) for amount in amounts if amount is not None and math.isfinite(amount) and amount != 0]
    if not stated:
        return MoneyForm()
    per_thousand = min(stated) < CENT
    smallest = min(stated) * (PER_THOUSAND if per_thousand else 1)
    # Placed by the smallest member as rounded to two figures, so one that rounds across a power of ten (0.0996 to
    # 0.10) is written to the decimals of what it became.
    exponent = int(f"{smallest:.1e}".split("e")[1])
    return MoneyForm(per_thousand=per_thousand, decimals=max(2, 1 - exponent))


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
    "UNPRICED",
    "MoneyForm",
    "format_number",
    "format_signed",
    "money_form",
    "per_thousand_digits",
    "spell_dollars",
    "spell_multiple",
]
