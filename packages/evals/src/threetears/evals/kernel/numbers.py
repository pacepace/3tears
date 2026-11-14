"""How a number is spelled for a reader — the one rule the analysis surfaces share (arm table, decision surface, reference resolver, chart compiler).

A leaf on purpose: the arm table, the decision-surface table, the reference resolver, the chart
compiler and the MCP renders all state numbers from one analysis, often on one line. Two formatters
would spell one analysis's latency ``1.235e+04`` in its value column and ``12,346`` in the interval
beside it. Anything that
shows a reader a number imports it from here, and nothing here imports anything of the engine's.
It lives in ``kernel``, the package both ``run`` and ``analysis`` may import, so a launch's refusal spells its
figures by the same rule as a report; ``threetears.evals.analysis`` re-exports its public names.
Over the whole package and the kit's script, that is checked from the source by
``tests/test_one_owner_of_number_forms.py``: every site outside this module, ``analysis/report/forms.py`` and its
parts under ``analysis/report/number_forms/`` that decides a figure's form is listed there, the list only shrinks,
and the report trees (``_SPELLING_TREES`` there) are held to the stricter rule, which names each deliberate fixed
spelling it allows (a share's percentage, an SVG coordinate) and flags every other format spec that chooses digits.
The engine's trees and the numbers side of the analysis package reach no page's forms, so a fixed form they state (a
refusal's figure, a ledger's dollars, a disclosure's share) is spelled by a function of this module and by nothing
else.

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

**A set reads at one precision** (owner ruling, 2026-11-08; docs/design.md, Presentation rulings). The rule above is a
lone figure's. Figures a reader compares (an interval's two ends, a column of one measure, a chart's labels) share
their decimals (:func:`set_places`, :func:`spell_at`, :func:`ends_places`): a judged score at two decimals, a change in
points at one, any other measured figure at two significant figures of the set's least figure but zero, so ``-29.0 to
+57.6`` and ``1.4`` beside ``1.5`` read alike, and a whole member keeps the set's decimals the lone rule would drop.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

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


def spell_count(count: int) -> str:
    """A count as the package spells it in prose, a cell and a disclosure alike: whole, its thousands separated
    (``2,969``, ``10,000``, ``6,224 of 6,224``), so one count reads one way on every surface.

    :param count: The count.
    :ptype count: int
    :return: Its text.
    :rtype: str
    """
    return group_thousands(str(int(count)))


#: A count a stored sentence carries bare: ``6224 of 6224 score(s)``, ``1234 result(s)``, a share's ``1234 with {...}``.
#: Never a digit run inside a name, a decimal or a setting (``gpt-4o-2024``, ``1.2345``, ``max_tokens=10240``).
_STORED_COUNT = re.compile(r"(?<![\w.,=-])\d{4,}(?=(?: of \d+)? (?:score|result)\(s\)| with \{)")


def spell_stored_counts(sentence: str) -> str:
    """A sentence the bundle stores, with its counts as a page prints them (:func:`spell_count`): ``6224 of 6224
    score(s)`` reads ``6,224 of 6,224 score(s)``. The stored sentence stays as assembled, since the bundle fingerprints
    it; only its printing groups the counts, as :func:`~threetears.evals.analysis.report.words.reason_words` rewords
    a stored reason.

    :param sentence: The sentence as stored.
    :ptype sentence: str
    :return: The sentence as printed.
    :rtype: str
    """
    return _STORED_COUNT.sub(lambda count: spell_count(int(count.group())), sentence)


def spell_dollars(amount: float) -> str:
    """An amount of money as the package spells it, alone: a total in the money form (:func:`money_form`), so one
    amount reads alike in a sentence, a byline and the table beside it: ``$0`` for zero, two significant figures
    and cents at least (owner ruling), the whole dollars grouped by thousands, a negative amount's sign before the
    dollar sign.

    The one money rule a run's summary (:func:`~threetears.evals.analysis.summary.dollars_text`, which adds the word
    for an amount no client priced) and a chart's notes read; why each spelling is chosen is said there.

    :param amount: Dollars, as stored.
    :ptype amount: float
    :return: ``$0``, ``$0.50``, ``$1,234.50``, ``$0.012``, ``-$1.20``.
    :rtype: str
    """
    return money_form([amount]).spell(amount)


#: What an amount no client could price reads as. Never ``$0``: an unpriced call may have been billed.
UNPRICED = "unpriced"


#: The decimals a ledger's dollars read to: the precision one call's price needs, which cents would read as nothing.
_LEDGER_PLACES = 4


def spell_ledger_dollars(amount: float) -> str:
    """An amount of money as a spend ledger states it, to four decimals, its ``$`` before the number (``$0.0012``):
    a call's cost and its priced ceiling, a ledger's totals. A fixed form, the same on a page
    (:meth:`~threetears.evals.analysis.report.forms.PageForms.money_ledger`) and off one.

    :param amount: Dollars, as stored.
    :ptype amount: float
    :return: ``$0.0012``, ``$12.5000``.
    :rtype: str
    """
    return f"${amount:.{_LEDGER_PLACES}f}"


def spell_share(value: float | None) -> str:
    """A share of a whole (0 to 1) in whole percent (``37%``): a different sentence from a measured rate, "37% of
    it", at its own fixed precision. A fixed form, the same on a page
    (:meth:`~threetears.evals.analysis.report.forms.PageForms.share`) and off one.

    :param value: The share; None for none.
    :ptype value: float | None
    :return: ``37%``, ``95%``; :data:`ABSENT` for an absent or non-finite share.
    :rtype: str
    """
    if value is None or not math.isfinite(value):
        return ABSENT
    return f"{value:.0%}"


#: The least amount, in magnitude, that cents state: a set of money per something whose least non-zero member is under
#: it is drawn per a thousand times more of what it counts (:func:`~threetears.evals.analysis.report.cost_unit.money_ruler`).
CENT = 0.01

#: The significant figures money is written to (owner ruling): two, one more only where two would print an
#: interval's two ends alike or leave a rate off the total it sits beside (:data:`RECONCILED_WITHIN`).
MONEY_FIGURES = 2

#: How far a rate's printed figures may round it where a total computed from it is printed beside it (a cost per 1,000
#: beside its monthly cost at the declared volume): the printed rate times the printed volume lands within this share
#: of the printed total, or the rate takes one more figure (owner ruling, "one more figure only where needed").
RECONCILED_WITHIN = 0.005

#: The least amount a rate reads in whole dollars from: two significant figures of $75 are its dollars, and money is
#: written in cents or whole dollars, never one decimal; from $100 a rate's two figures round into the dollars ($123
#: reads ``$120``). A total reads to cents whatever its size (owner ruling).
RATE_WHOLE_FROM = 10.0


@dataclass(frozen=True)
class MoneyForm:
    """One spelling of money for a whole set of amounts: a column, a chart's axis, a figure-key family
    (``cost_per_1k.*``). Every member of the set is written by one rule (#993, as the 2026-11-07 ruling reads it: one
    count of significant figures, never one count of decimals), which :func:`money_form` chooses from the set: each
    amount to two significant figures, so $0.008 reads ``$0.0080`` beside ``$0.11``, never ``$0.1100`` (owner
    ruling). A total reads to cents from a dollar up whatever its size (``$1,208.00``); a rate (money per something)
    reads to cents from a dollar and in whole dollars from :data:`RATE_WHOLE_FROM` (``$75 [$73, $77]``), its two
    figures rounding into the dollars from $100 (``$120`` for $123), never one decimal.

    - ``figures``: the significant figures each member is written to (:data:`MONEY_FIGURES`). ``None`` for a form that
      writes every member to ``decimals`` alike: one a report stored, or a chart's ticks read back from its format.
    - ``decimals``: every member's decimals where ``figures`` is ``None``, a negative count rounding into the dollars
      (``-1`` writes $123 as ``$120``); else the set's largest member's, never fewer than whole dollars, which a
      chart's ticks are written to, so no tick carries more decimals than the amounts beside it.
    - ``at_most``: the most decimals a member is written to, where its figures would ask for more: the precision the
      intervals a set of changes is read with support (:meth:`for_changes`); ``None`` for no bound.
    - ``rate``: whether the set is money per something (a cost per reply, per 1,000), which reads in whole dollars from
      :data:`RATE_WHOLE_FROM`; a total (a run's cost, a month's) reads to cents.
    - ``reconciles``: whether a total computed from each member is printed beside it (a cost per 1,000 at a declared
      monthly volume): a member that two figures round by more than :data:`RECONCILED_WITHIN` of itself takes one
      more, so ``$0.302 per 1,000 calls`` sits beside ``$1,208.00 a month``, never ``$0.30``.

    An exact zero reads ``$0`` in every form: it is no measurement of a precision. ``None`` reads :data:`UNPRICED`,
    never ``$0``. A non-finite amount reads :data:`ABSENT`. Thousands are grouped (``$1,208.00``) and a negative
    amount's sign goes before the dollar sign.

    :ivar decimals: Every member's decimals without ``figures``; with them, the set's largest member's.
    :ivar figures: The significant figures each member is written to; ``None`` for every member at ``decimals``.
    :ivar at_most: The most decimals any member is written to; ``None`` for no bound.
    :ivar rate: Whether the set is money per something rather than a total.
    :ivar reconciles: Whether a total computed from each member is printed beside it.
    """

    decimals: int = 2
    figures: int | None = None
    at_most: int | None = None
    rate: bool = False
    reconciles: bool = False

    def places(self, amount: float) -> int:
        """The decimals this form writes one amount to: those that give it the form's significant figures (one more
        where it reconciles and two round it too far), cents at least below :data:`RATE_WHOLE_FROM` or for a total,
        and ``at_most`` at most; ``decimals`` itself where the form names no figures or the amount is zero. A rate's
        count is negative where its figures end before the dollar (``-1`` for $123, written ``$120``).

        :param amount: Dollars, as stored.
        :ptype amount: float
        :return: The decimals.
        :rtype: int
        """
        if self.figures is None or amount == 0 or not math.isfinite(amount):
            return self.decimals
        magnitude = abs(amount)
        places = _member_decimals(magnitude, self.figures, rate=self.rate)
        if self.reconciles and abs(round(magnitude, places) - magnitude) > RECONCILED_WITHIN * magnitude:
            places = _member_decimals(magnitude, self.figures + 1, rate=self.rate)
        return places if self.at_most is None else min(places, self.at_most)

    def places_of(
        self, amounts: Iterable[float | None], *, ends: tuple[float | None, float | None] | None = None
    ) -> int:
        """The decimals a table cell holding these amounts (its value and its interval's ends) writes every one of
        them to, so a cell reads at one precision: the finest any of them is written to (:meth:`places`); and, where
        the cell's interval ``ends`` would read alike at those decimals though they differ, one significant figure
        more, for this cell alone (owner ruling: "one more only where").

        :param amounts: The cell's amounts, in dollars as stored; ``None``, zero and non-finite ones are passed over.
        :ptype amounts: Iterable[float | None]
        :param ends: The cell's interval, ``(low, high)``; None for none.
        :ptype ends: tuple[float | None, float | None] | None
        :return: The decimals; the form's own where no amount is stated.
        :rtype: int
        """
        stated = [amount for amount in amounts if amount is not None and math.isfinite(amount) and amount != 0]
        if not stated:
            return self.decimals
        places = max(self.places(amount) for amount in stated)
        if self.figures is not None and _alike(ends, places):
            finer = replace(self, figures=self.figures + 1)
            places = max(places, max(finer.places(amount) for amount in stated))
        return places

    def spell(self, amount: float | None, *, signed: bool = False) -> str:
        """One member of the set, in the set's form.

        :param amount: Dollars, as stored; ``None`` for an amount no client priced.
        :ptype amount: float | None
        :param signed: Whether a positive amount carries its ``+`` (a change); a zero, or an amount that rounds to
            zero in the form, carries none.
        :ptype signed: bool
        :return: ``$1.89``, ``-$0.30``, ``+$0.49``, ``$0``; :data:`UNPRICED` for ``None``.
        :rtype: str
        """
        if amount is None:
            return UNPRICED
        if not math.isfinite(amount):
            return ABSENT
        return self._at(amount, self.places(amount), signed=signed)

    def spell_ends(self, low: float, high: float, *, signed: bool = False) -> tuple[str, str]:
        """An interval's two ends in the set's form: each to its own figures, and both to one figure more where they
        would otherwise read alike though they differ (``$0.0451`` and ``$0.0454``, never ``$0.045`` twice).

        :param low: The low end, in dollars as stored.
        :ptype low: float
        :param high: The high end, in dollars as stored.
        :ptype high: float
        :param signed: Whether a positive end carries its ``+`` (a change's interval).
        :ptype signed: bool
        :return: The two ends' texts.
        :rtype: tuple[str, str]
        """
        said = (self.spell(low, signed=signed), self.spell(high, signed=signed))
        if said[0] != said[1] or low == high or self.figures is None:
            return said
        finer = replace(self, figures=self.figures + 1)
        return finer.spell(low, signed=signed), finer.spell(high, signed=signed)

    def _at(self, amount: float, places: int, *, signed: bool) -> str:
        if amount == 0:
            return "$0"
        magnitude = abs(amount)
        digits = _money_digits(magnitude, places)
        zero = float(digits) == 0
        sign = "-" if amount < 0 and not zero else "+" if signed and amount > 0 and not zero else ""
        return f"{sign}${group_thousands(digits)}"

    def for_changes(self, changes: Iterable[float | None], widths: Iterable[float | None] = ()) -> MoneyForm:
        """The form changes between members of the set read in: the set's own scale, figures and kind (a rate or a
        total), so a change reads in the form of the amounts it compares, to the precision of the changes themselves
        and of the intervals they are read with, never the amounts' own.

        Each change is written to the set's significant figures (:meth:`places`), so a $0.40 change between monthly
        totals of $1,208 does not read ``$0``; and to no more decimals than two significant figures of the widest
        interval support, so a change of $0.4924 on an interval $0.64 wide reads ``+$0.49``, never ``+$0.492``. Money
        keeps cents or whole dollars, never one decimal.

        :param changes: The changes, in dollars as stored; ``None`` and non-finite ones are passed over.
        :ptype changes: Iterable[float | None]
        :param widths: The widths of the intervals they are read with, in dollars as stored; none for no interval,
            which leaves the changes' own precision.
        :ptype widths: Iterable[float | None]
        :return: The changes' form.
        :rtype: MoneyForm
        """
        figures = self.figures or MONEY_FIGURES
        stated = [abs(change) for change in changes if change is not None and math.isfinite(change)]
        own = _coarsest(stated, figures, rate=self.rate)
        spans = [width for width in widths if width is not None and math.isfinite(width) and width > 0]
        if not spans:
            return MoneyForm(decimals=own, figures=figures, rate=self.rate)
        supported = max(0, 1 - math.floor(math.log10(max(spans))))
        supported = 2 if supported == 1 else supported
        return MoneyForm(
            decimals=min(own, supported),
            figures=figures,
            at_most=supported,
            rate=self.rate,
        )


def money_form(amounts: Iterable[float | None], *, rate: bool = False, reconciles: bool = False) -> MoneyForm:
    """The one form a set of amounts is spelled in (:class:`MoneyForm`), chosen from the whole set: plain dollars, each
    amount to two significant figures (owner ruling, 2026-11-07). A total reads to cents from a dollar up whatever
    its size; a rate reads in whole dollars from :data:`RATE_WHOLE_FROM`, to its two figures ($123 reads ``$120``). An
    interval whose two ends would read alike takes one figure more where it is spelled (:meth:`MoneyForm.spell_ends`,
    :meth:`MoneyForm.places_of`), never the whole set. ``None`` and non-finite members say nothing of the form and are passed over. Charts, figure keys and
    tables all read money by this one rule, so one amount reads alike on each.

    :param amounts: Every amount the set holds, in dollars as stored, the ends of its intervals included.
    :ptype amounts: Iterable[float | None]
    :param rate: Whether the set is money per something (a cost per reply, per 1,000); a total by default.
    :ptype rate: bool
    :param reconciles: Whether a total computed from each member is printed beside it (:class:`MoneyForm`).
    :ptype reconciles: bool
    :return: The set's form.
    :rtype: MoneyForm
    """
    stated = [abs(amount) for amount in amounts if amount is not None and math.isfinite(amount) and amount != 0]
    return MoneyForm(
        decimals=_coarsest(stated, MONEY_FIGURES, rate=rate),
        figures=MONEY_FIGURES,
        rate=rate,
        reconciles=reconciles and rate,
    )


def _member_decimals(magnitude: float, figures: int, *, rate: bool) -> int:
    """The decimals that give a non-zero magnitude ``figures`` significant figures: never fewer than cents for a total
    or a rate under :data:`RATE_WHOLE_FROM`; a rate from it up to its figures where they end at the dollar or before
    it (``-1`` for $123: ``$120``, owner ruling), else to cents, since money is never written to one decimal. Placed
    by the magnitude as rounded to those figures, so one that rounds across a power of ten (0.0996 to 0.10) is written
    to the decimals of what it became."""
    exponent = int(f"{magnitude:.{figures - 1}e}".split("e")[1])
    places = figures - 1 - exponent
    if rate and magnitude >= RATE_WHOLE_FROM:
        return places if places <= 0 else 2
    return max(2, places)


def _coarsest(magnitudes: Iterable[float], figures: int, *, rate: bool) -> int:
    """The decimals a set's largest non-zero magnitude is written to (:func:`_member_decimals`), whole dollars at the
    fewest, as a chart's ticks are written; cents for none."""
    stated = [magnitude for magnitude in magnitudes if magnitude != 0]
    if not stated:
        return 2
    return max(0, _member_decimals(max(stated), figures, rate=rate))


def _money_digits(magnitude: float, places: int) -> str:
    """A magnitude's digits at ``places`` decimals; a negative count rounds into the whole part (``120`` for 123 at
    ``-1``)."""
    if places >= 0:
        return f"{magnitude:.{places}f}"
    return f"{round(magnitude, places):.0f}"


def _alike(ends: tuple[float | None, float | None] | None, places: int) -> bool:
    """Whether an interval's two ends, which differ, read alike at ``places`` decimals."""
    if ends is None:
        return False
    low, high = ends
    if low is None or high is None or not (math.isfinite(low) and math.isfinite(high)) or low == high:
        return False
    return _money_digits(abs(low), places) == _money_digits(abs(high), places) and (low < 0) == (high < 0)


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


#: The significant figures a set of measured figures (neither a rate, money nor a judged score) reads to: a count per
#: trial, a time, any other measured unit: a count reads at two significant figures. Two, of the set's least figure but
#: zero, so every member reads at the decimals that figure needs: ``1.4`` beside ``1.5`` items, ``99`` beside ``100``
#: ms, on the scorecard as in the Full results.
SET_FIGURES = 2

#: The decimals a judged score on a rubric scale (1–5) reads at, its figures and its changes alike: a hundredth of a
#: point is the step a mean over tens of cases distinguishes, and two decimals keep ``4.86`` beside ``5.00`` one
#: precision.
SCORE_PLACES = 2

#: The decimals a percent or a point reads at in a set: a change in a rate (``-7.6 pts``, ``0.0 pts``), an interval's
#: ends (``-6.9 to +11.0``), whatever its members' digits.
POINT_PLACES = 1


def set_places(values: Iterable[float | None], *, figures: int = SET_FIGURES) -> int:
    """The decimals one set of figures reads at, so its members share their precision: those that give the set's least
    magnitude but zero ``figures`` significant figures, and never fewer than none (``99`` and ``100``, never ``100``
    rounded to ``1.0e2``).

    A set is the figures a reader compares: an interval's two ends, a column of one measure, an axis, a chart's end
    labels. The lone figure keeps :func:`format_number`'s rule, where a whole value drops its decimal; a member of a set
    keeps the set's decimals, so ``-29.0 to +57.6`` and ``1.4`` beside ``1.5`` read alike.

    :param values: The set's figures, as they are drawn; ``None``, non-finite ones and those under :data:`WRITTEN_FROM`
        (zero among them) say nothing of its precision.
    :ptype values: Iterable[float | None]
    :param figures: The significant figures its least member reads to.
    :ptype figures: int
    :return: The decimals, at most :data:`WRITTEN_FROM`'s six; none for a set with no figure at or above it.
    :rtype: int
    """
    # A figure under WRITTEN_FROM reads as the bound it is under (spell_at), so it asks for no decimals: a change of
    # float residue (5.5e-17) never spells the whole column to eighteen places.
    stated = [
        abs(value) for value in values if value is not None and math.isfinite(value) and abs(value) >= WRITTEN_FROM
    ]
    if not stated:
        return 0
    least = min(stated)
    # Placed by the least figure as rounded to its figures, so 0.0996 (two figures: 0.10) reads at two decimals.
    exponent = int(f"{least:.{figures - 1}e}".split("e")[1])
    return min(max(0, figures - 1 - exponent), _WRITTEN_PLACES)


def spell_at(value: float | None, places: int, *, signed: bool = False) -> str:
    """One member of a set at the set's decimals (:func:`set_places`): ``-6.0``, ``0.0``, ``4.86``; :data:`ABSENT` for
    an absent or non-finite value.

    A zero reads at the set's decimals too (``0.0`` beside ``-5.0``), and a value that rounds to zero there carries no
    sign: ``+0.0`` would claim a direction the digits do not show.

    :param value: The figure, as drawn.
    :ptype value: float | None
    :param places: The set's decimals.
    :ptype places: int
    :param signed: Whether a positive figure carries its ``+`` (a change).
    :ptype signed: bool
    :return: Its text.
    :rtype: str
    """
    if value is None or not math.isfinite(value):
        return ABSENT
    text = f"{abs(value):.{max(0, places)}f}"
    if float(text) == 0:
        # A figure under WRITTEN_FROM that is not zero reads as the bound it is under, as a lone one does, never as a
        # zero the digits would claim.
        return format_number(value) if value != 0 and abs(value) < WRITTEN_FROM else text
    return f"-{text}" if value < 0 else f"+{text}" if signed else text


def ends_places(low: float, high: float, places: int) -> int:
    """The decimals an interval's two ends read at: the set's, or more where the set's would print a non-zero end as a
    zero (``-0.0`` for -0.03, an interval that excludes zero reading as if it touched it) or two ends that differ alike;
    both ends take them, so the interval still reads at one precision (as money's ends take one figure more, owner
    ruling).

    :param low: The low end, as drawn.
    :ptype low: float
    :param high: The high end, as drawn.
    :ptype high: float
    :param places: The set's decimals.
    :ptype places: int
    :return: The decimals.
    :rtype: int
    """
    needed = places
    for end in (low, high):
        # An end under WRITTEN_FROM reads as its bound (:func:`spell_at`), so it asks for no decimals: float noise of
        # 1e-9 never spells the interval to nine places.
        if abs(end) >= WRITTEN_FROM and math.isfinite(end) and float(f"{abs(end):.{needed}f}") == 0:
            needed = max(needed, set_places([end], figures=1))
    if low != high and abs(high - low) >= WRITTEN_FROM and spell_at(low, needed) == spell_at(high, needed):
        needed = max(needed, set_places([high - low], figures=1))
    return min(needed, _WRITTEN_PLACES)


#: The most decimals a set reads at: :data:`WRITTEN_FROM`'s, past which a figure reads as its bound.
_WRITTEN_PLACES = 6

#: :data:`WRITTEN_FROM` as the bound a smaller value reads as.
_WRITTEN_FROM_TEXT = f"{WRITTEN_FROM:.6f}"


def round_half_even(value: float, places: int) -> str:
    """A finite value's digits at ``places`` decimals (none below zero), rounded half to even: the one rounding a
    decided figure goes through. It rounds the value as stored, as Python's float formatting does and as every set
    spelling here does (:func:`spell_at`), so ``0.125`` reads ``0.12`` and ``2.675``, stored a hair under its
    spelling, reads ``2.67``. A value that rounds to zero carries no sign: ``-0.0`` would claim a direction the digits
    do not show.

    :param value: The value, finite.
    :ptype value: float
    :param places: The decimals.
    :ptype places: int
    :return: Its digits, signed where it is negative and does not round to zero.
    :rtype: str
    """
    text = f"{value:.{max(0, places)}f}"
    return text.removeprefix("-") if float(text) == 0 else text


def spell_declared(value: float) -> str:
    """A value as its declarer wrote it: the shortest general form, so ``30.0`` reads ``30`` and ``0.25`` reads
    ``0.25``. It exists for the engine messages that echo a declared setting (a timeout, a lease, a scale bound) back
    to the host that declared it, which should recognise its own spelling; a measured figure goes through
    :func:`format_number` or a set's places instead.

    :param value: The declared value, finite.
    :ptype value: float
    :return: Its text, as ``%g`` writes it (six significant figures, an exponent for the very large or small).
    :rtype: str
    """
    return f"{value:g}"


#: The least p-value written as its digits (APA): one under it reads as the bound ``< 0.001``.
P_WRITTEN_FROM = 0.001

#: The p-value from which two decimals are written, three below it (APA, 7th ed., §6.36).
P_TWO_PLACES_FROM = 0.1

#: The decimals an effect size or a correlation reads at.
EFFECT_PLACES = 2


def spell_p(p: float | None) -> str:
    """A p-value as APA writes it: ``< 0.001`` under a thousandth (an exact zero included, which no test reads), three
    decimals below 0.1 (``0.003``), two from it up (``0.50``, ``1.00``).

    :param p: The p-value, or ``None`` when there is none.
    :ptype p: float | None
    :return: Its text; :data:`ABSENT` for ``None``, NaN or an infinity.
    :rtype: str
    """
    if p is None or not math.isfinite(p):
        return ABSENT
    if p < P_WRITTEN_FROM:
        return f"< {P_WRITTEN_FROM}"
    return round_half_even(p, 2 if p >= P_TWO_PLACES_FROM else 3)


def spell_effect(value: float | None) -> str:
    """An effect size or a correlation (a Hedges' g, a pairing r) at :data:`EFFECT_PLACES` decimals, its sign kept:
    ``0.08``, ``-0.19``.

    :param value: The effect, or ``None`` when there is none.
    :ptype value: float | None
    :return: Its text; :data:`ABSENT` for ``None``, NaN or an infinity.
    :rtype: str
    """
    if value is None or not math.isfinite(value):
        return ABSENT
    return round_half_even(value, EFFECT_PLACES)


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
    "spell_count",
    "spell_dollars",
    "spell_stored_counts",
    "spell_multiple",
]
