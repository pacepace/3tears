"""What a page's forms hand out and decide by: a figure as decided (:class:`Printed`), an interval, the ledger's
entry, money left to its column's form, a judge's preference shares, and one measure's form (:class:`MeasureForm`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from threetears.evals.analysis.quantities import MONEY, PERCENT
from threetears.evals.analysis.report.blocks import TableCell
from threetears.evals.analysis.report.cost_unit import MoneyRuler
from threetears.evals.analysis.report.number_forms.kinds import Role, _MeasureKind, _ReadsAs
from threetears.evals.analysis.report.number_forms.policy import policy_percent
from threetears.evals.analysis.report.parts.look import Quantity
from threetears.evals.analysis.report.unit_once import interval_alone
from threetears.evals.kernel.numbers import MoneyForm


@dataclass(frozen=True)
class Printed:
    """One figure as decided: its value and its text, with the parts a renderer may lay out apart.

    :ivar value: The figure as given (as stored, before any scale); None for an absent one.
    :ivar text: The whole text, as every surface prints it.
    :ivar digits: The number with its sign, never its unit: money's without its ``$`` (``-0.30`` of ``-$0.30``), so a
        cell under a head that names the unit prints it alone.
    :ivar unit: The unit written beside it (``usd`` for money); empty for none.
    :ivar held: On a code page, the number a table's cell holds for the figure beside its text, which a program sorts
        by: scaled into the unit it is printed in and rounded to the precision it reads at; None for an absent figure,
        and on a written page.
    """

    value: float | None
    text: str
    digits: str
    unit: str
    held: float | None = None


@dataclass(frozen=True)
class _PrintedInterval:
    """An interval as decided: its two ends and its text, with each end's text apart.

    :ivar low: The low end as given.
    :ivar high: The high end as given.
    :ivar text: The whole text, as every surface prints it.
    :ivar low_text: The low end's text, as the page prints it: on a written page without its unit, which the interval
        writes once after its high end (``-13`` of ``-13 to -2 pts``).
    :ivar high_text: The high end's text, as the page prints it (``-2 pts``).
    :ivar unit: The interval's unit; empty for none.
    :ivar low_digits: The low end's number with its sign, never its unit, on every page.
    :ivar high_digits: The high end's number with its sign, never its unit.
    :ivar digits: The whole interval with neither end's unit, in the one form every page joins an interval in
        (``-6.0 to +1.6``, ``-3 to +8``): what a cell under a head that names the unit prints.
    """

    low: float
    high: float
    text: str
    low_text: str
    high_text: str
    unit: str
    low_digits: str
    high_digits: str
    digits: str

    @property
    def ends_beside(self) -> tuple[str, str]:
        """The two ends as written beside their figure, which carries the unit: their digits (``80`` and ``89`` of
        ``85% (80 to 89)``), money's each keeping its ``$`` (``$16`` and ``$25``), as :attr:`beside` joins them."""
        if self.unit == MONEY:
            return self.low_text, self.high_text
        return self.low_digits, self.high_digits

    @property
    def beside(self) -> str:
        """The interval as written beside its figure, which carries the unit (number rule 3: ``85% (80 to 89)``,
        ``+10 pts, -29 to +58``): its digits; money's ends each keep their ``$``, as money's form writes them."""
        return self.text if self.unit == MONEY else self.digits


@dataclass(frozen=True)
class _Issued:
    """One text a page's forms handed out: what the ledger holds.

    :ivar measure: The measure's key; empty for a figure that is no measurement.
    :ivar kind: The measure's kind; empty for a figure that is no measurement.
    :ivar role: What the figure is to its measure, or its fixed form.
    :ivar value: The figure, or an interval's two ends, as given.
    :ivar text: Its text.
    """

    measure: str
    kind: _MeasureKind
    role: Role
    value: float | tuple[float, float] | None
    text: str


@dataclass(frozen=True)
class ColumnMoney:
    """An amount of money a table's builder leaves to its set's one form (:meth:`PageForms.money_columns`): the form is
    chosen over every amount the column (or the row's figure type) holds, which no one cell's builder sees.

    :ivar value: The amount, in dollars on the table's ruler.
    :ivar low: Its interval's low end, on the same ruler; None for none.
    :ivar high: Its interval's high end; None for none.
    :ivar flag: The marker its cell prints beside it; None for none.
    :ivar measure: The measure it is a figure of, which the ledger records it under; empty for none.
    :ivar kind: The measure's kind.
    """

    value: float
    low: float | None = None
    high: float | None = None
    flag: str | None = None
    measure: str = ""
    kind: _MeasureKind = None

    @property
    def amounts(self) -> tuple[float, ...]:
        """The amount and its interval's ends, each that is set."""
        return tuple(number for number in (self.value, self.low, self.high) if number is not None)


@dataclass(frozen=True)
class PreferenceShares:
    """A judge's preference on the share scale, in percent, at the one precision the side-by-side table, the figure,
    its line and the answer print it at (:meth:`PageForms.preference`).

    :ivar candidate: The candidate's share, at :attr:`places`.
    :ivar control: The control's: 100 less the candidate's as printed, so the two printed shares sum to 100.
    :ivar interval: The interval on the candidate's share; ``None`` where the pass read none.
    :ivar places: The decimals all three are read to.
    """

    candidate: float
    control: float
    interval: tuple[float, float] | None
    places: int

    def share_text(self, value: float) -> str:
        """One of the three as text, without its ``%``: :meth:`share_percent`'s digits, so the table, the line, the
        answer and the strip's marks read one text.

        :param value: One of the three.
        :ptype value: float
        :return: Its text.
        :rtype: str
        """
        return self.share_percent(value).removesuffix(PERCENT)

    def share_percent(self, value: float) -> str:
        """One of the three with its ``%``, rounded half to even at :attr:`places` whatever its significant figures
        (:func:`policy_percent`): an end of ``99.998`` short of all, a share of ``49.9996`` off even, and two shares
        that sum to 100 as printed (``91.071`` and ``8.929``). The three are one set, and a set reads at one precision,
        so each keeps the set's decimals, a whole one too (``80.000`` beside ``99.998``); only an exact zero reads
        whole, ``0%``: a share of none states no precision.

        :param value: One of the three.
        :ptype value: float
        :return: Its text.
        :rtype: str
        """
        return policy_percent(value, self.places)

    def cell(self, value: float) -> TableCell:
        """A share as a table cell, decided once for both shares, its text :meth:`share_text`'s: a number where the
        set reads whole, else the text alone, since a share past its set's decimals sorts by no figure a reader sees.
        The two share columns of a row are always one kind.

        :param value: One of the shares.
        :ptype value: float
        :return: The cell.
        :rtype: TableCell
        """
        text = self.share_text(value)
        return TableCell(value=value, value_text=text) if self.places == 0 else TableCell(value=text)

    @property
    def interval_text(self) -> str | None:
        """The interval standing alone, its ``%`` once after its high end (``33 to 100%``); ``None`` without one."""
        if self.ends is None:
            return None
        return interval_alone(*self.ends, PERCENT)

    @property
    def fmt(self) -> str:
        """The kit's format a chart draws the shares in, naming the set's decimals where they read whole too
        (``pct:3`` for ``49.998%``, ``pct:0``): a format that names none leaves the kit's own "as many decimals as it
        needs" to any share drawn without its text."""
        return f"pct:{self.places}"

    @property
    def ends(self) -> tuple[str, str] | None:
        """The interval's two ends as text (:meth:`share_text`); ``None`` without an interval."""
        if self.interval is None:
            return None
        return (self.share_text(self.interval[0]), self.share_text(self.interval[1]))


@dataclass(frozen=True)
class MeasureForm:
    """How every figure of one measure reads on a page: the one decision its levels, its changes and their intervals
    share, wherever the page prints them.

    :ivar key: The measure's key.
    :ivar kind: The measure's kind.
    :ivar reads_as: Whether the measure reads as a rate, a score, money or another measured unit.
    :ivar unit: The unit written beside a level (``%``, ``ms``, ``notices``); empty for a bare figure.
    :ivar change_unit: The unit written beside a change (``pts`` for a rate's).
    :ivar factor: What a level as given is multiplied by to be printed (100 for a share read in percent).
    :ivar change_factor: What a change as given is multiplied by to be printed.
    :ivar money: The form money levels are written in; None for no money.
    :ivar change_money: The form money changes are written in; None for no money.
    :ivar span: How far a change can reach either way, as printed (100 points, a rubric's width); None for unbounded.
    :ivar levels: How the written page draws a level (its unit, its factor, its money); None on a code page.
    :ivar changes: How the written page draws a change; None on a code page.
    :ivar ruler: The code page's money ruler; None elsewhere.
    :ivar scale: The scale a score's name carries on a written page (``1–5``); empty for any other measure.
    :ivar unnamed_span: How far a change can reach for a comparison that names no kind, which the written page reads
        by its measure's figures alone (:func:`~threetears.evals.analysis.report.figure_keys.is_rubric_score` with
        no kind), where one that names ``measure`` is never read as a score; the same as ``span`` for any other kind.
    :ivar places: The decimals every figure of the measure reads at under the number policy, its levels, changes and
        interval ends alike, read from its uncertainty (:func:`policy_places`); None for money, which reads by its own
        form.
    :ivar half_width: The narrowest 95% half-width among the measure's headline intervals (each arm's level, each
        comparison), as printed, which ``places`` is read from; None for a measure with no interval.
    """

    key: str
    kind: _MeasureKind
    reads_as: _ReadsAs
    unit: str
    change_unit: str
    factor: float
    change_factor: float
    money: MoneyForm | None = None
    change_money: MoneyForm | None = None
    span: float | None = None
    levels: Quantity | None = field(default=None, compare=False)
    changes: Quantity | None = field(default=None, compare=False)
    ruler: MoneyRuler | None = field(default=None, compare=False)
    scale: str = ""
    unnamed_span: float | None = None
    places: int | None = None
    half_width: float | None = None


__all__ = [
    "ColumnMoney",
    "MeasureForm",
    "PreferenceShares",
    "Printed",
]
