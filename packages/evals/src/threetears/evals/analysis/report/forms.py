"""One owner of a page's number forms: every figure's form is decided here, once per page, by its measure.

A figure's form is its digits, sign, unit and interval as printed. Spelling one number is the leaf's
(:mod:`threetears.evals.kernel.numbers`); deciding which figures form a set, and how many digits the set reads at, is
this module's. A set is the measure across the whole page (owner ruling, 2026-11-08, Q2), so every figure of one
measure is decided by one :class:`MeasureForm`, never by the surface that prints it.

- :class:`PageForms` is built once per page, before any part is assembled: :meth:`PageForms.of` for a written page
  (a report input), :meth:`PageForms.of_campaign` for a campaign's code page (its readings and its aspects' pages).
  It holds one :class:`MeasureForm` per measure the page can state, and keeps a ledger (:class:`Issued`) of every text
  it hands out, by measure and role, which is what a test reads to hold one figure to one text across surfaces.
- A figure knows its set by its measure key plus its role: :meth:`PageForms.level`, :meth:`PageForms.change`,
  :meth:`PageForms.interval`. A figure that is no measurement takes a fixed form: :meth:`PageForms.count`,
  :meth:`PageForms.p`, :meth:`PageForms.effect`, :meth:`PageForms.confidence`, :meth:`PageForms.declared`,
  :meth:`PageForms.share`, :meth:`PageForms.multiple`, :meth:`PageForms.money_total`.
- Each returns a :class:`Printed` (or a :class:`PrintedInterval`): the value and its decided text, with the parts a
  renderer may lay out apart.

**Today's decisions, relocated, not rewritten.** Until the policy flip every form here reproduces what the page
prints now, so routing a surface through this module moves no golden. Where the page decides a measure's form, that
decision lives here and its old module calls it: the Full results' quantities and decimals (a measure's levels, its
changes, how far a change reaches, a change's interval's ends) and the scorecard's column of measured figures'
decimals. A surface that calls one of these free functions still decides there, so the ratchet counts each such call
as a site (``tests/test_one_owner_of_number_forms.py``); only :class:`PageForms`, asked by measure and role, is the
migrated route. What a measure is (a rate, money on a ruler, shown at all) is
:mod:`~threetears.evals.analysis.report.campaign.measure_kinds`'s, and where a decision is a public function of
another module (a measure's unit, its set's places by
:func:`~threetears.evals.analysis.report.figure_keys.figure_places`), the form calls it. The
differences between surfaces that reproduce today's page (a chart's note spelling a figure alone, ``alone``) are named
where they are taken, so the flip removes them in one place.

**Why at page assembly.** The bundle is fingerprinted evidence and the report input a stored snapshot that recorded
writer runs replay, so neither carries presentation. Page assembly is the first place that sees the whole page, and
both page kinds already pass one object through every part, so a page's forms ride beside it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.campaign_readings import CampaignReadings, MeasureReading
from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS, UNSPACED_UNITS
from threetears.evals.analysis.report.campaign.measure_kinds import money_rulers, rate_measures, shown_measures
from threetears.evals.analysis.report.cost_unit import MoneyRuler
from threetears.evals.analysis.report.figure_keys import RUBRIC_SCALE, figure_places, held_means
from threetears.evals.analysis.report.blocks import TableCell, interval_text
from threetears.evals.analysis.report.parts.charting import figure_unit, interval_level, interval_words, is_rubric_score
from threetears.evals.analysis.report.parts.look import Quantity, figure_text, in_change_form, quantity, quantity_of
from threetears.evals.analysis.report_input import Contrast, ContrastFamily, ReportInput
from threetears.evals.kernel.numbers import (
    POINT_PLACES,
    MoneyForm,
    ends_places,
    format_number,
    set_places,
    spell_at,
    spell_count,
    spell_dollars,
    spell_multiple,
)

if TYPE_CHECKING:
    # A page is built over its aspects' pages, and the module that defines a page loads the parts that read these
    # forms, so the type is named for the checker alone.
    from threetears.evals.analysis.report.model import Report

#: How a measure's figures read: a rate in percent (its changes in points), a judged score on the 1–5 rubric, money,
#: or any other measured unit (a count per trial, a time).
ReadsAs = Literal["rate", "score", "money", "measured"]

#: Which family a measure's figures come from, as a contrast names it: ``measure``, ``judged``, or ``trial`` (how each
#: trial ended, the pass rate); None where a contrast names none, which every decision reads as a measure's own. Free
#: text, as the report input stores it.
MeasureKind = str | None

#: What a figure is to its measure, or which fixed form a figure that is no measurement takes.
Role = Literal[
    "level",
    "change",
    "interval",
    "count",
    "p",
    "effect",
    "confidence",
    "declared",
    "share",
    "multiple",
    "money_total",
]

#: Which page a set of forms is for: a written page (a report input) or a campaign's code page.
PageKind = Literal["written", "code"]

#: The most decimals a measured figure on the scorecard is stated to, before its column's precision is applied.
MEASURED_PLACES = 4


@dataclass(frozen=True)
class Printed:
    """One figure as decided: its value and its text, with the parts a renderer may lay out apart.

    :ivar value: The figure as given (as stored, before any scale); None for an absent one.
    :ivar text: The whole text, as every surface prints it.
    :ivar digits: The number with its sign, never its unit: money's without its ``$`` (``-0.30`` of ``-$0.30``), so a
        cell under a head that names the unit prints it alone.
    :ivar unit: The unit written beside it (``usd`` for money); empty for none.
    """

    value: float | None
    text: str
    digits: str
    unit: str


@dataclass(frozen=True)
class PrintedInterval:
    """An interval as decided: its two ends and its text, with each end's text apart.

    :ivar low: The low end as given.
    :ivar high: The high end as given.
    :ivar text: The whole text, as every surface prints it.
    :ivar low_text: The low end's text, as the page prints it (with its unit on a written page: ``-6.0 pts``).
    :ivar high_text: The high end's text, as the page prints it.
    :ivar unit: The interval's unit; empty for none.
    :ivar low_digits: The low end's number with its sign, never its unit, on every page.
    :ivar high_digits: The high end's number with its sign, never its unit.
    :ivar digits: The whole interval with neither end's unit, joined as the page joins it (``-6.0 to +1.6``,
        ``[-3.5, 7.9]``): what a cell under a head that names the unit prints.
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


@dataclass(frozen=True)
class Issued:
    """One text a page's forms handed out: what the ledger holds.

    :ivar measure: The measure's key; empty for a figure that is no measurement.
    :ivar kind: The measure's kind; empty for a figure that is no measurement.
    :ivar role: What the figure is to its measure, or its fixed form.
    :ivar value: The figure, or an interval's two ends, as given.
    :ivar text: Its text.
    """

    measure: str
    kind: MeasureKind
    role: Role
    value: float | tuple[float, float] | None
    text: str


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
    :ivar level_decimals: The decimals every level reads at; None where the lone figure rule spells each level.
    :ivar change_decimals: The decimals every change reads at; None where the lone figure rule spells each change.
    :ivar money: The form money levels are written in; None for no money.
    :ivar change_money: The form money changes are written in; None for no money.
    :ivar span: How far a change can reach either way, as printed (100 points, a rubric's width); None for unbounded.
    :ivar levels: How the written page draws a level (its unit, its factor, its money); None on a code page.
    :ivar changes: How the written page draws a change; None on a code page.
    :ivar ruler: The code page's money ruler; None elsewhere.
    """

    key: str
    kind: MeasureKind
    reads_as: ReadsAs
    unit: str
    change_unit: str
    factor: float
    change_factor: float
    level_decimals: int | None
    change_decimals: int | None
    money: MoneyForm | None = None
    change_money: MoneyForm | None = None
    span: float | None = None
    levels: Quantity | None = field(default=None, compare=False)
    changes: Quantity | None = field(default=None, compare=False)
    ruler: MoneyRuler | None = field(default=None, compare=False)


class PageForms:
    """Every form one page prints, decided once before the page is assembled, and the ledger of every text handed out.

    :param page: Which page the forms are for.
    :ptype page: PageKind
    :param measures: Each measure's form, by its key and kind.
    :ptype measures: Mapping[tuple[str, MeasureKind], MeasureForm]
    """

    def __init__(self, page: PageKind, measures: Mapping[tuple[str, MeasureKind], MeasureForm]) -> None:
        self.page = page
        self._measures = dict(measures)
        self._issued: list[Issued] = []

    @classmethod
    def of(cls, report_input: ReportInput) -> PageForms:
        """A written page's forms: one per measure, judged dimension and the pass rate its input holds a figure of,
        each as the page's Full results decide it.

        :param report_input: The page's report input.
        :ptype report_input: ReportInput
        :return: The page's forms.
        :rtype: PageForms
        """
        kinds: dict[tuple[str, MeasureKind], None] = {}
        for key in report_input.measures:
            kinds[key, "measure"] = None
        for key in report_input.judged:
            kinds[key, "judged"] = None
        for contrast in report_input.contrasts:
            kinds[contrast.measure, _measure_kind(contrast.kind)] = None
        measures = {(key, kind): _written_form(report_input, key, kind) for key, kind in kinds}
        return cls("written", measures)

    @classmethod
    def of_campaign(cls, readings: CampaignReadings, reports: Mapping[str, Report]) -> PageForms:
        """A campaign code page's forms: the pass rate a change reads, and every measure its scorecard states, each as
        the scorecard decides it.

        :param readings: What code read off the campaign.
        :ptype readings: CampaignReadings
        :param reports: Each aspect's own page, by its label.
        :ptype reports: Mapping[str, Report]
        :return: The page's forms.
        :rtype: PageForms
        """
        measures: dict[tuple[str, MeasureKind], MeasureForm] = {}
        if readings.readings:
            # A change's readings hold the pass rate's changes and intervals in points already, and its levels as
            # shares; the scorecard prints both at a point's one decimal.
            measures[TRIAL_PASS_NAME, "trial"] = MeasureForm(
                key=TRIAL_PASS_NAME,
                kind="trial",
                reads_as="rate",
                unit=PERCENT,
                change_unit=POINTS,
                factor=100.0,
                change_factor=1.0,
                level_decimals=POINT_PLACES,
                change_decimals=POINT_PLACES,
                span=100.0,
            )
        shown = shown_measures(readings)
        rulers = money_rulers(readings, reports)
        rates = rate_measures(readings, reports, {})
        for key in dict.fromkeys(reading.measure for reading in shown):
            measures[key, "measure"] = _scorecard_form(key, shown, rates, rulers)
        return cls("code", measures)

    # --- measures ----------------------------------------------------------------------------------------------------

    def measure(self, key: str, *, kind: MeasureKind = "measure") -> MeasureForm:
        """One measure's form.

        :param key: The measure's key.
        :ptype key: str
        :param kind: The measure's kind; None reads as ``measure``, as every decision reads it.
        :ptype kind: MeasureKind
        :return: Its form.
        :rtype: MeasureForm
        :raises KeyError: The page states no figure of that measure.
        """
        return self._measures[key, _measure_kind(kind)]

    @property
    def measures(self) -> tuple[MeasureForm, ...]:
        """Every measure's form, in the order the page's input holds them."""
        return tuple(self._measures.values())

    def level(self, key: str, value: float | None, *, kind: MeasureKind = "measure", alone: bool = False) -> Printed:
        """A level of a measure (an arm's mean, a slice's), in its measure's form.

        :param key: The measure's key.
        :ptype key: str
        :param value: The level as given; None for none.
        :ptype value: float | None
        :param kind: The measure's kind.
        :ptype kind: MeasureKind
        :param alone: Whether the level is printed by itself (a chart's note, a key no table prints), which today reads
            by the lone figure rule rather than at its set's decimals.
        :ptype alone: bool
        :return: The level.
        :rtype: Printed
        """
        form = self.measure(key, kind=kind)
        if value is None or not math.isfinite(value):
            return self._issue(form.key, form.kind, "level", value, _ABSENT_TEXT, _ABSENT_TEXT, "")
        if self.page == "written":
            text = _written_level(form, value, alone=alone)
        else:
            text = _code_level(form, value)
        return self._issue(form.key, form.kind, "level", value, text, _digits_of(text, form.unit), form.unit)

    def change(self, key: str, value: float | None, *, kind: MeasureKind = "measure", alone: bool = False) -> Printed:
        """A change of a measure (an arm less the control), signed, in its measure's form.

        :param key: The measure's key.
        :ptype key: str
        :param value: The change as given; None for none.
        :ptype value: float | None
        :param kind: The measure's kind.
        :ptype kind: MeasureKind
        :param alone: Whether the change is printed by itself, which today reads by the lone figure rule.
        :ptype alone: bool
        :return: The change.
        :rtype: Printed
        """
        form = self.measure(key, kind=kind)
        if value is None or not math.isfinite(value):
            return self._issue(form.key, form.kind, "change", value, _ABSENT_TEXT, _ABSENT_TEXT, "")
        if self.page == "written":
            text = _written_change(form, value, alone=alone)
        else:
            text = _code_change(form, value)
        unit = form.change_unit
        return self._issue(form.key, form.kind, "change", value, text, _digits_of(text, unit), unit)

    def interval(
        self, key: str, low: float, high: float, *, kind: MeasureKind = "measure", change: bool = False
    ) -> PrintedInterval:
        """An interval on a level or a change of a measure, both ends in its measure's form.

        :param key: The measure's key.
        :ptype key: str
        :param low: The low end as given.
        :ptype low: float
        :param high: The high end as given.
        :ptype high: float
        :param kind: The measure's kind.
        :ptype kind: MeasureKind
        :param change: Whether the interval is on a change (its ends signed).
        :ptype change: bool
        :return: The interval.
        :rtype: PrintedInterval
        """
        form = self.measure(key, kind=kind)
        if self.page == "written":
            low_text, high_text, text = _written_interval(form, low, high, change=change)
        else:
            low_text, high_text, text = _code_interval(form, low, high, change=change)
        unit = form.change_unit if change else form.unit
        low_digits, high_digits = _digits_of(low_text, unit), _digits_of(high_text, unit)
        self._issued.append(Issued(form.key, form.kind, "interval", (low, high), text))
        return PrintedInterval(
            low=low,
            high=high,
            text=text,
            low_text=low_text,
            high_text=high_text,
            unit=unit,
            low_digits=low_digits,
            high_digits=high_digits,
            digits=_interval_join(self.page, low_digits, high_digits),
        )

    # --- figures that are no measurement -----------------------------------------------------------------------------

    def count(self, count: int) -> Printed:
        """A count: whole, its thousands grouped (``2,969``).

        :param count: The count.
        :ptype count: int
        :return: The count.
        :rtype: Printed
        """
        text = spell_count(count)
        return self._issue("", "", "count", float(count), text, text, "")

    def p(self, value: float | None) -> Printed:
        """A p-value, as the page prints one today: by the lone figure rule.

        :param value: The p-value; None for none.
        :ptype value: float | None
        :return: The p-value.
        :rtype: Printed
        """
        text = format_number(value)
        return self._issue("", "", "p", value, text, text, "")

    def effect(self, value: float | None) -> Printed:
        """An effect size or a correlation, as the page prints one today: by the lone figure rule.

        :param value: The effect; None for none.
        :ptype value: float | None
        :return: The effect.
        :rtype: Printed
        """
        text = format_number(value)
        return self._issue("", "", "effect", value, text, text, "")

    def confidence(self, level: str, family: ContrastFamily | None = None) -> Printed:
        """An interval's level, as every surface prints it (:func:`~threetears.evals.analysis.report.parts.charting.
        interval_level`): one decimal, more only where one would read a level under 100 as ``100%``, and what the
        family holds at once beside it where it corrects its comparisons together.

        :param level: The level as the input states it (``99.69%``).
        :ptype level: str
        :param family: The contrast family the interval belongs to; None for none.
        :ptype family: ContrastFamily | None
        :return: The level's words.
        :rtype: Printed
        """
        text = interval_level(level, family)
        return self._issue("", "", "confidence", None, text, text, "")

    def declared(self, value: float | None, unit: str = "") -> Printed:
        """A value the input declares (a margin, a floor, a setting), as the page prints one today: by the lone figure
        rule, in its unit.

        :param value: The value; None for none.
        :ptype value: float | None
        :param unit: Its unit; empty for none.
        :ptype unit: str
        :return: The value.
        :rtype: Printed
        """
        digits = format_number(value)
        text = _joined(digits, unit) if value is not None and math.isfinite(value) else digits
        return self._issue("", "", "declared", value, text, digits, unit if text != digits else "")

    def share(self, value: float | None) -> Printed:
        """A share of a whole (0 to 1) in whole percent (``37%``): a different sentence from a measured rate, "37% of
        it", at its own fixed precision.

        :param value: The share; None for none.
        :ptype value: float | None
        :return: The share.
        :rtype: Printed
        """
        if value is None or not math.isfinite(value):
            return self._issue("", "", "share", value, _ABSENT_TEXT, _ABSENT_TEXT, "")
        text = f"{value:.0%}"
        return self._issue("", "", "share", value, text, text.removesuffix(PERCENT), PERCENT)

    def multiple(self, ratio: float) -> Printed:
        """A ratio read as a multiple (:func:`~threetears.evals.kernel.numbers.spell_multiple`); the unit word is the
        caller's.

        :param ratio: The multiple.
        :ptype ratio: float
        :return: The multiple.
        :rtype: Printed
        """
        text = spell_multiple(ratio)
        return self._issue("", "", "multiple", ratio, text, text, "")

    def money_total(self, amount: float) -> Printed:
        """An amount of money alone, a total (:func:`~threetears.evals.kernel.numbers.spell_dollars`).

        :param amount: Dollars, as stored.
        :ptype amount: float
        :return: The amount.
        :rtype: Printed
        """
        text = spell_dollars(amount)
        return self._issue("", "", "money_total", amount, text, _digits_of(text, MONEY), MONEY)

    # --- the ledger --------------------------------------------------------------------------------------------------

    @property
    def ledger(self) -> tuple[Issued, ...]:
        """Every text handed out, in the order it was."""
        return tuple(self._issued)

    def issued(self, key: str, *, kind: MeasureKind = "measure", role: Role | None = None) -> tuple[Issued, ...]:
        """The texts handed out for one measure, optionally of one role.

        :param key: The measure's key.
        :ptype key: str
        :param kind: The measure's kind.
        :ptype kind: MeasureKind
        :param role: The role; None for every role.
        :ptype role: Role | None
        :return: The entries, in the order they were handed out.
        :rtype: tuple[Issued, ...]
        """
        return tuple(
            entry
            for entry in self._issued
            if (entry.measure, entry.kind) == (key, _measure_kind(kind)) and (role is None or entry.role == role)
        )

    def _issue(
        self, key: str, kind: MeasureKind, role: Role, value: float | None, text: str, digits: str, unit: str
    ) -> Printed:
        self._issued.append(Issued(key, kind, role, value, text))
        return Printed(value=value, text=text, digits=digits, unit=unit)


#: What an absent figure reads as.
_ABSENT_TEXT = format_number(None)


def _joined(digits: str, unit: str) -> str:
    """Digits and their unit, spaced as the package writes the unit."""
    if not unit:
        return digits
    return f"{digits}{unit}" if unit in UNSPACED_UNITS else f"{digits} {unit}"


def _digits_of(text: str, unit: str) -> str:
    """A figure's text without its unit: money's ``$`` taken from inside its sign or bound (``-$0.30``, ``< $0.000001``),
    any other unit from its end."""
    if unit == MONEY:
        return text.replace("$", "", 1)
    if not unit:
        return text
    return text.removesuffix(unit).rstrip() if text.endswith(unit) else text


def _interval_join(page: PageKind, low: str, high: str) -> str:
    """Two ends joined as the page joins an interval: ``low to high`` on a written page, ``[low, high]`` on a code
    page's scorecard."""
    return f"[{low}, {high}]" if page == "code" else f"{low} to {high}"


def _measure_kind(kind: MeasureKind) -> str:
    """A measure's kind as its forms are keyed: a contrast that names none is a measure's own, as every decision reads
    it, so one measure never takes two forms."""
    return "measure" if kind is None else kind


# --- the written page: the Full results' decisions ---------------------------------------------------------------------


def level_quantity(report_input: ReportInput, key: str, *, kind: str | None = None) -> Quantity:
    """How a measure's levels are drawn on a written page, its Full results' every table and its forms alike: by the
    one owner of money's scale over every figure the input holds of it, so its money is drawn per one count everywhere
    (per 1,000 replies for money per reply, per 1,000,000 where it is under a cent per 1,000); any other unit as it is.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure's or judged dimension's key.
    :ptype key: str
    :param kind: A contrast's kind where the key is one's; None for a measure's.
    :ptype kind: str | None
    :return: The quantity its levels are drawn as.
    :rtype: Quantity
    """
    unit = figure_unit(report_input, key, kind=kind)
    return quantity_of(unit, held_means(report_input, key)) if kind != "judged" else quantity(unit)


def change_quantity(
    report_input: ReportInput, measure: str, kind: str | None, contrasts: Sequence[Contrast]
) -> Quantity:
    """How a measure's changes are drawn on a written page: on its levels' own ruler (:func:`level_quantity`), so a
    change is drawn per the count its measure's name says, in the changes' own form capped by their widest interval
    (#993); any other unit's change quantity as it is.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param measure: The measure's key.
    :ptype measure: str
    :param kind: The comparisons' kind.
    :ptype kind: str | None
    :param contrasts: Every comparison on the measure the page states.
    :ptype contrasts: Sequence[Contrast]
    :return: The quantity its changes are drawn as.
    :rtype: Quantity
    """
    amounts = level_quantity(report_input, measure, kind=kind)
    if amounts.unit != MONEY:
        return quantity(figure_unit(report_input, measure, kind=kind), change=True)
    return in_change_form(
        amounts,
        [float(c.delta) * amounts.factor for c in contrasts if c.delta is not None],
        [
            abs(float(c.interval[1]) - float(c.interval[0])) * amounts.factor
            for c in contrasts
            if c.interval is not None
        ],
    )


def level_decimals(report_input: ReportInput, key: str, drawn: Quantity, *, kind: str | None = None) -> int | None:
    """The decimals every level of a measure reads at in every Full results table, by the one owner the keys citing it
    read too (:func:`~threetears.evals.analysis.report.figure_keys.figure_places`), as drawn: a judged score at two,
    any other measured figure (a count per trial, a time) at two significant figures of the least of every mean the
    input holds of it, so its column reads at one precision (``1.4`` beside ``1.5``). None for a rate, whose column
    rule spells it, and for money, whose form does.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure's or judged dimension's key.
    :ptype key: str
    :param drawn: How its levels are drawn (:func:`level_quantity`).
    :ptype drawn: Quantity
    :param kind: A contrast's kind where the key is one's; None for a measure's.
    :ptype kind: str | None
    :return: The decimals, or None.
    :rtype: int | None
    """
    if drawn.unit in (MONEY, PERCENT, POINTS):
        return None
    return figure_places(report_input, key, judged=kind == "judged", factor=drawn.factor)


def change_decimals(report_input: ReportInput, measure: str, kind: str | None, drawn: Quantity) -> int | None:
    """The decimals a comparison's change and its interval's ends read at: a rate's in points at one decimal (``-6.0
    pts`` beside ``-7.6 pts``); a score's at two and any other measured change at two significant figures of the least
    change the page states of its measure, as drawn, by the one owner the keys citing it read too
    (:func:`~threetears.evals.analysis.report.figure_keys.figure_places`); None for money, whose form spells it.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param measure: The measure's key.
    :ptype measure: str
    :param kind: The comparison's kind.
    :ptype kind: str | None
    :param drawn: How its changes are drawn (:func:`change_quantity`).
    :ptype drawn: Quantity
    :return: The decimals, or None.
    :rtype: int | None
    """
    if drawn.unit == MONEY:
        return None
    if drawn.unit == POINTS:
        return POINT_PLACES
    return figure_places(report_input, measure, change=True, judged=kind == "judged", factor=drawn.factor)


def change_span(report_input: ReportInput, measure: str, kind: str | None, drawn: Quantity) -> float | None:
    """How far a change can reach either way, drawn: a rate's 100 points, a score's the width of its scale (4 on 1–5);
    None for any other measure, whose range the input does not bound.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param measure: The measure's key.
    :ptype measure: str
    :param kind: The comparison's kind.
    :ptype kind: str | None
    :param drawn: How its changes are drawn.
    :ptype drawn: Quantity
    :return: The reach, or None.
    :rtype: float | None
    """
    if drawn.unit == POINTS:
        return 100.0
    if is_rubric_score(report_input, measure, kind=kind):
        return RUBRIC_SCALE[1] - RUBRIC_SCALE[0]
    return None


def at_decimals(value: float, places: int, unit: str, *, signed: bool = False) -> str:
    """A figure at its set's decimals (:func:`~threetears.evals.kernel.numbers.spell_at`), its unit spaced as the
    package writes it (``1.4 notices``, ``4.75``, ``-6.0 pts``). A level that is exactly zero reads ``0`` (nothing
    happened, which no precision qualifies, as ``0%`` reads in a column of percents); a change reads its zero at its
    set's decimals (``0.0 pts`` beside ``-5.0 pts``).

    :param value: The figure, as drawn.
    :ptype value: float
    :param places: Its set's decimals.
    :ptype places: int
    :param unit: Its unit; empty for none.
    :ptype unit: str
    :param signed: Whether it is a change, read signed.
    :ptype signed: bool
    :return: Its text.
    :rtype: str
    """
    text = "0" if value == 0 and not signed else spell_at(value, places, signed=signed)
    return _joined(text, unit)


def change_ends(low: float, high: float, places: int, unit: str, *, span: float | None) -> tuple[str, str]:
    """A change's interval's two ends, as drawn, at one precision: kept within how far the measure can move (``span``),
    at the set's decimals or more where those would print a non-zero end as zero or two ends alike
    (:func:`~threetears.evals.kernel.numbers.ends_places`), each signed in its unit.

    :param low: The low end, as drawn.
    :ptype low: float
    :param high: The high end, as drawn.
    :ptype high: float
    :param places: The changes' set's decimals.
    :ptype places: int
    :param unit: Their unit; empty for none.
    :ptype unit: str
    :param span: How far a change can reach either way, as drawn; None for unbounded.
    :ptype span: float | None
    :return: The two ends' texts.
    :rtype: tuple[str, str]
    """
    if span is not None:
        # A change never reaches past what its measure can move: a score's by its scale's width, a rate's by 100 points.
        low, high = max(low, -span), min(high, span)
    at = ends_places(low, high, places)
    return at_decimals(low, at, unit, signed=True), at_decimals(high, at, unit, signed=True)


def _written_form(report_input: ReportInput, key: str, kind: MeasureKind) -> MeasureForm:
    """One measure's form on a written page, from the Full results' decisions; a measure's own kind is None there."""
    stated: str | None = None if kind == "measure" else kind
    levels = level_quantity(report_input, key, kind=stated)
    held = [c for c in report_input.contrasts if (c.measure, _measure_kind(c.kind)) == (key, kind)]
    changes = change_quantity(report_input, key, kind, held)
    reads_as: ReadsAs = (
        "money"
        if levels.unit == MONEY
        else "rate"
        if levels.unit == PERCENT
        else "score"
        if is_rubric_score(report_input, key, kind=stated)
        else "measured"
    )
    return MeasureForm(
        key=key,
        kind=kind,
        reads_as=reads_as,
        unit=levels.unit,
        change_unit=changes.unit,
        factor=levels.factor,
        change_factor=changes.factor,
        level_decimals=level_decimals(report_input, key, levels, kind=stated),
        change_decimals=change_decimals(report_input, key, kind, changes),
        money=levels.money,
        change_money=changes.money,
        span=change_span(report_input, key, kind, changes),
        levels=levels,
        changes=changes,
    )


def _drawn(form: MeasureForm, *, change: bool) -> Quantity:
    drawn = form.changes if change else form.levels
    if drawn is None:
        raise ValueError(f"the measure {form.key!r} holds no written page's quantity")
    return drawn


def _written_level(form: MeasureForm, value: float, *, alone: bool) -> str:
    """A level as the Full results state it: at its set's decimals; a rate, money or a lone level by its quantity's
    note rule (:func:`~threetears.evals.analysis.report.parts.look.figure_text`)."""
    drawn = _drawn(form, change=False)
    if form.level_decimals is not None and not alone:
        return at_decimals(value * drawn.factor, form.level_decimals, drawn.unit)
    return figure_text(value * drawn.factor, drawn)


def _written_change(form: MeasureForm, value: float, *, alone: bool) -> str:
    """A change as the Full results' comparisons state it: signed at its set's decimals; money or a lone change by its
    quantity's note rule."""
    drawn = _drawn(form, change=True)
    if form.change_decimals is not None and not alone:
        return at_decimals(value * drawn.factor, form.change_decimals, drawn.unit, signed=True)
    return figure_text(value * drawn.factor, drawn)


def _written_interval(form: MeasureForm, low: float, high: float, *, change: bool) -> tuple[str, str, str]:
    """An interval as the written page states it, ``low to high``: a change's as the Full results' comparisons do, its
    ends kept within how far the measure can move and read at one precision; a level's as a chart's tooltip does."""
    drawn = _drawn(form, change=change)
    if not change or form.change_decimals is None:
        scaled = (low * drawn.factor, high * drawn.factor)
        text = interval_words(scaled[0], scaled[1], drawn)
        return figure_text(scaled[0], drawn), figure_text(scaled[1], drawn), text
    low_text, high_text = change_ends(
        low * drawn.factor, high * drawn.factor, form.change_decimals, drawn.unit, span=form.span
    )
    return low_text, high_text, _interval_join("written", low_text, high_text)


# --- the code page: the scorecard's decisions --------------------------------------------------------------------------


def measured_decimals(figures: Iterable[float]) -> int | None:
    """The decimals a column of measured figures (neither a rate nor money, whose columns have forms of their own)
    reads at on the scorecard: the set's (:func:`~threetears.evals.kernel.numbers.set_places`), so ``2.737`` beside
    ``0`` reads ``2.7`` and ``99.2345`` beside ``100`` reads ``99`` and ``100``; None for a column of zeros, which the
    number rule spells.

    :param figures: The column's figures, as its cells hold them.
    :ptype figures: Iterable[float]
    :return: The decimals, or None.
    :rtype: int | None
    """
    held = list(figures)
    if not any(figure != 0 for figure in held):
        return None
    return set_places(held)


def _scorecard_form(
    key: str, shown: Sequence[MeasureReading], rates: frozenset[str], rulers: Mapping[str, MoneyRuler]
) -> MeasureForm:
    """One measure's form on a measurement's scorecard: a rate in percent to a decimal, money on its ruler, any other
    measure at its column's decimals over every cell that holds one figure of it (a cell read under several setups
    holds none)."""
    if key in rates:
        return MeasureForm(
            key=key,
            kind="measure",
            reads_as="rate",
            unit=PERCENT,
            change_unit=POINTS,
            factor=100.0,
            change_factor=100.0,
            level_decimals=None,
            change_decimals=POINT_PLACES,
            span=100.0,
        )
    ruler = rulers.get(key)
    if ruler is not None:
        return MeasureForm(
            key=key,
            kind="measure",
            reads_as="money",
            unit=MONEY,
            change_unit=MONEY,
            factor=ruler.scale,
            change_factor=ruler.scale,
            level_decimals=None,
            change_decimals=None,
            money=ruler.form,
            ruler=ruler,
        )
    cells: dict[tuple[str, str], list[MeasureReading]] = {}
    for reading in shown:
        if reading.measure == key:
            cells.setdefault((reading.aspect, reading.arm), []).append(reading)
    figures = [
        round(found[0].value, MEASURED_PLACES)
        for found in cells.values()
        if len(found) == 1 and found[0].value is not None
    ]
    decimals = measured_decimals(figures)
    return MeasureForm(
        key=key,
        kind="measure",
        reads_as="measured",
        unit="",
        change_unit="",
        factor=1.0,
        change_factor=1.0,
        level_decimals=decimals,
        change_decimals=decimals,
    )


def _code_level(form: MeasureForm, value: float) -> str:
    """A level as the scorecard's cell states it: a rate in percent to a decimal, money on its ruler in its one form,
    any other figure at its column's decimals."""
    if form.ruler is not None:
        drawn = form.ruler.drawn(value)
        return TableCell(value=drawn, unit=MONEY, decimals=form.ruler.form.places_of([drawn])).figure
    if form.reads_as == "rate":
        decimals = form.level_decimals
        level = round(value * form.factor, POINT_PLACES)
        return TableCell(value=level, unit=PERCENT, decimals=decimals).figure
    return TableCell(value=round(value * form.factor, MEASURED_PLACES), decimals=form.level_decimals).figure


def _code_change(form: MeasureForm, value: float) -> str:
    """A change as the scorecard's cell states it: signed, in points at a point's one decimal for a rate."""
    decimals = form.change_decimals
    return TableCell(
        value=round(value * form.change_factor, POINT_PLACES if decimals is None else decimals),
        unit=form.change_unit,
        decimals=decimals,
        signed=True,
    ).figure


def _code_interval(form: MeasureForm, low: float, high: float, *, change: bool) -> tuple[str, str, str]:
    """An interval as the scorecard states it, ``[low, high]``: a change's in its unit's precision, the unit in its
    column's head; a level's ends each as its cell writes them."""
    if change:
        text = interval_text(low * form.change_factor, high * form.change_factor, form.change_unit)
        low_text, _, high_text = text.removeprefix("[").removesuffix("]").partition(", ")
        return low_text, high_text, text
    if form.ruler is not None:
        low_text, high_text = form.ruler.spell_ends(low, high)
        return low_text, high_text, f"[{low_text}, {high_text}]"
    cell = TableCell(value=0.0, unit=form.unit, decimals=form.level_decimals)
    ends = (cell.amount(low * form.factor), cell.amount(high * form.factor))
    return ends[0], ends[1], f"[{ends[0]}, {ends[1]}]"
