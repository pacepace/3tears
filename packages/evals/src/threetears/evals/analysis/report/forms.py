"""One owner of a page's number forms: every figure's form is decided here, once per page, by its measure.

A figure's form is its digits, sign, unit and interval as printed. Spelling one number is the leaf's
(:mod:`threetears.evals.kernel.numbers`); deciding which figures form a set, and how many digits the set reads at, is
this module's. A set is the measure across the whole page (owner ruling, 2026-11-08, Q2), so every figure of one
measure is decided by one :class:`MeasureForm`, never by the surface that prints it.

- :class:`PageForms` is built once per page, before any part is assembled: :meth:`PageForms.of` for a written page
  (a report input), :meth:`PageForms.of_campaign` for a campaign's code page (its readings and its aspects' pages),
  :meth:`PageForms.of_bundle` for an aspect's own page (its bundle, which holds no report input: its contrasts' and
  guardrails' sets, each surface's as today). A written page's forms carry their input
  (:attr:`PageForms.report_input`), so a figure key reads the input its forms were built from.
  It holds one :class:`MeasureForm` per measure the page can state, and keeps a ledger (:class:`_Issued`) of every text
  it hands out, by measure and role, which is what a test reads to hold one figure to one text across surfaces.
- A figure knows its set by its measure key plus its role: :meth:`PageForms.level`, :meth:`PageForms.change`,
  :meth:`PageForms.interval`; on a code page also :meth:`PageForms.drawn_level` (a level already in percent),
  :meth:`PageForms.bracketed` (a level with its interval in one cell), :meth:`PageForms.sizes` (an interval sized in
  a sentence) and :meth:`PageForms.half_width`. A figure that is no measurement takes a fixed form: :meth:`PageForms.count`,
  :meth:`PageForms.p`, :meth:`PageForms.effect`, :meth:`PageForms.confidence`, :meth:`PageForms.declared`,
  :meth:`PageForms.share`, :meth:`PageForms.multiple`, :meth:`PageForms.money_total`; on a code page also
  :meth:`PageForms.confidence_share`. A pass^k (:meth:`PageForms.pass_k`) is its measure's, by key and kind. A
  figure key's figure (:mod:`~threetears.evals.analysis.report.figure_keys`) reads its measure's set by key and kind
  (:meth:`PageForms.key_places`, :meth:`PageForms.key_number`) and an interval's ends at one precision
  (:meth:`PageForms.key_plain_ends`, :meth:`PageForms.key_points_ends`): every caller of the keys hands its page's
  forms in, since this module reads the keys' module and so it can read none of this one.
- Each returns a :class:`Printed` (or a :class:`_PrintedInterval`): the value and its decided text, with the parts a
  renderer may lay out apart.

**Today's decisions, relocated, not rewritten.** Until the policy flip every form here reproduces what the page
prints now, so routing a surface through this module moves no golden. Where the page decides a measure's form, that
decision lives here and its old module calls it: the Full results' quantities and decimals (a measure's levels, its
changes, how far a change reaches, a change's interval's ends) and the scorecard's column of measured figures'
decimals. A surface that calls one of these free functions still decides there, so the ratchet counts each such call
as a site (``tests/test_one_owner_of_number_forms.py``); only :class:`PageForms`, asked by measure and role, is the
migrated route. What a measure is (a rate, money on a ruler, shown at all) is
:mod:`~threetears.evals.analysis.report.campaign.measure_kinds`'s, and where a decision is a public function of
another module (a measure's unit, :func:`~threetears.evals.analysis.report.figure_keys.figure_unit`), the form calls
it; a measure's set's places, which the keys and the Full results both read (:func:`figure_places`), are decided here.
The
differences between surfaces that reproduce today's page (a chart's note spelling a figure alone, ``alone``) are named
where they are taken, so the flip removes them in one place.
A chart's forms (``PageForms.chart_*``) still decide over a chart's own drawn values, as today; each takes the measure
and kind of a measure's figure (``chart_amounts`` its measures' keys), and C5 re-decides them by measure there.

**Why at page assembly.** The bundle is fingerprinted evidence and the report input a stored snapshot that recorded
writer runs replay, so neither carries presentation. Page assembly is the first place that sees the whole page, and
both page kinds already pass one object through every part, so a page's forms ride beside it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, overload

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME, AnalysisContextBundle, FamilyComparison
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_readings import CampaignReadings, MeasureReading
from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS, UNSPACED_UNITS, is_money
from threetears.evals.analysis.report.campaign.measure_kinds import money_rulers, rate_measures, shown_measures
from threetears.evals.analysis.report.cost_unit import MoneyRuler
from threetears.evals.analysis.report.figures import ContrastPlaces, MeasureRuler, Rulers, report_rulers
from threetears.evals.analysis.report.figure_keys import (
    RUBRIC_SCALE,
    Figure,
    criterion_ruler,
    figure_unit,
    format_figure,
    held_kind,
    held_means,
    is_rubric_score,
    plus_signed,
    resolve,
    rubric_score,
)
from threetears.evals.analysis.report.blocks import CellValue, TableCell, interval_text
from threetears.evals.analysis.report.charts import percent_at
from threetears.evals.analysis.report.parts.look import Quantity, figure_text, in_change_form, quantity, quantity_of
from threetears.evals.analysis.report.parts.words import FAMILY_LEVEL_TEMPLATE
from threetears.evals.analysis.report_input import Contrast, ContrastFamily, Criterion, ReportInput
from threetears.evals.analysis.report_margins import aspect_board
from threetears.evals.analysis.report.verdict import MIN_CASES
from threetears.evals.kernel.metrics import METRIC_DESCRIPTORS
from threetears.evals.kernel.surface import DecisionSurface
from threetears.evals.kernel.numbers import (
    POINT_PLACES,
    SCORE_PLACES,
    WHOLE_FROM,
    MoneyForm,
    ends_places,
    format_number,
    format_signed,
    money_form,
    set_places,
    spell_at,
    spell_count,
    spell_dollars,
    spell_multiple,
)

__all__ = [
    "EVEN_SHARE",
    "MeasureForm",
    "PageForms",
    "PreferenceShares",
    "Printed",
    "Role",
]

if TYPE_CHECKING:
    # A page is built over its aspects' pages, and the module that defines a page loads the parts that read these
    # forms, so the type is named for the checker alone.
    from threetears.evals.analysis.report.model import Report

#: How a measure's figures read: a rate in percent (its changes in points), a judged score on the 1–5 rubric, money,
#: or any other measured unit (a count per trial, a time).
_ReadsAs = Literal["rate", "score", "money", "measured"]

#: Which family a measure's figures come from, as a contrast names it: ``measure``, ``judged``, or ``trial`` (how each
#: trial ended, the pass rate); None where a contrast names none, which every decision reads as a measure's own. Free
#: text, as the report input stores it.
_MeasureKind = str | None

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
    "lone",
    "reading",
    "echoed",
    "hits",
    "money",
    "check",
    "preference",
    "pass_k",
    "half_width",
]

#: Which page a set of forms is for: a written page (a report input), a campaign's code page, an aspect's own page
#: (code's page over one aspect's bundle), or an off-page surface (a command line's lines, a lens's text, the rung-5
#: page's side-by-side figures), which states no page's measure.
_PageKind = Literal["written", "code", "aspect", "off"]

#: Which of an aspect page's surfaces states a reading's changes or levels: the contrasts table (and the statistics
#: beside it), the verdict line's sentences, a program's contrast rows, or the guardrails table. Each reads its own set
#: today (the sentences' money over every contrast tested, one under the case floor too, where the table's is over the
#: contrasts it shows); the policy makes them one set per measure.
_ContrastSurface = Literal["contrasts", "sentences", "rows", "guardrails"]

#: The most decimals a measured figure on the scorecard is stated to, before its column's precision is applied.
_MEASURED_PLACES = 4


@dataclass(frozen=True)
class Printed:
    """One figure as decided: its value and its text, with the parts a renderer may lay out apart.

    :ivar value: The figure as given (as stored, before any scale); None for an absent one.
    :ivar text: The whole text, as every surface prints it.
    :ivar digits: The number with its sign, never its unit: money's without its ``$`` (``-0.30`` of ``-$0.30``), so a
        cell under a head that names the unit prints it alone.
    :ivar unit: The unit written beside it (``usd`` for money); empty for none.
    :ivar held: On a code page, the number a table's cell holds for the figure until a cell carries its decided text:
        scaled into the unit it is printed in and rounded to the precision it reads at (a rate's percent to a decimal,
        a measured figure to four decimals), so a column rule the table still applies after the cell is
        built (whole percents per column) reads the number it read before; None for an absent figure, and on a
        written page.
    :ivar decimals: On a code page, the decimals a cell holding the figure writes it at, where its form fixes them (a
        pass rate's level at one, money's in its form, a measured column at its set's); None where the number rule or
        the table's column rule spells it.
    """

    value: float | None
    text: str
    digits: str
    unit: str
    held: float | None = None
    decimals: int | None = None


@dataclass(frozen=True)
class _PrintedInterval:
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
        """One of the three as text, at :attr:`places` whatever its significant figures: an end of ``99.998`` short of
        all, a share of ``49.9996`` off even, and two shares that sum to 100 as printed (``91.071`` and ``8.929``). The
        three are one set, and a set reads at one precision, so each keeps the set's decimals, a whole one too
        (``80.000`` beside ``99.998``).

        :param value: One of the three.
        :ptype value: float
        :return: Its text.
        :rtype: str
        """
        return f"{value:.{self.places}f}"

    def cell(self, value: float) -> CellValue:
        """A share as a table cell, decided once for both shares: numbers where the set reads whole (the one number
        rule prints a whole share as :meth:`share_text` does), else the text of both, since that rule drops a whole
        share's decimals and keeps four significant figures where the set needs its own decimals. The two share
        columns of a row are always one kind.

        :param value: One of the shares.
        :ptype value: float
        :return: The cell's value.
        :rtype: CellValue
        """
        return value if self.places == 0 else self.share_text(value)

    @property
    def fmt(self) -> str:
        """The kit's format a chart draws the shares in, at the set's decimals, so its values table and its tooltips
        read each share as the line under it does (``pct:3`` for ``49.998%``); ``pct`` where they read whole."""
        return "pct" if self.places == 0 else f"pct:{self.places}"

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
    :ivar level_decimals: The decimals every level reads at; None where the lone figure rule spells each level.
    :ivar change_decimals: The decimals every change reads at; None where the lone figure rule spells each change.
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
    """

    key: str
    kind: _MeasureKind
    reads_as: _ReadsAs
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
    scale: str = ""
    unnamed_span: float | None = None


class PageForms:
    """Every form one page prints, decided once before the page is assembled, and the ledger of every text handed out.

    :param page: Which page the forms are for.
    :ptype page: _PageKind
    :param measures: Each measure's form, by its key and kind.
    :ptype measures: Mapping[tuple[str, _MeasureKind], MeasureForm]
    :param report_input: A written page's report input, which a figure key's set is read from
        (:meth:`key_places`); None for any other page.
    :ptype report_input: ReportInput | None
    """

    def __init__(
        self,
        page: _PageKind,
        measures: Mapping[tuple[str, _MeasureKind], MeasureForm],
        report_input: ReportInput | None = None,
    ) -> None:
        self.page = page
        self._measures = dict(measures)
        self._issued: list[_Issued] = []
        self._input = report_input
        # An aspect page's contrasts' and guardrails' decisions, by the surface that states them (:meth:`of_bundle`).
        self._bundle: AnalysisContextBundle | None = None
        self._contrast_places: dict[_ContrastSurface, dict[tuple[str, str], ContrastPlaces]] = {}
        self._change_money: dict[_ContrastSurface, dict[tuple[str, str], MoneyForm]] = {}

    @classmethod
    def of(cls, report_input: ReportInput) -> PageForms:
        """A written page's forms: one per measure, judged dimension and the pass rate its input holds a figure of,
        each as the page's Full results decide it. A slice (a stratum, a task, a cell) can hold a figure on a key the
        input declares no measure for, which its table prints as a measure's; and a scoreboard that counts passes
        states the pass rate, so each has its form too.

        :param report_input: The page's report input.
        :ptype report_input: ReportInput
        :return: The page's forms.
        :rtype: PageForms
        """
        kinds: dict[tuple[str, _MeasureKind], None] = {}
        for key in report_input.measures:
            kinds[key, "measure"] = None
        for key in report_input.judged:
            kinds[key, "judged"] = None
        for contrast in report_input.contrasts:
            kinds[contrast.measure, _measure_kind(contrast.kind)] = None
        slices = [
            *(stratum.per_arm for stratum in report_input.strata),
            *(task.per_arm for task in report_input.tasks),
            *(cell.per_arm for cell in report_input.cells.values()),
        ]
        for per_arm in slices:
            for figures in per_arm.values():
                for key in figures:
                    judged_only = key in report_input.judged and key not in report_input.measures
                    kinds.setdefault((key, "judged" if judged_only else "measure"), None)
        if _counts_passes(report_input):
            kinds.setdefault((TRIAL_PASS_NAME, "trial"), None)
        measures = {(key, kind): _written_form(report_input, key, kind) for key, kind in kinds}
        return cls("written", measures, report_input)

    @classmethod
    def off_page(cls) -> PageForms:
        """An off-page surface's forms: a command line's lines, a lens's text, a result's fields, the rung-5 page's
        side-by-side figures. It states no page's measure, so it holds none; its figures take the fixed forms and the
        off-page ones (:meth:`lone`, :meth:`reading`, :meth:`preference` ...), and its ledger holds what it handed out.

        :return: The surface's forms.
        :rtype: PageForms
        """
        return cls("off", {})

    @classmethod
    def of_campaign(cls, readings: CampaignReadings, reports: Mapping[str, Report]) -> PageForms:
        """A campaign code page's forms: the pass rate a change or a choice reads, and every measure its scorecard
        states, each as the scorecard decides it.

        :param readings: What code read off the campaign.
        :ptype readings: CampaignReadings
        :param reports: Each aspect's own page, by its label.
        :ptype reports: Mapping[str, Report]
        :return: The page's forms.
        :rtype: PageForms
        """
        measures: dict[tuple[str, _MeasureKind], MeasureForm] = {}
        if readings.readings or readings.pairs:
            # A change's readings and a choice's pairs hold the pass rate's changes and intervals in points already,
            # and its levels as shares; the scorecard prints both at a point's one decimal.
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

    @classmethod
    def of_bundle(cls, bundle: AnalysisContextBundle) -> PageForms:
        """An aspect's own page's forms (code's page over the aspect's bundle, which holds no report input): each
        reading's places and each money measure's form for its changes, as every surface that states the aspect's
        contrasts and guardrails decides them today, each over its own set (:data:`_ContrastSurface`).

        :param bundle: The aspect's evidence.
        :ptype bundle: AnalysisContextBundle
        :return: The page's forms.
        :rtype: PageForms
        """
        forms = cls("aspect", {})
        forms._bundle = bundle
        surface = bundle_decision_surface(bundle)
        rulers = report_rulers(surface, bundle.measure_catalog)
        compared = [comparison for family in bundle.multiple_comparisons.families for comparison in family.comparisons]
        # The sentences read each contrast once, by its arm, measure, reading and aspect, the last tested winning.
        tested = list(
            {
                (comparison.contrast.variant_key, comparison.name, comparison.reading, family.aspect): comparison
                for family in bundle.multiple_comparisons.families
                for comparison in family.comparisons
            }.values()
        )
        shown = [comparison for comparison in compared if _resolvable(comparison)]
        guardrails = bundle.guardrails.checks if bundle.guardrails is not None else []
        forms._contrast_places = {
            "contrasts": _contrast_places(bundle, surface, rulers, shown),
            "sentences": _contrast_places(bundle, surface, rulers, [c for c in tested if _resolvable(c)]),
            "rows": _contrast_places(bundle, surface, rulers, shown),
        }
        forms._change_money = {
            "contrasts": _change_money(rulers, ((c.name, c.reading, c.delta, c.interval) for c in shown)),
            "sentences": _change_money(rulers, ((c.name, c.reading, c.delta, c.interval) for c in tested)),
            "rows": _change_money(
                rulers, ((c.name, c.reading, c.delta, c.interval) for c in compared if not _under_floor(c))
            ),
            "guardrails": _change_money(
                rulers, ((check.name, check.reading, check.delta, check.interval) for check in guardrails)
            ),
        }
        return forms

    @property
    def bundle(self) -> AnalysisContextBundle:
        """The bundle an aspect page's forms were built from.

        :raises ValueError: The forms are another page's, which hold no bundle.
        """
        if self._bundle is None:
            raise ValueError(f"a {self.page} page's forms hold no aspect's bundle")
        return self._bundle

    def contrast_places(self, name: str, reading: str, *, stated_in: _ContrastSurface) -> ContrastPlaces | None:
        """How one reading's contrast figures read on one surface of an aspect's page: its set's places.

        :param name: The reading's measure or dimension.
        :ptype name: str
        :param reading: Which reading of it (``measure``, ``judged``, ``trial``).
        :ptype reading: str
        :param stated_in: The surface that states the figures.
        :ptype stated_in: _ContrastSurface
        :return: The places; None where the surface states no contrast of the reading that its set holds.
        :rtype: ContrastPlaces | None
        """
        return self._contrast_places.get(stated_in, {}).get((name, reading))

    def change_money(self, name: str, reading: str, *, stated_in: _ContrastSurface) -> MoneyForm | None:
        """The one form a money measure's changes read in on one surface of an aspect's page, a change and its
        interval's ends alike (#993); None for a measure of another unit.

        :param name: The measure's name.
        :ptype name: str
        :param reading: Which reading of it.
        :ptype reading: str
        :param stated_in: The surface that states the changes.
        :ptype stated_in: _ContrastSurface
        :return: The form, or None.
        :rtype: MoneyForm | None
        """
        return self._change_money.get(stated_in, {}).get((name, reading))

    # --- measures ----------------------------------------------------------------------------------------------------

    def measure(self, key: str, *, kind: _MeasureKind = "measure") -> MeasureForm:
        """One measure's form.

        :param key: The measure's key.
        :ptype key: str
        :param kind: The measure's kind; None reads as ``measure``, as every decision reads it.
        :ptype kind: _MeasureKind
        :return: Its form.
        :rtype: MeasureForm
        :raises KeyError: The page states no figure of that measure.
        """
        return self._measures[key, _measure_kind(kind)]

    @property
    def measures(self) -> tuple[MeasureForm, ...]:
        """Every measure's form, in the order the page's input holds them."""
        return tuple(self._measures.values())

    # --- the input a written page's forms are of ---------------------------------------------------------------------
    #
    # A figure key resolves on an input and prints in a page's forms. A caller that took the two apart could hand in
    # forms built from another input, and nothing would say so; so a written page's forms carry their input, a key's
    # reader reads it from them (:func:`~threetears.evals.analysis.report.figure_keys.render`), and a reader that is
    # handed both checks them (:meth:`checked`).

    @property
    def report_input(self) -> ReportInput:
        """The report input a written page's forms were built from, which its figure keys resolve on.

        :raises ValueError: The forms are a code page's or an off-page surface's, which hold no report input.
        """
        if self._input is None:
            raise ValueError(f"a {self.page} page's forms hold no report input, which a figure key resolves on")
        return self._input

    def checked(self, report_input: ReportInput) -> PageForms:
        """These forms, where they are ``report_input``'s: for a reader handed an input and its page's forms apart.

        :param report_input: The input the reader resolves its figures on.
        :ptype report_input: ReportInput
        :return: These forms.
        :rtype: PageForms
        :raises ValueError: The forms were built from another input, or hold none.
        """
        if self.report_input is not report_input:
            raise ValueError("a page's forms were handed in beside a report input they were not built from")
        return self

    def over(self, view: ReportInput) -> PageForms:
        """The same page's forms over a view of its input that adds no figure (a configs chart's grid,
        :func:`~threetears.evals.analysis.report_rows.config_grid`): its measures and its ledger are this page's, and
        a figure key resolved on the view reads its set there.

        :param view: The view: the page's input with its factors and configs the grid's.
        :ptype view: ReportInput
        :return: The forms over the view.
        :rtype: PageForms
        """
        viewed = PageForms(self.page, self._measures, view)
        viewed._issued = self._issued
        return viewed

    def level(self, key: str, value: float | None, *, kind: _MeasureKind = "measure", alone: bool = False) -> Printed:
        """A level of a measure (an arm's mean, a slice's), in its measure's form.

        :param key: The measure's key.
        :ptype key: str
        :param value: The level as given; None for none.
        :ptype value: float | None
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :param alone: Whether the level is printed by itself (a chart's note, a key no table prints), which today reads
            by the lone figure rule rather than at its set's decimals.
        :ptype alone: bool
        :return: The level.
        :rtype: Printed
        """
        form = self.measure(key, kind=kind)
        if value is None or not (math.isfinite(value) or self.page == "written"):
            # A written page spells a figure that is no number as its rule does, the dash in its unit (``— ms``).
            return self._issue(form.key, form.kind, "level", value, _ABSENT_TEXT, _ABSENT_TEXT, "")
        if self.page == "written":
            text = _written_level(form, value, alone=alone)
            return self._issue(form.key, form.kind, "level", value, text, _digits_of(text, form.unit), form.unit)
        return self._code_issue(form, "level", value, _code_scaled(form, value), alone=alone)

    def drawn_level(
        self, key: str, value: float | None, *, kind: _MeasureKind = "measure", alone: bool = False
    ) -> Printed:
        """A level of a measure given as the code page draws it (a rate already in percent: a change's baseline, the
        mean of its cases' shares), in its measure's form. A level computed in percent is rounded as computed, never
        divided back into a share first, so its digits are the ones the page read before.

        :param key: The measure's key.
        :ptype key: str
        :param value: The level as drawn; None for none.
        :ptype value: float | None
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :param alone: Whether the level is stated by itself in words (``from v1's 86.7%``), which today spells a rate
            at its form's decimals from the level as drawn, not from the cell's rounded number.
        :ptype alone: bool
        :return: The level.
        :rtype: Printed
        :raises ValueError: The forms are a written page's, whose levels are drawn from their quantity.
        """
        form = self.measure(key, kind=kind)
        if self.page != "code":
            raise ValueError(f"a level of {form.key!r} is given as drawn only on a code page")
        if value is None or not math.isfinite(value):
            return self._issue(form.key, form.kind, "level", value, _ABSENT_TEXT, _ABSENT_TEXT, "")
        if alone and form.reads_as == "rate":
            text = percent_at(value, POINT_PLACES if form.level_decimals is None else form.level_decimals)
            held = _code_held(form, value)
            return self._issue(
                form.key, form.kind, "level", value, text, _digits_of(text, form.unit), form.unit, held=held
            )
        return self._code_issue(form, "level", value, value)

    def change(self, key: str, value: float | None, *, kind: _MeasureKind = "measure", alone: bool = False) -> Printed:
        """A change of a measure (an arm less the control), signed, in its measure's form.

        :param key: The measure's key.
        :ptype key: str
        :param value: The change as given; None for none.
        :ptype value: float | None
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :param alone: Whether the change is printed by itself, which today reads by the lone figure rule.
        :ptype alone: bool
        :return: The change.
        :rtype: Printed
        """
        form = self.measure(key, kind=kind)
        if value is None or not (math.isfinite(value) or self.page == "written"):
            # As a level: a written page spells a change that is no number in its unit (``— pts``).
            return self._issue(form.key, form.kind, "change", value, _ABSENT_TEXT, _ABSENT_TEXT, "")
        if self.page == "written":
            text = _written_change(form, value, alone=alone)
        else:
            text = _code_change(form, value)
        unit = form.change_unit
        if self.page == "code":
            held, decimals = _code_change_held(form, value), form.change_decimals
            return self._issue(
                form.key, form.kind, "change", value, text, _digits_of(text, unit), unit, held=held, decimals=decimals
            )
        return self._issue(form.key, form.kind, "change", value, text, _digits_of(text, unit), unit)

    def half_width(self, key: str, value: float, *, kind: _MeasureKind = "measure") -> Printed:
        """How far a change's interval reaches either side of the change (a comparison's half-width, each arm's own),
        unsigned, in its measure's change unit at the changes' decimals: a code page's precision table.

        :param key: The measure's key.
        :ptype key: str
        :param value: The half-width as given.
        :ptype value: float
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :return: The half-width.
        :rtype: Printed
        :raises ValueError: The forms are a written page's, which states no half-width.
        """
        form = self.measure(key, kind=kind)
        if self.page != "code":
            raise ValueError(f"a half-width of {form.key!r} is stated only on a code page")
        held = _code_change_held(form, value)
        unit = form.change_unit
        text = TableCell(value=held, unit=unit, decimals=form.change_decimals).figure
        return self._issue(
            form.key,
            form.kind,
            "half_width",
            value,
            text,
            _digits_of(text, unit),
            unit,
            held=held,
            decimals=form.change_decimals,
        )

    def interval(
        self, key: str, low: float, high: float, *, kind: _MeasureKind = "measure", change: bool = False
    ) -> _PrintedInterval:
        """An interval on a level or a change of a measure, both ends in its measure's form.

        :param key: The measure's key.
        :ptype key: str
        :param low: The low end as given.
        :ptype low: float
        :param high: The high end as given.
        :ptype high: float
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :param change: Whether the interval is on a change (its ends signed).
        :ptype change: bool
        :return: The interval.
        :rtype: _PrintedInterval
        """
        form = self.measure(key, kind=kind)
        if self.page == "written":
            span = form.unnamed_span if kind is None else form.span
            low_text, high_text, text = _written_interval(form, low, high, change=change, span=span)
        else:
            low_text, high_text, text = _code_interval(form, low, high, change=change)
        unit = form.change_unit if change else form.unit
        low_digits, high_digits = _digits_of(low_text, unit), _digits_of(high_text, unit)
        self._issued.append(_Issued(form.key, form.kind, "interval", (low, high), text))
        return _PrintedInterval(
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

    def bracketed(
        self, key: str, value: float | None, low: float, high: float, *, kind: _MeasureKind = "measure"
    ) -> tuple[Printed, _PrintedInterval]:
        """A level and its interval as one code page cell states them, the interval drawn beside the level: money's
        level and both ends in one form, its decimals chosen over all three (one figure finer where the ends would
        read alike); any other measure's level and interval each in its measure's form, as :meth:`level` and
        :meth:`interval` state them.

        :param key: The measure's key.
        :ptype key: str
        :param value: The level as given; None for none.
        :ptype value: float | None
        :param low: The interval's low end as given.
        :ptype low: float
        :param high: The interval's high end as given.
        :ptype high: float
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :return: The level and the interval.
        :rtype: tuple[Printed, _PrintedInterval]
        :raises ValueError: The forms are a written page's, whose cells draw no interval beside a level this way.
        """
        form = self.measure(key, kind=kind)
        if self.page != "code":
            raise ValueError(f"a level of {form.key!r} is bracketed by its interval only on a code page")
        if form.ruler is None or value is None or not math.isfinite(value):
            return self.level(key, value, kind=kind), self.interval(key, low, high, kind=kind)
        drawn, drawn_low, drawn_high = form.ruler.drawn(value), form.ruler.drawn(low), form.ruler.drawn(high)
        places = form.ruler.form.places_of([drawn, drawn_low, drawn_high], ends=(drawn_low, drawn_high))
        cell = TableCell(value=drawn, unit=MONEY, decimals=places)
        text = cell.figure
        level = self._issue(
            form.key, form.kind, "level", value, text, _digits_of(text, MONEY), MONEY, held=drawn, decimals=places
        )
        low_text, high_text = cell.amount(drawn_low), cell.amount(drawn_high)
        joined = f"[{low_text}, {high_text}]"
        self._issued.append(_Issued(form.key, form.kind, "interval", (low, high), joined))
        low_digits, high_digits = _digits_of(low_text, MONEY), _digits_of(high_text, MONEY)
        interval = _PrintedInterval(
            low=low,
            high=high,
            text=joined,
            low_text=low_text,
            high_text=high_text,
            unit=MONEY,
            low_digits=low_digits,
            high_digits=high_digits,
            digits=_interval_join(self.page, low_digits, high_digits),
        )
        return level, interval

    def sizes(self, key: str, low: float, high: float, *, kind: _MeasureKind = "measure") -> _PrintedInterval:
        """A change's interval as a code page's sentence sizes it (``between 3.5 and 7.9 points worse``): each end's
        size unsigned, both at one precision, the changes' decimals or finer where those would print a non-zero end as
        zero or both ends alike (:func:`~threetears.evals.kernel.numbers.ends_places`).

        :param key: The measure's key.
        :ptype key: str
        :param low: The low end as given.
        :ptype low: float
        :param high: The high end as given.
        :ptype high: float
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :return: The interval, each end's text its size.
        :rtype: _PrintedInterval
        :raises ValueError: The forms are a written page's, whose sentences size an interval by their own keys.
        """
        form = self.measure(key, kind=kind)
        if self.page != "code":
            raise ValueError(f"an interval of {form.key!r} is sized this way only on a code page")
        scaled_low, scaled_high = low * form.change_factor, high * form.change_factor
        decimals = POINT_PLACES if form.change_decimals is None else form.change_decimals
        places = ends_places(scaled_low, scaled_high, decimals)
        low_text, high_text = spell_at(abs(scaled_low), places), spell_at(abs(scaled_high), places)
        text = f"{low_text} to {high_text}"
        # A sentence states each end's size apart (``between 3.5 and 7.9 points worse``), never the two joined, so
        # each is issued apart: the interval, by its ends, and the text the sentence prints of each.
        self._issued += [
            _Issued(form.key, form.kind, "interval", (low, high), low_text),
            _Issued(form.key, form.kind, "interval", (low, high), high_text),
        ]
        return _PrintedInterval(
            low=low,
            high=high,
            text=text,
            low_text=low_text,
            high_text=high_text,
            unit=form.change_unit,
            low_digits=low_text,
            high_digits=high_text,
            digits=text,
        )

    def stated_interval(self, key: str, low: float, high: float, *, kind: _MeasureKind = "measure") -> _PrintedInterval:
        """An interval as a code page's folded table of every measure states it, ``[low, high]``: each end as the fold
        states a value, by the lone figure rule at its cell's precision (a rate in percent to a decimal, any other
        measure to four decimals).

        :param key: The measure's key.
        :ptype key: str
        :param low: The low end as given.
        :ptype low: float
        :param high: The high end as given.
        :ptype high: float
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :return: The interval.
        :rtype: _PrintedInterval
        :raises ValueError: The forms are a written page's, which has no such fold.
        """
        form = self.measure(key, kind=kind)
        if self.page != "code":
            raise ValueError(f"an interval of {form.key!r} is stated this way only on a code page")
        low_text, high_text = (
            _joined(format_number(_code_held(form, _code_scaled(form, end))), form.unit) for end in (low, high)
        )
        text = f"[{low_text}, {high_text}]"
        self._issued.append(_Issued(form.key, form.kind, "interval", (low, high), text))
        low_digits, high_digits = _digits_of(low_text, form.unit), _digits_of(high_text, form.unit)
        return _PrintedInterval(
            low=low,
            high=high,
            text=text,
            low_text=low_text,
            high_text=high_text,
            unit=form.unit,
            low_digits=low_digits,
            high_digits=high_digits,
            digits=_interval_join(self.page, low_digits, high_digits),
        )

    # --- decisions that print nothing --------------------------------------------------------------------------------

    def drawn_held(self, key: str, value: float, *, kind: _MeasureKind = "measure") -> float:
        """The number a code page's cell would hold for a level given as drawn (:meth:`drawn_level`), for a decision
        that compares levels at the precision they read at (whether two baselines read alike) and prints nothing, so
        nothing is issued.

        :param key: The measure's key.
        :ptype key: str
        :param value: The level as drawn.
        :ptype value: float
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :return: The number.
        :rtype: float
        """
        return _code_held(self.measure(key, kind=kind), value)

    def told_apart(self, level: float) -> float:
        """An interval's level, held as a share, to the precision :meth:`confidence_share` prints it at, for a decision
        that compares levels (whether two read alike, or one reads as the nominal level) and prints nothing, so nothing
        is issued. It belongs to no measure, as :meth:`confidence_share`'s level does not.

        :param level: The level, as a share.
        :ptype level: float
        :return: The level to four decimals.
        :rtype: float
        """
        return round(level, 4)

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
        """An interval's level, as every surface prints it (:func:`interval_level`): one decimal, more only where one
        would read a level under 100 as ``100%``, and what the family holds at once beside it where it corrects its
        comparisons together.

        :param level: The level as the input states it (``99.69%``).
        :ptype level: str
        :param family: The contrast family the interval belongs to; None for none.
        :ptype family: ContrastFamily | None
        :return: The level's words.
        :rtype: Printed
        """
        text = interval_level(level, family)
        return self._issue("", "", "confidence", None, text, text, "")

    def confidence_share(self, level: float) -> Printed:
        """An interval's level held as a share (``0.95``, ``0.99688``), as a code page states it today: in percent to the
        hundredth that tells two levels apart (``99.69%``). Its ``held`` is the level to that precision, as a share,
        which is how the page tells whether two levels read alike or one is the nominal level. It belongs to no
        measure: a level is how a family's intervals were read, the same for every measure the family holds, as
        :meth:`confidence`'s is.

        :param level: The level, as a share.
        :ptype level: float
        :return: The level.
        :rtype: Printed
        """
        text = format_number(round(100 * level, 2)) + PERCENT
        return self._issue(
            "", "", "confidence", level, text, text.removesuffix(PERCENT), PERCENT, held=self.told_apart(level)
        )

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
        return self._issue("", "", "share", value, text, text.removesuffix(PERCENT), PERCENT, held=round(100 * value))

    def pass_k(self, key: str, value: float | None, *, kind: _MeasureKind = "measure") -> Printed:
        """A pass^k of a measure (the chance a case passes every one of k runs, 0 to 1), in percent to a decimal, as a
        code page's depth tables hold one today; the column's rule then spells it. It is recorded under its measure,
        which today's form does not read, and a page that states no other figure of the measure (a measurement's
        depth table) holds no form for it, so none is looked up.

        :param key: The measure's key.
        :ptype key: str
        :param value: The pass^k, as a share; None for none.
        :ptype value: float | None
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :return: The pass^k.
        :rtype: Printed
        """
        measure = _measure_kind(kind)
        if value is None or not math.isfinite(value):
            return self._issue(key, measure, "pass_k", value, _ABSENT_TEXT, _ABSENT_TEXT, "")
        held = round(100 * value, POINT_PLACES)
        text = _joined(format_number(held), PERCENT)
        return self._issue(key, measure, "pass_k", value, text, _digits_of(text, PERCENT), PERCENT, held=held)

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

    # --- an off-page surface's figures -------------------------------------------------------------------------------
    #
    # Each takes the measure a figure is of, keyword-only, so the ledger holds it under that measure and the policy
    # can decide it by measure here without a caller changing; a figure of no measure (a spend, a setting, a judge's
    # check, a level of confidence) passes none.

    def lone(
        self,
        value: float | None,
        *,
        signed: bool = False,
        money: bool = False,
        measure: str = "",
        kind: _MeasureKind = None,
    ) -> Printed:
        """A figure an off-page surface states by itself, as it prints one today: by the lone figure rule, a change
        signed, money with its ``$`` on the number inside its sign or bound (``-$1.23``, ``< $0.000001``). Of a
        measure: a pivot's cell, a series' point, a run's dimension mean, a label's precision. Of none: a priced
        band's end, a level of confidence, a result's field, a count of calls.

        :param value: The figure; None for none.
        :ptype value: float | None
        :param signed: Whether it is a change, read signed.
        :ptype signed: bool
        :param money: Whether it is dollars, read with its ``$``.
        :ptype money: bool
        :param measure: The measure's key where the figure is one of its figures; empty for a figure of no measure.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``); unread without a measure.
        :ptype kind: _MeasureKind
        :return: The figure.
        :rtype: Printed
        :raises ValueError: Both ``signed`` and ``money``: the lone figure rule spells no signed money.
        """
        if signed and money:
            raise ValueError("the lone figure rule spells no signed money")
        text = format_signed(value) if signed else format_number(value, prefix="$" if money else "")
        unit = MONEY if money else ""
        return self._issue(*_keyed(measure, kind), "lone", value, text, _digits_of(text, unit), unit)

    def reading(self, value: float, *, measure: str = "", kind: _MeasureKind = None) -> Printed:
        """A reading an off-page table or line states at a fixed two decimals whatever its measure, as each prints one
        today. Of a measure: an arm's mean in the rung-5 readings table. Of none: a judge's kappa and its lower bound
        in the quick print, which say how far the judge agrees with people, not what the eval measured.

        :param value: The reading.
        :ptype value: float
        :param measure: The measure's key where the figure is one of its figures; empty for a figure of no measure.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``); unread without a measure.
        :ptype kind: _MeasureKind
        :return: The reading.
        :rtype: Printed
        """
        text = f"{value:.{_READING_PLACES}f}"
        return self._issue(*_keyed(measure, kind), "reading", value, text, text, "")

    def echoed(self, value: float, *, measure: str = "", kind: _MeasureKind = None) -> Printed:
        """A value an author declared, echoed back in a refusal or an error as Python's general format writes it
        (``0``, ``1``, ``0.5``): the form the declaration is read in, so the message names what was written. Of a
        measure: an end of the range a grader declared its scores in.

        :param value: The declared value.
        :ptype value: float
        :param measure: The measure's key where the figure is one of its figures; empty for a figure of no measure.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``); unread without a measure.
        :ptype kind: _MeasureKind
        :return: The value.
        :rtype: Printed
        """
        text = f"{value:g}"
        return self._issue(*_keyed(measure, kind), "echoed", value, text, text, "")

    def hits(self, rate: float, n: int, *, measure: str = "", kind: _MeasureKind = None) -> Printed:
        """How many of ``n`` trials a rate counts, recovered whole from the rate (``75`` of 90 at 0.8333), as the quick
        print states it beside the rate: its digits bare. Always of a measure's rate, a measure's or a pass/fail
        judged dimension's.

        :param rate: The rate, 0 to 1.
        :ptype rate: float
        :param n: The trials it was read on.
        :ptype n: int
        :param measure: The measure's key where the figure is one of its figures; empty for a figure of no measure.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``); unread without a measure.
        :ptype kind: _MeasureKind
        :return: The count.
        :rtype: Printed
        """
        text = str(round(rate * n))
        return self._issue(*_keyed(measure, kind), "hits", rate, text, text, "")

    def money_set(
        self, amounts: Iterable[float | None], *, measure: str = "", kind: _MeasureKind = None
    ) -> tuple[Printed, ...]:
        """Amounts of money read as one set, in one money form
        (:func:`~threetears.evals.kernel.numbers.money_form`). Of none today: a pairwise pass's spend beside the part it
        ran under.

        :param amounts: The amounts, in dollars as stored; None for an absent one.
        :ptype amounts: Iterable[float | None]
        :param measure: The measure's key where the figure is one of its figures; empty for a figure of no measure.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``); unread without a measure.
        :ptype kind: _MeasureKind
        :return: Each amount, in the order given.
        :rtype: tuple[Printed, ...]
        """
        held = list(amounts)
        form = money_form(held)
        return tuple(self._money(amount, form.spell(amount), measure, kind) for amount in held)

    def money_on(self, ruler: MoneyRuler, amount: float, *, measure: str = "", kind: _MeasureKind = None) -> Printed:
        """An amount of money on a ruler a surface laid over its set, in the ruler's one form. Of none today: every
        arm's total spend, every role's.

        :param ruler: The set's ruler.
        :ptype ruler: MoneyRuler
        :param amount: Dollars, as stored.
        :ptype amount: float
        :param measure: The measure's key where the figure is one of its figures; empty for a figure of no measure.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``); unread without a measure.
        :ptype kind: _MeasureKind
        :return: The amount.
        :rtype: Printed
        """
        return self._money(amount, ruler.spell(amount), measure, kind)

    def check(self, value: float | None, *, measure: str = "", kind: _MeasureKind = None) -> float | None:
        """A judge's-check figure, a part from 0 to 1 or a lean of parts, in percent or points (x 100), at the fewest
        decimals that keep its sign (:func:`_places` of it alone): the number the rung-5 checks table's cell holds,
        which its serializer spells by the one rule. Of none today: order consistency and the first-slot lean say how
        the judge read, not what the eval measured.

        :param value: The figure as read; None for none.
        :ptype value: float | None
        :param measure: The measure's key where the figure is one of its figures; empty for a figure of no measure.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``); unread without a measure.
        :ptype kind: _MeasureKind
        :return: The figure in percent or points, at its decimals; None for none.
        :rtype: float | None
        """
        if value is None:
            return None
        points = value * 100
        decided = round(points, _places([points]))
        self._issue(*_keyed(measure, kind), "check", value, format_number(decided), format_number(decided), "")
        return decided

    def preference(self, share: float | None, difference: tuple[float, float] | None) -> PreferenceShares | None:
        """A candidate's share of a judge's preference and the interval the pass read on the difference of the two
        shares, on the share scale: the two shares run 0 to 1 and sum to 1, so a difference ``d`` is the candidate's
        share ``(1 + d) / 2``. Each end is held within 0 to 100%, the scale a share lies on. The shares and the ends
        read at one precision, the fewest decimals that keep each on its side of none, even and all, and the three
        apart.

        :param share: The candidate's share, 0 to 1; None when nothing was scored.
        :ptype share: float | None
        :param difference: The interval on the difference of the two shares; None for none.
        :ptype difference: tuple[float, float] | None
        :return: The shares; None when nothing was scored.
        :rtype: PreferenceShares | None
        """
        if share is None:
            return None
        points = share * 100
        ends = None if difference is None else [min(100.0, max(0.0, (1 + end) / 2 * 100)) for end in difference]
        places = _even_places([points, *(ends or [])])
        candidate = round(points, places)
        shares = PreferenceShares(
            candidate=candidate,
            control=round(100 - candidate, places),
            interval=None if ends is None else (round(ends[0], places), round(ends[1], places)),
            places=places,
        )
        self._issue("", "", "preference", share, shares.share_text(candidate), shares.share_text(candidate), PERCENT)
        return shares

    def _money(self, amount: float | None, text: str, measure: str, kind: _MeasureKind) -> Printed:
        return self._issue(*_keyed(measure, kind), "money", amount, text, _digits_of(text, MONEY), MONEY)

    # --- a chart's forms: today's, each chart deciding over its own drawn values -----------------------------------
    #
    # A written page's chart draws each quantity in the form its own drawn values give it (a money ruler from the
    # amounts it draws, a note by its quantity's rule), not yet its measure's across the page. These hold that
    # decision, relocated. A measure's figure names its measure and kind, which the ledger records it under; a figure
    # of no measure names neither. The flip re-decides them by measure here.

    @overload
    def chart_quantity(self, unit: str, values: Sequence[float] = (), *, change: bool = False) -> Quantity: ...

    @overload
    def chart_quantity(
        self, unit: str, values: Sequence[float] = (), *, change: bool = False, measure: str, kind: _MeasureKind
    ) -> Quantity: ...

    def chart_quantity(
        self,
        unit: str,
        values: Sequence[float] = (),
        *,
        change: bool = False,
        measure: str = "",
        kind: _MeasureKind = None,
    ) -> Quantity:
        """How a chart draws a quantity in ``unit`` (:func:`~threetears.evals.analysis.report.parts.look.quantity`):
        its factor, the kit's format, its unit, and money's one form from the values it draws.

        :param unit: The figures' unit, as the page reads it.
        :ptype unit: str
        :param values: The drawn values, after the unit's own factor; money's form is chosen from them today.
        :ptype values: Sequence[float]
        :param change: Whether the values are changes, read signed.
        :ptype change: bool
        :param measure: The measure the quantity draws figures of, by key; omitted for a figure of no measure (a
            preference, how often a rerun settles a line, a price per point, an empty chart's axis, a fallback
            format), whose ledger key is ``("", "")``.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``), required with a measure.
        :ptype kind: _MeasureKind
        :return: The quantity.
        :rtype: Quantity
        """
        return quantity(unit, values, change=change)

    @overload
    def chart_quantity_of(self, unit: str, figures: Iterable[float], *, change: bool = False) -> Quantity: ...

    @overload
    def chart_quantity_of(
        self, unit: str, figures: Iterable[float], *, change: bool = False, measure: str, kind: _MeasureKind
    ) -> Quantity: ...

    def chart_quantity_of(
        self,
        unit: str,
        figures: Iterable[float],
        *,
        change: bool = False,
        measure: str = "",
        kind: _MeasureKind = None,
    ) -> Quantity:
        """How a chart draws a quantity in ``unit`` from the figures it draws as the input states them, before any
        factor (:func:`~threetears.evals.analysis.report.parts.look.quantity_of`).

        :param unit: The figures' unit.
        :ptype unit: str
        :param figures: Every figure drawn in it, in the input's units.
        :ptype figures: Iterable[float]
        :param change: Whether the figures are changes.
        :ptype change: bool
        :param measure: The measure the figures are of, by key; omitted for a figure of no measure, whose ledger key is
            ``("", "")``.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``), required with a measure.
        :ptype kind: _MeasureKind
        :return: The quantity.
        :rtype: Quantity
        """
        return quantity_of(unit, figures, change=change)

    def chart_amounts(self, report_input: ReportInput, keys: Sequence[str]) -> Quantity:
        """The one ruler a change chart's money is drawn on: money's scale over every mean the input holds of every
        money measure the chart draws (``keys``, the measures themselves, each a measure's own), read in the first
        one's unit.

        :param report_input: The report input the chart reads.
        :ptype report_input: ReportInput
        :param keys: The money measures the chart draws, at least one.
        :ptype keys: Sequence[str]
        :return: The quantity their amounts are drawn as.
        :rtype: Quantity
        """
        means = [mean for key in keys for mean in held_means(report_input, key)]
        return quantity_of(figure_unit(report_input, keys[0]), means)

    @overload
    def chart_changes(self, amounts: Quantity, changes: Iterable[float], widths: Iterable[float] = ()) -> Quantity: ...

    @overload
    def chart_changes(
        self,
        amounts: Quantity,
        changes: Iterable[float],
        widths: Iterable[float] = (),
        *,
        measure: str,
        kind: _MeasureKind,
    ) -> Quantity: ...

    def chart_changes(
        self,
        amounts: Quantity,
        changes: Iterable[float],
        widths: Iterable[float] = (),
        *,
        measure: str = "",
        kind: _MeasureKind = None,
    ) -> Quantity:
        """How a chart draws changes between money amounts: on the amounts' ruler, in the changes' own form
        (:func:`~threetears.evals.analysis.report.parts.look.in_change_form`).

        :param amounts: The quantity the compared amounts are drawn as.
        :ptype amounts: Quantity
        :param changes: Every change drawn, at the amounts' factor.
        :ptype changes: Iterable[float]
        :param widths: Their intervals' widths, at the amounts' factor.
        :ptype widths: Iterable[float]
        :param measure: The measure the changes are of, by key; omitted for a figure of no measure, whose ledger key is
            ``("", "")``.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``), required with a measure.
        :ptype kind: _MeasureKind
        :return: The changes' quantity.
        :rtype: Quantity
        """
        return in_change_form(amounts, changes, widths)

    @overload
    def chart_note(self, value: float, drawn: Quantity) -> Printed: ...

    @overload
    def chart_note(self, value: float, drawn: Quantity, *, measure: str, kind: _MeasureKind) -> Printed: ...

    def chart_note(self, value: float, drawn: Quantity, *, measure: str = "", kind: _MeasureKind = None) -> Printed:
        """A drawn value as a chart's note, tooltip or line label writes it, by its quantity's rule
        (:func:`~threetears.evals.analysis.report.parts.look.figure_text`): a change where its quantity draws changes,
        a level otherwise, as the ledger records it.

        :param value: The value, as drawn.
        :ptype value: float
        :param drawn: Its quantity.
        :ptype drawn: Quantity
        :param measure: The measure the value is of, by key; omitted for a figure of no measure (a preference, how
            often a rerun settles a line, a price per point, a stored line no criterion keys), whose ledger key is
            ``("", "")``.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``), required with a measure.
        :ptype kind: _MeasureKind
        :return: The value.
        :rtype: Printed
        """
        text = figure_text(value, drawn)
        return self._chart_issue(measure, kind, "change" if drawn.change else "level", value, text, drawn.unit)

    @overload
    def chart_level(self, value: float, drawn: Quantity) -> Printed: ...

    @overload
    def chart_level(self, value: float, drawn: Quantity, *, measure: str, kind: _MeasureKind) -> Printed: ...

    def chart_level(self, value: float, drawn: Quantity, *, measure: str = "", kind: _MeasureKind = None) -> Printed:
        """A level (not a change) on a change chart, unsigned in its quantity's unit: money in its ruler's form at no
        fewer decimals than the chart's ticks, so a tooltip's two levels differ by its bar; any other unit by its
        quantity's rule.

        :param value: The level, as drawn.
        :ptype value: float
        :param drawn: The quantity the chart's levels are drawn as.
        :ptype drawn: Quantity
        :param measure: The measure the level is of, by key; omitted for a figure of no measure, whose ledger key is
            ``("", "")``.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``), required with a measure.
        :ptype kind: _MeasureKind
        :return: The level.
        :rtype: Printed
        """
        form = drawn.money
        if drawn.unit == MONEY and form is not None and form.figures is not None and value != 0:
            places = max(form.places(value), form.decimals)
            text = MoneyForm(per_thousand=form.per_thousand, decimals=places).spell(value)
        else:
            text = figure_text(value, replace(drawn, change=False))
        return self._chart_issue(measure, kind, "level", value, text, drawn.unit)

    @overload
    def chart_interval(self, low: float, high: float, drawn: Quantity) -> _PrintedInterval: ...

    @overload
    def chart_interval(
        self, low: float, high: float, drawn: Quantity, *, measure: str, kind: _MeasureKind
    ) -> _PrintedInterval: ...

    def chart_interval(
        self, low: float, high: float, drawn: Quantity, *, measure: str = "", kind: _MeasureKind = None
    ) -> _PrintedInterval:
        """An interval as a chart's tooltip writes it, ``low to high``, each end by its quantity's rule
        (:func:`_interval_words`).

        :param low: Its low end, as drawn.
        :ptype low: float
        :param high: Its high end, as drawn.
        :ptype high: float
        :param drawn: The quantity drawn.
        :ptype drawn: Quantity
        :param measure: The measure the interval is on, by key; omitted for one of no measure (a preference's),
            whose ledger key is ``("", "")``.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``), required with a measure.
        :ptype kind: _MeasureKind
        :return: The interval.
        :rtype: _PrintedInterval
        """
        low_text, high_text = figure_text(low, drawn), figure_text(high, drawn)
        text = _interval_words(low, high, drawn)
        self._issued.append(_Issued(*_keyed(measure, kind), "interval", (low, high), text))
        low_digits, high_digits = _digits_of(low_text, drawn.unit), _digits_of(high_text, drawn.unit)
        return _PrintedInterval(
            low=low,
            high=high,
            text=text,
            low_text=low_text,
            high_text=high_text,
            unit=drawn.unit,
            low_digits=low_digits,
            high_digits=high_digits,
            digits=_interval_join("written", low_digits, high_digits),
        )

    def chart_count(self, count: float) -> Printed:
        """A count as a chart's words print it, whole and grouped
        (:func:`_count_text`). A count of cases is no measure's
        figure, so it takes none.

        :param count: The count.
        :ptype count: float
        :return: The count.
        :rtype: Printed
        """
        text = _count_text(count)
        return self._issue("", "", "count", count, text, text, "")

    def count_alone(self, value: float | None) -> Printed:
        """A count as a chart's words print one today where they spell it by the lone figure rule, ungrouped
        (``1234``), unlike :meth:`count`. A count of passes, trials or cases is no measure's figure, so it takes none.

        :param value: The count; None for none.
        :ptype value: float | None
        :return: The count.
        :rtype: Printed
        """
        text = format_number(value)
        return self._issue("", "", "count", value, text, text, "")

    def gate_line(self, report_input: ReportInput, criterion: Criterion) -> Printed:
        """A criterion's threshold as a gate row's line states it: as its figure key reads it, money on its
        criterion's ruler with its count named once. It is a figure of what the criterion reads
        (``criterion.measure``), which the ledger records it under by the one rule for a criterion's kind
        (:func:`~threetears.evals.analysis.report.figure_keys.held_kind`).

        :param report_input: The report input.
        :ptype report_input: ReportInput
        :param criterion: The criterion.
        :ptype criterion: Criterion
        :return: The threshold.
        :rtype: Printed
        """
        line = resolve(f"criteria.{criterion.id}.threshold", report_input)
        threshold = float(criterion.threshold)
        key = (criterion.measure, held_kind(report_input, criterion.measure))
        if line.scale not in ("usd", "usd_per_1k", "usd_per_1m"):
            text = format_figure(line, forms=self)
            return self._issue(*key, "declared", threshold, text, text, "")
        ruler, _own = criterion_ruler(report_input, criterion, line)
        amount = ruler.spell(threshold)
        text = f"{amount} {ruler.words(line.per)}"
        return self._issue(*key, "declared", threshold, text, _digits_of(amount, MONEY), MONEY)

    def _chart_issue(self, measure: str, kind: _MeasureKind, role: Role, value: float, text: str, unit: str) -> Printed:
        """A chart's figure recorded under its measure and role, or under ``("", "")`` where it is of no measure."""
        return self._issue(*_keyed(measure, kind), role, value, text, _digits_of(text, unit), unit)

    # --- a figure key's figures: today's, each key's set read as its table prints it --------------------------------
    #
    # A figure key (:mod:`~threetears.evals.analysis.report.figure_keys`) prints a plain figure of a measure at the
    # precision the page's tables print that measure's set at, and an interval's two ends at one precision. These hold
    # that decision, relocated. Each prints a part of a key's text, which ``format_figure`` finishes with the key's
    # filters and unit, so nothing is issued; the flip decides a key's whole text here, by its measure and role.

    def key_places(self, key: str, *, kind: _MeasureKind, change: bool) -> int | None:
        """The decimals a figure key's plain figure of a measure reads at, as the page's tables print it: a judged
        score's two, any other measure's at its set's (:func:`figure_places`), its levels' or its changes'; None where
        the measure has no such set, and the number rule spells its figure. A decision that prints nothing, so nothing
        is issued.

        :param key: The measure's or judged dimension's key.
        :ptype key: str
        :param kind: The measure's kind: ``judged`` for a 1–5 score, which reads at a score's two decimals.
        :ptype kind: _MeasureKind
        :param change: Whether the figure is a change, read at its changes' set.
        :ptype change: bool
        :return: The decimals, or None.
        :rtype: int | None
        :raises ValueError: The forms hold no report input (a code page's, an off-page surface's), which a key's set
            is read from.
        """
        if self._input is None:
            raise ValueError(
                f"a figure key's set of {key!r} is read from a written page's report input, and none is held"
            )
        return figure_places(self._input, key, change=change, judged=_measure_kind(kind) == "judged")

    def key_number(self, figure: Figure, value: float) -> str:
        """A plain figure's number as a figure key prints it, unsigned and without its unit: at its measure's set's
        decimals (:meth:`key_places`) where the figure is one the page's tables print of that measure (a level, a
        change, an interval or one of its ends, a criterion's figure or its gap), else by the lone figure rule, as an
        adjusted p or a count of repeats keeps it.

        :param figure: What the key resolved to: its measure's key and kind and its statistic say which set it is of.
        :ptype figure: Figure
        :param value: The number, as the key spells it (an interval's end, or the figure's own value).
        :ptype value: float
        :return: Its digits.
        :rtype: str
        """
        places = self._key_figure_places(figure)
        return format_number(value) if places is None else spell_at(value, places)

    def key_plain_ends(
        self, figure: Figure, values: tuple[float, float], numbers: tuple[str, str], *, absolute: bool
    ) -> tuple[str, str] | None:
        """An interval's two plain ends at one precision, as every set reads: the decimals the page's tables print the
        figure's set at (:meth:`key_number`), else two significant figures of the interval's width, which is what its
        ends can tell apart (``2.84–3.00`` for 2.839 to 3), never padded to its finer end's digits; more only where an
        end that is not zero would read as one. An end from :data:`~threetears.evals.kernel.numbers.WHOLE_FROM` up, or
        one read as a bound, leaves both as they are.

        :param figure: What the key resolved to.
        :ptype figure: Figure
        :param values: The two ends, as the key reads them (sorted by size where ``absolute``).
        :ptype values: tuple[float, float]
        :param numbers: The two ends as the key spelled each alone, with the figure's unit.
        :ptype numbers: tuple[str, str]
        :param absolute: Whether the key reads the ends' sizes (``abs``), unsigned.
        :ptype absolute: bool
        :return: The two ends' texts, each with the figure's unit; None where they stay as spelled.
        :rtype: tuple[str, str] | None
        """
        unit = f" {figure.unit}" if figure.unit else ""
        bare = [number.removesuffix(unit) if unit else number for number in numbers]
        if any(abs(value) >= WHOLE_FROM for value in values) or any(not _PLAIN_END.fullmatch(n) for n in bare):
            return None
        shown = [abs(value) if absolute else value for value in values]
        held = self._key_figure_places(figure)
        places = ends_places(shown[0], shown[1], held if held is not None else set_places([shown[1] - shown[0]]))
        signed = figure.change and not absolute
        texts = [spell_at(value, places) for value in shown]
        low, high = (
            plus_signed(text, value) if signed and float(text) != 0 else text
            for text, value in zip(texts, shown, strict=True)
        )
        return low + unit, high + unit

    def key_points_ends(self, figure: Figure, values: tuple[float, float], *, absolute: bool) -> tuple[str, str]:
        """An interval in points with both ends at one precision, as every set reads: a point's one decimal, a whole
        end too (``-29.0 to +57.6``), or more where an end that is not zero would read as one (``-2.2 to -0.04``, never
        ``-2.2 to 0.0``, which would hide which side of zero it lies). A lone figure in points keeps the number rule,
        which writes a whole one whole.

        :param figure: What the key resolved to; a share's ends are read in points (x 100).
        :ptype figure: Figure
        :param values: The two ends, as the key reads them.
        :ptype values: tuple[float, float]
        :param absolute: Whether the key reads the ends' sizes (``abs``), unsigned.
        :ptype absolute: bool
        :return: The two ends' texts.
        :rtype: tuple[str, str]
        """
        hundred = [value * 100 if figure.scale == "share" else value for value in values]
        at = ends_places(hundred[0], hundred[1], POINT_PLACES)
        low, high = (
            spell_at(abs(value), at) if absolute else plus_signed(spell_at(value, at), value) for value in hundred
        )
        return low, high

    def _key_figure_places(self, figure: Figure) -> int | None:
        """A key's figure's set's decimals (:meth:`key_places`), where the figure is one the page's tables print of the
        measure it is of (:func:`_tabled`); None otherwise. A criterion's gap is a difference on its measure, read at
        its changes' set."""
        if figure.set_of is None or not _tabled(figure):
            return None
        key, kind = figure.set_of
        change = figure.change or (bool(figure.measure) and figure.statistic in _CRITERION_GAPS)
        return self.key_places(key, kind=kind, change=change)

    # --- the ledger --------------------------------------------------------------------------------------------------

    @property
    def ledger(self) -> tuple[_Issued, ...]:
        """Every text handed out, in the order it was."""
        return tuple(self._issued)

    def issued(self, key: str, *, kind: _MeasureKind = "measure", role: Role | None = None) -> tuple[_Issued, ...]:
        """The texts handed out for one measure, optionally of one role.

        :param key: The measure's key.
        :ptype key: str
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :param role: The role; None for every role.
        :ptype role: Role | None
        :return: The entries, in the order they were handed out.
        :rtype: tuple[_Issued, ...]
        """
        return tuple(
            entry
            for entry in self._issued
            if (entry.measure, entry.kind) == (key, _measure_kind(kind)) and (role is None or entry.role == role)
        )

    def _issue(
        self,
        key: str,
        kind: _MeasureKind,
        role: Role,
        value: float | None,
        text: str,
        digits: str,
        unit: str,
        *,
        held: float | None = None,
        decimals: int | None = None,
    ) -> Printed:
        self._issued.append(_Issued(key, kind, role, value, text))
        return Printed(value=value, text=text, digits=digits, unit=unit, held=held, decimals=decimals)

    def _code_issue(
        self, form: MeasureForm, role: Role, value: float, scaled: float, *, alone: bool = False
    ) -> Printed:
        """A code page's level, from its value scaled into the unit it is printed in: the number its cell holds, the
        decimals the cell writes it at, and its text; ``alone``, a rate or a measured figure by the lone figure rule at
        its cell's precision, as the folded table of every measure states it."""
        if form.ruler is not None:
            places = form.ruler.form.places_of([scaled])
            text = TableCell(value=scaled, unit=MONEY, decimals=places).figure
            return self._issue(
                form.key, form.kind, role, value, text, _digits_of(text, MONEY), MONEY, held=scaled, decimals=places
            )
        held = _code_held(form, scaled)
        if alone:
            text = _joined(format_number(held), form.unit)
            return self._issue(
                form.key, form.kind, role, value, text, _digits_of(text, form.unit), form.unit, held=held
            )
        text = _code_cell_text(form, held)
        return self._issue(
            form.key,
            form.kind,
            role,
            value,
            text,
            _digits_of(text, form.unit),
            form.unit,
            held=held,
            decimals=form.level_decimals,
        )


#: What an absent figure reads as.
_ABSENT_TEXT = format_number(None)

#: The decimals an off-page reading is stated to (:meth:`PageForms.reading`).
_READING_PLACES = 2

#: Where a share of a judge's preference is even, in percent: the judge preferred neither page.
EVEN_SHARE = 50.0

#: The most decimals a side-by-side figure is read to (:func:`_places`).
_MOST_SIDE_BY_SIDE_PLACES = 6


def _places(values: Sequence[float]) -> int:
    """The fewest decimals, none at first, at which every one of ``values`` keeps its sign (no end that is not 0 prints
    as 0, and none crosses zero) and values that differ still differ: an interval's ends read as distinct and a
    crossing stays visible, whatever the rounding (P19)."""
    for places in range(_MOST_SIDE_BY_SIDE_PLACES + 1):
        rounded = [round(value, places) for value in values]
        if all(
            (low > 0) == (value > 0) and (low < 0) == (value < 0) for low, value in zip(rounded, values, strict=True)
        ):
            if len(set(rounded)) == len(set(values)):
                return places
    return _MOST_SIDE_BY_SIDE_PLACES


def _even_places(values: Sequence[float]) -> int:
    """The fewest decimals at which shares in percent keep their side of none (0), of even (50) and of all (100), and
    shares that differ still differ (:func:`_places` about each): a lean off even, or an end short of none or all,
    stays visible whatever the rounding."""
    return max(_places([value - edge for value in values]) for edge in (0.0, EVEN_SHARE, 100.0))


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


def _interval_join(page: _PageKind, low: str, high: str) -> str:
    """Two ends joined as the page joins an interval: ``low to high`` on a written page, ``[low, high]`` on a code
    page's scorecard."""
    return f"[{low}, {high}]" if page == "code" else f"{low} to {high}"


def _keyed(measure: str, kind: _MeasureKind) -> tuple[str, str]:
    """The key an off-page or a chart's figure is held under in the ledger: its measure and kind, or neither for a
    figure of no measure, as the fixed forms are held."""
    return (measure, _measure_kind(kind)) if measure else ("", "")


def _counts_passes(report_input: ReportInput) -> bool:
    """Whether some scoreboard (the lead aspect's, or another aspect's own) counts an arm's passes over its usable
    trials, which a table then states as the arm's pass rate."""
    aspects = (None, *(aspect.id for aspect in report_input.aspects))
    return any(
        row.passed is not None and bool(row.usable)
        for aspect in aspects
        for row in aspect_board(report_input, aspect).values()
    )


def _measure_kind(kind: _MeasureKind) -> str:
    """A measure's kind as its forms are keyed: a contrast that names none is a measure's own, as every decision reads
    it, so one measure never takes two forms."""
    return "measure" if kind is None else kind


# --- an interval's level, an interval's words and a count, as every surface prints them -------------------------------


def interval_level(level: str, family: ContrastFamily | None) -> str:
    """An interval's level as every surface prints it, a chart's head and a table's cell alike: to one decimal
    (``99.69%`` reads ``99.7%``), more only where one would round a level under 100 to ``100%``, which no interval holds
    at (``99.95%`` stays ``99.95%``); a level stored in words, as stored. Where ``family`` corrects its comparisons
    together at this level, what the family holds at once follows (``99.7% (95% across 16 comparisons)``), so a reader
    never meets a bare ``99.69%``.

    :param level: The level as the input states it (``99.69%``).
    :ptype level: str
    :param family: The contrast family the interval belongs to, or None.
    :ptype family: ContrastFamily | None
    :return: The level's words.
    :rtype: str
    """
    shown = _readable_level(level)
    if family is None or family.level != level or family.alpha is None:
        return shown
    corrected = (family.correction or "").strip().lower() not in ("", "none")
    if not corrected or not family.family_size or family.family_size < 2:
        return shown
    joint = f"{format_number((1 - float(family.alpha)) * 100)}%"
    return FAMILY_LEVEL_TEMPLATE.format(level=shown, joint=joint, size=family.family_size)


def _readable_level(level: str) -> str:
    number = level.strip().removesuffix("%")
    try:
        value = float(number)
    except ValueError:
        # NOSILENT: a level stored in words is printed as the input wrote it; no number is lost.
        return level
    places = 1
    while value < 100 and round(value, places) >= 100 and places < len(number.partition(".")[2]):
        places += 1
    text = spell_at(value, places)
    return f"{text.rstrip('0').rstrip('.')}%"


def _interval_words(low: float, high: float, drawn: Quantity) -> str:
    """An interval in a tooltip: its two ends by the package's number rule, joined as prose joins them.

    :param low: Its low end, in display units.
    :ptype low: float
    :param high: Its high end, in display units.
    :ptype high: float
    :param drawn: The quantity drawn.
    :ptype drawn: Quantity
    :return: The words (``-10.8 to -3.4``).
    :rtype: str
    """
    return f"{figure_text(low, drawn)} to {figure_text(high, drawn)}"


def _count_text(count: float) -> str:
    """A count as a chart's words print it: the figure keys' whole-number spelling, thousands separated
    (``1,234``), so a count reads alike on a chart and in the prose beside it.

    :param count: The count.
    :ptype count: float
    :return: The text.
    :rtype: str
    """
    return spell_count(round(count))


#: How every family spells a level the tables print and its interval's two ends.
_LEVEL_FIGURES = frozenset({"mean", "low", "high"})

#: How the criteria families spell the level of the measure a criterion reads, and how they and the weakest margin
#: spell its gap from the threshold.
_CRITERION_LEVEL = "value"
_CRITERION_GAPS = frozenset({"gap", "margin"})

#: A plain end's digits: a sign, the digits and a decimal part, never a bound or grouped thousands.
_PLAIN_END = re.compile(r"[+\-]?\d+(?:\.\d+)?")


def _tabled(figure: Figure) -> bool:
    """Whether a key's figure is one the page's tables print of a measure or a judged score, so the key reads it at the
    table's precision. Decided by what the figure is, not by how its family spells it: a change of any family, its
    interval and that interval's ends (each resolver marks them :attr:`~threetears.evals.analysis.report.figure_keys.
    Figure.change`, whether its family spells the change ``delta``, ``difference``, ``effect``, ``change`` or
    ``vs_control``); any interval; and a level, which every family but a criterion's spells ``mean``, with its
    interval's two ends, or a key naming no statistic, which reads its family's own figure; a criterion's figure is the
    level of the measure it reads (``Figure.measure``), which its family spells ``value``, and its ``gap`` (the weakest
    margin's ``margin``) a difference on that measure. No other statistic is the table's figure: an adjusted p of
    0.0031 is never a score to two decimals, nor a half-width or a standard error a change."""
    statistic = figure.statistic
    if figure.change or figure.kind == "interval" or statistic is None:
        return True
    return statistic in _LEVEL_FIGURES or (bool(figure.measure) and statistic in (_CRITERION_LEVEL, *_CRITERION_GAPS))


def figure_places(
    report_input: ReportInput, key: str, *, change: bool = False, judged: bool = False, factor: float = 1.0
) -> int | None:
    """The decimals every plain figure of a measure or judged dimension reads at wherever the page prints it, its Full
    results and the keys its prose cites alike: a judged score's two (:data:`~threetears.evals.kernel.numbers.
    SCORE_PLACES`); any other figure at two significant figures of the least of its set as drawn
    (:func:`~threetears.evals.kernel.numbers.set_places`), its levels (every mean the input holds of it) or its changes
    (every corrected comparison's on it), each a set of its own. None for a figure with no such set: a judged level off
    the rubric, a key that is neither, or a measure the input holds no figure of in that set, whose figure keeps the
    number rule.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure's or judged dimension's key.
    :ptype key: str
    :param change: Whether the figures are changes.
    :ptype change: bool
    :param judged: Whether the figures are a judged dimension's (a comparison of kind ``judged``), not a measure's.
    :ptype judged: bool
    :param factor: What each figure is multiplied by where it is drawn: the set is placed as it prints.
    :ptype factor: float
    :return: The decimals, or None.
    :rtype: int | None
    """
    if rubric_score(report_input, key) and (judged or key not in report_input.measures):
        return SCORE_PLACES
    if change:
        held = [
            float(contrast.delta)
            for contrast in report_input.contrasts
            if contrast.measure == key and (contrast.kind == "judged") == judged and contrast.delta is not None
        ]
    elif judged or key not in report_input.measures:
        return None
    else:
        held = held_means(report_input, key)
    # A measure the input holds no such figure of has no set: its figure keeps the number rule, never zero decimals.
    return set_places(value * factor for value in held) if held else None


# --- the written page: the Full results' decisions ---------------------------------------------------------------------


def _level_quantity(report_input: ReportInput, key: str, *, kind: str | None = None) -> Quantity:
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


def _change_quantity(
    report_input: ReportInput, measure: str, kind: str | None, contrasts: Sequence[Contrast]
) -> Quantity:
    """How a measure's changes are drawn on a written page: on its levels' own ruler (:func:`_level_quantity`), so a
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
    amounts = _level_quantity(report_input, measure, kind=kind)
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


def _level_decimals(report_input: ReportInput, key: str, drawn: Quantity, *, kind: str | None = None) -> int | None:
    """The decimals every level of a measure reads at in every Full results table, by the one owner the keys citing it
    read too (:func:`figure_places`), as drawn: a judged score at two,
    any other measured figure (a count per trial, a time) at two significant figures of the least of every mean the
    input holds of it, so its column reads at one precision (``1.4`` beside ``1.5``). None for a rate, whose column
    rule spells it, and for money, whose form does.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure's or judged dimension's key.
    :ptype key: str
    :param drawn: How its levels are drawn (:func:`_level_quantity`).
    :ptype drawn: Quantity
    :param kind: A contrast's kind where the key is one's; None for a measure's.
    :ptype kind: str | None
    :return: The decimals, or None.
    :rtype: int | None
    """
    if drawn.unit in (MONEY, PERCENT, POINTS):
        return None
    return figure_places(report_input, key, judged=kind == "judged", factor=drawn.factor)


def _change_decimals(report_input: ReportInput, measure: str, kind: str | None, drawn: Quantity) -> int | None:
    """The decimals a comparison's change and its interval's ends read at: a rate's in points at one decimal (``-6.0
    pts`` beside ``-7.6 pts``); a score's at two and any other measured change at two significant figures of the least
    change the page states of its measure, as drawn, by the one owner the keys citing it read too
    (:func:`figure_places`); None for money, whose form spells it.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param measure: The measure's key.
    :ptype measure: str
    :param kind: The comparison's kind.
    :ptype kind: str | None
    :param drawn: How its changes are drawn (:func:`_change_quantity`).
    :ptype drawn: Quantity
    :return: The decimals, or None.
    :rtype: int | None
    """
    if drawn.unit == MONEY:
        return None
    if drawn.unit == POINTS:
        return POINT_PLACES
    return figure_places(report_input, measure, change=True, judged=kind == "judged", factor=drawn.factor)


def _change_span(report_input: ReportInput, measure: str, kind: str | None, drawn: Quantity) -> float | None:
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


def _at_decimals(value: float, places: int, unit: str, *, signed: bool = False) -> str:
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


def _change_ends(low: float, high: float, places: int, unit: str, *, span: float | None) -> tuple[str, str]:
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
    return _at_decimals(low, at, unit, signed=True), _at_decimals(high, at, unit, signed=True)


def _written_form(report_input: ReportInput, key: str, kind: _MeasureKind) -> MeasureForm:
    """One measure's form on a written page, from the Full results' decisions; a measure's own kind is None there."""
    stated: str | None = None if kind == "measure" else kind
    levels = _level_quantity(report_input, key, kind=stated)
    held = [c for c in report_input.contrasts if (c.measure, _measure_kind(c.kind)) == (key, kind)]
    changes = _change_quantity(report_input, key, kind, held)
    reads_as: _ReadsAs = (
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
        level_decimals=_level_decimals(report_input, key, levels, kind=stated),
        change_decimals=_change_decimals(report_input, key, kind, changes),
        money=levels.money,
        change_money=changes.money,
        span=_change_span(report_input, key, kind, changes),
        levels=levels,
        changes=changes,
        scale=_RUBRIC_SCALE_TEXT if reads_as == "score" else "",
        unnamed_span=_change_span(report_input, key, stated, changes),
    )


#: The 1–5 rubric as a score's name carries it (``Outcome (1–5)``), so its two decimals read on a scale a reader knows.
_RUBRIC_SCALE_TEXT = f"{format_number(RUBRIC_SCALE[0])}–{format_number(RUBRIC_SCALE[1])}"


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
        return _at_decimals(value * drawn.factor, form.level_decimals, drawn.unit)
    return figure_text(value * drawn.factor, drawn)


def _written_change(form: MeasureForm, value: float, *, alone: bool) -> str:
    """A change as the Full results' comparisons state it: signed at its set's decimals; money or a lone change by its
    quantity's note rule."""
    drawn = _drawn(form, change=True)
    if form.change_decimals is not None and not alone:
        return _at_decimals(value * drawn.factor, form.change_decimals, drawn.unit, signed=True)
    return figure_text(value * drawn.factor, drawn)


def _written_interval(
    form: MeasureForm, low: float, high: float, *, change: bool, span: float | None
) -> tuple[str, str, str]:
    """An interval as the written page states it, ``low to high``: a change's as the Full results' comparisons do, its
    ends kept within how far the measure can move and read at one precision; a level's as a chart's tooltip does."""
    drawn = _drawn(form, change=change)
    if not change or form.change_decimals is None:
        scaled = (low * drawn.factor, high * drawn.factor)
        text = _interval_words(scaled[0], scaled[1], drawn)
        return figure_text(scaled[0], drawn), figure_text(scaled[1], drawn), text
    low_text, high_text = _change_ends(
        low * drawn.factor, high * drawn.factor, form.change_decimals, drawn.unit, span=span
    )
    return low_text, high_text, _interval_join("written", low_text, high_text)


# --- the code page: the scorecard's decisions --------------------------------------------------------------------------


def _measured_decimals(figures: Iterable[float]) -> int | None:
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
    # A rate is decided before money, so a measure both a rate and money would print its money through the rate's form;
    # no measure the package reads is both, and the scorecard's cells check money first.
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
        round(found[0].value, _MEASURED_PLACES)
        for found in cells.values()
        if len(found) == 1 and found[0].value is not None
    ]
    decimals = _measured_decimals(figures)
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


def _code_scaled(form: MeasureForm, value: float) -> float:
    """A level as given, scaled into the unit the code page prints it in: money on its ruler, a share in percent."""
    return form.ruler.drawn(value) if form.ruler is not None else value * form.factor


def _code_held(form: MeasureForm, scaled: float) -> float:
    """The number a scorecard's cell holds for a level, scaled: a rate in percent to a decimal, any other figure to
    four decimals."""
    return round(scaled, POINT_PLACES if form.reads_as == "rate" else _MEASURED_PLACES)


def _code_cell_text(form: MeasureForm, held: float) -> str:
    """A level's text as the scorecard's cell states it, from the number the cell holds: a rate in percent, any other
    figure at its column's decimals."""
    unit = PERCENT if form.reads_as == "rate" else ""
    return TableCell(value=held, unit=unit, decimals=form.level_decimals).figure


def _code_change_held(form: MeasureForm, value: float) -> float:
    """The number a scorecard's cell holds for a change (or how far one reaches), scaled into its unit and rounded to
    the changes' decimals, a point's one where they have none."""
    decimals = form.change_decimals
    return round(value * form.change_factor, POINT_PLACES if decimals is None else decimals)


def _code_change(form: MeasureForm, value: float) -> str:
    """A change as the scorecard's cell states it: signed, in points at a point's one decimal for a rate."""
    return TableCell(
        value=_code_change_held(form, value), unit=form.change_unit, decimals=form.change_decimals, signed=True
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


# --- an aspect page's contrasts: today's sets, relocated --------------------------------------------------------------


def _under_floor(comparison: FamilyComparison) -> bool:
    """Whether a contrast's fewer side is under the case floor (:data:`~threetears.evals.analysis.report.verdict.
    MIN_CASES`), where it states no change."""
    return min(comparison.control.n_cases, comparison.contrast.n_cases) < MIN_CASES


def _resolvable(comparison: FamilyComparison) -> bool:
    """Whether a contrast can resolve, and so is a row of the contrasts table: both sides hold cases, at the floor."""
    return min(comparison.control.n_cases, comparison.contrast.n_cases) > 0 and not _under_floor(comparison)


def _reading_places(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    ruler: MeasureRuler,
    name: str,
    reading: str,
    levels: Iterable[float],
    changes: Iterable[float],
) -> ContrastPlaces:
    """One reading's :class:`ContrastPlaces`, from every level and change of it a surface states, drawn on its ruler.

    A rate reads its changes in points at one decimal; a judged score (a dimension on a scale other than 0 to 1) at
    two decimals, its levels and changes alike; money by its own form; every other measured figure (a count per
    trial, a time) at two significant figures of its set's least figure but zero (:func:`set_places`), its levels and
    its changes each a set of their own.
    """
    if is_money(ruler.unit):
        return ContrastPlaces(None, None, floor=0.0)
    if ruler.unit == PERCENT:
        return ContrastPlaces(None, POINT_PLACES, floor=0.0, ceiling=100.0)
    if reading == "judged":
        facts = surface.dimensions.get(name)
        bounds = facts.value_range if facts is not None else None
        if bounds is not None:
            low, high = (bound * ruler.factor for bound in bounds)
            scale = f"{format_number(low)}–{format_number(high)}"
            return ContrastPlaces(SCORE_PLACES, SCORE_PLACES, floor=low, ceiling=high, scale=scale)
    described = bundle.measure_catalog.get(name) or METRIC_DESCRIPTORS.get(name) if reading == "measure" else None
    floor = described.interval_floor if described is not None else None
    ceiling = described.value_range[1] if described is not None and described.value_range is not None else None
    return ContrastPlaces(
        set_places(levels),
        set_places(changes),
        floor=floor * ruler.factor if floor is not None else None,
        ceiling=ceiling * ruler.factor if ceiling is not None else None,
    )


def _contrast_places(
    bundle: AnalysisContextBundle, surface: DecisionSurface, rulers: Rulers, comparisons: Iterable[FamilyComparison]
) -> dict[tuple[str, str], ContrastPlaces]:
    """Each reading's :class:`ContrastPlaces` over ``comparisons``: its set is every arm's figure and every change on
    it, by name and reading."""
    held: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for comparison in comparisons:
        ruler = rulers.of(comparison.name, comparison.reading)
        levels, changes = held.setdefault((comparison.name, comparison.reading), ([], []))
        levels += [
            side.mean * ruler.factor for side in (comparison.control, comparison.contrast) if side.mean is not None
        ]
        if comparison.delta is not None:
            changes.append(comparison.delta * ruler.factor)
    return {
        (name, reading): _reading_places(bundle, surface, rulers.of(name, reading), name, reading, levels, changes)
        for (name, reading), (levels, changes) in held.items()
    }


def _change_money(
    rulers: Rulers, changes: Iterable[tuple[str, str, float | None, tuple[float, float] | None]]
) -> dict[tuple[str, str], MoneyForm]:
    """The one form each money measure's changes read in, a change and its interval's ends alike (#993): from every
    change a surface states of the measure and the widths of their intervals, on the measure's ruler
    (:meth:`~threetears.evals.kernel.numbers.MoneyForm.for_changes`), by name and reading. A measure of another unit
    has none."""
    held: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for name, reading, delta, interval in changes:
        ruler = rulers.of(name, reading)
        if ruler.money is None or not is_money(ruler.unit):
            continue
        deltas, widths = held.setdefault((name, reading), ([], []))
        if delta is not None:
            deltas.append(delta * ruler.factor)
        if interval is not None:
            widths.append(abs(interval[1] - interval[0]) * ruler.factor)
    forms: dict[tuple[str, str], MoneyForm] = {}
    for (name, reading), (deltas, widths) in held.items():
        money = rulers.of(name, reading).money
        assert money is not None
        forms[name, reading] = money.for_changes(deltas, widths)
    return forms
