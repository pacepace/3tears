"""One owner of a page's number forms: every figure's form is decided here, once per page, by its measure.

A figure's form is its digits, sign, unit and interval as printed. Spelling one number is the leaf's
(:mod:`threetears.evals.kernel.numbers`); deciding which figures form a set, and how many digits the set reads at, is
this module's, through its parts. A set is the measure across the whole page (owner ruling, 2026-11-08), so every
figure of one measure is decided by one :class:`MeasureForm`, never by the surface that prints it.

- :class:`PageForms` is built once per page, before any part is assembled: :meth:`PageForms.of` for a written page
  (a report input), :meth:`PageForms.of_campaign` for a campaign's code page (its readings and its aspects' pages),
  :meth:`PageForms.of_bundle` for an aspect's own page (its bundle, which holds no report input: one set per reading
  over every surface; :meth:`PageForms.of_aspects` for a campaign's aspects together, each set across the campaign). A written page's forms carry their input
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
  (:meth:`PageForms.key_plain_ends`, :meth:`PageForms.key_points_ends`), and every other number a key prints is
  spelled here by the filter that chose it (:meth:`PageForms.key_percent`, :meth:`PageForms.key_points`,
  :meth:`PageForms.key_money`, :meth:`PageForms.key_whole`, :meth:`PageForms.key_multiple`, its ends recorded by
  :meth:`PageForms.key_ends`): every caller of the keys hands its page's forms in, since this module reads the keys'
  parts and so none of them can read this one. A chart's interval level reads :meth:`PageForms.drawn_confidence`.
- Each returns a :class:`Printed` (or a :class:`_PrintedInterval`): the value and its decided text, with the parts a
  renderer may lay out apart.

**Today's decisions, relocated, not rewritten.** Until the policy flip every form here reproduces what the page prints
now, so routing a surface through the forms moves no golden. Where the page decides a measure's form, that decision
lives with the forms (in their parts) and its old module calls it: the Full results' quantities and decimals (a
measure's levels, its changes, how far a change reaches, a change's interval's ends) and the scorecard's column of
measured figures' decimals. A surface that calls one of these free functions still decides there, so the rule's test
counts each such call as a site (``tests/test_one_owner_of_number_forms.py``); only :class:`PageForms`, asked by measure
and role, is the migrated route. What a measure is (a rate, money on a ruler, shown at all) is
:mod:`~threetears.evals.analysis.report.campaign.measure_kinds`'s, and where a decision is a public function of another
module (a measure's unit, :func:`~threetears.evals.analysis.report.figure_keys.figure_unit`), the form calls it.

**A written page under the number policy.** A written page's measure reads at one precision on the whole page, read
from its uncertainty (:func:`policy_places` over its set's half-widths, :func:`_measure_sets`), and every figure of it,
in a table, a chart, a key's prose or alone, prints at that precision in the policy's one form (:func:`policy_rate`,
:func:`policy_figure`, :func:`policy_interval`); which figures share a set is :func:`campaign_set`'s alone. A chart's
forms (``PageForms.chart_*``) read a figure's measure's precision where they are handed its measure and kind, and
decide over the chart's own drawn values only where they are not. A written page's table respells no rate
(:meth:`PageForms.as_printed` records what each prints). Money's column rule (:meth:`PageForms.money_columns`) holds
on every page, deciding over the amounts a table's builder left to it (:class:`ColumnMoney`) and recording what every
money cell prints; and a stored table's glyph titles and scale line (:meth:`PageForms.glyph_said`,
:meth:`PageForms.scale_span`) join the texts its cells carry. A table cell holding a number carries the text its page
decided (``value_text``, and its interval's ends'), and is refused without it: nothing but these forms spells a cell's
number.

**Why at page assembly.** The bundle is fingerprinted evidence and the report input a stored snapshot that recorded
writer runs replay, so neither carries presentation. Page assembly is the first place that sees the whole page, and
both page kinds already pass one object through every part, so a page's forms ride beside it.

The parts :class:`PageForms` decides with live in :mod:`threetears.evals.analysis.report.number_forms`, one module
per part; this module holds the class, the contrasts an aspect page's sets read, and names what hosts import.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, overload

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME, AnalysisContextBundle, FamilyComparison
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS, delta_unit, is_money, number_in, with_unit
from threetears.evals.analysis.report.blocks import LANDS_AT, CellValue, TableCell
from threetears.evals.analysis.report.campaign.measure_kinds import money_rulers, rate_measures, shown_measures
from threetears.evals.analysis.report.charts import KitChart
from threetears.evals.analysis.report.cost_unit import MoneyRuler, money_ruler
from threetears.evals.analysis.report.figures import ContrastPlaces, MeasureRuler, Rulers, report_rulers
from threetears.evals.analysis.report.keys.figure import Figure
from threetears.evals.analysis.report.keys.prices import MULTIPLE_UNIT
from threetears.evals.analysis.report.keys.reads import criterion_ruler, held_means
from threetears.evals.analysis.report.keys.resolve import (
    figure_unit,
    format_figure,
    held_kind,
    plus_signed,
    resolve,
    rubric_score,
)
from threetears.evals.analysis.report.number_forms.code_page import (
    _code_cell_text,
    _code_change,
    _code_change_held,
    _code_change_places,
    _code_held,
    _code_interval,
    _code_places,
    _code_scaled,
    _rate_form,
    _said_held,
    _scorecard_form,
)
from threetears.evals.analysis.report.number_forms.intervals import (
    _PLAIN_END,
    _count_text,
    _interval_words,
    _tabled,
    interval_level,
)
from threetears.evals.analysis.report.number_forms.kinds import Role, _MeasureKind, _PageKind
from threetears.evals.analysis.report.number_forms.policy import (
    _ABSENT_TEXT,
    _PREFERENCE_BOUNDS,
    _RATE_BOUNDS,
    _ZERO_BOUND,
    policy_bounded_places,
    policy_declared,
    policy_effect,
    policy_figure,
    policy_interval,
    policy_level,
    policy_lone_interval,
    policy_p,
    policy_places,
    policy_rate,
)
from threetears.evals.analysis.report.number_forms.printed import (
    ColumnMoney,
    MeasureForm,
    PreferenceShares,
    Printed,
    _Issued,
    _PrintedInterval,
)
from threetears.evals.analysis.report.number_forms.sets import (
    EVEN_SHARE,
    MeasureSet,
    _GRID,
    _INTERVAL_ENDS,
    _READING_PLACES,
    _SCORE_HUNDREDTHS_FROM,
    _a_rate_cell,
    _aspect_page,
    _bundle_stated,
    _campaign_stated,
    _counts_passes,
    _decimals_in,
    _digits_of,
    _drawn_texts,
    _glyph_ends,
    _key_role,
    _key_ruler,
    _key_set,
    _keyed,
    _measure_kind,
    _measure_sets,
    _places,
    _rate_words,
    _set_widths,
    _signed_text,
)
from threetears.evals.analysis.report.number_forms.written import (
    _placed_text,
    _written_change,
    _written_form,
    _written_interval,
    _written_level,
    _written_stated,
)
from threetears.evals.analysis.report.parts.look import Quantity, figure_text, in_change_form, quantity, quantity_of
from threetears.evals.analysis.report.per_input import kept_for
from threetears.evals.analysis.report.said import Said, SaidFigure
from threetears.evals.analysis.report.unit_once import interval_alone, interval_beside, unit_after, unit_spaced
from threetears.evals.analysis.report.verdict import MIN_CASES, interval_words
from threetears.evals.analysis.report_input import ContrastFamily, Criterion, ReportInput
from threetears.evals.kernel.metrics import METRIC_DESCRIPTORS
from threetears.evals.kernel.numbers import (
    POINT_PLACES,
    SCORE_PLACES,
    WHOLE_FROM,
    MoneyForm,
    ends_places,
    format_number,
    format_signed,
    group_thousands,
    money_form,
    round_half_even,
    set_places,
    spell_at,
    spell_count,
    spell_dollars,
    spell_multiple,
    spell_stored_counts,
)
from threetears.evals.kernel.surface import DecisionSurface

__all__ = [
    "EVEN_SHARE",
    "ColumnMoney",
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
        # An aspect page's contrasts' and guardrails' decisions, one per reading (:meth:`of_bundle`).
        self._bundle: AnalysisContextBundle | None = None
        self._contrast_places: dict[tuple[str, str], ContrastPlaces] = {}
        self._change_money: dict[tuple[str, str], MoneyForm] = {}
        # Each table cell a builder made as a figure of a measure (:meth:`of_measure`), by the cell's id, with that
        # measure and its reading, so a column rule that spells the cell records it under them. The cell is held beside
        # its id, so the id names that cell for as long as these forms are read.
        self._cell_measures: dict[int, tuple[TableCell, str, str]] = {}

    @classmethod
    def of(cls, report_input: ReportInput) -> PageForms:
        """A written page's forms: one per measure, judged dimension and the pass rate its input holds a figure of,
        each as the page's Full results decide it. A slice (a stratum, a task, a cell) can hold a figure on a key the
        input declares no measure for, which its table prints as a measure's; and a scoreboard that counts passes
        states the pass rate, so each has its form too.

        The forms are pure in the input, which is frozen, so they are worked out once per input
        (:func:`~threetears.evals.analysis.report.per_input.kept_for`): an analyst tool that cites each figure on
        forms of its own reads the input's forms back rather than deciding every measure again. Each call is still a
        page of its own, with its own ledger of what it hands out.

        :param report_input: The page's report input.
        :ptype report_input: ReportInput
        :return: The page's forms.
        :rtype: PageForms
        """
        return cls(
            "written",
            kept_for(report_input).get(_written_measures, lambda: _written_measures(report_input)),
            report_input,
        )

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
        states, each at the places its uncertainty sets (:func:`policy_places`) over the campaign's set of it
        (:func:`campaign_set`): the page's own intervals and every aspect page's.

        :param readings: What code read off the campaign.
        :ptype readings: CampaignReadings
        :param reports: Each aspect's own page, by its label.
        :ptype reports: Mapping[str, Report]
        :return: The page's forms.
        :rtype: PageForms
        """
        campaign = _measure_sets(_campaign_stated([report.bundle for report in reports.values()], readings))
        measures: dict[tuple[str, _MeasureKind], MeasureForm] = {}
        if readings.readings or readings.pairs:
            # A change's readings and a choice's pairs hold the pass rate's changes and intervals in points already,
            # and its levels as shares.
            widths = _set_widths(campaign, "code", TRIAL_PASS_NAME, "trial")
            measures[TRIAL_PASS_NAME, "trial"] = _rate_form(TRIAL_PASS_NAME, "trial", widths, change_factor=1.0)
        shown = shown_measures(readings)
        rulers = money_rulers(readings, reports)
        rates = rate_measures(readings, reports, {})
        for key in dict.fromkeys(reading.measure for reading in shown):
            widths = _set_widths(campaign, "code", key, "measure")
            measures[key, "measure"] = _scorecard_form(key, shown, rates, rulers, widths)
        forms = cls("code", measures)
        # Each aspect's readings' sets, so a figure the page states on an aspect's ruler (a bar's value and interval)
        # reads as that aspect's page states it.
        for report in reports.values():
            bundle = report.bundle
            surface = bundle_decision_surface(bundle)
            compared = [
                comparison for family in bundle.multiple_comparisons.families for comparison in family.comparisons
            ]
            held = _contrast_places(bundle, surface, report_rulers(surface, bundle.measure_catalog), compared, campaign)
            forms._contrast_places = {**held, **forms._contrast_places}
        return forms

    @classmethod
    def of_bundle(cls, bundle: AnalysisContextBundle) -> PageForms:
        """An aspect's own page's forms, built alone (code's page over the aspect's bundle, which holds no report
        input), for an aspect that is its own campaign: its set of each reading is its own. An aspect of a campaign
        reads its campaign's sets (:meth:`of_aspects`, the aspects and the code page's readings together), and every
        surface that states its figures (its page, the listing's verdict, the quick line, a program's rows) takes those
        forms, so none prints the aspect at its own precision beside the campaign's page.

        :param bundle: The aspect's evidence.
        :ptype bundle: AnalysisContextBundle
        :return: The page's forms.
        :rtype: PageForms
        """
        return cls._of_bundle(bundle, _measure_sets(_campaign_stated([bundle])))

    @classmethod
    def of_aspects(
        cls, bundles: Sequence[AnalysisContextBundle], readings: CampaignReadings | None = None
    ) -> dict[str, PageForms]:
        """Every aspect page's forms of one campaign, by the aspect's label: each reading at the places the campaign's
        set of it sets (:func:`campaign_set`), its aspects' intervals and the code page's (``readings``), so a measure
        reads at one precision on every page of the campaign.

        :param bundles: The campaign's aspects' bundles.
        :ptype bundles: Sequence[AnalysisContextBundle]
        :param readings: What code read off the campaign, whose intervals join the set; None for none.
        :ptype readings: CampaignReadings | None
        :return: Each aspect's forms.
        :rtype: dict[str, PageForms]
        """
        campaign = _measure_sets(_campaign_stated(bundles, readings))
        return {bundle.aspect: cls._of_bundle(bundle, campaign) for bundle in bundles}

    @classmethod
    def _of_bundle(cls, bundle: AnalysisContextBundle, campaign: Mapping[MeasureSet, Sequence[float]]) -> PageForms:
        """An aspect's own page's forms: each reading's places, one set per reading over every surface the page states
        it on (its contrasts, the verdict line's sentences, a program's rows, its guardrails), read from the campaign's
        set of it; and each money measure's one form for its changes."""
        forms = cls("aspect", {})
        forms._bundle = bundle
        surface = bundle_decision_surface(bundle)
        rulers = report_rulers(surface, bundle.measure_catalog)
        compared = [comparison for family in bundle.multiple_comparisons.families for comparison in family.comparisons]
        forms._contrast_places = _contrast_places(bundle, surface, rulers, compared, campaign)
        guardrails = bundle.guardrails.checks if bundle.guardrails is not None else []
        forms._change_money = _change_money(
            rulers,
            [
                *((c.name, c.reading, c.delta, c.interval) for c in compared),
                *((check.name, check.reading, check.delta, check.interval) for check in guardrails),
            ],
        )
        return forms

    @property
    def bundle(self) -> AnalysisContextBundle:
        """The bundle an aspect page's forms were built from.

        :raises ValueError: The forms are another page's, which hold no bundle.
        """
        if self._bundle is None:
            raise ValueError(f"a {self.page} page's forms hold no aspect's bundle")
        return self._bundle

    def contrast_places(self, name: str, reading: str) -> ContrastPlaces | None:
        """How one reading's figures read on an aspect's page, on every surface that states them: its set's places.

        :param name: The reading's measure or dimension.
        :ptype name: str
        :param reading: Which reading of it (``measure``, ``judged``, ``trial``).
        :ptype reading: str
        :return: The places; None where the page states no contrast of the reading.
        :rtype: ContrastPlaces | None
        """
        return self._contrast_places.get((name, reading))

    def change_money(self, name: str, reading: str) -> MoneyForm | None:
        """The one form a money measure's changes read in on an aspect's page, a change and its interval's ends alike
        (#993), on every surface that states them; None for a measure of another unit.

        :param name: The measure's name.
        :ptype name: str
        :param reading: Which reading of it.
        :ptype reading: str
        :return: The form, or None.
        :rtype: MoneyForm | None
        """
        return self._change_money.get((name, reading))

    def recorded(
        self, rows: Sequence[Mapping[str, TableCell]], *, bare: Mapping[str, str] | None = None
    ) -> list[dict[str, TableCell]]:
        """A code or an aspect page's table, each rate and money cell recorded as it prints, under the measure its
        builder named (:meth:`of_measure`), and carrying that text (``value_text``): its builder decided the cell's
        form by its measure's set, so nothing respells it, and the ledger holds every rate and money figure the table
        prints, as it holds every other.

        :param rows: The table's rows, by column key, as their builders made them.
        :ptype rows: Sequence[Mapping[str, TableCell]]
        :param bare: The columns whose head names their unit (:func:`~threetears.evals.analysis.report.blocks.bare_columns`);
            None for none.
        :ptype bare: Mapping[str, str] | None
        :return: The rows, each rate and money cell recorded.
        :rtype: list[dict[str, TableCell]]
        """
        named = bare or {}
        return [
            {
                key: self._issue_cell(cell, cell)
                if _a_rate_cell(cell, key in named)
                or _rate_words(cell, key in named)
                or (cell.is_number and is_money(cell.unit))
                else cell
                for key, cell in row.items()
            }
            for row in rows
        ]

    # --- an aspect's own page: each figure at its reading's one set ---------------------------------------------------
    #
    # An aspect's page states each figure on its reading's ruler for the report
    # (:class:`~threetears.evals.analysis.report.figures.MeasureRuler`), at the places its reading's set reads at across
    # the campaign (:meth:`contrast_places`) and in the money form its changes read in (:meth:`change_money`), whichever
    # surface states it. Each method takes the ruler, reads the set's places off these forms where its caller hands
    # none, and records what it prints under the reading it is a figure of (its measure or dimension, and ``measure``,
    # ``judged`` or ``trial``).

    @staticmethod
    def small_n(comparison: FamilyComparison) -> int | None:
        """The fewer of a contrast's two case counts where it is under the case floor
        (:data:`~threetears.evals.analysis.report.verdict.MIN_CASES`), where the contrast states no change, interval,
        effect or p, and so joins no set of changes; None where both sides reach it.

        :param comparison: The contrast.
        :ptype comparison: FamilyComparison
        :return: The fewer count, or None.
        :rtype: int | None
        """
        n = min(comparison.control.n_cases, comparison.contrast.n_cases)
        return n if n < MIN_CASES else None

    def ruled_level(
        self,
        ruler: MeasureRuler,
        value: float,
        places: ContrastPlaces | None = None,
        *,
        measure: str = "",
        kind: _MeasureKind = None,
        role: Role = "level",
    ) -> Printed:
        """A level (an arm's mean, a spread's centre) on its reading's ruler, the unit on the figure where the unit
        sits there (``98.9%``, ``$0.0459``), bare where the row's heading names it (a latency's ``950`` under
        ``Turn time (ms)``); at the set's places for levels where ``places`` gives them (``2.83``, ``1.4``), a level of
        exactly zero ``0`` (nothing happened, which no precision qualifies), as a column of percents reads ``0%``;
        money in the ruler's one form (#993).

        :param ruler: The reading's ruler.
        :ptype ruler: MeasureRuler
        :param value: The level, as stored.
        :ptype value: float
        :param places: The surface's set's places; None where it holds none.
        :ptype places: ContrastPlaces | None
        :param measure: The measure or dimension it is a level of; empty for none.
        :ptype measure: str
        :param kind: Its reading (``measure``, ``judged``, ``trial``).
        :ptype kind: _MeasureKind
        :param role: What the figure is to its measure, as the ledger records it: ``level``, or ``share`` for a share
            at the level's digits that is no level of the measure (an outcome bar's passes of every trial it ran,
            lost ones too).
        :ptype role: Role
        :return: The level.
        :rtype: Printed
        """
        text = _ruled_level_text(ruler, value * ruler.factor, self._set_places(places, measure, kind))
        unit = _on_the_figure(ruler.unit)
        return self._issue(*_keyed(measure, kind), role, value, text, _digits_of(text, unit), unit)

    def ruled_spread(
        self,
        ruler: MeasureRuler,
        sem: float,
        *,
        mean: float | None = None,
        measure: str = "",
        kind: _MeasureKind = None,
    ) -> Printed:
        """A mean's standard error on its ruler, stated beside the mean as ``± spread``, at the mean's precision (rule
        1: one measure, one precision): money to the decimals the mean beside it reads to (``$3.33 ± $0.09``, never
        ``$0.088``), any other unit as a bare number at its reading's places where the page holds them, else at the
        decimals the mean beside it reads to; never rounded to a zero it is not, where it takes the figures that keep
        it off zero.

        :param ruler: The reading's ruler.
        :ptype ruler: MeasureRuler
        :param sem: The standard error, as stored.
        :ptype sem: float
        :param mean: The mean it is stated beside, as stored; None where it stands alone, which reads it by its own
            figures.
        :ptype mean: float | None
        :param measure: The measure or dimension it is the spread of; empty for none.
        :ptype measure: str
        :param kind: Its reading.
        :ptype kind: _MeasureKind
        :return: The spread.
        :rtype: Printed
        """
        scaled = sem * ruler.factor
        if is_money(ruler.unit) and ruler.money is not None:
            decimals = ruler.money.places_of([mean * ruler.factor]) if mean is not None else None
            if decimals is not None and (scaled == 0 or float(round_half_even(scaled, decimals)) != 0):
                text = MoneyForm(decimals=decimals).spell(scaled)
            else:
                text = money_form([scaled], rate=ruler.money.rate).spell(scaled)
            return self._issue(*_keyed(measure, kind), "half_width", sem, text, _digits_of(text, MONEY), MONEY)
        # At its reading's places, as the mean beside it reads, kept off zero: a spread is never none.
        places = self._set_places(None, measure, kind)
        if places is not None and places.change is not None:
            text = policy_figure(scaled, places.change, bounds=_ZERO_BOUND)
        elif mean is not None:
            beside = _ruled_level_text(ruler, mean * ruler.factor, None)
            text = policy_figure(scaled, _decimals_in(beside), bounds=_ZERO_BOUND)
        else:
            text = number_in(scaled, ruler.unit)
        return self._issue(*_keyed(measure, kind), "half_width", sem, text, text, "")

    def ruled_interval(
        self,
        ruler: MeasureRuler,
        low: float,
        high: float,
        places: ContrastPlaces | None = None,
        *,
        measure: str = "",
        kind: _MeasureKind = None,
    ) -> _PrintedInterval:
        """A level's interval in its one form, ``low to high``, on its ruler (an arm's figure's, a bar's value's): at
        its reading's set's places for levels, never past what a level can take (a count's never below zero, a score's
        within its scale), each end kept off zero and a rate's off its floor and ceiling (``88.1 to 99.7%``), the unit
        once after the high end (``0.45 to 0.55 s``); money's ends in the ruler's one form, each with its ``$``; with no
        set's places, each end at the unit's precision.

        :param ruler: The reading's ruler.
        :ptype ruler: MeasureRuler
        :param low: The low end, as stored.
        :ptype low: float
        :param high: The high end, as stored.
        :ptype high: float
        :param places: The surface's set's places; None where it holds none.
        :ptype places: ContrastPlaces | None
        :param measure: The measure or dimension it is an interval of; empty for none.
        :ptype measure: str
        :param kind: Its reading.
        :ptype kind: _MeasureKind
        :return: The interval.
        :rtype: _PrintedInterval
        """
        drawn = [low * ruler.factor, high * ruler.factor]
        places = self._set_places(places, measure, kind)
        if is_money(ruler.unit) and ruler.money is not None:
            texts = _money_ends(ruler.money.spell_ends(drawn[0], drawn[1]), drawn[0], drawn[1], change=False)
        elif places is not None and places.level is not None:
            if places.floor is not None:
                drawn = [max(end, places.floor) for end in drawn]
            if places.ceiling is not None:
                drawn = [min(end, places.ceiling) for end in drawn]
            bounds = _RATE_BOUNDS if ruler.unit == PERCENT else ()
            texts = _ends_at(drawn[0], drawn[1], places.level, ruler.unit, change=False, bounds=bounds)
        else:
            texts = _ends_in(drawn[0], drawn[1], ruler.unit, change=False)
        return self._ruled_interval(measure, kind, low, high, texts, ruler.unit)

    def ruled_change_interval(
        self,
        ruler: MeasureRuler,
        low: float,
        high: float,
        *,
        form: MoneyForm | None = None,
        places: ContrastPlaces | None = None,
        measure: str = "",
        kind: _MeasureKind = None,
    ) -> _PrintedInterval:
        """An interval on a change in its one form, ``low to high``, its ends signed, on its ruler in the unit a change
        is stated in: money's ends in the form its changes read in (``form``, :meth:`change_money`), each with its
        ``$``; any other unit's at its reading's set's places (``-6.9 to +11.0 pts``), each end kept off zero on its
        own (``-15 to -0.03 pts``), never past what a change can be (a score's from -4 to 4 on a 1–5 scale); a rate's in
        points at a point's one decimal where no set holds places; else each end by the unit's precision.

        :param ruler: The reading's ruler.
        :ptype ruler: MeasureRuler
        :param low: The low end, as stored.
        :ptype low: float
        :param high: The high end, as stored.
        :ptype high: float
        :param form: The form the reading's changes read in on the surface; None for none.
        :ptype form: MoneyForm | None
        :param places: The surface's set's places; None where it holds none.
        :ptype places: ContrastPlaces | None
        :param measure: The measure or dimension it is a change of; empty for none.
        :ptype measure: str
        :param kind: Its reading.
        :ptype kind: _MeasureKind
        :return: The interval.
        :rtype: _PrintedInterval
        """
        held = self._set_places(places, measure, kind)
        if held is not None and held.change is not None:
            texts = _change_ends(ruler, low, high, form=form, places=held.change, span=held.span())
        else:
            texts = _change_ends(ruler, low, high, form=form, places=None, span=None)
        return self._ruled_interval(measure, kind, low, high, texts, delta_unit(ruler.unit))

    def ruled_level_cell(
        self,
        ruler: MeasureRuler,
        value: float | None,
        places: ContrastPlaces | None = None,
        *,
        measure: str = "",
        kind: _MeasureKind = None,
    ) -> TableCell:
        """A level as a table cell: restated onto its ruler and carrying its unit; any unit but a percent and money at
        the set's places for levels where ``places`` gives them, as the contrasts table prints the same mean; money in
        the form its ruler chose over every figure of the measure, at the places the value needs in it, so two arms'
        means in one row read alike (#993).

        A cell a table's column rule spells (a rate's, money) is recorded there, as it prints
        (:meth:`money_columns`); any other is recorded here.

        :param ruler: The reading's ruler.
        :ptype ruler: MeasureRuler
        :param value: The level, as stored; None for none.
        :ptype value: float | None
        :param places: The surface's set's places; None where it holds none.
        :ptype places: ContrastPlaces | None
        :param measure: The measure or dimension it is a level of; empty for none.
        :ptype measure: str
        :param kind: Its reading.
        :ptype kind: _MeasureKind
        :return: The cell.
        :rtype: TableCell
        """
        if value is None:
            return TableCell(value=None)
        drawn = value * ruler.factor
        places = self._set_places(places, measure, kind)
        if ruler.money is not None and is_money(ruler.unit):
            money = ruler.money
            text = MoneyForm(decimals=money.places_of([drawn])).spell(drawn)
            return self._ruled_cell(measure, kind, "level", value, drawn, ruler.unit, text)
        if places is not None and places.level is not None:
            text = _ruled_level_text(ruler, drawn, places)
            if ruler.unit and ruler.unit != PERCENT:
                text = unit_spaced(text, ruler.unit)
            return self._ruled_cell(measure, kind, "level", value, drawn, ruler.unit, text)
        return self._ruled_cell(measure, kind, "level", value, drawn, ruler.unit, with_unit(drawn, ruler.unit))

    def ruled_change_cell(
        self,
        ruler: MeasureRuler,
        delta: float | None,
        *,
        interval: tuple[float, float] | None = None,
        margin: float | None = None,
        form: MoneyForm | None = None,
        places: int | None = None,
        measure: str = "",
        kind: _MeasureKind = None,
    ) -> TableCell:
        """A change as a table cell: on its ruler, in the unit a change is stated in, signed (its direction is half the
        fact), at the set's places for changes where ``places`` gives them (``-6.0 pts`` beside ``+3.6 pts``, never
        ``-6 pts``), a rate's at a point's one decimal otherwise. Handed the interval on it, the cell carries it, with
        the declared margin, on the same ruler, read against zero: what a page draws beside the figure as the row's
        glyph, its ends as the row's interval column prints them (:meth:`ruled_change_interval`). Handed the form the
        reading's changes read in (:meth:`change_money`), money's change is written in it, at the places its value and
        its interval's ends need.

        A cell a table's column rule spells (a rate's, money) is recorded there, as it prints; any other is recorded
        here.

        :param ruler: The reading's ruler.
        :ptype ruler: MeasureRuler
        :param delta: The change, as stored; None for none.
        :ptype delta: float | None
        :param interval: The interval on the change, as stored; None to carry none.
        :ptype interval: tuple[float, float] | None
        :param margin: The declared margin, as stored; None for none.
        :ptype margin: float | None
        :param form: The form the reading's changes read in on the surface; None for none.
        :ptype form: MoneyForm | None
        :param places: The surface's set's places for changes; None where it holds none.
        :ptype places: int | None
        :param measure: The measure or dimension it is a change of; empty for none.
        :ptype measure: str
        :param kind: Its reading.
        :ptype kind: _MeasureKind
        :return: The cell.
        :rtype: TableCell
        """
        if delta is None:
            return TableCell(value=None)
        unit = delta_unit(ruler.unit)
        held = self._set_places(None, measure, kind)
        if places is None and held is not None:
            places = held.change
        if places is None and ruler.unit == PERCENT:
            places = POINT_PLACES
        # The interval's ends read at the set's places, each kept off zero on its own; only the figure takes more to
        # stay off zero itself.
        set_places = places
        if places is not None and form is None:
            places = policy_bounded_places(
                delta * ruler.factor, places, bounds=_ZERO_BOUND if ruler.unit == PERCENT else ()
            )
        drawn = delta * ruler.factor
        ends = None if interval is None else (interval[0] * ruler.factor, interval[1] * ruler.factor)
        money: MoneyForm | None = None
        if form is not None:
            decided = form.places_of([float(drawn), *(ends or (None, None))], ends=ends or (None, None))
            money = MoneyForm(decimals=decided)
            text = _signed_text(money.spell(drawn), drawn)
        elif places is None:
            text = _signed_text(with_unit(drawn, unit), drawn)
        else:
            digits = policy_figure(drawn, places, bounds=_ZERO_BOUND if ruler.unit == PERCENT else (), signed=True)
            text = unit_spaced(digits, unit)
        if interval is None:
            return self._ruled_cell(measure, kind, "change", delta, drawn, unit, text)
        # The glyph's ends are the row's interval column's, each decided by its rule (:meth:`ruled_change_interval`),
        # so the title and the cell beside it print one interval one way (``-15 to -0.03``, never ``-15.09``).
        span = held.span() if held is not None else None
        low_text, high_text, _ = _change_ends(
            ruler,
            interval[0],
            interval[1],
            form=form,
            places=set_places,
            span=100.0 if span is None and ruler.unit == PERCENT else span,
        )
        if not is_money(unit):
            written = _on_the_figure(unit)
            low_text, high_text = _digits_of(low_text, written), _digits_of(high_text, written)
        return self._ruled_cell(
            measure,
            kind,
            "change",
            delta,
            drawn,
            unit,
            text,
            low=interval[0] * ruler.factor,
            high=interval[1] * ruler.factor,
            low_text=low_text,
            high_text=high_text,
            margin=abs(margin) * ruler.factor if margin is not None else None,
            zero_line=True,
        )

    def ruled_allows(
        self,
        arm: str,
        ruler: MeasureRuler,
        interval: tuple[float, float],
        *,
        form: MoneyForm | None = None,
        places: ContrastPlaces | None = None,
        higher_is_better: bool = True,
        measure: str = "",
        kind: _MeasureKind = None,
    ) -> str:
        """What an interval on an arm's change from the control allows, as the verdict line's figures say it
        (:func:`~threetears.evals.analysis.report.verdict.interval_words`): each end's size unsigned, as the contrasts
        table prints the change. Points bare, the sentence naming them once; money in the form its changes read in;
        any other unit beside the number. Where the set holds places for changes, the ends read as the interval's cell
        prints them: within what a change can be, at one precision, finer only where that would print the two ends
        alike or a non-zero end as zero (:func:`~threetears.evals.kernel.numbers.ends_places`), so ``6.0 points``
        never reads ``6 points`` beside ``-6.0 pts``.

        :param arm: The arm, as the page names it.
        :ptype arm: str
        :param ruler: The reading's ruler.
        :ptype ruler: MeasureRuler
        :param interval: The interval on the change, as stored.
        :ptype interval: tuple[float, float]
        :param form: The form the reading's changes read in on the surface; None for none.
        :ptype form: MoneyForm | None
        :param places: The surface's set's places; None where it holds none.
        :ptype places: ContrastPlaces | None
        :param higher_is_better: The reading's direction.
        :ptype higher_is_better: bool
        :param measure: The measure or dimension it is a change of; empty for none.
        :ptype measure: str
        :param kind: Its reading.
        :ptype kind: _MeasureKind
        :return: The sentence, with its closing stop.
        :rtype: str
        """
        low, high = (end * ruler.factor for end in interval)
        places = self._set_places(places, measure, kind)
        if places is not None and places.change is not None and (span := places.span()) is not None:
            low, high = max(low, -span), min(high, span)

        def size(value: float) -> str:
            return self._ruled_size(ruler, value, form=form, places=places, measure=measure, kind=kind)

        return interval_words(
            arm,
            low,
            high,
            amount=size,
            unit=" points" if ruler.unit == PERCENT else "",
            higher_is_better=higher_is_better,
        )

    def loss_share(self, share: float) -> Printed:
        """A share of an arm's trials lost (0 to 1), or the share past which loss differs between arms, in percent by
        the lone figure rule (``12.5%``), as the methods' disclosure states it.

        :param share: The share.
        :ptype share: float
        :return: The share.
        :rtype: Printed
        """
        digits = format_number(100 * share)
        return self._issue("", "", "share", share, digits + PERCENT, digits, PERCENT)

    def stored_counts(self, sentence: str) -> str:
        """A sentence the bundle stores, with its counts as a page prints a count (:meth:`count`): ``6224 of 6224
        score(s)`` reads ``6,224 of 6,224 score(s)``. The stored sentence stays as assembled, since the bundle
        fingerprints it; only its printing groups the counts.

        :param sentence: The sentence as stored.
        :ptype sentence: str
        :return: The sentence as printed.
        :rtype: str
        """
        return spell_stored_counts(sentence)

    def _set_places(self, places: ContrastPlaces | None, measure: str, kind: _MeasureKind) -> ContrastPlaces | None:
        """The places a ruled figure reads at: those its caller read off these forms, else its reading's set's on an
        aspect page, so every figure of a reading reads at one precision whichever table states it."""
        if places is not None or not measure:
            return places
        return self._contrast_places.get((measure, _measure_kind(kind)))

    def _ruled_size(
        self,
        ruler: MeasureRuler,
        value: float,
        *,
        form: MoneyForm | None,
        places: ContrastPlaces | None,
        measure: str,
        kind: _MeasureKind,
    ) -> str:
        """A change's size as a sentence states it, unsigned, from its value already on the ruler: money in its form,
        points bare, any other unit beside the number, at the set's places for changes where ``places`` gives them."""
        if form is not None and is_money(ruler.unit):
            text = form.spell(abs(value))
            self._issued.append(_Issued(*_keyed(measure, kind), "change", value, text))
            return text
        unit = delta_unit(ruler.unit)
        # Each end is kept off zero on its own, as the interval's cell prints it (``0.03 points``, never ``0 points``).
        number = (
            policy_figure(abs(value), places.change, bounds=_ZERO_BOUND)
            if places is not None and places.change is not None
            else number_in(abs(value), unit)
        )
        text = number if ruler.unit == PERCENT or not unit else f"{number} {unit}"
        self._issued.append(_Issued(*_keyed(measure, kind), "change", value, text))
        return text

    def _ruled_interval(
        self, measure: str, kind: _MeasureKind, low: float, high: float, texts: tuple[str, str, str], unit: str
    ) -> _PrintedInterval:
        """An aspect page's interval from its two ends' texts and its whole text, recorded under its reading."""
        low_text, high_text, text = texts
        self._issued.append(_Issued(*_keyed(measure, kind), "interval", (low, high), text))
        written = _on_the_figure(unit)
        low_digits, high_digits = _digits_of(low_text, written), _digits_of(high_text, written)
        return _PrintedInterval(
            low=low,
            high=high,
            text=text,
            low_text=low_text,
            high_text=high_text,
            unit=unit,
            low_digits=low_digits,
            high_digits=high_digits,
            digits=interval_beside(low_digits, high_digits),
        )

    def _ruled_cell(
        self,
        measure: str,
        kind: _MeasureKind,
        role: Role,
        value: float,
        drawn: float,
        unit: str,
        text: str,
        **interval: Any,
    ) -> TableCell:
        """A cell an aspect page's builder decided, holding ``drawn`` in ``unit`` and carrying its text
        (``value_text``) and, where ``interval`` gives one, its interval with its ends' texts: a rate's or money's held
        under its measure (:meth:`of_measure`), which the page's tables record as they print (:meth:`recorded`); any
        other recorded here, under its measure."""
        decided = TableCell(value=drawn, unit=unit, value_text=text, **interval)
        if unit in (PERCENT, POINTS) or is_money(unit):
            return self.of_measure(decided, measure=measure, kind=kind)
        self._issued.append(_Issued(*_keyed(measure, kind), role, value, text))
        return decided

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
        :func:`~threetears.evals.analysis.report.rows.config_grid`): its measures and its ledger are this page's, and
        a figure key resolved on the view reads its set there.

        :param view: The view: the page's input with its factors and configs the grid's.
        :ptype view: ReportInput
        :return: The forms over the view.
        :rtype: PageForms
        :raises ValueError: The view is no view of the input these forms were built from (any field but its factors and
            configs is another's), or the forms hold no input.
        """
        held = self.report_input
        if view is not held and any(
            getattr(view, name) is not getattr(held, name) for name in type(held).model_fields if name not in _GRID
        ):
            # A view shares every field but the grid with the input it is of (``model_copy``), so a figure read on it
            # is the page's own: another input handed in here would print in this page's forms unchecked.
            raise ValueError("a page's forms were handed a view of a report input they were not built from")
        viewed = PageForms(self.page, self._measures, view)
        viewed._issued = self._issued
        return viewed

    def level(self, key: str, value: float | None, *, kind: _MeasureKind = "measure") -> Printed:
        """A level of a measure (an arm's mean, a slice's), in its measure's form.

        :param key: The measure's key.
        :ptype key: str
        :param value: The level as given; None for none.
        :ptype value: float | None
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :return: The level.
        :rtype: Printed
        """
        form = self.measure(key, kind=kind)
        if value is None or not (math.isfinite(value) or self.page == "written"):
            # A written page spells a figure that is no number as its rule does, the dash in its unit (``— ms``).
            return self._issue(form.key, form.kind, "level", value, _ABSENT_TEXT, _ABSENT_TEXT, "")
        if self.page == "written":
            text = _written_level(form, value)
            return self._issue(form.key, form.kind, "level", value, text, _digits_of(text, form.unit), form.unit)
        return self._code_issue(form, "level", value, _code_scaled(form, value))

    def drawn_level(self, key: str, value: float | None, *, kind: _MeasureKind = "measure") -> Printed:
        """A level of a measure given as the code page draws it (a rate already in percent: a change's baseline, the
        mean of its cases' shares), in its measure's form, wherever the page states it, in a cell or in words.

        :param key: The measure's key.
        :ptype key: str
        :param value: The level as drawn; None for none.
        :ptype value: float | None
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :return: The level.
        :rtype: Printed
        :raises ValueError: The forms are a written page's, whose levels are drawn from their quantity.
        """
        form = self.measure(key, kind=kind)
        if self.page != "code":
            raise ValueError(f"a level of {form.key!r} is given as drawn only on a code page")
        if value is None or not math.isfinite(value):
            return self._issue(form.key, form.kind, "level", value, _ABSENT_TEXT, _ABSENT_TEXT, "")
        return self._code_issue(form, "level", value, value)

    def change(self, key: str, value: float | None, *, kind: _MeasureKind = "measure") -> Printed:
        """A change of a measure (an arm less the control), signed, in its measure's form.

        :param key: The measure's key.
        :ptype key: str
        :param value: The change as given; None for none.
        :ptype value: float | None
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :return: The change.
        :rtype: Printed
        """
        form = self.measure(key, kind=kind)
        if value is None or not (math.isfinite(value) or self.page == "written"):
            # As a level: a written page spells a change that is no number in its unit (``— pts``).
            return self._issue(form.key, form.kind, "change", value, _ABSENT_TEXT, _ABSENT_TEXT, "")
        if self.page == "written":
            text = _written_change(form, value)
        else:
            text = _code_change(form, value)
        unit = form.change_unit
        if self.page == "code":
            held = _code_change_held(form, value)
            return self._issue(form.key, form.kind, "change", value, text, _digits_of(text, unit), unit, held=held)
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
        decimals = _code_change_places(form, value)
        text = unit_spaced(policy_figure(value * form.change_factor, decimals, bounds=()), unit)
        return self._issue(form.key, form.kind, "half_width", value, text, _digits_of(text, unit), unit, held=held)

    def interval(
        self,
        key: str,
        low: float,
        high: float,
        *,
        kind: _MeasureKind = "measure",
        change: bool = False,
        drawn: bool = False,
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
        :param drawn: Whether the ends are given as a code page draws them (a rate already in percent: the mean of its
            aspects' levels), not as stored.
        :ptype drawn: bool
        :return: The interval.
        :rtype: _PrintedInterval
        :raises ValueError: Ends given as drawn on a written page, which draws its figures from their quantity.
        """
        form = self.measure(key, kind=kind)
        if self.page == "written":
            if drawn:
                raise ValueError(f"an interval of {form.key!r} is given as drawn only on a code page")
            span = form.unnamed_span if kind is None else form.span
            low_text, high_text, text = _written_interval(form, low, high, change=change, span=span)
        else:
            low_text, high_text, text = _code_interval(form, low, high, change=change, drawn=drawn)
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
            digits=interval_beside(low_digits, high_digits),
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
        money = MoneyForm(decimals=places)
        text = money.spell(drawn)
        level = self._issue(form.key, form.kind, "level", value, text, _digits_of(text, MONEY), MONEY, held=drawn)
        low_text, high_text = money.spell(drawn_low), money.spell(drawn_high)
        joined = interval_beside(low_text, high_text)
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
            digits=interval_beside(low_digits, high_digits),
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
        # Each end is sized off zero on its own, as the interval's cell prints it (``0.03 points``, never ``0``).
        low_text, high_text = (
            policy_figure(abs(end), _code_places(form), bounds=_ZERO_BOUND) for end in (scaled_low, scaled_high)
        )
        text = interval_beside(low_text, high_text)
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
        """An interval as a code page's folded table of every measure states it, in its one form, ``low to high``, at
        its measure's places, as the scorecard states it (#1127 item 8).

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
        # The fold's rows read at the measure's places, as the scorecard does (#1127 item 8).
        low_text, high_text, text = _code_interval(form, low, high, change=False)
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
            digits=interval_beside(low_digits, high_digits),
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
        :return: The level as its text states it (:func:`policy_level`), as a share.
        :rtype: float
        """
        return float(policy_level(100 * level).removesuffix(PERCENT)) / 100

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
        """A p-value as APA writes it (:func:`policy_p`): ``< 0.001`` under a thousandth, three decimals below 0.1, two
        from it up (``0.50``, ``1.00``). It is no measurement, so no measure's decimals reach it.

        :param value: The p-value; None for none.
        :ptype value: float | None
        :return: The p-value.
        :rtype: Printed
        """
        text = policy_p(value)
        return self._issue("", "", "p", value, text, text, "")

    def effect(self, value: float | None) -> Printed:
        """An effect size or a correlation at two decimals, its sign kept (:func:`policy_effect`): ``0.08``, ``0.12``.

        :param value: The effect; None for none.
        :ptype value: float | None
        :return: The effect.
        :rtype: Printed
        """
        text = policy_effect(value)
        return self._issue("", "", "effect", value, text, text, "")

    def confidence(self, level: str, family: ContrastFamily | None = None) -> Printed:
        """An interval's level, as every heading prints it (:func:`interval_level`): the joint guarantee where its
        family corrects its comparisons together (``95% (adjusted for 16 comparisons)``), any other level as stated.

        :param level: The level as the input states it (``99.69%``).
        :ptype level: str
        :param family: The contrast family the interval belongs to; None for none.
        :ptype family: ContrastFamily | None
        :return: The level's words.
        :rtype: Printed
        """
        text = interval_level(level, family)
        return self._issue("", "", "confidence", None, text, text, "")

    def drawn_confidence(self, level: float) -> Printed:
        """An interval's level as a written page's chart states it in its head (``95%``, ``99.7% computed``): a share
        (``0.95``) or a percent (``95``) alike, in percent by the lone figure rule. It belongs to no measure, as
        :meth:`confidence`'s does not.

        :param level: The level, as a share (at most 1) or in percent.
        :ptype level: float
        :return: The level.
        :rtype: Printed
        """
        text = policy_level(level * 100 if level <= 1 else level)
        return self._issue("", "", "confidence", level, text, text.removesuffix(PERCENT), PERCENT)

    def confidence_share(self, level: float) -> Printed:
        """An interval's level held as a share (``0.95``, ``0.99688``), as a code page states it: in percent, its distance
        from 100% to two significant figures (:func:`policy_level`: ``99.69%``). Its ``held`` is the level to that
        precision, as a share,
        which is how the page tells whether two levels read alike or one is the nominal level. It belongs to no
        measure: a level is how a family's intervals were read, the same for every measure the family holds, as
        :meth:`confidence`'s is.

        :param level: The level, as a share.
        :ptype level: float
        :return: The level.
        :rtype: Printed
        """
        text = policy_level(100 * level)
        return self._issue(
            "", "", "confidence", level, text, text.removesuffix(PERCENT), PERCENT, held=self.told_apart(level)
        )

    def declared(self, value: float | None, unit: str = "", *, signed: bool = False) -> Printed:
        """A value the input declares (a margin, a floor, a setting) exactly as declared, in its unit
        (:func:`policy_declared`): ``-15 pts``, never ``-15.0 pts``. It was chosen, not measured, so no measure's
        decimals reach it.

        :param value: The value; None for none.
        :ptype value: float | None
        :param unit: Its unit; empty for none.
        :ptype unit: str
        :param signed: Whether a positive value carries its ``+``, as one read beside a signed change does (``+15
            pts``).
        :ptype signed: bool
        :return: The value.
        :rtype: Printed
        """
        digits = policy_declared(value)
        if signed and value is not None and math.isfinite(value):
            digits = plus_signed(digits, value)
        text = unit_spaced(digits, unit) if value is not None and math.isfinite(value) else digits
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
        """A pass^k of a measure (the chance a case passes every one of k runs, 0 to 1), in percent: on a code page that
        holds the measure's form, at its places, kept off its floor and ceiling, as a rate of the measure; elsewhere,
        and on a page that states no other figure of the measure (a measurement's depth table), to a decimal by the
        lone figure rule. It is recorded under its measure.

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
        if self.page == "code" and (key, measure) in self._measures:
            # A code page reads a pass^k at its measure's places, as a rate of the measure, in its depth tables too.
            form = self._measures[key, measure]
            places = policy_bounded_places(100 * value, _code_places(form), bounds=_RATE_BOUNDS)
            held = float(round_half_even(100 * value, places))
            text = policy_rate(100 * value, _code_places(form))
            return self._issue(key, measure, "pass_k", value, text, _digits_of(text, PERCENT), PERCENT, held=held)
        held = round(100 * value, POINT_PLACES)
        text = unit_spaced(format_number(held), PERCENT)
        return self._issue(key, measure, "pass_k", value, text, _digits_of(text, PERCENT), PERCENT, held=held)

    def pass_k_change(self, key: str, value: float | None, *, kind: _MeasureKind = "measure") -> Printed:
        """A change of a pass^k (one run's less another's, as a share), signed, in points, as :meth:`pass_k` states its
        levels: on a code page that holds the measure's form, as that page states the measure's changes
        (:meth:`change`); elsewhere to a decimal by the lone figure rule. It is recorded under its measure.

        :param key: The measure's key.
        :ptype key: str
        :param value: The change, as a share; None for none.
        :ptype value: float | None
        :param kind: The measure's kind.
        :ptype kind: _MeasureKind
        :return: The change.
        :rtype: Printed
        """
        measure = _measure_kind(kind)
        if value is None or not math.isfinite(value):
            return self._issue(key, measure, "change", value, _ABSENT_TEXT, _ABSENT_TEXT, "")
        if self.page == "code" and (key, measure) in self._measures:
            return self.change(key, value, kind=kind)
        text = unit_spaced(format_signed(round(100 * value, POINT_PLACES)), POINTS)
        return self._issue(key, measure, "change", value, text, _digits_of(text, POINTS), POINTS)

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

    # --- sentences read before any page's forms exist: floors', campaign rows', frontier cells' --------------------

    def said(self, said: Said) -> str:
        """A sentence read before this page existed (a floor's statement and its reason, a rate's, a campaign row's
        reason read again from its evidence, a frontier pass^k cell's line), as this page says it, each figure it holds
        in its form: a margin as declared (:meth:`declared`), a level as :meth:`confidence_share` states one, a count as
        :meth:`count` states one (``1,200``), a pass^k as :meth:`pass_k` states one, and how far a rate's drop reaches,
        in points, unsigned, at the places this page reads that rate's changes at, kept off zero: its own form of the
        rate (a code page's pass rate), else the rate's set on an aspect page. A rate the page states no change of reads
        to a decimal, a whole number whole, as the campaign's readings freeze it
        (:meth:`~threetears.evals.analysis.report.said.Said.stored`). A figure the words state as a bound never reads
        past itself: a lower bound (``at least``) rounds down at those places, an upper one (``at most``) up, and where
        that would cross the margin the sentence compares it with, or zero, it takes a decimal more until it does not
        (``within 2.5 points (it is at most 2.4)``, never ``at most 3``). Each point figure is recorded under its rate's
        measure as a change.

        :param said: The sentence, as read.
        :ptype said: Said
        :return: The sentence.
        :rtype: str
        """
        return said.words(self._said_figure)

    def _said_figure(self, figure: SaidFigure) -> str:
        """One figure of a sentence read before the page (:meth:`said`); a change reaches here as its size, in points."""
        if figure.role == "margin":
            return self.declared(figure.value).text
        if figure.role == "level":
            return self.confidence_share(figure.value).text
        if figure.role == "count":
            return self.count(int(figure.value)).text
        if figure.role == "pass_k":
            if figure.measure is None:
                return self.pass_k(TRIAL_PASS_NAME, figure.value, kind="trial").text
            return self.pass_k(figure.measure, figure.value).text
        key, kinds = (
            (TRIAL_PASS_NAME, ("trial",)) if figure.measure is None else (figure.measure, ("measure", "judged"))
        )
        found = next(((kind, places) for kind in kinds if (places := self._said_places(key, kind)) is not None), None)
        if found is None:
            held, places = _said_held(figure, POINT_PLACES)
            # A figure at the rule's own decimal reads as the campaign's readings froze it, a whole number whole.
            kind, text = kinds[0], format_number(held) if places == POINT_PLACES else spell_at(held, places)
        else:
            held, places = _said_held(figure, policy_bounded_places(figure.value, found[1], bounds=_ZERO_BOUND))
            kind, text = found[0], spell_at(held, places)
        self._issued.append(_Issued(key, kind, "change", figure.value, text))
        return text

    def _said_places(self, key: str, kind: str) -> int | None:
        """The places a rate's changes read at on this page, read in points: its own form of the rate where it reads
        as one, else its set on an aspect page; None where the page states no change of it."""
        form = self._measures.get((key, kind))
        if form is not None and form.reads_as == "rate":
            if self.page == "code":
                return _code_places(form)
            if form.places is not None:
                return form.places
        held = self._contrast_places.get((key, kind))
        return held.change if held is not None else None

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

    def money_ledger(self, amount: float, *, measure: str = "", kind: _MeasureKind = None) -> Printed:
        """An amount of money the out-of-run spend ledger states, or a refusal prices one call at, to four decimals, its
        ``$`` before the number (``$0.0012``), as each prints one today: the precision a single call's price needs,
        which cents would read as nothing. Of none today: a call's cost and its priced ceiling, a ledger's totals, a
        judge call's priced floor.

        :param amount: Dollars, as stored.
        :ptype amount: float
        :param measure: The measure's key where the figure is one of its figures; empty for a figure of no measure.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``); unread without a measure.
        :ptype kind: _MeasureKind
        :return: The amount.
        :rtype: Printed
        """
        return self._money(amount, f"${amount:.4f}", measure, kind)

    def whole_seconds(self, seconds: float, *, measure: str = "", kind: _MeasureKind = None) -> Printed:
        """A duration an off-page line states in whole seconds, rounded half to even, its ``s`` after the number
        (``95s``), as it prints one today: a launch's refusal weighs a cell's deadline against a job's budget, where a
        fraction of a second decides nothing. Of none today: a writer cell's deadline, the run it could take, its job's
        budget.

        :param seconds: The duration, in seconds.
        :ptype seconds: float
        :param measure: The measure's key where the figure is one of its figures; empty for a figure of no measure.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``); unread without a measure.
        :ptype kind: _MeasureKind
        :return: The duration.
        :rtype: Printed
        """
        digits = f"{seconds:.0f}"
        return self._issue(*_keyed(measure, kind), "lone", seconds, digits + "s", digits, "s")

    def score_readings(
        self, reads: Mapping[str, tuple[float, int]], *, measure: str, kind: _MeasureKind
    ) -> dict[str, Printed]:
        """A judged score's readings across an off-page table's row, one per arm, each its mean at one precision for the
        row: a score states no interval there, so its decimals are rule 2's for a score's spread at that many cases, a
        hundredth from :data:`_SCORE_HUNDREDTHS_FROM` cases up and a tenth below, where a hundredth is noise (rung
        5's ``4.9`` at 7 cases, never ``4.86``).

        :param reads: Each arm's mean and the cases it is over, by arm.
        :ptype reads: Mapping[str, tuple[float, int]]
        :param measure: The judged dimension's key.
        :ptype measure: str
        :param kind: Its kind.
        :ptype kind: _MeasureKind
        :return: Each arm's reading, by arm.
        :rtype: dict[str, Printed]
        """
        fewest = min((n for _, n in reads.values()), default=0)
        places = SCORE_PLACES if fewest >= _SCORE_HUNDREDTHS_FROM else SCORE_PLACES - 1
        said = {}
        for arm, (mean, _) in reads.items():
            text = policy_figure(mean, places, bounds=())
            said[arm] = self._issue(*_keyed(measure, kind), "reading", mean, text, text, "")
        return said

    def rate_readings(
        self, reads: Mapping[str, tuple[float, int]], *, measure: str, kind: _MeasureKind
    ) -> dict[str, Printed]:
        """A rate's readings across an off-page table's row, one per arm, each a percent and the count it is of
        (``14% (1 of 7)``), never a fraction at two decimals (``0.14``), which reads as no rate. The row is one
        measure, so every arm's reads at one precision: the table reads no interval, so two significant figures of
        its least figure but zero, at most one decimal, each kept off 0% and 100% (:func:`policy_places`,
        :func:`policy_rate`). Of a measure: a boolean measure's rate, or a pass/fail judged dimension's, in the
        rung-5 readings table.

        :param reads: Each arm's rate, 0 to 1, and the trials it was read on, by arm.
        :ptype reads: Mapping[str, tuple[float, int]]
        :param measure: The measure's key.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``).
        :ptype kind: _MeasureKind
        :return: Each arm's reading, by arm.
        :rtype: dict[str, Printed]
        """
        places = policy_places((), [rate * 100 for rate, _ in reads.values()], capped=True)
        readings = {}
        for arm, (rate, n) in reads.items():
            percent = policy_rate(rate * 100, places)
            text = f"{percent} ({spell_count(round(rate * n))} of {spell_count(n)})"
            digits = percent.removesuffix(PERCENT)
            readings[arm] = self._issue(*_keyed(measure, kind), "reading", rate, text, digits, PERCENT)
        return readings

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
        """An amount of money on a ruler a surface laid over its set, in the ruler's one form. Of a measure: an arm's
        cost in the options table (the measure the page reads its cost from), a gate's figure and threshold on a money
        criterion (the measure the criterion reads). Of none: an arm's total spend, a role's.

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

    def check_cell(self, value: float | None, *, measure: str = "", kind: _MeasureKind = None) -> TableCell:
        """A judge's-check figure as the rung-5 checks table's cell holds it (:meth:`check`), its text by the one
        number rule.

        :param value: The figure as read; None for none.
        :ptype value: float | None
        :param measure: The measure's key where the figure is one of its figures; empty for a figure of no measure.
        :ptype measure: str
        :param kind: The measure's kind (None reads as ``measure``); unread without a measure.
        :ptype kind: _MeasureKind
        :return: The cell.
        :rtype: TableCell
        """
        decided = self.check(value, measure=measure, kind=kind)
        return TableCell(value=None) if decided is None else TableCell(value=decided, value_text=format_number(decided))

    def preference(self, share: float | None, difference: tuple[float, float] | None) -> PreferenceShares | None:
        """A candidate's share of a judge's preference and the interval the pass read on the difference of the two
        shares, on the share scale: the two shares run 0 to 1 and sum to 1, so a difference ``d`` is the candidate's
        share ``(1 + d) / 2``. Each end is held within 0 to 100%, the scale a share lies on. The shares and the ends
        read at one precision, read from the interval's uncertainty as a rate's is (:func:`policy_places`, at most one
        decimal: ``40 to 60%``), with just enough more that none prints on none, all or even, the share where neither
        arm is preferred (an end of ``49.7%``, never ``50%``; :func:`policy_bounded_places`), and that two ends do not
        print alike (``67.01 to 67.03``, never ``67.0 to 67.0``).

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
        widths = [] if ends is None else [(ends[1] - ends[0]) / 2]
        held = policy_places(widths, [points], capped=True)
        places = max(policy_bounded_places(value, held, bounds=_PREFERENCE_BOUNDS) for value in [points, *(ends or [])])
        if ends is not None:
            # Two ends that would print alike take the decimals that tell them apart (number rule 3), and the shares
            # read at them too, so they still sum to 100 at one precision.
            places = ends_places(ends[0], ends[1], places)
        candidate = round(points, places)
        shares = PreferenceShares(
            candidate=candidate,
            control=round(100 - candidate, places),
            interval=None if ends is None else (round(ends[0], places), round(ends[1], places)),
            places=places,
        )
        self._issue("", "", "preference", share, shares.share_text(candidate), shares.share_text(candidate), PERCENT)
        return shares

    def glyph_title(self, value: float, low: float, high: float, *, unit: str, margin: float | None = None) -> Printed:
        """What an interval glyph drawn with no table's text beside it states in its title: the figure in its unit,
        then its interval in its one form, its unit on the figure alone, both at two significant figures of the
        interval's width (:func:`policy_lone_interval`: ``88 to 100``, ``-4.2 to +2.0``), money in its own form
        (``-$0.01 to $0.06``), then a declared margin, ``±5 pts``. A table's glyph states its cells' text instead;
        this is the glyph's own, of no measure.

        :param value: The figure, in ``unit``.
        :ptype value: float
        :param low: The interval's low end, in ``unit``.
        :ptype low: float
        :param high: The interval's high end, in ``unit``.
        :ptype high: float
        :param unit: The unit the figures are in; empty for none.
        :ptype unit: str
        :param margin: A declared margin either side of zero, or None.
        :ptype margin: float | None
        :return: The title.
        :rtype: Printed
        """
        # The unit is on the figure alone, the interval beside it bare (number rule 3), but money's ``$`` on each end.
        if is_money(unit):
            interval = interval_beside(with_unit(low, unit), with_unit(high, unit))
            figure = with_unit(value, unit)
        else:
            # A change in points: the figure and its interval's ends carry their signs.
            digits, interval = policy_lone_interval(
                low, high, rate=unit in (PERCENT, POINTS), change=unit == POINTS, value=value
            )
            figure = unit_spaced(digits, unit)
        said = f"{figure}, interval {interval}"
        if margin is not None:
            said += f", margin ±{with_unit(margin, unit)}"
        return self._issue("", "", "lone", value, said, said, "")

    # --- a table's glyph: its cells' decided texts ----------------------------------------------------------------
    #
    # A table's interval glyphs are drawn on one scale per column (or per unit, on a matrix), whose ends the line under
    # the table states and whose figures each glyph's title states. A page's writer lays them out from a stored table,
    # which holds no page's forms, so each cell carries the texts its builder's forms decided for its figure and its
    # interval's ends, and these join them.

    def glyph_said(self, cell: TableCell, unit: str) -> Printed:
        """What a table glyph's title states: its cell's figure, its interval and its declared margin
        (``-5.0 pts, interval -6.9 to +11.0, margin ±15 pts``), each as the cell's builder decided it: the figure with
        its unit (the head's, under a head that names it), a difference's signed (``+0.2``); the interval in its one
        form, ``low to high``, its unit on the figure alone (number rule 3), money's ``$`` on each end; the margin as
        declared, as the chart's margin line reads it.

        :param cell: The glyph's cell, which carries an interval.
        :ptype cell: TableCell
        :param unit: The unit the cell's figures are read in: its own, else the one its column's head names.
        :ptype unit: str
        :return: The title.
        :rtype: Printed
        :raises ValueError: The cell carries no interval.
        """
        if cell.low is None or cell.low_text is None or cell.high_text is None:
            raise ValueError(f"a glyph is drawn for a cell's interval, and the cell holding {cell.value!r} has none")
        # A change's cell states the level it lands at before it (``78% · -8 pts``); its glyph draws the change alone.
        own = cell.figure.removeprefix(f"{cell.level_text}{LANDS_AT}") if cell.level_text is not None else cell.figure
        figure = own if cell.unit or not unit else unit_spaced(own, unit)
        value = float(cell.value)  # type: ignore[arg-type]
        # A difference's figure carries its sign, as its cell does (``+0.2``); its unit is on the figure alone, the
        # interval beside it written bare (number rule 3: ``+10 pts, interval -45 to +60``).
        signed = plus_signed(figure, value) if cell.zero_line else figure
        said = f"{signed}, interval {interval_beside(cell.low_text, cell.high_text)}"
        if cell.margin is not None:
            # A margin is declared, not measured: it reads as declared (``±15 pts``), never at the cells' decimals.
            said += f", margin ±{with_unit(cell.margin, unit)}"
        return self._issue("", "", "lone", value, said, said, "")

    def lone_rate(self, level: float, interval: tuple[float, float] | None) -> tuple[Printed, Printed | None]:
        """A rate and its interval, in percent, that no page's measure holds (a quick print's line): both at the
        decimals the interval's own half-width gives a rate (rules 1 and 2, at most one), the level kept off 0% and
        100% (``99.7%``, never ``100%``) unless it sits there, and the interval in its one form, ``low to high`` with
        its ``%`` once (rule 3), each end kept off its bounds. With no interval the level reads by the number rule.

        :param level: The rate, in percent.
        :ptype level: float
        :param interval: Its interval's ends, in percent; None for none.
        :ptype interval: tuple[float, float] | None
        :return: The level, and the interval or None.
        :rtype: tuple[Printed, Printed | None]
        """
        if interval is None:
            return self.in_unit(level, PERCENT, role="level"), None
        low, high = interval
        places = policy_places([(high - low) / 2], [low, high], capped=True)
        said = policy_rate(level, places)
        _, _, text = policy_interval(low, high, places, change=False, bounds=_RATE_BOUNDS, unit=PERCENT)
        self._issued.append(_Issued("", "", "interval", (low, high), text))
        return self._issue("", "", "level", level, said, said.removesuffix(PERCENT), PERCENT), Printed(
            value=None, text=text, digits=text, unit=""
        )

    def scale_span(self, cells: Sequence[TableCell], unit: str, low: float, high: float) -> Printed:
        """A glyph scale's ends as the line under its table states them, its unit once (``-7.8 to +35.4 pts``, ``84 to
        100%``, ``$0.031 to $0.10``): each end as the cells drawn on it print that figure, an interval's end as its
        cell's builder decided it; an end no cell prints (zero, a declared margin's, a rate's floor or ceiling, or the
        track either side of one figure) by the number rule, points at a tenth, signed on a scale of differences; a
        declared margin's ends as declared, zero whole.

        :param cells: The cells drawn on the scale, each carrying an interval.
        :ptype cells: Sequence[TableCell]
        :param unit: The unit the scale's figures are read in.
        :ptype unit: str
        :param low: The scale's low end.
        :ptype low: float
        :param high: The scale's high end.
        :ptype high: float
        :return: The span.
        :rtype: Printed
        """
        said: dict[float, str] = {}
        for cell in cells:
            if cell.low is not None and cell.high is not None and cell.low_text and cell.high_text:
                said.setdefault(cell.low, cell.low_text)
                said.setdefault(cell.high, cell.high_text)
        change = any(cell.zero_line or cell.margin is not None for cell in cells)
        # A declared margin's ends read as declared, and zero whole, as no measure's precision qualifies either.
        for cell in cells:
            if cell.margin is not None:
                said.setdefault(-cell.margin, format_signed(-cell.margin))
                said.setdefault(cell.margin, plus_signed(format_number(cell.margin), cell.margin))
        said.setdefault(0.0, with_unit(0.0, MONEY) if is_money(unit) else format_number(0.0))

        def end(number: float) -> str:
            if number in said:
                return said[number]
            if unit == POINTS:
                # A change in points reads at a point's tenth, as a change no set decides does.
                spelled = spell_at(number, ends_places(low, high, POINT_PLACES))
            else:
                spelled = number_in(number, unit) if not is_money(unit) else with_unit(number, MONEY)
            return plus_signed(spelled, number) if change else spelled

        text = interval_alone(end(low), end(high), unit)
        self._issued.append(_Issued("", "", "interval", (low, high), text))
        return Printed(value=None, text=text, digits=text, unit="")

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
        :return: The quantity, its format naming its measure's decimals (``pct:0``), so a value the kit writes with no
            text beside it reads as the page prints it.
        :rtype: Quantity
        """
        drawn = quantity(unit, values, change=change)
        return quantity(unit, values, change=change, places=self._named_places(drawn, values, measure, kind))

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
        :return: The quantity, its format naming its measure's decimals, as :meth:`chart_quantity`'s does.
        :rtype: Quantity
        """
        figures = tuple(figures)
        drawn = quantity_of(unit, figures, change=change)
        named = self._named_places(drawn, [figure * drawn.factor for figure in figures], measure, kind)
        return quantity_of(unit, figures, change=change, places=named)

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
        places = self._chart_places(drawn, measure, kind)
        text = figure_text(value, drawn) if places is None else _placed_text(value, drawn, places, change=drawn.change)
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
        places = self._chart_places(drawn, measure, kind)
        if places is not None:
            text = _placed_text(value, drawn, places, change=False)
        elif drawn.unit == MONEY and form is not None and form.figures is not None and value != 0:
            places = max(form.places(value), form.decimals)
            text = MoneyForm(decimals=places).spell(value)
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
        places = self._chart_places(drawn, measure, kind)
        if places is None:
            low_text, high_text = figure_text(low, drawn), figure_text(high, drawn)
            text = _interval_words(low, high, drawn)
        else:
            bounds = _RATE_BOUNDS if drawn.rate and not drawn.change else ()
            low_digits, high_digits, text = policy_interval(
                low, high, places, change=drawn.change, bounds=bounds, unit=drawn.unit
            )
            low_text, high_text = low_digits, unit_spaced(high_digits, drawn.unit)
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
            digits=interval_beside(low_digits, high_digits),
        )

    def _chart_places(self, drawn: Quantity, measure: str, kind: _MeasureKind) -> int | None:
        """The decimals a chart's figure of a measure reads at: its measure's (:attr:`MeasureForm.places`), as every
        other surface of the page prints it; None for money, which keeps its ruler's form, and for a figure of no
        measure the page holds (an off-page chart's, a price per point), which its quantity's rule spells."""
        if not measure or drawn.unit == MONEY:
            return None
        form = self._measures.get((measure, _measure_kind(kind)))
        return None if form is None else form.places

    def _named_places(self, drawn: Quantity, values: Iterable[float], measure: str, kind: _MeasureKind) -> int | None:
        """The decimals a chart's format names: its measure's (:meth:`_chart_places`); for a figure of no measure the
        page holds (a preference, an empty chart's axis), the unit's own where its format names them (a score's two),
        else its drawn values' as a set with no interval reads them, two significant figures of the least
        (:func:`set_places`), none for no value; None for money, whose digits are its one form's."""
        if drawn.unit == MONEY:
            return None
        places = self._chart_places(drawn, measure, kind)
        if places is not None:
            return places
        return None if ":" in drawn.fmt else set_places(values)

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

    def count_cell(self, count: CellValue, *, flag: str | None = None, ref: str | None = None) -> TableCell:
        """A count as a table's cell holds it, the count kept as given (a number a program sorts by) and its text
        whole and grouped, as every count reads (number rule 5: ``1,234``, :meth:`count`); a count that is no whole
        number (a median of repeats, ``2.5``) by the number rule; words (a range of counts, ``18 to 20``) and an
        absent count kept as they are.

        :param count: The count, words or None.
        :ptype count: CellValue
        :param flag: The marker the cell prints beside it; None for none.
        :ptype flag: str | None
        :param ref: The id on the page the cell links to; None for none.
        :ptype ref: str | None
        :return: The cell.
        :rtype: TableCell
        """
        if isinstance(count, int | float) and not isinstance(count, bool):
            whole = math.isfinite(count) and float(count).is_integer()
            text = self.count(int(count)).text if whole else self.count_alone(count).text
            return TableCell(value=count, value_text=text, flag=flag, ref=ref)
        return TableCell(value=count, flag=flag, ref=ref)

    def unit_cell(
        self,
        value: float | None,
        unit: str,
        interval: tuple[float, float] | None = None,
        *,
        measure: str = "",
        kind: _MeasureKind = None,
    ) -> TableCell:
        """A figure no set decides as a table's cell holds it, in its unit at the unit's own precision (a percent's or
        a point's one decimal) and by the number rule in any other, its interval's ends beside it as a glyph and an
        interval column print them (:meth:`interval_alone`): a confusion table's share, a floor's worst drop, a
        figure of a measure the page holds no form of.

        :param value: The figure, already in ``unit``; None for none.
        :ptype value: float | None
        :param unit: Its unit; empty for none.
        :ptype unit: str
        :param interval: Its interval's ends, in ``unit``; None for none.
        :ptype interval: tuple[float, float] | None
        :param measure: The measure it is a figure of; empty for none.
        :ptype measure: str
        :param kind: Its reading.
        :ptype kind: _MeasureKind
        :return: The cell.
        :rtype: TableCell
        """
        if value is None:
            return TableCell(value=None)
        text = self.in_unit(value, unit, role="lone", measure=measure, kind=kind).text
        if interval is None:
            return TableCell(value=value, unit=unit, value_text=text)
        low_text, high_text = _glyph_ends(*interval, unit=unit, places=None, change=False)
        return TableCell(
            value=value,
            unit=unit,
            value_text=text,
            low=interval[0],
            high=interval[1],
            low_text=low_text,
            high_text=high_text,
        )

    def money_cell(self, form: MoneyForm, value: float, ends: tuple[float, float] | None = None) -> TableCell:
        """An amount as a table's cell holds it, in a form its builder chose over its column (a price per point, in
        the form every price of its column and their open ends read in), at the places its value and its interval's
        ends need in it, each end with its ``$``.

        :param form: The column's money form.
        :ptype form: MoneyForm
        :param value: The amount, in dollars on the column's ruler.
        :ptype value: float
        :param ends: Its interval's ends, on the same ruler; None for none.
        :ptype ends: tuple[float, float] | None
        :return: The cell.
        :rtype: TableCell
        """
        places = form.places_of([value, *(ends or ())], ends=ends)
        money = MoneyForm(decimals=places)
        text = money.spell(value)
        self._issued.append(_Issued("", "", "money", value, text))
        if ends is None:
            return TableCell(value=value, unit=MONEY, value_text=text)
        return TableCell(
            value=value,
            unit=MONEY,
            value_text=text,
            low=ends[0],
            high=ends[1],
            low_text=money.spell(ends[0]),
            high_text=money.spell(ends[1]),
        )

    def money_in(self, form: MoneyForm, amount: float) -> Printed:
        """An amount in a form its surface chose over its set (an interval's one end, open above), at its own figures.

        :param form: The set's money form.
        :ptype form: MoneyForm
        :param amount: The amount, in dollars on the set's ruler.
        :ptype amount: float
        :return: The amount.
        :rtype: Printed
        """
        return self._money(amount, form.spell(amount), "", None)

    def money_ruler_of(self, amounts: Iterable[float | None], *, per: int, start: int = 0) -> MoneyRuler:
        """The ruler a table's column of money is stated on (a price per point, every price and open end of the column
        together), by the one owner of money's scale (:func:`~threetears.evals.analysis.report.cost_unit.money_ruler`):
        its one form, the scale its amounts are drawn at, and the count its head names. Its amounts print through
        :meth:`money_cell` and :meth:`money_in` in the ruler's form.

        :param amounts: Every amount the column holds, as stored; None for an absent one.
        :ptype amounts: Iterable[float | None]
        :param per: How many of what it counts one stored amount is per: 1,000 for a price per 1,000; 0 for a total.
        :ptype per: int
        :param start: The least count to state it per.
        :ptype start: int
        :return: The ruler.
        :rtype: MoneyRuler
        """
        return money_ruler(amounts, per=per, start=start)

    @staticmethod
    def unit_on(digits: str, unit: str) -> str:
        """A bare figure's decided digits with its unit, where no head names it (a rate's ``95`` in a column of ``(%)``
        read in a row of its own, ``95%``); money's ``$`` is on its digits already.

        :param digits: The figure's decided digits.
        :ptype digits: str
        :param unit: Its unit.
        :ptype unit: str
        :return: The figure with its unit.
        :rtype: str
        """
        return unit_after(digits, unit)

    @staticmethod
    def lands_at(level: Printed, change: Printed) -> str:
        """A change after the level it lands at, as a matrix's cell states it (``78% · -8 pts``), so a change is never
        read without where it lands; a glyph drawn for the cell states the change alone (:meth:`glyph_said`).

        :param level: The level, decided.
        :ptype level: Printed
        :param change: The change, decided.
        :ptype change: Printed
        :return: The cell's text.
        :rtype: str
        """
        return f"{level.text}{LANDS_AT}{change.text}"

    def interval_alone(self, cell: TableCell) -> Printed:
        """A cell's interval standing alone, in its own column: its ends as the cell's builder decided them, the unit
        written once, after the high end (``80 to 89%``, ``-6.9 to +11.0 pts``), money's ``$`` on each end (``$16 to
        $25``); a dash where the cell carries none.

        :param cell: The cell.
        :ptype cell: TableCell
        :return: The interval's text.
        :rtype: Printed
        """
        if cell.low_text is None or cell.high_text is None:
            return Printed(value=None, text=_ABSENT_TEXT, digits=_ABSENT_TEXT, unit="")
        text = interval_alone(cell.low_text, cell.high_text, cell.unit)
        return Printed(value=None, text=text, digits=text, unit="")

    def in_unit(self, value: float, unit: str, *, role: Role, measure: str = "", kind: _MeasureKind = None) -> Printed:
        """A figure in its unit, unsigned, at the unit's own precision (a percent's or a point's one decimal) and by
        the number rule in any other, as an aspect page states one today where no set decides it: a declared margin or
        line in the unit a change is stated in (``declared``), an interval's level in percent (``confidence``), a
        judged reading's rectifier (``change``), a quick print's figure beside its unit.

        :param value: The figure, already in ``unit``.
        :ptype value: float
        :param unit: Its unit; empty for none.
        :ptype unit: str
        :param role: What the figure is: a measure's, or the fixed form it takes.
        :ptype role: Role
        :param measure: The measure or dimension it is a figure of; empty for none.
        :ptype measure: str
        :param kind: Its reading (``measure``, ``judged``, ``trial``).
        :ptype kind: _MeasureKind
        :return: The figure.
        :rtype: Printed
        """
        text = with_unit(value, unit)
        written = MONEY if is_money(unit) else unit
        return self._issue(*_keyed(measure, kind), role, value, text, _digits_of(text, written), written)

    @staticmethod
    def projected_cost(value: float) -> float:
        """A projected cost to two significant figures, since the cases it prices are about (``2.7``, ``120``), nought
        for none: the one rounding every projected cost a page states is read at, which a table's set of projected
        costs chooses its money form over. A decision that prints nothing, so nothing is issued.

        :param value: The cost, in US dollars.
        :ptype value: float
        :return: The cost to two figures.
        :rtype: float
        """
        if value <= 0:
            return 0.0
        return round(value, 1 - math.floor(math.log10(value)))

    def projected_money(self, value: float, form: MoneyForm | None = None) -> Printed:
        """A projected cost as a page prints one, to two figures (:meth:`projected_cost`): by the lone figure rule with
        its ``$`` (``$2.7``, ``$120``), or, handed the form every projected cost the table states reads in (#993), in
        it (``$2.70`` beside ``$27.00``). A projected cost is of no measure: it prices cases not yet run.

        :param value: The cost, in US dollars.
        :ptype value: float
        :param form: The table's form for its projected costs; None for none.
        :ptype form: MoneyForm | None
        :return: The cost.
        :rtype: Printed
        """
        held = self.projected_cost(value)
        text = format_number(held, prefix="$") if form is None else form.spell(held)
        return self._issue("", "", "money", value, text, _digits_of(text, MONEY), MONEY)

    def chart_texts(
        self, data: KitChart, *, read_at: str = "", measure: str = "", kind: _MeasureKind = None
    ) -> KitChart:
        """A chart's data with every value it draws carried as decided text (``value_text``, ``lo_text``,
        ``hi_text`` and their kin), which the kit and the static page print in place of formatting a value, and its
        values table prints. Today's decision, relocated, not rewritten: each value as the chart's values table has
        always spelled it, in the chart's format at ``read_at``'s precision where the format names no unit (the
        chart's own ``spelled``, the contract's rule, which a host's chart without texts is spelled by too).

        A chart that draws one measure's figures (``measure``) records each value's text under it: a value or a range's
        end as the measure's level, or as its change where the chart's format signs its values (``+pts``), and an
        interval's end as its interval. A chart's rows name no measure, so a chart of several measures, or of none,
        records nothing, and neither does a figure that is no value of its measure (a count, a cost beside a score).

        :param data: The chart's data, as its resolver drew it.
        :ptype data: KitChart
        :param read_at: The unit its values are read at where its format names none, as its block reads them.
        :ptype read_at: str
        :param measure: The one measure or dimension whose figures the chart draws; empty for several, or none.
        :ptype measure: str
        :param kind: Its reading (``measure``, ``judged``, ``trial``).
        :ptype kind: _MeasureKind
        :return: The data, every value it draws carrying its text.
        :rtype: KitChart
        """
        spelled = data.spelled(read_at)
        if measure:
            signed = str(getattr(spelled, "fmt", "")).startswith("+")
            for field_name, value, text in _drawn_texts(spelled.model_dump(mode="json", by_alias=True)):
                role: Role = "interval" if field_name in _INTERVAL_ENDS else "change" if signed else "level"
                self._issued.append(_Issued(*_keyed(measure, kind), role, value, text))
        return spelled

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
    # that decision, relocated, and spell every other number a key prints by the filter that chose its spelling. Each
    # prints a part of a key's text, which ``format_figure`` finishes with the key's sign, filters and unit; each
    # records the part it prints, under the measure whose set it reads, so the ledger holds every figure a key prints
    # and a page-level test can hold a key's text to its table's. The flip decides a key's whole text here, by its
    # measure and role.

    def key_places(self, key: str, *, kind: _MeasureKind) -> int | None:
        """The decimals a figure key's figure of a measure reads at: its measure's, its levels, changes and interval
        ends alike (:attr:`MeasureForm.places`), as every table of the page prints it; None where the page holds no
        such measure, or its money keeps its own form, and the number rule spells its figure. A decision that prints
        nothing, so nothing is issued.

        :param key: The measure's or judged dimension's key.
        :ptype key: str
        :param kind: The measure's kind: ``judged`` for a judged dimension.
        :ptype kind: _MeasureKind
        :return: The decimals, or None.
        :rtype: int | None
        :raises ValueError: The forms hold no report input (a code page's, an off-page surface's), which a key's set
            is read from.
        """
        if self._input is None:
            raise ValueError(
                f"a figure key's set of {key!r} is read from a written page's report input, and none is held"
            )
        form = self._measures.get((key, _measure_kind(kind)))
        return None if form is None else form.places

    def key_number(self, figure: Figure, value: float) -> str:
        """A plain figure's number as a figure key prints it, unsigned and without its unit: at its measure's decimals
        (:meth:`key_places`), an exact zero whole, where the figure is one the page's tables print of that measure (a
        level, a change, an interval or one of its ends, a criterion's figure or its gap), else by the lone figure rule,
        as an adjusted p or a count of repeats keeps it.

        A count's thousands are grouped (``4,000,000``), as a sentence states a count. A number figure's digits are
        recorded as printed; an interval's ends are recorded by :meth:`key_plain_ends`, which reads them both.

        :param figure: What the key resolved to: its measure's key and kind and its statistic say which set it is of.
        :ptype figure: Figure
        :param value: The number, as the key spells it (an interval's end, or the figure's own value).
        :ptype value: float
        :return: Its digits.
        :rtype: str
        """
        places = self._key_figure_places(figure)
        text = format_number(value) if places is None else policy_figure(value, places, bounds=())
        text = group_thousands(text) if figure.scale == "count" else text
        if figure.kind != "interval":
            # Recorded where the tables hold it (_key_role), whatever set its digits read: a criterion's figure of a
            # measure with no set of levels is still that measure's level.
            self._issue(*_key_set(figure), _key_role(figure), value, text, text, "")
        return text

    def key_plain_ends(
        self, figure: Figure, values: tuple[float, float], numbers: tuple[str, str], *, absolute: bool
    ) -> tuple[str, str] | None:
        """An interval's two plain ends at one precision: its measure's decimals in the one interval form, each end kept
        off zero (:func:`policy_interval`), where the page's tables print the figure's set (:meth:`key_number`); else
        two significant figures of the interval's width, which is what its
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
        :return: The two ends' texts, the high one with the figure's unit where the measure's decimals read them, else
            each with it; None where they stay as spelled.
        :rtype: tuple[str, str] | None
        """
        unit = f" {figure.unit}" if figure.unit else ""
        bare = [number.removesuffix(unit) if unit else number for number in numbers]
        if any(abs(value) >= WHOLE_FROM for value in values) or any(not _PLAIN_END.fullmatch(n) for n in bare):
            # The ends stay as the key spelled each alone, which is what it prints.
            self._key_ends(figure, values, (numbers[0], numbers[1]))
            return None
        shown = [abs(value) if absolute else value for value in values]
        held = self._key_figure_places(figure)
        signed = figure.change and not absolute
        if held is not None:
            # The one interval form's ends at the measure's decimals, each kept off zero (policy_interval).
            low, high, _text = policy_interval(shown[0], shown[1], held, change=signed, bounds=())
            # The unit written once, after the high end, as the one interval form writes it (``-0.007 to +0.002
            # notices``).
            self._key_ends(figure, values, (low, high + unit))
            return low, high + unit
        places = ends_places(shown[0], shown[1], set_places([shown[1] - shown[0]]))
        texts = [spell_at(value, places) for value in shown]
        low, high = (
            plus_signed(text, value) if signed and float(text) != 0 else text
            for text, value in zip(texts, shown, strict=True)
        )
        self._key_ends(figure, values, (low + unit, high + unit))
        return low + unit, high + unit

    def key_points_ends(self, figure: Figure, values: tuple[float, float], *, absolute: bool) -> tuple[str, str]:
        """An interval in points with both ends at one precision: its rate's decimals where the page holds the rate,
        each end kept off zero (``-15.1 to -0.03``); else a point's one decimal, a whole end too (``-29.0 to +57.6``),
        or more where an end that is not zero would read as one (``-2.2 to -0.04``, never ``-2.2 to 0.0``, which would
        hide which side of zero it lies). A lone figure in points keeps the number rule,
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
        places = self._key_rate_places(figure)
        if places is not None:
            # The one interval form's ends at the rate's decimals, each kept off zero (policy_interval).
            shown = [abs(value) for value in hundred] if absolute else hundred
            low, high, _text = policy_interval(shown[0], shown[1], places, change=not absolute, bounds=())
            self._key_ends(figure, values, (low, high))
            return low, high
        at = ends_places(hundred[0], hundred[1], POINT_PLACES)
        low, high = (
            spell_at(abs(value), at) if absolute else plus_signed(spell_at(value, at), value) for value in hundred
        )
        self._key_ends(figure, values, (low, high))
        return low, high

    # Each spells a figure key's number by one of the filters that choose its spelling, as the key prints it today,
    # unsigned where the key adds its sign and without the unit it writes once. A number figure's text is recorded as
    # printed, under the measure whose set the key reads, by what the figure is to it; an interval's two ends are
    # recorded once both are decided (:meth:`key_ends`), since a key may respell them together.

    def key_percent(self, figure: Figure, hundred: float, *, tens: bool = False) -> str:
        """A rate's figure in percent, its digits without the ``%``: at its measure's decimals where the page holds the
        rate (:meth:`_key_rate_places`), else by the number rule for a percent; a descriptive ratio whole
        (:attr:`~threetears.evals.analysis.report.figure_keys.Figure.whole`); an interval's confidence level as every
        level prints as text (:func:`policy_level`; the per-interval level ``{family.level}`` cites), or the joint
        guarantee, which a figure in the package's own
        words holds as its value (:attr:`~threetears.evals.analysis.report.figure_keys.Figure.joint`); and ``tens`` to the nearest 10% (the ``pct10`` filter, an
        approximation the writer chooses).

        :param figure: What the key resolved to.
        :ptype figure: Figure
        :param hundred: The figure in percent (a share's x 100).
        :ptype hundred: float
        :param tens: Whether the key reads it to the nearest 10%.
        :ptype tens: bool
        :return: Its digits.
        :rtype: str
        """
        places = self._key_rate_places(figure)
        if tens:
            text = format_number(float(round(hundred / 10) * 10))
        elif figure.level:
            # An interval's level as every level prints as text (:func:`policy_level`): the joint guarantee where the
            # package's own words state one (its value, from :func:`joint_level`), else the per-interval one
            # {family.level} cites, its figure exact and its text rounded (owner ruling, 2026-11-14).
            text = policy_level(hundred).removesuffix(PERCENT)
        elif figure.whole:
            text = number_in(float(round(hundred)), PERCENT)
        elif places is not None:
            # A rate's level at its measure's decimals, kept off its floor and its ceiling; a change kept off zero.
            text = policy_figure(hundred, places, bounds=_ZERO_BOUND if figure.change else _RATE_BOUNDS)
        else:
            text = number_in(hundred, PERCENT)
        return self._key_issue(figure, hundred, text)

    def key_points(self, figure: Figure, hundred: float) -> str:
        """A figure in points, its digits unsigned: at its rate's decimals where the page holds the rate, kept off
        zero (:meth:`_key_rate_places`), else by the number rule for points.

        :param figure: What the key resolved to; a share's figure is read in points (x 100).
        :ptype figure: Figure
        :param hundred: The figure in points.
        :ptype hundred: float
        :return: Its digits.
        :rtype: str
        """
        places = self._key_rate_places(figure)
        text = number_in(hundred, POINTS) if places is None else policy_figure(hundred, places, bounds=_ZERO_BOUND)
        return self._key_issue(figure, hundred, text)

    def key_money(
        self,
        figure: Figure,
        value: float,
        *,
        per: int,
        start: int = 0,
        finer: bool = False,
        signed: bool = False,
    ) -> tuple[str, str]:
        """A money figure in its family's one form (#993), and the words of the count it is stated per: over the
        family's whole set (:attr:`~threetears.evals.analysis.report.figure_keys.Figure.amounts`), never the figure
        alone, its count moved per 1,000 more where the set is under a cent; a change in the changes' form on the
        amounts' ruler, capped by its interval's width. A figure made without its set is a set of its own numbers.

        :param figure: What the key resolved to.
        :ptype figure: Figure
        :param value: The amount, as stored.
        :ptype value: float
        :param per: How many of what it counts one stored amount is per; 0 for a bare amount, which states no count
            and reads in plain dollars.
        :ptype per: int
        :param start: The least count to state it per.
        :ptype start: int
        :param finer: Whether the amount is an end of an interval whose two ends would read alike, which takes one
            significant figure more, for that interval alone.
        :ptype finer: bool
        :param signed: Whether it reads signed, as a change does.
        :ptype signed: bool
        :return: The amount, and the words of its count (empty for a bare amount).
        :rtype: tuple[str, str]
        """
        ruler = _key_ruler(figure, per=per, start=start, finer=finer)
        text = ruler.spell(value, signed=signed)
        words = ruler.words(figure.per) if per else ""
        return self._key_issue(figure, value, text), words

    def key_refused_whole(
        self, figure: Figure, value: float, floor: float, *, finer: bool = False, signed: bool = False
    ) -> tuple[str, str]:
        """What a ``usd_whole`` key's refusal names: the amount, a bare one in its family's form (:meth:`key_money`),
        and the least amount the filter reads, as declared. A refusal prints no figure, so nothing is recorded.

        :param figure: What the key resolved to.
        :ptype figure: Figure
        :param value: The amount, as stored.
        :ptype value: float
        :param floor: The least amount whole dollars read.
        :ptype floor: float
        :param finer: Whether the amount is an end of an interval whose two ends would read alike.
        :ptype finer: bool
        :param signed: Whether it reads signed, as a change does.
        :ptype signed: bool
        :return: The amount and the floor.
        :rtype: tuple[str, str]
        """
        return _key_ruler(figure, per=0, finer=finer).spell(value, signed=signed), format_number(floor)

    def key_whole_count(self, value: float) -> int | None:
        """A figure as a whole count of none or more, for the ``words`` filter to name; None where it is no whole
        count. A decision that prints nothing, so nothing is recorded.

        :param value: The figure.
        :ptype value: float
        :return: The count, or None.
        :rtype: int | None
        """
        whole = round(value)
        return whole if whole == value and whole >= 0 else None

    def key_whole(self, figure: Figure, value: float) -> str:
        """A figure to the nearest whole number, its thousands grouped (``4,000,000``), unsigned where the key adds its
        sign: the ``int``, ``usd_whole`` and ``words`` filters' digits.

        :param figure: What the key resolved to.
        :ptype figure: Figure
        :param value: The number.
        :ptype value: float
        :return: Its digits.
        :rtype: str
        """
        return self._key_issue(figure, value, group_thousands(format_number(float(round(value)))))

    def key_multiple(self, figure: Figure, value: float) -> str:
        """A figure in a unit the key keeps (a multiple, ``21 times``), its digits without the unit word
        (:func:`~threetears.evals.kernel.numbers.spell_multiple`).

        :param figure: What the key resolved to.
        :ptype figure: Figure
        :param value: The multiple.
        :ptype value: float
        :return: Its digits.
        :rtype: str
        """
        return self._key_issue(figure, value, spell_multiple(value))

    def key_interval(
        self,
        figure: Figure,
        values: tuple[float, float],
        spelled: Callable[[float, bool], tuple[str, str]],
        *,
        spelling: str,
        absolute: bool,
        joined: bool,
    ) -> str:
        """An interval as a key prints it: each end spelled by the filter the key chose (``spelled``, the grammar's,
        which refuses a filter that does not suit), one significant figure more for an interval whose two differing ends
        would read alike (owner ruling), both ends at one precision where the spelling sets one (plain, points), and
        the two joined: one figure where they read alike, ``low to high`` where a count is said once after them (money
        per what it counts, the count after), each end's own unit where the two differ, else the one interval form,
        its unit once after the high end, signed or not (``-13 to -2 pts``, ``27 to 28%``; number rule 3).

        :param figure: What the key resolved to.
        :ptype figure: Figure
        :param values: The two ends, as the key reads them (sorted by size where ``absolute``).
        :ptype values: tuple[float, float]
        :param spelled: One end's number and the suffix unit it carries, by the key's spelling; ``True`` for one
            significant figure more.
        :ptype spelled: Callable[[float, bool], tuple[str, str]]
        :param spelling: The spelling the key's filters chose (``plain``, ``pts``, ``pct``, ``usd`` ...).
        :ptype spelling: str
        :param absolute: Whether the key reads the ends' sizes (``abs``), unsigned.
        :ptype absolute: bool
        :param joined: Whether the key asks for its ends joined by ``to`` (the ``interval`` filter).
        :ptype joined: bool
        :return: The interval's text.
        :rtype: str
        """
        low, high = values
        ends = [spelled(end, False) for end in values]
        if ends[0] == ends[1] and low != high:
            # Two ends that differ never read alike: one significant figure more, for this interval alone.
            ends = [spelled(end, True) for end in values]
        if spelling == "plain":
            respelled = self.key_plain_ends(figure, values, (ends[0][0], ends[1][0]), absolute=absolute)
            if respelled is not None:
                ends = [(text, end[1]) for text, end in zip(respelled, ends, strict=True)]
        elif spelling == "pts":
            points = self.key_points_ends(figure, values, absolute=absolute)
            ends = [(text, end[1]) for text, end in zip(points, ends, strict=True)]
        elif spelling != "words":
            # The plain and points ends are recorded as they are respelled; a count's words are no figure's digits.
            self.key_ends(figure, values, (ends[0][0], ends[1][0]))
        (first, first_unit), (second, second_unit) = ends
        if ends[0] == ends[1]:
            return first + first_unit
        if first_unit == second_unit and first_unit.startswith(" per "):
            return interval_beside(first, second) + first_unit
        if first_unit != second_unit:
            return interval_beside(first + first_unit, second + second_unit)
        # The one interval form, signed or not (number rule 3): its unit written once, after the high end (``-13 to -2
        # pts``, ``27 to 28%``).
        return interval_beside(first, second) + second_unit

    def key_ends(self, figure: Figure, values: tuple[float, float], numbers: tuple[str, str]) -> tuple[str, str]:
        """An interval's two ends as a key prints them, once both are decided, each recorded apart under the measure
        whose set the key reads (a key's sentence may join them by a dash, a ``to`` or its unit). The plain and points
        ends record themselves (:meth:`key_plain_ends`, :meth:`key_points_ends`).

        :param figure: What the key resolved to.
        :ptype figure: Figure
        :param values: The two ends, as the key reads them.
        :ptype values: tuple[float, float]
        :param numbers: The two ends' texts, without the unit the key writes once.
        :ptype numbers: tuple[str, str]
        :return: The two texts, as given.
        :rtype: tuple[str, str]
        """
        self._key_ends(figure, values, numbers)
        return numbers

    def _key_issue(self, figure: Figure, value: float, text: str) -> str:
        """A key's number as printed, recorded where the figure is one number; an interval's ends wait for
        :meth:`key_ends`."""
        if figure.kind != "interval":
            self._issue(*_key_set(figure), _key_role(figure), value, text, text, "")
        return text

    def _key_ends(self, figure: Figure, values: tuple[float, float], texts: tuple[str, str]) -> None:
        """An interval's two ends as a key prints them, each recorded apart (a key's sentence may join them by a dash,
        a ``to`` or its unit), under the measure whose set the key reads."""
        key = _key_set(figure)
        self._issued += [_Issued(*key, "interval", values, text) for text in texts]

    def _key_figure_places(self, figure: Figure) -> int | None:
        """A key's figure's measure's decimals (:meth:`key_places`), where the figure is one the page's tables print of
        the measure it is of (:func:`_tabled`); None otherwise. Only a plain figure of a measure or a judged score reads
        here: a rate's and a change in points read theirs by their spellings (:meth:`_key_rate_places`), and money, a
        count and a multiple keep their spellings' own rules."""
        if figure.set_of is None or not _tabled(figure) or figure.scale != "plain" or figure.unit == MULTIPLE_UNIT:
            return None
        key, kind = figure.set_of
        if kind == "trial" or (kind == "judged" and not rubric_score(self.report_input, key)):
            return None
        return self.key_places(key, kind=kind)

    def _key_rate_places(self, figure: Figure) -> int | None:
        """The decimals a key's rate or change in points reads at: its measure's, where the figure is one the page's
        tables print of a rate the page holds (:func:`_tabled`); None otherwise (an interval's confidence level, a
        figure of no measure), which its spelling's own rule prints."""
        if figure.set_of is None or figure.level or not _tabled(figure) or self._input is None:
            return None
        key, kind = figure.set_of
        form = self._measures.get((key, _measure_kind(kind)))
        return form.places if form is not None and form.reads_as == "rate" else None

    # --- a table's column rules: today's, each deciding over the cells its table already built --------------------
    #
    # A table states each set of its money in one form, decided over the column (or the row) after its cells are
    # built: a cell keeps its value and glyph, and takes the form its set reads in. It records the text every cell it
    # reads prints, under the measure its builder named (:meth:`of_measure`), and a number carries that text as its
    # decided text (``value_text``), so the ledger holds what a money column prints, not only what its cells were built
    # from. A rate's cell prints its measure's text, decided where it is built.

    def money_columns(
        self,
        rows: Sequence[Mapping[str, TableCell | ColumnMoney]],
        *,
        by_row: Sequence[str | None] | None = None,
        reconciles: Collection[str] = (),
    ) -> list[dict[str, TableCell]]:
        """A table's rows with each set of its money in one form (#993). A set is every amount of one figure type the
        table shows, across its rows and columns: by default a column's (a table of arms by measure); where ``by_row``
        names a row's figure type (a table of slices by arm, each row one measure across the arms), every money amount
        of the rows naming the same one. Every amount a set holds (:class:`ColumnMoney`, which its builder leaves to
        its set), and the ends of its intervals, choose the set's :func:`~threetears.evals.kernel.numbers.money_form`
        from the values themselves, never from one amount, and each is written in it: plain dollars to two significant
        figures, every set a rate (money per the count the table's head names), in whole dollars from $10. Each cell
        writes its value and its interval's ends at one precision, the finest of them needs, and one significant
        figure more where its own interval's ends would otherwise read alike
        (:meth:`~threetears.evals.kernel.numbers.MoneyForm.places_of`), so $0.008 reads ``$0.0080`` beside ``$0.11``,
        never ``$0.1100``. A table states the count its money is drawn per in its head or its row, never as a mark on
        the amount. The values are as stored, so the decimals state no digit a value does not hold; a cell its builder
        decided (words, another unit, or money in a form chosen from a wider set) is left as it is.

        Every money number it reads is recorded as it prints (``column``), a form its builder chose too, under the
        measure its builder named (:meth:`of_measure`), and carries the text recorded (``value_text``).

        :param rows: The table's rows, by column key: each cell as its builder made it, or an amount left to its set.
        :ptype rows: Sequence[Mapping[str, TableCell | ColumnMoney]]
        :param by_row: Each row's figure type, where the table lays one per row (``None`` for a row whose money is set
            by its column); unset for a table whose every set is a column.
        :ptype by_row: Sequence[str | None] | None
        :param reconciles: The sets a total computed from each amount is printed beside (a cost per 1,000 at a declared
            monthly volume), whose amounts take a third figure where two would leave them off that total
            (:class:`~threetears.evals.kernel.numbers.MoneyForm`).
        :ptype reconciles: Collection[str]
        :return: The rows, each money cell carrying its set's form.
        :rtype: list[dict[str, TableCell]]
        """

        def set_of(at: int, key: str) -> tuple[str, str]:
            kind = by_row[at] if by_row is not None else None
            return ("row", kind) if kind is not None else ("column", key)

        held: dict[tuple[str, str], list[float]] = {}
        for at, row in enumerate(rows):
            for key, cell in row.items():
                if isinstance(cell, ColumnMoney):
                    held.setdefault(set_of(at, key), []).extend(cell.amounts)
        forms = {
            name: money_form(amounts, rate=True, reconciles=name[1] in reconciles) for name, amounts in held.items()
        }
        return [
            {
                key: self._column_cell(cell, forms[set_of(at, key)])
                if isinstance(cell, ColumnMoney)
                else self._issue_cell(cell, cell)
                if cell.is_number and is_money(cell.unit)
                else cell
                for key, cell in row.items()
            }
            for at, row in enumerate(rows)
        ]

    def _column_cell(self, amount: ColumnMoney, form: MoneyForm) -> TableCell:
        """An amount written in its set's form, at the places its value and its interval's ends need in it, recorded as
        it prints under the measure its builder named."""
        ends = (amount.low, amount.high)
        money = MoneyForm(decimals=form.places_of(list(amount.amounts), ends=ends))
        text = money.spell(amount.value)
        self._issued.append(_Issued(*_keyed(amount.measure, amount.kind), "column", amount.value, text))
        interval: dict[str, Any] = {}
        if amount.low is not None and amount.high is not None:
            interval = {
                "low": amount.low,
                "high": amount.high,
                "low_text": money.spell(amount.low),
                "high_text": money.spell(amount.high),
            }
        cell = TableCell(value=amount.value, unit=MONEY, value_text=text, flag=amount.flag, **interval)
        return self.of_measure(cell, measure=amount.measure, kind=amount.kind) if amount.measure else cell

    def _issue_cell(self, cell: TableCell, built: TableCell) -> TableCell:
        """A cell a column rule read, its text recorded as it prints, without its flag (which says how the cell reads,
        not its figure): under the measure its builder made ``built``, the cell the rule was handed, a figure of
        (:meth:`of_measure`), or under none where it named none. A number carries the text recorded (``value_text``),
        so every reader prints what the ledger holds."""
        value = float(cell.value) if cell.is_number else None  # type: ignore[arg-type]
        _held, measure, kind = self._cell_measures.get(id(built), (built, "", ""))
        self._issued.append(_Issued(measure, kind, "column", value, cell.figure))
        return self.of_measure(cell, measure=measure, kind=kind) if measure and cell.is_number else cell

    def of_measure(self, cell: TableCell, *, measure: str, kind: _MeasureKind) -> TableCell:
        """A cell a builder made, held as a figure of ``measure`` read as ``kind``, so a column rule that spells it
        (:meth:`money_columns`) records what it prints under that measure: the ledger then holds a
        rate's or money's table cell by its measure, as it holds every other figure the page prints. Nothing is
        printed, so nothing is issued.

        :param cell: The cell, as its builder made it.
        :ptype cell: TableCell
        :param measure: The measure or dimension whose figure it prints; empty for none, and nothing is held.
        :ptype measure: str
        :param kind: Its reading (``measure``, ``judged``, ``trial``).
        :ptype kind: _MeasureKind
        :return: The cell, as it was.
        :rtype: TableCell
        """
        if measure:
            self._cell_measures[id(cell)] = (cell, *_keyed(measure, kind))
        return cell

    def as_printed(
        self,
        rows: Sequence[Mapping[str, TableCell]],
        *,
        bare: Mapping[str, str] | None = None,
        keep: Collection[str] = (),
    ) -> list[dict[str, TableCell]]:
        """A table's rows as their builders decided them, each rate's cell recorded as it prints, under the measure its
        builder named (:meth:`of_measure`), and a number carrying the text recorded (``value_text``). Every rate a
        written page's table prints is its measure's decided text, so no column respells it; a cell a builder laid
        from a key's words (a gate's line, a pass rate a figure key states) is recorded here, so the ledger holds what
        every rate cell prints.

        :param rows: The table's rows, by column key.
        :ptype rows: Sequence[Mapping[str, TableCell]]
        :param bare: The columns whose head names their unit (:func:`~threetears.evals.analysis.report.blocks.
            bare_columns`), whose bare figures are a rate's; None for none.
        :ptype bare: Mapping[str, str] | None
        :param keep: The columns that state no rate (an interval's level), recorded where they were decided.
        :ptype keep: Collection[str]
        :return: The rows, each rate's number carrying its text.
        :rtype: list[dict[str, TableCell]]
        """
        under = {key: unit for key, unit in (bare or {}).items() if key not in keep}
        return [
            {
                key: self._issue_cell(cell, cell)
                if key not in keep and (_a_rate_cell(cell, key in under) or _rate_words(cell, key in under))
                else cell
                for key, cell in row.items()
            }
            for row in rows
        ]

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
    ) -> Printed:
        self._issued.append(_Issued(key, kind, role, value, text))
        return Printed(value=value, text=text, digits=digits, unit=unit, held=held)

    def _code_issue(self, form: MeasureForm, role: Role, value: float, scaled: float) -> Printed:
        """A code page's level, from its value scaled into the unit it is printed in: the number its cell holds, the
        decimals the cell writes it at, and its text, at its measure's places wherever the page states it, the folded
        table of every measure too (#1127 item 8)."""
        if form.ruler is not None:
            text = MoneyForm(decimals=form.ruler.form.places_of([scaled])).spell(scaled)
            return self._issue(form.key, form.kind, role, value, text, _digits_of(text, MONEY), MONEY, held=scaled)
        held = _code_held(form, scaled)
        unit = form.unit
        text = _code_cell_text(form, scaled)
        return self._issue(form.key, form.kind, role, value, text, _digits_of(text, unit), unit, held=held)


# --- an aspect page's contrasts: one set per reading -----------------------------------------------------------------


def _resolvable(comparison: FamilyComparison) -> bool:
    """Whether a contrast can resolve, and so is a row of the contrasts table: both sides hold cases, at the floor."""
    return min(comparison.control.n_cases, comparison.contrast.n_cases) > 0 and PageForms.small_n(comparison) is None


def _reading_places(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    ruler: MeasureRuler,
    name: str,
    reading: str,
    widths: Sequence[float],
    figures: Iterable[float],
) -> ContrastPlaces:
    """One reading's :class:`ContrastPlaces`: its levels and its changes at the places its narrowest interval sets
    (:func:`policy_places`, drawn on its ruler), a rate's capped at one decimal, a reading with no interval at two
    significant figures of its least figure; money by its own form. A level never runs past what it can take."""
    drawn = [width * ruler.factor for width in widths]
    if is_money(ruler.unit):
        return ContrastPlaces(None, None, floor=0.0)
    if ruler.unit == PERCENT:
        places = policy_places(drawn, figures, capped=True)
        return ContrastPlaces(places, places, floor=0.0, ceiling=100.0)
    places = policy_places(drawn, figures)
    if reading == "judged":
        facts = surface.dimensions.get(name)
        bounds = facts.value_range if facts is not None else None
        if bounds is not None:
            low, high = (bound * ruler.factor for bound in bounds)
            scale = f"{format_number(low)}–{format_number(high)}"
            return ContrastPlaces(places, places, floor=low, ceiling=high, scale=scale)
    described = bundle.measure_catalog.get(name) or METRIC_DESCRIPTORS.get(name) if reading == "measure" else None
    floor = described.interval_floor if described is not None else None
    ceiling = described.value_range[1] if described is not None and described.value_range is not None else None
    return ContrastPlaces(
        places,
        places,
        floor=floor * ruler.factor if floor is not None else None,
        ceiling=ceiling * ruler.factor if ceiling is not None else None,
    )


def _contrast_places(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    rulers: Rulers,
    comparisons: Iterable[FamilyComparison],
    campaign: Mapping[MeasureSet, Sequence[float]],
) -> dict[tuple[str, str], ContrastPlaces]:
    """Each reading's :class:`ContrastPlaces` over every contrast an aspect's page states and every reading it holds
    a half-width of, by name and reading: its places from its set's half-widths across the campaign (``campaign``,
    :func:`campaign_set`), its figures where it has none."""
    # A reading the page states no contrast of (a bar's, a guardrail's, a per-label statistic's) still has its set: from
    # its half-widths, else from every arm's figure of it.
    held: dict[tuple[str, str], list[float]] = {(width.name, width.reading): [] for width in _bundle_stated(bundle)}
    # The page states every arm's pass rate (its outcome bar's note) whether or not it compares one: where the campaign
    # holds intervals on the rate (a choice's pairs, a change's readings), the note reads at their places, as the
    # campaign's scorecard prints the same passes, never at its ruler's own one decimal beside them.
    if _set_widths(campaign, _aspect_page(bundle.aspect), TRIAL_PASS_NAME, "trial"):
        held.setdefault((TRIAL_PASS_NAME, "trial"), [])
    for cell in surface.cells:
        for summary in cell.measures.measures:
            figure = summary.rate if summary.rate is not None else summary.mean
            if figure is not None:
                held.setdefault((summary.name, "measure"), []).append(
                    figure * rulers.of(summary.name, "measure").factor
                )
        for judged in cell.judged:
            if judged.mean is not None:
                held.setdefault((judged.dimension, "judged"), []).append(
                    judged.mean * rulers.of(judged.dimension, "judged").factor
                )
    for comparison in comparisons:
        ruler = rulers.of(comparison.name, comparison.reading)
        figures = held.setdefault((comparison.name, comparison.reading), [])
        figures += [
            side.mean * ruler.factor for side in (comparison.control, comparison.contrast) if side.mean is not None
        ]
        if comparison.delta is not None:
            figures.append(comparison.delta * ruler.factor)
    return {
        (name, reading): _reading_places(
            bundle,
            surface,
            rulers.of(name, reading),
            name,
            reading,
            _set_widths(campaign, _aspect_page(bundle.aspect), name, reading),
            figures,
        )
        for (name, reading), figures in held.items()
    }


def _change_money(
    rulers: Rulers, changes: Iterable[tuple[str, str, float | None, tuple[float, float] | None]]
) -> dict[tuple[str, str], MoneyForm]:
    """The one form each money measure's changes read in, a change and its interval's ends alike (#993): from every
    change an aspect's page states of the measure and the widths of their intervals, on the measure's ruler
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


def _on_the_figure(unit: str) -> str:
    """The unit an aspect page writes on a figure itself (money's ``$``, a percent's ``%``); empty for any other unit,
    which the row's heading or the interval's end names once."""
    if is_money(unit):
        return MONEY
    return unit if unit == PERCENT else ""


def _ruled_level_text(ruler: MeasureRuler, scaled: float, places: ContrastPlaces | None) -> str:
    """A level's text on an aspect page, from its value on its ruler: money in its ruler's form; at its reading's places
    where its set holds them, a rate with its ``%`` and kept off its floor and ceiling (``99.7%``, never ``100%``), any
    other bare, its unit in the row's heading (an exact zero ``0``); else by the unit's precision."""
    if is_money(ruler.unit) and ruler.money is not None:
        return ruler.money.spell(scaled)
    if places is not None and places.level is not None:
        if ruler.unit == PERCENT:
            return policy_rate(scaled, places.level)
        return policy_figure(scaled, places.level, bounds=())
    if ruler.unit == PERCENT or is_money(ruler.unit):
        return with_unit(scaled, ruler.unit)
    return number_in(scaled, ruler.unit)


def _change_ends(
    ruler: MeasureRuler,
    low: float,
    high: float,
    *,
    form: MoneyForm | None,
    places: int | None,
    span: float | None,
) -> tuple[str, str, str]:
    """An interval on a change on its ruler (:meth:`PageForms.ruled_change_interval`): its two ends' texts and its
    whole text, money's in the form its changes read in, any other unit's at ``places``, each end kept off zero on its
    own and never past ``span`` either side of zero, a rate's in points at a point's one decimal where no places are
    given, else each end by its unit's precision."""
    drawn_low, drawn_high = low * ruler.factor, high * ruler.factor
    unit = delta_unit(ruler.unit)
    if form is not None and is_money(ruler.unit):
        return _money_ends(form.spell_ends(drawn_low, drawn_high), drawn_low, drawn_high, change=True)
    if places is not None:
        if span is not None:
            drawn_low, drawn_high = max(drawn_low, -span), min(drawn_high, span)
        return _ends_at(drawn_low, drawn_high, places, unit, change=True)
    if ruler.unit == PERCENT:
        return _ends_at(max(drawn_low, -100.0), min(drawn_high, 100.0), POINT_PLACES, POINTS, change=True)
    return _ends_in(drawn_low, drawn_high, unit, change=True)


def _ends_at(
    low: float, high: float, places: int, unit: str, *, change: bool, bounds: Collection[float] = ()
) -> tuple[str, str, str]:
    """An interval's two ends in its one form at its reading's places (:func:`policy_interval`): ``-6.9 to +11.0 pts``,
    ``62.3 to 65.1%``. Each end's text carries the unit where it sits on the number, a percent's; the unit is written
    once, after the high end."""
    low_text, high_text, _ = policy_interval(low, high, places, change=change, bounds=bounds)
    if is_money(unit):
        # Money's ``$`` is a prefix, on every end (``-$0.01 to +$0.02``).
        return _money_ends((_dollars_on(low_text), _dollars_on(high_text)), low, high, change=False)
    on = _on_the_figure(unit)
    return unit_spaced(low_text, on), unit_spaced(high_text, on), interval_alone(low_text, high_text, unit)


def _dollars_on(digits: str) -> str:
    """Decided digits with money's ``$`` on the number, inside its sign or bound (``-$0.01``, ``< $0.000001``)."""
    found = re.match(r"([+\-−]?(?:[<>] )?)(.*)", digits)
    assert found is not None  # The pattern matches any text.
    return f"{found[1]}${found[2]}"


def _ends_in(low: float, high: float, unit: str, *, change: bool) -> tuple[str, str, str]:
    """An interval's two ends, each at its unit's precision, in the one form: ``88.1 to 100%``, ``0.45 to 0.55 s``; a
    change's ends signed; money's each with its ``$`` (``-$0.01 to +$0.02``), the one unit a prefix writes on every
    end."""
    if is_money(unit):
        texts = (with_unit(low, unit), with_unit(high, unit))
        return _money_ends(texts, low, high, change=change)
    ends = [number_in(end, "") for end in (low, high)]
    if change:
        ends = [plus_signed(end, value) for end, value in zip(ends, (low, high), strict=True)]
    on = _on_the_figure(unit)
    return unit_spaced(ends[0], on), unit_spaced(ends[1], on), interval_alone(ends[0], ends[1], unit)


def _money_ends(texts: tuple[str, str], low: float, high: float, *, change: bool) -> tuple[str, str, str]:
    """Two money ends, each in its form with its ``$``, and the interval in its one form: ``$0.033 to $0.035``, a
    change's ends signed (``-$0.0030 to +$0.0019``)."""
    low_text, high_text = texts
    if change:
        low_text, high_text = plus_signed(low_text, low), plus_signed(high_text, high)
    return low_text, high_text, interval_beside(low_text, high_text)


def _written_measures(report_input: ReportInput) -> dict[tuple[str, _MeasureKind], MeasureForm]:
    """A written page's form of each measure, judged dimension and the pass rate its input holds a figure of
    (:meth:`PageForms.of`), as its Full results decide them.

    :param report_input: The page's report input.
    :ptype report_input: ReportInput
    :return: Each form, by its measure's key and kind.
    :rtype: dict[tuple[str, _MeasureKind], MeasureForm]
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
    stated, set_figures = _written_stated(report_input, kinds)
    # The campaign's one table (:func:`_measure_sets`) over what the input mirrors of it: each interval the code
    # page and the aspect pages state of an arm's level, a comparison's change or a guardrail check, its provenance
    # decided as theirs is (:data:`_PROVENANCE`). test_report_forms holds the pages to one form per measure.
    sets = _measure_sets(stated)
    measures = {(key, kind): _written_form(report_input, key, kind, sets, set_figures) for key, kind in kinds}
    return measures
