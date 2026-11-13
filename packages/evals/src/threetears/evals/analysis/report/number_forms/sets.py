"""Which figures share a measure's set and the precision it reads at: a set's name (:func:`campaign_set`), the
half-widths each page states, gathered across the campaign, and the helpers a page's forms read a figure's text,
signs and ends with.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME, AnalysisContextBundle
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS, is_money, number_in, with_unit
from threetears.evals.analysis.report.blocks import TableCell
from threetears.evals.analysis.report.cost_unit import MoneyRuler, money_ruler
from threetears.evals.analysis.report.keys.figure import Figure
from threetears.evals.analysis.report.keys.resolve import plus_signed
from threetears.evals.analysis.report.number_forms.intervals import _tabled
from threetears.evals.analysis.report.number_forms.kinds import Role, _MeasureKind
from threetears.evals.analysis.report.unit_once import unit_spaced
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_margins import aspect_board
from threetears.evals.analysis.stats import mean_interval
from threetears.evals.kernel.numbers import POINT_PLACES, MoneyForm, ends_places, spell_at


#: The values a chart draws of its measure, by the field each is held in, whose texts a chart of one measure records
#: (:meth:`PageForms.chart_texts`); the rest (a count, a cost, a point's other axis, a detail) are no value of it.
_MEASURE_VALUES = frozenset({"value", "values", "lo", "hi", "low", "high", "from", "to"})

#: The fields among them that are an interval's ends.
_INTERVAL_ENDS = frozenset({"lo", "hi", "low", "high"})

#: The lines a chart draws at a declared value (a floor, a budget, a margin, a price), which are no value of its
#: measure either.
_CHART_LINES = frozenset({"reference", "x_reference", "price_line"})

#: The fields a view of a page's input replaces (:meth:`PageForms.over`): a configs chart's grid. Every other field is
#: the input's own.
_GRID = frozenset({"factors", "configs"})

#: The fewest cases a judged score with no interval reads to a hundredth at; below it the score reads a tenth. A judged
#: score's spread across cases is typically half a point to a point on a 1 to 5 scale, so below about 30 cases its 95%
#: half-width is about 0.3 or wider, which rule 2 reads at a tenth: a hundredth there is noise.
_SCORE_HUNDREDTHS_FROM = 30

#: The decimals an off-page reading is stated to (:meth:`PageForms.reading`).
_READING_PLACES = 2

#: Where a share of a judge's preference is even, in percent: the judge preferred neither page.
EVEN_SHARE = 50.0

#: The most decimals a side-by-side figure is read to (:func:`_places`).
_MOST_SIDE_BY_SIDE_PLACES = 6


def _drawn_texts(node: object) -> Iterator[tuple[str, float | None, str]]:
    """Every value a chart's dumped data draws of its measure (:data:`_MEASURE_VALUES`) with the text it carries, by
    the value's field: a list of values beside a list of texts, one by one; a value with no text left out."""
    if isinstance(node, list):
        for item in node:
            yield from _drawn_texts(item)
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        text = node.get(f"{key}_text") if key in _MEASURE_VALUES else None
        if isinstance(text, str):
            yield key, float(value) if isinstance(value, int | float) else None, text
        elif isinstance(text, list) and isinstance(value, list):
            for one, said in zip(value, text, strict=False):
                if isinstance(said, str):
                    yield key, float(one) if isinstance(one, int | float) else None, said
        elif not key.endswith("_text") and key not in _CHART_LINES:
            yield from _drawn_texts(value)


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


def _decimals_in(text: str) -> int:
    """How many decimals a figure's text is written to: ``2`` for ``$3.33``, none for ``90%``."""
    found = re.search(r"\.(\d+)", text)
    return len(found[1]) if found is not None else 0


def _digits_of(text: str, unit: str) -> str:
    """A figure's text without its unit: money's ``$`` taken from inside its sign or bound (``-$0.30``, ``< $0.000001``),
    any other unit from its end."""
    if unit == MONEY:
        return text.replace("$", "", 1)
    if not unit:
        return text
    return text.removesuffix(unit).rstrip() if text.endswith(unit) else text


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


#: A measure's set as :func:`campaign_set` names it: the scope it holds across, then the measure's kind and key.
MeasureSet = tuple[str, str, str]

#: What a measure's set holds across: its whole campaign (the owner's set-scope ruling), or each page alone (``"page"``),
#: which the owner may yet choose. The one line that decides it, read by :func:`campaign_set` alone.
_SET_SCOPE: Literal["campaign", "page"] = "campaign"


def campaign_set(page: str, measure: str, kind: _MeasureKind) -> MeasureSet:
    """The set a figure of ``measure`` stated on ``page`` joins: two figures share a set, and so its decimals, exactly
    when this names one set for both.

    The one place that decides a set's scope. A set is the measure across the whole campaign, its written page, its
    code page and every aspect page alike (the owner's set-scope ruling), so the scope is ``"campaign"`` wherever the
    figure is stated. The owner may yet let each page decide: that is :data:`_SET_SCOPE` reading ``"page"``, one line,
    and the scope is then ``page``. Every page reads its sets from its campaign's one table (:func:`_measure_sets`),
    each half-width in it keyed here by the page that states it, so nothing else moves.

    :param page: The name of the page that states the figure: ``written``, ``code``, or an aspect's page
        (:func:`_aspect_page`).
    :ptype page: str
    :param measure: The measure's key.
    :ptype measure: str
    :param kind: The measure's kind; None for a measure's own.
    :ptype kind: _MeasureKind
    :return: The set's name.
    :rtype: MeasureSet
    """
    return (page if _SET_SCOPE == "page" else "campaign", _measure_kind(kind), measure)


def _signed_text(text: str, value: float) -> str:
    """A change's text with the ``+`` a positive one carries, where it reads as more than zero and states no bound."""
    shown = re.search(r"[1-9]", text) is not None
    return f"+{text}" if value > 0 and shown and not text.startswith(("+", "<", ">")) else text


def _amount(number: float, unit: str, places: int | None, money: MoneyForm | None, *, change: bool) -> str:
    """A number of a table cell's, written in the form its builder decided: money in its form; a percent, a bare
    figure, points or another unit at ``places`` (a level of exactly zero whole, ``0%``, ``0 notices``); with no
    places, in its unit by the number rule (:func:`~threetears.evals.analysis.quantities.with_unit`)."""
    if money is not None and is_money(unit):
        return money.spell(number)
    if places is None:
        return with_unit(number, unit)
    if unit == PERCENT:
        # An exact zero reads whole, ``0%``, which states no precision.
        return "0%" if number == 0 else f"{spell_at(number, places)}%"
    if unit == POINTS:
        # A change at its unit's precision: a whole one keeps its decimal beside its siblings (``10.0 pts``).
        return f"{spell_at(number, places)} {POINTS}"
    shown = "0" if number == 0 and not change else spell_at(number, places)
    return unit_spaced(shown, unit)


def _glyph_ends(
    low: float,
    high: float,
    *,
    unit: str,
    places: int | None,
    money: MoneyForm | None = None,
    change: bool,
) -> tuple[str, str]:
    """An interval's two ends as a glyph's title and the line under its table write them beside the cell's figure,
    which carries the unit: both at the cell's one precision (``-6.9 to +11.0``, never ``11``), finer only where
    that would print the two alike (:func:`~threetears.evals.kernel.numbers.ends_places`), a change's ends signed;
    a percent's without its ``%`` and money's each with its ``$``; with no places, at the unit's precision.

    :param low: The low end, in ``unit``.
    :ptype low: float
    :param high: The high end, in ``unit``.
    :ptype high: float
    :param unit: The unit the ends are read in: the cell's own, else the one its column's head names.
    :ptype unit: str
    :param places: The decimals the cell's figure reads at; None where its unit's precision spells it.
    :ptype places: int | None
    :param money: Money's form, where the cell's figure is written in one.
    :ptype money: MoneyForm | None
    :param change: Whether the figure is a change, whose ends carry their signs.
    :ptype change: bool
    :return: The two ends' texts.
    :rtype: tuple[str, str]
    """

    def signed(text: str, value: float) -> str:
        # A change's ends carry their signs; a level's do not.
        return plus_signed(text, value) if change else text

    held = None if is_money(unit) or unit == PERCENT else places
    if held is None and unit == POINTS:
        held = POINT_PLACES
    if held is not None and unit != PERCENT:
        at = ends_places(low, high, held)
        return signed(spell_at(low, at), low), signed(spell_at(high, at), high)
    if unit in (PERCENT, "") or is_money(unit):

        def spell(number: float) -> str:
            if unit or places is not None:
                return _amount(number, unit, places, money, change=change)
            return with_unit(number, unit)

        low_text, high_text = spell(low).removesuffix(PERCENT), spell(high).removesuffix(PERCENT)
        return signed(low_text, low), signed(high_text, high)
    return signed(number_in(low, unit), low), signed(number_in(high, unit), high)


def _key_set(figure: Figure) -> tuple[str, str]:
    """The key a figure key's printed part is held under in the ledger: the measure the page's tables hold it under
    where its family names one its path does not (:attr:`~threetears.evals.analysis.report.figure_keys.Figure.
    ledger_as`, an arm's cost per reply under the page's cost), else the measure whose set it reads, by its kind, or
    neither for a figure of no measure's set (a count, an adjusted p). The decimals it prints at read its set alone
    (:meth:`PageForms._key_figure_places`), so where it is held moves no digit."""
    if figure.ledger_as is not None:
        key, kind, _role = figure.ledger_as
        return (key, _measure_kind(kind))
    if figure.set_of is None:
        return ("", "")
    key, kind = figure.set_of
    return (key, _measure_kind(kind))


#: The scales a figure key's money is on.
_MONEY_SCALES = frozenset({"usd", "usd_per_1k", "usd_per_1m"})


def _key_role(figure: Figure) -> Role:
    """What a key's figure is to the measure it is held under: the role the page's tables hold it in where its family
    names one its statistic does not (:attr:`~threetears.evals.analysis.report.figure_keys.Figure.ledger_as`, a
    percentile as a level); else its level or its change where it is a figure the page's tables print of that measure
    (:func:`_tabled`), whether or not its digits read at the measure's set; else a lone figure."""
    if figure.ledger_as is not None:
        return figure.ledger_as[2]
    if figure.set_of is None or not _tabled(figure):
        return "lone"
    if figure.change:
        return "change"
    # Money's level is held as the money it is, as a table's cost cell holds it (:meth:`PageForms.money_on`).
    return "money" if figure.scale in _MONEY_SCALES else "level"


def _key_ruler(figure: Figure, *, per: int, start: int = 0, finer: bool = False) -> MoneyRuler:
    """How a money key's family is stated (#993), by the one owner of money's scale (:func:`money_ruler`): over the
    family's whole set, never the figure alone; a change in the changes' form on the amounts' ruler, capped by its
    interval's width. ``finer``: an interval whose two ends would read alike, one significant figure more, for it
    alone."""
    own = (
        figure.value if isinstance(figure.value, tuple) else (figure.value,) if isinstance(figure.value, float) else ()
    )
    ruler = money_ruler(
        figure.amounts or own, per=per, start=start, rate=not figure.total, reconciles=figure.reconciles
    )
    if finer and ruler.form.figures is not None:
        ruler = replace(ruler, form=replace(ruler.form, figures=ruler.form.figures + 1))
    if not figure.change:
        return ruler
    return ruler.for_changes(own, (figure.spread,) if figure.spread is not None else ())


def _a_rate_cell(cell: TableCell, bare: bool) -> bool:
    """Whether a cell states a rate's figure as a number a table's rate column prints: in percent, in points, or bare
    under a head naming its unit."""
    return cell.is_number and (cell.unit in (PERCENT, POINTS) or (bare and not cell.unit))


def _rate_words(cell: TableCell, bare: bool) -> bool:
    """Whether a cell states a rate's figure in its words, which a table's rate column reads and may respell: a
    percent or a change in points (``[1.0%, 9.0%]``, ``85% (765 of 897)``, ``-7.6 pts``), or any figure under a head
    naming its unit (``[17.2, 25]``)."""
    if cell.is_number or not isinstance(cell.value, str) or not _A_DIGIT.search(cell.value):
        return False
    return bare or PERCENT in cell.value or POINTS in cell.value


#: A digit, which a cell's words must hold to state a figure.
_A_DIGIT = re.compile(r"\d")


# --- which figures share a measure's set: the campaign's ----------------------------------------------------------


#: The key the pass rate's half-widths are held under, whichever page states it: a change's or a choice's readings
#: (``trial``), a measurement's scorecard (its ``passed`` measure) and an aspect page's ``trial`` contrasts alike.
_PASS_RATE = (TRIAL_PASS_NAME, "trial")


def _set_key(name: str, reading: str) -> tuple[str, str]:
    """The key one reading's half-widths are held under across the campaign's pages: its name and reading, the pass
    rate's one key wherever it is stated."""
    return _PASS_RATE if name == TRIAL_PASS_NAME else (name, reading)


def _half_width(interval: tuple[float, float] | None, scale: float = 1.0) -> float | None:
    """An interval's half-width, as stored (``scale`` divides one held in points back into a share); None for none, and
    for one of no width, which says nothing of the precision."""
    if interval is None or interval[1] == interval[0]:
        return None
    return abs(float(interval[1]) - float(interval[0])) / 2 / scale


def _aspect_page(label: str) -> str:
    """An aspect's own page's name, as :func:`campaign_set` reads it."""
    return f"aspect {label}"


#: Where a half-width comes from: ``compared``, an interval on a figure a reader compares (an arm's level, a
#: comparison's change); ``pooled``, one on a figure pooled over the arms' or the comparisons' own (the code page's
#: campaign-wide pass rate and its change); ``derived``, one no page prints, the engine draws from a level's standard
#: error (every page states a judged score's level as its mean and its spread). Only a ``compared`` one sets a
#: measure's digits (owner ruling, set digits): the rest print at the form it sets (:func:`_measure_sets`).
_Provenance = Literal["compared", "pooled", "derived"]

#: What a gathered half-width is an interval on, as every page's gatherer names it: an arm's ``level`` on a measure; a
#: comparison's ``change`` (a contrast, a reading's or a pair's interval on the pass rate, a guardrail's check); a
#: judged score's level (``judged_level``); or a figure ``pooled`` over the arms or the comparisons.
_Evidence = Literal["level", "change", "judged_level", "pooled"]

#: The one place a half-width's provenance is decided (:data:`_Provenance`), by what it is an interval on: the written
#: page's gatherer (:func:`_written_stated`), the code page's (:func:`_readings_stated`) and every aspect page's
#: (:func:`_bundle_stated`) name the evidence, and this decides alike for all three, so one figure counts the same on
#: every page of the campaign.
_PROVENANCE: dict[_Evidence, _Provenance] = {
    "level": "compared",
    "change": "compared",
    "judged_level": "derived",
    "pooled": "pooled",
}

#: One half-width as it is gathered: its reading's key, its width as stored, and what it is an interval on.
_Gathered = tuple[tuple[str, str], float | None, _Evidence]


@dataclass(frozen=True)
class _Width:
    """One half-width a page of the campaign states: the page, its reading's name and kind, its width as stored, and
    where it comes from (:data:`_Provenance`)."""

    page: str
    name: str
    reading: str
    half_width: float
    provenance: _Provenance


def _stated(page: str, widths: Iterable[_Gathered]) -> list[_Width]:
    """Half-widths one page states, by their reading's key, each with its provenance (:data:`_PROVENANCE`); an absent
    one, and one that is no number, left out."""
    return [
        _Width(page, name, reading, float(width), _PROVENANCE[evidence])
        for (name, reading), width, evidence in widths
        if width is not None and math.isfinite(width) and width > 0
    ]


def _reading_widths(
    within: tuple[float, float] | None,
    gain: tuple[float, float] | None,
    drop: tuple[float, float] | None,
    precision: float | None,
) -> list[float | None]:
    """A pass-rate reading's half-widths as shares (it is held in points), on the code page and on the written page
    alike: its nominal interval's (the half-width its precision states, ``precision``, where it holds none), its
    gain's and its drop's."""
    nominal = _half_width(within, 100.0)
    if nominal is None and precision is not None:
        nominal = float(precision) / 100.0
    return [nominal, _half_width(gain, 100.0), _half_width(drop, 100.0)]


def _readings_stated(readings: CampaignReadings) -> list[_Width]:
    """The half-widths a campaign's code page states, as stored, by reading and what each is an interval on
    (:data:`_PROVENANCE` decides each one's provenance): each reading's and each pair's interval on the pass rate's
    change (held in points, so restated as shares, :func:`_reading_widths`), each arm's interval on a measure's level,
    and each arm's interval pooled across the aspects, its level's and its change's."""
    found: list[_Gathered] = []
    for row in readings.readings:
        found += [
            (_PASS_RATE, width, "change")
            for width in _reading_widths(
                row.within_interval,
                row.gain_interval,
                row.drop_interval,
                row.precision.half_width if row.precision is not None else None,
            )
        ]
    found += [(_PASS_RATE, _half_width(pair.interval, 100.0), "change") for pair in readings.pairs]
    found += [
        (_set_key(reading.measure, "measure"), _half_width(reading.interval), "level") for reading in readings.measures
    ]
    if readings.aggregate is not None:
        for arm in readings.aggregate.arms:
            found.append((_PASS_RATE, _half_width(arm.pass_rate.interval, 100.0), "pooled"))
            if arm.change is not None:
                found.append((_PASS_RATE, _half_width(arm.change.interval, 100.0), "pooled"))
    return _stated("code", found)


def _bundle_stated(bundle: AnalysisContextBundle) -> list[_Width]:
    """The half-widths an aspect's page states, as stored, by reading and what each is an interval on
    (:data:`_PROVENANCE` decides each one's provenance): each contrast's and each guardrail check's interval on its
    change, each arm's own interval on a measure's level as its cell holds it, and a judged score's level's, drawn
    from its cell's mean and spread inside the dimension's scale, which the page states as the mean and its spread."""
    surface = bundle_decision_surface(bundle)
    found: list[_Gathered] = [
        (_set_key(comparison.name, comparison.reading), _half_width(comparison.interval), "change")
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
    ]
    for cell in surface.cells:
        for summary in cell.measures.measures:
            if summary.ci_low is not None and summary.ci_high is not None:
                found.append(
                    (_set_key(summary.name, "measure"), _half_width((summary.ci_low, summary.ci_high)), "level")
                )
        for judged in cell.judged:
            facts = surface.dimensions.get(judged.dimension)
            if judged.mean is None or judged.sem is None or facts is None:
                continue
            interval = mean_interval(judged.mean, judged.sem, judged.n_independent, value_range=facts.value_range)
            found.append(((judged.dimension, "judged"), _half_width(interval), "judged_level"))
    if bundle.guardrails is not None:
        found += [
            (_set_key(check.name, check.reading), _half_width(check.interval), "change")
            for check in bundle.guardrails.checks
        ]
    return _stated(_aspect_page(bundle.aspect), found)


def _campaign_stated(
    bundles: Iterable[AnalysisContextBundle], readings: CampaignReadings | None = None
) -> list[_Width]:
    """Every half-width a campaign's aspect pages and its code page state, as stored, each by the page that states it
    and where it comes from."""
    stated = [width for bundle in bundles for width in _bundle_stated(bundle)]
    return stated if readings is None else [*stated, *_readings_stated(readings)]


def _measure_sets(stated: Iterable[_Width]) -> dict[MeasureSet, tuple[float, ...]]:
    """The campaign's one table: every compared half-width its pages state, by the set each joins
    (:func:`campaign_set`), which the written page, the code page and every aspect page read their decimals from
    alike. The one place the owner's set-digits ruling is decided: a pooled or derived half-width joins no set, so it
    never sets a measure's digits, and its figure prints at the form the compared ones set."""
    held: dict[MeasureSet, list[float]] = {}
    for width in stated:
        if width.provenance != "compared":
            continue
        held.setdefault(campaign_set(width.page, *_set_key(width.name, width.reading)), []).append(width.half_width)
    return {key: tuple(found) for key, found in held.items()}


def _set_widths(sets: Mapping[MeasureSet, Sequence[float]], page: str, name: str, reading: str) -> tuple[float, ...]:
    """The half-widths a reading's figures on ``page`` read their places from: every one its set holds in the
    campaign's table (:func:`campaign_set`)."""
    return tuple(sets.get(campaign_set(page, *_set_key(name, reading)), ()))


__all__ = [
    "EVEN_SHARE",
]
