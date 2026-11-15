"""The written page's decisions: how its Full results draw a measure's levels and changes, the half-widths and
figures it states, and each measure's form and texts on it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS
from threetears.evals.analysis.report.keys.reads import held_means, held_stats
from threetears.evals.analysis.report.keys.resolve import RUBRIC_SCALE, figure_unit, is_rubric_score
from threetears.evals.analysis.report.number_forms.intervals import _interval_words
from threetears.evals.analysis.report.number_forms.kinds import _MeasureKind, _ReadsAs
from threetears.evals.analysis.report.number_forms.policy import (
    _ABSENT_TEXT,
    _RATE_BOUNDS,
    _ZERO_BOUND,
    policy_figure,
    policy_interval,
    policy_places,
    policy_rate,
)
from threetears.evals.analysis.report.number_forms.printed import MeasureForm
from threetears.evals.analysis.report.number_forms.sets import (
    MeasureSet,
    _Evidence,
    _Gathered,
    _Width,
    _half_width,
    _measure_kind,
    _reading_widths,
    _set_key,
    _set_widths,
    _stated,
    campaign_set,
)
from threetears.evals.analysis.report.parts.look import Quantity, figure_text, in_change_form, quantity, quantity_of
from threetears.evals.analysis.report.unit_once import unit_spaced
from threetears.evals.analysis.report_input import Contrast, Reading, ReportInput, Stat
from threetears.evals.analysis.report_margins import aspect_board
from threetears.evals.kernel.numbers import format_number


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


def _written_stated(
    report_input: ReportInput, kinds: Iterable[tuple[str, _MeasureKind]]
) -> tuple[list[_Width], dict[MeasureSet, list[float]]]:
    """A written page's own half-widths and its sets' figures (:func:`campaign_set`): the half-width of each interval
    the input mirrors of what the campaign's pages state, as stored, named by what it is an interval on so that
    :data:`_PROVENANCE` decides its provenance as it does on the code and aspect pages (each arm's level on a measure,
    on the page and on each aspect; a judged score's level, which every page states as its mean and spread; each
    comparison's change, the pass rate's readings and pairs among them, held in points, so restated as shares, through
    :func:`_reading_widths` as the code page reads them; each guardrail check's change), and each set's figures as
    printed, which a set with no interval reads its decimals from.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param kinds: Each measure the page states, by its key and kind.
    :ptype kinds: Iterable[tuple[str, _MeasureKind]]
    :return: The half-widths, stated on the written page, and each set's figures.
    :rtype: tuple[list[_Width], dict[MeasureSet, list[float]]]
    """
    found: list[_Gathered] = []
    figures: dict[MeasureSet, list[float]] = {}
    for key, kind in kinds:
        stated: str | None = None if kind == "measure" else kind
        levels = _level_quantity(report_input, key, kind=stated)
        if levels.unit == MONEY:
            # Money keeps its own ruled form, chosen over its amounts (owner ruling), which no half-width reaches.
            continue
        changes = quantity(figure_unit(report_input, key, kind=kind), change=True)
        set_key = _set_key(key, _measure_kind(_set_kind(report_input, key, kind)))
        held_figures = figures.setdefault(campaign_set("written", *set_key), [])
        held = [c for c in report_input.contrasts if (c.measure, _measure_kind(c.kind)) == (key, kind)]
        found += [(set_key, _half_width(c.interval), "change") for c in held]
        held_figures += [float(c.delta) * changes.factor for c in held if c.delta is not None]
        if kind == "trial":
            for aspect in report_input.aspects:
                found += [
                    (set_key, width, "change")
                    for reading in aspect.readings
                    for width in _reading_widths(
                        reading.within_interval,
                        reading.gain_interval,
                        None,
                        _precision_half_width(reading),
                    )
                ]
                found += [(set_key, _half_width(pair.interval, 100.0), "change") for pair in aspect.pairs]
                held_figures += [
                    100.0 * row.passed / row.usable
                    for row in aspect_board(report_input, aspect.id).values()
                    if row.passed is not None and row.usable
                ]
            continue
        table = report_input.judged.get(key) if kind == "judged" else report_input.measures.get(key)
        if table is not None:
            stats = [*table.per_arm.values(), *(stat for arms in table.by_aspect.values() for stat in arms.values())]
            evidence: _Evidence = "judged_level" if kind == "judged" else "level"
            found += [(set_key, _stat_width(stat), evidence) for stat in stats]
        # A level is any statistic the input holds of an arm (its mean, a p50 or a p95 where it states no mean), so a
        # measure with no interval reads at two figures of the least of them (rule 2), never at none.
        held_figures += [
            float(level) * levels.factor
            for stat in held_stats(report_input, key)
            for level in (stat.mean, stat.p50, stat.p95)
            if level is not None
        ]
    found += [
        (guarded, _half_width(_stored_interval(check.get("interval"))), "change")
        for check in report_input.guardrails
        if (guarded := _guardrail_key(check)) is not None
    ]
    return _stated("written", found), figures


def _set_kind(report_input: ReportInput, key: str, kind: _MeasureKind) -> _MeasureKind:
    """The kind a written page's figure joins its set under: a comparison that names no kind, or ``measure``, on a key
    the input holds only as a judged dimension is that dimension's, so one dimension never reads at two precisions."""
    if kind == "measure" and key not in report_input.measures and key in report_input.judged:
        return "judged"
    return kind


def _stat_width(stat: Stat) -> float | None:
    """An arm's level's half-width, as given: the one it states, else its interval's."""
    return float(stat.half_width) if stat.half_width is not None else _half_width(stat.ci)


def _precision_half_width(reading: Reading) -> float | None:
    """The half-width a reading's precision states, in points; None where it states none."""
    if reading.precision is None or reading.precision.half_width is None:
        return None
    return float(reading.precision.half_width)


def _guardrail_key(check: Mapping[str, object]) -> tuple[str, str] | None:
    """The key a stored guardrail check's interval joins its set under, as the aspect page keys it: its measure and
    reading where the pipeline wrote them, else the judged dimension a host's own input names; None for a check that
    names neither, which states no measure's interval."""
    name, reading = check.get("name"), check.get("reading")
    if isinstance(name, str) and isinstance(reading, str):
        return _set_key(name, reading)
    dimension = check.get("dimension")
    return (dimension, "judged") if isinstance(dimension, str) else None


def _stored_interval(value: object) -> tuple[float, float] | None:
    """An interval as the report input stores a guardrail check's (a list of its two ends); None for none."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = value
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return None
    return float(low), float(high)


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


def _written_form(
    report_input: ReportInput,
    key: str,
    kind: _MeasureKind,
    sets: Mapping[MeasureSet, Sequence[float]],
    figures: Mapping[MeasureSet, Sequence[float]],
) -> MeasureForm:
    """One measure's form on a written page: how its figures are drawn, and the decimals its set reads at, read from
    its uncertainty across the campaign (``sets``, as stored, drawn in its unit here; :func:`policy_places`), a rate's
    and its changes' to at most one, else from its figures; a measure's own kind is None there. Money keeps the form
    its ruler chooses."""
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
    places: int | None = None
    narrowest: float | None = None
    if reads_as != "money":
        set_key = _set_key(key, _measure_kind(_set_kind(report_input, key, kind)))
        widths = [width * levels.factor for width in _set_widths(sets, "written", *set_key)]
        narrowest = min(widths) if widths else None
        places = policy_places(widths, figures.get(campaign_set("written", *set_key), ()), capped=reads_as == "rate")
    return MeasureForm(
        key=key,
        kind=kind,
        reads_as=reads_as,
        unit=levels.unit,
        change_unit=changes.unit,
        factor=levels.factor,
        change_factor=changes.factor,
        money=levels.money,
        change_money=changes.money,
        span=_change_span(report_input, key, kind, changes),
        levels=levels,
        changes=changes,
        scale=_RUBRIC_SCALE_TEXT if reads_as == "score" else "",
        unnamed_span=_change_span(report_input, key, stated, changes),
        places=places,
        half_width=narrowest,
    )


#: The 1–5 rubric as a score's name carries it (``Outcome (1–5)``), so its two decimals read on a scale a reader knows.
_RUBRIC_SCALE_TEXT = f"{format_number(RUBRIC_SCALE[0])}–{format_number(RUBRIC_SCALE[1])}"


def _drawn(form: MeasureForm, *, change: bool) -> Quantity:
    drawn = form.changes if change else form.levels
    if drawn is None:
        raise ValueError(f"the measure {form.key!r} holds no written page's quantity")
    return drawn


def _written_level(form: MeasureForm, value: float) -> str:
    """A level as every surface of a written page states it, at its measure's decimals: a rate kept off its floor and
    its ceiling (:func:`policy_rate`), any other figure in its unit (:func:`policy_figure`), an exact zero whole
    (``0 notices``); money, and a figure that is no number, by its quantity's note rule
    (:func:`~threetears.evals.analysis.report.parts.look.figure_text`)."""
    drawn = _drawn(form, change=False)
    scaled = value * drawn.factor
    if form.places is None:
        return figure_text(scaled, drawn)
    if not math.isfinite(scaled):
        # A figure that is no number reads as the dash in its unit (``— items``), as the number rule spells it.
        return unit_spaced(_ABSENT_TEXT, drawn.unit)
    if form.reads_as == "rate":
        return policy_rate(scaled, form.places)
    return unit_spaced(policy_figure(scaled, form.places, bounds=()), drawn.unit)


def _written_change(form: MeasureForm, value: float) -> str:
    """A change as every surface of a written page states it, signed at its measure's decimals: a change in points
    kept off zero (``-0.03 pts``, never ``-0.0 pts``), any other at its measure's decimals alone; money, and a figure
    that is no number, by its quantity's note rule."""
    drawn = _drawn(form, change=True)
    scaled = value * drawn.factor
    if form.places is None:
        return figure_text(scaled, drawn)
    if not math.isfinite(scaled):
        return unit_spaced(_ABSENT_TEXT, drawn.unit)
    bounds = _ZERO_BOUND if drawn.unit == POINTS else ()
    return unit_spaced(policy_figure(scaled, form.places, bounds=bounds, signed=True), drawn.unit)


def _written_interval(
    form: MeasureForm, low: float, high: float, *, change: bool, span: float | None
) -> tuple[str, str, str]:
    """An interval as every surface of a written page states it, in its one form (:func:`policy_interval`): ``low to
    high`` at its measure's decimals, the unit once (``-13 to -2 pts``), a change's ends signed and kept within how far
    the measure can move, each end kept off zero, and a rate level's off its floor and its ceiling. Money's, and an
    interval with an end that is no number, by its quantity's note rule. Each end's text is as the interval prints it,
    the unit once, after the high end."""
    drawn = _drawn(form, change=change)
    scaled = (low * drawn.factor, high * drawn.factor)
    if form.places is None or not all(math.isfinite(end) for end in scaled):
        return figure_text(scaled[0], drawn), figure_text(scaled[1], drawn), _interval_words(*scaled, drawn)
    if change and span is not None:
        # A change never reaches past what its measure can move: a score's by its scale's width, a rate's by 100 points.
        scaled = (max(scaled[0], -span), min(scaled[1], span))
    bounds = _RATE_BOUNDS if form.reads_as == "rate" and not change else ()
    low_digits, high_digits, text = policy_interval(
        scaled[0], scaled[1], form.places, change=change, bounds=bounds, unit=drawn.unit
    )
    return low_digits, unit_spaced(high_digits, drawn.unit), text


def _placed_text(value: float, drawn: Quantity, places: int, *, change: bool) -> str:
    """A chart's drawn figure of a measure at its measure's decimals, as the page's tables print it: a change signed, in
    points kept off zero; a rate's level kept off its floor and its ceiling; any other in its unit."""
    if not math.isfinite(value):
        return figure_text(value, drawn)
    if change:
        bounds = _ZERO_BOUND if drawn.unit == POINTS else ()
        return unit_spaced(policy_figure(value, places, bounds=bounds, signed=True), drawn.unit)
    if drawn.unit == PERCENT:
        return policy_rate(value, places)
    return unit_spaced(policy_figure(value, places, bounds=()), drawn.unit)


__all__: list[str] = []
