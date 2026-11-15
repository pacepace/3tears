"""Separation: an arm's spread, two arms' difference, and the cases that would settle a reading against a
criterion, a hypothetical line or another arm.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from threetears.evals.analysis.cell_mean import NoCellMean, cell_mean_interval
from threetears.evals.analysis.report.keys.figure import Figure, Scale
from threetears.evals.analysis.report.keys.prices import _apart
from threetears.evals.analysis.report.keys.reads import _cell_spelling, _reading_arms, _scale_of
from threetears.evals.analysis.report.keys.vocabulary import (
    _asked,
    _control,
    _criterion,
    _interval,
    _level,
    _number,
    _refuse,
    _unstated,
    on_lead_aspect,
)
from threetears.evals.analysis.report_input import Contrast, Criterion, ReportInput, Stat
from threetears.evals.analysis.report_margins import aspect_board, aspect_of, criterion_figures, criterion_other_aspect
from threetears.evals.analysis.separation import (
    POWER,
    NoSpread,
    Spread,
    cases_to_separate,
    independent_difference,
    level_share,
    spread_interval,
    stat_spread,
    wrong_side_share,
)
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, MATCH_MEASURE


# --- the families --------------------------------------------------------------------------------------------------


def _arms(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [arm.id for arm in report_input.arms]


def _aspects(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [aspect.id for aspect in report_input.aspects]


def _asked_reading_arms(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return _reading_arms(_asked(report_input))


def _named_reading_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _reading_arms(aspect_of(report_input, prior[0]))


# --- separation: an arm's spread, two arms' difference, the cases that would settle a reading ------------------------
#
# ``spread.*``, ``compare.*`` and ``settle.*`` read a figure's spread and what it says (:mod:`~threetears.evals.analysis
# .separation` holds the arithmetic and its design note). Each names where it reads with one ``<where>`` slot:
# ``overall`` (the arm's own figure), ``task.<task>``, ``cells.<row>.<column>`` or ``strata.<stratum>`` (a stratum whose
# id the input holds once). Every figure is derived: computed here, read off the figures' stated spreads (a normal
# approximation), never a test the pipeline ran.

#: Where a separation key reads an arm's own figure, rather than one within a slice.
_OVERALL = "overall"

#: What an arm's spread on a figure reads: its mean, its standard error, the spread of one case, the count it rests on
#: and what that count counts (cases, trials or observations), the distinct cases behind a count of trials and the
#: trials each ran today, the level a result read from it is stated at, what the standard error was read from, and the
#: interval it gives at that level and its two ends (the mean plus and minus the normal quantile times the standard
#: error, a rate's ends held to none to all: :func:`~threetears.evals.analysis.separation.spread_interval`).
_SPREAD_FIELDS = (
    "mean",
    "se",
    "per_case",
    "n",
    "counts",
    "cases",
    "repeats",
    "level",
    "basis",
    "interval",
    "low",
    "high",
)

#: What two arms' difference reads: the first minus the second, its interval, the interval's ends and half-width, its
#: level, and whether it is the paired estimate a contrast states or two independent arms' spreads combined.
_COMPARE_FIELDS = ("difference", "interval", "low", "high", "half_width", "level", "basis")

#: What settling two arms apart reads: their difference, the cases each needs for an interval centred on it to clear
#: zero (``cases_needed``: half such designs clear; ``cases_needed_80``: four in five) and the trials those take at
#: today's repeats, how often a fresh run of today's size reads the other arm ahead, and the level, its sidedness
#: (two-sided) in its words.
_SETTLE_PAIR_FIELDS = (
    "difference",
    "cases_needed",
    "cases_needed_80",
    "trials_needed",
    "trials_needed_80",
    "wrong_side",
    "level",
)

#: What settling an arm against a line reads: its distance from the line toward passing (negative where today's figure
#: misses), the cases a one-sided bound needs to sit wholly on today's side of the line, the cases the two-sided
#: interval a page prints needs for the same, the trials each size takes at today's repeats, how often a fresh run of
#: today's size lands on the other side, and the level, its sidedness (one-sided) in its words.
_SETTLE_LINE_FIELDS = (
    "distance",
    "cases_needed",
    "cases_needed_80",
    "cases_needed_two_sided",
    "cases_needed_two_sided_80",
    "trials_needed",
    "trials_needed_80",
    "trials_needed_two_sided",
    "trials_needed_two_sided_80",
    "wrong_side",
    "level",
)

#: A hypothetical line's fields: the line itself, then what settling an arm against it reads.
_HYPOTHETICAL_FIELDS = ("line", *_SETTLE_LINE_FIELDS)

#: A margin's or floor's fields: its line (minus the margin, in points), the candidate's change from the control it reads
#: (in points), then what settling that change against the line reads.
_MARGIN_FIELDS = ("line", "change", *_SETTLE_LINE_FIELDS)

#: How each basis of a spread reads, in words.
_BASIS_WORDS: dict[str, str] = {
    "interval": "its stated interval",
    "half_width": "its stated half-width",
    "sem": "its standard error",
    "binomial": "a pass/fail rate's binomial spread, from its count",
    "cells": (
        "the standard error derived from its target-matrix cells, the figure being their plain mean: the root of "
        "the cells' squared standard errors over their count, the cells read as independent, at the cells' stated "
        "level; a size read from it counts cases at today's mix of cells"
    ),
}

#: A cell-derived spread's basis where a case-weighted mean of the cells fits the figure as well as the plain one.
_CELLS_BOTH_WORDS = (
    "the standard error derived from its target-matrix cells, the figure fitting both their plain and their "
    "case-weighted mean: the wider of the two readings', the cells read as independent, at the cells' stated level; "
    "a size read from it counts cases at today's mix of cells"
)


@dataclass(frozen=True)
class _Place:
    """One slice a separation key reads within: how ``<where>`` spells it, each arm's figures there, and its cases."""

    where: str
    per_arm: Mapping[str, Mapping[str, Stat]]
    n_cases: int | None


def _separation_places(report_input: ReportInput) -> list[_Place]:
    """Every slice a separation key can read within: each task, each cell its spelling names alone, and each stratum
    whose id the input holds once."""
    spellings = [_cell_spelling(cell) for cell in report_input.cells.values()]
    ids = [stratum.id for stratum in report_input.strata]
    return [
        *(_Place(f"task.{task.id}", task.per_arm, task.n_cases) for task in report_input.tasks),
        *(
            _Place(f"cells.{spelled}", cell.per_arm, cell.n_cases)
            for cell, spelled in zip(report_input.cells.values(), spellings, strict=True)
            if spellings.count(spelled) == 1
        ),
        *(
            _Place(f"strata.{stratum.id}", stratum.per_arm, stratum.n_cases)
            for stratum in report_input.strata
            if ids.count(stratum.id) == 1
        ),
    ]


#: What a count a spread rests on counts: distinct cases, trials (a case run again counts again), or observations (a
#: statistic's own ``n``, where nothing says whether it counts cases or trials).
_Counted = Literal["cases", "trials", "observations"]


def _separation_place(report_input: ReportInput, where: str) -> _Place | None:
    """The one slice a separation key's ``<where>`` names (:func:`_separation_places`), found by its kind and id alone,
    since every key reads it once per arm and place; None where it names none, or a cell or stratum held twice."""
    kind, _dot, named = where.partition(".")
    found: list[_Place] = []
    if kind == "task":
        found = [_Place(where, task.per_arm, task.n_cases) for task in report_input.tasks if task.id == named]
    elif kind == "cells":
        cells = report_input.cells.values()
        found = [_Place(where, cell.per_arm, cell.n_cases) for cell in cells if _cell_spelling(cell) == named]
    elif kind == "strata":
        strata = report_input.strata
        found = [_Place(where, stratum.per_arm, stratum.n_cases) for stratum in strata if stratum.id == named]
    return found[0] if len(found) == 1 else None


@dataclass(frozen=True)
class _Held:
    """An arm's statistic at one place, and what its spread's count is.

    :ivar stat: The statistic.
    :ivar count: The count its spread rests on; None where none is stated.
    :ivar counted: What that count counts.
    :ivar cases: The distinct cases behind a count of trials, where stated and fewer; None otherwise.
    :ivar source: Where the count was read, in words.
    """

    stat: Stat
    count: float | None
    counted: _Counted | None
    cases: float | None
    source: str


#: Where a spread's count was read, in words: the figure's own, the slice's stated cases, or the arm's scoreboard.
_COUNT_SOURCES: dict[str, str] = {
    "own": "its own count",
    "task": "the task's stated cases",
    "cells": "the cell's stated cases",
    "strata": "the stratum's stated cases",
    "trials": "the arm's usable trials",
    "cases": "the arm's cases",
}


def _held_where(
    report_input: ReportInput, arm_id: str, key: str, where: str, aspect: str | None = None
) -> _Held | None:
    """An arm's statistic on a measure or judged dimension at ``where``, with the count it rests on, what that count
    counts and where it was read: its own ``n`` (observations; cases where its ``n_cases`` equals it, trials where its
    ``n_cases`` is fewer), else its ``n_cases`` (cases), else the place's: the slice's cases, or overall the arm's
    usable trials, else its cases. A count of trials carries the distinct cases behind it where the figure, or overall
    the arm's scoreboard, states them. Overall, ``aspect`` names an aspect other than the lead whose figures
    (``by_aspect``) and scoreboard are read instead (a criterion declared on it); a slice holds no other aspect's
    figures. None where the input states no mean there."""
    fallback: tuple[int | None, _Counted | None, int | None, str] = (None, None, None, "")
    if aspect is not None and where != _OVERALL:
        return None
    if where == _OVERALL:
        table = report_input.measures.get(key) or report_input.judged.get(key)
        per_arm = None if table is None else table.per_arm if aspect is None else table.by_aspect.get(aspect, {})
        stat = per_arm.get(arm_id) if per_arm is not None else None
        held = {candidate.id for candidate in report_input.aspects}
        board = (aspect_board(report_input, aspect) if aspect is None or aspect in held else {}).get(arm_id)
        if board is not None and board.usable is not None:
            fallback = (board.usable, "trials", board.cases, _COUNT_SOURCES["trials"])
        elif board is not None:
            fallback = (board.cases, "cases", None, _COUNT_SOURCES["cases"])
    else:
        place = _separation_place(report_input, where)
        stat = place.per_arm.get(arm_id, {}).get(key) if place is not None else None
        if place is not None:
            fallback = (place.n_cases, "cases", None, _COUNT_SOURCES[where.partition(".")[0]])
    if stat is None or stat.mean is None:
        return None
    count: int | None
    counted: _Counted | None
    cases: int | None = None
    source = _COUNT_SOURCES["own"]
    if stat.n is not None:
        count = stat.n
        counted = "observations" if stat.n_cases is None else "cases" if stat.n <= stat.n_cases else "trials"
        cases = stat.n_cases if counted == "trials" else None
    elif stat.n_cases is not None:
        count, counted = stat.n_cases, "cases"
    else:
        count, counted, cases, source = fallback
    if count is None:
        return _Held(stat, None, None, None, "")
    distinct = float(cases) if cases is not None and 0 < cases < count else None
    return _Held(stat, float(count), counted, distinct, source)


def _stat_where(
    report_input: ReportInput, arm_id: str, key: str, where: str, aspect: str | None = None
) -> tuple[Stat, float | None, _Counted | None] | None:
    """An arm's statistic at ``where`` (on another aspect, where ``aspect`` names one), the count it rests on and what
    that count counts (:func:`_held_where`)."""
    held = _held_where(report_input, arm_id, key, where, aspect)
    return None if held is None else (held.stat, held.count, held.counted)


def _wheres(report_input: ReportInput, arms: Sequence[str], key: str) -> list[str]:
    """Where every one of ``arms`` holds a mean on ``key``: overall first, then each slice in the input's order."""
    candidates = [_OVERALL, *(place.where for place in _separation_places(report_input))]
    return [where for where in candidates if all(_stat_where(report_input, arm, key, where) for arm in arms)]


def _separated_keys(report_input: ReportInput, arms: Sequence[str]) -> list[str]:
    """The measures and judged dimensions every one of ``arms`` holds a mean on somewhere."""
    keys = dict.fromkeys([*report_input.measures, *report_input.judged])
    return [key for key in keys if _wheres(report_input, arms, key)]


#: The measures graded by class label whose figure is a share of counted trials: the ones the report input's build grades
#: ``class_label`` (``report.input_build``'s grade rule). A figure graded so by another name may be a derived mean of
#: shares (a balanced accuracy: the mean of each class's recall), which no binomial spread describes.
_TRIAL_SHARE_LABELS = frozenset({MATCH_MEASURE, ACCURACY_MEASURE})


def _rate_of(report_input: ReportInput, key: str) -> Literal["share", "percent"] | None:
    """The scale a figure that is a share of counted trials is on, which a binomial spread reads: a measure graded
    pass/fail (each observation a pass or a fail, the figure the share that passed), one graded by class label that the
    build's grade rule names a share of trials (:data:`_TRIAL_SHARE_LABELS`), a pass/fail judged dimension. None for any
    other figure: a derived mean of shares graded by class label under another name reads no binomial spread."""
    measure = report_input.measures.get(key)
    judged = report_input.judged.get(key)
    trial_share = measure is not None and (
        measure.grade == "pass_fail" or (measure.grade == "class_label" and key in _TRIAL_SHARE_LABELS)
    )
    graded = trial_share or (judged is not None and judged.scale == "pass_fail")
    scale, _unit = _scale_of(report_input, key)
    if graded and scale in ("share", "percent"):
        return "share" if scale == "share" else "percent"
    return None


def _stated_level(report_input: ReportInput, key: str) -> float | None:
    """The level a measure's or judged dimension's stated intervals are at, as a share; None where it states none."""
    table = report_input.measures.get(key) or report_input.judged.get(key)
    return None if table is None else level_share(table.level)


def _where_words(where: str) -> str:
    return "overall" if where == _OVERALL else f"in {where}"


def _read_spread(report_input: ReportInput, arm_id: str, key: str, where: str, aspect: str | None = None) -> Spread:
    """An arm's spread on a figure at ``where`` (:func:`~threetears.evals.analysis.separation.stat_spread`), on another
    aspect's figure where ``aspect`` names one (:func:`_stat_where`).

    :raises NoSpread: The input gives none; its words are what a refusal says after the key.
    """
    held = _held_where(report_input, arm_id, key, where, aspect)
    if held is None:
        raise NoSpread(f"names {arm_id}'s mean on {key!r} {_where_words(where)}, which the input does not state")
    stat = held.stat
    if where == _OVERALL and aspect is None and stat.ci is None and stat.half_width is None and stat.sem is None:
        cells = _cells_spread(report_input, arm_id, key, held)
        if cells is not None:
            return cells
    try:
        return stat_spread(
            float(stat.mean) if stat.mean is not None else None,
            ci=None if stat.ci is None else (float(stat.ci[0]), float(stat.ci[1])),
            half_width=None if stat.half_width is None else float(stat.half_width),
            sem=None if stat.sem is None else float(stat.sem),
            n=held.count,
            stated_level=_stated_level(report_input, key),
            rate=_rate_of(report_input, key),
            cases=held.cases,
        )
    except NoSpread as none:
        raise NoSpread(
            f"reads {arm_id}'s spread on {key!r} {_where_words(where)}, and the figure has none: {none}"
        ) from none


def _weighted_too(report_input: ReportInput, key: str, arm_id: str) -> bool:
    """Whether a cell-derived spread's figure fits its cells' case-weighted mean as well as their plain one."""
    try:
        return cell_mean_interval(report_input, key, arm_id).weighted_too
    except NoCellMean:
        # NOSILENT: read only for a spread whose basis is the cells', which qualified; an overall that does not is
        # no weighted reading, which False says.
        return False


def _cells_spread(report_input: ReportInput, arm_id: str, key: str, held: _Held) -> Spread | None:
    """An arm's overall spread read off its target-matrix cells, where the overall states none of its own and is the
    plain mean of its cells (:func:`~threetears.evals.analysis.cell_mean.cell_mean_interval`, the interval a gate on it
    reads): its standard error the cells', combined; one case's spread that times the root of the overall's count. None
    where the overall does not qualify, so its spread is read as before."""
    try:
        derived = cell_mean_interval(report_input, key, arm_id)
    except NoCellMean:
        # NOSILENT: an overall that is no plain mean of its cells is the input's shape, not a fault: its spread is
        # read from what it states, as before, which is what None tells the caller.
        return None
    count, distinct = held.count, held.cases
    per_case = derived.se * math.sqrt(distinct or count) if count is not None else None
    return Spread(derived.mean, derived.se, count, per_case, derived.level, "cells", distinct)


def _spread(
    report_input: ReportInput, arm_id: str, key: str, where: str, path: str, aspect: str | None = None
) -> Spread:
    """An arm's spread on a figure at ``where`` (on another aspect's, where ``aspect`` names one), refused as unstated,
    with why, where the input gives none."""
    try:
        return _read_spread(report_input, arm_id, key, where, aspect)
    except NoSpread as none:
        raise _unstated(path, str(none)) from none


def _settled(path: str, settle: Callable[[], float]) -> float:
    """A settle computation's number, refused as unstated, with why, where the arithmetic has none."""
    try:
        return settle()
    except NoSpread as none:
        raise _unstated(path, f"has no figure: {none}") from none


def _spread_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, key, where, field = values
    spread = _spread(report_input, arm_id, key, where, path)
    if field == "mean":
        scale, unit = _scale_of(report_input, key)
        return _number(path, spread.mean, "the arm's mean", scale, unit=unit)
    if field in ("se", "per_case"):
        value = spread.se if field == "se" else spread.per_case
        if value is None:
            raise _unstated(path, "names the spread of one case, which needs the count the figure rests on")
        return _apart(report_input, path, key, value, f"the figure's {field}", change=False)
    if field == "n":
        return _number(path, spread.n, "the count the figure's spread rests on", "count")
    if field == "counts":
        held = _stat_where(report_input, arm_id, key, where)
        counted = None if held is None else held[2]
        if counted is None:
            raise _unstated(path, "names what the figure's count counts, and it rests on no count")
        return Figure(path=path, kind="text", value=counted)
    if field == "cases":
        return _number(path, spread.cases, "the distinct cases behind a count of trials", "count")
    if field == "repeats":
        return _number(path, spread.repeats, "the trials each case ran, where the count is trials", "plain")
    if field == "level":
        return _level(path, spread.level, "the level a result read from it is stated at")
    if field in ("interval", "low", "high"):
        try:
            low, high = spread_interval(spread, rate=_rate_of(report_input, key))
        except NoSpread as none:
            raise _unstated(path, str(none)) from none
        scale, unit = _scale_of(report_input, key)
        if field == "interval":
            return _interval(path, (low, high), "the interval", scale, unit=unit)
        return _number(path, low if field == "low" else high, f"the interval's {field} end", scale, unit=unit)
    return Figure(path=path, kind="text", value=_basis_words(report_input, arm_id, key, where, spread))


def _basis_words(
    report_input: ReportInput, arm_id: str, key: str, where: str, spread: Spread, aspect: str | None = None
) -> str:
    """What a spread was read from, in words; a binomial one says where its count was read (its own, the slice's
    stated cases, or the arm's scoreboard), on another aspect's figures where ``aspect`` names one."""
    if spread.basis == "cells" and _weighted_too(report_input, key, arm_id):
        return _CELLS_BOTH_WORDS
    if spread.basis != "binomial":
        return _BASIS_WORDS[spread.basis]
    held = _held_where(report_input, arm_id, key, where, aspect)
    source = held.source if held is not None and held.source else _COUNT_SOURCES["own"]
    words = f"a pass/fail rate's binomial spread, from {source}"
    if spread.cases is None:
        return words
    return f"{words}, read on its distinct cases, each counted once: {_REPEATS_BOUND}"


#: What a binomial spread on a repeated design is, as its basis says it: the wide end of what its repeats allow.
_REPEATS_BOUND = (
    "the widest spread its repeats allow (repeats read as independent would read it narrower by the root of the "
    "repeats), so an interval or a size read from it is never narrower than the cases support"
)


def _family_level(report_input: ReportInput, contrast: Contrast) -> float | None:
    """A contrast's interval's level: its own, else the one family correcting its aspect's."""
    own = level_share(contrast.level)
    if own is not None:
        return own
    families = [family for family in report_input.contrast_families if family.aspect == contrast.aspect]
    return level_share(families[0].level) if len(families) == 1 else None


def correction_words(report_input: ReportInput, contrast: Contrast) -> str:
    """Whether a contrast's interval is corrected, as the one family on its aspect states it (``correction``), never
    asserted: corrected by the method it names, not corrected where it names ``none``, and unstated where it names none
    or no one family is on the contrast's aspect. A ``compare.*.basis`` key and a compare block's words say it alike.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param contrast: The contrast.
    :ptype contrast: Contrast
    :return: The words, after the contrast's "its own interval, ".
    :rtype: str
    """
    families = [family for family in report_input.contrast_families if family.aspect == contrast.aspect]
    if len(families) != 1:
        return "and no one contrast family on its aspect says whether it is corrected"
    correction = (families[0].correction or "").strip()
    if not correction:
        return "and its family states no correction, so whether it is corrected is unstated"
    if correction.lower() == "none":
        return "not corrected across its family (the family's correction is none)"
    return f"corrected within its family ({correction})"


def _paired(report_input: ReportInput, arm_id: str, other: str, key: str) -> Contrast | None:
    """The one contrast stating one of the two arms against the other (the control) on ``key``, with its difference
    and interval, on the lead aspect whose figures ``overall`` reads (:func:`on_lead_aspect`): the paired estimate, which the
    compare family reads overall in place of the arms' own spreads."""
    control = _control(report_input)
    if control is None or control.id not in (arm_id, other):
        return None
    candidate = other if arm_id == control.id else arm_id
    held = [
        contrast
        for contrast in report_input.contrasts
        if contrast.arm == candidate
        and contrast.measure == key
        and on_lead_aspect(report_input, contrast)
        and contrast.control in (None, control.id)
        and contrast.delta is not None
        and contrast.interval is not None
    ]
    return held[0] if len(held) == 1 else None


def _both_bases(
    report_input: ReportInput, arm_id: str, other: str, key: str, where: str, first: Spread, second: Spread
) -> str:
    """What two arms' spreads were read from, in words: once where both read alike, else each, the first's first."""
    one = _basis_words(report_input, arm_id, key, where, first)
    two = _basis_words(report_input, other, key, where, second)
    return one if one == two else f"{one}; {two}"


def pairing_contrast(report_input: ReportInput, arm_id: str, other: str, key: str, where: str) -> Contrast | None:
    """The contrast a ``compare.<arm>.<other>.<key>.<where>`` key reads as its paired estimate: overall, the one
    contrast stating one of the two arms against the other (the control) on the lead aspect, with its difference and
    interval; None where the compare family reads the two arms' own spreads as independent.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm_id: The first arm.
    :ptype arm_id: str
    :param other: The second.
    :ptype other: str
    :param key: The measure or judged dimension.
    :ptype key: str
    :param where: The place, as a separation key spells it.
    :ptype where: str
    :return: The contrast; None for none.
    :rtype: Contrast | None
    """
    return _paired(report_input, arm_id, other, key) if where == _OVERALL else None


@dataclass(frozen=True)
class _Difference:
    """One arm minus another on a figure at one place, with its interval: the compare family's reading, which the lever
    family reads too, so the two never disagree on whether an interval exists or what it is.

    :ivar difference: The first minus the second, in the figure's own units: a pairing contrast's difference where one
        pairs them, else the two means'.
    :ivar low: The interval's low end, in the same units.
    :ivar high: Its high end.
    :ivar level: Its level, as a share; None where a contrast's interval states none.
    :ivar basis: Paired or independent, and from what, in words.
    """

    difference: float
    low: float
    high: float
    level: float | None
    basis: str


def _mean_difference(report_input: ReportInput, arm_id: str, other: str, key: str, where: str) -> float | None:
    """The first arm's mean minus the second's at ``where``; None where either states none."""
    means = [_stat_where(report_input, arm, key, where) for arm in (arm_id, other)]
    if means[0] is None or means[1] is None or means[0][0].mean is None or means[1][0].mean is None:
        return None
    return float(means[0][0].mean) - float(means[1][0].mean)


def _read_difference(report_input: ReportInput, arm_id: str, other: str, key: str, where: str) -> _Difference:
    """The first arm minus the second on a figure at ``where``, with its interval: overall, the contrast pairing them
    where one does (:func:`_paired`), else the two arms' own spreads combined as independent
    (:func:`~threetears.evals.analysis.separation.independent_difference`).

    :raises NoSpread: Neither a pairing contrast nor both arms' spreads give an interval; its words say why.
    """
    contrast = pairing_contrast(report_input, arm_id, other, key, where)
    if contrast is not None and contrast.delta is not None and contrast.interval is not None:
        sign = 1.0 if contrast.arm == arm_id else -1.0
        low, high = sorted((sign * float(contrast.interval[0]), sign * float(contrast.interval[1])))
        return _Difference(
            difference=sign * float(contrast.delta),
            low=low,
            high=high,
            level=_family_level(report_input, contrast),
            basis=(
                f"paired: the contrast of {contrast.arm} against the control on {key!r}, its own interval, "
                f"{correction_words(report_input, contrast)}"
            ),
        )
    first = _read_spread(report_input, arm_id, key, where)
    second = _read_spread(report_input, other, key, where)
    difference, half = independent_difference(first, second, first.level)
    return _Difference(
        difference=difference,
        low=difference - half,
        high=difference + half,
        level=first.level,
        basis=(
            f"independent: each arm's own spread ({_both_bases(report_input, arm_id, other, key, where, first, second)}), "
            "combined as if the arms ran different cases, which reads wider than a paired interval where they ran the "
            "same"
        ),
    )


def _compare_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, other, key, where, field = values
    if arm_id == other:
        raise _refuse(path, "compares an arm with itself; name two arms")
    contrast = pairing_contrast(report_input, arm_id, other, key, where)
    if field == "difference" and contrast is None:
        difference = _mean_difference(report_input, arm_id, other, key, where)
        if difference is None:
            raise _unstated(path, f"names the two arms' means on {key!r} {_where_words(where)}, which one lacks")
        return _apart(report_input, path, key, difference, "the two arms' difference", change=True)
    try:
        read = _read_difference(report_input, arm_id, other, key, where)
    except NoSpread as none:
        raise _unstated(path, str(none)) from none
    difference, low, high, level, basis = read.difference, read.low, read.high, read.level, read.basis
    if field == "basis":
        return Figure(path=path, kind="text", value=basis)
    if field == "level":
        if level is None:
            raise _unstated(path, "names the level of the contrast's interval, which the input does not state")
        return _level(path, level, "the interval's level")
    if field == "interval":
        scale, unit = _scale_of(report_input, key)
        factor = 100.0 if scale == "share" else 1.0
        shown: Scale = "points" if scale in ("share", "percent") else scale
        return Figure(
            path=path,
            kind="interval",
            value=(low * factor, high * factor),
            scale=shown,
            unit="" if shown == "points" else unit,
            change=True,
        )
    value = {"difference": difference, "low": low, "high": high, "half_width": (high - low) / 2}[field]
    return _apart(report_input, path, key, value, f"the difference's {field}", change=field != "half_width")


def _settle_pair_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, other, key, where, field = values
    if arm_id == other:
        raise _refuse(path, "settles an arm against itself; name two arms")
    first = _spread(report_input, arm_id, key, where, path)
    second = _spread(report_input, other, key, where, path)
    difference = first.mean - second.mean
    if field == "difference":
        return _apart(report_input, path, key, difference, "the two arms' difference", change=True)
    if field == "level":
        return _sided_level(path, first.level, 2)
    if field == "wrong_side":
        share = _settled(path, lambda: wrong_side_share(difference, math.hypot(first.se, second.se)))
        return _number(path, share, "how often a fresh run of today's size reads the other arm ahead", "share")
    if first.per_case is None or second.per_case is None:
        raise _unstated(path, "names the cases a design needs, which needs the count each arm's figure rests on")
    per_case = math.hypot(first.per_case, second.per_case)
    power = POWER if field.endswith("_80") else 0.5
    cases = _settled(path, lambda: cases_to_separate(per_case, difference, first.level, sided=2, power=power))
    if field.startswith("trials_"):
        if first.repeats != second.repeats:
            raise _unstated(
                path,
                "names the trials each arm needs, and the two arms ran their cases a different number of times: "
                "read each arm's repeats (`spread.<arm>.<measure>.<where>.repeats`)",
            )
        return _trials(path, cases, first)
    return _number(path, cases, "the cases each arm needs", "count")


def _trials(path: str, cases: float, spread: Spread) -> Figure:
    """The trials a size in cases takes, each case run as often as today's design runs one."""
    if spread.repeats is None:
        raise _unstated(
            path,
            "names the trials a size takes, and the figure states no count of trials over fewer distinct cases: its "
            "size counts what its count counts (`spread.<arm>.<measure>.<where>.counts`)",
        )
    return _number(path, math.ceil(cases * spread.repeats - 1e-9), "the trials the cases take", "count")


def _sided_level(path: str, level: float, sided: Literal[1, 2]) -> Figure:
    """A settle figure's level, said with its sidedness (``95% one-sided``), so no citation of it reads as the
    two-sided intervals a page prints when it is not one: the level a number the page's forms print, as every level."""
    read = _level(path, level, "the level the design is sized at")
    return replace(read, said_between=("", " one-sided" if sided == 1 else " two-sided"))


def _settle_line(
    report_input: ReportInput,
    path: str,
    spread: Spread,
    key: str,
    line: float,
    higher_is_better: bool,
    field: str,
    *,
    in_points: bool = False,
) -> Figure:
    """What settling an arm's figure against a line reads, at ``field``: its distance toward passing, and the sizes at
    which an interval centred on it sits wholly on today's side of the line (clearing it where today's figure passes,
    confirming the miss where it fails), one-sided and, beside it, two-sided as a page prints its intervals; each size
    in cases, and the trials it takes at today's repeats. ``in_points``: the figure is a change already in points (a
    reading against a margin or a floor), not one on ``key``'s scale."""
    distance = spread.mean - line if higher_is_better else line - spread.mean
    if field == "distance" and in_points:
        return _number(path, distance, "the figure's distance from the line", "points", change=True)
    if field == "distance":
        # A distance is a change on ``key``, which a criterion's path names only by the criterion: it carries the measure
        # so it reads at that measure's changes' set, as the criterion's own gap does.
        apart = _apart(report_input, path, key, distance, "the figure's distance from the line", change=True)
        return replace(apart, measure=key)
    if field == "level":
        return _sided_level(path, spread.level, 1)
    if field == "wrong_side":
        share = _settled(path, lambda: wrong_side_share(distance, spread.se))
        return _number(path, share, "how often a fresh run of today's size lands on the other side", "share")
    per_case = spread.per_case
    if per_case is None:
        raise _unstated(path, "names the cases a design needs, which needs the count the figure rests on")
    power = POWER if field.endswith("_80") else 0.5
    sided: Literal[1, 2] = 2 if "two_sided" in field else 1
    cases = _settled(path, lambda: cases_to_separate(per_case, distance, spread.level, sided=sided, power=power))
    if field.startswith("trials_"):
        return _trials(path, cases, spread)
    return _number(path, cases, "the cases a design needs", "count")


#: What a criterion's settle reads beside its sizes: the count the arm's spread rests on, what that count counts, and
#: what the spread was read from, each on the criterion's own aspect (the ``spread.`` key's, on the lead aspect).
_SETTLE_COUNT_FIELDS = ("n", "counts", "cases", "repeats", "basis")


def _criterion_count(
    report_input: ReportInput, path: str, spread: Spread, at: tuple[str, str, str, str | None], field: str
) -> Figure:
    """A criterion settle's ``n``, ``counts``, ``repeats`` or ``basis``: what ``spread.<arm>.<measure>.<where>``
    reads, on the criterion's aspect."""
    if field == "n":
        return _number(path, spread.n, "the count the figure's spread rests on", "count")
    if field == "repeats":
        return _number(path, spread.repeats, "the trials each case ran, where the count is trials", "plain")
    if field == "cases":
        return _number(path, spread.cases, "the distinct cases behind a count of trials", "count")
    if field == "counts":
        arm_id, key, where, aspect = at
        held = _stat_where(report_input, arm_id, key, where, aspect)
        counted = None if held is None else held[2]
        if counted is None:
            raise _unstated(path, "names what the figure's count counts, and it rests on no count")
        return Figure(path=path, kind="text", value=counted)
    arm_id, key, where, aspect = at
    return Figure(path=path, kind="text", value=_basis_words(report_input, arm_id, key, where, spread, aspect))


def _criterion_wheres(report_input: ReportInput, criterion: Criterion, arm_id: str) -> list[str]:
    """Where a criterion reads an arm, as ``<where>`` spells it."""
    found: list[str] = []
    for figure in criterion_figures(report_input, criterion, arm_id):
        if figure.cell is not None:
            found.append(f"cells.{_cell_spelling(report_input.cells[figure.cell])}")
        else:
            found.append(_OVERALL if figure.task is None else f"task.{figure.task}")
    return found


def _settle_criterion_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, arm_id, where, field = values
    criterion = _criterion(report_input, criterion_id)
    if criterion.statistic != "mean":
        raise _unstated(
            path, f"reads a {criterion.statistic}, which states no spread to size a design by; settle reads a mean"
        )
    aspect = criterion_other_aspect(report_input, criterion)
    spread = _spread(report_input, arm_id, criterion.measure, where, path, aspect)
    if field in _SETTLE_COUNT_FIELDS:
        return _criterion_count(report_input, path, spread, (arm_id, criterion.measure, where, aspect), field)
    higher = criterion.direction == "higher_is_better"
    return _settle_line(report_input, path, spread, criterion.measure, float(criterion.threshold), higher, field)


def _directed(report_input: ReportInput, key: str) -> bool | None:
    """Which way a measure or judged dimension is better: a judged dimension higher; None where a measure states none."""
    if key in report_input.judged:
        return True
    return report_input.measures[key].higher_is_better


def _settle_hypothetical_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    line_text, arm_id, key, where, field = values
    line = float(line_text)
    if field == "line":
        scale, unit = _scale_of(report_input, key, (line,))
        return _number(path, line, "the hypothetical line", scale, unit=unit)
    higher = _directed(report_input, key)
    if higher is None:
        raise _refuse(path, f"reads a line on {key!r}, which states no direction (higher or lower is better)")
    spread = _spread(report_input, arm_id, key, where, path)
    return _settle_line(report_input, path, spread, key, line, higher, field)


__all__ = [
    "pairing_contrast",
    "correction_words",
]
