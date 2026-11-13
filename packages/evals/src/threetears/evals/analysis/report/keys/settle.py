"""Settling a reading: a candidate's change against its aspect's margin or a floor, two arms apart on an aspect's
pass rate, and the separation families read with an assumed count.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.report.keys.counts import _cell_spellings
from threetears.evals.analysis.report.keys.figure import Figure
from threetears.evals.analysis.report.keys.prices import _means_of
from threetears.evals.analysis.report.keys.reads import (
    _MARGIN_LEDGER,
    _a_floor_s_change,
    _a_floor_s_rate,
    _aspect_strata,
    _cell_spelling,
    _costed_candidates,
    _of_the_pass_rate,
    _reading_arms,
    _scale_of,
    _strata,
    _the_pass_rate,
)
from threetears.evals.analysis.report.keys.separation import (
    _COMPARE_FIELDS,
    _COUNT_SOURCES,
    _HYPOTHETICAL_FIELDS,
    _MARGIN_FIELDS,
    _OVERALL,
    _REPEATS_BOUND,
    _SETTLE_COUNT_FIELDS,
    _SETTLE_LINE_FIELDS,
    _SETTLE_PAIR_FIELDS,
    _SPREAD_FIELDS,
    _arms,
    _aspects,
    _compare_figure,
    _criterion_wheres,
    _held_where,
    _rate_of,
    _separated_keys,
    _settle_criterion_figure,
    _settle_hypothetical_figure,
    _settle_line,
    _settle_pair_figure,
    _settled,
    _sided_level,
    _spread_figure,
    _stated_level,
    _wheres,
)
from threetears.evals.analysis.report.keys.vocabulary import (
    _Family,
    _Ids,
    _LedgerAs,
    _Number,
    _RateOf,
    _Slot,
    _Statistic,
    _Word,
    _a_percentile_is_a_level,
    _control,
    _criterion,
    _number,
    _reading,
    _refuse,
    _unstated,
)
from threetears.evals.analysis.report_input import Aspect, Reading, ReportInput, ScoreboardRow, Stat
from threetears.evals.analysis.report_margins import aspect_board, aspect_of, criterion_figures, on_aspect
from threetears.evals.analysis.separation import (
    POWER,
    NoSpread,
    Spread,
    cases_to_separate,
    difference_spread,
    level_share,
    stat_spread,
    wrong_side_share,
)
from threetears.evals.kernel.metrics import goal_check_of


# --- settling a candidate's reading against its aspect's margin, or a floor read on it --------------------------------


def _margined_aspects(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    """The aspects some reading is read against a margin on: the reading's own margin, else the aspect's."""
    return [
        aspect.id
        for aspect in report_input.aspects
        if any(_reading_margin(aspect, reading) is not None for reading in aspect.readings)
    ]


def _reading_margin(aspect: Aspect, reading: Reading) -> float | None:
    margin = reading.margin if reading.margin is not None else aspect.margin
    return None if margin is None else float(margin)


def _margined_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect = aspect_of(report_input, prior[-1])
    return [reading.arm for reading in aspect.readings if _reading_margin(aspect, reading) is not None]


def _reading_spread(aspect: Aspect, arm_id: str, path: str) -> Spread:
    """A candidate's reading on an aspect as a spread on its change, in points: its gain interval at its stated level
    (:func:`~threetears.evals.analysis.separation.difference_spread`), with the cases each arm ran."""
    reading = _reading(aspect, arm_id)
    if reading.gain_interval is None:
        raise _unstated(
            path, f"reads {arm_id}'s reading on {aspect.id} by its interval, which the reading does not state"
        )
    try:
        return difference_spread(
            float(reading.change),
            (float(reading.gain_interval[0]), float(reading.gain_interval[1])),
            level_share(reading.gain_confidence),
            tuple(float(count) for count in reading.cases),
        )
    except NoSpread as none:
        raise _unstated(path, f"reads {arm_id}'s reading on {aspect.id}, which has no spread: {none}") from none


def _settle_margin_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id, field = values
    aspect = aspect_of(report_input, aspect_id)
    margin = _reading_margin(aspect, _reading(aspect, arm_id))
    if margin is None:
        raise _unstated(path, f"reads {arm_id}'s reading on {aspect_id} against its margin, and none is declared")
    if field == "line":
        # A declared line prints as declared, never at a measure's changes' precision, so it is not marked a change.
        return _number(path, -margin, "the margin's line: the drop the reading may show", "points")
    spread = _reading_spread(aspect, arm_id, path)
    if field == "change":
        return _number(path, spread.mean, "the reading's change from the control", "points", change=True)
    return _settle_line(report_input, path, spread, "", -margin, True, field, in_points=True)


def _floored_aspects(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return list(dict.fromkeys(read.aspect for read in report_input.floors if read.margin is not None))


def _floored_keys(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    reads = report_input.floors
    return list(dict.fromkeys(read.floor for read in reads if read.aspect == prior[-1] and read.margin is not None))


def _floored_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    """The arms a floor is read on with a margin stated, which a ``settle.floor.`` key settles against; an arm whose
    read states no margin has no line to settle against."""
    aspect_id, floor = prior[-2:]
    return [
        read.arm
        for read in report_input.floors
        if (read.aspect, read.floor) == (aspect_id, floor) and read.margin is not None
    ]


def _floor_measure(report_input: ReportInput, rate: str, path: str) -> str:
    """The one measure a floor's rate names: the measure of that key, else the check measure minted for it."""
    named = [key for key in report_input.measures if key == rate or goal_check_of(key) == rate]
    if len(named) != 1:
        found = "no measure" if not named else f"{len(named)} measures"
        raise _unstated(path, f"reads the floor's rate {rate!r}, which names {found} the input holds")
    return named[0]


def _aspect_spread(report_input: ReportInput, key: str, aspect_id: str, arm_id: str, path: str) -> Spread:
    """An arm's spread on a measure on one aspect, from its own statistic there: its interval, half-width, standard
    error, or a pass/fail rate's binomial spread from its own count."""
    table = report_input.measures[key]
    stat = on_aspect(report_input, table, aspect_id).get(arm_id)
    if stat is None:
        raise _unstated(path, f"reads {arm_id}'s figure on {key!r} on {aspect_id}, which the input does not state")
    count = stat.n if stat.n is not None else stat.n_cases
    distinct = stat.n_cases if stat.n is not None and stat.n_cases is not None and stat.n_cases < stat.n else None
    try:
        return stat_spread(
            None if stat.mean is None else float(stat.mean),
            ci=None if stat.ci is None else (float(stat.ci[0]), float(stat.ci[1])),
            half_width=None if stat.half_width is None else float(stat.half_width),
            sem=None if stat.sem is None else float(stat.sem),
            n=None if count is None else float(count),
            stated_level=_stated_level(report_input, key),
            rate=_rate_of(report_input, key),
            cases=None if distinct is None else float(distinct),
        )
    except NoSpread as none:
        raise _unstated(path, f"reads {arm_id}'s spread on {key!r} on {aspect_id}: {none}") from none


def _floor_spread(report_input: ReportInput, aspect_id: str, floor: str, arm_id: str, path: str) -> Spread:
    """A floor's drop on a candidate as a spread, in points: the aspect's reading where the floor speaks for its pass
    rate, else the candidate's rate on the floor's own measure minus the control's, their spreads combined as
    independent."""
    read = next(
        held for held in report_input.floors if (held.aspect, held.floor, held.arm) == (aspect_id, floor, arm_id)
    )
    if read.reading is None:
        aspect = next((held for held in report_input.aspects if held.id == aspect_id), None)
        if aspect is None or arm_id not in _reading_arms(aspect):
            raise _unstated(path, f"reads the pass rate's drop on {aspect_id}, which holds no reading of {arm_id}")
        return _reading_spread(aspect, arm_id, path)
    control = _control(report_input)
    if control is None:
        raise _unstated(path, "reads a drop from the control, and the input names no control")
    key = _floor_measure(report_input, read.reading, path)
    mine = _aspect_spread(report_input, key, aspect_id, arm_id, path)
    theirs = _aspect_spread(report_input, key, aspect_id, control.id, path)
    factor = 100.0 if _scale_of(report_input, key)[0] == "share" else 1.0
    per_case = (
        None
        if mine.per_case is None or theirs.per_case is None
        else factor * math.hypot(mine.per_case, theirs.per_case)
    )
    count = None if mine.n is None or theirs.n is None else min(mine.n, theirs.n)
    return Spread(
        factor * (mine.mean - theirs.mean),
        factor * math.hypot(mine.se, theirs.se),
        count,
        per_case,
        mine.level,
        mine.basis,
        mine.cases if mine.cases == theirs.cases else None,
    )


def _settle_floor_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, floor, arm_id, field = values
    read = next(
        held for held in report_input.floors if (held.aspect, held.floor, held.arm) == (aspect_id, floor, arm_id)
    )
    if read.margin is None:
        raise _unstated(
            path, f"reads {arm_id}'s read of floor {floor} against its margin, which the read does not state"
        )
    if field == "line":
        # A declared line prints as declared, never at a measure's changes' precision, so it is not marked a change.
        return _number(path, -float(read.margin), "the floor's line: the drop it allows", "points")
    spread = _floor_spread(report_input, aspect_id, floor, arm_id, path)
    if field == "change":
        return _number(
            path, spread.mean, "the candidate's change from the control on the floor's rate", "points", change=True
        )
    return _settle_line(report_input, path, spread, "", -float(read.margin), True, field, in_points=True)


# --- settling two arms apart on an aspect's pass rate: trials passed over usable, on its scoreboard -------------------

#: What settling two arms apart on an aspect's pass rate reads: each arm's rate, their difference, the sizes and trials
#: (as two arms apart on a measure), how often a fresh run reads the other arm ahead, the level, the trials each case ran
#: today, and what the spread was read from (its count basis).
_PASSED_FIELDS = (
    "first",
    "second",
    "difference",
    "cases_needed",
    "cases_needed_80",
    "trials_needed",
    "trials_needed_80",
    "wrong_side",
    "level",
    "repeats",
    "basis",
)


def _passing_board(report_input: ReportInput, aspect_id: str) -> dict[str, ScoreboardRow]:
    """The arms an aspect's scoreboard states a pass rate for: trials passed, and usable trials above none."""
    return {
        arm: row
        for arm, row in aspect_board(report_input, aspect_id).items()
        if row.passed is not None and row.usable is not None and row.usable > 0
    }


def _passed_aspects(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [aspect.id for aspect in report_input.aspects if len(_passing_board(report_input, aspect.id)) >= 2]


def _passed_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(_passing_board(report_input, prior[-1]))


def _passed_others(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [arm for arm in _passing_board(report_input, prior[-2]) if arm != prior[-1]]


@dataclass(frozen=True)
class _PassedApart:
    """Two arms apart on an aspect's pass rate, in points: the difference, its spread, the level, today's repeats where
    both arms ran each case alike, and what the spread was read from."""

    difference: float
    se: float
    per_case: float | None
    level: float
    repeats: float | None
    basis: str


def _board_repeats(rows: Sequence[ScoreboardRow]) -> float | None:
    """The trials each case ran today, where every arm states its cases and ran each as often; None otherwise."""
    repeats = {
        float(row.usable) / float(row.cases)
        for row in rows
        if row.usable is not None and row.cases is not None and row.cases > 0
    }
    return repeats.pop() if len(repeats) == 1 and all(row.cases for row in rows) else None


def _passed_reading(
    report_input: ReportInput, aspect_id: str, arm_id: str, other: str
) -> tuple[Reading, tuple[float, float], float] | None:
    """The reading that pairs the two arms on the aspect, its interval, and the sign that makes it the first minus the
    second: one of them is the control, and the other's reading states an interval at a level; None otherwise."""
    control = _control(report_input)
    if control is None or control.id not in (arm_id, other):
        return None
    candidate = other if arm_id == control.id else arm_id
    aspect = aspect_of(report_input, aspect_id)
    held = [
        reading
        for reading in aspect.readings
        if reading.arm == candidate
        and reading.gain_interval is not None
        and level_share(reading.gain_confidence) is not None
    ]
    if len(held) != 1 or held[0].gain_interval is None:
        return None
    low, high = held[0].gain_interval
    return held[0], (float(low), float(high)), 1.0 if candidate == arm_id else -1.0


#: Why an arm's spread read on its trials may be too narrow.
_TRIALS_NARROWER = "which reads narrower than its cases would where a case ran more than once"

#: What a paired row's two estimates are, beside its difference: not the same figures, so said once in its basis.
_PAIRED_ESTIMATES = (
    "first and second are each arm's pooled pass rate on the aspect's scoreboard (trials passed over usable), while "
    "the difference is the reading's own change, the mean of per-case shares over the cases it pairs (under one test "
    "setup where several ran), so it need not equal first minus second: it differs where cases ran unequal usable "
    "trials, the reading pairs fewer cases than the scoreboard counts, or one setup was read; cite the difference "
    "with the sizes, never first minus second"
)


def _passed_apart(report_input: ReportInput, aspect_id: str, arm_id: str, other: str, path: str) -> _PassedApart:
    """Two arms apart on an aspect's pass rate (its scoreboard's trials passed over usable), in points: paired where the
    aspect's reading of a candidate against the control states its interval (its own change, interval and level, the
    cases each arm ran), else each arm's binomial spread from its own count, read on its distinct cases where its
    scoreboard states them, combined as independent."""
    board = _passing_board(report_input, aspect_id)
    rows = (board[arm_id], board[other])
    repeats = _board_repeats(rows)
    paired = _passed_reading(report_input, aspect_id, arm_id, other)
    if paired is not None:
        reading, interval, sign = paired
        try:
            spread = difference_spread(
                float(reading.change),
                interval,
                level_share(reading.gain_confidence),
                tuple(float(count) for count in reading.cases),
            )
        except NoSpread as none:
            raise _unstated(
                path, f"reads {reading.arm}'s reading on {aspect_id}, which has no spread: {none}"
            ) from none
        return _PassedApart(
            difference=sign * spread.mean,
            se=spread.se,
            per_case=spread.per_case,
            level=spread.level,
            repeats=repeats,
            basis=(
                f"paired: {reading.arm}'s reading on {aspect_id} against the control, its own change and interval at "
                "its stated level, over the cases each arm ran (the reading's cases); a size counts cases each arm "
                f"runs, each run as often as today. {_PAIRED_ESTIMATES}"
            ),
        )
    spreads = []
    for arm, row in zip((arm_id, other), rows, strict=True):
        try:
            spreads.append(
                stat_spread(
                    float(row.passed or 0) / float(row.usable or 1),
                    n=float(row.usable or 0),
                    rate="share",
                    cases=None if row.cases is None else float(row.cases),
                )
            )
        except NoSpread as none:
            raise _unstated(path, f"reads {arm}'s pass rate on {aspect_id}, which has no spread: {none}") from none
    first, second = spreads
    per_case = (
        None if first.per_case is None or second.per_case is None else 100 * math.hypot(first.per_case, second.per_case)
    )
    on_cases = [spread.cases is not None for spread in spreads]
    if all(on_cases):
        counted = f"read on its distinct cases, each counted once: {_REPEATS_BOUND}"
    elif not any(on_cases):
        counted = f"read on its usable trials as if each were independent, {_TRIALS_NARROWER}"
    else:
        cased, trialled = (arm_id, other) if on_cases[0] else (other, arm_id)
        counted = (
            f"read for {cased} on its distinct cases, each counted once ({_REPEATS_BOUND}), and for {trialled}, whose "
            f"scoreboard states no cases, on its usable trials as if each were independent, {_TRIALS_NARROWER}"
        )
    return _PassedApart(
        difference=100 * (first.mean - second.mean),
        se=100 * math.hypot(first.se, second.se),
        per_case=per_case,
        level=first.level,
        repeats=repeats,
        basis=(
            f"independent: each arm's trials passed over usable on {aspect_id}'s scoreboard, a pass/fail rate's "
            f"binomial spread {counted}; combined as if the arms ran different cases"
        ),
    )


def _settle_passed_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id, other, field = values
    if arm_id == other:
        raise _refuse(path, "settles an arm against itself; name two arms")
    board = _passing_board(report_input, aspect_id)
    if field in ("first", "second"):
        row = board[arm_id if field == "first" else other]
        rate = float(row.passed or 0) / float(row.usable or 1)
        return _number(path, rate, "the arm's pass rate on the aspect: trials passed over usable", "share")
    apart = _passed_apart(report_input, aspect_id, arm_id, other, path)
    if field == "difference":
        return _number(path, apart.difference, "the two arms' difference in pass rate", "points", change=True)
    if field == "basis":
        return Figure(path=path, kind="text", value=apart.basis)
    if field == "level":
        return _sided_level(path, apart.level, 2)
    if field == "repeats":
        return _number(path, apart.repeats, "the trials each case ran today, where both arms ran each alike", "plain")
    if field == "wrong_side":
        share = _settled(path, lambda: wrong_side_share(apart.difference, apart.se))
        return _number(path, share, "how often a fresh run of today's size reads the other arm ahead", "share")
    per_case = apart.per_case
    if per_case is None:
        raise _unstated(path, "names the cases a design needs, which needs the count each arm's rate rests on")
    power = POWER if field.endswith("_80") else 0.5
    cases = _settled(path, lambda: cases_to_separate(per_case, apart.difference, apart.level, sided=2, power=power))
    if field.startswith("trials_"):
        if apart.repeats is None:
            raise _unstated(
                path,
                "names the trials a size takes, and the two arms' scoreboards state no cases, or ran them a different "
                "number of times",
            )
        return _number(path, math.ceil(cases * apart.repeats - 1e-9), "the trials the cases take", "count")
    return _number(path, cases, "the cases each arm needs", "count")


def passed_aspects(report_input: ReportInput, arms: Sequence[str]) -> list[str]:
    """The aspects whose scoreboard states a pass rate for every one of ``arms`` (trials passed over usable), which a
    ``settle.passed.<aspect>.`` key settles them apart on, in the input's aspect order.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arms: The arms, by id.
    :ptype arms: Sequence[str]
    :return: The aspects, by id; empty where none holds every arm's pass rate.
    :rtype: list[str]
    """
    return [
        aspect for aspect in _passed_aspects(report_input) if set(arms) <= set(_passing_board(report_input, aspect))
    ]


def separation_wheres(report_input: ReportInput, arms: Sequence[str], key: str) -> list[str]:
    """Where a ``spread.``, ``compare.`` or ``settle.`` key can read every one of ``arms`` on a measure or judged
    dimension, as its ``<where>`` slot spells it: ``overall`` first, then ``task.<task>``, ``cells.<row>.<column>`` and
    ``strata.<stratum>``, each where every arm states a mean.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arms: The arms, by id.
    :ptype arms: Sequence[str]
    :param key: The measure or judged dimension, by key.
    :ptype key: str
    :return: The places, in the input's order; empty where the arms share none.
    :rtype: list[str]
    """
    return _wheres(report_input, arms, key)


def criterion_wheres(report_input: ReportInput, criterion_id: str, arm: str) -> list[str]:
    """Where a ``settle.criteria.`` key reads an arm against a declared criterion, as its ``<where>`` slot spells it:
    ``overall``, ``task.<task>`` or ``cells.<row>.<column>``, each place the criterion reads a figure for the arm.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion_id: The criterion, by id.
    :ptype criterion_id: str
    :param arm: The arm, by id.
    :ptype arm: str
    :return: The places, in the criterion's order; empty where it reads no figure for the arm.
    :rtype: list[str]
    :raises StopIteration: The input declares no criterion of that id.
    """
    return _criterion_wheres(report_input, _criterion(report_input, criterion_id), arm)


def _other_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [arm.id for arm in report_input.arms if arm.id != prior[-1]]


def _pair_keys(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _separated_keys(report_input, prior[-2:])


def _pair_wheres(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _wheres(report_input, prior[-3:-1], prior[-1])


def _one_arm_keys(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _separated_keys(report_input, prior[-1:])


def _one_arm_wheres(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _wheres(report_input, prior[-2:-1], prior[-1])


def _criterion_figure_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    criterion = _criterion(report_input, prior[0])
    return [arm.id for arm in report_input.arms if criterion_figures(report_input, criterion, arm.id)]


def _criterion_figure_wheres(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    criterion_id, arm_id = prior
    return _criterion_wheres(report_input, _criterion(report_input, criterion_id), arm_id)


def _keyed(name: str, read: Callable[[ReportInput], Iterable[str]]) -> _Ids:
    return _Ids(name, lambda report_input, _prior: read(report_input))


_ARM = _Ids("<arm>", _arms)
_CANDIDATE = _Ids("<candidate>", _costed_candidates)
#: The slots every key on one aspect begins with.
_ON_ASPECT = (_Word("aspects"), _Ids("<aspect>", _aspects))
#: The slots a strata key begins with: unqualified, and on one aspect.
_STRATUM = (_Word("strata"), _Ids("<stratum>", _strata))
_ASPECT_STRATUM = (*_ON_ASPECT, _Word("strata"), _Ids("<stratum>", _aspect_strata))
#: How a cell is named, and a criterion's place.
_CELL = _Ids("<cell>", _cell_spellings)


def _family_of(
    *slots: _Slot,
    what: str,
    resolve: Callable[[ReportInput, tuple[str, ...], str], Figure],
    derived: bool = False,
    derived_statistics: tuple[str, ...] = (),
    example: str = "",
    derived_note: str = "",
    compares: Callable[[ReportInput, tuple[str, ...]], Iterable[float] | None] | None = None,
    ledger_as: _LedgerAs = _a_percentile_is_a_level,
    rate: _RateOf | None = None,
) -> _Family:
    return _Family(
        slots=slots,
        what=what,
        resolve=resolve,
        derived=derived,
        derived_statistics=derived_statistics,
        example=example,
        derived_note=derived_note,
        compares=compares,
        ledger_as=ledger_as,
        rate=rate,
    )


#: What the grammar says of a separation figure, in place of "descriptive, no interval".
_SEPARATION_NOTE = (
    "Derived: computed here from the figures' stated spreads (normal approximation), never a test the pipeline ran."
)

#: The separation families (an arm's spread, two arms' difference, the cases that would settle a reading), read by
#: ``arms_compare`` and ``settle_cases``; every figure derived.
_SEPARATION_FAMILIES: tuple[_Family, ...] = (
    _family_of(
        _Word("spread"),
        _ARM,
        _Ids("<measure>", _one_arm_keys),
        _Ids("<where>", _one_arm_wheres),
        _Statistic(_SPREAD_FIELDS, "<field>"),
        what="an arm's spread on a measure or judged dimension, `overall` or within a slice (`task.<task>`, "
        "`cells.<row>.<column>`, `strata.<stratum>`): `se` the standard error of its mean and `per_case` one case's "
        "spread (its mean's over today's repeats, where the count is trials), in points for a rate; `n` the count it "
        "rests on and `counts` what that count counts (`cases`; `trials`, where a repeated case counts again; or "
        "`observations`, a statistic's own n that says neither); `cases` the distinct cases behind a count of trials "
        "and `repeats` the trials each ran (a binomial spread there is read on the cases, each counted once: the "
        "widest its repeats allow); `level` the level a result read from it is stated at; `basis` what the "
        "standard error was read from (a stated interval or half-width at a stated level, a standard error, or a "
        "pass/fail rate's binomial spread from a count, naming where the count was read: its own, the slice's stated "
        "cases, or the arm's scoreboard); `interval`, `low` and `high` the interval it gives at that level (the mean "
        "plus and minus the normal quantile times `se`, a rate's ends held to none to all; none for a rate of none or "
        "all), which a chart of a rate draws where the input states no interval",
        resolve=_spread_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="spread.fast.v2.tone.kept.overall.se",
    ),
    _family_of(
        _Word("compare"),
        _ARM,
        _Ids("<arm>", _other_arms),
        _Ids("<measure>", _pair_keys),
        _Ids("<where>", _pair_wheres),
        _Statistic(_COMPARE_FIELDS, "<field>"),
        what="the first arm minus the second on a measure or judged dimension, `overall` or within a slice, in points "
        "for a rate, with its interval: paired where a contrast states one against the other (the control) "
        "overall, else the two arms' own spreads combined as independent; `basis` says which",
        resolve=_compare_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        compares=lambda ri, values: _means_of(ri, values[2]),
        example="compare.fast.v2.base.tone.kept.overall.interval|pts",
    ),
    _family_of(
        _Word("settle"),
        _ARM,
        _Ids("<arm>", _other_arms),
        _Ids("<measure>", _pair_keys),
        _Ids("<where>", _pair_wheres),
        _Statistic(_SETTLE_PAIR_FIELDS, "<field>"),
        what="the cases each of two arms needs for a two-sided interval on their difference, centred on today's, to "
        "clear zero: `cases_needed` (half such designs clear if the truth equals today's difference), "
        "`cases_needed_80` (four in five do), in cases where the figures state their cases (each run as often as "
        "today), else in what their count counts (`spread.<arm>.<measure>.<where>.counts`); `trials_needed` and "
        "`trials_needed_80` the trials those take at today's repeats; `wrong_side` how often a fresh run of today's "
        "size reads the other arm ahead; `level` the level, with its sidedness (`95% two-sided`)",
        resolve=_settle_pair_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.fast.v2.base.tone.kept.overall.cases_needed",
    ),
    _family_of(
        _Word("settle"),
        _Word("criteria"),
        _keyed("<criterion>", lambda ri: [criterion.id for criterion in ri.criteria]),
        _Ids("<arm>", _criterion_figure_arms),
        _Ids("<where>", _criterion_figure_wheres),
        _Statistic((*_SETTLE_LINE_FIELDS, *_SETTLE_COUNT_FIELDS), "<field>"),
        what="an arm's mean against a declared criterion's threshold, where it reads it, on the aspect the criterion "
        "reads: `distance` toward passing "
        "(negative where today's figure misses); the cases a design needs for a one-sided bound centred on today's "
        "figure to sit wholly on today's side of the line, clearing it where the figure passes and confirming the "
        "miss where it misses (`cases_needed`: half such designs; `cases_needed_80`: four in five), and the same for "
        "the two-sided interval a page prints (`cases_needed_two_sided`, `cases_needed_two_sided_80`), each in cases "
        "where the figure states its cases (each run as often as today), else in what its count counts "
        "(`spread.<arm>.<measure>.<where>.counts`), and each with the trials it takes at today's repeats "
        "(`trials_needed`, `trials_needed_80`, `trials_needed_two_sided`, `trials_needed_two_sided_80`); on a "
        "repeated design a binomial spread is read on the cases, each counted once, so a size is the most cases the "
        "repeats could need; `wrong_side`, how often a fresh run of today's size lands on the other side; `level` the "
        "level, with its sidedness (`95% one-sided`); `n`, `counts`, `cases`, `repeats` and `basis` the count the "
        "arm's spread rests on, what it counts, the distinct cases and trials each ran, and what the spread was read "
        "from, as `spread.` reads them, on the criterion's aspect",
        resolve=_settle_criterion_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.criteria.cell_floor.fast.v2.cells.short.en.cases_needed",
    ),
    _family_of(
        _Word("settle"),
        _Word("hypothetical"),
        _Number("<line>"),
        _ARM,
        _Ids("<measure>", _one_arm_keys),
        _Ids("<where>", _one_arm_wheres),
        _Statistic(_HYPOTHETICAL_FIELDS, "<field>"),
        what="an arm's mean against a hypothetical line the snapshot does not declare, written in the figure's own "
        "units: what a declared criterion's settle reads, the line itself as `line`; every figure hypothetical",
        resolve=_settle_hypothetical_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.hypothetical.0.9.fast.v2.tone.kept.overall.wrong_side",
    ),
    _family_of(
        _Word("settle"),
        _Word("margin"),
        _Ids("<aspect>", _margined_aspects),
        _Ids("<candidate>", _margined_arms),
        _Statistic(_MARGIN_FIELDS, "<field>"),
        what="a candidate's reading on an aspect against the aspect's margin: `line` minus the margin, `change` the "
        "reading's change from the control (in points), then what a criterion's settle reads, the spread read off "
        "the reading's own interval at its stated level, each arm running the size in cases",
        resolve=_settle_margin_figure,
        rate=_the_pass_rate,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.margin.answers.fast.v2.cases_needed",
        ledger_as=_MARGIN_LEDGER,
    ),
    _family_of(
        _Word("settle"),
        _Word("floor"),
        _Ids("<aspect>", _floored_aspects),
        _Ids("<floor>", _floored_keys),
        _Ids("<candidate>", _floored_arms),
        _Statistic(_MARGIN_FIELDS, "<field>"),
        what="a candidate's change from the control against a floor read on it: `line` minus the floor's margin, "
        "`change` the change (in points), then what a criterion's settle reads; the change is the aspect's reading "
        "where the floor speaks for its pass rate, else the candidate's rate on the floor's own measure minus the "
        "control's, their spreads combined as independent",
        resolve=_settle_floor_figure,
        rate=_a_floor_s_rate,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.floor.answers.answers/rate_drop/*.fast.v2.wrong_side",
        ledger_as=_a_floor_s_change,
    ),
    _family_of(
        _Word("settle"),
        _Word(TRIAL_PASS_NAME),
        _Ids("<aspect>", _passed_aspects),
        _Ids("<arm>", _passed_arms),
        _Ids("<arm>", _passed_others),
        _Statistic(_PASSED_FIELDS, "<field>"),
        what="two arms apart on an aspect's pass rate (its scoreboard's trials passed over usable), the first minus "
        "the second: `first` and `second` each arm's pooled rate, `difference` in points (on a paired row the "
        "reading's own change, which need not equal first minus second; `basis` says so), then what two arms' "
        "settle reads "
        "(`cases_needed`, `cases_needed_80`, `trials_needed`, `trials_needed_80`, `wrong_side`, `level` two-sided); "
        "`repeats` the trials each case ran today; `basis` the count basis: paired where the aspect's reading of a "
        "candidate against the control states its interval (its own change and interval, over the cases each arm "
        "ran), else each arm's binomial spread from its own trials, read on its distinct cases where its scoreboard "
        "states them, combined as independent",
        resolve=_settle_passed_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.passed.answers.fast.v2.base.cases_needed",
        ledger_as=_of_the_pass_rate({"first": "level", "second": "level", "difference": "change"}),
        rate=_the_pass_rate,
    ),
)

# --- an assumed count: the separation families read as if each slice stating no count held one ---------------------

#: How ``assumed.share.`` reads a slice's count: the arm's own distinct cases on the measure, times the slice's
#: traffic share (and its trials times the share, on a repeated design).
ASSUMED_SHARE = "share"

#: What the grammar says of a figure read under an assumed count, in place of "descriptive, no interval".
_ASSUMED_NOTE = (
    "Hypothetical: read as if each slice stating no count of its own held the assumed cases, never a figure the input "
    "states; computed as its plain key's is."
)


def _assumed_count(path: str, spelled: str) -> int:
    """The cases an ``assumed.<cases>.`` key spells, refused unless a whole number above none."""
    value = float(spelled)
    if not value.is_integer() or value < 1:
        raise _refuse(
            path, f"assumes {spelled} cases a slice, and an assumed count is a whole number of cases, one or more"
        )
    return int(value)


def _slice_share(report_input: ReportInput, where: str) -> float | None:
    """A slice's stated share of traffic: a task's or a stratum's; None for a cell, or one that states none."""
    kind, _dot, named = where.partition(".")
    if kind == "task":
        shares = [task.share for task in report_input.tasks if task.id == named]
    elif kind == "strata":
        shares = [stratum.share for stratum in report_input.strata if stratum.id == named]
    else:
        shares = []
    return float(shares[0]) if len(shares) == 1 and shares[0] is not None else None


def _assumed_cases(
    report_input: ReportInput, assumed: int | None, arm_id: str, key: str, where: str, path: str
) -> dict[str, int] | None:
    """The count an assumption gives one arm's figure at a slice, where its count is unstated, as the statistic's own
    fields (``n_cases``, and ``n`` on a repeated design); None where the figure states its count (the assumption leaves
    it as it is). ``assumed`` None reads the arm's own distinct cases times the slice's traffic share, and on a
    repeated design its trials times the share as ``n``, so the slice keeps the arm's repeats."""
    held = _held_where(report_input, arm_id, key, where)
    if held is None or held.count is not None:
        return None
    if assumed is not None:
        return {"n_cases": assumed}
    share = _slice_share(report_input, where)
    overall = _held_where(report_input, arm_id, key, _OVERALL)
    cases = None
    if overall is not None and overall.count is not None:
        cases = overall.cases if overall.cases is not None else overall.count if overall.counted == "cases" else None
    if share is None or overall is None or cases is None:
        lacking = "a traffic share" if share is None else f"{arm_id}'s own distinct cases on {key!r}"
        raise _unstated(
            path, f"reads {where}'s cases as {arm_id}'s cases times its traffic share, and it states no {lacking}"
        )
    filled = {"n_cases": max(1, round(cases * share))}
    if overall.cases is not None and overall.count is not None:
        filled["n"] = max(filled["n_cases"], round(overall.count * share))
    return filled


def _assumed_input(
    report_input: ReportInput, assumed: int | None, arms: Sequence[str], key: str, where: str, path: str
) -> ReportInput:
    """The input as if each of ``arms``' figures on ``key`` at slice ``where`` that states no count held the assumed
    cases; refused where ``where`` is no slice, or every one of them states its count (the plain key reads it)."""
    if where == _OVERALL:
        raise _refuse(
            path, "assumes a count overall, and an assumed count fills only a slice: cite the key without `assumed.`"
        )
    filled = {arm: _assumed_cases(report_input, assumed, arm, key, where, path) for arm in arms}
    filled = {arm: cases for arm, cases in filled.items() if cases is not None}
    if not filled:
        raise _refuse(
            path,
            f"assumes a count at {where}, where every figure it reads states its own: cite the key without `assumed.`",
        )
    kind, _dot, named = where.partition(".")

    def counted(per_arm: Mapping[str, Mapping[str, Stat]]) -> dict[str, dict[str, Stat]]:
        return {
            arm: {
                measure: stat.model_copy(update=filled[arm]) if arm in filled and measure == key else stat
                for measure, stat in figures.items()
            }
            for arm, figures in per_arm.items()
        }

    if kind == "task":
        tasks = [
            task.model_copy(update={"per_arm": counted(task.per_arm)}) if task.id == named else task
            for task in report_input.tasks
        ]
        return report_input.model_copy(update={"tasks": tasks})
    if kind == "strata":
        strata = [
            stratum.model_copy(update={"per_arm": counted(stratum.per_arm)}) if stratum.id == named else stratum
            for stratum in report_input.strata
        ]
        return report_input.model_copy(update={"strata": strata})
    cells = {
        cell_id: cell.model_copy(update={"per_arm": counted(cell.per_arm)}) if _cell_spelling(cell) == named else cell
        for cell_id, cell in report_input.cells.items()
    }
    return report_input.model_copy(update={"cells": cells})


#: What one separation family's values name, for an assumption to fill: the arms, the measure, and the place.
_Reads = Callable[[ReportInput, tuple[str, ...]], tuple[tuple[str, ...], str, str]]


def _assumed_resolver(
    inner: _Family, reads: _Reads, *, counted: bool
) -> Callable[[ReportInput, tuple[str, ...], str], Figure]:
    """A separation family read under an assumed count: ``assumed.<cases>.`` (``counted``) or ``assumed.share.``."""

    def resolve_assumed(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
        assumed = _assumed_count(path, values[0]) if counted else None
        rest = values[1:] if counted else values
        arms, key, where = reads(report_input, rest)
        figure = inner.resolve(_assumed_input(report_input, assumed, arms, key, where, path), rest, path)
        if isinstance(figure.value, str) and figure.kind == "text" and path.endswith(".basis"):
            how = (
                f"{assumed} cases"
                if assumed is not None
                else "the arm's own distinct cases times the slice's traffic share"
            )
            own = f"from {_COUNT_SOURCES['own']}"
            said = figure.value.replace(own, f"from an assumed count ({how})")
            return replace(figure, value=said if said != figure.value else f"{figure.value}; the count assumed: {how}")
        return figure

    return resolve_assumed


def _stripped(slot: _Slot, prefix: int) -> _Slot:
    """An inner family's slot, its options read without the assumption's own values before it."""
    if not isinstance(slot, _Ids) or not prefix:
        return slot
    options = slot.options
    return _Ids(slot.name, lambda report_input, prior: options(report_input, prior[prefix:]))


def _criterion_reads(report_input: ReportInput, values: tuple[str, ...]) -> tuple[tuple[str, ...], str, str]:
    criterion_id, arm_id, where, _field = values
    return (arm_id,), _criterion(report_input, criterion_id).measure, where


#: Each separation family an assumption reads, with what its values name.
_ASSUMABLE: tuple[tuple[_Family, _Reads], ...] = (
    (_SEPARATION_FAMILIES[0], lambda _ri, values: ((values[0],), values[1], values[2])),
    (_SEPARATION_FAMILIES[1], lambda _ri, values: ((values[0], values[1]), values[2], values[3])),
    (_SEPARATION_FAMILIES[2], lambda _ri, values: ((values[0], values[1]), values[2], values[3])),
    (_SEPARATION_FAMILIES[3], _criterion_reads),
    (_SEPARATION_FAMILIES[4], lambda _ri, values: ((values[1],), values[2], values[3])),
)


def _assumed_family(inner: _Family, reads: _Reads, *, counted: bool) -> _Family:
    head: tuple[_Slot, ...] = (_Word("assumed"), _Number("<cases>")) if counted else (_Word("assumed.share"),)
    how = (
        "as if each slice stating no count of its own held `<cases>` cases"
        if counted
        else "as if each slice stating no count of its own held the arm's own distinct cases times the slice's traffic share"
    )
    return _family_of(
        *head,
        *(_stripped(slot, 1 if counted else 0) for slot in inner.slots),
        what=f"{inner.form()} read {how}; refused overall, and at a slice where every figure states its count",
        resolve=_assumed_resolver(inner, reads, counted=counted),
        derived=True,
        derived_note=_ASSUMED_NOTE,
    )


#: The separation families read under an assumed count, each by a number of cases and by traffic share.
_ASSUMED_FAMILIES: tuple[_Family, ...] = tuple(
    _assumed_family(inner, reads, counted=counted) for counted in (True, False) for inner, reads in _ASSUMABLE
)


def assumable(report_input: ReportInput, arms: Sequence[str], key: str, where: str) -> bool:
    """Whether an assumed count would fill something at a separation key's place: it is a slice, and one of ``arms``
    states a mean on ``key`` there and no count.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arms: The arms, by id.
    :ptype arms: Sequence[str]
    :param key: The measure or judged dimension, by key.
    :ptype key: str
    :param where: The place, as ``<where>`` spells it.
    :ptype where: str
    :return: Whether an ``assumed.`` key reads there.
    :rtype: bool
    """
    if where == _OVERALL:
        return False
    held = [_held_where(report_input, arm, key, where) for arm in arms]
    return any(one is not None and one.count is None for one in held)


__all__ = [
    "assumable",
    "criterion_wheres",
    "separation_wheres",
    "passed_aspects",
]
