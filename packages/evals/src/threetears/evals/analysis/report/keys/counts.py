"""What a sentence counts or compares: gates and criteria, the examples a snapshot froze and the trials the run
failed, whether two aspects ran the same cases, a classifier's confusion, an aspect's pair readings, floors and bars,
and target-matrix cells.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from threetears.evals.analysis.report.keys.figure import Figure, Scale, _count
from threetears.evals.analysis.report.keys.reads import _cell_spelling, _scale_of, _slice_figure
from threetears.evals.analysis.report.keys.vocabulary import (
    _criterion,
    _end,
    _interval,
    _level,
    _number,
    _refuse,
    _text,
    _unstated,
)
from threetears.evals.analysis.report.keys.weakest import _cell_name
from threetears.evals.analysis.report_input import Criterion, FailedTrials, ReportInput
from threetears.evals.analysis.report_margins import (
    aspect_of,
    criterion_misses,
    criterion_passes,
    criterion_places,
    criterion_unknowns,
    surviving_arms,
    undecided_arms,
)


# --- counts: what a sentence counts, keyed so prose never types one ------------------------------------------------

#: What ``count.<what>`` counts in the campaign: its arms (every option, the control among them), aspects, tasks,
#: strata, target-matrix cells, configs of a factorial grid, gates and guardrails, every declared criterion, the
#: options surviving every gate and guardrail, the options undecided (no gate figure fails them, and a gate has no
#: figure for them somewhere), the pairs of arms its pair readings read apart and the pairs of arms there are, the
#: failed trials frozen as examples and the failed trials the run holds (:func:`_failures_held`).
_COUNTED = (
    "arms",
    "aspects",
    "tasks",
    "strata",
    "cells",
    "configs",
    "gates",
    "criteria",
    "surviving",
    "undecided",
    "pairs",
    "arm_pairs",
    "examples",
    "failures",
)

#: The criterion kinds that remove an option that misses them, as the gates count them.
_GATE_KINDS = frozenset({"gate", "guardrail"})


def _gates(report_input: ReportInput) -> list[Criterion]:
    return [criterion for criterion in report_input.criteria if criterion.kind in _GATE_KINDS]


def _campaign_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (what,) = values
    counted: dict[str, int] = {
        "arms": len(report_input.arms),
        "aspects": len(report_input.aspects),
        "tasks": len(report_input.tasks),
        "strata": len(report_input.strata),
        "cells": len(report_input.cells),
        "configs": len(report_input.configs),
        "gates": len(_gates(report_input)),
        "criteria": len(report_input.criteria),
    }
    if what == "surviving":
        return _count(path, len(surviving_arms(report_input)))
    if what == "undecided":
        return _count(path, len(undecided_arms(report_input)))
    if what == "pairs":
        read = {frozenset(pair.arms) for aspect in report_input.aspects for pair in aspect.pairs}
        return _count(path, len(read))
    if what == "arm_pairs":
        return _count(path, math.comb(len(report_input.arms), 2))
    if what == "examples":
        return _count(path, len(report_input.examples))
    if what == "failures":
        held = _failures_held(report_input, [aspect.id for aspect in report_input.aspects], path)
        return _count(path, sum(row.failed for row in held))
    return _count(path, counted[what])


def _gate_ids(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [criterion.id for criterion in _gates(report_input)]


def _criterion_ids(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [criterion.id for criterion in report_input.criteria]


def _passing_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion = _criterion(report_input, values[0])
    return _count(path, sum(criterion_passes(report_input, criterion, arm.id) for arm in report_input.arms))


def _places_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    return _count(path, len(criterion_places(report_input, _criterion(report_input, values[0]))))


def _failing_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, arm_id = values
    return _count(path, criterion_misses(report_input, _criterion(report_input, criterion_id), arm_id))


def _unknown_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, arm_id = values
    return _count(path, criterion_unknowns(report_input, _criterion(report_input, criterion_id), arm_id))


def _failed_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (arm_id,) = values
    missed = [criterion for criterion in _gates(report_input) if criterion_misses(report_input, criterion, arm_id)]
    return _count(path, len(missed))


def _factor_ids(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [factor.id for factor in report_input.factors]


def _levels_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    factor = next(factor for factor in report_input.factors if factor.id == values[0])
    return _count(path, len(factor.levels))


# --- the examples a snapshot froze, and the trials the run failed --------------------------------------------------


def _failures_held(report_input: ReportInput, aspects: Sequence[str], path: str) -> list[FailedTrials]:
    """The input's failed-trial counts on ``aspects``, refused as unstated where it counts none on one of them: an input
    built without them (an uncounted one, a bare one), or an aspect whose bundle left failures out, reads as not known,
    never as none."""
    held = report_input.failed_trials
    counted = {row.aspect for row in held} if held is not None else set()
    missing = [aspect for aspect in aspects if aspect not in counted]
    if held is None or missing:
        where = "" if held is None else f" on {', '.join(missing)}"
        raise _unstated(
            path,
            f"counts the failed trials the run holds{where}, which the snapshot does not count: not known, so a share "
            "of them is not either",
        )
    return [row for row in held if row.aspect in aspects]


def _example_aspects(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    """The aspects an example or a failed-trial count names, in the input's aspect order, then any other."""
    named = [example.aspect for example in report_input.examples if example.aspect is not None]
    named += [row.aspect for row in report_input.failed_trials or ()]
    order = [aspect.id for aspect in report_input.aspects]
    return [aspect for aspect in order if aspect in named] + [a for a in dict.fromkeys(named) if a not in order]


def _example_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    """The arms an example or a failed-trial count names on one aspect, in the input's arm order."""
    aspect = prior[-1]
    named = {example.arm for example in report_input.examples if example.aspect == aspect}
    named |= {row.arm for row in report_input.failed_trials or () if row.aspect == aspect}
    return [arm.id for arm in report_input.arms if arm.id in named]


def _example_strata(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    """The strata or tasks an example or a failed-trial count names for one aspect and arm."""
    aspect, arm = prior[-2:]
    named = [e.stratum for e in report_input.examples if (e.aspect, e.arm) == (aspect, arm)]
    named += [row.stratum for row in report_input.failed_trials or () if (row.aspect, row.arm) == (aspect, arm)]
    return [stratum for stratum in dict.fromkeys(named) if stratum is not None]


def _kept_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect, arm, *stratum = values
    kept = [
        example
        for example in report_input.examples
        if (example.aspect, example.arm) == (aspect, arm) and (not stratum or example.stratum == stratum[0])
    ]
    return _count(path, len(kept))


def _failed_trials_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect, arm, *stratum = values
    held = _failures_held(report_input, [aspect], path)
    rows = [row for row in held if row.arm == arm and (not stratum or row.stratum == stratum[0])]
    return _count(path, sum(row.failed for row in rows))


# --- whether two aspects ran the same cases -------------------------------------------------------------------------

#: What two aspects' cases are, as ``overlap.<aspect>.<aspect>.same`` reads it: every case in both, some, none, or not
#: known.
OVERLAP_WORDS: dict[str, str] = {
    "same": "the same cases",
    "partly": "partly the same cases",
    "apart": "different cases",
    "unknown": "not known: the snapshot does not hold the two aspects' case ids",
}

#: What an overlap key reads: whether the two aspects ran the same cases, in words, and the counts behind it.
_OVERLAP_FIELDS = ("same", "shared", "only_first", "only_second")


def _other_aspects(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [aspect.id for aspect in report_input.aspects if aspect.id != prior[-1]]


def _overlap(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    first, second, field = values
    held = report_input.case_overlap or ()
    found = next((row for row in held if row.aspects in ((first, second), (second, first))), None)
    if found is None:
        if field == "same":
            return Figure(path=path, kind="text", value=OVERLAP_WORDS["unknown"])
        raise _unstated(
            path, f"counts the cases {first} and {second} share, which the snapshot does not hold: not known"
        )
    flipped = found.aspects != (first, second)
    only_first, only_second = (
        (found.only_second, found.only_first) if flipped else (found.only_first, found.only_second)
    )
    if field == "same":
        word = "same" if not only_first and not only_second else "apart" if not found.shared else "partly"
        return Figure(path=path, kind="text", value=OVERLAP_WORDS[word])
    counts = {"shared": found.shared, "only_first": only_first, "only_second": only_second}
    return _count(path, counts[field])


# --- a classifier's confusion: where each expected label's predictions went ------------------------------------------


def _confusion_axes(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return list(report_input.confusion)


def _confusion_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(report_input.confusion[prior[-1]].per_arm)


def _confusion_labels(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    axis = next(value for value in prior if value in report_input.confusion)
    return list(report_input.confusion[axis].labels)


def _confusion(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    axis, arm, expected, predicted = values
    table = report_input.confusion[axis]
    matrix = table.per_arm[arm]
    row, column = table.labels.index(expected), table.labels.index(predicted)
    if row >= len(matrix) or column >= len(matrix[row]):
        raise _unstated(path, f"names a cell of {arm}'s {axis} confusion its matrix does not hold")
    scale: Scale = "percent" if table.unit == "row_percent" else "count"
    return Figure(path=path, kind="number", value=float(matrix[row][column]), scale=scale)


# --- an aspect's pair readings, floors and bars ----------------------------------------------------------------------

_PAIR_STATISTICS = ("change", "interval", "low", "high", "confidence", "p_holm", "ahead", "why")


def _pair_firsts(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(pair.arms[0] for pair in aspect_of(report_input, prior[0]).pairs))


def _pair_seconds(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, first = prior
    pairs = aspect_of(report_input, aspect_id).pairs
    return list(dict.fromkeys(pair.arms[1] for pair in pairs if pair.arms[0] == first))


def _pair(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, first, second, statistic = values
    held = [pair for pair in aspect_of(report_input, aspect_id).pairs if pair.arms == (first, second)]
    if len(held) > 1:
        raise _refuse(path, f"names the pair {first} and {second}, which aspect {aspect_id} holds {len(held)} times")
    pair = held[0]
    if statistic == "change":
        return _number(path, pair.change, "the pair's change", "points", change=True)
    if statistic == "interval":
        return _interval(path, pair.interval, "the pair's interval", "points", change=True)
    if statistic in ("low", "high"):
        end = _end(path, pair.interval, statistic, "the pair's interval")
        return _number(path, end, f"the interval's {statistic} end", "points", change=True)
    if statistic == "confidence":
        return _level(path, pair.confidence, "the interval's level")
    if statistic == "p_holm":
        return _number(path, pair.p_holm, "the Holm-adjusted p", "plain")
    if statistic == "ahead":
        if pair.ahead is None:
            raise _unstated(path, "names the arm shown ahead, which the input does not state: neither is shown ahead")
        return Figure(path=path, kind="arm", value=pair.ahead)
    return _text(path, pair.why, "why the pair reads as it does")


_FLOOR_FIELDS = (
    "status",
    "reading",
    "speaks_for",
    "interval",
    "low",
    "high",
    "worst_drop",
    "margin",
    "short_by",
    "reason",
    "held_above",
    "broken_at",
)
#: The floor fields computed here: the margin above which the floor holds (``-low``), and at or below which its
#: interval's best end breaks it (``-high``).
_FLOOR_DERIVED = ("held_above", "broken_at")


def _floor_aspects(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    """The aspects floors are read on, as each read names its own: an aspect the input holds a record for or not."""
    return list(dict.fromkeys(read.aspect for read in report_input.floors))


def _floor_keys(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(read.floor for read in report_input.floors if read.aspect == prior[0]))


def _floor_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, floor = prior
    return [read.arm for read in report_input.floors if (read.aspect, read.floor) == (aspect_id, floor)]


def _floor(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, floor, arm_id, field = values
    held = [read for read in report_input.floors if (read.aspect, read.floor, read.arm) == (aspect_id, floor, arm_id)]
    if len(held) > 1:
        raise _refuse(path, f"names floor {floor} on {arm_id}, which aspect {aspect_id} holds {len(held)} times")
    read = held[0]
    if field == "interval":
        return _interval(path, read.interval, "the floor's interval", "points", change=True)
    if field in ("low", "high"):
        end = _end(path, read.interval, field, "the floor's interval")
        return _number(path, end, f"the interval's {field} end", "points", change=True)
    if field in ("held_above", "broken_at"):
        end = _end(path, read.interval, "low" if field == "held_above" else "high", "the floor's interval")
        return _number(path, -float(end), f"the floor's {field}", "points")
    if field in ("worst_drop", "margin", "short_by"):
        return _number(path, getattr(read, field), f"the floor's {field}", "points")
    return _text(path, getattr(read, field), f"the floor's {field}")


_BAR_FIELDS = ("status", "value", "bar", "interval", "low", "high", "why")


def _bar_aspects(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    """The aspects bars are read on, as each read names its own: an aspect the input holds a record for or not."""
    return list(dict.fromkeys(read.aspect for read in report_input.bars))


def _bar_measures(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(read.measure for read in report_input.bars if read.aspect == prior[0]))


def _bar_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, measure = prior
    return [read.arm for read in report_input.bars if (read.aspect, read.measure) == (aspect_id, measure)]


def _bar(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, measure, arm_id, field = values
    held = [read for read in report_input.bars if (read.aspect, read.measure, read.arm) == (aspect_id, measure, arm_id)]
    if len(held) > 1:
        raise _refuse(
            path, f"names the bar on {measure} for {arm_id}, which aspect {aspect_id} holds {len(held)} times"
        )
    read = held[0]
    scale, unit = _scale_of(report_input, measure, (float(read.bar),))
    if field in ("value", "bar"):
        return _number(path, getattr(read, field), f"the bar's {field}", scale, unit=unit)
    if field == "interval":
        return _interval(path, read.interval, "the bar's interval", scale, unit=unit)
    if field in ("low", "high"):
        end = _end(path, read.interval, field, "the bar's interval")
        return _number(path, end, f"the interval's {field} end", scale, unit=unit)
    return _text(path, getattr(read, field), f"the bar's {field}")


# --- target-matrix cells -------------------------------------------------------------------------------------------

_CELL_FIELDS = ("name", "n_cases")


def _cell_spellings(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [_cell_spelling(cell) for cell in report_input.cells.values()]


def _spelled_cell(report_input: ReportInput, spelled: str) -> str:
    return next(cell_id for cell_id, cell in report_input.cells.items() if _cell_spelling(cell) == spelled)


def _cell_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(report_input.cells[_spelled_cell(report_input, prior[0])].per_arm)


def _cell_keys(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    spelled, arm_id = prior
    return list(report_input.cells[_spelled_cell(report_input, spelled)].per_arm[arm_id])


def _cell_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    spelled, field = values
    cell_id = _spelled_cell(report_input, spelled)
    if field == "name":
        return _text(path, _cell_name(report_input, cell_id), "the cell's name")
    return _number(path, report_input.cells[cell_id].n_cases, "the cell's cases", "count")


def _cell_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    spelled, arm_id, key, statistic = values
    per_arm = report_input.cells[_spelled_cell(report_input, spelled)].per_arm
    return _slice_figure(report_input, per_arm, (arm_id, key, statistic), path, f"cell {spelled}")


__all__: list[str] = []
