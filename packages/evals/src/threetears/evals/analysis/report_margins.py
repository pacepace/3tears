"""How far each arm's figures sit from the lines they are read against, and which one sits nearest: the weakest margin.

A report's Watch slot opens with the weakest margin under the call (GENERATOR §2.5): the reading nearest to changing
what the page decides. Which reading that is, is a fact, so code finds it (owner, 2026-10-25: code computes facts)
and the planner cites it by figure key (``weakest.*``,
:mod:`~threetears.evals.analysis.report.figure_keys`); the plan checks refuse a decide plan whose Watch cites none.

**The lines a figure is read against**, each with the figure it reads:

- *A declared criterion* (a gate, floor, bar or guardrail): the arm's figure at the criterion's statistic, where its
  scope reads it (overall, in its one task, in every task, in every cell), against the threshold. One margin per
  figure, so a criterion read in every task gives one per task.
- *An aspect's margin*: the worst end of a reading's interval (``lo``) against ``-margin``. A reading is within the
  margin when its worst drop is under it (GENERATOR §3), so this is the distance at which it flips.
- *A floor the pipeline read*: the worst end of its interval against ``-margin``, the floor's own margin.

**A margin's gap** is signed toward passing: how far the figure clears its line (positive) or misses it (negative), in
the line's units. **The weakest margin** is the one whose gap is smallest relative to its line (``|gap| / |line|``),
since a gap in milliseconds and one in points compare only as shares of their lines; a line at zero compares at its
gap's size. Ties go to the first in the order above: criteria in the input's order, then readings, then floors, each
arm in the input's order.

**Only a margin that could change the call alone counts.** An option survives when it clears every gate and guardrail
figure, has a figure for each at every place its scope reads (:func:`criterion_places`), and breaches no guardrail the pipeline read (GENERATOR §3, as
:func:`~threetears.evals.analysis.report_decision.surviving_options` reads it). Every margin of a surviving option
counts: missing a gate would drop it, and a floor, bar, reading or margin moves how it is read. An option that does
not survive counts only by the one gate or guardrail figure it misses, when that is all that keeps it out, so
clearing it alone would let it back in; an option out on two figures, or for any other reason, is not one figure
from returning, and nothing it clears changes the call.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from threetears.evals.analysis.report_input import Criterion, Judged, Measure, Reading, ReportInput, Stat

__all__ = [
    "CriterionFigure",
    "CriterionPlace",
    "Margin",
    "criterion_figures",
    "criterion_places",
    "criterion_place",
    "flip_margin",
    "margins",
    "weakest_margin",
]

#: What a margin is read against: a declared criterion, an aspect's margin, or a floor the pipeline read.
_MarginKind = Literal["criterion", "margin", "floor"]

#: Where a criterion reads its figures: the arm's overall value, each task's (or its one task's), or each cell's.
CriterionPlace = Literal["overall", "task", "cell"]

#: The criterion kinds that remove an option that misses them.
_GATING = frozenset({"gate", "guardrail"})


@dataclass(frozen=True)
class CriterionFigure:
    """One figure a criterion reads for one arm, and where it sits.

    :ivar value: The arm's figure, at the criterion's statistic.
    :ivar task: The task it was read in; None when not read in a task.
    :ivar cell: The target-matrix cell it was read in, by id; None when not read in a cell.
    :ivar ci: The interval on the figure, low then high, where the input states one. Only a mean carries one: the
        input's interval is on the mean, so a p50 or p95 figure has none.
    :ivar half_width: The interval's half-width, where the input states only that; for a mean alone, like ``ci``.
    """

    value: float
    task: str | None = None
    cell: str | None = None
    ci: tuple[float, float] | None = None
    half_width: float | None = None


@dataclass(frozen=True)
class Margin:
    """One figure's distance from the line it is read against (this module's account).

    :ivar arm: The arm whose figure it is.
    :ivar kind: What the line is: a declared criterion, an aspect's margin, or a floor.
    :ivar source: The criterion's id, the aspect's id, or the floor's key.
    :ivar measure: The measure or judged dimension the criterion reads; None for a margin or a floor.
    :ivar aspect: The aspect a margin or floor is read on; None for a criterion.
    :ivar task: The task a criterion's figure was read in; None otherwise.
    :ivar cell: The cell a criterion's figure was read in; None otherwise.
    :ivar value: The figure read: the criterion's statistic, or the interval's worst end, in points.
    :ivar line: The threshold, or ``-margin``, in the same units.
    :ivar gap: How far the figure clears the line, signed toward passing.
    """

    arm: str
    kind: _MarginKind
    source: str
    measure: str | None
    aspect: str | None
    task: str | None
    cell: str | None
    value: float
    line: float
    gap: float

    @property
    def relative(self) -> float:
        """The gap as a share of its line; at a line of zero, the gap's size."""
        return abs(self.gap) / abs(self.line) if self.line else abs(self.gap)


def flip_margin(reading: Reading) -> float | None:
    """The margin at which a reading flips (GENERATOR §3, UPSTREAM U6): its interval's worst drop, ``-lo``, never below
    zero. The reading is within any margin above it and outside any at or below it.

    :param reading: An arm's reading on an aspect.
    :ptype reading: Reading
    :return: The flip margin, in points; None where the reading states no interval to read it on.
    :rtype: float | None
    """
    if reading.within_interval is None:
        return None
    return max(0.0, -float(reading.within_interval[0]))


def criterion_place(criterion: Criterion) -> CriterionPlace:
    """Where a criterion reads its figures, from its declared scope and task: every cell, a task (each task, or the
    one it names), else the arm's overall value. The one reading of ``Criterion.scope``, so every reader agrees.

    :param criterion: The declared criterion.
    :ptype criterion: Criterion
    :return: ``cell``, ``task`` or ``overall``.
    :rtype: CriterionPlace
    """
    if criterion.scope == "each_cell":  # prose-canary: allow — Criterion.scope is the input's declared enum, not prose
        return "cell"
    if criterion.scope == "each_task" or criterion.task is not None:  # prose-canary: allow — the same declared enum
        return "task"
    return "overall"


def criterion_places(report_input: ReportInput, criterion: Criterion) -> list[tuple[str | None, str | None]]:
    """Every place a criterion reads a figure at, by its scope: the arm's overall value, its one task or every task,
    or every cell. An arm that lacks the figure at any of them is not shown to pass it.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion: The declared criterion.
    :ptype criterion: Criterion
    :return: Each place as ``(task, cell)``: ``(None, None)`` overall, a task's id, or a cell's id, in the input's
        order; empty where the scope covers no place the input holds.
    :rtype: list[tuple[str | None, str | None]]
    """
    place = criterion_place(criterion)
    if place == "cell":
        return [(None, key) for key in report_input.cells]
    if place == "task":
        return [(task.id, None) for task in report_input.tasks if criterion.task in (None, task.id)]
    return [(None, None)]


def _stat_at(
    report_input: ReportInput, criterion: Criterion, arm: str, task: str | None, cell: str | None
) -> Stat | None:
    """The arm's stat on the criterion's measure at one of its places; None where the input states none."""
    if cell is not None:
        return report_input.cells[cell].per_arm.get(arm, {}).get(criterion.measure)
    if task is not None:
        held = next(candidate for candidate in report_input.tasks if candidate.id == task)
        return held.per_arm.get(arm, {}).get(criterion.measure)
    holders: tuple[Mapping[str, Measure | Judged], ...] = (report_input.measures, report_input.judged)
    for table in holders:
        figures = table.get(criterion.measure)
        if figures is not None and arm in figures.per_arm:
            return figures.per_arm[arm]
    return None


def criterion_figures(report_input: ReportInput, criterion: Criterion, arm: str) -> list[CriterionFigure]:
    """The arm's figures a criterion reads, at its statistic, at each of its places (:func:`criterion_places`).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion: The declared criterion.
    :ptype criterion: Criterion
    :param arm: The arm's id.
    :ptype arm: str
    :return: One figure per place the criterion reads it (overall, its task, every task or every cell), each where the
        input states it as a number, with the interval the input states on it. A figure that is not a finite number
        (NaN, an infinity) is no figure: it neither meets nor misses a line, so it is left out like a missing one, and
        an arm lacking it is not shown to pass.
    :rtype: list[CriterionFigure]
    """
    found: list[CriterionFigure] = []
    for task, cell in criterion_places(report_input, criterion):
        stat = _stat_at(report_input, criterion, arm, task, cell)
        value = getattr(stat, criterion.statistic) if stat is not None else None
        if stat is None or value is None or not math.isfinite(value):
            continue
        mean = criterion.statistic == "mean"
        ci = (float(stat.ci[0]), float(stat.ci[1])) if mean and stat.ci is not None else None
        half_width = float(stat.half_width) if mean and stat.half_width is not None else None
        found.append(CriterionFigure(float(value), task, cell, ci, half_width))
    return found


def _gap(criterion: Criterion, value: float) -> float:
    threshold = float(criterion.threshold)
    return value - threshold if criterion.direction == "higher_is_better" else threshold - value


def _missed_gates(report_input: ReportInput, arm: str) -> int | None:
    """How many gate and guardrail figures an arm misses; None when it is out for another reason: a gate with no figure
    for it at one of its places, or a guardrail the pipeline read ``breached`` on it."""
    if any(str(read.get("arm")) == arm and read.get("decision") == "breached" for read in report_input.guardrails):
        return None
    missed = 0
    for criterion in report_input.criteria:
        if criterion.kind not in _GATING:
            continue
        figures = criterion_figures(report_input, criterion, arm)
        if not figures or len(figures) < len(criterion_places(report_input, criterion)):
            return None
        missed += sum(_gap(criterion, figure.value) < 0 for figure in figures)
    return missed


def _criterion_margins(report_input: ReportInput, surviving: set[str]) -> Iterator[Margin]:
    """Every criterion figure's margin that could change the call alone (this module's account)."""
    for criterion in report_input.criteria:
        for arm in report_input.arms:
            for figure in criterion_figures(report_input, criterion, arm.id):
                gap = _gap(criterion, figure.value)
                if arm.id not in surviving and not (gap < 0 and criterion.kind in _GATING):
                    continue
                yield Margin(
                    arm=arm.id,
                    kind="criterion",
                    source=criterion.id,
                    measure=criterion.measure,
                    aspect=None,
                    task=figure.task,
                    cell=figure.cell,
                    value=figure.value,
                    line=float(criterion.threshold),
                    gap=gap,
                )


def _interval_margins(report_input: ReportInput, surviving: set[str]) -> Iterator[Margin]:
    """Every reading's distance from its aspect's margin, then every floor's from its own, read on intervals."""
    for aspect in report_input.aspects:
        for reading in aspect.readings:
            margin = reading.margin if reading.margin is not None else aspect.margin
            if reading.arm not in surviving or reading.within_interval is None or margin is None:
                continue
            low = float(reading.within_interval[0])
            if not math.isfinite(low):
                continue
            yield Margin(
                arm=reading.arm,
                kind="margin",
                source=aspect.id,
                measure=None,
                aspect=aspect.id,
                task=None,
                cell=None,
                value=low,
                line=-float(margin),
                gap=low + float(margin),
            )
    for floor in report_input.floors:
        if floor.arm not in surviving or floor.interval is None or floor.margin is None:
            continue
        low = float(floor.interval[0])
        if not math.isfinite(low):
            continue
        yield Margin(
            arm=floor.arm,
            kind="floor",
            source=floor.floor,
            measure=None,
            aspect=floor.aspect,
            task=None,
            cell=None,
            value=low,
            line=-float(floor.margin),
            gap=low + float(floor.margin),
        )


def margins(report_input: ReportInput) -> list[Margin]:
    """Every margin that could change the call alone, in this module's order.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Criteria in the input's order, then readings, then floors.
    :rtype: list[Margin]
    """
    missed = {arm.id: _missed_gates(report_input, arm.id) for arm in report_input.arms}
    surviving = {arm for arm, count in missed.items() if count == 0}
    one_short = {arm for arm, count in missed.items() if count == 1}
    found = [*_criterion_margins(report_input, surviving), *_interval_margins(report_input, surviving)]
    return [margin for margin in found if margin.arm in surviving or margin.arm in one_short]


def weakest_margin(report_input: ReportInput) -> Margin | None:
    """The weakest margin: the one whose gap is smallest relative to its line, the first on a tie.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The weakest margin; None when the input reads no figure against a criterion, margin or floor.
    :rtype: Margin | None
    """
    found = margins(report_input)
    if not found:
        return None
    least = min(margin.relative for margin in found)
    return next(margin for margin in found if math.isclose(margin.relative, least, rel_tol=0, abs_tol=1e-12))
