"""How far each arm's figures sit from the lines they are read against, and which one sits nearest: the weakest margin.

A report's Watch slot opens with the weakest margin under the call (GENERATOR §2.5): the reading nearest to changing
what the page decides. Which reading that is, is a fact, so code finds it (owner, 2026-10-25: code computes facts)
and the planner cites it by figure key (``weakest.*``,
:mod:`~threetears.evals.analysis.report.figure_keys`); the plan checks refuse a decide plan whose Watch cites none.

**The lines a figure is read against**, each with the figure it reads:

- *A declared criterion* (a gate, floor, bar or guardrail): the arm's figure at the criterion's statistic, where its
  scope reads it (overall, in its one task, in every task, in every cell), against the threshold. One margin per
  figure, so a criterion read in every task gives one per task.

  **Which aspect's figure** (:func:`criterion_aspect`): a criterion naming an aspect reads that aspect's figures, one
  naming none the lead aspect's (the first asked, else the first), as every by-arm map holds them. Overall, the lead
  aspect's figure is the measure's or judged dimension's ``per_arm`` and any other's its ``by_aspect``. A task's and a
  cell's figures name no aspect, so a criterion on an aspect other than the lead reads none there: that place is
  unknown, never read off the lead's figure. Every reader of a criterion's figures reads them here
  (:func:`criterion_figures`), so the gate check, the margins, the keys and the analyst's analyses cannot disagree.
- *An aspect's margin*: the worst end of a reading's interval (``lo``) against ``-margin``. A reading is within the
  margin when its worst drop is under it (GENERATOR §3), so this is the distance at which it flips.
- *A floor the pipeline read*: the worst end of its interval against ``-margin``, the floor's own margin.

**A margin's gap** is signed toward passing: how far the figure clears its line (positive) or misses it (negative), in
the line's units. **The weakest margin** is the one whose gap is smallest relative to its line (``|gap| / |line|``),
since a gap in milliseconds and one in points compare only as shares of their lines; a line at zero compares at its
gap's size. Ties go to the first in the order above: criteria in the input's order, then readings, then floors, each
arm in the input's order.

**Unknown is not a miss.** A place a criterion reads with no figure for an arm (none stated, or not a finite number)
is *unknown*: :func:`criterion_misses` counts only figures on the wrong side of the line, :func:`criterion_unknowns`
the places with none, and :func:`criterion_passes` needs every place read and none missed. An option no gate or
guardrail figure fails, and no breached guardrail removes, but with a gate it has no figure for somewhere is
*undecided* (:func:`undecided_arms`): neither shown to pass, so it does not survive, nor removed by a figure.

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

from threetears.evals.analysis.cell_mean import NoCellMean, cell_mean_interval
from threetears.evals.analysis.report_input import (
    Aspect,
    Criterion,
    Judged,
    Measure,
    Reading,
    ReportInput,
    ScoreboardRow,
    Stat,
)

__all__ = [
    "CriterionFigure",
    "CriterionPlace",
    "Margin",
    "aspect_board",
    "aspect_of",
    "criterion_aspect",
    "criterion_figures",
    "criterion_misses",
    "criterion_other_aspect",
    "criterion_passes",
    "criterion_places",
    "criterion_place",
    "criterion_unknowns",
    "flip_margin",
    "lead_aspect",
    "margins",
    "on_aspect",
    "surviving_arms",
    "undecided_arms",
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
    :ivar derived: Whether ``ci`` was derived here rather than stated: an overall mean that is the plain mean of its
        target-matrix cells, its interval combined from theirs (:mod:`~threetears.evals.analysis.cell_mean`).
    """

    value: float
    task: str | None = None
    cell: str | None = None
    ci: tuple[float, float] | None = None
    half_width: float | None = None
    derived: bool = False


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
    :ivar corrected: Whether a margin's worst end is read on the multiplicity-corrected interval of the arm's contrast
        (the pass-rate contrast) rather than the plain interval at the nominal level.
    :ivar level: The corrected interval's confidence level, as the input states it (``99.69%``); None for a figure read
        on the plain interval, a criterion, or a level the input does not state.
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
    corrected: bool = False
    level: str | None = None

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


def _corrected_interval(
    report_input: ReportInput, aspect: Aspect, reading: Reading
) -> tuple[tuple[float, float], str | None] | None:
    """The multiplicity-corrected interval on a reading's change, in points, where the input holds one: the pass-rate
    contrast of the arm against the control (kind ``trial``, how each trial ended) on the reading's aspect, which its
    family corrected together. The plain interval a reading carries (``within_interval``) is at the nominal level with
    no correction, so it can end well short of where the corrected one does; where both exist the worst case is read on
    the corrected one.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param aspect: The aspect the reading is on.
    :ptype aspect: Aspect
    :param reading: An arm's reading on that aspect.
    :ptype reading: Reading
    :return: The interval's two ends in points and its level as the input states it; None where the input holds no
        corrected interval for the reading (no such contrast, an untested one, or an end that is not finite).
    :rtype: tuple[tuple[float, float], str | None] | None
    """
    for contrast in report_input.contrasts:
        if contrast.kind != "trial" or contrast.arm != reading.arm or contrast.interval is None:
            continue
        # A contrast naming no aspect speaks for the report's one aspect.
        if contrast.aspect != aspect.id and not (contrast.aspect is None and len(report_input.aspects) == 1):
            continue
        low, high = (float(end) * 100.0 for end in contrast.interval)
        if not (math.isfinite(low) and math.isfinite(high)):
            continue
        level = contrast.level or next(
            (
                family.level
                for family in report_input.contrast_families
                if family.aspect in (contrast.aspect, None) and family.level
            ),
            None,
        )
        return (low, high), level
    return None


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


def lead_aspect(report_input: ReportInput) -> str | None:
    """The lead aspect, whose figures the input's by-arm maps hold: the first asked, else the first.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Its id; None where the input holds no aspect.
    :rtype: str | None
    """
    asked = [aspect.id for aspect in report_input.aspects if aspect.asked]
    return asked[0] if asked else next((aspect.id for aspect in report_input.aspects), None)


def aspect_of(report_input: ReportInput, aspect_id: str) -> Aspect:
    """One of the input's aspects, by id.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param aspect_id: The aspect's id.
    :ptype aspect_id: str
    :return: The aspect.
    :rtype: Aspect
    :raises KeyError: The input holds no aspect of that id; the message names it and the ids the input holds.
    """
    for aspect in report_input.aspects:
        if aspect.id == aspect_id:
            return aspect
    held = ", ".join(repr(aspect.id) for aspect in report_input.aspects) or "none"
    raise KeyError(f"the report input holds no aspect {aspect_id!r}; its aspects are {held}")


def aspect_board(report_input: ReportInput, aspect_id: str | None) -> Mapping[str, ScoreboardRow]:
    """How each arm's trials came out on one aspect: its own scoreboard, else, on the lead aspect, the input's.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param aspect_id: The aspect, by id; None for the lead aspect's, which the input's scoreboard holds.
    :ptype aspect_id: str | None
    :return: Each arm's row, by arm; empty on an aspect other than the lead that holds no scoreboard.
    :rtype: Mapping[str, ScoreboardRow]
    :raises KeyError: The input holds no aspect of that id (:func:`aspect_of`).
    """
    if aspect_id is None:
        return report_input.scoreboard
    board = aspect_of(report_input, aspect_id).scoreboard
    if board is not None:
        return board
    return report_input.scoreboard if aspect_id == lead_aspect(report_input) else {}


def on_aspect(report_input: ReportInput, table: Measure | Judged, aspect_id: str | None) -> Mapping[str, Stat]:
    """A measure's or judged dimension's figures on one aspect, by arm: ``per_arm`` on the lead aspect, else its
    ``by_aspect`` entry, which an aspect it was not read on lacks.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param table: The measure or judged dimension.
    :ptype table: Measure | Judged
    :param aspect_id: The aspect, by id; None for the lead aspect.
    :ptype aspect_id: str | None
    :return: Each arm's stat, by arm.
    :rtype: Mapping[str, Stat]
    """
    if aspect_id is None or aspect_id == lead_aspect(report_input):
        return table.per_arm
    return table.by_aspect.get(aspect_id, {})


def criterion_aspect(report_input: ReportInput, criterion: Criterion) -> str | None:
    """The aspect whose figures a criterion reads (this module's Which aspect's figure): the one it names, else the
    lead aspect (the first asked, else the first).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion: The declared criterion.
    :ptype criterion: Criterion
    :return: The aspect's id; None where the criterion names none and the input holds no aspect.
    :rtype: str | None
    """
    return criterion.aspect if criterion.aspect is not None else lead_aspect(report_input)


def criterion_other_aspect(report_input: ReportInput, criterion: Criterion) -> str | None:
    """The aspect a criterion reads where it is not the lead one, whose figures sit in ``by_aspect``; None for the
    lead's.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion: The declared criterion.
    :ptype criterion: Criterion
    :return: The aspect's id, or None.
    :rtype: str | None
    """
    if criterion.aspect is None or criterion.aspect == lead_aspect(report_input):
        return None
    return criterion.aspect


def _stat_at(
    report_input: ReportInput, criterion: Criterion, arm: str, task: str | None, cell: str | None
) -> Stat | None:
    """The arm's stat on the criterion's measure at one of its places, on the criterion's aspect; None where the input
    states none (a task's or cell's figure for an aspect other than the lead, which the input never holds)."""
    other = criterion_other_aspect(report_input, criterion)
    if (cell is not None or task is not None) and other is not None:
        return None
    if cell is not None:
        return report_input.cells[cell].per_arm.get(arm, {}).get(criterion.measure)
    if task is not None:
        held = next(candidate for candidate in report_input.tasks if candidate.id == task)
        return held.per_arm.get(arm, {}).get(criterion.measure)
    holders: tuple[Mapping[str, Measure | Judged], ...] = (report_input.measures, report_input.judged)
    for table in holders:
        figures = table.get(criterion.measure)
        if figures is None:
            continue
        per_arm = on_aspect(report_input, figures, other)
        if arm in per_arm:
            return per_arm[arm]
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
        input states it as a number, with the interval the input states on it; an overall mean that states none, and
        is the plain mean of its target-matrix cells whose every cell states a spread, carries the interval derived
        from theirs, marked ``derived``. A figure that is not a finite number
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
        derived = _derived_interval(report_input, criterion, arm) if mean and task is None and cell is None else None
        if ci is None and half_width is None and derived is not None:
            found.append(CriterionFigure(float(value), task, cell, derived, None, derived=True))
            continue
        found.append(CriterionFigure(float(value), task, cell, ci, half_width))
    return found


def _derived_interval(report_input: ReportInput, criterion: Criterion, arm: str) -> tuple[float, float] | None:
    """The interval derived for an arm's overall mean on the criterion's lead-aspect figure, where it is the plain mean
    of its target-matrix cells and every cell states a spread (:func:`~threetears.evals.analysis.cell_mean
    .cell_mean_interval`); None where it is not, or the criterion reads another aspect, whose figures no cell holds."""
    if criterion_other_aspect(report_input, criterion) is not None:
        return None
    try:
        return cell_mean_interval(report_input, criterion.measure, arm).interval
    except NoCellMean:
        # NOSILENT: an overall that is no plain mean of its cells, or whose cells state no spread, is the input's
        # shape and not a fault: it stays a point estimate, which the point_estimate caveat then says.
        return None


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


def criterion_misses(report_input: ReportInput, criterion: Criterion, arm: str) -> int:
    """How many of the places a criterion reads an arm's figure at it misses: each figure on the wrong side of the
    threshold. A place with no figure is unknown, not missed (:func:`criterion_unknowns`).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion: The declared criterion.
    :ptype criterion: Criterion
    :param arm: The arm's id.
    :ptype arm: str
    :return: The places missed on a figure, 0 where none is (a borderline figure clears its line).
    :rtype: int
    """
    return sum(_gap(criterion, figure.value) < 0 for figure in criterion_figures(report_input, criterion, arm))


def criterion_unknowns(report_input: ReportInput, criterion: Criterion, arm: str) -> int:
    """How many of the places a criterion reads an arm at hold no figure for it (GENERATOR §1.1: a figure the input
    does not hold is not shown passed, and is not shown missed either).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion: The declared criterion.
    :ptype criterion: Criterion
    :param arm: The arm's id.
    :ptype arm: str
    :return: The places with no finite figure for the arm.
    :rtype: int
    """
    return len(criterion_places(report_input, criterion)) - len(criterion_figures(report_input, criterion, arm))


def criterion_passes(report_input: ReportInput, criterion: Criterion, arm: str) -> bool:
    """Whether an arm clears a criterion everywhere it reads: it reads somewhere, every place holds a figure, and no
    figure misses.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion: The declared criterion.
    :ptype criterion: Criterion
    :param arm: The arm's id.
    :ptype arm: str
    :return: True where every place the criterion reads holds a figure clearing it.
    :rtype: bool
    """
    return (
        bool(criterion_places(report_input, criterion))
        and criterion_unknowns(report_input, criterion, arm) == 0
        and criterion_misses(report_input, criterion, arm) == 0
    )


def surviving_arms(report_input: ReportInput) -> list[str]:
    """The arms that clear every gate and guardrail figure at every place it reads, and breach no guardrail the
    pipeline read: the options that survive (this module's account, GENERATOR §3).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Their ids, in the input's arm order.
    :rtype: list[str]
    """
    return [arm.id for arm in report_input.arms if _missed_gates(report_input, arm.id) == 0]


def undecided_arms(report_input: ReportInput) -> list[str]:
    """The options neither shown to pass nor removed (this module's Unknown is not a miss): no gate or guardrail figure
    misses, no guardrail the pipeline read is breached, and some gate or guardrail has no figure for the arm at a
    place it reads (or reads at no place the input holds).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Their ids, in the input's arm order.
    :rtype: list[str]
    """
    breached = {str(read.get("arm")) for read in report_input.guardrails if read.get("decision") == "breached"}
    gating = [criterion for criterion in report_input.criteria if criterion.kind in _GATING]
    return [
        arm.id
        for arm in report_input.arms
        if arm.id not in breached
        and not any(criterion_misses(report_input, criterion, arm.id) for criterion in gating)
        and any(
            criterion_unknowns(report_input, criterion, arm.id) or not criterion_places(report_input, criterion)
            for criterion in gating
        )
    ]


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
            # The worst case is read on the corrected interval wherever the input holds one, and the margin says so.
            corrected = _corrected_interval(report_input, aspect, reading)
            if corrected is not None:
                low = corrected[0][0]
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
                corrected=corrected is not None,
                level=corrected[1] if corrected is not None else None,
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
