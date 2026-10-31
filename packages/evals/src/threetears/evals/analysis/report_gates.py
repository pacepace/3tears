"""The gate check (GENERATOR §3): each arm's status against the gates, and why, one reason per figure.

Every part of a page that says whether an arm passes reads it here, as the options that survive
(:func:`~threetears.evals.analysis.report_decision.surviving_options`) do, so a page never shows an arm passing that
the decision dropped.

**What is checked.** Every criterion of kind ``gate`` or ``guardrail``, at the figures its scope reads
(:func:`~threetears.evals.analysis.report_margins.criterion_place` and
:func:`~threetears.evals.analysis.report_margins.criterion_figures`): overall, in its one task, in every task, or in
every cell. A check narrowed to a task reads only the criteria read in tasks, at that task's figures; one narrowed to
a cell, only those read in cells, at that cell's; one narrowed to a criterion, only that one, wherever it reads (a gate
matrix's row per gate). Floors and bars warn and never gate; a guardrail the pipeline read ``breached`` is the
decision's to drop, not a figure here.

**Each figure**, on its point estimate:

- *fails* when it misses the threshold: under it where higher is better, over it where lower is;
- else is *borderline* when its interval crosses the threshold: the low end under it where higher is better, the high
  end over it where lower is. The interval is the figure's ``Stat.ci``, else the value plus or minus its
  ``half_width``, else, for an overall mean that is the plain mean of its target-matrix cells, the interval derived
  from theirs (:mod:`~threetears.evals.analysis.cell_mean`), which its reason calls derived. Only a mean carries one,
  so a p50 or p95 is never borderline;
- else *passes*.

A gate with no figure for the arm at a place the check reads it is *unknown* there: a figure the input does not hold
is not shown passed (GENERATOR §1.1), and neither is one that is not a number (a NaN or an infinity), which reads as no
figure. Its finding (:func:`gate_findings`) says ``unknown``, never ``fail``, so a reader that keys each figure tells a
place with no figure from a figure that misses; the check's status (:func:`gate_check`) counts it among the fails,
since an arm not shown to pass a gate does not pass it. Each comparison is written as the figure *meeting* the line,
so a value no comparison holds for meets none. Unnarrowed, a gate read in every task (or every cell) is unknown at
each task (cell) that lacks the arm's figure, so the unnarrowed check agrees with every narrowed one.

**The status** is ``fail`` when any figure fails or is unknown, ``borderline`` when none fails and one is borderline, else
``pass``. Borderline counts as passing for the decision (GENERATOR §3).

**Reasons** are code's templates, one per figure, in the criteria's order. Each names what it reads (the statistic
for a p50 or p95, the criterion's reader name for a mean) and states the figure by the package's number rule:
``p95 3.6 s > 3 s``, ``Resolves the issue 74% < 90%``, ``Resolves the issue 90% ±4.4``, ``Cost per reply $0.00541,
interval $0.00522–$0.00559``, ``Latency budget (p95): no figure``, ``Overall quality floor: no figure in Billing &
account Q&A``. An unnarrowed check adds where the figure sits
(``in Billing & account Q&A``, ``in Complex × News``).

**Routing** (GENERATOR §3, read by the gate matrix, :func:`routing_picks`): for each routable task, the cheapest arm
that passes there and in every cross-cutting task. Cost (:func:`task_costs`) is the task's figure on the input's one
measure stated per 1,000 (of what its label counts), else on its one dollar measure on the cost merit axis, times
1,000; an arm with no cost there is never the cheapest. The costs carry the measure they were read on, so what they are
per is that measure's, never the page's cost basis.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from threetears.evals.analysis.report.cost_unit import MoneyRuler, money_ruler
from threetears.evals.analysis.report.figure_keys import Figure, format_figure, reader_name, resolve
from threetears.evals.analysis.report_input import Criterion, ReportInput
from threetears.evals.analysis.report_margins import (
    CriterionFigure,
    CriterionPlace,
    criterion_figures,
    criterion_place,
    criterion_places,
)

__all__ = [
    "criterion_ruler",
    "GateCheck",
    "GateFinding",
    "GateStatus",
    "gate_check",
    "gate_findings",
    "routing_picks",
    "task_costs",
]

#: An arm's status against the gates: every figure passes, one sits borderline and none fails, or one fails.
GateStatus = Literal["pass", "borderline", "fail"]

#: The criterion kinds that gate an option.
_GATING = frozenset({"gate", "guardrail"})

#: The scales whose interval ends read in points beside their figure (``90% ±4.4``).
_RATE_SCALES = frozenset({"share", "percent", "points"})

#: How a missed line reads, by the criterion's direction.
_MISSED = {"higher_is_better": "<", "lower_is_better": ">"}

#: Where a reason says its figure sits, when the check was not narrowed to that place.
_WHERE_TEMPLATE = "{reason} in {place}"
#: What a gate with no figure for the arm reads.
_NO_FIGURE_TEMPLATE = "{name}: no figure"
#: How a cell's two levels read together.
_CELL_TEMPLATE = "{row} × {column}"


@dataclass(frozen=True)
class GateCheck:
    """One arm checked against the gates (this module's account).

    :ivar arm: The arm's id.
    :ivar status: ``fail``, ``borderline`` or ``pass``.
    :ivar fails: Why it fails, one reason per failing figure, in the criteria's order.
    :ivar borders: Why it is borderline, one reason per borderline figure, in the criteria's order.
    """

    arm: str
    status: GateStatus
    fails: tuple[str, ...]
    borders: tuple[str, ...]


@dataclass(frozen=True)
class GateFinding:
    """One figure a gate check fails or holds borderline, as data: which criterion, where, and its reason.

    :ivar criterion: The gate's or guardrail's id.
    :ivar status: ``fail`` (the figure misses), ``borderline`` (its interval crosses the line) or ``unknown`` (the input
        states no figure there).
    :ivar task: The task the figure is read in; None when it is not read in a task.
    :ivar cell: The target-matrix cell it is read in, by id; None when it is not read in a cell.
    :ivar figure: The arm's figure there; None where the input states none (``unknown``).
    :ivar reason: The reason :func:`gate_check` gives for it, its figures written by the package's number rule.
    """

    criterion: str
    status: Literal["fail", "borderline", "unknown"]
    task: str | None
    cell: str | None
    figure: CriterionFigure | None
    reason: str


def _applies(criterion: Criterion, task: str | None, cell: str | None, only: str | None) -> bool:
    """Whether a check narrowed to ``task`` or ``cell`` (neither: unnarrowed), and to the criterion ``only`` (None: to
    none), reads ``criterion``."""
    if criterion.kind not in _GATING or only not in (None, criterion.id):
        return False
    place: CriterionPlace = criterion_place(criterion)
    if task is not None:
        return place == "task" and criterion.task in (None, task)
    if cell is not None:
        return place == "cell"
    return True


def _cell_words(report_input: ReportInput, cell_id: str) -> str:
    cell = report_input.cells[cell_id]
    matrix = report_input.target_matrix
    if matrix is None:
        return cell_id
    row = next((level.name or level.id for level in matrix.rows.levels if level.id == cell.row), cell.row)
    column = next((level.name or level.id for level in matrix.columns.levels if level.id == cell.column), cell.column)
    return _CELL_TEMPLATE.format(row=row, column=column)


def _placed(report_input: ReportInput, reason: str, task: str | None, cell: str | None, narrowed: bool) -> str:
    if narrowed:
        return reason
    if task is not None:
        return _WHERE_TEMPLATE.format(reason=reason, place=reader_name("task", task, report_input))
    if cell is not None:
        return _WHERE_TEMPLATE.format(reason=reason, place=_cell_words(report_input, cell))
    return reason


#: The scales a criterion's line is money on.
_MONEY_SCALES = frozenset({"usd", "usd_per_1k", "usd_per_1m"})


def criterion_ruler(report_input: ReportInput, criterion: Criterion, line: Figure) -> tuple[MoneyRuler, int]:
    """The ruler a money criterion's figures are stated on, where its count is named once beside them (a gate's
    reason, a gate row's line): the one owner of money's scale over the measure's every arm's figure and the line, per
    1,000 of what it counts at least, never a per-1,000 mark.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion: The criterion, on a money measure.
    :ptype criterion: Criterion
    :param line: Its threshold's figure (``criteria.<id>.threshold``).
    :ptype line: Figure
    :return: The ruler, and the count its measure's own unit is per.
    :rtype: tuple[MoneyRuler, int]
    """
    own = _SCALE_COUNTS[line.scale]
    measure = report_input.measures.get(criterion.measure)
    amounts = [
        float(stat.mean) for stat in (measure.per_arm.values() if measure is not None else ()) if stat.mean is not None
    ]
    return money_ruler([*amounts, float(criterion.threshold)], per=own, start=max(own, 1000)), own


#: How many of what it counts a figure on each dollar scale is per.
_SCALE_COUNTS = {"usd": 1, "usd_per_1k": 1000, "usd_per_1m": 1_000_000}


def _interval(figure: CriterionFigure) -> tuple[float, float] | None:
    if figure.ci is not None:
        return figure.ci
    if figure.half_width is not None:
        return figure.value - figure.half_width, figure.value + figure.half_width
    return None


def _spread(line: Figure, figure: CriterionFigure) -> str:
    """The figure's interval as a reason states it after the figure: ``±4.4``, ``interval 85.6–94.4%``, or ``derived
    interval 89.8–90.4%`` where code derived it from the cells (the figure's ``derived``)."""
    if figure.ci is not None:
        words = "derived interval " if figure.derived else "interval "
        return words + format_figure(replace(line, kind="interval", value=figure.ci))
    assert figure.half_width is not None
    width = replace(line, value=figure.half_width)
    return "±" + format_figure(width, ("pts", "abs") if line.scale in _RATE_SCALES else ())


def _read(
    report_input: ReportInput, criterion: Criterion, figure: CriterionFigure, line: Figure
) -> tuple[GateStatus, str]:
    """One figure against its criterion's line: its status, and the reason it would give."""
    name = (
        criterion.statistic if criterion.statistic != "mean" else reader_name("criterion", criterion.id, report_input)
    )
    threshold = float(criterion.threshold)
    higher = criterion.direction == "higher_is_better"
    meets = figure.value >= threshold if higher else figure.value <= threshold
    interval = _interval(figure)
    borderline = interval is not None and not (interval[0] >= threshold if higher else interval[1] <= threshold)
    if line.scale in _MONEY_SCALES:
        # Money on the one owner's ruler (#993): both figures in its form, its count named once, in the criterion's
        # name (its own count replaced), never a per-1,000 mark.
        ruler, own = criterion_ruler(report_input, criterion, line)
        name = ruler.label(name, line.per, own)
        if not meets:
            return "fail", f"{name} {ruler.spell(figure.value)} {_MISSED[criterion.direction]} {ruler.spell(threshold)}"
        if borderline and interval is not None:
            if figure.ci is not None:
                words = "derived interval " if figure.derived else "interval "
                spread = f", {words}{ruler.spell(interval[0])}–{ruler.spell(interval[1])}"
            else:
                spread = f" ±{ruler.spell(float(figure.half_width or 0))}"
            return "borderline", f"{name} {ruler.spell(figure.value)}{spread}"
        return "pass", ""
    stated = format_figure(replace(line, value=figure.value))
    if not meets:
        return "fail", f"{name} {stated} {_MISSED[criterion.direction]} {format_figure(line)}"
    if borderline:
        separator = ", " if figure.ci is not None else " "
        return "borderline", f"{name} {stated}{separator}{_spread(line, figure)}"
    return "pass", ""


def gate_findings(
    report_input: ReportInput,
    arm: str,
    *,
    task: str | None = None,
    cell: str | None = None,
    criterion: str | None = None,
) -> tuple[GateFinding, ...]:
    """Every figure one arm fails, holds borderline or has none at, overall or in one task or cell, or on one criterion,
    as data:
    what :func:`gate_check` reads its status and reasons off, for a reader that keys each figure rather than reading
    its reason.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The arm's id.
    :ptype arm: str
    :param task: The one task to check in, by id; None to check wherever each criterion reads.
    :ptype task: str | None
    :param cell: The one target-matrix cell to check in, by id; None to check wherever each criterion reads.
    :ptype cell: str | None
    :param criterion: The one gate or guardrail to check against, by id; None for every one.
    :ptype criterion: str | None
    :return: Each failing, borderline and unknown figure, in the criteria's order and each criterion's places'
        order.
    :rtype: tuple[GateFinding, ...]
    :raises ValueError: Both a task and a cell were named, or a task, cell or gating criterion the input does not hold.
    """
    if task is not None and cell is not None:
        raise ValueError(f"a gate check reads one place at most; it was given task {task!r} and cell {cell!r}")
    if task is not None and task not in {held.id for held in report_input.tasks}:
        raise ValueError(f"the input holds no task {task!r}")
    if cell is not None and cell not in report_input.cells:
        raise ValueError(f"the input holds no cell {cell!r}")
    if criterion is not None and criterion not in {held.id for held in report_input.criteria if held.kind in _GATING}:
        raise ValueError(f"the input holds no gate or guardrail {criterion!r}")
    narrowed = task is not None or cell is not None
    found: list[GateFinding] = []
    for held in report_input.criteria:
        if not _applies(held, task, cell, criterion):
            continue
        places = [
            (at_task, at_cell)
            for at_task, at_cell in criterion_places(report_input, held)
            if (task is None or at_task == task) and (cell is None or at_cell == cell)
        ]
        no_figure = _NO_FIGURE_TEMPLATE.format(name=reader_name("criterion", held.id, report_input))
        if not places:
            found.append(GateFinding(held.id, "unknown", None, None, None, no_figure))
            continue
        figures: dict[tuple[str | None, str | None], CriterionFigure] = {
            (figure.task, figure.cell): figure for figure in criterion_figures(report_input, held, arm)
        }
        line = resolve(f"criteria.{held.id}.threshold", report_input)
        for at_task, at_cell in places:
            figure = figures.get((at_task, at_cell))
            if figure is None or not math.isfinite(figure.value):
                reason = _placed(report_input, no_figure, at_task, at_cell, narrowed)
                found.append(GateFinding(held.id, "unknown", at_task, at_cell, None, reason))
                continue
            read, reason = _read(report_input, held, figure, line)
            if read != "pass":
                placed = _placed(report_input, reason, at_task, at_cell, narrowed)
                found.append(GateFinding(held.id, read, at_task, at_cell, figure, placed))
    return tuple(found)


def gate_check(
    report_input: ReportInput,
    arm: str,
    *,
    task: str | None = None,
    cell: str | None = None,
    criterion: str | None = None,
) -> GateCheck:
    """Check one arm against every gate and guardrail, overall or in one task or cell, or against one of them (this
    module's account).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The arm's id.
    :ptype arm: str
    :param task: The one task to check in, by id; None to check wherever each criterion reads.
    :ptype task: str | None
    :param cell: The one target-matrix cell to check in, by id; None to check wherever each criterion reads.
    :ptype cell: str | None
    :param criterion: The one gate or guardrail to check against, by id; None for every one.
    :ptype criterion: str | None
    :return: The arm's status, with a reason per failing (an unknown figure among them) and per borderline figure.
    :rtype: GateCheck
    :raises ValueError: Both a task and a cell were named, or a task, cell or gating criterion the input does not hold.
    """
    found = gate_findings(report_input, arm, task=task, cell=cell, criterion=criterion)
    fails = tuple(finding.reason for finding in found if finding.status in ("fail", "unknown"))
    borders = tuple(finding.reason for finding in found if finding.status == "borderline")
    status: GateStatus = "fail" if fails else "borderline" if borders else "pass"
    return GateCheck(arm=arm, status=status, fails=fails, borders=borders)


class TaskCosts(dict[str, float]):
    """Each arm's cost per 1,000 (of what it counts) in one task, by arm, and the measure it was read on.

    :ivar measure: The measure the costs were read on, by key; empty where the task states no cost.
    """

    def __init__(self, costs: Mapping[str, float] | None = None, measure: str = "") -> None:
        super().__init__(costs or {})
        self.measure = measure


def task_costs(report_input: ReportInput, task_id: str) -> TaskCosts:
    """Each arm's cost per 1,000 (of what it counts) in one task (this module's Routing): what a gate matrix's cell
    prices and routing picks the cheapest by, with the measure it was read on (what the costs are per is that
    measure's).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param task_id: The task, by id.
    :ptype task_id: str
    :return: Each arm's cost per 1,000 (of what it counts) there, by arm, and its measure; empty where the task states
        no cost.
    :rtype: TaskCosts
    :raises ValueError: The input holds no such task.
    """
    task = next((held for held in report_input.tasks if held.id == task_id), None)
    if task is None:
        raise ValueError(f"the input holds no task {task_id!r}")
    for unit, merit, factor in (("usd_per_1k", None, 1.0), ("usd", "cost", 1000.0)):
        keys = [
            key
            for key, measure in report_input.measures.items()
            if measure.unit == unit and (merit is None or measure.merit_axis == merit)
        ]
        if len(keys) != 1:
            continue
        means = {arm: figures[keys[0]].mean for arm, figures in task.per_arm.items() if keys[0] in figures}
        costs = {arm: float(mean) * factor for arm, mean in means.items() if mean is not None}
        if costs:
            return TaskCosts(costs, keys[0])
    return TaskCosts()


def routing_picks(report_input: ReportInput, arms: Sequence[str]) -> dict[str, str]:
    """For each routable task, the cheapest of ``arms`` passing there and in every cross-cutting task (this module's
    Routing), by task id; a task no costed arm passes has no pick. The first in ``arms`` order wins a tie.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arms: The arms that may be picked, in the order a tie goes by.
    :ptype arms: Sequence[str]
    :return: Each routable task's pick, by task id.
    :rtype: dict[str, str]
    """
    cross_cutting = [task.id for task in report_input.tasks if task.cross_cutting]
    eligible = [
        arm for arm in arms if all(gate_check(report_input, arm, task=task).status != "fail" for task in cross_cutting)
    ]
    picks: dict[str, str] = {}
    for task in report_input.tasks:
        if not task.routable:
            continue
        costs = task_costs(report_input, task.id)
        passing = [
            arm for arm in eligible if arm in costs and gate_check(report_input, arm, task=task.id).status != "fail"
        ]
        if passing:
            picks[task.id] = min(passing, key=lambda arm: costs[arm])
    return picks
