"""A criterion's status per arm: each gate and guardrail read on an arm's figures, as arithmetic over the report input.

What the decision reads to drop an option (:func:`~threetears.evals.analysis.answer.surviving_options`), and what a
page's gate check words (:func:`~threetears.evals.analysis.report.gates.gate_findings`), so a page never shows an arm
passing that the decision dropped, and no wording can move the call.

**What is read.** Every criterion of kind ``gate`` or ``guardrail``, at the figures its scope reads
(:func:`~threetears.evals.analysis.report_margins.criterion_place` and
:func:`~threetears.evals.analysis.report_margins.criterion_figures`): overall, in its one task, in every task, or in
every cell. A read narrowed to a task reads only the criteria read in tasks, at that task's figures; one narrowed to a
cell, only those read in cells; one narrowed to a criterion, only that one, wherever it reads.

**Each figure**, on its point estimate, *fails* when it misses the threshold (under it where higher is better, over it
where lower is); else is *borderline* when its interval crosses the threshold (its ``ci``, else the value plus or minus
its ``half_width``); else *passes*. A place with no figure for the arm, or one that is not a number, is *unknown*: a
figure the input does not hold is not shown passed.

**An arm's status** (:func:`gate_status`) is ``fail`` when any figure fails or is unknown, ``borderline`` when none
does and one is borderline, else ``pass``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from threetears.evals.analysis.per_input import kept_for
from threetears.evals.analysis.report_input import Criterion, ReportInput
from threetears.evals.analysis.report_margins import (
    CriterionFigure,
    CriterionPlace,
    criterion_figures,
    criterion_place,
    criterion_places,
)

__all__ = ["CriterionRead", "CriterionStatus", "GateStatus", "criterion_reads", "gate_status"]

#: One figure against its criterion's line.
CriterionStatus = Literal["pass", "borderline", "fail", "unknown"]

#: An arm's status against the gates: every figure passes, one sits borderline and none fails, or one fails.
GateStatus = Literal["pass", "borderline", "fail"]

#: The criterion kinds that gate an option.
_GATING = frozenset({"gate", "guardrail"})


@dataclass(frozen=True)
class CriterionRead:
    """One place a gate or guardrail reads an arm's figure, and its status there.

    :ivar criterion: The criterion's id.
    :ivar status: ``pass``, ``borderline``, ``fail`` or ``unknown``.
    :ivar task: The task the figure is read in; None when it is not read in a task.
    :ivar cell: The target-matrix cell it is read in, by id; None when it is not read in a cell.
    :ivar figure: The arm's figure there; None where the input states none (``unknown``).
    """

    criterion: str
    status: CriterionStatus
    task: str | None
    cell: str | None
    figure: CriterionFigure | None


def _applies(criterion: Criterion, task: str | None, cell: str | None, only: str | None) -> bool:
    """Whether a read narrowed to ``task`` or ``cell`` (neither: unnarrowed), and to the criterion ``only`` (None: to
    none), reads ``criterion``."""
    if criterion.kind not in _GATING or only not in (None, criterion.id):
        return False
    place: CriterionPlace = criterion_place(criterion)
    if task is not None:
        return place == "task" and criterion.task in (None, task)
    if cell is not None:
        return place == "cell"
    return True


def _interval(figure: CriterionFigure) -> tuple[float, float] | None:
    if figure.ci is not None:
        return figure.ci
    if figure.half_width is not None:
        return figure.value - figure.half_width, figure.value + figure.half_width
    return None


def _status(criterion: Criterion, figure: CriterionFigure) -> CriterionStatus:
    threshold = float(criterion.threshold)
    higher = criterion.direction == "higher_is_better"
    if not (figure.value >= threshold if higher else figure.value <= threshold):
        return "fail"
    interval = _interval(figure)
    if interval is not None and not (interval[0] >= threshold if higher else interval[1] <= threshold):
        return "borderline"
    return "pass"


def criterion_reads(
    report_input: ReportInput,
    arm: str,
    *,
    task: str | None = None,
    cell: str | None = None,
    criterion: str | None = None,
) -> tuple[CriterionRead, ...]:
    """Every place one arm's figure is read against a gate or guardrail, overall or in one task or cell, or on one
    criterion, with its status there (this module's account).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The arm's id.
    :ptype arm: str
    :param task: The one task to read in, by id; None to read wherever each criterion reads.
    :ptype task: str | None
    :param cell: The one target-matrix cell to read in, by id; None to read wherever each criterion reads.
    :ptype cell: str | None
    :param criterion: The one gate or guardrail to read, by id; None for every one.
    :ptype criterion: str | None
    :return: Each place's read, in the criteria's order and each criterion's places' order; a criterion read at no
        place is one ``unknown`` read with no task and no cell.
    :rtype: tuple[CriterionRead, ...]
    :raises ValueError: Both a task and a cell were named, or a task, cell or gating criterion the input does not
        hold.
    """
    if task is not None and cell is not None:
        raise ValueError(f"a gate check reads one place at most; it was given task {task!r} and cell {cell!r}")
    if task is not None and task not in {held.id for held in report_input.tasks}:
        raise ValueError(f"the input holds no task {task!r}")
    if cell is not None and cell not in report_input.cells:
        raise ValueError(f"the input holds no cell {cell!r}")
    if criterion is not None and criterion not in {held.id for held in report_input.criteria if held.kind in _GATING}:
        raise ValueError(f"the input holds no gate or guardrail {criterion!r}")
    found: list[CriterionRead] = []
    for held in report_input.criteria:
        if not _applies(held, task, cell, criterion):
            continue
        places = [
            (at_task, at_cell)
            for at_task, at_cell in criterion_places(report_input, held)
            if (task is None or at_task == task) and (cell is None or at_cell == cell)
        ]
        if not places:
            found.append(CriterionRead(held.id, "unknown", None, None, None))
            continue
        figures = {(figure.task, figure.cell): figure for figure in criterion_figures(report_input, held, arm)}
        for at_task, at_cell in places:
            figure = figures.get((at_task, at_cell))
            if figure is None or not math.isfinite(figure.value):
                found.append(CriterionRead(held.id, "unknown", at_task, at_cell, None))
            else:
                found.append(CriterionRead(held.id, _status(held, figure), at_task, at_cell, figure))
    return tuple(found)


def gate_status(report_input: ReportInput, arm: str, *, task: str | None = None) -> GateStatus:
    """One arm's status against every gate and guardrail, overall or in one task (this module's account).

    Pure in the input, which is frozen, so it is read once per input, arm and task
    (:func:`~threetears.evals.analysis.per_input.kept_for`).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The arm's id.
    :ptype arm: str
    :param task: The one task to read in, by id; None to read wherever each criterion reads.
    :ptype task: str | None
    :return: ``fail``, ``borderline`` or ``pass``.
    :rtype: GateStatus
    :raises ValueError: The input holds no such task.
    """

    def read() -> GateStatus:
        statuses = {held.status for held in criterion_reads(report_input, arm, task=task)}
        if statuses & {"fail", "unknown"}:
            return "fail"
        return "borderline" if "borderline" in statuses else "pass"

    return kept_for(report_input).get((gate_status, arm, task), read)
