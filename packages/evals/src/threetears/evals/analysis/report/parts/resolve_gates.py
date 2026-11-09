"""The gate family's charts: the gate matrix, the gate scatter and the config lattice.

Every status a chart here shows is the gate check's (:func:`~threetears.evals.analysis.report_gates.gate_check`), the
one GENERATOR §3 function the decision's survivors read too, so a chart never shows an arm passing that the decision
dropped; each pill states its status in words (:data:`~threetears.evals.analysis.report.parts.words.GATE_WORDS`),
so a reader without the colour (Markdown, a screen reader) reads it too. Every line a chart draws at a gate is the
gate's own threshold, read through its figure key (``criteria.<id>.threshold``): the criterion that reads the chart's
measure where the chart reads it (over the whole comparison, or in the task a panel draws), at the statistic it draws.

**gate_matrix**, a table of pills, reasons and costs:

==========  ==============================  =========================  ===============================================
source      rows                            columns                    cell
==========  ==============================  =========================  ===============================================
tasks       each task, a cross-cutting one  the arms                   the check in the task, its binding reasons
            flagged as applying to all                                 (the failures, else what is borderline); the
            traffic                                                    task's cost per 1,000 (of what it counts); the
                                                                       routing pick
                                                                       (:func:`~threetears.evals.analysis.report_gates.routing_picks`)
                                                                       on a routable task. A task no gate is read in
                                                                       has no status: its cell is neutral, with the
                                                                       query's figures there
readings    each aspect with readings       the candidates             GENERATOR §3 by the direction of the change
                                                                       (worse bad, better good, not separated
                                                                       neutral), the margin verdict in the words; the
                                                                       change and its interval, and each floor on it
                                                                       that reads another rate than the pass rate
floors      each floor                      the candidates             held good, unclear borderline, broken bad; the
                                                                       worst drop against the floor's margin
measures,   each gate and guardrail (those  the arms, or each config   the check against that one criterion, wherever
configs     on the query's measures), then  named by its levels        it reads. A measure no gate reads has no
            each measure the query names                               status, as a task no gate reads has none: its
            (every one, where it names                                 cells are neutral, with each arm's figure
            none and no gate is declared)
            that no gate reads
==========  ==============================  =========================  ===============================================

**gate_scatter** draws each arm (each config) at the figures the two gates read, the gates' own statistics (a p95
for a latency budget), with both gates' lines and the region meeting both shaded and named; a point that fails the
gate check is hollow, and its key says so; every point carries its arm's tag, its label where it has no other. Its
axes are the two measures the query names, else the two the gates read; the one a gate reads as a floor is up. A
config's point takes its first factor's colour, which a legend names. No line joins one model's configs: the input
does not say a factor's levels are ordered, and a line over unordered categories draws a trend that is not there
(GENERATOR §5.1). Over tasks it panels by task, each panel at that task's gates. An x axis whose values span more than ten times is drawn on a log scale, which a caption says
(GENERATOR §5.2).

**lattice** tiles a factorial grid: rows the first factor's levels, columns the second's (every combination of the rest,
where there are more), each tile the config's figure on the query's first measure, its cost per 1,000 (of what the cost
counts), its latency's p95 and its gate status; where no gate or guardrail is declared, no gate reads a config, and its
tile carries no status and no pill, since a pill on every tile saying nothing is gated is noise. A gate matrix's ungated
cell says "Not gated here" only where some gate is declared elsewhere, which is news; where none is, its cell carries no
word, only its figures.

Dot strips, range bars and the frontier scatter, which draw rates against the same lines, are
:mod:`~threetears.evals.analysis.report.parts.resolve_rates`'.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.quantities import PERCENT, with_unit
from threetears.evals.analysis.report.charts import (
    ChartColumn,
    ChartReference,
    ChartShape,
    ChartStatus,
    GateCell,
    GateMatrixChart,
    GateRow,
    GateScatterChart,
    GateScatterPanel,
    LatticeCell,
    LatticeChart,
    LatticeRow,
    ScatterPoint,
)
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.report.cost_unit import priced_in_part, per_count, per_noun
from threetears.evals.analysis.report.figure_keys import cost_basis, resolve, figure_unit
from threetears.evals.analysis.report.parts.charting import (
    ChartContext,
    Drawn,
    KindResolver,
    chart_block,
    money_drawn,
    criterion_label,
    factor_legend,
    with_hollow_key,
    factor_token,
    figure_stat,
    figure_kind,
    measure_words,
    money_words,
    point_labels,
    slice_name,
    stat_value,
)
from threetears.evals.analysis.report.parts.look import (
    GATE_STATUS,
    Quantity,
    band_domain,
    log_spread,
    series_token,
)
from threetears.evals.analysis.report.parts.words import (
    CROSS_CUTTING_FLAG,
    FLOOR_ON_TEMPLATE,
    FLOOR_READS_TEMPLATE,
    FLOOR_REASON_TEMPLATE,
    FLOOR_WORDS,
    GATE_HEADERS,
    GATE_LINE_WORDS,
    GATE_PLACE_WORDS,
    GATE_WORDS,
    LOG_AXIS_NOTE,
    NO_READING,
    NOT_GATED,
    PASSES_EVERY_GATE,
    PICK_LABEL,
    REGION_LABEL,
    TASK_DETAIL_TEMPLATE,
    TASK_SHARE_TEMPLATE,
    reading_words,
)
from threetears.evals.analysis.report_gates import GateCheck, gate_check, routing_picks, task_costs
from threetears.evals.analysis.report_input import (
    Aspect,
    Criterion,
    FloorRead,
    Reading,
    ReportInput,
    Stat,
    Task,
)
from threetears.evals.analysis.report_margins import criterion_place, flip_margin
from threetears.evals.analysis.report_rows import query_aspect, query_rows

__all__ = ["RESOLVERS"]

#: The criterion kinds that gate an option (GENERATOR §3).
_GATING = frozenset({"gate", "guardrail"})

#: A reading's direction to its pill (GENERATOR §3: by the direction of the change, never by the margin).
_READING_STATUS: dict[str, ChartStatus] = {"worse": "bad", "better": "good", "": "neutral"}

#: A floor's status to its pill.
_FLOOR_STATUS: dict[str, ChartStatus] = {"held": "good", "unclear": "warn", "broken": "bad"}

#: The room a log axis leaves past its least and its largest value, as factors.
_LOG_PAD = (0.8, 1.25)


# --- what every kind here reads -------------------------------------------------------------------------------------


def _ungated(gating: Sequence[Criterion]) -> str:
    """An ungated cell's word: "Not gated here" where some gate is declared elsewhere, else none (no pill), since a
    word on every cell saying nothing is gated is noise."""
    return NOT_GATED if gating else ""


def _gating(report_input: ReportInput) -> list[Criterion]:
    return [criterion for criterion in report_input.criteria if criterion.kind in _GATING]


def _reads_in(criterion: Criterion, task: str | None) -> bool:
    """Whether a criterion reads its figure where a chart does: over the whole comparison (``task`` None), or in
    ``task``."""
    place = criterion_place(criterion)
    if task is None:
        return place == "overall"
    return place == "task" and criterion.task in (None, task)


def _criterion_line(context: ChartContext, criterion: Criterion, drawn: Quantity) -> ChartReference:
    """A criterion's threshold as a chart's line, in display units: a floor where higher is better, else a budget."""
    figure = resolve(f"criteria.{criterion.id}.threshold", context.report_input)
    if not isinstance(figure.value, float):
        raise context.refuse(f"the criterion {criterion.id!r} states no single threshold")
    value = figure.value * drawn.factor
    kind: Literal["floor", "budget"] = "floor" if criterion.direction == "higher_is_better" else "budget"
    return ChartReference(value=value, label=criterion_label(context, criterion, value, drawn), kind=kind)


def _arm_name(context: ChartContext, arm: str) -> str:
    return context.look.arm(arm).name if arm in context.look.arms else arm


def _config_levels(report_input: ReportInput, arm: str) -> dict[str, str]:
    """A config's level of each factor, by factor id; empty for an arm that ran no config."""
    config = next((config for config in report_input.configs if config.arm == arm), None)
    return dict(config.levels) if config is not None else {}


def _level_name(report_input: ReportInput, factor_id: str, level_id: str) -> str:
    factor = next(factor for factor in report_input.factors if factor.id == factor_id)
    level = next((level for level in factor.levels if level.id == level_id), None)
    return (level.name if level is not None else None) or level_id


def _config_name(report_input: ReportInput, arm: str) -> str:
    """A config as its levels name it, in the factors' order (``Brisk-S · Low · 1 retry``)."""
    levels = _config_levels(report_input, arm)
    names = [
        _level_name(report_input, factor.id, levels[factor.id])
        for factor in report_input.factors
        if factor.id in levels
    ]
    return " · ".join(names) or arm


def _gate_cell(check: GateCheck, *, cost: float | None = None, pick: bool = False) -> GateCell:
    """A gate check as a matrix cell: its pill in words, its binding reasons (the failures, else the borderline)."""
    reasons = list(check.fails or check.borders)
    return GateCell(
        status=GATE_STATUS[check.status],
        label=GATE_WORDS[check.status],
        reasons=reasons or None,
        cost=cost,
        pick=True if pick else None,
    )


def _check_lines(check: GateCheck) -> list[str]:
    """A tooltip's lines for a gate check: each reason, else that it passes every gate."""
    return [*check.fails, *check.borders] or [PASSES_EVERY_GATE]


# --- gate_matrix ----------------------------------------------------------------------------------------------------


def _task_detail(context: ChartContext, task: Task) -> str | None:
    parts = []
    if task.n_cases is not None:
        parts.append(TASK_DETAIL_TEMPLATE.format(cases=context.forms.chart_count(task.n_cases).text))
    if task.share:
        parts.append(TASK_SHARE_TEMPLATE.format(share=with_unit(float(task.share) * 100, PERCENT)))
    return " · ".join(parts) or None


def _figures_drawn(context: ChartContext, key: str, stats: Iterable[Stat]) -> Quantity:
    """How one measure's figures are written in a matrix's cells: from every figure the matrix states of it, so its
    money reads in one form (#993)."""
    reads = [stat_value(context.report_input, key, stat) for stat in stats]
    figures = [read[0] for read in reads if read is not None]
    unit, kind = figure_unit(context.report_input, key), figure_kind(context, key)
    return context.forms.chart_quantity_of(unit, figures, measure=key, kind=kind)


def _figure_words(context: ChartContext, key: str, stat: Stat, drawn: Quantity) -> str | None:
    """One figure in a cell's words: the measure's reader name, then its value in its unit, as its matrix draws the
    measure (``drawn``)."""
    read = stat_value(context.report_input, key, stat)
    if read is None:
        return None
    statistic = f" ({read[1]})" if read[1] != "mean" else ""
    said = context.forms.chart_note(read[0] * drawn.factor, drawn, measure=key, kind=figure_kind(context, key)).text
    return f"{money_words(context.report_input, key, drawn)}{statistic} {said}"


def _task_matrix(context: ChartContext) -> Drawn:
    report_input = context.report_input
    tasks = [row.item for row in context.rows() if isinstance(row.item, Task)]
    if not tasks:
        raise context.refuse("it selects no task")
    allowed = set(context.query.arms) or set(context.look.arms)
    arms = context.arm_order(arm for task in tasks for arm in task.per_arm if arm in allowed)
    picks = routing_picks(report_input, arms)
    gating = _gating(report_input)
    rows: list[GateRow] = []
    costs_drawn: list[float] = []
    cost_measures: set[str] = set()
    measures_drawn = {
        key: _figures_drawn(
            context, key, (task.per_arm[arm][key] for task in tasks for arm in arms if key in task.per_arm.get(arm, {}))
        )
        for key in context.query.measures
    }
    for task in tasks:
        costs = task_costs(report_input, task.id)
        if any(arm in costs for arm in arms):
            cost_measures.add(costs.measure)
        gated = any(_reads_in(criterion, task.id) for criterion in gating)
        cells = []
        for arm in arms:
            cost = costs.get(arm)
            if cost is not None:
                costs_drawn.append(cost)
            # A cost over part of the arm's calls is a smaller total, never stated bare.
            share = priced_in_part(report_input, arm) if cost is not None else None
            if gated:
                check = gate_check(report_input, arm, task=task.id, forms=context.forms)
                cell = _gate_cell(check, cost=cost, pick=picks.get(task.id) == arm)
                if share is not None:
                    cell = cell.model_copy(update={"reasons": [*(cell.reasons or []), share]})
                cells.append(cell)
                continue
            figures = task.per_arm.get(arm, {})
            # The cost the cell prints beside its reasons is no figure to say twice.
            said = [
                _figure_words(context, key, figures[key], measures_drawn[key])
                for key in context.query.measures
                if key in figures and key != costs.measure
            ]
            reasons = [line for line in said if line] + ([share] if share is not None else [])
            cells.append(GateCell(status="neutral", label=_ungated(gating), reasons=reasons or None, cost=cost))
        rows.append(
            GateRow(
                label=slice_name(context, task),
                cells=cells,
                detail=_task_detail(context, task),
                flag=CROSS_CUTTING_FLAG if task.cross_cutting else None,
            )
        )
    # The costs are their tasks' figures on a cost measure, of one noun (refused below otherwise): named by the first.
    cost_measure = min(cost_measures, default="")
    priced = context.forms.chart_quantity("usd_per_1k", costs_drawn, measure=cost_measure, kind="measure")
    if priced.factor != 1:
        # Costs under a cent per 1,000 are drawn per 1,000,000 (look's money), every cell alike.
        rows = [
            row.model_copy(
                update={
                    "cells": [
                        cell.model_copy(update={"cost": cell.cost * priced.factor}) if cell.cost is not None else cell
                        for cell in row.cells
                    ]
                }
            )
            for row in rows
        ]
    # Each task's costs are its own figures on a cost measure: what they are per is that measure's, not the page's.
    nouns = {per_noun(report_input.measures[key]) for key in cost_measures}
    if len(nouns) > 1:
        raise context.refuse(
            f"its tasks' costs are read on measures counting different things ({', '.join(sorted(cost_measures))}), "
            "and a matrix prints one cost unit"
        )
    data = GateMatrixChart(
        columns=[ChartColumn(label=_arm_name(context, arm)) for arm in arms],
        rows=rows,
        row_header=GATE_HEADERS["tasks"],
        cost_fmt=priced.fmt if costs_drawn else None,
        cost_unit=per_count(next(iter(nouns)), priced.counted) if costs_drawn else None,
        pick_label=PICK_LABEL if picks else None,
    )
    # The costs' count is the matrix's own cost unit words; each measure's money says it in a line under the chart.
    return money_drawn(context, data, money=list(measures_drawn.items()))


def _direction(reading: Reading) -> str:
    """GENERATOR §3: worse where the interval lies under zero, better where above, else not separated."""
    if reading.within_interval is None:
        return ""
    low, high = (float(end) for end in reading.within_interval)
    return "worse" if high < 0 else "better" if low > 0 else ""


def _reading_aspects(context: ChartContext) -> list[Aspect]:
    report_input = context.report_input
    if context.query.aspect:
        aspect = query_aspect(context.query.aspect, report_input)
        return [aspect] if aspect is not None else []
    return [aspect for aspect in report_input.aspects if aspect.readings]


def _reading_cell(context: ChartContext, aspect: Aspect, reading: Reading | None) -> GateCell:
    if reading is None:
        return GateCell(status="neutral", label=NO_READING)
    direction = _direction(reading)
    margin = reading.margin if reading.margin is not None else aspect.margin
    flip = flip_margin(reading)
    # Within the margin by the decision's own rule: the reading flips at its worst drop, never below zero.
    within = flip < float(margin) if margin is not None and flip is not None else None
    # A reading is the pipeline's change in the pass rate.
    forms, rate = context.forms, TRIAL_PASS_NAME
    points = forms.chart_quantity("points", change=True, measure=rate, kind="trial")
    said = forms.chart_note(float(reading.change), points, measure=rate, kind="trial").text
    if reading.within_interval is not None:
        low, high = (float(end) for end in reading.within_interval)
        interval = forms.chart_interval(low, high, points, measure=rate, kind="trial").text
        said += f" ({interval})"
    reasons = [said]
    for floor in context.report_input.floors:
        if floor.aspect == aspect.id and floor.arm == reading.arm and floor.reading:
            status = FLOOR_WORDS.get(floor.status, floor.status)
            reasons.append(
                FLOOR_ON_TEMPLATE.format(
                    rate=measure_words(context.report_input, floor.reading, prose=True), status=status
                )
            )
    return GateCell(status=_READING_STATUS[direction], label=reading_words(direction, within), reasons=reasons)


def _reading_matrix(context: ChartContext) -> Drawn:
    aspects = _reading_aspects(context)
    selected: dict[str, dict[str, Reading]] = {}
    for aspect in aspects:
        query = context.query.model_copy(update={"aspect": aspect.id})
        held = {row.arm: row.item for row in query_rows(query, context.report_input) if isinstance(row.item, Reading)}
        if held:
            selected[aspect.id] = held
    if not selected:
        raise context.refuse("it selects no reading")
    arms = context.arm_order(arm for held in selected.values() for arm in held)
    rows = [
        GateRow(
            label=context.look.reader("aspect", aspect.id),
            cells=[_reading_cell(context, aspect, selected[aspect.id].get(arm)) for arm in arms],
        )
        for aspect in aspects
        if aspect.id in selected
    ]
    data = GateMatrixChart(
        columns=[ChartColumn(label=_arm_name(context, arm)) for arm in arms],
        rows=rows,
        row_header=GATE_HEADERS["readings"],
    )
    return Drawn(chart_block(context, data))


def _floor_label(context: ChartContext, floor: FloorRead) -> str:
    """A floor's row: its aspect's reader name, where the floor's key opens with it, else the key in words."""
    aspect = next((aspect for aspect in context.report_input.aspects if aspect.id == floor.aspect), None)
    if aspect is not None and floor.floor.startswith(f"{aspect.id}/"):
        return context.look.reader("aspect", aspect.id)
    return context.look.reader("criterion", floor.floor)


def _floor_cell(context: ChartContext, floor: FloorRead | None) -> GateCell:
    if floor is None:
        return GateCell(status="neutral", label=NO_READING)
    # A floor's drop and margin are points of the rate it is read on.
    rate = floor.reading or floor.speaks_for or ""
    kind = figure_kind(context, rate)
    points = context.forms.chart_quantity("points", change=True, measure=rate, kind=kind)
    reasons = []
    if floor.worst_drop is not None and floor.margin is not None:
        drop = context.forms.chart_note(float(floor.worst_drop), points, measure=rate, kind=kind).text
        margin = context.forms.chart_note(float(floor.margin), points, measure=rate, kind=kind).text
        reasons.append(FLOOR_REASON_TEMPLATE.format(drop=drop, margin=margin))
    if floor.reading:
        reasons.append(FLOOR_READS_TEMPLATE.format(rate=measure_words(context.report_input, floor.reading, prose=True)))
    return GateCell(
        status=_FLOOR_STATUS.get(floor.status, "neutral"),
        label=FLOOR_WORDS.get(floor.status, floor.status),
        reasons=reasons or None,
    )


def _floor_matrix(context: ChartContext) -> Drawn:
    floors = [row.item for row in context.rows() if isinstance(row.item, FloorRead)]
    if not floors:
        raise context.refuse("it selects no floor")
    keys = list(dict.fromkeys(floor.floor for floor in floors))
    arms = context.arm_order(floor.arm for floor in floors)
    by_place = {(floor.floor, floor.arm): floor for floor in floors}
    first = {key: next(floor for floor in floors if floor.floor == key) for key in keys}
    rows = [
        GateRow(
            label=_floor_label(context, first[key]),
            cells=[_floor_cell(context, by_place.get((key, arm))) for arm in arms],
        )
        for key in keys
    ]
    data = GateMatrixChart(
        columns=[ChartColumn(label=_arm_name(context, arm)) for arm in arms],
        rows=rows,
        row_header=GATE_HEADERS["floors"],
    )
    return Drawn(chart_block(context, data))


def _gate_detail(context: ChartContext, criterion: Criterion) -> str:
    """A gate row's line: where it sits, by its direction, and where it is read beyond the whole comparison."""
    line = context.forms.gate_line(context.report_input, criterion).text
    said = GATE_LINE_WORDS[criterion.direction].format(line=line)
    if criterion.statistic != "mean":
        said = f"{criterion.statistic} {said}"
    place = criterion_place(criterion)
    if place == "cell":
        said += f" {GATE_PLACE_WORDS['cell']}"
    elif place == "task" and criterion.task is None:
        said += f" {GATE_PLACE_WORDS['task']}"
    elif place == "task" and criterion.task is not None:
        said += " " + GATE_PLACE_WORDS["one_task"].format(task=context.look.reader("task", criterion.task))
    return said


def _criterion_matrix(context: ChartContext) -> Drawn:
    report_input = context.report_input
    measures = set(context.query.measures)
    criteria = [criterion for criterion in _gating(report_input) if not measures or criterion.measure in measures]
    rows_read = context.rows()
    gated = {criterion.measure for criterion in _gating(report_input)}
    named = list(context.query.measures) or ([] if criteria else list(dict.fromkeys(row.key for row in rows_read)))
    ungated = [key for key in named if key not in gated]
    if not criteria and not ungated:
        raise context.refuse("it selects no gate, guardrail or measure to show")
    selected = list(dict.fromkeys(row.arm for row in rows_read if row.arm))
    if context.query.source == "configs":
        order = [config.arm for config in report_input.configs]
        arms = [arm for arm in order if arm in selected] if not context.query.arms else context.arm_order(selected)
        names = [_config_name(report_input, arm) for arm in arms]
    else:
        arms = context.arm_order(selected)
        names = [_arm_name(context, arm) for arm in arms]
    if not arms:
        raise context.refuse("it selects no arm")
    rows = [
        GateRow(
            label=context.look.reader("criterion", criterion.id),
            detail=_gate_detail(context, criterion),
            cells=[
                _gate_cell(gate_check(report_input, arm, criterion=criterion.id, forms=context.forms)) for arm in arms
            ],
        )
        for criterion in criteria
    ]
    figures = {(row.key, row.arm): row.item for row in rows_read if isinstance(row.item, Stat)}
    money: list[tuple[str, Quantity]] = []
    for key in ungated:
        cells = []
        drawn = _figures_drawn(context, key, (stat for (held, _), stat in figures.items() if held == key))
        for arm in arms:
            stat = figures.get((key, arm))
            said = _figure_words(context, key, stat, drawn) if stat is not None else None
            cells.append(
                GateCell(status="neutral", label=_ungated(_gating(report_input)), reasons=[said] if said else None)
            )
        rows.append(GateRow(label=money_words(report_input, key, drawn), cells=cells))
        money.append((key, drawn))
    data = GateMatrixChart(
        columns=[ChartColumn(label=name) for name in names],
        rows=rows,
        row_header=GATE_HEADERS[context.query.source],
    )
    return money_drawn(context, data, money=money)


def _gate_matrix(context: ChartContext) -> Drawn:
    source = context.query.source
    if source == "tasks":
        return _task_matrix(context)
    if source == "readings":
        return _reading_matrix(context)
    if source == "floors":
        return _floor_matrix(context)
    if source in ("measures", "configs"):
        return _criterion_matrix(context)
    raise context.refuse("a gate matrix draws tasks, readings, floors, measures or configs")


# --- the scatters ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Axis:
    """One axis of a scatter: how it is drawn, its title, and the measure it draws."""

    drawn: Quantity
    title: str
    key: str


@dataclass(frozen=True)
class _Point:
    arm: str
    x: float
    y: float
    check: GateCheck


@dataclass
class _Panel:
    """One panel's points and lines (a whole chart's, where it draws no panels)."""

    title: str | None
    points: list[_Point] = field(default_factory=list)
    y_line: ChartReference | None = None
    x_line: ChartReference | None = None


def _axis_keys(context: ChartContext) -> tuple[str, str]:
    """The gate scatter's two measures, x then y: the two its query names, else the two its gates read (over tasks,
    the two every task reads); up is the one a gate reads as a floor (higher is better), across the one read as a
    budget."""
    report_input = context.report_input
    # Over tasks, the gates every task reads (a task's own gates draw in its panel only where they are the axes').
    place = "task" if context.query.source == "tasks" else "overall"
    gating = [
        criterion
        for criterion in _gating(report_input)
        if criterion_place(criterion) == place and criterion.task is None
    ]
    keys = list(dict.fromkeys(context.query.measures))
    if len(keys) != 2:
        keys = list(dict.fromkeys(criterion.measure for criterion in gating))
    if len(keys) != 2:
        raise context.refuse(f"a gate scatter draws two measures, and it reads {keys}")
    floors = {criterion.measure for criterion in gating if criterion.direction == "higher_is_better"}
    first, second = keys
    return (second, first) if first in floors and second not in floors else (first, second)


def _axis_criterion(context: ChartContext, key: str, task: str | None) -> Criterion:
    """The one gate or guardrail reading ``key`` where a gate scatter reads it."""
    found = [
        criterion
        for criterion in _gating(context.report_input)
        if criterion.measure == key and _reads_in(criterion, task)
    ]
    if len(found) != 1:
        where = f" in {task!r}" if task is not None else ""
        raise context.refuse(f"a gate scatter draws the one gate on each axis, and {len(found)} read {key!r}{where}")
    return found[0]


def _axis_title(context: ChartContext, key: str, statistic: str, drawn: Quantity) -> str:
    name = money_words(context.report_input, key, drawn)
    return f"{name} ({statistic})" if statistic != "mean" else name


def _panel_points(
    context: ChartContext, keys: tuple[str, str], statistics: tuple[str, str], task: Task | None
) -> list[_Point]:
    """Each arm's point: its two figures, overall or in one task, at each gate's statistic."""
    report_input = context.report_input
    x_key, y_key = keys
    if task is not None:
        allowed = set(context.query.arms) or set(context.look.arms)
        arms = [arm for arm in task.per_arm if arm in allowed]
        figures = {(key, arm): task.per_arm[arm][key] for arm in arms for key in keys if key in task.per_arm[arm]}
    else:
        arms = list(dict.fromkeys(row.arm for row in context.rows() if isinstance(row.item, Stat)))
        figures = {
            (key, arm): stat
            for arm in arms
            for key in keys
            if (stat := figure_stat(report_input, key, arm, context.query.aspect)) is not None
        }
    points: list[_Point] = []
    for arm in arms:
        if (x_key, arm) not in figures or (y_key, arm) not in figures:
            continue
        x, y = getattr(figures[x_key, arm], statistics[0]), getattr(figures[y_key, arm], statistics[1])
        if x is None or y is None:
            continue
        check = gate_check(report_input, arm, task=task.id if task is not None else None, forms=context.forms)
        points.append(_Point(arm, float(x), float(y), check))
    return points


def _axis_domain(values: Sequence[float], drawn: Quantity, *, log: bool) -> tuple[float, float]:
    if log:
        return min(values) * _LOG_PAD[0], max(values) * _LOG_PAD[1]
    return band_domain(values, drawn)


def _scatter_point(context: ChartContext, point: _Point, axes: tuple[_Axis, _Axis], count: int) -> ScatterPoint:
    configs = context.query.source == "configs"
    x_axis, y_axis = axes
    x, y = point.x * x_axis.drawn.factor, point.y * y_axis.drawn.factor
    name = _config_name(context.report_input, point.arm) if configs else _arm_name(context, point.arm)
    label, narrow = point_labels(context, point.arm, name, count)
    color: str | None = factor_token(context.report_input, point.arm) if configs else None
    shape: ChartShape | None = None
    if color is None and point.arm in context.look.arms:
        look = context.look.arm(point.arm)
        color, shape = look.token, look.shape
    x_said = context.forms.chart_note(x, x_axis.drawn, measure=x_axis.key, kind=figure_kind(context, x_axis.key)).text
    y_said = context.forms.chart_note(y, y_axis.drawn, measure=y_axis.key, kind=figure_kind(context, y_axis.key)).text
    tip = [name, f"{x_axis.title} {x_said} · {y_axis.title} {y_said}"]
    return ScatterPoint(
        id=point.arm,
        x=x,
        y=y,
        label=label,
        narrow_label=narrow,
        color=color,
        shape=shape,
        hollow=True if point.check.status == "fail" else None,
        tip=[*tip, *_check_lines(point.check)],
        tag=context.look.arm(point.arm).tag if point.arm in context.look.arms else None,
    )


def _gate_scatter(context: ChartContext) -> Drawn:
    if context.query.source not in ("measures", "judged", "configs", "tasks"):
        raise context.refuse("a gate scatter draws measures, judged, configs or tasks")
    x_key, y_key = _axis_keys(context)
    slices = (
        [row.item for row in context.rows() if isinstance(row.item, Task)] if context.query.source == "tasks" else []
    )
    if context.query.source == "tasks" and not slices:
        raise context.refuse("it selects no task")
    places: list[Task | None] = list(slices) or [None]
    panels: list[_Panel] = []
    x_values: list[float] = []
    y_values: list[float] = []
    statistics: tuple[str, str] = ("mean", "mean")
    criteria = {}
    for task in places:
        where = task.id if task is not None else None
        x_criterion = _axis_criterion(context, x_key, where)
        y_criterion = _axis_criterion(context, y_key, where)
        statistics = (x_criterion.statistic, y_criterion.statistic)
        criteria[where] = (x_criterion, y_criterion, _panel_points(context, (x_key, y_key), statistics, task))
    # Each axis in one form, from every point it draws (#993).
    points_drawn = [point for _, _, held in criteria.values() for point in held]
    x_unit, y_unit = figure_unit(context.report_input, x_key), figure_unit(context.report_input, y_key)
    x_kind, y_kind = figure_kind(context, x_key), figure_kind(context, y_key)
    x_drawn = context.forms.chart_quantity_of(x_unit, [point.x for point in points_drawn], measure=x_key, kind=x_kind)
    y_drawn = context.forms.chart_quantity_of(y_unit, [point.y for point in points_drawn], measure=y_key, kind=y_kind)
    for task in places:
        where = task.id if task is not None else None
        panel = _Panel(title=slice_name(context, task) if task is not None else None)
        x_criterion, y_criterion, points = criteria[where]
        statistics = (x_criterion.statistic, y_criterion.statistic)
        panel.x_line = _criterion_line(context, x_criterion, x_drawn)
        panel.y_line = _criterion_line(context, y_criterion, y_drawn)
        panel.points = points
        x_values += [point.x * x_drawn.factor for point in panel.points]
        y_values += [point.y * y_drawn.factor for point in panel.points]
        x_values += [panel.x_line.value] if panel.x_line is not None else []
        y_values += [panel.y_line.value] if panel.y_line is not None else []
        panels.append(panel)
    count = max(len(panel.points) for panel in panels)
    if not count:
        raise context.refuse(f"no arm holds a figure on both {x_key!r} and {y_key!r}")
    x_axis = _Axis(x_drawn, _axis_title(context, x_key, statistics[0], x_drawn), x_key)
    y_axis = _Axis(y_drawn, _axis_title(context, y_key, statistics[1], y_drawn), y_key)
    x_log = log_spread(x_values)
    x_domain = _axis_domain(x_values, x_drawn, log=x_log)
    y_domain = context.spec.domain or _axis_domain(y_values, y_drawn, log=False)
    drawn_points = [
        [_scatter_point(context, point, (x_axis, y_axis), count) for point in panel.points] for panel in panels
    ]
    arms = [point.arm for panel in panels for point in panel.points]
    legend = factor_legend(context.report_input, arms) if context.query.source == "configs" else None
    common: dict[str, object] = {
        "x_domain": (x_domain[0], x_domain[1]),
        "y_domain": (y_domain[0], y_domain[1]),
        "x_fmt": x_drawn.fmt,
        "y_fmt": y_drawn.fmt,
        "x_log": True if x_log else None,
        "x_title": x_axis.title,
        "y_title": y_axis.title,
        "emphasis": context.emphasis,
        "legend": with_hollow_key(legend, (point for points in drawn_points for point in points)),
    }
    notes = (LOG_AXIS_NOTE.format(axis=x_axis.title),) if x_log else ()
    data: GateScatterChart
    if len(panels) == 1:
        only = panels[0]
        whole = {"points": drawn_points[0], "reference": only.y_line, "x_reference": only.x_line}
        data = GateScatterChart.model_validate({**common, **whole, "region_label": REGION_LABEL})
    else:
        gate_panels = [
            GateScatterPanel(title=panel.title, points=points, reference=panel.y_line, x_reference=panel.x_line)
            for panel, points in zip(panels, drawn_points, strict=True)
        ]
        data = GateScatterChart.model_validate({**common, "region_label": REGION_LABEL, "panels": gate_panels})
    return money_drawn(context, data, notes, money=[(x_key, x_drawn), (y_key, y_drawn)])


# --- lattice --------------------------------------------------------------------------------------------------------


def _latency_key(report_input: ReportInput) -> str | None:
    """The latency a tile states: the measure a criterion reads at its p95, else the one in ms or s with p95s."""
    read = list(dict.fromkeys(criterion.measure for criterion in report_input.criteria if criterion.statistic == "p95"))
    if len(read) == 1:
        return read[0]
    timed = [
        key
        for key, measure in report_input.measures.items()
        if measure.unit in ("ms", "s") and any(stat.p95 is not None for stat in measure.per_arm.values())
    ]
    return timed[0] if len(timed) == 1 else None


def _lattice(context: ChartContext) -> Drawn:
    report_input = context.report_input
    if context.query.source != "configs":
        raise context.refuse("a lattice draws configs")
    factors = report_input.factors
    if len(factors) < 2:
        raise context.refuse("a lattice crosses two factors, and the input declares fewer")
    keys = list(context.query.measures) or list(dict.fromkeys(row.key for row in context.rows()))
    if not keys:
        raise context.refuse("it selects no measure")
    key = keys[0]
    rows_factor, column_factors = factors[0], factors[1:]
    allowed = set(context.query.arms) or {config.arm for config in report_input.configs}
    by_levels = {
        tuple(config.levels.get(factor.id, "") for factor in factors): config.arm
        for config in report_input.configs
        if config.arm in allowed
    }
    columns = [combination for combination in product(*(factor.levels for factor in column_factors))]
    latency_key = _latency_key(report_input)
    latency_unit = figure_unit(report_input, latency_key) if latency_key is not None else ""
    values: list[float] = []
    costs: list[float] = []
    tiles: list[list[tuple[str, float, float | None, float | None]]] = []
    unit = figure_unit(report_input, key)
    for level in rows_factor.levels:
        row = []
        for combination in columns:
            arm = by_levels.get((level.id, *(one.id for one in combination)))
            if arm is None:
                where = " · ".join(one.name or one.id for one in (level, *combination))
                raise context.refuse(f"its grid has no config at {where}")
            stat = figure_stat(report_input, key, arm, context.query.aspect)
            read = stat_value(report_input, key, stat) if stat is not None else None
            if read is None:
                raise context.refuse(f"the config {arm!r} holds no figure on {key!r}")
            cost = _cost(report_input, arm)
            latency = None
            if latency_key is not None:
                held = report_input.measures[latency_key].per_arm.get(arm)
                latency = float(held.p95) if held is not None and held.p95 is not None else None
            row.append((arm, read[0], cost, latency))
            values.append(read[0])
            if cost is not None:
                costs.append(cost)
        tiles.append(row)
    forms, kind = context.forms, figure_kind(context, key)
    cost_key, cost_kind = _cost_measure(report_input)
    drawn = forms.chart_quantity_of(unit, values, measure=key, kind=kind)  # the format follows the drawn sizes
    priced = forms.chart_quantity("usd_per_1k", costs, measure=cost_key, kind=cost_kind)
    timed = (
        forms.chart_quantity(latency_unit, measure=latency_key, kind=figure_kind(context, latency_key))
        if latency_key is not None
        else None
    )
    # A config no gate reads has no status: its tile carries no pill, never a pass where nothing was gated.
    gated = bool(_gating(report_input))
    lattice_rows = []
    for place, (level, row) in enumerate(zip(rows_factor.levels, tiles, strict=True)):
        cells = []
        for arm, value, cost, latency in row:
            check = gate_check(report_input, arm, forms=context.forms) if gated else None
            shown = value * drawn.factor
            said = forms.chart_note(shown, drawn, measure=key, kind=kind).text
            tip = [_config_name(report_input, arm), f"{money_words(report_input, key, drawn)} {said}"]
            if cost is not None:
                share = priced_in_part(report_input, arm)
                priced_at = forms.chart_note(cost * priced.factor, priced, measure=cost_key, kind=cost_kind).text
                tip.append(
                    f"{priced_at} {per_count(cost_basis(report_input).noun, priced.counted)}"
                    + (f" ({share})" if share is not None else "")
                )
            if latency is not None and timed is not None:
                latency_kind = figure_kind(context, latency_key or "")
                timed_at = forms.chart_note(latency * timed.factor, timed, measure=latency_key or "", kind=latency_kind)
                tip.append(f"p95 {timed_at.text}")
            cells.append(
                LatticeCell(
                    id=arm,
                    value=shown,
                    cost=cost * priced.factor if cost is not None else None,
                    latency=latency * timed.factor if latency is not None and timed is not None else None,
                    status=GATE_STATUS[check.status] if check is not None else None,
                    status_label=GATE_WORDS[check.status] if check is not None else None,
                    tip=[*tip, *_check_lines(check)] if check is not None else tip,
                )
            )
        lattice_rows.append(LatticeRow(label=level.name or level.id, color=series_token(place), cells=cells))
    shown_values = [value * drawn.factor for value in values]
    bar = context.spec.domain or band_domain(shown_values, drawn)
    data = LatticeChart(
        columns=[ChartColumn(label=" · ".join(one.name or one.id for one in combination)) for combination in columns],
        rows=lattice_rows,
        fmt=drawn.fmt,
        bar_domain=(bar[0], bar[1]),
        cost_fmt=priced.fmt if costs else None,
        cost_unit=per_count(cost_basis(report_input).noun, priced.counted) if costs else None,
        latency_fmt=timed.fmt if timed is not None else None,
        emphasis=context.emphasis,
    )
    # The costs' count is the matrix's own cost unit words; the measure's money says it in a line under the chart.
    return money_drawn(context, data, money=[(key, drawn)])


def _cost_measure(report_input: ReportInput) -> tuple[str, str]:
    """The measure a page's cost per 1,000 is a figure of, and its kind: the cost measure it is read from, else the
    scoreboard's cost per reply; neither where the input states no cost."""
    source = cost_basis(report_input).source
    if source == _SCOREBOARD_COST:
        return _REPLY_COST, "measure"
    return (source, "measure") if source else ("", "")


#: The cost source that is the scoreboard's cost per reply, and the scoreboard figure it reads.
_SCOREBOARD_COST, _REPLY_COST = "scoreboard", "reply_cost_usd"


def _cost(report_input: ReportInput, arm: str) -> float | None:
    """An arm's cost per 1,000 of what it counts, by the page's one cost rule (``cost_per_1k.<arm>``); None where it
    has none."""
    try:
        figure = resolve(f"cost_per_1k.{arm}", report_input)
    except UnresolvableReference:
        # NOSILENT: a config whose cost no source states is an expected absence, not a failure: its tile prints no
        # cost, and the lattice's cost format reads the configs that state one.
        return None
    return figure.value if isinstance(figure.value, float) else None


#: The kinds this module resolves.
RESOLVERS: dict[str, KindResolver] = {
    "gate_matrix": _gate_matrix,
    "gate_scatter": _gate_scatter,
    "lattice": _lattice,
}
