"""The Full results part (GENERATOR §2.11): every table the plan lists, each a table block of the input's figures.

The plan names the tables (``full_results``, filled by code at completion from what the input holds); this module
draws each, in the plan's order, its rows in the input's order and every arm a column or a row in the input's order:

====================  ============================================================================================
table                 rows and cells
====================  ============================================================================================
``arms_by_measure``   an arm per row, its chip first: its pass rate (passes over usable trials beside it), its mean
                      on each judged dimension, its figure on each measure (its mean, else the statistic a
                      criterion on it reads, which the cell names), and, where no measure states a cost, its cost
                      per 1,000 (of what it counts) by the package's one cost rule
``contrasts``         a corrected comparison per row: the candidate, the measure (and the aspect, where the
                      comparisons span several), the change and its interval in the measure's unit, the interval's
                      level, the adjusted p, the verdict as a pill, the confounds named on it
``strata``            a slice per row and measure: every stratum, then, as its own table, every task, each with its
                      cases and every arm's figure on each measure the slice holds
``gate_matrix_full``  each gate and guardrail at each place it is read (overall, a task, a cell) per row, its line,
                      and every arm's status there as a pill, its figure beside it
``configs``           an arm per row: its level of each factor
``cells``             a target-matrix cell per row and measure: its cases and every arm's figure
====================  ============================================================================================

**Numbers by the page's one rule.** A figure is stated as a chart states it
(:func:`~threetears.evals.analysis.report.plan_page.charting.figure_unit`,
:func:`~threetears.evals.analysis.report.plan_page.look.quantity`,
:func:`~threetears.evals.analysis.report.plan_page.look.figure_text`): a rate as a percent, a change in points, money
per reply per 1,000 replies, which the measure's name then says. A figure the input does not hold is a dash.

**A status is said in words.** Every pill's text is its status (``Passes``, ``Regressed``), so Markdown and a screen
reader, which lose the colour, lose nothing.

**The gate matrix reads the gate check.** Each cell is
:func:`~threetears.evals.analysis.report_gates.gate_check` narrowed to the row's place, over the input with that row's
one criterion, so a cell's status is the one the options table, the survivors and every gate chart read: the matrix
is the gate check spelled out, never a second reading of it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.quantities import MONEY
from threetears.evals.analysis.report.blocks import TableBlock, TableCell, TableColumn
from threetears.evals.analysis.report.charts import ChartStatus
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.cost_unit import per_count, per_noun
from threetears.evals.analysis.report.figure_keys import cost_basis, format_figure, resolve
from threetears.evals.analysis.report.plan_page.charting import figure_unit, measure_words, stat_value
from threetears.evals.analysis.report.plan_page.look import GATE_STATUS, PageLook, figure_text, quantity
from threetears.evals.analysis.report.plan_page.model import PLANNED_SECTION
from threetears.evals.analysis.report.plan_page.words import (
    CONTRAST_VERDICT_WORDS,
    COST_HEADER,
    FULL_RESULTS_HEADERS,
    FULL_RESULTS_TITLES,
    GATE_WORDS,
    LINE_TEMPLATES,
    NO_FIGURE,
    OVERALL,
    PASSES_TEMPLATE,
)
from threetears.evals.analysis.report_gates import GateCheck, gate_check
from threetears.evals.analysis.report_input import Contrast, Criterion, ReportInput, Stat
from threetears.evals.analysis.report_margins import criterion_figures, criterion_places
from threetears.evals.analysis.style import humanize
from threetears.evals.kernel.report_plan import ReportPlan

__all__ = ["full_results"]


#: Each corrected comparison's verdict's pill: a separated gain or equivalence good, a regression bad, the rest
#: neutral.
_VERDICT_STATUS: dict[str, ChartStatus] = {"improved": "good", "equivalent": "good", "regressed": "bad"}

#: The criterion kinds the gate check reads.
_GATING = frozenset({"gate", "guardrail"})

#: A slice the strata, tasks and cells tables lay out: its reader name, its cases, each arm's figures by measure.
_Slice = tuple[str, int | None, Mapping[str, Mapping[str, Stat]]]

#: A Full results table's builder: the input and the page's look in; its tables out.
_TableBuilder = Callable[[ReportInput, PageLook], list[TableBlock]]


def _table(name: str, columns: Sequence[TableColumn], rows: list[dict[str, TableCell]], order: str) -> TableBlock:
    return TableBlock(
        section=PLANNED_SECTION,
        name=name,
        title=FULL_RESULTS_TITLES[name],
        columns=list(columns),
        rows=rows,
        order=order,
        total_rows=len(rows),
    )


def _column(key: str, header: str | None = None) -> TableColumn:
    return TableColumn(key=key, header=header or FULL_RESULTS_HEADERS[key])


def _chip(look: PageLook, arm: str) -> TableCell:
    """An arm's cell, drawn as its chip; an arm the page has no look for, by its id."""
    if arm not in look.arms:
        return TableCell(value=arm)
    seen = look.arm(arm)
    return TableCell(value=seen.name, tag=seen.tag, color=seen.token)


def _arm_columns(report_input: ReportInput, look: PageLook) -> list[TableColumn]:
    """A column per arm, keyed ``arm:<id>`` and headed by its tag and name, in the input's order."""
    return [_column(f"arm:{arm.id}", look.arm(arm.id).label) for arm in report_input.arms if arm.id in look.arms]


def _named(report_input: ReportInput, key: str, *, kind: str | None = None) -> str:
    """A measure's reader name, and where its figures are money per reply, that the table states them per 1,000
    replies, as every chart does."""
    words = measure_words(report_input, key, kind=kind)
    drawn = quantity(figure_unit(report_input, key, kind=kind))
    measure = report_input.measures.get(key) if kind is None else None
    if measure is None or drawn.unit != MONEY or drawn.factor == 1.0:
        return words
    return f"{words}, {per_count(per_noun(measure))}"


def _stated(report_input: ReportInput, key: str, stat: Stat | None, *, kind: str | None = None) -> TableCell:
    """One figure on a measure or judged dimension as a cell: its mean (else the statistic a criterion on it reads,
    named beside it) in the measure's unit, by the page's number rule; a dash where the input holds none."""
    read = stat_value(report_input, key, stat) if stat is not None else None
    if read is None:
        return TableCell(value=None)
    value, statistic = read
    drawn = quantity(figure_unit(report_input, key, kind=kind), [value])
    return TableCell(value=figure_text(value * drawn.factor, drawn), flag=statistic if statistic != "mean" else None)


# --- arms by measure -------------------------------------------------------------------------------------------------


def _pass_rate(report_input: ReportInput, arm: str) -> TableCell:
    row = report_input.scoreboard.get(arm)
    if row is None or row.passed is None or not row.usable:
        return TableCell(value=None)
    drawn = quantity("share")
    return TableCell(
        value=figure_text(row.passed / row.usable * drawn.factor, drawn),
        flag=PASSES_TEMPLATE.format(passed=row.passed, usable=row.usable),
    )


def _cost(report_input: ReportInput, arm: str) -> TableCell:
    """The arm's cost per 1,000 of what it counts, by the package's one cost rule (the ``cost_per_1k`` figure key)."""
    try:
        return TableCell(value=format_figure(resolve(f"cost_per_1k.{arm}", report_input), ("usd",)))
    except UnresolvableReference:
        # NOSILENT: the key is code's, built from an arm id, so one that does not resolve means the input holds no
        # cost for the arm, which the cell says with a dash.
        return TableCell(value=None)


def _arms_by_measure(report_input: ReportInput, look: PageLook) -> list[TableBlock]:
    arms = [arm.id for arm in report_input.arms]
    passing = any(_pass_rate(report_input, arm).value is not None for arm in arms)

    judged = [key for key, held in report_input.judged.items() if any(arm in held.per_arm for arm in arms)]
    measures = [
        key
        for key, held in report_input.measures.items()
        if any(_stated(report_input, key, held.per_arm.get(arm)).value is not None for arm in arms)
    ]
    # The cost the decision reads, where no measure column already states a cost: the one cost rule's figure.
    costed = not any(quantity(figure_unit(report_input, key)).unit == MONEY for key in measures) and any(
        _cost(report_input, arm).value is not None for arm in arms
    )
    columns = [
        _column("arm"),
        *([_column("pass_rate")] if passing else []),
        *(_column(f"judged:{key}", _named(report_input, key, kind="judged")) for key in judged),
        *(_column(f"measure:{key}", _named(report_input, key)) for key in measures),
        *([_column("cost_per_1k", COST_HEADER.format(per=per_count(cost_basis(report_input).noun)))] if costed else []),
    ]
    rows = []
    for arm in arms:
        cells = {"arm": _chip(look, arm)}
        if passing:
            cells["pass_rate"] = _pass_rate(report_input, arm)
        for key in judged:
            cells[f"judged:{key}"] = _stated(
                report_input, key, report_input.judged[key].per_arm.get(arm), kind="judged"
            )
        for key in measures:
            cells[f"measure:{key}"] = _stated(report_input, key, report_input.measures[key].per_arm.get(arm))
        if costed:
            cells["cost_per_1k"] = _cost(report_input, arm)
        rows.append(cells)
    return [_table("arms_by_measure", columns, rows, "in the campaign's arm order")]


# --- the corrected comparisons ---------------------------------------------------------------------------------------


def _change(report_input: ReportInput, contrast: Contrast) -> tuple[TableCell, TableCell]:
    """A comparison's change and its interval, in its measure's unit (a rate's in points)."""
    drawn = quantity(figure_unit(report_input, contrast.measure, kind=contrast.kind), change=True)
    change = (
        TableCell(value=figure_text(float(contrast.delta) * drawn.factor, drawn))
        if contrast.delta is not None
        else TableCell(value=None)
    )
    if contrast.interval is None:
        return change, TableCell(value=None)
    low, high = (figure_text(float(end) * drawn.factor, drawn) for end in contrast.interval)
    return change, TableCell(value=f"{low} to {high}")


def _level(report_input: ReportInput, contrast: Contrast) -> str | None:
    """A comparison's interval level: its own, else its family's (the family on its aspect)."""
    if contrast.level:
        return contrast.level
    return next(
        (
            family.level
            for family in report_input.contrast_families
            if family.aspect == contrast.aspect and family.level
        ),
        None,
    )


def _verdict(contrast: Contrast) -> TableCell:
    words = CONTRAST_VERDICT_WORDS.get(contrast.verdict) or humanize("dimension", contrast.verdict).capitalize()
    return TableCell(value=words, status=_VERDICT_STATUS.get(contrast.verdict, "neutral"))


def _contrasts(report_input: ReportInput, look: PageLook) -> list[TableBlock]:
    held = report_input.contrasts
    by_aspect = len({contrast.aspect for contrast in held}) > 1
    columns = [
        _column("arm"),
        _column("measure"),
        *([_column("aspect")] if by_aspect else []),
        _column("change"),
        _column("interval"),
        _column("level"),
        _column("p_adjusted"),
        _column("verdict"),
        _column("confounds"),
    ]
    rows = []
    for contrast in held:
        change, interval = _change(report_input, contrast)
        cells = {
            "arm": _chip(look, contrast.arm),
            "measure": TableCell(value=_named(report_input, contrast.measure, kind=contrast.kind)),
            "change": change,
            "interval": interval,
            "level": TableCell(value=_level(report_input, contrast)),
            "p_adjusted": TableCell(
                value=format_number(float(contrast.p_adjusted)) if contrast.p_adjusted is not None else None
            ),
            "verdict": _verdict(contrast),
            "confounds": TableCell(
                value=", ".join(humanize("dimension", confound) for confound in contrast.confounds) or None
            ),
        }
        if by_aspect:
            cells["aspect"] = TableCell(
                value=look.reader("aspect", contrast.aspect) if contrast.aspect is not None else None
            )
        rows.append(cells)
    return [_table("contrasts", columns, rows, "in the pipeline's order")]


# --- strata, tasks and cells -----------------------------------------------------------------------------------------


def _slices_table(
    name: str, slice_header: str, slices: Sequence[_Slice], report_input: ReportInput, look: PageLook
) -> TableBlock:
    """A slice per row and measure, its cases, then every arm's figure on the measure there."""
    measures = list(dict.fromkeys(key for _, _, per_arm in slices for figures in per_arm.values() for key in figures))
    several = len(measures) > 1
    columns = [
        _column(name, slice_header),
        *([_column("measure")] if several else []),
        _column("cases"),
        *_arm_columns(report_input, look),
    ]
    arms = [column.key.removeprefix("arm:") for column in columns if column.key.startswith("arm:")]
    rows = []
    for label, cases, per_arm in slices:
        for key in measures:
            if not any(key in per_arm.get(arm, {}) for arm in arms):
                continue
            kind = "judged" if key in report_input.judged and key not in report_input.measures else None
            cells = {name: TableCell(value=label), "cases": TableCell(value=cases)}
            if several:
                cells["measure"] = TableCell(value=_named(report_input, key, kind=kind))
            for arm in arms:
                cells[f"arm:{arm}"] = _stated(report_input, key, per_arm.get(arm, {}).get(key), kind=kind)
            rows.append(cells)
    return _table(name, columns, rows, "in the input's order, each slice's measures in the order it holds them")


def _strata(report_input: ReportInput, look: PageLook) -> list[TableBlock]:
    tables = []
    if report_input.strata:
        strata = [
            (look.reader("stratum", stratum.id), stratum.n_cases, stratum.per_arm) for stratum in report_input.strata
        ]
        tables.append(_slices_table("strata", FULL_RESULTS_HEADERS["stratum"], strata, report_input, look))
    if report_input.tasks:
        tasks = [(look.reader("task", task.id), task.n_cases, task.per_arm) for task in report_input.tasks]
        tables.append(_slices_table("tasks", FULL_RESULTS_HEADERS["task"], tasks, report_input, look))
    return tables


def _cell_name(report_input: ReportInput, cell_id: str) -> str:
    """A target-matrix cell as a reader is shown it: its row's level, then its column's."""
    cell = report_input.cells[cell_id]
    matrix = report_input.target_matrix
    if matrix is None:
        return humanize("dimension", cell_id)
    row = next((level.name or level.id for level in matrix.rows.levels if level.id == cell.row), cell.row)
    column = next((level.name or level.id for level in matrix.columns.levels if level.id == cell.column), cell.column)
    return f"{row} × {column}"


def _cells(report_input: ReportInput, look: PageLook) -> list[TableBlock]:
    cells = [(_cell_name(report_input, key), cell.n_cases, cell.per_arm) for key, cell in report_input.cells.items()]
    return [_slices_table("cells", FULL_RESULTS_HEADERS["cell"], cells, report_input, look)] if cells else []


# --- the gate matrix -------------------------------------------------------------------------------------------------


def _places(
    report_input: ReportInput, criterion: Criterion, look: PageLook
) -> Iterator[tuple[str | None, str | None, str]]:
    """Every place a criterion is read at, with its name in the Where column."""
    for task, cell in criterion_places(report_input, criterion):
        if task is not None:
            yield task, cell, look.reader("task", task)
        elif cell is not None:
            yield task, cell, _cell_name(report_input, cell)
        else:
            yield task, cell, OVERALL


def _gate_cell(
    report_input: ReportInput, criterion: Criterion, arm: str, task: str | None, cell: str | None
) -> TableCell:
    """One arm against one criterion at one place: the gate check's status in words, the figure beside it."""
    checked: GateCheck = gate_check(report_input, arm, task=task, cell=cell, criterion=criterion.id)
    figure = next(
        (held for held in criterion_figures(report_input, criterion, arm) if (held.task, held.cell) == (task, cell)),
        None,
    )
    line = resolve(f"criteria.{criterion.id}.threshold", report_input)
    said = format_figure(replace(line, value=figure.value)) if figure is not None else NO_FIGURE
    return TableCell(value=GATE_WORDS[checked.status], status=GATE_STATUS[checked.status], flag=said)


def _gate_matrix_full(report_input: ReportInput, look: PageLook) -> list[TableBlock]:
    gating = [criterion for criterion in report_input.criteria if criterion.kind in _GATING]
    arm_columns = _arm_columns(report_input, look)
    columns = [_column("where"), _column("criterion"), _column("line"), *arm_columns]
    rows = []
    for criterion in gating:
        line = format_figure(resolve(f"criteria.{criterion.id}.threshold", report_input))
        for task, cell, where in _places(report_input, criterion, look):
            cells = {
                "where": TableCell(value=where),
                "criterion": TableCell(value=look.reader("criterion", criterion.id)),
                "line": TableCell(value=LINE_TEMPLATES[criterion.direction].format(line=line)),
            }
            for column in arm_columns:
                arm = column.key.removeprefix("arm:")
                cells[column.key] = _gate_cell(report_input, criterion, arm, task, cell)
            rows.append(cells)
    return [_table("gate_matrix_full", columns, rows, "each criterion as declared, at each place it is read")]


# --- the configurations ----------------------------------------------------------------------------------------------


def _configs(report_input: ReportInput, look: PageLook) -> list[TableBlock]:
    factors = report_input.factors
    columns = [
        _column("arm"),
        *(_column(f"factor:{factor.id}", factor.name or humanize("dimension", factor.id)) for factor in factors),
    ]
    rows = []
    for config in report_input.configs:
        cells = {"arm": _chip(look, config.arm)}
        for factor in factors:
            level = config.levels.get(factor.id)
            named = next((held.name or held.id for held in factor.levels if held.id == level), level)
            cells[f"factor:{factor.id}"] = TableCell(value=named)
        rows.append(cells)
    return [_table("configs", columns, rows, "in the input's order")]


#: Each table the plan may list, to its builder.
_TABLES: dict[str, _TableBuilder] = {
    "arms_by_measure": _arms_by_measure,
    "contrasts": _contrasts,
    "strata": _strata,
    "gate_matrix_full": _gate_matrix_full,
    "configs": _configs,
    "cells": _cells,
}


def full_results(plan: ReportPlan, report_input: ReportInput, look: PageLook) -> list[TableBlock]:
    """Every Full results table the plan lists, in its order (this module's account).

    :param plan: The stored plan.
    :ptype plan: ReportPlan
    :param report_input: The report input it was written against.
    :ptype report_input: ReportInput
    :param look: The page's look.
    :ptype look: PageLook
    :return: The tables; a listed table the input holds no rows for is laid out empty, saying so.
    :rtype: list[TableBlock]
    """
    return [table for name in plan.full_results for table in _TABLES[name](report_input, look)]
