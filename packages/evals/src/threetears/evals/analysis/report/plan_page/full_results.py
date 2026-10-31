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

**More than one aspect.** Where more than one aspect holds a figure, every arm by measure is a row per aspect and arm,
grouped by aspect in the input's order, the aspect named in its own column and every figure that aspect's own (a
measure's ``per_arm`` on the lead aspect, its ``by_aspect`` entry on another; a dash where the aspect did not read it).

**Every table the input holds** (:func:`every_table`): a report document's full-results appendix, which no plan
lists. It is each table above, in that order, where the input holds a row of it: every arm by measure always, the
rest as the input holds contrasts, strata or tasks, gates or guardrails, configurations and cells. Which tables print
is read off the input's rows alone.

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

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace

from threetears.evals.kernel.numbers import format_number
from threetears.evals.analysis.quantities import MONEY
from threetears.evals.analysis.report.blocks import TableBlock, TableCell, TableColumn, money_columns
from threetears.evals.analysis.report.charts import ChartStatus
from threetears.evals.analysis.report.cost_unit import money_ruler, priced_in_part
from threetears.evals.analysis.report.figure_keys import (
    aspect_costs_per_thousand,
    cost_basis,
    cost_ruler,
    format_figure,
    held_figure,
    resolve,
)
from threetears.evals.analysis.report.plan_page.charting import (
    figure_unit,
    held_means,
    money_words,
    stat_value,
)
from threetears.evals.analysis.report.plan_page.look import (
    GATE_STATUS,
    PageLook,
    Quantity,
    figure_text,
    in_change_form,
    quantity,
    quantity_of,
)
from threetears.evals.analysis.report.plan_page.model import PLANNED_SECTION
from threetears.evals.analysis.report.plan_page.tables import pass_rate_cell
from threetears.evals.analysis.report.plan_page.words import (
    ASPECTS_TABLE_ORDER,
    CONTRAST_VERDICT_WORDS,
    COST_HEADER,
    FULL_RESULTS_HEADERS,
    FULL_RESULTS_TITLES,
    GATE_UNKNOWN,
    GATE_WORDS,
    LINE_TEMPLATES,
    NO_FIGURE,
    OVERALL,
)
from threetears.evals.analysis.report_gates import GateCheck, gate_check
from threetears.evals.analysis.report_input import Aspect, Contrast, Criterion, Judged, Measure, ReportInput, Stat
from threetears.evals.analysis.report_margins import aspect_board, criterion_figures, criterion_places, on_aspect
from threetears.evals.analysis.style import humanize
from threetears.evals.kernel.report_plan import ReportPlan

__all__ = ["every_table", "full_results"]


#: Each corrected comparison's verdict's pill: a separated gain or equivalence good, a regression bad, the rest
#: neutral.
_VERDICT_STATUS: dict[str, ChartStatus] = {"improved": "good", "equivalent": "good", "regressed": "bad"}

#: The criterion kinds the gate check reads.
_GATING = frozenset({"gate", "guardrail"})

#: A slice the strata, tasks and cells tables lay out: its reader name, its cases, each arm's figures by measure.
_Slice = tuple[str, int | None, Mapping[str, Mapping[str, Stat]]]

#: A Full results table's builder: the input and the page's look in; its tables out.
_TableBuilder = Callable[[ReportInput, PageLook], list[TableBlock]]


def _table(
    name: str,
    columns: Sequence[TableColumn],
    rows: list[dict[str, TableCell]],
    order: str,
    *,
    by_row: Sequence[str | None] | None = None,
) -> TableBlock:
    """A Full results table, each figure type's money in one form chosen from its own values (#993): a column's, or,
    where ``by_row`` names each row's measure (a slice table, each row one measure across the arms), every row's on
    that measure, across the table. Every column's head or row names the count its money is drawn per, so it never
    carries the per-1,000 mark beside it."""
    return TableBlock(
        section=PLANNED_SECTION,
        name=name,
        title=FULL_RESULTS_TITLES[name],
        columns=list(columns),
        rows=money_columns(rows, by_row=by_row, marked=False),
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


def _drawn(report_input: ReportInput, key: str, *, kind: str | None = None) -> Quantity:
    """How a measure's figures are drawn in every Full results table: by the one owner of money's scale over every
    figure the input holds of it, so its money is drawn per one count in every table (per 1,000 replies for money per
    reply, per 1,000,000 where it is under a cent per 1,000); any other unit as it is."""
    unit = figure_unit(report_input, key, kind=kind)
    return quantity_of(unit, held_means(report_input, key)) if kind != "judged" else quantity(unit)


def _named(report_input: ReportInput, key: str, *, kind: str | None = None) -> str:
    """A measure's reader name, with the count its money is drawn per in it (the one owner of money's words: the
    label's own count replaced, never a second), as every chart names it."""
    return money_words(report_input, key, _drawn(report_input, key, kind=kind), kind=kind)


def _stated(report_input: ReportInput, key: str, stat: Stat | None, *, kind: str | None = None) -> TableCell:
    """One figure on a measure or judged dimension as a cell: its mean (else the statistic a criterion on it reads,
    named beside it) in the measure's unit, by the page's number rule; a dash where the input holds none."""
    read = stat_value(report_input, key, stat) if stat is not None else None
    if read is None:
        return TableCell(value=None)
    value, statistic = read
    drawn = _drawn(report_input, key, kind=kind)
    flag = statistic if statistic != "mean" else None
    if drawn.unit == MONEY:
        # A number, which its column's one money form spells (:func:`_table`), never a form of its own.
        return TableCell(value=value * drawn.factor, unit=MONEY, flag=flag)
    return TableCell(value=figure_text(value * drawn.factor, drawn), flag=flag)


# --- arms by measure -------------------------------------------------------------------------------------------------


def _cost(report_input: ReportInput, arm: str) -> TableCell:
    """The arm's cost by the package's one cost rule (the ``cost_per_1k`` figure), a number drawn per the count the
    column's head names (:func:`~threetears.evals.analysis.report.figure_keys.cost_ruler`), its priced share beside it
    where only part of its calls were priced."""
    figure = held_figure(f"cost_per_1k.{arm}", report_input)
    if figure is None or not isinstance(figure.value, float):
        return TableCell(value=None)
    # The form is chosen over every arm's cost, a partly priced one's included, where the options table leaves that arm
    # out of its choice: this table prints the partly priced cost, its share beside it, and the options table prints
    # the share in its place, so each column's form fits the numbers it prints.
    return TableCell(
        value=cost_ruler(report_input).drawn(figure.value), unit=MONEY, flag=priced_in_part(report_input, arm)
    )


def _arms_by_measure(report_input: ReportInput, look: PageLook) -> list[TableBlock]:
    if len(aspects := _measured_aspects(report_input)) > 1:
        return _arms_by_aspect_and_measure(report_input, look, aspects)
    arms = [arm.id for arm in report_input.arms]
    # The scoreboard's own column, drawn where it counts passes: where it counts none, the ``pass_rate`` measure the
    # lead aspect's cell falls back to is already a measure column below, and one figure takes one column.
    board = aspect_board(report_input, None)
    passing = any((row := board.get(arm)) is not None and row.passed is not None and row.usable for arm in arms)

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
        *(
            [
                _column(
                    "cost_per_1k", COST_HEADER.format(per=cost_ruler(report_input).words(cost_basis(report_input).noun))
                )
            ]
            if costed
            else []
        ),
    ]
    rows = []
    for arm in arms:
        cells = {"arm": _chip(look, arm)}
        if passing:
            cells["pass_rate"] = pass_rate_cell(report_input, arm)
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


def _measured_aspects(report_input: ReportInput) -> list[Aspect]:
    """The aspects some arm holds a figure on, in the input's order: a usable trial, a measure's or a judged
    dimension's figure."""
    tables: list[Measure | Judged] = [*report_input.measures.values(), *report_input.judged.values()]
    return [
        aspect
        for aspect in report_input.aspects
        if any(row.usable for row in aspect_board(report_input, aspect.id).values())
        or any(on_aspect(report_input, table, aspect.id) for table in tables)
    ]


def _arms_by_aspect_and_measure(report_input: ReportInput, look: PageLook, aspects: list[Aspect]) -> list[TableBlock]:
    """Every arm by measure where more than one aspect holds figures: a row per aspect and arm, the aspect named in
    its own column, each figure the aspect's own; the cost column, where no measure states one, by the one cost rule
    read on each aspect, every aspect's cost in one form."""
    arms = [arm.id for arm in report_input.arms]
    places = [(aspect, arm) for aspect in aspects for arm in arms]
    passing = {(aspect.id, arm): pass_rate_cell(report_input, arm, aspect.id) for aspect, arm in places}

    def held(table: Measure | Judged, key: str, *, kind: str | None = None) -> bool:
        return any(
            _stated(report_input, key, on_aspect(report_input, table, aspect.id).get(arm), kind=kind).value is not None
            for aspect, arm in places
        )

    judged = [key for key, table in report_input.judged.items() if held(table, key, kind="judged")]
    measures = [key for key, table in report_input.measures.items() if held(table, key)]
    costs = {aspect.id: aspect_costs_per_thousand(report_input, aspect.id) for aspect in aspects}
    # Over every cost this column prints, a partly priced arm's included (:func:`_cost`'s reason).
    ruler = money_ruler([cost for held_costs in costs.values() for cost in held_costs.values()], per=1000, start=1000)
    costed = not any(quantity(figure_unit(report_input, key)).unit == MONEY for key in measures) and any(
        arm in costs[aspect.id] for aspect, arm in places
    )
    columns = [
        _column("arm"),
        _column("aspect"),
        *([_column("pass_rate")] if any(cell.value is not None for cell in passing.values()) else []),
        *(_column(f"judged:{key}", _named(report_input, key, kind="judged")) for key in judged),
        *(_column(f"measure:{key}", _named(report_input, key)) for key in measures),
        *(
            [_column("cost_per_1k", COST_HEADER.format(per=ruler.words(cost_basis(report_input).noun)))]
            if costed
            else []
        ),
    ]
    keys = {column.key for column in columns}
    rows = []
    for aspect, arm in places:
        cells = {"arm": _chip(look, arm), "aspect": TableCell(value=aspect.name or aspect.id)}
        if "pass_rate" in keys:
            cells["pass_rate"] = passing[aspect.id, arm]
        for key in judged:
            cells[f"judged:{key}"] = _stated(
                report_input, key, on_aspect(report_input, report_input.judged[key], aspect.id).get(arm), kind="judged"
            )
        for key in measures:
            cells[f"measure:{key}"] = _stated(
                report_input, key, on_aspect(report_input, report_input.measures[key], aspect.id).get(arm)
            )
        if costed:
            cost = costs[aspect.id].get(arm)
            cells["cost_per_1k"] = (
                TableCell(value=ruler.drawn(cost), unit=MONEY, flag=priced_in_part(report_input, arm, aspect.id))
                if cost is not None
                else TableCell(value=None)
            )
        rows.append(cells)
    return [_table("arms_by_measure", columns, rows, ASPECTS_TABLE_ORDER)]


# --- the corrected comparisons ---------------------------------------------------------------------------------------


def _changes_drawn(report_input: ReportInput, contrasts: Sequence[Contrast]) -> dict[tuple[str, str | None], Quantity]:
    """How each measure's changes are drawn in the comparisons table: on its figures' own ruler (:func:`_drawn`), so a
    change is drawn per the count its measure's name says, in the changes' own form capped by their widest interval
    (#993); any other unit's change quantity as it is."""
    drawn: dict[tuple[str, str | None], Quantity] = {}
    for measure, kind in dict.fromkeys((c.measure, c.kind) for c in contrasts):
        amounts = _drawn(report_input, measure, kind=kind)
        held = [c for c in contrasts if (c.measure, c.kind) == (measure, kind)]
        drawn[measure, kind] = (
            in_change_form(
                amounts,
                [float(c.delta) * amounts.factor for c in held if c.delta is not None],
                [
                    abs(float(c.interval[1]) - float(c.interval[0])) * amounts.factor
                    for c in held
                    if c.interval is not None
                ],
            )
            if amounts.unit == MONEY
            else quantity(figure_unit(report_input, measure, kind=kind), change=True)
        )
    return drawn


def _change(
    report_input: ReportInput, contrast: Contrast, drawn_as: Mapping[tuple[str, str | None], Quantity]
) -> tuple[TableCell, TableCell]:
    """A comparison's change and its interval, in its measure's unit (a rate's in points), money in the table's one
    form for its unit."""
    drawn = drawn_as[contrast.measure, contrast.kind]
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
    drawn_as = _changes_drawn(report_input, held)
    for contrast in held:
        change, interval = _change(report_input, contrast, drawn_as)
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
    kinds: list[str | None] = []
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
            kinds.append(key)
    order = "in the input's order, each slice's measures in the order it holds them"
    return _table(name, columns, rows, order, by_row=kinds)


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
    """One arm against one criterion at one place: the gate check's status in words, the figure beside it; where the
    input holds no figure there, ``Unknown (no figure)``, never a fail: the gate check counts an unknown figure among
    an arm's fails, but a reader of the grid reads what is known at that place, as the rest of the page does."""
    figure = next(
        (held for held in criterion_figures(report_input, criterion, arm) if (held.task, held.cell) == (task, cell)),
        None,
    )
    if figure is None or not math.isfinite(figure.value):
        return TableCell(value=GATE_UNKNOWN, status="neutral", flag=NO_FIGURE)
    checked: GateCheck = gate_check(report_input, arm, task=task, cell=cell, criterion=criterion.id)
    line = resolve(f"criteria.{criterion.id}.threshold", report_input)
    said = format_figure(replace(line, value=figure.value))
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


def every_table(report_input: ReportInput, look: PageLook) -> list[TableBlock]:
    """Every Full results table the input holds a row of, in this module's order (this module's account).

    :param report_input: The report input a report document was written from.
    :ptype report_input: ReportInput
    :param look: The page's look.
    :ptype look: PageLook
    :return: The tables, each with one row at least.
    :rtype: list[TableBlock]
    """
    return [table for build in _TABLES.values() for table in build(report_input, look) if table.rows]
