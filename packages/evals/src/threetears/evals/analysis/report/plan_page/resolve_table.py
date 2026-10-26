"""The plain table: the rows a chart query selects, laid out as a table block, for any source.

A row per selected row, in the input's order (the arms in the chart's order), columns by source:

==============  ===================================================================================================
source          columns
==============  ===================================================================================================
readings        the arm, its change in points with its interval, the pipeline's reading
contrasts       the arm, the measure, the change (a rate's in points) with its interval, the verdict
measures,       the arm, the measure, its figure (its mean, else the statistic a criterion on it reads) with its
judged,         interval; money per reply stated per 1,000 replies
configs
scoreboard      the arm, the figure, its value (a count by cause as its total)
floors          the floor, the arm, its status, the largest drop its interval allows, its margin
strata, tasks   the slice, the arm, the measure, its figure with its interval
examples        the trial, the case, the stratum or task, the arm, the outcome, the reason
cells           the cell (its row level and column level), the arm, the measure, its figure, its cases
confusion       the axis, the arm, the true label, the predicted label, the share of the row (or the count)
human_pref      the slice, the share of ratings preferring the control with its interval
==============  ===================================================================================================

An arm is the kit's chip (its tag and name); every number is spelled by the package's number rule, an absent one a
dash. A query's arms keep the rows of those arms, on sources whose rows are of no one arm too.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from threetears.evals.analysis.quantities import MONEY
from threetears.evals.analysis.report.blocks import CellValue, TableBlock, TableCell, TableColumn
from threetears.evals.analysis.report.cost_unit import per_count, per_noun
from threetears.evals.analysis.report.plan_page.charting import (
    ChartContext,
    Drawn,
    KindResolver,
    figure_unit,
    measure_words,
    slice_name,
    stat_interval,
    stat_value,
)
from threetears.evals.analysis.report.plan_page.look import quantity
from threetears.evals.analysis.report.plan_page.model import PLANNED_SECTION
from threetears.evals.analysis.report_input import (
    Cell,
    ConfusionTable,
    Contrast,
    Example,
    FloorRead,
    PreferenceRow,
    Reading,
    ScoreboardRow,
    Stat,
    Stratum,
    Task,
)
from threetears.evals.analysis.report.plan_page.words import (
    CELL_TEMPLATE,
    TABLE_HEADERS,
    TABLE_ORDER,
    TABLE_PER_THOUSAND,
)
from threetears.evals.analysis.style import humanize

__all__ = ["RESOLVERS"]


#: One table row's cells, by column key.
_Row = dict[str, TableCell]


def _arm(context: ChartContext, arm: str) -> TableCell:
    if arm in context.look.arms:
        look = context.look.arm(arm)
        return TableCell(value=look.name, tag=look.tag, color=look.token)
    return TableCell(value=arm or None)


def _number(value: float | None, unit: str, interval: tuple[float, float] | None = None) -> TableCell:
    if value is None:
        return TableCell(value=None)
    low, high = interval if interval is not None else (None, None)
    return TableCell(value=value, unit=unit, low=low, high=high)


def _words(value: str | None) -> TableCell:
    return TableCell(value=value or None)


def _kept_arms(context: ChartContext, held: Iterator[str]) -> list[str]:
    allowed = set(context.query.arms) or set(context.look.arms)
    return context.arm_order(arm for arm in held if arm in allowed)


def _figure_cell(context: ChartContext, key: str, stat: Stat | None) -> tuple[TableCell, str]:
    """A figure's cell, and its measure's name as the table writes it (per 1,000 replies for money per reply)."""
    report_input = context.report_input
    name = measure_words(report_input, key)
    read = stat_value(report_input, key, stat) if stat is not None else None
    drawn = quantity(figure_unit(report_input, key), [read[0]] if read is not None else [])
    measure = report_input.measures.get(key)
    if drawn.unit == MONEY and drawn.factor != 1 and measure is not None:
        name = TABLE_PER_THOUSAND.format(name=name, per=per_count(per_noun(measure)))
    if stat is None or read is None:
        return TableCell(value=None), name
    interval = stat_interval(stat) if read[1] == "mean" else None
    shown = (interval[0] * drawn.factor, interval[1] * drawn.factor) if interval is not None else None
    return _number(read[0] * drawn.factor, drawn.unit, shown), name


def _readings(context: ChartContext) -> Iterator[_Row]:
    points = quantity("points", change=True)
    by_arm = {row.arm: row.item for row in context.rows() if isinstance(row.item, Reading)}
    for arm in context.arm_order(by_arm):
        reading = by_arm[arm]
        interval = reading.within_interval
        yield {
            "arm": _arm(context, arm),
            "change": _number(
                float(reading.change),
                points.unit,
                (float(interval[0]), float(interval[1])) if interval is not None else None,
            ),
            "reading": _words(humanize("dimension", reading.reading)),
        }


def _contrasts(context: ChartContext) -> Iterator[_Row]:
    report_input = context.report_input
    contrasts = [row.item for row in context.rows() if isinstance(row.item, Contrast)]
    for key in dict.fromkeys(contrast.measure for contrast in contrasts):
        on_key = {contrast.arm: contrast for contrast in contrasts if contrast.measure == key}
        for arm in context.arm_order(on_key):
            contrast = on_key[arm]
            drawn = quantity(figure_unit(report_input, key, kind=contrast.kind), change=True)
            interval = contrast.interval
            yield {
                "arm": _arm(context, arm),
                "measure": _words(measure_words(report_input, key, kind=contrast.kind)),
                "change": _number(
                    float(contrast.delta) * drawn.factor if contrast.delta is not None else None,
                    drawn.unit,
                    (float(interval[0]) * drawn.factor, float(interval[1]) * drawn.factor)
                    if interval is not None and contrast.delta is not None
                    else None,
                ),
                "verdict": _words(humanize("dimension", contrast.verdict)),
            }


def _figures(context: ChartContext) -> Iterator[_Row]:
    rows = context.rows()
    figures = {(row.key, row.arm): row.item for row in rows if isinstance(row.item, Stat)}
    for key in dict.fromkeys(row.key for row in rows):
        for arm in context.arm_order(arm for (on, arm) in figures if on == key):
            cell, name = _figure_cell(context, key, figures[key, arm])
            yield {"arm": _arm(context, arm), "measure": _words(name), "value": cell}


def _scoreboard(context: ChartContext) -> Iterator[_Row]:
    boards = {row.arm: row.item for row in context.rows() if isinstance(row.item, ScoreboardRow)}
    keys = list(context.query.measures) or list(ScoreboardRow.model_fields)
    for arm in context.arm_order(boards):
        for key in keys:
            held = getattr(boards[arm], key, None)
            value: CellValue = (
                sum(held.values()) if isinstance(held, dict) else held if isinstance(held, int | float) else None
            )
            if value is None and not context.query.measures:
                continue
            yield {
                "arm": _arm(context, arm),
                "figure": _words(humanize("dimension", key)),
                "value": TableCell(value=value),
            }


def _floors(context: ChartContext) -> Iterator[_Row]:
    points = quantity("points", change=True)
    for row in context.rows():
        floor = row.item
        assert isinstance(floor, FloorRead)
        yield {
            "floor": _words(humanize("dimension", floor.floor)),
            "arm": _arm(context, floor.arm),
            "status": _words(floor.status),
            "worst_drop": _number(float(floor.worst_drop) if floor.worst_drop is not None else None, points.unit),
            "margin": _number(float(floor.margin) if floor.margin is not None else None, points.unit),
        }


def _slices(context: ChartContext) -> Iterator[_Row]:
    slices = [row.item for row in context.rows() if isinstance(row.item, Stratum | Task)]
    named = set(context.query.measures)
    for item in slices:
        for arm in _kept_arms(context, iter(item.per_arm)):
            for key, stat in item.per_arm[arm].items():
                if named and key not in named:
                    continue
                cell, name = _figure_cell(context, key, stat)
                yield {
                    "slice": _words(slice_name(context, item)),
                    "arm": _arm(context, arm),
                    "measure": _words(name),
                    "value": cell,
                }


def _examples(context: ChartContext) -> Iterator[_Row]:
    for row in context.rows():
        example = row.item
        assert isinstance(example, Example)
        yield {
            "trial": _words(example.trial),
            "case": _words(example.case),
            "stratum": _words(context.look.reader("stratum", example.stratum) if example.stratum else None),
            "arm": _arm(context, example.arm or ""),
            "outcome": _words(example.outcome),
            "reason": _words(example.reason),
        }


def _level_names(context: ChartContext) -> dict[str, str]:
    matrix = context.report_input.target_matrix
    factors = [*context.report_input.factors, *((matrix.rows, matrix.columns) if matrix is not None else ())]
    return {level.id: level.name for factor in factors for level in factor.levels if level.name}


def _cells(context: ChartContext) -> Iterator[_Row]:
    names = _level_names(context)
    named = set(context.query.measures)
    for row in context.rows():
        cell = row.item
        assert isinstance(cell, Cell)
        place = CELL_TEMPLATE.format(row=names.get(cell.row, cell.row), column=names.get(cell.column, cell.column))
        for arm in _kept_arms(context, iter(cell.per_arm)):
            for key, stat in cell.per_arm[arm].items():
                if named and key not in named:
                    continue
                figure, name = _figure_cell(context, key, stat)
                yield {
                    "cell": _words(place),
                    "arm": _arm(context, arm),
                    "measure": _words(name),
                    "value": figure,
                    "cases": TableCell(value=cell.n_cases),
                }


def _confusion(context: ChartContext) -> Iterator[_Row]:
    names = _level_names(context)
    for row in context.rows():
        table = row.item
        assert isinstance(table, ConfusionTable)
        unit = "%" if table.unit == "row_percent" else ""
        for arm in _kept_arms(context, iter(table.per_arm)):
            for expected, shares in zip(table.labels, table.per_arm[arm], strict=True):
                for predicted, share in zip(table.labels, shares, strict=True):
                    yield {
                        "axis": _words(humanize("dimension", row.key)),
                        "arm": _arm(context, arm),
                        "expected": _words(names.get(expected, expected)),
                        "predicted": _words(names.get(predicted, predicted)),
                        "value": _number(float(share), unit),
                    }


def _human_pref(context: ChartContext) -> Iterator[_Row]:
    for row in context.rows():
        preference = row.item
        assert isinstance(preference, PreferenceRow)
        yield {
            "slice": _words(preference.name or preference.id),
            "prefers_control": _number(float(preference.prefers_control), "%", row.interval),
        }


#: Each source's rows, and its columns in order.
_SOURCES: dict[str, tuple[Callable[[ChartContext], Iterator[_Row]], tuple[str, ...]]] = {
    "readings": (_readings, ("arm", "change", "reading")),
    "contrasts": (_contrasts, ("arm", "measure", "change", "verdict")),
    "measures": (_figures, ("arm", "measure", "value")),
    "judged": (_figures, ("arm", "measure", "value")),
    "configs": (_figures, ("arm", "measure", "value")),
    "scoreboard": (_scoreboard, ("arm", "figure", "value")),
    "floors": (_floors, ("floor", "arm", "status", "worst_drop", "margin")),
    "strata": (_slices, ("slice", "arm", "measure", "value")),
    "tasks": (_slices, ("slice", "arm", "measure", "value")),
    "examples": (_examples, ("trial", "case", "stratum", "arm", "outcome", "reason")),
    "cells": (_cells, ("cell", "arm", "measure", "value", "cases")),
    "confusion": (_confusion, ("axis", "arm", "expected", "predicted", "value")),
    "human_pref": (_human_pref, ("slice", "prefers_control")),
}


def _table(context: ChartContext) -> Drawn:
    reader, keys = _SOURCES[context.query.source]
    rows = list(reader(context))
    if not rows:
        raise context.refuse("it selects no row to lay out")
    table = TableBlock(
        section=PLANNED_SECTION,
        name=context.query.source,
        title=context.title,
        columns=[TableColumn(key=key, header=TABLE_HEADERS[key]) for key in keys],
        rows=rows,
        order=TABLE_ORDER,
        total_rows=len(rows),
    )
    return Drawn(table)


#: The kinds this module resolves.
RESOLVERS: dict[str, KindResolver] = {"table": _table}
