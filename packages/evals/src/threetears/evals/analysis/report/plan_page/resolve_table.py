"""The plain table: the rows a chart query selects, laid out as a table block, for any source.

A row per selected row, in the input's order (the arms in the chart's order), columns by source:

==============  ===================================================================================================
source          columns
==============  ===================================================================================================
readings        the arm, its change in points with its interval, the pipeline's reading
contrasts       the arm, the measure (naming the aspect where two aspects' contrasts share it), the change (a
                rate's in points) with its interval, the verdict
measures,       wide: a row per arm, a column per measure holding its figure (its mean, else the statistic a
judged,         criterion on it reads) with its interval; money stated per 1,000 of what it counts (per 1,000
                replies for money per reply)
configs
scoreboard      the arm, the figure, its value (a count by cause as its total)
floors          the floor, the arm, its status, the largest drop its interval allows, its margin
strata, tasks   wide: a row per slice and arm, a column per measure holding its figure with its interval
examples        the trial, the case, the stratum or task, the arm, the outcome, the reason
cells           wide: a row per cell (its row level and column level) and arm, a column per measure holding its
                figure, then its cases
confusion       the axis, the arm, the true label, the predicted label, the share of the row (or the count)
human_pref      the slice, the share of ratings preferring the control with its interval
prices          the step along the price frontier (from one option to the next), its price per point with its
                interval: a month's at the declared volume, else per 1,000 of what the cost counts
bar_reads       the aspect, the measure, the arm, its value with the interval its bar was read on, the bar, the status
compare         the place, each arm's mean, the first less the second, its interval, level and basis (in reader
                names: a paired one names the candidate and the measure, never their ids)
settle          the arm, the line (and place), today's estimate, the line, the distance, which way it settles, the
                cases a one-sided and a two-sided design need, the trials that takes and the repeats of a case,
                today's cases and today's trials (each where its count states it), how often a fresh run reads the
                other side, the level
levers          the setting (or the two an interaction crosses), the move, the effect (or difference), its interval,
                level, and an interaction's verdict
==============  ===================================================================================================

**Wide, so paired figures sit side by side.** An arm's pass rate, cost and latency are one row, each in its
measure's column headed by its reader name, rather than three rows a reader pairs by eye.

**An interval is a column of its own.** Every column whose figures carry an interval is followed by one spelling
each interval's ends, headed by the chart's interval level (``95% interval``), so a reader without the glyph (the
Markdown page, a values table) reads the interval and its level too.

An arm is the kit's chip (its tag and name); every number is spelled by the package's number rule, an absent one a dash;
a row an analysis computed (compare, settle, levers) prints each figure as its key reads it, so the table and prose
citing the key read alike.
A month's prices are whole dollars where every one on the axis is :data:`_WHOLE_DOLLARS_FROM` or more, as the
``usd_whole`` figure key filter writes them: the cents of a monthly amount are noise, and under that whole dollars would
round away too much, so every price keeps its cents. A query's arms keep the rows of those arms, on sources whose rows
are of no one arm too.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from threetears.evals.analysis.quantities import MONEY, with_unit
from threetears.evals.analysis.report.blocks import CellValue, TableBlock, TableCell, TableColumn
from threetears.evals.analysis.report.charts import OPEN_ABOVE
from threetears.evals.analysis.report.cost_unit import per_count, per_noun
from threetears.evals.analysis.report.plan_page.charting import (
    ChartContext,
    Drawn,
    KindResolver,
    computed_notes,
    contrast_groups,
    contrast_words,
    figure_interval,
    figure_unit,
    measure_words,
    slice_name,
    stat_interval,
    stat_value,
)
from threetears.evals.analysis.report.plan_page.look import quantity
from threetears.evals.analysis.report.plan_page.model import PLANNED_SECTION
from threetears.evals.analysis.report.plan_page.resolve_changes import compare_basis
from threetears.evals.analysis.report.plan_page.resolve_magnitudes import price_notes, settle_label, settle_notes
from threetears.evals.analysis.analyst.cited import CitedFigure, cite_if
from threetears.evals.analysis.analyst.compare import ComparisonRow
from threetears.evals.analysis.analyst.levers import EffectRow, InteractionRow
from threetears.evals.analysis.analyst.settle import SettleRow
from threetears.evals.analysis.report_input import (
    BarRead,
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
from threetears.evals.analysis.report_rows import PriceStep, lever_arms
from threetears.evals.analysis.report.plan_page.words import (
    PRICE_STEP_TEMPLATE,
    CELL_TEMPLATE,
    INTERVAL_HEADER,
    OVERALL_WORD,
    TABLE_HEADERS,
    TABLE_ORDER,
    TABLE_PER_THOUSAND,
)
from threetears.evals.analysis.style import humanize

__all__ = ["RESOLVERS"]


#: One table row's cells, by column key.
_Row = dict[str, TableCell]

#: A column a reader lays out beyond its source's fixed ones: its key and its header.
_Laid = list[tuple[str, str]]

#: The prefix a wide table's measure column is keyed by, before the measure's key.
_MEASURE_COLUMN = "measure:"

#: The suffix an interval column is keyed by, after the key of the column whose figures it spells.
_INTERVAL_COLUMN = ":interval"


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


def _figure_cell(context: ChartContext, key: str, stat: Stat | None, arm: str = "") -> tuple[TableCell, str]:
    """A figure's cell, and its measure's name as the table writes it (per 1,000 replies for money per reply); an
    arm's overall figure carries the interval code computes where the table's level is computed here."""
    report_input = context.report_input
    name = measure_words(report_input, key)
    read = stat_value(report_input, key, stat) if stat is not None else None
    drawn = quantity(figure_unit(report_input, key), [read[0]] if read is not None else [])
    measure = report_input.measures.get(key)
    if drawn.unit == MONEY and drawn.factor != 1 and measure is not None:
        name = TABLE_PER_THOUSAND.format(name=name, per=per_count(per_noun(measure)))
    if stat is None or read is None:
        return TableCell(value=None), name
    if read[1] != "mean":
        interval = None
    else:
        interval = figure_interval(context, key, arm, stat) if arm else stat_interval(stat)
    shown = (interval[0] * drawn.factor, interval[1] * drawn.factor) if interval is not None else None
    return _number(read[0] * drawn.factor, drawn.unit, shown), name


def _readings(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
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


def _contrasts(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
    report_input = context.report_input
    contrasts = [row.item for row in context.rows() if isinstance(row.item, Contrast)]
    for group in contrast_groups(context, contrasts):
        key = group.key
        for arm in context.arm_order(group.by_arm):
            contrast = group.by_arm[arm]
            drawn = quantity(figure_unit(report_input, key, kind=contrast.kind), change=True)
            interval = contrast.interval
            yield {
                "arm": _arm(context, arm),
                "measure": _words(contrast_words(report_input, contrast, group.aspect if group.split else None)),
                "change": _number(
                    float(contrast.delta) * drawn.factor if contrast.delta is not None else None,
                    drawn.unit,
                    (float(interval[0]) * drawn.factor, float(interval[1]) * drawn.factor)
                    if interval is not None and contrast.delta is not None
                    else None,
                ),
                "verdict": _words(humanize("dimension", contrast.verdict)),
            }


def _wide(
    context: ChartContext, places: list[tuple[_Row, dict[str, Stat]]], laid: _Laid, arms: list[str] | None = None
) -> Iterator[_Row]:
    """Each place's row (its own cells, then a cell per measure it holds a figure on, a dash where it holds none),
    each measure's column added to ``laid`` once, in the order the places first hold it; ``arms`` names the arm each
    place is, where it is an arm's overall figures."""
    keys = list(dict.fromkeys(key for _, figures in places for key in figures))
    named: dict[str, str] = {}
    built: list[_Row] = []
    for index, (own, figures) in enumerate(places):
        row = dict(own)
        for key in keys:
            cell, name = _figure_cell(context, key, figures.get(key), arms[index] if arms is not None else "")
            named.setdefault(key, name)
            row[f"{_MEASURE_COLUMN}{key}"] = cell
        built.append(row)
    laid += [(f"{_MEASURE_COLUMN}{key}", named[key]) for key in keys]
    yield from built


def _figures(context: ChartContext, laid: _Laid) -> Iterator[_Row]:
    rows = context.rows()
    figures = {(row.key, row.arm): row.item for row in rows if isinstance(row.item, Stat)}
    arms = context.arm_order(arm for (_, arm) in figures)
    places = [
        ({"arm": _arm(context, arm)}, {key: stat for (key, on), stat in figures.items() if on == arm}) for arm in arms
    ]
    yield from _wide(context, places, laid, arms)


def _scoreboard(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
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


def _floors(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
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


def _slices(context: ChartContext, laid: _Laid) -> Iterator[_Row]:
    slices = [row.item for row in context.rows() if isinstance(row.item, Stratum | Task)]
    named = set(context.query.measures)
    places = [
        (
            {"slice": _words(slice_name(context, item)), "arm": _arm(context, arm)},
            {key: stat for key, stat in item.per_arm[arm].items() if not named or key in named},
        )
        for item in slices
        for arm in _kept_arms(context, iter(item.per_arm))
    ]
    yield from _wide(context, [place for place in places if place[1]], laid)


def _examples(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
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


def _cells(context: ChartContext, laid: _Laid) -> Iterator[_Row]:
    names = _level_names(context)
    named = set(context.query.measures)
    places: list[tuple[_Row, dict[str, Stat]]] = []
    for row in context.rows():
        cell = row.item
        assert isinstance(cell, Cell)
        place = CELL_TEMPLATE.format(row=names.get(cell.row, cell.row), column=names.get(cell.column, cell.column))
        places += [
            (
                {"cell": _words(place), "arm": _arm(context, arm), "cases": TableCell(value=cell.n_cases)},
                {key: stat for key, stat in cell.per_arm[arm].items() if not named or key in named},
            )
            for arm in _kept_arms(context, iter(cell.per_arm))
        ]
    yield from _wide(context, [place for place in places if place[1]], laid)


def _axis_names(context: ChartContext) -> dict[str, str]:
    """Each declared factor's (or target-matrix axis's) reader name, by id."""
    matrix = context.report_input.target_matrix
    factors = [*context.report_input.factors, *((matrix.rows, matrix.columns) if matrix is not None else ())]
    return {factor.id: factor.name for factor in factors if factor.name}


def _confusion(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
    names = _level_names(context)
    axes = _axis_names(context)
    for row in context.rows():
        table = row.item
        assert isinstance(table, ConfusionTable)
        unit = "%" if table.unit == "row_percent" else ""
        for arm in _kept_arms(context, iter(table.per_arm)):
            for expected, shares in zip(table.labels, table.per_arm[arm], strict=True):
                for predicted, share in zip(table.labels, shares, strict=True):
                    yield {
                        "axis": _words(axes.get(row.key) or humanize("dimension", row.key)),
                        "arm": _arm(context, arm),
                        "expected": _words(names.get(expected, expected)),
                        "predicted": _words(names.get(predicted, predicted)),
                        "value": _number(float(share), unit),
                    }


def _human_pref(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
    for row in context.rows():
        preference = row.item
        assert isinstance(preference, PreferenceRow)
        yield {
            "slice": _words(preference.name or preference.id),
            "prefers_control": _number(float(preference.prefers_control), "%", row.interval),
        }


def _bar_reads(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
    for row in context.rows():
        read = row.item
        assert isinstance(read, BarRead)
        drawn = quantity(figure_unit(context.report_input, read.measure))
        interval = read.interval
        yield {
            "aspect": _words(context.look.reader("aspect", read.aspect)),
            "measure": _words(measure_words(context.report_input, read.measure)),
            "arm": _arm(context, read.arm),
            "value": _number(
                float(read.value) * drawn.factor if read.value is not None else None,
                drawn.unit,
                (float(interval[0]) * drawn.factor, float(interval[1]) * drawn.factor)
                if interval is not None
                else None,
            ),
            "bar": _number(float(read.bar) * drawn.factor, drawn.unit),
            "status": _words(read.status),
        }


def _read(figure: CitedFigure | None) -> TableCell:
    """A cited figure's cell: the figure as its key reads it, so the table prints what prose citing the key does."""
    return TableCell(value=figure.reads if figure is not None else None)


def _compare(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
    for row in context.rows():
        item = row.item
        assert isinstance(item, ComparisonRow)
        yield {
            "place": _words(str(item.name.value) if item.name is not None else OVERALL_WORD),
            "first": _read(item.first),
            "second": _read(item.second),
            "difference": _read(item.difference),
            "interval": _read(item.interval),
            "level": _read(item.level),
            "basis": _words(compare_basis(context, item, row.key) or None),
        }


def _settle(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
    for row in context.rows():
        item = row.item
        assert isinstance(item, SettleRow)
        yield {
            "arm": _arm(context, row.arm),
            "against": _words(settle_label(context, item)),
            "estimate": _read(item.estimate[0] if item.estimate else None),
            "line": _read(item.line),
            "distance": _read(item.distance),
            "settles": _words(item.settles),
            "cases_needed": _read(item.cases_needed),
            "cases_needed_two_sided": _read(item.cases_needed_two_sided),
            "trials_needed": _read(item.trials_needed),
            "repeats": _read(item.repeats[0] if item.repeats else None),
            **_today(context, row.arm, row.key, item),
            "wrong_side": _read(item.wrong_side),
            "level": _read(item.level),
        }


def _today(context: ChartContext, arm: str, key: str, row: SettleRow) -> _Row:
    """Today's count beside a settle row's sizes, in both units the sizes are: its distinct cases (the count itself
    where it counts cases, else the spread's ``cases`` key) and its trials (the count where it counts trials), each
    as its key reads it, so a size in cases never sits against a count of trials unlabelled."""
    count = row.counts[0] if row.counts else None
    counted = str(row.count_of[0].value) if row.count_of else ""
    cases: CitedFigure | None = count if counted == "cases" else None
    if cases is None and counted == "trials" and key:
        cases = cite_if(f"spread.{arm}.{key}.{row.where}.cases", context.report_input)
    return {"today_cases": _read(cases), "today_trials": _read(count if counted == "trials" else None)}


def _levers(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
    for row in context.rows():
        item = row.item
        assert isinstance(item, EffectRow | InteractionRow)
        by_field = {figure.key.rpartition(".")[2]: figure for figure in item.figures}
        if isinstance(item, EffectRow):
            start, end = lever_arms(item, context.report_input)
            moved = PRICE_STEP_TEMPLATE.format(first=_arm(context, start).text, second=_arm(context, end).text)
            what = humanize("dimension", item.setting)
        else:
            arms = lever_arms(item, context.report_input)
            moved = " vs ".join(_arm(context, arm).text for arm in arms[::2])
            what = f"{humanize('dimension', item.setting)} × {humanize('dimension', item.across)}"
        yield {
            "setting": _words(what),
            "move": _words(moved),
            "difference": _read(item.figures[0]),
            "interval": _read(by_field.get("interval")),
            "level": _read(by_field.get("level")),
            "verdict": _words(item.verdict if isinstance(item, InteractionRow) else None),
        }


#: The least monthly price a table writes in whole dollars, as the ``usd_whole`` filter reads one.
_WHOLE_DOLLARS_FROM = 10.0


def _prices(context: ChartContext, _laid: _Laid) -> Iterator[_Row]:
    steps = [(row.item, row.interval) for row in context.rows() if isinstance(row.item, PriceStep)]
    monthly = any(step.monthly for step, _ in steps)
    drawn = quantity("dollars" if monthly else "usd_per_1k", [step.value for step, _ in steps])
    amounts = [
        abs(value)
        for step, interval in steps
        for value in (step.value, *(interval or ()), *((step.open_from,) if step.open_from is not None else ()))
    ]
    whole = monthly and min(amounts, default=0.0) * drawn.factor >= _WHOLE_DOLLARS_FROM

    def dollars(value: float) -> float:
        return float(round(value * drawn.factor)) if whole else value * drawn.factor

    def name(arm: str) -> str:
        return context.look.arm(arm).name if arm in context.look.arms else arm

    for step, interval in steps:
        row = {
            "step": _words(PRICE_STEP_TEMPLATE.format(first=name(step.first), second=name(step.second))),
            "per_point": _number(
                dollars(step.value),
                drawn.unit,
                (dollars(interval[0]), dollars(interval[1])) if interval is not None else None,
            ),
        }
        if step.open_from is not None and context.level:
            # An interval open above has one end, which no cell's two ends hold: its interval cell is laid out here.
            low = with_unit(dollars(step.open_from), drawn.unit)
            row[f"per_point{_INTERVAL_COLUMN}"] = _words(OPEN_ABOVE.format(low=low))
        yield row


#: Each source's rows, and its columns in order: its fixed ones, with the measure columns its reader lays out (a wide
#: table's) where :data:`_MEASURE_COLUMN` stands.
_SOURCES: dict[str, tuple[Callable[[ChartContext, _Laid], Iterator[_Row]], tuple[str, ...]]] = {
    "readings": (_readings, ("arm", "change", "reading")),
    "contrasts": (_contrasts, ("arm", "measure", "change", "verdict")),
    "measures": (_figures, ("arm", _MEASURE_COLUMN)),
    "judged": (_figures, ("arm", _MEASURE_COLUMN)),
    "configs": (_figures, ("arm", _MEASURE_COLUMN)),
    "scoreboard": (_scoreboard, ("arm", "figure", "value")),
    "floors": (_floors, ("floor", "arm", "status", "worst_drop", "margin")),
    "strata": (_slices, ("slice", "arm", _MEASURE_COLUMN)),
    "tasks": (_slices, ("slice", "arm", _MEASURE_COLUMN)),
    "examples": (_examples, ("trial", "case", "stratum", "arm", "outcome", "reason")),
    "cells": (_cells, ("cell", "arm", _MEASURE_COLUMN, "cases")),
    "confusion": (_confusion, ("axis", "arm", "expected", "predicted", "value")),
    "human_pref": (_human_pref, ("slice", "prefers_control")),
    "prices": (_prices, ("step", "per_point")),
    "bar_reads": (_bar_reads, ("aspect", "measure", "arm", "value", "bar", "status")),
    "compare": (_compare, ("place", "first", "second", "difference", "interval", "level", "basis")),
    "settle": (
        _settle,
        (
            "arm",
            "against",
            "estimate",
            "line",
            "distance",
            "settles",
            "cases_needed",
            "cases_needed_two_sided",
            "trials_needed",
            "repeats",
            "today_cases",
            "today_trials",
            "wrong_side",
            "level",
        ),
    ),
    "levers": (_levers, ("setting", "move", "difference", "interval", "level", "verdict")),
}


def _interval_text(cell: TableCell) -> TableCell:
    """A figure's interval as its own cell, its two ends in the figure's unit; a dash where it carries none."""
    if cell.low is None or cell.high is None:
        return TableCell(value=None)
    return TableCell(value=f"{with_unit(cell.low, cell.unit)} to {with_unit(cell.high, cell.unit)}")


def _with_intervals(columns: list[TableColumn], rows: list[_Row], level: str) -> tuple[list[TableColumn], list[_Row]]:
    """The columns, each one whose figures carry an interval followed by its interval column, and the rows with it: a
    row's own interval cell where its reader laid one out (an interval open above, which no cell's two ends hold),
    else its figure's two ends."""
    laid: list[TableColumn] = []
    spelled: list[str] = []
    for column in columns:
        laid.append(column)
        own = f"{column.key}{_INTERVAL_COLUMN}"
        if any(row[column.key].low is not None or own in row for row in rows if column.key in row):
            key = f"{column.key}{_INTERVAL_COLUMN}"
            laid.append(TableColumn(key=key, header=INTERVAL_HEADER.format(level=level).strip().capitalize()))
            spelled.append(column.key)
    filled = [
        {
            **row,
            **{
                f"{key}{_INTERVAL_COLUMN}": row.get(f"{key}{_INTERVAL_COLUMN}")
                or (_interval_text(row[key]) if key in row else TableCell(value=None))
                for key in spelled
            },
        }
        for row in rows
    ]
    return laid, [{column.key: row[column.key] for column in laid if column.key in row} for row in filled]


def _table(context: ChartContext) -> Drawn:
    reader, keys = _SOURCES[context.query.source]
    measured: _Laid = []
    rows = list(reader(context, measured))
    if not rows:
        raise context.refuse("it selects no row to lay out")
    columns: list[TableColumn] = []
    for key in keys:
        if key == _MEASURE_COLUMN:
            columns += [TableColumn(key=on, header=header) for on, header in measured]
        else:
            columns.append(TableColumn(key=key, header=TABLE_HEADERS[key]))
    columns, rows = _with_intervals(columns, rows, context.spec.level)
    notes = (
        price_notes(context)
        if context.query.source == "prices"
        else settle_notes(context)
        if context.query.source == "settle"
        else computed_notes(context)
    )
    table = TableBlock(
        section=PLANNED_SECTION,
        name=context.query.source,
        title=context.title,
        columns=columns,
        rows=rows,
        order=TABLE_ORDER,
        total_rows=len(rows),
    )
    return Drawn(table, notes)


#: The kinds this module resolves.
RESOLVERS: dict[str, KindResolver] = {"table": _table}
