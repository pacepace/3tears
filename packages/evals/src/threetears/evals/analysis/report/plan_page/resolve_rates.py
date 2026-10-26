"""The charts of rates against a line: dot strips, range bars, and the frontier scatter.

**dot_interval** marks each figure with a faint interval against its floor:

- *readings*, *contrasts*: each change as the diverging chart reads it (the one owner of how a reading or a contrast
  reads as a change, :mod:`~threetears.evals.analysis.report.plan_page.resolve_changes`), a dot per row, panels by
  measure where it reads several;
- *measures*, *judged*: an arm per row, a panel per measure where it reads several;
- *tasks*, *strata*: an arm per row, a panel per slice (and measure) where there are several, or a panel per arm, a
  row per slice, where the plan asks for panels by arm;
- *floors*: an arm per row, a panel per floor where there are several: each dot is the largest drop its interval
  allows, against the floor's margin below zero, which a caption says; a broken floor is hollow.

The line is the stored one, else the one criterion on the measure read where the chart reads it (overall, in every
task, or in the one task a panel draws). A dot that misses its line is hollow. A strip always states a level (the
kit's contract): the stored one, else the one level the figures it reads declare, else none, when no interval is
drawn.

**range_bars** draw each latency from its p50 to its p95: a group per task (or per measure, over measures), a row per
arm; a figure holding only its p95 is drawn as its end mark alone, which a caption names. The line is the stored one,
else the one p95 criterion read where the chart reads it, and a bar ending past its budget is hollow.

**scatter_frontier** plots two measures that trade off, an arm (or config) per point, with the Pareto line: the
query's two measures, else the two its rows hold; up is the one where higher is better, across the other. Over tasks,
a panel per task. A point that fails a gate (:func:`~threetears.evals.analysis.report_gates.gate_check`, in the task
a panel draws) is hollow; on a summary, an arm that does not survive is faded. Each axis carries its criterion's line
(the stored line and domain are the y axis's). An x axis spanning more than ten times is logarithmic, which a
caption says.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.charts import (
    BarsChart,
    ChartReference,
    DotIntervalChart,
    DotIntervalPanel,
    DotIntervalRow,
    RangeBarsChart,
    RangeGroup,
    RangeRow,
    ScatterChart,
    ScatterPanel,
    ScatterPoint,
)
from threetears.evals.analysis.report.plan_page import resolve_changes
from threetears.evals.analysis.report.plan_page.charting import (
    ChartContext,
    CriterionWhere,
    Drawn,
    KindResolver,
    chart_block,
    criterion_line,
    figure_unit,
    interval_words,
    measure_words,
    misses_line,
    missing_note,
    reference_line,
    slice_keys,
    slice_name,
    stat_interval,
    stat_value,
)
from threetears.evals.analysis.report.plan_page.look import (
    Quantity,
    band_domain,
    centred_domain,
    figure_text,
    magnitude_domain,
    quantity,
)
from threetears.evals.analysis.report.plan_page.words import REFERENCE_TEMPLATE
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report_input import FloorRead, ScoreboardRow, Stat, Stratum, Task
from threetears.evals.analysis.style import humanize

__all__ = ["RESOLVERS"]

#: The caption under a dot strip of floors.
_FLOOR_NOTE = "Each dot is the largest drop its interval allows; a floor holds where the dot stays above its line."
#: A dot's tooltip line where it misses its line.
_MISSES_TEMPLATE = "Misses the {line}"
#: The caption under a scatter whose x axis is logarithmic.
_LOG_NOTE = "The horizontal axis is logarithmic: each step is a multiple."
#: A range bar's tooltip line: its two ends.
_RANGE_TEMPLATE = "p50 {p50} · p95 {p95}"
#: A range bar's tooltip line where the input holds only its p95, which is drawn as its end mark alone.
_P95_TEMPLATE = "p95 {p95}; no p50 in the evidence"
#: The caption naming the range bars drawn as their p95 alone.
_NO_MEDIAN_TEMPLATE = "Drawn at the p95 alone, with no p50 in the evidence: {missing}."

#: The spread of an x axis past which it is drawn logarithmic.
_LOG_SPREAD = 10.0

#: The scoreboard's figures that are money, each to how it is stated, and those that count by cause.
_SCOREBOARD_UNITS = {"reply_cost_usd": "usd", "test_cost_usd": "dollars"}
_BY_CAUSE = frozenset({"lost", "failed"})
#: The scoreboard's figures where less is better.
_SCOREBOARD_LOWER = frozenset({"reply_cost_usd", "test_cost_usd", "lost", "failed"})


def _arm_label(context: ChartContext, arm: str) -> str:
    return context.look.arm(arm).label if arm in context.look.arms else arm


def _arm_name(context: ChartContext, arm: str) -> str:
    return context.look.arm(arm).name if arm in context.look.arms else arm


def _with_line(tip: list[str], value: float, line: ChartReference | None) -> list[str]:
    return [*tip, _MISSES_TEMPLATE.format(line=line.label or line.kind)] if line and misses_line(value, line) else tip


# --- dot_interval -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Dot:
    """One drawn dot: its row's id and label, the arm it is (for its colour and mark), its value and interval."""

    id: str
    label: str
    arm: str
    value: float
    interval: tuple[float, float] | None
    tip: list[str]
    strong: bool | None = None
    hollow: bool = False


@dataclass(frozen=True)
class _Strip:
    """One strip (a panel, or the whole chart): its title, the arm it stands for (for a chip), its dots and line."""

    title: str
    arm: str
    dots: list[_Dot]
    line: ChartReference | None


def _declared_level(context: ChartContext, keys: Iterable[str]) -> str:
    """The one interval level the measures and judged dimensions a strip reads declare; empty for none or several."""
    report_input = context.report_input
    levels = {
        level
        for key in keys
        for table in (report_input.measures, report_input.judged)
        if key in table and (level := table[key].level)
    }
    return levels.pop() if len(levels) == 1 else ""


def _dot(
    context: ChartContext,
    *,
    row_id: str,
    label: str,
    arm: str,
    value: float,
    interval: tuple[float, float] | None,
    drawn: Quantity,
) -> _Dot:
    tip = [label, figure_text(value, drawn)]
    if interval is not None:
        tip[1] += f", {interval_words(*interval, drawn)}"
    strong = True if context.summary and arm and arm == context.look.recommended else None
    return _Dot(id=row_id, label=label, arm=arm, value=value, interval=interval, tip=tip, strong=strong)


def _change_strips(context: ChartContext) -> tuple[list[_Strip], Quantity, str, tuple[str, ...], bool]:
    """readings or contrasts: the diverging chart's rows as dots, a strip per group (a measure, where several)."""
    as_bars = dataclasses.replace(context, spec=context.spec.model_copy(update={"kind": "diverging_bars"}))
    drawn = resolve_changes.RESOLVERS["diverging_bars"](as_bars)
    data = drawn.block.data if isinstance(drawn.block, ChartBlock) else None
    assert isinstance(data, BarsChart) and data.rows is not None, "readings and contrasts diverge as rows"
    unit = quantity("points", change=True)
    grouped: dict[str, list[_Dot]] = {}
    for row in data.rows:
        if row.value is None:
            continue
        interval = (row.lo, row.hi) if row.lo is not None and row.hi is not None else None
        arm = row.id or ""
        grouped.setdefault(row.group or "", []).append(
            _Dot(
                id=arm,
                label=row.label,
                arm=arm,
                value=row.value,
                interval=interval,
                tip=list(row.tip or [row.label]),
                strong=row.strong,
            )
        )
    fmt_unit = Quantity(1.0, data.fmt or unit.fmt, unit.unit, change=True)
    line = data.reference
    strips = [_Strip(title=title, arm="", dots=dots, line=line) for title, dots in grouped.items()]
    return strips, fmt_unit, context.level, drawn.notes, True


def _figure_strips(context: ChartContext) -> tuple[list[_Strip], Quantity, str, tuple[str, ...], bool]:
    """measures or judged: an arm per row, a strip per measure."""
    report_input = context.report_input
    rows = context.rows()
    keys = list(dict.fromkeys(row.key for row in rows))
    level = context.level or _declared_level(context, keys)
    figures = {(row.key, row.arm): row.item for row in rows if isinstance(row.item, Stat)}
    reads = {
        place: read for place, stat in figures.items() if (read := stat_value(report_input, place[0], stat)) is not None
    }
    drawn = quantity(figure_unit(report_input, keys[0]) if keys else "", [read[0] for read in reads.values()])
    strips = []
    for key in keys:
        statistics = {read[1] for (on, _), read in reads.items() if on == key}
        line = reference_line(context, drawn) or (
            criterion_line(context, key, drawn, where="overall", statistic=statistics.pop())
            if len(statistics) == 1
            else None
        )
        dots = []
        for arm in context.arm_order(arm for (on, arm) in reads if on == key):
            value, statistic = reads[key, arm]
            interval = stat_interval(figures[key, arm]) if level and statistic == "mean" else None
            shown = (interval[0] * drawn.factor, interval[1] * drawn.factor) if interval is not None else None
            dot = _dot(
                context,
                row_id=arm,
                label=_arm_label(context, arm),
                arm=arm,
                value=value * drawn.factor,
                interval=shown,
                drawn=drawn,
            )
            dots.append(dataclasses.replace(dot, tip=_with_line(dot.tip, dot.value, line)))
        strips.append(_Strip(title=measure_words(report_input, key), arm="", dots=dots, line=line))
    return strips, drawn, level, (), False


def _slice_strips(context: ChartContext) -> tuple[list[_Strip], Quantity, str, tuple[str, ...], bool]:
    """tasks or strata: an arm per row and a strip per slice and measure, or a strip per arm (a row per slice)."""
    report_input = context.report_input
    slices = [row.item for row in context.rows() if isinstance(row.item, Stratum | Task)]
    keys = slice_keys(context, slices)
    level = context.level or _declared_level(context, keys)
    allowed = list(context.query.arms) or list(context.look.arms)
    arms = [arm for arm in context.arm_order(allowed) if any(arm in item.per_arm for item in slices)]
    found: list[tuple[Stratum | Task, str, str, float, str, tuple[float, float] | None]] = []
    for item in slices:
        for key in keys:
            for arm in arms:
                stat = item.per_arm.get(arm, {}).get(key)
                read = stat_value(report_input, key, stat) if stat is not None else None
                if stat is None or read is None:
                    continue
                interval = stat_interval(stat) if level and read[1] == "mean" else None
                found.append((item, key, arm, read[0], read[1], interval))
    drawn = quantity(figure_unit(report_input, keys[0]), [value for *_, value, _, _ in found])
    tasks = context.query.source == "tasks"

    def line_of(key: str, item: Stratum | Task | None, statistics: set[str]) -> ChartReference | None:
        stored = reference_line(context, drawn)
        if stored is not None or len(statistics) != 1:
            return stored
        if not tasks:
            return None
        where: CriterionWhere = ("task", item.id) if item is not None else "tasks"
        return criterion_line(context, key, drawn, where=where, statistic=next(iter(statistics)))

    def place(item: Stratum | Task, key: str) -> str:
        name = slice_name(context, item)
        return name if len(keys) == 1 else f"{name} · {measure_words(report_input, key)}"

    strips: list[_Strip] = []
    if context.query.small_multiples_by == "arm":
        line = line_of(keys[0], None, {statistic for *_, statistic, _ in found}) if len(keys) == 1 else None
        for arm in arms:
            dots = []
            for item, key, on, value, _, interval in found:
                if on != arm:
                    continue
                shown = (interval[0] * drawn.factor, interval[1] * drawn.factor) if interval is not None else None
                dot = _dot(
                    context,
                    row_id=f"{item.id}/{key}",
                    label=place(item, key),
                    arm=arm,
                    value=value * drawn.factor,
                    interval=shown,
                    drawn=drawn,
                )
                dots.append(dataclasses.replace(dot, tip=_with_line(dot.tip, dot.value, line)))
            strips.append(_Strip(title=_arm_name(context, arm), arm=arm, dots=dots, line=line))
        return strips, drawn, level, (), False
    for item in slices:
        for key in keys:
            here = [entry for entry in found if entry[0] is item and entry[1] == key]
            if not here:
                continue
            line = line_of(key, item, {entry[4] for entry in here})
            dots = []
            for _, _, arm, value, _, interval in here:
                shown = (interval[0] * drawn.factor, interval[1] * drawn.factor) if interval is not None else None
                dot = _dot(
                    context,
                    row_id=arm,
                    label=_arm_label(context, arm),
                    arm=arm,
                    value=value * drawn.factor,
                    interval=shown,
                    drawn=drawn,
                )
                dots.append(dataclasses.replace(dot, tip=_with_line(dot.tip, dot.value, line)))
            strips.append(_Strip(title=place(item, key), arm="", dots=dots, line=line))
    return strips, drawn, level, (), False


def _floor_strips(context: ChartContext) -> tuple[list[_Strip], Quantity, str, tuple[str, ...], bool]:
    """floors: an arm per row at the largest drop its interval allows, a strip per floor against its margin."""
    drawn = quantity("points", change=True)
    level = context.level
    floors: dict[str, list[FloorRead]] = {}
    for row in context.rows():
        assert isinstance(row.item, FloorRead)
        floors.setdefault(row.item.floor, []).append(row.item)
    strips = []
    missing: list[str] = []
    for floor, reads in floors.items():
        margins = {float(read.margin) for read in reads if read.margin is not None}
        line = reference_line(context, drawn)
        if line is None and len(margins) == 1:
            below = -next(iter(margins))
            line = ChartReference(
                value=below,
                label=REFERENCE_TEMPLATE.format(kind="floor", value=figure_text(below, drawn)),
                kind="floor",
            )
        dots = []
        by_arm = {read.arm: read for read in reads}
        for arm in context.arm_order(by_arm):
            read = by_arm[arm]
            low = (
                -float(read.worst_drop)
                if read.worst_drop is not None
                else float(read.interval[0])
                if read.interval is not None
                else None
            )
            if low is None:
                missing.append(f"{_arm_name(context, arm)}, {humanize('dimension', floor)}")
                continue
            interval = (float(read.interval[0]), float(read.interval[1])) if level and read.interval else None
            dot = _dot(
                context, row_id=arm, label=_arm_label(context, arm), arm=arm, value=low, interval=interval, drawn=drawn
            )
            dots.append(dataclasses.replace(dot, tip=[*dot.tip, read.status], hollow=read.status == "broken"))
        strips.append(_Strip(title=humanize("dimension", floor), arm="", dots=dots, line=line))
    return strips, drawn, level, (_FLOOR_NOTE, *missing_note(missing)), True


def _dot_row(context: ChartContext, dot: _Dot, line: ChartReference | None) -> DotIntervalRow:
    look = context.look.arm(dot.arm) if dot.arm in context.look.arms else None
    hollow = misses_line(dot.value, line) or dot.hollow
    return DotIntervalRow(
        id=dot.id,
        label=dot.label,
        value=dot.value,
        lo=dot.interval[0] if dot.interval is not None else None,
        hi=dot.interval[1] if dot.interval is not None else None,
        color=look.token if look is not None else None,
        shape=look.shape if look is not None else None,
        hollow=True if hollow else None,
        strong=dot.strong,
        tip=dot.tip,
    )


def _dot_interval(context: ChartContext) -> Drawn:
    source = context.query.source
    if source in ("readings", "contrasts"):
        strips, drawn, level, notes, centred = _change_strips(context)
    elif source in ("measures", "judged"):
        strips, drawn, level, notes, centred = _figure_strips(context)
    elif source in ("tasks", "strata"):
        strips, drawn, level, notes, centred = _slice_strips(context)
    elif source == "floors":
        strips, drawn, level, notes, centred = _floor_strips(context)
    else:
        raise context.refuse("a dot strip draws readings, contrasts, measures, judged, floors, tasks or strata")
    strips = [strip for strip in strips if strip.dots]
    if not strips:
        raise context.refuse("it selects no figure to draw")
    ends = [end for strip in strips for dot in strip.dots for end in (dot.value, *(dot.interval or ()))]
    ends += [strip.line.value for strip in strips if strip.line is not None]
    domain = context.spec.domain or (centred_domain(ends, drawn) if centred else band_domain(ends, drawn))
    if len(strips) == 1:
        (strip,) = strips
        data = DotIntervalChart(
            rows=[_dot_row(context, dot, strip.line) for dot in strip.dots],
            domain=(domain[0], domain[1]),
            fmt=drawn.fmt,
            level=level,
            reference=strip.line,
            emphasis=context.emphasis,
        )
        return Drawn(chart_block(context, data), notes)
    panels = []
    for strip in strips:
        look = context.look.arm(strip.arm) if strip.arm in context.look.arms else None
        panels.append(
            DotIntervalPanel(
                title=strip.title,
                tag=look.tag if look is not None else None,
                color=look.token if look is not None else None,
                rows=[_dot_row(context, dot, strip.line) for dot in strip.dots],
                reference=strip.line,
            )
        )
    data = DotIntervalChart(
        domain=(domain[0], domain[1]), fmt=drawn.fmt, level=level, emphasis=context.emphasis, panels=panels
    )
    return Drawn(chart_block(context, data), notes)


# --- range_bars ---------------------------------------------------------------------------------------------------


def _spread_stat(stat: Stat | None) -> tuple[float | None, float] | None:
    """A latency's p50 (None where the input holds only its p95) and p95; None without a p95."""
    if stat is None or stat.p95 is None:
        return None
    return (float(stat.p50) if stat.p50 is not None else None), float(stat.p95)


def _range_bars(context: ChartContext) -> Drawn:
    report_input = context.report_input
    source = context.query.source
    groups: list[tuple[str, CriterionWhere, str, list[tuple[str, tuple[float | None, float]]]]] = []
    if source == "measures":
        rows = context.rows()
        keys = list(dict.fromkeys(row.key for row in rows if isinstance(row.item, Stat) and _spread_stat(row.item)))
        for key in keys:
            spreads = {
                row.arm: spread
                for row in rows
                if row.key == key and isinstance(row.item, Stat) and (spread := _spread_stat(row.item))
            }
            groups.append(
                (
                    measure_words(report_input, key),
                    "overall",
                    key,
                    [(arm, spreads[arm]) for arm in context.arm_order(spreads)],
                )
            )
    elif source == "tasks":
        slices = [row.item for row in context.rows() if isinstance(row.item, Task)]
        held = [
            key
            for key in dict.fromkeys(key for item in slices for figures in item.per_arm.values() for key in figures)
            if any(_spread_stat(figures.get(key)) for item in slices for figures in item.per_arm.values())
        ]
        keys = list(context.query.measures) or held
        allowed = list(context.query.arms) or list(context.look.arms)
        for item in slices:
            for key in keys:
                spreads = {
                    arm: spread
                    for arm in allowed
                    if (spread := _spread_stat(item.per_arm.get(arm, {}).get(key))) is not None
                }
                if spreads:
                    name = slice_name(context, item)
                    label = name if len(keys) == 1 else f"{name} · {measure_words(report_input, key)}"
                    groups.append(
                        (label, ("task", item.id), key, [(arm, spreads[arm]) for arm in context.arm_order(spreads)])
                    )
    else:
        raise context.refuse("range bars draw measures or tasks")
    if not groups:
        raise context.refuse("it selects no figure with a p95")
    units = {figure_unit(report_input, key) for _, _, key, _ in groups}
    if len(units) != 1:
        raise context.refuse(f"its measures are in more than one unit ({sorted(units)})")
    values = [end for *_, bars in groups for _, spread in bars for end in spread if end is not None]
    drawn = quantity(units.pop(), values)
    keys_drawn = {key for _, _, key, _ in groups}
    line = reference_line(context, drawn)
    if line is None and len(keys_drawn) == 1:
        (key,) = keys_drawn
        where: CriterionWhere = "overall" if source == "measures" else "tasks"
        line = criterion_line(context, key, drawn, where=where, statistic="p95")
    drawn_groups = []
    no_median: list[str] = []
    for label, _, _, bars in groups:
        drawn_rows = []
        for arm, (p50, p95) in bars:
            high = p95 * drawn.factor
            low = p50 * drawn.factor if p50 is not None else high
            look = context.look.arm(arm) if arm in context.look.arms else None
            said = (
                _RANGE_TEMPLATE.format(p50=figure_text(low, drawn), p95=figure_text(high, drawn))
                if p50 is not None
                else _P95_TEMPLATE.format(p95=figure_text(high, drawn))
            )
            if p50 is None:
                no_median.append(f"{_arm_name(context, arm)}, {label}")
            tip = [f"{_arm_name(context, arm)} · {label}", said]
            drawn_rows.append(
                RangeRow.model_validate(
                    {
                        "label": _arm_label(context, arm),
                        "from": low,
                        "to": high,
                        "color": look.token if look is not None else None,
                        "shape": look.shape if look is not None else None,
                        "hollow": True if misses_line(high, line) else None,
                        "tip": _with_line(tip, high, line),
                    }
                )
            )
        drawn_groups.append(RangeGroup(label=label, rows=drawn_rows))
    ends = [*(value * drawn.factor for value in values), *([line.value] if line is not None else [])]
    domain = context.spec.domain or magnitude_domain(ends)
    data = RangeBarsChart(groups=drawn_groups, domain=(domain[0], domain[1]), fmt=drawn.fmt, reference=line)
    notes = (_NO_MEDIAN_TEMPLATE.format(missing="; ".join(no_median)),) if no_median else ()
    return Drawn(chart_block(context, data), notes)


# --- scatter_frontier ---------------------------------------------------------------------------------------------


def _higher_is_better(context: ChartContext, key: str) -> bool | None:
    """Which way a scatter's measure is better: its declaration, a judged dimension's always up, a scoreboard
    figure's by what it counts, else the criteria on it; None where nothing says."""
    report_input = context.report_input
    if context.query.source == "scoreboard":
        return key not in _SCOREBOARD_LOWER
    measure = report_input.measures.get(key)
    if measure is not None and measure.higher_is_better is not None:
        return measure.higher_is_better
    if key in report_input.judged and measure is None:
        return True
    directions = {criterion.direction for criterion in report_input.criteria if criterion.measure == key}
    return directions.pop() == "higher_is_better" if len(directions) == 1 else None


def _axes(context: ChartContext, keys: Sequence[str]) -> tuple[str, str]:
    """``(x, y)``: up the one where higher is better, across the other; else the first named up."""
    if len(keys) != 2:
        raise context.refuse(f"a scatter reads two measures, and it reads {list(keys)}")
    first, second = keys
    ups = [_higher_is_better(context, key) for key in keys]
    if ups == [False, True] or ups == [None, True] or ups == [False, None]:
        return first, second
    return second, first


@dataclass(frozen=True)
class _Point:
    """One point: the arm, the task it sits in (for its gates; none overall), and its two read values."""

    arm: str
    task: str | None
    x: float
    y: float
    x_statistic: str
    y_statistic: str


def _scoreboard_value(row: ScoreboardRow, key: str) -> float | None:
    held = getattr(row, key, None)
    if key in _BY_CAUSE and isinstance(held, dict):
        return float(sum(held.values()))
    return float(held) if isinstance(held, int | float) else None


def _scatter_points(context: ChartContext) -> tuple[list[_Point], str, str, str, str]:
    """Every point, the x and y keys, and the x and y units."""
    report_input = context.report_input
    source = context.query.source
    rows = context.rows()
    points: list[_Point] = []
    if source == "scoreboard":
        x, y = _axes(context, list(context.query.measures))
        boards = {row.arm: row.item for row in rows if isinstance(row.item, ScoreboardRow)}
        for arm in context.arm_order(boards):
            across, up = _scoreboard_value(boards[arm], x), _scoreboard_value(boards[arm], y)
            if across is not None and up is not None:
                points.append(_Point(arm, None, across, up, "mean", "mean"))
        return points, x, y, _SCOREBOARD_UNITS.get(x, ""), _SCOREBOARD_UNITS.get(y, "")
    if source == "tasks":
        slices = [row.item for row in rows if isinstance(row.item, Task)]
        keys = list(context.query.measures) or list(
            dict.fromkeys(key for item in slices for figures in item.per_arm.values() for key in figures)
        )
        x, y = _axes(context, keys)
        allowed = list(context.query.arms) or list(context.look.arms)
        for item in slices:
            for arm in context.arm_order(arm for arm in allowed if arm in item.per_arm):
                held = item.per_arm[arm]
                xs = stat_value(report_input, x, held[x]) if x in held else None
                ys = stat_value(report_input, y, held[y]) if y in held else None
                if xs is not None and ys is not None:
                    points.append(_Point(arm, item.id, xs[0], ys[0], xs[1], ys[1]))
    else:
        keys = list(context.query.measures) or list(dict.fromkeys(row.key for row in rows))
        x, y = _axes(context, keys)
        figures = {(row.key, row.arm): row.item for row in rows if isinstance(row.item, Stat)}
        arms = dict.fromkeys(arm for (_, arm) in figures)
        for arm in context.arm_order(arms):
            xs = stat_value(report_input, x, figures[x, arm]) if (x, arm) in figures else None
            ys = stat_value(report_input, y, figures[y, arm]) if (y, arm) in figures else None
            if xs is not None and ys is not None:
                points.append(_Point(arm, None, xs[0], ys[0], xs[1], ys[1]))
    return points, x, y, figure_unit(report_input, x), figure_unit(report_input, y)


def _log_domain(values: Sequence[float]) -> tuple[float, float]:
    return 10.0 ** math.floor(math.log10(min(values))), 10.0 ** math.ceil(math.log10(max(values)))


def _scatter(context: ChartContext) -> Drawn:
    if context.query.source not in ("measures", "judged", "scoreboard", "configs", "tasks"):
        raise context.refuse("a scatter draws measures, judged, scoreboard, configs or tasks")
    points, x_key, y_key, x_unit, y_unit = _scatter_points(context)
    if not points:
        raise context.refuse("no arm holds both of its figures")
    x_drawn = quantity(x_unit, [point.x for point in points])
    y_drawn = quantity(y_unit, [point.y for point in points])
    xs = [point.x * x_drawn.factor for point in points]
    ys = [point.y * y_drawn.factor for point in points]
    tasks = list(dict.fromkeys(point.task for point in points))
    scoreboard = context.query.source == "scoreboard"

    def lines(task: str | None) -> tuple[ChartReference | None, ChartReference | None]:
        if scoreboard:
            return reference_line(context, y_drawn), None
        here = [point for point in points if point.task == task]
        where: CriterionWhere = ("task", task) if task is not None else "overall"
        y_stats = {point.y_statistic for point in here}
        x_stats = {point.x_statistic for point in here}
        y_line = reference_line(context, y_drawn) or (
            criterion_line(context, y_key, y_drawn, where=where, statistic=y_stats.pop()) if len(y_stats) == 1 else None
        )
        x_line = (
            criterion_line(context, x_key, x_drawn, where=where, statistic=x_stats.pop()) if len(x_stats) == 1 else None
        )
        return y_line, x_line

    def drawn_points(task: str | None) -> list[ScatterPoint]:
        drawn = []
        for point in points:
            if point.task != task:
                continue
            look = context.look.arm(point.arm) if point.arm in context.look.arms else None
            fails = (
                bool(context.report_input.criteria)
                and gate_check(context.report_input, point.arm, task=point.task).status == "fail"
            )
            x_value, y_value = point.x * x_drawn.factor, point.y * y_drawn.factor
            drawn.append(
                ScatterPoint(
                    id=point.arm,
                    x=x_value,
                    y=y_value,
                    label=_arm_label(context, point.arm),
                    narrow_label=look.tag if look is not None else None,
                    color=look.token if look is not None else None,
                    shape=look.shape if look is not None else None,
                    hollow=True if fails else None,
                    muted=True if context.summary and point.arm not in context.look.survivors else None,
                    tip=[
                        _arm_name(context, point.arm),
                        f"{measure_words(context.report_input, x_key)} {figure_text(x_value, x_drawn)}",
                        f"{measure_words(context.report_input, y_key)} {figure_text(y_value, y_drawn)}",
                    ],
                )
            )
        return drawn

    x_ends = xs + [line.value for task in tasks if (line := lines(task)[1]) is not None]
    y_ends = ys + [line.value for task in tasks if (line := lines(task)[0]) is not None]
    positive = [value for value in x_ends if value > 0]
    x_log = len(positive) == len(x_ends) and max(positive) / min(positive) > _LOG_SPREAD
    stored = context.spec.domain
    x_domain = _log_domain(positive) if x_log else band_domain(x_ends, x_drawn)
    y_domain = (stored[0], stored[1]) if stored else band_domain(y_ends, y_drawn)

    def chart(**part: object) -> ScatterChart:
        return ScatterChart.model_validate(
            {
                "x_domain": x_domain,
                "y_domain": y_domain,
                "x_fmt": x_drawn.fmt,
                "y_fmt": y_drawn.fmt,
                "x_log": True if x_log else None,
                "x_title": humanize("dimension", x_key) if scoreboard else measure_words(context.report_input, x_key),
                "y_title": humanize("dimension", y_key) if scoreboard else measure_words(context.report_input, y_key),
                "frontier": True,
                "emphasis": context.emphasis,
                **part,
            }
        )

    notes = (_LOG_NOTE,) if x_log else ()
    if len(tasks) == 1:
        y_line, x_line = lines(tasks[0])
        data = chart(points=drawn_points(tasks[0]), reference=y_line, x_reference=x_line)
        return Drawn(chart_block(context, data), notes)
    panels = []
    for task in tasks:
        y_line, x_line = lines(task)
        title = next((slice_name(context, item) for item in context.report_input.tasks if item.id == task), task or "")
        panels.append(ScatterPanel(title=title, points=drawn_points(task), reference=y_line, x_reference=x_line))
    return Drawn(chart_block(context, chart(panels=panels)), notes)


#: The kinds this module resolves.
RESOLVERS: dict[str, KindResolver] = {
    "dot_interval": _dot_interval,
    "range_bars": _range_bars,
    "scatter_frontier": _scatter,
}
