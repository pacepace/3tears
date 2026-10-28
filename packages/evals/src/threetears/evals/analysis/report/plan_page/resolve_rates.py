"""The charts of rates against a line: dot strips, range bars, and the frontier scatter.

**dot_interval** marks each figure with a faint interval against its floor:

- *readings*, *contrasts*: each change as the diverging chart reads it (the one owner of how a reading or a contrast
  reads as a change, :mod:`~threetears.evals.analysis.report.plan_page.resolve_changes`), a dot per row, panels by
  measure where it reads several;
- *measures*, *judged*: an arm per row, a panel per measure where it reads several;
- *tasks*, *strata*: an arm per row, a panel per slice (and measure) where there are several, or a panel per arm, a
  row per slice, where the plan asks for panels by arm;
- *floors*: an arm per row, a panel per floor where there are several: each dot is the largest drop its interval
  allows, against the floor's margin below zero, which a caption says; a broken floor is hollow;
- *bar_reads*: an arm per row at its value with the interval the bar was read on (the pipeline's, at its interval
  level), a panel per aspect, measure and bar, against the bar's line; a missed bar is hollow;
- *compare*, *levers*: as the diverging chart reads them, a dot per row, a strip per group.

The line is the stored one, else the one criterion on the measure read where the chart reads it (overall, in every
task, or in the one task a panel draws). A dot that misses its line is hollow. A strip always states a level (the
kit's contract): the stored one, else the one level the figures it reads declare, else none, when no interval is
drawn. A rate's interval is drawn within 0 to 100, where a rate lies. Each strip is drawn in its own measure's unit
(the one-unit rule, :func:`~threetears.evals.analysis.report.plan_page.charting.unit_refusal`): where the measures
it reads are in several, each panel carries its unit's format and span, the panels of one unit sharing one span; a
strip per arm holds every measure, so its measures must share one unit, each drawn at its own scale.

**range_bars** draw each latency from its p50 to its p95: a group per task (or per measure, over measures), a row per
arm; a figure holding only its p95 is drawn as its end mark alone, which a caption names. The line is the stored one,
else the one p95 criterion read where the chart reads it, and a bar ending past its budget is hollow, its tooltip
saying so. A legend names the arms, in the page's order. Measures in more than one unit are refused: the bars share
one axis, on which measures drawn alike (a share beside a percent) are each drawn at their own scale.

**scatter_frontier** plots two measures that trade off, an arm (or config) per point, with the Pareto line: the
query's two measures, else the two its rows hold; up is the one where higher is better, across the other. Over tasks,
a panel per task. A point that fails a gate (:func:`~threetears.evals.analysis.report_gates.gate_check`, in the task
a panel draws) is hollow; on a summary, an arm that does not survive is faded. Each axis carries its criterion's line
(the stored line and domain are the y axis's). An x axis spanning more than ten times is logarithmic, which a
caption says. A config takes its first factor's colour, which a legend names
(:func:`~threetears.evals.analysis.report.plan_page.charting.factor_token`); a panel of more than eight points labels
only the stressed and the recommended one
(:func:`~threetears.evals.analysis.report.plan_page.charting.point_labels`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from threetears.evals.analysis.quantities import MONEY
from threetears.evals.analysis.report.cost_unit import priced_in_part
from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.charts import (
    BarsChart,
    ChartReference,
    DotIntervalChart,
    DotIntervalPanel,
    DotIntervalRow,
    LegendItem,
    RangeBarsChart,
    RangeGroup,
    RangeRow,
    ScatterChart,
    ScatterPanel,
    ScatterPoint,
)
from threetears.evals.analysis.report.plan_page import resolve_changes
from threetears.evals.analysis.report.plan_page.charting import (
    STORED_SPAN_REFUSAL,
    ChartContext,
    CriterionWhere,
    Drawn,
    KindResolver,
    chart_block,
    computed_notes,
    criterion_line,
    drawn_keys,
    factor_legend,
    factor_token,
    figure_interval,
    figure_unit,
    interval_words,
    money_words,
    misses_line,
    missing_note,
    point_labels,
    reference_line,
    slice_keys,
    slice_name,
    stat_interval,
    stat_value,
    unit_quantities,
    unit_refusal,
)
from threetears.evals.analysis.report.plan_page.look import (
    Quantity,
    band_domain,
    centred_domain,
    figure_text,
    log_domain,
    log_spread,
    magnitude_domain,
    quantity,
    quantity_of,
)
from threetears.evals.analysis.report.plan_page.words import (
    BAR_STRIP_NOTE,
    BAR_WORD,
    FLOOR_STRIP_NOTE,
    LOG_X_NOTE,
    MISSES_TEMPLATE,
    NO_MEDIAN_TEMPLATE,
    P95_TEMPLATE,
    RANGE_TEMPLATE,
    REFERENCE_TEMPLATE,
)
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report_input import BarRead, FloorRead, ScoreboardRow, Stat, Stratum, Task
from threetears.evals.analysis.style import humanize

__all__ = ["RESOLVERS"]

#: The sources whose rows an analysis computes, unrounded: a values table spells them at their quantity's precision.
_COMPUTED = frozenset({"compare", "levers"})


#: The scoreboard's figures that count by cause.
_BY_CAUSE = frozenset({"lost", "failed"})
#: The scoreboard's figures where less is better.
_SCOREBOARD_LOWER = frozenset({"reply_cost_usd", "test_cost_usd", "lost", "failed"})


def _arm_label(context: ChartContext, arm: str) -> str:
    return context.look.arm(arm).label if arm in context.look.arms else arm


def _arm_name(context: ChartContext, arm: str) -> str:
    return context.look.arm(arm).name if arm in context.look.arms else arm


def _with_line(tip: list[str], value: float, line: ChartReference | None) -> list[str]:
    return [*tip, MISSES_TEMPLATE.format(line=line.label or line.kind)] if line and misses_line(value, line) else tip


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
    """One strip (a panel, or the whole chart): its title, the arm it stands for (for a chip), its dots and line, and
    the quantity its dots are drawn as."""

    title: str
    arm: str
    dots: list[_Dot]
    line: ChartReference | None
    drawn: Quantity
    key: str = ""


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


def _priced_part(context: ChartContext, arm: str, drawn: Quantity) -> str:
    """What a money figure of an arm whose cost counts only part of its calls says beside it: `` (850 of 897 calls
    priced)``; empty for any other figure."""
    share = priced_in_part(context.report_input, arm) if drawn.unit == MONEY and arm else None
    return f" ({share})" if share is not None else ""


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
    if interval is not None and drawn.rate:
        # A rate's interval stated as plus or minus a half-width can run past 0 or 100; a rate never does.
        interval = (max(interval[0], 0.0), min(interval[1], 100.0))
    tip = [label, figure_text(value, drawn)]
    if interval is not None:
        tip[1] += f", {interval_words(*interval, drawn)}"
    if drawn.unit == MONEY and arm and (share := priced_in_part(context.report_input, arm)) is not None:
        # A cost over part of the arm's calls is a smaller total, never stated bare.
        tip[1] += f" ({share})"
    strong = True if context.summary and arm and arm == context.look.recommended else None
    return _Dot(id=row_id, label=label, arm=arm, value=value, interval=interval, tip=tip, strong=strong)


def _change_strips(context: ChartContext) -> tuple[list[_Strip], str, tuple[str, ...], bool]:
    """readings or contrasts: the diverging chart's rows as dots, a strip per group (a measure, where several, and the
    aspect too where a measure's contrasts sit on several). The strips are the small multiples, so the bars are read
    unpanelled: a strip already names the aspect wherever two aspects' contrasts share it."""
    query = context.spec.query.model_copy(update={"small_multiples_by": "none"})
    as_bars = dataclasses.replace(
        context, spec=context.spec.model_copy(update={"kind": "diverging_bars", "query": query})
    )
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
    strips = [_Strip(title=title, arm="", dots=dots, line=line, drawn=fmt_unit) for title, dots in grouped.items()]
    per = drawn.block.drawn_per if isinstance(drawn.block, ChartBlock) else []
    return strips, context.level, (*drawn.notes, *per), True


def _figure_strips(context: ChartContext) -> tuple[list[_Strip], str, tuple[str, ...], bool]:
    """measures or judged: an arm per row, a strip per measure, each in its measure's unit."""
    report_input = context.report_input
    rows = context.rows()
    keys = drawn_keys(context.spec.kind, context.query, report_input)
    level = context.level or _declared_level(context, keys)
    figures = {(row.key, row.arm): row.item for row in rows if isinstance(row.item, Stat)}
    reads = {
        place: read for place, stat in figures.items() if (read := stat_value(report_input, place[0], stat)) is not None
    }
    quantities = unit_quantities(
        report_input, {key: [r[0] for (on, _), r in reads.items() if on == key] for key in keys}
    )
    strips = []
    for key in keys:
        drawn = quantities[key]
        statistics = {read[1] for (on, _), read in reads.items() if on == key}
        line = reference_line(context, drawn) or (
            criterion_line(context, key, drawn, where="overall", statistic=statistics.pop())
            if len(statistics) == 1
            else None
        )
        dots = []
        for arm in context.arm_order(arm for (on, arm) in reads if on == key):
            value, statistic = reads[key, arm]
            interval = figure_interval(context, key, arm, figures[key, arm]) if level and statistic == "mean" else None
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
        title = money_words(report_input, key, drawn)
        strips.append(_Strip(title=title, arm="", dots=dots, line=line, drawn=drawn, key=key))
    return strips, level, computed_notes(context), False


def _slice_strips(context: ChartContext) -> tuple[list[_Strip], str, tuple[str, ...], bool]:
    """tasks or strata: an arm per row and a strip per slice and measure, each in its measure's unit, or a strip per
    arm (a row per slice), in the one unit its measures share."""
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
    quantities = unit_quantities(report_input, {key: [entry[3] for entry in found if entry[1] == key] for key in keys})
    tasks = context.query.source == "tasks"

    def line_of(key: str, item: Stratum | Task | None, statistics: set[str]) -> ChartReference | None:
        drawn = quantities[key]
        stored = reference_line(context, drawn)
        if stored is not None or len(statistics) != 1:
            return stored
        if not tasks:
            return None
        where: CriterionWhere = ("task", item.id) if item is not None else "tasks"
        return criterion_line(context, key, drawn, where=where, statistic=next(iter(statistics)))

    def place(item: Stratum | Task, key: str) -> str:
        name = slice_name(context, item)
        return name if len(keys) == 1 else f"{name} · {money_words(report_input, key, quantities[key])}"

    strips: list[_Strip] = []
    if context.query.small_multiples_by == "arm":
        # A strip per arm holds every measure on one axis: the one-unit rule has refused measures in more than one,
        # and those drawn alike (a share beside a percent) share one quantity, each dot at its own measure's scale.
        drawn = quantities[keys[0]]
        line = line_of(keys[0], None, {statistic for *_, statistic, _ in found}) if len(keys) == 1 else None
        for arm in arms:
            dots = []
            for item, key, on, value, _, interval in found:
                if on != arm:
                    continue
                own = quantities[key]
                shown = (interval[0] * own.factor, interval[1] * own.factor) if interval is not None else None
                dot = _dot(
                    context,
                    row_id=f"{item.id}/{key}",
                    label=place(item, key),
                    arm=arm,
                    value=value * own.factor,
                    interval=shown,
                    drawn=own,
                )
                dots.append(dataclasses.replace(dot, tip=_with_line(dot.tip, dot.value, line)))
            strips.append(
                _Strip(title=_arm_name(context, arm), arm=arm, dots=dots, line=line, drawn=drawn, key=keys[0])
            )
        return strips, level, (), False
    for item in slices:
        for key in keys:
            here = [entry for entry in found if entry[0] is item and entry[1] == key]
            if not here:
                continue
            drawn = quantities[key]
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
            strips.append(_Strip(title=place(item, key), arm="", dots=dots, line=line, drawn=drawn, key=key))
    return strips, level, (), False


def _floor_strips(context: ChartContext) -> tuple[list[_Strip], str, tuple[str, ...], bool]:
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
        strips.append(_Strip(title=humanize("dimension", floor), arm="", dots=dots, line=line, drawn=drawn))
    return strips, level, (FLOOR_STRIP_NOTE, *missing_note(missing)), True


def _bar_read_strips(context: ChartContext) -> tuple[list[_Strip], str, tuple[str, ...], bool]:
    """bar_reads: an arm per row at its value with the interval the bar was read on, a strip per aspect, measure and
    bar, against the bar's line; a missed bar is hollow."""
    report_input = context.report_input
    reads = [row.item for row in context.rows() if isinstance(row.item, BarRead)]
    groups: dict[tuple[str, str, float, str], list[BarRead]] = {}
    for read in reads:
        groups.setdefault((read.aspect, read.measure, float(read.bar), read.direction), []).append(read)
    several_aspects = len({aspect for aspect, *_ in groups}) > 1
    strips = []
    for (aspect, measure, bar, direction), held in groups.items():
        # One form for the strip: its line and every read's value and interval ends (#993).
        drawn = quantity_of(
            figure_unit(report_input, measure),
            [
                bar,
                *(
                    float(number)
                    for read in held
                    for number in (read.value, *(read.interval or ()))
                    if number is not None
                ),
            ],
        )
        kind: Literal["floor", "budget"] = "floor" if direction == "higher_is_better" else "budget"
        line = ChartReference(
            value=bar * drawn.factor,
            label=REFERENCE_TEMPLATE.format(kind=BAR_WORD, value=figure_text(bar * drawn.factor, drawn)),
            kind=kind,
        )
        dots = []
        by_arm = {read.arm: read for read in held}
        for arm in context.arm_order(by_arm):
            read = by_arm[arm]
            if read.value is None:
                continue
            interval = (
                (float(read.interval[0]) * drawn.factor, float(read.interval[1]) * drawn.factor)
                if context.level and read.interval is not None
                else None
            )
            dot = _dot(
                context,
                row_id=arm,
                label=_arm_label(context, arm),
                arm=arm,
                value=float(read.value) * drawn.factor,
                interval=interval,
                drawn=drawn,
            )
            words = [read.status, *([read.why] if read.why else [])]
            dots.append(dataclasses.replace(dot, tip=[*dot.tip, *words], hollow=read.status == "missed"))
        name = money_words(report_input, measure, drawn)
        title = f"{context.look.reader('aspect', aspect)} · {name}" if several_aspects else name
        strips.append(_Strip(title=title, arm="", dots=dots, line=line, drawn=drawn, key=measure))
    return strips, context.level, (BAR_STRIP_NOTE,), False


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


def _strip_spans(context: ChartContext, strips: Sequence[_Strip], centred: bool) -> dict[Quantity, tuple[float, float]]:
    """Each quantity's span: the stored one, else one fitted to every dot, interval and line its strips draw."""
    spans: dict[Quantity, tuple[float, float]] = {}
    for drawn in dict.fromkeys(strip.drawn for strip in strips):
        mine = [strip for strip in strips if strip.drawn == drawn]
        ends = [end for strip in mine for dot in strip.dots for end in (dot.value, *(dot.interval or ()))]
        ends += [strip.line.value for strip in mine if strip.line is not None]
        domain = context.spec.domain or (centred_domain(ends, drawn) if centred else band_domain(ends, drawn))
        spans[drawn] = (domain[0], domain[1])
    return spans


def _dot_interval(context: ChartContext) -> Drawn:
    source = context.query.source
    refusal = unit_refusal(context.spec.kind, context.query, context.report_input)
    if refusal:
        raise context.refuse(refusal)
    if source in ("readings", "contrasts", "compare", "levers"):
        strips, level, notes, centred = _change_strips(context)
    elif source == "bar_reads":
        strips, level, notes, centred = _bar_read_strips(context)
    elif source in ("measures", "judged"):
        strips, level, notes, centred = _figure_strips(context)
    elif source in ("tasks", "strata"):
        strips, level, notes, centred = _slice_strips(context)
    elif source == "floors":
        strips, level, notes, centred = _floor_strips(context)
    else:
        raise context.refuse(
            "a dot strip draws readings, contrasts, measures, judged, floors, tasks, strata, bar reads, two arms "
            "compared or lever effects"
        )
    strips = [strip for strip in strips if strip.dots]
    if not strips:
        raise context.refuse("it selects no figure to draw")
    spans = _strip_spans(context, strips, centred)
    several = len(spans) > 1
    if several and context.spec.domain:
        raise context.refuse(STORED_SPAN_REFUSAL)
    first = strips[0].drawn
    # What an analysis computes is spelled at its quantity's precision in the values table (chart_block's read_at).
    read_at = first.unit if source in _COMPUTED and not several else ""
    if len(strips) == 1:
        (strip,) = strips
        data = DotIntervalChart(
            rows=[_dot_row(context, dot, strip.line) for dot in strip.dots],
            domain=spans[first],
            fmt=first.fmt,
            level=level,
            reference=strip.line,
            emphasis=context.emphasis,
        )
        return Drawn(chart_block(context, data, read_at=read_at, money=_strip_money(strips)), notes)
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
                domain=spans[strip.drawn] if several else None,
                fmt=strip.drawn.fmt if several else None,
            )
        )
    data = DotIntervalChart(domain=spans[first], fmt=first.fmt, level=level, emphasis=context.emphasis, panels=panels)
    return Drawn(chart_block(context, data, read_at=read_at, money=_strip_money(strips)), notes)


def _strip_money(strips: Sequence[_Strip]) -> list[tuple[str, Quantity]]:
    """Each money figure a strip chart draws, by key, with the quantity its strip draws it as."""
    return list(dict.fromkeys((strip.key, strip.drawn) for strip in strips if strip.key))


# --- range_bars ---------------------------------------------------------------------------------------------------


def _spread_stat(stat: Stat | None) -> tuple[float | None, float] | None:
    """A latency's p50 (None where the input holds only its p95) and p95; None without a p95."""
    if stat is None or stat.p95 is None:
        return None
    return (float(stat.p50) if stat.p50 is not None else None), float(stat.p95)


def _range_bars(context: ChartContext) -> Drawn:
    report_input = context.report_input
    source = context.query.source
    refusal = unit_refusal(context.spec.kind, context.query, report_input)
    if refusal:
        raise context.refuse(refusal)
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
            groups.append(("", "overall", key, [(arm, spreads[arm]) for arm in context.arm_order(spreads)]))
    elif source == "tasks":
        slices = [row.item for row in context.rows() if isinstance(row.item, Task)]
        keys = drawn_keys(context.spec.kind, context.query, report_input)
        allowed = list(context.query.arms) or list(context.look.arms)
        for item in slices:
            for key in keys:
                spreads = {
                    arm: spread
                    for arm in allowed
                    if (spread := _spread_stat(item.per_arm.get(arm, {}).get(key))) is not None
                }
                if spreads:
                    groups.append(
                        (
                            slice_name(context, item),
                            ("task", item.id),
                            key,
                            [(arm, spreads[arm]) for arm in context.arm_order(spreads)],
                        )
                    )
    else:
        raise context.refuse("range bars draw measures or tasks")
    if not groups:
        raise context.refuse("it selects no figure with a p95")
    ends_of: dict[str, list[float]] = {}
    for _, _, key, bars in groups:
        ends_of.setdefault(key, []).extend(end for _, spread in bars for end in spread if end is not None)
    # One axis: the one-unit rule has refused range bars whose measures are in more than one unit, and those drawn
    # alike (a share beside a percent) share one quantity, each group at its own measure's scale.
    quantities = unit_quantities(report_input, ends_of)
    drawn = quantities[groups[0][2]]
    keys_drawn = {key for _, _, key, _ in groups}
    line = reference_line(context, drawn)
    if line is None and len(keys_drawn) == 1:
        (key,) = keys_drawn
        where: CriterionWhere = "overall" if source == "measures" else "tasks"
        line = criterion_line(context, key, drawn, where=where, statistic="p95")
    drawn_groups = []
    no_median: list[str] = []
    several = len({key for _, _, key, _ in groups}) > 1
    for name, _, key, bars in groups:
        factor = quantities[key].factor
        # A group's words name what its money is drawn per (the one owner of money's words), its measure's when
        # it names one alone and after its slice where several share the chart.
        measure = money_words(report_input, key, quantities[key])
        label = measure if not name else f"{name} · {measure}" if several else name
        drawn_rows = []
        for arm, (p50, p95) in bars:
            high = p95 * factor
            low = p50 * factor if p50 is not None else high
            look = context.look.arm(arm) if arm in context.look.arms else None
            said = (
                RANGE_TEMPLATE.format(p50=figure_text(low, drawn), p95=figure_text(high, drawn))
                if p50 is not None
                else P95_TEMPLATE.format(p95=figure_text(high, drawn))
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
    ends = [
        *(end * quantities[key].factor for key, figures in ends_of.items() for end in figures),
        *([line.value] if line is not None else []),
    ]
    domain = context.spec.domain or magnitude_domain(ends)
    drawn_arms = dict.fromkeys(arm for *_, bars in groups for arm, _ in bars if arm in context.look.arms)
    legend = [
        LegendItem(label=look.name, color=look.token, mark=look.shape)
        for look in (context.look.arm(arm) for arm in context.look.ordered(drawn_arms))
    ]
    data = RangeBarsChart(
        groups=drawn_groups, domain=(domain[0], domain[1]), fmt=drawn.fmt, reference=line, legend=legend or None
    )
    notes = (NO_MEDIAN_TEMPLATE.format(missing="; ".join(no_median)),) if no_median else ()
    money = [(key, quantities[key]) for key in dict.fromkeys(key for _, _, key, _ in groups)]
    return Drawn(chart_block(context, data, money=money), notes)


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
        x_unit = figure_unit(report_input, x, source="scoreboard")
        return points, x, y, x_unit, figure_unit(report_input, y, source="scoreboard")
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


def _scatter(context: ChartContext) -> Drawn:
    if context.query.source not in ("measures", "judged", "scoreboard", "configs", "tasks"):
        raise context.refuse("a scatter draws measures, judged, scoreboard, configs or tasks")
    points, x_key, y_key, x_unit, y_unit = _scatter_points(context)
    if not points:
        raise context.refuse("no arm holds both of its figures")
    x_drawn = quantity_of(x_unit, [point.x for point in points])
    y_drawn = quantity_of(y_unit, [point.y for point in points])
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

    configs = context.query.source == "configs"

    def drawn_points(task: str | None) -> list[ScatterPoint]:
        drawn = []
        here = [point for point in points if point.task == task]
        for point in here:
            look = context.look.arm(point.arm) if point.arm in context.look.arms else None
            label, narrow = point_labels(context, point.arm, _arm_label(context, point.arm), len(here))
            colour = factor_token(context.report_input, point.arm) if configs else None
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
                    label=label,
                    narrow_label=narrow,
                    color=colour or (look.token if look is not None else None),
                    shape=look.shape if look is not None and colour is None else None,
                    hollow=True if fails else None,
                    muted=True if context.summary and point.arm not in context.look.survivors else None,
                    tip=[
                        _arm_name(context, point.arm),
                        f"{money_words(context.report_input, x_key, x_drawn)} {figure_text(x_value, x_drawn)}"
                        + _priced_part(context, point.arm, x_drawn),
                        f"{money_words(context.report_input, y_key, y_drawn)} {figure_text(y_value, y_drawn)}"
                        + _priced_part(context, point.arm, y_drawn),
                    ],
                )
            )
        return drawn

    x_ends = xs + [line.value for task in tasks if (line := lines(task)[1]) is not None]
    y_ends = ys + [line.value for task in tasks if (line := lines(task)[0]) is not None]
    x_log = log_spread(x_ends)
    stored = context.spec.domain
    x_domain = log_domain(x_ends) if x_log else band_domain(x_ends, x_drawn)
    y_domain = (stored[0], stored[1]) if stored else band_domain(y_ends, y_drawn)

    def chart(**part: object) -> ScatterChart:
        return ScatterChart.model_validate(
            {
                "x_domain": x_domain,
                "y_domain": y_domain,
                "x_fmt": x_drawn.fmt,
                "y_fmt": y_drawn.fmt,
                "x_log": True if x_log else None,
                "x_title": money_words(
                    context.report_input, x_key, x_drawn, name=humanize("dimension", x_key) if scoreboard else ""
                ),
                "y_title": money_words(
                    context.report_input, y_key, y_drawn, name=humanize("dimension", y_key) if scoreboard else ""
                ),
                "frontier": True,
                "emphasis": context.emphasis,
                "legend": factor_legend(context.report_input, (point.arm for point in points)) if configs else None,
                **part,
            }
        )

    notes = (LOG_X_NOTE,) if x_log else ()
    if len(tasks) == 1:
        y_line, x_line = lines(tasks[0])
        data = chart(points=drawn_points(tasks[0]), reference=y_line, x_reference=x_line)
        return Drawn(chart_block(context, data, money=[(x_key, x_drawn), (y_key, y_drawn)]), notes)
    panels = []
    for task in tasks:
        y_line, x_line = lines(task)
        title = next((slice_name(context, item) for item in context.report_input.tasks if item.id == task), task or "")
        panels.append(ScatterPanel(title=title, points=drawn_points(task), reference=y_line, x_reference=x_line))
    return Drawn(chart_block(context, chart(panels=panels), money=[(x_key, x_drawn), (y_key, y_drawn)]), notes)


#: The kinds this module resolves.
RESOLVERS: dict[str, KindResolver] = {
    "dot_interval": _dot_interval,
    "range_bars": _range_bars,
    "scatter_frontier": _scatter,
}
