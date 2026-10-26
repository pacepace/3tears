"""The charts of magnitude: bars from zero, and lines over an ordered x.

**bars** draw each figure from zero:

- *measures*, *judged*: an arm per row (on the query's aspect where it names one), grouped under the measure's reader
  name where the chart reads several; money per reply drawn per 1,000 replies;
- *configs*: a config per row, named by its other factors' levels and grouped under its first factor's level;
- *scoreboard*: a row per arm on each scoreboard figure the query names (a count, or a cost);
- *strata*, *tasks*: each slice's figures, in panels by task (or by arm) where the plan asks for them, else an arm
  per row grouped under each slice (a single slice needs no group);
- *human_pref*: a row per slice, the share of ratings preferring the control.

A summary chart's rows follow the options table, fade an arm that does not survive the gates and bold the
recommended one (GENERATOR §5.3). A figure without a mean is drawn at the statistic a criterion on it reads, else its
p95, which a caption says. A floor or budget line sits at its criterion's threshold.

**lines** draw a value over an ordered x, a line per arm:

- *strata*, *tasks*: the slices are the x, in the input's order; an arm missing a slice's figure is left out and
  named in a caption;
- *measures*, *judged*: a measure's bins (each arm's share in each), else the aspects it is read on.

Bands sit on at most two lines, and only where the chart states a level.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from threetears.evals.analysis.report.charts import BarRow, BarsChart, BarsPanel, LineSeries, LinesChart
from threetears.evals.analysis.report.plan_page.charting import (
    ChartContext,
    Drawn,
    KindResolver,
    chart_block,
    figure_stat,
    figure_unit,
    interval_words,
    measure_words,
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
    figure_text,
    magnitude_domain,
    quantity,
)
from threetears.evals.analysis.report.plan_page.words import STATISTIC_NOTE
from threetears.evals.analysis.report_input import (
    Measure,
    PreferenceRow,
    ScoreboardRow,
    Stat,
    Stratum,
    Task,
)

__all__ = ["RESOLVERS"]

#: The most lines that carry a band (GENERATOR §5.1: bands on at most two series).
_MOST_BANDS = 2

#: The scoreboard figures that are money, and how each is stated.
_SCOREBOARD_UNITS = {"reply_cost_usd": "usd", "test_cost_usd": "dollars"}

#: A scoreboard's figures that count by cause, drawn as their total.
_BY_CAUSE = frozenset({"lost", "failed"})


@dataclass(frozen=True)
class _Bar:
    """One drawn bar: its row's id, label, colour, tag and group, its value and interval, and how it reads."""

    id: str
    label: str
    value: float
    interval: tuple[float, float] | None = None
    color: str | None = None
    tag: str | None = None
    group: str | None = None
    arm: str = ""
    tip: tuple[str, ...] = ()


def _statistic_note(statistics: set[str]) -> tuple[str, ...]:
    drawn_at = sorted(statistics - {"mean"})
    return tuple(STATISTIC_NOTE.format(statistic=statistic) for statistic in drawn_at)


def _arm_bar(
    context: ChartContext, arm: str, value: float, interval: tuple[float, float] | None, *, group: str = ""
) -> _Bar:
    """One arm's bar, in its colour and with its tag; a row naming no arm of the input by its id alone."""
    if arm in context.look.arms:
        look = context.look.arm(arm)
        return _Bar(
            id=arm,
            label=look.name,
            value=value,
            interval=interval,
            color=look.token,
            tag=look.tag,
            arm=arm,
            group=group,
        )
    return _Bar(id=arm, label=arm, value=value, interval=interval, arm=arm, group=group)


def _figure_bars(context: ChartContext) -> tuple[list[_Bar], Quantity, set[str]]:
    """measures, judged or configs: an arm's (or a config's) figure per row, grouped by measure where there are
    several, or, for configs, by the first factor's level."""
    report_input = context.report_input
    rows = context.rows()
    keys = list(dict.fromkeys(row.key for row in rows))
    figures = {(row.key, row.arm): row.item for row in rows if isinstance(row.item, Stat)}
    unit = figure_unit(report_input, keys[0]) if keys else ""
    values: list[float] = []
    statistics: set[str] = set()
    picked: list[tuple[str, str, float, tuple[float, float] | None]] = []
    for key in keys:
        arms = [arm for (on, arm) in figures if on == key]
        for arm in context.arm_order(arms):
            stat = figures[key, arm]
            read = stat_value(report_input, key, stat)
            if read is None:
                continue
            statistics.add(read[1])
            interval = stat_interval(stat) if read[1] == "mean" else None
            picked.append((key, arm, read[0], interval))
            values.append(read[0])
    drawn = quantity(unit, values)
    bars: list[_Bar] = []
    configs = {config.arm: config for config in report_input.configs} if context.query.source == "configs" else {}
    for key, arm, value, interval in picked:
        shown = (interval[0] * drawn.factor, interval[1] * drawn.factor) if interval is not None else None
        group = measure_words(report_input, key) if len(keys) > 1 else None
        if arm in configs:
            first, *rest = _config_levels(context, arm)
            label = " · ".join(rest) or first
            bar = _arm_bar(context, arm, value * drawn.factor, shown, group=first if rest else (group or ""))
            bars.append(_relabel(bar, label))
        else:
            bars.append(_arm_bar(context, arm, value * drawn.factor, shown, group=group or ""))
    if configs:
        order = {arm: index for index, arm in enumerate(config.arm for config in report_input.configs)}
        bars.sort(key=lambda bar: (_first_level_place(context, bar.arm), order.get(bar.arm, len(order))))
    return bars, drawn, statistics


def _config_levels(context: ChartContext, arm: str) -> list[str]:
    """A config's factor levels as a reader names them, in the factors' order."""
    config = next(config for config in context.report_input.configs if config.arm == arm)
    names = []
    for factor in context.report_input.factors:
        level = config.levels.get(factor.id)
        if level is None:
            continue
        held = next((one for one in factor.levels if one.id == level), None)
        names.append((held.name if held is not None else None) or level)
    return names


def _first_level_place(context: ChartContext, arm: str) -> int:
    factors = context.report_input.factors
    if not factors:
        return 0
    config = next(config for config in context.report_input.configs if config.arm == arm)
    ids = [level.id for level in factors[0].levels]
    level = config.levels.get(factors[0].id)
    return ids.index(level) if level in ids else len(ids)


def _relabel(bar: _Bar, label: str) -> _Bar:
    return _Bar(
        id=bar.id,
        label=label,
        value=bar.value,
        interval=bar.interval,
        color=bar.color,
        tag=bar.tag,
        group=bar.group,
        arm=bar.arm,
        tip=bar.tip,
    )


def _scoreboard_bars(context: ChartContext) -> tuple[list[_Bar], Quantity, set[str]]:
    keys = list(context.query.measures)
    if not keys:
        raise context.refuse("a scoreboard chart names the scoreboard figures it draws, and this one names none")
    units = {_SCOREBOARD_UNITS.get(key, "") for key in keys}
    if len(units) > 1:
        raise context.refuse(f"its figures {keys} are in more than one unit")
    rows = {row.arm: row.item for row in context.rows() if isinstance(row.item, ScoreboardRow)}
    picked: list[tuple[str, str, float]] = []
    for key in keys:
        for arm in context.arm_order(rows):
            held = getattr(rows[arm], key)
            value = sum(held.values()) if key in _BY_CAUSE else held
            if value is not None:
                picked.append((key, arm, float(value)))
    drawn = quantity(units.pop(), [value for _, _, value in picked])
    bars = [
        _arm_bar(
            context,
            arm,
            value * drawn.factor,
            None,
            group=measure_words(context.report_input, key) if len(keys) > 1 else "",
        )
        for key, arm, value in picked
    ]
    return bars, drawn, {"mean"}


def _slice_bars(context: ChartContext) -> tuple[list[_Bar], Quantity, set[str]]:
    """strata or tasks: each arm's figure in each slice, its slice as the bar's group (by slice, then arm)."""
    report_input = context.report_input
    slices = [row.item for row in context.rows() if isinstance(row.item, Stratum | Task)]
    keys = slice_keys(context, slices)
    allowed = set(context.query.arms) or set(context.look.arms)
    statistics: set[str] = set()
    picked: list[tuple[str, str, str, float, tuple[float, float] | None]] = []
    for key in keys:
        for item in slices:
            name = slice_name(context, item)
            where = name if len(keys) == 1 else f"{name} · {measure_words(report_input, key)}"
            for arm in context.arm_order(arm for arm in item.per_arm if arm in allowed):
                stat = item.per_arm[arm].get(key)
                read = stat_value(report_input, key, stat) if stat is not None else None
                if stat is None or read is None:
                    continue
                statistics.add(read[1])
                interval = stat_interval(stat) if read[1] == "mean" else None
                picked.append((where, key, arm, read[0], interval))
    drawn = quantity(figure_unit(report_input, keys[0]), [value for *_, value, _ in picked])
    bars = []
    for where, _key, arm, value, interval in picked:
        shown = (interval[0] * drawn.factor, interval[1] * drawn.factor) if interval is not None else None
        bars.append(_arm_bar(context, arm, value * drawn.factor, shown, group=where))
    return bars, drawn, statistics


def _preference_bars(context: ChartContext) -> tuple[list[_Bar], Quantity, set[str]]:
    drawn = quantity("percent")
    bars = []
    for row in context.rows():
        assert isinstance(row.item, PreferenceRow)
        item = row.item
        width = item.half_width
        value = float(item.prefers_control)
        interval = (value - float(width), value + float(width)) if width is not None else None
        bars.append(_Bar(id=item.id, label=item.name or item.id, value=value, interval=interval))
    return bars, drawn, {"mean"}


def _bar_row(context: ChartContext, bar: _Bar, drawn: Quantity) -> BarRow:
    interval = bar.interval if context.level else None
    tip = [bar.label, figure_text(bar.value, drawn)]
    if interval is not None:
        tip[1] += f", {interval_words(*interval, drawn)}"
    strong = muted = None
    if context.summary and bar.arm:
        strong = True if bar.arm == context.look.recommended else None
        muted = True if bar.arm not in context.look.survivors else None
    return BarRow(
        id=bar.id,
        label=bar.label,
        tag=bar.tag,
        value=bar.value,
        lo=interval[0] if interval is not None else None,
        hi=interval[1] if interval is not None else None,
        color=bar.color,
        strong=strong,
        muted=muted,
        note=figure_text(bar.value, drawn),
        tip=tip,
        group=bar.group or None,
    )


def _ends(bars: Sequence[_Bar], level: str) -> Iterator[float]:
    for bar in bars:
        yield bar.value
        if level and bar.interval is not None:
            yield from bar.interval


def _panels(context: ChartContext, bars: Sequence[_Bar], drawn: Quantity) -> list[BarsPanel] | None:
    """Small multiples where the plan asks for them and the chart has more than one: by arm (a panel per arm, a row
    per group), else by the bars' group (a panel per slice or task, a row per arm)."""
    by = context.query.small_multiples_by
    if by == "none":
        return None
    if by == "arm":
        keyed: dict[str, list[_Bar]] = {bar.arm: [] for bar in bars if bar.arm}
        for bar in bars:
            if bar.arm:
                keyed[bar.arm].append(_relabel(_ungrouped(bar), bar.group or bar.label))
        if len(keyed) < 2:
            return None
        return [
            BarsPanel(
                title=context.look.arm(arm).name if arm in context.look.arms else arm,
                tag=context.look.arm(arm).tag if arm in context.look.arms else None,
                color=context.look.arm(arm).token if arm in context.look.arms else None,
                rows=[_bar_row(context, bar, drawn) for bar in rows],
            )
            for arm, rows in keyed.items()
        ]
    grouped: dict[str, list[_Bar]] = {}
    for bar in bars:
        grouped.setdefault(bar.group or "", []).append(_ungrouped(bar))
    if len(grouped) < 2:
        return None
    return [
        BarsPanel(title=title, rows=[_bar_row(context, bar, drawn) for bar in rows]) for title, rows in grouped.items()
    ]


def _ungrouped(bar: _Bar) -> _Bar:
    return _Bar(
        id=bar.id,
        label=bar.label,
        value=bar.value,
        interval=bar.interval,
        color=bar.color,
        tag=bar.tag,
        group=None,
        arm=bar.arm,
        tip=bar.tip,
    )


def _bars(context: ChartContext) -> Drawn:
    source = context.query.source
    if source in ("measures", "judged", "configs"):
        bars, drawn, statistics = _figure_bars(context)
    elif source == "scoreboard":
        bars, drawn, statistics = _scoreboard_bars(context)
    elif source in ("strata", "tasks"):
        bars, drawn, statistics = _slice_bars(context)
    elif source == "human_pref":
        bars, drawn, statistics = _preference_bars(context)
    else:
        raise context.refuse("bars draw measures, judged, scoreboard, strata, tasks, configs or human preference")
    if not bars:
        raise context.refuse("it selects no figure to draw")
    if len({bar.group for bar in bars}) == 1:
        bars = [_ungrouped(bar) for bar in bars]
    reference = reference_line(context, drawn)
    ends = [*_ends(bars, context.level), *([reference.value] if reference is not None else [])]
    domain = context.spec.domain or magnitude_domain(ends, rate=drawn.rate)
    panels = _panels(context, bars, drawn)
    data = BarsChart(
        rows=None if panels is not None else [_bar_row(context, bar, drawn) for bar in bars],
        panels=panels,
        domain=(domain[0], domain[1]),
        fmt=drawn.fmt,
        tick_fmt=drawn.tick_fmt,
        level=context.level or None,
        reference=reference,
        emphasis=context.emphasis,
    )
    return Drawn(chart_block(context, data), _statistic_note(statistics))


# --- lines ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Line:
    arm: str
    values: list[float]
    bands: list[tuple[float, float]] | None


def _slice_lines(context: ChartContext) -> tuple[list[str], list[_Line], Quantity, list[str]]:
    report_input = context.report_input
    slices = [row.item for row in context.rows() if isinstance(row.item, Stratum | Task)]
    keys = slice_keys(context, slices)
    if len(keys) != 1:
        raise context.refuse(f"a line chart reads one measure, and the query names {keys}")
    (key,) = keys
    categories = [slice_name(context, item) for item in slices]
    allowed = list(context.query.arms) or list(context.look.arms)
    lines: list[_Line] = []
    missing: list[str] = []
    for arm in context.arm_order(allowed):
        stats = [item.per_arm.get(arm, {}).get(key) for item in slices]
        reads = [stat_value(report_input, key, stat) if stat is not None else None for stat in stats]
        if any(read is None for read in reads):
            gaps = [name for name, read in zip(categories, reads, strict=True) if read is None]
            if len(gaps) < len(categories):
                missing += [f"{_arm_name(context, arm)}, {gap}" for gap in gaps]
            continue
        intervals = [stat_interval(stat) if stat is not None else None for stat in stats]
        bands = [band for band in intervals if band is not None]
        lines.append(
            _Line(
                arm=arm,
                values=[read[0] for read in reads if read is not None],
                bands=bands if len(bands) == len(stats) else None,
            )
        )
    values = [value for line in lines for value in line.values]
    return categories, lines, quantity(figure_unit(report_input, key), values), missing


def _arm_name(context: ChartContext, arm: str) -> str:
    return context.look.arm(arm).name if arm in context.look.arms else arm


def _figure_lines(context: ChartContext) -> tuple[list[str], list[_Line], Quantity, list[str]]:
    """measures or judged: a measure's bins, each arm's share in each; else the aspects it is read on."""
    report_input = context.report_input
    keys = list(dict.fromkeys([*context.query.measures, *(row.key for row in context.rows())]))
    if len(keys) != 1:
        raise context.refuse(f"a line chart reads one measure, and the query selects {keys}")
    (key,) = keys
    arms = context.arm_order(dict.fromkeys(row.arm for row in context.rows()))
    measure = report_input.measures.get(key)
    if isinstance(measure, Measure) and measure.bins:
        categories = [level.name or level.id for level in measure.bins]
        lines = []
        for arm in arms:
            stat = figure_stat(report_input, key, arm, context.query.aspect)
            if stat is not None and stat.shares is not None and len(stat.shares) == len(categories):
                lines.append(_Line(arm=arm, values=[float(share) for share in stat.shares], bands=None))
        values = [value for line in lines for value in line.values]
        return categories, lines, quantity("percent", values), []
    aspects = [aspect.id for aspect in report_input.aspects]
    categories = [context.look.reader("aspect", aspect) for aspect in aspects]
    lines = []
    missing: list[str] = []
    for arm in arms:
        reads = []
        for aspect in aspects:
            stat = figure_stat(report_input, key, arm, aspect if aspect != aspects[0] else "")
            reads.append(stat_value(report_input, key, stat) if stat is not None else None)
        if any(read is None for read in reads):
            missing.append(_arm_name(context, arm))
            continue
        lines.append(_Line(arm=arm, values=[read[0] for read in reads if read is not None], bands=None))
    values = [value for line in lines for value in line.values]
    return categories, lines, quantity(figure_unit(report_input, key), values), missing


def _lines(context: ChartContext) -> Drawn:
    source = context.query.source
    if source in ("strata", "tasks"):
        categories, lines, drawn, missing = _slice_lines(context)
    elif source in ("measures", "judged"):
        categories, lines, drawn, missing = _figure_lines(context)
    else:
        raise context.refuse("lines draw strata, tasks, measures or judged")
    if not lines:
        raise context.refuse("no arm holds a figure at every point of its x")
    banded = bool(context.level) and len(lines) <= _MOST_BANDS and all(line.bands for line in lines)
    series = []
    ends: list[float] = []
    for line in lines:
        values = [value * drawn.factor for value in line.values]
        ends += values
        bands = [(low * drawn.factor, high * drawn.factor) for low, high in line.bands or []] if banded else None
        if bands:
            ends += [end for band in bands for end in band]
        arm = context.look.arm(line.arm) if line.arm in context.look.arms else None
        series.append(
            LineSeries(
                name=arm.label if arm is not None else line.arm,
                end_label=arm.name if arm is not None else line.arm,
                values=values,
                color=arm.token if arm is not None else None,
                lo=[low for low, _ in bands] if bands else None,
                hi=[high for _, high in bands] if bands else None,
                strong=True
                if line.arm == context.emphasis or (context.summary and line.arm == context.look.recommended)
                else None,
            )
        )
    domain = context.spec.domain or band_domain(ends, drawn)
    data = LinesChart(
        categories=categories,
        series=series,
        domain=(domain[0], domain[1]),
        fmt=drawn.fmt,
        level=context.level if banded else None,
    )
    return Drawn(chart_block(context, data), missing_note(missing))


#: The kinds this module resolves.
RESOLVERS: dict[str, KindResolver] = {"bars": _bars, "lines": _lines}
