"""The ``timeseries`` arm — one reading followed across the campaign's builds or days, a panel per series.

**One panel per series, one ink.** Time takes the horizontal axis and the value the vertical one, so a
series cannot sit on an axis the way every other chart's categories do — and identity never rides on hue
here, because the palette recycles and an axis does not. So each series gets its own row of a faceted
figure, named in the row header exactly as a distribution names its cohorts (the same gutter, the same
placement decision), and every panel shares one value axis so a level in one row reads against a level in
another. Overlaying the lines in hues would have been the shorter figure and the one this package refuses.

**Time is ordinal, and its order is stated.** The positions are builds or days, earliest first, as the
payload lists them; a categorical axis left to sort itself sorts alphabetically, and a line drawn through
``0.10`` before ``0.9`` is a trend the campaign never had. So the axis states its order as an explicit list,
and the gate refuses a line that does not (:func:`~threetears.evals.analysis.viz.policy._check_line_order`).

**A gap breaks the line.** A position a series has no point at is not interpolated across: each run of
consecutive points is its own segment, and the gap is disclosed under the figure with its reason.

**The value is written beside its point**, offset along the time axis rather than the value axis, so the
number sits at the height it names.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.viz.compiler import (
    DISPLAY_FIELD,
    KIND_FIELD,
    VALUE_LABEL_OFFSET,
    VALUE_TEXT_FIELD,
    VEGA_LITE_SCHEMA,
    ChartColumn,
    CompiledChart,
    ValueAxis,
    _axis_title,
    _Categories,
    _interval_caption,
    _interval_disclosures,
    _title_spec,
    display_scale,
    point_radius,
)
from threetears.evals.analysis.viz.palette import font_sizes, font_weights, geometry
from threetears.evals.analysis.viz.payloads import TimeseriesPayload

#: The row key holding a point's time position — what the time axis encodes.
_POSITION_FIELD = "position"

#: The row key naming the run of consecutive points a row belongs to. A line connects the rows sharing one,
#: so a gap — a position with no point — ends one segment and the next point starts another.
_SEGMENT_FIELD = "segment"

#: A point's mark area in px², ~9px across.
_POINT_SIZE = 72

#: How far a point extends past the value it is centred on, in px.
_POINT_RADIUS = point_radius(_POINT_SIZE)


def _segments(payload: TimeseriesPayload) -> dict[tuple[str, str], int]:
    """Number each series' runs of consecutive positions, so a gap ends a line rather than bridging it.

    Args:
        payload: The validated payload.

    Returns:
        ``(series label, position) -> segment number`` for every drawn point.
    """
    order = {position: index for index, position in enumerate(payload.positions)}
    numbered: dict[tuple[str, str], int] = {}
    for line in payload.series:
        segment, previous = 0, None
        for point in line.points:
            index = order[point.position]
            if previous is not None and index != previous + 1:
                segment += 1
            numbered[(line.label, point.position)] = segment
            previous = index
    return numbered


def _gap_lines(payload: TimeseriesPayload) -> list[str]:
    """One disclosure line per reason a point is missing, naming each series and the positions it lacks.

    Grouped by reason because the reason is what the sentence spends its words on, and one sentence per gap
    repeats it as many times as the axis is long.
    """
    by_reason: dict[str, dict[str, list[str]]] = {}
    for gap in payload.gaps:
        by_reason.setdefault(gap.reason, {}).setdefault(gap.series, []).append(gap.position)
    return [
        f"Not drawn ({reason}): "
        + "; ".join(f"{series} at {', '.join(where)}" for series, where in named.items())
        + "."
        for reason, named in by_reason.items()
    ]


def compile_timeseries(payload: TimeseriesPayload) -> CompiledChart:
    """Compile one reading across the time axis as a panel per series, each point with its interval.

    Args:
        payload: The validated timeseries payload.

    Returns:
        The compiled chart.
    """
    sizes = geometry()
    intervals = [point.ci for line in payload.series for point in line.points]
    scale, unit = display_scale([bound for ci in intervals for bound in (ci.low, ci.high)], payload.unit)
    time_title = f"Build ({payload.release_label})" if payload.basis == "release" else "Day (UTC)"
    categories = _Categories.of("series", [line.label for line in payload.series])
    width = int(sizes["plot_width"])
    # A panel is a plot of its own, so it takes the floor a plot is never drawn below rather than a row step.
    height = int(sizes["plot_min_height"])

    segments = _segments(payload)
    points: list[dict[str, Any]] = []
    for line in payload.series:
        for point in line.points:
            points.append(
                {
                    "series": line.label,
                    DISPLAY_FIELD: categories.display[line.label],
                    _POSITION_FIELD: point.position,
                    _SEGMENT_FIELD: f"{line.label}#{segments[(line.label, point.position)]}",
                    "mean": point.ci.mean * scale,
                    "low": point.ci.low * scale,
                    "high": point.ci.high * scale,
                    "n": point.n,
                }
            )
    marks = [row | {KIND_FIELD: "point"} for row in points] + [
        {
            DISPLAY_FIELD: row[DISPLAY_FIELD],
            _POSITION_FIELD: row[_POSITION_FIELD],
            "mean": row["mean"],
            VALUE_TEXT_FIELD: format_number(row["mean"]),
            KIND_FIELD: "value",
        }
        for row in points
    ]

    # The quantity is named once, in the heading, for the reason the distribution arm gives: an axis title in
    # every panel is the same words repeated, and the vertical one Vega would turn.
    value_axis = ValueAxis.position("", [row[key] for row in points for key in ("low", "high")], height)
    time: dict[str, Any] = {
        "field": _POSITION_FIELD,
        "type": "ordinal",
        # The order is the payload's — earliest first — and stated, never left to Vega, which would sort the
        # names alphabetically and draw the line through them in that order.
        "sort": list(payload.positions),
        "scale": {"domain": list(payload.positions)},
        "axis": {
            "title": time_title,
            "labelAngle": 0,
            "grid": False,
            "labelFontSize": font_sizes()["tick"],
            "labelFontWeight": font_weights()["tick"],
        },
    }
    value = value_axis.encoding("mean")

    def of_kind(kind: str) -> list[dict[str, Any]]:
        return [{"filter": {"field": KIND_FIELD, "equal": kind}}]

    layers: list[dict[str, Any]] = [
        {
            "transform": of_kind("point"),
            "mark": {"type": "line", "strokeWidth": 2},
            "encoding": {"x": time, "y": value, "detail": {"field": _SEGMENT_FIELD, "type": "nominal"}},
        },
        {
            "transform": of_kind("point"),
            "mark": {"type": "rule", "strokeWidth": 2, "tooltip": True},
            "encoding": {"x": time, "y": value_axis.encoding("low"), "y2": {"field": "high"}},
        },
        {
            "transform": of_kind("point"),
            "mark": {"type": "point", "filled": True, "size": _POINT_SIZE, "opacity": 1, "tooltip": True},
            "encoding": {"x": time, "y": value},
        },
        {
            # The value beside its point, at the height it names: offset along time, never along the value.
            "transform": of_kind("value"),
            "mark": {
                "type": "text",
                "align": "left",
                "baseline": "middle",
                "dx": _POINT_RADIUS + VALUE_LABEL_OFFSET,
                "fontSize": font_sizes()["value"],
                "fontWeight": font_weights()["value"],
            },
            "encoding": {"x": time, "y": value, "text": {"field": VALUE_TEXT_FIELD, "type": "nominal"}},
        },
    ]
    title = f"{_axis_title(payload.metric, unit)} over {'builds' if payload.basis == 'release' else 'days'}"
    spec: dict[str, Any] = {
        "$schema": VEGA_LITE_SCHEMA,
        "title": _title_spec(title, categories.figure_width()),
        "description": _interval_caption(intervals),
        "data": {"values": marks},
        "facet": {
            "row": {
                "field": DISPLAY_FIELD,
                "type": "nominal",
                "sort": categories.drawn(),
                "header": categories.facet_header(),
            }
        },
        "spacing": int(sizes["panel_gap"]),
        # One grid, so every row's header starts on one left edge — see the distribution arm.
        "align": "all",
        "spec": {"width": width, "height": height, "layer": layers},
    }

    order_line = (
        f"Builds of {payload.release_label} are in the order each first ran."
        if payload.basis == "release"
        else "Days are UTC, in calendar order."
    )
    disclosures = [order_line, *_interval_disclosures(intervals), *_gap_lines(payload)]
    columns: list[ChartColumn] = [
        {"key": "series", "header": "Series"},
        {"key": _POSITION_FIELD, "header": time_title},
        {"key": "mean", "header": _axis_title("Mean", unit)},
        {"key": "low", "header": _axis_title("Low", unit)},
        {"key": "high", "header": _axis_title("High", unit)},
        {"key": "n", "header": "n"},
    ]
    table = [{column["key"]: row[column["key"]] for column in columns} for row in points]
    return CompiledChart(spec=spec, columns=columns, rows=table, unit=unit, disclosures=disclosures, title=title)


__all__ = [
    "compile_timeseries",
]
