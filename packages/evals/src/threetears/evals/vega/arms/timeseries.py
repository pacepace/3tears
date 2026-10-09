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
and the gate refuses a line that does not (:func:`~threetears.evals.vega.spec_policy._check_line_order`).

**A gap breaks the line.** A position a series has no point at is not interpolated across: each run of
consecutive points is its own segment, and the gap is disclosed under the figure with its reason.

**The value is written beside its point**, offset along the time axis rather than the value axis, so the
number sits at the height it names.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.vega.compiler import (
    DISPLAY_FIELD,
    KIND_FIELD,
    VALUE_LABEL_OFFSET,
    VALUE_TEXT_FIELD,
    VEGA_LITE_SCHEMA,
    ValueAxis,
    _Categories,
    _identity,
    _number,
    _title_spec,
    _value_axis,
    point_radius,
)
from threetears.evals.analysis.viz.intent import ChartIntent
from threetears.evals.contracts.host import ChartFont
from threetears.evals.analysis.viz.intents.timeseries import POSITION_FIELD
from threetears.evals.vega.palette import font_sizes, font_weights, geometry

#: The row key naming the run of consecutive points a row belongs to. A line connects the rows sharing one,
#: so a gap — a position with no point — ends one segment and the next point starts another.
_SEGMENT_FIELD = "segment"

#: A point's mark area in px², ~9px across.
_POINT_SIZE = 72

#: How far a point extends past the value it is centred on, in px.
_POINT_RADIUS = point_radius(_POINT_SIZE)


def _segments(intent: ChartIntent, order: list[str]) -> list[str]:
    """Name each point's run of consecutive positions, so a gap ends a line rather than bridging it.

    Args:
        intent: The timeseries intent, whose data lists each series' points in axis order.
        order: The time axis's positions, earliest first.

    Returns:
        One segment name per data row, in data order.
    """
    index = {position: number for number, position in enumerate(order)}
    names: list[str] = []
    previous: dict[str, tuple[int, int]] = {}
    for row in intent.data:
        series, at = str(row["series"]), index[str(row[POSITION_FIELD])]
        segment, last = previous.get(series, (0, at - 1))
        if at != last + 1:
            segment += 1
        previous[series] = (segment, at)
        names.append(f"{series}#{segment}")
    return names


def compile_timeseries(intent: ChartIntent, *, font: ChartFont | None = None) -> dict[str, Any]:
    """Draw one reading across the time axis as a panel per series, each point with its interval.

    Args:
        intent: The timeseries intent.
        font: The typeface the chart is laid out in; ``None`` for the packaged face.

    Returns:
        The Vega-Lite spec.
    """
    sizes = geometry()
    time_axis = _value_axis(intent, "time")
    categories = _Categories.of("series", _identity(intent).order, font=font)
    width = int(sizes["plot_width"])
    # A panel is a plot of its own, so it takes the floor a plot is never drawn below rather than a row step.
    height = int(sizes["plot_min_height"])

    points: list[dict[str, Any]] = [
        dict(row) | {DISPLAY_FIELD: categories.display[str(row["series"])], _SEGMENT_FIELD: segment}
        for row, segment in zip(intent.data, _segments(intent, time_axis.order), strict=True)
    ]
    marks = [row | {KIND_FIELD: "point"} for row in points] + [
        {
            DISPLAY_FIELD: row[DISPLAY_FIELD],
            POSITION_FIELD: row[POSITION_FIELD],
            "mean": row["mean"],
            VALUE_TEXT_FIELD: format_number(_number(row["mean"])),
            KIND_FIELD: "value",
        }
        for row in points
    ]

    # The quantity is named once, in the heading: an axis title in every panel is the same words repeated,
    # and the vertical one Vega would turn.
    value_axis = ValueAxis.position("", [_number(row[key]) for row in points for key in ("low", "high")], height)
    time: dict[str, Any] = {
        "field": POSITION_FIELD,
        "type": "ordinal",
        # The order is the intent's — earliest first — and stated, never left to Vega, which would sort the
        # names alphabetically and draw the line through them in that order.
        "sort": list(time_axis.order),
        "scale": {"domain": list(time_axis.order)},
        "axis": {
            "title": time_axis.quantity,
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
    return {
        "$schema": VEGA_LITE_SCHEMA,
        "title": _title_spec(intent.title, categories.figure_width(), font=font),
        "description": intent.intervals,
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


__all__ = [
    "compile_timeseries",
]
