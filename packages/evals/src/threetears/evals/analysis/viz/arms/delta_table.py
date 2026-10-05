"""The ``delta_table`` arm — an A-vs-B comparison as a dumbbell on relative change.

The rows are different metrics in different units, so the one thing they can share
is a unitless axis. Everything here follows from that: what is plotted is relative
change, what is tabulated is each row's own unit, and a row that cannot produce a
relative change is named rather than dropped.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.viz.compiler import (
    VEGA_LITE_SCHEMA,
    MarkValue,
    ValueAxis,
    _bar_mark,
    _Categories,
    _composed,
    _identity,
    _layers,
    _number,
    _title_spec,
    _value_axis,
    _zero_rule,
    point_radius,
    value_label_layers,
)
from threetears.evals.analysis.viz.intent import ChartIntent

#: The axis floor for a relative-change comparison, as a fraction.
#:
#: An axis scaled only to its data puts the largest change at the edge every
#: time, so a set of 3% changes draws identically to a set of 300% ones and bar
#: length becomes a constant that still reads as magnitude. Ten percent is a
#: change most operators would act on, which makes it the right reference for
#: "small" without being so wide that real movement disappears.
_CHANGE_AXIS_FLOOR = 0.1

#: The point mark's size in px², which is the mark that IS the value.
_POINT_SIZE = 80

#: How far that point extends past the value it is centred on, in px.
#:
#: Derived from the size rather than written beside it: a point is sized by area and
#: an axis has to be told a distance, and two numbers that must agree are two numbers
#: that will not. The conversion is :func:`~threetears.evals.analysis.viz.compiler.point_radius`,
#: which is where Vega's bounding-box convention is stated once.
#:
#: Two things read it, and they are the same fact twice. It is what the value axis
#: reserves beyond the largest change — a point centred on the domain's end is drawn
#: half outside the plot and renders as a clipped half-circle jammed against the
#: identity labels. And it is what holds the value label off the point: the label is
#: set from the mark's EDGE, so a number offset from the centre instead is written
#: 1.5px from a disc it is supposed to sit beside, which reads as touching it.
_POINT_RADIUS = point_radius(_POINT_SIZE)

#: The dumbbell connector's thickness, in px.
#:
#: **Stated in px, not as a fraction of the band, and that is :func:`_bar_mark`'s own
#: rule rather than a preference here.** The band varies with the row count — one row
#: gets the whole 168px floor, three rows get 56px each, ten get 48 — so a fraction of
#: it encodes the row count in the mark. At one row it inverted the dumbbell outright:
#: 18% of the band drew a 30px connector under a 10px point, which is a plain bar, and
#: a plain bar is exactly the "two marks competing to be read as the quantity" this
#: shape exists to avoid.
#:
#: Three px is what the other two dumbbells in this package already draw —
#: ``null_result``'s interval rule and ``distribution``'s estimate rule, both under a
#: point of the same order — so the recession reads the same across one report.
_CONNECTOR_HEIGHT = 3


def compile_delta_table(intent: ChartIntent) -> dict[str, Any]:
    """Draw an A-vs-B intent as a dumbbell on one shared relative-change axis.

    A is pinned at zero because it is the baseline the comparison runs *from*, so the bar's length is
    the size of the change and its side is the direction. The axis is floored so a set of small changes
    still renders small.

    Args:
        intent: The comparison's intent.

    Returns:
        The Vega-Lite spec.
    """
    plotted = intent.data
    reach = max((abs(_number(entry["change"])) for entry in plotted), default=0.0)
    axis_title = _value_axis(intent, "change").quantity
    categories = _Categories.of("metric", _identity(intent).order)
    width, height = categories.plot_size()
    # Stated, not inferred from the bar lengths: the reader needs to know how wide
    # "the edge of the chart" is before a bar's length means anything. The floor and
    # the point's own radius are both stated here and resolved there, because turning
    # either of them into a domain is the value→px arithmetic `ValueAxis` owns.
    value_axis = ValueAxis.symmetric(
        axis_title, reach, width, floor=_CHANGE_AXIS_FLOOR, mark_clearance=_POINT_RADIUS, number_format="+.0%"
    )
    axis = value_axis.encoding("change")
    identity = categories.axis(domain=not value_axis.marks_form_the_edge())
    layers: list[dict[str, Any]] = [
        {
            # A BAR from the axis zero, not a rule between two fields. The two draw
            # identically and mean different things: this length is a magnitude
            # measured from the baseline, so it belongs to the rule that a length
            # encoding starts at zero — which a span between two data values does
            # not, and would have escaped by looking like an uncertainty interval.
            #
            # Thinner than the fixed row thickness, and it is the one bar here that stays
            # one: this is the connector of a dumbbell, sized to stay recessive under the
            # point that marks the value.
            "mark": _bar_mark(height=_CONNECTOR_HEIGHT),
            "encoding": {"y": identity, "x": axis},
        },
        {
            "mark": {"type": "point", "filled": True, "size": _POINT_SIZE, "tooltip": True},
            "encoding": {
                "y": identity,
                "x": axis,
                "tooltip": [
                    {"field": "metric", "type": "nominal", "title": "Metric"},
                    {"field": "change", "type": "quantitative", "title": axis_title, "format": "+.1%"},
                    {"field": "effect", "type": "nominal", "title": "Effect"},
                ],
            },
        },
    ]
    # The value as the axis states it — a percentage, at the precision the tooltip and
    # the values table use, so one row reads the same in all three places.
    # `filled=False`: what this arm draws AT a value is a 3px connector and a point, and
    # 3px under a ~10px glyph band is nothing to knock a number out of.
    # `thickness`: a label pushed inward is pushed along the connector, so it is lifted
    # clear of it rather than laid on it.
    # `radius`: the point is CENTRED on the change it names, so the label is set from
    # the mark's edge rather than its centre.
    labels = [
        MarkValue(
            display=categories.display[str(entry["metric"])],
            end=_number(entry["change"]),
            text=f"{_number(entry['change']):+.1%}",
            radius=_POINT_RADIUS,
            filled=False,
            thickness=_CONNECTOR_HEIGHT,
        )
        for entry in plotted
    ]
    return _composed(
        {
            "$schema": VEGA_LITE_SCHEMA,
            "title": _title_spec(intent.title, categories.figure_width()),
            "data": {"values": categories.labelled(plotted)},
            "layer": _layers(_zero_rule(value_axis), *layers, *value_label_layers(labels, value_axis, identity)),
            "width": width,
            "height": height,
        },
        categories,
    )


__all__ = [
    "compile_delta_table",
]
