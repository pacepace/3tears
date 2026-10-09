"""The ``null_result`` arm — an established null's arms as intervals, overlap shaded.

The overlap band is geometry and is drawn as geometry. Neither the shading nor the
prose beside it may read as a verdict: what settles the comparison is a test on the
difference, and that reaches the reader through the finding, not through this
picture.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.vega.compiler import (
    ANCHOR_FIELD,
    DISPLAY_FIELD,
    SECONDARY_OPACITY,
    VALUE_LABEL_OFFSET,
    VALUE_TEXT_FIELD,
    VEGA_LITE_SCHEMA,
    MarkValue,
    Placement,
    ValueAxis,
    _Categories,
    _composed,
    _identity,
    _number,
    _title_spec,
    _value_axis,
    centred_value_placements,
    point_radius,
    value_label_mark,
)
from threetears.evals.analysis.viz.intent import ChartIntent
from threetears.evals.contracts.host import ChartFont
from threetears.evals.vega.palette import font_sizes

#: The interval rule's thickness, in px.
#:
#: The same 3px the other two dumbbells in this package draw — ``delta_table``'s
#: connector and ``distribution``'s estimate rule — so an interval recedes under its
#: point the same way across one report.
_INTERVAL_HEIGHT = 3

#: The mean's point mark, in px² (Vega's bounding-box convention — see
#: :func:`~threetears.evals.vega.compiler.point_radius`).
#:
#: Named because the value label has to clear it: the label is anchored at the mean,
#: which is where this point is centred.
_POINT_SIZE = 70


def _value_lift() -> float:
    """How far each arm's value label is raised above its row's centreline, in px.

    The label names the MEAN and is anchored there, which is the middle of the
    interval — occupied by the span rule and by the mean's own point. So it is lifted
    onto a line of its own above them, the way ``distribution`` writes its estimate:
    the point's radius, then the :data:`VALUE_LABEL_OFFSET` daylight every value label
    keeps from the edge of its mark, then half the label's line box, since the label is
    drawn ``baseline: middle``. The rule is 3px and sits well inside the point's radius.

    Returns:
        The lift, in px.
    """
    return point_radius(_POINT_SIZE) + VALUE_LABEL_OFFSET + font_sizes()["value"] / 2


def compile_null_result(intent: ChartIntent, *, font: ChartFont | None = None) -> dict[str, Any]:
    """Draw an established null's intent: each arm as an interval, with any overlap shaded.

    The overlap band is drawn because the reader needs to see where the arms coincide. It is **not**
    evidence of the null: two marginal intervals can overlap while the difference between the means is
    real, and reading overlap as a verdict once published a false null over arms 33% apart.

    Args:
        intent: The null result's intent.
        font: The typeface the chart is laid out in; ``None`` for the packaged face.

    Returns:
        The Vega-Lite spec.
    """
    rows = intent.data
    axis_title = _value_axis(intent, "value").quantity
    bounds = [_number(row[key]) for row in rows for key in ("low", "high")]
    # The overlap is [max of lows, min of highs] — non-empty only if the arms overlap.
    overlap_low = max(_number(row["low"]) for row in rows)
    overlap_high = min(_number(row["high"]) for row in rows)
    overlaps = overlap_high >= overlap_low

    categories = _Categories.of("label", _identity(intent).order, font=font)
    # The taller step: each row carries its value on a line of its own above the mark.
    width, height = categories.plot_size(value_above=True)
    # A position, so the axis crops to the arms and states that it did.
    value_axis = ValueAxis.position(axis_title, bounds, width)
    identity = categories.axis()
    layers: list[dict[str, Any]] = []
    if overlaps:
        layers.append(
            {
                "data": {"values": [{"low": overlap_low, "high": overlap_high}]},
                "mark": {"type": "rect", "opacity": SECONDARY_OPACITY},
                "encoding": {"x": value_axis.encoding("low"), "x2": {"field": "high"}},
            }
        )
    layers.append(
        {
            "data": {"values": categories.labelled(rows)},
            "mark": {"type": "rule", "size": _INTERVAL_HEIGHT, "tooltip": True},
            "encoding": {
                "y": identity,
                "x": value_axis.encoding("low"),
                "x2": {"field": "high"},
            },
        }
    )
    layers.append(
        {
            "data": {"values": categories.labelled(rows)},
            "mark": {"type": "point", "filled": True, "size": _POINT_SIZE, "tooltip": True},
            "encoding": {"y": identity, "x": value_axis.encoding("mean")},
        }
    )
    # Anchored at the MEAN it prints. It used to be anchored at the interval's HIGH end
    # while printing the mean, so every number sat at an x-position it did not name: an
    # arm with mean 17.18 over 14.4-19.96 printed "17.18" beside x≈19.96. Lifted clear
    # of the rule and the point rather than set beside them, because the mean is inside
    # the mark. No fill style: an interval rule and a point have no fill under a label
    # drawn above them.
    lift = _value_lift()
    estimates = [
        MarkValue(display=row[DISPLAY_FIELD], end=_number(row["mean"]), text=format_number(_number(row["mean"])))
        for row in categories.labelled(rows)
    ]
    for placement, marks in centred_value_placements(estimates, value_axis, font=font).items():
        layers.append(
            {
                "data": {
                    "values": [
                        {DISPLAY_FIELD: mark.display, ANCHOR_FIELD: mark.end, VALUE_TEXT_FIELD: mark.text}
                        for mark in marks
                    ]
                },
                "mark": value_label_mark(Placement(align=placement, inside=False, lift=lift)),
                "encoding": {
                    "y": identity,
                    "x": value_axis.encoding(ANCHOR_FIELD),
                    "text": {"field": VALUE_TEXT_FIELD, "type": "nominal"},
                },
            }
        )

    spec: dict[str, Any] = _composed(
        {
            "$schema": VEGA_LITE_SCHEMA,
            "title": _title_spec(intent.title, categories.figure_width(), "", font=font),
            "layer": layers,
            "width": width,
            "height": height,
        },
        categories,
        # A name drawn above its row stacks above the value line rather than through it:
        # the value's line box reaches half its height past the lift.
        clearance_above=lift + font_sizes()["value"] / 2,
    )
    # Never empty here: an arm's `ci` is required and the payload holds at least two, so the intent
    # always states its intervals — which is what the spec gate reads.
    spec["description"] = intent.intervals
    return spec


__all__ = [
    "compile_null_result",
]
