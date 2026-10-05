"""The ``null_result`` arm — an established null's arms as intervals, overlap shaded.

The overlap band is geometry and is drawn as geometry. Neither the shading nor the
prose beside it may read as a verdict: what settles the comparison is a test on the
difference, and that reaches the reader through the finding, not through this
picture.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.viz.compiler import (
    DISPLAY_FIELD,
    SECONDARY_OPACITY,
    VEGA_LITE_SCHEMA,
    MarkValue,
    ValueAxis,
    _Categories,
    _composed,
    _identity,
    _number,
    _title_spec,
    _value_axis,
    value_label_layers,
)
from threetears.evals.analysis.viz.intent import ChartIntent

#: The interval rule's thickness, in px.
#:
#: The same 3px the other two dumbbells in this package draw — ``delta_table``'s
#: connector and ``distribution``'s estimate rule — so an interval recedes under its
#: point the same way across one report. Named rather than written into the mark
#: alone because the label placement has to know it: a number pushed inward lands on
#: this rule, and how far it must be lifted off it is this number's business.
_INTERVAL_HEIGHT = 3


def compile_null_result(intent: ChartIntent) -> dict[str, Any]:
    """Draw an established null's intent: each arm as an interval, with any overlap shaded.

    The overlap band is drawn because the reader needs to see where the arms coincide. It is **not**
    evidence of the null: two marginal intervals can overlap while the difference between the means is
    real, and reading overlap as a verdict once published a false null over arms 33% apart.

    Args:
        intent: The null result's intent.

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

    categories = _Categories.of("label", _identity(intent).order)
    width, height = categories.plot_size()
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
            "mark": {"type": "point", "filled": True, "size": 70, "tooltip": True},
            "encoding": {"y": identity, "x": value_axis.encoding("mean")},
        }
    )
    layers.extend(
        value_label_layers(
            [
                # `filled=False`: this arm draws a 3px rule and a point, and 3px under a
                # ~10px glyph band is nothing to knock a number out of — the knockout ink
                # IS the chart surface, so a label taking it there would be painted on the
                # background in the background's own colour.
                #
                # `thickness`: inward of the value is not EMPTY either. The label is anchored
                # at the high end and pushed inward when the axis has no room past it, and
                # inward is back along the rule, so it is lifted off the rule rather than
                # drawn through it.
                #
                # No `radius`: the label is anchored at the interval's HIGH end, where the
                # rule stops — a butt cap at the value, whose 3px is thickness across the row
                # and nothing along the axis.
                MarkValue(
                    display=row[DISPLAY_FIELD],
                    end=_number(row["high"]),
                    text=format_number(_number(row["mean"])),
                    filled=False,
                    thickness=_INTERVAL_HEIGHT,
                )
                for row in categories.labelled(rows)
            ],
            value_axis,
            identity,
        )
    )

    spec: dict[str, Any] = _composed(
        {
            "$schema": VEGA_LITE_SCHEMA,
            "title": _title_spec(intent.title, categories.figure_width(), ""),
            "layer": layers,
            "width": width,
            "height": height,
        },
        categories,
    )
    # Never empty here: an arm's `ci` is required and the payload holds at least two, so the intent
    # always states its intervals — which is what the spec gate reads.
    spec["description"] = intent.intervals
    return spec


__all__ = [
    "compile_null_result",
]
