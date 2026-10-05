"""The ``attribution`` arm — a whole and a part on one shared signed axis.

The type exists to draw a subtraction that may not be earned, so the remainder's
row is the shape that matters: it is always present, and only its MARK changes
between "this much was left over" and "this cannot be placed".
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.viz.compiler import (
    SECONDARY_OPACITY,
    VALUE_LABEL_OFFSET,
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
    value_label_layers,
)
from threetears.evals.analysis.viz.intent import ChartIntent
from threetears.evals.analysis.viz.intents.attribution import REMAINDER_SCOPE
from threetears.evals.analysis.viz.quantities import signed_with_unit

#: What an `attribution` draws in the remainder's row when the subtraction was not
#: earned. Words rather than a zero-length bar: the row has to stay on the axis so
#: the picture records that the question was asked, and anything MEASURED against
#: the value axis there would read as "nothing was left over".
_NOT_PLACEABLE = "not placeable"


def compile_attribution(intent: ChartIntent) -> dict[str, Any]:
    """Draw a whole-vs-part intent onto one shared signed axis.

    **Two bars side by side, never stacked, and never a waterfall.** A waterfall lays the part
    end-to-end against the whole and closes the gap with a remainder, which draws a balanced partition —
    the exact attribution the finding says it cannot make. Bars from a common zero state each movement
    as its own magnitude.

    **The remainder always occupies a row; only its MARK changes.** Where the arithmetic is earned it is
    a bar, drawn at a lower opacity than the two measured movements because it is derived rather than
    observed. Where it is withheld the same row carries the words *not placeable*: omitting the row
    leaves the picture silent about a question the finding asked, and drawing it EMPTY reads as "the
    movement was fully accounted for" — the one misreading this type exists to prevent.

    Args:
        intent: The attribution's intent.

    Returns:
        The Vega-Lite spec.
    """
    plotted = intent.data
    quantified = any(row["scope"] == REMAINDER_SCOPE for row in plotted)
    axis_title = _value_axis(intent, "change").quantity
    categories = _Categories.of("scope", _identity(intent).order)
    width, height = categories.plot_size()
    value_axis = ValueAxis.magnitude(axis_title, [_number(row["delta"]) for row in plotted], width)
    identity = categories.axis(domain=not value_axis.marks_form_the_edge())
    bars: dict[str, Any] = {
        "data": {"values": categories.labelled(plotted)},
        "mark": _bar_mark(tooltip=True),
        "encoding": {
            "y": identity,
            "x": value_axis.encoding("delta"),
            # Opacity, not hue: the remainder is a different KIND of quantity, not a
            # different category, and identity is already on the axis labels.
            "opacity": {
                "condition": {"test": "datum.derived", "value": SECONDARY_OPACITY},
                "value": 1,
            },
            "tooltip": [
                {"field": "scope", "type": "nominal", "title": "Scope"},
                {"field": "measure", "type": "nominal", "title": "Measure"},
                {"field": "delta", "type": "quantitative", "title": axis_title},
            ],
        },
    }
    not_placeable: dict[str, Any] | None = None
    if not quantified:
        # Words, not an absence. Anchored at the zero line so it starts where a bar
        # would, and it is a `text` mark rather than a zero-length one precisely
        # because nothing about it can be read off the value axis.
        #
        # Which SIDE of zero it reads into is not cosmetic. Zero sits at the domain's
        # edge whenever every movement shares a sign, so on an all-negative chart
        # zero is the RIGHT edge, and a left-aligned label runs off the plot. Clipped,
        # the row renders empty — the "fully accounted for" misreading. So the text
        # reads back into the plot, asked of the drawn domain.
        anchors_right = value_axis.high == 0
        not_placeable = {
            "data": {"values": categories.labelled([{"scope": REMAINDER_SCOPE, "delta": 0, "label": _NOT_PLACEABLE}])},
            "mark": {
                "type": "text",
                "align": "right" if anchors_right else "left",
                "dx": -VALUE_LABEL_OFFSET if anchors_right else VALUE_LABEL_OFFSET,
                "fontStyle": "italic",
            },
            "encoding": {
                "y": identity,
                "x": value_axis.encoding("delta"),
                "text": {"field": "label", "type": "nominal"},
            },
        }
    labels = [
        # Signed, and without the unit the axis already states: the direction is half
        # the fact on this chart, while the unit written on every mark is the axis
        # title copied.
        MarkValue(
            display=categories.display[str(row["scope"])],
            end=_number(row["delta"]),
            text=signed_with_unit(_number(row["delta"]), ""),
        )
        for row in plotted
    ]
    return _composed(
        {
            "$schema": VEGA_LITE_SCHEMA,
            "title": _title_spec(intent.title, categories.figure_width()),
            "layer": _layers(
                _zero_rule(value_axis), bars, not_placeable, *value_label_layers(labels, value_axis, identity)
            ),
            "width": width,
            "height": height,
        },
        categories,
    )


__all__ = [
    "compile_attribution",
]
