"""The ``attribution`` arm — a whole and a part on one shared signed axis.

The type exists to draw a subtraction that may not be earned, so the remainder's
row is the shape that matters: it is always present, and only its MARK changes
between "this much was left over" and "this cannot be placed".
"""

from __future__ import annotations

from typing import Any

from threetears.evals.vega.compiler import (
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
from threetears.evals.kernel.host import ChartFont
from threetears.evals.analysis.viz.intents.attribution import REMAINDER_SCOPE
from threetears.evals.analysis.viz.quantities import signed_with_unit
from threetears.evals.vega.palette import CONTEXT_STYLE

#: What an `attribution` draws in the remainder's row when the subtraction was not
#: earned. Words rather than a zero-length bar: the row has to stay on the axis so
#: the picture records that the question was asked, and anything MEASURED against
#: the value axis there would read as "nothing was left over".
_NOT_PLACEABLE = "not placeable"


def compile_attribution(intent: ChartIntent, *, font: ChartFont | None = None) -> dict[str, Any]:
    """Draw a whole-vs-part intent onto one shared signed axis.

    **Two bars side by side, never stacked, and never a waterfall.** A waterfall lays the part
    end-to-end against the whole and closes the gap with a remainder, which draws a balanced partition —
    the exact attribution the finding says it cannot make. Bars from a common zero state each movement
    as its own magnitude.

    **The remainder always occupies a row; only its MARK changes.** Where the arithmetic is earned it is
    a bar, drawn in the ``chart-context`` style rather than the full ink of the two measured movements
    because it is derived rather than observed. Where it is withheld the same row carries the words *not placeable*: omitting the row
    leaves the picture silent about a question the finding asked, and drawing it EMPTY reads as "the
    movement was fully accounted for" — the one misreading this type exists to prevent.

    Args:
        intent: The attribution's intent.
        font: The typeface the chart is laid out in; ``None`` for the packaged face.

    Returns:
        The Vega-Lite spec.
    """
    plotted = intent.data
    quantified = any(row["scope"] == REMAINDER_SCOPE for row in plotted)
    axis_title = _value_axis(intent, "change").quantity
    categories = _Categories.of("scope", _identity(intent).order, font=font)
    width, height = categories.plot_size()
    value_axis = ValueAxis.magnitude(axis_title, [_number(row["delta"]) for row in plotted], width)
    identity = categories.axis(domain=not value_axis.marks_form_the_edge())
    labelled = categories.labelled(plotted)

    def bars(derived: bool) -> dict[str, Any] | None:
        """One bar layer per weight: the measured movements at full ink, the remainder receding.

        The remainder is a different KIND of quantity (computed, not observed), so it
        recedes; identity is already on the axis labels, so hue carries nothing. The
        recession is asked for by NAME, as the frontier's dominated marks do, because a
        compiled alpha chosen for emphasis is right on at most one of the two surfaces a
        stored spec is drawn onto. A weight with no row is no layer at all.
        """
        if not any(bool(row.get("derived")) == derived for row in plotted):
            return None
        return {
            "data": {"values": labelled},
            "transform": [{"filter": "datum.derived" if derived else "!datum.derived"}],
            "mark": _bar_mark(tooltip=True) | ({"style": CONTEXT_STYLE} if derived else {}),
            "encoding": {
                "y": identity,
                "x": value_axis.encoding("delta"),
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
            "title": _title_spec(intent.title, categories.figure_width(), font=font),
            "layer": _layers(
                _zero_rule(value_axis),
                bars(derived=False),
                bars(derived=True),
                not_placeable,
                *value_label_layers(labels, value_axis, identity, font=font),
            ),
            "width": width,
            "height": height,
        },
        categories,
    )


__all__ = [
    "compile_attribution",
]
