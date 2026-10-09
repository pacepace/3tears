"""The ``frontier`` arm — contestants placed on cost against quality.

The one type here whose identity axis is quantitative on both sides: every other
chart puts its categories on an axis and reads one quantity against them, and this
one reads two quantities against each other and puts the categories beside their
marks. So it takes the point-plot geometry rather than the row-based figure width,
and both of its axes are positions, which means both may crop — and their labelled
ticks are what disclose it, since there is no crop footnote.

**Dominance rides on shape and recession, never hue.** A contestant not shown
dominated draws a circle, one never tested a square, a dominated one a diamond, a
disqualified one a cross, and the recessive classes are additionally drawn in the
neutral that carries no identity.
Two reasons the shape is load-bearing rather than decorative: identity may not ride
on hue, since the palette recycles and an axis does not; and a distinction drawn
only in weight is one a reader with low contrast vision does not receive, which is
why the recession is the second channel and never the only one.

**The recession is asked for by NAME, not stated as an opacity.** The recessive
marks are their own layer wearing the ``chart-context`` style, so each renderer
decides what receding looks like on the surface it is painting — the same alpha is
not the same recession on obsidian and on pearl, and this one spec is drawn onto
both.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.vega.compiler import (
    VEGA_LITE_SCHEMA,
    ValueAxis,
    _name_font_size,
    _number,
    _title_spec,
    _value_axis,
    point_radius,
)
from threetears.evals.analysis.viz.intent import ChartIntent
from threetears.evals.analysis.viz.intents.frontier import CLASS_FIELD, CLASS_SHAPES, DISPLAY_FIELD, EMPHATIC_CLASSES
from threetears.evals.vega.palette import CONTEXT_STYLE, font_weights, geometry

#: The classes drawn in the context neutral rather than in the chart's own ink.
#:
#: Every class but the emphatic ones the intent names, so a class added later cannot
#: silently draw at full weight.
_RECESSIVE_CLASSES: tuple[str, ...] = tuple(name for name, _ in CLASS_SHAPES if name not in EMPHATIC_CLASSES)

#: How far a contestant's name sits to the right of its mark's EDGE, in px.
#:
#: From the edge, not from the point's position: this arm's mark is the largest in the
#: package — ~20px across, because the shape channel has to survive being read — so a
#: name offset from the CENTRE started 0.9px INSIDE the disc, measured in the rendered
#: SVG. The offset is added to :data:`_POINT_RADIUS` where the label is placed.
_LABEL_OFFSET = 8

#: A contestant's mark area in px², which is ~20px across.
#:
#: Larger than the dot an interval carries, and the reason is the shape channel
#: rather than emphasis: a circle and a diamond are the same silhouette at 12px, so
#: a mark too small to tell the two apart drops the redundant channel back onto
#: opacity alone — the one encoding this chart may not rely on. The plot is small by
#: design; the marks in it are not.
_POINT_SIZE = 320

#: How far a contestant's mark extends past the point it is placed at, in px.
#:
#: :data:`_POINT_SIZE` is an area and a label offset is a distance, so the conversion
#: is stated once — in :func:`~threetears.evals.vega.compiler.point_radius`, which is also
#: where Vega's bounding-box convention is written down.
_POINT_RADIUS = point_radius(_POINT_SIZE)


def _by_weight(drawn: list[dict[str, Any]]) -> list[tuple[bool, list[str]]]:
    """Split the drawn points into the weights their marks are painted at.

    Two layers at most, and only the ones with something in them: an empty layer
    is a filter matching nothing, which Vega draws as an invisible mark and a
    reader never sees — but it still joins the scale resolution, and a figure whose
    no contestant recedes should not carry a recessive layer at all.

    Args:
        drawn: The plotted rows, each carrying its contention class.

    Returns:
        ``(recessive, class names)`` per layer, the emphatic layer first so it is
        composed UNDER nothing and the receding marks cannot cover it.
    """
    partitions = []
    for recessive in (False, True):
        classes = [name for name, _ in CLASS_SHAPES if (name in _RECESSIVE_CLASSES) == recessive]
        present = [name for name in classes if any(row[CLASS_FIELD] == name for row in drawn)]
        if present:
            partitions.append((recessive, present))
    return partitions


def compile_frontier(intent: ChartIntent) -> dict[str, Any]:
    """Draw a cost-against-quality intent as a point plot.

    Args:
        intent: The frontier's intent.

    Returns:
        The Vega-Lite spec.
    """
    sizes = geometry()
    width, height = int(sizes["point_width"]), int(sizes["point_height"])
    cost_title = _value_axis(intent, "cost").quantity
    quality_title = _value_axis(intent, "quality").quantity
    drawn = intent.data
    bars = [reference.value for reference in intent.references if reference.axis == "quality"]

    # The bar joins the y values so the rule it draws lands inside the plot. A
    # quality bar above everything measured is the informative case — nothing
    # cleared it — and an axis cropped to the data alone would put that rule off
    # the top edge.
    qualities = [_number(row["quality"]) for row in drawn] + bars
    cost_axis = ValueAxis.position(cost_title, [_number(row["cost"]) for row in drawn], width)
    # The y axis names its own quantity, drawn FLAT above the axis rather than turned.
    quality_axis = ValueAxis.position(quality_title, qualities, height)

    present = list(intent.shapes.items())
    shape: dict[str, Any] = {
        "field": CLASS_FIELD,
        "type": "nominal",
        "scale": {"domain": [name for name, _ in present], "range": [symbol for _, symbol in present]},
        # NO legend, ever. A shape key's swatches are filled from the MARK, so a
        # dominated point drawn in the context neutral still appeared in the key in the
        # primary hue — telling the reader the greying means nothing. A colour ENCODING
        # would fix the swatch and is refused by the direct-label rule, correctly. Every
        # mark here is already labelled in place with its contestant's name, and the
        # shape vocabulary reaches the reader as a disclosure line.
        "legend": None,
    }

    # One point layer per weight, each filtered out of the SAME table rather than
    # drawn from a dataset of its own: the shape scale carries an explicit domain,
    # so Vega-Lite shares it across the layers. Two layers rather than one mark with
    # an opacity encoding: a style name defers what "receded" looks like to the
    # renderer's theme, and a style is a property of a mark.
    layers: list[dict[str, Any]] = [
        {
            "transform": [{"filter": {"field": CLASS_FIELD, "oneOf": list(classes)}}],
            # `opacity: 1` opts OUT of Vega-Lite's 0.7 point default: these are a handful
            # of answers rather than a cloud. The recession is the STYLE, and it is
            # load-bearing: a dominated or disqualified point differs in shape AND in ink.
            "mark": {"type": "point", "filled": True, "size": _POINT_SIZE, "tooltip": True, "opacity": 1}
            | ({"style": CONTEXT_STYLE} if recessive else {}),
            "encoding": {
                "x": cost_axis.encoding("cost"),
                "y": quality_axis.encoding("quality", upright_title=True),
                "shape": shape,
            },
        }
        for recessive, classes in _by_weight(drawn)
    ]
    layers.append(
        {
            # The name beside the mark, which is where identity lives on every chart
            # this compiler emits.
            "mark": {
                "type": "text",
                "align": "left",
                "baseline": "middle",
                # From the mark's edge: the point is centred on the contestant's
                # position, so its radius is distance the name never had.
                "dx": _POINT_RADIUS + _LABEL_OFFSET,
                "fontSize": _name_font_size(),
                "fontWeight": font_weights()["label"],
            },
            "encoding": {
                "x": cost_axis.encoding("cost"),
                "y": quality_axis.encoding("quality", upright_title=True),
                "text": {"field": DISPLAY_FIELD, "type": "nominal"},
            },
        }
    )
    for bar in bars:
        layers.insert(
            0,
            {
                # Under the marks: the bar is the standard they are read against, and a
                # rule drawn over a contestant sitting on it would cut the mark in half.
                # Context ink, by name: the bar is furniture, so it recedes for the same
                # reason a zero rule does. The dash is geometry and stays in the spec.
                "data": {"values": [{"bar": bar}]},
                "mark": {"type": "rule", "strokeDash": [4, 4], "style": CONTEXT_STYLE},
                "encoding": {"y": quality_axis.encoding("bar", upright_title=True)},
            },
        )

    # No crop footnote. Both axes are cropped and both say so already, in the only
    # way a reader of a point plot consults: the labelled ticks.
    return {
        "$schema": VEGA_LITE_SCHEMA,
        "title": _title_spec(intent.title, width),
        "data": {"values": drawn},
        "layer": layers,
        "width": width,
        "height": height,
    }


__all__ = [
    "compile_frontier",
]
