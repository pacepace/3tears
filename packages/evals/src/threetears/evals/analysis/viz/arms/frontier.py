"""The ``frontier`` arm — contestants placed on cost against quality.

The one type here whose identity axis is quantitative on both sides: every other
chart puts its categories on an axis and reads one quantity against them, and this
one reads two quantities against each other and puts the categories beside their
marks. So it takes the point-plot geometry rather than the row-based figure width,
and both of its axes are positions, which means both may crop — and their labelled
ticks are what disclose it, since there is no crop footnote.

**Dominance rides on shape and recession, never hue.** An on-frontier contestant
draws a circle, a dominated one a diamond, a disqualified one a cross, and the
recessive classes are additionally drawn in the neutral that carries no identity.
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

from collections.abc import Sequence
from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.viz.compiler import (
    DISPLAY_FIELD,
    VEGA_LITE_SCHEMA,
    ChartColumn,
    CompiledChart,
    _axis_title,
    _name_font_size,
    strip_common_prefix,
    _title_spec,
    ValueAxis,
    display_scale,
    point_radius,
)
from threetears.evals.analysis.viz.palette import CONTEXT_STYLE, font_weights, geometry
from threetears.evals.analysis.viz.payloads import FrontierPayload, FrontierVizPoint

#: The point symbol each contention class draws, and the order a key lists them in.
#:
#: Ordered best-to-worst rather than alphabetically, so a legend reads down the
#: same ranking the chart is about. The symbols are Vega-Lite shape names, which
#: name geometry and never a colour — the whole reason this channel can carry a
#: distinction that hue is not allowed to.
_CLASS_SHAPES: tuple[tuple[str, str], ...] = (
    ("On frontier", "circle"),
    ("Dominated", "diamond"),
    ("Disqualified", "cross"),
)

#: The classes drawn in the context neutral rather than in the chart's own ink.
#:
#: Derived from :data:`_CLASS_SHAPES` rather than listed again, so a fourth class
#: cannot be added to the shape key and silently draw at full weight. The head of
#: that tuple is the class a reader is being pointed AT; everything after it is
#: there for comparison.
_RECESSIVE_CLASSES: tuple[str, ...] = tuple(name for name, _ in _CLASS_SHAPES[1:])

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
#: is stated once — in :func:`~threetears.evals.analysis.viz.compiler.point_radius`, which is also
#: where Vega's bounding-box convention is written down.
_POINT_RADIUS = point_radius(_POINT_SIZE)

#: The row key holding a point's contention class — what its shape is drawn from.
_CLASS_FIELD = "status"

#: The class of a contestant that carries no cost, and so is not on the plot at all.
#:
#: Deliberately absent from :data:`_CLASS_SHAPES`: it has no symbol because it has
#: no mark. It exists so the values table can say why a row is not in the picture
#: rather than filing it under a verdict nothing measured.
_UNPRICED_CLASS = "Not priced"


def _classify(dominated: bool, disqualified: bool, *, priced: bool) -> str:
    """Which contention class a point belongs to.

    Disqualification outranks domination: a contestant out on a safety bar is out
    whatever its cost bought, and drawing it as merely dominated would file a
    two-pillar failure under "lost on price".

    An unpriced contestant is its own class rather than the default one, and that
    is a correctness rule rather than a label. Domination is a claim about both
    axes, so a point with no cost cannot be known to be on the frontier OR off it —
    reporting it as "on frontier" would put a contestant nothing was measured
    against beside the winner, in the column a reader reads the verdict from.

    Args:
        dominated: Whether some other point beats it on both axes.
        disqualified: Whether it failed a two-pillar / safety bar.
        priced: Whether it carries a production-replicating cost.

    Returns:
        The class name, as the key and any legend spell it.
    """
    if disqualified:
        return "Disqualified"
    if dominated:
        return "Dominated"
    return "On frontier" if priced else _UNPRICED_CLASS


def _joined_names(labels: Sequence[str]) -> str:
    """Join contestant names so the sentence around them reads as English.

    Args:
        labels: The names, in the order the payload gave them.

    Returns:
        The one name, two joined by ``and``, or a comma list with ``and`` before the last.
    """
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


def _disqualifications(points: Sequence[FrontierVizPoint]) -> list[tuple[str, list[str]]]:
    """The disqualified contestants, grouped under the reason they share.

    Grouped rather than stated once per contestant, because the reason is what the sentence
    spends its words on: a whole class usually fails for one cause, so a per-contestant
    sentence repeats that cause verbatim as many times as the class is large, and the
    disclosure grows with the size of the thing it is disclosing. A disclosure nobody finishes
    reading discloses nothing.

    Reasons are matched exactly and kept in the order they first appear: two wordings of one
    cause are two groups, which is the honest read of two strings this module cannot know are
    the same claim.

    Args:
        points: The payload's contestants.

    Returns:
        ``(reason, labels)`` for each distinct reason, in first-appearance order.
    """
    grouped: dict[str, list[str]] = {}
    for point in points:
        if point.disqualified and point.disqualified_reason:
            grouped.setdefault(point.disqualified_reason.strip(), []).append(point.label)
    return list(grouped.items())


def _by_weight(drawn: list[dict[str, Any]]) -> list[tuple[bool, list[str]]]:
    """Split the drawn points into the weights their marks are painted at.

    Two layers at most, and only the ones with something in them: an empty layer
    is a filter matching nothing, which Vega draws as an invisible mark and a
    reader never sees — but it still joins the scale resolution, and a figure whose
    every contestant is on the frontier should not carry a recessive layer at all.

    Args:
        drawn: The plotted rows, each carrying its contention class.

    Returns:
        ``(recessive, class names)`` per layer, the emphatic layer first so it is
        composed UNDER nothing and the receding marks cannot cover it.
    """
    partitions = []
    for recessive in (False, True):
        classes = [name for name, _ in _CLASS_SHAPES if (name in _RECESSIVE_CLASSES) == recessive]
        present = [name for name in classes if any(row[_CLASS_FIELD] == name for row in drawn)]
        if present:
            partitions.append((recessive, present))
    return partitions


def compile_frontier(payload: FrontierPayload) -> CompiledChart:
    """Compile a cost-against-quality trade-off as a point plot.

    Args:
        payload: The validated frontier payload.

    Returns:
        The compiled chart.
    """
    sizes = geometry()
    width, height = int(sizes["point_width"]), int(sizes["point_height"])
    cost_title = payload.cost_label or "Cost"
    quality_title = payload.quality_label or "Quality"

    # A point with no cost has no x to be placed at. It is disclosed in its own line
    # and kept in the values table rather than dropped, because "we never priced
    # this one" and "this one was not in the running" are different facts and the
    # chart must not turn the first into the second. Placing it at zero would be
    # worse still: an unpriced contestant would draw as the cheapest thing here.
    placeable = [point for point in payload.points if point.cost is not None]
    unpriced = [point.label for point in payload.points if point.cost is None]

    # Latency is restated in the largest unit that keeps two significant figures, the
    # same rule every other quantity in this report obeys — 50 s, never 50300 ms. It
    # is the only quantity here with a declared unit to restate: cost and quality
    # carry captions the generator wrote, not units this compiler knows how to ladder.
    latencies = [point.latency_ms for point in payload.points if point.latency_ms is not None]
    latency_scale, latency_unit = display_scale(latencies, "ms" if latencies else None)

    display = strip_common_prefix([point.label for point in payload.points])
    rows: list[dict[str, Any]] = [
        {
            "label": point.label,
            DISPLAY_FIELD: display[point.label],
            "cost": point.cost,
            "quality": point.quality,
            "latency": None if point.latency_ms is None else point.latency_ms * latency_scale,
            _CLASS_FIELD: _classify(point.dominated, point.disqualified, priced=point.cost is not None),
        }
        for point in payload.points
    ]
    drawn = [row for row in rows if row["cost"] is not None]

    # The bar joins the y values so the rule it draws lands inside the plot. A
    # quality bar above everything measured is the informative case — nothing
    # cleared it — and an axis cropped to the data alone would put that rule off
    # the top edge, drawing the one chart where the bar matters as one with no bar.
    qualities = [point.quality for point in placeable]
    if payload.bar is not None:
        qualities.append(payload.bar)
    cost_axis = ValueAxis.position(cost_title, [point.cost for point in placeable if point.cost is not None], width)
    # The y axis names its own quantity, drawn FLAT above the axis rather than
    # turned. Leaving it unnamed was the earlier reading of the no-rotated-text
    # rule and it cost the reader more than it saved: the quantity was recoverable
    # only from the heading two lines up, so a figure read on its own had one
    # labelled axis and one bare one.
    quality_axis = ValueAxis.position(quality_title, qualities, height)

    present = [(name, symbol) for name, symbol in _CLASS_SHAPES if any(row[_CLASS_FIELD] == name for row in drawn)]
    shape: dict[str, Any] = {
        "field": _CLASS_FIELD,
        "type": "nominal",
        "scale": {"domain": [name for name, _ in present], "range": [symbol for _, symbol in present]},
        # NO legend, ever. Two attempts to make one honest both failed, and the
        # second failure is the informative one: a shape key's swatches are filled
        # from the MARK, so a dominated point drawn in the context neutral still
        # appeared in the key in the primary hue — telling the reader the greying
        # means nothing. `config.legend.symbolFillColor` does not win that, measured
        # on a hard reload. A colour ENCODING would fix the swatch and is refused by
        # the direct-label rule, correctly: a colour legend outside a facet is
        # precisely what that rule exists to stop.
        #
        # Which leaves the rule's own answer. Every mark here is already labelled
        # in place with its contestant's name, so identity never needed the key; what
        # the key carried was the SHAPE VOCABULARY, and one sentence says that
        # without asking the reader to hold a swatch in memory and walk back to it.
        "legend": None,
    }

    # One point layer per weight, each filtered out of the SAME table rather than
    # drawn from a dataset of its own: the shape scale carries an explicit domain,
    # so Vega-Lite shares it across the layers and the key still lists every class
    # the figure has — including one whose marks all sit in the recessive layer.
    #
    # Two layers rather than one mark with an opacity encoding, and that is the
    # whole retrofit: an opacity encoding puts the recession's VALUE in the spec,
    # which decides what "receded" looks like before either theme is known. A style
    # name defers that to the renderer, and a style is a property of a mark — so the
    # marks have to be partitioned to carry two of them.
    layers: list[dict[str, Any]] = [
        {
            "transform": [{"filter": {"field": _CLASS_FIELD, "oneOf": list(classes)}}],
            # `opacity: 1` opts OUT of a renderer default rather than stating an
            # appearance: Vega-Lite draws a point at 0.7 so overlapping points read
            # as density, and a frontier's marks are a handful of answers rather
            # than a cloud — at 0.7 every one of them draws in a blend of the
            # palette hue and whatever is behind it, which differs per theme. Full
            # ink is the same instruction in both, which is why it is not the
            # recession this arm compiles.
            #
            # The recession is the STYLE below, and it is load-bearing rather than
            # decorative: a dominated or disqualified point differs from an eligible
            # one in shape AND in ink, so the distinction survives a reader who
            # cannot resolve the shapes at this size. Nothing else carries it — the
            # encoding here is `x`/`y`/`shape` and a compiled spec may not carry a
            # colour at all — so deleting the style deletes the channel, and no test
            # goes red when it does.
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
            # this compiler emits. There is no legend that could carry it: a key of
            # model IDs is the reader holding a position in memory and walking back
            # to it, which is the thing direct labels exist to spare them.
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
    if payload.bar is not None:
        layers.insert(
            0,
            {
                # Under the marks: the bar is the standard they are read against, and
                # a rule drawn over a contestant sitting on it would cut the mark in half.
                "data": {"values": [{"bar": payload.bar}]},
                # Context ink, by name. The bar is furniture — the standard the
                # contestants are judged against, not one of them — so it recedes for
                # the same reason a zero rule does, and by the same mechanism: the
                # renderer decides how far, because how far is a property of the
                # surface. The dash is geometry and stays in the spec.
                "mark": {"type": "rule", "strokeDash": [4, 4], "style": CONTEXT_STYLE},
                "encoding": {"y": quality_axis.encoding("bar", upright_title=True)},
            },
        )

    # No crop footnote. Both axes are cropped and both say so already, in the only
    # way a reader of a point plot actually consults: the labelled ticks, which
    # begin where the axis begins. A dot's value is read off them rather than
    # compared against another dot's width, so the sentence restates the axis
    # instead of disclosing anything — and three figures each carrying it is what
    # teaches a reader to stop reading subtitles (the scope is in the zero-baseline
    # rule). So this figure passes no
    # subtitle at all, rather than an empty one it then has to filter back out.
    spec: dict[str, Any] = {
        "$schema": VEGA_LITE_SCHEMA,
        "title": _title_spec(f"{quality_title} against {cost_title}", width),
        "data": {"values": drawn},
        "layer": layers,
        "width": width,
        "height": height,
    }

    # What the figure cannot say for itself, one line per idea and never joined onto
    # the author's caption. The ORDER is the reading order: the key a reader needs to
    # read the marks at all, then who is out and why, then who is missing, then the
    # standard the rest are held to.
    disclosures: list[str] = []
    # The shape vocabulary, as a line, because the figure draws no key — see the
    # `legend` comment above. Only where a distinction exists: with one class present
    # there is nothing to tell apart and the line is furniture.
    if len(present) > 1:
        disclosures.append(", ".join(f"{symbol} = {name.lower()}" for name, symbol in present) + ".")
    # The glyph says a contestant is out; only this says why. A shape the reader
    # cannot account for is the disclosure half-made — and one line per REASON rather
    # than per contestant, since the reason is the disclosure and repeating it is how
    # this text once grew past being read.
    disclosures.extend(
        f"{_joined_names(labels)} {'is' if len(labels) == 1 else 'are'} disqualified: {reason}."
        for reason, labels in _disqualifications(payload.points)
    )
    if unpriced:
        disclosures.append(
            f"Not drawn — no production-replicating cost was recorded for {_joined_names(unpriced)}; "
            "the values below carry what was measured."
        )
    if payload.bar is not None:
        disclosures.append(f"The quality bar is {format_number(payload.bar)}.")

    columns: list[ChartColumn] = [
        {"key": "label", "header": "Contestant"},
        {"key": "cost", "header": cost_title},
        {"key": "quality", "header": quality_title},
    ]
    if latencies:
        columns.append({"key": "latency", "header": _axis_title("Latency", latency_unit)})
    columns.append({"key": _CLASS_FIELD, "header": "Contention"})
    return CompiledChart(
        spec=spec,
        columns=columns,
        rows=rows,
        unit="",
        disclosures=disclosures,
        title=f"{quality_title} against {cost_title}",
    )


__all__ = [
    "compile_frontier",
]
