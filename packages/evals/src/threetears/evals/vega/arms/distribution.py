"""The ``distribution`` arm — per-group spreads as one faceted panel, plus marginals.

A distribution of a quantity already on an axis is a MARGINAL of that axis, so it
is drawn in the row with the estimate it belongs to rather than in a panel of its
own. The one exception is a payload whose every group recorded only pre-binned
counts: nothing places on a value axis there, so the counts are the whole chart.

**The geometry values this module reads are its own.** ``geometry()`` and
``font_weights()`` are imported here from :mod:`threetears.evals.vega.palette`
directly, so they resolve in THIS module's namespace: a test that replaces
``geometry`` on :mod:`threetears.evals.vega.compiler` — which is where the shared
row-step and bar-thickness arithmetic reads it — does not reach the panel gap or
the rug threshold below. Patch this module to move those.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.vega.compiler import (
    ANCHOR_FIELD,
    DISPLAY_FIELD,
    KIND_FIELD,
    RISE_FIELD,
    SECONDARY_OPACITY,
    VALUE_TEXT_FIELD,
    VEGA_LITE_SCHEMA,
    MarkValue,
    Placement,
    ValueAxis,
    _Categories,
    _identity,
    _name_font_size,
    _number,
    _title_spec,
    _value_axis,
    centred_value_placements,
    value_label_mark,
)
from threetears.evals.analysis.viz.intent import ChartIntent
from threetears.evals.contracts.host import ChartFont
from threetears.evals.vega.palette import font_weights, geometry

#: How tall the cap at a known interval bound is drawn, in px.
_CAP_HEIGHT = 12

#: How tall a row's marginal draws, in px.
#:
#: Well inside :func:`~threetears.evals.vega.palette.geometry`'s ``row_step_marginal``,
#: because the rest of the row is the estimate the marginal sits under.
_MARGINAL_RISE = 26

#: Clear space between a row's marginal and the row below it, in px.
#:
#: Not decoration. The rows sit against each other with no gap, so a marginal drawn
#: hard against the row's floor is nearer the NEXT row's estimate than its own, and
#: reads as belonging to it.
_MARGINAL_FLOOR = 8

#: How far above the estimate band a value label is lifted, in px.
#:
#: The estimate label names the MEAN and so is anchored at it, which is the middle of
#: the interval rather than beside it — occupied by the span rule and by the mean's own
#: point mark. Lifting is what buys the space the old end-of-interval anchor was
#: reaching for, without the cost that anchor had: a number drawn at an x-position it
#: does not name.
#:
#: **What 16 clears, measured rather than asserted.** The label is drawn ``baseline:
#: middle``, so at the value step (15px) its lower edge sits 8.5px above the anchor. The
#: tallest thing in the estimate band is the interval cap, which rises 6px above centre
#: (half of :data:`_CAP_HEIGHT`; the mean's point mark is smaller, r≈4.7 at size 70). So
#: the label's underside clears the cap by 2.5px. Going higher is what the row cannot
#: afford: on a row carrying a marginal the estimate already sits above the marginal
#: band, and at this lift the label's top edge is ~1.5px inside the cell.
#:
#: Label-above-mark is the established rule for this figure rather than a choice made
#: here.
_ESTIMATE_LABEL_LIFT = 16


def compile_distribution(intent: ChartIntent, *, font: ChartFont | None = None) -> dict[str, Any]:
    """Draw per-group spreads as one faceted panel over one shared value axis.

    **One panel, one axis, one quantity.** This type used to draw two: an interval
    panel titled with the measure, and a count panel below it plotting the
    distribution of that same measure against bin names. A distribution of a quantity
    already on an axis is a MARGINAL of that axis, so it is drawn in the row with the
    estimate it belongs to.

    **The facet is not optional.** This type exists to show whether cohorts
    separate, and a pooled marginal is the one view that cannot answer that. A
    single-group payload is the pooled case already, and gets there by having one
    row rather than by taking a different path.

    Below :func:`~threetears.evals.vega.palette.geometry`'s ``rug_max_per_series``
    observations every one is drawn as its own tick, because at n=5 a histogram is
    a bin-width decision imposed on data small enough to show whole; above it the
    ticks would merge, so a binned band takes over at the same row height.

    Args:
        intent: The distribution's intent.
        font: The typeface the chart is laid out in; ``None`` for the packaged face.

    Returns:
        The Vega-Lite spec.
    """
    if intent.axis("value") is None:
        # Nothing places on a value axis, so there is no axis for a marginal to be
        # marginal TO: the counts are the chart rather than a duplicate of one.
        return _compile_binned_distribution(intent, font=font)
    ordering = _identity(intent).order
    categories = _Categories.of("label", ordering, font=font)
    estimates_by_label = {str(row["label"]): row for row in intent.data if "mean" in row}
    samples: dict[str, list[float]] = {label: [] for label in ordering}
    for row in intent.data:
        if "sample" in row:
            samples[str(row["label"])].append(_number(row["sample"]))
    marginal = any(samples.values())
    width, height = categories.plot_size(marginal=marginal)
    # Every facet cell is one row of the same panel, so the row step IS the cell.
    layout = _RowLayout(cell=max(1, height // len(ordering)), marginal=marginal)
    # A spread is a POSITION, not a length, so the axis crops to the data. Framed,
    # because with the marginal in the row the axis rule is the only bright line left
    # in the figure, and a rule that ends where the data ends states the extent of
    # what was measured without a caption having to.
    values = [_number(row[key]) for row in estimates_by_label.values() for key in ("low", "high", "mean")]
    values.extend(value for group in samples.values() for value in group)
    value_axis = ValueAxis.position("", values, width, framed=True)

    bins = _bin_count(samples)
    rows: list[dict[str, Any]] = _marginal_rows(samples, categories, value_axis, bins)
    estimates: list[MarkValue] = []
    for label in ordering:
        estimate = estimates_by_label.get(label)
        if estimate is None:
            continue
        drawn = categories.display[label]
        low, high, mean = _number(estimate["low"]), _number(estimate["high"]), _number(estimate["mean"])
        rows.extend(_span_rows(drawn, low, high))
        rows.append({DISPLAY_FIELD: drawn, KIND_FIELD: "mean", ANCHOR_FIELD: mean, "n": estimate["n"]})
        # Anchored at the MEAN it names, which is the values-on-the-mark rule applied
        # to a mark that is an interval rather than a bar. Anchoring it at the far end
        # put every number at an x-position it did not name.
        estimates.append(MarkValue(display=drawn, end=mean, text=format_number(mean)))
    for placement, marks in centred_value_placements(estimates, value_axis, font=font).items():
        rows.extend(
            {
                DISPLAY_FIELD: mark.display,
                KIND_FIELD: f"value-{placement}",
                ANCHOR_FIELD: mark.end,
                VALUE_TEXT_FIELD: mark.text,
            }
            for mark in marks
        )

    spec: dict[str, Any] = {
        "$schema": VEGA_LITE_SCHEMA,
        "title": _title_spec(intent.title, categories.figure_width(), intent.footnote, font=font),
        "data": {"values": rows},
        "facet": {
            "row": {
                "field": DISPLAY_FIELD,
                "type": "nominal",
                "sort": categories.drawn(),
                "header": categories.facet_header(),
            }
        },
        # The rows are rows of ONE panel, not panels beside each other, so they sit
        # against each other with no gap — the 32px that separates panels would make
        # each row read as a chart of its own.
        "spacing": 0,
        # One shared grid across the cells, which is what puts every row's header on a
        # common left edge. Without it Vega sizes each cell's header column to its OWN
        # text, so two cohorts whose names differ in length start at two different x
        # positions and the column reads as though one were indented.
        "align": "all",
        "spec": {
            "width": width,
            "height": layout.cell,
            "layer": _distribution_layers(
                value_axis, layout, width / bins, intent.title, {row[KIND_FIELD] for row in rows}
            ),
        },
    }
    if intent.intervals:
        # The spec's own statement of what its intervals are and what they span — the
        # spec gate reads it: a spec that draws a span and says nothing about what it
        # varies over is refused.
        spec["description"] = intent.intervals
    return spec


def _compile_binned_distribution(intent: ChartIntent, *, font: ChartFont | None = None) -> dict[str, Any]:
    """Draw a distribution whose every group recorded only pre-binned counts.

    Bin ranges are label strings — the payload records them as names, not as edges —
    so they are ordered as given and never placed on a value axis, which would mean
    inventing coordinates the payload never wrote down.

    Args:
        intent: The distribution's intent, whose data is the counts per group and bin.
        font: The typeface the chart is laid out in; ``None`` for the packaged face.

    Returns:
        The Vega-Lite spec.
    """
    categories = _Categories.of("label", _identity(intent).order, font=font)
    rows = intent.data
    width, _height = categories.plot_size()
    # One domain across every cell: the cells are read against each other, so a bin
    # drawn taller in one panel than another has to mean a larger count rather than
    # a smaller neighbour.
    count_axis = ValueAxis.magnitude("", [_number(row["count"]) for row in rows], width, tick_count=3)
    panel = {
        "data": {"values": categories.labelled(rows)},
        # The quantity this panel counts, named ABOVE it rather than beside it. Vega
        # draws a y-axis title rotated a quarter turn, and no chart here turns its
        # words; the axis then states nothing rather than the same word sideways.
        "title": _value_axis(intent, "count").quantity,
        "facet": {
            "row": {
                "field": DISPLAY_FIELD,
                "type": "nominal",
                "sort": categories.drawn(),
                "header": categories.facet_header(),
            }
        },
        "spec": {
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "x": {
                    "field": "range",
                    "type": "nominal",
                    "sort": None,
                    "axis": {
                        "title": None,
                        "labelAngle": 0,
                        # Bin ranges are names, so they draw at the name step — this
                        # axis's labels are the one place on the figure where the
                        # category step lands on x rather than y.
                        "labelFontSize": _name_font_size(),
                        "labelFontWeight": font_weights()["label"],
                        # Counts rise from a shared baseline and the bins sit on it,
                        # so both the grid and the domain line are drawn by the bars.
                        "grid": False,
                        "domain": False,
                    },
                },
                # A count is a magnitude — the bar's height IS the number — and it is
                # titleless because the panel's own heading names it.
                "y": count_axis.encoding("count"),
            },
            "width": width,
            "height": 60,
        },
    }
    # The panel names what it counts and the figure names what was measured — two
    # different statements, so the figure is a one-panel concat rather than a single
    # view whose one `title` slot would carry only one of them.
    return {
        "$schema": VEGA_LITE_SCHEMA,
        "title": _title_spec(intent.title, categories.figure_width(), font=font),
        "vconcat": [panel],
        "spacing": geometry()["panel_gap"],
    }


@dataclass(frozen=True)
class _RowLayout:
    """Where one facet row's two bands sit inside its cell, in px.

    A row carrying a marginal is read in two registers: the estimate on top, the
    shape it was drawn from underneath. Both are positioned by ``yOffset``, which
    Vega measures from the cell's CENTRE, so every number here is that offset rather
    than a distance from the top — computed once because getting the two out of step
    puts a cohort's shape nearer its neighbour's estimate than its own.
    """

    cell: int
    """The row's full height."""

    marginal: bool
    """Whether any row of this figure draws a marginal.

    Per figure rather than per row, because a facet's cells are one height: a row
    whose cohort recorded no raw values still lives in a cell sized for the ones
    that did, and centring its estimate would put it out of line with theirs.
    """

    @property
    def _band(self) -> float:
        """How much of the cell the marginal and its floor gap take."""
        return _MARGINAL_RISE + _MARGINAL_FLOOR if self.marginal else 0.0

    @property
    def estimate(self) -> float:
        """The estimate's offset from the cell centre — centred in what is left above."""
        return (self.cell - self._band) / 2 - self.cell / 2

    @property
    def marginal_centre(self) -> float:
        """The marginal band's offset from the cell centre."""
        return self.cell / 2 - _MARGINAL_FLOOR - _MARGINAL_RISE / 2

    @property
    def floor(self) -> float:
        """How far the marginal's own zero is lifted off the cell's floor."""
        return _MARGINAL_FLOOR


def _distribution_layers(
    axis: ValueAxis, layout: _RowLayout, band: float, quantity: str, kinds: Iterable[str]
) -> list[dict[str, Any]]:
    """The marks one facet cell draws, back to front, each filtered to its own kind.

    Every layer reads the SAME faceted table, because a layer carrying ``data`` of
    its own is not faceted — Vega-Lite hands the partition to the cell, and a layer
    that brought its own rows would draw all of them in every row of the figure.

    Order is a contract here, unlike elsewhere: the marginal is the row's ground and
    the estimate is read against it, so the estimate is drawn last and over.

    **A layer with no rows is not emitted at all.** An empty layer is invisible in
    the picture and fully present to anything reading the spec, which is not a
    harmless difference: the gate that asks a frame drawing a span to say what the
    span varies over reads the ENCODING, so a figure with no intervals in it would
    be refused for failing to describe the intervals it does not draw.

    Args:
        axis: The shared value axis every mark is placed on.
        layout: Where the row's two bands sit inside its cell.
        band: One bin's width in px, which the mark states because a bar on a
            CONTINUOUS axis is otherwise drawn at Vega's own fixed width and a
            histogram becomes a row of unconnected sticks.
        quantity: What the axis measures, for the tooltips — the axis itself states
            nothing, so a tooltip that named no quantity would name none anywhere.
        kinds: Which kinds of row the figure actually carries.

    Returns:
        The cell's layers, in drawn order.
    """
    drawn = set(kinds)
    return [layer for layer in _every_distribution_layer(axis, layout, band, quantity) if _kind_of(layer) in drawn]


def _kind_of(layer: dict[str, Any]) -> str:
    """Which kind of row a cell layer draws, read back off its own filter."""
    return str(layer["transform"][0]["filter"]["equal"])


def _every_distribution_layer(axis: ValueAxis, layout: _RowLayout, band: float, quantity: str) -> list[dict[str, Any]]:
    """Every layer a distribution cell can draw, whether or not this one does.

    Args:
        axis: The shared value axis every mark is placed on.
        layout: Where the row's two bands sit inside its cell.
        band: One bin's width in px.
        quantity: What the axis measures, for the tooltips.

    Returns:
        Every candidate layer, in drawn order.
    """
    rise = {
        "field": RISE_FIELD,
        "type": "quantitative",
        "scale": {"domain": [0, layout.cell], "zero": True, "nice": False},
        # Suppressed rather than absent, and the difference is the whole rule: an
        # absent axis is Vega-Lite filling the title with the field name, while
        # `title: None` is the frame's heading having already named the quantity.
        # The height itself is deliberately unlabelled — a marginal states shape,
        # and the counts behind it are exact in the values table.
        "axis": {"title": None, "labels": False, "ticks": False, "domain": False, "grid": False},
    }
    return [
        {
            "transform": _of_kind("bin"),
            "mark": {
                "type": "bar",
                "opacity": SECONDARY_OPACITY,
                # `width`, not `continuousBandSize`: the latter is the documented
                # property for a bar on a continuous axis and vl-convert ignores it
                # outright — measured, at the pinned version — leaving every bin at
                # Vega's 5px default, which is a row of sticks rather than a
                # histogram. Both renderers honour `width`.
                "width": band,
                "yOffset": -layout.floor,
                "tooltip": True,
            },
            "encoding": {"x": axis.encoding(ANCHOR_FIELD), "y": rise},
        },
        {
            "transform": _of_kind("rug"),
            # One tick per observation, occupying the same band of the row a binned
            # marginal would, so the two forms of the same statement sit in one place.
            "mark": {"type": "tick", "thickness": 1, "height": _MARGINAL_RISE, "yOffset": layout.marginal_centre},
            "encoding": {
                "x": axis.encoding(ANCHOR_FIELD),
                "tooltip": [{"field": ANCHOR_FIELD, "type": "quantitative", "title": quantity}],
            },
        },
        {
            "transform": _of_kind("span"),
            "mark": {"type": "rule", "size": 3, "yOffset": layout.estimate},
            "encoding": {
                "x": axis.encoding("low"),
                "x2": {"field": "high"},
            },
        },
        {
            "transform": _of_kind("cap"),
            "mark": {"type": "tick", "thickness": 2, "height": _CAP_HEIGHT, "yOffset": layout.estimate},
            "encoding": {"x": axis.encoding(ANCHOR_FIELD)},
        },
        {
            "transform": _of_kind("mean"),
            "mark": {"type": "point", "filled": True, "size": 70, "yOffset": layout.estimate, "tooltip": True},
            "encoding": {"x": axis.encoding(ANCHOR_FIELD)},
        },
        *(
            {
                "transform": _of_kind(f"value-{placement}"),
                # Lifted clear of the estimate rather than set beside it, because the
                # anchor is now the mean — inside the interval, where the span rule and
                # the mean's point already are.
                #
                # A centred label takes no `dx`: the offset exists to hold text off the
                # mark it sits BESIDE, and a centred one sits above, so applying it
                # would shift the number off the value it names by exactly the amount
                # this figure was fixed for.
                # Built by the SHARED helper, not restated here. A helper the arm beside
                # it bypasses is a helper that stops being the answer — and the thing it
                # decides is which labels take the knockout ink, so a second copy is a
                # second contrast decision. `filled=False` because this arm's label is
                # lifted clear of the mark rather than laid on it: an interval rule and a
                # point have no fill under a label above them.
                #
                "mark": value_label_mark(
                    Placement(align=placement, inside=False),
                    yOffset=layout.estimate - _ESTIMATE_LABEL_LIFT,
                ),
                "encoding": {
                    "x": axis.encoding(ANCHOR_FIELD),
                    "text": {"field": VALUE_TEXT_FIELD, "type": "nominal"},
                },
            }
            for placement in ("center", "left", "right")
        ),
    ]


def _of_kind(kind: str) -> list[dict[str, Any]]:
    """The transform selecting one layer's rows out of the shared faceted table."""
    return [{"filter": {"field": KIND_FIELD, "equal": kind}}]


def _span_rows(display: str, low: float, high: float) -> list[dict[str, Any]]:
    """One interval's rows: a solid span, and a hard cap at each bound.

    Every interval states the coverage level it was computed at
    (:class:`~threetears.evals.analysis.viz.payloads.ConfidenceInterval` requires it), so its
    ends are known bounds and are drawn as caps.

    Args:
        display: The drawn category name, which places the rows in their facet.
        low: The interval's lower bound, in the axis's units.
        high: Its upper bound.

    Returns:
        The span row and its two cap rows.
    """
    return [
        {DISPLAY_FIELD: display, KIND_FIELD: "span", "low": low, "high": high},
        *({DISPLAY_FIELD: display, KIND_FIELD: "cap", ANCHOR_FIELD: bound} for bound in (low, high)),
    ]


def _marginal_rows(
    samples: dict[str, list[float]], categories: _Categories, axis: ValueAxis, bins: int
) -> list[dict[str, Any]]:
    """Every row's marginal: each observation drawn, or a band binned from them.

    Below ``rug_max_per_series`` observations each one is its own tick — that costs
    nothing, states the sample size visually, and cannot be distorted by a bucket
    boundary nobody chose. Past it the ticks merge into a smear, so a binned band
    takes over at the same row height. The decision is per cohort.

    **Computed for the whole figure rather than per group, because both halves of a
    binned marginal are shared.** The bins are laid out on the shared domain, and
    their heights are normalised against the tallest bin ANYWHERE, so a
    five-observation cohort cannot draw as tall as a five-hundred-observation one.

    Args:
        samples: Each group's observations, already in the drawn unit, in drawn order.
        categories: The figure's resolved labels, which place each row in its facet.
        axis: The shared value axis, which decides the bins' extent.
        bins: How many bins the figure divides that extent into.

    Returns:
        The marginal rows for every group that recorded raw values.
    """
    edges = _bin_edges(axis, bins)
    rug: list[dict[str, Any]] = []
    binned: list[dict[str, Any]] = []
    for label, values in samples.items():
        display = categories.display[label]
        if not values:
            continue
        if len(values) <= geometry()["rug_max_per_series"]:
            rug.extend({DISPLAY_FIELD: display, KIND_FIELD: "rug", ANCHOR_FIELD: value} for value in values)
            continue
        counts = _bin_counts(values, axis, bins)
        binned.extend(
            {
                DISPLAY_FIELD: display,
                KIND_FIELD: "bin",
                ANCHOR_FIELD: (edges[index] + edges[index + 1]) / 2,
                "count": count,
            }
            for index, count in enumerate(counts)
            if count
        )
    tallest = max((row["count"] for row in binned), default=1)
    return rug + [row | {RISE_FIELD: row["count"] / tallest * _MARGINAL_RISE} for row in binned]


def _bin_counts(samples: Sequence[float], axis: ValueAxis, bins: int) -> list[int]:
    """How many of ``samples`` fall in each of ``bins`` equal spans of the axis."""
    counts = [0] * bins
    span = axis.high - axis.low
    for value in samples:
        # The last bin owns its upper edge; every other bin is half-open. Without
        # that the single largest observation falls outside every bin and the
        # marginal quietly draws one fewer than it was given.
        counts[min(int((value - axis.low) / span * bins), bins - 1)] += 1
    return counts


def _bin_count(samples: dict[str, list[float]]) -> int:
    """How many bins every binned row of this figure is divided into.

    The square root of the largest cohort's observation count — the conventional
    default — taken over the FIGURE rather than per group, so every binned row is
    drawn to one ruler.

    Args:
        samples: Each group's observations; a group with none contributes nothing.

    Returns:
        The bin count, at least one.
    """
    largest = max((len(values) for values in samples.values()), default=0)
    return max(1, round(math.sqrt(largest)))


def _bin_edges(axis: ValueAxis, bins: int) -> list[float]:
    """Equal-width bin edges spanning the axis, shared by every row that bins."""
    step = (axis.high - axis.low) / bins
    return [axis.low + index * step for index in range(bins + 1)]


__all__ = [
    "compile_distribution",
]
