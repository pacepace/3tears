"""The ``distribution`` arm — per-group spreads as one faceted panel, plus marginals.

A distribution of a quantity already on an axis is a MARGINAL of that axis, so it
is drawn in the row with the estimate it belongs to rather than in a panel of its
own. The one exception is a payload whose every group recorded only pre-binned
counts: nothing places on a value axis there, so the counts are the whole chart.

**The geometry values this module reads are its own.** ``geometry()`` and
``font_weights()`` are imported here from :mod:`threetears.evals.analysis.viz.palette`
directly, so they resolve in THIS module's namespace: a test that replaces
``geometry`` on :mod:`threetears.evals.analysis.viz.compiler` — which is where the shared
row-step and bar-thickness arithmetic reads it — does not reach the panel gap or
the rug threshold below. Patch this module to move those.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.viz.compiler import (
    ANCHOR_FIELD,
    DISPLAY_FIELD,
    KIND_FIELD,
    RISE_FIELD,
    SECONDARY_OPACITY,
    VALUE_TEXT_FIELD,
    VEGA_LITE_SCHEMA,
    ChartColumn,
    CompiledChart,
    _axis_title,
    _Categories,
    _interval_caption,
    _interval_disclosures,
    MarkValue,
    _name_font_size,
    Placement,
    _title_spec,
    ValueAxis,
    display_scale,
    value_label_mark,
)
from threetears.evals.analysis.viz.palette import font_sizes, font_weights, geometry
from threetears.evals.analysis.viz.payloads import DistributionGroup, DistributionPayload
from threetears.evals.analysis.viz.text_metrics import text_width

#: What a panel of pre-binned counts is measuring, where those counts are the whole
#: chart rather than one row's marginal.
#:
#: A heading over the panel rather than a y-axis title, because Vega draws a y title
#: rotated a quarter turn and no chart here rotates text. The axis then states
#: nothing rather than the same word sideways.
_COUNT_TITLE = "observations"

#: How tall the cap at a known interval bound is drawn, in px.
_CAP_HEIGHT = 12

#: How tall a row's marginal draws, in px.
#:
#: Well inside :func:`~threetears.evals.analysis.viz.palette.geometry`'s ``row_step_marginal``,
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


def compile_distribution(payload: DistributionPayload) -> CompiledChart:
    """Compile per-group spreads into one faceted panel over one shared value axis.

    **One panel, one axis, one quantity.** This type used to draw two: an interval
    panel titled with the measure, and a count panel below it plotting the
    distribution of that same measure against bin names. Three labels for one
    variable on two different rulers, stacked so the vertical alignment between
    them looked meaningful while meaning nothing. A distribution of a quantity
    already on an axis is a MARGINAL of that axis, so it is drawn in the row with
    the estimate it belongs to.

    **The facet is not optional.** This type exists to show whether cohorts
    separate, and a pooled marginal is the one view that cannot answer that. A
    single-group payload is the pooled case already, and gets there by having one
    row rather than by taking a different path.

    Every row is drawn against the same domain, which is the point: spreads that do
    not share a ruler cannot be compared, and drawing them in one frame while
    scaling each to its own values invites exactly that comparison. The counts a
    binned row draws are shared the same way and for the same reason.

    Shape is drawn from values and never inferred from two endpoints. Below
    :func:`~threetears.evals.analysis.viz.palette.geometry`'s ``rug_max_per_series``
    observations every one is drawn as its own tick, because at n=5 a histogram is
    a bin-width decision imposed on data small enough to show whole; above it the
    ticks would merge, so a binned band takes over at the same row height. A group
    reporting only an interval gets the interval and an explicit statement that its
    shape is unknown — not a fabricated bell.

    **Pre-binned buckets do not reach the axis, and the figure says so.** Their
    bins are *label strings*, so placing them would mean inventing numeric edges
    the payload never gave — and drawing them against bin names instead is the
    second ruler this restructure exists to remove. The counts stay legible in the
    values table, where they are exact, and the footnote names the cohorts whose
    recorded shape could not be placed. Same principle as ``not placeable``: draw
    the absence, do not leave it to be inferred.

    Args:
        payload: The validated per-group distribution payload.

    Returns:
        The compiled chart.
    """
    values = [value for group in payload.groups for value in _group_values(group)]
    scale, unit = display_scale(values, payload.unit)
    # The measure names itself ONCE, and it is here rather than on the axis: a
    # heading over the panel and a title beside the ticks are the same words twice,
    # and the axis is the one of the two that cannot say them without Vega turning
    # them a quarter turn. So the unit rides into the heading as a parenthetical and
    # the axis states nothing.
    title = _axis_title(payload.x_label or "Distribution", unit)

    ordering = [group.label for group in payload.groups]
    categories = _Categories.of("label", ordering)
    if not any(group.ci is not None or group.samples for group in payload.groups):
        # Nothing places on the value axis, so there is no axis for a marginal to be
        # marginal TO — and the rule against a second panel of one quantity has
        # nothing to bite on, because there is no first panel. The counts are the
        # chart rather than a duplicate of one, so they keep the shape they had.
        return _compile_binned_distribution(payload, categories, scale, unit)
    marginal = any(group.samples for group in payload.groups)
    width, height = categories.plot_size(marginal=marginal)
    # Every facet cell is one row of the same panel, so the row step IS the cell.
    layout = _RowLayout(cell=max(1, height // len(ordering)), marginal=marginal)
    # A spread is a POSITION, not a length, so the axis crops to the data — two
    # spreads six seconds apart on a 0-60s axis are the same picture — and says so.
    # Framed, because with the marginal in the row the axis rule is the only bright
    # line left in the figure, and a rule that ends where the data ends states the
    # extent of what was measured without a caption having to.
    value_axis = ValueAxis.position("", [value * scale for value in values], width, framed=True)

    bins = _bin_count(payload.groups)
    rows: list[dict[str, Any]] = _marginal_rows(payload.groups, categories, scale, value_axis, bins)
    estimates: list[MarkValue] = []
    for group in payload.groups:
        drawn = categories.display[group.label]
        if group.ci is None:
            continue
        low, high, mean = group.ci.low * scale, group.ci.high * scale, group.ci.mean * scale
        rows.extend(_span_rows(drawn, low, high))
        rows.append({DISPLAY_FIELD: drawn, KIND_FIELD: "mean", ANCHOR_FIELD: mean, "n": group.n})
        # Anchored at the MEAN it names, which is the values-on-the-mark rule applied
        # to a mark that is an interval rather than a bar. It was anchored at `high`
        # instead — the interval's far end, on the reasoning that the end is the only
        # place beside the mark not already occupied by it — and that put every number
        # at an x-position it did not name: a group with mean 12.5 over a 10.0-15.0
        # interval printed "12.5" at x≈15, and the reader maps it to the wrong value.
        # Vertical space is what the end-of-mark reasoning was actually short of, and
        # `_ESTIMATE_LABEL_LIFT` is where that comes from now.
        estimates.append(MarkValue(display=drawn, end=mean, text=format_number(mean)))
    for placement, marks in _estimate_label_placements(estimates, value_axis).items():
        rows.extend(
            {
                DISPLAY_FIELD: mark.display,
                KIND_FIELD: f"value-{placement}",
                ANCHOR_FIELD: mark.end,
                VALUE_TEXT_FIELD: mark.text,
            }
            for mark in marks
        )

    unplaceable = [group.label for group in payload.groups if group.buckets and not group.samples]
    footnote = _unplaceable_shape_footnote(unplaceable)
    spec: dict[str, Any] = {
        "$schema": VEGA_LITE_SCHEMA,
        "title": _title_spec(title, categories.figure_width(), footnote),
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
        # each row read as a chart of its own, which is the reading this restructure
        # removes.
        "spacing": 0,
        # One shared grid across the cells, which is what puts every row's header on a
        # common left edge. Without it Vega sizes each cell's header column to its OWN
        # text, so two cohorts whose names differ in length start at two different x
        # positions and the column reads as though one were indented — a long name
        # such as `model-alpha-large` sits hard left of a short one such as `m-b`. Neither label
        # ALIGNMENT fixes that, which is the trap: anchoring the text moves it within a
        # column that is itself the wrong width. The grid is the thing to fix.
        "align": "all",
        "spec": {
            "width": width,
            "height": layout.cell,
            "layer": _distribution_layers(value_axis, layout, width / bins, title, {row[KIND_FIELD] for row in rows}),
        },
    }
    intervals = [group.ci for group in payload.groups]
    qualification = _interval_caption(intervals)
    if qualification:
        # The spec's own statement of what its intervals are and what they span.
        # Carried here rather than only in the disclosures because the policy gate
        # reads it: a spec that draws a span and says nothing about what it varies
        # over is refused.
        spec["description"] = qualification

    columns: list[ChartColumn] = [{"key": "label", "header": "Group"}]
    if any(group.ci is not None for group in payload.groups):
        columns.extend(
            [
                {"key": "mean", "header": _axis_title("Mean", unit)},
                {"key": "low", "header": _axis_title("Low", unit)},
                {"key": "high", "header": _axis_title("High", unit)},
            ]
        )
    # An absent `n` is stated by absence, never as a column of em dashes — the same
    # rule the breakdown case states, and it has to hold here or it holds by accident.
    if any(group.n is not None for group in payload.groups):
        columns.append({"key": "n", "header": "n"})
    columns.append({"key": "shape", "header": "Shape"})
    return CompiledChart(
        spec=spec,
        columns=columns,
        rows=[_distribution_row(group, scale) for group in payload.groups],
        unit=unit,
        disclosures=_interval_disclosures(intervals),
        title=title,
    )


def _compile_binned_distribution(
    payload: DistributionPayload, categories: _Categories, scale: float, unit: str
) -> CompiledChart:
    """Compile a distribution whose every group recorded only pre-binned counts.

    The one shape of this type that is NOT a marginal. A marginal is a distribution
    of a quantity already on an axis, drawn in the row that quantity's estimate sits
    in; here no estimate and no raw value places anything on a value axis, so the
    counts are the whole chart and there is no second ruler for them to be a second
    ruler to.

    Bin ranges are label strings — the payload records them as names, not as edges —
    so they are ordered as given and never placed on a value axis, which would mean
    inventing coordinates the payload never wrote down.

    Args:
        payload: The distribution, every group of which carries only buckets.
        categories: The figure's resolved labels and placement.
        scale: The unit restatement factor, carried for the values table.
        unit: The restated unit.

    Returns:
        The compiled chart.
    """
    rows: list[dict[str, Any]] = [
        {"label": group.label, "range": bucket.range, "count": bucket.count}
        for group in payload.groups
        for bucket in (group.buckets or [])
    ]
    width, _height = categories.plot_size()
    # One domain across every cell: the cells are read against each other, so a bin
    # drawn taller in one panel than another has to mean a larger count rather than
    # a smaller neighbour.
    count_axis = ValueAxis.magnitude("", [row["count"] for row in rows], width, tick_count=3)
    panel = {
        "data": {"values": categories.labelled(rows)},
        # The quantity this panel counts, named ABOVE it rather than beside it. Vega
        # draws a y-axis title rotated a quarter turn, and no chart here turns its
        # words; the axis then states nothing rather than the same word sideways.
        "title": _COUNT_TITLE,
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
                # titleless because the panel's own heading names it, which is what
                # keeps the words upright.
                "y": count_axis.encoding("count"),
            },
            "width": width,
            "height": 60,
        },
    }
    # The panel names what it counts and the figure names what was measured — two
    # different statements, and a single view flattened into the figure would have
    # one `title` slot for them, so the panel's heading would silently overwrite the
    # figure's. The measure carries no unit here: what this chart draws is a count,
    # and the measure's own unit is inside the bin names the payload wrote.
    title = payload.x_label or "Distribution"
    spec: dict[str, Any] = {
        "$schema": VEGA_LITE_SCHEMA,
        "title": _title_spec(title, categories.figure_width()),
        "vconcat": [panel],
        "spacing": geometry()["panel_gap"],
    }
    columns: list[ChartColumn] = [{"key": "label", "header": "Group"}]
    if any(group.n is not None for group in payload.groups):
        columns.append({"key": "n", "header": "n"})
    columns.append({"key": "shape", "header": "Shape"})
    return CompiledChart(
        spec=spec,
        columns=columns,
        rows=[_distribution_row(group, scale) for group in payload.groups],
        unit=unit,
        title=title,
    )


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


def _estimate_label_placements(values: Sequence[MarkValue], axis: ValueAxis) -> dict[str, list[MarkValue]]:
    """Group each estimate label by the alignment that keeps it inside the plot.

    **A different question from :func:`~threetears.evals.analysis.viz.compiler._aligned_values`,
    which is why this does not call it.** That one asks which side of a mark's END has
    room, because a bar's label goes beside the bar. This label is anchored at an
    INTERIOR point — the mean — and is lifted clear of the mark rather than set beside
    it, so the only question left is whether the text box fits the plot on both sides.
    Centred where it does; otherwise pushed to whichever side the text has to grow
    into. Both are still bounded by ``value_label_max_marks``, which is a statement
    about how many numbers a figure can carry rather than about where they sit.

    Without this a label near the domain's edge overran the plot: a mean of 14.9 on an
    axis ending at 15.0 printed past the right edge, overlapping its own interval cap
    and making Vega grow the frame — the figure leaving its column for a label.

    Args:
        values: One entry per group carrying an estimate, anchored at its mean.
        axis: The shared value axis.

    Returns:
        ``center``/``left``/``right`` → the marks taking it, or an empty mapping
        where no value is written at all.
    """
    sizes = geometry()
    if not values or len(values) > sizes["value_label_max_marks"]:
        return {}
    size = font_sizes()["value"]
    placed: dict[str, list[MarkValue]] = {}
    for mark in values:
        half = text_width(mark.text, size) / 2
        from_left = axis.offset(mark.end)
        from_right = axis.plot_span - from_left
        if from_left >= half and from_right >= half:
            placement = "center"
        elif from_right < half:
            # Not enough plot to the right, so the text grows LEFT from the anchor —
            # which is what Vega calls a right alignment.
            placement = "right"
        else:
            placement = "left"
        placed.setdefault(placement, []).append(mark)
    return placed


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
    groups: Sequence[DistributionGroup], categories: _Categories, scale: float, axis: ValueAxis, bins: int
) -> list[dict[str, Any]]:
    """Every row's marginal: each observation drawn, or a band binned from them.

    Below ``rug_max_per_series`` observations each one is its own tick — that costs
    nothing, states the sample size visually, and cannot be distorted by a bucket
    boundary nobody chose. Past it the ticks merge into a smear, so a binned band
    takes over at the same row height. The decision is per cohort: one figure may
    carry both, and each is answering the same question at the size it can.

    **Computed for the whole figure rather than per group, because both halves of a
    binned marginal are shared.** The bins are laid out on the shared domain, so a
    bin in one row covers the same values as the bin above it; and their heights are
    normalised against the tallest bin ANYWHERE, so a five-observation cohort cannot
    draw as tall as a five-hundred-observation one. Per-group normalisation is the
    same defect as a per-group domain, one axis over.

    Args:
        groups: Every group, whose ``samples`` are what a marginal is drawn from.
        categories: The figure's resolved labels, which place each row in its facet.
        scale: The unit restatement factor every plotted value takes.
        axis: The shared value axis, which decides the bins' extent.
        bins: How many bins the figure divides that extent into.

    Returns:
        The marginal rows for every group that recorded raw values.
    """
    edges = _bin_edges(axis, bins)
    rug: list[dict[str, Any]] = []
    binned: list[dict[str, Any]] = []
    for group in groups:
        samples = [value * scale for value in (group.samples or [])]
        display = categories.display[group.label]
        if not samples:
            continue
        if len(samples) <= geometry()["rug_max_per_series"]:
            rug.extend({DISPLAY_FIELD: display, KIND_FIELD: "rug", ANCHOR_FIELD: value} for value in samples)
            continue
        counts = _bin_counts(samples, axis, bins)
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


def _bin_count(groups: Sequence[DistributionGroup]) -> int:
    """How many bins every binned row of this figure is divided into.

    The square root of the largest cohort's observation count — the conventional
    default — taken over the FIGURE rather than per group. Per group it would give
    different widths in different rows, and the reader would be comparing two
    histograms drawn to two rulers, which is the defect one axis over.

    Args:
        groups: Every group; the ones with no samples contribute no observations.

    Returns:
        The bin count, at least one.
    """
    largest = max((len(group.samples or []) for group in groups), default=0)
    return max(1, round(math.sqrt(largest)))


def _bin_edges(axis: ValueAxis, bins: int) -> list[float]:
    """Equal-width bin edges spanning the axis, shared by every row that bins."""
    step = (axis.high - axis.low) / bins
    return [axis.low + index * step for index in range(bins + 1)]


def _unplaceable_shape_footnote(labels: Sequence[str]) -> str:
    """State the cohorts whose recorded shape could not be drawn on the axis.

    A payload may record a group's shape as pre-binned counts whose bins are label
    strings. Those cannot be placed on a value axis without inventing the numeric
    edges the payload never gave, and drawing them against bin NAMES instead is the
    second ruler this chart exists without. So the shape is absent from the picture,
    and the picture says which cohorts it is absent for — the counts themselves stay
    exact in the values table.

    Args:
        labels: The groups whose only recorded shape is pre-binned.

    Returns:
        One sentence, or ``""`` where every group's shape is drawable.
    """
    if not labels:
        return ""
    # "Shape" is what this module calls the distribution internally, and it means
    # nothing to a reader looking at the figure — it could equally be the mark, the
    # interval, or the whole row. Name what is missing by what they would have SEEN:
    # the individual runs, drawn as ticks under the interval on every other cohort.
    return (
        f"Individual runs not drawn for {', '.join(labels)}: recorded as pre-binned counts, whose bins name "
        "ranges rather than locating them. The counts are in the values table."
    )


def _distribution_row(group: DistributionGroup, scale: float) -> dict[str, Any]:
    """One group's line in the values table, including what its shape rests on.

    Pre-binned counts are written out here rather than summarised, and that is not
    a nicety: their bins name ranges instead of locating them, so they cannot be
    drawn on the value axis at all, and this table is the whole of what the reader
    gets of them. "3 bins" would state that a shape exists and withhold it.

    Args:
        group: One cohort of the distribution.
        scale: The unit restatement factor the chart's values took.

    Returns:
        The values-table row.
    """
    observed = len(group.samples or [])
    buckets = group.buckets or []
    if observed:
        shape = f"{observed} samples"
    elif buckets:
        # Bucket ranges are LABEL STRINGS the payload recorded, so a restated row can sit
        # a `45–50k` bin beside a `Mean (s)` of 52 — one row against two units. Known and
        # filed rather than patched: the duration ladder cannot restate prose, and the
        # honest fix needs numeric bin edges the payload does not carry, which is the same
        # gap this type's docstring already records for placing them on the axis.
        shape = "; ".join(f"{bucket.range}: {bucket.count}" for bucket in buckets)
    else:
        # Said out loud. An interval alone gives no shape, and a reader shown only
        # a band will supply a bell that the data never stated.
        shape = "unknown — interval only"
    row: dict[str, Any] = {"label": group.label, "n": group.n, "shape": shape}
    if group.ci is not None:
        row |= {"mean": group.ci.mean * scale, "low": group.ci.low * scale, "high": group.ci.high * scale}
    return row


def _group_values(group: DistributionGroup) -> list[float]:
    """Every numeric value a distribution group contributes to the shared domain."""
    values = list(group.samples or [])
    if group.ci is not None:
        values.extend([group.ci.low, group.ci.high, group.ci.mean])
    return values


__all__ = [
    "compile_distribution",
]
