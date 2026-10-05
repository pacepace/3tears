"""The ``delta_table`` arm — an A-vs-B comparison as a dumbbell on relative change.

The rows are different metrics in different units, so the one thing they can share
is a unitless axis. Everything here follows from that: what is plotted is relative
change, what is tabulated is each row's own unit, and a row that cannot produce a
relative change is named rather than dropped.
"""

from __future__ import annotations

import math
from typing import Any

from threetears.evals.analysis.reporting import format_significance
from threetears.evals.analysis.viz.compiler import (
    VEGA_LITE_SCHEMA,
    ChartColumn,
    CompiledChart,
    _bar_mark,
    _Categories,
    _composed,
    _layers,
    MarkValue,
    _render_cell,
    _signed_with_unit,
    _title_spec,
    value_label_layers,
    ValueAxis,
    _with_unit,
    _zero_rule,
    display_scale,
    point_radius,
)
from threetears.evals.analysis.viz.payloads import DeltaRow, DeltaTablePayload, PayloadError

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


def compile_delta_table(payload: DeltaTablePayload) -> CompiledChart:
    """Compile an A-vs-B comparison as a dumbbell on one shared relative-change axis.

    **The axis carries relative change, not the deltas themselves.** The rows are
    different metrics in different units, so their absolute deltas cannot share a
    scale — and a magnitude encoding whose rows each use their own scale is not an
    encoding at all: both markers land at the same two positions on every row, so
    a 3% difference draws exactly like a tenfold one. Relative change is the
    unitless quantity that makes bar length mean the same thing on every row, and
    the axis is floored so a set of small changes still renders small.

    A is pinned at zero because it is the baseline the comparison runs *from*, so
    the bar's length is the size of the change and its side is the direction.

    Args:
        payload: The validated comparison payload.

    Returns:
        The compiled chart.
    """
    a_label = payload.a_label or "A"
    b_label = payload.b_label or "B"
    plotted: list[dict[str, Any]] = []
    undrawn: list[str] = []
    for row in payload.rows:
        change = _relative_change(row.a, row.b) if row.data_type == "numeric" else None
        if change is None:
            undrawn.append(row.metric)
            continue
        entry: dict[str, Any] = {"metric": row.metric, "change": change, "effect": _effect_read(row)}
        if row.p is not None:
            entry["p"] = row.p
        if row.d_z is not None:
            entry["d_z"] = row.d_z
        plotted.append(entry)

    reach = max((abs(entry["change"]) for entry in plotted), default=0.0)
    axis_title = f"change vs {a_label}"
    ordering = [entry["metric"] for entry in plotted]
    categories = _Categories.of("metric", ordering)
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
            # point that marks the value. A 28px bar with a marker at its end is two marks
            # competing to be read as the quantity.
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
    # 3px under a ~10px glyph band is nothing to knock a number out of — surface-coloured
    # glyphs would be legible over the stripe and invisible above and below it. So the
    # label takes the ordinary ink. It rode the `filled` default while the connector was
    # a fraction of the band, thick enough to pass for a fill at one row and at no other
    # row count.
    # `thickness`: the connector is not therefore ABSENT under the label, and reading
    # `filled=False` as "chart surface and nothing else" is what drew a bar through
    # every number this arm writes. A label pushed inward — which the row deciding the
    # axis always is, the room past it being the point's radius — is pushed along the
    # connector, so it is lifted clear of it rather than laid on it. Measured on the
    # rendered SVG before the lift: the connector's band and the glyphs' band were the
    # same 3px of row, at one row and at every other row count.
    # `radius`: the point is CENTRED on the change it names, so the value's position is
    # the middle of the mark rather than its edge. Handed over so the label is set from
    # the edge — without it the 6px gap is spent inside the disc and 1.5px of it reaches
    # the reader, which renders as a number touching its own point.
    labels = [
        MarkValue(
            display=categories.display[entry["metric"]],
            end=entry["change"],
            text=f"{entry['change']:+.1%}",
            radius=_POINT_RADIUS,
            filled=False,
            thickness=_CONNECTOR_HEIGHT,
        )
        for entry in plotted
    ]
    title = f"{a_label} vs {b_label}"
    spec: dict[str, Any] = _composed(
        {
            "$schema": VEGA_LITE_SCHEMA,
            "title": _title_spec(title, categories.figure_width()),
            "data": {"values": categories.labelled(plotted)},
            "layer": _layers(_zero_rule(value_axis), *layers, *value_label_layers(labels, value_axis, identity)),
            "width": width,
            "height": height,
        },
        categories,
    )

    columns: list[ChartColumn] = [
        {"key": "metric", "header": "Metric"},
        {"key": "a", "header": a_label},
        {"key": "b", "header": b_label},
        {"key": "delta", "header": f"Δ ({b_label}−{a_label})"},
        {"key": "change", "header": axis_title},
        {"key": "effect", "header": "Effect"},
    ]
    rows = [_delta_row(row) for row in payload.rows]
    disclosures: list[str] = []
    if immaterial := [row.metric for row in payload.rows if row.materiality == "immaterial"]:
        # Labelled, not hidden: the row is still drawn and tabulated, and this says which changes the
        # host declared too small to act on, so a reader does not act on one.
        disclosures.append(
            f"{len(immaterial)} of {len(payload.rows)} changes are below their measure's materiality threshold "
            f"({', '.join(immaterial)}) — immaterial: too small to act on, however clearly they clear their noise."
        )
    if undrawn:
        # Named, never silently dropped: the table below the chart still counts
        # these rows, so an unexplained gap reads as a rendering fault.
        disclosures.append(
            f"{len(undrawn)} of {len(payload.rows)} metrics are not drawn "
            f"({', '.join(undrawn)}) — a relative axis cannot place a non-numeric value or a change from a zero baseline."
        )
    return CompiledChart(spec=spec, columns=columns, rows=rows, unit="", disclosures=disclosures, title=title)


def _delta_row(row: DeltaRow) -> dict[str, Any]:
    """One comparison row, rendered for the values table in its own metric's unit.

    The restatement ladder is applied PER ROW here, unlike every other chart,
    and that is the same one-unit-per-quantity rule rather than an exception to it: one unit per
    *quantity*, and each row of this table is a different quantity. A latency
    stated in seconds beside a cost stated in dollars is correct; the same
    latency stated as 16162 ms is not.

    Args:
        row: One metric's A and B values, with whatever statistics it carries.

    Returns:
        The values-table row.
    """
    change = _relative_change(row.a, row.b) if row.data_type == "numeric" else None
    if row.data_type != "numeric":
        return {
            "metric": row.metric,
            "a": _render_cell(row.a),
            "b": _render_cell(row.b),
            "delta": None,
            "change": None,
            "effect": _effect_read(row),
        }
    a = _numeric_side(row, "a", row.a)
    b = _numeric_side(row, "b", row.b)
    delta = row.delta if row.delta is not None else b - a
    scale, unit = display_scale([a, b, delta], row.unit)
    return {
        "metric": row.metric,
        "a": _with_unit(a * scale, unit),
        "b": _with_unit(b * scale, unit),
        "delta": _signed_with_unit(delta * scale, unit) + (" (immaterial)" if row.materiality == "immaterial" else ""),
        "change": f"{change:+.1%}" if change is not None else None,
        "effect": _effect_read(row),
    }


def _numeric_side(row: DeltaRow, name: str, value: float | str | bool | None) -> float:
    """One side of a numeric row, as the number :class:`DeltaRow` already guarantees it is.

    The payload model refuses a numeric row whose sides are not finite numbers, so this
    raises only for a row built around that validation.

    Args:
        row: The row the value belongs to, named in the refusal.
        name: Which side it is, ``"a"`` or ``"b"``.
        value: The side's value.

    Returns:
        The value as a number.

    Raises:
        PayloadError: The value is not a number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PayloadError(f"metric {row.metric!r} is numeric but {name} is {value!r}")
    return value


def _effect_read(row: DeltaRow) -> str:
    """The paired-test read for one row, as words.

    "Not significant" and "not tested" are different facts, and the payload
    model refuses a significance flag with no statistic behind it — so a row
    reaching here either carries the statistic its verdict came from or makes no
    verdict at all.

    Delegated rather than branched here: this read is the same rule the MCP
    compare table prints, and the two hand-written copies had already drifted on
    both the not-tested predicate and the number formatting.

    Pairing is read OFF THE ROW. It was passed as a literal ``True`` here, on the
    reading that a field named ``d_z`` could only be holding a paired effect size
    — but the payload never said that, and the browser kit rendering the same
    stored dict read an absent ``paired`` as unpaired, so one row printed d_z on
    the report and d in the kit's table. A field's NAME cannot decide what a
    number is; only the test that produced it can.

    Args:
        row: One metric's comparison row.

    Returns:
        The verdict, with its statistics where the row states any.
    """
    return format_significance(significant=row.significant, paired=row.paired, p=row.p, effect=row.d_z, n=row.n)


def _relative_change(a: float | str | bool | None, b: float | str | bool | None) -> float | None:
    """Change from A as a signed fraction of A, or ``None`` where it has no finite value.

    A zero baseline has no relative change — every non-zero B is infinitely far
    from it — so the caller states the two values and the reason rather than
    drawing both markers at the centre, which reads as "no change".

    Args:
        a: The baseline value, of whatever type the row declared.
        b: The compared value.

    Returns:
        The signed fraction, or ``None`` where the pair cannot produce one.
    """
    if not isinstance(a, int | float) or not isinstance(b, int | float) or isinstance(a, bool) or isinstance(b, bool):
        return None
    if a == 0 or not math.isfinite(a) or not math.isfinite(b):
        return None
    return (b - a) / abs(a)


__all__ = [
    "compile_delta_table",
]
