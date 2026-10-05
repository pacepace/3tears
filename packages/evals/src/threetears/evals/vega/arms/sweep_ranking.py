"""The ``sweep_ranking`` arm — configurations ranked, with their levels as a barcode.

**A combination is not a series; it is a row.** A campaign sweeping two or more
levers produces combinations, and drawing them as coloured series is what the
categorical palette cannot survive — thirty-six cells against four validated hues.
So the configuration goes on the row, and the ranked measure takes position.

The configuration is drawn as a **fused barcode**: one cell per swept lever, cells
touching rather than spaced, so a lever that drives the ranking shows up as a
contiguous block in its column and the reader finds it by scanning for the column
that sorted itself. That is the whole reason the cells are fused — a gap between
them turns a block into a list of separate marks.

**Two ink vocabularies, and they never share a scale.** An ordered lever's levels
run low to high, so they take the ordered ramp by rank, which Vega samples at
however many steps the sweep has. A categorical lever's levels have no order, so
they take the validated hues and their names are written in the values table. The
two are separate layers with an independently resolved colour scale, which is
admissible precisely because their domains are disjoint: a concurrency rank and a
model name are never the same category, so the harm a shared scale exists to
prevent — one category drawn in two hues — cannot arise here.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.vega.compiler import (
    DISPLAY_FIELD,
    VEGA_LITE_SCHEMA,
    MarkValue,
    ValueAxis,
    _identity,
    _number,
    _title_spec,
    _value_axis,
    plot_size,
    point_radius,
    value_label_layers,
)
from threetears.evals.analysis.viz.intent import ChartIntent
from threetears.evals.analysis.viz.intents.sweep_ranking import CONFIG_FIELD, LEVER_KEY_PREFIX
from threetears.evals.vega.palette import CONTEXT_STYLE, SEQUENTIAL_RANGE, geometry
from threetears.evals.analysis.viz.payloads import ABSENT_LEVEL
from threetears.evals.vega.spec_policy import RANKING_SPEC_NAME

#: The row key holding a configuration's identity — what both panels align on.
#:
#: The compiler's own display field rather than a name of this arm's, because the
#: value-label machinery keys on it.
_ROW_FIELD = DISPLAY_FIELD

#: The row key holding which swept lever a barcode cell belongs to.
_DIMENSION_FIELD = "dimension"

#: The row key holding the level a cell draws, as text.
_LEVEL_FIELD = "level"

#: The row key holding an ordered level's position in its own lever's range.
#:
#: A rank rather than the level's own number: the ramp should step evenly between
#: the levels that were actually run, and a lever swept at 1, 2 and 64 would
#: otherwise draw its first two levels indistinguishably — encoding the arithmetic
#: gap between levels, which the sweep never measured, instead of the order.
_RANK_FIELD = "rank"

#: The gap between the barcode and the ranking, in px. Zero inside the barcode is
#: what fuses it; this is the separation between the glyph and the measurement.
_PANEL_GAP = 16

#: A configuration's mark area in px², which is ~12px across. Smaller than a
#: frontier contestant's, which has to carry a shape distinction this one does not.
_POINT_SIZE = 120

#: How far that point extends past the value it is centred on, in px.
#:
#: The label's clearance is measured from it: this point is the widest in the package,
#: so a number offset from the CENTRE instead cleared it by half a pixel and rendered
#: welded to its own mark.
_POINT_RADIUS = point_radius(_POINT_SIZE)


def compile_sweep_ranking(intent: ChartIntent) -> dict[str, Any]:
    """Draw a ranked configuration sweep as a barcode beside a ranking.

    The configuration is drawn as a **fused barcode**: one cell per swept lever, cells
    touching rather than spaced, so a lever that drives the ranking shows up as a
    contiguous block in its column — the reader finds it by scanning for the column
    that sorted itself.

    Args:
        intent: The sweep's intent.

    Returns:
        The Vega-Lite spec.
    """
    levers = [
        encoding.field.removeprefix(LEVER_KEY_PREFIX) for encoding in intent.encodings if encoding.role == "level"
    ]
    ramps = {
        colours.field.removeprefix(LEVER_KEY_PREFIX): colours.domain
        for colours in intent.colours
        if colours.scheme == "sequential"
    }
    keys = _identity(intent).order
    ranked_title = _value_axis(intent, "ranked").quantity
    sizes = geometry()
    _, height = plot_size(max(len(keys), 1))
    identity = {"field": _ROW_FIELD, "type": "nominal", "sort": keys, "axis": None}

    marks = [
        {
            _ROW_FIELD: entry[CONFIG_FIELD],
            _DIMENSION_FIELD: name,
            _LEVEL_FIELD: entry[f"{LEVER_KEY_PREFIX}{name}"],
            _RANK_FIELD: _level_rank(str(entry[f"{LEVER_KEY_PREFIX}{name}"]), ramps.get(name)),
        }
        for entry in intent.data
        for name in levers
    ]
    ranking: list[dict[str, Any]] = [
        {_ROW_FIELD: entry[CONFIG_FIELD], "ranked": entry["ranked"], "secondary": entry["secondary"]}
        for entry in intent.data
    ]
    value_axis = ValueAxis.position(ranked_title, [_number(entry["ranked"]) for entry in ranking], sizes["plot_width"])
    # Rendered here rather than by a Vega `format`: the two renderers must draw the
    # same characters. `filled=False`: the ranking panel draws POINTS, so a label
    # placed inward sits on the chart surface. `radius`: the point is centred on the
    # value, so the clearance is measured from its edge.
    labels = [
        MarkValue(
            display=str(entry[_ROW_FIELD]),
            end=_number(entry["ranked"]),
            text=format_number(_number(entry["ranked"])),
            radius=_POINT_RADIUS,
            filled=False,
        )
        for entry in ranking
    ]
    barcode = _barcode_panel(marks, identity, levers, set(ramps), height)
    return {
        "$schema": VEGA_LITE_SCHEMA,
        "title": _title_spec(intent.title, sizes["figure_width"], intent.footnote),
        "hconcat": [barcode, _ranking_panel(ranking, identity, value_axis, labels, height)],
        "spacing": _PANEL_GAP,
        # The rows must line up across the two panels or the barcode describes a
        # different configuration from the mark beside it. Vega-Lite resolves a
        # concat's positional scales independently by default.
        "resolve": {"scale": {"y": "shared"}},
    }


def _level_rank(level: str, ramp: list[str] | None) -> float | None:
    """Where a level sits in its lever's ramp, as a 0-based rank.

    Args:
        level: The level, as text.
        ramp: The lever's levels low to high when the intent gives it a ramp; None for a lever drawn in
            categorical slots. Read from the intent rather than re-derived: a rank and the categorical
            domain are two halves of one partition, and a lever answered "ordered" by one rule and
            "categorical" by another produces cells matching NEITHER barcode layer.

    Returns:
        The rank, or ``None`` for a level with no place in an order — the absence
        sentinel, or any level of a categorical lever.
    """
    if ramp is None:
        return None
    return float(ramp.index(level)) if level in ramp else None


def _barcode_panel(
    marks: list[dict[str, Any]],
    identity: dict[str, Any],
    order: list[str],
    ordered_levers: set[str],
    height: int,
) -> dict[str, Any]:
    """The glyph: one fused cell per lever per configuration.

    Two layers rather than one, because the two ink vocabularies cannot share a
    scale — a rank and a model name are not values of one thing. Their domains are
    disjoint by construction, which is what makes the independent resolution
    admissible rather than the defect the rule usually names.

    Args:
        marks: One entry per cell.
        identity: The row encoding both panels align on.
        order: The dimension names, in column order.
        ordered_levers: Which of them carry an order.
        height: The panel height in px, shared with the ranking beside it.

    Returns:
        The barcode view.
    """
    sizes = geometry()
    column = {
        "field": _DIMENSION_FIELD,
        "type": "nominal",
        "sort": order,
        # No labels: a lever's name does not fit a cell's width and the label rules forbid
        # both truncating it and shrinking it, so the columns are named in a
        # disclosure line and spelled in full in the values table instead.
        "axis": None,
    }
    # The absence sentinel is excluded from BOTH vocabularies, and that is a
    # correctness rule rather than tidiness. A lever this configuration never set has
    # no level, so a hue for it would draw an absence as though it were a value the
    # reader could compare — and left in neither domain it draws in whatever Vega
    # picks for an out-of-domain datum, which is the same defect arrived at by
    # accident. It gets the neutral that carries no identity, which is what the
    # absence IS.
    categorical = sorted(
        {
            mark[_LEVEL_FIELD]
            for mark in marks
            if mark[_DIMENSION_FIELD] not in ordered_levers and mark[_LEVEL_FIELD] != ABSENT_LEVEL
        }
    )
    ranks = [mark[_RANK_FIELD] for mark in marks if mark[_RANK_FIELD] is not None]
    layers: list[dict[str, Any]] = []
    if any(mark[_LEVEL_FIELD] == ABSENT_LEVEL for mark in marks):
        layers.append(
            {
                "transform": [{"filter": {"field": _LEVEL_FIELD, "equal": ABSENT_LEVEL}}],
                "mark": {"type": "rect", "tooltip": True, "style": CONTEXT_STYLE},
                "encoding": {"x": column, "y": identity, "tooltip": _cell_tooltip()},
            }
        )
    if ranks:
        layers.append(
            {
                "transform": [{"filter": {"field": _RANK_FIELD, "valid": True}}],
                "mark": {"type": "rect", "tooltip": True},
                "encoding": {
                    "x": column,
                    "y": identity,
                    # A LINEAR scale over the rank, never an ordinal one. An ordinal
                    # scale recycles past the ramp's authored stops, so a sweep with
                    # more levels than stops would draw its darkest level in its
                    # lightest one's ink and the block would read as having wrapped.
                    # A linear scale samples the ramp instead, which is why a lever
                    # that reaches THIS layer has no ceiling on how many levels it
                    # may have. The ceiling rations hues, and a lever drawn from the
                    # ramp spends none — including the case this file decides, where
                    # a lever declared ordered over levels that are words draws in
                    # hues instead and counts like any other.
                    "color": {
                        "field": _RANK_FIELD,
                        "type": "quantitative",
                        "scale": {"domain": [0, max(ranks) or 1], "range": SEQUENTIAL_RANGE},
                        "legend": None,
                    },
                    "tooltip": _cell_tooltip(),
                },
            }
        )
    if categorical:
        layers.append(
            {
                "transform": [
                    {"filter": {"field": _RANK_FIELD, "valid": False}},
                    {"filter": {"field": _LEVEL_FIELD, "oneOf": categorical}},
                ],
                "mark": {"type": "rect", "tooltip": True},
                "encoding": {
                    "x": column,
                    "y": identity,
                    "color": {
                        "field": _LEVEL_FIELD,
                        "type": "nominal",
                        # An explicit domain, so a level keeps its hue from one
                        # figure of a report to the next; no range at all, so the
                        # validated slots come from the renderer's own config.
                        "scale": {"domain": categorical},
                        # Direct labels, never a key: the values table names every
                        # level of every configuration, which is the label a barcode
                        # cell is too narrow to carry itself.
                        "legend": None,
                    },
                    "tooltip": _cell_tooltip(),
                },
            }
        )
    view: dict[str, Any] = {
        "data": {"values": marks},
        "width": sizes["gutter_left"],
        "height": height,
    }
    if len(layers) == 1:
        return view | layers[0]
    return view | {
        "layer": layers,
        # Admissible because the domains are disjoint — see this module's header and
        # `policy._check_shared_scales`, which reads the domains rather than
        # refusing the declaration on sight.
        "resolve": {"scale": {"color": "independent"}},
    }


def _cell_tooltip() -> list[dict[str, str]]:
    """What a barcode cell says when a reader asks it directly.

    The cells carry no text of their own — a lever name and a level do not fit the
    width, and the label rules refuse to shrink either — so this and the values table are
    where the levels are actually readable.
    """
    return [
        {"field": _DIMENSION_FIELD, "type": "nominal", "title": "Lever"},
        {"field": _LEVEL_FIELD, "type": "nominal", "title": "Level"},
    ]


def _ranking_panel(
    ranking: list[dict[str, Any]],
    identity: dict[str, Any],
    value_axis: ValueAxis,
    labels: list[MarkValue],
    height: int,
) -> dict[str, Any]:
    """The measurement: one mark per configuration on the ranked axis.

    A point rather than a bar, and that follows the zero-baseline rule rather than
    a preference: this is a POSITION on a measure, so it crops to the data — its
    labelled ticks disclose where the axis starts — where a length would have to
    start at zero and would flatten the very differences the ranking is about.

    Args:
        ranking: One entry per drawn configuration.
        identity: The row encoding both panels align on.
        value_axis: The ranked measure's axis.
        labels: Each mark's value and where its mark ends.
        height: The panel height in px, shared with the barcode beside it.

    Returns:
        The ranking view.
    """
    return {
        # Declaring the frame a ranking is what SUBMITS it to rule 10 — the gate
        # reads this name and checks the stated row order against the quantities
        # actually plotted. An undeclared frame is an ordinary ordered list and is
        # not judged, so omitting it here would not relax the rule, it would make
        # the rule unreachable on the one chart it was written for.
        "name": RANKING_SPEC_NAME,
        "data": {"values": ranking},
        "width": geometry()["plot_width"],
        "height": height,
        "layer": [
            {
                # Full ink rather than the point mark's density default: these are a
                # handful of answers, not a cloud, and at 0.7 each would draw in a
                # blend of the palette hue and whatever sits behind it — a different
                # colour in each theme, arrived at by not deciding.
                "mark": {"type": "point", "filled": True, "size": _POINT_SIZE, "tooltip": True, "opacity": 1},
                "encoding": {"x": value_axis.encoding("ranked"), "y": identity},
            },
            *value_label_layers(labels, value_axis, identity),
        ],
    }


__all__ = [
    "compile_sweep_ranking",
]
