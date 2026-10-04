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
from threetears.evals.analysis.viz.compiler import (
    DISPLAY_FIELD,
    VEGA_LITE_SCHEMA,
    ChartColumn,
    CompiledChart,
    _axis_title,
    MarkValue,
    plot_size,
    _title_spec,
    value_label_layers,
    ValueAxis,
    _with_unit,
    display_scale,
    point_radius,
)
from threetears.evals.analysis.viz.palette import CONTEXT_STYLE, SEQUENTIAL_RANGE, geometry
from threetears.evals.analysis.viz.payloads import ABSENT_LEVEL, ResolvedDimension, SweepRankingPayload, SweepRow
from threetears.evals.analysis.viz.policy import RANKING_SPEC_NAME

#: How many configurations a figure draws before it starts omitting them.
#:
#: Past this the figure keeps :data:`_ROWS_WHEN_TRUNCATED` and states what it
#: dropped. Thirty-six configurations at the row step is a 1728px figure, which is
#: not a tall chart but a chart the reader scrolls instead of comparing.
_MAX_ROWS = 12

#: How many configurations survive truncation. Two fewer than the threshold, so the
#: annotation row has somewhere to go and the figure does not grow to accommodate
#: the sentence that says it shrank.
_ROWS_WHEN_TRUNCATED = 10

#: The row key holding a configuration's identity — what both panels align on.
#:
#: The compiler's own display field rather than a name of this arm's, because the
#: value-label machinery keys on it: a configuration's ranked value is placed by the
#: same arithmetic that puts a number beside a bar, and a private key here would
#: have meant reimplementing the 56px clearance rule and the mark-count suppression
#: alongside it.
_ROW_FIELD = DISPLAY_FIELD

#: The row key holding which swept lever a barcode cell belongs to.
_DIMENSION_FIELD = "dimension"

#: The row key holding the level a cell draws, as text.
_LEVEL_FIELD = "level"

#: The row key holding an ordered level's position in its own lever's range.
#:
#: A rank rather than the level's own number: the ramp should step evenly between
#: the levels that were actually run, and a lever swept at 1, 2 and 64 would
#: otherwise draw its first two levels indistinguishably and its third alone at the
#: far end — encoding the arithmetic gap between levels, which the sweep never
#: measured, instead of the order, which it did.
_RANK_FIELD = "rank"

#: Namespace for a swept lever's column in the values-as-drawn table.
#:
#: The table holds two kinds of column — one per lever, plus the two measures and
#: the observation count — and a lever's name comes straight from the generator. A
#: lever named `ranked`, `secondary` or `n` would land on a measure's key, and the
#: level would be overwritten by that measure's value in every row while the two
#: columns drew under one key. That is not cosmetic here: both panels suppress the
#: row axis, so this table is the ONLY place a barcode row's identity is readable,
#: and the collision replaces the identity with a number.
#:
#: Separated by NAMESPACE rather than by a reserved-name list, and the difference is
#: that a namespace cannot rot. A blocklist is correct only while someone remembers
#: to extend it: add a column to this table and every unreserved name it introduces
#: is a live collision, with the suite green because nothing relates the two lists.
#: It also has to be taught to the generator, since a payload refused for a bound it
#: was never given costs a whole paid analysis rather than just its chart — so the
#: blocklist buys a prompt rule, a validator, and a test to hold them together,
#: where the prefix buys nothing and forecloses the case.
#:
#: Known residual, accepted: the prefix separates KEYS, not HEADERS, so a lever
#: named `n` still draws a column headed `n` beside the observation count's. That is
#: a reader disambiguating two headers by position and value, which is recoverable —
#: where the collision it replaces was a level silently overwritten by a number, and
#: the blocklist's answer to it was to discard a whole paid analysis over a lever
#: name that is otherwise perfectly good.
_LEVER_KEY_PREFIX = "lever."

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


def _row_key(row: SweepRow, order: list[str]) -> str:
    """One configuration's identity, as the axis and the values table spell it.

    Its levels, joined — a configuration has no name of its own unless the payload
    gave it one, and the levels ARE what distinguishes it. Never drawn: both panels
    suppress their row axis, because the barcode is the label. It exists so the two
    panels can align, and so the values table can name a row when a reader asks
    which one a mark belongs to.

    Args:
        row: One configuration.
        order: The dimension names, in barcode column order.

    Returns:
        The row's key.
    """
    return row.label or " · ".join(row.config[name] for name in order)


def compile_sweep_ranking(payload: SweepRankingPayload) -> CompiledChart:
    """Compile a ranked configuration sweep as a barcode beside a ranking.

    Args:
        payload: The validated sweep payload.

    Returns:
        The compiled chart.
    """
    resolved = payload.dimensions_resolved()
    order = [lever.name for lever in resolved]
    ordered_levers = {lever.name for lever in resolved if lever.ordered}

    # Sorted by the ranked measure and by nothing else. The block pattern in a
    # column IS the finding, so a sort on that column would manufacture the very
    # thing the reader is being invited to discover. Descending, because rank 1 sits
    # at the top and the axis puts high values at the right — the two orderings
    # agree and the marks descend as a staircase.
    # An empty slice is a RESULT and it is drawn as one: the figure falls back to
    # the sweep the slice was taken from, so the reader sees what was searched and
    # can read off the secondary values why nothing fell inside the tolerance. The
    # disclosures are what say the slice is empty. Neither alternative is acceptable —
    # a frame with no marks is the empty chart the reporting rules forbid, and refusing the
    # payload would discard a whole paid analysis over a measurement outcome.
    qualifying = payload.qualifying() or list(payload.rows)
    ranked = sorted(qualifying, key=lambda row: (-row.ranked_value, _row_key(row, order)))
    drawn, dropped = ranked[:_ROWS_WHEN_TRUNCATED], ranked[_ROWS_WHEN_TRUNCATED:]
    if len(ranked) <= _MAX_ROWS:
        drawn, dropped = ranked, []
    qualifying = ranked

    scale, ranked_unit = display_scale([row.ranked_value for row in qualifying], payload.ranked.unit)
    secondary_scale, secondary_unit = display_scale(
        [row.secondary_value for row in payload.rows], payload.secondary.unit
    )
    ranked_title = _axis_title(payload.ranked.measure, ranked_unit)

    keys = [_row_key(row, order) for row in drawn]
    sizes = geometry()
    _, height = plot_size(max(len(drawn), 1))
    identity = {"field": _ROW_FIELD, "type": "nominal", "sort": keys, "axis": None}

    marks = [
        {
            _ROW_FIELD: key,
            _DIMENSION_FIELD: name,
            _LEVEL_FIELD: row.config[name],
            _RANK_FIELD: _level_rank(name, row.config[name], payload, ordered_levers),
        }
        for row, key in zip(drawn, keys, strict=True)
        for name in order
    ]
    # Vega data rows, keyed by the field names the spec encodes.
    ranking: list[dict[str, Any]] = [
        {_ROW_FIELD: key, "ranked": row.ranked_value * scale, "secondary": row.secondary_value * secondary_scale}
        for row, key in zip(drawn, keys, strict=True)
    ]

    value_axis = ValueAxis.position(ranked_title, [entry["ranked"] for entry in ranking], sizes["plot_width"])
    # Rendered here rather than by a Vega `format`, for the reason every number in
    # this compiler is: the two renderers must draw the same characters, and a
    # format string is resolved by whichever engine happens to be running.
    # `filled=False`: the ranking panel draws POINTS, so a label placed inward sits on
    # the chart surface rather than on a mark — and the knockout ink IS that surface.
    # `radius`: the same point, from the other side. It is centred on the value, so its
    # body sits between the value and the label; the clearance is measured from its edge.
    labels = [
        MarkValue(
            display=entry[_ROW_FIELD],
            end=entry["ranked"],
            text=format_number(entry["ranked"]),
            radius=_POINT_RADIUS,
            filled=False,
        )
        for entry in ranking
    ]
    barcode = _barcode_panel(marks, identity, order, ordered_levers, payload, height)
    spec: dict[str, Any] = {
        "$schema": VEGA_LITE_SCHEMA,
        "title": _title_spec(_title(payload), sizes["figure_width"], _footnote(payload, dropped, scale)),
        "hconcat": [barcode, _ranking_panel(ranking, identity, value_axis, labels, height)],
        "spacing": _PANEL_GAP,
        # The rows must line up across the two panels or the barcode describes a
        # different configuration from the mark beside it. Vega-Lite resolves a
        # concat's positional scales independently by default, which is right for
        # two panels measuring two quantities and catastrophic for two halves of
        # one row.
        "resolve": {"scale": {"y": "shared"}},
    }

    # A lever's name reaches the reader through the HEADER; the key is only how the
    # two surfaces address the cell. Prefixing the key therefore costs a reader
    # nothing and forecloses the collision by construction.
    columns: list[ChartColumn] = [{"key": f"{_LEVER_KEY_PREFIX}{name}", "header": name} for name in order]
    columns.append({"key": "ranked", "header": ranked_title})
    columns.append({"key": "secondary", "header": _axis_title(payload.secondary.measure, secondary_unit)})
    if any(row.n is not None for row in payload.rows):
        columns.append({"key": "n", "header": "n"})
    rows = [
        {
            **{f"{_LEVER_KEY_PREFIX}{name}": level for name, level in row.config.items()},
            "ranked": row.ranked_value * scale,
            "secondary": row.secondary_value * secondary_scale,
            "n": row.n,
        }
        for row in qualifying
    ]
    return CompiledChart(
        spec=spec,
        columns=columns,
        rows=rows,
        unit=ranked_unit,
        disclosures=_disclosures(payload, resolved, order, secondary_scale, secondary_unit, dropped, scale),
        title=_title(payload),
    )


def _title(payload: SweepRankingPayload) -> str:
    """The figure's own heading — what is ranked, and against what."""
    return f"{payload.ranked.measure} by configuration"


def _level_rank(dimension: str, level: str, payload: SweepRankingPayload, ordered_levers: set[str]) -> float | None:
    """Where a level sits in its lever's range, as a 0-based rank.

    Args:
        dimension: The lever the level belongs to.
        level: The level, as text.
        payload: The sweep, which knows every level the lever was swept at.
        ordered_levers: The levers the payload's own resolution says draw as a
            ramp. Passed in rather than re-derived here, because a rank and the
            categorical domain are two halves of one partition: a lever answered
            "ordered" by one rule and "categorical" by another produces cells that
            match NEITHER barcode layer, and a cell matching no layer is not a
            visible error but a blank column under a disclosure still promising a
            ramp.

    Returns:
        The rank, or ``None`` for a level with no place in an order — the absence
        sentinel, or any level of a categorical lever.
    """
    if dimension not in ordered_levers:
        return None
    levels = _ordered_levels(dimension, payload)
    return float(levels.index(level)) if level in levels else None


def _ordered_levels(dimension: str, payload: SweepRankingPayload) -> list[str]:
    """A lever's levels sorted low to high.

    Sorted numerically rather than as text, which is the whole reason orderedness
    turns on parsability: ``10`` sorts before ``9`` as a string, and a ramp built
    on that order would draw the block pattern backwards for exactly the levers a
    reader is most likely to be scanning for.

    Only ever called for a lever the payload's resolution reports as ordered, and
    that resolution requires every level to parse — so the numeric key cannot meet
    a level it can't convert.
    """
    levels = {row.config[dimension] for row in payload.rows}
    return sorted((level for level in levels if level != ABSENT_LEVEL), key=float)


def _barcode_panel(
    marks: list[dict[str, Any]],
    identity: dict[str, Any],
    order: list[str],
    ordered_levers: set[str],
    payload: SweepRankingPayload,
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
        payload: The sweep, for the categorical domain.
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


def _footnote(payload: SweepRankingPayload, dropped: list[SweepRow], scale: float) -> str:
    """What the figure did to its rows, in the picture.

    Args:
        payload: The sweep.
        dropped: Configurations this compilation did not draw.
        scale: The restatement factor the ranked measure took, so the band is
            stated in the unit the marks beside it are labelled in.

    Returns:
        The footnote, or ``""``.
    """
    # The cropped-axis sentence is deliberately absent: this figure's marks are
    # points, and a point's value comes off the labelled ticks, which already begin
    # where the axis begins. What a reader CANNOT recover from the picture is a
    # configuration that is not in it, so that is what the footnote carries.
    return _omission_sentence(payload, dropped, scale)


def _omission_sentence(payload: SweepRankingPayload, dropped: list[SweepRow], scale: float) -> str:
    """State every configuration missing from the picture, whoever dropped it.

    Two sources and one sentence: the producer may have sent a truncated sweep and
    declared what it left out, and this compilation truncates again past its own
    row bound. A reader counting marks is owed the total, not whichever half the
    last stage happened to know about.

    Args:
        payload: The sweep, which may declare its own omission.
        dropped: Configurations this compilation did not draw.
        scale: The restatement factor the ranked measure took, so the band is
            stated in the unit the axis and the drawn values are in. Without it a
            figure whose measure was restated would name a band in the payload's
            original unit beside marks labelled in the restated one — two numbers
            for one quantity, in the sentence whose whole job is to let the reader
            judge whether what was dropped could have changed the verdict.

    Returns:
        One sentence naming the count and the band, or ``""``.
    """
    declared = payload.omitted
    if not dropped and declared is None:
        return ""
    count = len(dropped) + (declared.count if declared else 0)
    bounds = [row.ranked_value * scale for row in dropped]
    if declared:
        bounds.extend([declared.low * scale, declared.high * scale])
    return (
        f"{count} further configuration{'' if count == 1 else 's'} ranked between "
        f"{format_number(min(bounds))} and {format_number(max(bounds))} and are not drawn."
    )


def _disclosures(
    payload: SweepRankingPayload,
    resolved: list[ResolvedDimension],
    order: list[str],
    secondary_scale: float,
    secondary_unit: str,
    dropped: list[SweepRow],
    scale: float,
) -> list[str]:
    """Everything the barcode cannot say for itself, one line per idea.

    Three obligations, and none of them is optional. The columns have to be NAMED,
    because the cells are too narrow to carry a lever's name and the reader
    otherwise has a glyph with no key. The secondary measure's spread has to be
    STATED — held within a tolerance, or free and its range given — because without
    it a block in the prompt column could equally mean "v3 configurations cost
    more", and the finding the barcode carries would be confounded by spend. And
    where a lever's orderedness was INFERRED rather than declared, that has to be
    admitted, since a ramp asserts an order the payload never claimed.

    A fourth follows from the third by the same reasoning, in the other direction:
    where a lever was declared ordered and drew as hues anyway — its levels state
    no order to place them in — that is admitted too. Overriding a declaration in
    silence would leave the generator's stated intent contradicted by the figure
    with nothing on the page saying so.

    Args:
        payload: The sweep.
        resolved: Each lever with its orderedness and whether that was declared.
        order: The lever names, in column order.
        secondary_scale: The restatement factor the secondary measure took.
        secondary_unit: Its restated unit.
        dropped: Configurations this compilation did not draw.
        scale: The restatement factor the ranked measure took.

    Returns:
        The lines, in reading order.
    """
    inferred = [lever.name for lever in resolved if not lever.declared]
    ramped = [lever.name for lever in resolved if lever.ordered]
    demoted = [lever.name for lever in resolved if lever.demoted]
    values = [row.secondary_value * secondary_scale for row in payload.rows]
    secondary = payload.secondary.measure
    if payload.held_fixed is not None:
        held = payload.held_fixed
        centre = _with_unit(held.value * secondary_scale, secondary_unit)
        window = _with_unit(held.tolerance * secondary_scale, secondary_unit)
        spread = (
            f"{secondary} is held at {centre} ± {window}"
            if payload.qualifying()
            else (
                f"No configuration fell within ± {window} of {centre} for {secondary}, so the slice is empty — "
                f"every configuration the slice was taken from is drawn instead"
            )
        )
    else:
        spread = (
            f"{secondary} runs from {_with_unit(min(values), secondary_unit)} to "
            f"{_with_unit(max(values), secondary_unit)} and is not held — the ranking is not controlled for it"
        )
    return [
        sentence
        for sentence in (
            f"Columns, left to right: {', '.join(order)}.",
            f"{spread}.",
            f"{', '.join(ramped)} draw{'s' if len(ramped) == 1 else ''} as a light-to-dark ramp." if ramped else "",
            (
                f"Whether {', '.join(inferred)} {'is' if len(inferred) == 1 else 'are'} ordered was inferred from the "
                "levels rather than declared."
                if inferred
                else ""
            ),
            (
                f"{', '.join(demoted)} {'was' if len(demoted) == 1 else 'were'} declared ordered but "
                f"{'draws' if len(demoted) == 1 else 'draw'} as hues: the levels state no order to place them in."
                if demoted
                else ""
            ),
            _omission_sentence(payload, dropped, scale),
        )
        if sentence
    ]


__all__ = [
    "compile_sweep_ranking",
]
