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
    ChartColumn,
    CompiledChart,
    _axis_title,
    _bar_mark,
    _Categories,
    _composed,
    _layers,
    MarkValue,
    _signed_with_unit,
    _title_spec,
    value_label_layers,
    ValueAxis,
    _with_unit,
    _zero_rule,
    display_scale,
)
from threetears.evals.analysis.viz.payloads import AttributionPayload

#: What an `attribution` draws in the remainder's row when the subtraction was not
#: earned. Words rather than a zero-length bar: the row has to stay on the axis so
#: the picture records that the question was asked, and anything MEASURED against
#: the value axis there would read as "nothing was left over".
_NOT_PLACEABLE = "not placeable"


def compile_attribution(payload: AttributionPayload) -> CompiledChart:
    """Compile a whole-vs-part movement onto one shared signed axis.

    **Two bars side by side, never stacked, and never a waterfall.** A waterfall
    lays the part end-to-end against the whole and closes the gap with a
    remainder, which draws a balanced partition — the exact attribution the
    finding says it cannot make. Bars from a common zero state each movement as
    its own magnitude and leave the reader to see that one is far larger than the
    other, which is the fact.

    **The remainder always occupies a row; only its MARK changes.** Where the
    arithmetic is earned it is a bar, drawn at a lower opacity than the two
    measured movements because it is derived rather than observed — the same
    reason raw sample dots are recessive against the estimate they surround.
    Where it is withheld the same row carries the words *not placeable* instead.

    Three options and only one of them is honest. Omitting the row leaves the
    picture silent about a question the finding asked, so a reader who never
    opens the values table cannot tell "we could not place this" from "nobody
    asked". Drawing the row EMPTY is worse: a category band with no mark sits
    exactly where a zero-length bar would, so it reads as "the movement was fully
    accounted for" — the one misreading this whole type exists to prevent. A text
    mark occupies the row unmistakably and cannot be measured against the axis,
    which is the property the other two lack.

    Args:
        payload: The validated whole-vs-part payload.

    Returns:
        The compiled chart.
    """
    # The whole first, then the part, then what is left over: read down the axis it
    # is the finding's own sentence — the run moved this much, the part under test
    # moved this much, and this much is unplaced.
    measured = [("End-to-end", payload.end_to_end), ("Subsystem", payload.subsystem)]
    remainder_scope = "Unattributed"
    unattributed = payload.unattributed_delta
    quantified = unattributed is not None

    # The display unit is chosen from the MOVEMENTS, not from the levels they run
    # between. A pair of 50 ms deltas between two ~44 s levels is a real movement,
    # and restating it in seconds to suit the levels rounds it to 0.05 — a change
    # the reader would take for none. The levels are then stated in that same unit
    # rather than their own, which is the one-unit-per-quantity rule holding: a
    # measure's delta and its levels are the same quantity.
    magnitudes = [movement.delta for _, movement in measured]
    if unattributed is not None:
        magnitudes.append(unattributed)
    scale, unit = display_scale(magnitudes, payload.unit)

    counted = any(movement.n is not None for _, movement in measured)
    plotted: list[dict[str, Any]] = [
        {"scope": scope, "measure": movement.measure, "delta": movement.delta * scale, "derived": False}
        for scope, movement in measured
    ]
    if unattributed is not None:
        plotted.append({"scope": remainder_scope, "measure": "", "delta": unattributed * scale, "derived": True})

    axis_title = _axis_title("change", unit)
    # The remainder's band exists either way, so the axis reads the same whether the
    # arithmetic was earned or refused — a chart that loses a row when a number is
    # unavailable makes the absence look like the row was never asked for.
    ordering = [scope for scope, _ in measured] + [remainder_scope]
    categories = _Categories.of("scope", ordering)
    width, height = categories.plot_size()
    # A bar's length is a magnitude measured from no change, so the baseline is where
    # no change is. The side of it carries direction.
    value_axis = ValueAxis.magnitude(axis_title, [row["delta"] for row in plotted], width)
    identity = categories.axis(domain=not value_axis.marks_form_the_edge())
    title = f"{payload.end_to_end.measure} vs {payload.subsystem.measure}"
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
        # edge whenever every movement shares a sign, so on an all-negative chart —
        # the type's headline case, a whole-run duration that IMPROVED while the part
        # did not move — zero is the RIGHT edge, and a left-aligned label runs off the
        # plot. Clipped, the row renders empty, which is exactly the "fully accounted
        # for" misreading this type exists to prevent. So the text reads back into the
        # plot: away from whichever edge zero is on, asked of the drawn domain rather
        # than re-derived from the values it was built from.
        anchors_right = value_axis.high == 0
        not_placeable = {
            "data": {"values": categories.labelled([{"scope": remainder_scope, "delta": 0, "label": _NOT_PLACEABLE}])},
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
        # the fact on this chart, while the unit written eleven times is the axis
        # title copied onto every mark.
        MarkValue(display=categories.display[row["scope"]], end=row["delta"], text=_signed_with_unit(row["delta"], ""))
        for row in plotted
    ]
    spec: dict[str, Any] = _composed(
        {
            "$schema": VEGA_LITE_SCHEMA,
            "title": _title_spec(title, categories.figure_width()),
            "layer": _layers(
                _zero_rule(value_axis), bars, not_placeable, *value_label_layers(labels, value_axis, identity)
            ),
            "width": width,
            "height": height,
        },
        categories,
    )

    columns: list[ChartColumn] = [
        {"key": "scope", "header": "Scope"},
        {"key": "measure", "header": "Measure"},
        {"key": "a", "header": payload.a_label or "A"},
        {"key": "b", "header": payload.b_label or "B"},
        {"key": "delta", "header": _axis_title("Δ", unit)},
    ]
    if counted:
        # Worth a column here more than on any other chart: the two movements have
        # their OWN observation counts, and two different n's side by side are the
        # plainest statement that these are different populations — which is the
        # whole reason the remainder below them may not be a number.
        columns.append({"key": "n", "header": "n"})
    rows: list[dict[str, Any]] = [
        {
            "scope": scope,
            "measure": movement.measure,
            "a": _with_unit(movement.a * scale, unit) if movement.a is not None else None,
            "b": _with_unit(movement.b * scale, unit) if movement.b is not None else None,
            "delta": _signed_with_unit(movement.delta * scale, unit),
            **({"n": movement.n} if counted else {}),
        }
        for scope, movement in measured
    ]
    rows.append(
        {
            "scope": remainder_scope,
            "measure": None,
            "a": None,
            "b": None,
            # `None` is what an unplaceable remainder is owed here; a disclosure carries
            # the reason it cannot be a number. What the reader SEES of that null differs
            # by surface and is not this layer's to assert — `values_as_drawn()` renders
            # an em dash, while the browser's cell renderer maps null to "not recorded",
            # which reads as a measurement that went missing rather than one refused.
            #
            # **That split is now settled rather than pending.** It was recorded here as
            # a defect awaiting reconciliation; it is DELIBERATE — a cell that looks blank in a screen-reader table cannot say
            # which kind of absence it is, so the browser spells it out. Making the two
            # agree would cost that, and is a decision against the rule rather than a
            # tidy-up. The numeric rule is what the two surfaces share.
            "delta": _signed_with_unit(unattributed * scale, unit) if unattributed is not None else None,
        }
    )
    return CompiledChart(
        spec=spec,
        columns=columns,
        rows=rows,
        unit=unit,
        disclosures=_attribution_disclosures(payload),
        title=title,
    )


def _attribution_disclosures(payload: AttributionPayload) -> list[str]:
    """State which levels were compared, and what may be said about the remainder.

    The withheld sentence is carried VERBATIM rather than summarised: it names
    which of the ways a subtraction goes wrong applies to this pair, and a reader
    deciding whether the finding bounds a decision needs that reason rather than
    the fact that there was one.

    Args:
        payload: The whole-vs-part payload, carrying the lever and either a
            quantified remainder or the sentence withholding it.

    Returns:
        The comparison line, where the payload names its lever, then the remainder line.
    """
    parts: list[str] = []
    if payload.lever:
        levels = f": {payload.a_label} → {payload.b_label}" if payload.a_label and payload.b_label else ""
        parts.append(f"Comparing {payload.lever}{levels}.")
    if payload.unattributed_withheld:
        parts.append(payload.unattributed_withheld)
    else:
        # Past tense, and deliberately. The containment is a fact the payload
        # CARRIES, recorded when the analysis was generated; the compiler never
        # re-reads the live catalog, so a present-tense claim here would speak for a
        # registry this code has not consulted and that may since have moved.
        parts.append(
            f"Unattributed is {payload.end_to_end.measure} minus {payload.subsystem.measure}, "
            f"which the measure catalog declared a component of it when this was generated."
        )
    return parts


__all__ = [
    "compile_attribution",
]
