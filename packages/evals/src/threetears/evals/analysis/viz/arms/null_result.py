"""The ``null_result`` arm — an established null's arms as intervals, overlap shaded.

The overlap band is geometry and is drawn as geometry. Neither the shading nor the
prose beside it may read as a verdict: what settles the comparison is a test on the
difference, and that reaches the reader through the finding, not through this
picture.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.viz.compiler import (
    DISPLAY_FIELD,
    SECONDARY_OPACITY,
    VEGA_LITE_SCHEMA,
    ChartColumn,
    CompiledChart,
    _axis_title,
    _Categories,
    _composed,
    _interval_caption,
    _interval_disclosures,
    MarkValue,
    _title_spec,
    value_label_layers,
    ValueAxis,
    _with_unit,
    display_scale,
)
from threetears.evals.analysis.viz.payloads import NullResultPayload

#: The interval rule's thickness, in px.
#:
#: The same 3px the other two dumbbells in this package draw — ``delta_table``'s
#: connector and ``distribution``'s estimate rule — so an interval recedes under its
#: point the same way across one report. Named rather than written into the mark
#: alone because the label placement has to know it: a number pushed inward lands on
#: this rule, and how far it must be lifted off it is this number's business.
_INTERVAL_HEIGHT = 3


def compile_null_result(payload: NullResultPayload) -> CompiledChart:
    """Compile an established null's arms as intervals, with any overlap shaded.

    The overlap band is drawn because the reader needs to see where the arms
    coincide. It is **not** evidence of the null, and neither the prose beside the
    chart nor the chart may say it is: two marginal intervals can overlap while the difference
    between the means is real, and reading overlap as a verdict once published a
    false null over arms 33% apart. What settles the comparison is a test on the
    DIFFERENCE, which reaches the reader through the finding's verdict and the
    mechanism below the chart.

    Args:
        payload: The validated null-result payload.

    Returns:
        The compiled chart.
    """
    bounds = [bound for arm in payload.groups for bound in (arm.ci.low, arm.ci.high)]
    scale, unit = display_scale(bounds, payload.unit)
    title = payload.metric or "Null result"
    axis_title = _axis_title(payload.metric or "value", unit)

    ordering = [arm.label for arm in payload.groups]
    rows: list[dict[str, Any]] = [
        {
            "label": arm.label,
            "mean": arm.ci.mean * scale,
            "low": arm.ci.low * scale,
            "high": arm.ci.high * scale,
            "n": arm.n,
        }
        for arm in payload.groups
    ]

    # The overlap is [max of lows, min of highs] — non-empty only if the arms overlap.
    overlap_low = max(arm.ci.low for arm in payload.groups) * scale
    overlap_high = min(arm.ci.high for arm in payload.groups) * scale
    overlaps = overlap_high >= overlap_low

    categories = _Categories.of("label", ordering)
    width, height = categories.plot_size()
    # An arm's interval is a LOCATION on the measure, so the axis crops to the arms
    # and states that it did. Forcing zero here would be the defect this type is
    # drawn to expose: two arms that differ by a third of their value look identical
    # once the axis spends most of its width on territory neither occupies.
    value_axis = ValueAxis.position(axis_title, [bound * scale for bound in bounds], width)
    identity = categories.axis()
    layers: list[dict[str, Any]] = []
    if overlaps:
        layers.append(
            {
                "data": {"values": [{"low": overlap_low, "high": overlap_high}]},
                "mark": {"type": "rect", "opacity": SECONDARY_OPACITY},
                "encoding": {"x": value_axis.encoding("low"), "x2": {"field": "high"}},
            }
        )
    layers.append(
        {
            "data": {"values": categories.labelled(rows)},
            "mark": {"type": "rule", "size": _INTERVAL_HEIGHT, "tooltip": True},
            "encoding": {
                "y": identity,
                "x": value_axis.encoding("low"),
                "x2": {"field": "high"},
            },
        }
    )
    layers.append(
        {
            "data": {"values": categories.labelled(rows)},
            "mark": {"type": "point", "filled": True, "size": 70, "tooltip": True},
            "encoding": {"y": identity, "x": value_axis.encoding("mean")},
        }
    )
    layers.extend(
        value_label_layers(
            [
                # `filled=False`: this arm draws a 3px rule and a point, and 3px under a
                # ~10px glyph band is nothing to knock a number out of — the knockout ink
                # IS the chart surface, so a label taking it there would be painted on the
                # background in the background's own colour.
                #
                # `thickness`: inward of the value is not EMPTY either, which is the half
                # the sentence above does not answer. The label is anchored at the high end
                # and pushed inward when the axis has no room past it — a wide label on the
                # outermost arm, which the domain pad leaves about 57px for — and inward is
                # back along the rule. So it is lifted off the rule rather than drawn
                # through it.
                #
                # No `radius`, and that is a measurement rather than an oversight: the label
                # is anchored at the interval's HIGH end, where the rule stops — a butt cap
                # at the value, whose 3px is thickness across the row and nothing along the
                # axis. The point sits at the mean, hundreds of px away. Rendered, the gap
                # here is the full 6px the constant names, where the arms whose label sits
                # on a POINT had 1.5px and 0.5px of it.
                MarkValue(
                    display=row[DISPLAY_FIELD],
                    end=row["high"],
                    text=format_number(row["mean"]),
                    filled=False,
                    thickness=_INTERVAL_HEIGHT,
                )
                for row in categories.labelled(rows)
            ],
            value_axis,
            identity,
        )
    )

    spec: dict[str, Any] = _composed(
        {
            "$schema": VEGA_LITE_SCHEMA,
            "title": _title_spec(title, categories.figure_width(), ""),
            "layer": layers,
            "width": width,
            "height": height,
        },
        categories,
    )
    # Never empty here, so there is no absent-description branch to write: an arm's
    # `ci` is required and the payload holds at least two, so there is always an
    # interval for `_interval_caption` to qualify.
    intervals = [arm.ci for arm in payload.groups]
    spec["description"] = _interval_caption(intervals)

    # Geometry, stated as geometry. Neither branch may carry a verdict: overlap
    # does not establish a null, and non-overlap does not refute the finding's own
    # conclusion — the test that settles either is not in this picture.
    if overlaps:
        overlap_sentence = (
            f"Intervals overlap on [{_with_unit(overlap_low, unit)}, {_with_unit(overlap_high, unit)}]. "
            "Overlap alone does not establish a null."
        )
    else:
        overlap_sentence = "Intervals do not overlap."
    # One line per idea: the geometry, then what the intervals are, then the author's
    # stated mechanism — carried verbatim, as its own line, because it is the payload's
    # reason the lever cannot act and a reader must be able to find where it starts.
    disclosures = [line for line in (overlap_sentence, *_interval_disclosures(intervals), payload.mechanism) if line]

    columns: list[ChartColumn] = [
        {"key": "label", "header": "Arm"},
        {"key": "mean", "header": _axis_title("Mean", unit)},
        {"key": "low", "header": _axis_title("Low", unit)},
        {"key": "high", "header": _axis_title("High", unit)},
    ]
    if any(arm.n is not None for arm in payload.groups):
        columns.append({"key": "n", "header": "n"})
    return CompiledChart(spec=spec, columns=columns, rows=rows, unit=unit, disclosures=disclosures, title=title)


__all__ = [
    "compile_null_result",
]
