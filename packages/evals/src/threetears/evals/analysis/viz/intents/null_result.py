"""The ``null_result`` intent — an established null's arms as intervals, their overlap stated as geometry.

Neither the overlap nor the words beside it may read as a verdict: what settles the comparison is a test
on the difference, and that reaches the reader through the finding, not through this chart.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.viz.intent import ChartAxis, ChartColumn, ChartEncoding, ChartIdentity, ChartIntent
from threetears.evals.analysis.viz.payloads import NullResultPayload
from threetears.evals.analysis.viz.quantities import (
    axis_title,
    display_scale,
    interval_disclosures,
    interval_sources,
    interval_statement,
    with_unit,
)


def null_result_intent(payload: NullResultPayload) -> ChartIntent:
    """Decide an established null's chart: each arm's interval as a position, the overlap stated.

    An arm's interval is a LOCATION on the measure, so its axis crops to the arms rather than starting
    at zero — forcing zero would make two arms a third apart look identical, which is the defect this
    type is drawn to expose.

    Args:
        payload: The validated null-result payload.

    Returns:
        The intent.
    """
    bounds = [bound for arm in payload.groups for bound in (arm.ci.low, arm.ci.high)]
    scale, unit = display_scale(bounds, payload.unit)
    title = payload.metric or "Null result"
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
    if overlap_high >= overlap_low:
        overlap_sentence = (
            f"Intervals overlap on [{with_unit(overlap_low, unit)}, {with_unit(overlap_high, unit)}]. "
            "Overlap alone does not establish a null."
        )
    else:
        overlap_sentence = "Intervals do not overlap."
    intervals = [arm.ci for arm in payload.groups]
    # One line per idea: the geometry, then what the intervals are, then the author's stated mechanism —
    # carried verbatim, as its own line, because it is the payload's reason the lever cannot act.
    disclosures = [line for line in (overlap_sentence, *interval_disclosures(intervals), payload.mechanism) if line]
    spans = interval_sources(intervals)
    columns = [
        ChartColumn(key="label", header="Arm"),
        ChartColumn(key="mean", header=axis_title("Mean", unit)),
        ChartColumn(key="low", header=axis_title("Low", unit)),
        ChartColumn(key="high", header=axis_title("High", unit)),
    ]
    counted = any(arm.n is not None for arm in payload.groups)
    if counted:
        columns.append(ChartColumn(key="n", header="Cases"))
    return ChartIntent(
        type="null_result",
        title=title,
        payload=payload.model_dump(mode="json"),
        scale=scale,
        unit=unit,
        data=rows,
        encodings=[
            ChartEncoding(field="label", role="identity"),
            ChartEncoding(field="mean", role="position", axis="value"),
            ChartEncoding(field="low", role="interval_low", axis="value", varies_over=spans),
            ChartEncoding(field="high", role="interval_high", axis="value", varies_over=spans),
            *([ChartEncoding(field="n", role="label")] if counted else []),
        ],
        axes=[
            ChartAxis(
                name="value", quantity=axis_title(payload.metric or "value", unit), unit=unit, zero_baseline=False
            )
        ],
        identity=ChartIdentity(
            field="label", order=[arm.label for arm in payload.groups], ordered_by="as the payload lists the arms"
        ),
        direct_labels=True,
        intervals=interval_statement(intervals),
        columns=columns,
        rows=rows,
        disclosures=disclosures,
    )


__all__ = [
    "null_result_intent",
]
