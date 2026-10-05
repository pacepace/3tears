"""The ``timeseries`` intent — one reading followed across a campaign's builds or days, a series per row.

**Time is ordinal, and its order is stated.** The positions are builds or days, earliest first, as the
payload lists them; a categorical axis left to sort itself sorts alphabetically, and a line through
``0.10`` before ``0.9`` is a trend the campaign never had. **A gap is a fact**: a position a series has no
point at is not interpolated across, and is disclosed with its reason. **Identity never rides on hue**:
each series is a row of its own, named as a distribution names its cohorts.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.viz.intent import ChartAxis, ChartColumn, ChartEncoding, ChartIdentity, ChartIntent
from threetears.evals.analysis.viz.payloads import TimeseriesPayload
from threetears.evals.analysis.viz.quantities import (
    axis_title,
    display_scale,
    interval_disclosures,
    interval_sources,
    interval_statement,
)

#: The data key holding a point's time position.
POSITION_FIELD = "position"


def _gap_lines(payload: TimeseriesPayload) -> list[str]:
    """One disclosure line per reason a point is missing, naming each series and the positions it lacks.

    Grouped by reason because the reason is what the sentence spends its words on, and one sentence per
    gap repeats it as many times as the axis is long.
    """
    by_reason: dict[str, dict[str, list[str]]] = {}
    for gap in payload.gaps:
        by_reason.setdefault(gap.reason, {}).setdefault(gap.series, []).append(gap.position)
    return [
        f"Not drawn ({reason}): "
        + "; ".join(f"{series} at {', '.join(where)}" for series, where in named.items())
        + "."
        for reason, named in by_reason.items()
    ]


def timeseries_intent(payload: TimeseriesPayload) -> ChartIntent:
    """Decide one reading across the time axis: each series' points as positions, each with its interval.

    Args:
        payload: The validated timeseries payload.

    Returns:
        The intent.
    """
    intervals = [point.ci for line in payload.series for point in line.points]
    scale, unit = display_scale([bound for ci in intervals for bound in (ci.low, ci.high)], payload.unit)
    time_title = f"Build ({payload.release_label})" if payload.basis == "release" else "Day (UTC)"
    points: list[dict[str, Any]] = [
        {
            "series": line.label,
            POSITION_FIELD: point.position,
            "mean": point.ci.mean * scale,
            "low": point.ci.low * scale,
            "high": point.ci.high * scale,
            "n": point.n,
        }
        for line in payload.series
        for point in line.points
    ]
    spans = interval_sources(intervals)
    order_line = (
        f"Builds of {payload.release_label} are in the order each first ran."
        if payload.basis == "release"
        else "Days are UTC, in calendar order."
    )
    columns = [
        ChartColumn(key="series", header="Series"),
        ChartColumn(key=POSITION_FIELD, header=time_title),
        ChartColumn(key="mean", header=axis_title("Mean", unit)),
        ChartColumn(key="low", header=axis_title("Low", unit)),
        ChartColumn(key="high", header=axis_title("High", unit)),
        ChartColumn(key="n", header="n"),
    ]
    return ChartIntent(
        type="timeseries",
        title=f"{axis_title(payload.metric, unit)} over {'builds' if payload.basis == 'release' else 'days'}",
        payload=payload.model_dump(mode="json"),
        scale=scale,
        unit=unit,
        data=points,
        encodings=[
            ChartEncoding(field="series", role="identity"),
            ChartEncoding(field=POSITION_FIELD, role="ordinal", axis="time"),
            ChartEncoding(field="mean", role="position", axis="value"),
            ChartEncoding(field="low", role="interval_low", axis="value", varies_over=spans),
            ChartEncoding(field="high", role="interval_high", axis="value", varies_over=spans),
            ChartEncoding(field="n", role="label"),
        ],
        axes=[
            ChartAxis(name="time", quantity=time_title, zero_baseline=False, order=list(payload.positions)),
            ChartAxis(name="value", quantity=axis_title(payload.metric, unit), unit=unit, zero_baseline=False),
        ],
        identity=ChartIdentity(
            field="series", order=[line.label for line in payload.series], ordered_by="as the payload lists the series"
        ),
        direct_labels=True,
        intervals=interval_statement(intervals),
        columns=columns,
        rows=[{column.key: row[column.key] for column in columns} for row in points],
        disclosures=[order_line, *interval_disclosures(intervals), *_gap_lines(payload)],
    )


__all__ = [
    "POSITION_FIELD",
    "timeseries_intent",
]
