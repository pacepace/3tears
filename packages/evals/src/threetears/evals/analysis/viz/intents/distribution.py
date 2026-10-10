"""The ``distribution`` intent — per-group spreads on one shared value axis, or pre-binned counts alone.

A distribution of a quantity already on an axis is a MARGINAL of that axis, so every group's estimate
and its observations are placed on one ruler — and so are pre-binned counts whose bins state numeric
edges. The one exception is a payload whose every group recorded only label-only bins: nothing places
on a value axis there, so the counts are the whole chart.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import SMALL_N_BAND_FLOOR
from threetears.evals.analysis.viz.intent import ChartAxis, ChartColumn, ChartEncoding, ChartIdentity, ChartIntent
from threetears.evals.analysis.viz.payloads import DistributionGroup, DistributionPayload
from threetears.evals.analysis.viz.quantities import (
    axis_title,
    display_scale,
    interval_disclosures,
    interval_sources,
    interval_statement,
)

#: What a panel of pre-binned counts is measuring, where those counts are the whole chart.
COUNT_TITLE = "observations"

#: A group's ``shape`` in the values table when it reported only an interval: no observation and no bin says
#: what lies between the ends, so the table says so rather than leave a reader to supply a bell.
SHAPE_UNKNOWN = "unknown — interval only"


def distribution_intent(payload: DistributionPayload) -> ChartIntent:
    """Decide a per-group distribution: each group's estimate and observations as positions on one axis.

    **One axis, one quantity.** Every group is placed against the same domain, which is the point:
    spreads that do not share a ruler cannot be compared. A spread is a POSITION, not a length, so the
    axis crops to the data. Shape is drawn from values and never inferred from two endpoints: a group
    reporting only an interval gets the interval and an explicit statement that its shape is unknown.

    **Pre-binned buckets reach the axis only by their edges.** A bin that states ``low`` and ``high`` is
    placed on the shared ruler beside the sampled cohorts, restated with everything else. A bin with only
    its label is never placed: reading numbers out of a label would invent edges the payload never gave.
    Those counts stay exact in the values table, and the footnote names the cohorts whose recorded shape
    could not be placed.

    Args:
        payload: The validated per-group distribution payload.

    Returns:
        The intent.
    """
    values = [value for group in payload.groups for value in _group_values(group)]
    scale, unit = display_scale(values, payload.unit)
    order = [group.label for group in payload.groups]
    identity = ChartIdentity(field="label", order=order, ordered_by="as the payload lists the groups")
    if not any(group.ci is not None or group.samples or _edged(group) for group in payload.groups):
        return _binned_intent(payload, identity, scale, unit)

    # The measure names itself ONCE: the unit rides into the heading, and the axis states the same words.
    title = axis_title(payload.x_label or "Distribution", unit)
    data: list[dict[str, Any]] = []
    for group in payload.groups:
        if group.ci is not None:
            data.append(
                {
                    "label": group.label,
                    "mean": group.ci.mean * scale,
                    "low": group.ci.low * scale,
                    "high": group.ci.high * scale,
                    "n": group.n,
                }
            )
    # Each observation is a mark placed by its value — the marginal a renderer draws as ticks or as a band.
    data.extend(
        {"label": group.label, "sample": value * scale} for group in payload.groups for value in group.samples or []
    )
    # A cohort's recorded bins, placed by the edges the payload stated and restated like every other value.
    # Its samples outrank them where it carries both: the observations ARE the shape the bins summarise.
    data.extend(
        {"label": group.label, "bin_low": low * scale, "bin_high": high * scale, "count": count}
        for group in payload.groups
        if not group.samples
        for low, high, count in _edges(group)
    )
    placed_bins = any(_edged(group) and not group.samples for group in payload.groups)
    intervals = [group.ci for group in payload.groups]
    spans = interval_sources(intervals)
    has_interval = any(group.ci is not None for group in payload.groups)
    columns = [ChartColumn(key="label", header="Group")]
    if has_interval:
        columns.extend(
            [
                ChartColumn(key="mean", header=axis_title("Mean", unit)),
                ChartColumn(key="low", header=axis_title("Low", unit)),
                ChartColumn(key="high", header=axis_title("High", unit)),
            ]
        )
    # An absent `n` is stated by absence, never as a column of em dashes.
    if any(group.n is not None for group in payload.groups):
        columns.append(ChartColumn(key="n", header="Cases"))
    columns.append(ChartColumn(key="shape", header="Shape"))
    unplaceable = [group.label for group in payload.groups if group.buckets and not group.samples and not _edged(group)]
    return ChartIntent(
        type="distribution",
        title=title,
        payload=payload.model_dump(mode="json"),
        scale=scale,
        unit=unit,
        data=data,
        encodings=[
            ChartEncoding(field="label", role="identity"),
            *(
                [
                    ChartEncoding(field="mean", role="position", axis="value"),
                    ChartEncoding(field="low", role="interval_low", axis="value", varies_over=spans),
                    ChartEncoding(field="high", role="interval_high", axis="value", varies_over=spans),
                ]
                if has_interval
                else []
            ),
            *(
                [ChartEncoding(field="sample", role="position", axis="value")]
                if any(g.samples for g in payload.groups)
                else []
            ),
            *(
                [
                    ChartEncoding(field="bin_low", role="position", axis="value"),
                    ChartEncoding(field="bin_high", role="position", axis="value"),
                    ChartEncoding(field="count", role="count", axis="count"),
                ]
                if placed_bins
                else []
            ),
        ],
        axes=[
            ChartAxis(name="value", quantity=title, unit=unit, zero_baseline=False),
            *([ChartAxis(name="count", quantity=COUNT_TITLE, zero_baseline=True)] if placed_bins else []),
        ],
        identity=identity,
        direct_labels=True,
        intervals=interval_statement(intervals),
        footnote=_unplaceable_shape_footnote(unplaceable),
        columns=columns,
        rows=[_distribution_row(group, scale, unit, payload.unit) for group in payload.groups],
        disclosures=[*interval_disclosures(intervals), *_unbanded_disclosures(payload.groups)],
    )


def _unbanded_disclosures(groups: Sequence[DistributionGroup]) -> list[str]:
    """Say why a group of a few values carries no interval band (#677).

    A group drawn from fewer than :data:`~threetears.evals.analysis.stats.SMALL_N_BAND_FLOOR` values and no
    interval is drawn as those values alone: a t interval over so few is too wide and unstable to draw as a
    band, so its absence is a decision the reader is told about, not a gap.

    Args:
        groups: The payload's groups.

    Returns:
        One line naming every such group, or none.
    """
    few = [
        group
        for group in groups
        if group.ci is None and not group.buckets and group.samples and len(group.samples) < SMALL_N_BAND_FLOOR
    ]
    if not few:
        return []
    named = ", ".join(f"{group.label} ({len(group.samples or [])})" for group in few)
    return [
        f"Drawn as its values with no interval band: {named}. Below {SMALL_N_BAND_FLOOR} values a t interval is too "
        "wide and unstable to draw; a value from a case run more than once is that case's mean."
    ]


def _binned_intent(payload: DistributionPayload, identity: ChartIdentity, scale: float, unit: str) -> ChartIntent:
    """Decide a distribution whose every group recorded only pre-binned counts.

    The one shape of this type that is NOT a marginal: no estimate and no raw value places anything on
    a value axis, so the counts are the whole chart. Bin ranges are label strings, so they are ordered as
    given and never placed on a value axis. The measure carries no unit in the title: what this chart
    draws is a count, and the measure's own unit is inside the bin names the payload wrote.

    Args:
        payload: The distribution, every group of which carries only buckets.
        identity: The groups, in drawn order.
        scale: The unit restatement factor, carried for the values table.
        unit: The restated unit.

    Returns:
        The intent.
    """
    data: list[dict[str, Any]] = [
        {"label": group.label, "range": bucket.range, "count": bucket.count}
        for group in payload.groups
        for bucket in (group.buckets or [])
    ]
    bins = list(dict.fromkeys(str(row["range"]) for row in data))
    columns = [ChartColumn(key="label", header="Group")]
    if any(group.n is not None for group in payload.groups):
        columns.append(ChartColumn(key="n", header="Cases"))
    columns.append(ChartColumn(key="shape", header="Shape"))
    title = payload.x_label or "Distribution"
    return ChartIntent(
        type="distribution",
        title=title,
        payload=payload.model_dump(mode="json"),
        scale=scale,
        unit=unit,
        data=data,
        encodings=[
            ChartEncoding(field="label", role="identity"),
            ChartEncoding(field="range", role="ordinal", axis="bins"),
            ChartEncoding(field="count", role="count", axis="count"),
        ],
        axes=[
            ChartAxis(name="bins", quantity=title, zero_baseline=False, order=bins),
            ChartAxis(name="count", quantity=COUNT_TITLE, zero_baseline=True),
        ],
        identity=identity,
        direct_labels=True,
        columns=columns,
        rows=[_distribution_row(group, scale, unit, payload.unit) for group in payload.groups],
    )


def _unplaceable_shape_footnote(labels: Sequence[str]) -> str:
    """State the cohorts whose recorded shape could not be drawn on the axis.

    Named by what a reader would have SEEN — the individual runs drawn under the interval on every
    other cohort — rather than by this module's word for it.

    Args:
        labels: The groups whose only recorded shape is pre-binned.

    Returns:
        One sentence, or ``""`` where every group's shape is drawable.
    """
    if not labels:
        return ""
    return (
        f"Individual runs not drawn for {', '.join(labels)}: recorded as pre-binned counts, whose bins name "
        "ranges rather than locating them. The counts are in the values table."
    )


def _edged(group: DistributionGroup) -> bool:
    """Whether a group's recorded bins state numeric edges (the payload holds a group to all or none)."""
    return bool(group.buckets) and all(bucket.edged for bucket in group.buckets or [])


def _edges(group: DistributionGroup) -> list[tuple[float, float, int]]:
    """A group's edged bins as ``(low, high, count)`` in the payload's unit, low to high; empty for label-only bins."""
    return sorted(
        (bucket.low, bucket.high, bucket.count)
        for bucket in group.buckets or []
        if bucket.low is not None and bucket.high is not None
    )


def _distribution_row(group: DistributionGroup, scale: float, unit: str, recorded: str | None) -> dict[str, Any]:
    """One group's line in the values table, including what its shape rests on.

    Pre-binned counts are written out here rather than summarised: a label-only bin cannot be drawn on
    the value axis at all, and this table is the whole of what the reader gets of it.

    **One row, one unit.** A bin with edges is spelled from them, restated with every other value in the
    row and followed by the unit it is now in. A label-only bin cannot be restated — its label is prose —
    so where the chart restated the unit, the cell names the unit its labels were recorded in; otherwise
    a `45–50k` bin would sit beside a `Mean (s)` of 52 with nothing saying they are two rulers.

    Args:
        group: One cohort of the distribution.
        scale: The unit restatement factor the chart's values took.
        unit: The unit the chart's values are stated in, after that restatement.
        recorded: The payload's own unit, which every recorded value — bin labels included — is in.

    Returns:
        The values-table row.
    """
    observed = len(group.samples or [])
    buckets = group.buckets or []
    if observed:
        shape = f"{observed} samples"
    elif _edged(group):
        shape = "; ".join(
            f"{format_number(low * scale)}–{format_number(high * scale)}: {count}" for low, high, count in _edges(group)
        )
        if unit:
            shape = f"{shape} ({unit})"
    elif buckets:
        shape = "; ".join(f"{bucket.range}: {bucket.count}" for bucket in buckets)
        if recorded and unit != recorded:
            shape = f"{shape} (bins in {recorded})"
    else:
        # Said out loud: a reader shown only a band will supply a bell the data never stated.
        shape = SHAPE_UNKNOWN
    row: dict[str, Any] = {"label": group.label, "n": group.n, "shape": shape}
    if group.ci is not None:
        row |= {"mean": group.ci.mean * scale, "low": group.ci.low * scale, "high": group.ci.high * scale}
    return row


def _group_values(group: DistributionGroup) -> list[float]:
    """Every numeric value a distribution group contributes to the shared domain."""
    values = list(group.samples or [])
    if not group.samples:
        values.extend(edge for low, high, _ in _edges(group) for edge in (low, high))
    if group.ci is not None:
        values.extend([group.ci.low, group.ci.high, group.ci.mean])
    return values


__all__ = [
    "COUNT_TITLE",
    "SHAPE_UNKNOWN",
    "distribution_intent",
]
