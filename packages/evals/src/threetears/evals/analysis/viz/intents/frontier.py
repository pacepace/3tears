"""The ``frontier`` intent — contestants placed on cost against quality, contention carried by shape.

The one type whose identity is not on an axis: it reads two quantities against each other and names each
contestant beside its mark. Both axes are positions and may crop. **Dominance rides on shape, never hue**:
an on-frontier contestant is a circle, a dominated one a diamond, a disqualified one a cross — geometry a
reader with low contrast vision still receives, and a channel the palette's recycling cannot reach.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.viz.intent import (
    ChartAxis,
    ChartColumn,
    ChartEncoding,
    ChartIdentity,
    ChartIntent,
    ChartReference,
)
from threetears.evals.analysis.viz.payloads import FrontierPayload, FrontierVizPoint
from threetears.evals.analysis.viz.quantities import axis_title, display_scale, strip_common_prefix

#: The symbol each contention class is drawn as, best to worst — the order a reader is told them in.
CLASS_SHAPES: tuple[tuple[str, str], ...] = (
    ("On frontier", "circle"),
    ("Dominated", "diamond"),
    ("Disqualified", "cross"),
)

#: The row key holding a point's contention class — what its shape is drawn from.
CLASS_FIELD = "status"

#: The row key holding the name drawn beside a contestant's mark — its label with the shared prefix stripped.
DISPLAY_FIELD = "display"

#: The class of a contestant that carries no cost, and so is not on the plot at all. It has no symbol
#: because it has no mark; it exists so the values table can say why a row is not in the picture.
UNPRICED_CLASS = "Not priced"


def _classify(dominated: bool, disqualified: bool, *, priced: bool) -> str:
    """Which contention class a point belongs to.

    Disqualification outranks domination: a contestant out on a safety bar is out whatever its cost
    bought. An unpriced contestant is its own class rather than the default one: domination is a claim
    about both axes, so a point with no cost cannot be known to be on the frontier OR off it.

    Args:
        dominated: Whether some other point beats it on both axes.
        disqualified: Whether it failed a two-pillar / safety bar.
        priced: Whether it carries a production-replicating cost.

    Returns:
        The class name.
    """
    if disqualified:
        return "Disqualified"
    if dominated:
        return "Dominated"
    return "On frontier" if priced else UNPRICED_CLASS


def _joined_names(labels: Sequence[str]) -> str:
    """Join contestant names so the sentence around them reads as English."""
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


def _disqualifications(points: Sequence[FrontierVizPoint]) -> list[tuple[str, list[str]]]:
    """The disqualified contestants, grouped under the reason they share, in first-appearance order.

    Grouped rather than stated once per contestant, because the reason is what the sentence spends its
    words on, and a disclosure that grows with the size of the thing it discloses stops being read.
    Reasons are matched exactly: two wordings of one cause are two groups.
    """
    grouped: dict[str, list[str]] = {}
    for point in points:
        if point.disqualified and point.disqualified_reason:
            grouped.setdefault(point.disqualified_reason.strip(), []).append(point.label)
    return list(grouped.items())


def frontier_intent(payload: FrontierPayload) -> ChartIntent:
    """Decide a cost-against-quality trade-off: each priced contestant a point, its class a shape.

    Args:
        payload: The validated frontier payload.

    Returns:
        The intent.
    """
    cost_title = payload.cost_label or "Cost"
    quality_title = payload.quality_label or "Quality"
    # A point with no cost has no x to be placed at: it is disclosed and kept in the values table rather
    # than dropped, and never placed at zero, where an unpriced contestant would draw as the cheapest.
    unpriced = [point.label for point in payload.points if point.cost is None]
    # Latency is restated in the largest unit that keeps two significant figures — the only quantity
    # here with a declared unit to restate.
    latencies = [point.latency_ms for point in payload.points if point.latency_ms is not None]
    latency_scale, latency_unit = display_scale(latencies, "ms" if latencies else None)
    display = strip_common_prefix([point.label for point in payload.points])
    rows: list[dict[str, Any]] = [
        {
            "label": point.label,
            DISPLAY_FIELD: display[point.label],
            "cost": point.cost,
            "quality": point.quality,
            "latency": None if point.latency_ms is None else point.latency_ms * latency_scale,
            CLASS_FIELD: _classify(point.dominated, point.disqualified, priced=point.cost is not None),
        }
        for point in payload.points
    ]
    drawn = [row for row in rows if row["cost"] is not None]
    present = [(name, symbol) for name, symbol in CLASS_SHAPES if any(row[CLASS_FIELD] == name for row in drawn)]

    # What the figure cannot say for itself, one line per idea, in reading order: the key a reader needs
    # to read the marks at all, then who is out and why, then who is missing, then the standard the rest
    # are held to. The shape vocabulary is a line because the figure draws no key — every mark is named
    # in place, so identity never needed one — and only where a distinction exists.
    disclosures: list[str] = []
    if len(present) > 1:
        disclosures.append(", ".join(f"{symbol} = {name.lower()}" for name, symbol in present) + ".")
    disclosures.extend(
        f"{_joined_names(labels)} {'is' if len(labels) == 1 else 'are'} disqualified: {reason}."
        for reason, labels in _disqualifications(payload.points)
    )
    if unpriced:
        disclosures.append(
            f"Not drawn — no production-replicating cost was recorded for {_joined_names(unpriced)}; "
            "the values below carry what was measured."
        )
    if payload.bar is not None:
        disclosures.append(f"The quality bar is {format_number(payload.bar)}.")

    columns = [
        ChartColumn(key="label", header="Contestant"),
        ChartColumn(key="cost", header=cost_title),
        ChartColumn(key="quality", header=quality_title),
    ]
    if latencies:
        columns.append(ChartColumn(key="latency", header=axis_title("Latency", latency_unit)))
    columns.append(ChartColumn(key=CLASS_FIELD, header="Contention"))
    return ChartIntent(
        type="frontier",
        title=f"{quality_title} against {cost_title}",
        payload=payload.model_dump(mode="json"),
        scale=1.0,
        unit="",
        data=drawn,
        encodings=[
            ChartEncoding(field="label", role="identity"),
            ChartEncoding(field=DISPLAY_FIELD, role="label"),
            ChartEncoding(field="cost", role="position", axis="cost"),
            ChartEncoding(field="quality", role="position", axis="quality"),
            ChartEncoding(field=CLASS_FIELD, role="class"),
        ],
        # Cost and quality carry captions the generator wrote, not units this engine knows how to ladder.
        axes=[
            ChartAxis(name="cost", quantity=cost_title, zero_baseline=False),
            ChartAxis(name="quality", quantity=quality_title, zero_baseline=False),
        ],
        identity=ChartIdentity(
            field="label", order=[str(row["label"]) for row in drawn], ordered_by="as the payload lists the contestants"
        ),
        references=[ChartReference(axis="quality", value=payload.bar, label="quality bar")]
        if payload.bar is not None
        else [],
        shapes=dict(present),
        direct_labels=True,
        columns=columns,
        rows=rows,
        disclosures=disclosures,
    )


__all__ = [
    "CLASS_FIELD",
    "CLASS_SHAPES",
    "DISPLAY_FIELD",
    "UNPRICED_CLASS",
    "frontier_intent",
]
