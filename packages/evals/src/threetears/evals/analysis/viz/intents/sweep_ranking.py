"""The ``sweep_ranking`` intent — configurations ranked by one measure, each row's levels beside it.

**A combination is not a series; it is a row.** A campaign sweeping two or more levers produces
combinations, and drawing them as coloured series is what the categorical palette cannot survive. So the
configuration goes on the row, the ranked measure takes position, and each swept lever's level is carried
in a colour scheme of its own: an ordered lever's levels take a ramp by rank, a categorical lever's take
the validated slots.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.viz.intent import (
    ChartAxis,
    ChartColours,
    ChartColumn,
    ChartEncoding,
    ChartIdentity,
    ChartIntent,
)
from threetears.evals.analysis.viz.payloads import ABSENT_LEVEL, ResolvedDimension, SweepRankingPayload, SweepRow
from threetears.evals.analysis.viz.quantities import axis_title, display_scale, with_unit

#: How many configurations a figure draws before it starts omitting them. Past this the figure keeps
#: :data:`ROWS_WHEN_TRUNCATED` and states what it dropped: thirty-six configurations at one row step is a
#: chart the reader scrolls instead of comparing.
MAX_ROWS = 12

#: How many configurations survive truncation — two fewer than the threshold, so the sentence saying the
#: figure shrank has somewhere to go.
ROWS_WHEN_TRUNCATED = 10

#: The data key holding a configuration's identity.
CONFIG_FIELD = "config"

#: Namespace for a swept lever's field, in the data and in the values table.
#:
#: A lever's name comes straight from the generator, and the table also holds the two measures and the
#: observation count: a lever named `ranked` or `n` would land on a measure's key and be overwritten by
#: it. Separated by NAMESPACE rather than by a reserved-name list, because a namespace cannot rot. Known
#: residual, accepted: the prefix separates KEYS, not HEADERS, so a lever named `n` still heads a column
#: `n` beside the observation count's — recoverable by position and value.
LEVER_KEY_PREFIX = "lever."


def row_key(row: SweepRow, order: list[str]) -> str:
    """One configuration's identity: its label, or its levels joined in column order.

    Args:
        row: One configuration.
        order: The dimension names, in column order.

    Returns:
        The row's key.
    """
    return row.label or " · ".join(row.config[name] for name in order)


def ordered_levels(dimension: str, payload: SweepRankingPayload) -> list[str]:
    """A lever's levels sorted low to high — numerically, which is why orderedness turns on parsability.

    Only ever called for a lever the payload's resolution reports as ordered, and that resolution requires
    every level to parse, so the numeric key cannot meet a level it can't convert.

    Args:
        dimension: The lever.
        payload: The sweep, which knows every level the lever was swept at.

    Returns:
        The levels, low to high, without the absence sentinel.
    """
    levels = {row.config[dimension] for row in payload.rows}
    return sorted((level for level in levels if level != ABSENT_LEVEL), key=float)


def sweep_ranking_intent(payload: SweepRankingPayload) -> ChartIntent:
    """Decide a ranked configuration sweep: rows by the ranked measure, each lever's level a colour slot.

    **Sorted by the ranked measure and by nothing else.** The block pattern in a lever's column IS the
    finding, so a sort on that column would manufacture the very thing the reader is invited to discover.
    Descending, so rank 1 is first. An empty slice is a RESULT: the chart falls back to the sweep the slice
    was taken from, and the disclosures say the slice is empty.

    Args:
        payload: The validated sweep payload.

    Returns:
        The intent.
    """
    resolved = payload.dimensions_resolved()
    order = [lever.name for lever in resolved]
    ordered = {lever.name for lever in resolved if lever.ordered}
    qualifying = payload.qualifying() or list(payload.rows)
    ranked = sorted(qualifying, key=lambda row: (-row.ranked_value, row_key(row, order)))
    drawn, dropped = ranked[:ROWS_WHEN_TRUNCATED], ranked[ROWS_WHEN_TRUNCATED:]
    if len(ranked) <= MAX_ROWS:
        drawn, dropped = ranked, []

    scale, ranked_unit = display_scale([row.ranked_value for row in ranked], payload.ranked.unit)
    secondary_scale, secondary_unit = display_scale(
        [row.secondary_value for row in payload.rows], payload.secondary.unit
    )
    ranked_title = axis_title(payload.ranked.measure, ranked_unit)
    data: list[dict[str, Any]] = [
        {
            CONFIG_FIELD: row_key(row, order),
            "ranked": row.ranked_value * scale,
            "secondary": row.secondary_value * secondary_scale,
            **{f"{LEVER_KEY_PREFIX}{name}": row.config[name] for name in order},
        }
        for row in drawn
    ]
    # The ABSENT sentinel is excluded from every scheme: a lever this configuration never set has no level,
    # so a hue for it would draw an absence as a value. It is named as uncoloured instead.
    categorical = sorted(
        {
            str(entry[f"{LEVER_KEY_PREFIX}{name}"])
            for entry in data
            for name in order
            if name not in ordered and entry[f"{LEVER_KEY_PREFIX}{name}"] != ABSENT_LEVEL
        }
    )
    colours: list[ChartColours] = []
    for name in order:
        field = f"{LEVER_KEY_PREFIX}{name}"
        # One categorical domain across every categorical lever, so a level keeps its slot from one column to
        # the next. A lever whose every drawn level is absent takes no scheme: nothing of it is coloured.
        domain = ordered_levels(name, payload) if name in ordered else categorical
        if any(entry[field] in domain for entry in data):
            colours.append(
                ChartColours(
                    field=field,
                    scheme="sequential" if name in ordered else "categorical",
                    domain=domain,
                    uncoloured=[ABSENT_LEVEL],
                )
            )
    columns = [ChartColumn(key=f"{LEVER_KEY_PREFIX}{name}", header=name) for name in order]
    columns.append(ChartColumn(key="ranked", header=ranked_title))
    columns.append(ChartColumn(key="secondary", header=axis_title(payload.secondary.measure, secondary_unit)))
    if any(row.n is not None for row in payload.rows):
        columns.append(ChartColumn(key="n", header="n"))
    rows = [
        {
            **{f"{LEVER_KEY_PREFIX}{name}": level for name, level in row.config.items()},
            "ranked": row.ranked_value * scale,
            "secondary": row.secondary_value * secondary_scale,
            "n": row.n,
        }
        for row in ranked
    ]
    return ChartIntent(
        type="sweep_ranking",
        title=f"{payload.ranked.measure} by configuration",
        payload=payload.model_dump(mode="json"),
        scale=scale,
        unit=ranked_unit,
        data=data,
        encodings=[
            ChartEncoding(field=CONFIG_FIELD, role="identity"),
            *(ChartEncoding(field=f"{LEVER_KEY_PREFIX}{name}", role="level") for name in order),
            ChartEncoding(field="ranked", role="position", axis="ranked"),
            ChartEncoding(field="secondary", role="label"),
        ],
        axes=[ChartAxis(name="ranked", quantity=ranked_title, unit=ranked_unit, zero_baseline=False)],
        identity=ChartIdentity(
            field=CONFIG_FIELD,
            order=[str(entry[CONFIG_FIELD]) for entry in data],
            ordered_by=f"{payload.ranked.measure}, highest first",
            ranked_by="ranked",
        ),
        colours=colours,
        # The barcode cells carry no text — a lever's name and level do not fit a cell — so the levels are
        # named in the values table, and the payload refuses more categorical levels than validated slots.
        direct_labels=False,
        footnote=_omission_sentence(payload, dropped, scale),
        columns=columns,
        rows=rows,
        disclosures=_disclosures(payload, resolved, order, secondary_scale, secondary_unit, dropped, scale),
    )


def _omission_sentence(payload: SweepRankingPayload, dropped: list[SweepRow], scale: float) -> str:
    """State every configuration missing from the chart, whoever dropped it.

    Two sources and one sentence: the producer may have sent a truncated sweep and declared what it left
    out, and the chart truncates again past its own row bound. The band is stated in the drawn unit.

    Args:
        payload: The sweep, which may declare its own omission.
        dropped: Configurations the chart does not draw.
        scale: The restatement factor the ranked measure took.

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

    The columns are NAMED, because the cells are too narrow to carry a lever's name. The secondary
    measure's spread is STATED — held within a tolerance, or free and its range given — because without it
    a block in one column could equally mean the configurations there cost more. An orderedness that was
    INFERRED rather than declared is admitted, and a lever declared ordered that draws as hues is too.

    Args:
        payload: The sweep.
        resolved: Each lever with its orderedness and whether that was declared.
        order: The lever names, in column order.
        secondary_scale: The restatement factor the secondary measure took.
        secondary_unit: Its restated unit.
        dropped: Configurations the chart does not draw.
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
        centre = with_unit(held.value * secondary_scale, secondary_unit)
        window = with_unit(held.tolerance * secondary_scale, secondary_unit)
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
            f"{secondary} runs from {with_unit(min(values), secondary_unit)} to "
            f"{with_unit(max(values), secondary_unit)} and is not held — the ranking is not controlled for it"
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
    "CONFIG_FIELD",
    "LEVER_KEY_PREFIX",
    "MAX_ROWS",
    "ROWS_WHEN_TRUNCATED",
    "ordered_levels",
    "row_key",
    "sweep_ranking_intent",
]
