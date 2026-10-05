"""The ``delta_table`` intent — an A-vs-B comparison as relative change from A, each row in its own unit.

The rows are different metrics in different units, so the one thing they can share is a unitless axis:
what is plotted is relative change, what is tabulated is each row's own unit, and a row that cannot
produce a relative change is named rather than dropped.
"""

from __future__ import annotations

import math
from typing import Any

from threetears.evals.analysis.reporting import format_significance
from threetears.evals.analysis.viz.intent import ChartAxis, ChartColumn, ChartEncoding, ChartIdentity, ChartIntent
from threetears.evals.analysis.viz.payloads import DeltaRow, DeltaTablePayload, PayloadError
from threetears.evals.analysis.viz.quantities import display_scale, render_cell, signed_with_unit, with_unit


def delta_table_intent(payload: DeltaTablePayload) -> ChartIntent:
    """Decide an A-vs-B comparison: each drawable row's relative change from A, as a length from zero.

    **The axis carries relative change, not the deltas themselves.** A magnitude encoding whose rows each
    use their own scale is not an encoding at all — a 3% difference would draw exactly like a tenfold
    one — so the axis is the unitless change from A, which is the baseline the comparison runs FROM:
    the length is the size of the change and its side is the direction.

    Args:
        payload: The validated comparison payload.

    Returns:
        The intent.
    """
    a_label = payload.a_label or "A"
    b_label = payload.b_label or "B"
    plotted: list[dict[str, Any]] = []
    undrawn: list[str] = []
    for row in payload.rows:
        change = relative_change(row.a, row.b) if row.data_type == "numeric" else None
        if change is None:
            undrawn.append(row.metric)
            continue
        entry: dict[str, Any] = {"metric": row.metric, "change": change, "effect": _effect_read(row)}
        if row.p is not None:
            entry["p"] = row.p
        if row.d_z is not None:
            entry["d_z"] = row.d_z
        plotted.append(entry)

    change_title = f"change vs {a_label}"
    disclosures: list[str] = []
    if immaterial := [row.metric for row in payload.rows if row.materiality == "immaterial"]:
        # Labelled, not hidden: the row is still drawn and tabulated, and this says which changes the
        # host declared too small to act on, so a reader does not act on one.
        disclosures.append(
            f"{len(immaterial)} of {len(payload.rows)} changes are below their measure's materiality threshold "
            f"({', '.join(immaterial)}) — immaterial: too small to act on, however clearly they clear their noise."
        )
    if undrawn:
        # Named, never silently dropped: the table below the chart still counts these rows, so an
        # unexplained gap reads as a rendering fault.
        disclosures.append(
            f"{len(undrawn)} of {len(payload.rows)} metrics are not drawn "
            f"({', '.join(undrawn)}) — a relative axis cannot place a non-numeric value or a change from a zero baseline."
        )
    return ChartIntent(
        type="delta_table",
        title=f"{a_label} vs {b_label}",
        payload=payload.model_dump(mode="json"),
        scale=1.0,
        unit="",
        data=plotted,
        encodings=[
            ChartEncoding(field="metric", role="identity"),
            ChartEncoding(field="change", role="length", axis="change"),
            ChartEncoding(field="effect", role="label"),
        ],
        axes=[ChartAxis(name="change", quantity=change_title, zero_baseline=True)],
        identity=ChartIdentity(
            field="metric", order=[entry["metric"] for entry in plotted], ordered_by="as the payload lists the metrics"
        ),
        direct_labels=True,
        columns=[
            ChartColumn(key="metric", header="Metric"),
            ChartColumn(key="a", header=a_label),
            ChartColumn(key="b", header=b_label),
            ChartColumn(key="delta", header=f"Δ ({b_label}−{a_label})"),
            ChartColumn(key="change", header=change_title),
            ChartColumn(key="effect", header="Effect"),
        ],
        rows=[_delta_row(row) for row in payload.rows],
        disclosures=disclosures,
    )


def _delta_row(row: DeltaRow) -> dict[str, Any]:
    """One comparison row, rendered for the values table in its own metric's unit.

    The restatement ladder is applied PER ROW here, unlike every other chart, and that is the
    one-unit-per-quantity rule rather than an exception to it: each row of this table is a different
    quantity. A latency stated in seconds beside a cost stated in dollars is correct; the same latency
    stated as 16162 ms is not.

    Args:
        row: One metric's A and B values, with whatever statistics it carries.

    Returns:
        The values-table row.
    """
    change = relative_change(row.a, row.b) if row.data_type == "numeric" else None
    if row.data_type != "numeric":
        return {
            "metric": row.metric,
            "a": render_cell(row.a),
            "b": render_cell(row.b),
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
        "a": with_unit(a * scale, unit),
        "b": with_unit(b * scale, unit),
        "delta": signed_with_unit(delta * scale, unit) + (" (immaterial)" if row.materiality == "immaterial" else ""),
        "change": f"{change:+.1%}" if change is not None else None,
        "effect": _effect_read(row),
    }


def _numeric_side(row: DeltaRow, name: str, value: float | str | bool | None) -> float:
    """One side of a numeric row, as the number :class:`DeltaRow` already guarantees it is.

    Args:
        row: The row the value belongs to, named in the refusal.
        name: Which side it is, ``"a"`` or ``"b"``.
        value: The side's value.

    Returns:
        The value as a number.

    Raises:
        PayloadError: The value is not a number — only for a row built around the payload's validation.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PayloadError(f"metric {row.metric!r} is numeric but {name} is {value!r}")
    return value


def _effect_read(row: DeltaRow) -> str:
    """The paired-test read for one row, as words.

    Delegated rather than branched here: this read is the same rule the compare table prints. Pairing is
    read OFF THE ROW — a field's NAME cannot decide what a number is; only the test that produced it can.

    Args:
        row: One metric's comparison row.

    Returns:
        The verdict, with its statistics where the row states any.
    """
    return format_significance(significant=row.significant, paired=row.paired, p=row.p, effect=row.d_z, n=row.n)


def relative_change(a: float | str | bool | None, b: float | str | bool | None) -> float | None:
    """Change from A as a signed fraction of A, or ``None`` where it has no finite value.

    A zero baseline has no relative change — every non-zero B is infinitely far from it — so the caller
    states the two values and the reason rather than drawing both markers at the centre, which reads as
    "no change".

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
    "delta_table_intent",
    "relative_change",
]
