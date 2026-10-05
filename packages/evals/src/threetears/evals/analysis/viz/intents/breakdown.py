"""The ``breakdown`` intent — part-to-whole, the parts sorted largest first and measured from zero."""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.viz.intent import ChartAxis, ChartColumn, ChartEncoding, ChartIdentity, ChartIntent
from threetears.evals.analysis.viz.payloads import BreakdownPayload
from threetears.evals.analysis.viz.quantities import axis_title, display_scale, with_unit


def breakdown_intent(payload: BreakdownPayload) -> ChartIntent:
    """Decide a part-to-whole chart: the parts as lengths from zero, largest first.

    Sorted descending because the reader's question is which part dominates, and a sort makes that a
    glance rather than a scan. Each part is a LENGTH, so its axis starts at zero: a truncated baseline
    draws a ratio the values do not contain.

    Args:
        payload: The validated part-to-whole payload.

    Returns:
        The intent.
    """
    ordered = sorted(payload.parts, key=lambda part: (-part.value, part.label))
    scale, unit = display_scale([part.value for part in ordered], payload.unit)
    rows: list[dict[str, Any]] = [{"label": part.label, "value": part.value * scale, "n": part.n} for part in ordered]
    # The generator's own words for the measure, verbatim — a title assembled around it ("<measure> by
    # part") reads as machine output the moment the measure is itself a phrase.
    title = payload.measure or "Breakdown"
    # Only offer `n` when some part actually carries one; a column of "n: null" states an absence the
    # payload already states by omission.
    counted = any(part.n is not None for part in ordered)
    columns = [ChartColumn(key="label", header="Part"), ChartColumn(key="value", header=axis_title("Value", unit))]
    if counted:
        columns.append(ChartColumn(key="n", header="n"))
    return ChartIntent(
        type="breakdown",
        title=title,
        payload=payload.model_dump(mode="json"),
        scale=scale,
        unit=unit,
        data=rows,
        encodings=[
            ChartEncoding(field="label", role="identity"),
            ChartEncoding(field="value", role="length", axis="value"),
            *([ChartEncoding(field="n", role="label")] if counted else []),
        ],
        axes=[
            ChartAxis(
                name="value", quantity=axis_title(payload.measure or "value", unit), unit=unit, zero_baseline=True
            )
        ],
        identity=ChartIdentity(
            field="label", order=[part.label for part in ordered], ordered_by="value, largest first"
        ),
        direct_labels=True,
        columns=columns,
        rows=rows,
        disclosures=_breakdown_disclosures(payload, scale, unit),
    )


def _breakdown_disclosures(payload: BreakdownPayload, scale: float, unit: str) -> list[str]:
    """State the whole the parts divide, when the payload knows it.

    Without it the bars are magnitudes; with it they are shares. This line is the only place that
    difference is stated, because the chart draws the parts either way. A whole SENTENCE rather than a
    bare fragment like ``total 100% over n=49``, which reads as a rendering fault beside the author's
    prose, stated in the chart's restated unit for the same reason the axis is.

    Args:
        payload: The part-to-whole payload, which may or may not state a whole.
        scale: The unit restatement factor the chart's values took.
        unit: The restated unit.

    Returns:
        The one line, or nothing where the payload states no whole.
    """
    if payload.total is None and payload.total_n is None:
        return []
    total = f"a total of {with_unit(payload.total * scale, unit)}" if payload.total is not None else "a total"
    over = f" over n={payload.total_n}" if payload.total_n is not None else ""
    return [f"The parts divide {total}{over}."]


__all__ = [
    "breakdown_intent",
]
