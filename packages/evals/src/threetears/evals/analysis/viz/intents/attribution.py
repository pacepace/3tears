"""The ``attribution`` intent — a whole and a part on one shared signed axis, and what is left over.

The type exists to state a subtraction that may not be earned, so the remainder's row is the one that
matters: it is always present, carrying a number where the subtraction was earned and none where it was
withheld, so the chart records that the question was asked either way.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.viz.intent import ChartAxis, ChartColumn, ChartEncoding, ChartIdentity, ChartIntent
from threetears.evals.analysis.viz.payloads import AttributionPayload
from threetears.evals.analysis.viz.quantities import axis_title, display_scale, signed_with_unit, with_unit

#: The remainder's row — always present, whether or not it carries a number.
REMAINDER_SCOPE = "Unattributed"


def attribution_intent(payload: AttributionPayload) -> ChartIntent:
    """Decide a whole-vs-part movement: each movement a signed length from no change, the remainder last.

    The whole first, then the part, then what is left over: read down the axis it is the finding's own
    sentence. A movement is a LENGTH measured from no change, so the axis starts at zero and the side
    of it carries direction. ``derived`` marks the remainder: computed, not observed.

    Args:
        payload: The validated whole-vs-part payload.

    Returns:
        The intent.
    """
    measured = [("End-to-end", payload.end_to_end), ("Subsystem", payload.subsystem)]
    unattributed = payload.unattributed_delta
    # The display unit is chosen from the MOVEMENTS, not from the levels they run between: a pair of
    # 50 ms deltas between two ~44 s levels restated in seconds rounds to 0.05, a change the reader
    # would take for none. The levels are then stated in that same unit.
    magnitudes = [movement.delta for _, movement in measured]
    if unattributed is not None:
        magnitudes.append(unattributed)
    scale, unit = display_scale(magnitudes, payload.unit)

    plotted: list[dict[str, Any]] = [
        {"scope": scope, "measure": movement.measure, "delta": movement.delta * scale, "derived": False}
        for scope, movement in measured
    ]
    if unattributed is not None:
        plotted.append({"scope": REMAINDER_SCOPE, "measure": "", "delta": unattributed * scale, "derived": True})

    counted = any(movement.n is not None for _, movement in measured)
    columns = [
        ChartColumn(key="scope", header="Scope"),
        ChartColumn(key="measure", header="Measure"),
        ChartColumn(key="a", header=payload.a_label or "A"),
        ChartColumn(key="b", header=payload.b_label or "B"),
        ChartColumn(key="delta", header=axis_title("Δ", unit)),
    ]
    if counted:
        # The two movements have their OWN observation counts, and two different n's side by side are
        # the plainest statement that these are different populations — the whole reason the
        # remainder below them may not be a number.
        columns.append(ChartColumn(key="n", header="Cases"))
    rows: list[dict[str, Any]] = [
        {
            "scope": scope,
            "measure": movement.measure,
            "a": with_unit(movement.a * scale, unit) if movement.a is not None else None,
            "b": with_unit(movement.b * scale, unit) if movement.b is not None else None,
            "delta": signed_with_unit(movement.delta * scale, unit),
            **({"n": movement.n} if counted else {}),
        }
        for scope, movement in measured
    ]
    # An unplaceable remainder is owed `None`; a disclosure carries the reason it cannot be a number.
    rows.append(
        {
            "scope": REMAINDER_SCOPE,
            "measure": None,
            "a": None,
            "b": None,
            "delta": signed_with_unit(unattributed * scale, unit) if unattributed is not None else None,
        }
    )
    return ChartIntent(
        type="attribution",
        title=f"{payload.end_to_end.measure} vs {payload.subsystem.measure}",
        payload=payload.model_dump(mode="json"),
        scale=scale,
        unit=unit,
        data=plotted,
        encodings=[
            ChartEncoding(field="scope", role="identity"),
            ChartEncoding(field="delta", role="length", axis="change"),
            ChartEncoding(field="measure", role="label"),
        ],
        axes=[ChartAxis(name="change", quantity=axis_title("change", unit), unit=unit, zero_baseline=True)],
        # The remainder's band exists either way, so the axis reads the same whether the arithmetic was
        # earned or refused.
        identity=ChartIdentity(
            field="scope",
            order=[scope for scope, _ in measured] + [REMAINDER_SCOPE],
            ordered_by="the whole, then the part, then what is left over",
        ),
        direct_labels=True,
        columns=columns,
        rows=rows,
        disclosures=_attribution_disclosures(payload),
    )


def _attribution_disclosures(payload: AttributionPayload) -> list[str]:
    """State which levels were compared, and what may be said about the remainder.

    The withheld sentence is carried VERBATIM rather than summarised: it names which of the ways a
    subtraction goes wrong applies to this pair, and a reader deciding whether the finding bounds a
    decision needs that reason rather than the fact that there was one.

    Args:
        payload: The whole-vs-part payload, carrying the lever and either a quantified remainder or the
            sentence withholding it.

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
        # Past tense, and deliberately: the containment is a fact the payload CARRIES, recorded when the
        # analysis was generated, and nothing here re-reads the live catalog.
        parts.append(
            f"Unattributed is {payload.end_to_end.measure} minus {payload.subsystem.measure}, "
            f"which the measure catalog declared a component of it when this was generated."
        )
    return parts


__all__ = [
    "REMAINDER_SCOPE",
    "attribution_intent",
]
