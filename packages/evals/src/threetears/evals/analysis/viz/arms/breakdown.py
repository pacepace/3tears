"""The ``breakdown`` arm — part-to-whole as a sorted horizontal bar chart.

Shape only. Every rule this chart could break as a *spec* is gated by
:mod:`threetears.evals.analysis.viz.policy`, which reads the compiled output rather than
trusting the producer; what lives here is the layout arithmetic and the marks.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.viz.compiler import (
    VEGA_LITE_SCHEMA,
    ChartColumn,
    CompiledChart,
    _axis_title,
    _bar_mark,
    _Categories,
    _composed,
    _layers,
    MarkValue,
    _title_spec,
    value_label_layers,
    ValueAxis,
    _with_unit,
    _zero_rule,
    display_scale,
)
from threetears.evals.analysis.viz.payloads import BreakdownPayload


def compile_breakdown(payload: BreakdownPayload) -> CompiledChart:
    """Compile a part-to-whole payload into a sorted horizontal bar chart.

    Sorted descending because the reader's question is which part dominates, and
    a sort makes that a glance rather than a scan. Horizontal because the parts
    are named things and their names are the identity channel — a column chart
    would rotate those labels or truncate them.

    Args:
        payload: The validated part-to-whole payload.

    Returns:
        The compiled chart.
    """
    ordered = sorted(payload.parts, key=lambda part: (-part.value, part.label))
    scale, unit = display_scale([part.value for part in ordered], payload.unit)
    rows: list[dict[str, Any]] = [{"label": part.label, "value": part.value * scale, "n": part.n} for part in ordered]
    # The generator's own words for the measure, verbatim — the report renders its
    # prose everywhere else, and a title assembled around it ("<measure> by part")
    # reads as machine output the moment the measure is itself a phrase.
    title = payload.measure or "Breakdown"
    axis_title = _axis_title(payload.measure or "value", unit)
    # Only offer `n` in the tooltip when some part actually carries one; a column of
    # "n: null" states an absence the payload already states by omission.
    counted = any(part.n is not None for part in ordered)

    categories = _Categories.of("label", [part.label for part in ordered])
    width, height = categories.plot_size()
    # Zero-based, always. A bar encodes magnitude by length, so a truncated baseline
    # draws a ratio the values do not contain.
    value_axis = ValueAxis.magnitude(axis_title, [row["value"] for row in rows], width)
    identity = categories.axis(domain=not value_axis.marks_form_the_edge())
    bars: dict[str, Any] = {
        "data": {"values": categories.labelled(rows)},
        "mark": _bar_mark(tooltip=True),
        "encoding": {
            "y": identity,
            "x": value_axis.encoding("value"),
            "tooltip": [
                {"field": "label", "type": "nominal", "title": "Part"},
                {"field": "value", "type": "quantitative", "title": axis_title},
                *([{"field": "n", "type": "quantitative", "title": "n"}] if counted else []),
            ],
        },
    }
    labels = [
        MarkValue(display=categories.display[row["label"]], end=row["value"], text=format_number(row["value"]))
        for row in rows
    ]
    spec: dict[str, Any] = _composed(
        {
            "$schema": VEGA_LITE_SCHEMA,
            "title": _title_spec(title, categories.figure_width()),
            "layer": _layers(_zero_rule(value_axis), bars, *value_label_layers(labels, value_axis, identity)),
            "width": width,
            "height": height,
        },
        categories,
    )
    columns: list[ChartColumn] = [
        {"key": "label", "header": "Part"},
        {"key": "value", "header": _axis_title("Value", unit)},
    ]
    if counted:
        columns.append({"key": "n", "header": "n"})
    return CompiledChart(
        spec=spec,
        columns=columns,
        rows=rows,
        unit=unit,
        disclosures=_breakdown_disclosures(payload, scale, unit),
        title=title,
    )


def _breakdown_disclosures(payload: BreakdownPayload, scale: float, unit: str) -> list[str]:
    """State the whole the parts divide, when the payload knows it.

    Without it the bars are magnitudes; with it they are shares. This line is the
    only place that difference is stated, because the chart draws the parts either
    way.

    A whole SENTENCE rather than a bare fragment like ``total 100% over n=49``,
    which reads as a rendering fault beside the author's prose. The number is
    stated in the chart's restated unit for the same reason the axis is — one
    quantity, one ruler.

    Args:
        payload: The part-to-whole payload, which may or may not state a whole.
        scale: The unit restatement factor the chart's values took.
        unit: The restated unit.

    Returns:
        The one line, or nothing where the payload states no whole.
    """
    if payload.total is None and payload.total_n is None:
        return []
    total = f"a total of {_with_unit(payload.total * scale, unit)}" if payload.total is not None else "a total"
    over = f" over n={payload.total_n}" if payload.total_n is not None else ""
    return [f"The parts divide {total}{over}."]


__all__ = [
    "compile_breakdown",
]
