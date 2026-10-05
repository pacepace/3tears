"""The ``breakdown`` arm — part-to-whole as a sorted horizontal bar chart.

Shape only. What the chart says — the order, the unit, the values, the disclosures — arrives decided in
its :class:`~threetears.evals.analysis.viz.intent.ChartIntent`; what lives here is the layout arithmetic
and the marks, gated by :mod:`threetears.evals.vega.spec_policy`, which reads the spec rather
than trusting the producer.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.vega.compiler import (
    VEGA_LITE_SCHEMA,
    MarkValue,
    ValueAxis,
    _bar_mark,
    _Categories,
    _composed,
    _identity,
    _layers,
    _number,
    _title_spec,
    _value_axis,
    _zero_rule,
    value_label_layers,
)
from threetears.evals.analysis.viz.intent import ChartIntent


def compile_breakdown(intent: ChartIntent) -> dict[str, Any]:
    """Draw a part-to-whole intent as a sorted horizontal bar chart.

    Horizontal because the parts are named things and their names are the identity channel — a column
    chart would rotate those labels or truncate them.

    Args:
        intent: The breakdown's intent.

    Returns:
        The Vega-Lite spec.
    """
    rows = intent.data
    axis_title = _value_axis(intent, "value").quantity
    counted = intent.encoding("n") is not None

    categories = _Categories.of("label", _identity(intent).order)
    width, height = categories.plot_size()
    # Zero-based, always: the intent declares the part a length.
    value_axis = ValueAxis.magnitude(axis_title, [_number(row["value"]) for row in rows], width)
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
        MarkValue(
            display=categories.display[str(row["label"])],
            end=_number(row["value"]),
            text=format_number(_number(row["value"])),
        )
        for row in rows
    ]
    return _composed(
        {
            "$schema": VEGA_LITE_SCHEMA,
            "title": _title_spec(intent.title, categories.figure_width()),
            "layer": _layers(_zero_rule(value_axis), bars, *value_label_layers(labels, value_axis, identity)),
            "width": width,
            "height": height,
        },
        categories,
    )


__all__ = [
    "compile_breakdown",
]
