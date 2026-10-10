"""Relative change, checked against what it means on the scale it is computed on (#601).

The delta table draws every numeric reading of two cells on one axis of RELATIVE change,
``(b - a) / |a|`` (:func:`~threetears.evals.analysis.viz.intents.delta_table.relative_change`), and a chart
reference may name a judged dimension among its readings. Relative change is meaningful on a ratio scale —
milliseconds, dollars, a share — where zero is "none of it". A judged 1–5 score is an interval scale: its
zero is an arbitrary point below the scale, so a "percent change" in it depends on where the scale happens
to start rather than on the move.

Exact, no simulation: the known answer is that a statistic of an interval-scale reading is unchanged when
the scale is relabelled (``1-5`` written as ``0-4``), and that one point up the scale is one point up
wherever it starts. So a row the measure declares interval-scale (``MetricDescriptor.scale``) states the
points it moved and no relative change.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.viz.intent import ChartIntent
from threetears.evals.analysis.viz.intents.delta_table import delta_table_intent, relative_change
from threetears.evals.analysis.viz.payloads import DeltaRow, DeltaTablePayload
from threetears.evals.kernel.metrics import describe_rubric_dim


def test_on_a_ratio_scale_relative_change_is_the_ratio() -> None:
    """Latency from 2,000 ms to 2,500 ms is 25% more, whatever unit it is written in."""
    assert relative_change(2000.0, 2500.0) == pytest.approx(0.25)
    assert relative_change(2.0, 2.5) == pytest.approx(0.25)


def _table(rows: list[tuple[str, float, float]], scale: str | None) -> ChartIntent:
    return delta_table_intent(
        DeltaTablePayload(
            caption="a judged score moved",
            rows=[DeltaRow(metric=metric, a=a, b=b, unit=None, scale=scale) for metric, a, b in rows],
        )
    )


def test_on_a_judged_scale_one_point_is_one_point_wherever_it_starts() -> None:
    """The #601 finding: a 1–5 score's one-point rise read +50% from 2 to 3 and +25% from 4 to 5, and
    relabelling 1–5 as 0–4 turned 2→3 into +100%. On an interval scale the table states the points moved,
    which is the same wherever the move starts and however the scale is numbered, and states no percent."""
    intent = _table([("low", 2.0, 3.0), ("high", 4.0, 5.0), ("relabelled", 1.0, 2.0)], "interval")
    assert {row["delta"] for row in intent.rows} == {"+1"}
    assert all(row["change"] is None for row in intent.rows)
    assert intent.data == [], "no interval-scale row is drawn on the relative axis"
    assert any("never as a percent" in text for text in intent.disclosures)


def test_a_judged_dimension_reaches_the_table_as_an_interval_scale() -> None:
    """The scale is the measure's own declaration: a 1-5 dimension is interval, a pass/fail one (a rate) is ratio."""
    assert describe_rubric_dim("tone", scale="ordinal").scale == "interval"
    assert describe_rubric_dim("tone", scale="pass_fail").scale == "ratio"


def test_a_ratio_scale_row_is_still_drawn_as_relative_change() -> None:
    intent = _table([("latency", 2000.0, 2500.0)], "ratio")
    assert intent.data[0]["change"] == pytest.approx(0.25)
