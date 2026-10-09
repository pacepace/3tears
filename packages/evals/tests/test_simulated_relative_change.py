"""Relative change, checked against what it means on the scale it is computed on (#601).

The delta table draws every numeric reading of two cells on one axis of RELATIVE change,
``(b - a) / |a|`` (:func:`~threetears.evals.analysis.viz.intents.delta_table.relative_change`), and a chart
reference may name a judged dimension among its readings. Relative change is meaningful on a ratio scale —
milliseconds, dollars, a share — where zero is "none of it". A judged 1–5 score is an interval scale: its
zero is an arbitrary point below the scale, so a "percent change" in it depends on where the scale happens
to start rather than on the move.

Exact, no simulation: the known answer is that a statistic of an interval-scale reading is unchanged when
the scale is relabelled (``1-5`` written as ``0-4``), and that one point up the scale is one point up
wherever it starts.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.viz.intents.delta_table import relative_change


def test_on_a_ratio_scale_relative_change_is_the_ratio() -> None:
    """Latency from 2,000 ms to 2,500 ms is 25% more, whatever unit it is written in."""
    assert relative_change(2000.0, 2500.0) == pytest.approx(0.25)
    assert relative_change(2.0, 2.5) == pytest.approx(0.25)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "#601 finding: the delta table states a 1-5 judged score's movement as relative change, which an interval "
        "scale does not support: the same one-point rise reads +50% from 2 to 3 and +25% from 4 to 5, and "
        "relabelling 1-5 as 0-4 turns 2->3 from +50% into +100%."
    ),
)
def test_on_a_judged_scale_one_point_is_one_point_wherever_it_starts() -> None:
    assert relative_change(2.0, 3.0) == pytest.approx(relative_change(4.0, 5.0))
    assert relative_change(2.0, 3.0) == pytest.approx(relative_change(2.0 - 1.0, 3.0 - 1.0))
