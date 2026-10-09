"""An interval stays inside the scale its measure is declared on, and always contains its own estimate.

Two defects drew these lines. A perfect rate's Wilson upper bound came out a hair under 1.0 — 4 of 4
gave ``0.9999999999999999`` — so the drawn interval missed its own estimate by one ulp, the chart
payload refused it, and every perfect label's precision and recall chart was dropped from the report
(those figures are now the code-only report's per-label table, held here the same way).
And ``accuracy``, a 0/1 measure derived from ``match``, took the symmetric t interval, so 8 of 10
read ``[0.498, 1.102]`` while ``match`` over the same observations read its Wilson ``[0.490, 0.943]``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import campaign_report, report_markdown
from threetears.evals.analysis.stats import mean_interval, observed_mean_interval, wilson_interval
from threetears.evals.analysis.viz.payloads import ConfidenceInterval
from threetears.evals.ops import CampaignDefinition, campaign_create
from threetears.evals.quick import callable_host, run_eval
from threetears.evals.run import list_results
from packages.evals.tests.bundle_support import one_batch_bundle

SCOPE = "bounded-interval-tests"


# =============================================================================
# The Wilson interval contains its rate exactly
# =============================================================================


@pytest.mark.parametrize("n", range(1, 61))
def test_a_perfect_rate_has_an_upper_bound_of_exactly_one(n: int) -> None:
    interval = wilson_interval(n, n)
    assert interval is not None
    assert interval[1] == 1.0


@pytest.mark.parametrize("n", range(1, 61))
def test_a_zero_rate_has_a_lower_bound_of_exactly_zero(n: int) -> None:
    interval = wilson_interval(0, n)
    assert interval is not None
    assert interval[0] == 0.0


@pytest.mark.parametrize("n", range(1, 41))
def test_every_wilson_interval_contains_its_rate(n: int) -> None:
    for n_true in range(n + 1):
        interval = wilson_interval(n_true, n)
        assert interval is not None
        low, high = interval
        assert 0.0 <= low <= n_true / n <= high <= 1.0, (n_true, n)


def test_four_of_four_is_drawable_as_a_chart_interval() -> None:
    """The exact case the report dropped: 4/4, whose upper bound was ``0.9999999999999999``."""
    interval = wilson_interval(4, 4)
    assert interval is not None
    ConfidenceInterval(low=interval[0], high=interval[1], mean=4 / 4, level=0.95, variability="across 4 cases")


# =============================================================================
# A numeric mean's interval stays inside the declared scale
# =============================================================================


def test_a_mean_interval_is_clipped_to_the_declared_scale() -> None:
    unclipped = mean_interval(4.8, 0.3, 5)
    clipped = mean_interval(4.8, 0.3, 5, value_range=(1.0, 5.0))
    assert unclipped is not None and clipped is not None
    assert unclipped[1] > 5.0
    assert clipped == (unclipped[0], 5.0)


def test_a_mean_interval_with_no_declared_scale_is_the_symmetric_t_interval() -> None:
    interval = mean_interval(10.0, 1.0, 4)
    assert interval is not None
    assert interval[0] + interval[1] == pytest.approx(20.0)


def test_zero_one_observations_on_a_unit_scale_take_the_proportions_wilson_interval() -> None:
    values = [1.0] * 8 + [0.0] * 2
    assert observed_mean_interval(values, value_range=(0.0, 1.0)) == wilson_interval(8, 10)


def test_zero_one_observations_on_no_declared_scale_keep_the_t_interval() -> None:
    """Only a declared unit scale makes 0/1 values trials; an undeclared measure is not read as a proportion."""
    interval = observed_mean_interval([1.0] * 8 + [0.0] * 2)
    assert interval is not None
    assert interval[1] > 1.0


def test_continuous_values_on_a_unit_scale_take_the_clipped_t_interval() -> None:
    values = [1.0, 1.0, 0.9, 1.0, 0.95]
    interval = observed_mean_interval(values, value_range=(0.0, 1.0))
    assert interval is not None
    low, high = interval
    assert high == 1.0
    assert low <= sum(values) / len(values) <= high


@pytest.mark.parametrize("values", [[], [1.0]], ids=["none", "one"])
def test_below_two_observations_a_numeric_mean_has_no_interval(values: list[float]) -> None:
    assert observed_mean_interval(values, value_range=(0.0, 1.0)) is None


# =============================================================================
# End to end: run_eval → campaign_report
# =============================================================================

_CASES = [
    {"text": text, "expected": expected}
    for text, expected in [
        ("The delivery came two days late and the box was crushed.", "negative"),
        ("Exactly what I ordered, and it arrived early.", "positive"),
        ("It works.", "neutral"),
        ("Great price, terrible battery.", "negative"),
        ("Not bad at all.", "positive"),
    ]
]


async def _strict(case: Mapping[str, Any]) -> str:
    words = str(case["text"]).lower()
    if any(word in words for word in ("late", "terrible")):
        return "negative"
    return "positive" if "great" in words else "neutral"


async def _loose(case: Mapping[str, Any]) -> str:
    words = str(case["text"]).lower()
    if "late" in words:
        return "negative"
    return "positive" if any(word in words for word in ("exactly", "bad", "great")) else "neutral"


def _expected(case: Mapping[str, Any]) -> str:
    return str(case["expected"])


async def test_a_two_classifier_campaign_report_draws_every_interval_inside_its_scale() -> None:
    host = callable_host()
    strict = await run_eval(_CASES, _strict, scope_id=SCOPE, expected=_expected, host=host, k=2)
    loose = await run_eval(_CASES, _loose, scope_id=SCOPE, expected=_expected, host=host, k=2)

    # The bundle: accuracy states the very interval match does, and none exceeds the unit scale.
    bundle = one_batch_bundle(list_results(host.storage, loose.run_id, SCOPE), profile=host.profile)
    (cell,) = bundle.cell_measures
    measures = {measure.name: measure for measure in cell.measures.measures}
    accuracy, match = measures["accuracy"], measures["match"]
    assert accuracy.mean == pytest.approx(0.8)
    assert (accuracy.ci_low, accuracy.ci_high) == (match.ci_low, match.ci_high)
    assert accuracy.ci_high is not None and accuracy.ci_high <= 1.0

    campaign = campaign_create(
        host,
        CampaignDefinition(
            name="strict vs loose", subject_id="sentiment", behavior="classify", run_ids=[strict.run_id, loose.run_id]
        ),
        SCOPE,
        created_by="test",
    )
    markdown = report_markdown(campaign_report(host, campaign.id, SCOPE))

    assert "cannot be drawn" not in markdown
    # Every label's precision and recall is in the per-label table with its interval, each inside [0, 1] and
    # around its own rate — a perfect label's included.
    per_label = markdown.split("**Per-label precision, recall and F1**", 1)[1].split("\n\n>", 1)[0]
    negative = [row.split("|") for row in per_label.splitlines() if row.startswith("| negative |")]
    assert len(negative) == 2 and all("[" in row[3] and "[" in row[4] for row in negative)
    figures = re.findall(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", per_label)
    assert figures and all(0.0 <= float(low) <= float(rate) <= float(high) <= 1.0 for rate, low, high in figures)
    accuracy_chart = markdown.split("**Chart: accuracy**", 1)[1].split("**Per-label", 1)[0]
    highs = [float(row.split("|")[4]) for row in accuracy_chart.splitlines() if row.startswith("| model=")]
    assert len(highs) == 2
    assert all(high <= 1.0 for high in highs)
