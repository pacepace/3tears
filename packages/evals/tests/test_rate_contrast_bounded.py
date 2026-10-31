"""A rate's family contrast is drawn on its bounded interval, and its verdict needs that interval to exclude zero.

A rate is a reading whose per-case value is a share on 0 to 1: the overall pass rate, a check's or a goal check's pass
rate, a classifier's accuracy, and a judged dimension scored pass or fail. A campaign's reading of the pass rate is the
bounded test's on the same per-case shares, so the contrast is drawn by the same method and the reading's interval, at
its looser level, sits inside the contrast's. The p's stay the t test's, so every adjusted p in the family is Holm's
over the raw t p's whatever the intervals are, and a measure that is no rate keeps its t interval.

The change below drops from eight passing cases of twenty to one, one trial per case: the paired t interval at the
family's level excludes zero, the bounded one does not, so the rate is not separated.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``_corrected_family``: drawing a rate on its t interval again (the old behaviour), so the pass rate reads
  ``regressed`` beside an interval its reading does not nest in.
- ``_is_rate``: reading a judged dimension as a rate whatever its scale, so the ordinal one loses its t interval.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, ComparisonFamily, FamilyComparison
from threetears.evals.analysis.stats import (
    bounded_difference_interval,
    composite_significance,
    difference_interval,
    holm_adjust,
)
from packages.evals.tests.report_shapes_support import ArmTrials, AspectTrials, Shape, shape_of


@pytest.fixture(scope="module")
def drop() -> Shape:
    """A control passing eight cases of twenty and a candidate passing one, one trial per case, on one aspect."""
    return shape_of(
        "rate drop",
        [AspectTrials("billing", [ArmTrials("production", passing=8), ArmTrials("v2", passing=1)])],
        control=True,
    )


@pytest.fixture(scope="module")
def judged() -> Shape:
    """Two arms judged on a pass/fail dimension and an ordinal one, with no checks: a rate and a measure that is none."""
    correct = {"production": [1] * 14 + [0] * 6, "v2": [1] * 6 + [0] * 14}
    clear = {"production": [5, 4, 4, 3, 5] * 4, "v2": [3, 2, 4, 2, 3] * 4}
    return shape_of(
        "judged",
        [
            AspectTrials(
                "tone",
                [
                    ArmTrials(arm, scores={"answer.correct": correct[arm], "answer.clear": clear[arm]})
                    for arm in ("production", "v2")
                ],
                checks=False,
            )
        ],
        control=True,
    )


def _family(bundle: AnalysisContextBundle) -> ComparisonFamily:
    (family,) = bundle.multiple_comparisons.families
    return family


def _one(family: ComparisonFamily, reading: str, name: str) -> FamilyComparison:
    (comparison,) = [c for c in family.comparisons if (c.reading, c.name) == (reading, name)]
    return comparison


def _t_interval(comparison: FamilyComparison, level: float) -> tuple[float, float] | None:
    assert comparison.case_values is not None
    return difference_interval(
        *comparison.case_values, paired=comparison.test == "paired", confidence=level, value_range=(0.0, 1.0)
    )


def test_a_rate_contrast_is_drawn_on_its_bounded_interval_at_the_family_s_level(drop: Shape) -> None:
    family = _family(drop.bundles["billing"])
    comparison = _one(family, "trial", "passed")
    assert family.interval_level is not None and comparison.case_values is not None

    bounded = bounded_difference_interval(
        *comparison.case_values, paired=True, value_range=(0.0, 1.0), confidence=family.interval_level
    )
    assert comparison.interval == comparison.bounded_interval == bounded
    assert comparison.basis == "t", "the p is still the t test's; only the interval it is drawn on is bounded"


def test_a_rate_the_t_interval_separates_and_the_bounded_one_does_not_is_not_separated(drop: Shape) -> None:
    family = _family(drop.bundles["billing"])
    comparison = _one(family, "trial", "passed")
    assert family.interval_level is not None and comparison.interval is not None

    t = _t_interval(comparison, family.interval_level)
    assert t is not None and t[1] < 0.0, "the t interval excludes zero"
    assert comparison.p_adjusted is not None and comparison.p_adjusted < family.alpha, "and the adjusted p is under α"
    assert comparison.interval[0] < 0.0 <= comparison.interval[1], "but the bounded interval reaches zero"
    assert comparison.verdict == "not_separated"


def test_every_rate_in_the_family_follows_its_bounded_interval(drop: Shape) -> None:
    family = _family(drop.bundles["billing"])
    rates = [c for c in family.comparisons if c.reading == "trial" or c.name.startswith("goal_state:")]
    assert {c.reading for c in rates} == {"trial", "measure"}, "the pass rate and its checks' pass rates"

    for comparison in rates:
        if comparison.p_adjusted is None:
            continue
        assert comparison.interval == comparison.bounded_interval, comparison.name
        assert comparison.verdict not in ("improved", "regressed"), comparison.name


def test_a_reading_s_interval_nests_inside_its_family_contrast_s(drop: Shape) -> None:
    family = _family(drop.bundles["billing"])
    contrast = _one(family, "trial", "passed")
    assert contrast.interval is not None and family.interval_level is not None
    low, high = (100.0 * end for end in contrast.interval)
    (reading,) = drop.readings.on_aspect("billing")

    read = [
        (interval, level)
        for interval, level in (
            (reading.gain_interval, reading.gain_confidence),
            (reading.drop_interval, reading.drop_confidence),
        )
        if interval is not None and level is not None
    ]
    assert read, "the reading states its interval"
    for (reading_low, reading_high), level in read:
        assert level <= family.interval_level
        assert low - 1e-9 <= reading_low <= reading_high <= high + 1e-9, (
            f"the reading's {level:.2%} interval [{reading_low}, {reading_high}] lies outside the contrast's "
            f"{family.interval_level:.2%} interval [{low}, {high}]"
        )


def test_a_judged_dimension_scored_pass_or_fail_is_a_rate_and_an_ordinal_one_keeps_its_t_interval(
    judged: Shape,
) -> None:
    family = _family(judged.bundles["tone"])
    assert family.interval_level is not None
    correct = _one(family, "judged", "answer.correct")
    clear = _one(family, "judged", "answer.clear")

    assert correct.interval == correct.bounded_interval
    assert clear.basis == "t" and clear.interval is not None
    assert clear.interval == _t_interval_on(clear, family.interval_level, (1.0, 5.0))
    assert clear.interval != clear.bounded_interval


def _t_interval_on(
    comparison: FamilyComparison, level: float, value_range: tuple[float, float]
) -> tuple[float, float] | None:
    assert comparison.case_values is not None
    return difference_interval(
        *comparison.case_values, paired=comparison.test == "paired", confidence=level, value_range=value_range
    )


@pytest.mark.parametrize("shape", ["drop", "judged"])
def test_every_adjusted_p_is_holm_s_over_the_t_test_s_raw_p_s(shape: str, request: pytest.FixtureRequest) -> None:
    built: Shape = request.getfixturevalue(shape)
    for bundle in built.bundles.values():
        family = _family(bundle)
        tested = [c for c in family.comparisons if c.p_raw is not None]
        raw = [p for c in tested for p in (c.p_raw, c.equivalence_p_raw) if p is not None]
        adjusted = [p for c in tested for p in (c.p_adjusted, c.equivalence_p_adjusted) if p is not None]

        assert adjusted == holm_adjust(raw, max_true=family.family_size)
        for comparison in tested:
            if comparison.basis == "t":
                assert comparison.case_values is not None
                assert comparison.p_raw == composite_significance(*comparison.case_values, paired=True)[2]
