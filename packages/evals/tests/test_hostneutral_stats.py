"""The one interval the analysis engine reports on a mean, on bare numbers.

:func:`threetears.evals.analysis.stats.ci_half_width` and :data:`threetears.evals.analysis.stats.INTERVAL_LEVEL` own
the width and the level of every interval the engine states. Their input is a standard error
and a count, so no host type or vocabulary can reach it. These tests are the host-neutral
evidence for that ownership: the level and the width are decided in one place, a single
observation has no interval, and small samples take the t multiplier rather than a
large-sample constant.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis import stats
from threetears.evals.analysis.stats import INTERVAL_LEVEL, ci_half_width, t_critical_two_sided


def test_the_reported_level_is_ninety_five_percent() -> None:
    assert INTERVAL_LEVEL == 0.95


@pytest.mark.parametrize("n", [0, 1])
def test_below_two_observations_there_is_no_interval_not_a_zero_width_one(n: int) -> None:
    """A zero half-width would print as a 95% interval around a single point."""
    assert ci_half_width(0.5, n) is None


def test_three_observations_take_the_t_multiplier_not_the_large_sample_constant() -> None:
    """At two degrees of freedom the 95% multiplier is about 4.30, well over twice 1.96."""
    half = ci_half_width(1.0, 3)

    assert half == pytest.approx(4.3027, abs=1e-3)
    assert half > 2 * 1.96


def test_the_width_scales_with_the_standard_error() -> None:
    assert ci_half_width(2.0, 10) == pytest.approx(2 * ci_half_width(1.0, 10))


def test_the_width_is_computed_at_the_level_the_module_declares(monkeypatch: pytest.MonkeyPatch) -> None:
    """Moving the level moves the width, so a caption read off the constant cannot lie."""
    at_default = ci_half_width(1.0, 8)
    monkeypatch.setattr(stats, "INTERVAL_LEVEL", 0.8)

    assert ci_half_width(1.0, 8) == pytest.approx(t_critical_two_sided(0.8, 7))
    assert ci_half_width(1.0, 8) < at_default
