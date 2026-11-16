"""The quick path's re-readable comparison: a newcomer adds cases and reads again, as often as they like, and the error
each reading states still holds (owner ruling, 2026-11-17).

The quick path files its comparison ``rereadable`` (``AspectDesign.rereadable``), and each paired contrast on a
declared range is read by the mixture bet (:func:`~threetears.evals.analysis.stats.bounded_separation_p` and
:func:`~threetears.evals.analysis.stats.bounded_difference_interval` with ``mixture``): the mean of ten bets, each
staking a fixed share ``j / 10`` of the cap on every pair, over the pairs in the order the cases were added, hedged
over the two directions. No stake depends on a declared count, so every reading is one capital read again. Held here:

- the decisions are the simulated construction's (``looks-bet-simulation-1116.md``, "Mixture stakes"), decision for
  decision on fixed seeds, and the interval excludes zero exactly where the test separates;
- under the null, re-reading as cases are added (adds sized from what the reading showed, up to five reads)
  separates at most α of the time, for two arms and for three arms under Holm's correction;
- at one stated alternative, a single read's power against the reading it replaces;
- the rung 3 comparison prints the one sentence that says so, and reads cost by its t test, once.

Each simulation is seeded, so a run is one fixed draw; its docstring states the rate it holds and the bound it is
checked against.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from threetears.evals.analysis.bundle.comparisons import REREAD_SENTENCE
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    bounded_difference_interval,
    bounded_separation_p,
    composite_significance,
    exact_rate_interval,
    holm_adjust,
)
from threetears.evals.kernel.declaration import AspectDesign

from packages.evals.tests.factories import minimal_aspect_design
from packages.evals.tests.ladder_support import load, offline_copy

#: The pass rate's range, which the per-case shares and their differences are read on.
_RATE = (0.0, 1.0)


def _pair(rng: random.Random, k: int, control: float, arm: float) -> tuple[float, float]:
    """One case's pass share on each arm: ``k`` trials each, every trial passing at its arm's rate."""
    return (
        sum(rng.random() < control for _ in range(k)) / k,
        sum(rng.random() < arm for _ in range(k)) / k,
    )


def _separates(control: list[float], arm: list[float]) -> bool:
    p = bounded_separation_p(control, arm, paired=True, value_range=_RATE, mixture=True)
    assert p is not None
    return p < SIGNIFICANCE_ALPHA


# --- the construction ------------------------------------------------------------------------------------------------


def _reference_crossing(diffs: list[float]) -> tuple[int, int]:
    """The simulated construction, ``reference-mix.py``'s ``first_crossing`` on its ``mix_grid`` arm, one row at a time
    in plain Python: z = (d + 1) / 2, null 1/2, ten fractions λ_j = (j / 10) · 0.9 / 0.5, each direction's log capital a
    cumulative sum of log1p, the ten combined by successive logaddexp less log 10, hedged as their logaddexp less log
    2, and the first index at which the hedged log capital reaches log(1/α); the direction there the larger side.

    :return: ``(index, direction)``: the 0-based index of the first crossing, or ``len(diffs)`` where none, and +1 up,
        -1 down, 0 none.
    """
    sides = []
    for zz in ([(d + 1) / 2 for d in diffs], [1 - (d + 1) / 2 for d in diffs]):
        acc: list[float] | None = None
        for j in range(1, 11):
            lam = j / 10 * 0.9 / 0.5
            running, path = 0.0, []
            for z in zz:
                running += math.log1p(lam * (z - 0.5))
                path.append(running)
            if acc is None:
                acc = path
            else:
                acc = [max(a, b) + math.log1p(math.exp(-abs(a - b))) for a, b in zip(acc, path, strict=True)]
        assert acc is not None
        sides.append([value - math.log(10) for value in acc])
    up, down = sides
    for index, (u, d) in enumerate(zip(up, down, strict=True)):
        hedged = max(u, d) + math.log1p(math.exp(-abs(u - d))) - math.log(2)
        if hedged >= math.log(1 / SIGNIFICANCE_ALPHA):
            return index, 1 if u >= d else -1
    return len(diffs), 0


def _sides(diffs: list[float]) -> tuple[list[float], list[float]]:
    """Two arms on the 0-1 range whose paired differences are ``diffs``."""
    return [max(0.0, -d) for d in diffs], [max(0.0, d) for d in diffs]


@pytest.mark.parametrize(("k", "control", "arm"), [(1, 0.5, 0.5), (1, 0.8, 0.6), (3, 0.6, 0.75), (3, 0.9, 0.9)])
def test_decisions_agree_with_the_simulated_construction_read_for_read_on_fixed_seeds(
    k: int, control: float, arm: float
) -> None:
    """Every read of 60 seeded rows a cell, at 20, 40, 60 and 80 pairs, separates exactly where the construction's
    hedged capital first crossed 1/α within those pairs, in its direction; and the interval at 95% excludes zero
    exactly where the test separates, on the side the construction crossed to."""
    rng = random.Random(f"rereadable-construction-{k}-{control}-{arm}")
    for _ in range(60):
        draws = [_pair(rng, k, control, arm) for _ in range(80)]
        diffs = [b - a for a, b in draws]
        crossed, direction = _reference_crossing(diffs)
        for read in (20, 40, 60, 80):
            low_side, high_side = _sides(diffs[:read])
            separated = _separates(low_side, high_side)
            interval = bounded_difference_interval(
                low_side, high_side, paired=True, value_range=_RATE, confidence=0.95, mixture=True
            )
            assert separated == (crossed <= read - 1)
            assert interval is not None
            side = 1 if interval[0] > 0 else -1 if interval[1] < 0 else 0
            assert side == (direction if separated else 0)


def test_a_reading_read_again_after_cases_are_added_is_the_same_capital_read_later() -> None:
    """A case appended is bet on after every case before it: the p over more cases is never above the p over fewer,
    and the interval over more lies inside the one over fewer, because each is one capital read later."""
    rng = random.Random("rereadable-prefix")
    draws = [_pair(rng, 3, 0.7, 0.8) for _ in range(90)]
    control, arm = [a for a, _ in draws], [b for _, b in draws]
    ps, intervals = [], []
    for read in (15, 30, 45, 60, 75, 90):
        ps.append(bounded_separation_p(control[:read], arm[:read], paired=True, value_range=_RATE, mixture=True))
        intervals.append(
            bounded_difference_interval(control[:read], arm[:read], paired=True, value_range=_RATE, mixture=True)
        )
    assert all(later <= earlier for earlier, later in zip(ps, ps[1:]))  # type: ignore[operator]
    for earlier, later in zip(intervals, intervals[1:]):
        assert earlier is not None and later is not None
        assert earlier[0] <= later[0] + 1e-9 and later[1] <= earlier[1] + 1e-9


def test_the_mixture_reads_paired_and_for_no_declared_count() -> None:
    with pytest.raises(ValueError, match="paired"):
        bounded_separation_p([0.0, 1.0], [1.0, 1.0], paired=False, value_range=_RATE, mixture=True)
    with pytest.raises(ValueError, match="planned_n"):
        bounded_difference_interval([0.0, 1.0], [1.0, 1.0], paired=True, value_range=_RATE, planned_n=4, mixture=True)


# --- error under re-reads --------------------------------------------------------------------------------------------


def _reread_until_separated(rng: random.Random, k: int, rate: float, arms: int) -> bool:
    """One newcomer under the null (every arm passing at ``rate``): launch 20 cases, read every arm against the
    control, and while nothing separates add cases and read again, at most five reads in all. Each add is sized from
    the reading: ten cases, plus up to ninety more the closer the smallest p came to α. Several arms are read under
    Holm's correction over their contrasts, as the quick path's family is."""
    cases: list[tuple[float, ...]] = []
    n = 20
    for _ in range(5):
        while len(cases) < n:
            cases.append(tuple(sum(rng.random() < rate for _ in range(k)) / k for _ in range(arms)))
        control = [case[0] for case in cases]
        raw = [
            bounded_separation_p(control, [case[i] for case in cases], paired=True, value_range=_RATE, mixture=True)
            for i in range(1, arms)
        ]
        ps = [p for p in raw if p is not None]
        if any(p < SIGNIFICANCE_ALPHA for p in holm_adjust(ps)):
            return True
        n += 10 + int(90 * (1.0 - min(1.0, min(ps))))
    return False


@pytest.mark.parametrize(("k", "rate", "arms", "seeded_at"), [(1, 0.5, 2, 66), (3, 0.8, 2, 24), (1, 0.8, 3, 53)])
def test_re_reading_as_cases_are_added_separates_under_the_null_at_most_alpha(
    k: int, rate: float, arms: int, seeded_at: int
) -> None:
    """2,000 seeded newcomers a cell, each reading up to five times with adds sized from the data, separate no more
    often than α = 5%: the Clopper-Pearson 95% upper bound on the rate is below α. The seeded draw gives 66 of 2,000
    at k = 1, base 0.5 (3.3%, upper bound 4.2%); 24 at k = 3, base 0.8 (1.2%, upper 1.8%); and 53 for three arms at
    k = 1, base 0.8 under Holm (2.7%, upper 3.5%). The counts are pinned to two either side, so a change to the bet
    that moves them is seen; the bound is the claim."""
    rng = random.Random(f"rereadable-null-{k}-{rate}-{arms}")
    replicates = 2000
    separated = sum(_reread_until_separated(rng, k, rate, arms) for _ in range(replicates))
    bounds = exact_rate_interval(separated, replicates)
    assert bounds is not None and bounds[1] < SIGNIFICANCE_ALPHA
    assert abs(separated - seeded_at) <= 2


# --- power on a single read ------------------------------------------------------------------------------------------


def test_a_single_read_gives_up_some_power_against_the_reading_it_replaces() -> None:
    """At k = 1, the control passing 80% and the arm 60%, on 60 paired cases read once, 2,000 seeded draws: the
    mixture separates on 742 (37.1%), where the reading it replaces (the paired t test's p below α and the bounded
    test's 95% interval, fixed for those 60 cases, excluding zero) separates on 872 (43.6%). That is the price of
    reading again freely: six and a half points of power at this alternative, held to no more than fifteen."""
    rng = random.Random("rereadable-power")
    mixture = one_shot = 0
    replicates = 2000
    for _ in range(replicates):
        draws = [_pair(rng, 1, 0.8, 0.6) for _ in range(60)]
        control, arm = [a for a, _ in draws], [b for _, b in draws]
        mixture += _separates(control, arm)
        t_p = composite_significance(control, arm, paired=True).p_value
        fixed = bounded_difference_interval(control, arm, paired=True, value_range=_RATE)
        one_shot += bool(
            t_p is not None and t_p < SIGNIFICANCE_ALPHA and fixed is not None and (fixed[0] > 0 or fixed[1] < 0)
        )
    assert mixture < one_shot <= mixture + 0.15 * replicates
    assert abs(mixture - 742) <= 10 and abs(one_shot - 872) <= 10


# --- the quick path --------------------------------------------------------------------------------------------------


async def test_rung_3_s_comparison_is_read_by_the_bet_and_says_so_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rung 3a, replayed offline: the print says the one sentence under its first line; every paired reading on a range
    is the bet's, and cost is read by its t test, once; a bet reading's adjusted p is Holm's over the family with the t
    reading's p held at 1, so no t p lets a bet reading step down."""
    offline_copy("3-invoices", tmp_path, monkeypatch)
    module = load(Path("compare.py"), "compare", monkeypatch)
    result = await module.compare.run(store=":memory:")

    printed = str(result).splitlines()
    assert printed[1] == REREAD_SENTENCE and str(result).count(REREAD_SENTENCE) == 1
    (family,) = result.report.bundle.multiple_comparisons.families
    assert family.rereadable and REREAD_SENTENCE in family.disclosure
    on_t = [comparison for comparison in family.comparisons if comparison.basis == "t"]
    on_bet = [comparison for comparison in family.comparisons if comparison.basis == "bounded"]
    assert [comparison.name for comparison in on_t] == ["production_replicating_cost"]
    assert {comparison.interval_basis for comparison in on_bet} == {"bounded"} and len(on_bet) == 5
    raw = [comparison.p_raw for comparison in family.comparisons]
    held = holm_adjust([1.0 if comparison.basis == "t" else p for comparison, p in zip(family.comparisons, raw)])
    for comparison, adjusted in zip(family.comparisons, held):
        if comparison.basis == "bounded":
            assert comparison.p_adjusted == pytest.approx(adjusted)


def test_a_design_is_rereadable_or_plans_its_looks_never_both() -> None:
    assert "rereadable" not in minimal_aspect_design().model_dump(mode="json")
    assert minimal_aspect_design(rereadable=True).model_dump(mode="json")["rereadable"] is True
    with pytest.raises(ValueError, match="rereadable or plans its looks"):
        minimal_aspect_design(rereadable=True, planned_looks=3)
    assert AspectDesign.model_fields["rereadable"].default is False
