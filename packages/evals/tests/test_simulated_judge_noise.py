"""Re-score noise against a planted truth: the judge's share of a contrast's width, and its ceiling on pass^k.

Seeded, 2,000 replicates per row, through the functions the bundle's filler calls
(:mod:`threetears.evals.analysis.judge_noise`: :func:`rescore_of` for an arm's figures, :func:`judge_share` for a
contrast's). The within-tolerance bars were fixed before the first run; the range bars were set from a measured run of
the interval below and checked to fail for each wrong range named after the table.

**The share.** Each replicate draws ``G`` cases and ``k`` trials per case per arm. A trial's answer is an integer
whose distribution differs by case and arm; the judge scores it with noise of a known variance: on 1 to 5, the answer
plus -1, 0 or +1 with chances ``p/2``, ``1 - p``, ``p/2`` (answers kept in 2 to 4, so the variance is exactly ``p``);
on pass/fail, the answer flipped with chance ``e`` (variance ``e (1 - e)``). Twenty of each arm's results, from at
least ten cases (the floor), are scored again by the same judge. The planted share is the true re-score variance
over the variance of this replicate's per-case differences: the contrast's own width is what it is, and the share
says how much of it the judge's noise is. A row passes when the estimate is within the scale's tolerance (C-12) at
least as often as its bar, and when each range, the share's and each arm's variance's, covers its planted value at
least as often as its low bar, no more often than its high bar, and is on average no wider, over that value, than its
width bar. The ranges are at the 95% level (``INTERVAL_LEVEL``); the low bar 0.93 leaves room for the Wilson and
Hilferty quantile and the cluster standard error at twenty results.

| Row | Scale, noise | G | k | Repeated results | Within tolerance: bar (measured) | Share's range covers: bars (measured); width: bar (measured) | Arm's range covers: bars (measured); width: bar (measured) |
|---|---|---|---|---|---|---|---|
| 1 | 1-5, p = 0.5 | 20 | 1 | 20 | ±30%: 0.80 (0.876) | 0.93-0.99 (0.977); 1.05 (0.93) | 0.93-0.99 (0.969); 1.45 (1.30) |
| 2 | 1-5, p = 0.5 | 60 | 3 | 20 | ±30%: 0.80 (0.884) | 0.93-0.99 (0.974); 1.05 (0.93) | 0.93-0.99 (0.974); 1.45 (1.30) |
| 3 | 1-5, p = 0.5 | 100 | 3 | 200 | ±30%: 0.98 (1.000) | 0.93-0.99 (0.982); 0.32 (0.28) | 0.93-0.99 (0.976); 0.45 (0.40) |
| 4 | pass/fail, e = 0.05 | 20 | 1 | 20 | ±70%: 0.75 (0.813) | 0.93-0.99 (0.960); 2.5 (2.22) | 0.93-0.995 (0.989); 3.7 (3.36) |
| 5 | pass/fail, e = 0.05 | 60 | 3 | 20 | ±70%: 0.75 (0.835) | 0.93-0.99 (0.969); 2.5 (2.22) | 0.93-0.995 (0.980); 3.7 (3.36) |

The tolerance is about one standard error of one arm's variance at the floor (twenty half squared differences on 1 to
5; about two flips on pass/fail); a share reads two arms' variances, so it holds more often than one standard error
would. The bar falls to near zero for a wrong formula: a variance not halved reads twice the truth, and a share without
``1/k`` reads three times it at k = 3 (rows 2, 3 and 5). Rows 1 and 4 hold the share at k = 1, where only repeats can
measure the judge. Checked by mutation (400 replicates): a variance not halved met the 1-5 rows' bar on 0.03 of
replicates and row 5's on 0.47; a share without ``1/k`` on 0.00 and 0.27. The arm's pass/fail high bar is 0.995: on
about two flips the range is Garwood's on a count, which covers more than its level by construction. Each range's
bars fail for a wrong range (measured on these seeds): the share taken at each end of the arms' ranges covered
0.998-1.000 on 1 to 5 and was 1.53, 0.40 and 4.4 times its planted value wide; an arm's variance-form chi-square
range covered 0.883 and 0.911 on pass/fail and was 1.53 wide at the 1-5 floor; degrees of freedom not held to one per
result covered 0.903 and 0.917 (rows 1 and 2, the arm); a range of zero to the most a score can vary covered 1.000.

**The ceiling.** ``e`` is planted on a judge that fails a passing answer with chance ``e`` and never passes a failing
one, so a repeat of a first reading that passed fails with chance exactly ``e``. Its Wilson interval over cases covers
``e``, and so the ceiling range covers ``(1 - e)^(k - 1)``, at least 0.93 and at most 0.99 at k = 3 over 100 cases, at
the floor and at 200 repeated results (measured 0.958, 0.964 and 0.971 at the floor for e = 0.05, 0.10 and 0.20; 0.972, 0.968 and 0.962
at 200). The review's table (B-3) is held exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from threetears.evals.analysis.judge_noise import (
    RESCORE_MIN_CASES,
    RESCORE_MIN_RESULTS,
    Repeated,
    SideNoise,
    judge_share,
    pass_k_ceiling,
    rescore_of,
)

_REPLICATES = 2_000

#: The 1 to 5 scale, and pass/fail, as the filler reads them.
_ORDINAL = (1.0, 5.0)
_PASS_FAIL = (0.0, 1.0)


@dataclass(frozen=True)
class _Bars:
    """How often a range must cover its planted value, from both sides, and how wide it may be on average."""

    covers_at_least: float
    covers_at_most: float
    width_at_most: float


@dataclass(frozen=True)
class _Row:
    """One row of the share's table."""

    scale: str
    noise: float
    cases: int
    k: int
    repeats: int
    tolerance: float
    within_bar: float
    share: _Bars
    arm: _Bars

    @property
    def planted_variance(self) -> float:
        return self.noise if self.scale == "ordinal" else self.noise * (1.0 - self.noise)


@dataclass
class _Measured:
    """What a row measured: within tolerance, and each range's coverage and mean width over its planted value."""

    within: float = 0.0
    share_covers: float = 0.0
    share_width: float = 0.0
    arm_covers: float = 0.0
    arm_width: float = 0.0


def _score(rng: random.Random, row: _Row, answer: int) -> int:
    """One judge reading of ``answer``."""
    if row.scale == "ordinal":
        return answer + rng.choices((-1, 0, 1), (row.noise / 2, 1.0 - row.noise, row.noise / 2))[0]
    return 1 - answer if rng.random() < row.noise else answer


def _answer(rng: random.Random, row: _Row, centre: float) -> int:
    """One trial's answer near a case's centre: 2 to 4 on 1 to 5, a pass with chance ``centre`` on pass/fail."""
    if row.scale == "ordinal":
        return min(4, max(2, round(centre + rng.gauss(0.0, 0.6))))
    return 1 if rng.random() < centre else 0


def _arm(rng: random.Random, row: _Row, centres: list[float]) -> tuple[list[tuple[str, int, int]], SideNoise | None]:
    """One arm: every trial ``(case, answer, first reading)``, and its side as the share reads it."""
    trials = [
        (f"c{case}", answer, _score(rng, row, answer))
        for case, centre in enumerate(centres)
        for answer in (_answer(rng, row, centre) for _ in range(row.k))
    ]
    while True:
        chosen = rng.sample(range(len(trials)), row.repeats)
        if len({trials[index][0] for index in chosen}) >= RESCORE_MIN_CASES:
            break
    repeated = [
        Repeated(trials[index][0], float(trials[index][2]), (float(_score(rng, row, trials[index][1])),))
        for index in chosen
    ]
    _, variance = rescore_of(
        repeated, value_range=_ORDINAL if row.scale == "ordinal" else _PASS_FAIL, k=row.k, passed=None
    )
    side = None if variance is None else SideNoise(variance, {f"c{case}": row.k for case in range(len(centres))})
    return trials, side


def _case_means(trials: list[tuple[str, int, int]]) -> dict[str, float]:
    sums: dict[str, list[int]] = {}
    for case, _, reading in trials:
        sums.setdefault(case, []).append(reading)
    return {case: sum(values) / len(values) for case, values in sums.items()}


def _share_row(row: _Row, seed: str) -> _Measured:
    """Each replicate's share against the planted share, and each arm's variance range against the planted variance.

    A replicate whose per-case differences have no spread (pass/fail at k = 1, rarely) has no share; it is not counted.
    """
    rng = random.Random(seed)
    measured = _Measured()
    shares = arms = 0
    for _ in range(_REPLICATES):
        if row.scale == "ordinal":
            control = [rng.uniform(2.3, 3.7) for _ in range(row.cases)]
            candidate = [centre + rng.gauss(0.2, 0.3) for centre in control]
        else:
            control = [rng.uniform(0.5, 0.95) for _ in range(row.cases)]
            candidate = [min(0.99, max(0.01, centre + rng.gauss(0.05, 0.1))) for centre in control]
        control_trials, control_side = _arm(rng, row, control)
        candidate_trials, candidate_side = _arm(rng, row, candidate)
        assert control_side is not None and candidate_side is not None
        for side in (control_side, candidate_side):
            arms += 1
            measured.arm_covers += side.variance.low <= row.planted_variance <= side.variance.high
            measured.arm_width += (side.variance.high - side.variance.low) / row.planted_variance
        before, after = _case_means(control_trials), _case_means(candidate_trials)
        cases = sorted(before)
        differences = [after[case] - before[case] for case in cases]
        share = judge_share(control_side, candidate_side, cases, differences)
        if share is None:
            continue
        shares += 1
        mean = sum(differences) / len(differences)
        spread = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
        planted = row.planted_variance * (2.0 / row.k) / spread
        measured.within += abs(share.point - planted) <= row.tolerance * planted
        measured.share_covers += share.low <= planted <= share.high
        measured.share_width += (share.high - share.low) / planted
    assert shares >= 0.99 * _REPLICATES, f"only {shares} replicates had a share"
    measured.within /= shares
    measured.share_covers /= shares
    measured.share_width /= shares
    measured.arm_covers /= arms
    measured.arm_width /= arms
    return measured


#: The bars of the table above.
_ORDINAL_SHARE, _ORDINAL_ARM = _Bars(0.93, 0.99, 1.05), _Bars(0.93, 0.99, 1.45)
_PASS_FAIL_SHARE, _PASS_FAIL_ARM = _Bars(0.93, 0.99, 2.5), _Bars(0.93, 0.995, 3.7)
_SHARE_ROWS = {
    "1-5, G 20, k 1, at the floor": _Row(
        "ordinal", 0.5, 20, 1, RESCORE_MIN_RESULTS, 0.30, 0.80, _ORDINAL_SHARE, _ORDINAL_ARM
    ),
    "1-5, G 60, k 3, at the floor": _Row(
        "ordinal", 0.5, 60, 3, RESCORE_MIN_RESULTS, 0.30, 0.80, _ORDINAL_SHARE, _ORDINAL_ARM
    ),
    "1-5, G 100, k 3, 200 repeats": _Row(
        "ordinal", 0.5, 100, 3, 200, 0.30, 0.98, _Bars(0.93, 0.99, 0.32), _Bars(0.93, 0.99, 0.45)
    ),
    "pass/fail, G 20, k 1, at the floor": _Row(
        "pass_fail", 0.05, 20, 1, RESCORE_MIN_RESULTS, 0.70, 0.75, _PASS_FAIL_SHARE, _PASS_FAIL_ARM
    ),
    "pass/fail, G 60, k 3, at the floor": _Row(
        "pass_fail", 0.05, 60, 3, RESCORE_MIN_RESULTS, 0.70, 0.75, _PASS_FAIL_SHARE, _PASS_FAIL_ARM
    ),
}


class TestTheShareRecoversAPlantedValue:
    @pytest.mark.parametrize("name", list(_SHARE_ROWS))
    def test_within_its_scale_s_tolerance_and_covered_by_its_range(self, name: str) -> None:
        row = _SHARE_ROWS[name]
        measured = _measured(name)
        assert measured.within >= row.within_bar, f"{name}: within ±{row.tolerance:.0%} on {measured.within:.3f}"
        _hold(f"{name}: the share's range", row.share, measured.share_covers, measured.share_width)

    @pytest.mark.parametrize("name", list(_SHARE_ROWS))
    def test_each_arm_s_variance_range_covers_its_planted_variance(self, name: str) -> None:
        row = _SHARE_ROWS[name]
        measured = _measured(name)
        _hold(f"{name}: an arm's variance range", row.arm, measured.arm_covers, measured.arm_width)


_MEASURED: dict[str, _Measured] = {}


def _measured(name: str) -> _Measured:
    """A row's measurement, made once per session for both of its tests."""
    if name not in _MEASURED:
        _MEASURED[name] = _share_row(_SHARE_ROWS[name], f"judge-noise-share/{name}")
    return _MEASURED[name]


def _hold(what: str, bars: _Bars, covers: float, width: float) -> None:
    """A range covers its planted value at least and at most as often as its bars, and is no wider on average."""
    assert covers >= bars.covers_at_least, f"{what} covered on {covers:.3f}, under {bars.covers_at_least}"
    assert covers <= bars.covers_at_most, f"{what} covered on {covers:.3f}, over {bars.covers_at_most}: too wide"
    assert width <= bars.width_at_most, f"{what} was {width:.2f} times its planted value wide on average"


def _ceiling_row(e: float, k: int, repeats: int, seed: str) -> float:
    """How often the ceiling range covers ``(1 - e)^(k - 1)``, over 2,000 arms of 100 cases."""
    rng = random.Random(seed)
    truth = pass_k_ceiling(e, k)
    covered = 0
    for _ in range(_REPLICATES):
        rates = [rng.uniform(0.6, 1.0) for _ in range(100)]
        trials = [(f"c{case}", 1 if rng.random() < rate else 0) for case, rate in enumerate(rates) for _ in range(k)]

        def read(answer: int) -> int:
            # Fails a passing answer with chance e; never passes a failing one.
            return 0 if answer == 1 and rng.random() < e else answer

        chosen = rng.sample(range(len(trials)), repeats)
        repeated = [
            Repeated(trials[index][0], float(read(trials[index][1])), (float(read(trials[index][1])),))
            for index in chosen
        ]
        rescore, _ = rescore_of(repeated, value_range=_PASS_FAIL, k=k, passed=1.0)
        assert rescore.ceiling_low is not None and rescore.ceiling_high is not None
        covered += rescore.ceiling_low <= truth <= rescore.ceiling_high
    return covered / _REPLICATES


class TestTheCeilingOnPassK:
    @pytest.mark.parametrize(
        ("e", "k", "ceiling"), [(0.05, 3, 0.90), (0.10, 2, 0.90), (0.10, 3, 0.81), (0.20, 3, 0.64)]
    )
    def test_the_review_s_table(self, e: float, k: int, ceiling: float) -> None:
        # Review B-3: the ceiling is (1 - e)^(k - 1), never (1 - f)^(k - 1) with f = 2e(1 - e) the flip rate.
        assert round(pass_k_ceiling(e, k), 2) == ceiling
        assert round((1.0 - 2.0 * e * (1.0 - e)) ** (k - 1), 2) != ceiling

    def test_the_design_s_example(self) -> None:
        assert pass_k_ceiling(0.10, 3) == pytest.approx(0.81)

    def test_it_is_no_ceiling_at_one_attempt(self) -> None:
        assert pass_k_ceiling(0.3, 1) == 1.0

    @pytest.mark.parametrize("repeats", [RESCORE_MIN_RESULTS, 200])
    @pytest.mark.parametrize("e", [0.05, 0.10, 0.20])
    def test_the_range_covers_the_planted_ceiling(self, e: float, repeats: int) -> None:
        covered = _ceiling_row(e, 3, repeats, f"judge-noise-ceiling/{e}/{repeats}")
        assert 0.93 <= covered <= 0.99, f"e {e}, {repeats} repeats: the ceiling range covered {covered:.3f}"
