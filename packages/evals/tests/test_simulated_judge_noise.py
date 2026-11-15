"""Re-score noise against a planted truth: the judge's share of a contrast's width, and its ceiling on pass^k.

Seeded, 2,000 replicates per row, through the functions the bundle's filler calls
(:mod:`threetears.evals.analysis.judge_noise`: :func:`rescore_of` for an arm's figures, :func:`judge_share` for a
contrast's). Bars fixed before the run.

**The share.** Each replicate draws ``G`` cases and ``k`` trials per case per arm. A trial's answer is an integer
whose distribution differs by case and arm; the judge scores it with noise of a known variance: on 1 to 5, the answer
plus -1, 0 or +1 with chances ``p/2``, ``1 - p``, ``p/2`` (answers kept in 2 to 4, so the variance is exactly ``p``);
on pass/fail, the answer flipped with chance ``e`` (variance ``e (1 - e)``). Twenty of each arm's results, from at
least ten cases (the floor), are scored again by the same judge. The planted share is the true re-score variance
over the variance of this replicate's per-case differences: the contrast's own width is what it is, and the share
says how much of it the judge's noise is. A row passes when the estimate is within the scale's tolerance (C-12) at
least as often as its bar, and the range covers the planted share at least as often as its bar.

| Row | Scale, noise | G | k | Repeated results | Within tolerance: bar (measured) | Range covers: bar (measured) |
|---|---|---|---|---|---|---|
| 1 | 1-5, p = 0.5 | 20 | 1 | 20 | ±30%: 0.80 (0.877) | 0.93 (1.000) |
| 2 | 1-5, p = 0.5 | 60 | 3 | 20 | ±30%: 0.80 (0.884) | 0.93 (1.000) |
| 3 | 1-5, p = 0.5 | 100 | 3 | 200 | ±30%: 0.98 (1.000) | 0.93 (0.998) |
| 4 | pass/fail, e = 0.05 | 20 | 1 | 20 | ±70%: 0.75 (0.813) | 0.93 (0.962) |
| 5 | pass/fail, e = 0.05 | 60 | 3 | 20 | ±70%: 0.75 (0.835) | 0.93 (0.979) |

The tolerance is about one standard error of one arm's variance at the floor (twenty half squared differences on 1 to
5; about two flips on pass/fail); a share reads two arms' variances, so it holds more often than one standard error
would. The bar falls to near zero for a wrong formula: a variance not halved reads twice the truth, and a share without
``1/k`` reads three times it at k = 3 (rows 2, 3 and 5). Rows 1 and 4 hold the share at k = 1, where only repeats can
measure the judge. Checked by mutation (400 replicates): a variance not halved met the 1-5 rows' bar on 0.03 of
replicates and row 5's on 0.47; a share without ``1/k`` on 0.00 and 0.27. The range is conservative: each end takes both arms' variances at that end.

**The ceiling.** ``e`` is planted on a judge that fails a passing answer with chance ``e`` and never passes a failing
one, so a repeat of a first reading that passed fails with chance exactly ``e``. Its Wilson interval over cases covers
``e``, and so the ceiling range covers ``(1 - e)^(k - 1)``, at least 0.93 at k = 3 over 100 cases, at the floor and at
200 repeated results (measured 0.958, 0.964 and 0.971 at the floor for e = 0.05, 0.10 and 0.20; 0.972, 0.968 and 0.962
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
class _Row:
    """One row of the share's table."""

    scale: str
    noise: float
    cases: int
    k: int
    repeats: int
    tolerance: float
    within_bar: float
    covers_bar: float

    @property
    def planted_variance(self) -> float:
        return self.noise if self.scale == "ordinal" else self.noise * (1.0 - self.noise)


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


def _share_row(row: _Row, seed: str) -> tuple[float, float]:
    """The share of replicates within tolerance of the planted share, and the share whose range covers it."""
    rng = random.Random(seed)
    within = covers = 0
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
        before, after = _case_means(control_trials), _case_means(candidate_trials)
        cases = sorted(before)
        differences = [after[case] - before[case] for case in cases]
        share = judge_share(control_side, candidate_side, cases, differences)
        assert share is not None
        mean = sum(differences) / len(differences)
        spread = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
        planted = row.planted_variance * (2.0 / row.k) / spread
        within += abs(share.point - planted) <= row.tolerance * planted
        covers += share.low <= planted <= share.high
    return within / _REPLICATES, covers / _REPLICATES


_SHARE_ROWS = {
    "1-5, G 20, k 1, at the floor": _Row("ordinal", 0.5, 20, 1, RESCORE_MIN_RESULTS, 0.30, 0.80, 0.93),
    "1-5, G 60, k 3, at the floor": _Row("ordinal", 0.5, 60, 3, RESCORE_MIN_RESULTS, 0.30, 0.80, 0.93),
    "1-5, G 100, k 3, 200 repeats": _Row("ordinal", 0.5, 100, 3, 200, 0.30, 0.98, 0.93),
    "pass/fail, G 20, k 1, at the floor": _Row("pass_fail", 0.05, 20, 1, RESCORE_MIN_RESULTS, 0.70, 0.75, 0.93),
    "pass/fail, G 60, k 3, at the floor": _Row("pass_fail", 0.05, 60, 3, RESCORE_MIN_RESULTS, 0.70, 0.75, 0.93),
}


class TestTheShareRecoversAPlantedValue:
    @pytest.mark.parametrize("name", list(_SHARE_ROWS))
    def test_within_its_scale_s_tolerance_and_covered_by_its_range(self, name: str) -> None:
        row = _SHARE_ROWS[name]
        within, covers = _share_row(row, f"judge-noise-share/{name}")
        assert within >= row.within_bar, f"{name}: within ±{row.tolerance:.0%} on {within:.3f} of replicates"
        assert covers >= row.covers_bar, f"{name}: the range covered the planted share on {covers:.3f}"


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
        assert covered >= 0.93, f"e {e}, {repeats} repeats: the ceiling range covered {covered:.3f}"
