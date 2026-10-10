"""Judge drift and inter-judge agreement, checked against simulated judges with a known answer (#597, #646).

Two judges score the same evidence. The drift reading (:func:`~threetears.evals.analysis.judge_drift`) says per
dimension whether the scores moved, with an interval on the movement; the agreement reading
(:func:`~threetears.evals.analysis.inter_judge_agreement`) says how far the two agree, with bounds on kappa. Each is
checked here against judges whose truth is known:

- **No drift.** The second judge is the first plus symmetric noise (or a per-case move whose sign is a coin), so
  the true movement is exactly 0 on every dimension. The family of dimensions may be called ``separated`` at most
  α = 5% of the time — including when a case is judged three times and its three movements are one movement, which
  a reading that took each result as a draw would call separated far more often.
- **A known shift.** The second judge is the first plus 1 plus symmetric noise, on scores kept inside 1-5 by
  construction, so the true movement is exactly +1. The interval (at ``1 − α/m``) covers +1 at least that often,
  and the reading separates it.
- **A known kappa.** The second judge copies the first with probability 0.6 and otherwise draws from the same
  marginal (:func:`~packages.evals.tests.simulation_support.draw_rater_pairs`), so the population kappa is 0.6 under
  any cost: the published figure is centred on it, and its one-sided 95% lower bound sits at or below it at least
  95% of the time.

Seeded; each tolerance is Monte-Carlo error (:mod:`packages.evals.tests.simulation_support`).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

from threetears.evals.analysis import inter_judge_agreement, judge_drift
from threetears.evals.contracts.models import EvalResult, RubricScore, SecondJudge, SecondJudgeScore, SecondJudging
from packages.evals.tests.factories import make_eval_result
from packages.evals.tests.simulation_support import TOLERANCE_Z, at_least, at_most, draw_rater_pairs

_DIMS = ("dim.a", "dim.b")
_SECOND = SecondJudge(model="judge/second")
_ALPHA = 0.05
_BASE = make_eval_result(rubric_scores=[], goal_state_outcomes=[])


def _results(
    rng: random.Random, n_cases: int, repeats: int, draw: Callable[[random.Random, str, str], tuple[int, int]]
) -> list[EvalResult]:
    """``n_cases`` cases judged ``repeats`` times each, every result carrying both judges' scores on every dim."""
    results = []
    for case in range(n_cases):
        for repeat in range(repeats):
            scores = []
            for dim in _DIMS:
                first, second = draw(rng, dim, f"case-{case}")
                scores.append(
                    SecondJudgeScore(
                        dim=dim,
                        scale="ordinal",
                        first_score=first,
                        first_served_model="judge/first",
                        first_judge_config_id=None,
                        second=RubricScore(dim=dim, scale="ordinal", score=second, served_model="judge/second"),
                    )
                )
            judging = SecondJudging(pass_id="pass", judge=_SECOND, sample_fraction=1.0, sample_seed=0, scores=scores)
            results.append(
                _BASE.model_copy(
                    update={"id": f"r-{case}-{repeat}", "test_case_id": f"case-{case}", "judge_seconds": [judging]}
                )
            )
    return results


def _noise(rng: random.Random, dim: str, case: str) -> tuple[int, int]:
    """No drift: the first score in 2-4, the second the first plus -1, 0 or +1 — true movement 0."""
    first = rng.choice((2, 3, 4))
    return first, first + rng.choice((-1, 0, 1))


def _shifted(rng: random.Random, dim: str, case: str) -> tuple[int, int]:
    """Drift on ``dim.a`` only: the first score in 1-3, the second the first plus 1 plus -1, 0 or +1 — true +1."""
    if dim != "dim.a":
        return _noise(rng, dim, case)
    first = rng.choice((1, 2, 3))
    return first, first + 1 + rng.choice((-1, 0, 1))


def _false_separation_rate(
    seed: str,
    draw: Callable[[random.Random, str, str], tuple[int, int]],
    *,
    n_cases: int,
    repeats: int,
    replicates: int,
) -> float:
    rng = random.Random(seed)
    hits = 0
    for _ in range(replicates):
        drift = judge_drift(_results(rng, n_cases, repeats, draw))
        hits += any(row.verdict == "separated" for row in drift.dimensions)
    return hits / replicates


class TestNoDriftIsCalledDriftAtMostAlpha:
    def test_independent_results(self) -> None:
        """20 cases judged once, two dims: the family's false 'separated' rate is at most 5%, over 1,000 replicates."""
        rate = _false_separation_rate("drift-null", _noise, n_cases=20, repeats=1, replicates=1000)
        assert rate <= at_most(_ALPHA, 1000), f"false separation {rate:.3f}"

    def test_a_case_judged_three_times_is_one_draw(self) -> None:
        """10 cases, each judged three times with one shared ±1 movement: still at most 5%. Read per result, the same
        data's 30 'draws' would show the case-level coin as a consistent movement far more often."""

        rng = random.Random("drift-null-clustered")
        hits = 0
        replicates = 1000
        for _ in range(replicates):
            signs = {(dim, f"case-{case}"): rng.choice((-1, 1)) for dim in _DIMS for case in range(10)}

            def draw(r: random.Random, dim: str, case: str) -> tuple[int, int]:
                first = r.choice((2, 3, 4))
                return first, first + signs[(dim, case)]

            drift = judge_drift(_results(rng, 10, 3, draw))
            hits += any(row.verdict == "separated" for row in drift.dimensions)
        assert hits / replicates <= at_most(_ALPHA, replicates), f"false separation {hits / replicates:.3f}"


class TestAKnownShiftIsCoveredAndSeparated:
    def test_the_interval_covers_the_true_movement_and_the_shift_separates(self) -> None:
        """20 cases, dim.a moved by exactly +1: the interval at 1 − α/2 covers +1 at least 97.5% of the time (within
        Monte-Carlo error over 1,000 replicates), and the shift reads 'separated' in nearly every replicate."""
        rng = random.Random("drift-shift")
        replicates = 1000
        covered = separated = 0
        for _ in range(replicates):
            rows = {row.rubric_dim: row for row in judge_drift(_results(rng, 20, 1, _shifted)).dimensions}
            moved = rows["dim.a"]
            assert moved.interval is not None and moved.interval_level is not None
            covered += moved.interval[0] <= 1.0 <= moved.interval[1]
            separated += moved.verdict == "separated" and (moved.delta or 0) > 0
        nominal = 1 - _ALPHA / 2
        assert covered / replicates >= at_least(nominal, replicates), f"coverage {covered / replicates:.3f}"
        assert separated / replicates >= 0.99, f"power {separated / replicates:.3f}"


class TestInterJudgeKappaIsTheKnownOne:
    def test_the_figure_is_centred_and_its_lower_bound_holds(self) -> None:
        """40 results, a second judge agreeing at a quadratic kappa of 0.6 on a peaked 1-5 marginal: the mean figure is
        within 0.03 of 0.6 (kappa's small-sample bias at 40 is ~0.01) plus 4 Monte-Carlo SEs, and the one-sided 95%
        lower bound is at or below 0.6 at least 95% of the time, over 400 replicates."""
        rng = random.Random("inter-judge-kappa")
        replicates = 400
        estimates = []
        held = 0
        for _ in range(replicates):
            pairs = iter(
                draw_rater_pairs(rng, 40 * len(_DIMS), [1, 2, 3, 4, 5], marginal=[1, 2, 4, 2, 1], agreement=0.6)
            )

            def draw(_r: random.Random, _dim: str, _case: str) -> tuple[int, int]:
                return next(pairs)

            (row, _other) = inter_judge_agreement(_results(rng, 40, 1, draw)).dimensions
            assert row.kappa is not None and row.agreement_interval is not None
            estimates.append(row.kappa)
            held += row.agreement_interval[0] <= 0.6
        mean = sum(estimates) / len(estimates)
        se = math.sqrt(sum((value - mean) ** 2 for value in estimates) / (len(estimates) - 1)) / math.sqrt(
            len(estimates)
        )
        assert abs(mean - 0.6) <= 0.03 + TOLERANCE_Z * se, f"mean kappa {mean:.4f}"
        assert held / replicates >= at_least(0.95, replicates), f"lower bound held {held / replicates:.3f}"
