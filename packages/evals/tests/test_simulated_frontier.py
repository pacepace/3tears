"""The frontier's dominance flag and its pass^k interval, against known truths (#601).

:func:`~threetears.evals.analysis.reporting.compute_frontier` ranks each subject's contestants on pass^k ×
production-replicating cost × mean total latency and flags a contestant ``dominated`` when another is shown
better on every axis it measured. A dominance flag is a claim that one contestant is worse, held to the α
every other between-arm claim in the engine is held to. Decided on point estimates, it read noise as an
ordering: of two IDENTICAL contestants (5 cases × k=3) one was flagged dominated in 0.32 of replicates.

- Identical contestants: same per-case pass probability (cases differ in difficulty, alike for both), same
  cost and latency distributions. Any flag is false.
- The least favourable truth for the rule: one contestant really is dearer and slower, and equal on pass^k.
  Two of the three axes are then genuinely separated, so a false flag needs only the third to err; the
  intersection–union test bounds it by α/2 there.
- A contestant worse on every axis is flagged: the rule is not vacuous.
- The pass^k interval the bar is read by covers the true pass^k at its level.
- The verdict's "cheapest" is decided by test too. Picked on point cost, it named one of two identical
  contestants the cheapest in every replicate; now a winner is named on identical contestants at most α of
  the time, and the rest name the set the data cannot order.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest

from threetears.evals.analysis.reporting import compute_frontier
from threetears.evals.analysis.stats import INTERVAL_LEVEL, SIGNIFICANCE_ALPHA, case_rate_interval
from threetears.evals.contracts import EvalResult, EvalRun, GoalStateOutcome, LatencyMetrics, RoleUsage, RubricScore
from threetears.evals.contracts.scoring import case_pass_hat_k
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.simulation_support import at_least, at_most

_MODELS = ("model-one", "model-two")


def _identical_contestants(rng: random.Random, n_cases: int, repeats: int) -> tuple[list[EvalRun], list[EvalResult]]:
    """Two runs of two contestants drawn from one distribution, over the same cases."""
    runs = [make_eval_run(status="completed", candidate_model=model) for model in _MODELS]
    difficulty = [rng.gauss(0.0, 1.0) for _ in range(n_cases)]
    results = []
    for run, model in zip(runs, _MODELS, strict=True):
        for case in range(n_cases):
            pass_probability = 1.0 / (1.0 + math.exp(-(1.5 + difficulty[case])))
            for repeat in range(1, repeats + 1):
                passed = rng.random() < pass_probability
                results.append(
                    make_eval_result(
                        eval_run_id=run.id,
                        scope_id=run.scope_id,
                        model=model,
                        test_case_id=f"tc-{case}",
                        k_iteration=repeat,
                        goal_state_outcomes=[GoalStateOutcome(expression="ok", passed=passed)],
                        rubric_scores=[RubricScore(dim="reply.quality", score=4 if passed else 2, scale="ordinal")],
                        usage=[RoleUsage(role="candidate", cost_usd=round(rng.lognormvariate(-4.0, 0.5), 6))],
                        latency=LatencyMetrics(total_ms=round(rng.gauss(2000.0, 300.0), 3)),
                    )
                )
    return runs, results


def test_identical_contestants_are_not_flagged_dominated_beyond_alpha() -> None:
    """300 replicates: SE at α is 0.0126, so the bound is 0.100."""
    rng = random.Random("frontier-identical")
    replicates = 300
    flagged = 0
    for _ in range(replicates):
        runs, results = _identical_contestants(rng, n_cases=5, repeats=3)
        (subject,) = compute_frontier(runs, results).subjects
        assert len(subject.points) == 2
        flagged += any(point.dominated for point in subject.points)
    rate = flagged / replicates
    assert rate <= at_most(SIGNIFICANCE_ALPHA, replicates), f"one identical contestant dominated in {rate:.3f}"


def _two_contestants(
    rng: random.Random,
    n_cases: int,
    repeats: int,
    *,
    logit_shift: float,
    cost_factor: float,
    latency_factor: float,
) -> tuple[list[EvalRun], list[EvalResult]]:
    """``model-one`` and ``model-two`` over the same cases; ``model-two`` moved by the given amounts.

    Cases differ in difficulty, cost and latency alike for both contestants (what pairing on the case
    buys). ``model-two``'s pass logit is ``model-one``'s plus ``logit_shift`` at every case, and its cost
    and latency are ``cost_factor`` and ``latency_factor`` times ``model-one``'s case levels.
    """
    runs = [make_eval_run(status="completed", candidate_model=model) for model in _MODELS]
    difficulty = [rng.gauss(0.0, 1.0) for _ in range(n_cases)]
    case_cost = [rng.gauss(0.0, 0.5) for _ in range(n_cases)]
    case_latency = [rng.gauss(0.0, 300.0) for _ in range(n_cases)]
    results = []
    for index, (run, model) in enumerate(zip(runs, _MODELS, strict=True)):
        moved = index == 1
        for case in range(n_cases):
            logit = 1.5 + difficulty[case] + (logit_shift if moved else 0.0)
            pass_probability = 1.0 / (1.0 + math.exp(-logit))
            for repeat in range(1, repeats + 1):
                passed = rng.random() < pass_probability
                cost = (cost_factor if moved else 1.0) * math.exp(-4.0 + case_cost[case] + rng.gauss(0.0, 0.3))
                latency = (latency_factor if moved else 1.0) * (2000.0 + case_latency[case]) + rng.gauss(0.0, 200.0)
                results.append(
                    make_eval_result(
                        eval_run_id=run.id,
                        scope_id=run.scope_id,
                        model=model,
                        test_case_id=f"tc-{case}",
                        k_iteration=repeat,
                        goal_state_outcomes=[GoalStateOutcome(expression="ok", passed=passed)],
                        rubric_scores=[RubricScore(dim="reply.quality", score=4 if passed else 2, scale="ordinal")],
                        usage=[RoleUsage(role="candidate", cost_usd=round(cost, 6))],
                        latency=LatencyMetrics(total_ms=round(latency, 3)),
                    )
                )
    return runs, results


def _second_flagged_share(seed: str, replicates: int, **truth: float) -> float:
    rng = random.Random(seed)
    flagged = 0
    for _ in range(replicates):
        runs, results = _two_contestants(rng, n_cases=8, repeats=3, **truth)  # type: ignore[arg-type]
        (subject,) = compute_frontier(runs, results).subjects
        second = next(point for point in subject.points if point.model == "model-two")
        flagged += second.dominated
    return flagged / replicates


def test_a_dearer_slower_contestant_equal_on_pass_k_is_not_flagged_beyond_alpha() -> None:
    """The least favourable truth: 3x the cost, 2x the latency, the same pass^k, 8 cases x k=3.

    Measured near 0.03 (the bound of the rule is α/2 = 0.025 per direction). 400 replicates: SE at α is
    0.0109, so the bound is 0.094.
    """
    replicates = 400
    rate = _second_flagged_share(
        "frontier-least-favourable", replicates, logit_shift=0.0, cost_factor=3.0, latency_factor=2.0
    )
    assert rate <= at_most(SIGNIFICANCE_ALPHA, replicates), f"flagged dominated on a pass^k tie in {rate:.3f}"


def test_a_contestant_worse_on_every_axis_is_flagged() -> None:
    """Pass logit 3 lower, 3x the cost, 2x the latency, 8 cases x k=3: measured near 0.98.

    300 replicates; held to at least 0.90 (less 4 SE, 0.83), so a rule that never flags fails it.
    """
    replicates = 300
    rate = _second_flagged_share("frontier-power", replicates, logit_shift=-3.0, cost_factor=3.0, latency_factor=2.0)
    assert rate >= at_least(0.90, replicates), f"a contestant worse on every axis was flagged in only {rate:.3f}"


def test_the_frontier_states_pass_k_with_the_interval_over_its_cases() -> None:
    """Wiring: the point's interval is :func:`case_rate_interval` over its per-case unbiased estimates."""
    rng = random.Random("frontier-interval-wiring")
    runs, results = _identical_contestants(rng, n_cases=6, repeats=3)
    (subject,) = compute_frontier(runs, results).subjects
    for point in subject.points:
        by_case: dict[str, list[bool]] = {}
        for result in results:
            if result.model == point.model:
                by_case.setdefault(result.test_case_id, []).append(all(o.passed for o in result.goal_state_outcomes))
        estimates = [case_pass_hat_k(attempts, subject.k) for attempts in by_case.values()]
        expected = case_rate_interval(
            [e for e in estimates if e is not None], max_effective_n=sum(map(len, by_case.values())) / subject.k
        )
        assert expected is not None
        assert (point.pass_hat_k_ci_low, point.pass_hat_k_ci_high) == pytest.approx(expected, rel=1e-12)


def _true_pass_k(base_logit: float, case_sd: float, k: int) -> float:
    """``E[p^k]`` over cases whose pass probability is ``logistic(base + case_sd z)``, by a fine midpoint rule."""
    steps, span = 4000, 16.0
    total = 0.0
    for index in range(steps):
        z = -span / 2 + span * (index + 0.5) / steps
        weight = math.exp(-z * z / 2) / math.sqrt(2 * math.pi) * span / steps
        total += weight * (1.0 / (1.0 + math.exp(-(base_logit + case_sd * z)))) ** k
    return total


#: (cases, k, attempts per case, base logit, between-case SD). The corners where an interval on a rate
#: is weakest: two cases; truth near 1 with each case seen once; truth near 0 at depth 3; every case alike.
_INTERVAL_CONFIGS = [
    (2, 1, 3, 1.5, 1.0),
    (5, 3, 3, 1.5, 1.0),
    (5, 3, 3, 5.0, 0.0),
    (8, 3, 5, -1.0, 0.5),
    (15, 1, 1, 5.0, 0.5),
    (15, 3, 3, 5.0, 1.0),
    (15, 5, 5, -1.0, 0.5),
]


@pytest.mark.parametrize(("n_cases", "k", "depth", "base_logit", "case_sd"), _INTERVAL_CONFIGS)
def test_the_pass_k_interval_covers_the_true_pass_k(
    n_cases: int, k: int, depth: int, base_logit: float, case_sd: float
) -> None:
    """1,000 replicates a configuration: SE at 95% is 0.0069, so the bound is 0.922.

    Wilson's shape on the same effective size covered 0.87 at 15 cases seen once with the truth at 0.99;
    Clopper-Pearson's holds there.
    """
    rng = random.Random(f"pass-k-interval-{n_cases}-{k}-{depth}-{base_logit}-{case_sd}")
    truth = _true_pass_k(base_logit, case_sd, k)
    replicates = 1000
    covered = 0
    for _ in range(replicates):
        estimates = []
        for _ in range(n_cases):
            p = 1.0 / (1.0 + math.exp(-(base_logit + case_sd * rng.gauss(0.0, 1.0))))
            attempts = [rng.random() < p for _ in range(depth)]
            estimate = case_pass_hat_k(attempts, k)
            assert estimate is not None
            estimates.append(estimate)
        interval = case_rate_interval(estimates, max_effective_n=n_cases * depth / k)
        assert interval is not None
        covered += interval[0] <= truth <= interval[1]
    coverage = covered / replicates
    assert coverage >= at_least(INTERVAL_LEVEL, replicates), f"covered the true pass^{k} {truth:.3f} in {coverage:.3f}"


def _priced_contestants(
    rng: random.Random, n_cases: int, repeats: int, cost_factors: Sequence[float]
) -> tuple[list[EvalRun], list[EvalResult]]:
    """One contestant per cost factor, alike in every other respect, over the same cases.

    Every attempt passes, so each contestant clears any bar its interval can reach and the verdict turns on
    cost alone. Cases differ in cost alike for every contestant (what pairing on the case buys); contestant
    ``i``'s cost is ``cost_factors[i]`` times the case's level, with lognormal noise per attempt.
    """
    models = [f"model-{index}" for index in range(len(cost_factors))]
    runs = [make_eval_run(status="completed", candidate_model=model) for model in models]
    case_cost = [rng.gauss(0.0, 0.5) for _ in range(n_cases)]
    results = []
    for run, model, factor in zip(runs, models, cost_factors, strict=True):
        for case in range(n_cases):
            for repeat in range(1, repeats + 1):
                cost = factor * math.exp(-4.0 + case_cost[case] + rng.gauss(0.0, 0.3))
                results.append(
                    make_eval_result(
                        eval_run_id=run.id,
                        scope_id=run.scope_id,
                        model=model,
                        test_case_id=f"tc-{case}",
                        k_iteration=repeat,
                        goal_state_outcomes=[GoalStateOutcome(expression="ok", passed=True)],
                        rubric_scores=[RubricScore(dim="reply.quality", score=4, scale="ordinal")],
                        usage=[RoleUsage(role="candidate", cost_usd=round(cost, 6))],
                        latency=LatencyMetrics(total_ms=round(rng.gauss(2000.0, 300.0), 3)),
                    )
                )
    return runs, results


def _shown_cheapest(seed: str, replicates: int, cost_factors: Sequence[float]) -> tuple[float, str]:
    """The share of replicates whose verdict names a single cheapest, and the model named most often."""
    rng = random.Random(seed)
    named = 0
    picks: dict[str, int] = {}
    for _ in range(replicates):
        runs, results = _priced_contestants(rng, n_cases=8, repeats=3, cost_factors=cost_factors)
        (subject,) = compute_frontier(runs, results, bar=0.5).subjects
        assert subject.verdict is not None and subject.n_cleared_bar == len(cost_factors)
        if subject.verdict.cost_decision == "shown_cheapest":
            named += 1
            picks[subject.verdict.model] = picks.get(subject.verdict.model, 0) + 1
        else:
            # Every cleared contestant the pick was not shown cheaper than is named beside it.
            assert subject.verdict.cost_decision in ("not_separated", "untested")
            assert subject.verdict.tied_with
    return named / replicates, max(picks, key=picks.__getitem__) if picks else ""


@pytest.mark.parametrize("n_contestants", [2, 3])
def test_identical_contestants_name_no_cheapest_beyond_alpha(n_contestants: int) -> None:
    """Identical contestants, 8 cases x k=3: picked on point cost, one was named cheapest every time.

    Measured 0.043 at two (the pick always leans its own way, so the two-sided test's whole α lands on it)
    and 0.005 at three, where the pick must separate from both rivals. 400 replicates: SE at α
    is 0.0109, so the bound is 0.094.
    """
    replicates = 400
    rate, _ = _shown_cheapest(f"frontier-cheapest-identical-{n_contestants}", replicates, [1.0] * n_contestants)
    assert rate <= at_most(SIGNIFICANCE_ALPHA, replicates), f"named a cheapest of identical contestants in {rate:.3f}"


def test_a_contestant_at_half_the_cost_is_named_cheapest() -> None:
    """One contestant at half the others' cost, 8 cases x k=3: measured 0.975.

    200 replicates; held to at least 0.90 (less 4 SE, 0.82), so a rule that never names a winner fails it.
    """
    replicates = 200
    rate, named = _shown_cheapest("frontier-cheapest-power", replicates, [1.0, 0.5, 1.0])
    assert rate >= at_least(0.90, replicates), f"the half-cost contestant was named cheapest in only {rate:.3f}"
    assert named == "model-1"
