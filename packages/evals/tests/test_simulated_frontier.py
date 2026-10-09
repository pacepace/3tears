"""The frontier's dominance flag on two identical contestants (#601).

:func:`~threetears.evals.analysis.reporting.compute_frontier` ranks each subject's contestants on pass^k ×
production-replicating cost × mean total latency and flags a contestant ``dominated`` when another is no
worse on every axis and strictly better on one. The comparison is of point estimates, so it reads noise
as an ordering: two contestants drawn from ONE distribution differ on every continuous axis, and one of
them is "dominated" whenever chance orders all three the same way.

The truth here is that the two contestants are the same: same per-case pass probability (cases differ in
difficulty, alike for both), same cost distribution, same latency distribution. A dominance flag between them
is a false claim that one is worse, held to the α every other between-arm claim in the engine is held to.
"""

from __future__ import annotations

import math
import random

import pytest

from threetears.evals.analysis.reporting import compute_frontier
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA
from threetears.evals.contracts import EvalResult, GoalStateOutcome, LatencyMetrics, RoleUsage, RubricScore
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.simulation_support import at_most

_MODELS = ("model-one", "model-two")


def _identical_contestants(rng: random.Random, n_cases: int, repeats: int) -> tuple[list, list[EvalResult]]:
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#601 finding: the frontier decides dominance on point estimates, so of two IDENTICAL contestants "
        "(5 cases x k=3) one is flagged dominated in 0.32 of replicates against the 0.05 any between-arm claim "
        "is held to. (It also ranks latency on mean_total_ms, where measuring-soundly says rank latency on p95.)"
    ),
)
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
