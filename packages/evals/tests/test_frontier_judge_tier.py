"""A frontier verdict made on a judged pass^k names the evidence tier of the judges behind it (#679, gap 21).

``scope_frontier(bar=0.5)`` answered "<pick> is the cheapest clearing the bar (only_cleared)", where the pass^k the
bar was read on conjoined criteria a judge scored, and named no tier — #679 gave a judged bar's verdict on a
campaign's decision surface its judges' tier, and never reached the frontier. A verdict is a recommendation, so the
tier rides on it: the weakest tier among the judges that scored the pick's criteria, ``undetermined`` where nothing
measured them, and none at all for a pick no judge scored.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from threetears.evals.analysis import MIN_CASES, frontier_text
from threetears.evals.analysis.lenses.frontier import FrontierResult, compute_frontier
from threetears.evals.analysis.reads import frontier
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.evidence_tiers import (
    CALIBRATION_MIN_AGREEMENT,
    CALIBRATION_MIN_RESULTS,
    SEPARATION_MIN_AGREEMENT,
    SEPARATION_MIN_RESULTS,
    JudgeEvidenceTier,
    TierCriterion,
)
from threetears.evals.schema import EvalRun, GoalStateOutcome, RoleUsage, RubricScore
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_calibration_rating, make_eval_result, make_eval_run

_DIM = "reply.quality"
_JUDGE = "judge/model"
_CASES = 20


def _score(case: int) -> int:
    """The judge's score on a case: 4 or 5 — both passing, and varied, so people agreeing with it is agreement."""
    return 4 + case % 2


def _corpus(*, judged: bool) -> tuple[EvalRun, list[Any]]:
    """One contestant over twenty cases, every attempt passing — its criteria judged, or only goal-checked."""
    run = make_eval_run(id="run-a", status="completed", k_runs=1, candidate_model="a")
    results = [
        make_eval_result(
            id=f"res-{case}",
            eval_run_id=run.id,
            scope_id=run.scope_id,
            model="a",
            variant_key="vk-a",
            test_case_id=f"tc{case}",
            rubric_scores=(
                [
                    RubricScore(
                        dim=_DIM,
                        score=_score(case),
                        scale="ordinal",
                        served_model=_JUDGE,
                        judge_temperature=0.0,
                        axis="capability",
                    )
                ]
                if judged
                else []
            ),
            goal_state_outcomes=[GoalStateOutcome(expression="ok", passed=True)],
            usage=[RoleUsage(role="candidate", cost_usd=0.01)],
        )
        for case in range(_CASES)
    ]
    return run, results


def _calibrated() -> JudgeEvidenceTier:
    return JudgeEvidenceTier(
        rubric_dim=_DIM,
        scale="ordinal",
        judge_model=_JUDGE,
        judge_config_id=None,
        judge_temperature=0.0,
        tier="calibrated",
        calibration=TierCriterion(
            state="met",
            n=CALIBRATION_MIN_RESULTS,
            results=CALIBRATION_MIN_RESULTS,
            agreement=1.0,
            lower=0.99,
            upper=1.0,
            threshold=CALIBRATION_MIN_AGREEMENT,
            min_results=CALIBRATION_MIN_RESULTS,
        ),
        separation=TierCriterion(
            state="insufficient",
            n=0,
            results=0,
            agreement=None,
            threshold=SEPARATION_MIN_AGREEMENT,
            min_results=SEPARATION_MIN_RESULTS,
        ),
    )


def _verdict(result: FrontierResult) -> Any:
    (subject,) = result.subjects
    assert subject.verdict is not None
    return subject.verdict


def test_a_verdict_on_a_judged_pass_k_carries_its_judges_tier() -> None:
    run, results = _corpus(judged=True)
    verdict = _verdict(compute_frontier([run], results, bar=0.5, archived_run_ids=None, tiers=[_calibrated()]))
    assert verdict.judge_evidence_tier == "calibrated"


def test_a_judge_nothing_measured_reads_undetermined_and_the_text_says_so() -> None:
    run, results = _corpus(judged=True)
    result = compute_frontier([run], results, bar=0.5, archived_run_ids=None)
    assert _verdict(result).judge_evidence_tier == "undetermined"
    assert "judge tier undetermined" in frontier_text(result)


def test_a_pick_no_judge_scored_carries_no_tier() -> None:
    run, results = _corpus(judged=False)
    result = compute_frontier([run], results, bar=0.5, archived_run_ids=None)
    assert _verdict(result).judge_evidence_tier is None
    assert "judge tier" not in frontier_text(result)


def test_the_scope_frontier_reads_its_judges_tiers_from_the_scope() -> None:
    """People's ratings stored in the scope decide the judge's tier, and the scope's frontier names it.

    Without them the answer would be ``undetermined``, which is also what a frontier that read no tier at all says;
    only a stored rating that decides a tier tells the two apart.
    """
    storage = EvalStorage(InMemoryDocumentStore())
    run, results = _corpus(judged=True)
    storage.save_eval_run(run)
    for case, result in enumerate(results):
        storage.save_eval_result(result)
        storage.save_calibration_rating(
            make_calibration_rating(
                scope_id=run.scope_id, run_id=run.id, result_id=result.id, rubric_dim=_DIM, score=_score(case)
            )
        )
    answered = frontier(storage, run.scope_id, list_runs=lambda scope_id, **_: [run], bar=0.5)
    (subject,) = answered["subjects"]
    assert subject["verdict"]["judge_evidence_tier"] == "calibrated"


def test_a_pass_k_over_fewer_cases_than_the_floor_reads_as_a_count_and_over_enough_as_a_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "_CASES", 3)
    run, results = _corpus(judged=False)
    few = frontier_text(compute_frontier([run], results, bar=0.5, archived_run_ids=None))
    assert (
        "pass^1 3 of 3 case(s), too few to state as a rate" in few and "[" not in few.split("pass^1")[1].split(",")[0]
    )

    monkeypatch.setattr(sys.modules[__name__], "_CASES", MIN_CASES)
    run, results = _corpus(judged=False)
    enough = frontier_text(compute_frontier([run], results, bar=0.5, archived_run_ids=None))
    assert f"over {MIN_CASES} case(s)" in enough and "too few to state" not in enough
