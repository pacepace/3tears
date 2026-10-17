"""``compare_two_runs`` reports each run's subject under the engine's own word for it.

The side-by-side view hands back what the host's ``subject_detail`` callable said about each run.
Its keys are part of the lens's public answer, so they are named for the engine's concept — a
subject — and carry no host's word for one.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis import compare_two_runs
from threetears.evals.ops import RunsCompared, runs_compared_text
from threetears.evals.schema import EvalRun, EvalTemplate, GoalStateOutcome, RubricScore
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.scoring import compute_per_case_composites
from packages.evals.tests.factories import make_eval_result, make_eval_run, make_template
from threetears.evals.storage import InMemoryDocumentStore


def _detail(run: EvalRun) -> dict[str, dict[str, Any]]:
    return {run.subject_snapshot.subject_id: {"run": run.id}}


def _load_template(template_id: str) -> EvalTemplate:
    return make_template(id=template_id)


def test_each_run_s_subject_detail_is_reported_under_the_subject_word() -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    run_a = make_eval_run(id="run-a", status="completed")
    run_b = make_eval_run(id="run-b", status="completed")
    storage.save_eval_run(run_a)
    storage.save_eval_run(run_b)

    view = compare_two_runs(
        storage, "run-a", "run-b", run_a.scope_id, load_template=_load_template, subject_detail=_detail
    )

    assert view["subject_detail_a"] == _detail(run_a)
    assert view["subject_detail_b"] == _detail(run_b)
    assert not [key for key in view if "persona" in key]


def _scored_run(storage: EvalStorage, run_id: str, scores: dict[str, int]) -> EvalRun:
    """A completed run of one template, one result per case, each case's composite set by its rubric score."""
    run = make_eval_run(id=run_id, status="completed", template_id="tpl-1", test_case_ids=sorted(scores))
    storage.save_eval_run(run)
    for case, score in scores.items():
        storage.save_eval_result(
            make_eval_result(
                eval_run_id=run.id,
                scope_id=run.scope_id,
                model=run.candidate_model,
                test_case_id=case,
                rubric_scores=[
                    RubricScore(
                        dim="reply.quality", score=score, scale="ordinal", axis="capability", judge_temperature=0.0
                    )
                ],
            )
        )
    return run


def test_the_composites_and_delta_are_over_the_cases_the_paired_test_read() -> None:
    """B scored a fifth case A never ran: the paired test leaves it out, and so do the means beside it."""
    storage = EvalStorage(InMemoryDocumentStore())
    run_a = _scored_run(storage, "run-a", {"c1": 2, "c2": 3, "c3": 3, "c4": 4})
    _scored_run(storage, "run-b", {"c1": 3, "c2": 3, "c3": 4, "c4": 5, "c5": 1})

    arm = compare_two_runs(
        storage, "run-a", "run-b", run_a.scope_id, load_template=_load_template, subject_detail=_detail
    )["comparison"]["arm"]

    assert arm["paired"] is True and arm["n_pairs"] == 4
    assert (arm["n_cases_a"], arm["n_cases_b"], arm["n_left_out_a"], arm["n_left_out_b"]) == (4, 4, 0, 1)
    per_case = compute_per_case_composites(storage.query_eval_results_by_run("run-b", run_a.scope_id))
    tested_b = [value for (_m, _r, case), value in per_case.items() if case != "c5"]
    assert arm["composite_b"] == pytest.approx(sum(tested_b) / 4)
    assert arm["composite_delta"] == pytest.approx(arm["composite_b"] - arm["composite_a"])
    low, high = arm["composite_interval"]
    assert low <= arm["composite_delta"] <= high
    # The interval is the one the test inverts: it excludes zero exactly when p is below alpha.
    assert (low > 0 or high < 0) == (arm["p"] < 0.05)


def _goal_checked_run(storage: EvalStorage, run_id: str, cases: list[str], *, k: int = 2) -> EvalRun:
    """A completed run of one template judged by nothing: every attempt of every case passed its one goal check."""
    run = make_eval_run(id=run_id, status="completed", template_id="tpl-1", test_case_ids=cases, k_runs=k)
    storage.save_eval_run(run)
    for case in cases:
        for iteration in range(1, k + 1):
            storage.save_eval_result(
                make_eval_result(
                    id=f"{run_id}-{case}-{iteration}",
                    eval_run_id=run.id,
                    scope_id=run.scope_id,
                    model=run.candidate_model,
                    test_case_id=case,
                    k_iteration=iteration,
                    rubric_scores=[],
                    goal_state_outcomes=[GoalStateOutcome(expression="state.done == true", passed=True)],
                )
            )
    return run


class TestTwoRunsNoJudgeScored:
    """Two goal-check-only runs (k=2, four shared cases): no composite exists, so nothing is paired over it (gap 19).

    Read as "paired over 0 shared case(s)" with ``paired=True``, contradicting ``n_pairs``, and its pass^k line was
    stated under a judged criterion the attempts never had, over "0 vs 0 case(s)" — the composite's case count,
    which is not the cases pass^k averaged.
    """

    def _compared(self) -> tuple[dict[str, Any], str]:
        storage = EvalStorage(InMemoryDocumentStore())
        cases = ["c1", "c2", "c3", "c4"]
        run_a = _goal_checked_run(storage, "run-a", cases)
        _goal_checked_run(storage, "run-b", cases)
        view = compare_two_runs(
            storage, "run-a", "run-b", run_a.scope_id, load_template=_load_template, subject_detail=_detail
        )
        compared = RunsCompared(
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            comparison=view,
            completeness_disclosures={},
            measurement_window_disclosure=None,
            cassette_mode_disclosure=None,
        )
        return view["comparison"]["arm"], runs_compared_text(compared)

    def test_no_composite_is_paired_over_no_case(self) -> None:
        arm, text = self._compared()
        assert arm["paired"] is False and arm["n_pairs"] is None
        assert arm["composite_untested_reason"] == "neither run's results carry a judged composite"
        assert "paired over 0" not in text
        assert "mean composite: none — no paired cases: neither run's results carry a judged composite" in text

    def test_pass_k_names_the_criterion_its_attempts_had_and_the_cases_it_read(self) -> None:
        arm, text = self._compared()
        assert (arm["pass_hat_k_n_cases_a"], arm["pass_hat_k_n_cases_b"]) == (4, 4)
        assert arm["pass_criteria_a"] == arm["pass_criteria_b"] == ["goal_checks"]
        assert "pass^k (k=2, every goal check): 1 vs 1 (delta 0; 4 vs 4 case(s))" in text
        assert "criterion >= 3 of 5" not in text
