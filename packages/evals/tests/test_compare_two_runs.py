"""``compare_two_runs`` reports each run's subject under the engine's own word for it.

The side-by-side view hands back what the host's ``subject_detail`` callable said about each run.
Its keys are part of the lens's public answer, so they are named for the engine's concept — a
subject — and carry no host's word for one.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis import compare_two_runs
from threetears.evals.contracts import EvalRun, EvalStorage, EvalTemplate, RubricScore
from threetears.evals.contracts.scoring import compute_per_case_composites
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
                rubric_scores=[RubricScore(dim="reply.quality", score=score, scale="ordinal")],
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
