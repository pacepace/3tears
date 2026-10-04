"""``compare_two_runs`` reports each run's subject under the engine's own word for it.

The side-by-side view hands back what the host's ``subject_detail`` callable said about each run.
Its keys are part of the lens's public answer, so they are named for the engine's concept — a
subject — and carry no host's word for one.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis import compare_two_runs
from threetears.evals.contracts import EvalRun, EvalStorage, EvalTemplate
from packages.evals.tests.factories import make_eval_run, make_template
from packages.evals.tests.memory_store import InMemoryDocumentStore


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
