"""Each trial and how it ended, and each arm's census of them, read off the results once for every reader.

A trial is one case run once on one arm, and it ends in exactly one state: **lost** to the test setup (the harness
faulted it, or the judge could not tell on a dimension it needs), **failed** by the candidate (its model failed, or a
check or a judged dimension failed it), or **passed**. A judge that could not tell on one dimension does not lose a
trial a failed check has already decided. The census counts every trial, the rows carry what a reader drills into,
capped with the lost and failed kept first, and the writer reads the rows without the excerpts.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from threetears.evals.analysis.bundle import trials as trials_module
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, ArmScoreboard
from threetears.evals.analysis.generator import build_user_message
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.schema import EvalResult, EvalTrace, GoalStateOutcome, JudgedArtifact, JudgeEvidence
from threetears.evals.schema.models import eval_trace_doc_id
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile


class _TracedStorage(ToyhostStorage):
    """The toy store, holding a trace for the results that say they have one."""

    def __init__(self, *args: Any, traces: dict[str, EvalTrace], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.traces = traces
        self.trace_reads: list[str] = []

    def load_eval_trace(self, result_id: str, scope_id: str) -> EvalTrace | None:
        self.trace_reads.append(result_id)
        return self.traces.get(result_id)


def _result(case: str, k: int, **update: Any) -> EvalResult:
    return make_eval_result(test_case_id=case, k_iteration=k, **update)


#: Every way a trial ends, one result each, on one arm.
RESULTS = [
    _result("case-a", 1),
    _result("case-a", 2, goal_state_outcomes=[GoalStateOutcome(expression="state.done == true", passed=False)]),
    _result("case-b", 1, candidate_error="model refused"),
    _result("case-b", 2, infra_error="simulator crashed"),
    _result("case-c", 1, judge_cannot_tell={"conversation.tone": "the transcript is empty"}, rubric_scores=[]),
    _result(
        "case-c",
        2,
        judge_cannot_tell={"conversation.tone": "the transcript is empty"},
        rubric_scores=[],
        goal_state_outcomes=[GoalStateOutcome(expression="state.done == true", passed=False)],
    ),
]


def _bundle(results: list[EvalResult], traces: dict[str, EvalTrace] | None = None) -> AnalysisContextBundle:
    run = make_eval_run(status="completed", rubric_scales={"conversation.tone": "ordinal"})
    campaign = EvalCampaign(
        scope_id=run.scope_id,
        name="trials",
        subject_id=run.subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[run.id],
        created_by="test:fixture",
    )
    members = [result.model_copy(update={"eval_run_id": run.id, "scope_id": run.scope_id}) for result in results]
    storage = _TracedStorage([run], {run.id: members}, traces=traces or {})
    return assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())


def test_every_trial_ends_in_one_state_and_names_why() -> None:
    bundle = _bundle(RESULTS)
    ended = [(trial.case_id, trial.repeat, trial.state, trial.cause) for trial in bundle.trials]
    assert ended == [
        ("case-a", 1, "passed", None),
        ("case-a", 2, "failed", "check"),
        ("case-b", 1, "failed", "model_failed"),
        ("case-b", 2, "lost", "infra"),
        ("case-c", 1, "lost", "judge_cannot_tell"),
        ("case-c", 2, "failed", "check"),
    ]
    (failed_check,) = bundle.trials[1].failed_checks
    assert failed_check.expression == "state.done == true"
    assert [trial.trial_id for trial in bundle.trials] == ["T1", "T2", "T3", "T4", "T5", "T6"]


def test_the_census_counts_every_trial_and_never_a_score() -> None:
    (board,) = _bundle(RESULTS).scoreboard
    assert (board.ran, board.usable, board.passed) == (6, 4, 1)
    assert board.lost == {"infra": 1, "judge_cannot_tell": 1}
    assert board.failed == {"check": 2, "model_failed": 1}
    assert (board.cases, board.usable_cases) == (3, 3)
    assert board.test_cost_usd == pytest.approx(0.06)


def test_a_census_whose_parts_do_not_make_its_whole_is_refused() -> None:
    with pytest.raises(ValueError, match="lost .* usable != .* ran"):
        ArmScoreboard(variant_key="v", ran=3, lost={"infra": 1}, usable=1, passed=1, failed={}, cases=1, usable_cases=1)


def test_the_exclusions_are_the_kernels_count_and_the_census_reads_the_same_classification() -> None:
    bundle = _bundle(RESULTS)
    (exclusions,) = bundle.exclusions
    # The bundle's exclusions are `trial_exclusion`'s, counted once by the assembler: both can't-tell trials are
    # left out of the measures needing every dimension, the one a failed check decided included.
    assert exclusions.by_cause == {"infra": 1, "judge_cannot_tell": 2}
    flagged = {(entry.cause, round(entry.share, 3)) for entry in bundle.differential_exclusions}
    assert flagged == {("infra", 0.167), ("judge_cannot_tell", 0.333)}
    # The census reads that classification per trial, as pass^k does: the can't-tell trial a check already failed
    # is the fail it is, so the census loses one fewer to the judge than the measures leave out.
    (board,) = bundle.scoreboard
    assert board.lost == {"infra": 1, "judge_cannot_tell": 1}
    assert board.lost["infra"] == exclusions.by_cause["infra"]


def test_a_trial_shows_what_the_judge_was_handed_its_tail_and_the_case_label_its_first_input() -> None:
    long = "x" * 2500 + " the reply ends here"
    traced = _result("case-a", 1, has_trace=True)
    evidence = JudgeEvidence(case_material="Queue three songs for the morning show.\nMore detail.", artifact=long)
    trace = EvalTrace(
        id=eval_trace_doc_id(traced.id),
        scope_id="uni-1",
        result_id=traced.id,
        eval_run_id="run-1",
        judge_evidence=evidence,
        judged_artifact=JudgedArtifact.TRANSCRIPT,
    )
    (trial,) = _bundle([traced], {traced.id: trace}).trials
    assert trial.case_label == "Queue three songs for the morning show. More detail."
    assert trial.first_input.startswith("Queue three songs")
    assert trial.excerpt.startswith("[… 520 characters earlier]") and trial.excerpt.endswith("the reply ends here")
    assert len(trial.excerpt) < 2100


def test_the_trials_are_capped_with_the_lost_and_failed_kept_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trials_module, "MAX_TRIALS", 3)
    bundle = _bundle(RESULTS)
    assert [trial.state for trial in bundle.trials] == ["failed", "lost", "lost"]
    assert bundle.trials_omitted == 3
    assert bundle.scoreboard[0].ran == 6, "the census counts every trial, kept or not"


def test_the_writer_reads_the_trials_without_what_a_reader_drills_into() -> None:
    payload = json.loads(build_user_message(_bundle(RESULTS)).removeprefix("CONTEXT BUNDLE (JSON):\n"))
    assert "package_version" not in payload
    for trial in payload["trials"]:
        assert not {"first_input", "excerpt", "error_words"} & trial.keys()
        assert {"trial_id", "state", "cause", "failed_checks"} <= trial.keys()


def test_the_build_is_carried_and_left_out_of_the_fingerprint() -> None:
    bundle = _bundle(RESULTS)
    assert bundle.schema_version == 57
    assert bundle.package_version is not None
    assert bundle.model_copy(update={"package_version": "9.9.9"}).fingerprint() == bundle.fingerprint()
