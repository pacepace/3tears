"""A witnessed run can be judged: it names the template it is judged against, and each cell is scored as it is recorded.

A witnessed session has no template that set it, and the only judging path the engine had — a re-judge —
needs a template and a judge error, which a witnessed cell had neither of, so real sessions could never
be scored. :func:`~threetears.evals.run.stamp_witnessed_judge` gives the run the template and the judge
apparatus a launch would stamp, before its identity is stamped; :func:`~threetears.evals.run.record_witnessed_cell`
then scores each cell through the runner's own judge phase from what the run recorded, and the cell is
one a later re-judge reads like any run's.

Each refusal is driven beside the accepted shape on the same fixture.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.contracts import (
    CandidateOutput,
    ConversationStopCause,
    EvalRun,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
    RubricDim,
    ValidationFailedError,
)
from threetears.evals.contracts.host import EvalHost
from threetears.evals.run import record_witnessed_cell, rejudge_result, stamp_witnessed_judge
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host

_SCOPE = "toy"
_KIND = "table-session"
_JUDGE = "judge/scripted"
_DIM = "play.fair_rulings"
_TEMPLATE_EDITED = "2026-10-01T09:00:00+00:00"
_SESSION_CAPTURED = "2026-10-05T12:00:00+00:00"

_EVIDENCE = JudgeEvidence(
    subject="The game master, at a real table.",
    case_material="Two players; the session ended when they left.",
    artifact="Player: I check the floor.\nGM: Roll Perception.",
)


@dataclass(frozen=True)
class _Completion:
    content: str
    stop_reason: str = "end_turn"
    input_tokens: int = 10
    output_tokens: int = 5
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    model: str = _JUDGE
    served_model: str | None = None


@dataclass
class _ScriptedJudge:
    """Scores every dim 4, or answers unparseable text for a failing one, and records what it was sent."""

    failing: set[str] = field(default_factory=set)
    calls: list[tuple[str, str]] = field(default_factory=list)
    model_name: str = _JUDGE

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> _Completion:
        match = re.search(r'the single key "(.+?)"', system)
        dim = match.group(1) if match else "?"
        self.calls.append((dim, user))
        reply = {"reasoning": "the ruling followed the dice", "criteria_scores": {dim: 4}}
        return _Completion(content="not json" if dim in self.failing else json.dumps(reply))

    async def aclose(self) -> None:
        """Nothing to release."""


def _setup(judge: _ScriptedJudge) -> tuple[EvalHost, EvalTemplate, EvalRun, EvalTestCase]:
    """A host that judges with ``judge``, the template a session is judged against, and the unjudged witnessed run."""
    host = toyhost_host(clients=lambda role, model, *, temperature=None: judge)
    template = EvalTemplate(
        scope_id=_SCOPE,
        name="fair table",
        intent="Rule what the dice say, at a table of real players.",
        candidate_kind=_KIND,
        rubric=[RubricDim(name=_DIM, description="the GM rules what the dice say", scale="ordinal")],
        created_at=_TEMPLATE_EDITED,
        updated_at=_TEMPLATE_EDITED,
    )
    host.storage.save_template(template)
    case = EvalTestCase(id="session-7", scope_id=_SCOPE, template_id=None, variation_params={"table": "tuesday"})
    host.storage.save_test_case(case)
    run = make_eval_run(
        id="run-witnessed",
        scope_id=_SCOPE,
        template_id=None,
        candidate_kind=_KIND,
        candidate_model="gm/model",
        test_case_ids=[case.id],
        apparatus_provenance="witnessed",
        created_at=_SESSION_CAPTURED,
    )
    return host, template, run, case


def _stamped(host: EvalHost, template: EvalTemplate, run: EvalRun) -> EvalRun:
    return stamp_witnessed_judge(host, run, template, judge_model=_JUDGE, judged_artifact=JudgedArtifact.DOCUMENT)


def _output(**overrides: Any) -> CandidateOutput:
    fields: dict[str, Any] = {
        "output": [{"role": "gm", "content": "Roll Perception."}],
        "judge_evidence": _EVIDENCE,
        "stop_cause": ConversationStopCause.PARTICIPANTS_ENDED,
    }
    fields.update(overrides)
    return CandidateOutput(**fields)


async def _record(host: EvalHost, run: EvalRun, case: EvalTestCase, output: CandidateOutput) -> Any:
    return await record_witnessed_cell(
        host,
        run,
        case,
        output,
        k_iteration=1,
        result_id="result-7",
        scored_at=_SESSION_CAPTURED,
        judged_artifact=JudgedArtifact.DOCUMENT,
    )


def test_stamping_gives_the_run_the_template_and_the_judge_a_launch_would() -> None:
    host, template, run, _case = _setup(_ScriptedJudge())

    stamped = _stamped(host, template, run)

    assert (stamped.template_id, stamped.judge_model) == (template.id, _JUDGE)
    assert (stamped.effective_judges, stamped.effective_judges_source) == ({_DIM: _JUDGE}, "recorded")
    assert (stamped.judge_config_ids, stamped.judge_config_provenance) == ({}, {})
    assert stamped.judge_request_settings == JUDGE_REQUEST_SETTINGS
    assert stamped.rubric_scales == {_DIM: "ordinal"}
    assert stamped.model_role_provenance["judge"] == "chosen"
    assert (run.template_id, run.judge_model) == (None, None), "a copy; the run as built is unchanged"


async def test_a_judged_witnessed_cell_is_scored_from_the_evidence_through_the_runner_s_judge_phase() -> None:
    judge = _ScriptedJudge()
    host, template, run, case = _setup(judge)

    result, trace = await _record(host, _stamped(host, template, run), case, _output())

    assert [score.dim for score in result.rubric_scores] == [_DIM]
    assert result.rubric_scores[0].score == 4
    assert (result.judge_model, result.judge_error) == (_JUDGE, None)
    assert result.latency is not None and result.latency.judge_ms is not None
    assert [dim for dim, _ in judge.calls] == [_DIM], "a document is asked its rubric alone"
    ((_, user),) = judge.calls
    assert user.endswith(f"# Output under review\n{_EVIDENCE.artifact}")
    assert trace.judge_evidence == _EVIDENCE


async def test_a_judged_witnessed_cell_whose_judge_failed_is_rejudged_like_any_run_s() -> None:
    """The witnessed cell is the record a run's judged cell is, so the re-judge path reads it unchanged."""
    judge = _ScriptedJudge(failing={_DIM})
    host, template, run, case = _setup(judge)
    stamped = _stamped(host, template, run).model_copy(update={"status": "completed"})
    result, trace = await _record(host, stamped, case, _output())
    assert result.rubric_scores == [] and result.judge_error is not None
    host.storage.save_eval_run(stamped)
    host.storage.save_eval_result(result, trace)
    judge.failing.clear()

    rejudged = await rejudge_result(host, result.id, _SCOPE)

    assert [score.dim for score in rejudged.rubric_scores] == [_DIM] and rejudged.judge_error is None


async def test_an_unjudged_witnessed_run_calls_no_judge_and_an_empty_output_is_not_judged() -> None:
    judge = _ScriptedJudge()
    host, template, run, case = _setup(judge)

    unjudged, _ = await record_witnessed_cell(
        host,
        run,
        case,
        _output(judge_evidence=None),
        k_iteration=1,
        result_id="result-7",
        scored_at=_SESSION_CAPTURED,
        judged_artifact=JudgedArtifact.UNJUDGED,
    )
    empty, _ = await _record(host, _stamped(host, template, run), case, _output(output=[], judge_evidence=None))

    assert unjudged.judge_model is None and unjudged.rubric_scores == []
    assert empty.judge_model == _JUDGE and empty.rubric_scores == [] and empty.judge_error is None
    assert judge.calls == [], "nothing produced, nothing judged — the runner's own gate"


class TestStampingRefuses:
    def test_a_commissioned_run(self) -> None:
        host, template, run, _ = _setup(_ScriptedJudge())
        with pytest.raises(ValueError, match="a launch stamps a commissioned run's judge"):
            _stamped(host, template, run.model_copy(update={"apparatus_provenance": "commissioned"}))

    def test_a_run_already_judged(self) -> None:
        host, template, run, _ = _setup(_ScriptedJudge())
        with pytest.raises(ValueError, match="a run's judge is stamped once"):
            _stamped(host, template, _stamped(host, template, run))

    def test_a_run_whose_identity_is_already_stamped(self) -> None:
        host, template, run, _ = _setup(_ScriptedJudge())
        with pytest.raises(ValueError, match="stamp the judge first, then the identity"):
            _stamped(host, template, run.model_copy(update={"context_key": "ctx-1"}))

    def test_a_template_of_another_scope_or_kind(self) -> None:
        host, template, run, _ = _setup(_ScriptedJudge())
        with pytest.raises(ValueError, match="a run is judged against a template of its own scope"):
            _stamped(host, template.model_copy(update={"scope_id": "elsewhere"}), run)
        with pytest.raises(ValueError, match="written for another kind's output"):
            _stamped(host, template.model_copy(update={"candidate_kind": "router"}), run)

    def test_an_unjudged_kind(self) -> None:
        host, template, run, _ = _setup(_ScriptedJudge())
        with pytest.raises(ValueError, match="unjudged kind has no judge to build"):
            stamp_witnessed_judge(host, run, template, judge_model=_JUDGE, judged_artifact=JudgedArtifact.UNJUDGED)


class TestRecordingAJudgedCellRefuses:
    async def test_a_template_edited_after_the_run_was_created(self) -> None:
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        stamped = _stamped(host, template, run)
        host.storage.save_template(template.model_copy(update={"updated_at": "2026-10-06T00:00:00+00:00"}))

        with pytest.raises(
            ValidationFailedError, match="the judge would read an intent the run was not judged against"
        ):
            await _record(host, stamped, case, _output())
        assert judge.calls == []

    async def test_request_settings_other_than_the_ones_a_call_sends_now(self) -> None:
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        stale = _stamped(host, template, run).model_copy(
            update={"judge_request_settings": JUDGE_REQUEST_SETTINGS.model_copy(update={"max_tokens": 1})}
        )

        with pytest.raises(ValidationFailedError, match="a new call would be asked differently"):
            await _record(host, stale, case, _output())
        assert judge.calls == []

    async def test_a_kind_that_declares_nothing_a_judge_reads(self) -> None:
        host, template, run, case = _setup(_ScriptedJudge())
        with pytest.raises(ValueError, match="declares nothing a judge reads"):
            await record_witnessed_cell(
                host,
                _stamped(host, template, run),
                case,
                _output(judge_evidence=None),
                k_iteration=1,
                result_id="result-7",
                scored_at=_SESSION_CAPTURED,
                judged_artifact=JudgedArtifact.UNJUDGED,
            )
