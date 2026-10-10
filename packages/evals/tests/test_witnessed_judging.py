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

import asyncio
import dataclasses
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.kernel import CandidateOutput, CandidateTelemetry, LeverCoordinateError, ValidationFailedError
from threetears.evals.schema import (
    RoleUsage,
    ConversationStopCause,
    EvalRun,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
    RubricDim,
)
from threetears.evals.kernel.host import EvalHost
from threetears.evals.schema import SweepableValue
from threetears.evals.run import (
    BudgetStoppedError,
    WitnessedJudging,
    record_witnessed_cell,
    rejudge_result,
    stamp_witnessed_judge,
)
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from packages.evals.tests.factories import make_eval_result, make_eval_run
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


#: The host's run cost ceiling, which a judged witnessed run inherits when it names none.
_CEILING = 5.0


def _stamped(host: EvalHost, template: EvalTemplate, run: EvalRun, **ceiling: Any) -> EvalRun:
    resolved: dict[str, Any] = {"configured_max_cost_usd": _CEILING, "enforcement_enabled": True, **ceiling}
    return stamp_witnessed_judge(
        host, run, template, judge_model=_JUDGE, judged_artifact=JudgedArtifact.DOCUMENT, **resolved
    )


def _output(**overrides: Any) -> CandidateOutput:
    fields: dict[str, Any] = {
        "output": [{"role": "gm", "content": "Roll Perception."}],
        "judge_evidence": _EVIDENCE,
        "stop_cause": ConversationStopCause.PARTICIPANTS_ENDED,
    }
    fields.update(overrides)
    return CandidateOutput(**fields)


async def _record(
    host: EvalHost,
    run: EvalRun,
    case: EvalTestCase,
    output: CandidateOutput,
    *,
    judging: WitnessedJudging | None = None,
    result_id: str = "result-7",
) -> Any:
    """Record one cell of ``run`` — through ``judging`` when given, else through judging of its own for this call."""
    return await record_witnessed_cell(
        host,
        run,
        case,
        output,
        k_iteration=1,
        result_id=result_id,
        scored_at=_SESSION_CAPTURED,
        judged_artifact=JudgedArtifact.DOCUMENT,
        judging=judging if judging is not None else WitnessedJudging(run.id),
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
            stamp_witnessed_judge(
                host,
                run,
                template,
                judge_model=_JUDGE,
                judged_artifact=JudgedArtifact.UNJUDGED,
                configured_max_cost_usd=_CEILING,
                enforcement_enabled=True,
            )

    @pytest.mark.parametrize("cap", [0.0, -1.0])
    def test_a_ceiling_that_is_not_positive(self, cap: float) -> None:
        host, template, run, _ = _setup(_ScriptedJudge())
        with pytest.raises(ValidationFailedError, match=f"max_cost_usd must be > 0 \\(got {cap}\\)"):
            _stamped(host, template, run, max_cost_usd=cap)


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
                judging=WitnessedJudging(run.id),
            )

    async def test_no_judging_or_judging_for_another_run(self) -> None:
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        stamped = _stamped(host, template, run)

        with pytest.raises(ValueError, match="none was given: build WitnessedJudging\\('run-witnessed'\\)"):
            await record_witnessed_cell(
                host,
                stamped,
                case,
                _output(),
                k_iteration=1,
                result_id="result-7",
                scored_at=_SESSION_CAPTURED,
                judged_artifact=JudgedArtifact.DOCUMENT,
            )
        with pytest.raises(ValueError, match="the one given judges run run-other"):
            await _record(host, stamped, case, _output(), judging=WitnessedJudging("run-other"))
        assert judge.calls == []
        result, _ = await _record(host, stamped, case, _output(), judging=WitnessedJudging(stamped.id))
        assert [score.dim for score in result.rubric_scores] == [_DIM], "the accepted shape, on the same fixture"


# =============================================================================
# Nothing is paid for a cell that cannot be recorded, and a judged run is held to its ceiling
# =============================================================================


class TestNothingIsJudgedForACellThatCannotBeRecorded:
    """The runner makes these refusals before its judge phase; a judged witnessed cell makes them before its judge."""

    async def test_a_kind_landing_the_derived_accuracy(self) -> None:
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)

        with pytest.raises(ValueError, match="which the engine derives from each observation's 'match'"):
            await _record(
                host, _stamped(host, template, run), case, _output(host_measures={"match": True, "accuracy": 1.0})
            )
        assert judge.calls == [], "the judge was never paid for a record that could not be built"

    async def test_background_spend_reported_twice(self) -> None:
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        usage = [RoleUsage(role="inner_agent", model="scout/model", call_count=1, cost_usd=0.1, price_source="script")]

        with pytest.raises(ValueError, match="reported inner_agent usage on its telemetry"):
            await _record(host, _stamped(host, template, run), case, _output(telemetry=CandidateTelemetry(usage=usage)))
        assert judge.calls == []

    async def test_a_variant_identity_the_host_cannot_derive(self) -> None:
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        uncarried = run.model_copy(
            update={
                "subject_snapshot": run.subject_snapshot.model_copy(
                    update={"components": {"no_lever_carries_this": SweepableValue.of("rev 2", display="rev 2")}}
                )
            }
        )

        with pytest.raises(LeverCoordinateError, match="no_lever_carries_this that no variant lever"):
            await _record(host, _stamped(host, template, uncarried), case, _output())
        assert judge.calls == []


def _judged_at(host: EvalHost, run: EvalRun, case: EvalTestCase, *, result_id: str, judge_cost: float | None) -> None:
    """Save a cell of ``run`` whose judging cost ``judge_cost``, as a host saves each recorded cell."""
    result = make_eval_result(
        id=result_id,
        scope_id=_SCOPE,
        eval_run_id=run.id,
        test_case_id=case.id,
        usage=[RoleUsage(role="judge", model=_JUDGE, cost_usd=judge_cost, price_source="script")],
    )
    host.storage.save_eval_result(result)


class TestAJudgedRunIsHeldToItsCeiling:
    def test_stamping_records_the_ceiling_a_launch_would(self) -> None:
        host, template, run, _ = _setup(_ScriptedJudge())

        assert (_stamped(host, template, run).max_cost_usd, _stamped(host, template, run).max_cost_usd_origin) == (
            _CEILING,
            "inherited",
        )
        chosen = _stamped(host, template, run, max_cost_usd=1.5)
        assert (chosen.max_cost_usd, chosen.max_cost_usd_origin) == (1.5, "chosen")
        uncapped = _stamped(host, template, run, enforcement_enabled=False)
        assert (uncapped.max_cost_usd, uncapped.max_cost_usd_origin) == (None, "uncapped")

    async def test_a_cell_past_the_ceiling_is_refused_before_its_judge(self) -> None:
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        stamped = _stamped(host, template, run, max_cost_usd=1.0)
        host.storage.save_eval_run(stamped)
        _judged_at(host, stamped, case, result_id="earlier", judge_cost=0.6)

        # Inside the ceiling: judged.
        result, _ = await _record(host, stamped, case, _output())
        assert [score.dim for score in result.rubric_scores] == [_DIM]

        _judged_at(host, stamped, case, result_id="later", judge_cost=0.6)
        with pytest.raises(BudgetStoppedError, match=r"\$1\.2000 spent against a \$1\.0000 cap"):
            await _record(host, stamped, case, _output())
        assert len(judge.calls) == 1, "only the cell inside the ceiling was judged"

    async def test_a_judged_cell_the_host_has_not_saved_counts_against_the_next(self) -> None:
        """The store holds only what the host saved; the run's judging counts what it judged and the host has not."""
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        stamped = _stamped(host, template, run, max_cost_usd=1.0)
        _judged_at(host, stamped, case, result_id="earlier", judge_cost=0.9995)
        judging = WitnessedJudging(stamped.id)

        await _record(host, stamped, case, _output(), judging=judging, result_id="first")
        with pytest.raises(BudgetStoppedError, match=r"\$1\.0005 spent against a \$1\.0000 cap"):
            await _record(host, stamped, case, _output(), judging=judging, result_id="second")
        assert len(judge.calls) == 1, "the unsaved first cell's judging stopped the second"

    async def test_cells_recorded_concurrently_are_admitted_one_at_a_time(self) -> None:
        """Two cells checked together would both see room for one; through one judging the second sees the first."""
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        stamped = _stamped(host, template, run, max_cost_usd=1.0)
        _judged_at(host, stamped, case, result_id="earlier", judge_cost=0.9995)
        judging = WitnessedJudging(stamped.id)

        outcomes = await asyncio.gather(
            _record(host, stamped, case, _output(), judging=judging, result_id="first"),
            _record(host, stamped, case, _output(), judging=judging, result_id="second"),
            return_exceptions=True,
        )

        assert sum(isinstance(outcome, BudgetStoppedError) for outcome in outcomes) == 1, outcomes
        assert len(judge.calls) == 1, "the ceiling was crossed by one cell, the runner's bound, never by the batch"

    async def test_a_cell_the_host_saved_is_counted_once(self) -> None:
        """Once the store holds a judged cell, its spend is read from there and not again from the judging."""
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        stamped = _stamped(host, template, run, max_cost_usd=1.0)
        _judged_at(host, stamped, case, result_id="earlier", judge_cost=0.9985)
        judging = WitnessedJudging(stamped.id)

        first, trace = await _record(host, stamped, case, _output(), judging=judging, result_id="first")
        host.storage.save_eval_result(first, trace)
        second, _ = await _record(host, stamped, case, _output(), judging=judging, result_id="second")

        assert [score.dim for score in second.rubric_scores] == [_DIM], (
            "$0.9995 spent, not $1.0005 with the saved cell counted twice"
        )

    @pytest.mark.parametrize("saved_between", [False, True], ids=["unsaved", "saved-between"])
    async def test_a_cell_recorded_twice_counts_both_judgements(self, saved_between: bool) -> None:
        """Re-recording a cell judges it again, and both judgements were paid for, though the store keeps one."""
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        stamped = _stamped(host, template, run, max_cost_usd=1.0)
        _judged_at(host, stamped, case, result_id="earlier", judge_cost=0.9985)
        judging = WitnessedJudging(stamped.id)

        for _ in range(2):
            result, trace = await _record(host, stamped, case, _output(), judging=judging, result_id="first")
            if saved_between:
                host.storage.save_eval_result(result, trace)
        with pytest.raises(BudgetStoppedError, match=r"\$1\.0005 spent against a \$1\.0000 cap"):
            await _record(host, stamped, case, _output(), judging=judging, result_id="second")
        assert len(judge.calls) == 2

    async def test_re_recording_a_cell_the_store_held_counts_the_judgement_it_replaces(self) -> None:
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        stamped = _stamped(host, template, run, max_cost_usd=1.0)
        _judged_at(host, stamped, case, result_id="earlier", judge_cost=0.9985)
        _judged_at(host, stamped, case, result_id="first", judge_cost=0.001)
        judging = WitnessedJudging(stamped.id)

        result, trace = await _record(host, stamped, case, _output(), judging=judging, result_id="first")
        host.storage.save_eval_result(result, trace)
        with pytest.raises(BudgetStoppedError, match=r"\$1\.0005 spent against a \$1\.0000 cap"):
            await _record(host, stamped, case, _output(), judging=judging, result_id="second")

    async def test_unpriced_judge_spend_stops_an_enforced_ceiling(self) -> None:
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        stamped = _stamped(host, template, run)
        _judged_at(host, stamped, case, result_id="earlier", judge_cost=None)

        with pytest.raises(BudgetStoppedError, match="could not be priced"):
            await _record(host, stamped, case, _output())
        assert judge.calls == []

    async def test_an_uncapped_run_judges_whatever_it_has_spent(self) -> None:
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        stamped = _stamped(host, template, run, enforcement_enabled=False)
        _judged_at(host, stamped, case, result_id="earlier", judge_cost=None)

        result, _ = await _record(host, stamped, case, _output())
        assert [score.dim for score in result.rubric_scores] == [_DIM]

    async def test_a_judged_run_recording_no_ceiling_is_refused(self) -> None:
        judge = _ScriptedJudge()
        host, template, run, case = _setup(judge)
        unbounded = _stamped(host, template, run).model_copy(update={"max_cost_usd": None, "max_cost_usd_origin": None})

        with pytest.raises(ValueError, match="records no cost ceiling"):
            await _record(host, unbounded, case, _output())
        assert judge.calls == []


_GOAL_CHECK = 'call_count("t.act") == 1'


def _with_a_goal_check(template: EvalTemplate) -> EvalTemplate:
    """``template`` declaring one goal check; the caller saves it over the stored one, which the run's judge reads."""
    return template.model_copy(update={"goal_state_checks": [_GOAL_CHECK]})


async def test_a_failed_witnessed_cell_stores_every_check_of_the_template_it_is_judged_against_as_failed() -> None:
    """The stored result — not a helper's return — carries the not-evaluated, failed outcome, so per-check rates count it."""
    judge = _ScriptedJudge()
    host, template, run, case = _setup(judge)
    template = _with_a_goal_check(template)
    host.storage.save_template(template)

    result, _trace = await _record(
        host, _stamped(host, template, run), case, _output(candidate_errors=["the player walked out mid-ruling"])
    )

    assert [(fact.expression, fact.passed) for fact in result.goal_state_outcomes] == [(_GOAL_CHECK, False)]
    assert result.goal_state_outcomes[0].detail.startswith("not evaluated: the candidate failed")


async def test_a_clean_witnessed_cell_that_graded_none_of_the_template_s_checks_is_refused_before_the_judge_is_paid() -> (
    None
):
    judge = _ScriptedJudge()
    host, template, run, case = _setup(judge)
    template = _with_a_goal_check(template)
    host.storage.save_template(template)

    with pytest.raises(ValueError, match="without grading the template's goal check"):
        await _record(host, _stamped(host, template, run), case, _output())

    assert judge.calls == [], "refused before any judge call was bought"


class _ThreadRecordingStore:
    """The host's store, recording the thread each call ran on."""

    def __init__(self, inner: Any, threads: list[tuple[str, int]]) -> None:
        self._inner = inner
        self._threads = threads

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute

        def recorded(*args: Any, **kwargs: Any) -> Any:
            self._threads.append((name, threading.get_ident()))
            return attribute(*args, **kwargs)

        return recorded


async def test_no_store_call_from_recording_a_judged_witnessed_cell_runs_on_the_loop_thread() -> None:
    """The template and config loads, and the ceiling's read of saved cells, all leave the event loop."""
    judge = _ScriptedJudge()
    host, template, run, case = _setup(judge)
    stamped = _stamped(host, template, run)
    threads: list[tuple[str, int]] = []
    loop_thread = threading.get_ident()
    recording = dataclasses.replace(host, storage=_ThreadRecordingStore(host.storage, threads))

    result, _trace = await _record(recording, stamped, case, _output())

    assert [score.dim for score in result.rubric_scores] == [_DIM]
    called = {name for name, _ in threads}
    assert {"load_template", "query_eval_results_by_run"} <= called, (
        f"the precondition: the store was read, so no call passes by not happening — {sorted(called)}"
    )
    assert all(ident != loop_thread for _, ident in threads), f"a store call ran on the loop: {threads}"
