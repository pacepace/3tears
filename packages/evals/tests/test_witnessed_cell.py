"""Recording a cell a host witnessed: the runner's own assembly, its refusals, and what the host writes around it.

:func:`~threetears.evals.run.record_witnessed_cell` builds a cell's result and trace from a
:class:`~threetears.evals.contracts.CandidateOutput` the host observed, through the same function the
runner assembles every completed cell with. The first test is the one that holds that: the toy host's
real run, every cell's output captured as the runner received it, re-recorded through the public
operation under the runner's own ids — and the two records are equal, field for field.

Then each refusal fires (the run's provenance, a judge model, the case's placement, the repeat, a
simulator's stop cause, the kind's declaration), and the shapes a witnessed session needs that a
commissioned one never did: a case under no template, a conversation its participants ended, and the
run's completeness, which the host stamps itself.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.contracts import (
    CandidateKindDefect,
    CandidateOutput,
    CandidateTelemetry,
    CellSummary,
    ConversationStopCause,
    EvalResult,
    EvalRun,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
    RoleUsage,
    summarize_completeness,
)
from threetears.evals.contracts.host import CellTrace, EvalHost
from threetears.evals.contracts.models import eval_trace_doc_id
from threetears.evals.run import RunnerOptions, execute_run, record_witnessed_cell
from threetears.evals.run.runner import RunCallbacks
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND, ScriptedExtractionClient, ToyExtractorKind
from packages.evals.tests.fixtures.toyhost.run import (
    RUN_MODELS,
    toyhost_arming_template,
    toyhost_run,
    toyhost_test_cases,
)

#: The instant every hand-recorded cell here is stamped with.
_AT = "2026-10-05T12:00:00+00:00"


class _RecordingKind:
    """The toy extractor, wrapped so each output it hands the runner is kept for the cell it ended.

    Delegates every call; the only addition is :attr:`last`, read by the run's ``on_result`` callback,
    which fires once per cell after that cell's ``invoke`` — so the pairing is exact.
    """

    def __init__(self, inner: ToyExtractorKind) -> None:
        self.inner = inner
        self.judged_artifact = inner.judged_artifact
        self.last: CandidateOutput | None = None

    async def prepare(self, **kwargs: Any) -> Any:
        return await self.inner.prepare(**kwargs)

    async def invoke(self, instance: Any, test_case: EvalTestCase, sink: Any) -> CandidateOutput:
        self.last = await self.inner.invoke(instance, test_case, sink)
        return self.last


async def _runner_cells() -> tuple[EvalHost, EvalRun, list[EvalTestCase], list[tuple[EvalResult, CandidateOutput]]]:
    """Run the toy host's arming template on one model through the real runner, keeping each cell's output.

    Returns:
        The host, the run, its cases, and each stored result beside the output its kind returned.
    """
    host = toyhost_host()
    world = host.profile.world
    assert world is not None
    template = toyhost_arming_template()
    inner = ToyExtractorKind(
        client=ScriptedExtractionClient(), world=world, judged=False, goal_checks=tuple(template.goal_state_checks)
    )
    kind = _RecordingKind(inner)
    cases = toyhost_test_cases(template)
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=inner, world=world)
    host.storage.save_eval_run(run)
    paired: list[tuple[EvalResult, CandidateOutput]] = []

    async def on_result(result: EvalResult) -> None:
        assert kind.last is not None
        paired.append((result, kind.last))

    started = datetime.now(UTC)
    await execute_run(
        host,
        run=run,
        template=template,
        test_cases=cases,
        judge_service=None,
        options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}),
        callbacks=RunCallbacks(on_result=on_result),
    )
    # The runner still mints each cell's own id and stamp at the cell, now passed into the assembly.
    finished = datetime.now(UTC)
    assert all(started <= datetime.fromisoformat(result.scored_at) <= finished for result, _ in paired)
    assert len({result.id for result, _ in paired}) == len(paired)
    # The run names the cases it ran — the denominator the witnessed op holds a case to.
    assert run.test_case_ids == [case.id for case in cases]
    return host, run, cases, paired


async def test_a_witnessed_cell_is_the_runners_cell_for_the_same_output() -> None:
    """One assembly path: every cell the runner recorded, re-recorded from its output, is the same record."""
    host, run, cases, paired = await _runner_cells()
    assert run.apparatus_provenance == "witnessed"
    by_id = {case.id: case for case in cases}
    assert len(paired) == len(cases) * run.k_runs
    # The arming template fires a trigger, so the world-event and end-state hops are exercised, not empty.
    assert all(result.world_events for result, _ in paired)
    for runner_result, output in paired:
        runner_trace = host.storage.load_eval_trace(runner_result.id, runner_result.scope_id)
        assert runner_trace is not None
        result, trace = record_witnessed_cell(
            host,
            run,
            by_id[runner_result.test_case_id],
            output,
            k_iteration=runner_result.k_iteration,
            result_id=runner_result.id,
            scored_at=runner_result.scored_at,
            judged_artifact=JudgedArtifact.UNJUDGED,
            world_events=runner_result.world_events,
            end_state=runner_trace.end_state,
        )
        # ``has_trace`` is the store's fact about the write, set when the runner's pair was saved.
        assert result.model_dump(exclude={"has_trace"}) == runner_result.model_dump(exclude={"has_trace"})
        assert trace.model_dump() == runner_trace.model_dump()


def _witnessed_run(**overrides: Any) -> tuple[EvalHost, EvalRun, EvalTestCase]:
    """A witnessed run under no template, with one template-less case, as a capturing host writes them."""
    host = toyhost_host()
    case = EvalTestCase(
        id="session-1", scope_id="toy", template_id=None, variation_params={"table": "tuesday"}, host_payload={}
    )
    fields: dict[str, Any] = {
        "id": "run-witnessed",
        "scope_id": "toy",
        "template_id": None,
        "candidate_model": "gm/model",
        "candidate_kind": TOY_EXTRACTOR_KIND,
        "test_case_ids": [case.id],
        "apparatus_provenance": "witnessed",
    }
    fields.update(overrides)
    return host, make_eval_run(**fields), case


def _output(**overrides: Any) -> CandidateOutput:
    fields: dict[str, Any] = {
        "output": [{"role": "player", "content": "we open the door"}, {"role": "gm", "content": "it creaks"}],
        "telemetry": CandidateTelemetry(
            usage=[RoleUsage(role="candidate", model="gm/model", call_count=2, cost_usd=0.5, price_source="script")]
        ),
        "stop_cause": ConversationStopCause.PARTICIPANTS_ENDED,
    }
    fields.update(overrides)
    return CandidateOutput(**fields)


def _record(host: EvalHost, run: EvalRun, case: EvalTestCase, output: CandidateOutput, **overrides: Any) -> Any:
    call: dict[str, Any] = {
        "k_iteration": 1,
        "result_id": "result-1",
        "scored_at": _AT,
        "judged_artifact": JudgedArtifact.UNJUDGED,
    }
    call.update(overrides)
    return record_witnessed_cell(host, run, case, output, **call)


def test_a_session_its_participants_ended_is_recorded_under_no_template_with_the_callers_ids() -> None:
    host, run, case = _witnessed_run()
    result, trace = _record(host, run, case, _output())
    assert (result.id, result.scored_at, trace.id) == ("result-1", _AT, eval_trace_doc_id("result-1"))
    assert result.stop_cause is ConversationStopCause.PARTICIPANTS_ENDED
    assert (result.test_case_id, result.candidate_kind, result.model) == ("session-1", TOY_EXTRACTOR_KIND, "gm/model")
    assert result.judge_model is None and result.termination == "completed"
    assert result.cost_usd == 0.5
    # Persisted by the host, not by the op: the case and the pair round-trip through the engine's store.
    host.storage.save_test_case(case)
    host.storage.save_eval_result(result, trace)
    assert host.storage.load_test_case("session-1", "toy") == case
    stored = host.storage.load_eval_result("result-1", "toy")
    assert stored is not None and stored.stop_cause is ConversationStopCause.PARTICIPANTS_ENDED


def test_recording_the_same_observation_twice_is_the_same_record() -> None:
    host, run, case = _witnessed_run()
    first = _record(host, run, case, _output())
    second = _record(host, run, case, _output())
    assert [doc.model_dump() for doc in first] == [doc.model_dump() for doc in second]


def test_spans_the_host_collected_land_on_the_trace_and_the_latency() -> None:
    host, run, case = _witnessed_run()
    spans = CellTrace(spans=[{"name": "agent.invoke"}], total_ms=1200.0, llm_ms=900.0, tool_ms=None)
    result, trace = _record(host, run, case, _output(), spans=spans)
    assert trace.otel_trace == [{"name": "agent.invoke"}]
    assert result.latency is not None
    assert (result.latency.total_ms, result.latency.llm_ms, result.latency.tool_ms) == (1200.0, 900.0, None)


def test_a_witnessed_runs_completeness_is_summarized_from_the_cells_its_host_recorded() -> None:
    host, run, case = _witnessed_run(k_runs=2)
    cells = [
        CellSummary.from_result(_record(host, run, case, _output(), k_iteration=k, result_id=f"r-{k}")[0], persisted=p)
        for k, p in ((1, True), (2, False))
    ]
    completeness = summarize_completeness(run, cells)
    assert (completeness.expected_cells, completeness.produced_cells, completeness.persisted_cells) == (2, 2, 1)
    assert completeness.degraded


@pytest.mark.parametrize(
    ("run_overrides", "case_overrides", "call_overrides", "names"),
    [
        pytest.param(
            {"apparatus_provenance": "commissioned"}, {}, {}, "only a witnessed run's cells", id="commissioned-run"
        ),
        pytest.param({"judge_model": "judge/model"}, {}, {}, "names judge model", id="run-names-a-judge"),
        pytest.param({"test_case_ids": ["session-2"]}, {}, {}, "is not one of run", id="case-outside-the-run"),
        pytest.param({}, {"scope_id": "elsewhere"}, {}, "a witnessed cell's case belongs", id="case-in-another-scope"),
        pytest.param({}, {"template_id": "tpl-1"}, {}, "a witnessed cell's case belongs", id="case-under-a-template"),
        pytest.param({}, {}, {"k_iteration": 2}, r"outside run run-witnessed's repeats \(1\.\.1\)", id="k-past-k-runs"),
        pytest.param({}, {}, {"k_iteration": 0}, r"outside run run-witnessed's repeats", id="k-zero"),
    ],
)
def test_a_cell_that_does_not_belong_to_a_witnessed_run_is_refused(
    run_overrides: dict[str, Any], case_overrides: dict[str, Any], call_overrides: dict[str, Any], names: str
) -> None:
    host, run, case = _witnessed_run(**run_overrides)
    case = case.model_copy(update=case_overrides)
    with pytest.raises(ValueError, match=names):
        _record(host, run, case, _output(), **call_overrides)


@pytest.mark.parametrize("cause", [ConversationStopCause.USER_DONE, ConversationStopCause.SIMULATOR_ERROR])
def test_a_simulators_stop_cause_is_refused_on_a_session_with_no_simulator(cause: ConversationStopCause) -> None:
    host, run, case = _witnessed_run()
    with pytest.raises(ValueError, match=f"cannot have stopped on '{cause.value}'"):
        _record(host, run, case, _output(stop_cause=cause))


@pytest.mark.parametrize("cause", [None, ConversationStopCause.MAX_TURNS, ConversationStopCause.APPARATUS_ERROR])
def test_every_other_stop_cause_is_recorded(cause: ConversationStopCause | None) -> None:
    host, run, case = _witnessed_run()
    assert _record(host, run, case, _output(stop_cause=cause))[0].stop_cause == cause


def test_output_contradicting_the_kinds_declaration_is_refused() -> None:
    host, run, case = _witnessed_run()
    with pytest.raises(CandidateKindDefect):
        _record(host, run, case, _output(), judged_artifact=JudgedArtifact.TRANSCRIPT)
    evidence = JudgeEvidence(case_material="the dungeon", artifact="we open the door")
    with pytest.raises(CandidateKindDefect):
        _record(host, run, case, _output(judge_evidence=evidence))
    # The declaration honoured: evidence rendered for a transcript kind is kept with its declaration.
    _, trace = _record(host, run, case, _output(judge_evidence=evidence), judged_artifact=JudgedArtifact.TRANSCRIPT)
    assert (trace.judge_evidence, trace.judged_artifact) == (evidence, JudgedArtifact.TRANSCRIPT)


def test_background_spend_reported_twice_is_refused() -> None:
    host, run, case = _witnessed_run()
    usage = [RoleUsage(role="inner_agent", model="scout/model", call_count=1, cost_usd=0.1, price_source="script")]
    with pytest.raises(ValueError, match="reported inner_agent usage on its telemetry"):
        _record(host, run, case, _output(telemetry=CandidateTelemetry(usage=usage)))


def test_a_kind_landing_the_derived_accuracy_is_refused() -> None:
    host, run, case = _witnessed_run()
    with pytest.raises(ValueError, match="which the engine derives from each observation's 'match'"):
        _record(host, run, case, _output(host_measures={"match": True, "accuracy": 1.0}))
    # The verdict alone is the shape a classifier lands.
    assert _record(host, run, case, _output(host_measures={"match": True}))[0].host_measures == {"match": True}


# --- a case under no template --------------------------------------------------------------------


def test_a_case_must_state_its_template_even_when_it_has_none() -> None:
    with pytest.raises(ValidationError, match="template_id"):
        EvalTestCase(id="c", scope_id="toy")  # type: ignore[call-arg]
    assert EvalTestCase(id="c", scope_id="toy", template_id=None).template_id is None
