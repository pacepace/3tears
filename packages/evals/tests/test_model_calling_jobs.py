"""Every action whose work calls a model runs it as a job: started in the call, answered through ``job_poll``.

``judge_second``, ``judge_drift_check`` and ``judge_temperature`` used to make their calls inside the call and return
their read; their spend was the only thing that persisted. Each now prices and admits in the call, starts its work as
a job, and the job writes a record of its own — a second judge's pass, a temperature comparison — which ``job_poll``
reads back as the result. A host action with no domain record runs the same way on the generic action record.
Pinned here, over a judged host (:mod:`packages.evals.tests.jobs_conformance_support`):

- **start to result through job_poll** for each converted action, and the foreground form (``await_job``);
- **a refusal in the call** spends nothing and starts nothing;
- **cancel** asked of a held job, from the job contract, lands ``cancelled`` and stops its calls;
- **lost** — a record that has not ended and whose lease lapsed (here: one carrying none) reads lost, written once;
- **an action job's result** is the action's typed result, and its failure the error it raised.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from threetears.evals.actions import Caller
from threetears.evals.kernel import NotFoundError, ValidationFailedError
from threetears.evals.ops import (
    JobStatus,
    OpsHost,
    SecondJudgeRead,
    await_job,
    job_cancel,
    job_poll,
    judge_drift_check,
    judge_second,
    judge_temperature,
    start_action_job,
)
from threetears.evals.run import EvalActionJob, EvalJudgeTemperatureCheck, EvalSecondJudgePass
from threetears.evals.schema import SecondJudge
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.analysis import JudgeTemperatureComparison
from threetears.evals.actions import ConformanceStage
from threetears.evals.testing import CompletionGuard, JobsConformanceHost
from packages.evals.tests.jobs_conformance_support import CONFORMANCE_SCOPE, ScriptedClients, conformance_host
from packages.evals.tests.off_loop import off_loop

_SECOND = "judge/second"


async def _settled(host: OpsHost, job_id: str) -> JobStatus:
    async with asyncio.timeout(10):
        while not (status := await job_poll(host, job_id, CONFORMANCE_SCOPE)).done:
            await asyncio.sleep(0.01)
    return status


class _HeldClients(ScriptedClients):
    """The scripted factory, every call of the second judge held until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.held = 0

    def __call__(self, role: Any, model: str | None, *, temperature: float | None = None) -> Any:
        client = super().__call__(role, model, temperature=temperature)
        if model != _SECOND:
            return client
        generate = client.generate
        clients = self

        async def held(**request: Any) -> Any:
            clients.held += 1
            await clients.gate.wait()
            return await generate(**request)

        client.generate = held  # type: ignore[method-assign]
        return client


async def _judged(clients: ScriptedClients | None = None) -> tuple[JobsConformanceHost, str]:
    built = off_loop(conformance_host, CompletionGuard(), clients)
    stage = ConformanceStage(host=built.host, caller=built.caller, sample=built.sample)
    return built, await stage.finished_run()


async def test_a_second_judge_is_a_job_whose_poll_returns_its_read() -> None:
    built, run_id = await _judged()
    started = await judge_second(built.host, run_id, CONFORMANCE_SCOPE, judge=SecondJudge(model=_SECOND))

    (job,) = started.jobs
    assert job.kind == "second_judge" and job.job_id.startswith("second-judge:") and job.target_id == run_id
    status = await _settled(built.host, job.job_id)

    assert (status.state, status.run_id) == ("completed", run_id) and status.result is not None
    read = SecondJudgeRead.model_validate(status.result)
    assert read.report.pass_id == job.job_id.split(":", 1)[1] and read.report.judged
    assert {row.rubric_dim for row in read.agreement.dimensions} == {"answer.faithful", "answer.clear"}
    stored, _ = off_loop(
        built.host.eval_host.storage.load_job_record, EvalSecondJudgePass, read.report.pass_id, CONFORMANCE_SCOPE
    )
    assert stored is not None and stored.asked_by == "judge_second" and stored.report == read.report


async def test_a_drift_check_is_the_same_job_over_every_result() -> None:
    built, run_id = await _judged()
    (status,) = await await_job(
        built.host,
        await judge_drift_check(built.host, run_id, CONFORMANCE_SCOPE, judge=SecondJudge(model=_SECOND)),
        CONFORMANCE_SCOPE,
    )

    assert status.state == "completed" and status.result is not None
    read = SecondJudgeRead.model_validate(status.result)
    assert read.report.sample_fraction == 1.0 and len(read.report.judged) == read.report.eligible
    stored, _ = off_loop(
        built.host.eval_host.storage.load_job_record, EvalSecondJudgePass, read.report.pass_id, CONFORMANCE_SCOPE
    )
    assert stored is not None and stored.asked_by == "judge_drift_check"


async def test_a_temperature_comparison_is_a_job_whose_poll_returns_the_comparison() -> None:
    built, run_id = await _judged()
    started = await judge_temperature(built.host, run_id, CONFORMANCE_SCOPE, selection="all", repeats=2)

    (job,) = started.jobs
    assert job.kind == "judge_temperature" and job.job_id.startswith("judge-temperature:")
    status = await _settled(built.host, job.job_id)

    assert status.state == "completed" and status.result is not None
    comparison = JudgeTemperatureComparison.model_validate(status.result)
    assert comparison.run_id == run_id and comparison.repeats == 2
    check, _ = off_loop(
        built.host.eval_host.storage.load_job_record,
        EvalJudgeTemperatureCheck,
        job.job_id.split(":", 1)[1],
        CONFORMANCE_SCOPE,
    )
    assert check is not None and check.answers is not None and check.answers.calls_made > 0


async def test_a_judge_action_refused_in_its_call_spends_nothing_and_starts_nothing() -> None:
    clients = ScriptedClients()
    built, run_id = await _judged(clients)
    before = len(clients.built)

    with pytest.raises(ValidationFailedError, match="judge repeat"):
        await judge_second(built.host, run_id, CONFORMANCE_SCOPE, judge=SecondJudge(model="judge/first"))

    assert (
        off_loop(built.host.eval_host.storage.query_out_of_run_spend, CONFORMANCE_SCOPE, purpose="second_judge") == []
    )
    assert len(clients.built) == before, "refused before any client was built"


async def test_a_held_second_judge_is_cancelled_through_the_job_contract() -> None:
    clients = _HeldClients()
    built, run_id = await _judged(clients)
    (job,) = (await judge_second(built.host, run_id, CONFORMANCE_SCOPE, judge=SecondJudge(model=_SECOND))).jobs
    async with asyncio.timeout(10):
        while not clients.held:
            await asyncio.sleep(0.01)

    await job_cancel(built.host, job.job_id, CONFORMANCE_SCOPE, reason="wrong judge")
    cancelled = await _settled(built.host, job.job_id)

    assert (cancelled.state, cancelled.status, cancelled.result) == ("cancelled", "cancelled", None)
    assert cancelled.detail is not None and "wrong judge" in cancelled.detail
    assert not clients.gate.is_set(), "the pass stopped while its calls were held, not after they returned"
    results = off_loop(built.host.eval_host.storage.query_eval_results_by_run, run_id, CONFORMANCE_SCOPE)
    assert all(result.judge_seconds == [] for result in results), "no second score was recorded"
    with pytest.raises(ValidationFailedError, match="not running"):
        await job_cancel(built.host, job.job_id, CONFORMANCE_SCOPE)


@pytest.mark.parametrize(
    "record",
    [
        EvalSecondJudgePass(
            scope_id=CONFORMANCE_SCOPE,
            run_id="run-1",
            asked_by="judge_second",
            judge=SecondJudge(model=_SECOND),
            sample_fraction=1.0,
            sample_seed=0,
        ),
        EvalJudgeTemperatureCheck(scope_id=CONFORMANCE_SCOPE, run_id="run-1", selection="all", repeats=2),
        EvalActionJob(scope_id=CONFORMANCE_SCOPE, action="rubric_propose"),
    ],
    ids=lambda record: record.doc_type,
)
async def test_a_job_record_not_ended_whose_lease_is_gone_reads_lost_written_once(record: Any) -> None:
    built = off_loop(conformance_host, CompletionGuard())
    storage = built.host.eval_host.storage
    off_loop(storage.save_job_record, record.to_dict(), if_match=None)
    prefix = {
        "eval_second_judge_pass": "second-judge:",
        "eval_judge_temperature_check": "judge-temperature:",
        "eval_action_job": "action:",
    }[record.doc_type]

    lost = await job_poll(built.host, f"{prefix}{record.id}", CONFORMANCE_SCOPE)

    assert (lost.state, lost.status, lost.done) == ("lost", "cancelled", True)
    assert lost.detail is not None and "no process can be running it" in lost.detail
    written, _ = off_loop(storage.load_job_record, type(record), record.id, CONFORMANCE_SCOPE)
    assert written is not None and written.lost is not None and written.outcome == "cancelled"
    again = await job_poll(built.host, f"{prefix}{record.id}", CONFORMANCE_SCOPE)
    assert again == lost, "the loss is the record's, read the same every time"


class _Draft(EvalBaseModel):
    """A drafting action's result."""

    text: str


async def test_an_action_job_returns_its_typed_result_and_its_failure() -> None:
    built = off_loop(conformance_host, CompletionGuard())

    async def draft() -> _Draft:
        return _Draft(text="a rubric")

    async def broken() -> _Draft:
        raise ValidationFailedError("the proposer refused the subject", remedy="propose for another subject")

    (done,) = await await_job(
        built.host,
        await start_action_job(built.host, CONFORMANCE_SCOPE, action="rubric_propose", run=draft, label="tone"),
        CONFORMANCE_SCOPE,
    )
    (failed,) = await await_job(
        built.host,
        await start_action_job(built.host, CONFORMANCE_SCOPE, action="rubric_propose", run=broken),
        CONFORMANCE_SCOPE,
    )

    assert (done.state, done.result) == ("completed", {"text": "a rubric"})
    assert (failed.state, failed.result, failed.detail) == (
        "failed",
        None,
        "the proposer refused the subject; propose for another subject",
    )
    record, _ = off_loop(
        built.host.eval_host.storage.load_job_record, EvalActionJob, done.job_id.split(":", 1)[1], CONFORMANCE_SCOPE
    )
    assert record is not None and record.result_model == "_Draft" and record.label == "tone"


async def test_an_action_job_is_answered_only_in_its_scope() -> None:
    built = off_loop(conformance_host, CompletionGuard())

    async def draft() -> _Draft:
        return _Draft(text="x")

    (job,) = (await start_action_job(built.host, CONFORMANCE_SCOPE, action="rubric_propose", run=draft)).jobs
    await _settled(built.host, job.job_id)
    other = Caller(scope_id="another", identity="agent:other")

    with pytest.raises(NotFoundError):
        await job_poll(built.host, job.job_id, other.scope_id)
