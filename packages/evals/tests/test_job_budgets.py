"""A job's time budget and its ending: what the budget bounds, how a job it stops reads, and how its ending lands.

- **A job whose budget runs out reads timed out** — state ``stopped``, status ``timed_out``, a detail naming the
  budget — never ``cancelled``, which is someone deciding to stop it; an operator's cancel of the same work reads
  ``cancelled``.
- **A launch's budget is its preparation's**: a launch whose run waits for a concurrency slot longer than that
  budget is not cancelled for it, and a preparation that hangs past it ends ``timed_out`` with nothing started.
- **Every job kind reads a timeout alike**: a sweep and an analysis generation whose own budget runs out end
  ``timed_out`` (state ``stopped``), never ``cancelled``; the sweep's arms in flight are stopped and none launched after.
- **A job's ending is not lost to a store that fails for a moment**: the write is tried again, and the result lands;
  one the store refuses every time is logged whole, so what the job paid for can be recovered.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest
from pydantic import Field

import threetears.evals.ops.sweeps as sweeps
import threetears.evals.run.job_work as job_work
import threetears.evals.run.launch_job as launch_job
from threetears.evals.kernel import EvalAnalysisAttempt, EvalSweep
from threetears.evals.ops import (
    LaunchArguments,
    SweepArguments,
    SweepArm,
    analysis_generate,
    job_cancel,
    job_poll,
    run_launch,
    start_action_job,
    sweep_launch,
)
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.storage import InMemoryDocumentStore
import packages.evals.tests.fixtures.toyhost.launch as toyhost_launch
from packages.evals.tests.fixtures.toyhost.kind import ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import RUN_MODELS, TOYHOST_SCOPE, TOYHOST_SUBJECT, ops_fixture, settled
from packages.evals.tests.off_loop import off_loop


class _Draft(EvalBaseModel):
    """What a drafting action produces."""

    text: str = Field(description="The draft.")


async def _never_answers() -> _Draft:
    """Work whose model never answers."""
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


def _launch(model: str) -> LaunchArguments:
    return LaunchArguments(template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=[model])


def _held_extraction(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Hold every extraction until the returned event is set, so a launched run stays running."""
    gate = asyncio.Event()
    extract = ScriptedExtractionClient.extract

    async def held(self: ScriptedExtractionClient, request: Any) -> Any:
        await gate.wait()
        return await extract(self, request)

    monkeypatch.setattr(ScriptedExtractionClient, "extract", held)
    return gate


async def test_a_job_whose_budget_runs_out_reads_timed_out_and_a_cancelled_one_reads_cancelled() -> None:
    fixture = off_loop(ops_fixture)
    (timed,) = (
        await start_action_job(fixture.host, TOYHOST_SCOPE, action="draft", run=_never_answers, budget_s=0.1)
    ).jobs
    (asked,) = (
        await start_action_job(fixture.host, TOYHOST_SCOPE, action="draft", run=_never_answers, budget_s=60.0)
    ).jobs

    ran_out = await settled(fixture.host, timed.job_id)
    await job_cancel(fixture.host, asked.job_id, TOYHOST_SCOPE, reason="not needed after all")
    stopped = await settled(fixture.host, asked.job_id)

    assert (ran_out.state, ran_out.status) == ("stopped", "timed_out")
    assert ran_out.detail is not None and "timed out: its 0s wall-clock budget ran out" in ran_out.detail
    assert "cancelled" not in ran_out.detail
    assert (stopped.state, stopped.status) == ("cancelled", "cancelled")
    assert stopped.detail is not None and "not needed after all" in stopped.detail


async def test_a_launch_whose_run_waits_for_a_slot_past_its_preparation_budget_is_not_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _held_extraction(monkeypatch)
    monkeypatch.setattr(launch_job, "LAUNCH_PREPARATION_BUDGET_S", 0.2)
    fixture = off_loop(ops_fixture)
    host = dataclasses.replace(fixture.host, launch=dataclasses.replace(fixture.host.launch, max_concurrent_jobs=1))
    (first,) = (await run_launch(host, _launch(RUN_MODELS[0]), TOYHOST_SCOPE)).jobs
    (second,) = (await run_launch(host, _launch(RUN_MODELS[1]), TOYHOST_SCOPE)).jobs

    await asyncio.sleep(0.6)  # the second launch's run is queued behind the first's for three budgets
    gate.set()

    for job in (first, second):
        status = await settled(host, job.job_id)
        assert (status.state, status.detail) == ("completed", None), job.job_id


async def test_a_launch_whose_preparation_hangs_past_its_budget_ends_timed_out_with_nothing_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def hangs(*_args: Any, **_kwargs: Any) -> Any:
        await asyncio.Event().wait()

    monkeypatch.setattr(launch_job, "LAUNCH_PREPARATION_BUDGET_S", 0.2)
    monkeypatch.setattr(toyhost_launch, "launch_run", hangs)
    fixture = off_loop(ops_fixture)

    (job,) = (await run_launch(fixture.host, _launch(RUN_MODELS[0]), TOYHOST_SCOPE)).jobs
    status = await settled(fixture.host, job.job_id)

    assert (status.state, status.status, status.run_ids) == ("stopped", "timed_out", [])
    assert fixture.host.launch.job_manager.admitted_count == 0, "the launch's reserved room was handed back"


class EndingRefusingStore(InMemoryDocumentStore):
    """Refuses the next ``refusals`` writes of an action job's ending — a store failing for a moment, or for good."""

    def __init__(self, refusals: int) -> None:
        """Refuse that many endings."""
        super().__init__()
        self.refusals = refusals

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Refuse an action job's ending while refusals remain; write anything else."""
        if document.get("doc_type") == "eval_action_job" and document.get("outcome") != "running" and self.refusals:
            self.refusals -= 1
            raise RuntimeError("the store is not answering")
        super().upsert(document, if_match=if_match, create_only=create_only)


async def _drafted() -> _Draft:
    return _Draft(text="the paid-for draft")


async def test_a_job_whose_ending_write_fails_for_a_moment_still_ends_with_its_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(job_work, "ENDING_WRITE_RETRY_DELAYS_S", (0.01, 0.01))
    fixture = off_loop(ops_fixture, store=EndingRefusingStore(refusals=2))

    (job,) = (await start_action_job(fixture.host, TOYHOST_SCOPE, action="draft", run=_drafted)).jobs
    status = await settled(fixture.host, job.job_id)

    assert (status.state, status.result) == ("completed", {"text": "the paid-for draft"})


async def test_a_job_whose_ending_is_never_written_logs_it_whole(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(job_work, "ENDING_WRITE_RETRY_DELAYS_S", (0.01, 0.01))
    fixture = off_loop(ops_fixture, store=EndingRefusingStore(refusals=100))

    (job,) = (await start_action_job(fixture.host, TOYHOST_SCOPE, action="draft", run=_drafted)).jobs
    record_id = job.job_id.split(":", 1)[1]
    async with asyncio.timeout(5):
        while fixture.host.launch.job_manager.is_task_active(record_id):
            await asyncio.sleep(0.01)

    (logged,) = [record for record in caplog.records if "its ending was NOT written after 3 attempts" in record.message]
    assert "the paid-for draft" in logged.message and '"outcome":"completed"' in logged.message


async def test_a_sweep_whose_budget_runs_out_reads_timed_out_and_stops_its_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _held_extraction(monkeypatch)
    monkeypatch.setattr(sweeps, "SWEEP_ARM_BUDGET_S", 0.1)
    fixture = off_loop(ops_fixture)
    arguments = SweepArguments(
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        campaign_name="prompt styles",
        campaign_behavior="extraction accuracy",
        k_runs=1,
        arms=[
            SweepArm(model=RUN_MODELS[0], label=style, overlays={"prompt_style": style})
            for style in ("terse", "verbose")
        ],
    )

    (job,) = (await sweep_launch(fixture.host, arguments, TOYHOST_SCOPE, created_by="agent:test")).jobs
    status = await settled(fixture.host, job.job_id)

    assert (status.state, status.status) == ("stopped", "timed_out")
    assert status.detail is not None and status.detail.startswith("timed out: its 0s wall-clock budget ran out")
    storage = fixture.host.eval_host.storage
    sweep, _ = off_loop(storage.load_job_record, EvalSweep, job.job_id.removeprefix("sweep:"), TOYHOST_SCOPE)
    assert sweep is not None and sweep.arms[0].run_id is not None and sweep.arms[1].run_id is None
    run = off_loop(storage.load_eval_run, sweep.arms[0].run_id, TOYHOST_SCOPE)
    assert run is not None and (run.status, run.cancellation_reason) == ("cancelled", sweeps.SWEEP_TIMEOUT_REASON)
    assert status.progress["arms"]["verbose"] == "not_launched"


async def test_a_generation_whose_budget_runs_out_reads_timed_out() -> None:
    gate = asyncio.Event()
    fixture = off_loop(ops_fixture, gate=gate)
    assert fixture.host.generation is not None
    host = dataclasses.replace(fixture.host, generation=dataclasses.replace(fixture.host.generation, budget_s=0.1))

    (job,) = (await analysis_generate(host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    status = await settled(host, job.job_id)

    assert (status.state, status.status) == ("stopped", "timed_out")
    attempt, _ = off_loop(
        host.eval_host.storage.load_job_record, EvalAnalysisAttempt, job.job_id.rsplit(":", 1)[1], TOYHOST_SCOPE
    )
    assert attempt is not None and attempt.error is not None
    assert attempt.error.startswith("timed out: its 0s wall-clock budget ran out")
    assert (await job_poll(host, job.job_id, TOYHOST_SCOPE)).done
