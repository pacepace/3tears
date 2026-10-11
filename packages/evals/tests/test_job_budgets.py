"""A job's time budget: what it bounds, and how a job it stops reads, over the toy host.

- **A job whose budget runs out reads timed out** — state ``stopped``, status ``timed_out``, a detail naming the
  budget — never ``cancelled``, which is someone deciding to stop it; an operator's cancel of the same work reads
  ``cancelled``.
- **A launch's budget is its preparation's**: a launch whose run waits for a concurrency slot longer than that
  budget is not cancelled for it, and a preparation that hangs past it ends ``timed_out`` with nothing started.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest
from pydantic import Field

import threetears.evals.run.launch_job as launch_job
from threetears.evals.ops import LaunchArguments, job_cancel, run_launch, start_action_job
from threetears.evals.schema.base import EvalBaseModel
import packages.evals.tests.fixtures.toyhost.launch as toyhost_launch
from packages.evals.tests.fixtures.toyhost.kind import ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import RUN_MODELS, TOYHOST_SCOPE, TOYHOST_SUBJECT, ops_fixture, settled


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
    fixture = ops_fixture()
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
    fixture = ops_fixture()
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
    fixture = ops_fixture()

    (job,) = (await run_launch(fixture.host, _launch(RUN_MODELS[0]), TOYHOST_SCOPE)).jobs
    status = await settled(fixture.host, job.job_id)

    assert (status.state, status.status, status.run_ids) == ("stopped", "timed_out", [])
    assert fixture.host.launch.job_manager.admitted_count == 0, "the launch's reserved room was handed back"
