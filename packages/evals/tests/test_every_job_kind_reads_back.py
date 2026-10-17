"""A completed job of every kind reads back through ``job_poll``'s own render.

``job_poll`` renders a completed job's result as its own action renders it, and a kind's result model was imported
only for type checking, so every poll of a completed temperature comparison raised ``NameError`` while the job and
its record were fine. Each kind is driven here to a completed job through the engine's own work — every long action's
happy path over the judged conformance host (an analysis generation over the ops fixture, whose writer authors a
memo the contract takes), the runs a launch names, and a host action — and its status is rendered
through the catalogue, as a transport does. The kinds are read off ``JobKind``, so a kind added without a way here to
complete one fails rather than going unread.
"""

from __future__ import annotations

import asyncio
from typing import get_args

from threetears.evals.actions import (
    ActionOutcome,
    Caller,
    ConformanceStage,
    MountedTool,
    eval_catalogue,
    standard_tools,
)
from threetears.evals.ops import JobKind, JobStatus, OpsHost, job_poll, start_action_job
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.testing import CompletionGuard
from packages.evals.tests.jobs_conformance_support import CONFORMANCE_SCOPE, conformance_host
from packages.evals.tests.ops_support import CALLER, ops_fixture
from packages.evals.tests.off_loop import off_loop


class _Drafted(EvalBaseModel):
    """A host action's result."""

    text: str


async def _settled(host: OpsHost, caller: Caller, job_id: str) -> JobStatus:
    async with asyncio.timeout(30):
        while not (status := await job_poll(host, job_id, caller.scope_id)).done:
            await asyncio.sleep(0.01)
    return status


async def _poll(tool: MountedTool, host: OpsHost, caller: Caller, job_id: str) -> ActionOutcome:
    return await tool.call({"action": "job_poll", "job_id": job_id}, host=host, caller=caller)


async def test_a_completed_job_of_every_kind_renders_through_job_poll() -> None:
    built = off_loop(conformance_host, CompletionGuard())
    host, caller = built.host, built.caller
    catalogue = eval_catalogue()
    evals, _admin = catalogue.mount_all(standard_tools())
    stage = ConformanceStage(host=host, caller=caller, sample=built.sample)

    #: Each job to settle and render, with the host and caller it is polled through.
    pending: list[tuple[OpsHost, Caller, str]] = []
    analysis = off_loop(ops_fixture)
    generated = await evals.call(
        {"action": "analysis_generate", "campaign_id": analysis.campaign.id}, host=analysis.host, caller=CALLER
    )
    assert generated.structured is not None, generated.text
    pending += [(analysis.host, CALLER, job["job_id"]) for job in generated.structured["jobs"]]
    for action in catalogue.actions:
        if not action.long_running or action.name == "analysis_generate":
            continue
        assert action.happy_path is not None, f"{action.name} starts jobs and declares no happy path"
        params = action.params.model_validate(dict(await action.happy_path(stage)))
        started = await action.handler(host, caller, params)
        pending += [(host, caller, job.job_id) for job in started.jobs]

    async def draft() -> _Drafted:
        return _Drafted(text="a rubric")

    acted = await start_action_job(host, CONFORMANCE_SCOPE, action="rubric_propose", run=draft, budget_s=60.0)
    pending += [(host, caller, job.job_id) for job in acted.jobs]

    statuses: dict[str, tuple[OpsHost, Caller, JobStatus]] = {}
    while pending:
        on, as_caller, job_id = pending.pop()
        status = await _settled(on, as_caller, job_id)
        statuses[job_id] = (on, as_caller, status)
        pending += [(on, as_caller, f"run:{run}") for run in status.run_ids if f"run:{run}" not in statuses]

    kinds = {status.kind for _, _, status in statuses.values()}
    assert kinds == set(get_args(JobKind)), f"no job here completes a {sorted(set(get_args(JobKind)) - kinds)}"
    for job_id, (on, as_caller, status) in statuses.items():
        assert status.state == "completed", (job_id, status.state, status.detail)
        outcome = await _poll(evals, on, as_caller, job_id)
        assert not outcome.is_error, (status.kind, outcome.text)
        assert f"job {job_id}: completed" in outcome.text
        if status.result is not None:
            assert "result" in outcome.text or status.kind in {"second_judge", "judge_temperature"}, outcome.text
