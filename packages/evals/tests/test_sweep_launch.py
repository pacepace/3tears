"""A sweep: arms that differ beyond the model, launched by one call, one after another, into one campaign (#632).

A toy host launches three arms differing in overlays with one call, watches them run one at a time, cancels
part-way, and finds the finished runs already members of the campaign.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from threetears.evals.actions import Caller
from threetears.evals.contracts import EvalStorage, ValidationFailedError
from threetears.evals.ops import (
    JobStatus,
    OpsHost,
    SweepArguments,
    SweepArm,
    job_cancel,
    job_poll,
    sweep_launch,
)
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.kind import TOY_SCRIPTS, ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template

STYLES = ("terse", "standard", "verbose")


def _host(*, latency_s: float) -> tuple[OpsHost, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    client = ScriptedExtractionClient(
        tuple(type(script)(**{**vars(script), "latency_s": latency_s}) for script in TOY_SCRIPTS)
    )
    launch, _client = toyhost_launch_host(storage=storage, client=client)
    return OpsHost(launch=launch), storage


def _sweep(**overrides: Any) -> SweepArguments:
    fields: dict[str, Any] = {
        "template_id": toyhost_template().id,
        "subject_id": TOYHOST_SUBJECT.subject_id,
        "campaign_name": "prompt styles",
        "campaign_behavior": "extraction accuracy",
        "k_runs": 1,
        "arms": [SweepArm(model=RUN_MODELS[0], label=style, overlays={"prompt_style": style}) for style in STYLES],
        **overrides,
    }
    return SweepArguments(**fields)


async def _watch(host: OpsHost, job_id: str, until: Any, executing: list[int]) -> JobStatus:
    """Poll a sweep until ``until(status)`` holds, sampling how many runs execute at once."""
    async with asyncio.timeout(20):
        while True:
            executing.append(host.launch.job_manager.executing_count)
            status = await job_poll(host, job_id, TOYHOST_SCOPE)
            if until(status):
                return status
            await asyncio.sleep(0.005)


async def test_a_three_arm_sweep_runs_its_arms_one_at_a_time_into_its_campaign() -> None:
    host, storage = _host(latency_s=0.02)

    started = await sweep_launch(host, _sweep(), TOYHOST_SCOPE, created_by="agent:test")
    (job,) = started.jobs
    assert job.kind == "sweep" and job.job_id.startswith("sweep:")
    executing: list[int] = []
    done = await _watch(host, job.job_id, lambda status: status.done, executing)

    assert (done.state, done.progress["arms_finished"]) == ("completed", 3)
    assert max(executing) == 1, "the arms ran one after another"
    campaign = storage.load_campaign(job.target_id, TOYHOST_SCOPE)
    assert campaign is not None and done.campaign_id == campaign.id
    runs = storage.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
    assert [run.overlays["prompt_style"] for run in runs] == list(STYLES), "each arm ran its own overlays"
    assert {run.status for run in runs} == {"completed"}


async def test_cancelling_part_way_stops_the_arm_in_flight_and_keeps_the_finished_ones_as_members() -> None:
    host, storage = _host(latency_s=0.05)
    (job,) = (await sweep_launch(host, _sweep(), TOYHOST_SCOPE, created_by="agent:test")).jobs
    executing: list[int] = []

    second_running = await _watch(
        host,
        job.job_id,
        lambda status: status.progress.get("arms", {}).get("standard") == "running",
        executing,
    )
    assert second_running.progress["arms"] == {"terse": "completed", "standard": "running", "verbose": "not_launched"}
    await job_cancel(host, job.job_id, TOYHOST_SCOPE)
    ended = await _watch(host, job.job_id, lambda status: status.done, executing)

    assert ended.state == "cancelled"
    assert ended.progress["arms"] == {"terse": "completed", "standard": "cancelled", "verbose": "not_launched"}
    campaign = storage.load_campaign(job.target_id, TOYHOST_SCOPE)
    assert campaign is not None
    members = storage.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
    assert [(run.overlays["prompt_style"], run.status) for run in members] == [
        ("terse", "completed"),
        ("standard", "cancelled"),
    ], "the finished arm and the one cut short are already members; the arm never launched is not"
    with pytest.raises(ValidationFailedError, match="nothing to cancel"):
        await job_cancel(host, job.job_id, TOYHOST_SCOPE)


async def test_raising_the_concurrency_runs_arms_side_by_side() -> None:
    host, _storage = _host(latency_s=0.05)
    (job,) = (await sweep_launch(host, _sweep(max_concurrent_arms=3), TOYHOST_SCOPE, created_by="agent:test")).jobs
    executing: list[int] = []
    done = await _watch(host, job.job_id, lambda status: status.done, executing)
    assert done.state == "completed"
    assert max(executing) > 1


async def test_an_arm_its_launch_would_refuse_refuses_the_sweep_before_anything_is_created() -> None:
    host, storage = _host(latency_s=0.0)
    arms = [
        SweepArm(model=RUN_MODELS[0], label="ok", overlays={"prompt_style": "terse"}),
        SweepArm(model=RUN_MODELS[0], label="bad", overlays={"prompt_style": "shouting"}),
    ]
    with pytest.raises(ValidationFailedError, match="arm 'bad' would be refused"):
        await sweep_launch(host, _sweep(arms=arms), TOYHOST_SCOPE, created_by="agent:test")
    assert storage.list_campaigns(TOYHOST_SCOPE) == [] and storage.query_eval_runs(TOYHOST_SCOPE) == []


def test_two_arms_that_are_one_condition_are_refused() -> None:
    same = SweepArm(model=RUN_MODELS[0], overlays={"prompt_style": "terse"})
    with pytest.raises(ValueError, match="one arm measured twice"):
        _sweep(arms=[same, same.model_copy(update={"label": "again"})])


async def test_the_action_launches_a_sweep_from_flat_parameters() -> None:
    from threetears.evals.actions import eval_catalogue

    host, _storage = _host(latency_s=0.0)
    action = eval_catalogue().get("sweep_launch")
    assert action is not None
    params = action.params.model_validate(
        {
            "template_id": toyhost_template().id,
            "subject_id": TOYHOST_SUBJECT.subject_id,
            "campaign_name": "styles",
            "campaign_behavior": "accuracy",
            "k_runs": 1,
            "arms": [{"model": RUN_MODELS[0], "overlays": {"prompt_style": style}} for style in STYLES[:2]],
        }
    )
    started = await action.handler(host, Caller(scope_id=TOYHOST_SCOPE, identity="agent:test"), params)
    (job,) = started.jobs
    done = await _watch(host, job.job_id, lambda status: status.done, [])
    assert done.state == "completed" and done.progress["arms_total"] == 2


async def test_an_arm_whose_run_fails_ends_the_sweep_launching_nothing_after_it() -> None:
    host, storage = _host(latency_s=0.0)
    arms = [
        # No script covers this model, so every cell of its run raises and the run ends failed: a harness fault.
        SweepArm(model="extractor-unscripted", label="broken"),
        SweepArm(model=RUN_MODELS[0], label="fine"),
    ]
    # A cap the sweep names, so the unpriceable arm is admitted under it rather than refused up front.
    (job,) = (await sweep_launch(host, _sweep(arms=arms, max_cost_usd=1.0), TOYHOST_SCOPE, created_by="t")).jobs
    done = await _watch(host, job.job_id, lambda status: status.done, [])

    assert done.state == "failed"
    assert done.detail is not None and "arm 'broken'" in done.detail and "ended failed" in done.detail
    assert done.progress["arms"] == {"broken": "failed", "fine": "not_launched"}
    campaign = storage.load_campaign(job.target_id, TOYHOST_SCOPE)
    assert campaign is not None and len(campaign.run_ids) == 1
