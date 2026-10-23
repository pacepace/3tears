"""Two processes over one store, a store clock a test sets, and a launched run held mid-flight: lease test support.

Shared by ``test_job_leases.py``, ``test_run_resume.py`` and ``test_sweep_resume.py``. Two "processes" are two launching hosts over one store,
each with its own job manager and its own leases, so its own holder id — which is what two processes sharing a
database are to the engine.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.evals.ops import (
    LaunchArguments,
    OpsHost,
    SweepArguments,
    SweepArm,
    job_poll,
    parse_job_id,
    run_job_id,
    run_launch,
)
from threetears.evals.schema.models import ArmSchedule
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.kind import ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT

__all__ = [
    "T0",
    "ClockedStore",
    "HeldGate",
    "held_extraction",
    "launched_arm",
    "launched_run",
    "other_process",
    "style_sweep",
    "until",
]

#: When the clocked store's clock starts.
T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


class ClockedStore(InMemoryDocumentStore):
    """An in-memory store keeping a clock of its own, which the test sets — a database server's clock, in effect."""

    def __init__(self, at: datetime = T0) -> None:
        """Start the clock at ``at``."""
        super().__init__()
        self.at = at

    def now(self) -> datetime:
        """The store's time, as the test set it."""
        return self.at


def other_process(host: OpsHost) -> OpsHost:
    """Another process over the same store: its own job manager and leases, so its own holder id."""
    return dataclasses.replace(host, launch=dataclasses.replace(host.launch))


async def until(condition: Any, *, timeout: float = 10.0) -> None:
    """Wait until ``condition()`` holds, failing after ``timeout`` seconds."""
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


class HeldGate(asyncio.Event):
    """The gate held extractions wait on, counting them: how many it held, and how many of those have since ended."""

    def __init__(self) -> None:
        """Nothing held yet."""
        super().__init__()
        self.held = 0
        self.released = 0


def held_extraction(monkeypatch: pytest.MonkeyPatch, *, passing: int = 0) -> HeldGate:
    """Hold every extraction after the first ``passing`` until the returned event is set, so a run stays running.

    :param monkeypatch: Patches the toy extractor's client.
    :ptype monkeypatch: pytest.MonkeyPatch
    :param passing: How many extractions go through at once before the rest wait.
    :ptype passing: int
    :return: The gate; setting it lets every held extraction, and every later one, through.
    :rtype: HeldGate
    """
    gate = HeldGate()
    extract = ScriptedExtractionClient.extract
    let_through = [passing]

    async def held(self: ScriptedExtractionClient, request: Any) -> Any:
        if let_through[0] > 0:
            let_through[0] -= 1
            return await extract(self, request)
        gate.held += 1
        try:
            await gate.wait()
            return await extract(self, request)
        finally:
            gate.released += 1

    monkeypatch.setattr(ScriptedExtractionClient, "extract", held)
    return gate


async def launched_run(host: OpsHost, k_runs: int = 1, *, max_cost_usd: float | None = None) -> str:
    """Launch one arm of ``k_runs`` repeats on ``host`` and wait until its run has started; the run's id.

    Started is read as no longer ``pending``: a run its cap stops at its first cell can be ``budget_stopped`` before
    any poll catches it ``running``.
    """
    _launch_id, run_id = await launched_arm(host, k_runs, max_cost_usd=max_cost_usd)
    return run_id


async def launched_arm(host: OpsHost, k_runs: int = 1, *, max_cost_usd: float | None = None) -> tuple[str, str]:
    """:func:`launched_run`, answering the launch's record id beside the run's.

    A test that lets the holder go on after its run is lost waits for the launch's task to end as well: until it
    ends, the launch's record is stored running, and a read once its lease has lapsed writes it lost too.
    """
    arguments = LaunchArguments(
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
        k_runs=k_runs,
        max_cost_usd=max_cost_usd,
    )
    (job,) = (await run_launch(host, arguments, TOYHOST_SCOPE)).jobs
    async with asyncio.timeout(10):
        while not (run_ids := (await job_poll(host, job.job_id, TOYHOST_SCOPE)).run_ids):
            await asyncio.sleep(0.01)
        while (await job_poll(host, run_job_id(run_ids[0]), TOYHOST_SCOPE)).status == "pending":
            await asyncio.sleep(0.01)
    _kind, launch_id, _attempt = parse_job_id(job.job_id)
    return launch_id, run_ids[0]


def style_sweep(styles: tuple[str, ...], *, arm_schedule: ArmSchedule | None, k_runs: int = 1) -> SweepArguments:
    """A sweep of one arm per prompt style, each on the first toy model and labelled by its style."""
    return SweepArguments(
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        campaign_name="prompt styles",
        campaign_behavior="extraction accuracy",
        k_runs=k_runs,
        arms=[SweepArm(model=RUN_MODELS[0], label=style, overlays={"prompt_style": style}) for style in styles],
        arm_schedule=arm_schedule,
    )
