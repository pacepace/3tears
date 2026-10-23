"""Two processes over one store, a store clock a test sets, and a launched run held mid-flight: lease test support.

Shared by ``test_job_leases.py`` and ``test_run_resume.py``. Two "processes" are two launching hosts over one store,
each with its own job manager and its own leases, so its own holder id — which is what two processes sharing a
database are to the engine.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.evals.ops import LaunchArguments, OpsHost, job_poll, run_job_id, run_launch
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.kind import ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT

__all__ = ["T0", "ClockedStore", "held_extraction", "launched_run", "other_process", "until"]

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


def held_extraction(monkeypatch: pytest.MonkeyPatch, *, passing: int = 0) -> asyncio.Event:
    """Hold every extraction after the first ``passing`` until the returned event is set, so a run stays running.

    :param monkeypatch: Patches the toy extractor's client.
    :ptype monkeypatch: pytest.MonkeyPatch
    :param passing: How many extractions go through at once before the rest wait.
    :ptype passing: int
    :return: The gate; setting it lets every held extraction, and every later one, through.
    :rtype: asyncio.Event
    """
    gate = asyncio.Event()
    extract = ScriptedExtractionClient.extract
    let_through = [passing]

    async def held(self: ScriptedExtractionClient, request: Any) -> Any:
        if let_through[0] > 0:
            let_through[0] -= 1
        else:
            await gate.wait()
        return await extract(self, request)

    monkeypatch.setattr(ScriptedExtractionClient, "extract", held)
    return gate


async def launched_run(host: OpsHost, k_runs: int = 1, *, max_cost_usd: float | None = None) -> str:
    """Launch one arm of ``k_runs`` repeats on ``host`` and wait until its run reads running; the run's id."""
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
        while (await job_poll(host, run_job_id(run_ids[0]), TOYHOST_SCOPE)).status != "running":
            await asyncio.sleep(0.01)
    return run_ids[0]
