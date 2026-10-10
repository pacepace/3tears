"""Test support for ``EvalJobManager`` suites: an in-memory run store and job-control helpers.

Everything here reaches the manager through its front door. A job's task is found the way any
asyncio caller can find one -- it is the task ``start_group`` adds to the loop -- and a job's
end is observed through ``is_active``, never through the manager's own bookkeeping.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from threetears.evals.schema.models import EvalRun
from threetears.evals.kernel.storage import JobStore
from threetears.evals.run.jobs import EvalJobManager, WorkFn

__all__ = ["InMemoryRunStore", "blocked_work", "cancelled", "settled", "start_tracked"]


class InMemoryRunStore(JobStore):
    """In-memory storage covering the EvalRun surface the job manager touches."""

    def __init__(self) -> None:
        self.runs: dict[str, EvalRun] = {}

    def save_eval_run(self, run: EvalRun, *, if_match: str | None = None) -> None:
        self.runs[run.id] = run

    def load_eval_run_with_etag(self, run_id: str, scope_id: str) -> tuple[EvalRun | None, str | None]:
        run = self.runs.get(run_id)
        return (run, "etag-1") if run is not None else (None, None)


async def cancelled(task: asyncio.Task[Any]) -> None:
    """Await a task, swallowing only its expected CancelledError."""
    with pytest.raises(asyncio.CancelledError):
        await task


def blocked_work() -> tuple[Any, asyncio.Event]:
    """A work fn that parks forever plus the event proving it started."""
    started = asyncio.Event()

    async def work(progress: Any) -> None:
        started.set()
        await asyncio.Event().wait()

    return work, started


async def start_tracked(
    manager: EvalJobManager, members: Sequence[tuple[EvalRun, WorkFn, float | None]]
) -> set[asyncio.Task[Any]]:
    """Start a launch group and return the job tasks it added to the loop.

    ``start_group`` finishes its saves before it creates a task and creates exactly one task per
    member, so the tasks running after it returns that were not running before are the group's.

    Args:
        manager: The manager to start the group on.
        members: ``(run, work, job_timeout_s)`` per run, as ``start_group`` takes them.

    Returns:
        The group's job tasks.
    """
    before = asyncio.all_tasks()
    await manager.start_group(members)
    tasks = asyncio.all_tasks() - before
    assert len(tasks) == len(members), f"expected one task per member, found {len(tasks)} for {len(members)}"
    return tasks


async def settled(manager: EvalJobManager, job_id: str, *, timeout: float = 5.0) -> None:
    """Wait until the manager no longer reports ``job_id`` active, failing after ``timeout`` seconds.

    Args:
        manager: The manager running the job.
        job_id: The job's run id.
        timeout: Seconds to wait before failing.
    """
    async with asyncio.timeout(timeout):
        while manager.is_active(job_id):
            await asyncio.sleep(0.001)
