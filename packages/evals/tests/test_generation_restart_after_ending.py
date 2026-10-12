"""A generation ``job_poll`` reads as done may be started again at once.

A generation's task writes its attempt's ending, then unwinds: it releases its client and its slot, and only then
stops holding the campaign's key in this process. A start made inside that window, by a caller who had just been
told the first generation completed, was refused as "already running". Under load the window is wide enough to
reach: the lease suite's restart-after-completion test failed that way on a busy machine.

- **A start inside the window waits for the unwinding task** and then starts, rather than refusing.
- **A task that never finishes unwinding is still refused as running**, once the wait gives up.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import threetears.evals.ops.analysis as analysis_ops
from threetears.evals.kernel import ConflictError
from threetears.evals.ops import analysis_generate, job_poll
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture, settled
from packages.evals.tests.off_loop import off_loop


def _unwinding_slowly(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Make every generation hold its task open after it has written its ending, until the returned event is set."""
    unwound = asyncio.Event()
    generate = analysis_ops.run_analysis_generation

    async def ended_then_held(*args: Any, **kwargs: Any) -> Any:
        ended = await generate(*args, **kwargs)
        await unwound.wait()
        return ended

    monkeypatch.setattr(analysis_ops, "run_analysis_generation", ended_then_held)
    return unwound


async def _done_but_held(fixture: Any, job_id: str) -> None:
    """Wait until ``job_id`` polls done while its task still holds the campaign's key."""
    async with asyncio.timeout(30):
        while not (await job_poll(fixture.host, job_id, TOYHOST_SCOPE)).done:
            await asyncio.sleep(0.01)
    assert fixture.host.launch.job_manager.is_task_active(job_id.rsplit(":", 1)[1]), "the task is still unwinding"


async def test_a_start_inside_the_unwinding_window_waits_and_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    unwound = _unwinding_slowly(monkeypatch)
    fixture = off_loop(ops_fixture)
    (first,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    await _done_but_held(fixture, first.job_id)

    second_start = asyncio.ensure_future(analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE))
    await asyncio.sleep(0.05)
    assert not second_start.done(), "the start waits for the unwinding task rather than refusing"
    unwound.set()
    (second,) = (await second_start).jobs

    assert second.job_id != first.job_id
    assert (await settled(fixture.host, second.job_id)).state == "completed"


async def test_a_task_that_never_finishes_unwinding_is_refused_as_running(monkeypatch: pytest.MonkeyPatch) -> None:
    unwound = _unwinding_slowly(monkeypatch)
    monkeypatch.setattr(analysis_ops, "ENDED_TASK_UNWIND_S", 0.05)
    fixture = off_loop(ops_fixture)
    (first,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    await _done_but_held(fixture, first.job_id)

    with pytest.raises(ConflictError, match="already running"):
        await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)
    unwound.set()
