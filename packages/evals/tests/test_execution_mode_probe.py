"""The ``execution_mode`` covariate counts runs that execute, never runs queued for a slot.

A launch reads :attr:`EvalJobManager.executing_count` as its ``concurrent_eval_jobs_probe``, and
:func:`derive_covariates` stamps ``concurrent`` whenever that reads above 1. The probe once read
``active_count``, which counts every task not yet done: under a cap of one, a run executing alone
with others queued behind it read 3, so a deliberately serial baseline was stamped ``concurrent``.
These drive the manager with real queueing and read the covariate the way the runner does, from
inside the executing work.
"""

from __future__ import annotations

import asyncio
from typing import Any

from threetears.evals.kernel.covariates import derive_covariates
from threetears.evals.run.jobs import EvalJobManager
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.job_support import InMemoryRunStore, settled


def _mode(manager: EvalJobManager) -> str | float:
    """The ``execution_mode`` a cell sampling the manager now would record."""
    return derive_covariates(usage=[], concurrent_eval_jobs=manager.executing_count)["execution_mode"]


async def test_a_run_executing_alone_under_a_cap_of_one_is_serial_though_others_queue() -> None:
    manager = EvalJobManager(InMemoryRunStore(), max_concurrent=1)
    runs = [make_eval_run() for _ in range(3)]
    queued = asyncio.Event()
    modes: list[str | float] = []
    queue_depths: list[int] = []

    async def work(progress: Any) -> None:
        # The first run samples only once the other two are queued behind it, which is the case at issue.
        await queued.wait()
        queue_depths.append(manager.active_count)
        modes.append(_mode(manager))

    for run in runs:
        await manager.start_group([(run, work, None)])
    queued.set()
    for run in runs:
        await settled(manager, run.id)

    assert queue_depths[0] == 3, "the arm under test needs two runs queued behind the first"
    assert modes == ["serial", "serial", "serial"], "a run that executed alone was stamped concurrent"
    assert manager.executing_count == 0
    await manager.shutdown()


async def test_two_runs_holding_slots_at_the_same_time_are_concurrent() -> None:
    manager = EvalJobManager(InMemoryRunStore(), max_concurrent=2)
    runs = [make_eval_run() for _ in range(2)]
    started = [asyncio.Event() for _ in runs]
    modes: dict[str, str | float] = {}

    def work_for(index: int) -> Any:
        async def work(progress: Any) -> None:
            started[index].set()
            await asyncio.gather(*(event.wait() for event in started))
            modes[runs[index].id] = _mode(manager)

        return work

    for index, run in enumerate(runs):
        await manager.start_group([(run, work_for(index), None)])
    for run in runs:
        await settled(manager, run.id)

    assert modes == {run.id: "concurrent" for run in runs}
    await manager.shutdown()


async def test_the_arms_of_one_launch_execute_side_by_side_in_their_one_slot() -> None:
    """A group takes one slot but its members run at once, so they contend with each other."""
    manager = EvalJobManager(InMemoryRunStore(), max_concurrent=1)
    runs = [make_eval_run() for _ in range(2)]
    started = [asyncio.Event() for _ in runs]
    modes: list[str | float] = []

    def work_for(index: int) -> Any:
        async def work(progress: Any) -> None:
            started[index].set()
            await asyncio.gather(*(event.wait() for event in started))
            modes.append(_mode(manager))

        return work

    await manager.start_group([(run, work_for(index), None) for index, run in enumerate(runs)])
    for run in runs:
        await settled(manager, run.id)

    assert modes == ["concurrent", "concurrent"]
    await manager.shutdown()


async def test_a_run_cancelled_while_queued_never_counted_as_executing() -> None:
    manager = EvalJobManager(InMemoryRunStore(), max_concurrent=1)
    first, waiting = make_eval_run(), make_eval_run()
    release = asyncio.Event()
    readings: list[int] = []

    async def first_work(progress: Any) -> None:
        await release.wait()
        readings.append(manager.executing_count)

    async def never_runs(progress: Any) -> None:
        raise AssertionError("a run cancelled while queued must not reach its work")

    await manager.start_group([(first, first_work, None)])
    await manager.start_group([(waiting, never_runs, None)])
    await asyncio.sleep(0)
    assert manager.cancel_job(waiting.id, reason="not needed")
    await settled(manager, waiting.id)
    release.set()
    await settled(manager, first.id)

    assert readings == [1]
    assert manager.executing_count == 0
    await manager.shutdown()
