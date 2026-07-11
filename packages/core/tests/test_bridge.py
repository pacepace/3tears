"""Tests for the sync-to-async bridge, focused on task lifecycle ownership."""

from __future__ import annotations

import asyncio
import gc

import pytest

from threetears.core import _bridge


@pytest.mark.asyncio
async def test_fire_and_forget_survives_gc_on_running_loop() -> None:
    """fire_and_forget task must not be dropped by the garbage collector.

    asyncio keeps only a weak reference to a task returned by ``create_task``.
    Without a strong reference held elsewhere, a ``gc.collect()`` before the
    task gets a chance to run can finalize it, silently dropping the coroutine.
    The bridge must hold a strong reference until the task completes.
    """
    completed = asyncio.Event()

    async def _work() -> None:
        # yield control so the task is still pending when gc runs below
        await asyncio.sleep(0.05)
        completed.set()

    _bridge.fire_and_forget(_work())

    # the task must be tracked while pending
    assert len(_bridge._pending_tasks) == 1

    # force a collection cycle while the task is still pending; a weakly-held
    # task would be eligible for finalization here
    gc.collect()

    await asyncio.wait_for(completed.wait(), timeout=1.0)

    # done callback must clear the strong reference to avoid leaking tasks
    await asyncio.sleep(0)
    assert len(_bridge._pending_tasks) == 0


@pytest.mark.asyncio
async def test_fire_and_forget_propagates_side_effect() -> None:
    """The scheduled coroutine actually runs and mutates observable state."""
    box: list[int] = []

    async def _work() -> None:
        box.append(1)

    _bridge.fire_and_forget(_work())

    for _ in range(100):
        if box:
            break
        await asyncio.sleep(0.01)

    assert box == [1]


def test_fire_and_forget_without_running_loop_uses_background() -> None:
    """From pure sync code (no running loop) the background loop runs the coro."""
    box: list[int] = []

    async def _work() -> None:
        box.append(7)

    _bridge.fire_and_forget(_work())
    _bridge.drain()

    assert box == [7]
