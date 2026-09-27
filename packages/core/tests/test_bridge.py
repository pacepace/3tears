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


def test_threads_first_bridging_at_once_share_one_background_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """concurrent first callers of ``sync_await`` all run on ONE background loop.

    the lazy start checked ``is_running()``, which stays False between ``Thread.start()``
    and the new thread entering ``run_forever``. a caller that took the lock in that gap
    saw no running loop and started a second one, replacing the first -- and any
    loop-bound resource (a connection pool) made on one loop fails when used from the
    other. the loop here is slow to start, so the gap is certainly open.
    """
    import threading
    import time

    class _SlowStartingLoop(asyncio.SelectorEventLoop):
        """a loop whose thread takes a moment to begin running it."""

        def run_forever(self) -> None:
            time.sleep(0.05)
            super().run_forever()

    _bridge.shutdown()
    monkeypatch.setattr(asyncio, "new_event_loop", _SlowStartingLoop)
    threads_count = 16
    start = threading.Barrier(threads_count)
    loops: list[asyncio.AbstractEventLoop] = []
    record = threading.Lock()

    async def _which_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    def call() -> None:
        start.wait()
        loop = _bridge.sync_await(_which_loop())
        with record:
            loops.append(loop)

    try:
        workers = [threading.Thread(target=call) for _ in range(threads_count)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
        assert len(loops) == threads_count
        assert len({id(loop) for loop in loops}) == 1
        bridge_threads = [t for t in threading.enumerate() if t.name == "threetears-async-bridge"]
        assert len(bridge_threads) == 1, f"{len(bridge_threads)} background loops are running"
    finally:
        _bridge.shutdown()
