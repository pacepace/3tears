"""Tests for the sync-to-async bridge, focused on task lifecycle ownership."""

from __future__ import annotations

import asyncio
import gc
import weakref

import pytest

from threetears.core import fire_and_forget
from threetears.core.testing import drain_and_shutdown_bridge


@pytest.mark.asyncio
async def test_fire_and_forget_survives_gc_on_running_loop() -> None:
    """fire_and_forget task must not be dropped by the garbage collector.

    asyncio keeps only a weak reference to a task returned by ``create_task``.
    Without a strong reference held elsewhere, a ``gc.collect()`` before the
    task gets a chance to run can finalize it, silently dropping the coroutine.
    The bridge must hold a strong reference until the task completes -- and
    only until then, or every scheduled task leaks.

    The task waits on a future nothing else references, so the event loop holds
    no strong reference to it either: if the bridge did not hold one, the
    collection below would reclaim it.
    """
    tasks: list[weakref.ref[asyncio.Task[object]]] = []

    async def _work() -> None:
        task = asyncio.current_task()
        assert task is not None
        tasks.append(weakref.ref(task))
        await asyncio.get_running_loop().create_future()

    fire_and_forget(_work())
    await asyncio.sleep(0)  # the task starts and suspends on its unreferenced future

    gc.collect()
    pending = tasks[0]()
    assert pending is not None, "the pending task was garbage-collected: nothing held it"
    assert not pending.done()

    pending.cancel()
    await asyncio.wait({pending})
    del pending
    await asyncio.sleep(0)  # the done callback runs

    gc.collect()
    assert tasks[0]() is None, "the finished task is still held: every scheduled task would leak"


@pytest.mark.asyncio
async def test_fire_and_forget_propagates_side_effect() -> None:
    """The scheduled coroutine actually runs and mutates observable state."""
    box: list[int] = []

    async def _work() -> None:
        box.append(1)

    fire_and_forget(_work())

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

    fire_and_forget(_work())
    drain_and_shutdown_bridge()

    assert box == [7]


def test_threads_first_bridging_at_once_share_one_background_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """concurrent first callers of the bridge, from threads with no running loop, share ONE background loop.

    the lazy start checked ``is_running()``, which stays False between ``Thread.start()``
    and the new thread entering ``run_forever``. a caller that took the lock in that gap
    saw no running loop and started a second one, replacing the first -- and any
    loop-bound resource (a connection pool) made on one loop fails when used from the
    other. the loop here is slow to start, so the gap is certainly open.

    driven through ``fire_and_forget``, the bridge's public entry: from a thread with no
    running loop it starts (or reuses) the background loop exactly as the synchronous
    collection accessors do.
    """
    import threading
    import time

    class _SlowStartingLoop(asyncio.SelectorEventLoop):
        """a loop whose thread takes a moment to begin running it."""

        def run_forever(self) -> None:
            time.sleep(0.05)
            super().run_forever()

    drain_and_shutdown_bridge()
    monkeypatch.setattr(asyncio, "new_event_loop", _SlowStartingLoop)
    threads_count = 16
    start = threading.Barrier(threads_count)
    loops: list[asyncio.AbstractEventLoop] = []
    record = threading.Lock()

    async def _record_loop() -> None:
        with record:
            loops.append(asyncio.get_running_loop())

    def call() -> None:
        start.wait()
        fire_and_forget(_record_loop())

    try:
        workers = [threading.Thread(target=call) for _ in range(threads_count)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
        deadline = time.monotonic() + 30
        while len(loops) < threads_count and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(loops) == threads_count
        assert len({id(loop) for loop in loops}) == 1
        bridge_threads = [t for t in threading.enumerate() if t.name == "threetears-async-bridge"]
        assert len(bridge_threads) == 1, f"{len(bridge_threads)} background loops are running"
    finally:
        drain_and_shutdown_bridge()
