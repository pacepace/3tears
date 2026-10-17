"""``threetears.core.sync.run``: one async call, run to the end from synchronous code.

A CLI or batch script has no event loop; these tests call ``run`` the way it does, from a plain
thread, and pin what it promises: the result or the work's own exception comes back, one shared
loop serves every call (so what one call opens the next can use), a timeout cancels the work, and a
call from inside a running loop is refused with a message saying what to do instead.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

import pytest

from threetears.core import sync
from threetears.core.testing import drain_and_shutdown_bridge


@pytest.fixture(autouse=True)
def _fresh_bridge() -> Iterator[None]:
    """stop the shared loop after each test, so none inherits another's queued work."""
    yield
    drain_and_shutdown_bridge()


def test_a_coroutine_returns_its_result() -> None:
    async def answer() -> int:
        await asyncio.sleep(0)
        return 42

    assert sync.run(answer()) == 42


def test_a_factory_is_called_on_the_shared_loop() -> None:
    seen: list[asyncio.AbstractEventLoop] = []

    async def where() -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        seen.append(loop)
        return loop

    first = sync.run(where)
    second = sync.run(lambda: where())
    assert first is second, "every call runs on one loop, so a resource opened by one serves the next"
    assert first.is_running()


def test_a_loop_bound_resource_opened_by_one_call_serves_the_next() -> None:
    async def make_lock() -> asyncio.Lock:
        lock = asyncio.Lock()
        async with lock:  # binds the lock to this loop
            pass
        return lock

    lock = sync.run(make_lock)

    async def use(held: asyncio.Lock) -> bool:
        async with held:
            return True

    assert sync.run(lambda: use(lock))


def test_the_works_own_exception_is_raised_unchanged() -> None:
    class Boom(Exception):
        pass

    async def fail() -> None:
        raise Boom("no")

    with pytest.raises(Boom, match="no"):
        sync.run(fail)


def test_a_timeout_inside_the_work_is_not_mistaken_for_runs_own() -> None:
    async def times_out() -> None:
        raise TimeoutError("the upstream's")

    with pytest.raises(TimeoutError, match="the upstream's"):
        sync.run(times_out, timeout=5)


def test_a_timeout_cancels_the_work() -> None:
    cancelled = threading.Event()

    async def forever() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(TimeoutError, match="did not finish within 0.05s"):
        sync.run(forever, timeout=0.05)
    assert cancelled.wait(timeout=5), "the timed-out work was left running on the shared loop"


@pytest.mark.parametrize("timeout", [0, -1])
def test_a_timeout_that_is_not_positive_is_refused(timeout: float) -> None:
    async def nothing() -> None:
        return None

    with pytest.raises(ValueError, match="timeout must be positive"):
        sync.run(nothing(), timeout=timeout)


def test_concurrent_callers_from_several_threads_each_get_their_own_result() -> None:
    results: dict[int, int] = {}

    def call(n: int) -> None:
        async def double() -> int:
            await asyncio.sleep(0.01)
            return n * 2

        results[n] = sync.run(double)

    threads = [threading.Thread(target=call, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert results == {n: n * 2 for n in range(8)}


async def test_inside_a_running_loop_it_is_refused_with_what_to_do_instead() -> None:
    async def nothing() -> None:
        return None

    coro = nothing()
    with pytest.raises(RuntimeError, match="await the work instead"):
        sync.run(coro)
    # the refused coroutine was closed, never left to warn that it was not awaited
    assert coro.cr_frame is None


async def test_inside_a_running_loop_a_factory_is_never_called() -> None:
    called: list[bool] = []

    async def nothing() -> None:
        called.append(True)

    with pytest.raises(RuntimeError, match="running event loop"):
        sync.run(nothing)
    assert not called
