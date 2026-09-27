"""Background event loop for sync-to-async bridging.

Provides a singleton daemon thread running its own event loop. Sync code
(like __getitem__) submits async coroutines via run_coroutine_threadsafe
and blocks on the result.

``fire_and_forget`` is loop-aware: when called from a thread that already
has a running event loop (e.g., an ASGI handler), it schedules the task on
that loop via ``create_task`` so that async resources (e.g., asyncpg
connection pools) stay on the correct loop. When called from pure sync code
with no running loop, it falls back to the background loop.

Pattern borrowed from fsspec (pandas/dask/xarray ecosystem).
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Coroutine, TypeVar

from threetears.core.config import DEFAULT_BRIDGE_LOOP_START_TIMEOUT_SECONDS
from threetears.observe import BuildOnce

__all__ = [
    "T",
    "drain",
    "fire_and_forget",
    "shutdown",
    "sync_await",
]

T = TypeVar("T")


@dataclass(frozen=True)
class _BridgeLoop:
    """the background loop and the daemon thread running it.

    :ivar loop: the loop, running once this is stored
    :ivar thread: the thread whose ``run_forever`` it is
    """

    loop: asyncio.AbstractEventLoop
    thread: threading.Thread


#: the one background loop, built on first use and rebuilt if it stopped. several threads reach
#: the bridge at once, so it is built through ``BuildOnce``: a second loop would strand the work
#: already queued on the first, and a loop-bound resource made on one would later be used from
#: the other.
_background: BuildOnce[str, _BridgeLoop] = BuildOnce(is_current=lambda bridge: bridge.loop.is_running())

#: the one key :data:`_background` holds.
_ONLY = "only"

# strong references to tasks scheduled on a caller's running loop via
# ``create_task``. asyncio keeps only a weak reference to such tasks, so an
# unreferenced task can be garbage-collected mid-flight, silently dropping the
# coroutine. holding the task here until it completes prevents that.
_pending_tasks: set[asyncio.Task[Any]] = set()


def _start_loop() -> _BridgeLoop:
    """start a background loop on a daemon thread, returning only once it is RUNNING.

    ``is_running()`` is what :data:`_background` judges a stored loop by, and it stays False between
    ``Thread.start()`` and the new thread entering ``run_forever``. a loop published in that gap
    would read as stale to the next caller, which would start a second one and replace the first
    while work was already queued on it -- so a loop-bound resource made on one loop was later
    used from the other.

    :return: the running loop and its thread
    :rtype: _BridgeLoop
    :raises RuntimeError: if the loop does not start in time
    """
    loop = asyncio.new_event_loop()
    running = threading.Event()
    # the first callback a loop runs, so it fires once run_forever has begun
    loop.call_soon(running.set)
    thread = threading.Thread(
        target=loop.run_forever,
        daemon=True,
        name="threetears-async-bridge",
    )
    thread.start()
    # bounded, so a loop that never starts is an error naming itself rather than a caller
    # hung here forever holding the lock every later caller waits on
    if not running.wait(timeout=DEFAULT_BRIDGE_LOOP_START_TIMEOUT_SECONDS):
        raise RuntimeError(
            f"the threetears async bridge loop did not start within {DEFAULT_BRIDGE_LOOP_START_TIMEOUT_SECONDS}s"
        )
    return _BridgeLoop(loop=loop, thread=thread)


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """the running background event loop, started on first use.

    :return: the running background loop
    :rtype: asyncio.AbstractEventLoop
    """
    return _background.get(_ONLY, _start_loop).loop


def sync_await(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from sync code, blocking until complete."""
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def fire_and_forget(coro: Coroutine[Any, Any, Any]) -> None:
    """Submit an async coroutine without blocking.

    When called from a thread with a running event loop, schedules the task
    on that loop (``create_task``). When called from pure sync code, uses
    the background loop (``run_coroutine_threadsafe``).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        bg_loop = _ensure_loop()
        asyncio.run_coroutine_threadsafe(coro, bg_loop)
    else:
        task = loop.create_task(coro)
        _pending_tasks.add(task)
        task.add_done_callback(_pending_tasks.discard)


def drain() -> None:
    """Wait for all pending tasks on the background loop to complete."""
    bridge = _background.peek(_ONLY)
    if bridge is None or not bridge.loop.is_running():
        return
    loop = bridge.loop

    async def _drain() -> None:
        # Get all tasks on this loop and wait for them
        tasks = [t for t in asyncio.all_tasks(loop) if not t.done() and t is not asyncio.current_task()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    future = asyncio.run_coroutine_threadsafe(_drain(), loop)
    future.result(timeout=10)


def shutdown() -> None:
    """Stop the background loop and join the thread. For clean teardown."""
    bridge = _background.pop(_ONLY)
    if bridge is None:
        return
    if bridge.loop.is_running():
        bridge.loop.call_soon_threadsafe(bridge.loop.stop)
    bridge.thread.join(timeout=5)
    bridge.loop.close()
