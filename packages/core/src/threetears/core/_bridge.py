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
from typing import Any, Coroutine, TypeVar

__all__ = [
    "T",
    "drain",
    "fire_and_forget",
    "shutdown",
    "sync_await",
]

T = TypeVar("T")

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None

# strong references to tasks scheduled on a caller's running loop via
# ``create_task``. asyncio keeps only a weak reference to such tasks, so an
# unreferenced task can be garbage-collected mid-flight, silently dropping the
# coroutine. holding the task here until it completes prevents that.
_pending_tasks: set[asyncio.Task[Any]] = set()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Lazily start the background event loop on first use."""
    global _loop, _thread
    if _loop is not None and _loop.is_running():
        return _loop
    with _lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()
        _thread = threading.Thread(
            target=_loop.run_forever,
            daemon=True,
            name="threetears-async-bridge",
        )
        _thread.start()
        return _loop


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
    if _loop is None or not _loop.is_running():
        return

    async def _drain() -> None:
        # Get all tasks on this loop and wait for them
        tasks = [t for t in asyncio.all_tasks(_loop) if not t.done() and t is not asyncio.current_task()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    future = asyncio.run_coroutine_threadsafe(_drain(), _loop)
    future.result(timeout=10)


def shutdown() -> None:
    """Stop the background loop and join the thread. For clean teardown."""
    global _loop, _thread
    if _loop is not None and _loop.is_running():
        _loop.call_soon_threadsafe(_loop.stop)
    if _thread is not None:
        _thread.join(timeout=5)
        _thread = None
    if _loop is not None:
        _loop.close()
        _loop = None
