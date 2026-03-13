"""Background event loop for sync-to-async bridging.

Provides a singleton daemon thread running its own event loop. Sync code
(like __getitem__) submits async coroutines via run_coroutine_threadsafe
and blocks on the result. The ASGI event loop is never touched.

Pattern borrowed from fsspec (pandas/dask/xarray ecosystem).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


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
    """Submit an async coroutine to run on the background loop without blocking."""
    loop = _ensure_loop()
    asyncio.run_coroutine_threadsafe(coro, loop)


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
