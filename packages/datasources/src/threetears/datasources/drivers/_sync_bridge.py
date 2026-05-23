"""async-to-sync executor bridge for drivers backed by sync backend libraries.

three of the planned 3tears drivers ship a sync backend library:

- :class:`RedshiftDriver` -- ``redshift_connector`` is blocking
- :class:`SnowflakeDriver` -- ``snowflake.connector`` is blocking
- :class:`BigQueryDriver` -- ``google.cloud.bigquery.Client`` is blocking

each of them needs the same shape: a bounded
:class:`concurrent.futures.ThreadPoolExecutor`, a method that submits a
sync callable + wraps the resulting future as an awaitable, and
cancellation that fires the backend's sync cancel hook on
:class:`asyncio.CancelledError`. :class:`AsyncSyncBridge` is that
shape, factored out so the three drivers reuse one tested
implementation instead of three drifting copies (reuse-review I5).

design points:

- bounded executor sized by the driver's ConnectionConfig
  (``executor_max_workers``); never default in the driver code -- the
  ``test_no_hardcoded_pool_params`` enforcement test catches drift.
- ``close()`` uses ``shutdown(wait=False)``. ``wait=True`` deadlocks
  the asyncio event loop because the executor's worker threads may be
  awaiting a coroutine that can't run while the loop is blocked. the
  threads drain naturally once their current call returns.
- cancellation: ``to_thread_with_cancel`` fires ``cancel_cb`` on
  :class:`asyncio.CancelledError`. ``cancel_cb`` MUST be safe to call
  from the event loop (i.e. fast / non-blocking); the canonical use
  is the backend lib's "abort the in-flight statement" hook (e.g.
  ``redshift_connector.Connection.cancel``).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from threetears.observe import get_logger

__all__ = ["AsyncSyncBridge"]

log = get_logger(__name__)

T = TypeVar("T")


class AsyncSyncBridge:
    """bounded executor + cancel-aware ``to_thread`` for sync-backed drivers.

    one bridge per driver instance; concrete drivers construct the
    bridge in their ``__init__`` from the relevant ConnectionConfig
    field and close it from their ``close()``.

    example (shape concrete drivers follow)::

        class RedshiftDriver(Driver):
            def __init__(self, config: RedshiftConnectionConfig) -> None:
                self._bridge = AsyncSyncBridge(
                    max_workers=config.executor_max_workers,
                    name="rs-bridge",
                )

            async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
                conn = await self._acquire_connection()
                return await self._bridge.to_thread_with_cancel(
                    lambda: _sync_fetch(conn, sql, params),
                    cancel_cb=conn.cancel,
                )

            async def close(self) -> None:
                await self._bridge.close()

    :param max_workers: bounded executor size. drivers MUST source
        this from their ConnectionConfig (the enforcement test catches
        inlined literals)
    :ptype max_workers: int
    :param name: thread-name prefix; surfaces in stack traces and
        process listings. drivers pass a short identifier
        (``"rs-bridge"`` / ``"sf-bridge"`` / ``"bq-bridge"``)
    :ptype name: str
    """

    def __init__(self, *, max_workers: int, name: str) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=name,
        )
        self._name = name
        self._max_workers = max_workers
        self._closed = False

    @property
    def max_workers(self) -> int:
        """configured executor ceiling.

        :return: max worker count set at construction
        :rtype: int
        """
        return self._max_workers

    @property
    def name(self) -> str:
        """thread-name prefix.

        :return: prefix string used for worker thread names
        :rtype: str
        """
        return self._name

    async def to_thread_with_cancel(
        self,
        fn: Callable[[], T],
        *,
        cancel_cb: Callable[[], Any],
    ) -> T:
        """submit ``fn`` to the executor; await result; cancel on cancellation.

        on :class:`asyncio.CancelledError` the bridge calls
        ``cancel_cb`` (the backend's "abort the in-flight statement"
        hook) BEFORE re-raising the cancellation. callbacks may be
        sync or async; sync callables whose return is a coroutine are
        awaited.

        the bridge does NOT wait for the underlying executor future
        to finish after cancellation -- the worker thread continues
        until its current syscall returns, but the asyncio caller is
        released immediately. the ``cancel_cb`` is the only signal
        that prompts the worker to bail out promptly.

        :param fn: zero-arg sync callable to run in the executor
        :ptype fn: Callable[[], T]
        :param cancel_cb: backend cancel hook to invoke on
            :class:`asyncio.CancelledError`. may be sync or async;
            sync callables returning a coroutine are awaited
        :ptype cancel_cb: Callable[[], Any]
        :return: the callable's result
        :rtype: T
        :raises RuntimeError: if the bridge was previously closed
        :raises asyncio.CancelledError: re-raised after best-effort
            backend cancellation
        """
        if self._closed:
            raise RuntimeError(f"AsyncSyncBridge[{self._name}] is closed; cannot submit new work")
        loop = asyncio.get_running_loop()
        future = self._executor.submit(fn)
        wrapped = asyncio.wrap_future(future, loop=loop)
        try:
            result = await wrapped
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                cancel_result = cancel_cb()
                if asyncio.iscoroutine(cancel_result):
                    await cancel_result
            raise
        return result

    async def close(self) -> None:
        """release the executor.

        uses ``shutdown(wait=False)`` per DS-09-12. workers in-flight
        at close time continue until their current call returns;
        callers waiting on a ``to_thread_with_cancel`` result will
        see their wrapped future complete or cancel naturally.

        idempotent: calling ``close()`` twice is a no-op on the
        second call.

        :return: nothing
        :rtype: None
        """
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=False)
