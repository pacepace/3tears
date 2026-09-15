"""the thing that flushes a write-behind collection's buffer.

A write-behind collection puts its L3 write in a :class:`~threetears.core.collections.flush.WriteBuffer`
and something has to drain it. Nothing in 3tears did: ``flush_pending`` had no production caller
anywhere, so a write-behind counter would have sat unflushed until its consumer happened to call
one. Rather than adding a lifecycle hook to four consumer repos, the primitive that declares
write-behind owns a flusher and starts it on its first write.

The interval is the whole exposure: a broker wipe loses at most the increments written since the
last flush, which is the trade counters accept and revocations do not.
"""

from __future__ import annotations

import asyncio
from typing import Final

from threetears.core.collections.flush import WriteBuffer, flush_pending
from threetears.core.collections.registry import CollectionRegistry
from threetears.observe import get_logger

__all__ = ["PeriodicFlusher"]

log = get_logger(__name__)

#: default seconds between flushes. One wipe inside this window costs a counter its increments,
#: so it is short enough to bound that and long enough to keep the batching worth having.
DEFAULT_FLUSH_INTERVAL_SECONDS: Final = 5.0


class PeriodicFlusher:
    """drains one write buffer on an interval until stopped.

    Idempotent to start: several primitives sharing a buffer share its flusher, and each calls
    :meth:`ensure_running` on every write.
    """

    def __init__(
        self,
        buffer: WriteBuffer,
        registry: CollectionRegistry,
        *,
        interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
    ) -> None:
        """configure the flusher; start nothing yet.

        :param buffer: the write buffer to drain
        :ptype buffer: WriteBuffer
        :param registry: the registry whose collections persist the buffered rows
        :ptype registry: CollectionRegistry
        :param interval_seconds: seconds between flushes; must be positive
        :ptype interval_seconds: float
        :return: nothing
        :rtype: None
        :raises ValueError: when the interval is not positive
        """
        if interval_seconds <= 0:
            raise ValueError(f"PeriodicFlusher interval_seconds must be positive, got {interval_seconds}")
        self._buffer = buffer
        self._registry = registry
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        """whether the flush loop is live.

        :return: ``True`` while the task exists and has not finished
        :rtype: bool
        """
        return self._task is not None and not self._task.done()

    def ensure_running(self) -> None:
        """start the flush loop if it is not already running.

        Called from the write path, so it must be cheap and must never raise on a repeat call.

        :return: nothing
        :rtype: None
        """
        if self.running:
            return
        self._task = asyncio.get_running_loop().create_task(self._run(), name="coordination-flush")

    async def aclose(self) -> None:
        """stop the loop and flush what is still buffered.

        The final flush is the difference between a clean shutdown losing nothing and losing one
        interval, so it runs even though the loop is already cancelled.

        :return: nothing
        :rtype: None
        """
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._flush_once()

    async def _run(self) -> None:
        """flush on the interval until cancelled.

        :return: nothing
        :rtype: None
        """
        while True:
            await asyncio.sleep(self._interval)
            await self._flush_once()

    async def _flush_once(self) -> None:
        """flush the buffer, reporting a failure without killing the loop.

        A flush failure is an L3 outage, and the next interval retries it. Letting it out would
        end the loop and leave every later write unflushed with nothing saying so.

        :return: nothing
        :rtype: None
        """
        try:
            flushed = await flush_pending(self._buffer, self._registry)
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- an L3 outage must not end the loop
            log.error(
                "coordination write-behind flush failed; retrying on the next interval",
                extra={"extra_data": {"error": f"{type(exc).__name__}: {exc}", "interval_seconds": self._interval}},
            )
            return
        if flushed:
            log.debug("coordination write-behind flush", extra={"extra_data": {"rows": flushed}})
