"""the thing that flushes a write-behind collection's buffer.

A write-behind collection puts its L3 write in a :class:`~threetears.core.collections.flush.WriteBuffer`
and something has to drain it. Nothing in 3tears did: ``flush_pending`` had no production caller
anywhere, so a write-behind counter would have sat unflushed until its consumer happened to call
one. Rather than adding a lifecycle hook to four consumer repos, the COLLECTION arms a flusher
from its own write paths -- ``CoordinationCollection.l2_cas_mutate`` and ``save_entity``, which
call ``ensure_flushing`` after every write -- and ``CollectionRegistry.close_collections()``
stops it.

**Not the primitive.** An earlier draft put the arming in each primitive, and a primitive that
forgot the call would buffer rows nothing ever flushed: silent durability loss, which is the one
thing write-behind exists to bound. Making it the collection's invariant means a wave-2 primitive
gets it without knowing it exists.

The interval is the whole exposure: a broker wipe loses at most the increments written since the
last flush, which is the trade counters accept and revocations do not.
"""

from __future__ import annotations

from typing import Final

from threetears.core.collections.flush import WriteBuffer, flush_pending
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.exceptions import GenerationUnavailableError
from threetears.observe import PeriodicTask, get_logger

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
        # flush on the interval, sleeping first; _flush_on_interval reports its own failure at ERROR.
        self._loop = PeriodicTask(
            self._flush_on_interval, interval=interval_seconds, name="coordination-flush", logger=log
        )

    @property
    def running(self) -> bool:
        """whether the flush loop is live.

        :return: ``True`` while the task exists and has not finished
        :rtype: bool
        """
        return self._loop.running

    def ensure_running(self) -> None:
        """start the flush loop if it is not already running.

        Called from the write path, so it must be cheap and must never raise on a repeat call.

        :return: nothing
        :rtype: None
        """
        self._loop.start()

    async def aclose(self) -> None:
        """stop the loop and flush what is still buffered.

        The final flush is the difference between a clean shutdown losing nothing and losing one
        interval, so it runs even though the loop is already cancelled.

        A switched-on table's write generation that could not be advanced for the final flush is
        raised here, after the loop has stopped: the rows were written, and whoever closes the
        flusher is the one caller left to hear that a follower cannot tell they changed.

        :return: nothing
        :rtype: None
        :raises GenerationUnavailableError: when the final flush wrote its rows and a switched-on
            table's write generation could not be advanced for them
        """
        await self._loop.stop()
        await self._flush_once()

    async def _flush_on_interval(self) -> None:
        """one interval's flush: every failure is reported, and none ends the loop.

        The loop has no caller to raise to, so a write generation that could not be advanced is
        reported where it happened, as a subscript write reports one.

        :return: nothing
        :rtype: None
        """
        await self._flush_once(raise_generation_failure=False)

    async def _flush_once(self, *, raise_generation_failure: bool = True) -> None:
        """flush the buffer, reporting a failure.

        A flush failure is an L3 outage, and the next interval retries it. Letting it out would
        end the loop and leave every later write unflushed with nothing saying so.

        A write generation that could not be advanced is not that. ``flush_pending`` raises it only
        after every row landed and was acknowledged, so nothing is retried for it: the rows are in
        L3 and the generation did not move, so a pod following the table cannot tell from it that
        they changed. It is logged as such, and raised when there is a caller to hear it.

        :param raise_generation_failure: whether a write generation that could not be advanced is
            raised after it is logged; ``False`` on the interval, which has no caller
        :ptype raise_generation_failure: bool
        :return: nothing
        :rtype: None
        :raises GenerationUnavailableError: when the rows were written and a switched-on table's
            write generation could not be advanced for them
        """
        try:
            flushed = await flush_pending(self._buffer, self._registry)
        except GenerationUnavailableError as exc:
            log.error(
                "coordination write-behind flush: the rows were written, but a table's write "
                "generation did not move; they are not retried, and a pod following the table "
                "cannot tell from it that they changed",
                extra={"extra_data": {"error": str(exc)}},
            )
            if raise_generation_failure:
                raise
            return
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- an L3 outage must not end the loop
            log.error(
                "coordination write-behind flush failed; retrying on the next interval",
                extra={"extra_data": {"error": f"{type(exc).__name__}: {exc}", "interval_seconds": self._interval}},
            )
            return
        if flushed:
            log.debug("coordination write-behind flush", extra={"extra_data": {"rows": flushed}})
