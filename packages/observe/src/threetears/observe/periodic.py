"""one start/stop/interval loop for every periodic background job.

a sweeper, a health check, a catch-up tick and a write-buffer flush all
need the same shell: start a task, run a coroutine every N seconds,
survive a bad tick, stop cleanly. 3tears carried that shell eight times
and its consumers carry more, and the copies drifted -- some sleep
before the first tick and some after, some ``stop`` without waiting for
the task to end, one bad tick kills some loops and not others.

``PeriodicTask`` is that shell, once:

- ``start()`` is idempotent while the loop is live; ``await stop()`` is
  idempotent and returns only once the loop has ended (a tick in flight
  is cancelled);
- a tick that raises is logged at WARNING with ``exc_info`` (under the
  caller's ``failure_message`` when given) and the loop carries on;
  ``CancelledError`` always propagates, and a stop is honoured even by a
  tick that swallowed it;
- the loop sleeps one interval before the first tick by default --
  ``first_delay`` sets that first sleep alone (``0`` ticks at once);
- a tick may return a number of seconds to use as the NEXT delay only
  (fast retry until a first success, backoff after a failure) -- ``None``
  keeps the configured interval;
- the task is spawned with ``spawn_background`` so its end is logged
  like every other background task.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any, cast

from threetears.observe.background import spawn_background

__all__ = ["PeriodicTask", "TickResult"]

#: what a tick may return: ``None`` (keep the interval) or the seconds to wait before the next tick.
TickResult = float | int | None


def _as_delay(returned: object) -> TickResult:
    """a tick's return value as a next delay: a finite, non-negative number; anything else is ``None``.

    :param returned: what the tick returned
    :ptype returned: object
    :return: the delay in seconds, or ``None`` to keep the interval
    :rtype: TickResult
    """
    valid = (
        isinstance(returned, int | float)
        and not isinstance(returned, bool)
        and math.isfinite(returned)
        and returned >= 0
    )
    return cast("float | int", returned) if valid else None


def _seconds(value: float | timedelta, *, label: str, allow_zero: bool = False) -> float:
    """normalise ``value`` to seconds, refusing a negative one (and zero unless allowed).

    :param value: seconds, or a ``timedelta``
    :ptype value: float | timedelta
    :param label: the parameter's name, for the error
    :ptype label: str
    :param allow_zero: whether zero is valid
    :ptype allow_zero: bool
    :return: the value in seconds
    :rtype: float
    :raises ValueError: the value is negative, or zero when zero is not allowed
    """
    seconds = value.total_seconds() if isinstance(value, timedelta) else float(value)
    if seconds < 0 or (seconds == 0 and not allow_zero):
        raise ValueError(f"{label} must be {'non-negative' if allow_zero else 'positive'}, got {value!r}")
    return seconds


class PeriodicTask:
    """run ``tick`` every ``interval`` on a background task, surviving failed ticks.

    :param tick: zero-argument coroutine function run once per period; may return the next delay
    :ptype tick: Callable[[], Awaitable[TickResult]]
    :param interval: seconds (or a ``timedelta``) between ticks; must be positive
    :ptype interval: float | timedelta
    :param name: task name, used for the task and in every log line
    :ptype name: str
    :param logger: logger for failed ticks and the task's end
    :ptype logger: logging.Logger
    :param first_delay: the sleep before the FIRST tick only; ``None`` (default) sleeps one
        interval, ``0`` ticks at once
    :ptype first_delay: float | timedelta | None
    :param failure_message: the WARNING logged for a failed tick (default
        ``"periodic tick failed: <name>"``) -- so a loop moved onto this keeps the message its
        operators already alert on
    :ptype failure_message: str | None
    """

    def __init__(
        self,
        tick: Callable[[], Awaitable[Any]],
        *,
        interval: float | timedelta,
        name: str,
        logger: logging.Logger,
        first_delay: float | timedelta | None = None,
        failure_message: str | None = None,
    ) -> None:
        self._tick = tick
        self.interval = _seconds(interval, label="interval")
        self.first_delay = (
            self.interval if first_delay is None else _seconds(first_delay, label="first_delay", allow_zero=True)
        )
        self.name = name
        self._logger = logger
        self._failure_message = failure_message if failure_message is not None else f"periodic tick failed: {name}"
        self._task: asyncio.Task[Any] | None = None

    @property
    def running(self) -> bool:
        """whether the loop is live and not being stopped.

        :return: ``True`` from :meth:`start` until :meth:`stop` begins
        :rtype: bool
        """
        task = self._task
        return task is not None and not task.done() and task.cancelling() == 0

    def start(self) -> None:
        """start the loop; a no-op while it is running.

        called while a :meth:`stop` is still unwinding, it starts a fresh loop beside the one
        ending.

        :return: nothing
        :rtype: None
        """
        if not self.running:
            self._task = spawn_background(self._loop(), name=self.name, logger=self._logger)

    async def stop(self) -> None:
        """stop the loop and wait for it to end; a no-op when it is not running.

        a tick in flight is cancelled. any number of callers may stop at once, and each returns
        only once the loop has ended. a tick may stop its own loop: the call returns at once and the
        loop ends when the tick does.

        :return: nothing
        :rtype: None
        """
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        if task is asyncio.current_task():
            return  # awaiting our own end would never return; the loop ends when this tick does
        await asyncio.wait([task])
        if self._task is task:
            self._task = None

    async def run_once(self) -> TickResult:
        """run one tick with the loop's failure isolation, without starting the loop.

        :return: the tick's requested next delay, or ``None`` (also after a failed tick)
        :rtype: TickResult
        """
        result: TickResult = None
        try:
            returned = await self._tick()
        except asyncio.CancelledError:
            raise
        except (
            Exception
        ):  # prawduct:allow prawduct/broad-except -- one bad tick must never end the loop; logged with its traceback
            error = sys.exc_info()[1]
            self._logger.warning(
                self._failure_message,
                extra={"extra_data": {"task_name": self.name, "error": f"{type(error).__name__}: {error}"}},
                exc_info=True,
            )
        else:
            result = _as_delay(returned)
        return result

    async def _loop(self) -> None:
        """sleep, tick, repeat -- until cancelled.

        :return: nothing (runs until cancelled)
        :rtype: None
        """
        delay = self.first_delay
        while True:
            await asyncio.sleep(delay)
            requested = await self.run_once()
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                # a stop landed during a tick that swallowed its CancelledError; honour it anyway
                raise asyncio.CancelledError
            delay = self.interval if requested is None else float(requested)
