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
- a tick that raises is logged at WARNING with ``exc_info`` and the loop
  carries on; ``CancelledError`` always propagates;
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
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from threetears.observe.background import spawn_background

__all__ = ["PeriodicTask", "TickResult"]

#: what a tick may return: ``None`` (keep the interval) or the seconds to wait before the next tick.
TickResult = float | int | None


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
    """

    def __init__(
        self,
        tick: Callable[[], Awaitable[Any]],
        *,
        interval: float | timedelta,
        name: str,
        logger: logging.Logger,
        first_delay: float | timedelta | None = None,
    ) -> None:
        self._tick = tick
        self.interval = _seconds(interval, label="interval")
        self.first_delay = (
            self.interval if first_delay is None else _seconds(first_delay, label="first_delay", allow_zero=True)
        )
        self.name = name
        self._logger = logger
        self._task: asyncio.Task[Any] | None = None

    @property
    def running(self) -> bool:
        """whether the loop task is live.

        :return: ``True`` while the loop runs
        :rtype: bool
        """
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """start the loop; a no-op while it is already running.

        :return: nothing
        :rtype: None
        """
        if not self.running:
            self._task = spawn_background(self._loop(), name=self.name, logger=self._logger)

    async def stop(self) -> None:
        """stop the loop and wait for it to end; a no-op when it is not running.

        a tick in flight is cancelled. safe to call any number of times.

        :return: nothing
        :rtype: None
        """
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

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
            self._logger.warning(
                f"periodic tick failed: {self.name}",
                extra={"extra_data": {"task_name": self.name}},
                exc_info=True,
            )
        else:
            result = returned if isinstance(returned, int | float) and not isinstance(returned, bool) else None
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
            delay = float(requested) if requested is not None and requested >= 0 else self.interval
