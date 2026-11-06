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
- a tick may return :class:`NextDelay` to set the NEXT delay only (fast
  retry until a first success, backoff after a failure); anything else it
  returns -- ``None``, a count, a list -- keeps the configured interval. A
  bare number is not a delay: a pass that returns how much it did (almost
  always ``0``) would otherwise re-tick with no sleep at all;
- a requested delay is floored at ``min_delay`` (one second by default), so
  even an explicit ``NextDelay(0)`` cannot spin the loop;
- the task is spawned with ``spawn_background`` so its end is logged
  like every other background task.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from threetears.observe.background import spawn_background
from threetears.observe.containment import contain

__all__ = ["NextDelay", "PeriodicTask", "TickResult"]

#: the floor under a delay a tick requests: low enough for a fast retry, high enough that a request for no
#: delay at all costs one tick a second, not a core.
_DEFAULT_MIN_DELAY = 1.0


@dataclass(frozen=True, slots=True)
class NextDelay:
    """a tick's request for the delay before the NEXT tick only; the configured interval applies after it.

    :param seconds: seconds (or a ``timedelta``) to wait; finite and non-negative. The loop floors it at the
        task's ``min_delay``
    :ptype seconds: float | timedelta
    :raises ValueError: ``seconds`` is negative or not finite
    """

    seconds: float | timedelta

    def __post_init__(self) -> None:
        """refuse a delay that is negative or not finite.

        :return: nothing
        :rtype: None
        :raises ValueError: the delay is negative or not finite
        """
        _seconds(self.seconds, label="NextDelay.seconds", allow_zero=True)


#: what a tick may return to steer the loop: :class:`NextDelay`, or ``None`` to keep the interval. Any other
#: value is ignored, and the interval holds.
TickResult = NextDelay | None


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
    if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and not allow_zero):
        raise ValueError(f"{label} must be {'non-negative' if allow_zero else 'positive'}, got {value!r}")
    return seconds


class PeriodicTask:
    """run ``tick`` every ``interval`` on a background task, surviving failed ticks.

    :param tick: zero-argument coroutine function run once per period; may return :class:`NextDelay`
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
    :param min_delay: the least a tick's :class:`NextDelay` may set the next sleep to; must be positive
        (default one second). The configured ``interval`` and ``first_delay`` are not floored
    :ptype min_delay: float | timedelta
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
        min_delay: float | timedelta = _DEFAULT_MIN_DELAY,
    ) -> None:
        self._tick = tick
        self.interval = _seconds(interval, label="interval")
        self.min_delay = _seconds(min_delay, label="min_delay")
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
        only once the loop has ended. a tick may stop its own loop: the call returns at once, and the
        tick is cancelled at its next ``await`` (or the loop ends when it returns).

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

    async def run_once(self) -> float | None:
        """run one tick with the loop's failure isolation, without starting the loop.

        :return: the seconds the tick's :class:`NextDelay` sets the next sleep to, floored at ``min_delay``;
            ``None`` when it returned anything else, and after a failed tick
        :rtype: float | None
        """
        result: float | None = None
        try:
            returned = await self._tick()
        except asyncio.CancelledError:
            raise
        except Exception as error:  # prawduct:allow prawduct/broad-except -- one bad tick must never end the loop
            contain(
                error,
                log=self._logger,
                what=self._failure_message,
                extra={"task_name": self.name, "error": f"{type(error).__name__}: {error}"},
            )
        else:
            if isinstance(returned, NextDelay):
                result = max(self.min_delay, _seconds(returned.seconds, label="NextDelay.seconds", allow_zero=True))
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
            delay = self.interval if requested is None else requested
