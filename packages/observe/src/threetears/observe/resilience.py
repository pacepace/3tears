"""startup resilience utilities for platform services.

provides retry-with-exponential-backoff for service initialization
steps that depend on external infrastructure (NATS, database, KV).
services must survive starting in any order and tolerate temporary
unavailability of their dependencies.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from threetears.observe import get_logger

__all__ = ["retry_bounded", "retry_until_done", "retry_with_backoff"]

_logger = get_logger(__name__)

_T = TypeVar("_T")


async def retry_with_backoff(
    operation: Callable[[], Awaitable[None]],
    name: str,
    max_attempts: int | None = 30,
    initial_backoff: float = 2.0,
    max_backoff: float = 30.0,
) -> bool:
    """retry async operation with exponential backoff.

    intended for service startup steps that depend on external
    infrastructure. logs warnings on retry and errors on final
    failure. never raises.

    two modes, selected by ``max_attempts``:

    - **finite** (``max_attempts`` is an int, the default ``30``):
      best-effort. after the ceiling is reached it logs an error and
      returns ``False`` so the caller can degrade a genuinely-optional
      step. the backoff schedule is deterministic (no jitter).
    - **infinite** (``max_attempts`` is ``None``): startup-critical.
      the operation is retried FOREVER -- until it succeeds -- with the
      same bounded exponential backoff plus jitter. this is the mode for
      a NATS handler / subscription / responder ``.start()`` the service
      cannot serve without: rather than give up and let the caller
      proceed to a falsely-ready state with a dead handler, the call
      blocks here so the readiness gate stays closed and the orchestrator
      holds traffic, then self-heals the instant the dependency returns.
      in this mode the function only ever returns ``True`` (on eventual
      success); it never returns ``False``. jitter decorrelates a fleet
      of pods all retrying against the same recovering dependency so they
      do not reconnect in lockstep.

    :param operation: async callable to retry
    :ptype operation: Callable[[], Awaitable[None]]
    :param name: human-readable name for logging
    :ptype name: str
    :param max_attempts: maximum retry attempts, or ``None`` to retry
        forever (startup-critical mode)
    :ptype max_attempts: int | None
    :param initial_backoff: initial backoff seconds
    :ptype initial_backoff: float
    :param max_backoff: maximum backoff seconds
    :ptype max_backoff: float
    :return: True if operation succeeded; False only when a finite
        ``max_attempts`` is exhausted (infinite mode never returns False)
    :rtype: bool
    """
    backoff = initial_backoff
    result = False
    attempt = 0
    while True:
        attempt += 1
        try:
            await operation()
            if attempt > 1:
                _logger.info(
                    "%s succeeded on attempt %d",
                    name,
                    attempt,
                )
            result = True
            break
        except Exception as exc:
            if max_attempts is not None and attempt >= max_attempts:
                _logger.error(
                    "%s failed after %d attempts: %s",
                    name,
                    max_attempts,
                    exc,
                )
                break
            if max_attempts is None:
                _logger.warning(
                    "%s attempt %d failed (retrying in %.1fs; startup-critical, "
                    "will retry until the dependency is up): %s",
                    name,
                    attempt,
                    backoff,
                    exc,
                )
                # equal jitter on the startup-critical path: sleep 50-100% of
                # the current backoff so a fleet of pods does not stampede a
                # recovering dependency in lockstep. a deterministic minimum
                # half-backoff keeps the loop from busy-spinning.
                sleep_for = backoff * (0.5 + random.random() * 0.5)
            else:
                _logger.warning(
                    "%s attempt %d/%d failed (retrying in %.1fs): %s",
                    name,
                    attempt,
                    max_attempts,
                    backoff,
                    exc,
                )
                sleep_for = backoff
            await asyncio.sleep(sleep_for)
            backoff = min(backoff * 2, max_backoff)
    return result


async def retry_until_done(
    attempt: Callable[[], Awaitable[bool]],
    *,
    first_delay: float,
    max_delay: float,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> int:
    """run ``attempt`` until it reports done, pausing with capped doubling backoff between attempts.

    the engine for work that must never be abandoned while its owner runs -- putting back what a
    NATS restart took, writing a catalog back into its bucket. it never gives up and never logs:
    ``attempt`` returns ``True`` when there is nothing left to do and ``False`` when it should run
    again, and it logs its own failures, because only the caller can name what is still missing.
    an exception ``attempt`` lets out propagates; a caller that must retry through one catches it
    inside ``attempt``. only cancellation of the caller's task ends it early.

    unlike :func:`retry_with_backoff`, which serves BOUNDED startup steps and returns ``False`` on
    exhaustion, this has no ceiling: its callers have nothing to fall back to.

    :param attempt: one attempt; ``True`` when done
    :ptype attempt: Callable[[], Awaitable[bool]]
    :param first_delay: pause after the first attempt that was not done, in seconds (> 0)
    :ptype first_delay: float
    :param max_delay: ceiling the doubling pause is clamped to, in seconds (>= ``first_delay``)
    :ptype max_delay: float
    :param sleep: how a pause is taken; ``None`` (production) is ``asyncio.sleep``, looked up when
        the pause is taken
    :ptype sleep: Callable[[float], Awaitable[None]] | None
    :return: how many attempts ran, the last of them the one that was done
    :rtype: int
    :raises ValueError: when ``first_delay`` is not positive or ``max_delay`` is below it
    """
    if first_delay <= 0 or max_delay < first_delay:
        raise ValueError(f"retry_until_done needs 0 < first_delay <= max_delay, got {first_delay!r} and {max_delay!r}")
    delay = first_delay
    attempts = 1
    while not await attempt():
        await (sleep or asyncio.sleep)(delay)
        delay = min(delay * 2, max_delay)
        attempts += 1
    return attempts


async def retry_bounded(
    attempt: Callable[[], Awaitable[_T]],
    *,
    retry_on: Callable[[Exception], bool],
    first_delay: float,
    max_delay: float,
    max_attempts: int | None = None,
    deadline_seconds: float | None = None,
    backs_off: Callable[[Exception], bool] | None = None,
    on_retry: Callable[[Exception, int, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] | None = None,
) -> _T:
    """run ``attempt`` until it returns, retrying the failures ``retry_on`` accepts, within a bound.

    the sibling of :func:`retry_until_done` for work that must GIVE UP: a bind waiting for a
    bucket's declarer, a pod's first connect to a platform still starting. each failure is
    classified as it arrives:

    - one ``retry_on`` refuses is raised at once -- a refusal no retry can clear;
    - one it accepts is retried after a pause that starts at ``first_delay`` and doubles up to
      ``max_delay``, unless ``backs_off`` says the attempt already waited for it (it paced itself),
      in which case it is retried at once and the pause does not advance;
    - once ``max_attempts`` attempts have run, or ``deadline_seconds`` have passed since the first,
      the LAST failure is raised unchanged, so the caller turns it into its own error and message.

    a pause never runs past the deadline. only ``Exception`` is classified: cancellation, a
    ``BaseException``, always propagates.

    :param attempt: one attempt; its return value is returned
    :ptype attempt: Callable[[], Awaitable[_T]]
    :param retry_on: whether a failure is worth another attempt
    :ptype retry_on: Callable[[Exception], bool]
    :param first_delay: the first pause, in seconds (> 0)
    :ptype first_delay: float
    :param max_delay: ceiling the doubling pause is clamped to (>= ``first_delay``)
    :ptype max_delay: float
    :param max_attempts: how many attempts may run; ``None`` for no attempt bound
    :ptype max_attempts: int | None
    :param deadline_seconds: how long after the first attempt began a retry may still start;
        ``None`` for no time bound. at least one bound is required
    :ptype deadline_seconds: float | None
    :param backs_off: whether a retried failure takes a pause; ``None`` pauses after every one
    :ptype backs_off: Callable[[Exception], bool] | None
    :param on_retry: told of each failure that will be retried, its attempt number (from 1) and
        the pause before the next attempt -- where the caller logs
    :ptype on_retry: Callable[[Exception, int, float], None] | None
    :param sleep: how a pause is taken; ``None`` (production) is ``asyncio.sleep``, looked up when
        the pause is taken
    :ptype sleep: Callable[[float], Awaitable[None]] | None
    :param clock: a monotonic clock in seconds for the deadline; ``None`` is the running loop's
    :ptype clock: Callable[[], float] | None
    :return: what the attempt that succeeded returned
    :rtype: _T
    :raises ValueError: when neither bound is given, or the schedule cannot back off
    :raises Exception: a failure ``retry_on`` refused, or the last failure once the bound is spent
    """
    if max_attempts is None and deadline_seconds is None:
        raise ValueError("retry_bounded needs a bound: max_attempts, deadline_seconds, or both")
    if first_delay <= 0 or max_delay < first_delay:
        raise ValueError(f"retry_bounded needs 0 < first_delay <= max_delay, got {first_delay!r} and {max_delay!r}")
    now = clock or asyncio.get_running_loop().time
    deadline = None if deadline_seconds is None else now() + deadline_seconds
    delay = first_delay
    attempts = 0
    while True:
        attempts += 1
        try:
            return await attempt()
        except Exception as exc:
            if not retry_on(exc):
                raise
            remaining = None if deadline is None else deadline - now()
            spent = (max_attempts is not None and attempts >= max_attempts) or (
                remaining is not None and remaining <= 0
            )
            if spent:
                raise
            pause = 0.0
            if backs_off is None or backs_off(exc):
                pause = delay if remaining is None else min(delay, remaining)
                delay = min(delay * 2, max_delay)
            if on_retry is not None:
                on_retry(exc, attempts, pause)
            if pause > 0:
                await (sleep or asyncio.sleep)(pause)
