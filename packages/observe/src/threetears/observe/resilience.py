"""startup resilience utilities for platform services.

provides retry-with-exponential-backoff for service initialization
steps that depend on external infrastructure (NATS, database, KV).
services must survive starting in any order and tolerate temporary
unavailability of their dependencies.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable

from threetears.observe import get_logger

__all__ = ["retry_with_backoff"]

_logger = get_logger(__name__)


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
