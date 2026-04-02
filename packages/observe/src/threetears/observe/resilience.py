"""startup resilience utilities for platform services.

provides retry-with-exponential-backoff for service initialization
steps that depend on external infrastructure (NATS, database, KV).
services must survive starting in any order and tolerate temporary
unavailability of their dependencies.
"""

import asyncio
from collections.abc import Awaitable, Callable

from threetears.observe import get_logger

_logger = get_logger(__name__)


async def retry_with_backoff(
    operation: Callable[[], Awaitable[None]],
    name: str,
    max_attempts: int = 30,
    initial_backoff: float = 2.0,
    max_backoff: float = 30.0,
) -> bool:
    """retry async operation with exponential backoff.

    intended for service startup steps that depend on external
    infrastructure. logs warnings on retry and errors on final
    failure. never raises -- returns False on exhaustion so
    service can decide whether to continue degraded or abort.

    :param operation: async callable to retry
    :ptype operation: Callable[[], Awaitable[None]]
    :param name: human-readable name for logging
    :ptype name: str
    :param max_attempts: maximum retry attempts
    :ptype max_attempts: int
    :param initial_backoff: initial backoff seconds
    :ptype initial_backoff: float
    :param max_backoff: maximum backoff seconds
    :ptype max_backoff: float
    :return: True if operation succeeded, False if all attempts exhausted
    :rtype: bool
    """
    backoff = initial_backoff
    result = False
    for attempt in range(1, max_attempts + 1):
        try:
            await operation()
            if attempt > 1:
                _logger.info(
                    "%s succeeded on attempt %d",
                    name, attempt,
                )
            result = True
            break
        except Exception as exc:
            if attempt >= max_attempts:
                _logger.error(
                    "%s failed after %d attempts: %s",
                    name, max_attempts, exc,
                )
                break
            _logger.warning(
                "%s attempt %d/%d failed (retrying in %.1fs): %s",
                name, attempt, max_attempts, backoff, exc,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
    return result
