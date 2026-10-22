"""How a call the provider declined is sent again: :class:`DeclinedRetry`.

A provider that declines a call at that moment (a 429 rate limit, a 529 overloaded, a 503 unavailable: the
``provider`` owner of :class:`~threetears.evals.providers.ProviderError`) bills nothing and may serve the same request
a moment later, so a short wait and a resend keeps the trial rather than excluding it. Only those are sent again: a
failure the account owns (a spent quota, a refused key) or the candidate owns (a 500, a 502, a bad request) would meet
the same answer, or is the model's outcome.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

__all__ = ["DeclinedRetry"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DeclinedRetry:
    """How many times a declined call is sent, and how long each resend waits.

    The wait before attempt ``n + 1`` is the provider's ``retry-after`` (seconds, or an HTTP date) when the answer
    carried one, else ``first_wait_s`` doubled for each attempt already made; either way at most ``max_wait_s``. No
    resend starts once the request's own timeout would have run out since the first attempt, so a retried call never
    outlives the time one request was given.

    :ivar attempts: Attempts in all, the first included; ``1`` sends a declined call once and raises its answer.
    :ivar first_wait_s: The wait before the second attempt when the provider named none, in seconds.
    :ivar max_wait_s: The longest any one wait is, the provider's own ``retry-after`` included, in seconds.
    :ivar sleep: How a wait is waited: ``asyncio.sleep``, or a host's own (a test's that waits for nothing).
    """

    attempts: int = 3
    first_wait_s: float = 1.0
    max_wait_s: float = 20.0
    sleep: Callable[[float], Awaitable[None]] = field(default=asyncio.sleep, compare=False)

    def __post_init__(self) -> None:
        """Refuse a policy that sends nothing, or waits a negative or unbounded time.

        :raises ValueError: ``attempts`` below one, or a wait that is negative or not finite.
        """
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or self.attempts < 1:
            raise ValueError(f"DeclinedRetry.attempts {self.attempts!r} is not a whole number of at least one")
        for name, value in (("first_wait_s", self.first_wait_s), ("max_wait_s", self.max_wait_s)):
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"DeclinedRetry.{name} {value!r} is not a finite number of seconds at least zero")

    def wait_s(self, made: int, retry_after: str | None) -> float:
        """How long to wait before the next attempt, after ``made`` attempts.

        :param made: Attempts made so far, the first is one.
        :ptype made: int
        :param retry_after: The answer's ``retry-after`` header, or ``None``.
        :ptype retry_after: str | None
        :return: Seconds, at most ``max_wait_s``.
        :rtype: float
        """
        asked = _seconds(retry_after)
        wanted = asked if asked is not None else self.first_wait_s * 2 ** (made - 1)
        return min(wanted, self.max_wait_s)


def _seconds(retry_after: str | None) -> float | None:
    """A ``retry-after`` value in seconds from now: a number of seconds, or an HTTP date; ``None`` when unreadable."""
    if retry_after is None or not retry_after.strip():
        return None
    text = retry_after.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except TypeError, ValueError:
            # NOSILENT: a header that is neither seconds nor a date names no wait; the backoff's own wait is used.
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - datetime.now(UTC)).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(seconds, 0.0)
