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
    carried one, else ``first_wait_s`` doubled for each attempt already made, at most ``max_wait_s``. A provider that
    asks for longer than ``max_wait_s`` is not sent the call again: its answer is raised at once, since a resend
    before the time it named would meet the same answer.

    **The time bound.** Every wait and every resend ends within one request's timeout (the client's read timeout,
    ``Model(timeout=)``) of the call's first attempt: no wait starts that would end past it, and a resend still
    running then is cut, raising the decline before it. The first attempt is bounded by the client's timeout as any
    request is. So a call ends one request's timeout after it started, or when its first attempt does if that took
    longer.

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

    def wait_s(self, made: int, retry_after: str | None) -> float | None:
        """How long to wait before the next attempt, after ``made`` attempts, or ``None`` to send no more.

        :param made: Attempts made so far, the first is one.
        :ptype made: int
        :param retry_after: The answer's ``retry-after`` header, or ``None``.
        :ptype retry_after: str | None
        :return: Seconds, at most ``max_wait_s``; ``None`` when the provider asked for longer than that.
        :rtype: float | None
        """
        asked = _seconds(retry_after)
        if asked is not None:
            return asked if asked <= self.max_wait_s else None
        backoff: float = self.first_wait_s * 2.0 ** (made - 1)
        return min(backoff, self.max_wait_s)


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
