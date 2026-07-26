"""In-memory store implementations, for tests.

Shipped rather than left to each consumer to hand-roll. Both services this
package was factored out of had written their own in-memory KV double, and a
fake that drifts from the real implementation is how a store bug ships green:
the test double answers the question the test asks, the real store answers a
different one, and nothing compares them.

These honour TTLs against an injectable clock and implement redemption as a
genuine single claim, so a concurrency test written against them means
something. They are NOT thread-safe and hold everything in a dict -- for tests
and local development only.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from threetears.iam.stores.base import (
    AttemptWindow,
    TicketIssue,
    hash_ticket,
    new_ticket_secret,
)

__all__ = ["MemoryAttemptLimiter", "MemoryStateStore", "MemoryTicketStore"]


@dataclass
class _Entry:
    payload: Mapping[str, Any]
    expires_at: float


class MemoryTicketStore:
    """In-memory :class:`~threetears.iam.stores.base.SingleUseTicketStore`."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._entries: dict[str, _Entry] = {}
        self._clock = clock

    async def issue(self, payload: Mapping[str, Any], *, ttl: timedelta) -> TicketIssue:
        secret = new_ticket_secret()
        hashed = hash_ticket(secret)
        self._entries[hashed] = _Entry(payload=dict(payload), expires_at=self._clock() + ttl.total_seconds())
        return TicketIssue(secret=secret, hashed=hashed)

    async def redeem(self, secret: str) -> Mapping[str, Any] | None:
        # pop, not get-then-delete: the removal IS the claim, so a second redemption of the
        # same ticket finds nothing regardless of interleaving.
        entry = self._entries.pop(hash_ticket(secret), None)
        if entry is None or entry.expires_at <= self._clock():
            return None
        return entry.payload


class MemoryStateStore:
    """In-memory :class:`~threetears.iam.stores.base.StateStore`."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._entries: dict[str, _Entry] = {}
        self._clock = clock

    async def put(self, key: str, payload: Mapping[str, Any], *, ttl: timedelta) -> None:
        self._entries[key] = _Entry(payload=dict(payload), expires_at=self._clock() + ttl.total_seconds())

    async def take(self, key: str) -> Mapping[str, Any] | None:
        entry = self._entries.pop(key, None)
        if entry is None or entry.expires_at <= self._clock():
            return None
        return entry.payload

    async def get(self, key: str) -> Mapping[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None or entry.expires_at <= self._clock():
            return None
        return entry.payload


class MemoryAttemptLimiter:
    """In-memory :class:`~threetears.iam.stores.base.AttemptLimiter`, fixed-window."""

    def __init__(
        self,
        *,
        max_attempts: int = 5,
        window: timedelta = timedelta(minutes=15),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_attempts = max_attempts
        self._window = window.total_seconds()
        self._clock = clock
        self._counts: dict[str, tuple[int, float]] = {}

    def _current(self, key: str) -> tuple[int, float]:
        count, started = self._counts.get(key, (0, self._clock()))
        if self._clock() - started >= self._window:
            return 0, self._clock()
        return count, started

    def _verdict(self, count: int, started: float) -> AttemptWindow:
        limited = count >= self._max_attempts
        remaining = self._window - (self._clock() - started)
        return AttemptWindow(
            count=count,
            limited=limited,
            retry_after=timedelta(seconds=max(remaining, 0.0)) if limited else None,
        )

    async def record_failure(self, key: str) -> AttemptWindow:
        count, started = self._current(key)
        count += 1
        self._counts[key] = (count, started)
        return self._verdict(count, started)

    async def check(self, key: str) -> AttemptWindow:
        count, started = self._current(key)
        return self._verdict(count, started)

    async def clear(self, key: str) -> None:
        self._counts.pop(key, None)
