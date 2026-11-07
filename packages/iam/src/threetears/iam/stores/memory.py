"""In-memory store implementations: for tests, and for a single-process service.

Shipped rather than left to each consumer to hand-roll. Both services this
package was factored out of had written their own in-memory KV double, and a
fake that drifts from the real implementation is how a store bug ships green:
the test double answers the question the test asks, the real store answers a
different one, and nothing compares them.

These honour TTLs against an injectable clock and implement redemption as a
genuine single claim, so a concurrency test written against them means
something.

**Supported in production on stated terms:** one process, with one event loop.
State lives in a dict, so it is lost on restart -- for OAuth state or a sign-in
ticket that costs whoever is mid-sign-in a retry -- and it is not shared with any
other process. The stores are NOT thread-safe: every call must come from the same
event loop. A deployment with more than one process needs a shared store
(:mod:`~threetears.iam.stores.nats_kv`, :mod:`~threetears.iam.stores.postgres`).

**Memory is bounded by expiry, and optionally by a cap.** An expired entry is swept
on the next write, not only when it is read: a sign-in that is started and abandoned
is never read again, and anyone can start one, so a store that dropped entries only
on read would grow without bound. Each write first removes whatever has expired, in
expiry order, at amortised constant cost. ``max_entries`` caps the live entries on
top of that: a write that would add one past it raises :class:`MemoryStoreFullError`
rather than evicting someone's in-flight state. Overwriting a live key is never
refused.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import Any

from threetears.iam.stores.base import (
    AttemptWindow,
    TicketIssue,
    hash_ticket,
    new_ticket_secret,
)

__all__ = ["MemoryAttemptLimiter", "MemoryStateStore", "MemoryStoreFullError", "MemoryTicketStore"]

#: Stale heap slots (keys taken, cleared or rewritten before they expired) tolerated beyond
#: the live entries before the heap is rebuilt from them. Rebuilding is linear, so doing it
#: only once the slack doubles keeps each write amortised constant.
_HEAP_SLACK = 64


class MemoryStoreFullError(Exception):
    """A write would add a live entry past the store's ``max_entries``.

    Raised rather than evicting: the entry that would go is someone's in-flight sign-in.
    Expired entries are swept before the count is taken, so this means that many entries
    are genuinely live.

    :ivar max_entries: the cap that was reached
    """

    def __init__(self, max_entries: int) -> None:
        super().__init__(f"the in-memory store holds its maximum of {max_entries} live entries")
        self.max_entries = max_entries


class _ExpiringEntries[V]:
    """A dict whose expired entries are swept on write, in expiry order.

    A min-heap of ``(expires_at, key)`` orders the sweep. A slot whose key was since removed
    or rewritten with a later expiry is skipped when it surfaces, so removal stays a plain
    ``dict.pop``.
    """

    def __init__(self, *, clock: Callable[[], float], max_entries: int | None) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, or None for no cap; got {max_entries}")
        self._clock = clock
        self._max_entries = max_entries
        self._entries: dict[str, tuple[V, float]] = {}
        self._expiries: list[tuple[float, str]] = []

    def put(self, key: str, value: V, *, expires_at: float) -> None:
        """Store ``value`` under ``key`` until ``expires_at``, after sweeping what has expired.

        :raises MemoryStoreFullError: ``key`` is new and the live entries are at the cap
        """
        self._sweep()
        if self._max_entries is not None and key not in self._entries and len(self._entries) >= self._max_entries:
            raise MemoryStoreFullError(self._max_entries)
        self._entries[key] = (value, expires_at)
        heapq.heappush(self._expiries, (expires_at, key))

    def pop(self, key: str) -> V | None:
        """Remove and return the live value under ``key``, or ``None`` when absent or expired."""
        found = self._entries.pop(key, None)
        if found is None or found[1] <= self._clock():
            return None
        return found[0]

    def get(self, key: str) -> V | None:
        """The live value under ``key``, or ``None`` when absent or expired."""
        found = self._entries.get(key)
        if found is None or found[1] <= self._clock():
            return None
        return found[0]

    def _sweep(self) -> None:
        now = self._clock()
        while self._expiries and self._expiries[0][0] <= now:
            _, key = heapq.heappop(self._expiries)
            found = self._entries.get(key)
            if found is not None and found[1] <= now:
                del self._entries[key]
        if len(self._expiries) > 2 * len(self._entries) + _HEAP_SLACK:
            self._expiries = [(expires_at, key) for key, (_, expires_at) in self._entries.items()]
            heapq.heapify(self._expiries)


class MemoryTicketStore:
    """In-memory :class:`~threetears.iam.stores.base.SingleUseTicketStore`.

    :param clock: seconds, monotonic; injectable so TTL tests do not sleep
    :param max_entries: the most live tickets held at once, or ``None`` for no cap
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, max_entries: int | None = None) -> None:
        self._entries: _ExpiringEntries[Mapping[str, Any]] = _ExpiringEntries(clock=clock, max_entries=max_entries)
        self._clock = clock

    async def issue(self, payload: Mapping[str, Any], *, ttl: timedelta) -> TicketIssue:
        """:raises MemoryStoreFullError: the store holds ``max_entries`` live tickets"""
        secret = new_ticket_secret()
        hashed = hash_ticket(secret)
        self._entries.put(hashed, dict(payload), expires_at=self._clock() + ttl.total_seconds())
        return TicketIssue(secret=secret, hashed=hashed)

    async def redeem(self, secret: str) -> Mapping[str, Any] | None:
        # pop, not get-then-delete: the removal IS the claim, so a second redemption of the
        # same ticket finds nothing regardless of interleaving.
        return self._entries.pop(hash_ticket(secret))


class MemoryStateStore:
    """In-memory :class:`~threetears.iam.stores.base.StateStore`.

    :param clock: seconds, monotonic; injectable so TTL tests do not sleep
    :param max_entries: the most live keys held at once, or ``None`` for no cap
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, max_entries: int | None = None) -> None:
        self._entries: _ExpiringEntries[Mapping[str, Any]] = _ExpiringEntries(clock=clock, max_entries=max_entries)
        self._clock = clock

    async def put(self, key: str, payload: Mapping[str, Any], *, ttl: timedelta) -> None:
        """:raises MemoryStoreFullError: ``key`` is new and the store holds ``max_entries`` live keys"""
        self._entries.put(key, dict(payload), expires_at=self._clock() + ttl.total_seconds())

    async def take(self, key: str) -> Mapping[str, Any] | None:
        return self._entries.pop(key)

    async def get(self, key: str) -> Mapping[str, Any] | None:
        return self._entries.get(key)


class MemoryAttemptLimiter:
    """In-memory :class:`~threetears.iam.stores.base.AttemptLimiter`, fixed-window.

    A key's count is swept on a later write once its window has passed, so keys that fail
    once and never return do not accumulate. There is no cap: refusing to count a failure
    would be refusing to limit it.
    """

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
        self._counts: _ExpiringEntries[tuple[int, float]] = _ExpiringEntries(clock=clock, max_entries=None)

    def _current(self, key: str) -> tuple[int, float]:
        # an absent or elapsed window reads as a fresh one starting now
        return self._counts.get(key) or (0, self._clock())

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
        self._counts.put(key, (count, started), expires_at=started + self._window)
        return self._verdict(count, started)

    async def check(self, key: str) -> AttemptWindow:
        count, started = self._current(key)
        return self._verdict(count, started)

    async def clear(self, key: str) -> None:
        self._counts.pop(key)
