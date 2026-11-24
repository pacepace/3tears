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
on a later write, not only when it is read: a sign-in that is started and abandoned
is never read again, and anyone can start one, so a store that dropped entries only
on read would grow without bound. The entries live in append-only frozen layers,
merged log-structured as they grow, so no map is ever changed in place and a write
costs amortised O(log n) (see :class:`_ExpiringEntries`). ``max_entries`` caps the
live entries on top of that: a write that would add one past it raises
:class:`MemoryStoreFullError` rather than evicting someone's in-flight state.
Overwriting a live key is never refused.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from datetime import timedelta
from types import MappingProxyType
from typing import Any, NamedTuple

from threetears.iam.stores.base import (
    AttemptWindow,
    TicketIssue,
    hash_ticket,
    new_ticket_secret,
)

__all__ = ["MemoryAttemptLimiter", "MemoryStateStore", "MemoryStoreFullError", "MemoryTicketStore"]

#: Marks a key as taken (popped, redeemed, cleared): it hides every older layer's record for it.
_TAKEN: Any = object()


class _Record(NamedTuple):
    """One key's newest word in a layer.

    :ivar value: the value, or :data:`_TAKEN` for a take marker
    :ivar expires_at: when the value stops being live (a marker carries the expiry it hides)
    :ivar holds_until: the latest expiry this record or any older record it hides reaches; a
        layer is dropped only once every record in it is past this, so dropping a layer can
        never let an older still-live record for the same key show through
    """

    value: Any
    expires_at: float
    holds_until: float


class _Layer(NamedTuple):
    """A frozen map of records, with the latest ``holds_until`` among them."""

    records: Mapping[str, _Record]
    holds_until: float


def _layer(records: Mapping[str, _Record]) -> _Layer:
    return _Layer(MappingProxyType(dict(records)), max((r.holds_until for r in records.values()), default=-math.inf))


def _merged(older: _Layer, newer: _Layer, *, bottom: bool, now: float) -> _Layer:
    """``newer`` over ``older`` as one layer; the bottom layer also sheds markers and expired values.

    Only the bottom layer may shed them: above it, a marker or an expired value still hides an
    older record for its key. Below the bottom there is nothing left to hide.
    """
    records = {**older.records, **newer.records}  # function-local, frozen by _layer
    if bottom:
        records = {k: r for k, r in records.items() if r.value is not _TAKEN and r.expires_at > now}
    return _layer(records)


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
    """Keys with values that expire, held in append-only frozen layers, newest first.

    A write or a take appends a one-record layer; whenever the newest layer is at least as
    large as the one below it the two merge (log-structured, like a binary counter), so there
    are O(log n) layers and each record is copied O(log n) times: a write costs amortised
    O(log n) and a lookup checks O(log n) layers, newest first. A take is a marker record that
    hides the key in every older layer. Every write first sweeps: a layer whose records are all
    past their ``holds_until`` is dropped whole, and the layers left are merged back into order.
    No map is ever changed in place -- each layer is a ``MappingProxyType`` built once.

    The expiry is what an entry means (a sign-in's life, an attempt window), not a cache TTL:
    an entry past it reads as absent, exactly as before.

    The cap counts live entries exactly. ``_bound`` is an upper bound on them, exact until the
    earliest expiry written since the last compaction (``_exact_until``); a write at the bound
    after that compacts every layer into one (dropping markers and expired values) to count
    again. A compaction is linear, and only happens at the cap once something has expired.
    """

    def __init__(self, *, clock: Callable[[], float], max_entries: int | None) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, or None for no cap; got {max_entries}")
        self._clock = clock
        self._max_entries = max_entries
        self._layers: tuple[_Layer, ...] = ()
        self._bound = 0
        self._exact_until = math.inf

    def put(self, key: str, value: V, *, expires_at: float) -> None:
        """Store ``value`` under ``key`` until ``expires_at``, after sweeping what has expired.

        :raises MemoryStoreFullError: ``key`` is new and the live entries are at the cap
        """
        now = self._clock()
        self._sweep(now)
        newest = self._newest(key)
        live = newest is not None and newest.value is not _TAKEN and newest.expires_at > now
        if self._max_entries is not None and not live and self._bound >= self._max_entries:
            if now >= self._exact_until:
                self._compact(now)
            if self._bound >= self._max_entries:
                raise MemoryStoreFullError(self._max_entries)
        hidden = newest.holds_until if newest is not None else -math.inf
        self._append(key, _Record(value, expires_at, max(expires_at, hidden)), now)
        if not live:
            self._bound += 1
        self._exact_until = min(self._exact_until, expires_at)

    def pop(self, key: str) -> V | None:
        """Remove and return the live value under ``key``, or ``None`` when absent or expired."""
        now = self._clock()
        newest = self._newest(key)
        if newest is None or newest.value is _TAKEN or newest.expires_at <= now:
            # absent, taken, or expired: an expired newest record already reads as absent
            return None
        self._append(key, _Record(_TAKEN, newest.expires_at, newest.holds_until), now)
        self._bound -= 1
        found: V = newest.value
        return found

    def get(self, key: str) -> V | None:
        """The live value under ``key``, or ``None`` when absent or expired."""
        newest = self._newest(key)
        if newest is None or newest.value is _TAKEN or newest.expires_at <= self._clock():
            return None
        found: V = newest.value
        return found

    def _newest(self, key: str) -> _Record | None:
        for layer in self._layers:
            found = layer.records.get(key)
            if found is not None:
                return found
        return None

    def _append(self, key: str, record: _Record, now: float) -> None:
        self._layers = self._in_order((_layer({key: record}), *self._layers), now)

    @staticmethod
    def _in_order(layers: tuple[_Layer, ...], now: float) -> tuple[_Layer, ...]:
        """``layers`` with each one smaller than the one below it, merging from the top down."""
        kept: tuple[_Layer, ...] = ()
        for layer in reversed(layers):  # oldest first, so kept[-1] is the newest so far
            while kept and len(layer.records) >= len(kept[-1].records):
                layer = _merged(kept[-1], layer, bottom=len(kept) == 1, now=now)
                kept = kept[:-1]
            kept = (*kept, layer)
        return tuple(reversed(kept))

    def _sweep(self, now: float) -> None:
        kept = tuple(layer for layer in self._layers if layer.holds_until > now)
        if len(kept) != len(self._layers):
            self._layers = self._in_order(kept, now)

    def _compact(self, now: float) -> None:
        bottom = _Layer(MappingProxyType({}), -math.inf)
        for layer in reversed(self._layers):
            bottom = _merged(bottom, layer, bottom=True, now=now)
        self._layers = (bottom,) if bottom.records else ()
        self._bound = len(bottom.records)
        self._exact_until = min((r.expires_at for r in bottom.records.values()), default=math.inf)


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
