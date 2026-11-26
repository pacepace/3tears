"""What a key pool counts, and the hook its metrics are exported through (KP-K1).

The pool reports each event to a :class:`KeyPoolObserver`. It always keeps its own
:class:`KeyPoolCounters`, which ``KeyPool.counters`` exposes, and also reports to an observer it is
given, such as a Prometheus exporter. Every label an event carries is a pool name or a key label from
configuration, so their number is bounded by configuration.

None of these counts is deduplicated: a ``NoKeyAvailableError`` raise is counted every time, even when
its log line is suppressed.
"""

from __future__ import annotations

import enum
import threading
from collections import Counter
from typing import Protocol

__all__ = ["KeyPoolCounters", "KeyPoolObserver", "MoveReason"]


class MoveReason(enum.StrEnum):
    """Why the pool moved its active key (KP-D7)."""

    #: the active key's credit or spend limit is used up (logged at WARNING)
    EXHAUSTED = "exhausted"
    #: the provider no longer accepts the active key (logged at WARNING)
    REVOKED = "revoked"
    #: the active key may not use the model (logged at WARNING)
    FORBIDDEN = "forbidden"
    #: the active key is rate-limited (logged at INFO)
    RATE_LIMITED = "rate_limited"
    #: the active key's circuit breaker is open in this process (logged at INFO)
    TRIPPED = "tripped"
    #: the active key's room fell below ``min_headroom`` (logged at INFO)
    LOW_ROOM = "low_room"
    #: a more-preferred key, by distinct priority, can serve again (logged at INFO)
    PULL_BACK = "pull_back"
    #: the billing period rolled over, which returns the pool to its most-preferred key (logged at INFO)
    PERIOD_ROLLOVER = "period_rollover"
    #: the stored active key is one this process isn't configured with (logged at INFO)
    NOT_CONFIGURED = "not_configured"


class KeyPoolObserver(Protocol):
    """Receives a key pool's events, to count or export them.

    Each method is called once per event, synchronously, on the pool's own task. A failure it raises is
    contained and logged; it never fails the call.
    """

    def on_move(self, pool: str, from_label: str, to_label: str, reason: MoveReason) -> None:
        """The pool moved its active key.

        :param pool: the pool's name
        :ptype pool: str
        :param from_label: the key it moved off
        :ptype from_label: str
        :param to_label: the key it moved to
        :ptype to_label: str
        :param reason: why
        :ptype reason: MoveReason
        """
        ...

    def on_room_refusal(self, pool: str, label: str) -> None:
        """A reservation didn't fit on a key's room, so the call went elsewhere.

        :param pool: the pool's name
        :ptype pool: str
        :param label: the key that refused it
        :ptype label: str
        """
        ...

    def on_orphaned(self, pool: str, label: str, count: int) -> None:
        """The expiry sweep settled reservations a call never closed, at their full amount.

        :param pool: the pool's name
        :ptype pool: str
        :param label: the key they were on
        :ptype label: str
        :param count: how many this sweep found
        :ptype count: int
        """
        ...

    def on_no_key_available(self, pool: str) -> None:
        """The pool raised ``NoKeyAvailableError``.

        :param pool: the pool's name
        :ptype pool: str
        """
        ...

    def on_store_failure(self, pool: str, operation: str) -> None:
        """A store call failed, and the pool failed the call closed (KP-G5).

        :param pool: the pool's name
        :ptype pool: str
        :param operation: the store method that failed, e.g. ``"reserve"``
        :ptype operation: str
        """
        ...


class KeyPoolCounters:
    """A :class:`KeyPoolObserver` that counts in memory: what ``KeyPool.counters`` reads.

    Safe across threads: every count changes under one lock.

    :ivar moves: moves, by ``(from_label, to_label, reason)``
    :ivar room_refusals: reservations refused for room, by key label
    :ivar orphaned: orphaned reservations swept, by key label
    :ivar no_key_available: ``NoKeyAvailableError`` raises
    :ivar store_failures: failed store calls, by store method
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.moves: Counter[tuple[str, str, MoveReason]] = Counter()
        self.room_refusals: Counter[str] = Counter()
        self.orphaned: Counter[str] = Counter()
        self.no_key_available = 0
        self.store_failures: Counter[str] = Counter()

    def on_move(self, pool: str, from_label: str, to_label: str, reason: MoveReason) -> None:
        """Count a move; see :meth:`KeyPoolObserver.on_move`."""
        with self._lock:
            self.moves[(from_label, to_label, reason)] += 1

    def on_room_refusal(self, pool: str, label: str) -> None:
        """Count a refusal for room; see :meth:`KeyPoolObserver.on_room_refusal`."""
        with self._lock:
            self.room_refusals[label] += 1

    def on_orphaned(self, pool: str, label: str, count: int) -> None:
        """Count orphaned reservations; see :meth:`KeyPoolObserver.on_orphaned`."""
        with self._lock:
            self.orphaned[label] += count

    def on_no_key_available(self, pool: str) -> None:
        """Count a ``NoKeyAvailableError``; see :meth:`KeyPoolObserver.on_no_key_available`."""
        with self._lock:
            self.no_key_available += 1

    def on_store_failure(self, pool: str, operation: str) -> None:
        """Count a store failure; see :meth:`KeyPoolObserver.on_store_failure`."""
        with self._lock:
            self.store_failures[operation] += 1
