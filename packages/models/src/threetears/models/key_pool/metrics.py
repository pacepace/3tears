"""A key pool's Prometheus metrics (KP-K1), behind the ``prometheus`` extra.

:class:`KeyPoolMetrics` is a :class:`~threetears.models.key_pool.observe.KeyPoolObserver`: pass it as
``KeyPool(observer=...)`` and the pool's events move its counters as they happen. One exporter serves
any number of pools, since every series carries the pool's name.

The gauges come from the pool's own snapshot, ``KeyPool.status()``, which :meth:`KeyPoolMetrics.refresh`
reads and writes into them. They are refreshed by that explicit call, not by a callback at scrape time:
``status()`` is a coroutine that reads the shared store, and a scrape runs synchronously on whatever
thread serves ``/metrics``. Driving the coroutine from there would block that thread on the store, and
deadlock where ``/metrics`` is served from the pool's own event loop; and a store failure during a
scrape would have nowhere to go but a log line. A consumer calls ``refresh`` on its own loop -- from its
health endpoint, which gets the same snapshot back, or from a periodic task -- and a store failure
raises to it. Between refreshes the gauges hold the last snapshot.

Every label value is a pool name or a key label from configuration, or a member of an enum this
package defines (a move's reason, a key's state, a store method), so the number of series is bounded
by configuration. No secret or fingerprint is ever a label or a value (KP-B1).
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

from threetears.observe import BuildOnce

from threetears.models.key_pool.observe import MoveReason
from threetears.models.key_pool.status import KeyPoolStatus, KeyState

try:
    from prometheus_client import REGISTRY, Counter, Gauge

    _PROMETHEUS = True
except ImportError:  # NOSILENT: the prometheus extra is optional; building KeyPoolMetrics without it raises
    _PROMETHEUS = False

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

    from threetears.models.key_pool.pool import KeyPool

__all__ = ["KeyPoolMetrics"]

_KEY = ("pool", "key")

#: the states ``status()`` reports. ``TRIED`` is only ever in a ``NoKeyAvailableError``, about one call
#: that already tried the key, so it is never a key's state in a snapshot and gets no series.
_SNAPSHOT_STATES = tuple(state for state in KeyState if state is not KeyState.TRIED)

#: the reasons a call can fail over for: the three key-moving verdicts, a breaker that refused the
#: call after the pool chose its key, and a request too large for the key (``low_room``). A rate limit
#: is retried on the same key, so it is none. The other move reasons are the pool's own decisions
#: between calls.
_FAILOVER_REASONS = (
    MoveReason.EXHAUSTED,
    MoveReason.REVOKED,
    MoveReason.FORBIDDEN,
    MoveReason.TRIPPED,
    MoveReason.LOW_ROOM,
)

#: the store methods a failure is counted by: every method of the ``KeyPoolStore`` Protocol.
_STORE_OPERATIONS = (
    "reserve",
    "settle",
    "release",
    "sweep_expired",
    "snapshot",
    "read_record",
    "compare_and_set_record",
)


@dataclass(frozen=True, slots=True)
class _Instruments:
    """The instruments registered on one registry, and a weak reference to that registry."""

    registry: weakref.ref[CollectorRegistry]
    spent: Gauge
    reserved: Gauge
    remaining: Gauge
    budget: Gauge
    state: Gauge
    active: Gauge
    moves: Counter
    failovers: Counter
    room_refusals: Counter
    orphaned: Counter
    no_key_available: Counter
    store_failures: Counter


def _register(registry: CollectorRegistry) -> _Instruments:
    """Register the key pool's instruments on ``registry``."""
    return _Instruments(
        registry=weakref.ref(registry),
        spent=Gauge(
            "threetears_key_pool_spent_usd",
            "Money a key settled this billing period, orphaned reservations included",
            _KEY,
            registry=registry,
        ),
        reserved=Gauge(
            "threetears_key_pool_reserved_usd",
            "Money held on a key by calls in flight this billing period",
            _KEY,
            registry=registry,
        ),
        remaining=Gauge(
            "threetears_key_pool_remaining_usd",
            "A key's budget less its spend and reservations this billing period, never below zero",
            _KEY,
            registry=registry,
        ),
        budget=Gauge(
            "threetears_key_pool_budget_usd",
            "A key's budget for the billing period, as this process configures it",
            _KEY,
            registry=registry,
        ),
        state=Gauge(
            "threetears_key_pool_key_state",
            "1 for the state a key is in, 0 for every other state",
            (*_KEY, "state"),
            registry=registry,
        ),
        active=Gauge(
            "threetears_key_pool_key_active",
            "1 when the key is its pool's active key, else 0",
            _KEY,
            registry=registry,
        ),
        moves=Counter(
            "threetears_key_pool_moves_total",
            "Moves of a pool's active key, by the key moved off, the key moved to and why",
            ("pool", "from", "to", "reason"),
            registry=registry,
        ),
        failovers=Counter(
            "threetears_key_pool_failovers_total",
            "Calls a key refused that ran their next attempt on another key, by why the key refused",
            ("pool", "reason"),
            registry=registry,
        ),
        room_refusals=Counter(
            "threetears_key_pool_room_refusals_total",
            "Reservations a key refused for room, so the call went to another key",
            _KEY,
            registry=registry,
        ),
        orphaned=Counter(
            "threetears_key_pool_orphaned_reservations_total",
            "Reservations no call closed, settled by the expiry sweep at their full amount",
            _KEY,
            registry=registry,
        ),
        no_key_available=Counter(
            "threetears_key_pool_no_key_available_total",
            "Calls a pool refused with NoKeyAvailableError",
            ("pool",),
            registry=registry,
        ),
        store_failures=Counter(
            "threetears_key_pool_store_failures_total",
            "Store calls that failed, so the pool failed the call closed, by store method",
            ("pool", "operation"),
            registry=registry,
        ),
    )


def _registry_alive(instruments: _Instruments) -> bool:
    """Whether the registry ``instruments`` were registered on still exists.

    The cache is keyed by ``id(registry)``, which a new registry can reuse once the old one is gone;
    while the old one is alive its id is its own, so a live weak reference means the same registry.
    """
    return instruments.registry() is not None


#: a registry refuses a second instrument of the same name, so every exporter on one registry shares
#: one set, registered once however many threads build an exporter first.
_INSTRUMENTS: BuildOnce[int, _Instruments] = BuildOnce(is_current=_registry_alive)


class KeyPoolMetrics:
    """A key pool's Prometheus exporter (KP-K1): counters moved by the pool's events, gauges by
    :meth:`refresh`.

    Pass it to every pool it should count, as ``KeyPool(observer=metrics)``. Exporters built on one
    registry share its instruments, so building a second is harmless.

    Gauges, per ``(pool, key)``: ``threetears_key_pool_spent_usd``, ``_reserved_usd``,
    ``_remaining_usd`` and ``_budget_usd``; ``threetears_key_pool_key_state`` (with a ``state`` label,
    1 for the key's state and 0 for the others); and ``threetears_key_pool_key_active``.

    Counters: ``threetears_key_pool_moves_total`` by ``pool``, ``from``, ``to`` and ``reason``;
    ``threetears_key_pool_failovers_total`` by ``pool`` and ``reason``;
    ``threetears_key_pool_room_refusals_total`` and ``threetears_key_pool_orphaned_reservations_total``
    by ``pool`` and ``key``; ``threetears_key_pool_no_key_available_total`` by ``pool``; and
    ``threetears_key_pool_store_failures_total`` by ``pool`` and ``operation``.

    :param registry: the registry to register on; ``None`` (the default) uses prometheus_client's
        default registry
    :ptype registry: CollectorRegistry | None
    :raises ImportError: when the ``prometheus`` extra is not installed
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        if not _PROMETHEUS:
            raise ImportError("key pool metrics need the prometheus extra: install 3tears-models[prometheus]")
        target = registry if registry is not None else REGISTRY
        self._instruments = _INSTRUMENTS.get(id(target), lambda: _register(target))

    async def refresh(self, pool: KeyPool) -> KeyPoolStatus:
        """Read ``pool.status()`` and write it into the gauges.

        Also starts each of the pool's per-key and per-pool counters at zero, so a rate over them is
        defined before their first event.

        :param pool: the pool to read
        :ptype pool: KeyPool
        :return: the snapshot the gauges now show, for a health endpoint to return as well
        :rtype: KeyPoolStatus
        :raises KeyPoolStoreError: when the store failed; the gauges keep the previous snapshot
        """
        status = await pool.status()
        self._show(status)
        return status

    def _show(self, status: KeyPoolStatus) -> None:
        """Write one snapshot into the gauges, and start its keys' counters."""
        metrics = self._instruments
        metrics.no_key_available.labels(status.pool)
        for reason in _FAILOVER_REASONS:
            metrics.failovers.labels(status.pool, str(reason))
        for operation in _STORE_OPERATIONS:
            metrics.store_failures.labels(status.pool, operation)
        for key in status.keys:
            series = (status.pool, key.label)
            metrics.spent.labels(*series).set(float(key.spent))
            metrics.reserved.labels(*series).set(float(key.reserved))
            metrics.remaining.labels(*series).set(float(key.remaining))
            metrics.budget.labels(*series).set(float(key.budget))
            metrics.active.labels(*series).set(1 if key.active else 0)
            for state in _SNAPSHOT_STATES:
                metrics.state.labels(*series, str(state)).set(1 if key.state is state else 0)
            metrics.room_refusals.labels(*series)
            metrics.orphaned.labels(*series)

    # -- KeyPoolObserver ----------------------------------------------------------------------------

    def on_move(self, pool: str, from_label: str, to_label: str, reason: MoveReason) -> None:
        """Count a move; see :meth:`~threetears.models.key_pool.observe.KeyPoolObserver.on_move`."""
        self._instruments.moves.labels(pool, from_label, to_label, str(reason)).inc()

    def on_failover(self, pool: str, reason: MoveReason) -> None:
        """Count a failover; see :meth:`~threetears.models.key_pool.observe.KeyPoolObserver.on_failover`."""
        self._instruments.failovers.labels(pool, str(reason)).inc()

    def on_room_refusal(self, pool: str, label: str) -> None:
        """Count a refusal for room; see :meth:`~threetears.models.key_pool.observe.KeyPoolObserver.on_room_refusal`."""
        self._instruments.room_refusals.labels(pool, label).inc()

    def on_orphaned(self, pool: str, label: str, count: int) -> None:
        """Count orphaned reservations; see :meth:`~threetears.models.key_pool.observe.KeyPoolObserver.on_orphaned`."""
        self._instruments.orphaned.labels(pool, label).inc(count)

    def on_no_key_available(self, pool: str) -> None:
        """Count a ``NoKeyAvailableError``; see
        :meth:`~threetears.models.key_pool.observe.KeyPoolObserver.on_no_key_available`."""
        self._instruments.no_key_available.labels(pool).inc()

    def on_store_failure(self, pool: str, operation: str) -> None:
        """Count a store failure; see :meth:`~threetears.models.key_pool.observe.KeyPoolObserver.on_store_failure`."""
        self._instruments.store_failures.labels(pool, operation).inc()
