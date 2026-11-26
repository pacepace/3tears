"""A key pool's Prometheus metrics: each counter moves on its event, the gauges show the pool's own
snapshot, and the exposition holds no secret and only labels bounded by configuration (KP-K1, KP-B1)."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from threetears.models.key_pool import (
    KeyPool,
    KeyPoolMetrics,
    KeyPoolStore,
    KeyPoolStoreError,
    KeySpec,
    KeyState,
    MoveReason,
    NoKeyAvailableError,
)

from .key_pool_support import (
    MODEL,
    POOL,
    FakeFlakyStore,
    ManualClock,
    exhausted,
    fail_on,
    make_pool,
    secret_for,
    serve,
    spec,
)


def _pool(metrics: KeyPoolMetrics, *keys: KeySpec, **settings: Any) -> KeyPool:
    return make_pool(*keys, observer=metrics, **settings)


def _sample(registry: CollectorRegistry, name: str, **labels: str) -> float | None:
    return registry.get_sample_value(name, labels)


@pytest.fixture
def registry() -> CollectorRegistry:
    return CollectorRegistry()


@pytest.fixture
def metrics(registry: CollectorRegistry) -> KeyPoolMetrics:
    return KeyPoolMetrics(registry)


class TestCounters:
    async def test_a_move_is_counted_by_from_to_and_reason(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        pool = _pool(metrics, spec("a"), spec("b"))
        await fail_on(pool, exhausted())
        moves = "threetears_key_pool_moves_total"
        assert _sample(registry, moves, pool=POOL, **{"from": "a", "to": "b", "reason": "exhausted"}) == 1
        assert pool.counters.moves == {("a", "b", MoveReason.EXHAUSTED): 1}

    async def test_a_failover_is_counted_by_reason(self, registry: CollectorRegistry, metrics: KeyPoolMetrics) -> None:
        pool = _pool(metrics, spec("a"), spec("b"))
        pool.record_failover(MoveReason.RATE_LIMITED)
        pool.record_failover(MoveReason.RATE_LIMITED)
        pool.record_failover(MoveReason.TRIPPED)
        failovers = "threetears_key_pool_failovers_total"
        assert _sample(registry, failovers, pool=POOL, reason="rate_limited") == 2
        assert _sample(registry, failovers, pool=POOL, reason="tripped") == 1
        assert pool.counters.failovers == {MoveReason.RATE_LIMITED: 2, MoveReason.TRIPPED: 1}

    async def test_a_failover_reason_outside_the_enum_is_refused(self, metrics: KeyPoolMetrics) -> None:
        pool = _pool(metrics, spec("a"))
        with pytest.raises(TypeError, match="MoveReason"):
            pool.record_failover("rate_limited")  # type: ignore[arg-type]
        assert not pool.counters.failovers

    async def test_a_reservation_too_big_for_a_key_counts_a_refusal_for_room(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        pool = _pool(metrics, spec("a", "1"), spec("b", "10"))
        lease = await pool.acquire(MODEL, Decimal("5"))
        await lease.settle(Decimal("5"))
        assert lease.label == "b"
        assert _sample(registry, "threetears_key_pool_room_refusals_total", pool=POOL, key="a") == 1
        assert _sample(registry, "threetears_key_pool_room_refusals_total", pool=POOL, key="b") is None

    async def test_an_orphaned_reservation_is_counted_on_its_key(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        clock = ManualClock()
        pool = _pool(metrics, spec("a", "10"), clock=clock, reservation_margin=timedelta(minutes=1))
        await pool.acquire(MODEL, Decimal("2"))  # never closed: the sweep orphans it
        await pool.acquire(MODEL, Decimal("2"))
        clock.advance(minutes=2)
        await serve(pool)
        assert _sample(registry, "threetears_key_pool_orphaned_reservations_total", pool=POOL, key="a") == 2

    async def test_every_no_key_available_raise_is_counted_per_pool(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        pool = _pool(metrics, spec("a"))
        await fail_on(pool, exhausted())
        for _ in range(3):
            with pytest.raises(NoKeyAvailableError):
                await pool.acquire(MODEL, Decimal("0.01"))
        assert _sample(registry, "threetears_key_pool_no_key_available_total", pool=POOL) == 3

    async def test_a_store_failure_is_counted_by_store_method(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        store = FakeFlakyStore()
        store.down = True
        pool = _pool(metrics, spec("a"), store=store)
        with pytest.raises(KeyPoolStoreError):
            await pool.acquire(MODEL, Decimal("0.01"))
        assert pool.counters.store_failures == {"read_record": 1}
        assert _sample(registry, "threetears_key_pool_store_failures_total", pool=POOL, operation="read_record") == 1

    async def test_one_exporter_keeps_each_pools_counts_apart(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        one, two = _pool(metrics, spec("a"), name="one"), _pool(metrics, spec("a"), name="two")
        await fail_on(one, exhausted())
        for _ in range(3):
            with pytest.raises(NoKeyAvailableError):
                await one.acquire(MODEL, Decimal("0.01"))
        await serve(two)
        assert _sample(registry, "threetears_key_pool_no_key_available_total", pool="one") == 3
        assert _sample(registry, "threetears_key_pool_no_key_available_total", pool="two") is None


class TestGauges:
    async def test_refresh_shows_each_keys_money_state_and_whether_it_is_active(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        pool = _pool(metrics, spec("a", "10"), spec("b", "20"))
        await serve(pool, amount="4")
        held = await pool.acquire(MODEL, Decimal("1.5"))
        await metrics.refresh(pool)
        assert _sample(registry, "threetears_key_pool_spent_usd", pool=POOL, key="a") == 4
        assert _sample(registry, "threetears_key_pool_reserved_usd", pool=POOL, key="a") == 1.5
        assert _sample(registry, "threetears_key_pool_remaining_usd", pool=POOL, key="a") == 4.5
        assert _sample(registry, "threetears_key_pool_budget_usd", pool=POOL, key="a") == 10
        assert _sample(registry, "threetears_key_pool_budget_usd", pool=POOL, key="b") == 20
        assert _sample(registry, "threetears_key_pool_remaining_usd", pool=POOL, key="b") == 20
        assert _sample(registry, "threetears_key_pool_key_active", pool=POOL, key="a") == 1
        assert _sample(registry, "threetears_key_pool_key_active", pool=POOL, key="b") == 0
        assert _sample(registry, "threetears_key_pool_key_state", pool=POOL, key="a", state="ok") == 1
        assert _sample(registry, "threetears_key_pool_key_state", pool=POOL, key="a", state="exhausted") == 0
        await held.settle(Decimal("1.5"))

    async def test_a_move_shows_on_the_next_refresh_and_not_before(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        pool = _pool(metrics, spec("a"), spec("b"))
        await metrics.refresh(pool)
        await fail_on(pool, exhausted())
        assert _sample(registry, "threetears_key_pool_key_state", pool=POOL, key="a", state="ok") == 1
        await metrics.refresh(pool)
        state = "threetears_key_pool_key_state"
        assert _sample(registry, state, pool=POOL, key="a", state="exhausted") == 1
        assert _sample(registry, state, pool=POOL, key="a", state="ok") == 0
        assert _sample(registry, "threetears_key_pool_key_active", pool=POOL, key="a") == 0
        assert _sample(registry, "threetears_key_pool_key_active", pool=POOL, key="b") == 1

    async def test_each_key_has_one_series_per_snapshot_state(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        pool = _pool(metrics, spec("a"))
        await metrics.refresh(pool)
        shown = {
            sample.labels["state"]
            for family in text_string_to_metric_families(generate_latest(registry).decode())
            if family.name == "threetears_key_pool_key_state"
            for sample in family.samples
        }
        assert shown == {str(state) for state in KeyState} - {"tried"}

    async def test_refresh_returns_the_snapshot_it_showed(self, metrics: KeyPoolMetrics) -> None:
        pool = _pool(metrics, spec("a"), spec("b"))
        await serve(pool, amount="2")
        assert await metrics.refresh(pool) == await pool.status()

    async def test_refresh_starts_the_pools_counters_at_zero(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        pool = _pool(metrics, spec("a"), spec("b"))
        await metrics.refresh(pool)
        for key in ("a", "b"):
            assert _sample(registry, "threetears_key_pool_room_refusals_total", pool=POOL, key=key) == 0
            assert _sample(registry, "threetears_key_pool_orphaned_reservations_total", pool=POOL, key=key) == 0
        assert _sample(registry, "threetears_key_pool_no_key_available_total", pool=POOL) == 0
        failover_reasons = ("exhausted", "revoked", "forbidden", "rate_limited", "tripped")
        for reason in failover_reasons:
            assert _sample(registry, "threetears_key_pool_failovers_total", pool=POOL, reason=reason) == 0
        for reason in {str(reason) for reason in MoveReason} - set(failover_reasons):
            # the pool's own decisions between calls are never a failover's reason, so get no series
            assert _sample(registry, "threetears_key_pool_failovers_total", pool=POOL, reason=reason) is None
        store_methods = {name for name in vars(KeyPoolStore) if not name.startswith("_")}
        assert store_methods  # the Protocol's own methods: every operation a failure is counted by
        for operation in store_methods:
            assert _sample(registry, "threetears_key_pool_store_failures_total", pool=POOL, operation=operation) == 0

    async def test_a_store_failure_raises_from_refresh_and_the_gauges_keep_the_last_snapshot(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        clock = ManualClock()
        store = FakeFlakyStore()
        pool = _pool(metrics, spec("a", "10"), store=store, clock=clock)
        await serve(pool, amount="3")
        await metrics.refresh(pool)
        store.down = True
        with pytest.raises(KeyPoolStoreError):
            await metrics.refresh(pool)
        assert _sample(registry, "threetears_key_pool_spent_usd", pool=POOL, key="a") == 3


class TestRegistries:
    async def test_two_exporters_on_one_registry_share_its_instruments(self, registry: CollectorRegistry) -> None:
        first, second = KeyPoolMetrics(registry), KeyPoolMetrics(registry)
        await fail_on(_pool(first, spec("a"), spec("b"), name="one"), exhausted())
        await fail_on(_pool(second, spec("a"), spec("b"), name="two"), exhausted())
        moves = "threetears_key_pool_moves_total"
        for name in ("one", "two"):
            assert _sample(registry, moves, pool=name, **{"from": "a", "to": "b", "reason": "exhausted"}) == 1

    async def test_each_new_registry_gets_its_own_instruments(self) -> None:
        # a registry's id can be reused once it is gone; each new one must still get instruments of its own
        for _ in range(20):
            registry = CollectorRegistry()
            await fail_on(_pool(KeyPoolMetrics(registry), spec("a"), spec("b")), exhausted())
            moves = "threetears_key_pool_moves_total"
            assert _sample(registry, moves, pool=POOL, **{"from": "a", "to": "b", "reason": "exhausted"}) == 1
            del registry


class TestSecrecy:
    """KP-B1: the exposition names keys by label only, and every label is bounded by configuration."""

    async def test_no_secret_and_only_bounded_labels_in_the_exposition(
        self, registry: CollectorRegistry, metrics: KeyPoolMetrics
    ) -> None:
        store = FakeFlakyStore()
        pool = _pool(metrics, spec("a", "1"), spec("b"), store=store)
        await metrics.refresh(pool)
        lease = await pool.acquire(MODEL, Decimal("2"))  # too big for a: a refusal for room, served on b
        await lease.settle(Decimal("2"))
        await fail_on(pool, exhausted())  # a is exhausted: a move to b
        await serve(pool, amount="0.5")
        await fail_on(pool, exhausted())  # b is exhausted too
        with pytest.raises(NoKeyAvailableError):
            await pool.acquire(MODEL, Decimal("0.01"))
        store.failing["snapshot"] = ConnectionError("unreachable")
        with pytest.raises(KeyPoolStoreError):
            await metrics.refresh(pool)
        store.failing.clear()
        await metrics.refresh(pool)

        text = generate_latest(registry).decode()
        for label in ("a", "b"):
            assert secret_for(label) not in text
        assert "sk-ant" not in text

        bounded = {
            "pool": {POOL},
            "key": {"a", "b"},
            "from": {"a", "b"},
            "to": {"a", "b"},
            "reason": {str(reason) for reason in MoveReason},
            "state": {str(state) for state in KeyState},
            "operation": {
                "reserve",
                "settle",
                "release",
                "sweep_expired",
                "snapshot",
                "read_record",
                "compare_and_set_record",
            },
        }
        seen: dict[str, set[str]] = {}
        for family in text_string_to_metric_families(text):
            if not family.name.startswith("threetears_key_pool_"):
                continue
            for sample in family.samples:
                for name, value in sample.labels.items():
                    seen.setdefault(name, set()).add(value)
        assert set(seen) <= set(bounded)
        for name, values in seen.items():
            assert values <= bounded[name], name
        assert seen["from"] and seen["reason"] and seen["operation"]
