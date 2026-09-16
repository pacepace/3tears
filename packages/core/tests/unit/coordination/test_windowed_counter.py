"""tests for :class:`WindowedCounter`: generic windowed attempt counter/rate-limiter.

The contract this pins (unchanged by the move from a KV bucket onto the coordination tables; only
the wiring in these fixtures changed):

- a fresh key's first attempt records count=1; repeated attempts within the window increment;
- distinct keys count independently, and distinct purposes over one table do too;
- the counter is SHARED, not per-process: two instances, and two replicas, see each other's
  recorded attempts;
- keys are hashed before storage (the raw identifier is never a stored key);
- `count`/`is_over_threshold` are read-only -- they never themselves record an attempt;
- an expired window resets the count back to 1 on the next attempt, not to count+1, judged on the
  counter's own clock;
- a row carries the expiry its window implies, so a closed window is absent at every tier;
- a non-positive ``window_seconds`` and an empty purpose are rejected at construction;
- fail-closed (default) propagates a storage failure, and exhausted compare-and-swap
  contention, which is this counter's expected shape under a burst; fail-open returns 0 for
  both instead;
- the count survives an L2 wipe when there is an L3 behind it.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest import mock

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination import WindowedCounter
from threetears.core.coordination.tables import CoordinationCountersCollection, coordination_collection
from threetears.core.exceptions import ConcurrentModificationError
from threetears.core.testing.kv import FakeNatsClient
from threetears.nats import KvError

_SCOPE = "counter-principal"


class _Nats(FakeNatsClient):
    """the shared collections bucket; no replica in these tests subscribes to invalidations."""

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        return None


class _Store:
    """an in-process L3 for the durable half."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    async def fetch_one(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        del table, conn
        row = self.rows.get((pk["purpose"], pk["key"]))
        return dict(row) if row is not None else None

    async def upsert(self, table: str, data: dict[str, Any], **kwargs: Any) -> int:
        del table, kwargs
        self.rows[(data["purpose"], data["key"])] = dict(data)
        return 1

    async def delete(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> None:
        del table, conn
        self.rows.pop((pk["purpose"], pk["key"]), None)

    async def scan(self, table: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        del table, filters
        return [dict(row) for row in self.rows.values()]

    async def execute(self, query: str, *params: Any, namespace: str | None = None) -> str:
        del query, params, namespace
        return "DELETE 0"


def _registry(nats: _Nats | None = None, store: _Store | None = None) -> CollectionRegistry:
    l1 = SQLiteBackend(db_name=f"counter_{uuid.uuid4().hex[:8]}")
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=l1,
        l2_client=nats,
        l3_pool=store,  # type: ignore[arg-type]
        kv_key_scope=_SCOPE if nats is not None else None,
    )
    return registry


def _counter(
    registry: CollectionRegistry, *, purpose: str = "throttle", window_seconds: int = 60, **kwargs: Any
) -> WindowedCounter:
    return WindowedCounter(registry, purpose=purpose, window_seconds=window_seconds, **kwargs)


def _counters(registry: CollectionRegistry) -> CoordinationCountersCollection:
    """the one counters collection on this registry, typed for the assertions below."""
    return coordination_collection(registry, CoordinationCountersCollection)  # type: ignore[return-value]


class _FailingNats(FakeNatsClient):
    """a client whose bucket refuses every operation, the way an unreachable broker does."""

    async def kv_bucket(self, **kwargs: Any) -> Any:
        del kwargs
        raise KvError("kv down")

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        return None


def _now() -> float:
    """a clock base near the wall clock.

    A row's ``expires_at`` is read against the wall clock by the tiers below, so a test clock has
    to sit near it; time travel to 1970 would make every row read as long expired, which is a
    property of the tiers rather than of the counter.

    :return: epoch seconds
    :rtype: float
    """
    return datetime.now(UTC).timestamp()


class TestWindowedCounter:
    @pytest.mark.asyncio
    async def test_first_attempt_records_count_one(self) -> None:
        counter = _counter(_registry(_Nats()))
        assert await counter.record_attempt("k") == 1

    @pytest.mark.asyncio
    async def test_repeated_attempts_increment_within_window(self) -> None:
        counter = _counter(_registry(_Nats()))
        assert await counter.record_attempt("k") == 1
        assert await counter.record_attempt("k") == 2
        assert await counter.record_attempt("k") == 3

    @pytest.mark.asyncio
    async def test_distinct_keys_count_independently(self) -> None:
        counter = _counter(_registry(_Nats()))
        assert await counter.record_attempt("a") == 1
        assert await counter.record_attempt("b") == 1
        assert await counter.record_attempt("a") == 2

    @pytest.mark.asyncio
    async def test_distinct_purposes_count_independently_over_one_table(self) -> None:
        # what separate buckets used to do: an edge throttle and a core throttle over the "same"
        # logical key must not share a count.
        registry = _registry(_Nats())
        edge = _counter(registry, purpose="edge_login")
        core = _counter(registry, purpose="core_login")
        assert await edge.record_attempt("k") == 1
        assert await core.record_attempt("k") == 1
        assert await edge.record_attempt("k") == 2

    @pytest.mark.asyncio
    async def test_shared_across_instances_over_the_same_purpose(self) -> None:
        registry = _registry(_Nats())
        first = _counter(registry, purpose="shared")
        second = _counter(registry, purpose="shared")
        assert await first.record_attempt("k") == 1
        assert await second.record_attempt("k") == 2

    @pytest.mark.asyncio
    async def test_shared_across_replicas(self) -> None:
        nats = _Nats()
        store = _Store()
        first = _counter(_registry(nats, store), purpose="shared")
        second = _counter(_registry(nats, store), purpose="shared")
        assert await first.record_attempt("k") == 1
        assert await second.record_attempt("k") == 2

    @pytest.mark.asyncio
    async def test_read_only_methods_do_not_record(self) -> None:
        counter = _counter(_registry(_Nats()))
        assert await counter.count("k") == 0
        assert await counter.is_over_threshold("k", threshold=1) is False
        await counter.record_attempt("k")
        assert await counter.count("k") == 1
        assert await counter.count("k") == 1  # still 1 -- count() never increments
        assert await counter.is_over_threshold("k", threshold=1) is True
        assert await counter.is_over_threshold("k", threshold=2) is False

    @pytest.mark.asyncio
    async def test_expired_window_resets_to_one(self) -> None:
        clock = [_now()]
        counter = _counter(_registry(_Nats()), clock=lambda: clock[0])
        assert await counter.record_attempt("k") == 1
        assert await counter.record_attempt("k") == 2
        clock[0] += 61.0  # past the 60s window, on this counter's own clock
        assert await counter.record_attempt("k") == 1
        assert await counter.count("k") == 1

    @pytest.mark.asyncio
    async def test_a_window_still_open_is_not_reset(self) -> None:
        clock = [_now()]
        counter = _counter(_registry(_Nats()), clock=lambda: clock[0])
        assert await counter.record_attempt("k") == 1
        clock[0] += 59.0
        assert await counter.record_attempt("k") == 2

    @pytest.mark.asyncio
    async def test_the_window_is_anchored_at_the_first_attempt_not_refreshed(self) -> None:
        # a steady stream of attempts must not extend the window the caller asked for.
        clock = [_now()]
        counter = _counter(_registry(_Nats()), clock=lambda: clock[0])
        for _ in range(3):
            await counter.record_attempt("k")
            clock[0] += 25.0  # 0s, 25s, 50s: all inside one 60s window
        assert await counter.record_attempt("k") == 1, "the window was extended by its own attempts"

    @pytest.mark.asyncio
    async def test_a_row_carries_the_expiry_its_window_implies(self) -> None:
        store = _Store()
        registry = _registry(_Nats(), store)
        counter = _counter(registry, window_seconds=90)
        await counter.record_attempt("k")
        await _counters(registry).aclose()  # write-behind: flush first
        row = next(iter(store.rows.values()))
        assert row["expires_at"] - row["window_start"] == timedelta(seconds=90)

    @pytest.mark.asyncio
    async def test_hashed_keys_not_stored_raw(self) -> None:
        store = _Store()
        counter = _counter(_registry(_Nats(), store))
        await counter.record_attempt("super-secret-identifier")
        assert all("super-secret-identifier" not in key for key in store.rows)
        assert all("super-secret-identifier" not in json.dumps(row, default=str) for row in store.rows.values())

    @pytest.mark.asyncio
    async def test_the_count_survives_an_l2_wipe_once_flushed(self) -> None:
        # the write-behind trade, stated exactly: what has reached L3 survives a wipe, and what is
        # still buffered does not. That is why a counter accepts it and a revocation does not.
        nats = _Nats()
        store = _Store()
        registry = _registry(nats, store)
        counter = _counter(registry)
        await counter.record_attempt("k")
        await counter.record_attempt("k")
        await _counters(registry).aclose()  # the flush interval, forced
        (await nats.kv_bucket(name="collections")).wipe()
        assert await counter.record_attempt("k") == 3, "a broker wipe reset the throttle"

    @pytest.mark.asyncio
    async def test_an_unflushed_count_is_what_a_wipe_costs(self) -> None:
        nats = _Nats()
        store = _Store()
        counter = _counter(_registry(nats, store))
        await counter.record_attempt("k")
        await counter.record_attempt("k")
        (await nats.kv_bucket(name="collections")).wipe()
        assert await counter.record_attempt("k") == 1, "an unflushed increment was reported as durable"

    @pytest.mark.asyncio
    async def test_storage_failure_propagates_fail_closed_by_default(self) -> None:
        counter = _counter(_registry(_FailingNats()))  # type: ignore[arg-type]
        with pytest.raises(KvError):
            await counter.record_attempt("x")

    @pytest.mark.asyncio
    async def test_storage_failure_swallowed_when_fail_open(self) -> None:
        counter = _counter(_registry(_FailingNats()), fail_open=True)  # type: ignore[arg-type]
        assert await counter.record_attempt("x") == 0
        assert await counter.count("x") == 0
        assert await counter.is_over_threshold("x", threshold=1) is False

    @pytest.mark.asyncio
    async def test_exhausted_contention_propagates_fail_closed_by_default(self) -> None:
        counter = _counter(_registry(_Nats()))
        with (
            mock.patch.object(
                CoordinationCountersCollection,
                "l2_cas_mutate",
                side_effect=ConcurrentModificationError("coordination_counters", ("throttle", "x"), None),
            ),
            pytest.raises(ConcurrentModificationError),
        ):
            await counter.record_attempt("x")

    @pytest.mark.asyncio
    async def test_exhausted_contention_fails_open_when_the_posture_says_so(self) -> None:
        """A burst against ONE key is this counter's expected shape, not an anomaly.

        `l2_cas_mutate` raises `ConcurrentModificationError` when its budget runs out, and that
        budget is raised to 30 precisely because a credential-stuffing run contends on one key.
        `ConcurrentModificationError` is not in `STORAGE_FAILURES`, so a `fail_open` counter used
        to propagate it -- turning the attack into a 500 on the request path of the control that
        exists to answer it. identity-edge's route throttles are the ones that would have worn it.
        """
        counter = _counter(_registry(_Nats()), fail_open=True)
        with mock.patch.object(
            CoordinationCountersCollection,
            "l2_cas_mutate",
            side_effect=ConcurrentModificationError("coordination_counters", ("throttle", "x"), None),
        ):
            assert await counter.record_attempt("x") == 0

    @pytest.mark.asyncio
    async def test_clear_resets_the_counter_and_never_raises(self) -> None:
        counter = _counter(_registry(_Nats()))
        await counter.record_attempt("k")
        await counter.clear("k")
        assert await counter.count("k") == 0
        failing = _counter(_registry(_FailingNats()))  # type: ignore[arg-type]
        await failing.clear("k")  # fail-open regardless of posture: never turns a login into an error

    @pytest.mark.asyncio
    async def test_concurrent_attempts_on_same_key_all_counted(self) -> None:
        counter = _counter(_registry(_Nats()), purpose="race")
        results = await asyncio.gather(*[counter.record_attempt("dup") for _ in range(8)])
        assert sorted(results) == list(range(1, 9))

    @pytest.mark.asyncio
    async def test_every_counter_over_the_table_shares_one_collection(self) -> None:
        # identity-edge builds seven of these in one process.
        registry = _registry(_Nats())
        counters = [_counter(registry, purpose=f"route-{n}") for n in range(7)]
        for counter in counters:
            await counter.record_attempt("k")
        assert registry.get_collection("coordination_counters") is coordination_collection(
            registry, CoordinationCountersCollection
        )

    @pytest.mark.asyncio
    async def test_a_write_behind_counter_starts_its_own_flusher(self) -> None:
        registry = _registry(_Nats(), _Store())
        counter = _counter(registry)
        await counter.record_attempt("k")
        collection = _counters(registry)
        assert collection._flusher is not None, "nothing would ever drain the write buffer"  # noqa: SLF001
        await collection.aclose()

    def test_non_positive_window_rejected(self) -> None:
        registry = _registry(_Nats())
        with pytest.raises(ValueError, match="window_seconds"):
            _counter(registry, window_seconds=0)
        with pytest.raises(ValueError, match="window_seconds"):
            _counter(registry, window_seconds=-5)

    def test_empty_purpose_rejected(self) -> None:
        with pytest.raises(ValueError, match="purpose"):
            _counter(_registry(_Nats()), purpose="  ")

    def test_fail_open_property_reflects_constructor_arg(self) -> None:
        registry = _registry(_Nats())
        assert _counter(registry).fail_open is False
        assert _counter(registry, fail_open=True).fail_open is True

    def test_the_clock_is_readable_so_a_retry_after_uses_one_clock(self) -> None:
        clock = [123.0]
        counter = _counter(_registry(_Nats()), clock=lambda: clock[0])
        assert counter.clock() == 123.0

    @pytest.mark.asyncio
    async def test_state_reports_the_window_start_for_a_truthful_retry_after(self) -> None:
        opened = _now()
        clock = [opened]
        counter = _counter(_registry(_Nats()), clock=lambda: clock[0])
        await counter.record_attempt("k")
        clock[0] += 10.0
        state = await counter.state("k")
        assert state is not None
        assert state.count == 1
        # the window opened 10s ago, so a caller reports 50s of a 60s window left, not 60.
        assert state.window_start == pytest.approx(opened, abs=1.0)
        assert clock[0] - state.window_start == pytest.approx(10.0, abs=1.0)
