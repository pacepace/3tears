"""Unit tests for per-``kind`` dispatch routing in the tick engine.

dsj-task-01. Before per-kind routing existed the pump took ONE callback,
the due-row scan was kind-blind, and whichever dispatcher happened to be
wired absorbed every row regardless of its ``kind``. On a Hub with
billing export enabled that meant a dataset row would never reach a
dataset handler AND would fire an unscheduled billing export that
advanced the billing watermark, recorded as a success. Nothing would
look wrong. That is the defect these tests pin shut.

Cases:

- a row of kind A reaches ONLY kind A's handler.
- two routed kinds each reach their own handler, never the other's.
- a row of an UNROUTED kind is inert: no handler runs, the schedule is
  NOT claimed (the occurrence stays recoverable), and the fire fails
  with the named ``unrouted_kind`` reason plus a failure metric.
- the routing table is consulted by exact key -- decoy ``'*'`` / ``''``
  entries prove a future glob-or-first-handler default would fail here.
- the due-row scan is handed exactly the routed kinds, and the default
  store pushes that filter into SQL.
- the billing-correctness assertion: a dataset row on a pump routing
  ``billing_export`` does not fire a billing export.
- a pump constructed with no routes, or with a bare callable (the
  pre-dsj-task-01 signature), fails loudly at the call site.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scheduled_jobs import tick as tick_mod
from threetears.scheduled_jobs.collections import ScheduledJobCollection
from threetears.scheduled_jobs.protocols import DueSchedule, FireStore, ScheduleStore
from threetears.scheduled_jobs.tick import UNROUTED_KIND_REASON
from threetears.scheduled_jobs.types import JobFireResult, JobTrigger


_BILLING_KIND = "billing_export"
_DATASET_KIND = "dataset_build"


def _now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class _FakeDueSchedule(DueSchedule):
    """A plain due-schedule row carrying an arbitrary ``kind``.

    # parity-with: threetears.scheduled_jobs.protocols.DueSchedule
    """

    def __init__(self, *, kind: str, job_id: UUID | None = None) -> None:
        self._kind = kind
        self._job_id = job_id or uuid4()
        self._partition_key = uuid4()

    @property
    def partition_key(self) -> UUID:
        return self._partition_key

    @property
    def job_id(self) -> UUID:
        return self._job_id

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def payload(self) -> dict[str, Any]:
        return {"k": self._kind}

    @property
    def schedule_type(self) -> str:
        return "interval"

    @property
    def schedule_config(self) -> dict[str, Any]:
        return {"seconds": 60}

    @property
    def missed_fire_policy(self) -> str:
        return "coalesce"

    @property
    def next_fire_at(self) -> datetime | None:
        return _now() - timedelta(seconds=5)

    @property
    def last_fired_at(self) -> datetime | None:
        return None

    @property
    def name(self) -> str | None:
        return f"{self._kind}-job"


class _FakeScheduleStore(ScheduleStore):
    """Store that HONOURS the ``kinds`` filter, as the real scan does.

    # parity-with: threetears.scheduled_jobs.protocols.ScheduleStore
    """

    def __init__(self, due: list[DueSchedule]) -> None:
        self._due = due
        self.claims: list[UUID] = []
        self.scan_calls: list[dict[str, Any]] = []

    async def list_due_for_tick(
        self,
        now: datetime,
        *,
        kinds: Sequence[str],
        limit: int = 200,
    ) -> list[DueSchedule]:
        self.scan_calls.append({"now": now, "kinds": tuple(kinds), "limit": limit})
        return [row for row in self._due if row.kind in set(kinds)]

    async def claim_and_reschedule(
        self,
        *,
        partition_key: UUID,
        job_id: UUID,
        expected_next_fire: datetime,
        computed_next_fire: datetime | None,
        new_status: str,
        now: datetime,
    ) -> bool:
        self.claims.append(job_id)
        return True


class _LeakyScheduleStore(_FakeScheduleStore):
    """Store that IGNORES the ``kinds`` filter -- models a store-contract bug.

    The engine's routing table is the second line of defence behind the
    SQL filter; this store is how the tests reach it.

    # parity-with: threetears.scheduled_jobs.protocols.ScheduleStore
    """

    async def list_due_for_tick(
        self,
        now: datetime,
        *,
        kinds: Sequence[str],
        limit: int = 200,
    ) -> list[DueSchedule]:
        self.scan_calls.append({"now": now, "kinds": tuple(kinds), "limit": limit})
        return list(self._due)


class _FakeFireStore(FireStore):
    """Records every fire-store call for assertions.

    # parity-with: threetears.scheduled_jobs.protocols.FireStore
    """

    def __init__(self) -> None:
        self.created: list[UUID] = []
        self.succeeded: list[UUID] = []
        self.failed: list[dict[str, Any]] = []
        self.reap_calls: list[dict[str, Any]] = []

    async def create_dispatching(
        self,
        *,
        fire_id: UUID,
        job_id: UUID,
        partition_key: UUID,
        scheduled_fire_at: datetime,
        actual_fired_at: datetime,
    ) -> None:
        self.created.append(job_id)

    async def finalize_success(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        status: str = "succeeded",
        output: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> None:
        self.succeeded.append(fire_id)

    async def finalize_failed(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        error: str,
        latency_ms: int | None = None,
    ) -> None:
        self.failed.append({"fire_id": fire_id, "error": error})

    async def reap_stale_dispatching(
        self,
        now: datetime,
        *,
        older_than: timedelta,
        kinds: Sequence[str],
    ) -> int:
        self.reap_calls.append({"older_than": older_than, "kinds": tuple(kinds)})
        return 0


class _RecordingEmitter:
    """Captures the failure reasons the tick emits."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def observe_tick_duration(self, _seconds: float) -> None: ...

    def observe_drift(self, _seconds: float) -> None: ...

    def inc_fire(self, **_kwargs: Any) -> None: ...

    def inc_failure(self, *, reason: str) -> None:
        self.failures.append(reason)


class _CtxHealthy:
    """Async context manager that acquires cleanly and yields the body."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: Any) -> bool:
        return False


class _RecordingPool:
    """asyncpg-pool-shaped recorder returning canned rows.

    The raw asyncpg pool has no production protocol (``l3_pool`` is typed
    ``Any`` by design), so there is nothing to declare parity against.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((sql, args))
        return []


def _patch_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the cross-pod lock with a healthy no-op context manager."""

    def _factory(_client: Any, _key: str, **_kwargs: Any) -> Any:
        return _CtxHealthy()

    monkeypatch.setattr("threetears.nats.nats_distributed_lock", _factory)


def _recorder(monkeypatch: pytest.MonkeyPatch) -> _RecordingEmitter:
    """Install and return a metrics recorder for the tick module."""
    emitter = _RecordingEmitter()
    monkeypatch.setattr(tick_mod, "get_scheduled_jobs_emitter", lambda *_a, **_k: emitter)
    return emitter


def _handler(seen: list[str]) -> Any:
    """Build a dispatch callback appending each trigger's kind to ``seen``."""

    async def _callback(trigger: JobTrigger, _fire_id: UUID) -> JobFireResult:
        seen.append(trigger.kind)
        return JobFireResult(status="succeeded")

    return _callback


class TestRoutesByKind:
    """A routed row reaches its own handler and no other."""

    async def test_row_of_kind_a_reaches_only_kind_a_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch)
        seen_a: list[str] = []
        seen_b: list[str] = []
        store = _FakeScheduleStore([_FakeDueSchedule(kind="kind_a")])
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(
            store,
            fires,
            {"kind_a": _handler(seen_a), "kind_b": _handler(seen_b)},
            nats_client=object(),
        )

        assert seen_a == ["kind_a"]
        assert seen_b == []

    async def test_two_kinds_reach_their_own_handlers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch)
        seen_a: list[str] = []
        seen_b: list[str] = []
        store = _FakeScheduleStore([_FakeDueSchedule(kind="kind_a"), _FakeDueSchedule(kind="kind_b")])
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(
            store,
            fires,
            {"kind_a": _handler(seen_a), "kind_b": _handler(seen_b)},
            nats_client=object(),
        )

        assert seen_a == ["kind_a"]
        assert seen_b == ["kind_b"]
        assert len(fires.created) == 2


class TestUnroutedKindIsInert:
    """An unrouted kind is neither misdelivered nor silently absorbed."""

    async def test_unrouted_kind_reaches_no_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The decoy ``'*'`` / ``''`` routes make a future fall-through default fail here.

        A ``routes.get(kind, routes['*'])`` glob default, or a
        ``next(iter(routes.values()))`` first-handler default, would put
        the unrouted row into one of these lists.
        """
        _patch_lock(monkeypatch)
        _recorder(monkeypatch)
        seen_a: list[str] = []
        seen_glob: list[str] = []
        seen_empty: list[str] = []
        store = _LeakyScheduleStore([_FakeDueSchedule(kind="kind_z")])
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(
            store,
            fires,
            {"kind_a": _handler(seen_a), "*": _handler(seen_glob), "": _handler(seen_empty)},
            nats_client=object(),
        )

        assert seen_a == []
        assert seen_glob == []
        assert seen_empty == []

    async def test_unrouted_kind_does_not_claim_the_schedule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Refusing WITHOUT claiming keeps the occurrence recoverable.

        A claim would advance ``next_fire_at`` and destroy the fire, so a
        one-shot build whose handler was merely not registered yet would
        be lost forever.
        """
        _patch_lock(monkeypatch)
        _recorder(monkeypatch)
        store = _LeakyScheduleStore([_FakeDueSchedule(kind="kind_z")])
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(store, fires, {"kind_a": _handler([])}, nats_client=object())

        assert store.claims == []
        assert fires.created == []
        assert fires.succeeded == []

    async def test_unrouted_kind_emits_named_failure_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch)
        emitter = _recorder(monkeypatch)
        store = _LeakyScheduleStore([_FakeDueSchedule(kind="kind_z")])
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(store, fires, {"kind_a": _handler([])}, nats_client=object())

        assert UNROUTED_KIND_REASON == "unrouted_kind"
        assert emitter.failures == [UNROUTED_KIND_REASON]

    async def test_unrouted_row_does_not_poison_the_routed_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch)
        _recorder(monkeypatch)
        seen_a: list[str] = []
        store = _LeakyScheduleStore([_FakeDueSchedule(kind="kind_z"), _FakeDueSchedule(kind="kind_a")])
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(store, fires, {"kind_a": _handler(seen_a)}, nats_client=object())

        assert seen_a == ["kind_a"]
        assert len(fires.created) == 1


class TestDueScanFiltersOnKind:
    """The kind filter is pushed into the scan, not applied after it."""

    async def test_scan_receives_exactly_the_routed_kinds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch)
        store = _FakeScheduleStore([])
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(
            store,
            fires,
            {"kind_b": _handler([]), "kind_a": _handler([])},
            nats_client=object(),
        )

        assert len(store.scan_calls) == 1
        assert store.scan_calls[0]["kinds"] == ("kind_a", "kind_b")

    async def test_due_row_query_constrains_kind_in_sql(self) -> None:
        """A refactor cannot quietly move the kind filter back into Python."""
        pool = _RecordingPool()
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        collection = ScheduledJobCollection(registry=registry, config=cfg, nats_client=None)

        await collection.list_due_for_tick(now=_now(), kinds=(_BILLING_KIND,))

        sql, args = pool.calls[-1]
        assert "kind = ANY($2)" in sql
        assert args[1] == [_BILLING_KIND]

    async def test_no_routed_kinds_scans_nothing(self) -> None:
        """An empty kind list must never widen to "every kind"."""
        pool = _RecordingPool()
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        collection = ScheduledJobCollection(registry=registry, config=cfg, nats_client=None)

        rows = await collection.list_due_for_tick(now=_now(), kinds=())

        assert rows == []
        assert pool.calls == []


class TestBillingCorrectness:
    """The defect this shard exists for: a dataset row must not bill."""

    async def test_dataset_row_does_not_fire_billing_export(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Hub with billing export enabled, and a dataset row in the table.

        The billing pump routes only ``billing_export``. The dataset row
        must reach neither the billing handler nor any claim -- firing it
        would advance the billing watermark and record the bogus export
        as a success.
        """
        _patch_lock(monkeypatch)
        _recorder(monkeypatch)
        billed: list[str] = []
        dataset_row = _FakeDueSchedule(kind=_DATASET_KIND)
        billing_row = _FakeDueSchedule(kind=_BILLING_KIND)
        store = _FakeScheduleStore([dataset_row, billing_row])
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(
            store,
            fires,
            {_BILLING_KIND: _handler(billed)},
            nats_client=object(),
        )

        assert billed == [_BILLING_KIND]
        assert store.claims == [billing_row.job_id]
        assert dataset_row.job_id not in fires.created

    async def test_dataset_row_leaked_by_the_scan_still_does_not_bill(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Defence in depth: even a kind-blind scan cannot misdeliver."""
        _patch_lock(monkeypatch)
        emitter = _recorder(monkeypatch)
        billed: list[str] = []
        store = _LeakyScheduleStore([_FakeDueSchedule(kind=_DATASET_KIND)])
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(
            store,
            fires,
            {_BILLING_KIND: _handler(billed)},
            nats_client=object(),
        )

        assert billed == []
        assert store.claims == []
        assert emitter.failures == [UNROUTED_KIND_REASON]


class TestRoutingTableIsValidated:
    """A mis-wired pump fails at the call site, never at runtime silently."""

    async def test_empty_routes_raises(self) -> None:
        store = _FakeScheduleStore([])
        fires = _FakeFireStore()
        with pytest.raises(ValueError, match="dispatch_routes"):
            await tick_mod.scheduled_tick_job(store, fires, {}, nats_client=object())

    async def test_bare_callable_raises(self) -> None:
        """The pre-dsj-task-01 single-callback signature is gone, not shimmed."""
        store = _FakeScheduleStore([])
        fires = _FakeFireStore()

        async def _legacy(_trigger: JobTrigger, _fire_id: UUID) -> JobFireResult:
            return JobFireResult()

        with pytest.raises(TypeError, match="dispatch_routes"):
            await tick_mod.scheduled_tick_job(store, fires, _legacy, nats_client=object())  # type: ignore[arg-type]

    async def test_routes_mapping_is_accepted_by_protocol_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Any ``Mapping`` works -- the engine never mutates the table."""
        _patch_lock(monkeypatch)
        seen: list[str] = []
        routes: Mapping[str, Any] = dict({"kind_a": _handler(seen)})
        store = _FakeScheduleStore([_FakeDueSchedule(kind="kind_a")])
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(store, fires, routes, nats_client=object())

        assert seen == ["kind_a"]
