"""Unit tests for per-``kind`` stale-dispatch reap thresholds.

dsj-task-01. ``DEFAULT_DISPATCH_REAP_AFTER_SECONDS = 900`` used to be
ONE value per pump, so a kind whose work legitimately runs for hours (a
dataset build carries a 4h statement ceiling) was falsely reaped and
recorded as a failure while it was still running.

The threshold is now a per-kind lookup with the existing constant as the
fallback, so a kind that configures nothing behaves exactly as before.
The tick groups its routed kinds by resolved threshold and issues ONE
sweep per distinct threshold -- two kinds sharing the default are one
query, not two.

**This alone does not fix false reaping.** The age is still measured
from dispatch start rather than last activity. The pair is completed by
``dsh-task-09``'s progress-conditioned renewal keyed on
``date_last_progress`` (``dsh-task-04b``, DSH-04B-02); a larger constant
by itself only moves the cliff. Neither half is sufficient alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scheduled_jobs import tick as tick_mod
from threetears.scheduled_jobs.collections import REAPED_DISPATCH_ERROR, JobFireCollection
from threetears.scheduled_jobs.config import (
    DEFAULT_DISPATCH_REAP_AFTER_SECONDS,
    DEFAULT_TICK_DUE_LIMIT,
    DEFAULT_TICK_LOCK_KEY,
    JobConfig,
    reap_after_seconds_for_kind,
)
from threetears.scheduled_jobs.protocols import DueSchedule, FireStore, ScheduleStore
from threetears.scheduled_jobs.types import JobFireResult, JobTrigger


_BILLING_KIND = "billing_export"
_DATASET_KIND = "dataset_build"

# A dataset build can legitimately run for hours; four hours mirrors the
# statement ceiling the executor enforces.
_DATASET_REAP_SECONDS = 4 * 60 * 60


def _now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class _PerKindConfig:
    """:class:`JobConfig` carrying a per-kind reap-threshold override.

    Structurally satisfies the pure-read config Protocol; every knob
    other than the by-kind map returns the platform baseline.
    """

    def __init__(self, by_kind: Mapping[str, int]) -> None:
        self._by_kind = MappingProxyType(dict(by_kind))

    @property
    def tick_lock_key(self) -> str:
        return DEFAULT_TICK_LOCK_KEY

    @property
    def tick_due_limit(self) -> int:
        return DEFAULT_TICK_DUE_LIMIT

    @property
    def dispatch_reap_after_seconds(self) -> int:
        return DEFAULT_DISPATCH_REAP_AFTER_SECONDS

    @property
    def dispatch_reap_after_seconds_by_kind(self) -> Mapping[str, int]:
        return self._by_kind


class _FakeScheduleStore(ScheduleStore):
    """Due-scan that returns nothing -- these tests only exercise the sweep.

    # parity-with: threetears.scheduled_jobs.protocols.ScheduleStore
    """

    async def list_due_for_tick(
        self,
        now: datetime,
        *,
        kinds: Sequence[str],
        limit: int = 200,
    ) -> list[DueSchedule]:
        return []

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
        return False


class _FakeFireStore(FireStore):
    """Records each reap sweep's kind group and age threshold.

    # parity-with: threetears.scheduled_jobs.protocols.FireStore
    """

    def __init__(self, *, reap_counts: Mapping[str, int] | None = None) -> None:
        self.reap_calls: list[dict[str, Any]] = []
        self._reap_counts = reap_counts or {}

    async def create_dispatching(
        self,
        *,
        fire_id: UUID,
        job_id: UUID,
        partition_key: UUID,
        scheduled_fire_at: datetime,
        actual_fired_at: datetime,
    ) -> None:
        return None

    async def finalize_success(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        status: str = "succeeded",
        output: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> None:
        return None

    async def finalize_failed(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        error: str,
        latency_ms: int | None = None,
    ) -> None:
        return None

    async def reap_stale_dispatching(
        self,
        now: datetime,
        *,
        older_than: timedelta,
        kinds: Sequence[str],
    ) -> int:
        group = tuple(kinds)
        self.reap_calls.append({"now": now, "older_than": older_than, "kinds": group})
        return sum(self._reap_counts.get(kind, 0) for kind in group)


class _RaisingFireStore(_FakeFireStore):
    """Fails the sweep for one kind group only.

    # parity-with: threetears.scheduled_jobs.protocols.FireStore
    """

    def __init__(self, *, failing_kind: str) -> None:
        super().__init__()
        self._failing_kind = failing_kind

    async def reap_stale_dispatching(
        self,
        now: datetime,
        *,
        older_than: timedelta,
        kinds: Sequence[str],
    ) -> int:
        if self._failing_kind in kinds:
            raise RuntimeError("reaper db hiccup")
        return await super().reap_stale_dispatching(now, older_than=older_than, kinds=kinds)


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

    def __init__(self, *, fetch_rows: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._fetch_rows = fetch_rows or []

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((sql, args))
        return self._fetch_rows


def _patch_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the cross-pod lock with a healthy no-op context manager."""

    def _factory(_client: Any, _key: str, **_kwargs: Any) -> Any:
        return _CtxHealthy()

    monkeypatch.setattr("threetears.nats.nats_distributed_lock", _factory)


async def _noop(_trigger: JobTrigger, _fire_id: UUID) -> JobFireResult:
    """Dispatch callback that is never reached by these tests."""
    return JobFireResult()


def _sweeps(fires: _FakeFireStore) -> dict[tuple[str, ...], timedelta]:
    """Project the recorded sweeps to ``kind group -> age threshold``."""
    return {call["kinds"]: call["older_than"] for call in fires.reap_calls}


class TestThresholdLookup:
    """The lookup falls back to the platform constant."""

    def test_unconfigured_kind_gets_the_default(self) -> None:
        config: JobConfig = _PerKindConfig({})
        assert reap_after_seconds_for_kind(config, _BILLING_KIND) == DEFAULT_DISPATCH_REAP_AFTER_SECONDS

    def test_configured_kind_gets_its_override(self) -> None:
        config: JobConfig = _PerKindConfig({_DATASET_KIND: _DATASET_REAP_SECONDS})
        assert reap_after_seconds_for_kind(config, _DATASET_KIND) == _DATASET_REAP_SECONDS

    def test_override_does_not_leak_to_other_kinds(self) -> None:
        config: JobConfig = _PerKindConfig({_DATASET_KIND: _DATASET_REAP_SECONDS})
        assert reap_after_seconds_for_kind(config, _BILLING_KIND) == DEFAULT_DISPATCH_REAP_AFTER_SECONDS


class TestTwoKindsReapOnDifferentThresholds:
    """The headline verification: two kinds, two thresholds, one tick."""

    async def test_each_kind_sweeps_at_its_own_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch)
        fires = _FakeFireStore()
        config: JobConfig = _PerKindConfig({_DATASET_KIND: _DATASET_REAP_SECONDS})

        await tick_mod.scheduled_tick_job(
            _FakeScheduleStore(),
            fires,
            {_BILLING_KIND: _noop, _DATASET_KIND: _noop},
            nats_client=object(),
            config=config,
        )

        assert _sweeps(fires) == {
            (_BILLING_KIND,): timedelta(seconds=DEFAULT_DISPATCH_REAP_AFTER_SECONDS),
            (_DATASET_KIND,): timedelta(seconds=_DATASET_REAP_SECONDS),
        }

    async def test_kinds_sharing_a_threshold_sweep_together(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two unconfigured kinds are one query, not two."""
        _patch_lock(monkeypatch)
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(
            _FakeScheduleStore(),
            fires,
            {_BILLING_KIND: _noop, _DATASET_KIND: _noop},
            nats_client=object(),
            config=_PerKindConfig({}),
        )

        assert _sweeps(fires) == {
            (_BILLING_KIND, _DATASET_KIND): timedelta(seconds=DEFAULT_DISPATCH_REAP_AFTER_SECONDS),
        }

    async def test_sweep_never_widens_beyond_the_routed_kinds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pump must not reap another pump's in-flight fires."""
        _patch_lock(monkeypatch)
        fires = _FakeFireStore()

        await tick_mod.scheduled_tick_job(
            _FakeScheduleStore(),
            fires,
            {_BILLING_KIND: _noop},
            nats_client=object(),
            config=_PerKindConfig({_DATASET_KIND: _DATASET_REAP_SECONDS}),
        )

        assert _sweeps(fires) == {(_BILLING_KIND,): timedelta(seconds=DEFAULT_DISPATCH_REAP_AFTER_SECONDS)}

    async def test_reaped_rows_count_as_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch)
        emitter = _RecordingEmitter()
        monkeypatch.setattr(tick_mod, "get_scheduled_jobs_emitter", lambda *_a, **_k: emitter)
        fires = _FakeFireStore(reap_counts={_BILLING_KIND: 2, _DATASET_KIND: 1})

        await tick_mod.scheduled_tick_job(
            _FakeScheduleStore(),
            fires,
            {_BILLING_KIND: _noop, _DATASET_KIND: _noop},
            nats_client=object(),
            config=_PerKindConfig({_DATASET_KIND: _DATASET_REAP_SECONDS}),
        )

        assert emitter.failures == ["reaped", "reaped", "reaped"]

    async def test_one_failing_sweep_does_not_block_the_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Per-group isolation: a hiccup on one kind must not skip the rest."""
        _patch_lock(monkeypatch)
        fires = _RaisingFireStore(failing_kind=_BILLING_KIND)

        await tick_mod.scheduled_tick_job(
            _FakeScheduleStore(),
            fires,
            {_BILLING_KIND: _noop, _DATASET_KIND: _noop},
            nats_client=object(),
            config=_PerKindConfig({_DATASET_KIND: _DATASET_REAP_SECONDS}),
        )

        assert _sweeps(fires) == {(_DATASET_KIND,): timedelta(seconds=_DATASET_REAP_SECONDS)}


class TestStoreSweepIsKindScoped:
    """The default store pushes the kind scope into the sweep's SQL."""

    async def test_sweep_sql_constrains_kind(self) -> None:
        pool = _RecordingPool(fetch_rows=[{"partition_key": uuid4(), "fire_id": uuid4()}])
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        collection = JobFireCollection(registry=registry, config=cfg, nats_client=None)
        now = _now()

        reaped = await collection.reap_stale_dispatching(
            now,
            older_than=timedelta(seconds=_DATASET_REAP_SECONDS),
            kinds=(_DATASET_KIND,),
        )

        assert reaped == 1
        sql, args = pool.calls[-1]
        assert "kind = ANY($3)" in sql
        assert args[0] == REAPED_DISPATCH_ERROR
        assert args[1] == now - timedelta(seconds=_DATASET_REAP_SECONDS)
        assert args[2] == [_DATASET_KIND]

    async def test_empty_kind_group_sweeps_nothing(self) -> None:
        """An empty group must never widen to "every kind"."""
        pool = _RecordingPool()
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        collection = JobFireCollection(registry=registry, config=cfg, nats_client=None)

        reaped = await collection.reap_stale_dispatching(_now(), older_than=timedelta(seconds=900), kinds=())

        assert reaped == 0
        assert pool.calls == []
