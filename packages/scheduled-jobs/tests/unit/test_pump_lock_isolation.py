"""Unit tests proving two tick pumps do not contend on one cross-pod lock.

dsj-task-01, DSJ-01-05. The lock-key seam already existed --
:attr:`~threetears.scheduled_jobs.config.JobConfig.tick_lock_key`, whose
platform default is
:data:`~threetears.scheduled_jobs.config.DEFAULT_TICK_LOCK_KEY` -- but
nothing in the package varied it, so nothing exercised it. These tests
wire it: two pumps constructed in one process hold DIFFERENT keys and
both run, where two pumps sharing a key serialise and one is skipped.

The fake lock here is a real mutex keyed by string (a held key raises
:class:`~threetears.nats.LockHeld`), so the "no contention" assertion is
load-bearing rather than a lock that never blocks anything. The
serialisation test is its control: it proves the fake CAN block.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any
from uuid import UUID

import pytest

from threetears.nats import LockHeld

from threetears.scheduled_jobs import tick as tick_mod
from threetears.scheduled_jobs.config import (
    DEFAULT_DISPATCH_REAP_AFTER_SECONDS,
    DEFAULT_JOB_CONFIG,
    DEFAULT_TICK_DUE_LIMIT,
    DEFAULT_TICK_LOCK_KEY,
    JobConfig,
)
from threetears.scheduled_jobs.protocols import DueSchedule, FireStore, ScheduleStore
from threetears.scheduled_jobs.types import JobFireResult, JobTrigger


_BILLING_LOCK_KEY = "billing_export_tick"
_DATASET_LOCK_KEY = "dataset_build_tick"


class _PumpConfig:
    """:class:`JobConfig` pinning one pump's cross-pod lock key.

    Structurally satisfies the pure-read config Protocol; every knob
    other than the lock key returns the platform baseline.
    """

    def __init__(self, lock_key: str) -> None:
        self._lock_key = lock_key

    @property
    def tick_lock_key(self) -> str:
        return self._lock_key

    @property
    def tick_due_limit(self) -> int:
        return DEFAULT_TICK_DUE_LIMIT

    @property
    def dispatch_reap_after_seconds(self) -> int:
        return DEFAULT_DISPATCH_REAP_AFTER_SECONDS

    @property
    def dispatch_reap_after_seconds_by_kind(self) -> Mapping[str, int]:
        return MappingProxyType({})


class _SlowScheduleStore(ScheduleStore):
    """Due-scan that yields to the event loop so two pumps can overlap.

    # parity-with: threetears.scheduled_jobs.protocols.ScheduleStore
    """

    def __init__(self, label: str, scans: list[str]) -> None:
        self._label = label
        self._scans = scans

    async def list_due_for_tick(
        self,
        now: datetime,
        *,
        kinds: Sequence[str],
        limit: int = 200,
    ) -> list[DueSchedule]:
        self._scans.append(self._label)
        # yield long enough that a contending pump reaches the lock while
        # this body still holds it -- otherwise "no contention" would pass
        # trivially on a body that never overlaps.
        await asyncio.sleep(0.02)
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
    """Inert fire store -- these tests never dispatch.

    # parity-with: threetears.scheduled_jobs.protocols.FireStore
    """

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
        return 0


class _KeyedLockManager:
    """A real in-process mutex keyed by lock-key string.

    Models the cross-pod lock's contract: acquiring a key another holder
    already owns raises :class:`LockHeld`; distinct keys never interact.
    """

    def __init__(self) -> None:
        self.held: set[str] = set()
        self.requested: list[str] = []

    def context(self, _client: Any, key: str, **_kwargs: Any) -> Any:
        """Return an async context manager guarding ``key``."""
        self.requested.append(key)
        return _KeyedLockContext(self, key)


class _KeyedLockContext:
    """One acquisition attempt against a :class:`_KeyedLockManager` key."""

    def __init__(self, manager: _KeyedLockManager, key: str) -> None:
        self._manager = manager
        self._key = key
        self._acquired = False

    async def __aenter__(self) -> None:
        if self._key in self._manager.held:
            raise LockHeld(f"held: {self._key}")
        self._manager.held.add(self._key)
        self._acquired = True

    async def __aexit__(self, *_: Any) -> bool:
        if self._acquired:
            self._manager.held.discard(self._key)
        return False


class _CtxHealthy:
    """Async context manager that acquires cleanly and yields the body."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: Any) -> bool:
        return False


async def _noop(_trigger: JobTrigger, _fire_id: UUID) -> JobFireResult:
    """Dispatch callback that is never reached by these tests."""
    return JobFireResult()


def _install_lock_manager(monkeypatch: pytest.MonkeyPatch) -> _KeyedLockManager:
    """Install and return the keyed-mutex lock stand-in."""
    manager = _KeyedLockManager()
    monkeypatch.setattr("threetears.nats.nats_distributed_lock", manager.context)
    return manager


class TestLockKeySeam:
    """The pre-existing seam carries the key the engine acquires."""

    def test_platform_default_key_is_the_documented_constant(self) -> None:
        assert DEFAULT_JOB_CONFIG.tick_lock_key == DEFAULT_TICK_LOCK_KEY == "scheduled_jobs_tick"

    def test_two_pump_configs_in_one_process_hold_different_keys(self) -> None:
        """The enforcement assertion: the seam is actually varied."""
        billing: JobConfig = _PumpConfig(_BILLING_LOCK_KEY)
        dataset: JobConfig = _PumpConfig(_DATASET_LOCK_KEY)
        assert billing.tick_lock_key != dataset.tick_lock_key

    async def test_engine_acquires_the_configured_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = _install_lock_manager(monkeypatch)
        await tick_mod.scheduled_tick_job(
            _SlowScheduleStore("solo", []),
            _FakeFireStore(),
            {"billing_export": _noop},
            nats_client=object(),
            config=_PumpConfig(_BILLING_LOCK_KEY),
        )
        assert manager.requested == [_BILLING_LOCK_KEY]


class TestTwoPumpsDoNotContend:
    """Distinct keys let two pumps run concurrently; one key serialises."""

    async def test_distinct_keys_both_bodies_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = _install_lock_manager(monkeypatch)
        scans: list[str] = []

        await asyncio.gather(
            tick_mod.scheduled_tick_job(
                _SlowScheduleStore("billing", scans),
                _FakeFireStore(),
                {"billing_export": _noop},
                nats_client=object(),
                config=_PumpConfig(_BILLING_LOCK_KEY),
            ),
            tick_mod.scheduled_tick_job(
                _SlowScheduleStore("dataset", scans),
                _FakeFireStore(),
                {"dataset_build": _noop},
                nats_client=object(),
                config=_PumpConfig(_DATASET_LOCK_KEY),
            ),
        )

        assert sorted(scans) == ["billing", "dataset"]
        assert sorted(manager.requested) == sorted([_BILLING_LOCK_KEY, _DATASET_LOCK_KEY])
        assert manager.held == set()

    async def test_shared_key_skips_the_second_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Control for the test above: the fake lock really can block."""
        _install_lock_manager(monkeypatch)
        scans: list[str] = []

        await asyncio.gather(
            tick_mod.scheduled_tick_job(
                _SlowScheduleStore("billing", scans),
                _FakeFireStore(),
                {"billing_export": _noop},
                nats_client=object(),
                config=_PumpConfig(_BILLING_LOCK_KEY),
            ),
            tick_mod.scheduled_tick_job(
                _SlowScheduleStore("dataset", scans),
                _FakeFireStore(),
                {"dataset_build": _noop},
                nats_client=object(),
                config=_PumpConfig(_BILLING_LOCK_KEY),
            ),
        )

        assert len(scans) == 1

    async def test_no_nats_client_skips_locking_entirely(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single-pod dev mode: the per-row CAS is the only guard."""
        monkeypatch.setattr("threetears.nats.nats_distributed_lock", lambda *_a, **_k: _CtxHealthy())
        scans: list[str] = []
        await tick_mod.scheduled_tick_job(
            _SlowScheduleStore("solo", scans),
            _FakeFireStore(),
            {"billing_export": _noop},
            nats_client=None,
            config=_PumpConfig(_BILLING_LOCK_KEY),
        )
        assert scans == ["solo"]
