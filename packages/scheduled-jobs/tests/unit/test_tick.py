"""Unit tests for :func:`threetears.scheduled_jobs.tick.scheduled_tick_job`.

Drive the tick engine over a fake store + fake fire store (both
fake-parity-declared against the production Protocols) and a recording
dispatch callback. No DB, no NATS server -- the lock is monkeypatched at
``threetears.nats.nats_distributed_lock`` (resolved by the local import
inside the engine), matching agent-wake's ``test_tick_degrade_open``
pattern.

Cases:

- cross-pod lock held -> the tick body is skipped (no due-scan, no fires).
- ``KvError`` (lock infra gone) -> degrade open: the body runs anyway.
- due-enumeration + CAS-claim happy path -> a fire is dispatched and
  finalized success.
- a losing CAS claim -> the row is skipped (no dispatch, no fire row).
- per-row failure isolation -> one raising dispatch does not poison the
  rest of the tick.
- drift recorded -> the emitter observes the actual-minus-scheduled gap.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.nats import LockHeld
from threetears.nats.errors import KvError

from threetears.scheduled_jobs import tick as tick_mod
from threetears.scheduled_jobs.protocols import DueSchedule, FireStore, ScheduleStore
from threetears.scheduled_jobs.types import JobFireResult, JobTrigger


def _now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class _FakeDueSchedule(DueSchedule):
    """A plain due-schedule row.

    # parity-with: threetears.scheduled_jobs.protocols.DueSchedule
    """

    def __init__(
        self,
        *,
        job_id: UUID | None = None,
        partition_key: UUID | None = None,
        kind: str = "demo",
        schedule_type: str = "interval",
        schedule_config: dict[str, Any] | None = None,
        missed_fire_policy: str = "coalesce",
        next_fire_at: datetime | None = None,
        last_fired_at: datetime | None = None,
        payload: dict[str, Any] | None = None,
        name: str | None = "demo-job",
    ) -> None:
        self._job_id = job_id or uuid4()
        self._partition_key = partition_key or uuid4()
        self._kind = kind
        self._schedule_type = schedule_type
        self._schedule_config = schedule_config if schedule_config is not None else {"seconds": 60}
        self._missed_fire_policy = missed_fire_policy
        self._next_fire_at = next_fire_at if next_fire_at is not None else _now() - timedelta(seconds=5)
        self._last_fired_at = last_fired_at
        self._payload = payload if payload is not None else {"x": 1}
        self._name = name

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
        return self._payload

    @property
    def schedule_type(self) -> str:
        return self._schedule_type

    @property
    def schedule_config(self) -> dict[str, Any]:
        return self._schedule_config

    @property
    def missed_fire_policy(self) -> str:
        return self._missed_fire_policy

    @property
    def next_fire_at(self) -> datetime | None:
        return self._next_fire_at

    @property
    def last_fired_at(self) -> datetime | None:
        return self._last_fired_at

    @property
    def name(self) -> str | None:
        return self._name


class _FakeScheduleStore(ScheduleStore):
    """Records claim calls; ``claim_outcomes`` controls per-job CAS result.

    # parity-with: threetears.scheduled_jobs.protocols.ScheduleStore
    """

    def __init__(
        self,
        due: list[DueSchedule],
        *,
        claim_outcomes: dict[UUID, bool] | None = None,
    ) -> None:
        self._due = due
        self._claim_outcomes = claim_outcomes or {}
        self.claims: list[dict[str, Any]] = []

    async def list_due_for_tick(self, now: datetime, *, limit: int = 200) -> list[DueSchedule]:
        return list(self._due)

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
        self.claims.append(
            {
                "partition_key": partition_key,
                "job_id": job_id,
                "expected_next_fire": expected_next_fire,
                "computed_next_fire": computed_next_fire,
                "new_status": new_status,
            }
        )
        return self._claim_outcomes.get(job_id, True)


class _FakeFireStore(FireStore):
    """Records every fire-store call for assertions.

    # parity-with: threetears.scheduled_jobs.protocols.FireStore
    """

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.succeeded: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []

    async def create_dispatching(
        self,
        *,
        fire_id: UUID,
        job_id: UUID,
        partition_key: UUID,
        scheduled_fire_at: datetime,
        actual_fired_at: datetime,
    ) -> None:
        self.created.append({"fire_id": fire_id, "job_id": job_id, "partition_key": partition_key})

    async def finalize_success(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        status: str = "succeeded",
        output: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> None:
        self.succeeded.append({"fire_id": fire_id, "status": status, "output": output, "latency_ms": latency_ms})

    async def finalize_failed(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        error: str,
        latency_ms: int | None = None,
    ) -> None:
        self.failed.append({"fire_id": fire_id, "error": error, "latency_ms": latency_ms})


class _CtxRaisingOnEnter:
    """Async context manager whose ``__aenter__`` raises -- models a lock
    whose acquisition fails (``KvError``) or is already held (``LockHeld``)."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def __aenter__(self) -> None:
        raise self._exc

    async def __aexit__(self, *_: Any) -> bool:
        return False


class _CtxHealthy:
    """Async context manager that acquires cleanly and yields the body."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: Any) -> bool:
        return False


def _patch_lock(monkeypatch: pytest.MonkeyPatch, ctx: Any) -> None:
    """Replace ``threetears.nats.nats_distributed_lock`` (resolved by the
    local import inside the engine) with a factory returning ``ctx``."""

    def _factory(_client: Any, _key: str, **_kw: Any) -> Any:
        return ctx

    monkeypatch.setattr("threetears.nats.nats_distributed_lock", _factory)


async def _record_success(_trigger: JobTrigger, _fire_id: UUID) -> JobFireResult:
    return JobFireResult(status="succeeded", output={"ok": True}, latency_ms=12)


class TestLockControlFlow:
    """The cross-pod lock gates the tick body, with degrade-open on KvError."""

    async def test_lock_held_skips_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch, _CtxRaisingOnEnter(LockHeld("held: scheduled_jobs_tick")))
        store = _FakeScheduleStore([_FakeDueSchedule()])
        fires = _FakeFireStore()
        await tick_mod.scheduled_tick_job(store, fires, _record_success, nats_client=object())
        # LockHeld skips the WHOLE body -- no claim, no fire.
        assert store.claims == []
        assert fires.created == []

    async def test_kverror_degrades_open_and_runs_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch, _CtxRaisingOnEnter(KvError("nats: no response from stream")))
        sched = _FakeDueSchedule()
        store = _FakeScheduleStore([sched])
        fires = _FakeFireStore()
        # Must NOT raise -- KvError is degraded to a warning + run.
        await tick_mod.scheduled_tick_job(store, fires, _record_success, nats_client=object())
        assert len(store.claims) == 1
        assert len(fires.created) == 1
        assert len(fires.succeeded) == 1


class TestHappyPath:
    """A healthy lock + a claimable due row dispatches and finalizes."""

    async def test_due_row_dispatches_and_finalizes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch, _CtxHealthy())
        sched = _FakeDueSchedule(schedule_type="interval", schedule_config={"seconds": 60})
        store = _FakeScheduleStore([sched])
        fires = _FakeFireStore()

        seen: list[JobTrigger] = []

        async def _cb(trigger: JobTrigger, fire_id: UUID) -> JobFireResult:
            seen.append(trigger)
            return JobFireResult(status="succeeded", output={"done": 1}, latency_ms=7)

        await tick_mod.scheduled_tick_job(store, fires, _cb, nats_client=object())

        # one claim, one fire created + finalized success
        assert len(store.claims) == 1
        claim = store.claims[0]
        assert claim["job_id"] == sched.job_id
        assert claim["expected_next_fire"] == sched.next_fire_at
        # interval is NOT terminal -> claim re-arms (active) with a future next-fire
        assert claim["new_status"] == "active"
        assert claim["computed_next_fire"] is not None
        assert len(fires.created) == 1
        assert len(fires.succeeded) == 1
        assert fires.succeeded[0]["output"] == {"done": 1}
        assert fires.succeeded[0]["latency_ms"] == 7
        # the trigger carries the opaque kind + payload verbatim
        assert len(seen) == 1
        assert seen[0].kind == "demo"
        assert seen[0].payload == {"x": 1}
        assert seen[0].job_id == sched.job_id

    async def test_consumer_defined_terminal_status_flows_through_intact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The engine is status-agnostic: a non-default terminal status the callback reports is
        persisted verbatim via ``finalize_success`` (NOT coerced to ``'succeeded'``).

        This is what lets a richer consumer — agent-wake's ``'yielded'`` / ``'fired_silent'`` /
        ``'skipped_busy'`` — delegate to this engine instead of keeping its own tick. ``JobFireResult.
        status`` is an open ``str``; only ``'failed'`` is engine-interpreted (→ finalize_failed)."""
        _patch_lock(monkeypatch, _CtxHealthy())
        store = _FakeScheduleStore([_FakeDueSchedule(schedule_type="interval", schedule_config={"seconds": 60})])
        fires = _FakeFireStore()

        async def _cb(_t: JobTrigger, _f: UUID) -> JobFireResult:
            # a consumer-specific terminal status the generic vocabulary does not enumerate
            return JobFireResult(status="skipped_busy", output={"deferred": True})

        await tick_mod.scheduled_tick_job(store, fires, _cb, nats_client=object())

        # finalized as the consumer's status (success path, not failed), persisted verbatim
        assert len(fires.succeeded) == 1
        assert fires.succeeded[0]["status"] == "skipped_busy"
        assert fires.failed == []

    async def test_one_shot_claim_marks_expired(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A terminal one-shot type claims with ``new_status='expired'`` + NULL next-fire."""
        _patch_lock(monkeypatch, _CtxHealthy())
        sched = _FakeDueSchedule(
            schedule_type="one_shot_at",
            schedule_config={"fire_at_iso": "2026-06-01T11:59:00+00:00"},
        )
        store = _FakeScheduleStore([sched])
        fires = _FakeFireStore()
        await tick_mod.scheduled_tick_job(store, fires, _record_success, nats_client=object())
        assert store.claims[0]["new_status"] == "expired"
        assert store.claims[0]["computed_next_fire"] is None


class TestCasMissSkips:
    """A losing CAS claim skips the row -- no dispatch, no fire row."""

    async def test_lost_claim_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch, _CtxHealthy())
        sched = _FakeDueSchedule()
        store = _FakeScheduleStore([sched], claim_outcomes={sched.job_id: False})
        fires = _FakeFireStore()

        called = False

        async def _cb(_t: JobTrigger, _f: UUID) -> JobFireResult:
            nonlocal called
            called = True
            return JobFireResult()

        await tick_mod.scheduled_tick_job(store, fires, _cb, nats_client=object())

        assert len(store.claims) == 1  # the claim WAS attempted
        assert fires.created == []  # but no fire row was written
        assert called is False  # and the callback never ran


class TestPerRowFailureIsolation:
    """One raising / failing dispatch does not poison the rest of the tick."""

    async def test_raising_dispatch_isolated_and_recorded_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch, _CtxHealthy())
        bad = _FakeDueSchedule(kind="bad")
        good = _FakeDueSchedule(kind="good")
        store = _FakeScheduleStore([bad, good])
        fires = _FakeFireStore()

        async def _cb(trigger: JobTrigger, _f: UUID) -> JobFireResult:
            if trigger.kind == "bad":
                raise RuntimeError("boom")
            return JobFireResult(status="succeeded")

        # Must NOT raise -- the bad row is isolated.
        await tick_mod.scheduled_tick_job(store, fires, _cb, nats_client=object())

        # both rows were claimed + got an in-flight fire
        assert len(store.claims) == 2
        assert len(fires.created) == 2
        # the bad one finalized failed, the good one finalized success
        assert len(fires.failed) == 1
        assert "boom" in fires.failed[0]["error"]
        assert len(fires.succeeded) == 1

    async def test_returned_failed_status_routes_to_finalize_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A callback that returns ``status='failed'`` (no raise) records the error."""
        _patch_lock(monkeypatch, _CtxHealthy())
        store = _FakeScheduleStore([_FakeDueSchedule()])
        fires = _FakeFireStore()

        async def _cb(_t: JobTrigger, _f: UUID) -> JobFireResult:
            return JobFireResult(status="failed", error="downstream rejected")

        await tick_mod.scheduled_tick_job(store, fires, _cb, nats_client=object())
        assert len(fires.failed) == 1
        assert fires.failed[0]["error"] == "downstream rejected"
        assert fires.succeeded == []


class TestDriftRecorded:
    """The emitter observes drift (actual - scheduled)."""

    async def test_drift_observed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch, _CtxHealthy())
        observed: list[float] = []

        class _RecordingEmitter:
            def observe_tick_duration(self, _s: float) -> None: ...
            def observe_drift(self, s: float) -> None:
                observed.append(s)

            def inc_fire(self, **_kw: Any) -> None: ...
            def inc_failure(self, **_kw: Any) -> None: ...

        monkeypatch.setattr(tick_mod, "get_scheduled_jobs_emitter", lambda *a, **k: _RecordingEmitter())

        # schedule whose planned fire is 90s before now -> drift ~= 90s
        sched = _FakeDueSchedule(next_fire_at=_now() - timedelta(seconds=90))
        store = _FakeScheduleStore([sched])
        fires = _FakeFireStore()
        await tick_mod.scheduled_tick_job(store, fires, _record_success, nats_client=object())

        assert len(observed) == 1
        assert observed[0] >= 89.0  # the tick's ``now`` is slightly after _now(); drift is at least ~90s
