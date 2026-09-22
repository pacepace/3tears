"""Unit tests for :class:`threetears.scheduled_jobs.background.BackgroundDispatch`.

The tick engine awaits each dispatch callback inline, so a pump whose fires take minutes has a
tick that lasts the sum of them: a kind scheduled every 60 seconds waits behind every slow one.
``BackgroundDispatch`` hands each fire off (the engine leaves its row in flight), runs the body as
a task of its own, and finalizes the row with the real outcome. These cases pin what that promises:

- the wrapped callback returns at once, handed off, while the body is still running;
- the body's own result, failure, exception or timeout is what the fire row records;
- one fire per kind is in flight at a time, in-process and (through the NATS lock) across pods;
- ``max_concurrent`` bounds how many bodies run at once;
- a fire timeout that would outlive the reaper's threshold is refused at construction;
- ``aclose`` cancels what is still running and records each fire as failed, saying why.

No DB and no NATS server: the fire store is a recording fake, and the cross-pod lock is
monkeypatched at ``threetears.nats.nats_distributed_lock`` exactly as the tick tests do.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.nats import LockHeld
from threetears.nats.errors import KvError

from threetears.scheduled_jobs.background import IN_FLIGHT_SKIP_OUTPUT_KEY, BackgroundDispatch, in_flight_lock_key
from threetears.scheduled_jobs.config import DEFAULT_JOB_CONFIG
from threetears.scheduled_jobs.protocols import FireStore
from threetears.scheduled_jobs.types import JobFireResult, JobTrigger

_NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _trigger(kind: str = "demo", schedule_type: str = "interval") -> JobTrigger:
    return JobTrigger(
        job_id=uuid4(),
        partition_key=uuid4(),
        kind=kind,
        schedule_type=schedule_type,
        fired_at=_NOW,
        scheduled_fire_at=_NOW,
    )


class _FakeFireStore(FireStore):
    """Records the finalize calls a background fire makes; ``raise_on_finalize`` models a DB outage."""

    def __init__(self, *, raise_on_finalize: bool = False) -> None:
        self.succeeded: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []
        self._raise_on_finalize = raise_on_finalize

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
        if self._raise_on_finalize:
            raise ConnectionError("database unreachable")
        self.succeeded.append({"fire_id": fire_id, "status": status, "output": output, "latency_ms": latency_ms})

    async def finalize_failed(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        error: str,
        latency_ms: int | None = None,
    ) -> None:
        if self._raise_on_finalize:
            raise ConnectionError("database unreachable")
        self.failed.append({"fire_id": fire_id, "error": error, "latency_ms": latency_ms})

    async def reap_stale_dispatching(
        self,
        now: datetime,
        *,
        older_than: timedelta,
        kinds: Sequence[str],
    ) -> int:
        return 0


class _RecordingEmitter:
    # parity-exempt: records only inc_fire and inc_failure, the two counters a background fire reports, a strict subset of ScheduledJobsMetricsEmitter
    def __init__(self) -> None:
        self.fires: list[tuple[str, str]] = []
        self.failures: list[str] = []

    def inc_fire(self, *, status: str, schedule_type: str) -> None:
        self.fires.append((status, schedule_type))

    def inc_failure(self, *, reason: str) -> None:
        self.failures.append(reason)


class _Config:
    # parity-exempt: a JobConfig stand-in carrying only reap thresholds, the one thing BackgroundDispatch reads from it
    def __init__(self, *, fallback: int = 900, by_kind: dict[str, int] | None = None) -> None:
        self.tick_lock_key = DEFAULT_JOB_CONFIG.tick_lock_key
        self.tick_due_limit = DEFAULT_JOB_CONFIG.tick_due_limit
        self.dispatch_reap_after_seconds = fallback
        self.dispatch_reap_after_seconds_by_kind = MappingProxyType(by_kind or {})


class _CtxRaisingOnEnter:
    """An async context manager whose entry raises -- a held or broken cross-pod lock."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def __aenter__(self) -> None:
        raise self._exc

    async def __aexit__(self, *_: Any) -> bool:
        return False


def _patch_lock(monkeypatch: pytest.MonkeyPatch, ctx: Any, keys: list[str] | None = None) -> None:
    def _factory(_client: Any, key: str, **_kw: Any) -> Any:
        if keys is not None:
            keys.append(key)
        return ctx

    monkeypatch.setattr("threetears.nats.nats_distributed_lock", _factory)


# ---------------------------------------------------------------------------
# handoff and finalize
# ---------------------------------------------------------------------------


async def test_the_wrapped_callback_hands_off_at_once_and_the_task_finalizes_the_real_result() -> None:
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires)
    release = asyncio.Event()

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        await release.wait()
        return JobFireResult(status="succeeded", output={"rows": 3})

    wrapped = dispatch.wrap(_body)
    fire_id = uuid4()
    result = await asyncio.wait_for(wrapped(_trigger(), fire_id), timeout=1)

    assert result.handed_off is True
    assert fires.succeeded == [], "nothing is finalized while the body is still running"
    assert dispatch.in_flight == {"demo": fire_id}

    release.set()
    await dispatch.join()
    assert fires.succeeded[0]["fire_id"] == fire_id
    assert fires.succeeded[0]["status"] == "succeeded"
    assert fires.succeeded[0]["output"] == {"rows": 3}
    assert fires.succeeded[0]["latency_ms"] is not None
    assert dispatch.in_flight == {}


async def test_a_status_the_body_reports_is_persisted_verbatim() -> None:
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(status="skipped_busy", latency_ms=42)

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.succeeded[0]["status"] == "skipped_busy"
    assert fires.succeeded[0]["latency_ms"] == 42, "a latency the body measured itself wins"


async def test_a_failed_result_finalizes_failed_with_its_error() -> None:
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(status="failed", error="upstream said no")

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.failed[0]["error"] == "upstream said no"
    assert fires.succeeded == []


async def test_a_raised_exception_finalizes_failed_with_its_text() -> None:
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        raise ValueError("bad page 3")

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.failed[0]["error"] == "bad page 3"


async def test_an_exception_with_no_text_is_recorded_by_its_type_name() -> None:
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        raise RuntimeError()

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.failed[0]["error"] == "RuntimeError"


async def test_a_timed_out_body_is_cancelled_and_names_its_kind_and_limit() -> None:
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires, fire_timeout_seconds=0.05)
    cancelled = asyncio.Event()

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return JobFireResult()

    await dispatch.wrap(_body)(_trigger(kind="poll:warn_act"), uuid4())
    await dispatch.join()
    assert cancelled.is_set()
    assert fires.failed[0]["error"] == "poll:warn_act: fire exceeded its 0.05s limit and was cancelled"


async def test_a_per_kind_timeout_overrides_the_default() -> None:
    dispatch = BackgroundDispatch(
        _FakeFireStore(), fire_timeout_seconds=300, fire_timeout_seconds_by_kind={"poll:warn_act": 600}
    )
    assert dispatch.fire_timeout_for("poll:warn_act") == 600
    assert dispatch.fire_timeout_for("poll:fred") == 300


async def test_a_body_that_itself_hands_off_is_not_finalized_twice() -> None:
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(handed_off=True)

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.succeeded == []
    assert fires.failed == []


async def test_a_finalize_that_fails_is_logged_and_releases_the_kind(caplog: pytest.LogCaptureFixture) -> None:
    """A lost write leaves the row in flight for the reaper; it must not wedge the kind."""
    dispatch = BackgroundDispatch(_FakeFireStore(raise_on_finalize=True))

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult()

    await dispatch.wrap(_body)(_trigger(), uuid4())
    with caplog.at_level("ERROR"):
        await dispatch.join()
    assert dispatch.in_flight == {}
    assert any("finalize" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# one fire per kind; bounded concurrency
# ---------------------------------------------------------------------------


async def test_a_kind_already_in_flight_records_the_new_fire_as_skipped() -> None:
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires)
    release = asyncio.Event()
    runs = 0

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        nonlocal runs
        runs += 1
        await release.wait()
        return JobFireResult()

    wrapped = dispatch.wrap(_body)
    first = uuid4()
    await wrapped(_trigger(), first)
    second = await wrapped(_trigger(), uuid4())

    assert second.handed_off is False, "the engine finalizes the skip itself"
    assert second.status == "succeeded"
    assert second.output is not None
    assert second.output[IN_FLIGHT_SKIP_OUTPUT_KEY] == "previous fire still in flight"
    assert second.output["in_flight_fire_id"] == str(first)

    release.set()
    await dispatch.join()
    assert runs == 1


async def test_different_kinds_run_at_the_same_time() -> None:
    dispatch = BackgroundDispatch(_FakeFireStore())
    both_running = asyncio.Event()
    running: set[str] = set()

    async def _body(trigger: JobTrigger, _f: UUID) -> JobFireResult:
        running.add(trigger.kind)
        if len(running) == 2:
            both_running.set()
        await asyncio.wait_for(both_running.wait(), timeout=1)
        return JobFireResult()

    wrapped = dispatch.wrap(_body)
    await wrapped(_trigger(kind="poll:slow"), uuid4())
    await wrapped(_trigger(kind="poll:fast"), uuid4())
    await dispatch.join()
    assert both_running.is_set()


async def test_max_concurrent_bounds_the_bodies_running_at_once() -> None:
    dispatch = BackgroundDispatch(_FakeFireStore(), max_concurrent=2)
    release = asyncio.Event()
    active = 0
    peak = 0

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await release.wait()
        active -= 1
        return JobFireResult()

    wrapped = dispatch.wrap(_body)
    for kind in ("a", "b", "c", "d"):
        await wrapped(_trigger(kind=kind), uuid4())
    await asyncio.sleep(0.05)
    assert peak == 2
    release.set()
    await dispatch.join()
    assert peak == 2


# ---------------------------------------------------------------------------
# the cross-pod lock
# ---------------------------------------------------------------------------


async def test_each_fire_holds_the_in_flight_lock_for_its_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    keys: list[str] = []

    class _Healthy:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_: Any) -> bool:
            return False

    _patch_lock(monkeypatch, _Healthy(), keys)
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires, nats_client=object())

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult()

    await dispatch.wrap(_body)(_trigger(kind="poll:gdelt"), uuid4())
    await dispatch.join()
    assert keys == [in_flight_lock_key("poll:gdelt")]
    assert len(fires.succeeded) == 1


@pytest.mark.parametrize("kind", ["gdelt", "poll:gdelt", "backfill:eodhd", "a kind with spaces", "ünïcode", "x" * 300])
def test_every_in_flight_lock_key_is_a_valid_nats_kv_key(kind: str) -> None:
    """nats-server refuses a key outside its grammar (``poll:gdelt`` raised
    ``JetStream.InvalidKeyError`` against a real server, which the lock turned into a KvError)."""
    import re

    key = in_flight_lock_key(kind)
    assert re.fullmatch(r"[-/_=.a-zA-Z0-9]+", key), key
    assert key == in_flight_lock_key(kind), "stable for the kind"


def test_a_kind_inside_the_key_grammar_stays_readable() -> None:
    assert in_flight_lock_key("gdelt") == "scheduled_jobs_in_flight.gdelt"


def test_kinds_that_differ_only_in_punctuation_get_different_keys() -> None:
    assert in_flight_lock_key("poll:x") != in_flight_lock_key("poll;x")


async def test_a_kind_in_flight_on_another_pod_records_the_fire_as_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_lock(monkeypatch, _CtxRaisingOnEnter(LockHeld("held")))
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires, nats_client=object())
    ran = False

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        nonlocal ran
        ran = True
        return JobFireResult()

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert ran is False
    assert fires.succeeded[0]["output"][IN_FLIGHT_SKIP_OUTPUT_KEY] == "fire in flight on another pod"


async def test_a_broken_lock_degrades_open_and_the_body_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same stance as the tick lock: the lock only saves redundant work, so its absence must not
    silence the fire."""
    _patch_lock(monkeypatch, _CtxRaisingOnEnter(KvError("nats: no response from stream")))
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires, nats_client=object())

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(output={"ran": True})

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.succeeded[0]["output"] == {"ran": True}


# ---------------------------------------------------------------------------
# construction guards
# ---------------------------------------------------------------------------


def test_a_default_timeout_that_outlives_the_reap_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="reap"):
        BackgroundDispatch(_FakeFireStore(), fire_timeout_seconds=900, config=_Config(fallback=900))


def test_a_per_kind_timeout_that_outlives_its_reap_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="poll:warn_act"):
        BackgroundDispatch(
            _FakeFireStore(),
            fire_timeout_seconds_by_kind={"poll:warn_act": 1800},
            config=_Config(fallback=900),
        )


def test_the_default_timeout_is_checked_against_a_kind_whose_reap_threshold_is_lower() -> None:
    with pytest.raises(ValueError, match="poll:quick"):
        BackgroundDispatch(_FakeFireStore(), fire_timeout_seconds=300, config=_Config(by_kind={"poll:quick": 120}))


def test_a_long_per_kind_timeout_fits_under_a_raised_reap_threshold() -> None:
    dispatch = BackgroundDispatch(
        _FakeFireStore(),
        fire_timeout_seconds_by_kind={"poll:warn_act": 1800},
        config=_Config(by_kind={"poll:warn_act": 2400}),
    )
    assert dispatch.fire_timeout_for("poll:warn_act") == 1800


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_concurrent": 0}, "max_concurrent"),
        ({"fire_timeout_seconds": 0}, "fire_timeout_seconds"),
        ({"fire_timeout_seconds_by_kind": {"a": -1}}, "a"),
    ],
)
def test_nonsense_limits_are_refused(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        BackgroundDispatch(_FakeFireStore(), **kwargs)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


async def test_a_background_fire_reports_its_outcome_to_the_metrics() -> None:
    emitter = _RecordingEmitter()
    dispatch = BackgroundDispatch(_FakeFireStore(), emitter=emitter)  # type: ignore[arg-type]

    async def _ok(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult()

    async def _bad(_t: JobTrigger, _f: UUID) -> JobFireResult:
        raise ValueError("nope")

    await dispatch.wrap(_ok)(_trigger(kind="a", schedule_type="interval"), uuid4())
    await dispatch.wrap(_bad)(_trigger(kind="b", schedule_type="cron"), uuid4())
    await dispatch.join()
    assert sorted(emitter.fires) == [("failed", "cron"), ("succeeded", "interval")]
    assert emitter.failures == ["handler_exception"]


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


async def test_aclose_cancels_what_is_running_and_records_why() -> None:
    fires = _FakeFireStore()
    dispatch = BackgroundDispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        await asyncio.sleep(10)
        return JobFireResult()

    fire_id = uuid4()
    await dispatch.wrap(_body)(_trigger(kind="poll:slow"), fire_id)
    await asyncio.sleep(0)
    await dispatch.aclose()
    assert fires.failed == [
        {
            "fire_id": fire_id,
            "error": "poll:slow: cancelled because BackgroundDispatch was closed before the fire finished",
            "latency_ms": fires.failed[0]["latency_ms"],
        }
    ]
    assert dispatch.in_flight == {}


async def test_a_closed_dispatcher_refuses_new_fires_as_failed() -> None:
    dispatch = BackgroundDispatch(_FakeFireStore())
    await dispatch.aclose()

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult()

    result = await dispatch.wrap(_body)(_trigger(kind="poll:late"), uuid4())
    assert result.handed_off is False
    assert result.status == "failed"
    assert result.error == "poll:late: BackgroundDispatch is closed; the fire was not run"
