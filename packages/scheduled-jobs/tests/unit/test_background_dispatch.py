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
from threetears.scheduled_jobs.events import EVENT_FIRE_WAITING_EXCLUSION_GROUP
from threetears.scheduled_jobs.protocols import FireStore
from threetears.scheduled_jobs.types import JobFireResult, JobTrigger


def _trigger(kind: str = "demo", schedule_type: str = "interval", *, fired_ago: float = 0.0) -> JobTrigger:
    """A trigger stamped as the tick would stamp it: fired just now (or ``fired_ago`` seconds ago),
    because the reaper -- and so BackgroundDispatch's deadline -- counts from that instant."""
    fired_at = datetime.now(UTC) - timedelta(seconds=fired_ago)
    return JobTrigger(
        job_id=uuid4(),
        partition_key=uuid4(),
        kind=kind,
        schedule_type=schedule_type,
        fired_at=fired_at,
        scheduled_fire_at=fired_at,
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


def _dispatch(fire_store: FireStore | None = None, **kwargs: Any) -> BackgroundDispatch:
    """A BackgroundDispatch over the platform config unless a test names its own."""
    kwargs.setdefault("config", DEFAULT_JOB_CONFIG)
    return BackgroundDispatch(fire_store if fire_store is not None else _FakeFireStore(), **kwargs)


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
    dispatch = _dispatch(fires)
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
    dispatch = _dispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(status="skipped_busy", latency_ms=42)

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.succeeded[0]["status"] == "skipped_busy"
    assert fires.succeeded[0]["latency_ms"] == 42, "a latency the body measured itself wins"


async def test_a_failed_result_finalizes_failed_with_its_error() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(status="failed", error="upstream said no")

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.failed[0]["error"] == "upstream said no"
    assert fires.succeeded == []


async def test_a_raised_exception_finalizes_failed_with_its_text() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        raise ValueError("bad page 3")

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.failed[0]["error"] == "bad page 3"


async def test_an_exception_with_no_text_is_recorded_by_its_type_name() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        raise RuntimeError()

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.failed[0]["error"] == "RuntimeError"


async def test_a_timed_out_body_is_cancelled_and_names_its_kind_and_limit() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(fires, fire_timeout_seconds=0.05)
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
    assert fires.failed[0]["error"] == "poll:warn_act: fire exceeded its limit (0.05s) and was cancelled"


async def test_a_per_kind_timeout_overrides_the_default() -> None:
    dispatch = _dispatch(
        _FakeFireStore(), fire_timeout_seconds=300, fire_timeout_seconds_by_kind={"poll:warn_act": 600}
    )
    assert dispatch.fire_timeout_for("poll:warn_act") == 600
    assert dispatch.fire_timeout_for("poll:fred") == 300


async def test_a_body_that_itself_hands_off_is_not_finalized_twice() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(handed_off=True)

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.succeeded == []
    assert fires.failed == []


async def test_a_finalize_that_fails_is_logged_and_releases_the_kind(caplog: pytest.LogCaptureFixture) -> None:
    """A lost write leaves the row in flight for the reaper; it must not wedge the kind."""
    dispatch = _dispatch(_FakeFireStore(raise_on_finalize=True))

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
    dispatch = _dispatch(fires)
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
    dispatch = _dispatch(_FakeFireStore())
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
    dispatch = _dispatch(_FakeFireStore(), max_concurrent=2)
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
    dispatch = _dispatch(fires, nats_client=object())

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
    dispatch = _dispatch(fires, nats_client=object())
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
    dispatch = _dispatch(fires, nats_client=object())

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
        _dispatch(_FakeFireStore(), fire_timeout_seconds=900, config=_Config(fallback=900))


def test_a_per_kind_timeout_that_outlives_its_reap_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="poll:warn_act"):
        _dispatch(
            _FakeFireStore(),
            fire_timeout_seconds_by_kind={"poll:warn_act": 1800},
            config=_Config(fallback=900),
        )


def test_the_default_timeout_is_checked_against_a_kind_whose_reap_threshold_is_lower() -> None:
    with pytest.raises(ValueError, match="poll:quick"):
        _dispatch(_FakeFireStore(), fire_timeout_seconds=300, config=_Config(by_kind={"poll:quick": 120}))


def test_a_long_per_kind_timeout_fits_under_a_raised_reap_threshold() -> None:
    dispatch = _dispatch(
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
        _dispatch(_FakeFireStore(), **kwargs)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


async def test_a_background_fire_reports_its_outcome_to_the_metrics() -> None:
    emitter = _RecordingEmitter()
    dispatch = _dispatch(_FakeFireStore(), emitter=emitter)  # type: ignore[arg-type]

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
    dispatch = _dispatch(fires)

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
    dispatch = _dispatch(_FakeFireStore())
    await dispatch.aclose()

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult()

    result = await dispatch.wrap(_body)(_trigger(kind="poll:late"), uuid4())
    assert result.handed_off is False
    assert result.status == "failed"
    assert result.error == "poll:late: BackgroundDispatch is closed; the fire was not run"


# ---------------------------------------------------------------------------
# settle exactly once: the paths where a fire could be finalized twice, never, or falsely
# ---------------------------------------------------------------------------


class _LockReleasing:
    """A lock that is acquired cleanly and then either blocks or raises on release."""

    def __init__(self, *, on_exit: BaseException | None = None) -> None:
        self.releasing = asyncio.Event()
        self.release = asyncio.Event()
        self._on_exit = on_exit

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: Any) -> bool:
        self.releasing.set()
        if self._on_exit is not None:
            raise self._on_exit
        await self.release.wait()
        return False


class _GatedFireStore(_FakeFireStore):
    """A fire store whose success write parks until released -- a slow database write."""

    def __init__(self) -> None:
        super().__init__()
        self.writing = asyncio.Event()
        self.release = asyncio.Event()

    async def finalize_success(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        status: str = "succeeded",
        output: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> None:
        self.writing.set()
        await self.release.wait()
        await super().finalize_success(partition_key, fire_id, status=status, output=output, latency_ms=latency_ms)


async def test_a_body_that_hands_off_again_is_never_written_even_by_aclose() -> None:
    """Another owner holds that row; aclose must not stamp 'cancelled' onto it."""
    fires = _FakeFireStore()
    dispatch = _dispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(handed_off=True)

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    await dispatch.aclose()
    assert fires.succeeded == []
    assert fires.failed == []


async def test_a_cancel_during_lock_release_keeps_the_real_result(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = _LockReleasing()
    _patch_lock(monkeypatch, lock)
    fires = _FakeFireStore()
    dispatch = _dispatch(fires, nats_client=object())

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(output={"watermark": 42})

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await asyncio.wait_for(lock.releasing.wait(), timeout=1)
    await dispatch.aclose()
    assert fires.failed == []
    assert fires.succeeded[0]["output"] == {"watermark": 42}


async def test_a_kverror_on_lock_release_keeps_the_real_result_and_runs_the_body_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lock(monkeypatch, _LockReleasing(on_exit=KvError("nats: key delete failed")))
    fires = _FakeFireStore()
    dispatch = _dispatch(fires, nats_client=object())
    runs = 0

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        nonlocal runs
        runs += 1
        return JobFireResult(output={"ran": runs})

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert runs == 1
    assert fires.succeeded[0]["output"] == {"ran": 1}


@pytest.mark.parametrize("exc", [ValueError("ttl mismatch on scheduler-locks"), RuntimeError()])
async def test_a_lock_that_fails_otherwise_fails_the_fire_loudly_without_running_it(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    """Neither LockHeld nor KvError -- a misconfigured bucket, a TTL mismatch: the fire must be
    recorded as failed with the reason, not vanish with its task."""
    _patch_lock(monkeypatch, _CtxRaisingOnEnter(exc))
    fires = _FakeFireStore()
    dispatch = _dispatch(fires, nats_client=object())
    ran = False

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        nonlocal ran
        ran = True
        return JobFireResult()

    await dispatch.wrap(_body)(_trigger(kind="poll:x"), uuid4())
    await dispatch.join()
    assert ran is False
    error = fires.failed[0]["error"]
    assert error.startswith(f"poll:x: the in-flight lock failed ({type(exc).__name__}: ")
    assert error.endswith("the fire was not run")


async def test_a_cancel_during_the_finalize_write_does_not_cut_the_record_short() -> None:
    fires = _GatedFireStore()
    dispatch = _dispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(output={"done": True})

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await asyncio.wait_for(fires.writing.wait(), timeout=1)
    closing = asyncio.get_running_loop().create_task(dispatch.aclose())
    await asyncio.sleep(0.01)
    fires.release.set()
    await asyncio.wait_for(closing, timeout=1)
    assert fires.succeeded[0]["output"] == {"done": True}
    assert fires.failed == []


async def test_aclose_records_a_fire_whose_task_never_started() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(fires)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult()

    fire_id = uuid4()
    await dispatch.wrap(_body)(_trigger(kind="poll:never"), fire_id)
    await dispatch.aclose()  # before the loop ever ran the task
    assert [f["fire_id"] for f in fires.failed] == [fire_id]
    assert fires.failed[0]["error"] == (
        "poll:never: cancelled because BackgroundDispatch was closed before the fire finished"
    )
    assert fires.succeeded == []


# ---------------------------------------------------------------------------
# the reaper's clock
# ---------------------------------------------------------------------------


async def test_a_fire_with_no_time_left_before_its_reap_threshold_is_not_run() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(
        fires,
        fire_timeout_seconds_by_kind={"poll:k": 20},
        config=_Config(by_kind={"poll:k": 60}),
    )
    ran = False

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        nonlocal ran
        ran = True
        return JobFireResult()

    # 60s threshold, 30s margin, fired 45s ago: nothing left.
    await dispatch.wrap(_body)(_trigger(kind="poll:k", fired_ago=45), uuid4())
    await dispatch.join()
    assert ran is False
    assert fires.failed[0]["error"] == (
        "poll:k: waited so long for a free slot that running it now would outlive its 60s reap threshold; "
        "the fire was not run"
    )


async def test_a_fire_is_stopped_before_its_reap_threshold_even_inside_its_own_timeout() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(
        fires,
        fire_timeout_seconds_by_kind={"poll:k": 60},
        config=_Config(by_kind={"poll:k": 100}),
    )

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        await asyncio.sleep(10)
        return JobFireResult()

    # 100s threshold, 30s margin, fired 69.9s ago: ~0.1s left, far under its own 60s timeout.
    await dispatch.wrap(_body)(_trigger(kind="poll:k", fired_ago=69.9), uuid4())
    await asyncio.wait_for(dispatch.join(), timeout=5)
    assert fires.failed[0]["error"].startswith("poll:k: fire exceeded the time left before its reap threshold (")


async def test_a_fire_waiting_for_a_slot_is_recorded_at_its_reap_deadline() -> None:
    """One slot, held by a long fire. The other has 0.2s before its reap threshold (60s less the
    30s margin, fired 29.8s ago): it must be recorded then, while the holder still runs -- not
    when the slot frees, by which time the reaper could have taken its row."""
    fires = _FakeFireStore()
    dispatch = _dispatch(
        fires,
        max_concurrent=1,
        fire_timeout_seconds_by_kind={"poll:long": 20, "poll:short": 20},
        config=_Config(by_kind={"poll:long": 60, "poll:short": 60}),
    )
    body = _Gated("poll:long", "poll:short")
    wrapped = dispatch.wrap(body)
    await wrapped(_trigger(kind="poll:long"), uuid4())
    await body.started["poll:long"].wait()
    await wrapped(_trigger(kind="poll:short", fired_ago=29.8), uuid4())

    for _ in range(100):
        if fires.failed:
            break
        await asyncio.sleep(0.02)
    assert [row["error"] for row in fires.failed] == [
        "poll:short: waited so long for a free slot that running it now would outlive its 60s reap threshold; "
        "the fire was not run"
    ]
    body.release["poll:long"].set()
    await dispatch.join()
    assert not body.started["poll:short"].is_set()
    assert len(fires.failed) == 1


async def test_a_cancelled_fire_gives_its_slot_back() -> None:
    """One slot. Cancelling the fire that holds it must free it for the next one, or every
    cancelled fire would lose a slot until nothing ran at all."""
    fires = _FakeFireStore()
    dispatch = _dispatch(fires, max_concurrent=1)
    body = _Gated("poll:a", "poll:b")
    wrapped = dispatch.wrap(body)
    await wrapped(_trigger(kind="poll:a"), uuid4())
    await body.started["poll:a"].wait()
    await wrapped(_trigger(kind="poll:b"), uuid4())
    await asyncio.sleep(0.02)
    [holder] = [task for task in asyncio.all_tasks() if task.get_name() == "scheduled-jobs:poll:a"]

    holder.cancel()
    await asyncio.wait_for(body.started["poll:b"].wait(), timeout=1)
    body.release["poll:b"].set()
    await dispatch.join()
    assert len(fires.succeeded) == 1


async def test_a_finished_fire_gives_back_exactly_one_slot() -> None:
    """One slot. After a fire finishes, two more must still run one at a time: a slot released
    twice would quietly raise the concurrency cap by one per fire."""
    dispatch = _dispatch(_FakeFireStore(), max_concurrent=1)

    async def _quick(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult()

    await dispatch.wrap(_quick)(_trigger(kind="poll:first"), uuid4())
    await dispatch.join()

    body = _Gated("poll:b", "poll:c")
    wrapped = dispatch.wrap(body)
    await wrapped(_trigger(kind="poll:b"), uuid4())
    await wrapped(_trigger(kind="poll:c"), uuid4())
    await asyncio.sleep(0.05)
    assert [kind for kind, event in body.started.items() if event.is_set()] == ["poll:b"]
    for event in body.release.values():
        event.set()
    await dispatch.join()


def test_a_nan_timeout_is_refused() -> None:
    with pytest.raises(ValueError, match="fire_timeout_seconds"):
        _dispatch(fire_timeout_seconds=float("nan"))


def test_a_timeout_inside_the_threshold_but_inside_the_margin_is_refused() -> None:
    with pytest.raises(ValueError, match="margin"):
        _dispatch(fire_timeout_seconds=880, config=_Config(fallback=900))


# ---------------------------------------------------------------------------
# what gets recorded, and counted
# ---------------------------------------------------------------------------


async def test_a_timeout_the_body_raises_itself_is_recorded_as_its_own_error() -> None:
    """Only the dispatcher's own deadline reads as 'exceeded its limit'."""
    fires = _FakeFireStore()
    dispatch = _dispatch(fires, fire_timeout_seconds=60)

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        raise TimeoutError("upstream read timed out")

    await dispatch.wrap(_body)(_trigger(), uuid4())
    await dispatch.join()
    assert fires.failed[0]["error"] == "upstream read timed out"


async def test_a_timeout_is_counted_as_a_timeout_not_a_handler_exception(caplog: pytest.LogCaptureFixture) -> None:
    emitter = _RecordingEmitter()
    dispatch = _dispatch(fire_timeout_seconds=0.05, emitter=emitter)  # type: ignore[arg-type]

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        await asyncio.sleep(10)
        return JobFireResult()

    with caplog.at_level("ERROR"):
        await dispatch.wrap(_body)(_trigger(schedule_type="interval"), uuid4())
        await dispatch.join()
    assert emitter.fires == [("failed", "interval")]
    assert emitter.failures == ["timeout"]
    assert [r.getMessage() for r in caplog.records].count("3tears.scheduled_jobs.fire.failed") == 1


async def test_a_returned_failure_is_logged_as_a_failed_fire(caplog: pytest.LogCaptureFixture) -> None:
    dispatch = _dispatch()

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(status="failed", error="upstream said no")

    with caplog.at_level("ERROR"):
        await dispatch.wrap(_body)(_trigger(), uuid4())
        await dispatch.join()
    assert [r.getMessage() for r in caplog.records].count("3tears.scheduled_jobs.fire.failed") == 1


async def test_a_cross_pod_skip_is_counted_as_a_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_lock(monkeypatch, _CtxRaisingOnEnter(LockHeld("held")))
    emitter = _RecordingEmitter()
    dispatch = _dispatch(nats_client=object(), emitter=emitter)  # type: ignore[arg-type]

    async def _body(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult()

    await dispatch.wrap(_body)(_trigger(schedule_type="cron"), uuid4())
    await dispatch.join()
    assert emitter.fires == [("succeeded", "cron")]
    assert emitter.failures == []


# ---------------------------------------------------------------------------
# exclusion groups
# ---------------------------------------------------------------------------


class _Gated:
    """A body per kind that records when it starts and ends, and parks until that kind is released."""

    def __init__(self, *kinds: str) -> None:
        self.events: list[str] = []
        self.started = {kind: asyncio.Event() for kind in kinds}
        self.release = {kind: asyncio.Event() for kind in kinds}

    async def __call__(self, trigger: JobTrigger, _f: UUID) -> JobFireResult:
        self.events.append(f"start {trigger.kind}")
        self.started[trigger.kind].set()
        await self.release[trigger.kind].wait()
        self.events.append(f"end {trigger.kind}")
        return JobFireResult()


async def test_kinds_in_one_group_take_turns_in_arrival_order() -> None:
    body = _Gated("poll:g", "backfill:g", "backfill:g_old")
    dispatch = _dispatch(
        _FakeFireStore(),
        exclusion_groups={"poll:g": "g", "backfill:g": "g", "backfill:g_old": "g"},
    )
    wrapped = dispatch.wrap(body)
    for kind in ("backfill:g", "poll:g", "backfill:g_old"):
        assert (await wrapped(_trigger(kind=kind), uuid4())).handed_off
    await body.started["backfill:g"].wait()
    await asyncio.sleep(0.05)
    assert body.events == ["start backfill:g"], "a second kind of the group started while the first ran"

    body.release["backfill:g"].set()
    await body.started["poll:g"].wait()
    body.release["poll:g"].set()
    await body.started["backfill:g_old"].wait()
    body.release["backfill:g_old"].set()
    await dispatch.join()
    assert body.events == [
        "start backfill:g",
        "end backfill:g",
        "start poll:g",
        "end poll:g",
        "start backfill:g_old",
        "end backfill:g_old",
    ]


async def test_kinds_in_different_groups_or_none_run_at_the_same_time() -> None:
    body = _Gated("poll:a", "poll:b", "poll:free")
    dispatch = _dispatch(_FakeFireStore(), exclusion_groups={"poll:a": "a", "poll:b": "b"})
    wrapped = dispatch.wrap(body)
    for kind in ("poll:a", "poll:b", "poll:free"):
        await wrapped(_trigger(kind=kind), uuid4())
    for kind in ("poll:a", "poll:b", "poll:free"):
        await asyncio.wait_for(body.started[kind].wait(), timeout=1)
    for event in body.release.values():
        event.set()
    await dispatch.join()


async def test_a_fire_waiting_for_its_turn_holds_no_slot() -> None:
    """Two slots. The group's first kind takes one; its sibling waits for the group's turn. If the
    waiting sibling held the second slot, the ungrouped kind could not start."""
    body = _Gated("poll:g", "backfill:g", "poll:free")
    dispatch = _dispatch(_FakeFireStore(), max_concurrent=2, exclusion_groups={"poll:g": "g", "backfill:g": "g"})
    wrapped = dispatch.wrap(body)
    await wrapped(_trigger(kind="poll:g"), uuid4())
    await body.started["poll:g"].wait()
    await wrapped(_trigger(kind="backfill:g"), uuid4())
    await asyncio.sleep(0.02)
    await wrapped(_trigger(kind="poll:free"), uuid4())

    await asyncio.wait_for(body.started["poll:free"].wait(), timeout=1)
    assert not body.started["backfill:g"].is_set()
    for event in body.release.values():
        event.set()
    await dispatch.join()


async def test_waiting_for_a_turn_uses_none_of_the_fires_own_limit() -> None:
    """backfill:g has a 0.2s limit and waits 0.3s for poll:g (1s limit) to finish, then runs 0.01s.
    Were the wait counted against its limit, it would be cut off before it ever started."""
    fires = _FakeFireStore()
    dispatch = _dispatch(
        fires,
        fire_timeout_seconds=0.2,
        fire_timeout_seconds_by_kind={"poll:g": 1.0},
        exclusion_groups={"poll:g": "g", "backfill:g": "g"},
    )

    async def _body(trigger: JobTrigger, _f: UUID) -> JobFireResult:
        await asyncio.sleep(0.3 if trigger.kind == "poll:g" else 0.01)
        return JobFireResult(output={"kind": trigger.kind})

    wrapped = dispatch.wrap(_body)
    await wrapped(_trigger(kind="poll:g"), uuid4())
    await wrapped(_trigger(kind="backfill:g"), uuid4())
    await dispatch.join()

    assert fires.failed == []
    assert sorted(row["output"]["kind"] for row in fires.succeeded) == ["backfill:g", "poll:g"]


async def test_waiting_for_a_turn_still_runs_down_the_reap_clock() -> None:
    """The reaper counts from the tick, so a turn waited for is time gone: with 60s to its reap
    threshold less the 30s margin, fired 29.8s ago, the sibling has 0.2s. It is recorded at that
    deadline -- while the holder is still running -- never run, and never recorded twice."""
    fires = _FakeFireStore()
    dispatch = _dispatch(
        fires,
        fire_timeout_seconds_by_kind={"poll:k": 20, "backfill:k": 20},
        config=_Config(by_kind={"poll:k": 60, "backfill:k": 60}),
        exclusion_groups={"poll:k": "k", "backfill:k": "k"},
    )
    body = _Gated("poll:k", "backfill:k")
    wrapped = dispatch.wrap(body)
    await wrapped(_trigger(kind="poll:k"), uuid4())
    await body.started["poll:k"].wait()
    waiting = uuid4()
    await wrapped(_trigger(kind="backfill:k", fired_ago=29.8), waiting)

    for _ in range(100):
        if fires.failed:
            break
        await asyncio.sleep(0.02)
    assert not body.release["poll:k"].is_set(), "the holder is still running"
    assert fires.failed == [
        {
            "fire_id": waiting,
            "error": (
                "backfill:k: waited so long for its turn in exclusion group 'k' that running it now would "
                "outlive its 60s reap threshold; the fire was not run"
            ),
            "latency_ms": fires.failed[0]["latency_ms"],
        }
    ]

    body.release["poll:k"].set()
    await dispatch.join()
    assert not body.started["backfill:k"].is_set()
    assert len(fires.failed) == 1
    assert len(fires.succeeded) == 1


async def test_a_sibling_with_a_shorter_reap_threshold_is_recorded_before_the_reaper_could_take_it() -> None:
    """Kinds in one group need not share a reap threshold. The holder has 100s to its threshold;
    the waiter has 60s and was fired 29.8s ago. Its wait must end at ITS deadline, not the holder's
    -- otherwise the reaper fails its row and the dispatcher later writes it again."""
    fires = _FakeFireStore()
    dispatch = _dispatch(
        fires,
        fire_timeout_seconds_by_kind={"backfill:long": 60, "poll:short": 20},
        config=_Config(by_kind={"backfill:long": 100, "poll:short": 60}),
        exclusion_groups={"backfill:long": "g", "poll:short": "g"},
    )
    body = _Gated("backfill:long", "poll:short")
    wrapped = dispatch.wrap(body)
    await wrapped(_trigger(kind="backfill:long"), uuid4())
    await body.started["backfill:long"].wait()
    await wrapped(_trigger(kind="poll:short", fired_ago=29.8), uuid4())

    for _ in range(100):
        if fires.failed:
            break
        await asyncio.sleep(0.02)
    [failed] = fires.failed
    assert failed["error"].startswith("poll:short: waited so long for its turn in exclusion group 'g'"), failed
    body.release["backfill:long"].set()
    await dispatch.join()
    assert len(fires.failed) == 1


async def test_a_fire_that_fails_hands_the_turn_on() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(fires, exclusion_groups={"poll:g": "g", "backfill:g": "g"})
    queued = asyncio.Event()

    async def _body(trigger: JobTrigger, _f: UUID) -> JobFireResult:
        if trigger.kind == "backfill:g":
            await queued.wait()  # hold the turn until the sibling is really waiting for it
            raise RuntimeError("upstream down")
        return JobFireResult()

    wrapped = dispatch.wrap(_body)
    await wrapped(_trigger(kind="backfill:g"), uuid4())
    await asyncio.sleep(0.02)
    await wrapped(_trigger(kind="poll:g"), uuid4())
    await asyncio.sleep(0.02)
    queued.set()
    await asyncio.wait_for(dispatch.join(), timeout=1)

    assert [row["error"] for row in fires.failed] == ["upstream down"]
    assert len(fires.succeeded) == 1


async def test_a_cancelled_holder_hands_the_turn_on() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(fires, exclusion_groups={"poll:g": "g", "backfill:g": "g"})
    body = _Gated("backfill:g", "poll:g")
    wrapped = dispatch.wrap(body)
    await wrapped(_trigger(kind="backfill:g"), uuid4())
    await body.started["backfill:g"].wait()
    await wrapped(_trigger(kind="poll:g"), uuid4())
    await asyncio.sleep(0.02)
    [holder] = [task for task in asyncio.all_tasks() if task.get_name() == "scheduled-jobs:backfill:g"]

    holder.cancel()
    await asyncio.wait_for(body.started["poll:g"].wait(), timeout=1)
    body.release["poll:g"].set()
    await dispatch.join()

    assert [row["error"] for row in fires.failed] == ["backfill:g: cancelled before the fire finished"]
    assert len(fires.succeeded) == 1


async def test_a_fire_that_times_out_hands_the_turn_on() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(fires, fire_timeout_seconds=0.05, exclusion_groups={"poll:g": "g", "backfill:g": "g"})

    async def _body(trigger: JobTrigger, _f: UUID) -> JobFireResult:
        if trigger.kind == "backfill:g":
            await asyncio.sleep(10)
        return JobFireResult()

    wrapped = dispatch.wrap(_body)
    await wrapped(_trigger(kind="backfill:g"), uuid4())
    await wrapped(_trigger(kind="poll:g"), uuid4())
    await asyncio.wait_for(dispatch.join(), timeout=2)

    assert [row["error"] for row in fires.failed] == ["backfill:g: fire exceeded its limit (0.05s) and was cancelled"]
    assert len(fires.succeeded) == 1


async def test_aclose_cancels_a_fire_waiting_for_its_turn_and_records_why() -> None:
    fires = _FakeFireStore()
    dispatch = _dispatch(fires, exclusion_groups={"poll:g": "g", "backfill:g": "g"})
    body = _Gated("poll:g", "backfill:g")
    wrapped = dispatch.wrap(body)
    running, waiting = uuid4(), uuid4()
    await wrapped(_trigger(kind="poll:g"), running)
    await body.started["poll:g"].wait()
    await wrapped(_trigger(kind="backfill:g"), waiting)
    await asyncio.sleep(0.02)

    await dispatch.aclose()

    assert not body.started["backfill:g"].is_set()
    assert {row["fire_id"]: row["error"] for row in fires.failed} == {
        running: "poll:g: cancelled because BackgroundDispatch was closed before the fire finished",
        waiting: "backfill:g: cancelled because BackgroundDispatch was closed before the fire finished",
    }
    assert dispatch.in_flight == {}


async def test_a_turn_is_free_again_after_a_waiter_is_cancelled() -> None:
    """Cancelling a fire while it waits must not leave the group's turn held or its holder stale."""
    fires = _FakeFireStore()
    dispatch = _dispatch(fires, exclusion_groups={"poll:g": "g", "backfill:g": "g", "backfill:g_old": "g"})
    body = _Gated("poll:g", "backfill:g", "backfill:g_old")
    wrapped = dispatch.wrap(body)
    await wrapped(_trigger(kind="poll:g"), uuid4())
    await body.started["poll:g"].wait()
    await wrapped(_trigger(kind="backfill:g"), uuid4())
    await asyncio.sleep(0.02)
    [waiter] = [task for task in asyncio.all_tasks() if task.get_name() == "scheduled-jobs:backfill:g"]
    waiter.cancel()
    await asyncio.sleep(0.02)

    body.release["poll:g"].set()
    await wrapped(_trigger(kind="backfill:g_old"), uuid4())
    await asyncio.wait_for(body.started["backfill:g_old"].wait(), timeout=1)
    body.release["backfill:g_old"].set()
    await dispatch.join()
    assert not body.started["backfill:g"].is_set()
    assert fires.failed[0]["error"] == "backfill:g: cancelled before the fire finished"


async def test_a_fire_waiting_for_its_turn_is_logged_with_the_kind_holding_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dispatch = _dispatch(_FakeFireStore(), exclusion_groups={"poll:g": "g", "backfill:g": "g"})
    body = _Gated("poll:g", "backfill:g")
    wrapped = dispatch.wrap(body)
    caplog.set_level("INFO")
    await wrapped(_trigger(kind="poll:g"), uuid4())
    await body.started["poll:g"].wait()
    await wrapped(_trigger(kind="backfill:g"), uuid4())
    await asyncio.sleep(0.02)

    [record] = [r for r in caplog.records if r.getMessage() == EVENT_FIRE_WAITING_EXCLUSION_GROUP]
    detail = record.extra_data  # type: ignore[attr-defined]
    assert (detail["kind"], detail["group"], detail["holder_kind"]) == ("backfill:g", "g", "poll:g")

    for event in body.release.values():
        event.set()
    await dispatch.join()


async def test_the_waiting_event_names_whoever_holds_the_turn_now(caplog: pytest.LogCaptureFixture) -> None:
    dispatch = _dispatch(_FakeFireStore(), exclusion_groups={"a:g": "g", "b:g": "g", "c:g": "g"})
    body = _Gated("a:g", "b:g", "c:g")
    wrapped = dispatch.wrap(body)
    caplog.set_level("INFO")
    await wrapped(_trigger(kind="a:g"), uuid4())
    await body.started["a:g"].wait()
    await wrapped(_trigger(kind="b:g"), uuid4())
    await asyncio.sleep(0.02)
    body.release["a:g"].set()
    await body.started["b:g"].wait()
    await wrapped(_trigger(kind="c:g"), uuid4())
    await asyncio.sleep(0.02)

    waits = [
        (r.extra_data["kind"], r.extra_data["holder_kind"])  # type: ignore[attr-defined]
        for r in caplog.records
        if r.getMessage() == EVENT_FIRE_WAITING_EXCLUSION_GROUP
    ]
    assert waits == [("b:g", "a:g"), ("c:g", "b:g")]
    for event in body.release.values():
        event.set()
    await dispatch.join()


async def test_a_fire_arriving_while_the_turn_passes_still_logs_its_wait(caplog: pytest.LogCaptureFixture) -> None:
    """Releasing the holder and handing off a new fire in one synchronous step makes the newcomer
    check the group after the holder has released the turn but before the next in line has taken
    it -- the moment a lock reads free although a fire is queued. The newcomer still waits, so it
    must still be logged."""
    dispatch = _dispatch(_FakeFireStore(), exclusion_groups={"a:g": "g", "b:g": "g", "c:g": "g"})
    body = _Gated("a:g", "b:g", "c:g")
    wrapped = dispatch.wrap(body)
    caplog.set_level("INFO")
    await wrapped(_trigger(kind="a:g"), uuid4())
    await body.started["a:g"].wait()
    await wrapped(_trigger(kind="b:g"), uuid4())
    await asyncio.sleep(0.02)

    body.release["a:g"].set()
    await wrapped(_trigger(kind="c:g"), uuid4())  # hands off at once; no await in between
    await body.started["b:g"].wait()
    await asyncio.sleep(0.02)

    waiting = [
        r.extra_data["kind"]  # type: ignore[attr-defined]
        for r in caplog.records
        if r.getMessage() == EVENT_FIRE_WAITING_EXCLUSION_GROUP
    ]
    assert waiting == ["b:g", "c:g"]
    for event in body.release.values():
        event.set()
    await dispatch.join()


async def test_the_group_queue_empties_after_a_waiter_gives_up(caplog: pytest.LogCaptureFixture) -> None:
    """A waiter that runs out of time leaves the group's queue. Once the holder is done too, the
    next fire finds nobody ahead of it and must not be logged as waiting."""
    fires = _FakeFireStore()
    dispatch = _dispatch(
        fires,
        fire_timeout_seconds_by_kind={"poll:k": 20, "backfill:k": 20},
        config=_Config(by_kind={"poll:k": 60, "backfill:k": 60}),
        exclusion_groups={"poll:k": "k", "backfill:k": "k"},
    )
    body = _Gated("poll:k", "backfill:k")
    wrapped = dispatch.wrap(body)
    await wrapped(_trigger(kind="poll:k"), uuid4())
    await body.started["poll:k"].wait()
    await wrapped(_trigger(kind="backfill:k", fired_ago=29.8), uuid4())
    for _ in range(100):
        if fires.failed:
            break
        await asyncio.sleep(0.02)
    body.release["poll:k"].set()
    await dispatch.join()

    caplog.clear()
    caplog.set_level("INFO")
    body.release["backfill:k"].set()
    await wrapped(_trigger(kind="backfill:k"), uuid4())
    await dispatch.join()
    assert not [r for r in caplog.records if r.getMessage() == EVENT_FIRE_WAITING_EXCLUSION_GROUP]
    assert body.started["backfill:k"].is_set()


async def test_the_group_queue_empties_after_a_waiter_is_cancelled(caplog: pytest.LogCaptureFixture) -> None:
    """The same for a waiter cancelled while it waits."""
    dispatch = _dispatch(_FakeFireStore(), exclusion_groups={"poll:g": "g", "backfill:g": "g"})
    body = _Gated("poll:g", "backfill:g")
    wrapped = dispatch.wrap(body)
    await wrapped(_trigger(kind="poll:g"), uuid4())
    await body.started["poll:g"].wait()
    await wrapped(_trigger(kind="backfill:g"), uuid4())
    await asyncio.sleep(0.02)
    [waiter] = [task for task in asyncio.all_tasks() if task.get_name() == "scheduled-jobs:backfill:g"]
    waiter.cancel()
    await asyncio.sleep(0.02)
    body.release["poll:g"].set()
    await dispatch.join()

    caplog.clear()
    caplog.set_level("INFO")
    body.release["backfill:g"].set()
    await wrapped(_trigger(kind="backfill:g"), uuid4())
    await dispatch.join()
    assert not [r for r in caplog.records if r.getMessage() == EVENT_FIRE_WAITING_EXCLUSION_GROUP]
    assert body.started["backfill:g"].is_set()


def test_exclusion_group_for_reports_a_kinds_group() -> None:
    dispatch = _dispatch(exclusion_groups={"poll:g": "g"})
    assert dispatch.exclusion_group_for("poll:g") == "g"
    assert dispatch.exclusion_group_for("poll:free") is None


@pytest.mark.parametrize("group", ["", None, 3])
def test_a_group_name_that_is_not_a_non_empty_string_is_refused(group: Any) -> None:
    with pytest.raises(ValueError, match="exclusion_groups"):
        _dispatch(exclusion_groups={"poll:g": group})
