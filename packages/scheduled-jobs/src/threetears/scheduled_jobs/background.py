"""Run each scheduled fire as a task of its own, so one slow fire no longer holds up the tick.

:func:`~threetears.scheduled_jobs.tick.scheduled_tick_job` awaits every dispatch callback
inline, so a tick lasts as long as all its fires together. A pump with a few slow kinds (a
five-minute backfill, a websocket listened to for three) makes every other kind wait behind
them: measured in a consumer routing 52 kinds, ticks landed every ~18 minutes, and a kind
scheduled every 60 seconds fired 36 times in 11 hours.

:class:`BackgroundDispatch` wraps a pump's callbacks. The wrapped callback hands the fire off
(:attr:`JobFireResult.handed_off`, so the tick leaves the fire row ``'dispatching'`` and moves
on) and runs the real callback as its own task, which then finalizes the row with the real
outcome. Along the way it:

- keeps **one fire per kind in flight**. In this process, a kind that is still running records
  its next fire as a success whose output says it was skipped and names the fire in flight.
  Across pods, each fire holds :func:`threetears.nats.nats_distributed_lock` on
  :func:`in_flight_lock_key` of its kind for as long as it runs -- the job the tick lock used to do
  when fires ran inside it -- and a held lock records the same kind of skip. ``nats_client=None``
  is single-pod, exactly as for the tick.
- **bounds concurrency** (``max_concurrent``) and **each fire's duration**
  (``fire_timeout_seconds``, per kind through ``fire_timeout_seconds_by_kind``).
- **never lets the reaper take a live fire.** The reaper counts from the tick, so a fire's body
  runs under the smaller of its own timeout and the time left before its kind's reap threshold
  (less a margin); a fire that waited so long for a free slot that none is left is recorded as
  failed without running. At construction, a timeout that could not fit under that threshold is
  refused outright.
- **finalizes and counts** each fire exactly once (``finalize_success`` / ``finalize_failed``,
  ``inc_fire`` / ``inc_failure``); the tick counts nothing for a handed-off fire. A timeout names
  the kind and the limit, and an exception with empty text is recorded by its type name. The
  finalize write runs as its own shielded task, so cancelling a fire cannot cut its record short.

If the process dies mid-fire the row stays ``'dispatching'`` and the tick's reaper records it as
failed, exactly as for a pod that dies inside an inline fire. :meth:`BackgroundDispatch.aclose`
cancels what is still running and records each such fire as failed, saying why; a fire whose
body had already finished keeps its real result.

Usage (pass the SAME config to both, so the reap thresholds agree)::

    background = BackgroundDispatch(fire_store, config=config, nats_client=nats)
    routes = {kind: background.wrap(handler) for kind, handler in handlers.items()}
    await scheduled_tick_job(schedule_store, fire_store, routes, nats_client=nats, config=config)
    ...
    await background.aclose()  # on shutdown
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from uuid import UUID

from threetears.observe import get_logger

from threetears.scheduled_jobs.config import JobConfig, reap_after_seconds_for_kind
from threetears.scheduled_jobs.events import (
    EVENT_FIRE_COMPLETED,
    EVENT_FIRE_FAILED,
    EVENT_FIRE_SKIPPED_IN_FLIGHT,
)
from threetears.scheduled_jobs.metrics import ScheduledJobsMetricsEmitter, get_scheduled_jobs_emitter
from threetears.scheduled_jobs.protocols import FireStore
from threetears.scheduled_jobs.tick import DispatchCallback
from threetears.scheduled_jobs.types import JobFireResult, JobTrigger

if TYPE_CHECKING:
    from threetears.nats.kv import KvCapable

__all__ = [
    "DEFAULT_FIRE_TIMEOUT_SECONDS",
    "DEFAULT_MAX_CONCURRENT_FIRES",
    "IN_FLIGHT_LOCK_KEY_PREFIX",
    "IN_FLIGHT_SKIP_OUTPUT_KEY",
    "REAP_MARGIN_SECONDS",
    "BackgroundDispatch",
    "in_flight_lock_key",
]

log = get_logger(__name__)

#: Default cap on fire bodies running at once in one :class:`BackgroundDispatch`.
DEFAULT_MAX_CONCURRENT_FIRES: Final[int] = 8

#: Default per-fire limit, in seconds. Well inside the platform's 900s reap threshold.
DEFAULT_FIRE_TIMEOUT_SECONDS: Final[float] = 300.0

#: How long before its kind's reap threshold a fire is made to stop, so it is finalized before the
#: reaper (which counts from the tick) could take it. Also the headroom the construction check
#: demands between a fire timeout and that threshold.
REAP_MARGIN_SECONDS: Final[float] = 30.0

#: Prefix of the cross-pod lock key a fire holds while it runs (see :func:`in_flight_lock_key`).
IN_FLIGHT_LOCK_KEY_PREFIX: Final[str] = "scheduled_jobs_in_flight."

#: Output key marking a fire recorded as skipped because its kind was still running. A consumer
#: that measures progress from fire outputs should leave these rows out of that measurement.
IN_FLIGHT_SKIP_OUTPUT_KEY: Final[str] = "skipped"

# The JetStream KV key grammar nats-server enforces -- the same pattern core's collections check
# their L2 key bodies against. A key outside it is refused with JetStream.InvalidKeyError.
_KV_KEY_GRAMMAR: Final = re.compile(r"^[-/_=.a-zA-Z0-9]+$")

_SKIPPED_HERE: Final[str] = "previous fire still in flight"
_SKIPPED_ELSEWHERE: Final[str] = "fire in flight on another pod"

# Bounded failure reasons (metrics.py documents the set).
_HANDLER_EXCEPTION: Final[str] = "handler_exception"
_TIMEOUT: Final[str] = "timeout"
_CANCELLED: Final[str] = "cancelled"
_OTHER: Final[str] = "other"


@dataclasses.dataclass(frozen=True)
class _Outcome:
    """A fire's result plus the bounded reason to count it under if it failed."""

    result: JobFireResult
    failure_reason: str = _HANDLER_EXCEPTION


class BackgroundDispatch:
    """Hand each fire off from the tick and run it as its own task (see the module docstring).

    One instance serves every kind a pump routes through it: the concurrency cap and the
    one-fire-per-kind rule are shared across all the callbacks it wraps.
    """

    def __init__(
        self,
        fire_store: FireStore,
        *,
        config: JobConfig,
        nats_client: KvCapable | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_FIRES,
        fire_timeout_seconds: float = DEFAULT_FIRE_TIMEOUT_SECONDS,
        fire_timeout_seconds_by_kind: Mapping[str, float] | None = None,
        emitter: ScheduledJobsMetricsEmitter | None = None,
    ) -> None:
        """Build the dispatcher; validate its limits against the pump's reap thresholds.

        :param fire_store: the same fire store the pump uses; background fires finalize through it
        :ptype fire_store: FireStore
        :param config: the SAME config the pump is given; its reap thresholds bound every fire
        :ptype config: JobConfig
        :param nats_client: :class:`threetears.nats.NatsClient` for the cross-pod in-flight lock,
            or ``None`` for a single pod (the one-fire-per-kind rule then holds in-process only)
        :ptype nats_client: KvCapable | None
        :param max_concurrent: most fire bodies running at once
        :ptype max_concurrent: int
        :param fire_timeout_seconds: per-fire limit for kinds without their own
        :ptype fire_timeout_seconds: float
        :param fire_timeout_seconds_by_kind: per-kind limits overriding ``fire_timeout_seconds``
        :ptype fire_timeout_seconds_by_kind: Mapping[str, float] | None
        :param emitter: metrics emitter; defaults to the process-wide scheduled-jobs emitter
        :ptype emitter: ScheduledJobsMetricsEmitter | None
        :raises ValueError: when a limit is not a positive number, or a fire timeout does not fit
            below the reap threshold (less :data:`REAP_MARGIN_SECONDS`) of the kinds it applies to
        """
        if not max_concurrent >= 1:
            raise ValueError(f"max_concurrent must be at least 1, got {max_concurrent}")
        if not fire_timeout_seconds > 0:
            raise ValueError(f"fire_timeout_seconds must be a positive number, got {fire_timeout_seconds!r}")
        by_kind = dict(fire_timeout_seconds_by_kind or {})
        for kind, seconds in by_kind.items():
            if not seconds > 0:
                raise ValueError(f"fire_timeout_seconds_by_kind[{kind!r}] must be a positive number, got {seconds!r}")
        _check_timeouts_fit_reap(fire_timeout_seconds, by_kind, config)

        self._fire_store = fire_store
        self._config = config
        self._nats_client = nats_client
        self._slots = asyncio.Semaphore(max_concurrent)
        self._default_timeout = float(fire_timeout_seconds)
        self._timeouts_by_kind: Mapping[str, float] = MappingProxyType(by_kind)
        self._emitter = emitter
        self._in_flight: dict[str, UUID] = {}
        # Every fire handed off and not yet settled: the one record of what this instance still
        # owes a row. Popped exactly once, by whichever path settles the fire.
        self._pending: dict[UUID, tuple[JobTrigger, float]] = {}
        self._started: set[UUID] = set()
        self._tasks: dict[UUID, asyncio.Task[None]] = {}
        self._writes: set[asyncio.Task[None]] = set()
        self._closed = False

    @property
    def in_flight(self) -> Mapping[str, UUID]:
        """Kind -> fire id of every fire this instance is running now (a snapshot)."""
        return MappingProxyType(dict(self._in_flight))

    def fire_timeout_for(self, kind: str) -> float:
        """The configured limit for a fire of ``kind`` (before the reap-deadline clamp).

        :param kind: the job kind
        :ptype kind: str
        :return: seconds
        :rtype: float
        """
        return self._timeouts_by_kind.get(kind, self._default_timeout)

    def wrap(self, callback: DispatchCallback) -> DispatchCallback:
        """Return a callback the pump can route to: it hands each fire off and runs ``callback``.

        :param callback: the real dispatch callback; it runs in a task of its own, under the
            fire timeout, and must return a terminal :class:`JobFireResult` or raise
        :ptype callback: DispatchCallback
        :return: the callback to put in the pump's routing table
        :rtype: DispatchCallback
        """

        async def _hand_off(trigger: JobTrigger, fire_id: UUID) -> JobFireResult:
            return self._hand_off(callback, trigger, fire_id)

        return _hand_off

    async def join(self) -> None:
        """Wait until every fire handed off so far (and any handed off meanwhile) is recorded."""
        while self._tasks or self._writes:
            await asyncio.gather(*list(self._tasks.values()), *list(self._writes), return_exceptions=True)

    async def aclose(self) -> None:
        """Stop: refuse new fires, cancel the running ones, and record each as failed.

        A fire whose body had already finished keeps its real result; a fire cancelled before its
        task ever started is recorded here, since it never reached its own handler.
        """
        self._closed = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for fire_id, (trigger, handed_off_at) in list(self._pending.items()):
            if fire_id not in self._started:
                await self._finalize(trigger, fire_id, self._cancelled(trigger, handed_off_at))
        await self.join()

    # ------------------------------------------------------------------

    def _hand_off(self, callback: DispatchCallback, trigger: JobTrigger, fire_id: UUID) -> JobFireResult:
        running = self._in_flight.get(trigger.kind)
        result: JobFireResult
        if self._closed:
            result = JobFireResult(
                status="failed", error=f"{trigger.kind}: BackgroundDispatch is closed; the fire was not run"
            )
        elif running is not None:
            _log_skip(trigger, fire_id, _SKIPPED_HERE)
            result = JobFireResult(
                status="succeeded",
                output={
                    IN_FLIGHT_SKIP_OUTPUT_KEY: _SKIPPED_HERE,
                    "in_flight_fire_id": str(running),  # convert at border: job_fires.output is JSON
                },
            )
        else:
            self._in_flight[trigger.kind] = fire_id
            self._pending[fire_id] = (trigger, time.monotonic())
            task = asyncio.get_running_loop().create_task(
                self._run(callback, trigger, fire_id), name=f"scheduled-jobs:{trigger.kind}"
            )
            self._tasks[fire_id] = task
            task.add_done_callback(functools.partial(self._forget, trigger.kind, fire_id))
            result = JobFireResult(handed_off=True)
        return result

    def _forget(self, kind: str, fire_id: UUID, _done: asyncio.Task[None]) -> None:
        self._tasks.pop(fire_id, None)
        self._started.discard(fire_id)
        if self._in_flight.get(kind) == fire_id:
            del self._in_flight[kind]

    async def _run(self, callback: DispatchCallback, trigger: JobTrigger, fire_id: UUID) -> None:
        self._started.add(fire_id)
        # local import: the same optional-NATS stance as the tick engine's own lock import.
        from threetears.nats import LockHeld, nats_distributed_lock  # noqa: PLC0415
        from threetears.nats.errors import KvError  # noqa: PLC0415

        outcome: _Outcome | None = None
        try:
            try:
                async with nats_distributed_lock(self._nats_client, in_flight_lock_key(trigger.kind)):
                    outcome = await self._run_body(callback, trigger, fire_id)
            except LockHeld:
                _log_skip(trigger, fire_id, _SKIPPED_ELSEWHERE)
                outcome = _Outcome(
                    JobFireResult(status="succeeded", output={IN_FLIGHT_SKIP_OUTPUT_KEY: _SKIPPED_ELSEWHERE})
                )
            except KvError as exc:
                # The lock only saves duplicate work across pods; losing it must not silence the
                # fire (the tick lock takes the same stance). Raised AFTER the body ran, it is the
                # lock's release that failed, and the fire's own result stands.
                log.warning(
                    "scheduled_jobs background: in-flight lock unavailable",
                    extra={
                        "extra_data": {
                            "kind": trigger.kind,
                            "body_ran": outcome is not None,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    },
                )
                if outcome is None:
                    outcome = await self._run_body(callback, trigger, fire_id)
            except Exception as exc:  # noqa: BLE001 - boundary: a lock that fails for any other reason (a misconfigured bucket, a TTL mismatch) must fail the fire loudly, not vanish with the task
                log.exception(
                    "scheduled_jobs background: the in-flight lock failed",
                    extra={"extra_data": {"kind": trigger.kind, "body_ran": outcome is not None}},
                )
                if outcome is None:
                    outcome = _Outcome(
                        JobFireResult(
                            status="failed",
                            error=(
                                f"{trigger.kind}: the in-flight lock failed "
                                f"({type(exc).__name__}: {str(exc) or 'no detail'}); the fire was not run"
                            ),
                        ),
                        _OTHER,
                    )
        except asyncio.CancelledError:
            if outcome is None:
                outcome = _Outcome(self._cancelled(trigger, self._handed_off_at(fire_id)), _CANCELLED)
            raise
        finally:
            await self._settle(trigger, fire_id, outcome)

    async def _run_body(self, callback: DispatchCallback, trigger: JobTrigger, fire_id: UUID) -> _Outcome:
        configured = self.fire_timeout_for(trigger.kind)
        async with self._slots:
            reap = reap_after_seconds_for_kind(self._config, trigger.kind)
            left = (
                trigger.fired_at + timedelta(seconds=reap - REAP_MARGIN_SECONDS) - datetime.now(UTC)
            ).total_seconds()
            outcome: _Outcome
            if not left > 0:
                outcome = _Outcome(
                    JobFireResult(
                        status="failed",
                        error=(
                            f"{trigger.kind}: waited so long for a free slot that running it now would outlive "
                            f"its {reap}s reap threshold; the fire was not run"
                        ),
                    ),
                    _TIMEOUT,
                )
            else:
                outcome = await self._call(callback, trigger, fire_id, min(configured, left), clamped=left < configured)
        return outcome

    async def _call(
        self, callback: DispatchCallback, trigger: JobTrigger, fire_id: UUID, limit: float, *, clamped: bool
    ) -> _Outcome:
        started = time.monotonic()
        deadline = asyncio.timeout(limit)
        outcome: _Outcome
        try:
            async with deadline:
                outcome = _Outcome(await callback(trigger, fire_id))
        except TimeoutError as exc:
            if deadline.expired():
                what = "the time left before its reap threshold" if clamped else "its limit"
                outcome = _Outcome(
                    JobFireResult(
                        status="failed",
                        error=f"{trigger.kind}: fire exceeded {what} ({limit:g}s) and was cancelled",
                    ),
                    _TIMEOUT,
                )
            else:
                outcome = _Outcome(JobFireResult(status="failed", error=str(exc) or type(exc).__name__))
        except Exception as exc:  # noqa: BLE001 - boundary: a background fire's failure is its failed row
            log.exception(
                "scheduled_jobs background: a fire's body raised",
                extra={
                    "extra_data": {
                        "fire_id": str(fire_id),  # convert at border: log extra_data field
                        "kind": trigger.kind,
                        "error_type": type(exc).__name__,
                    }
                },
            )
            outcome = _Outcome(JobFireResult(status="failed", error=str(exc) or type(exc).__name__))
        if outcome.result.latency_ms is None and not outcome.result.handed_off:
            outcome = dataclasses.replace(
                outcome, result=dataclasses.replace(outcome.result, latency_ms=_elapsed_ms(started))
            )
        return outcome

    def _handed_off_at(self, fire_id: UUID) -> float:
        entry = self._pending.get(fire_id)
        return entry[1] if entry is not None else time.monotonic()

    def _cancelled(self, trigger: JobTrigger, handed_off_at: float) -> JobFireResult:
        reason = (
            "cancelled because BackgroundDispatch was closed before the fire finished"
            if self._closed
            else "cancelled before the fire finished"
        )
        return JobFireResult(status="failed", error=f"{trigger.kind}: {reason}", latency_ms=_elapsed_ms(handed_off_at))

    async def _settle(self, trigger: JobTrigger, fire_id: UUID, outcome: _Outcome | None) -> None:
        if fire_id not in self._pending:
            return
        if outcome is None:
            # Only a non-Exception BaseException (KeyboardInterrupt, SystemExit) gets here.
            outcome = _Outcome(
                JobFireResult(
                    status="failed",
                    error=f"{trigger.kind}: interrupted before the fire finished",
                    latency_ms=_elapsed_ms(self._handed_off_at(fire_id)),
                ),
                _OTHER,
            )
        if outcome.result.handed_off:
            # The body handed the fire on to yet another owner, which now holds the row.
            self._pending.pop(fire_id, None)
        else:
            await self._finalize(trigger, fire_id, outcome.result, outcome.failure_reason)

    async def _finalize(
        self, trigger: JobTrigger, fire_id: UUID, result: JobFireResult, failure_reason: str = _CANCELLED
    ) -> None:
        if self._pending.pop(fire_id, None) is None:
            return
        # Its own task, shielded: cancelling the fire (aclose) must not cut the record short.
        write = asyncio.get_running_loop().create_task(
            self._write(trigger, fire_id, result, failure_reason), name=f"scheduled-jobs:finalize:{trigger.kind}"
        )
        self._writes.add(write)
        write.add_done_callback(self._writes.discard)
        await asyncio.shield(write)

    async def _write(self, trigger: JobTrigger, fire_id: UUID, result: JobFireResult, failure_reason: str) -> None:
        failed = result.status == "failed"
        try:
            if failed:
                await self._fire_store.finalize_failed(
                    trigger.partition_key,
                    fire_id,
                    error=result.error or "dispatch returned status='failed' without an error string",
                    latency_ms=result.latency_ms,
                )
            else:
                await self._fire_store.finalize_success(
                    trigger.partition_key,
                    fire_id,
                    status=result.status,
                    output=result.output,
                    latency_ms=result.latency_ms,
                )
        except Exception:  # noqa: BLE001 - boundary: a lost write leaves the row to the reaper and must not wedge the kind
            log.exception(
                "scheduled_jobs background: could not finalize the fire row; the reaper will record it as failed",
                extra={
                    "extra_data": {
                        "fire_id": str(fire_id),  # convert at border: log extra_data field
                        "kind": trigger.kind,
                        "status": result.status,
                    }
                },
            )
            return
        emitter = self._emitter if self._emitter is not None else get_scheduled_jobs_emitter()
        emitter.inc_fire(status=result.status, schedule_type=trigger.schedule_type)
        detail = {
            "job_id": str(trigger.job_id),  # convert at border: log extra_data field
            "fire_id": str(fire_id),  # convert at border: log extra_data field
            "kind": trigger.kind,
            "status": result.status,
            "latency_ms": result.latency_ms,
        }
        if failed:
            emitter.inc_failure(reason=failure_reason)
            log.error(
                EVENT_FIRE_FAILED, extra={"extra_data": {**detail, "reason": failure_reason, "error": result.error}}
            )
        log.info(EVENT_FIRE_COMPLETED, extra={"extra_data": detail})


def in_flight_lock_key(kind: str) -> str:
    """The NATS KV key a fire of ``kind`` holds while it runs.

    A kind is an arbitrary string, and a KV key may not carry a colon, a space or most other
    punctuation (``poll:gdelt`` is refused with ``JetStream.InvalidKeyError``). Same rule as core's
    collection L2 keys: a kind inside the key grammar is used as-is, anything else by its SHA-256.

    :param kind: the job kind
    :ptype kind: str
    :return: a key inside the KV key grammar, stable for the kind
    :rtype: str
    """
    body = kind if _KV_KEY_GRAMMAR.match(kind) else hashlib.sha256(kind.encode("utf-8")).hexdigest()
    return f"{IN_FLIGHT_LOCK_KEY_PREFIX}{body}"


def _check_timeouts_fit_reap(default_timeout: float, by_kind: Mapping[str, float], config: JobConfig) -> None:
    """Refuse a fire timeout that could not finish before the reaper takes the fire.

    Written as ``not (timeout < budget)`` so a NaN, which compares false both ways, is refused.

    :param default_timeout: the limit for kinds without their own
    :ptype default_timeout: float
    :param by_kind: per-kind limits
    :ptype by_kind: Mapping[str, float]
    :param config: the pump's config, carrying the reap thresholds
    :ptype config: JobConfig
    :raises ValueError: naming the kind and both numbers
    """
    for kind in sorted(set(by_kind) | set(config.dispatch_reap_after_seconds_by_kind)):
        timeout = by_kind.get(kind, default_timeout)
        reap = reap_after_seconds_for_kind(config, kind)
        if not timeout < reap - REAP_MARGIN_SECONDS:
            raise ValueError(
                f"the {timeout!r}s fire timeout for kind {kind!r} does not fit below its {reap}s reap threshold "
                f"less the {REAP_MARGIN_SECONDS:g}s margin, so the reaper could record the live fire as failed; "
                f"lower the timeout or raise dispatch_reap_after_seconds_by_kind[{kind!r}]"
            )
    reap = config.dispatch_reap_after_seconds
    if not default_timeout < reap - REAP_MARGIN_SECONDS:
        raise ValueError(
            f"fire_timeout_seconds ({default_timeout!r}s) does not fit below the {reap}s reap threshold less the "
            f"{REAP_MARGIN_SECONDS:g}s margin, so the reaper could record a live fire as failed; lower it or "
            "raise dispatch_reap_after_seconds"
        )


def _log_skip(trigger: JobTrigger, fire_id: UUID, reason: str) -> None:
    log.info(
        EVENT_FIRE_SKIPPED_IN_FLIGHT,
        extra={
            "extra_data": {
                "fire_id": str(fire_id),  # convert at border: log extra_data field
                "kind": trigger.kind,
                "reason": reason,
            }
        },
    )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
