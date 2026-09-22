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
  (``fire_timeout_seconds``, per kind through ``fire_timeout_seconds_by_kind``). A timeout is
  refused at construction unless it is below the reap threshold the pump's ``JobConfig`` gives
  that kind, because otherwise the reaper would record a live fire as failed.
- **finalizes and counts** each fire itself (``finalize_success`` / ``finalize_failed``,
  ``inc_fire`` / ``inc_failure``); the tick counts nothing for a handed-off fire. A timeout names
  the kind and the limit; an exception with empty text is recorded by its type name.

If the process dies mid-fire the row stays ``'dispatching'`` and the tick's reaper records it as
failed, exactly as for a pod that dies inside an inline fire. :meth:`BackgroundDispatch.aclose`
cancels what is still running and records each such fire as failed, saying why.

Usage::

    background = BackgroundDispatch(fire_store, nats_client=nats, config=config)
    routes = {kind: background.wrap(handler) for kind, handler in handlers.items()}
    await scheduled_tick_job(schedule_store, fire_store, routes, nats_client=nats, config=config)
    ...
    await background.aclose()  # on shutdown

Time a fire spends waiting for a free slot counts against its kind's reap threshold but not
against its fire timeout; size ``max_concurrent`` so a backlog drains well inside that threshold.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import re
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from uuid import UUID

from threetears.observe import get_logger

from threetears.scheduled_jobs.config import DEFAULT_JOB_CONFIG, JobConfig, reap_after_seconds_for_kind
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
    "BackgroundDispatch",
    "in_flight_lock_key",
]

log = get_logger(__name__)

#: Default cap on fire bodies running at once in one :class:`BackgroundDispatch`.
DEFAULT_MAX_CONCURRENT_FIRES: Final[int] = 8

#: Default per-fire limit, in seconds. Below the platform's 900s reap threshold with room to spare.
DEFAULT_FIRE_TIMEOUT_SECONDS: Final[float] = 300.0

#: Prefix of the cross-pod lock key a fire holds while it runs (see :func:`in_flight_lock_key`).
IN_FLIGHT_LOCK_KEY_PREFIX: Final[str] = "scheduled_jobs_in_flight."

# The JetStream KV key grammar nats-server enforces -- the same pattern core's collections check
# their L2 key bodies against. A key outside it is refused with JetStream.InvalidKeyError.
_KV_KEY_GRAMMAR: Final = re.compile(r"^[-/_=.a-zA-Z0-9]+$")

#: Output key marking a fire recorded as skipped because its kind was still running. A consumer
#: that measures progress from fire outputs should leave these rows out of that measurement.
IN_FLIGHT_SKIP_OUTPUT_KEY: Final[str] = "skipped"

_SKIPPED_HERE: Final[str] = "previous fire still in flight"
_SKIPPED_ELSEWHERE: Final[str] = "fire in flight on another pod"

# The same bounded failure reason the tick records for a failed handler (metrics.py).
_HANDLER_EXCEPTION: Final[str] = "handler_exception"


class BackgroundDispatch:
    """Hand each fire off from the tick and run it as its own task (see the module docstring).

    One instance serves every kind a pump routes through it: the concurrency cap and the
    one-fire-per-kind rule are shared across all the callbacks it wraps.
    """

    def __init__(
        self,
        fire_store: FireStore,
        *,
        nats_client: KvCapable | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_FIRES,
        fire_timeout_seconds: float = DEFAULT_FIRE_TIMEOUT_SECONDS,
        fire_timeout_seconds_by_kind: Mapping[str, float] | None = None,
        config: JobConfig = DEFAULT_JOB_CONFIG,
        emitter: ScheduledJobsMetricsEmitter | None = None,
    ) -> None:
        """Build the dispatcher; validate its limits against the pump's reap thresholds.

        :param fire_store: the same fire store the pump uses; background fires finalize through it
        :ptype fire_store: FireStore
        :param nats_client: :class:`threetears.nats.NatsClient` for the cross-pod in-flight lock,
            or ``None`` for a single pod (the one-fire-per-kind rule then holds in-process only)
        :ptype nats_client: KvCapable | None
        :param max_concurrent: most fire bodies running at once
        :ptype max_concurrent: int
        :param fire_timeout_seconds: per-fire limit for kinds without their own
        :ptype fire_timeout_seconds: float
        :param fire_timeout_seconds_by_kind: per-kind limits overriding ``fire_timeout_seconds``
        :ptype fire_timeout_seconds_by_kind: Mapping[str, float] | None
        :param config: the pump's operational config; its reap thresholds bound the timeouts
        :ptype config: JobConfig
        :param emitter: metrics emitter; defaults to the process-wide scheduled-jobs emitter
        :ptype emitter: ScheduledJobsMetricsEmitter | None
        :raises ValueError: when a limit is not positive, or a fire timeout is not below the reap
            threshold of the kinds it applies to
        """
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be at least 1, got {max_concurrent}")
        if fire_timeout_seconds <= 0:
            raise ValueError(f"fire_timeout_seconds must be positive, got {fire_timeout_seconds:g}")
        by_kind = dict(fire_timeout_seconds_by_kind or {})
        for kind, seconds in by_kind.items():
            if seconds <= 0:
                raise ValueError(f"fire_timeout_seconds_by_kind[{kind!r}] must be positive, got {seconds:g}")
        _check_timeouts_fit_reap(fire_timeout_seconds, by_kind, config)

        self._fire_store = fire_store
        self._nats_client = nats_client
        self._slots = asyncio.Semaphore(max_concurrent)
        self._default_timeout = float(fire_timeout_seconds)
        self._timeouts_by_kind: Mapping[str, float] = MappingProxyType(by_kind)
        self._emitter = emitter
        self._in_flight: dict[str, UUID] = {}
        self._pending: dict[UUID, tuple[JobTrigger, float]] = {}
        self._tasks: dict[UUID, asyncio.Task[None]] = {}
        self._closed = False

    @property
    def in_flight(self) -> Mapping[str, UUID]:
        """Kind -> fire id of every fire this instance is running now (a snapshot)."""
        return MappingProxyType(dict(self._in_flight))

    def fire_timeout_for(self, kind: str) -> float:
        """The limit a fire of ``kind`` runs under.

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
        """Wait until every fire handed off so far (and any handed off meanwhile) is finalized."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)

    async def aclose(self) -> None:
        """Stop: refuse new fires, cancel the running ones, and record each as failed."""
        self._closed = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # A task cancelled before it ever started never reached its own handler.
        for fire_id, (trigger, started) in list(self._pending.items()):
            await self._finalize(trigger, fire_id, self._cancelled(trigger, started))

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
        if self._in_flight.get(kind) == fire_id:
            del self._in_flight[kind]

    async def _run(self, callback: DispatchCallback, trigger: JobTrigger, fire_id: UUID) -> None:
        # local import: the same optional-NATS stance as the tick engine's own lock import.
        from threetears.nats import LockHeld, nats_distributed_lock  # noqa: PLC0415
        from threetears.nats.errors import KvError  # noqa: PLC0415

        result: JobFireResult | None = None
        try:
            try:
                async with nats_distributed_lock(self._nats_client, in_flight_lock_key(trigger.kind)):
                    result = await self._run_body(callback, trigger, fire_id)
            except LockHeld:
                _log_skip(trigger, fire_id, _SKIPPED_ELSEWHERE)
                result = JobFireResult(status="succeeded", output={IN_FLIGHT_SKIP_OUTPUT_KEY: _SKIPPED_ELSEWHERE})
            except KvError as exc:
                # The lock only saves duplicate work across pods; losing it must not silence the
                # fire (the tick lock takes the same stance). If the body already ran, the lock's
                # release failed and the result stands.
                log.warning(
                    "scheduled_jobs background: in-flight lock unavailable; running the fire without it",
                    extra={"extra_data": {"kind": trigger.kind, "error_type": type(exc).__name__, "error": str(exc)}},
                )
                if result is None:
                    result = await self._run_body(callback, trigger, fire_id)
        except asyncio.CancelledError:
            started = self._pending.get(fire_id, (trigger, time.monotonic()))[1]
            await self._finalize(trigger, fire_id, self._cancelled(trigger, started))
            raise
        # Every path above sets a result; a body that itself handed the fire off keeps its own owner.
        if result is not None and not result.handed_off:
            await self._finalize(trigger, fire_id, result)

    async def _run_body(self, callback: DispatchCallback, trigger: JobTrigger, fire_id: UUID) -> JobFireResult:
        limit = self.fire_timeout_for(trigger.kind)
        async with self._slots:
            started = time.monotonic()
            deadline = asyncio.timeout(limit)
            try:
                async with deadline:
                    result = await callback(trigger, fire_id)
            except TimeoutError as exc:
                if deadline.expired():
                    message = f"{trigger.kind}: fire exceeded its {limit:g}s limit and was cancelled"
                else:
                    message = str(exc) or type(exc).__name__
                result = JobFireResult(status="failed", error=message)
            except Exception as exc:  # noqa: BLE001 - boundary: a background fire's failure is its failed row
                log.exception(
                    EVENT_FIRE_FAILED,
                    extra={
                        "extra_data": {
                            "job_id": str(trigger.job_id),  # convert at border: log extra_data field
                            "fire_id": str(fire_id),  # convert at border: log extra_data field
                            "kind": trigger.kind,
                            "error_type": type(exc).__name__,
                        }
                    },
                )
                result = JobFireResult(status="failed", error=str(exc) or type(exc).__name__)
            if result.latency_ms is None and not result.handed_off:
                result = dataclasses.replace(result, latency_ms=_elapsed_ms(started))
        return result

    def _cancelled(self, trigger: JobTrigger, started: float) -> JobFireResult:
        reason = (
            "cancelled because BackgroundDispatch was closed before the fire finished"
            if self._closed
            else "cancelled before the fire finished"
        )
        return JobFireResult(status="failed", error=f"{trigger.kind}: {reason}", latency_ms=_elapsed_ms(started))

    async def _finalize(self, trigger: JobTrigger, fire_id: UUID, result: JobFireResult) -> None:
        if self._pending.pop(fire_id, None) is None:
            return
        try:
            if result.status == "failed":
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
        if result.status == "failed":
            emitter.inc_failure(reason=_HANDLER_EXCEPTION)
        log.info(
            EVENT_FIRE_COMPLETED,
            extra={
                "extra_data": {
                    "fire_id": str(fire_id),  # convert at border: log extra_data field
                    "kind": trigger.kind,
                    "status": result.status,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                }
            },
        )


def _check_timeouts_fit_reap(default_timeout: float, by_kind: Mapping[str, float], config: JobConfig) -> None:
    """Refuse a fire timeout that would let the reaper record a live fire as failed.

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
        if timeout >= reap:
            raise ValueError(
                f"the {timeout:g}s fire timeout for kind {kind!r} is not below its {reap}s reap threshold, so the "
                "reaper would record the live fire as failed; lower the timeout or raise "
                f"dispatch_reap_after_seconds_by_kind[{kind!r}]"
            )
    if default_timeout >= config.dispatch_reap_after_seconds:
        raise ValueError(
            f"fire_timeout_seconds ({default_timeout:g}s) is not below the {config.dispatch_reap_after_seconds}s "
            "reap threshold, so the reaper would record a live fire as failed; lower it or raise "
            "dispatch_reap_after_seconds"
        )


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
