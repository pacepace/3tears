"""Unit tests for :func:`threetears.agent.wake.tick.wake_tick_job` lock handling.

The cross-pod ``"agent_wake_tick"`` lock is a redundant-work *optimization*,
not a correctness requirement: per-schedule mutual exclusion is the Postgres
optimistic-CAS in ``WakeScheduleCollection.claim_and_reschedule``. So a failure
to ACQUIRE the lock (``KvError`` -- bucket/stream gone, NATS unreachable) must
NOT suppress the tick body, or a single-node NATS wipe silences the wake
heartbeat for hours until a process restart.

These tests pin the three acquisition outcomes:

- ``KvError`` (lock infra unavailable) -> degrade open: tick body STILL runs,
  ``wake_tick_job`` does not raise. (regression for the prod incident where a
  JetStream wipe killed every tick)
- ``LockHeld`` (another pod holds it) -> skip the body, no exception.
- healthy lock -> body runs exactly once, inside the lock.

``_run_tick_body`` is replaced with an ``AsyncMock`` so these cases need no
database -- they assert only the lock-vs-body control flow.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from threetears.agent.wake import tick as tick_mod
from threetears.nats import LockHeld
from threetears.nats.errors import KvError


class _CtxRaisingOnEnter:
    """Async context manager whose ``__aenter__`` raises -- models a lock
    whose ACQUISITION fails (``KvError``) or is already held (``LockHeld``)."""

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
    local import inside ``wake_tick_job``) with a factory returning ``ctx``."""

    def _factory(_client: Any, _key: str, **_kw: Any) -> Any:
        return ctx

    monkeypatch.setattr("threetears.nats.nats_distributed_lock", _factory)


class TestTickDegradesOpenOnKvError:
    """A lock-infra failure must not suppress the tick body."""

    async def test_kverror_runs_body_anyway(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run_body = AsyncMock()
        monkeypatch.setattr(tick_mod, "_run_tick_body", run_body)
        _patch_lock(monkeypatch, _CtxRaisingOnEnter(KvError("nats: no response from stream")))

        pool = object()
        callback = AsyncMock()
        # Must NOT raise -- the KvError is degraded to a warning + run.
        await tick_mod.wake_tick_job(pool, nats_client=object(), dispatch_callback=callback)

        run_body.assert_awaited_once_with(pool, callback)


class TestTickLockHeldSkips:
    """Another pod holding the lock skips the body (existing behavior preserved)."""

    async def test_lockheld_skips_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run_body = AsyncMock()
        monkeypatch.setattr(tick_mod, "_run_tick_body", run_body)
        _patch_lock(monkeypatch, _CtxRaisingOnEnter(LockHeld("lock already held: agent_wake_tick")))

        await tick_mod.wake_tick_job(object(), nats_client=object(), dispatch_callback=AsyncMock())

        run_body.assert_not_awaited()


class TestTickHealthyLockRunsOnce:
    """A healthy lock runs the body exactly once."""

    async def test_healthy_lock_runs_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run_body = AsyncMock()
        monkeypatch.setattr(tick_mod, "_run_tick_body", run_body)
        _patch_lock(monkeypatch, _CtxHealthy())

        pool = object()
        callback = AsyncMock()
        await tick_mod.wake_tick_job(pool, nats_client=object(), dispatch_callback=callback)

        run_body.assert_awaited_once_with(pool, callback)
