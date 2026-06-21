"""Regression: ``wake_tick_job`` degrades open when the cross-pod lock infra fails.

The cross-pod ``"agent_wake_tick"`` lock is a redundant-work *optimization*, not a
correctness requirement: per-schedule mutual exclusion is the Postgres optimistic-CAS
in ``WakeScheduleCollection.claim_and_reschedule``. So a failure to ACQUIRE the lock
(``KvError`` -- bucket/stream gone, NATS unreachable) must NOT suppress the tick body,
or a single-node NATS wipe silences the wake heartbeat for hours until a process
restart (the prod incident this guards).

Since S-2 the lock control flow lives in the generic
:func:`threetears.scheduled_jobs.scheduled_tick_job` (whose own ``test_tick.py``
pins the three acquisition outcomes against fake stores). These tests pin the
SAME contract end-to-end **through ``wake_tick_job``** -- i.e. that wake's
delegation passes the ``nats_client`` + the ``"agent_wake_tick"`` lock key into
the engine so the engine's degrade-open actually protects wake:

- ``KvError`` (lock infra unavailable) -> degrade open: the due-scan STILL runs,
  ``wake_tick_job`` does not raise.
- ``LockHeld`` (another pod holds it) -> the body is skipped (no due-scan).

The schedule collection is replaced with a no-DB subclass so the test needs no
Postgres; only the lock-vs-body control flow is exercised. (The fire collection
is never reached because the due-scan returns nothing.)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from threetears.nats import LockHeld
from threetears.nats.errors import KvError

from threetears.agent.wake import tick as tick_mod
from threetears.agent.wake.collections import WakeScheduleCollection


class _NoDbScheduleCollection(WakeScheduleCollection):
    """A schedule collection whose due-scan hits no database.

    Subclasses the production collection (parity declared by subclass) and
    counts due-scans on a class attribute so the test can assert whether the
    tick body ran past the lock without holding the wake_tick_job-constructed
    instance.
    """

    due_scans: int = 0

    async def list_due_for_tick(self, now: Any, *, limit: int = 200) -> list[Any]:
        type(self).due_scans += 1
        return []


class _CtxRaisingOnEnter:
    """Async context manager whose ``__aenter__`` raises -- models a lock whose
    ACQUISITION fails (``KvError``) or is already held (``LockHeld``)."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def __aenter__(self) -> None:
        raise self._exc

    async def __aexit__(self, *_: Any) -> bool:
        return False


def _patch_lock(monkeypatch: pytest.MonkeyPatch, ctx: Any) -> None:
    """Replace ``threetears.nats.nats_distributed_lock`` (resolved by the local
    import inside the generic engine) with a factory returning ``ctx``."""

    def _factory(_client: Any, _key: str, **_kw: Any) -> Any:
        return ctx

    monkeypatch.setattr("threetears.nats.nats_distributed_lock", _factory)


@pytest.fixture(autouse=True)
def _no_db_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the schedule collection wake_tick_job constructs for the no-DB one
    and reset its due-scan counter per test."""
    _NoDbScheduleCollection.due_scans = 0
    monkeypatch.setattr(tick_mod, "WakeScheduleCollection", _NoDbScheduleCollection)


class TestTickDegradesOpenOnKvError:
    """A lock-infra failure must not suppress the tick body."""

    async def test_kverror_runs_body_anyway(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch, _CtxRaisingOnEnter(KvError("nats: no response from stream")))
        # Must NOT raise -- the KvError is degraded to a warning + run.
        await tick_mod.wake_tick_job(object(), nats_client=object(), dispatch_callback=AsyncMock())
        assert _NoDbScheduleCollection.due_scans == 1


class TestTickLockHeldSkips:
    """Another pod holding the lock skips the body (existing behavior preserved)."""

    async def test_lockheld_skips_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_lock(monkeypatch, _CtxRaisingOnEnter(LockHeld("lock already held: agent_wake_tick")))
        await tick_mod.wake_tick_job(object(), nats_client=object(), dispatch_callback=AsyncMock())
        assert _NoDbScheduleCollection.due_scans == 0


class TestTickHealthyLockRunsOnce:
    """A healthy lock runs the body exactly once."""

    async def test_healthy_lock_runs_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _CtxHealthy:
            async def __aenter__(self) -> None:
                return None

            async def __aexit__(self, *_: Any) -> bool:
                return False

        _patch_lock(monkeypatch, _CtxHealthy())
        await tick_mod.wake_tick_job(object(), nats_client=object(), dispatch_callback=AsyncMock())
        assert _NoDbScheduleCollection.due_scans == 1
