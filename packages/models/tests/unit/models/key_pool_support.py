"""Shared builders for the key pool's unit tests: a manual clock, key specs, verdicts, and a store that
can be taken down.

Every pool here runs on an injected clock and an in-memory store, with no network (KP-L1).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from threetears.models.circuit_breaker import CircuitBreakerRegistry
from threetears.models.key_pool import (
    AccountSnapshot,
    InMemoryKeyPoolStore,
    KeyPool,
    KeyPoolStore,
    KeySpec,
    Secret,
    Verdict,
    VerdictKind,
    VerdictScope,
)

PROVIDER = "anthropic"
POOL = "test-pool"
NOW = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
MODEL = "pool-test-model"
OTHER_MODEL = "pool-other-model"


class ManualClock:
    """A wall clock the test moves by hand."""

    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


class ManualMonotonic:
    """A monotonic clock for a breaker registry, moved by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class RecordingSleep:
    """An async sleep that advances a :class:`ManualClock` instead of waiting, and records each wait."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.clock.advance(seconds=seconds)


def secret_for(label: str) -> str:
    return f"sk-ant-api03-{label}-0123456789abcdef"


def spec(
    label: str,
    budget: str = "10",
    *,
    secret: str | None = None,
    priority: int | None = None,
    account: str | None = None,
) -> KeySpec:
    return KeySpec(
        Secret(secret if secret is not None else secret_for(label)),
        label,
        Decimal(budget),
        priority=priority,
        account=account,
    )


def make_pool(
    *keys: KeySpec,
    store: KeyPoolStore | None = None,
    clock: ManualClock | None = None,
    name: str = POOL,
    **settings: Any,
) -> KeyPool:
    return KeyPool(
        PROVIDER,
        name,
        keys,
        store if store is not None else InMemoryKeyPoolStore(),
        clock=clock if clock is not None else ManualClock(),
        **settings,
    )


def tripping_registry() -> tuple[CircuitBreakerRegistry, ManualMonotonic]:
    """A registry whose breakers open on one failure and recover after 30 s of its manual clock."""
    monotonic = ManualMonotonic()
    return CircuitBreakerRegistry(failure_threshold=1, recovery_timeout_seconds=30.0, clock=monotonic), monotonic


def exhausted(scope: VerdictScope = VerdictScope.ACCOUNT, until: datetime | None = None) -> Verdict:
    return Verdict(VerdictKind.EXHAUSTED, scope, may_have_billed=False, until=until)


def revoked() -> Verdict:
    return Verdict(VerdictKind.REVOKED, VerdictScope.KEY, may_have_billed=False)


def forbidden(scope: VerdictScope = VerdictScope.KEY_AND_MODEL) -> Verdict:
    return Verdict(VerdictKind.FORBIDDEN, scope, may_have_billed=False)


def rate_limited(until: datetime | None = None, scope: VerdictScope = VerdictScope.KEY) -> Verdict:
    return Verdict(VerdictKind.RATE_LIMITED, scope, may_have_billed=False, until=until)


def oversized() -> Verdict:
    return Verdict(VerdictKind.OVERSIZED, VerdictScope.KEY, may_have_billed=False)


def not_key_related(*, billed: bool) -> Verdict:
    return Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=billed)


async def serve(pool: KeyPool, model_id: str | None = MODEL, amount: str = "0.01") -> str:
    """One call that succeeds at its reserved amount; the label it ran on."""
    lease = await pool.acquire(model_id, Decimal(amount))
    await lease.settle(Decimal(amount))
    return lease.label


async def fail_on(pool: KeyPool, verdict: Verdict, model_id: str | None = MODEL) -> str:
    """One call that fails with ``verdict``; the label it ran on."""
    lease = await pool.acquire(model_id, Decimal("0.01"))
    await lease.fail(verdict)
    return lease.label


def lines(caplog: pytest.LogCaptureFixture, level: int, needle: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno == level and needle in record.getMessage()]


# parity-with: threetears.models.key_pool.store.KeyPoolStore
class FakeFlakyStore:
    """An in-memory store that can be taken down, or made to raise per operation.

    ``down`` makes every call raise ``ConnectionError``, as a store whose database is unreachable does;
    ``failing`` maps an operation to what it raises instead. ``calls`` names each operation made, and
    ``seen`` holds the ``repr`` of every argument passed in and every result handed back, for the
    secrecy test to search. ``accounts`` and ``records`` name every account and record an operation
    touched, so that test can read back everything the store holds.

    Each operation yields to the event loop first, as a store across a network does, so concurrent
    callers really interleave; the in-memory store alone never suspends.
    """

    def __init__(self) -> None:
        self.inner = InMemoryKeyPoolStore()
        self.down = False
        self.failing: dict[str, Exception] = {}
        self.calls: list[str] = []
        self.seen: list[str] = []
        self.accounts: set[str] = set()
        self.records: set[str] = set()

    async def _check(self, operation: str, *args: object) -> None:
        await asyncio.sleep(0)
        self.calls.append(operation)
        self.seen.append(repr(args))
        if self.down:
            raise ConnectionError("the ledger's database is unreachable")
        failure = self.failing.get(operation)
        if failure is not None:
            raise failure

    def _saw[T](self, result: T) -> T:
        self.seen.append(repr(result))
        return result

    async def reserve(
        self, account: str, reservation_id: str, amount: Decimal, budget: Decimal, expires_at: datetime
    ) -> bool:
        self.accounts.add(account)
        await self._check("reserve", account, reservation_id, amount, budget, expires_at)
        return self._saw(await self.inner.reserve(account, reservation_id, amount, budget, expires_at))

    async def settle(self, account: str, reservation_id: str, amount: Decimal) -> None:
        self.accounts.add(account)
        await self._check("settle", account, reservation_id, amount)
        await self.inner.settle(account, reservation_id, amount)

    async def release(self, account: str, reservation_id: str) -> None:
        self.accounts.add(account)
        await self._check("release", account, reservation_id)
        await self.inner.release(account, reservation_id)

    async def sweep_expired(self, accounts: Iterable[str], now: datetime) -> dict[str, list[str]]:
        names = list(accounts)
        self.accounts.update(names)
        await self._check("sweep_expired", names, now)
        return self._saw(await self.inner.sweep_expired(names, now))

    async def snapshot(self, accounts: Iterable[str]) -> dict[str, AccountSnapshot]:
        names = list(accounts)
        self.accounts.update(names)
        await self._check("snapshot", names)
        return self._saw(await self.inner.snapshot(names))

    async def read_record(self, name: str) -> tuple[int, dict[str, Any] | None]:
        self.records.add(name)
        await self._check("read_record", name)
        return self._saw(await self.inner.read_record(name))

    async def compare_and_set_record(
        self, name: str, expected_version: int, document: dict[str, Any]
    ) -> tuple[bool, int, dict[str, Any] | None]:
        self.records.add(name)
        await self._check("compare_and_set_record", name, expected_version, document)
        return self._saw(await self.inner.compare_and_set_record(name, expected_version, document))
