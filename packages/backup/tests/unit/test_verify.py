"""Unit tests for RestoreVerifier + its asyncpg-backed defaults (fakes; no database)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from threetears.backup.verify import (
    RestoreVerifier,
    count_public_tables,
    make_subprocess_hook,
    make_temp_db_provisioner,
)

_TEMP_DSN = "postgresql://tmp/verify_target"


# parity-with: threetears.backup.verify.SupportsRestore
class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def restore_into(self, target_dsn: str, key: str) -> None:
        self.calls.append((target_dsn, key))


class _FailingEngine:
    async def restore_into(self, target_dsn: str, key: str) -> None:
        raise RuntimeError("restore blew up")


def _provisioner(lifecycle: list[str], *, dsn: str = _TEMP_DSN):
    @asynccontextmanager
    async def provision() -> AsyncIterator[str]:
        lifecycle.append("enter")
        try:
            yield dsn
        finally:
            lifecycle.append("exit")

    return provision


@pytest.mark.asyncio
async def test_verify_restores_asserts_and_runs_hook() -> None:
    lifecycle: list[str] = []
    engine = _FakeEngine()
    hook_seen: list[str] = []

    async def assertions(dsn: str) -> Mapping[str, Any]:
        return {"public_tables": 3, "dsn": dsn}

    async def hook(dsn: str) -> None:
        hook_seen.append(dsn)

    verifier = RestoreVerifier(engine, _provisioner(lifecycle), assertions=assertions, post_restore_hook=hook)  # type: ignore[arg-type]
    result = await verifier.verify("backups/k.enc")

    assert result.ok is True
    assert result.hook_ran is True
    assert result.checks == {"public_tables": 3, "dsn": _TEMP_DSN}
    assert engine.calls == [(_TEMP_DSN, "backups/k.enc")]  # restored into the temp db
    assert hook_seen == [_TEMP_DSN]
    assert lifecycle == ["enter", "exit"]  # temp db torn down


@pytest.mark.asyncio
async def test_verify_without_assertions_or_hook() -> None:
    lifecycle: list[str] = []
    verifier = RestoreVerifier(_FakeEngine(), _provisioner(lifecycle))  # type: ignore[arg-type]

    result = await verifier.verify("backups/k.enc")

    assert result.checks == {}
    assert result.hook_ran is False
    assert lifecycle == ["enter", "exit"]


@pytest.mark.asyncio
async def test_temp_db_is_torn_down_even_when_restore_fails() -> None:
    lifecycle: list[str] = []
    verifier = RestoreVerifier(_FailingEngine(), _provisioner(lifecycle))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="restore blew up"):
        await verifier.verify("backups/k.enc")

    assert lifecycle == ["enter", "exit"]  # provisioner still cleaned up


# parity-exempt: asyncpg.Connection stand-in exercising only execute()/close(); mirroring the full Connection surface would be misleading, not safer
class _FakeAdmin:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def execute(self, sql: str) -> None:
        self._log.append(sql)

    async def close(self) -> None:
        self._log.append("close")


@pytest.mark.asyncio
async def test_make_temp_db_provisioner_creates_then_drops() -> None:
    log: list[str] = []

    async def connect(dsn: str) -> _FakeAdmin:
        return _FakeAdmin(log)

    provision = make_temp_db_provisioner("postgresql://admin@h/postgres", connect=connect, name_prefix="vr_")
    async with provision() as temp_dsn:
        assert temp_dsn.startswith("postgresql://admin@h/vr_")

    assert any(s.startswith('CREATE DATABASE "vr_') for s in log)
    assert any(s.startswith('DROP DATABASE IF EXISTS "vr_') for s in log)


# parity-exempt: asyncpg.Connection stand-in exercising only fetchval()/close(); the full Connection surface is out of scope for a table-count assertion
class _FakeConn:
    def __init__(self, value: int) -> None:
        self._value = value

    async def fetchval(self, query: str) -> int:
        return self._value

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_count_public_tables_reports_and_guards() -> None:
    async def connect_ok(dsn: str) -> _FakeConn:
        return _FakeConn(4)

    assert await count_public_tables(connect=connect_ok)("dsn") == {"public_tables": 4}

    async def connect_empty(dsn: str) -> _FakeConn:
        return _FakeConn(0)

    with pytest.raises(AssertionError, match="no public tables"):
        await count_public_tables(connect=connect_empty)("dsn")


@pytest.mark.asyncio
async def test_subprocess_hook_exports_dsn_to_the_child(tmp_path: Path) -> None:
    sink = tmp_path / "hook.out"
    hook = make_subprocess_hook(["sh", "-c", f'printf "%s" "$RESTORED_DATABASE_URL" > {sink}'])

    await hook("postgresql://tmp/verify_abc")

    assert sink.read_text() == "postgresql://tmp/verify_abc"
