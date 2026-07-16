"""Restore verification — prove a backup actually restores.

A backup you can't restore is a false comfort. :class:`RestoreVerifier` restores a backup into a
**throwaway temporary database** (never the source, never a shared schema — a real separate database
so ``pg_restore`` is fully isolated), runs assertions against it, and optionally invokes a
``post_restore_hook`` — the extension point for "spin a test stack against the restored data". The
temp database is created and dropped around the check.

Everything the verifier touches is injected: the temp-database *provisioner* (an async context
manager yielding a dsn), the *assertions*, and the *hook*. That keeps the orchestration unit-testable
with fakes; :func:`make_temp_db_provisioner` and :func:`count_tables` are the real
asyncpg-backed defaults for integration use.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse, urlunparse
from uuid import uuid7

from threetears.observe import get_logger

from threetears.backup.process import feed_stdin

__all__ = [
    "Assertions",
    "PostRestoreHook",
    "RestoreVerifier",
    "SupportsRestore",
    "TempDbProvisioner",
    "VerificationResult",
    "count_tables",
    "make_subprocess_hook",
    "make_temp_db_provisioner",
]

log = get_logger(__name__)

#: yields the dsn of a fresh, empty temporary database, and tears it down on exit.
TempDbProvisioner = Callable[[], AbstractAsyncContextManager[str]]
#: run against the restored temp dsn; returns named checks recorded on the result.
Assertions = Callable[[str], Awaitable[Mapping[str, Any]]]
#: the opt-in extension point (e.g. boot a stack against the restored dsn).
PostRestoreHook = Callable[[str], Awaitable[None]]


@runtime_checkable
class SupportsRestore(Protocol):
    """The only capability the verifier needs from the engine: restore a backup into a dsn.

    :class:`~threetears.backup.engine.BackupEngine` satisfies this structurally; segregating it
    keeps the verifier decoupled from the full engine surface.
    """

    async def restore_into(self, target_dsn: str, key: str) -> None: ...


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of verifying one backup."""

    key: str
    ok: bool
    checks: dict[str, Any] = field(default_factory=dict)
    hook_ran: bool = False


class RestoreVerifier:
    """Verify backups by restoring them into a temporary database.

    :param engine: anything that can :meth:`restore_into` a dsn (a :class:`BackupEngine`).
    :param provision_temp_db: factory yielding an async context manager over a fresh temp dsn.
    :param assertions: optional check run against the restored dsn (defaults to none).
    :param post_restore_hook: optional subprocess/stack hook run against the restored dsn.
    """

    def __init__(
        self,
        engine: SupportsRestore,
        provision_temp_db: TempDbProvisioner,
        *,
        assertions: Assertions | None = None,
        post_restore_hook: PostRestoreHook | None = None,
    ) -> None:
        self._engine = engine
        self._provision = provision_temp_db
        self._assertions = assertions
        self._hook = post_restore_hook

    async def verify(self, key: str) -> VerificationResult:
        """Restore ``key`` into a temp database, run assertions + the hook, then tear it down."""
        checks: dict[str, Any] = {}
        hook_ran = False
        async with self._provision() as temp_dsn:
            await self._engine.restore_into(temp_dsn, key)
            if self._assertions is not None:
                checks = dict(await self._assertions(temp_dsn))
            if self._hook is not None:
                await self._hook(temp_dsn)
                hook_ran = True
        log.info("restore verified", extra={"extra_data": {"key": key, "checks": checks, "hook_ran": hook_ran}})
        return VerificationResult(key=key, ok=True, checks=checks, hook_ran=hook_ran)


def _swap_database(dsn: str, database: str) -> str:
    parsed = urlparse(dsn)
    return urlunparse(parsed._replace(path=f"/{database}"))


def make_temp_db_provisioner(
    admin_dsn: str,
    *,
    connect: Callable[[str], Awaitable[Any]],
    name_prefix: str = "verify_restore_",
) -> TempDbProvisioner:
    """Build a provisioner that CREATEs a temp database and DROPs it on exit (asyncpg).

    :param admin_dsn: a dsn with rights to CREATE/DROP DATABASE (e.g. the maintenance db).
    :param connect: an async connect callable (``asyncpg.connect``); injected for testability.
    :param name_prefix: prefix for the generated temp database name.
    :return: a :data:`TempDbProvisioner`.
    """

    @asynccontextmanager
    async def provision() -> AsyncIterator[str]:
        database = f"{name_prefix}{uuid7().hex[:12]}"
        admin = await connect(admin_dsn)
        try:
            await admin.execute(f'CREATE DATABASE "{database}"')
        finally:
            await admin.close()
        try:
            yield _swap_database(admin_dsn, database)
        finally:
            cleanup = await connect(admin_dsn)
            try:
                await cleanup.execute(f'DROP DATABASE IF EXISTS "{database}"')
            finally:
                await cleanup.close()

    return provision


def count_tables(*, connect: Callable[[str], Awaitable[Any]]) -> Assertions:
    """Default assertion: the restored database has at least one user table (asyncpg).

    Counts base tables across every non-system schema (excluding ``pg_catalog`` /
    ``information_schema``), not just ``public`` — real apps put their tables in named schemas
    (scriob's live in ``platform``), so a ``public``-only check would read zero and falsely fail.

    :param connect: an async connect callable (``asyncpg.connect``); injected for testability.
    :return: an :data:`Assertions` recording ``{"tables": n}`` and asserting ``n > 0``.
    """

    async def assertion(dsn: str) -> Mapping[str, Any]:
        conn = await connect(dsn)
        try:
            count = await conn.fetchval(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_type = 'BASE TABLE' "
                "AND table_schema NOT IN ('pg_catalog', 'information_schema')"
            )
        finally:
            await conn.close()
        tables = int(count)
        if tables < 1:
            raise AssertionError("restored database has no user tables")
        return {"tables": tables}

    return assertion


def make_subprocess_hook(argv: list[str], *, dsn_env: str = "RESTORED_DATABASE_URL") -> PostRestoreHook:
    """Build a hook that runs ``argv`` with the restored dsn exported in the environment.

    The stubbed extension point for "start a test stack against the restored database": off unless a
    caller wires it in. The restored dsn is passed to the child via ``dsn_env`` (default
    ``RESTORED_DATABASE_URL``).

    :param argv: the command to run after a successful restore.
    :param dsn_env: environment variable the restored dsn is exported under.
    :return: a :data:`PostRestoreHook`.
    """

    async def hook(dsn: str) -> None:
        await feed_stdin(argv, _no_stdin(), env={**os.environ, dsn_env: dsn})

    return hook


async def _no_stdin() -> AsyncIterator[bytes]:
    """An empty byte stream — the hook command takes its input from the environment, not stdin."""
    return
    yield b""  # pragma: no cover - the bare `yield` only makes this an async generator
