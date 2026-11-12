"""a test's schema drop retries YugabyteDB's concurrent-catalog-write conflict (``threetears.core.testing.schemas``).

memory's YugabyteDB run failed a passing test at teardown: its ``DROP SCHEMA ... CASCADE`` met another session's DDL
and YugabyteDB answered ``SerializationError`` (kConflict), which it asks the client to retry (S/o19/o32-mem-tail-yb.raw).
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from threetears.core.testing.schemas import drop_schema


class _Conn:
    """answers ``conflicts`` drops with the conflict, then each after with ``then`` (None: the drop succeeds)."""

    def __init__(self, conflicts: int, then: BaseException | None = None) -> None:
        self.conflicts = conflicts
        self.then = then
        self.sent: list[str] = []

    async def execute(self, sql: str, *params: Any) -> str:
        self.sent.append(sql)
        if len(self.sent) <= self.conflicts:
            raise asyncpg.SerializationError("conflict with concurrently committed data")
        if self.then is not None:
            raise self.then
        return "DROP SCHEMA"


async def test_a_conflicting_drop_is_retried_until_it_lands() -> None:
    """Flips if a drop YugabyteDB refused for a concurrent catalog write is not tried again (the teardown error)."""
    conn = _Conn(conflicts=2)
    await drop_schema(conn, "mem_it_1", first_wait=0)
    assert conn.sent == ['DROP SCHEMA IF EXISTS "mem_it_1" CASCADE'] * 3


async def test_a_drop_that_always_conflicts_raises_after_its_attempts() -> None:
    """Flips if a conflict that never clears is swallowed (the schema would be left behind unsaid)."""
    conn = _Conn(conflicts=99)
    with pytest.raises(asyncpg.SerializationError):
        await drop_schema(conn, "mem_it_2", attempts=3, first_wait=0)
    assert len(conn.sent) == 3


async def test_any_other_error_is_raised_at_once() -> None:
    """Flips if an error that is not the conflict (a missing privilege) is retried or swallowed."""
    conn = _Conn(conflicts=0, then=asyncpg.InsufficientPrivilegeError("must be owner of schema"))
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        await drop_schema(conn, "mem_it_3", first_wait=0)
    assert len(conn.sent) == 1
