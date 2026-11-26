"""Integration: each index of the log keeper's marks is the one its keeper's reads plan through.

The plan tests the marks' DDL names (``threetears.core.ddl.keeper_marks``: ``-- purpose:`` / ``-- plan_test:``): a
restore's read of a lost log's marks by the position index, and the window's sweep of the old marks and the newest-mark
read by the time index. The marks are seeded past where the planner would rather scan, on Postgres and YugabyteDB alike.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.log_keeper import LogKeeper
from threetears.core.testing.plans import PlanRecorder, plan_uses_index

pytestmark = pytest.mark.integration


@dataclass
class _Marks:
    conn: asyncpg.Connection
    schema: str


class _NoLog:
    """the keeper's change log, never read by the reads under test."""

    #: names no database on its records: the keeper reads this at construction (EpochGenerationSource.database)
    database = None

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the reads under test read no change log ({name})")


class _Pool:
    """one connection as the keeper's pool."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def fetch(self, query: str, *args: Any) -> Any:
        return await self._conn.fetch(query, *args)

    async def execute(self, query: str, *args: Any) -> Any:
        return await self._conn.execute(query, *args)

    def acquire(self) -> Any:
        conn = self._conn

        class _Held:
            async def __aenter__(self) -> Any:
                return conn

            async def __aexit__(self, *exc: object) -> None:
                return None

        return _Held()


#: the marks seeded: this many logs, each marked this many times
_MARKED_LOGS = 200
_MARKS_EACH = 100
#: the newest seeded mark's time, less an hour: the g-th mark was written g hours before it
_MARKED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _keeper(conn: Any, schema: str) -> LogKeeper:
    return LogKeeper(_Pool(conn), schema=schema, log=_NoLog())


@pytest.fixture
async def marks(db_container: str) -> AsyncIterator[_Marks]:
    """the keeper's marks table, installed, holding many logs' marks an hour apart."""
    schema = f"markplan_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await _keeper(conn, schema).install()
        table = f"{schema}.threetears_change_note_marks"
        await conn.execute(
            f"INSERT INTO {table} (log_incarnation, seq, at) "  # noqa: S608 - test identifiers
            f"SELECT 'log_' || (g % {_MARKED_LOGS}), g / {_MARKED_LOGS}, $1::timestamptz - make_interval(hours => g) "
            f"FROM generate_series(1, {_MARKED_LOGS * _MARKS_EACH}) g",
            _MARKED_AT,
        )
        await conn.execute(f"ANALYZE {table}")
        yield _Marks(conn=conn, schema=schema)
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def test_threetears_change_note_marks_position_reads_a_restores_marks(marks: _Marks) -> None:
    """Flips if a restore's read of one lost log's marks stops planning through the marks' position index, or answers
    other than that log's marks past the floor in position order."""
    checked = PlanRecorder(marks.conn)
    found = await _keeper(checked, marks.schema).marks_past("log_7", 40)
    assert [int(mark["seq"]) for mark in found] == list(range(41, _MARKS_EACH))
    assert checked.plans and all(
        plan_uses_index(plan, "threetears_change_note_marks_position") for plan in checked.plans
    ), [p.plan for p in checked.plans]


async def test_threetears_change_note_marks_at_reads_the_windows_sweep(marks: _Marks) -> None:
    """Flips if the window's sweep of the marks older than its anchor, or the newest-mark read that names the log the
    keeper knows, stops planning through the marks' time index, or either answers other than its marks."""
    table = f"{marks.schema}.threetears_change_note_marks"
    # the 30 oldest marks (g = 19971 .. 20000) were written before it
    cut = _MARKED_AT - timedelta(hours=_MARKED_LOGS * _MARKS_EACH - 29.5)
    checked = PlanRecorder(marks.conn)
    deleted = await _keeper(checked, marks.schema).drop_marks_before(cut)
    assert deleted == 30
    assert await marks.conn.fetchval(f"SELECT count(*) FROM {table} WHERE at < $1", cut) == 0  # noqa: S608 - test identifiers
    newest = await _keeper(checked, marks.schema).newest_mark()
    assert newest is not None and newest.incarnation == "log_1" and newest.seq == 0, newest
    assert checked.plans and all(plan_uses_index(plan, "threetears_change_note_marks_at") for plan in checked.plans), [
        p.plan for p in checked.plans
    ]
