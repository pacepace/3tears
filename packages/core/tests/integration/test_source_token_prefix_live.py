"""Integration: a fence's rows a prefix names are cleared through its ``COLLATE "C"`` index, as a range.

The fence table is created from its declaration (:func:`source_tokens_schema`, the DDL ``to_sqlalchemy_table``
emits: ``signal COLLATE "C" ASC``), seeded with a ready signal's job rows among thousands of others, and
:func:`clear_signals` is planned and run: the prefix delete reads ``source_tokens_by_signal_c`` (a range, never the
table, on YugabyteDB never every tablet) and deletes exactly the rows ``starts_with`` names. On the session's
``db_container`` (PostgreSQL), or YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import cast, Any

import asyncpg
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from threetears.core.coordination.source_token import clear_signals, source_tokens_schema
from threetears.core.testing.plans import StatementPlan, explain_write, plan_uses_index

pytestmark = pytest.mark.integration

_OTHERS = 4000


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"fence_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        table = source_tokens_schema().to_sqlalchemy_table(sa.MetaData())
        await connection.execute(str(CreateTable(table).compile(dialect=cast(Any, postgresql).dialect())))
        for index in table.indexes:
            await connection.execute(str(CreateIndex(index).compile(dialect=cast(Any, postgresql).dialect())))
        names = (
            ["refresh_ready", "refresh_ready_other", "refresh_ready0"]
            + [f"refresh_ready/{job}" for job in range(40)]
            + [f"other/{n:05d}" for n in range(_OTHERS)]
        )
        await connection.executemany(
            "INSERT INTO source_tokens (signal, run_id, as_of, accepted_at, date_created, date_updated) "
            "VALUES ($1, 1, now(), now(), now(), now())",
            [(n,) for n in names],
        )
        await connection.execute("ANALYZE source_tokens")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


class _Planned:
    """a connection whose every write is planned (never run by the plan), then run."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn
        self.plans: list[StatementPlan] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.plans.append(await explain_write(self._conn, sql, *args))
        return str(await self._conn.execute(sql, *args))

    async def fetch(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        self.plans.append(await explain_write(self._conn, sql, *args))
        return list(await self._conn.fetch(sql, *args))


async def test_a_prefix_clear_reads_the_c_ordered_index_as_a_range(conn: asyncpg.Connection) -> None:
    """Flips if the fence's prefix delete is not a range on source_tokens_by_signal_c (a scan of 4,000 rows for 40;
    on YugabyteDB every tablet), or clears a row starts_with does not name ("refresh_ready_other",
    "refresh_ready0") or misses one it does, or the named signal's own row."""
    planned = _Planned(conn)
    named = sorted(
        row["signal"]
        for row in await conn.fetch("SELECT signal FROM source_tokens WHERE starts_with(signal, 'refresh_ready/')")
    )
    cleared = await clear_signals(planned, "source_tokens", signal="refresh_ready", prefix="refresh_ready/")
    assert cleared == sorted(["refresh_ready", *named]) and len(named) == 40
    left = {row["signal"] for row in await conn.fetch("SELECT signal FROM source_tokens WHERE signal LIKE 'refresh%'")}
    assert left == {"refresh_ready_other", "refresh_ready0"}
    (prefixed,) = [plan for plan in planned.plans if 'COLLATE "C"' in plan.sql]
    assert plan_uses_index(prefixed, "source_tokens_by_signal_c"), prefixed.plan
