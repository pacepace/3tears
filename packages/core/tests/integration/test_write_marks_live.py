"""live: a committed write leaves its mark in its own transaction, and the reconciler applies it.

Against a real database (the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to run it
against YugabyteDB). These are the gates Chunk 12 asked for:

- every switched-on table carries the trigger, in every schema, and a table without it is found;
- a write whose generation advance never happened (the writer cancelled or killed after its commit)
  leaves a mark the next pass applies, and a rolled-back write leaves none;
- two concurrent writers to one table both commit: the mark is an insert, so they never contend.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.core.data.migrations.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.data.migrations.session import ConnectionSession
from threetears.core.collections.write_marks import (
    WRITE_MARKS_TABLE,
    WriteMarkReconciler,
    install_write_mark_triggers,
    tables_missing_the_trigger,
    write_marks_ddl,
)

pytestmark = pytest.mark.integration

_TABLE = "marked_widgets"


class _Generations:
    """records each advance, as the epoch bucket's writer would perform it."""

    def __init__(self) -> None:
        self.advanced: list[str] = []

    async def advance(self, table_name: str) -> str:
        self.advanced.append(table_name)
        return f"inc:{len(self.advanced)}"

    async def current(self, table_name: str) -> str:
        return "inc:0"


@pytest.fixture
async def schemas(db_container: str) -> AsyncIterator[tuple[asyncpg.Pool, str, str]]:
    """a platform schema with the queue, an agent schema, and the switched-on table in both."""
    suffix = uuid.uuid4().hex[:8]
    platform, agent = f"marks_platform_{suffix}", f"marks_agent_{suffix}"
    pool = await asyncpg.create_pool(db_container, min_size=2, max_size=4)
    assert pool is not None
    try:
        for schema in (platform, agent):
            await pool.execute(f"CREATE SCHEMA {schema}")
            await pool.execute(f"CREATE TABLE {schema}.{_TABLE} (id TEXT PRIMARY KEY, n INTEGER)")
        for statement in write_marks_ddl(platform):
            await pool.execute(statement)
        yield pool, platform, agent
    finally:
        for schema in (platform, agent):
            await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def _marks(pool: asyncpg.Pool, platform: str) -> list[str]:
    rows = await pool.fetch(f"SELECT table_name FROM {platform}.{WRITE_MARKS_TABLE} ORDER BY id")
    return [row["table_name"] for row in rows]


async def test_every_switched_on_table_carries_the_trigger_in_every_schema(
    schemas: tuple[asyncpg.Pool, str, str],
) -> None:
    pool, platform, agent = schemas
    before = await tables_missing_the_trigger(pool, [_TABLE])
    assert {(platform, _TABLE), (agent, _TABLE)} <= set(before)

    await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE])
    # a second installer (another replica) finds nothing to do and raises nothing
    await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE])

    after = await tables_missing_the_trigger(pool, [_TABLE])
    assert not {(platform, _TABLE), (agent, _TABLE)} & set(after)


async def test_a_committed_write_whose_advance_never_ran_is_applied_by_the_next_pass(
    schemas: tuple[asyncpg.Pool, str, str],
) -> None:
    pool, platform, agent = schemas
    await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE])

    # the writer commits and dies before advancing anything
    await pool.execute(f"INSERT INTO {agent}.{_TABLE} VALUES ('a', 1)")
    # a rolled-back write leaves nothing behind
    async with pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        await conn.execute(f"INSERT INTO {agent}.{_TABLE} VALUES ('b', 2)")
        await transaction.rollback()

    assert await _marks(pool, platform) == [_TABLE]
    source = _Generations()
    reconciler = WriteMarkReconciler(pool, generation_source=source, marks_schema=platform, tables=lambda: [_TABLE])
    done = await reconciler.run_pass()

    assert source.advanced == [_TABLE]
    assert (done.applied, done.behind) == (1, False)
    assert await _marks(pool, platform) == []


async def test_a_cancelled_writer_whose_commit_landed_still_leaves_its_mark(
    schemas: tuple[asyncpg.Pool, str, str],
) -> None:
    pool, platform, agent = schemas
    await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE])

    async def write_then_be_cancelled() -> None:
        await pool.execute(f"UPDATE {agent}.{_TABLE} SET n = 3 WHERE id = 'none'")
        await asyncio.sleep(3600)  # the advance it would have made after the commit

    task = asyncio.create_task(write_then_be_cancelled())
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await _marks(pool, platform) == [_TABLE]


async def test_two_concurrent_writers_to_one_table_both_commit(schemas: tuple[asyncpg.Pool, str, str]) -> None:
    pool, platform, agent = schemas
    await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE])
    started = asyncio.Event()
    both_in = asyncio.Barrier(2)

    async def writer(key: str) -> None:
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(f"INSERT INTO {agent}.{_TABLE} VALUES ($1, 1)", key)
            # both transactions hold their mark uncommitted at once before either commits
            await both_in.wait()
            started.set()

    await asyncio.wait_for(asyncio.gather(writer("x"), writer("y")), timeout=60)

    assert started.is_set()
    assert sorted(await _marks(pool, platform)) == [_TABLE, _TABLE]
    rows = await pool.fetch(f"SELECT id FROM {agent}.{_TABLE} ORDER BY id")
    assert [row["id"] for row in rows] == ["x", "y"]


async def test_a_busy_ddl_lock_defers_the_install_and_the_drain_still_runs(
    schemas: tuple[asyncpg.Pool, str, str],
) -> None:
    pool, platform, agent = schemas
    # a write before any trigger exists, then one marked by hand as a trigger would
    await pool.execute(f"INSERT INTO {platform}.{WRITE_MARKS_TABLE} (table_name) VALUES ($1)", _TABLE)
    source = _Generations()
    reconciler = WriteMarkReconciler(pool, generation_source=source, marks_schema=platform, tables=lambda: [_TABLE])

    holder = await pool.acquire()
    try:
        async with database_ddl_lock(ConnectionSession(holder)):
            quick = DdlLockPolicy(poll_interval=0.05, max_wait=0.3)
            assert await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE], policy=quick) == []
            assert {(platform, _TABLE), (agent, _TABLE)} <= set(await tables_missing_the_trigger(pool, [_TABLE]))
            done = await reconciler.run_pass()
    finally:
        await pool.release(holder)

    assert (done.applied, source.advanced) == (1, [_TABLE])
