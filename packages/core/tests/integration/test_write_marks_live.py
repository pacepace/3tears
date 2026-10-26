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
from typing import Any

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections import WRITE_GENERATION, CollectionRegistry
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, TableSchema, collection_for_schema
from threetears.core.collections.schema_backed import Column as SchemaColumn
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeNatsClient

from threetears.core.data.migrations.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.data.migrations.session import ConnectionSession
from threetears.core.collections.write_marks import (
    MARK_FUNCTION,
    WRITE_MARK_TRIGGER,
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
        await pool.execute(f"INSERT INTO {agent}.{_TABLE} VALUES ('c', 3)")
        await asyncio.sleep(3600)  # the advance it would have made after the commit

    task = asyncio.create_task(write_then_be_cancelled())
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await _marks(pool, platform) == [_TABLE]


async def test_a_statement_that_changes_no_row_leaves_no_mark(schemas: tuple[asyncpg.Pool, str, str]) -> None:
    # the hub's idempotent namespace insert on every tool pod's handshake changed nothing and still marked, and
    # each mark is an advance no follower heard, so every follower dropped the table about twice a minute
    pool, platform, agent = schemas
    await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE])
    await pool.execute(f"INSERT INTO {agent}.{_TABLE} VALUES ('a', 1)")
    await pool.execute(f"DELETE FROM {platform}.{WRITE_MARKS_TABLE}")

    await pool.execute(
        f"INSERT INTO {agent}.{_TABLE} SELECT 'a', 1 WHERE NOT EXISTS (SELECT 1 FROM {agent}.{_TABLE} WHERE id = 'a')"
    )
    await pool.execute(f"UPDATE {agent}.{_TABLE} SET n = 2 WHERE false")
    await pool.execute(f"DELETE FROM {agent}.{_TABLE} WHERE false")
    await pool.execute(f"INSERT INTO {agent}.{_TABLE} VALUES ('a', 1) ON CONFLICT DO NOTHING")

    assert await _marks(pool, platform) == []


async def test_a_write_of_many_rows_in_one_transaction_leaves_one_mark(
    schemas: tuple[asyncpg.Pool, str, str],
) -> None:
    pool, platform, agent = schemas
    await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE])

    await pool.execute(f"INSERT INTO {agent}.{_TABLE} VALUES ('a', 1), ('b', 2), ('c', 3)")
    assert await _marks(pool, platform) == [_TABLE]
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(f"UPDATE {agent}.{_TABLE} SET n = n + 1")
        await conn.execute(f"DELETE FROM {agent}.{_TABLE} WHERE id = 'c'")
        await conn.execute(f"INSERT INTO {platform}.{_TABLE} VALUES ('z', 9)")  # another table: its own mark
    assert await _marks(pool, platform) == [_TABLE, _TABLE, _TABLE]


async def test_a_row_rolled_back_to_a_savepoint_takes_its_mark_back_and_a_later_row_marks_again(
    schemas: tuple[asyncpg.Pool, str, str],
) -> None:
    pool, platform, agent = schemas
    await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE])

    async with pool.acquire() as conn, conn.transaction():
        savepoint = conn.transaction()
        await savepoint.start()
        await conn.execute(f"INSERT INTO {agent}.{_TABLE} VALUES ('a', 1)")
        await savepoint.rollback()
        await conn.execute(f"INSERT INTO {agent}.{_TABLE} VALUES ('b', 2)")

    assert await _marks(pool, platform) == [_TABLE]


async def test_an_install_over_the_statement_trigger_replaces_it_so_a_write_marks_once(
    schemas: tuple[asyncpg.Pool, str, str],
) -> None:
    # a database 0.78.0 installed: the old function and a statement trigger on the table
    pool, platform, agent = schemas
    await pool.execute(
        f"CREATE OR REPLACE FUNCTION {platform}.{MARK_FUNCTION}() RETURNS trigger LANGUAGE plpgsql AS $m$ "
        f"BEGIN INSERT INTO {platform}.{WRITE_MARKS_TABLE} (table_name) VALUES (TG_TABLE_NAME); RETURN NULL; END $m$"
    )
    await pool.execute(
        f"CREATE TRIGGER {WRITE_MARK_TRIGGER} AFTER INSERT OR UPDATE OR DELETE ON {agent}.{_TABLE} "
        f"FOR EACH STATEMENT EXECUTE FUNCTION {platform}.{MARK_FUNCTION}()"
    )

    await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE])

    triggers = await pool.fetch(
        "SELECT g.tgname, (g.tgtype & 1) = 1 AS per_row FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 AND c.relname = $2 AND NOT g.tgisinternal",
        agent,
        _TABLE,
    )
    assert [(row["tgname"], row["per_row"]) for row in triggers] == [(WRITE_MARK_TRIGGER, True)]
    await pool.execute(f"INSERT INTO {agent}.{_TABLE} VALUES ('a', 1), ('b', 2)")
    await pool.execute(f"UPDATE {agent}.{_TABLE} SET n = 5 WHERE false")
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


async def test_the_migration_run_that_creates_a_switched_table_gives_it_its_trigger(db_container: str) -> None:
    from threetears.core.coordination.migrations import register as register_coordination
    from threetears.core.data.migrations import MigrationRunner

    schema = f"marks_run_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path TO {schema}")
        runner = MigrationRunner()
        register_coordination(runner)
        await runner.apply_for_platform_schema(ConnectionSession(conn))

        # from the same run: the revocation list carries its trigger before any write
        assert (schema, "coordination_revocations") not in set(
            await tables_missing_the_trigger(conn, ["coordination_revocations"])
        )
        await conn.execute(
            "INSERT INTO coordination_revocations (purpose, key, revoked_at, expires_at, date_created, date_updated) "
            "VALUES ('standing', 'k', now(), now() + interval '1 day', now(), now())"
        )
        marks = await conn.fetch(f"SELECT table_name FROM {schema}.{WRITE_MARKS_TABLE}")
        assert [row["table_name"] for row in marks] == ["coordination_revocations"]
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


_ROWS = TableSchema(
    name=_TABLE,
    primary_key="id",
    columns=[SchemaColumn("id", STRING_TYPE), SchemaColumn("n", BIGINT_TYPE, nullable=True)],
    on_conflict="update",
)


class _Row(BaseEntity):
    primary_key_field = "id"


class _SwitchedRows(collection_for_schema(_ROWS, entity_class=_Row)):  # type: ignore[misc]
    write_generation = WRITE_GENERATION


class _DiesBeforeClearing(SqlL3Backend):
    """L3 for a writer whose process ends between its advance and the delete of the marks it covered."""

    async def execute(self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any) -> str:
        if query.startswith("DELETE") and WRITE_MARKS_TABLE in query:
            return "DELETE 0"
        return await super().execute(query, *params, namespace=namespace, **kwargs)


async def _switched_writer(
    db_container: str, platform: str, agent: str, *, l3: type[SqlL3Backend] = SqlL3Backend
) -> tuple[Any, asyncpg.Pool, _Generations]:
    """a switched-on collection over the agent schema's table, its registry naming the platform's queue."""
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, server_settings={"search_path": agent})
    assert pool is not None
    source = _Generations()
    registry = CollectionRegistry()
    registry.configure(l2_client=FakeNatsClient(), kv_key_scope="marks-test", l3_pool=l3(pool))
    registry.set_generation_source(source)
    registry.set_write_marks_schema(platform)
    return _SwitchedRows(registry, DefaultCoreConfig()), pool, source


async def test_a_collection_write_that_advanced_itself_leaves_no_mark(
    db_container: str, schemas: tuple[asyncpg.Pool, str, str]
) -> None:
    # its own advance is heard; the reconciler's second one from its mark was not, and every follower dropped
    pool, platform, agent = schemas
    await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE])
    rows, writer_pool, source = await _switched_writer(db_container, platform, agent)
    try:
        await rows.save_entity(rows.create({"id": "a", "n": 1}))
        await rows.save_entity(rows.create({"id": "b", "n": 2}))
    finally:
        await writer_pool.close()

    assert source.advanced == [_TABLE, _TABLE]
    assert await _marks(pool, platform) == []


async def test_a_writer_that_dies_between_its_advance_and_the_delete_leaves_marks_one_pass_drains(
    db_container: str, schemas: tuple[asyncpg.Pool, str, str]
) -> None:
    pool, platform, agent = schemas
    await install_write_mark_triggers(pool, marks_schema=platform, tables=[_TABLE])
    rows, writer_pool, source = await _switched_writer(db_container, platform, agent, l3=_DiesBeforeClearing)
    try:
        await rows.save_entity(rows.create({"id": "a", "n": 1}))
    finally:
        await writer_pool.close()
    assert source.advanced == [_TABLE]
    assert await _marks(pool, platform) == [_TABLE]

    drain = _Generations()
    done = await WriteMarkReconciler(
        pool, generation_source=drain, marks_schema=platform, tables=lambda: [_TABLE]
    ).run_pass()

    assert drain.advanced == [_TABLE]
    assert done.applied == 1
    assert await _marks(pool, platform) == []
