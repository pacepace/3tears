"""live: the writers' backstop's own gates, with no L3 write a commit and no change-note ring.

Against a real database (the session ``db_container``; Postgres by default, set ``THREETEARS_TEST_POSTGRES_URL`` to run
it against YugabyteDB). The gates of the write-mark removal (Pace 2026-11-22):

- the writer's connection class keeps each transaction's start until it ends, and each statement's outside one;
- after coordination's migrations no write-mark trigger and no mark function remains, on the platform schema's tables
  or an agent schema's, and one an older replica puts back is taken down by the backstop;
- a table carrying an epoch is read as switched on; one without it is not.

What a dead writer's window announces is read from the tables' epoch ranges since the ring retired (append-only L3
chunk 7): ``test_epoch_range_readers`` holds it, its rows, its gone markers and its runs deleted whole.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.lost_writes import WriteBackstop, tables_switched_by_epoch
from threetears.core.coordination.lease import KVLease
from threetears.core.coordination.migrations import register as register_coordination
from threetears.core.data.migrations import MigrationRunner
from threetears.core.data.migrations.session import ConnectionSession
from threetears.core.ddl.write_marks import MARK_FUNCTION, WRITE_MARK_TRIGGER
from threetears.core.testing.kv import FakeNatsClient
from threetears.core.utils.write_ledger import WriteLedger, write_ledger_connection_class

pytestmark = pytest.mark.integration


@dataclass
class _Change:
    table: str
    rows: list[str] | None
    keys_unknown: bool
    tenant: str | None
    noted_between: tuple[datetime, datetime] | None
    prefixes: Any
    reason: str | None


class _Recorder:
    """the epoch bucket's writer, recording each change it is handed."""

    def __init__(self) -> None:
        self.changes: list[_Change] = []

    async def current(self, table_name: str) -> str:
        return "inc:0"

    async def advance(self, table_name: str) -> str:
        raise AssertionError("a recording source is advanced with its change")

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
        noted_between: tuple[datetime, datetime] | None = None,
        prefixes: Any = None,
        reason: str | None = None,
    ) -> str:
        self.changes.append(
            _Change(
                table_name,
                None if rows is None else sorted(row.ids[0] for row in rows),
                keys_unknown,
                tenant,
                noted_between,
                prefixes,
                reason,
            )
        )
        return f"inc:{len(self.changes)}"


@dataclass
class _Db:
    pool: asyncpg.Pool
    schema: str
    table: str
    ledger: WriteLedger
    dsn: str


@pytest.fixture
async def db(db_container: str) -> AsyncIterator[_Db]:
    """a schema holding one table carrying an epoch (switched on), on a pool keeping a writer ledger."""
    schema = f"lost_{uuid.uuid4().hex[:8]}"
    table = f"lost_rows_{uuid.uuid4().hex[:6]}"
    ledger = WriteLedger()
    setup = await asyncpg.connect(db_container)
    try:
        await setup.execute(f"CREATE SCHEMA {schema}")
        await setup.execute(
            f"CREATE TABLE {schema}.{table} (id TEXT PRIMARY KEY, n INTEGER, epoch BIGINT NOT NULL DEFAULT 0)"
        )
    finally:
        await setup.close()
    pool = await asyncpg.create_pool(
        db_container, min_size=1, max_size=3, connection_class=write_ledger_connection_class(ledger=ledger)
    )
    assert pool is not None
    try:
        yield _Db(pool=pool, schema=schema, table=table, ledger=ledger, dsn=db_container)
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def test_the_connection_class_keeps_each_write_s_start_until_it_ends(db: _Db) -> None:
    """Flips if an open transaction stops holding the watermark before it ends, or a connection closed mid-transaction
    holds it for ever."""
    async with db.pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        held = db.ledger.watermark()
        await conn.execute(f"INSERT INTO {db.schema}.{db.table} VALUES ('t', 1)")
        assert db.ledger.open_writes == 1
        assert db.ledger.watermark() == held
        await tx.commit()
    assert db.ledger.open_writes == 0
    # a statement outside a transaction is a write of its own, ended with the statement
    await db.pool.execute(f"INSERT INTO {db.schema}.{db.table} VALUES ('u', 1)")
    assert db.ledger.open_writes == 0
    lone = await asyncpg.connect(db.dsn, connection_class=write_ledger_connection_class(ledger=db.ledger))
    tx = lone.transaction()
    await tx.start()
    assert db.ledger.open_writes == 1
    lone.terminate()
    await asyncio.sleep(0)  # asyncpg calls a termination listener on the loop's next turn
    assert db.ledger.open_writes == 0


async def test_a_table_carrying_an_epoch_is_read_as_switched_on(db: _Db) -> None:
    await db.pool.execute(f"CREATE TABLE {db.schema}.plain_rows (id TEXT PRIMARY KEY)")
    found = await tables_switched_by_epoch(db.pool, db.schema, [db.table, "plain_rows", "absent_rows"])
    assert found == frozenset({db.table})


async def _mark_triggers(pool: asyncpg.Pool, schemas: Sequence[str]) -> list[tuple[str, str]]:
    rows = await pool.fetch(
        "SELECT n.nspname::text AS s, c.relname::text AS t FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE g.tgname = $1 AND n.nspname::text = ANY($2::text[]) "
        "ORDER BY 1, 2",
        WRITE_MARK_TRIGGER,
        list(schemas),
    )
    return [(str(row["s"]), str(row["t"])) for row in rows]


async def _mark_functions(pool: asyncpg.Pool, schema: str) -> int:
    return int(
        await pool.fetchval(
            "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            "WHERE p.proname::text = $1 AND n.nspname::text = $2",
            MARK_FUNCTION,
            schema,
        )
    )


async def _old_marks(pool: asyncpg.Pool, platform: str, tables: Sequence[tuple[str, str]]) -> None:
    """the mark function and its per-row triggers as an older release (or an older replica mid-roll) left them."""
    await pool.execute(
        f"CREATE OR REPLACE FUNCTION {platform}.{MARK_FUNCTION}() RETURNS trigger LANGUAGE plpgsql AS "
        f"$m$ BEGIN INSERT INTO {platform}.collection_write_marks (table_name) VALUES (TG_TABLE_NAME); "
        "RETURN NULL; END $m$"
    )
    for schema, table in tables:
        await pool.execute(f"DROP TRIGGER IF EXISTS {WRITE_MARK_TRIGGER} ON {schema}.{table}")
        await pool.execute(
            f"CREATE TRIGGER {WRITE_MARK_TRIGGER} AFTER INSERT OR UPDATE OR DELETE ON {schema}.{table} "
            f"FOR EACH ROW EXECUTE FUNCTION {platform}.{MARK_FUNCTION}()"
        )


async def test_no_write_mark_trigger_or_function_remains_after_the_migrations_and_the_backstop(
    db_container: str,
) -> None:
    """Flips if a mark trigger survives coordination v006 (each write would still pay a per-row trigger and an insert),
    or one an older replica puts back mid-roll stays."""
    suffix = uuid.uuid4().hex[:8]
    platform, agent = f"marks_gone_{suffix}", f"marks_gone_agent_{suffix}"
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2)
    assert pool is not None
    try:
        for schema in (platform, agent):
            await pool.execute(f"CREATE SCHEMA {schema}")
        await pool.execute(f"CREATE TABLE {agent}.pod_rows (id TEXT PRIMARY KEY)")
        async with pool.acquire() as conn:
            await conn.execute(f"SET search_path TO {platform}")
            runner = MigrationRunner()
            register_coordination(runner)
            # an older release's database: the queue, its function and the triggers, before v006
            await runner.apply_for_platform_schema(ConnectionSession(conn), target=5)
            marked = [(platform, "coordination_revocations"), (agent, "pod_rows")]
            await _old_marks(pool, platform, marked)
            assert await _mark_triggers(pool, (platform, agent)) == sorted(marked)
            await runner.apply_for_platform_schema(ConnectionSession(conn))
        assert await _mark_triggers(pool, (platform, agent)) == []
        assert await _mark_functions(pool, platform) == 0
        # the queue's table stays, empty, for an older replica mid-roll
        assert await pool.fetchval(f"SELECT count(*) FROM {platform}.collection_write_marks") == 0
        await pool.execute(f"INSERT INTO {agent}.pod_rows VALUES ('a')")
        assert await pool.fetchval(f"SELECT count(*) FROM {platform}.collection_write_marks") == 0

        # an older replica's install backstop puts them back; the new backstop takes them down
        await _old_marks(pool, platform, marked)
        backstop = WriteBackstop(
            pool,
            generation_source=_Recorder(),
            notes_schema=platform,
            lease=KVLease(FakeNatsClient(), pod_id="sweeper"),
            ledger=WriteLedger(),
        )
        assert sorted(await backstop.drop_leftover_write_mark_triggers()) == [f"{s}.{t}" for s, t in sorted(marked)]
        assert await _mark_triggers(pool, (platform, agent)) == []
        assert await _mark_functions(pool, platform) == 0
        assert await backstop.drop_leftover_write_mark_triggers() == []
    finally:
        for schema in (agent, platform):
            await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()
