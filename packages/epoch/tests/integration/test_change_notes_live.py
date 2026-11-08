"""Integration test: every write's change note commits with it, and a lost change log is put back from the notes.

Against a real database (the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to run it on
YugabyteDB) and, where the follower's catch-up is the point, a real broker. What these prove:

- a write and its note commit together or not at all: a committed write has its note, a rolled-back one
  has neither, and a writer killed mid-transaction leaves neither;
- after NATS loses the change log, the keeper puts it back from the notes, and a follower of a table that
  is NOT L2-first -- whose records nothing else in L3 keeps -- walks the restored log and catches up, never
  behind as ``record_lost``;
- the ring switches and empties its oldest table while writers write, and no note the coverage still
  needs is lost;
- the scrub reports an epoch the log holds whose note is not in L3.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import WRITE_GENERATION, CollectionRegistry
from threetears.core.collections.change_notes import ChangeNoteKeeper
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.generation import NotedTable
from threetears.core.collections.schema_backed import STRING_TYPE, TableSchema, collection_for_schema
from threetears.core.collections.schema_backed import Column as SchemaColumn
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.change_notes import NOTE_RING_SIZE, note_table
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import (
    EPOCH_CHANGES_BUCKET,
    EpochChangeRetention,
    EpochGenerationReader,
    EpochGenerationSource,
    generation_catchup_tick,
)
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

#: a fence well beyond any statement these tests run, and short enough for a test to pass several
_FENCE = timedelta(milliseconds=300)


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[tuple[asyncpg.Pool, str, str]]:
    """a schema holding the ring and one noted table, with its trigger; the table's name is unique."""
    suffix = uuid.uuid4().hex[:8]
    schema, table = f"notes_{suffix}", f"noted_widgets_{suffix}"
    pool = await asyncpg.create_pool(db_container, min_size=2, max_size=12, server_settings={"search_path": schema})
    assert pool is not None
    try:
        await pool.execute(f"CREATE SCHEMA {schema}")
        await pool.execute(f"CREATE TABLE {schema}.{table} (id TEXT PRIMARY KEY, owner TEXT, n TEXT)")
        yield pool, schema, table
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def _keeper(pool: asyncpg.Pool, schema: str, table: str, log: Any) -> ChangeNoteKeeper:
    keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=log, fence=_FENCE)
    await keeper.install()
    assert await keeper.install_triggers([NotedTable(table=table, keys=("id",), columns=("owner",))]) == [
        f"{schema}.{table}"
    ]
    assert await keeper.install_triggers([NotedTable(table=table, keys=("id",), columns=("owner",))]) == [], (
        "a second install finds nothing to do"
    )
    return keeper


async def _notes(
    pool: asyncpg.Pool, schema: str, *, indexes: tuple[int, ...] = tuple(range(NOTE_RING_SIZE))
) -> list[Any]:
    found: list[Any] = []
    for index in indexes:
        found.extend(await pool.fetch(f"SELECT table_name, keys, columns FROM {schema}.{note_table(index)}"))
    return found


class TestAWriteAndItsNoteCommitTogether:
    async def test_committed_rolled_back_and_killed_writes(
        self, ring: tuple[asyncpg.Pool, str, str], db_container: str
    ) -> None:
        pool, schema, table = ring
        await _keeper(pool, schema, table, EpochGenerationSource(FakeNatsClient()))

        await pool.execute(f"INSERT INTO {table} VALUES ('a', 'ada', '1')")
        async with pool.acquire() as conn:
            transaction = conn.transaction()
            await transaction.start()
            await conn.execute(f"INSERT INTO {table} VALUES ('b', 'bob', '2')")
            await transaction.rollback()

        # a writer process killed between its write and its commit: its socket closes with no COMMIT, and the
        # server ends its transaction. (Never pg_terminate_backend: behind a connection manager a backend serves
        # other clients too.)
        victim = await asyncpg.connect(db_container, server_settings={"search_path": schema})
        await victim.execute("BEGIN")
        await victim.execute(f"INSERT INTO {table} VALUES ('c', 'cy', '3')")
        victim.terminate()

        assert [row["id"] for row in await pool.fetch(f"SELECT id FROM {table} ORDER BY id")] == ["a"]
        notes = await _notes(pool, schema)
        assert [(note["table_name"], list(note["keys"])) for note in notes] == [(table, ["a"])], (
            "only the committed write has a note"
        )
        assert json.loads(notes[0]["columns"]) == {"owner": "ada"}, "the note carries the invalidation column"

    async def test_an_update_that_moves_a_column_notes_both_rows_and_a_delete_notes_the_old(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        await _keeper(pool, schema, table, EpochGenerationSource(FakeNatsClient()))
        await pool.execute(f"INSERT INTO {table} VALUES ('a', 'ada', '1')")
        await pool.execute(f"UPDATE {table} SET n = '2' WHERE id = 'a'")
        await pool.execute(f"UPDATE {table} SET owner = 'bea' WHERE id = 'a'")
        await pool.execute(f"DELETE FROM {table} WHERE id = 'a'")
        owners = sorted(json.loads(note["columns"])["owner"] for note in await _notes(pool, schema))
        # insert, quiet update, moving update (new and old), delete (old)
        assert owners == ["ada", "ada", "ada", "bea", "bea"]

    async def test_a_truncate_is_one_whole_table_note(self, ring: tuple[asyncpg.Pool, str, str]) -> None:
        pool, schema, table = ring
        await _keeper(pool, schema, table, EpochGenerationSource(FakeNatsClient()))
        await pool.execute(f"TRUNCATE {table}")
        notes = await _notes(pool, schema)
        assert [(note["table_name"], note["keys"]) for note in notes] == [(table, None)]


@asynccontextmanager
async def _nats(url: str) -> AsyncIterator[NatsClient]:
    namespace = f"notes{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(nats_url=url, nats_subject_namespace=namespace, client_name="notes") as nc:
        yield nc


class _Pod:
    """one registry over the noted table, with its own L1, NATS connection, source and reader."""

    def __init__(self, nc: NatsClient, pool: asyncpg.Pool, table: str, scope: str) -> None:
        schema = TableSchema(
            name=table,
            primary_key="id",
            columns=[
                SchemaColumn("id", STRING_TYPE),
                SchemaColumn("owner", STRING_TYPE, nullable=True),
                SchemaColumn("n", STRING_TYPE, nullable=True),
            ],
            on_conflict="update",
        )
        metadata = MetaData()
        schema.to_sqlalchemy_table(metadata)
        l1 = SQLiteBackend(db_name=f"notes_{uuid.uuid4().hex[:8]}")
        l1.initialize(metadata)
        self.registry = CollectionRegistry()
        self.registry.configure(l1_backend=l1, l2_client=nc, l3_pool=SqlL3Backend(pool), kv_key_scope=scope)
        self.source = EpochGenerationSource(nc, restored_from_l3=True, restore_wait=timedelta(seconds=5))
        self.registry.set_generation_source(self.source)
        self.reader = EpochGenerationReader(nc)
        declared = type(f"Noted_{scope}", (collection_for_schema(schema),), {"write_generation": WRITE_GENERATION})
        self.rows = declared(self.registry, DefaultCoreConfig())


class TestALostLogIsPutBackFromTheNotes:
    async def test_a_non_l2_first_tables_follower_catches_up_with_no_record_lost(
        self, ring: tuple[asyncpg.Pool, str, str], nats_container: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        pool, schema, table = ring
        async with _nats(nats_container) as nc:
            # two registries on one connection: the follower never starts its listener, so it hears nothing
            writer, follower = _Pod(nc, pool, table, "writer"), _Pod(nc, pool, table, "follower")
            await writer.source.start_log()
            keeper = await _keeper(pool, schema, table, writer.source)
            await keeper.run_pass()  # the keeper sees the log
            await asyncio.sleep(_FENCE.total_seconds() * 2)
            await keeper.run_pass()
            assert (await keeper.ring()).fence_seq is not None, "the fence after the ring's start passed"

            await writer.rows.save_entity(writer.rows.create({"id": "w1", "owner": "ada", "n": "1"}))
            follower.registry.follow_generation(table)
            await generation_catchup_tick(follower.registry, follower.reader)
            assert (await follower.rows.ensure("w1") or {}).get("n") == "1"

            # the follower hears nothing of this write: its listener never ran
            await writer.rows.save_entity(writer.rows.create({"id": "w1", "owner": "ada", "n": "2"}))
            await writer.rows.save_entity(writer.rows.create({"id": "w2", "owner": "bea", "n": "1"}))
            lost = (await writer.source.log_state()).incarnation  # type: ignore[union-attr]

            await nc.jetstream_context().delete_stream(f"KV_{nc.namespace}-{EPOCH_CHANGES_BUCKET}")
            restored = (await keeper.run_pass()).restored
            assert restored is not None and restored != lost
            continued = await follower.reader.continues(restored)
            assert continued is not None and continued[0] == lost and continued[1] is not None

            with caplog.at_level(logging.ERROR):
                await generation_catchup_tick(follower.registry, follower.reader)
            assert not follower.registry.behind_on(table), "the notes cover the follower's mark"
            behind = [r for r in caplog.records if r.getMessage().startswith("behind on a table")]
            assert behind == [], "never behind as record_lost"
            assert (follower.registry.generation_marks.recorded(table) or "").startswith(f"{restored}:")
            assert (await follower.rows.ensure("w1") or {}).get("n") == "2", "the missed write was applied"

            # writing on after the restore lands in the restored log, and the follower follows it
            await writer.rows.save_entity(writer.rows.create({"id": "w3", "owner": "cy", "n": "1"}))
            head = await follower.reader.read(table)
            assert head is not None and head.startswith(f"{restored}:")
            ring_state = await keeper.ring()
            assert ring_state.log_incarnation == restored and ring_state.floor == f"{restored}:0"


class TestTheRingSwitchesUnderWrites:
    async def test_no_note_the_coverage_needs_is_lost_across_switches_and_empties(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus, change_retention=EpochChangeRetention(max_bytes=256 * 1024))
        await source.start_log()
        keeper = await _keeper(pool, schema, table, source)
        written: list[tuple[str, int]] = []
        stop = asyncio.Event()

        async def writer(index: int) -> None:
            count = 0
            while not stop.is_set():
                key = f"w{index}-{count}"
                async with pool.acquire() as conn, conn.transaction():
                    await conn.execute(f"INSERT INTO {table} VALUES ($1, 'o', '1')", key)
                token = await source.advance_with_change(table, [ChangedRow(ids=[key], columns={"owner": "o"})])
                written.append((key, int(token.rpartition(":")[2])))
                count += 1
                await asyncio.sleep(0)

        async def keep() -> list[int]:
            switches: list[int] = []
            while not stop.is_set():
                done = await keeper.run_pass()
                if done.switched_to is not None:
                    switches.append(done.switched_to)
                state = await source.log_state()
                if state is not None and state.bytes > 64 * 1024:
                    # retention ages the log's oldest records out, so it never fills; which notes the coverage
                    # needs is checked below against every write made, not against what the log still holds
                    bus.change_log(EPOCH_CHANGES_BUCKET).age_out(state.last_seq - 50)
                if len(switches) >= 4:
                    stop.set()
                await asyncio.sleep(0.01)
            return switches

        keeping = asyncio.create_task(keep())
        writers = [asyncio.create_task(writer(index)) for index in range(8)]
        try:
            await asyncio.wait_for(stop.wait(), timeout=60)
        finally:
            stop.set()
        await asyncio.gather(*writers)
        switches = await keeping

        assert len(switches) >= 3, f"the ring went round: {switches}"
        state = await keeper.ring()
        assert state.floor is not None
        floor = int(state.floor.rpartition(":")[2])
        noted = {list(note["keys"])[0] for note in await _notes(pool, schema, indexes=(state.current, state.previous))}
        needed = [key for key, seq in written if seq > floor]
        assert needed, "writes after the floor were made"
        missing = [key for key in needed if key not in noted]
        assert missing == [], f"{len(missing)} notes the coverage needs were lost"


class TestTheScrubReportsAMissingNote:
    async def test_an_epoch_whose_write_left_no_note_is_reported_and_nothing_is_repaired(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        pool, schema, table = ring
        source = EpochGenerationSource(FakeNatsClient())
        await source.start_log()
        keeper = await _keeper(pool, schema, table, source)
        await keeper.run_pass()
        await asyncio.sleep(_FENCE.total_seconds() * 2)
        await keeper.run_pass()
        assert (await keeper.ring()).fence_seq is not None

        await pool.execute(f"INSERT INTO {table} VALUES ('noted', 'o', '1')")
        await source.advance_with_change(table, [ChangedRow(ids=["noted"])])
        # an epoch whose write left no note: appended with no L3 write behind it
        await source.advance_with_change(table, [ChangedRow(ids=["unnoted"])])
        await keeper.run_pass()  # the scrub checks up to the head one pass ago
        notes_before = len(await _notes(pool, schema))

        with caplog.at_level(logging.ERROR):
            scrub = (await keeper.run_pass()).scrub
        assert (scrub.checked, scrub.missing) == (2, 1)
        said = [r.__dict__["extra_data"] for r in caplog.records if "no change note" in r.getMessage()]
        assert [each["missing"] for each in said] == ["unnoted"]
        assert len(await _notes(pool, schema)) == notes_before, "the scrub only checks; it writes no note"
