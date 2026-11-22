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
- the scrub reports an epoch the log holds whose note is not in L3, on its own interval, through an index;
- a restore is labelled as continuing the lost log only after its last append: one refused part-way leaves
  the follower behind, and the next pass redoes it under fresh append ids; a write made meanwhile lands at
  once and is walked too; a restore that would not fit is refused, said, and leaves followers behind;
- a noted column whose text form is not exact gets no trigger, with an error naming it.
"""

from __future__ import annotations

import asyncio
import time
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import asyncpg
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from prometheus_client import REGISTRY
from sqlalchemy import MetaData

from threetears.core.backends.protocol import LoadFile
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import WRITE_GENERATION, BaseCollection, CollectionRegistry
from threetears.core.collections.change_note_keeper import ChangeNoteKeeper, ChangeNoteKeeping, ScrubPass
from threetears.core.collections.change_notes import (
    RESTORE_BUDGET_RATIO,
    PoolNotedKeysReader,
)
from threetears.core.collections.epoch_change import RESET_REASON, ChangedRow, key_text, read_epoch_change_body
from threetears.core.collections.generation import NotedTable, tenancy_of_table
from threetears.core.collections.l2_first import run_l2_first_worker
from threetears.core.collections.lost_writes import announce_window
from threetears.core.collections.schema_backed import STRING_TYPE, TableSchema, collection_for_schema
from threetears.core.collections.schema_backed import Column as SchemaColumn
from threetears.core.config import DefaultCoreConfig
from threetears.core.coordination.lease import KVLease
from threetears.core.ddl.change_notes import (
    WHOLE_COPY_ROWS,
    NOTE_REACH_SETTING,
    NOTE_RING_SIZE,
    NOTE_TRIGGER,
    NOTE_TRUNCATE_TRIGGER,
    note_function_name,
    note_table,
    note_trigger_sql,
    trigger_function_sql,
)
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.tenancy import TENANT_BY_SCHEMA, PodTenants, schema_tenant, tenant_column, value_tenant
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import (
    EPOCH_CHANGES_BUCKET,
    EpochChangeRetention,
    EpochGenerationReader,
    EpochGenerationSource,
    generation_catchup_tick,
)
from threetears.nats import NatsClient, set_default_namespace
from threetears.nats.errors import KvError
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.utils.write_ledger import CLOCK_SKEW

pytestmark = pytest.mark.integration

#: the shared truncate trigger an install before a whole-table note named its tenant gave each table
_LEGACY_TRUNCATE_TRIGGER = "threetears_change_note_truncate"

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


class _Recovery:
    """announces each lost write's window as a dead writer's recovery does (:func:`announce_window`): from the
    database's clock where the last pass ended (or the recovery started) to now, widened by the clocks' skew, over the
    test's own table alone (other tests' switched tables share the database)."""

    def __init__(self, pool: asyncpg.Pool, source: Any, schema: str, tables: dict[str, tuple[str, ...]]) -> None:
        self._pool, self._source, self._schema, self._tables = pool, source, schema, tables
        self._since: datetime | None = None

    @classmethod
    async def start(cls, pool: asyncpg.Pool, source: Any, schema: str, tables: dict[str, tuple[str, ...]]) -> _Recovery:
        recovery = cls(pool, source, schema, tables)
        recovery._since = await pool.fetchval("SELECT clock_timestamp()")
        return recovery

    def back(self, interval: timedelta) -> None:
        """the next window starts ``interval`` earlier: a dead writer whose oldest write began that long ago."""
        assert self._since is not None
        self._since -= interval

    async def run_pass(self) -> Any:
        assert self._since is not None
        until = await self._pool.fetchval("SELECT clock_timestamp()")
        done = await announce_window(
            self._pool,
            generation_source=self._source,
            notes_schema=self._schema,
            since=self._since - CLOCK_SKEW,
            until=until + CLOCK_SKEW,
            why="recovery",
            tables=self._tables,
        )
        self._since = until
        return done


async def _keeper(pool: asyncpg.Pool, schema: str, table: str, log: Any) -> ChangeNoteKeeper:
    keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=log, fence=_FENCE, scrub_every=timedelta(0))
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

    def __init__(
        self, nc: NatsClient, pool: asyncpg.Pool, table: str, scope: str, *, database: str | None = None
    ) -> None:
        schema = TableSchema(
            name=table,
            primary_key="id",
            columns=[
                SchemaColumn("id", STRING_TYPE),
                SchemaColumn("owner", STRING_TYPE, nullable=True),
                SchemaColumn("n", STRING_TYPE, nullable=True),
            ],
            on_conflict="update",
            l1_eviction=TEST_L1_TABLE,
        )
        metadata = MetaData()
        schema.to_sqlalchemy_table(metadata)
        l1 = SQLiteBackend(db_name=f"notes_{uuid.uuid4().hex[:8]}")
        l1.initialize(metadata)
        self.registry = CollectionRegistry()
        self.registry.configure(l1_backend=l1, l2_client=nc, l3_pool=SqlL3Backend(pool), kv_key_scope=scope)
        self.source = EpochGenerationSource(
            nc, restored_from_l3=True, restore_wait=timedelta(seconds=5), database=database
        )
        self.registry.set_generation_source(self.source)
        self.reader = EpochGenerationReader(nc, database=database)
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


async def _fenced(pool: asyncpg.Pool, schema: str, table: str, source: EpochGenerationSource) -> ChangeNoteKeeper:
    """a keeper whose ring has seen the log and passed its fence, so the coverage vouches from the log's start."""
    keeper = await _keeper(pool, schema, table, source)
    await keeper.run_pass()
    await pool.execute(f"UPDATE {schema}.threetears_change_note_ring SET switched_at = now() - interval '1 hour'")
    await keeper.run_pass()
    assert (await keeper.ring()).fence_seq is not None
    return keeper


class TestARestoreIsWholeOrNotTakenAtAll:
    """the label "continues the lost log" is written only after the restore's last append."""

    async def test_a_restore_refused_part_way_leaves_the_follower_behind_until_it_is_redone(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        bus = FakeNatsClient()
        writer, follower = _Pod(bus, pool, table, "writer"), _Pod(bus, pool, table, "follower")  # type: ignore[arg-type]
        await writer.source.start_log()
        keeper = await _fenced(pool, schema, table, writer.source)
        await writer.rows.save_entity(writer.rows.create({"id": "w1", "owner": "ada", "n": "1"}))
        follower.registry.follow_generation(table)
        await generation_catchup_tick(follower.registry, follower.reader)
        assert (await follower.rows.ensure("w1") or {}).get("n") == "1"
        # unheard by the follower, then NATS loses the log
        await writer.rows.save_entity(writer.rows.create({"id": "w1", "owner": "ada", "n": "2"}))
        await writer.rows.save_entity(writer.rows.create({"id": "w2", "owner": "bea", "n": "1"}))
        # another table's note, so the restore is two records and the second can be refused
        current = (await keeper.ring()).current
        await pool.execute(
            f"INSERT INTO {schema}.{note_table(current)} (table_name, keys, key_text) VALUES ('zz_other', ARRAY['x'], 'x')"
        )
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        lost = (await writer.source.log_state()).incarnation  # type: ignore[union-attr]
        log.vanish()

        ids: list[str] = []

        def refuse_after_the_first(msg_id: str) -> Exception | None:
            if not msg_id.startswith("restore-"):
                return None
            ids.append(msg_id)
            return KvError("refused") if len(ids) > 1 else None

        log.refuse_append = refuse_after_the_first
        with pytest.raises(GenerationUnavailableError):
            await keeper.run_pass()
        state = await writer.source.log_state()
        assert state is not None and state.restoring == lost and state.continues is None, "declared, not labelled"

        # a write meanwhile lands in the declared log at once: it does not wait out the restore
        await writer.rows.save_entity(writer.rows.create({"id": "w3", "owner": "cy", "n": "1"}))

        await generation_catchup_tick(follower.registry, follower.reader)
        assert follower.registry.behind_on(table), "a partial restore never makes the follower current"
        assert (await follower.rows.ensure("w1") or {}).get("n") == "1", "nothing was applied from it"

        # the next pass redoes the restore into the same log, with fresh append ids, and labels it
        log.refuse_append = None
        restored = (await keeper.run_pass()).restored
        assert restored == state.incarnation
        relabelled = await writer.source.log_state()
        assert relabelled is not None and relabelled.continues == lost and relabelled.restoring is None
        assert (await keeper.ring()).log_incarnation == restored

        await generation_catchup_tick(follower.registry, follower.reader)
        assert not follower.registry.behind_on(table)
        assert (await follower.rows.ensure("w1") or {}).get("n") == "2", "every noted key was applied"
        assert (await follower.rows.ensure("w2") or {}).get("owner") == "bea"
        assert (await follower.rows.ensure("w3") or {}).get("owner") == "cy", "the write made meanwhile too"

    async def test_a_rerun_appends_under_fresh_ids(self, ring: tuple[asyncpg.Pool, str, str]) -> None:
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus, restored_from_l3=True, restore_wait=timedelta(seconds=1))
        await source.start_log()
        keeper = await _fenced(pool, schema, table, source)
        current = (await keeper.ring()).current
        for other in ("t_a", "t_b", "t_c"):  # three tables' notes: three restored records
            await pool.execute(
                f"INSERT INTO {schema}.{note_table(current)} (table_name, keys, key_text) VALUES ($1, ARRAY['k'], 'k')",
                other,
            )
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        log.vanish()
        seen: list[str] = []

        def refuse_the_third(msg_id: str) -> Exception | None:
            seen.append(msg_id)
            return KvError("refused") if len(seen) == 2 else None

        log.refuse_append = refuse_the_third
        with pytest.raises(GenerationUnavailableError):
            await keeper.run_pass()
        first = set(seen)
        seen.clear()
        log.refuse_append = lambda msg_id: seen.append(msg_id) or None  # type: ignore[func-returns-value]
        assert (await keeper.run_pass()).restored is not None
        assert seen and not first & set(seen), "a re-run's appends are never dropped as repeats"

    async def test_a_restore_that_would_not_fit_is_refused_said_and_leaves_followers_behind(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(
            bus,
            change_retention=EpochChangeRetention(max_bytes=4096),
            restored_from_l3=True,
            restore_wait=timedelta(seconds=1),
        )
        await source.start_log()
        keeper = await _fenced(pool, schema, table, source)
        # many distinct keys: their records fold to far more than the log's bound
        await pool.execute(
            f"INSERT INTO {table} SELECT 'k' || g, 'o', '1' FROM generate_series(1, 400) g",
        )
        lost = (await source.log_state()).incarnation  # type: ignore[union-attr]
        bus.change_log(EPOCH_CHANGES_BUCKET).vanish()

        with caplog.at_level(logging.ERROR):
            assert (await keeper.run_pass()).restored is None
        assert any("NOT restored" in record.getMessage() for record in caplog.records)
        state = await source.log_state()
        assert state is not None and state.continues is None, "never labelled as continuing the lost log"
        assert state.last_seq == 0, "no record of the refused restore was appended"
        ring_state = await keeper.ring()
        assert ring_state.log_incarnation == state.incarnation != lost, "the new log is taken as a fresh one"
        # the next pass does not try again
        again = await keeper.run_pass()
        assert again.restored is None


async def _triggers(pool: asyncpg.Pool, schema: str, table: str) -> set[str]:
    return {
        str(row["tgname"])
        for row in await pool.fetch(
            "SELECT g.tgname FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 AND c.relname = $2",
            schema,
            table,
        )
    }


async def _give_as_its_schemas_run_does(pool: asyncpg.Pool, notes_schema: str, schema: str, noted: NotedTable) -> None:
    """a table of another schema given its note triggers as that schema's migration run gives them."""
    await pool.execute(
        trigger_function_sql(
            notes_schema=notes_schema,
            table_schema=schema,
            table=noted.table,
            keys=noted.keys,
            columns=noted.columns,
            tenancy=noted.tenancy,
        )
    )
    for statement in note_trigger_sql(notes_schema=notes_schema, table_schema=schema, table=noted.table).values():
        await pool.execute(statement)


class TestTheKeeperGivesTriggersInItsOwnSchemaOnly:
    async def test_a_table_of_the_same_name_in_another_schema_is_said_and_counted_never_given(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Flips if the keeper gives another schema's table of a noted name the triggers its own classes declare
        (a pod's table keyed or tenanted otherwise), or leaves one without them unsaid."""
        pool, schema, table = ring
        other = f"{schema}_pod"
        await pool.execute(f"CREATE SCHEMA {other}")
        try:
            # the pod's own table: the same name, another key
            await pool.execute(f"CREATE TABLE {other}.{table} (pod_key TEXT PRIMARY KEY, owner TEXT)")
            keeper = ChangeNoteKeeper(
                pool, notes_schema=schema, log=EpochGenerationSource(FakeNatsClient()), fence=_FENCE
            )
            await keeper.install()
            noted = [NotedTable(table=table, keys=("id",), columns=("owner",))]
            with caplog.at_level(logging.WARNING):
                assert await keeper.install_triggers(noted) == [f"{schema}.{table}"]
                assert await keeper.install_triggers(noted) == [], "nothing more to give, the pod's table never"
            assert {NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER} <= await _triggers(pool, schema, table)
            assert await _triggers(pool, other, table) == set(), "the keeper gave the pod's table nothing"
            said = [r.__dict__["extra_data"] for r in caplog.records if "another schema" in r.getMessage()]
            assert said == [{"table": f"{other}.{table}", "keeper_schema": schema}], "said once"
            assert REGISTRY.get_sample_value("threetears_change_notes_untriggered_tables") == 1

            # its schema's migration run gives it its own triggers: it is no longer counted, and the keeper's
            # next install leaves its function (its own key) alone
            pod_noted = NotedTable(table=table, keys=("pod_key",))
            await _give_as_its_schemas_run_does(pool, schema, other, pod_noted)
            assert await keeper.install_triggers(noted) == []
            assert REGISTRY.get_sample_value("threetears_change_notes_untriggered_tables") == 0
            await pool.execute(f"INSERT INTO {other}.{table} VALUES ('p1', 'o')")
            current = (await keeper.ring()).current
            keys = [
                list(note["keys"])
                for note in await pool.fetch(f"SELECT keys FROM {schema}.{note_table(current)}")  # noqa: S608
            ]
            assert keys == [["p1"]], "the pod's note names its own key"
        finally:
            await pool.execute(f"DROP SCHEMA IF EXISTS {other} CASCADE")


class TestAColumnWhoseTextIsNotExactGetsNoTrigger:
    async def test_a_boolean_key_or_a_timestamp_column_is_refused_with_an_error(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        pool, schema, table = ring
        odd = f"{table}_odd"
        await pool.execute(f"CREATE TABLE {schema}.{odd} (id BOOLEAN PRIMARY KEY, at TIMESTAMP)")
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=EpochGenerationSource(FakeNatsClient()), fence=_FENCE)
        await keeper.install()
        with caplog.at_level(logging.ERROR):
            given = await keeper.install_triggers(
                [NotedTable(table=odd, keys=("id",), columns=("at",)), NotedTable(table=table, keys=("id",))]
            )
        assert given == [f"{schema}.{table}"], "the exact table is given its trigger, the other is refused"
        said = [r.__dict__["extra_data"] for r in caplog.records if "no change-note trigger" in r.getMessage()]
        assert said == [{"table": f"{schema}.{odd}", "columns": ["id (boolean)", "at (timestamp without time zone)"]}]


class TestATimestamptzKeyIsNotedInOneSpelling:
    async def test_its_notes_name_each_instant_as_its_record_does_and_bind_back_to_its_rows(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        """Flips if a timestamptz key's note is spelled by the session's TimeZone or DateStyle rather than in UTC to
        the microsecond (so a note and its record name one row two ways), or a key read back from the notes fails to
        bind to its row."""
        pool, schema, table = ring
        snaps = f"{table}_snaps"
        await pool.execute(
            f"CREATE TABLE {schema}.{snaps} (office TEXT, at TIMESTAMPTZ, seen TIMESTAMPTZ, n INT, "
            "PRIMARY KEY (office, at))"
        )
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=EpochGenerationSource(FakeNatsClient()), fence=_FENCE)
        await keeper.install()
        assert await keeper.install_triggers([NotedTable(table=snaps, keys=("office", "at"), columns=("seen",))]) == [
            f"{schema}.{snaps}"
        ]
        pacific = timezone(timedelta(hours=-8))
        instants = [
            datetime(2026, 11, 3, 19, 0, 0, tzinfo=pacific),  # a whole second: no fraction in str()
            datetime(2026, 11, 4, 3, 0, 0, 82658, tzinfo=UTC),
            datetime(1999, 12, 31, 23, 59, 59, 999999, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        ]
        async with pool.acquire() as conn:
            # a writer whose session would spell a timestamptz otherwise
            await conn.execute("SET TimeZone = 'America/New_York'; SET DateStyle = 'SQL, DMY'")
            for n, at in enumerate(instants):
                await conn.execute(f"INSERT INTO {schema}.{snaps} VALUES ('pres', $1, $1, $2)", at, n)
        current = (await keeper.ring()).current
        notes = await pool.fetch(
            f"SELECT keys, columns FROM {schema}.{note_table(current)} WHERE table_name = $1",  # noqa: S608
            snaps,
        )
        named = sorted(tuple(note["keys"]) for note in notes)
        assert named == sorted(("pres", key_text(at)) for at in instants), "the record's spelling, exactly"
        assert sorted(json.loads(note["columns"])["seen"] for note in notes) == sorted(key_text(at) for at in instants)
        assert key_text(instants[0]) == "2026-11-04T03:00:00.000000+00:00"
        # the keys read back bind to their rows, as a follower's read by key binds them (each key's text cast to its
        # column's type, CopyFollower's keyed read)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT t.n, t.at FROM {schema}.{snaps} AS t JOIN (SELECT unnest($1::text[]) AS k0, "  # noqa: S608
                "unnest($2::text[]) AS k1) AS k ON t.office = k.k0 AND t.at = CAST(k.k1 AS timestamptz)",
                [key[0] for key in named],
                [key[1] for key in named],
            )
        assert sorted(row["n"] for row in rows) == [0, 1, 2]
        assert sorted(row["at"] for row in rows) == sorted(instants)


class TestTheScrubRunsOnItsOwnInterval:
    async def test_a_pass_inside_the_interval_does_not_scrub_and_the_notes_are_indexed(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        source = EpochGenerationSource(FakeNatsClient())
        await source.start_log()
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=source, fence=timedelta(minutes=10))
        await keeper.install()
        first = await keeper.run_pass()
        second = await keeper.run_pass()
        assert first.scrub is not second.scrub
        assert second.scrub == ScrubPass(), "a minute (a tenth of the fence) has not passed"
        indexed = {
            str(row["indexdef"])
            for row in await pool.fetch("SELECT indexdef FROM pg_indexes WHERE schemaname = $1", schema)
        }
        for index in range(NOTE_RING_SIZE):
            # Postgres says "(key_text, table_name)", YugabyteDB "lsm (key_text HASH, table_name ASC)"
            assert any(
                note_table(index) in each
                and 0 <= each.find("key_text") < each.find("table_name", each.find("key_text"))
                for each in indexed
            ), index


class TestARestoreTooBigForTheLogPutsBackItsNewestPart:
    async def test_the_floor_is_the_newest_fitting_mark_a_follower_past_it_catches_up_one_before_is_behind(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        pool, schema, table = ring
        bus = FakeNatsClient()
        retention = EpochChangeRetention(max_bytes=16 * 1024)

        def pod(scope: str) -> _Pod:
            made = _Pod(bus, pool, table, scope)  # type: ignore[arg-type]
            made.source = EpochGenerationSource(
                bus, change_retention=retention, restored_from_l3=True, restore_wait=timedelta(seconds=2)
            )
            made.registry.set_generation_source(made.source)
            return made

        writer, old, new = pod("writer"), pod("old"), pod("new")
        await writer.source.start_log()
        keeper = ChangeNoteKeeper(
            pool, notes_schema=schema, log=writer.source, fence=_FENCE, scrub_every=timedelta(0), switch_bytes=10**9
        )
        await keeper.install()
        await keeper.install_triggers([NotedTable(table=table, keys=("id",), columns=("owner",))])
        await keeper.run_pass()
        await pool.execute(f"UPDATE {schema}.threetears_change_note_ring SET switched_at = now() - interval '1 hour'")
        await keeper.run_pass()

        await writer.rows.save_entity(writer.rows.create({"id": "w0", "owner": "o", "n": "1"}))
        old.registry.follow_generation(table)
        await generation_catchup_tick(old.registry, old.reader)  # its mark: before the bulk below
        # far more distinct keys than the log can take back: notes with no epoch of their own
        await pool.execute(f"INSERT INTO {table} SELECT 'bulk-' || g, 'o', '1' FROM generate_series(1, 3000) g")
        await writer.rows.save_entity(writer.rows.create({"id": "w1", "owner": "o", "n": "1"}))
        # past half the wait, so the mark this pass takes (one every fence / 20) starts after the bulk
        await asyncio.sleep(_FENCE.total_seconds())
        await keeper.run_pass()
        mark = (await writer.source.log_state()).last_seq  # type: ignore[union-attr]
        state_lost = (await writer.source.log_state()).incarnation  # type: ignore[union-attr]
        # many later marks, each vouching for less: the binary search must still take the oldest that fits
        await pool.executemany(
            f"INSERT INTO {schema}.threetears_change_note_marks (log_incarnation, seq, at) "
            "VALUES ($1, $2, now() + make_interval(secs => $3))",
            [(state_lost, mark + n, float(n)) for n in range(1, 41)],
        )
        new.registry.follow_generation(table)
        await generation_catchup_tick(new.registry, new.reader)  # its mark: at the mark's position
        await writer.rows.save_entity(writer.rows.create({"id": "w2", "owner": "o", "n": "after"}))
        lost = (await writer.source.log_state()).incarnation  # type: ignore[union-attr]
        bus.change_log(EPOCH_CHANGES_BUCKET).vanish()

        with caplog.at_level(logging.ERROR):
            restored = (await keeper.run_pass()).restored
        assert restored is not None
        state = await writer.source.log_state()
        assert state is not None and state.continues == lost and state.covered_from == mark
        said = [r.__dict__["extra_data"] for r in caplog.records if "only the newest change notes" in r.getMessage()]
        assert len(said) == 1 and said[0]["covered_from"] == f"{lost}:{mark}" and said[0]["notes_left_out"] >= 3000

        await generation_catchup_tick(new.registry, new.reader)
        assert not new.registry.behind_on(table), "a follower at the floor catches up"
        assert (await new.rows.ensure("w2") or {}).get("n") == "after"
        await generation_catchup_tick(old.registry, old.reader)
        assert old.registry.behind_on(table), "a follower older than the floor is behind (record_lost)"


class TestTheSwitchSizeIsDerivedFromTheRestoreBudget:
    async def test_the_ring_switches_at_half_the_restore_budget_not_a_whole_log(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        max_bytes = 64 * 1024
        source = EpochGenerationSource(FakeNatsClient(), change_retention=EpochChangeRetention(max_bytes=max_bytes))
        await source.start_log()
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=source, fence=_FENCE, scrub_every=timedelta(0))
        await keeper.install()
        await keeper.run_pass()
        await pool.execute(f"UPDATE {schema}.threetears_change_note_ring SET switched_at = now() - interval '1 hour'")
        assert (await keeper.run_pass()).switched_to is None
        threshold = max_bytes * RESTORE_BUDGET_RATIO / 2
        switched: int | None = None
        while switched is None:
            await source.advance_with_change(table, [ChangedRow(ids=[f"k{uuid.uuid4().hex}"])])
            state = await source.log_state()
            assert state is not None
            switched = (await keeper.run_pass()).switched_to
            if switched is None:
                assert state.bytes < threshold * 1.2, "it would have switched by now"
        assert state is not None
        assert state.bytes < max_bytes / 2, "far before one whole log's worth"
        assert state.bytes >= threshold * 0.8


class TestALivingKeeperAndWorkerSayWhenTheyLastPassed:
    async def test_a_keeper_pass_and_a_worker_pass_move_their_heartbeats(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        source = EpochGenerationSource(FakeNatsClient())
        await source.start_log()
        keeper = await _keeper(pool, schema, table, source)
        before = time.time()
        await keeper.run_pass()
        beat = REGISTRY.get_sample_value("threetears_change_notes_keeper_last_pass_timestamp_seconds")
        assert beat is not None and beat >= before

        class _OnePass:
            lost_reason: object = None

        held = _OnePass()

        class _Pass:
            #: the worker turns a saver's batched reads on (its runner is the infrastructure identity)
            batched_reads: bool | None = None

            async def run_pass(self) -> None:
                held.lost_reason = "done"  # one pass, then the lease is gone

        worker_before = time.time()
        await run_l2_first_worker(held, saver=_Pass(), refiller=_Pass(), every=timedelta(0))  # type: ignore[arg-type]
        worker_beat = REGISTRY.get_sample_value("threetears_l2_first_worker_last_pass_timestamp_seconds")
        assert worker_beat is not None and worker_beat >= worker_before


class TestARestoreStreamsTheNotes:
    async def test_far_more_notes_than_the_log_s_bytes_imply_restore_as_their_distinct_rows(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus, restored_from_l3=True, restore_wait=timedelta(seconds=1))
        await source.start_log()
        keeper = await _fenced(pool, schema, table, source)
        await pool.execute(f"INSERT INTO {table} SELECT 'k' || g, 'o', '0' FROM generate_series(1, 10) g")
        # one raw-SQL statement per round, twenty thousand notes over ten rows, and one small epoch
        for round_ in range(2000):
            await pool.execute(f"UPDATE {table} SET n = $1", str(round_))
        await source.advance(table)
        notes = await pool.fetchval(f"SELECT count(*) FROM {schema}.{note_table((await keeper.ring()).current)}")
        assert notes >= 20_000
        bus.change_log(EPOCH_CHANGES_BUCKET).vanish()

        assert (await keeper.run_pass()).restored is not None
        reader = EpochGenerationReader(bus)
        state = await source.log_state()
        assert state is not None and state.continues is not None
        walk = await reader.changes_after(table, f"{state.incarnation}:0", f"{state.incarnation}:{state.last_seq}")
        restored = sorted(row.ids[0] for change in walk.changes for row in change.rows)
        assert restored == sorted(f"k{g}" for g in range(1, 11)), "each distinct row once"

    async def test_the_keeper_declares_then_reads_the_notes_then_appends(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus, restored_from_l3=True, restore_wait=timedelta(seconds=1))
        await source.start_log()
        steps: list[str] = []

        class _Ordered:
            """the keeper's log, saying in which order the keeper used it."""

            def __getattr__(self, name: str) -> object:
                return getattr(source, name)

            async def begin_restore(self, lost: str) -> str | None:
                restored = await source.begin_restore(lost)
                steps.append("declared")
                # a writer, as soon as the log is there: its row is written after the declaration
                await pool.execute(f"INSERT INTO {table} VALUES ('after-declare', 'o', '1')")
                await source.advance_with_change(table, [ChangedRow(ids=["after-declare"])])
                return restored

            async def finish_restore(self, changes: object, **kwargs: object) -> int:
                steps.append("appended")
                return await source.finish_restore(changes, **kwargs)  # type: ignore[arg-type]

        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=_Ordered(), fence=_FENCE, scrub_every=timedelta(0))  # type: ignore[arg-type]
        await keeper.install()
        await keeper.install_triggers([NotedTable(table=table, keys=("id",))])
        await keeper.run_pass()
        await pool.execute(f"UPDATE {schema}.threetears_change_note_ring SET switched_at = now() - interval '1 hour'")
        await keeper.run_pass()
        await pool.execute(f"INSERT INTO {table} VALUES ('before', 'o', '1')")
        bus.change_log(EPOCH_CHANGES_BUCKET).vanish()

        restored = (await keeper.run_pass()).restored
        assert restored is not None
        assert steps == ["declared", "appended"]
        state = await source.log_state()
        assert state is not None
        walk = await EpochGenerationReader(bus).changes_after(table, f"{restored}:0", f"{restored}:{state.last_seq}")
        named = [row.ids[0] for change in walk.changes for row in change.rows]
        assert named[0] == "after-declare", "the writer's record landed in the declared log first"
        assert "before" in named[1:] and "after-declare" in named[1:], "the notes were read after the declaration"


class TestTheRingSwitchesOnNotesToo:
    async def test_a_bulk_write_with_one_small_epoch_switches_the_ring(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        source = EpochGenerationSource(
            FakeNatsClient(), change_retention=EpochChangeRetention(max_bytes=64 * 1024 * 1024)
        )
        await source.start_log()
        keeper = ChangeNoteKeeper(
            pool, notes_schema=schema, log=source, fence=_FENCE, scrub_every=timedelta(0), switch_notes=500
        )
        await keeper.install()
        await keeper.install_triggers([NotedTable(table=table, keys=("id",))])
        await keeper.run_pass()
        await pool.execute(f"UPDATE {schema}.threetears_change_note_ring SET switched_at = now() - interval '1 hour'")
        await keeper.run_pass()
        await pool.execute(f"INSERT INTO {table} SELECT 'b' || g, 'o', '1' FROM generate_series(1, 600) g")
        await source.advance(table)  # one small record: far from the bytes the ring switches at
        await asyncio.sleep(_FENCE.total_seconds() / 10)  # past the count's interval
        assert (await keeper.run_pass()).switched_to == 1


class TestARecoveryNamesTheRowsTheNotesName:
    async def test_a_missed_write_s_epoch_names_its_rows_and_a_truncate_s_the_whole_table(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        await _keeper(pool, schema, table, source)
        reconciler = await _Recovery.start(pool, source, schema, {table: (schema,)})
        reader = EpochGenerationReader(bus)

        async def last() -> Any:
            token = await reader.read(table)
            assert token is not None
            return await reader.change(table, token)

        # raw SQL: a note, and no advance
        await pool.execute(f"INSERT INTO {table} VALUES ('raw-1', 'ada', '1'), ('raw-2', 'bea', '1')")
        assert (await reconciler.run_pass()).advanced == (table,)
        record = await last()
        assert not record.whole_table
        assert sorted((row.ids[0], (row.columns or {}).get("owner")) for row in record.rows) == [
            ("raw-1", "ada"),
            ("raw-2", "bea"),
        ], "the rows the notes name, with their invalidation columns: no follower drops the table"

        # a truncate leaves a whole-table note: a window holding it records the whole table
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(f"TRUNCATE {table}")
            await conn.execute(f"INSERT INTO {table} VALUES ('reloaded', 'o', '1')")
        await reconciler.run_pass()
        assert (await last()).whole_table, "a truncate's note names the whole table, and so does its epoch"

        # a window older than the coverage (the ring has switched since; a writer dead a day): the notes cannot vouch
        # for it, so keys unknown (every follower behind on the table, none told to drop it), said and counted
        await pool.execute(
            f"UPDATE {schema}.threetears_change_note_ring SET switch_no = switch_no + 1, previous_switched_at = now()"
        )
        await pool.execute(f"INSERT INTO {table} VALUES ('old', 'cy', '1')")
        reconciler.back(timedelta(days=1))
        with caplog.at_level(logging.WARNING):
            await reconciler.run_pass()
        unnamed = await last()
        assert unnamed.keys_unknown and not unnamed.whole_table
        said = [r.__dict__["extra_data"] for r in caplog.records if "advanced keys unknown" in r.getMessage()]
        assert said == [{"table": table, "reason": "not_covered", "whole_changes": 0}]


class TestARecoveryKeysEachSchemasChange:
    """under TENANT_BY_SCHEMA every agent schema's notes of a table share the ring, and a window names no schema."""

    async def test_one_schemas_truncate_never_hides_anothers_missed_rows(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        other = f"{schema}_b"

        class _PerSchema(BaseCollection[Any]):
            """the table, declared per schema, so a recovery and a pod's reader key it by schema."""

            tenancy = TENANT_BY_SCHEMA

            @property
            def table_name(self) -> str:
                return table

        assert tenancy_of_table(table) == _PerSchema.tenancy
        await pool.execute(f"CREATE SCHEMA {other}")
        try:
            await pool.execute(f"CREATE TABLE {other}.{table} (id TEXT PRIMARY KEY, owner TEXT, n TEXT)")
            bus = FakeNatsClient()
            source = EpochGenerationSource(bus)
            await source.start_log()
            keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=source, fence=_FENCE, scrub_every=timedelta(0))
            await keeper.install()
            noted = NotedTable(table=table, keys=("id",), tenancy=TENANT_BY_SCHEMA)
            await keeper.install_triggers([noted])
            # the other schema's table is given its triggers by its own schema's migration run, never the keeper
            await _give_as_its_schemas_run_does(pool, schema, other, noted)
            reconciler = await _Recovery.start(pool, source, schema, {table: (schema, other)})
            await pool.execute(f"INSERT INTO {other}.{table} VALUES ('seed', 'o', '1')")
            await reconciler.run_pass()
            mine = EpochGenerationReader(bus, tenants=PodTenants(schema=schema))
            theirs = EpochGenerationReader(bus, tenants=PodTenants(schema=other))
            mark_mine, mark_theirs = await mine.read(table), await theirs.read(table)
            assert mark_mine is not None and mark_theirs is not None

            # agent A's writer commits rows and dies before its advance; agent B's table is truncated meanwhile
            await pool.execute(f"INSERT INTO {schema}.{table} VALUES ('a1', 'o', '1'), ('a2', 'o', '1')")
            async with pool.acquire() as conn, conn.transaction():
                await conn.execute(f"TRUNCATE {other}.{table}")
                await conn.execute(f"INSERT INTO {other}.{table} VALUES ('b1', 'o', '1')")
            await reconciler.run_pass()

            head = await mine.read(table)
            assert head is not None and head != mark_mine, "A's pods see their own missed rows"
            walk = await mine.changes_after(table, mark_mine, head)
            assert walk.complete
            assert sorted(row.ids[0] for change in walk.changes for row in change.rows) == ["a1", "a2"]
            assert not any(change.whole_table for change in walk.changes), "B's truncate is not A's"
            head = await theirs.read(table)
            assert head is not None
            walk = await theirs.changes_after(table, mark_theirs, head)
            assert walk.complete and any(change.whole_table for change in walk.changes), "B's truncate is B's"
            assert not any(row.ids[0].startswith("a") for change in walk.changes for row in change.rows)
        finally:
            await pool.execute(f"DROP SCHEMA IF EXISTS {other} CASCADE")

    async def test_unnamed_writes_are_keys_unknown_under_every_schema_holding_the_table(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        other = f"{schema}_b"

        class _PerSchema(BaseCollection[Any]):
            tenancy = TENANT_BY_SCHEMA

            @property
            def table_name(self) -> str:
                return table

        await pool.execute(f"CREATE SCHEMA {other}")
        try:
            await pool.execute(f"CREATE TABLE {other}.{table} (id TEXT PRIMARY KEY, owner TEXT, n TEXT)")
            bus = FakeNatsClient()
            source = EpochGenerationSource(bus)
            await source.start_log()
            # no change-note ring at all: the notes cannot name anything (notes_unavailable)
            reconciler = await _Recovery.start(pool, source, schema, {table: (schema, other)})
            await pool.execute(f"INSERT INTO {schema}.{table} VALUES ('a1', 'o', '1')")
            await reconciler.run_pass()
            log = bus.change_log(EPOCH_CHANGES_BUCKET)
            entries = await log.records_after(f"*.{table}.e", 0, 100)
            assert sorted(entry.key for entry in entries) == sorted(
                [
                    f"{schema_tenant(schema)}.{table}.e",
                    f"{schema_tenant(other)}.{table}.e",
                ]
            ), "one keys-unknown record under each schema holding the table, none on _all"
            for entry in entries:
                assert read_epoch_change_body(entry.value, table, f"log:{entry.seq}").keys_unknown
        finally:
            await pool.execute(f"DROP SCHEMA IF EXISTS {other} CASCADE")


class TestARecoveredWriteTooLargeToRecordReachesAFollower:
    async def test_the_follower_applies_every_noted_key_and_is_current_without_a_restart(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        """Flips if a recovery records a lost write of more rows than one record names as keys unknown with no notes
        bracket: every follower is then behind on the table until it restarts, and the rows stay stale in it."""
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        await _keeper(pool, schema, table, source)
        reconciler = await _Recovery.start(pool, source, schema, {table: (schema,)})
        reader = EpochGenerationReader(bus)

        # a follower whose reader names a commit's keys from its notes, as a pod's broker transport does
        registry = CollectionRegistry()
        registry.follow_generation(table)
        registry.set_noted_keys_reader(PoolNotedKeysReader(pool, notes_schema=schema))
        heard: set[str] = set()
        dropped: list[bool] = []
        registry.register_derived_cache(
            table, on_row=lambda message: heard.add(str(message.ids[0])), on_table_dropped=lambda: dropped.append(True)
        )
        await pool.execute(f"INSERT INTO {table} VALUES ('seed', 'o', '1')")
        await reconciler.run_pass()
        await registry.catch_up_generation(table, await reader.read(table), changes=reader)
        assert registry.behind_reason(table) is None

        # one transaction writes more rows than one record names, and its writer dies before its advance
        many = WHOLE_COPY_ROWS + 1
        await pool.execute(
            f"INSERT INTO {table} SELECT 'many-' || g, 'o', '1' FROM generate_series(1, $1::int) g",  # noqa: S608
            many,
        )
        assert (await reconciler.run_pass()).advanced == (table,)
        for _ in range(2):
            # a second pass changes nothing: the follower is current, not behind until its next start
            await registry.catch_up_generation(table, await reader.read(table), changes=reader)
            assert registry.behind_reason(table) is None
        assert heard >= {f"many-{g}" for g in range(1, many + 1)}, "every row the lost advance changed is evicted"
        assert not dropped, "never by dropping the table"

    async def test_a_truncate_in_the_window_is_recorded_whole_beside_the_bracket(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        """Flips if a whole-table note inside a window of more rows than one record names is dropped: the truncate's
        own advance was lost too, so the bracket's keys alone leave every row the truncate removed in a follower (red on
        3d221b26e: only the bracket was recorded)."""
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        await _keeper(pool, schema, table, source)
        reconciler = await _Recovery.start(pool, source, schema, {table: (schema,)})
        reader = EpochGenerationReader(bus)
        registry = CollectionRegistry()
        registry.follow_generation(table)
        registry.set_noted_keys_reader(PoolNotedKeysReader(pool, notes_schema=schema))
        heard: set[str] = set()
        dropped: list[bool] = []
        registry.register_derived_cache(
            table, on_row=lambda message: heard.add(str(message.ids[0])), on_table_dropped=lambda: dropped.append(True)
        )
        await pool.execute(f"INSERT INTO {table} VALUES ('seed', 'o', '1')")
        await reconciler.run_pass()
        await registry.catch_up_generation(table, await reader.read(table), changes=reader)
        seen = await reader.read(table)

        # a truncate and a reload of more rows than one record names, in one transaction, whose advance was lost
        many = WHOLE_COPY_ROWS + 1
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(f"TRUNCATE {table}")
            await conn.execute(
                f"INSERT INTO {table} SELECT 'many-' || g, 'o', '1' FROM generate_series(1, $1::int) g",  # noqa: S608
                many,
            )
        assert (await reconciler.run_pass()).advanced == (table,)
        token = await reader.read(table)
        assert token is not None and seen is not None
        walk = await reader.changes_after(table, seen, token)
        kinds = sorted("whole" if change.whole_table else "bracket" for change in walk.changes)
        # the truncate's whole table (a platform table's: _all, every row) covers the bracket's keys, so it alone
        assert kinds == ["whole"], "the truncate recorded whole, and no bracket under the tenant it covers"
        await registry.catch_up_generation(table, token, changes=reader)
        assert registry.behind_reason(table) is None
        assert dropped, "the truncate reaches the follower: 'seed' is gone from it"


class TestAFollowerOnADirectPoolReadsTheNotesItself:
    async def test_a_bracketed_keys_unknown_record_is_applied_from_the_notes_and_the_follower_converges(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        """Flips if a registry on a direct L3 pool (the hub's own followers: access tables, registry auth) has no
        noted-keys reader: a keys-unknown record carrying its notes bracket then leaves it behind on the table on
        every catch-up pass until it restarts, the lost write's rows stale in it (RC1, 2026-11-16)."""
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        await _keeper(pool, schema, table, source)
        reconciler = await _Recovery.start(pool, source, schema, {table: (schema,)})
        reader = EpochGenerationReader(bus)

        # wired as the hub wires its own registries: on the direct pool, with a reader of the notes over it
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        registry.set_noted_keys_reader(PoolNotedKeysReader(pool, notes_schema=schema, every_tenant=True))
        registry.follow_generation(table)
        heard: set[str] = set()
        dropped: list[bool] = []
        registry.register_derived_cache(
            table, on_row=lambda message: heard.add(str(message.ids[0])), on_table_dropped=lambda: dropped.append(True)
        )
        await pool.execute(f"INSERT INTO {table} VALUES ('seed', 'o', '1')")
        await reconciler.run_pass()
        await registry.catch_up_generation(table, await reader.read(table), changes=reader)
        assert registry.behind_reason(table) is None

        # a write of more rows than one record names, whose advance was lost: recorded keys unknown with its bracket
        many = WHOLE_COPY_ROWS + 1
        await pool.execute(
            f"INSERT INTO {table} SELECT 'many-' || g, 'o', '1' FROM generate_series(1, $1::int) g",  # noqa: S608
            many,
        )
        assert (await reconciler.run_pass()).advanced == (table,)
        token = await reader.read(table)
        assert token is not None
        record = await reader.change(table, token)
        assert record is not None
        assert record.keys_unknown and record.noted_between is not None
        for _ in range(2):
            await registry.catch_up_generation(table, await reader.read(table), changes=reader)
            assert registry.behind_reason(table) is None, "current from the notes, not behind until a restart"
        assert heard >= {f"many-{g}" for g in range(1, many + 1)}, "every row the lost write changed is evicted"
        assert not dropped, "never by dropping the table"


class TestANoteCarriesItsTenant:
    async def test_the_trigger_names_the_tenant_as_the_records_do_and_the_restore_keys_by_it(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus, restored_from_l3=True, restore_wait=timedelta(seconds=1))
        await source.start_log()
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=source, fence=_FENCE, scrub_every=timedelta(0))
        await keeper.install()
        await keeper.install_triggers([NotedTable(table=table, keys=("id",), tenancy=tenant_column("owner"))])
        await keeper.run_pass()
        await pool.execute(f"UPDATE {schema}.threetears_change_note_ring SET switched_at = now() - interval '1 hour'")
        await keeper.run_pass()
        long_owner = "a b/" + "x" * 80
        await pool.execute(
            f"INSERT INTO {table} VALUES ('r1', 'ada', '1'), ('r2', $1, '1'), ('r3', NULL, '1')", long_owner
        )
        current = (await keeper.ring()).current
        tenants = {
            list(note["keys"])[0]: note["tenant"]
            for note in await pool.fetch(f"SELECT keys, tenant FROM {schema}.{note_table(current)}")
        }
        assert tenants == {"r1": value_tenant("ada"), "r2": value_tenant(long_owner), "r3": None}, (
            "the SQL token rule is the Python one, a long or unsafe value digested the same way"
        )

        lost = (await source.log_state()).incarnation  # type: ignore[union-attr]
        bus.change_log(EPOCH_CHANGES_BUCKET).vanish()
        restored = (await keeper.run_pass()).restored
        assert restored is not None
        state = await source.log_state()
        assert state is not None and state.continues == lost

        class _Owned(BaseCollection[Any]):
            """the ring's table, declared so a pod's reader reads it by its owner column."""

            tenancy = tenant_column("owner")

            @property
            def table_name(self) -> str:
                return table

        assert tenancy_of_table(table) == _Owned.tenancy

        async def ids(owner: str) -> list[str]:
            reader = EpochGenerationReader(bus, tenants=PodTenants(customer=owner))
            head = await reader.read(table)
            assert head is not None
            walk = await reader.changes_after(table, f"{restored}:0", head)
            return sorted(row.ids[0] for change in walk.changes for row in change.rows)

        assert await ids("ada") == ["r1", "r3"], "ada's row and the tenant-less row on _all"
        assert await ids(long_owner) == ["r2", "r3"]

    async def test_a_schema_tenant_is_the_rows_schema(self, ring: tuple[asyncpg.Pool, str, str]) -> None:
        pool, schema, table = ring
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=EpochGenerationSource(FakeNatsClient()), fence=_FENCE)
        await keeper.install()
        await keeper.install_triggers([NotedTable(table=table, keys=("id",), tenancy=TENANT_BY_SCHEMA)])
        await pool.execute(f"INSERT INTO {table} VALUES ('s1', 'o', '1')")
        current = (await keeper.ring()).current
        [tenant] = [note["tenant"] for note in await pool.fetch(f"SELECT tenant FROM {schema}.{note_table(current)}")]
        assert tenant == schema_tenant(schema)
        # a truncate of a schema table belongs to that schema alone: its whole-table note says so
        await pool.execute(f"TRUNCATE {table}")
        whole = [
            note["tenant"]
            for note in await pool.fetch(f"SELECT tenant FROM {schema}.{note_table(current)} WHERE key_text IS NULL")
        ]
        assert whole == [schema_tenant(schema)]

    async def test_an_install_replaces_the_shared_truncate_trigger_with_the_tables_own(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=EpochGenerationSource(FakeNatsClient()), fence=_FENCE)
        await keeper.install()
        noted = [NotedTable(table=table, keys=("id",), tenancy=TENANT_BY_SCHEMA)]
        await keeper.install_triggers(noted)
        # a table given its triggers by an install before a truncate's note named its tenant
        function = f"{schema}.{note_function_name(schema, table)}"
        await pool.execute(f"DROP TRIGGER {NOTE_TRUNCATE_TRIGGER} ON {table}")
        await pool.execute(
            f"CREATE TRIGGER {_LEGACY_TRUNCATE_TRIGGER} AFTER TRUNCATE ON {table} FOR EACH STATEMENT "
            f"EXECUTE FUNCTION {function}()"
        )
        await keeper.install_triggers(noted)
        triggers = {
            str(row["tgname"])
            for row in await pool.fetch(
                "SELECT g.tgname FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid WHERE c.relname = $1", table
            )
        }
        assert NOTE_TRUNCATE_TRIGGER in triggers and _LEGACY_TRUNCATE_TRIGGER not in triggers


@dataclass
class _Loaded:
    """a loaded table beside the ring: its name, a direct store over it, its keeper and the keeper's log."""

    table: str
    store: SqlL3Backend
    keeper: ChangeNoteKeeper
    bus: FakeNatsClient
    source: EpochGenerationSource


async def _loaded_table(pool: asyncpg.Pool, schema: str, table: str) -> _Loaded:
    """a table keyed by (race, geo) beside the ring, noted, a direct store over it, and a keeper whose ring has
    switched once, so its coverage holds the notes written next."""
    loaded = f"{table}_lsf"
    await pool.execute(f"CREATE TABLE {schema}.{loaded} (race TEXT, geo TEXT, votes TEXT, PRIMARY KEY (race, geo))")
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus, restored_from_l3=True, restore_wait=timedelta(seconds=1))
    await source.start_log()
    keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=source, fence=_FENCE, scrub_every=timedelta(0))
    await keeper.install()
    await keeper.install_triggers([NotedTable(table=loaded, keys=("race", "geo"))])
    await keeper.run_pass()
    await pool.execute(f"UPDATE {schema}.threetears_change_note_ring SET switched_at = now() - interval '1 hour'")
    await keeper.run_pass()
    store = SqlL3Backend(pool)
    store.register_schema(
        loaded,
        TableSchema(
            name=loaded,
            primary_key=("race", "geo"),
            columns=[
                SchemaColumn("race", STRING_TYPE),
                SchemaColumn("geo", STRING_TYPE),
                SchemaColumn("votes", STRING_TYPE),
            ],
            l1_eviction=TEST_L1_TABLE,
        ),
    )
    return _Loaded(table=loaded, store=store, keeper=keeper, bus=bus, source=source)


async def _load_race(store: SqlL3Backend, table: str, path: Any, race: str, *geos: str) -> None:
    """load ``geos`` of ``race`` from a parquet file, a transaction a row, naming the race as its key prefix."""
    file = path / f"{race}.parquet"
    pq.write_table(pa.Table.from_pylist([{"geo": geo, "votes": "1"} for geo in geos]), file)
    await store.load_files(
        table,
        [LoadFile(url=file.as_uri(), rows=len(geos))],
        columns=["geo", "votes"],
        constants={"race": race},
        rows_per_transaction=1,
        timeout_ms=10_000,
        key_prefixes=[{"race": race}],
    )


async def _restored_records(loaded: _Loaded) -> list[Any]:
    """lose the log, let the keeper put it back, and read what it appended for the loaded table."""
    source, table = loaded.source, loaded.table
    loaded.bus.change_log(EPOCH_CHANGES_BUCKET).vanish()
    assert (await loaded.keeper.run_pass()).restored is not None
    state = await source.log_state()
    assert state is not None and state.continues is not None
    records = [await source.epoch_at(seq) for seq in range(1, state.last_seq + 1)]
    return [record for record in records if record is not None and record.table == table]


class TestAWholeNoteNamesItsReachAndTheRestoreKeepsIt:
    async def test_a_load_s_note_names_its_prefixes_a_reset_s_the_reset_a_bare_truncate_nothing(
        self, ring: tuple[asyncpg.Pool, str, str], tmp_path: Any
    ) -> None:
        pool, schema, table = ring
        it = await _loaded_table(pool, schema, table)
        loaded, store, keeper = it.table, it.store, it.keeper
        await _load_race(store, loaded, tmp_path, "r1", "g1", "g2")
        await store.truncate(loaded, reason=RESET_REASON)
        await store.truncate(loaded)
        current = (await keeper.ring()).current
        notes = await pool.fetch(
            f"SELECT columns FROM {schema}.{note_table(current)} WHERE table_name = $1 AND key_text IS NULL "
            "ORDER BY noted_at",
            loaded,
        )
        reaches = [None if note["columns"] is None else json.loads(note["columns"]) for note in notes]
        assert reaches == [
            {"prefixes": [{"race": "r1"}]},
            {"prefixes": [{"race": "r1"}]},
            {"reason": RESET_REASON},
            None,
        ], "one whole note a load transaction naming its prefix; the reset's names the reset; a bare truncate's none"
        assert await pool.fetchval("SELECT current_setting($1, true)", NOTE_REACH_SETTING) in (None, ""), (
            "no connection is left naming a reach"
        )

    async def test_loads_restore_as_one_record_of_their_prefixes_and_a_row_beside_them_is_kept(
        self, ring: tuple[asyncpg.Pool, str, str], tmp_path: Any
    ) -> None:
        pool, schema, table = ring
        it = await _loaded_table(pool, schema, table)
        loaded, store = it.table, it.store
        await _load_race(store, loaded, tmp_path, "r1", "g1", "g2")
        await _load_race(store, loaded, tmp_path, "r2", "g1")
        await pool.execute(f"INSERT INTO {loaded} VALUES ('r3', 'g9', '1')")

        records = await _restored_records(it)
        whole = [record for record in records if record.whole_table]
        assert len(whole) == 1, "the table's whole notes fold into one record"
        assert sorted(prefix["race"] for prefix in whole[0].prefixes or []) == ["r1", "r2"], (
            "the loads' prefixes, never the plain whole table"
        )
        assert whole[0].reason is None
        named = [row.ids for record in records for row in record.rows]
        assert ["r3", "g9"] in named, "a row outside every prefix is restored by name"

    async def test_a_reset_restores_as_a_reset_whatever_was_loaded_after_it(
        self, ring: tuple[asyncpg.Pool, str, str], tmp_path: Any
    ) -> None:
        pool, schema, table = ring
        it = await _loaded_table(pool, schema, table)
        loaded, store = it.table, it.store
        await store.truncate(loaded, reason=RESET_REASON)
        await _load_race(store, loaded, tmp_path, "r1", "g1")

        records = await _restored_records(it)
        assert [(record.whole_table, record.reason, record.prefixes) for record in records] == [
            (True, RESET_REASON, None)
        ], "a reset, never a plain whole record, and nothing the reset already covers"

    async def test_a_bare_truncate_restores_plain_whole(
        self, ring: tuple[asyncpg.Pool, str, str], tmp_path: Any
    ) -> None:
        pool, schema, table = ring
        it = await _loaded_table(pool, schema, table)
        loaded, store = it.table, it.store
        await _load_race(store, loaded, tmp_path, "r1", "g1")
        await store.truncate(loaded)

        records = await _restored_records(it)
        assert [(record.whole_table, record.reason, record.prefixes) for record in records] == [(True, None, None)], (
            "a whole note naming no reach: the whole table, as an older writer's"
        )


class TestARingBesideALogAnotherKeeperKeeps:
    """a second database's ring (identity-core's) beside the change log the hub's keeper owns (RC3, 2026-11-16)."""

    async def test_it_never_declares_restores_or_marks_the_log_and_still_switches(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        """Flips if a ring beside another keeper's log declares the log or marks it (it would race the owner's
        restore), or if its scrub reads another database's epochs as missing notes."""
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus, database=schema)
        owners = EpochGenerationSource(bus, database=f"{schema}_owner")
        keeper = ChangeNoteKeeper(
            pool, notes_schema=schema, log=source, fence=_FENCE, scrub_every=timedelta(0), keeps_log=False
        )
        await keeper.install()
        await keeper.install_triggers([NotedTable(table=table, keys=("id",), columns=("owner",))])

        absent = await keeper.run_pass()
        assert (absent.started, absent.restored) == (None, None)
        assert await source.log_state() is None, "the log's owner declares it, never this ring"
        assert (await keeper.ring()).log_incarnation is None

        incarnation = await owners.start_log()  # the owner declares it
        adopted = await keeper.run_pass()
        assert adopted.adopted == incarnation and (await keeper.ring()).log_incarnation == incarnation
        await asyncio.sleep(_FENCE.total_seconds() * 2)
        await keeper.run_pass()
        assert (await keeper.ring()).fence_seq is not None, "the ring's fence passes as the owner's does"

        # an epoch of another database's table: no note of it here, and nothing says it is missing (its own scrub
        # skips it; RC8 gave the ring beside a scrub of its own tables, below)
        await owners.advance_with_change(f"{table}_theirs", [ChangedRow(ids=["another-database"])])
        await keeper.run_pass()
        scrub = (await keeper.run_pass()).scrub
        assert (scrub.checked, scrub.missing, scrub.others) == (0, 0, 1), "another database's epoch is skipped"
        assert await pool.fetchval(f"SELECT count(*) FROM {schema}.threetears_change_note_marks") == 0

        # its notes name its own writes, and the ring switches on the log's bytes as the owner's does
        await pool.execute(f"INSERT INTO {schema}.{table} VALUES ('mine', 'o', '1')")
        assert [row["keys"] for row in await _notes(pool, schema)] == [["mine"]]
        side = ChangeNoteKeeper(pool, notes_schema=schema, log=source, fence=_FENCE, switch_bytes=1, keeps_log=False)
        before = (await keeper.ring()).current
        assert (await side.run_pass()).switched_to == (before + 1) % NOTE_RING_SIZE

    async def test_its_scrub_reports_a_missing_note_of_its_own_tables_and_skips_every_other(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Flips if nobody scrubs the ring beside's tables (RC8: a missing note on an identity-core table was never
        reported, the owner's scrub skipping it as another database's), if its scrub reports under another metric
        than the owner's, or if it judges a table it does not note: another database's, or a plain table of its own
        schema whose name the owner's database also holds (identity's ``api_keys`` beside the hub's, on epochs)."""
        pool, schema, table = ring
        plain = f"{table}_plain"
        await pool.execute(f"CREATE TABLE {schema}.{plain} (id TEXT PRIMARY KEY)")
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus, database=schema)
        # the owner's writer, built before records named their database: its records are judged by their table
        owners = EpochGenerationSource(bus)
        await owners.start_log()  # the owner declares it
        side = ChangeNoteKeeper(
            pool, notes_schema=schema, log=source, fence=_FENCE, scrub_every=timedelta(0), keeps_log=False
        )
        await side.install()
        await side.install_triggers([NotedTable(table=table, keys=("id",), columns=("owner",))])
        await side.run_pass()
        await pool.execute(f"UPDATE {schema}.threetears_change_note_ring SET switched_at = now() - interval '1 hour'")
        await side.run_pass()
        assert (await side.ring()).fence_seq is not None

        await pool.execute(f"INSERT INTO {table} VALUES ('noted', 'o', '1')")
        await source.advance_with_change(table, [ChangedRow(ids=["noted"])])
        # an epoch of its own table whose write left no note: appended with no L3 write behind it
        await source.advance_with_change(table, [ChangedRow(ids=["unnoted"])])
        # the owner database's table of the same name as a plain table here, and one this database has not at all
        await owners.advance_with_change(plain, [ChangedRow(ids=["the-owners"])])
        await owners.advance_with_change(f"{table}_owners", [ChangedRow(ids=["the-owners"])])
        await side.run_pass()  # the scrub checks up to the head one pass ago
        counted = REGISTRY.get_sample_value("threetears_change_notes_missing_total", {"table": table}) or 0.0

        with caplog.at_level(logging.ERROR):
            scrub = (await side.run_pass()).scrub
        said = [r.__dict__["extra_data"]["missing"] for r in caplog.records if "no change note" in r.getMessage()]
        assert said == ["unnoted"], "its own table's missing note is said; no other table's epoch is"
        assert (scrub.checked, scrub.missing, scrub.others) == (2, 1, 2)
        assert REGISTRY.get_sample_value("threetears_change_notes_missing_total", {"table": table}) == counted + 1, (
            "counted on the owner's metric, so one alert covers every ring"
        )
        assert await pool.fetchval(f"SELECT count(*) FROM {schema}.threetears_change_note_marks") == 0, (
            "it scrubs, and still never marks the log"
        )

    @pytest.mark.parametrize("order", ["after_the_label", "before_the_label"])
    async def test_a_lost_log_gets_the_ring_besides_notes_back_and_its_follower_converges(
        self,
        ring: tuple[asyncpg.Pool, str, str],
        db_container: str,
        nats_container: str,
        order: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Flips if a lost log is put back from the owner's ring alone (RC5's restore gap): a follower of the other
        database's table carries its mark over and continues past that database's missed write, silently. Both
        orders: the ring beside puts its notes back after the owner's label, or into the restore before it."""
        pool, schema, table = ring
        beside, theirs = f"{schema}_beside", f"{table}_theirs"
        await pool.execute(f"CREATE SCHEMA {beside}")
        await pool.execute(f"CREATE TABLE {beside}.{theirs} (id TEXT PRIMARY KEY, owner TEXT, n TEXT)")
        # the other database: its own pool, its writers and followers over its own table and ring
        other = await asyncpg.create_pool(
            db_container, min_size=2, max_size=12, server_settings={"search_path": beside}
        )
        assert other is not None
        try:
            async with _nats(nats_container) as nc:
                writer = _Pod(nc, other, theirs, "writer", database=beside)
                follower = _Pod(nc, other, theirs, "follower", database=beside)
                owners = EpochGenerationSource(
                    nc, restored_from_l3=True, restore_wait=timedelta(seconds=5), database=schema
                )
                await owners.start_log()
                owner = await _fenced(pool, schema, table, owners)
                side = ChangeNoteKeeper(other, notes_schema=beside, log=writer.source, fence=_FENCE, keeps_log=False)
                await side.install()
                noted = NotedTable(table=theirs, keys=("id",), columns=("owner",))
                assert await side.install_triggers([noted]) == [f"{beside}.{theirs}"]
                await side.run_pass()
                await other.execute(
                    f"UPDATE {beside}.threetears_change_note_ring SET switched_at = now() - interval '1 hour'"
                )
                await side.run_pass()
                assert (await side.ring()).fence_seq is not None

                await writer.rows.save_entity(writer.rows.create({"id": "w1", "owner": "ada", "n": "1"}))
                follower.registry.follow_generation(theirs)
                await generation_catchup_tick(follower.registry, follower.reader)
                assert (await follower.rows.ensure("w1") or {}).get("n") == "1"
                # the follower hears nothing of this write: its listener never ran
                await writer.rows.save_entity(writer.rows.create({"id": "w1", "owner": "ada", "n": "2"}))
                lost = (await writer.source.log_state()).incarnation  # type: ignore[union-attr]

                await nc.jetstream_context().delete_stream(f"KV_{nc.namespace}-{EPOCH_CHANGES_BUCKET}")
                await side.run_pass()
                assert await writer.source.log_state() is None, "the ring beside never declares the log"
                if order == "before_the_label":
                    declared = await writer.source.begin_restore(lost)  # the owner's restore, declared, not labelled
                    side_pass = await side.run_pass()
                    restored = (await owner.run_pass()).restored
                    assert restored == declared
                else:
                    restored = (await owner.run_pass()).restored
                    side_pass = await side.run_pass()
                assert restored is not None and restored != lost
                continued = await follower.reader.continues(restored)
                assert continued is not None and continued[0] == lost, "the owner's label"

                with caplog.at_level(logging.ERROR):
                    await side.run_pass()  # the label's check, once it is written
                    await generation_catchup_tick(follower.registry, follower.reader)
                assert not follower.registry.behind_on(theirs)
                assert (await follower.rows.ensure("w1") or {}).get("n") == "2", (
                    "the other database's missed write reached its follower from its own ring's notes"
                )
                assert side_pass.put_back == restored
                beside_ring = await side.ring()
                assert beside_ring.log_incarnation == restored and beside_ring.floor == f"{restored}:0"
                keeper_errors = [
                    r.getMessage()
                    for r in caplog.records
                    if r.levelno >= logging.ERROR and r.name.endswith("change_note_keeper")
                ]
                assert keeper_errors == [], "both floors are the log's start: the label vouches for nothing uncovered"
        finally:
            await other.close()
            await pool.execute(f"DROP SCHEMA IF EXISTS {beside} CASCADE")

    async def test_a_ring_whose_floor_is_after_the_label_says_so(
        self, ring: tuple[asyncpg.Pool, str, str], nats_container: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Flips if a ring beside takes a restore whose label vouches for epochs its own notes may not cover (its floor
        is after the owner's covered_from) without saying so: a follower of its tables may have missed a write."""
        pool, schema, table = ring
        beside = f"{schema}_beside"
        await pool.execute(f"CREATE SCHEMA {beside}")
        try:
            async with _nats(nats_container) as nc:
                source = EpochGenerationSource(nc, restored_from_l3=True, restore_wait=timedelta(seconds=5))
                await source.start_log()
                owner = await _fenced(pool, schema, table, source)  # its floor: the log's start
                for key in ("a", "b", "c"):
                    await pool.execute(f"INSERT INTO {table} VALUES ($1, 'o', '1')", key)
                    await source.advance_with_change(table, [ChangedRow(ids=[key])])
                besides = EpochGenerationSource(
                    nc, restored_from_l3=True, restore_wait=timedelta(seconds=5), database=beside
                )
                side = ChangeNoteKeeper(pool, notes_schema=beside, log=besides, fence=_FENCE, keeps_log=False)
                await side.install()
                await side.run_pass()
                await pool.execute(
                    f"UPDATE {beside}.threetears_change_note_ring SET switched_at = now() - interval '1 hour'"
                )
                await side.run_pass()
                lost = (await source.log_state()).incarnation  # type: ignore[union-attr]
                assert (await side.ring()).fence_seq == f"{lost}:3", "the ring beside vouches only from here"

                await nc.jetstream_context().delete_stream(f"KV_{nc.namespace}-{EPOCH_CHANGES_BUCKET}")
                restored = (await owner.run_pass()).restored
                assert restored is not None and (await source.log_state()).covered_from == 0  # type: ignore[union-attr]
                with caplog.at_level(logging.ERROR):
                    await side.run_pass()
                said = [r.__dict__["extra_data"] for r in caplog.records if "label vouches" in r.getMessage()]
                assert said == [
                    {"lost": lost, "restored": restored, "covered_from": f"{lost}:0", "ring_floor": f"{lost}:3"}
                ]
                assert REGISTRY.get_sample_value(
                    "threetears_change_notes_put_back_short_total", {"reason": "floor_after_label"}
                ), "counted"
        finally:
            await pool.execute(f"DROP SCHEMA IF EXISTS {beside} CASCADE")

    async def test_the_owners_scrub_judges_only_the_tables_its_database_notes(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Flips if the owner's scrub reads another database's epochs on the shared log as missing notes (RC5: every
        identity write raised threetears_change_notes_missing_total), or stops judging a table of its own schema."""
        pool, schema, table = ring
        beside, theirs, bare = f"{schema}_beside", f"{table}_theirs", f"{table}_bare"
        await pool.execute(f"CREATE SCHEMA {beside}")
        await pool.execute(f"CREATE TABLE {beside}.{theirs} (id TEXT PRIMARY KEY, owner TEXT, n TEXT)")
        # a table of the owner's own schema with no note trigger (refused one, or lost it): still judged
        await pool.execute(f"CREATE TABLE {schema}.{bare} (id TEXT PRIMARY KEY)")
        try:
            source_bus = FakeNatsClient()
            source = EpochGenerationSource(source_bus)
            await source.start_log()
            owner = await _keeper(pool, schema, table, source)
            await owner.run_pass()
            await asyncio.sleep(_FENCE.total_seconds() * 2)
            await owner.run_pass()
            assert (await owner.ring()).fence_seq is not None
            besides = EpochGenerationSource(source_bus, database=beside)
            side = ChangeNoteKeeper(pool, notes_schema=beside, log=besides, fence=_FENCE, keeps_log=False)
            await side.install()
            await side.install_triggers([NotedTable(table=theirs, keys=("id",), columns=("owner",))])

            # the other database's write: its note is in its own ring, never the owner's
            await pool.execute(f"INSERT INTO {beside}.{theirs} VALUES ('theirs', 'o', '1')")
            await besides.advance_with_change(theirs, [ChangedRow(ids=["theirs"])])
            await pool.execute(f"INSERT INTO {table} VALUES ('noted', 'o', '1')")
            await source.advance_with_change(table, [ChangedRow(ids=["noted"])])
            await source.advance_with_change(bare, [ChangedRow(ids=["bare"])])
            await owner.run_pass()  # the scrub checks up to the head one pass ago

            with caplog.at_level(logging.ERROR):
                scrub = (await owner.run_pass()).scrub
            said = [r.__dict__["extra_data"]["missing"] for r in caplog.records if "no change note" in r.getMessage()]
            assert said == ["bare"], "the other database's epoch is no missing note; the owner's own bare table is"
            assert (scrub.checked, scrub.missing, scrub.others) == (2, 1, 1)
        finally:
            await pool.execute(f"DROP SCHEMA IF EXISTS {beside} CASCADE")


class TestTheKeepingRunsUnderItsOwnLease:
    async def test_one_replica_keeps_each_ring_and_two_rings_keep_at_once(
        self, ring: tuple[asyncpg.Pool, str, str], nats_container: str
    ) -> None:
        """Flips if two rings share one lease (one database's ring would never be kept) or two replicas of one ring
        both keep it."""
        pool, schema, table = ring
        async with _nats(nats_container) as nc:
            source = EpochGenerationSource(nc, database=schema)
            await source.start_log()
            keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=source, fence=_FENCE, keeps_log=False)
            first = ChangeNoteKeeping(keeper, nc, lease="rc3-ring-a", key_scope="svc-rc3")
            second = ChangeNoteKeeping(keeper, nc, lease="rc3-ring-a", key_scope="svc-rc3")
            other = ChangeNoteKeeping(keeper, nc, lease="rc3-ring-b", key_scope="svc-rc3")
            await first.start()
            keeping = asyncio.create_task(first.claim_and_keep("replica-1"))
            beside = asyncio.create_task(other.claim_and_keep("replica-1"))
            try:
                await asyncio.sleep(0.5)
                assert not keeping.done() and not beside.done(), "each ring is kept, under its own lease"
                await asyncio.wait_for(second.claim_and_keep("replica-2"), timeout=10)  # returns: another keeps it
                lease = KVLease(nc, key_scope="svc-rc3")
                assert await lease.holder("rc3-ring-a") == "replica-1"
            finally:
                keeping.cancel()
                beside.cancel()
                await asyncio.gather(keeping, beside, return_exceptions=True)


@asynccontextmanager
async def _two_databases(
    pool: asyncpg.Pool, schema: str, table: str, db_container: str, bus: Any
) -> AsyncIterator[
    tuple[asyncpg.Pool, str, EpochGenerationSource, EpochGenerationSource, ChangeNoteKeeper, ChangeNoteKeeper]
]:
    """the hub's database (``schema``, the log's owner) and identity-core's (a second schema, its own pool and ring
    beside the log), each noting a table of the one name ``table``; both rings fenced, both sources over ``bus``."""
    beside = f"{schema}_beside"
    await pool.execute(f"CREATE SCHEMA {beside}")
    try:
        await pool.execute(f"CREATE TABLE {beside}.{table} (id TEXT PRIMARY KEY, owner TEXT, n TEXT)")
        other = await asyncpg.create_pool(
            db_container, min_size=2, max_size=12, server_settings={"search_path": beside}
        )
        assert other is not None
        try:
            hub = EpochGenerationSource(bus, database=schema)
            identity = EpochGenerationSource(bus, database=beside)
            await hub.start_log()
            owner = await _fenced(pool, schema, table, hub)
            side = ChangeNoteKeeper(
                other, notes_schema=beside, log=identity, fence=_FENCE, scrub_every=timedelta(0), keeps_log=False
            )
            await side.install()
            assert await side.install_triggers([NotedTable(table=table, keys=("id",), columns=("owner",))]) == [
                f"{beside}.{table}"
            ]
            await side.run_pass()
            await other.execute(
                f"UPDATE {beside}.threetears_change_note_ring SET switched_at = now() - interval '1 hour'"
            )
            await side.run_pass()
            assert (await side.ring()).fence_seq is not None
            yield other, beside, hub, identity, owner, side
        finally:
            await other.close()
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {beside} CASCADE")


class TestATableNamedInTwoDatabases:
    """two databases on one change log each note a table of the same name (RC9, 2026-11-16): today the hub's and
    identity-core's ``coordination_revocations``, 3tears' coordination table migrated into each one's schema."""

    async def test_each_scrub_judges_only_its_own_databases_epochs_of_the_name(
        self, ring: tuple[asyncpg.Pool, str, str], db_container: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Flips if either ring's scrub judges the other database's epoch of the shared name: the owner reads every
        identity write of it as a missing note (the name is a table of its schema), and the ring beside every hub
        write (the name is a table it triggers), both on threetears_change_notes_missing_total -- or if either stops
        reporting its own write that left no note."""
        pool, schema, table = ring
        async with _two_databases(pool, schema, table, db_container, FakeNatsClient()) as databases:
            other, beside, hub, identity, owner, side = databases
            await pool.execute(f"INSERT INTO {schema}.{table} VALUES ('hubs', 'o', '1')")
            await hub.advance_with_change(table, [ChangedRow(ids=["hubs"])])
            await other.execute(f"INSERT INTO {beside}.{table} VALUES ('identitys', 'o', '1')")
            await identity.advance_with_change(table, [ChangedRow(ids=["identitys"])])
            # each database's epoch whose write left no note: appended with no L3 write behind it
            await hub.advance_with_change(table, [ChangedRow(ids=["hub-unnoted"])])
            await identity.advance_with_change(table, [ChangedRow(ids=["identity-unnoted"])])
            await owner.run_pass()  # each scrub checks up to the head one pass ago
            await side.run_pass()

            found: dict[str, tuple[list[str], ScrubPass]] = {}
            for name, keeper in (("owner", owner), ("beside", side)):
                caplog.clear()
                with caplog.at_level(logging.ERROR):
                    scrub = (await keeper.run_pass()).scrub
                said = [
                    r.__dict__["extra_data"]["missing"] for r in caplog.records if "no change note" in r.getMessage()
                ]
                found[name] = (said, scrub)
            assert found["owner"][0] == ["hub-unnoted"], "the owner says its own lost note, never identity's epochs"
            assert found["beside"][0] == ["identity-unnoted"], "the ring beside says its own, never the hub's"
            for name, (_, scrub) in found.items():
                assert (scrub.checked, scrub.missing, scrub.others) == (2, 1, 2), name

    async def test_a_follower_never_applies_the_other_databases_record_of_the_name(
        self, ring: tuple[asyncpg.Pool, str, str], db_container: str
    ) -> None:
        """Flips if identity-core's follower of its own table applies the hub's records of the hub's table of that
        name: a hub truncate drops its whole cache, a hub key evicts its row of the same key, and a hub keys-unknown
        record's keys are paged from identity's own ring -- behind (``not_covered``) when its bracket is older than
        that ring's coverage, until it restarts. Or if the hub's follower pages an identity bracket from the hub's
        ring, evicting whatever the hub wrote in it."""
        pool, schema, table = ring
        bus = FakeNatsClient()
        before_rings = datetime.now(tz=UTC) - timedelta(hours=2)
        async with _two_databases(pool, schema, table, db_container, bus) as databases:
            other, beside, hub, identity, _owner, _side = databases
            # identity's ring has switched since its install: a bracket from before it is one it no longer covers
            # (a ring switches on the log's bytes since its last switch, so one record of identity's goes first)
            await identity.advance_with_change(f"{table}_elsewhere", [ChangedRow(ids=["x"])])
            switching = ChangeNoteKeeper(
                other, notes_schema=beside, log=identity, fence=_FENCE, switch_bytes=1, keeps_log=False
            )
            assert (await switching.run_pass()).switched_to is not None
            followers: dict[str, tuple[CollectionRegistry, EpochGenerationReader, list[str], list[bool]]] = {}
            for name, (notes, notes_schema) in {"hub": (pool, schema), "identity": (other, beside)}.items():
                registry = CollectionRegistry()
                registry.configure(l3_pool=notes)
                registry.set_noted_keys_reader(PoolNotedKeysReader(notes, notes_schema=notes_schema, every_tenant=True))
                registry.follow_generation(table)
                heard: list[str] = []
                dropped: list[bool] = []
                registry.register_derived_cache(
                    table,
                    on_row=lambda message, heard=heard: heard.append(str(message.ids[0])),  # type: ignore[misc]
                    on_table_dropped=lambda dropped=dropped: dropped.append(True),  # type: ignore[misc]
                )
                reader = EpochGenerationReader(bus, database=notes_schema)
                await registry.catch_up_generation(table, await reader.read(table), changes=reader)
                followers[name] = (registry, reader, heard, dropped)

            opened = datetime.now(tz=UTC) - timedelta(seconds=1)
            await pool.execute(f"INSERT INTO {schema}.{table} VALUES ('same-key', 'o', '1')")
            await hub.advance_with_change(table, [ChangedRow(ids=["same-key"])])
            await hub.advance_with_change(table, None)
            await hub.advance_with_change(
                table, None, keys_unknown=True, noted_between=(before_rings, before_rings + timedelta(seconds=1))
            )
            await other.execute(f"INSERT INTO {beside}.{table} VALUES ('identitys', 'o', '1')")
            closed = datetime.now(tz=UTC) + timedelta(seconds=1)
            await identity.advance_with_change(table, None, keys_unknown=True, noted_between=(opened, closed))

            for name, (registry, reader, _heard, _dropped) in followers.items():
                await registry.catch_up_generation(table, await reader.read(table), changes=reader)
                assert registry.behind_reason(table) is None, f"{name} is behind on another database's record"
            _, _, hub_heard, hub_dropped = followers["hub"]
            _, _, identity_heard, identity_dropped = followers["identity"]
            assert identity_heard == ["identitys"], "identity applies its own bracket's key, no hub key"
            assert not identity_dropped, "a hub truncate never drops identity's table"
            assert hub_heard == ["same-key"] and hub_dropped == [True], (
                "the hub applies its own key and truncate once, never identity's bracket paged from the hub's ring"
            )
