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
from datetime import timedelta
from typing import Any

import asyncpg
import pytest
from prometheus_client import REGISTRY
from sqlalchemy import MetaData

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import WRITE_GENERATION, BaseCollection, CollectionRegistry
from threetears.core.collections.change_notes import RESTORE_BUDGET_RATIO, ChangeNoteKeeper, ScrubPass
from threetears.core.collections.epoch_change import ChangedRow, read_epoch_change_body
from threetears.core.collections.generation import NotedTable, tenancy_of_table
from threetears.core.collections.l2_first import run_l2_first_worker
from threetears.core.collections.write_marks import WriteMarkReconciler, install_write_mark_triggers, write_marks_ddl
from threetears.core.collections.schema_backed import STRING_TYPE, TableSchema, collection_for_schema
from threetears.core.collections.schema_backed import Column as SchemaColumn
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.change_notes import (
    NOTE_RING_SIZE,
    NOTE_TRUNCATE_TRIGGER,
    note_function_name,
    note_table,
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


class TestAColumnWhoseTextIsNotExactGetsNoTrigger:
    async def test_a_boolean_key_or_a_timestamp_column_is_refused_with_an_error(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        pool, schema, table = ring
        odd = f"{table}_odd"
        await pool.execute(f"CREATE TABLE {schema}.{odd} (id BOOLEAN PRIMARY KEY, at TIMESTAMPTZ)")
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=EpochGenerationSource(FakeNatsClient()), fence=_FENCE)
        await keeper.install()
        with caplog.at_level(logging.ERROR):
            given = await keeper.install_triggers(
                [NotedTable(table=odd, keys=("id",), columns=("at",)), NotedTable(table=table, keys=("id",))]
            )
        assert given == [f"{schema}.{table}"], "the exact table is given its trigger, the other is refused"
        said = [r.__dict__["extra_data"] for r in caplog.records if "no change-note trigger" in r.getMessage()]
        assert said == [{"table": f"{schema}.{odd}", "columns": ["id (boolean)", "at (timestamp with time zone)"]}]


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


class TestTheReconcilerNamesTheRowsTheNotesName:
    async def test_a_missed_write_s_epoch_names_its_rows_and_a_truncate_s_the_whole_table(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        pool, schema, table = ring
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        await _keeper(pool, schema, table, source)
        for statement in write_marks_ddl(schema):
            await pool.execute(statement)
        await install_write_mark_triggers(pool, marks_schema=schema, tables=[table])
        reconciler = WriteMarkReconciler(pool, generation_source=source, marks_schema=schema, tables=lambda: [table])
        reader = EpochGenerationReader(bus)

        async def last() -> Any:
            token = await reader.read(table)
            assert token is not None
            return await reader.change(table, token)

        # raw SQL: a mark and a note, and no advance
        await pool.execute(f"INSERT INTO {table} VALUES ('raw-1', 'ada', '1'), ('raw-2', 'bea', '1')")
        assert (await reconciler.run_pass()).advanced == (table,)
        record = await last()
        assert not record.whole_table
        assert sorted((row.ids[0], (row.columns or {}).get("owner")) for row in record.rows) == [
            ("raw-1", "ada"),
            ("raw-2", "bea"),
        ], "the rows the notes name, with their invalidation columns: no follower drops the table"

        # a truncate leaves no mark of its own, but a whole-table note: the next marked write's notes carry it
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(f"TRUNCATE {table}")
            await conn.execute(f"INSERT INTO {table} VALUES ('reloaded', 'o', '1')")
        await reconciler.run_pass()
        assert (await last()).whole_table, "a truncate's note names the whole table, and so does its epoch"

        # a mark older than the coverage (the ring has switched since): the notes cannot vouch for it, so keys
        # unknown (every follower behind on the table, none told to drop it), said and counted
        await pool.execute(
            f"UPDATE {schema}.threetears_change_note_ring SET switch_no = switch_no + 1, previous_switched_at = now()"
        )
        await pool.execute(f"INSERT INTO {table} VALUES ('old', 'cy', '1')")
        await pool.execute(f"UPDATE {schema}.collection_write_marks SET date_created = now() - interval '1 day'")
        with caplog.at_level(logging.WARNING):
            await reconciler.run_pass()
        unnamed = await last()
        assert unnamed.keys_unknown and not unnamed.whole_table
        said = [r.__dict__["extra_data"] for r in caplog.records if "advanced as keys unknown" in r.getMessage()]
        assert said == [{"table": table, "reason": "not_covered"}]


class TestTheReconcilerKeysEachSchemasChange:
    """under TENANT_BY_SCHEMA every agent schema's notes of a table share the ring, and a mark names no schema."""

    async def test_one_schemas_truncate_never_hides_anothers_missed_rows(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        other = f"{schema}_b"

        class _PerSchema(BaseCollection[Any]):
            """the table, declared per schema, so the reconciler and a pod's reader key it by schema."""

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
            await keeper.install_triggers([NotedTable(table=table, keys=("id",), tenancy=TENANT_BY_SCHEMA)])
            for statement in write_marks_ddl(schema):
                await pool.execute(statement)
            await install_write_mark_triggers(pool, marks_schema=schema, tables=[table])
            reconciler = WriteMarkReconciler(
                pool, generation_source=source, marks_schema=schema, tables=lambda: [table]
            )
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
            for statement in write_marks_ddl(schema):
                await pool.execute(statement)
            await install_write_mark_triggers(pool, marks_schema=schema, tables=[table])
            reconciler = WriteMarkReconciler(
                pool, generation_source=source, marks_schema=schema, tables=lambda: [table]
            )
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
