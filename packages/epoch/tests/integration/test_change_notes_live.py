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
from threetears.core.collections.change_notes import RESTORE_BUDGET_RATIO, ChangeNoteKeeper, ScrubPass
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.generation import NotedTable
from threetears.core.collections.schema_backed import STRING_TYPE, TableSchema, collection_for_schema
from threetears.core.collections.schema_backed import Column as SchemaColumn
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.change_notes import NOTE_RING_SIZE, note_table
from threetears.core.exceptions import GenerationUnavailableError
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
