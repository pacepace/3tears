"""Integration test: no save reads the ring row, a cached slot stays safe, and the idle table empties by deletes.

Against a real database (the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to run it on
YugabyteDB). What these prove:

- with the connection's cached slot present, the note trigger reads no ring row: a role with no SELECT on
  the ring writes; without the setting, or with an expired one, the same write needs the ring row;
- a process's writes through its pool read the ring row once, not once per save;
- a slot cached before a switch, used within its age, lands in a table the coverage still holds, and the
  ring cannot switch again (and so empty it) while that copy is usable;
- a copy older than its age is read again before it is applied;
- the keeper empties the idle table with batched deletes while eight writers write to the current one, with
  no error, and a switch never lands on a table whose emptying has not finished;
- a migration run that creates a noted table gives it its note triggers in that run;
- an agent schema's noted table gets its note triggers from that schema's migration run at every start, a table
  created before the run gave triggers included: the keeper's install gives them only in its own schema.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.collections import WRITE_GENERATION, CollectionRegistry
from threetears.core.collections.change_notes import ChangeNoteKeeper
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.generation import NotedTable
from threetears.core.collections.schema_backed import STRING_TYPE, TableSchema, collection_for_schema
from threetears.core.collections.schema_backed import Column as SchemaColumn
from threetears.core.data.migrations import MigrationRunner, MigrationScope, PackageMigrations
from threetears.core.data.store import DataStore
from threetears.core.ddl.change_notes import (
    NOTE_SLOT_SETTING,
    NOTE_TRIGGER,
    NOTE_TRUNCATE_TRIGGER,
    RING_TABLE,
    note_table,
)
from threetears.core.testing.kv import FakeNatsClient
from threetears.core.utils.change_note_slot import ChangeNoteSlot, change_note_slot_connection_class
from threetears.epoch import EpochGenerationSource

pytestmark = pytest.mark.integration

#: the shared truncate trigger an install before a whole-table note named its tenant gave each table
_LEGACY_TRUNCATE_TRIGGER = "threetears_change_note_truncate"

#: long enough that a copy read in a test stays valid (half of it) while the test acts, short enough to wait out
_FENCE = timedelta(seconds=4)


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[tuple[asyncpg.Pool, str, str]]:
    """a schema holding the ring (fence :data:`_FENCE`) and one noted table with its trigger."""
    suffix = uuid.uuid4().hex[:8]
    schema, table = f"slot_{suffix}", f"slot_widgets_{suffix}"
    pool = await asyncpg.create_pool(db_container, min_size=2, max_size=12, server_settings={"search_path": schema})
    assert pool is not None
    try:
        await pool.execute(f"CREATE SCHEMA {schema}")
        await pool.execute(f"CREATE TABLE {schema}.{table} (id TEXT PRIMARY KEY, owner TEXT)")
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=EpochGenerationSource(FakeNatsClient()), fence=_FENCE)
        await keeper.install()
        await keeper.install_triggers([NotedTable(table=table, keys=("id",))])
        yield pool, schema, table
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def _slot_pool(db_container: str, schema: str, slot: ChangeNoteSlot, size: int = 2) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(
        db_container,
        min_size=1,
        max_size=size,
        server_settings={"search_path": schema},
        connection_class=change_note_slot_connection_class(slot),
    )
    assert pool is not None
    return pool


async def _notes(pool: asyncpg.Pool, schema: str, index: int) -> list[Any]:
    return list(await pool.fetch(f"SELECT keys, ring_read FROM {schema}.{note_table(index)}"))


async def _db_epoch(pool: asyncpg.Pool) -> float:
    return float(await pool.fetchval("SELECT extract(epoch FROM now())::float8"))


class TestNoSaveReadsTheRingRow:
    async def test_with_the_setting_a_role_that_cannot_read_the_ring_still_writes(
        self, ring: tuple[asyncpg.Pool, str, str]
    ) -> None:
        pool, schema, table = ring
        role = f"slot_writer_{uuid.uuid4().hex[:8]}"
        await pool.execute(f"CREATE ROLE {role}")
        try:
            await pool.execute(f"GRANT USAGE ON SCHEMA {schema} TO {role}")
            await pool.execute(f"GRANT INSERT ON {schema}.{table} TO {role}")
            for index in range(3):
                await pool.execute(f"GRANT INSERT ON {schema}.{note_table(index)} TO {role}")
            # no SELECT on the ring row: any read of it fails the write
            future = await _db_epoch(pool) + 60
            async with pool.acquire() as conn:
                await conn.execute(f"SET ROLE {role}")
                try:
                    await conn.execute("SELECT set_config($1, $2, false)", NOTE_SLOT_SETTING, f"0:{future:.6f}")
                    await conn.execute(f"INSERT INTO {table} VALUES ('cached', 'o')")

                    await conn.execute(f"RESET {NOTE_SLOT_SETTING}")
                    with pytest.raises(asyncpg.InsufficientPrivilegeError):
                        await conn.execute(f"INSERT INTO {table} VALUES ('absent', 'o')")

                    past = future - 3600
                    await conn.execute("SELECT set_config($1, $2, false)", NOTE_SLOT_SETTING, f"0:{past:.6f}")
                    with pytest.raises(asyncpg.InsufficientPrivilegeError):
                        await conn.execute(f"INSERT INTO {table} VALUES ('expired', 'o')")
                finally:
                    await conn.execute("RESET ROLE")
            notes = await _notes(pool, schema, 0)
            assert [(list(note["keys"]), note["ring_read"]) for note in notes] == [(["cached"], None)]
        finally:
            await pool.execute(f"DROP OWNED BY {role}")
            await pool.execute(f"DROP ROLE {role}")

    async def test_a_process_reads_the_ring_once_for_many_saves_and_flags_none(
        self, ring: tuple[asyncpg.Pool, str, str], db_container: str
    ) -> None:
        pool, schema, table = ring
        slot = ChangeNoteSlot(notes_schema=schema)
        writer = await _slot_pool(db_container, schema, slot)
        try:
            for n in range(20):
                async with writer.acquire() as conn, conn.transaction():
                    await conn.execute(f"INSERT INTO {table} VALUES ($1, 'o')", f"k{n}")
            for n in range(20, 30):
                await writer.execute(f"INSERT INTO {table} VALUES ($1, 'o')", f"k{n}")
        finally:
            await writer.close()
        assert slot.reads == 1, "the ring row is read once per refresh, never per save"
        notes = await _notes(pool, schema, 0)
        assert len(notes) == 30
        assert {note["ring_read"] for note in notes} == {None}, "every note took the cached slot"

    async def test_a_write_with_no_setting_reads_the_ring_and_the_keeper_counts_it_at_the_switch(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        pool, schema, table = ring
        await pool.execute(f"INSERT INTO {table} VALUES ('raw', 'o')")
        notes = await _notes(pool, schema, 0)
        assert [note["ring_read"] for note in notes] == [True]

        source = EpochGenerationSource(FakeNatsClient())
        await source.start_log()
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=source, fence=_FENCE, switch_bytes=1)
        await keeper.run_pass()
        await pool.execute(f"UPDATE {schema}.{RING_TABLE} SET switched_at = now() - interval '1 hour'")
        await source.advance_with_change(table, [ChangedRow(ids=["raw"])])
        with caplog.at_level(logging.INFO):
            switched = [(await keeper.run_pass()).switched_to for _ in range(2)]
        assert 1 in switched
        said = [r.__dict__["extra_data"] for r in caplog.records if "without a cached slot" in r.getMessage()]
        assert said == [{"note_table": note_table(0), "tables": {table: 1}}]


class TestACachedSlot:
    async def test_a_copy_read_before_a_switch_lands_in_a_covered_table_and_blocks_the_next_switch(
        self, ring: tuple[asyncpg.Pool, str, str], db_container: str
    ) -> None:
        pool, schema, table = ring
        source = EpochGenerationSource(FakeNatsClient())
        await source.start_log()
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=source, fence=_FENCE, switch_bytes=1)
        await keeper.run_pass()
        # the wait since the ring's start has passed; the fence position is recorded
        await pool.execute(f"UPDATE {schema}.{RING_TABLE} SET switched_at = now() - interval '1 hour'")
        assert (await keeper.run_pass()).fenced

        slot = ChangeNoteSlot(notes_schema=schema)
        writer = await _slot_pool(db_container, schema, slot)
        try:
            await writer.execute(f"INSERT INTO {table} VALUES ('before', 'o')")  # the copy: slot 0
            assert slot.usable() is not None and slot.usable().startswith("0:")  # type: ignore[union-attr]

            await source.advance_with_change(table, [ChangedRow(ids=["before"])])
            assert (await keeper.run_pass()).switched_to == 1

            # the process's copy still names 0 and is still usable: its write goes there
            await writer.execute(f"INSERT INTO {table} VALUES ('after', 'o')")
            assert slot.reads == 1
        finally:
            await writer.close()

        state = await keeper.ring()
        assert (state.current, state.previous) == (1, 0), "table 0 is still in the coverage"
        assert sorted(list(note["keys"])[0] for note in await _notes(pool, schema, 0)) == ["after", "before"]

        # while that copy may be used, the ring cannot switch again, so table 0 cannot become the idle one
        await source.advance_with_change(table, [ChangedRow(ids=["after"])])
        again = await keeper.run_pass()
        assert again.switched_to is None and not again.fenced
        assert slot.max_age < state.wait_seconds, "a copy's age is inside the wait"

    async def test_a_copy_older_than_its_age_is_read_again_before_it_is_applied(
        self, ring: tuple[asyncpg.Pool, str, str], db_container: str
    ) -> None:
        pool, schema, table = ring
        now = [1000.0]
        slot = ChangeNoteSlot(notes_schema=schema, clock=lambda: now[0])
        writer = await _slot_pool(db_container, schema, slot, size=1)
        try:
            await writer.execute(f"INSERT INTO {table} VALUES ('a', 'o')")
            assert slot.reads == 1
            assert slot.max_age == pytest.approx(_FENCE.total_seconds() / 2), "the age derives from the told fence"
            assert slot.refresh_after == pytest.approx(slot.max_age / 2)

            # the ring moved; this process has not looked
            await pool.execute(f"UPDATE {schema}.{RING_TABLE} SET current_ring = 1")
            now[0] += slot.refresh_after / 2
            await writer.execute(f"INSERT INTO {table} VALUES ('b', 'o')")
            assert slot.reads == 1, "a young copy is used as is"

            now[0] += slot.max_age  # older than its age: never applied without a read
            assert slot.usable() is None
            await writer.execute(f"INSERT INTO {table} VALUES ('c', 'o')")
            assert slot.reads == 2
            assert (slot.usable() or "").startswith("1:")
        finally:
            await writer.close()
        assert sorted(list(note["keys"])[0] for note in await _notes(pool, schema, 0)) == ["a", "b"]
        assert [list(note["keys"])[0] for note in await _notes(pool, schema, 1)] == ["c"]


async def _fill(pool: asyncpg.Pool, schema: str, index: int, count: int) -> None:
    await pool.execute(
        f"INSERT INTO {schema}.{note_table(index)} (table_name, keys) "
        "SELECT 'filler', ARRAY[g::text] FROM generate_series(1, $1) g",
        count,
    )


class TestTheIdleTableEmptiesByDeletes:
    async def test_emptying_under_eight_writers_to_the_current_table_completes_with_no_error(
        self, ring: tuple[asyncpg.Pool, str, str], db_container: str
    ) -> None:
        pool, schema, table = ring
        await _fill(pool, schema, 1, 20_000)  # the idle table while 0 is current
        await pool.execute(f"UPDATE {schema}.{RING_TABLE} SET emptied = FALSE")
        source = EpochGenerationSource(FakeNatsClient())
        await source.start_log()
        keeper = ChangeNoteKeeper(
            pool,
            notes_schema=schema,
            log=source,
            fence=_FENCE,
            empty_batch=500,
            empty_pause=timedelta(milliseconds=5),
            empty_budget=timedelta(milliseconds=200),
        )
        slot = ChangeNoteSlot(notes_schema=schema)
        writers_pool = await _slot_pool(db_container, schema, slot, size=8)
        stop = asyncio.Event()
        written: list[str] = []
        errors: list[BaseException] = []

        async def writer(index: int) -> None:
            count = 0
            while not stop.is_set():
                key = f"w{index}-{count}"
                try:
                    async with writers_pool.acquire() as conn, conn.transaction():
                        await conn.execute(f"INSERT INTO {table} VALUES ($1, 'o')", key)
                    written.append(key)
                except Exception as exc:  # noqa: BLE001 - every error is the finding
                    errors.append(exc)
                count += 1

        tasks = [asyncio.create_task(writer(index)) for index in range(8)]
        passes = 0
        try:
            await asyncio.sleep(0.2)
            began = time.monotonic()
            while True:
                done = await keeper.run_pass()
                passes += 1
                if done.emptied is not None:
                    assert done.emptied == 1
                    break
                assert time.monotonic() - began < 120, "the emptying finishes"
            await asyncio.sleep(0.2)
        finally:
            stop.set()
            await asyncio.gather(*tasks)
            await writers_pool.close()

        assert errors == []
        assert passes > 1, "the emptying spanned passes, each bounded"
        assert await pool.fetchval(f"SELECT count(*) FROM {schema}.{note_table(1)}") == 0
        assert (await keeper.ring()).emptied
        noted = {list(note["keys"])[0] for note in await _notes(pool, schema, 0)}
        assert written and set(written) <= noted, "every writer's note is in the current table"

    async def test_a_switch_never_lands_on_a_table_still_being_emptied(
        self, ring: tuple[asyncpg.Pool, str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        pool, schema, table = ring
        source = EpochGenerationSource(FakeNatsClient())
        await source.start_log()
        keeper = ChangeNoteKeeper(
            pool,
            notes_schema=schema,
            log=source,
            fence=_FENCE,
            switch_bytes=1,
            empty_batch=10,
            empty_pause=timedelta(0),
            empty_budget=timedelta(0),
        )
        await keeper.run_pass()
        await pool.execute(f"UPDATE {schema}.{RING_TABLE} SET switched_at = now() - interval '1 hour'")
        assert (await keeper.run_pass()).fenced
        # the idle table holds notes still being emptied, and a switch is due
        await _fill(pool, schema, 1, 45)
        await pool.execute(f"UPDATE {schema}.{RING_TABLE} SET emptied = FALSE")
        await source.advance_with_change(table, [ChangedRow(ids=["x"])])

        waits: list[bool] = []
        with caplog.at_level(logging.WARNING):
            for _ in range(10):
                done = await keeper.run_pass()
                if done.switched_to is not None:
                    left = await pool.fetchval(f"SELECT count(*) FROM {schema}.{note_table(done.switched_to)}")
                    assert left == 0, "switched into an emptied table"
                    break
                waits.append(done.switch_waited)
                assert (await keeper.ring()).current == 0
        assert done.switched_to == 1
        assert waits == [True] * 4, "each batch pass waited, the last one emptied and switched"
        said = [r for r in caplog.records if "switch is due but waits" in r.getMessage()]
        assert len(said) == 1, "logged once per due switch"


class TestTheMigrationRunGivesANewNotedTableItsTriggers:
    async def test_a_table_created_by_a_run_is_noted_from_that_run(
        self, ring: tuple[asyncpg.Pool, str, str], db_container: str
    ) -> None:
        pool, schema, _ = ring
        table = f"hooked_{uuid.uuid4().hex[:8]}"
        declared = TableSchema(
            name=table,
            primary_key="id",
            columns=[SchemaColumn("id", STRING_TYPE), SchemaColumn("owner", STRING_TYPE, nullable=True)],
        )
        # a class this process defines: noted_tables() names its table from here on
        type(f"Hooked_{table}", (collection_for_schema(declared),), {"write_generation": WRITE_GENERATION})

        async def create(store: Any) -> None:
            await store.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, owner TEXT)")

        package = PackageMigrations(name=f"hook_{table}", scope=MigrationScope.AGENT)
        package.version(1)(create)
        package.mark_write_generation(table)  # the package declares the table it creates switched on
        runner = MigrationRunner()
        runner.register(package)
        run_pool = await asyncpg.create_pool(
            db_container, min_size=1, max_size=2, server_settings={"search_path": schema}
        )
        assert run_pool is not None
        try:
            registry = CollectionRegistry()
            registry.configure(l3_pool=run_pool)
            assert await DataStore(uuid.uuid4(), registry).run_migrations(runner) == 1
        finally:
            await run_pool.close()

        triggers = {
            str(row["tgname"])
            for row in await pool.fetch(
                "SELECT g.tgname FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 AND c.relname = $2",
                schema,
                table,
            )
        }
        assert {NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER} <= triggers
        assert _LEGACY_TRUNCATE_TRIGGER not in triggers
        await pool.execute(f"INSERT INTO {table} VALUES ('first', 'o')")
        assert [list(note["keys"]) for note in await _notes(pool, schema, 0)] == [["first"]], (
            "its first write left a note"
        )

    async def test_an_agent_schemas_table_from_before_the_hook_gets_its_triggers_at_every_run_start(
        self, db_container: str
    ) -> None:
        """Flips if an agent schema's noted table depends on the keeper for its triggers (the keeper gives them
        only in its own schema), or a run with nothing pending leaves one without them."""
        suffix = uuid.uuid4().hex[:8]
        database, platform, agent = f"notes_db_{suffix}", "platform", f"agent_{suffix}"
        table = f"agent_widgets_{suffix}"
        admin = await asyncpg.connect(db_container)
        try:
            # a database of its own: the run finds the ring as a deployment has it, one schema holding it
            await admin.execute(f"CREATE DATABASE {database}")
            url = db_container.rsplit("/", 1)[0] + f"/{database}"
            pool = await asyncpg.create_pool(url, min_size=1, max_size=4)
            assert pool is not None
            try:
                await pool.execute(f"CREATE SCHEMA {platform}")
                await pool.execute(f"CREATE SCHEMA {agent}")
                keeper = ChangeNoteKeeper(
                    pool, notes_schema=platform, log=EpochGenerationSource(FakeNatsClient()), fence=_FENCE
                )
                await keeper.install()
                declared = TableSchema(
                    name=table,
                    primary_key="id",
                    columns=[SchemaColumn("id", STRING_TYPE), SchemaColumn("owner", STRING_TYPE, nullable=True)],
                )
                type(f"Agent_{table}", (collection_for_schema(declared),), {"write_generation": WRITE_GENERATION})

                async def create(store: Any) -> None:
                    await store.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, owner TEXT)")

                async def run(marked: bool) -> int:
                    package = PackageMigrations(name=f"agent_{table}", scope=MigrationScope.AGENT)
                    package.version(1)(create)
                    if marked:
                        package.mark_write_generation(table)
                    runner = MigrationRunner()
                    runner.register(package)
                    run_pool = await asyncpg.create_pool(
                        url, min_size=1, max_size=2, server_settings={"search_path": agent}
                    )
                    assert run_pool is not None
                    try:
                        registry = CollectionRegistry()
                        registry.configure(l3_pool=run_pool)
                        return int(await DataStore(uuid.uuid4(), registry).run_migrations(runner))
                    finally:
                        await run_pool.close()

                async def triggers() -> set[str]:
                    return {
                        str(row["tgname"])
                        for row in await pool.fetch(
                            "SELECT g.tgname FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid "
                            "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 AND c.relname = $2",
                            agent,
                            table,
                        )
                    }

                # the table, created by a run before its package declared it switched on: no triggers
                assert await run(marked=False) == 1
                assert await triggers() == set()
                assert await keeper.install_triggers([NotedTable(table=table, keys=("id",))]) == [], (
                    "the keeper gives none in an agent schema"
                )
                assert await triggers() == set()
                # the next start, nothing pending: its run gives them
                assert await run(marked=True) == 0
                assert {NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER} <= await triggers()
                # lost again (a restore of the schema, say): the start after gives them again
                for name in (NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER):
                    await pool.execute(f"DROP TRIGGER {name} ON {agent}.{table}")
                assert await run(marked=True) == 0
                assert {NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER} <= await triggers()
                await pool.execute(f"INSERT INTO {agent}.{table} VALUES ('first', 'o')")
                current = (await keeper.ring()).current
                assert [list(note["keys"]) for note in await _notes(pool, platform, current)] == [["first"]]
            finally:
                await pool.close()
        finally:
            await admin.execute(f"DROP DATABASE IF EXISTS {database}")
            await admin.close()
