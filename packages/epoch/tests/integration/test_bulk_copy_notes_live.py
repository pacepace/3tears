"""Integration: a bulk copy is noted whole, not a note a row, and costs about what the database's COPY costs.

Every row a noted table is written leaves a change note in its own transaction, a second write. A bulk
copy of tens of thousands of rows (a reports pod reloading a state) wrote that many notes, of which the
write-mark reconciler can name at most :data:`WHOLE_COPY_ROWS`: past it the notes cost a write each and
name nothing. A copy of that many rows or more now marks its transaction whole (one note for the table),
as a file load always has. In a whole-marked transaction the row trigger does not fire at all (its ``WHEN``
is the setting's complement), and one statement trigger writes the note; outside one, every row is noted
as before. A direct connection copies the rows in one ``COPY`` (:func:`copy_bounds`), not a thousand at a time.

Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL
to run them there. The budget (twice a plain ``COPY`` of the same rows, calibrated on the same database) is
asserted on YugabyteDB, the platform's database; Postgres checks the notes only.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from threetears.core.backends.protocol import copy_bounds
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.change_notes import ChangeNoteKeeper
from threetears.core.collections.generation import NotedTable
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.ddl.change_notes import (
    NOTE_RING_SIZE,
    NOTE_TRIGGER,
    NOTE_WHOLE_SETTING,
    NOTE_WHOLE_TRIGGER,
    WHOLE_COPY_ROWS,
    note_table,
)
from threetears.core.testing.calibrate import budget_problem, plain_copy_seconds
from threetears.core.testing.kv import FakeNatsClient
from threetears.core.testing.plans import on_yugabyte
from threetears.epoch import EpochGenerationSource

pytestmark = pytest.mark.integration

_TABLE = "copied"

_SCHEMA = TableSchema(
    name=_TABLE,
    primary_key=("id",),
    columns=[
        Column("id", STRING_TYPE),
        Column("state", STRING_TYPE),
        Column("votes", BIGINT_TYPE, nullable=True),
        Column("pad", STRING_TYPE, nullable=True),
    ],
)

#: a large state's reload
_BULK_ROWS = 50_000

#: runs of the copy timed; the best is kept
_RUNS = 3


@dataclass
class _Noted:
    pool: asyncpg.Pool
    schema: str


@pytest.fixture
async def noted(db_container: str) -> AsyncIterator[_Noted]:
    """a schema holding the ring and one noted table, its trigger given."""
    schema = f"bulk_{uuid.uuid4().hex[:8]}"
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    try:
        await pool.execute(f"CREATE SCHEMA {schema}")
        await pool.execute(
            f"CREATE TABLE {schema}.{_TABLE} (id TEXT PRIMARY KEY, state TEXT NOT NULL, votes BIGINT, pad TEXT)"
        )
        keeper = ChangeNoteKeeper(pool, notes_schema=schema, log=EpochGenerationSource(FakeNatsClient()))
        await keeper.install()
        await keeper.install_triggers([NotedTable(table=_TABLE, keys=("id",))])
        yield _Noted(pool=pool, schema=schema)
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


def _rows(count: int) -> list[dict[str, Any]]:
    return [{"id": f"r{n:07d}", "state": "TX", "votes": n, "pad": "x" * 40} for n in range(count)]


async def _copy(noted: _Noted, rows: list[dict[str, Any]]) -> float:
    """copy ``rows`` in one transaction on a direct connection, committed, cut as a collection's copy cuts them; its seconds."""
    backend = SqlL3Backend(noted.pool)
    backend.register_schema(_TABLE, _SCHEMA)
    async with noted.pool.acquire() as conn:
        max_rows, _ = copy_bounds(conn, len(rows))
        assert max_rows is not None, "a direct connection streams its copy whole"
        began = time.perf_counter()
        async with conn.transaction():
            await backend.copy_many(_TABLE, rows, max_rows=max_rows, max_bytes=8 * 1024 * 1024, conn=conn)
        return time.perf_counter() - began


async def _notes(noted: _Noted) -> list[Any]:
    found: list[Any] = []
    for index in range(NOTE_RING_SIZE):
        found += await noted.pool.fetch(
            f"SELECT keys FROM {noted.schema}.{note_table(index)} WHERE table_name = $1", _TABLE
        )
    return found


class TestAWholeMarkedWriteIsOneNote:
    async def test_a_whole_marked_transaction_writes_one_note_for_every_kind_of_write(self, noted: _Noted) -> None:
        """Flips if a whole-marked transaction's inserts, updates or deletes are noted a row each, or not at all."""
        await noted.pool.execute(f"INSERT INTO {_TABLE} SELECT 'k' || n, 'TX', n, NULL FROM generate_series(1, 20) n")
        for index in range(NOTE_RING_SIZE):
            await noted.pool.execute(f"DELETE FROM {noted.schema}.{note_table(index)}")
        async with noted.pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config($1, '1', true)", NOTE_WHOLE_SETTING)
            await conn.execute(f"INSERT INTO {_TABLE} SELECT 'n' || n, 'TX', n, NULL FROM generate_series(1, 20) n")
            await conn.execute(f"UPDATE {_TABLE} SET votes = votes + 1 WHERE id LIKE 'k%'")
            await conn.execute(f"DELETE FROM {_TABLE} WHERE id LIKE 'n1%'")

        notes = await _notes(noted)
        assert [note["keys"] for note in notes] == [None]

    async def test_an_unmarked_transaction_still_notes_each_row(self, noted: _Noted) -> None:
        """Flips if the row trigger's ``WHEN`` skips rows outside a whole-marked transaction."""
        async with noted.pool.acquire() as conn, conn.transaction():
            await conn.execute(f"INSERT INTO {_TABLE} SELECT 'k' || n, 'TX', n, NULL FROM generate_series(1, 3) n")
            await conn.execute(f"UPDATE {_TABLE} SET votes = 0 WHERE id = 'k1'")
            await conn.execute(f"DELETE FROM {_TABLE} WHERE id = 'k2'")

        notes = await _notes(noted)
        assert len(notes) == 5, "three inserts, an update and a delete, a note each"
        assert all(note["keys"] is not None for note in notes)

    async def test_the_row_trigger_is_skipped_in_the_catalogue_not_in_its_function(self, noted: _Noted) -> None:
        """Flips if the row trigger fires (a function call a row) in a whole-marked transaction."""
        rows = await noted.pool.fetch(
            "SELECT g.tgname, (g.tgtype & 1) = 1 AS per_row, pg_get_triggerdef(g.oid) AS definition "
            "FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = $1 AND c.relname = $2 AND NOT g.tgisinternal",
            noted.schema,
            _TABLE,
        )
        found = {str(row["tgname"]): (bool(row["per_row"]), str(row["definition"])) for row in rows}
        per_row, definition = found[NOTE_TRIGGER]
        assert per_row and NOTE_WHOLE_SETTING in definition and "IS DISTINCT FROM" in definition
        per_row, definition = found[NOTE_WHOLE_TRIGGER]
        assert not per_row and NOTE_WHOLE_SETTING in definition


class TestABulkCopyIsNotedWhole:
    async def test_a_copy_past_the_reconcilers_cap_leaves_one_whole_note(self, noted: _Noted) -> None:
        """Flips if a bulk copy writes a note a row past what the notes can name."""
        await _copy(noted, _rows(WHOLE_COPY_ROWS))

        notes = await _notes(noted)
        assert [note["keys"] for note in notes] == [None]
        assert await noted.pool.fetchval(f"SELECT count(*) FROM {_TABLE}") == WHOLE_COPY_ROWS

    async def test_a_small_copy_still_names_its_rows(self, noted: _Noted) -> None:
        """Flips if a copy the notes can name is noted whole, which would drop the table at every follower."""
        await _copy(noted, _rows(500))

        notes = await _notes(noted)
        assert len(notes) == 500
        assert all(note["keys"] is not None for note in notes)

    async def test_a_bulk_copy_keeps_within_twice_a_plain_copy(self, noted: _Noted) -> None:
        """Flips if copying a state's rows into a noted table takes over twice a plain COPY of them here."""
        async with noted.pool.acquire() as conn:
            if not await on_yugabyte(conn):
                pytest.skip("the budget is YugabyteDB's, the platform's database; Postgres checks the notes only")
            rows = _rows(_BULK_ROWS)
            records = [(row["id"], row["state"], row["votes"], row["pad"]) for row in rows]
            baseline = await plain_copy_seconds(
                conn,
                [("id", "TEXT PRIMARY KEY"), ("state", "TEXT NOT NULL"), ("votes", "BIGINT"), ("pad", "TEXT")],
                records,
                runs=_RUNS,
            )
        best = float("inf")
        for _ in range(_RUNS):
            await noted.pool.execute(f"TRUNCATE {_TABLE}")
            for index in range(NOTE_RING_SIZE):
                await noted.pool.execute(f"DELETE FROM {noted.schema}.{note_table(index)}")
            best = min(best, await _copy(noted, rows))
        problem = budget_problem(f"a {_BULK_ROWS:,}-row copy", best, baseline)
        assert problem is None, problem
