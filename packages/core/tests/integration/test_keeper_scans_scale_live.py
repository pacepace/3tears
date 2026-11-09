"""Integration: the note keeper's periodic looks read the few notes they look for, not the ring, at night's size.

The scrub asks, every minute, whether each table an epoch names whole has a whole-table note, and whether a
table has any note; each ring switch counts the notes written without a cached slot. Without an index for
those few notes among a night's million, each look scanned a whole note table. These seed the ring's
current table with a million notes across two thousand tables, fifty of them noted whole and two hundred
notes written without a slot, and judge the keeper's real statements with the shared plan check: no more
rows scanned than 1.1x the rows returned, on YugabyteDB; Postgres checks the answers.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them
there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.change_notes import ChangeNoteKeeper
from threetears.core.collections.epoch_change import EpochChange
from threetears.core.ddl.change_notes import note_table
from threetears.core.testing.plans import PlanRecorder, on_yugabyte, plan_problems

pytestmark = pytest.mark.integration

_NOTES = 1_000_000
_TABLES = 2_000
#: every this many tables, one was loaded whole (a truncate, a file load)
_WHOLE_EVERY = 40
_RING_READS = 200
_SCAN_RATIO = 1.1


@dataclass
class _Ring:
    conn: asyncpg.Connection
    schema: str


class _NoLog:
    """the keeper's change log, never read by the looks under test."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the looks under test read no change log ({name})")


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[_Ring]:
    """a ring whose current table holds a night's notes: a few whole-table ones, a few written without a slot."""
    schema = f"keeper_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=1200)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await ChangeNoteKeeper(_Pool(conn), notes_schema=schema, log=_NoLog()).install()
        notes = f"{schema}.{note_table(0)}"
        await conn.execute(
            f"INSERT INTO {notes} (table_name, keys, key_text, ring_read) "
            f"SELECT 'table_' || (g % {_TABLES}), ARRAY[g::text], g::text, "
            f"CASE WHEN g <= {_RING_READS} THEN TRUE END FROM generate_series(1, {_NOTES}) g"
        )
        await conn.execute(
            f"INSERT INTO {notes} (table_name) SELECT 'table_' || g FROM generate_series(0, {_TABLES - 1}, {_WHOLE_EVERY}) g"
        )
        await conn.execute(f"ANALYZE {notes}")
        yield _Ring(conn=conn, schema=schema)
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


class _Pool:
    """one connection as the keeper's pool: ``fetch`` and ``execute`` on it, ``acquire`` handing it out."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def fetch(self, query: str, *args: Any) -> Any:
        return await self._conn.fetch(query, *args)

    async def execute(self, query: str, *args: Any) -> Any:
        return await self._conn.execute(query, *args)

    def acquire(self) -> Any:
        conn = self._conn

        class _Held:
            async def __aenter__(self) -> Any:
                return conn

            async def __aexit__(self, *exc: object) -> None:
                return None

        return _Held()


def _whole(table: str) -> EpochChange:
    return EpochChange(table=table, token="i:1", whole_table=True)


class TestTheKeepersLooksReadOnlyWhatTheyLookFor:
    async def test_the_scrub_finds_whole_and_noted_tables_by_index(self, ring: _Ring) -> None:
        """Flips if the scrub's whole-table or any-note look scans the note table rather than its index."""
        checked = PlanRecorder(ring.conn)
        keeper = ChangeNoteKeeper(_Pool(checked), notes_schema=ring.schema, log=_NoLog())
        state = await ChangeNoteKeeper(_Pool(ring.conn), notes_schema=ring.schema, log=_NoLog()).ring()
        asked = [f"table_{n}" for n in range(0, 200)]

        found = await keeper._found(state, [_whole(table) for table in asked])  # noqa: SLF001 - the scrub's look

        current = found[state.current]
        assert current.whole == {f"table_{n}" for n in range(0, 200, _WHOLE_EVERY)}
        assert current.any_note == set(asked)
        if await on_yugabyte(ring.conn):
            problems = plan_problems(checked.plans, scan_ratio=_SCAN_RATIO, together=True)
            assert not problems, problems

    async def test_a_switch_counts_the_notes_written_without_a_slot_by_index(self, ring: _Ring) -> None:
        """Flips if counting the notes written without a cached slot scans every note of the table."""
        checked = PlanRecorder(ring.conn)
        keeper = ChangeNoteKeeper(_Pool(checked), notes_schema=ring.schema, log=_NoLog())

        await keeper._count_ring_reads(0)  # noqa: SLF001 - the switch's count

        assert len(checked.plans) == 1
        [plan] = checked.plans
        assert plan.rows_returned == _RING_READS
        if await on_yugabyte(ring.conn):
            problems = plan_problems(checked.plans, scan_ratio=_SCAN_RATIO)
            assert not problems, problems
