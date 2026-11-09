"""Integration: a table's change notes since a time are read through an index, at election-night size.

The write-mark reconciler asks, every few seconds a marked table, which rows the change notes say that
table's writes changed since a time (:func:`noted_rows_since`). With no index leading on the table, each
ask scanned the whole note table (YugabyteDB: 4,602 rows scanned for none returned, about 0.35 s a marked
table at 1M notes). These seed a ring's current table with a night's notes across many tables, ask for one
table's recent notes, and judge the real plans with the shared plan check: rows scanned within 1.1x the
rows returned, and the whole ask within 2x the database's time to read those rows from a plain table.

Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL
to run them there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from threetears.core.collections.change_notes import NotedRows, noted_rows_since
from threetears.core.ddl.change_notes import NOTE_RING_SIZE, RING_TABLE, change_notes_ddl, note_table
from threetears.core.testing.calibrate import best_seconds, budget_problem, plain_copies
from threetears.core.testing.plans import PlanRecorder, on_yugabyte, plan_problems

pytestmark = pytest.mark.integration

#: notes in the ring's current table: a night's writes across many tables
_NOTES = 1_000_000
_TABLES = 2_000

#: the table asked about, and how many of its rows changed since the ask's time
_ASKED = "results_asked"
_CHANGED = 500

#: the most rows the ask scans for each row it answers
_SCAN_RATIO = 1.1

#: runs the ask and its baseline are each timed over, the best kept: a few milliseconds each, so one stall
#: in a short series would be the answer
_RUNS = 9


@dataclass
class _Ring:
    conn: asyncpg.Connection
    schema: str
    since: datetime


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[_Ring]:
    """a ring whose current table holds a night of notes, the asked table's newest among them."""
    schema = f"notes_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=1200)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        for statement in change_notes_ddl(schema):
            await conn.execute(statement)
        notes = f"{schema}.{note_table(0)}"
        night = datetime.now(UTC) - timedelta(hours=6)
        # every other table's notes, through the night
        await conn.execute(
            f"INSERT INTO {notes} (table_name, keys, key_text, noted_at) "
            f"SELECT 'table_' || (g % {_TABLES}), ARRAY[g::text], g::text, "
            f"$1::timestamptz + make_interval(secs => g * 0.02) FROM generate_series(1, {_NOTES}) g",
            night,
        )
        # the asked table: older notes, then the ones since the ask's time
        since = datetime.now(UTC) - timedelta(minutes=1)
        await conn.execute(
            f"INSERT INTO {notes} (table_name, keys, key_text, noted_at) "
            f"SELECT $1, ARRAY['old-' || g], 'old-' || g, $2::timestamptz + make_interval(secs => g) "
            "FROM generate_series(1, 2000) g",
            _ASKED,
            night,
        )
        await conn.execute(
            f"INSERT INTO {notes} (table_name, keys, key_text, noted_at) "
            f"SELECT $1, ARRAY['new-' || g], 'new-' || g, $2::timestamptz + make_interval(secs => 1) "
            f"FROM generate_series(1, {_CHANGED}) g",
            _ASKED,
            since,
        )
        await conn.execute(f"ANALYZE {notes}")
        yield _Ring(conn=conn, schema=schema, since=since)
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


class TestATablesNotesAreReadThroughAnIndex:
    async def test_the_reconcilers_ask_scans_about_its_own_notes(self, ring: _Ring) -> None:
        """Flips if a table's notes since a time are found by scanning every note, or answer other than the notes."""
        checked = PlanRecorder(ring.conn)

        noted = await noted_rows_since(checked, notes_schema=ring.schema, table=_ASKED, since=ring.since, cap=10_000)

        assert isinstance(noted, NotedRows)
        assert {row.ids[0] for row in noted.rows} == {f"new-{g}" for g in range(1, _CHANGED + 1)}
        assert not noted.whole_table
        if await on_yugabyte(ring.conn):
            problems = plan_problems(checked.plans, scan_ratio=_SCAN_RATIO, together=True)
            assert not problems, problems

    async def test_the_reconcilers_ask_keeps_within_twice_the_reads_it_must_make(self, ring: _Ring) -> None:
        """Flips if the ask takes over twice the bare reads it needs: the ring row, and the table's notes in the
        current and previous note tables, read from plain tables in one statement."""
        if not await on_yugabyte(ring.conn):
            pytest.skip("the budget is YugabyteDB's, the platform's database; Postgres checks correctness only")
        wait = await ring.conn.fetchval(f"SELECT wait_seconds FROM {ring.schema}.{RING_TABLE} WHERE id = 1")
        reach = ring.since - timedelta(seconds=float(wait)) / 2
        notes = "SELECT DISTINCT key_text, keys, columns, tenant FROM {table} WHERE table_name = $1 AND noted_at >= $2"
        sources = {
            "ring": (f"SELECT * FROM {ring.schema}.{RING_TABLE} WHERE id = 1", ()),  # noqa: S608
            "current": (notes.format(table=f"{ring.schema}.{note_table(0)}"), (_ASKED, reach)),
            "previous": (notes.format(table=f"{ring.schema}.{note_table(NOTE_RING_SIZE - 1)}"), (_ASKED, reach)),
        }
        async with plain_copies(ring.conn, sources) as plain:
            baseline = await best_seconds(
                lambda: ring.conn.fetch(
                    f"SELECT r.switch_no, n.* FROM {plain['ring']} r LEFT JOIN LATERAL "  # noqa: S608
                    f"(SELECT * FROM {plain['current']} UNION ALL SELECT * FROM {plain['previous']}) n ON TRUE"
                ),
                runs=_RUNS,
            )
        took = await best_seconds(
            lambda: noted_rows_since(ring.conn, notes_schema=ring.schema, table=_ASKED, since=ring.since, cap=10_000),
            runs=_RUNS,
        )
        problem = budget_problem("noted_rows_since", took, baseline)
        assert problem is None, problem
