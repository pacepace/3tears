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

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from threetears.core.collections.change_notes import NotedRows, install_change_notes, noted_rows_since
from threetears.core.ddl.change_notes import RING_TABLE, note_table
from threetears.core.testing.calibrate import budget_problem
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

#: runs each statement is given before it is timed: a prepared statement is planned for its values for its first
#: five, and keeps one plan from the sixth
_WARM_RUNS = 6


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
        await install_change_notes(conn, schema)
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
        # the current table has been current all night: the switch into it, and its fence, are hours past
        await conn.execute(f"UPDATE {schema}.{RING_TABLE} SET switched_at = $1 WHERE id = 1", night)
        yield _Ring(conn=conn, schema=schema, since=since)
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


class TestATablesNotesAreReadThroughAnIndex:
    @pytest.mark.parametrize("plan_cache_mode", ["auto", "force_generic_plan"])
    async def test_the_reconcilers_ask_scans_about_its_own_notes(self, ring: _Ring, plan_cache_mode: str) -> None:
        """Flips if a table's notes since a time are found by scanning every note, or answer other than the notes,
        under the plan made for the statement's own values or the one a prepared statement keeps (after its fifth
        run the reconciler's ask is planned once, generically: pg_prepared_statements counts it)."""
        await ring.conn.execute(f"SET plan_cache_mode = {plan_cache_mode}")
        checked = PlanRecorder(ring.conn)

        noted = await noted_rows_since(checked, notes_schema=ring.schema, table=_ASKED, since=ring.since, cap=10_000)

        assert isinstance(noted, NotedRows)
        assert {row.ids[0] for row in noted.rows} == {f"new-{g}" for g in range(1, _CHANGED + 1)}
        assert not noted.whole_table
        if await on_yugabyte(ring.conn) or plan_cache_mode != "auto":
            problems = plan_problems(checked.plans, scan_ratio=_SCAN_RATIO, together=True)
            assert not problems, problems

    async def test_the_reconcilers_ask_keeps_within_twice_the_reads_it_must_make(self, ring: _Ring) -> None:
        """Flips if the ask takes over twice the bare read it makes: the ring row and the table's notes since the
        reach, read in one statement through the notes' own index on the real table.

        That is the database's own cost for the operation, as every keyed-read baseline here is: the rows are
        found where they live, among a night of other tables' notes. The previous note table is past its fence
        (the ring switched at the night's start), so it holds none since and is not part of the read. An earlier
        baseline read the same rows from plain tables holding only them; it left out the index read every real
        ask pays, and the path stayed at about 2.7x it however the ask was built."""
        if not await on_yugabyte(ring.conn):
            pytest.skip("the budget is YugabyteDB's, the platform's database; Postgres checks correctness only")
        wait = await ring.conn.fetchval(f"SELECT wait_seconds FROM {ring.schema}.{RING_TABLE} WHERE id = 1")
        reach = ring.since - timedelta(seconds=float(wait)) / 2
        bare = (
            f"SELECT r.switch_no, n.* FROM {ring.schema}.{RING_TABLE} r LEFT JOIN LATERAL "  # noqa: S608
            f"(SELECT DISTINCT key_text, keys, columns, tenant FROM {ring.schema}.{note_table(0)} "
            "WHERE table_name = $1 AND noted_at >= $2) n ON TRUE WHERE r.id = 1"
        )
        # each statement prepared and past its first five runs, as the reconciler's is after its first minute: from
        # the sixth a prepared statement keeps one plan rather than planning each run (pg_prepared_statements)
        for _ in range(_WARM_RUNS):
            await ring.conn.fetch(bare, _ASKED, reach)
            await noted_rows_since(ring.conn, notes_schema=ring.schema, table=_ASKED, since=ring.since, cap=10_000)
        # timed in turn, run by run, so a stall of the database (a compaction after the night's notes went in)
        # falls on both alike rather than on whichever was timed second
        baseline = took = float("inf")
        for _ in range(_RUNS):
            began = time.perf_counter()
            await ring.conn.fetch(bare, _ASKED, reach)
            baseline = min(baseline, time.perf_counter() - began)
            began = time.perf_counter()
            await noted_rows_since(ring.conn, notes_schema=ring.schema, table=_ASKED, since=ring.since, cap=10_000)
            took = min(took, time.perf_counter() - began)
        problem = budget_problem("noted_rows_since", took, baseline)
        assert problem is None, problem


async def test_a_ring_just_switched_still_answers_from_the_previous_table(db_container: str) -> None:
    """Flips if the ask stops reading the previous table while a copy of the slot read before the switch may
    still be writing there (within the fence after the switch)."""
    schema = f"notes_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await install_change_notes(conn, schema)
        await conn.execute(
            f"UPDATE {schema}.{RING_TABLE} SET current_ring = 1, switch_no = 1, switched_at = now(), "
            "previous_switched_at = now() - interval '1 hour' WHERE id = 1"
        )
        since = datetime.now(UTC) - timedelta(seconds=10)
        await conn.execute(
            f"INSERT INTO {schema}.{note_table(0)} (table_name, keys, key_text) VALUES ($1, ARRAY['late'], 'late')",
            _ASKED,
        )
        noted = await noted_rows_since(conn, notes_schema=schema, table=_ASKED, since=since, cap=10)
        assert isinstance(noted, NotedRows)
        assert [row.ids for row in noted.rows] == [["late"]]
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()
