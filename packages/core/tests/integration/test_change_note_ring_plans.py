"""Integration: each index of the change-note ring is the one its keeper read plans through.

The plan tests the ring's DDL names (``threetears.core.ddl.change_notes``: ``-- purpose:`` / ``-- plan_test:``): the
scrub's keyed look by the joined-key index, its whole-table look by the partial whole-note index, its any-note look by
the table-and-time index, a switch's count of the notes written without a cached slot by the partial ring-read
index, and a restore's read of a lost log's marks and a switch's sweep of the old marks by the marks' two indexes.
The ring's current table (and its marks) are seeded past where the planner would rather scan, on Postgres and
YugabyteDB alike.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.change_note_keeper import ChangeNoteKeeper
from threetears.core.collections.change_notes import NotedRows, noted_keys_page, noted_rows_since
from threetears.core.collections.epoch_change import ChangedRow, EpochChange
from threetears.core.ddl.change_notes import note_table
from threetears.core.testing.plans import PlanRecorder, plan_uses_index

pytestmark = pytest.mark.integration

_NOTES = 50_000
_TABLES = 500
#: every this many tables, one was loaded whole
_WHOLE_EVERY = 50
_RING_READS = 20


@dataclass
class _Ring:
    conn: asyncpg.Connection
    schema: str


class _NoLog:
    """the keeper's change log, never read by the reads under test."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the reads under test read no change log ({name})")


class _Pool:
    """one connection as the keeper's pool."""

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


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[_Ring]:
    """a ring whose current table holds many keyed notes, a few whole-table ones and a few written without a slot."""
    schema = f"ringplan_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await ChangeNoteKeeper(_Pool(conn), notes_schema=schema, log=_NoLog()).install()
        notes = f"{schema}.{note_table(0)}"
        await conn.execute(
            f"INSERT INTO {notes} (table_name, keys, key_text, ring_read) "  # noqa: S608 - test identifiers
            f"SELECT 'table_' || (g % {_TABLES}), ARRAY[g::text], g::text, "
            f"CASE WHEN g <= {_RING_READS} THEN TRUE END FROM generate_series(1, {_NOTES}) g"
        )
        await conn.execute(
            f"INSERT INTO {notes} (table_name) "  # noqa: S608 - test identifiers
            f"SELECT 'table_' || g FROM generate_series(0, {_TABLES - 1}, {_WHOLE_EVERY}) g"
        )
        # ANALYZE reads every seeded note, so each run's plans see the same statistics. At the default target it
        # samples 30,000 of the 50,010 notes, and how many of the 10 whole-table notes the sample holds decides the
        # whole-note look's plan: with none or one (about 1 run in 600) the planner reads the joined-key index's NULL
        # entries instead of the partial whole-note index (null_frac 0 and 1/30000 chose threetears_change_notes_0_key,
        # cost 8.06 and 12.08 against 12.29; 2/30000 and up chose _whole: PG, 2026-11-16). ANALYZE samples 300 rows
        # per point of the target, so 1000 reads the whole table.
        await conn.execute("SET default_statistics_target = 1000")
        await conn.execute(f"ANALYZE {notes}")
        await conn.execute("RESET default_statistics_target")
        yield _Ring(conn=conn, schema=schema)
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _scrub_plans(ring: _Ring) -> list[Any]:
    """the plans of the scrub's looks for a keyed epoch and a whole-table one, on the ring's current table.

    :return: the plans
    :rtype: list[StatementPlan]
    """
    checked = PlanRecorder(ring.conn)
    keeper = ChangeNoteKeeper(_Pool(checked), notes_schema=ring.schema, log=_NoLog())
    state = await ChangeNoteKeeper(_Pool(ring.conn), notes_schema=ring.schema, log=_NoLog()).ring()
    records = [
        EpochChange(table="table_7", token="i:1", rows=[ChangedRow(ids=["7"]), ChangedRow(ids=["507"])]),
        EpochChange(table="table_50", token="i:2", whole_table=True),
    ]
    await keeper.notes_found(state, records)
    return [plan for plan in checked.plans if note_table(state.current) in plan.sql]


async def test_threetears_change_notes_0_key_reads_the_scrubs_keyed_look(ring: _Ring) -> None:
    """Flips if the scrub's look for the keys an epoch names stops planning through the joined-key index."""
    plans = await _scrub_plans(ring)
    assert any(plan_uses_index(plan, "threetears_change_notes_0_key") for plan in plans), [p.plan for p in plans]


async def test_threetears_change_notes_0_whole_reads_the_scrubs_whole_table_look(ring: _Ring) -> None:
    """Flips if the scrub's look for a table's whole-table notes stops planning through the partial whole-note index."""
    plans = await _scrub_plans(ring)
    assert any(plan_uses_index(plan, "threetears_change_notes_0_whole") for plan in plans), [p.plan for p in plans]


async def test_threetears_change_notes_0_table_reads_the_scrubs_any_note_look(ring: _Ring) -> None:
    """Flips if the scrub's look for whether a table has any note stops planning through the table-and-time index."""
    plans = await _scrub_plans(ring)
    assert any(plan_uses_index(plan, "threetears_change_notes_0_table") for plan in plans), [p.plan for p in plans]


async def test_threetears_change_notes_0_table_reads_a_tables_notes_since_a_time(ring: _Ring) -> None:
    """Flips if any read of one table's notes since a time stops planning through the table-and-time index: the
    scrub's any-note look, the write-mark reconciler's read (``noted_rows_since``, its notes read built by
    ``_noted_rows_step``) and a follower's keys page (``noted_keys_page``, each notes read built by
    ``_noted_keys_sql``), each on the ring's current table."""
    plans = await _scrub_plans(ring)
    assert any(plan_uses_index(plan, "threetears_change_notes_0_table") for plan in plans), [p.plan for p in plans]
    now = await ring.conn.fetchval("SELECT now()")
    reconciled = PlanRecorder(ring.conn)
    found = await noted_rows_since(
        _Pool(reconciled), notes_schema=ring.schema, table="table_7", since=now - timedelta(hours=1), cap=1_000
    )
    assert not isinstance(found, str) and len(found.rows) == _NOTES // _TABLES, found
    [plan] = reconciled.plans
    assert plan_uses_index(plan, "threetears_change_notes_0_table"), plan.plan
    paged = PlanRecorder(ring.conn)
    page = await noted_keys_page(
        paged,
        notes_schema=ring.schema,
        table="table_7",
        tenant_token=None,
        noted_between=(now - timedelta(hours=1), now + timedelta(minutes=1)),
        cursor=None,
        limit=10,
        max_bytes=1_000_000,
    )
    assert not isinstance(page, str) and 0 < len(page.keys) <= 10, page  # a page ends at a whole instant
    current = [plan for plan in paged.plans if f'"{note_table(0)}"' in plan.sql]
    assert current and all(plan_uses_index(plan, "threetears_change_notes_0_table") for plan in current), [
        p.plan for p in paged.plans
    ]


async def test_threetears_change_notes_0_whole_reads_the_whole_notes_past_the_reconcilers_cap(ring: _Ring) -> None:
    """Flips if ``noted_rows_since``'s read of a table's whole-table notes, made when the notes name more rows than
    its cap (``_noted_whole_since``), stops planning through the partial whole-note index: it would then visit every
    keyed note that took the read past its cap."""
    now = await ring.conn.fetchval("SELECT now()")
    recorded = PlanRecorder(ring.conn)
    found = await noted_rows_since(
        _Pool(recorded), notes_schema=ring.schema, table="table_0", since=now - timedelta(hours=1), cap=10
    )
    assert isinstance(found, NotedRows) and found.too_many_rows and found.whole_table and not found.rows, found
    # the look at the ring's current table (the one seeded); each slot's is its own statement
    whole = [plan for plan in recorded.plans if "key_text IS NULL" in plan.sql and f'"{note_table(0)}"' in plan.sql]
    assert whole and all(plan_uses_index(plan, "threetears_change_notes_0_whole", partial=True) for plan in whole), [
        p.plan for p in recorded.plans
    ]


async def test_threetears_change_notes_0_ring_read_reads_a_switchs_count(ring: _Ring) -> None:
    """Flips if a switch's count of the notes written without a cached slot stops planning through its partial
    index."""
    checked = PlanRecorder(ring.conn)
    await ChangeNoteKeeper(_Pool(checked), notes_schema=ring.schema, log=_NoLog()).count_ring_reads(0)
    [plan] = checked.plans
    assert plan_uses_index(plan, "threetears_change_notes_0_ring_read", partial=True), plan.plan


#: the marks seeded for the marks' plan tests: this many logs, each marked this many times
_MARKED_LOGS = 200
_MARKS_EACH = 100
#: the newest seeded mark's time, less an hour: the g-th mark was written g hours before it
_MARKED_AT = datetime(2026, 1, 1, tzinfo=UTC)


async def _seed_marks(ring: _Ring) -> None:
    """mark many logs many times each, an hour apart, so a read of one log's marks, or of the oldest, is worth an
    index."""
    marks = f"{ring.schema}.threetears_change_note_marks"
    await ring.conn.execute(
        f"INSERT INTO {marks} (log_incarnation, seq, at) "  # noqa: S608 - test identifiers
        f"SELECT 'log_' || (g % {_MARKED_LOGS}), g / {_MARKED_LOGS}, $1::timestamptz - make_interval(hours => g) "
        f"FROM generate_series(1, {_MARKED_LOGS * _MARKS_EACH}) g",
        _MARKED_AT,
    )
    await ring.conn.execute(f"ANALYZE {marks}")


async def test_threetears_change_note_marks_position_reads_a_restores_marks(ring: _Ring) -> None:
    """Flips if a restore's read of one lost log's marks past its floor stops planning through the marks' position
    index, or answers other than that log's marks past the floor in position order."""
    await _seed_marks(ring)
    checked = PlanRecorder(ring.conn)
    marks = await ChangeNoteKeeper(_Pool(checked), notes_schema=ring.schema, log=_NoLog()).marks_past("log_7", 40)
    assert [int(mark["seq"]) for mark in marks] == list(range(41, _MARKS_EACH))
    assert checked.plans and all(
        plan_uses_index(plan, "threetears_change_note_marks_position") for plan in checked.plans
    ), [p.plan for p in checked.plans]


async def test_threetears_change_note_marks_at_reads_a_switchs_sweep(ring: _Ring) -> None:
    """Flips if a switch's sweep of the marks older than the coverage stops planning through the marks' time index,
    or deletes other than those marks."""
    await _seed_marks(ring)
    marks = f"{ring.schema}.threetears_change_note_marks"
    # the 30 oldest marks (g = 19971 .. 20000) were written before it
    cut = _MARKED_AT - timedelta(hours=_MARKED_LOGS * _MARKS_EACH - 29.5)
    checked = PlanRecorder(ring.conn)
    deleted = await ChangeNoteKeeper(_Pool(checked), notes_schema=ring.schema, log=_NoLog()).drop_marks_before(cut)
    assert deleted == 30
    assert await ring.conn.fetchval(f"SELECT count(*) FROM {marks} WHERE at < $1", cut) == 0  # noqa: S608 - test identifiers
    assert checked.plans and all(plan_uses_index(plan, "threetears_change_note_marks_at") for plan in checked.plans), [
        p.plan for p in checked.plans
    ]
