"""Integration: each index of the change-note ring is the one its keeper read plans through.

The plan tests the ring's DDL names (``threetears.core.ddl.change_notes``: ``-- purpose:`` / ``-- plan_test:``): the
scrub's keyed look by the joined-key index, its whole-table look by the partial whole-note index, its any-note look by
the table-and-time index, and a switch's count of the notes written without a cached slot by the partial ring-read
index. The ring's current table is seeded past where the planner would rather scan, on Postgres and YugabyteDB alike.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.change_notes import ChangeNoteKeeper
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
        await conn.execute(f"ANALYZE {notes}")
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


async def test_threetears_change_notes_0_ring_read_reads_a_switchs_count(ring: _Ring) -> None:
    """Flips if a switch's count of the notes written without a cached slot stops planning through its partial
    index."""
    checked = PlanRecorder(ring.conn)
    await ChangeNoteKeeper(_Pool(checked), notes_schema=ring.schema, log=_NoLog()).count_ring_reads(0)
    [plan] = checked.plans
    assert plan_uses_index(plan, "threetears_change_notes_0_ring_read", partial=True), plan.plan
