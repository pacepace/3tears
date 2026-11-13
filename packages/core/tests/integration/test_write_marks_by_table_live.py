"""Integration: a commit's read of its tables' write marks reads those marks, however long the queue.

The hub's broker, on every commit it answers, reads the marks of the tables the commit wrote, to clear those
its advances cover: ``SELECT id, table_name FROM collection_write_marks WHERE table_name = ANY($1)``. With
only the id key that read scanned the queue, the undrained marks of every other table and the tombstones of
every mark cleared before. This queues marks for many tables, clears most of them, and judges the broker's
read of one table's few marks with the shared plan check, on YugabyteDB; Postgres checks the answer.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.core import keyed
from threetears.core.ddl.write_marks import WRITE_MARKS_TABLE, write_marks_ddl
from threetears.core.sql_fragments import quote_identifier
from threetears.core.testing.plans import PlanRecorder, on_yugabyte, plan_problems, plan_uses_index

pytestmark = pytest.mark.integration

#: marks queued for other tables, most of them since cleared
_QUEUED = 100_000
#: the commit's table's own marks
_OWN = 5


@pytest.fixture
async def marks(db_container: str) -> AsyncIterator[tuple[asyncpg.Connection, str]]:
    schema = f"marks_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        for statement in write_marks_ddl(schema):
            await conn.execute(statement)
        queue = f"{quote_identifier(schema)}.{quote_identifier(WRITE_MARKS_TABLE)}"
        await conn.execute(
            f"INSERT INTO {queue} (table_name) SELECT 'table_' || (g % 500) FROM generate_series(1, {_QUEUED}) g"
        )
        await conn.execute(f"DELETE FROM {queue} WHERE id % 10 <> 0")
        await conn.execute(f"INSERT INTO {queue} (table_name) SELECT 'written' FROM generate_series(1, {_OWN})")
        await conn.execute(f"ANALYZE {queue}")
        yield conn, queue
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def test_a_commits_marks_are_read_by_their_table(marks: tuple[asyncpg.Connection, str]) -> None:
    """Flips if the broker's read of a commit's marks scans the queue rather than that table's marks."""
    conn, queue = marks
    checked = PlanRecorder(conn)

    rows = await checked.fetch(
        f"SELECT id, table_name FROM {queue} WHERE table_name = ANY($1::text[])",  # noqa: S608 - the broker's read
        ["written"],
    )

    assert len(rows) == _OWN
    if await on_yugabyte(conn):
        problems = plan_problems(checked.plans, scan_ratio=1.1)
        assert not problems, problems


async def test_a_commits_marks_are_read_through_the_table_index(marks: tuple[asyncpg.Connection, str]) -> None:
    """The plan test of ``collection_write_marks_table``: the hub broker's read of a commit's marks
    (``QueryProxy._marks_for``: ``keyed.read_by`` pinned by ``table_name = ANY``, reading ``id`` and ``table_name``)
    finds them through the index, on Postgres and YugabyteDB. Flips if the index goes or the read stops naming
    the table."""
    conn, queue = marks
    checked = PlanRecorder(conn)

    # the hub names the queue ``<schema>.collection_write_marks``, unquoted, as read_by takes a table
    rows = await keyed.read_by(
        checked, queue.replace('"', ""), by={"table_name": keyed.AnyOf(["written"])}, columns=("id", "table_name")
    )

    assert len(rows) == _OWN
    [plan] = checked.plans
    assert plan_uses_index(plan, "collection_write_marks_table"), plan
