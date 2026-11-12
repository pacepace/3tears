"""Integration: a backup's own whole reads (declared ``backup_verification``, Pace 2026-11-12) keep a calibrated speed.

DriftComparator's _count_and_checksum (through _compare_table and _count) reads every row of a table up to the backup's
instant by nature: it is timed through compare against the bare count-and-checksum statement on the same table
(200,000 rows), turn about, and held to the calibrated budget
(:func:`~threetears.core.testing.calibrate.budget_problem`), never a figure in seconds. ClusterBackup._count_tables
needs none: its statement is the bare count itself, which the rule checks (coordinator ruling 2026-11-12).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import pytest

from threetears.backup.compare import DriftComparator
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem

pytestmark = pytest.mark.integration

_ROWS = 200_000

#: every base table, as the comparison lists them
_TABLES = (
    "SELECT table_schema, table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE' "
    "AND table_schema NOT IN ('pg_catalog', 'information_schema') ORDER BY table_schema, table_name"
)


@pytest.fixture
async def seeded(db_container: str) -> AsyncIterator[tuple[asyncpg.Connection, str]]:
    """a schema of its own holding ``events`` (200,000 rows, a ``date_created`` cutoff column)."""
    schema = f"backup_speed_{uuid.uuid4().hex[:10]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(
            f'CREATE TABLE "{schema}".events (id BIGINT PRIMARY KEY, body TEXT NOT NULL, date_created TIMESTAMPTZ NOT NULL)'
        )
        await conn.execute(
            f"INSERT INTO \"{schema}\".events SELECT g, md5(g::text), now() - g * interval '1 second' "
            "FROM generate_series(1, $1::int) g",
            _ROWS,
        )
        await conn.execute(f'ANALYZE "{schema}".events')
        yield conn, schema
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


async def test_drift_checksums_speed_at_real_size(db_container: str, seeded: tuple[asyncpg.Connection, str]) -> None:
    """DriftComparator.compare (its _compare_table, _count and _count_and_checksum), both sides the same database,
    against the same work by hand: both sides connected and listed, each table's count-and-checksum read on both."""
    conn, _ = seeded

    async def connect(_dsn: str) -> asyncpg.Connection:
        return await asyncpg.connect(db_container, command_timeout=600)

    comparator = DriftComparator(connect, "restored", "live")
    tables = [(row["table_schema"], row["table_name"]) for row in await conn.fetch(_TABLES)]

    async def bare() -> None:
        # the same work written by hand: both sides connected and their tables listed, then each table's count and
        # checksum read on both
        sides = [await connect("restored"), await connect("live")]
        try:
            for side in sides:
                await side.fetch(_TABLES)
            for schema, table in tables:
                statement = f'SELECT count(*) AS n, coalesce(sum(hashtext(t::text)::bigint), 0) AS s FROM "{schema}"."{table}" t'
                for side in sides:
                    await side.fetchrow(statement)
        finally:
            for side in sides:
                await side.close()

    async def path() -> None:
        await comparator.compare(as_of=datetime.now(UTC))

    took_bare, took = await best_seconds_interleaved(bare, path, runs=3)
    problem = budget_problem("DriftComparator.compare", took, took_bare)
    assert problem is None, problem
