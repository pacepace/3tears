"""Integration: each coordination table's expiry sweep finds its expired rows through the table's expiry index.

The sweep (``CoordinationCollection.sweep_expired``) deletes a bounded batch of the rows whose expiry passed. Each
table's statement (``sweep_expired_sql``) is the purpose of that table's ``idx_coordination_<table>_expiry``: seeded
past the size where the planner would rather scan, with few rows expired, its plan reads the index by its condition
on ``expires_at``, never every row of the table.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import pytest

from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    TableSchema,
)
from threetears.core.coordination.tables import (
    COORDINATION_TABLE_SCHEMAS,
    CoordinationClaimsCollection,
    CoordinationCountersCollection,
    CoordinationRedemptionsCollection,
    CoordinationRevocationsCollection,
    table_def_for,
)
from threetears.core.data.sql_builder import build_create_index_sql, build_create_table_sql
from threetears.core.testing.plans import explain_write, plan_uses_index

pytestmark = pytest.mark.integration

#: rows each table is seeded with; a hundredth of them expired
_ROWS = 50_000

#: a value of each column type a coordination table holds NOT NULL
_SEED_VALUES = {DATETIMETZ_TYPE: "now()", INT_TYPE: "0", BIGINT_TYPE: "0", STRING_TYPE: "'done'"}


def _seed_sql(declared: TableSchema) -> str:
    """seed ``declared`` with :data:`_ROWS` rows, a hundredth of them expired, every NOT NULL column given a value."""
    required = [column for column in declared.columns if not column.nullable and column.name not in {"purpose", "key"}]
    names = ", ".join(column.name for column in required)
    given = ", ".join(_SEED_VALUES[column.column_type] for column in required)
    return (
        f"INSERT INTO {declared.name} (purpose, key, expires_at, {names}) "  # noqa: S608 - the test's own tables
        f"SELECT 'p', 'k' || g, now() + make_interval(days => CASE WHEN g % 100 = 0 THEN -1 ELSE 30 END), {given} "
        f"FROM generate_series(1, {_ROWS}) g"
    )


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"coordexpiry_{uuid.uuid4().hex[:8]}"
    held = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await held.execute(f"CREATE SCHEMA {schema}")
        await held.execute(f"SET search_path = {schema}")
        for declared in COORDINATION_TABLE_SCHEMAS:
            table = table_def_for(declared)
            await held.execute(build_create_table_sql(table))
            for index in table.indexes:
                await held.execute(build_create_index_sql(table.name, index))
            await held.execute(_seed_sql(declared))
            await held.execute(f"ANALYZE {declared.name}")
        yield held
    finally:
        await held.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await held.close()


async def test_each_expiry_sweep_finds_its_rows_through_its_expiry_index(conn: asyncpg.Connection) -> None:
    """Flips if a table's sweep reaches its expired rows by any way but its expiry index's condition."""
    sweeps = {
        "idx_coordination_counters_expiry": CoordinationCountersCollection.sweep_expired_sql,
        "idx_coordination_claims_expiry": CoordinationClaimsCollection.sweep_expired_sql,
        "idx_coordination_revocations_expiry": CoordinationRevocationsCollection.sweep_expired_sql,
        "idx_coordination_redemptions_expiry": CoordinationRedemptionsCollection.sweep_expired_sql,
    }
    for index, sql in sweeps.items():
        plan = await explain_write(conn, sql, datetime.now(UTC), 1_000)
        assert plan_uses_index(plan, index), f"{index} is not how {sql[:60]} finds its rows: {plan.plan}"
