"""Integration: each coordination table's expiry sweep finds its expired rows through the table's expiry index.

The sweep (``CoordinationCollection.sweep_expired``) deletes a bounded batch of the rows whose expiry passed, one
``keyed.delete_before`` page; it is the purpose of each table's ``idx_coordination_<table>_expiry_range``
(``expires_at ASC``: range-sharded on YugabyteDB, where a hashed index answers no range). The statement the sweep
really sends for each table is captured (its collection over a connection that records it), then planned on a table
seeded past the size where the planner would rather scan, with few rows expired: it reads the index by its condition
on ``expires_at``, never every row of the table.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
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
    CoordinationCollection,
    CoordinationCountersCollection,
    CoordinationRedemptionsCollection,
    CoordinationRevocationsCollection,
    coordination_collection,
    table_def_for,
)
from threetears.core.data.sql_builder import build_create_index_sql, build_create_table_sql
from threetears.core.testing.captured import CapturedL3
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


# parity-with: threetears.core.collections.generation.GenerationSource
class _Generations:
    """the revocations table caches absences, so a registry with L3 names a generation source."""

    async def current(self, table_name: str) -> str:
        del table_name
        return "i:0"

    async def advance(self, table_name: str) -> str:
        del table_name
        return "i:1"


async def _swept(collection_class: type[CoordinationCollection]) -> tuple[str, tuple[Any, ...]]:
    """the statement ``collection_class``'s sweep sends, and its parameters.

    :param collection_class: the coordination table's collection
    :ptype collection_class: type[CoordinationCollection]
    :return: the statement and its parameters
    :rtype: tuple[str, tuple[Any, ...]]
    """
    captured = CapturedL3()
    registry = CollectionRegistry()
    registry.configure(l1_backend=SQLiteBackend(db_name=f"coordplan_{uuid.uuid4().hex[:8]}"), l3_pool=captured)
    registry.set_generation_source(_Generations())
    collection = coordination_collection(registry, collection_class, None)
    try:
        await collection.sweep_expired(now=datetime.now(UTC), batch=1_000)
    finally:
        await registry.close_collections()
    (asked, *rest) = captured.asked
    assert not rest, captured.asked
    return asked


async def test_each_expiry_sweep_finds_its_rows_through_its_expiry_index(conn: asyncpg.Connection) -> None:
    """Flips if a table's sweep reaches its expired rows by any way but its expiry index's condition."""
    sweeps = {
        "idx_coordination_counters_expiry_range": CoordinationCountersCollection,
        "idx_coordination_claims_expiry_range": CoordinationClaimsCollection,
        "idx_coordination_revocations_expiry_range": CoordinationRevocationsCollection,
        "idx_coordination_redemptions_expiry_range": CoordinationRedemptionsCollection,
    }
    for index, collection_class in sweeps.items():
        sql, args = await _swept(collection_class)
        assert f'FROM "{collection_class.schema.name}"' in sql, sql
        plan = await explain_write(conn, sql, *args)
        assert plan_uses_index(plan, index), f"{index} is not how {sql[:60]} finds its rows: {plan.plan}"
