"""Integration: a whole-key delete stays led by the key under a generic plan, at real size.

``build_whole_key_delete_sql`` names its keys as ``unnest`` arrays, which the planner gives a fixed row
estimate, and a prepared statement may keep a generic plan after its fifth run. Either could turn the delete
into a scan of the table for every batch of keys. Checked here, never run (a write is planned, not
analysed): under the custom plan and the forced generic one, the table is reached through its key, and no
sequential scan of it appears anywhere in the plan.

Asserted on YugabyteDB, the platform's database: against the session ``db_container`` (Postgres) it skips;
set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import asyncpg
import pytest

from threetears.core.backends.schema_sql import build_whole_key_delete_sql, build_whole_key_select_sql
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.testing.plans import explain_write, on_yugabyte

pytestmark = pytest.mark.integration

_SCHEMA = TableSchema(
    name="results",
    primary_key=("race", "geo"),
    columns=[Column("race", STRING_TYPE), Column("geo", STRING_TYPE), Column("votes", BIGINT_TYPE, nullable=True)],
)

_ROWS = 300_000
_KEYS = 1_000


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"wholekey_{uuid.uuid4().hex[:8]}"
    held = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await held.execute(f"CREATE SCHEMA {schema}")
        await held.execute(f"SET search_path = {schema}")
        await held.execute("CREATE TABLE results (race TEXT, geo TEXT, votes BIGINT, PRIMARY KEY (race, geo))")
        await held.execute(
            f"INSERT INTO results SELECT 'r' || (g % 500), 'g' || g, g FROM generate_series(1, {_ROWS}) g"
        )
        await held.execute("ANALYZE results")
        yield held
    finally:
        await held.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await held.close()


def _nodes(plan: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield plan
    for child in plan.get("Plans", ()):
        yield from _nodes(child)


@pytest.mark.parametrize("mode", ["auto", "force_generic_plan"])
async def test_a_whole_key_delete_and_read_reach_the_table_by_its_key(conn: asyncpg.Connection, mode: str) -> None:
    """Flips if a batch of whole keys is deleted, or read, by scanning the table rather than looking each key up."""
    if not await on_yugabyte(conn):
        # Postgres hashes the keys against a scan of every row their leading values hold, which here is the
        # whole table: a plan of its own, for the platform's test database only
        pytest.skip("the key-led plan is YugabyteDB's, the platform's database")
    await conn.execute(f"SET plan_cache_mode = {mode}")
    keys = [(f"r{n % 500}", f"g{n}") for n in range(1, _KEYS + 1)]
    arrays = [list(column) for column in zip(*keys, strict=True)]
    for sql in (build_whole_key_delete_sql(_SCHEMA), build_whole_key_select_sql(_SCHEMA)):
        planned = await explain_write(conn, sql, *arrays) if sql.startswith("DELETE") else None
        plan = planned.plan if planned is not None else await _explain(conn, sql, arrays)
        scans = [node for node in _nodes(plan) if node.get("Relation Name") == "results"]
        assert scans, f"the plan reaches no table: {plan}"
        assert all(node["Node Type"] != "Seq Scan" for node in scans), f"{sql[:40]} scans the table: {plan}"


async def _explain(conn: asyncpg.Connection, sql: str, arrays: list[list[str]]) -> Mapping[str, Any]:
    explained = await conn.fetchval(f"EXPLAIN (FORMAT JSON) {sql}", *arrays)
    plan: Mapping[str, Any] = json.loads(explained)[0]["Plan"]
    return plan
