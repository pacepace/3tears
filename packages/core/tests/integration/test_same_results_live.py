"""the equivalence harness on PostgreSQL and YugabyteDB: the same statement passes, a dropped filter and a lost order
fail, and every case leaves the table as it was.

The unit tests prove the harness's judgement on DuckDB; these prove its calls (``executemany``, a transaction rolled
back per case, ``$n`` binds) on the databases the conversions run on. Uses the session-scoped ``db_container``
fixture, a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

import asyncpg
import pytest

from threetears.core.testing.same_results import Seeded, assert_same_results

pytestmark = pytest.mark.integration


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"sameres_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(
            "CREATE TABLE t (tenant text NOT NULL, k integer NOT NULL, v text, s integer NOT NULL, "
            "PRIMARY KEY (tenant, k))"
        )
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _row(seed: Seeded) -> Mapping[str, Any]:
    return {"tenant": f"t{seed.tenant}", "k": seed.index, "v": None if seed.null else f"v{seed.index}", "s": seed.tie}


_OLD = "SELECT k, v FROM t WHERE tenant = $1 ORDER BY k"


def _asking(sql: str, *params: Any) -> Any:
    async def helper(conn: asyncpg.Connection) -> list[asyncpg.Record]:
        return list(await conn.fetch(sql, *params))

    return helper


async def test_the_harness_judges_on_the_database(conn: asyncpg.Connection) -> None:
    """Flips if the same statement fails, a dropped tenant filter or a lost order passes, or a case's seed outlives
    it, on PostgreSQL or YugabyteDB."""
    await conn.execute("INSERT INTO t VALUES ('kept', 1, 'x', 1)")
    await assert_same_results(conn, table="t", row=_row, old_sql=_OLD, params=["t0"], helper=_asking(_OLD, "t0"))
    with pytest.raises(AssertionError, match="a tenant boundary"):
        await assert_same_results(
            conn, table="t", row=_row, old_sql=_OLD, params=["t0"], helper=_asking("SELECT k, v FROM t ORDER BY k")
        )
    with pytest.raises(AssertionError, match="another order"):
        await assert_same_results(
            conn,
            table="t",
            row=_row,
            old_sql=_OLD,
            params=["t0"],
            helper=_asking("SELECT k, v FROM t WHERE tenant = $1 ORDER BY k DESC", "t0"),
        )
    assert [dict(r) for r in await conn.fetch("SELECT tenant, k FROM t")] == [{"tenant": "kept", "k": 1}]
