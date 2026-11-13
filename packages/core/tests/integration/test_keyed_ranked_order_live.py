"""``ranked_order`` on PostgreSQL: the rows and order ``ORDER BY first..., key(c0), key(c1)`` written per row answers,
NULLs in a leading sort and a ranked column, ties broken by a later ranked column, a ``LIMIT`` cut in a tie; and a
sort's own null order in an ordered step. Uses the session-scoped ``db_container`` fixture, a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.keyed import Fragment, Ranked, Sort, ordered, ranked_order, read_step, scan, value_ranks
from threetears.core.testing.same_results import Seeded, assert_same_results

pytestmark = pytest.mark.integration

_EPOCH = datetime(2026, 11, 3)


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"ranked_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(
            "CREATE TABLE held (state text, office text, run_id bigint PRIMARY KEY, instant timestamp, v bigint)"
        )
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _row(seed: Seeded) -> Mapping[str, Any]:
    return {
        "state": f"S{seed.tenant}",
        "office": None if seed.null else f"o{seed.tie % 3}{'0' * (seed.index % 2)}",
        "run_id": seed.index,
        "instant": None if seed.null else _EPOCH + timedelta(hours=seed.tie),
        "v": seed.index * 10,
    }


_TENANT = Fragment('t."state" = $1', ("S0",))
_RANKED = [Ranked("office", "length(v) DESC, v"), Ranked("state", "v")]
_FIRST = [Sort("instant", descending=True, nulls="last")]
_TERMS = "t.instant DESC NULLS LAST, length(t.office) DESC, t.office, t.state"


async def test_a_ranked_order_answers_the_order_by_its_keys_written_per_row(conn: asyncpg.Connection) -> None:
    """Flips if ranked_order answers other rows or another order than the per-row ``ORDER BY`` on PostgreSQL (its
    NULL order for a DESC sort is NULLS FIRST, so ``nulls="last"`` must be spelled), or a LIMIT cut in a tie keeps
    rows of another place in the order."""
    rel = scan("held", where=[_TENANT])
    await assert_same_results(
        conn,
        table="held",
        row=_row,
        old_sql=f"SELECT t.* FROM held AS t WHERE t.state = $1 ORDER BY {_TERMS}",
        params=["S0"],
        helper=lambda c: read_step(c, ranked_order(rel, _RANKED, first=_FIRST)),
        order_by=["instant", "office", "state"],
    )
    await assert_same_results(
        conn,
        table="held",
        row=_row,
        old_sql=f"SELECT t.instant, t.office, t.state FROM held AS t WHERE t.state = $1 ORDER BY {_TERMS} LIMIT 3",
        params=["S0"],
        helper=lambda c: read_step(c, ranked_order(rel, _RANKED, first=_FIRST, limit=3)),
        columns=["instant", "office", "state"],
    )


async def test_value_ranks_and_a_sorts_null_order_hold_on_postgresql(conn: asyncpg.Connection) -> None:
    """Flips if value_ranks ranks otherwise than its key orders the distinct values (NULL included: PostgreSQL puts a
    NULL key first descending), or an ordered step drops a sort's ``nulls`` on PostgreSQL."""
    await conn.execute(
        "INSERT INTO held VALUES ('S0', 'o10', 1, NULL, 1), ('S0', 'o9', 2, NULL, NULL), ('S0', NULL, 3, NULL, 3), "
        "('S0', 'o9', 4, NULL, 4)"
    )
    ranks = await read_step(conn, value_ranks("held", "office", "length(v) DESC, v"))
    assert sorted(((row["v"], row["rank"]) for row in ranks), key=lambda pair: pair[1]) == [
        (None, 1),
        ("o10", 2),
        ("o9", 3),
    ]
    down = await read_step(conn, ordered(scan("held", ["run_id", "v"]), [Sort("v", descending=True, nulls="last")]))
    assert [row["run_id"] for row in down] == [4, 3, 1, 2]
