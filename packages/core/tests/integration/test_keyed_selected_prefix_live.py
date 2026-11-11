"""Integration: a Selected column reads what the hand-written CAST read, and a write by prefix reads by the key's index.

``read_keyed(columns=[..., Selected("CAST(t.s AS VARCHAR)", "s")])`` against the hand-written select it replaces,
through the equivalence harness. ``delete_keyed`` / ``update_keyed`` ``prefix=`` is a ``COLLATE "C"`` range on the key:
planned on a table whose key has a C-ordered ASC index (the range shape YugabyteDB reads without visiting every tablet),
it reads by that index, and it writes exactly the rows ``starts_with`` names. On the session's ``db_container``
(PostgreSQL), or YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.keyed import (
    AnyOf,
    Fragment,
    KeyBatch,
    Selected,
    counts_by_relations,
    delete_keyed,
    read_keyed,
    read_past_marks,
    update_keyed,
)
from threetears.core.testing.plans import StatementPlan, explain_write, plan_uses_index
from threetears.core.testing.same_results import Seeded, assert_same_results

pytestmark = pytest.mark.integration

_SIGNALS = 4000
_START = datetime(2026, 11, 3, 20, tzinfo=UTC)


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"selpre_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(
            "CREATE TABLE t (tenant text NOT NULL, k integer NOT NULL, s timestamptz, PRIMARY KEY (tenant, k))"
        )
        await connection.execute("CREATE TABLE n (signal text NOT NULL, v integer NOT NULL)")
        await connection.execute('CREATE UNIQUE INDEX n_signal_c ON n ((signal COLLATE "C") ASC)')
        await connection.executemany(
            "INSERT INTO n VALUES ($1, 0)",
            [(f"job/{i}",) for i in range(40)] + [(f"other{i:05d}",) for i in range(_SIGNALS)] + [("job",), ("job0",)],
        )
        await connection.execute("ANALYZE n")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _row(seed: Seeded) -> Mapping[str, Any]:
    stamp = None if seed.null else _START + timedelta(minutes=seed.tie, microseconds=123456)
    return {"tenant": f"t{seed.tenant}", "k": seed.index, "s": stamp}


async def test_a_selected_column_reads_what_the_hand_written_cast_read(conn: asyncpg.Connection) -> None:
    """Flips if a Selected expression reads other text than the CAST it replaces (a typed value, another format),
    loses its name, or the tenant pin beside it is dropped."""
    keys = list(range(13))
    await assert_same_results(
        conn,
        table="t",
        row=_row,
        old_sql="SELECT k, CAST(s AS VARCHAR) AS s FROM t WHERE tenant = $1 AND k = ANY($2::int[])",
        params=["t0", keys],
        helper=lambda c: read_keyed(
            c,
            "t",
            keys=KeyBatch(("k",), [(k,) for k in keys], ("int",)),
            columns=["k", Selected("CAST(t.s AS VARCHAR)", "s")],
            by={"tenant": "t0"},
        ),
    )


# parity-with: asyncpg.Connection (execute / fetch -- the seam the write helpers drive)
class _Planned:
    """a connection whose every write is planned (EXPLAIN, never run) before it runs."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn
        self.plans: list[StatementPlan] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.plans.append(await explain_write(self._conn, sql, *args))
        return str(await self._conn.execute(sql, *args))

    async def fetch(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        self.plans.append(await explain_write(self._conn, sql, *args))
        return list(await self._conn.fetch(sql, *args))


async def test_a_write_by_prefix_reads_by_the_key_index_and_writes_only_its_rows(conn: asyncpg.Connection) -> None:
    """Flips if the prefix range is not sargable on the C-ordered key index (a full scan of 4,000 rows for 40, on
    YugabyteDB every tablet), or writes a row starts_with does not name ("job", "job0") or misses one it does."""
    planned = _Planned(conn)
    named = sorted(row["signal"] for row in await conn.fetch("SELECT signal FROM n WHERE starts_with(signal, 'job/')"))

    updated = await update_keyed(planned, "n", values={"v": 1}, prefix={"signal": "job/"})
    deleted = await delete_keyed(planned, "n", prefix={"signal": "job/"}, returning=["signal"])

    assert updated.count == len(named) == 40
    assert sorted(row["signal"] for row in deleted.rows) == named
    assert {row["signal"] for row in await conn.fetch("SELECT signal FROM n WHERE signal LIKE 'job%'")} == {
        "job",
        "job0",
    }
    for plan in planned.plans:
        assert plan_uses_index(plan, "n_signal_c"), plan.plan


async def test_a_packed_count_answers_what_the_unpacked_count_answered(conn: asyncpg.Connection) -> None:
    """Flips if the packed counts_by_relations answers other groups or counts than the UNION ALL of keyed counts it
    packs (a tenant's pin dropped, a group lost to the packing, a count misread), in one statement."""
    await conn.execute("CREATE TABLE t2 (LIKE t INCLUDING ALL)")
    statements: list[str] = []

    # parity-with: asyncpg.Connection (fetch -- the seam counts_by_relations drives)
    class _Counted:
        async def fetch(self, sql: str, *args: Any) -> list[asyncpg.Record]:
            statements.append(sql)
            return list(await conn.fetch(sql, *args))

    keys = list(range(14))

    async def packed(c: asyncpg.Connection) -> list[dict[str, Any]]:
        counted = await counts_by_relations(
            _Counted(),
            {"t": "t", "t2": "t2"},
            column="k",
            values=AnyOf(keys, "int"),
            by={"tenant": "t0"},
            packed_rows=3,
        )
        return [
            {"relation": index, "counted": value, "held": held}
            for index, label in enumerate(("t", "t2"))
            for value, held in counted[label].items()
        ]

    await assert_same_results(
        conn,
        table="t",
        row=_row,
        old_sql=(
            "SELECT 0 AS relation, k AS counted, count(*) AS held FROM t WHERE tenant = $1 AND k = ANY($2::int[]) "
            "GROUP BY k UNION ALL "
            "SELECT 1 AS relation, k AS counted, count(*) AS held FROM t2 WHERE tenant = $1 AND k = ANY($2::int[]) "
            "GROUP BY k"
        ),
        params=["t0", keys],
        helper=packed,
    )
    assert statements and all(sql.startswith("SELECT MAX(CASE") for sql in statements)
    assert len(statements) == 8, "a packed count is one statement a call (one call a seeded case)"


async def test_a_pinned_read_past_a_floor_reads_what_the_hand_written_one_read(conn: asyncpg.Connection) -> None:
    """Flips if read_past_marks' by=, where= or floor= reads other rows than the hand-written change read it replaces
    (another tenant's, rows at or before the floor, rows the filter leaves out), on PostgreSQL or YugabyteDB."""
    floor = _START + timedelta(minutes=4)

    async def moved(c: asyncpg.Connection) -> list[dict[str, Any]]:
        read = await read_past_marks(
            c,
            "t",
            columns=["k", "s"],
            marks={"s": None},
            by={"tenant": "t0"},
            where=[Fragment('"k" % 2 = $1', (0,))],
            floor={"s": floor},
            casts={"s": "TIMESTAMPTZ"},
        )
        return read.rows

    await assert_same_results(
        conn,
        table="t",
        row=_row,
        old_sql="SELECT k, s FROM t WHERE tenant = $1 AND k % 2 = $2 AND CAST(s AS TIMESTAMPTZ) > CAST($3 AS TIMESTAMPTZ)",
        params=["t0", 0, floor],
        helper=moved,
    )
