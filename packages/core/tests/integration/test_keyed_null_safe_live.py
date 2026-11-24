"""Integration: a null-safe key batch (keys with NULL parts) reads and writes by the key's index on PostgreSQL and
YugabyteDB, split by its NULL parts (``threetears.interfaces.keyed.rows.null_split``).

Sent as ``IS NOT DISTINCT FROM`` each key, 1,000 keys against this 200,000-row table took 3.9 s on PostgreSQL 16 and
5.3 s on YugabyteDB 2026.1 (a laptop, 2026-11-23; 11 s where first seen): no index reads by that operator, so the
table was read once a key, a nested loop over sequential scans. Split, each NULL pattern's NULL parts are pinned ``IS NULL``
beside equality on the rest: every pattern its index's own condition. A table keyed ``(scope, geo, id)`` whose scope
and geo may be NULL, in a schema of its own; 1,000 keys of which about a fifth hold a NULL part (and some name no
row), read, updated and deleted (rolled back), plain lists and Arrow alike. Against the session ``db_container``
(PostgreSQL); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them there.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import asyncpg
import pyarrow as pa
import pytest

from threetears.core.keyed import Increment, KeyBatch, delete_keyed, read_keyed, update_keyed
from threetears.core.testing.plans import PlanRecorder, WholeScanGuard, on_yugabyte, plan_uses_index

pytestmark = pytest.mark.integration

_ROWS = 200_000
_KEYS = 1_000
#: keys past the table: named, holding no row
_MISSING = 20

#: the read's wall time at most, seconds: split, 1,000 keys read in 3-6 ms on PostgreSQL 16 and 9-13 ms on YugabyteDB
#: 2026.1 (a laptop, best of three, 2026-11-23) where the IS NOT DISTINCT FROM join took 3.9 and 5.3 s; a budget
#: well clear of noise (a first, cold read) and still well under the old read
_READ_SECONDS = 1.0


def _scope(g: int) -> str | None:
    return None if g % 10 == 0 else f"S{g % 50:02d}"


def _geo(g: int) -> str | None:
    return None if g % 7 == 0 else f"G{g % 200:03d}"


@dataclass
class _Seeded:
    conn: asyncpg.Connection
    yugabyte: bool


@pytest.fixture
async def seeded(db_container: str) -> AsyncIterator[_Seeded]:
    """the table at real size, with fresh statistics, in a schema of its own."""
    schema = f"knull_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path = {schema}")
        await conn.execute("CREATE TABLE kn (scope TEXT, geo TEXT, id BIGINT NOT NULL, v BIGINT)")
        await conn.execute(
            f"""INSERT INTO kn SELECT
                CASE WHEN g % 10 = 0 THEN NULL ELSE 'S' || lpad((g % 50)::text, 2, '0') END,
                CASE WHEN g % 7 = 0 THEN NULL ELSE 'G' || lpad((g % 200)::text, 3, '0') END,
                g, g
            FROM generate_series(1, {_ROWS}) g"""
        )
        await conn.execute("CREATE INDEX kn_key ON kn (scope ASC, geo ASC, id ASC)")
        await conn.execute("ANALYZE kn")
        yield _Seeded(conn=conn, yugabyte=await on_yugabyte(conn))
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


def _named() -> tuple[list[list[Any]], list[int]]:
    """the keys (one list a column) and the ids of the rows they name: spread over the table, a fifth or so with a
    NULL part (every pattern: scope, geo, both), and :data:`_MISSING` past it."""
    step = _ROWS // _KEYS
    held = list(range(7, step * (_KEYS - _MISSING), step))[: _KEYS - _MISSING]
    past = list(range(_ROWS + 1, _ROWS + 1 + _MISSING))
    ids = [*held, *past]
    columns: list[list[Any]] = [[_scope(g) for g in ids], [_geo(g) for g in ids], ids]
    return columns, held


def _batch(columns: list[list[Any]], arrow: bool) -> KeyBatch:
    keys: list[Any] = (
        [pa.array(columns[0], pa.string()), pa.array(columns[1], pa.string()), pa.array(columns[2], pa.int64())]
        if arrow
        else columns
    )
    return KeyBatch(("scope", "geo", "id"), keys, ("text", "text", "bigint"), null_safe=True)


def _scans(node: Mapping[str, Any]) -> list[str]:
    """each relation scan's type in a plan."""
    own = [str(node["Node Type"])] if node.get("Relation Name") else []
    return [*own, *(scan for child in node.get("Plans", ()) for scan in _scans(child))]


@pytest.mark.parametrize("arrow", [False, True], ids=["plain", "arrow"])
async def test_a_null_safe_batch_reads_and_writes_by_the_key_s_index(seeded: _Seeded, arrow: bool) -> None:
    """Flips if a null-safe batch of 1,000 keys with NULL parts is read or written other than by the key's index (a
    sequential scan: IS NOT DISTINCT FROM a key, or a NULL pattern the index cannot condition on), reads past its
    :data:`_READ_SECONDS` budget, or names other rows than its keys do."""
    columns, held = _named()
    assert sum(None in key for key in zip(*columns, strict=True)) > _KEYS // 10
    recorder = PlanRecorder(seeded.conn)
    started = time.perf_counter()
    rows = await read_keyed(seeded.conn, "kn", keys=_batch(columns, arrow), columns=["v"])
    seconds = time.perf_counter() - started
    assert sorted(row["v"] for row in rows) == held
    assert sorted(row["v"] for row in await read_keyed(recorder, "kn", keys=_batch(columns, arrow), columns=["v"]))
    assert recorder.plans
    for plan in recorder.plans:
        assert "IS NOT DISTINCT FROM" not in plan.sql
        assert plan_uses_index(plan, "kn_key"), plan.plan
        assert "Seq Scan" not in _scans(plan.plan), _scans(plan.plan)
        # each key one lookup: the rows read are the rows answered (an index scan's, and on YugabyteDB the table's)
        assert plan.rows_scanned <= 2 * len(held), (plan.rows_scanned, len(held))
    assert seconds < _READ_SECONDS, f"read {_KEYS} null-safe keys in {seconds:.2f} s"
    print(f"null-safe read ({'arrow' if arrow else 'plain'}, yugabyte={seeded.yugabyte}): {seconds * 1000:.1f} ms")
    transaction = seeded.conn.transaction()
    await transaction.start()
    try:
        guard = WholeScanGuard(seeded.conn, threshold_rows=1_000)
        updated = await update_keyed(guard, "kn", values={"v": Increment(1)}, keys=_batch(columns, arrow))
        deleted = await delete_keyed(guard, "kn", keys=_batch(columns, arrow))
        read = await read_keyed(guard, "kn", keys=_batch(columns, arrow), columns=["v"])
    finally:
        await transaction.rollback()
    assert (updated.count, deleted.count, read) == (len(held), len(held), [])
    guard.check()
