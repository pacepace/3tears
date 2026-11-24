"""Integration: Arrow mode's key binding (each key column one text, split back in SQL) plans on PostgreSQL and
YugabyteDB as the plain lists' statements plan (by the key's index on YugabyteDB), names the rows they name, and
costs the serving process less CPU.

Tables at real size in a schema of their own: 300,000 rows keyed ``(state, geo, id)`` (the reports pod's results)
and 200,000 keyed by a UUID. 1,000, 10,000 and 50,000 keys of each, read, updated and deleted (rolled back), every
statement planned for real. Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to
a YugabyteDB URL to run them there.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pyarrow as pa
import pytest

from threetears.core.keyed import Increment, KeyBatch, binding, delete_keyed, read_keyed, update_keyed
from threetears.core.testing.calibrate import best_cpu_seconds
from threetears.core.testing.plans import PlanRecorder, StatementPlan, explain_write, on_yugabyte, plan_uses_index

pytestmark = pytest.mark.integration

_STATES, _GEOS, _IDS = 50, 200, 30
_UUID_ROWS = 200_000
_COUNTS = (1_000, 10_000, 50_000)

#: Arrow mode's serving-process CPU for a 50,000-key read, at most this share of the plain lists', by table: a composite
#: key with a number among its columns (measured 2026-11-23, PostgreSQL 16 and YugabyteDB 2026.1 on a laptop: 0.56-0.62,
#: the driver's encoding of 50,000 values a column gone) and UUIDs held as text (0.85-1.01: asyncpg sends a list of str
#: about as cheaply as Arrow joins and checks it)
_ARROW_CPU_SHARE: dict[str, float] = {"kc": 0.8, "ku": 1.1}


@dataclass
class _Seeded:
    conn: asyncpg.Connection
    yugabyte: bool


@pytest.fixture
async def seeded(db_container: str) -> AsyncIterator[_Seeded]:
    """the two tables at real size, with fresh statistics, in a schema of their own."""
    schema = f"kbind_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path = {schema}")
        await conn.execute(
            "CREATE TABLE kc (state TEXT NOT NULL, geo TEXT NOT NULL, id BIGINT NOT NULL, v BIGINT, "
            "PRIMARY KEY (state, geo, id))"
        )
        await conn.execute(
            f"""INSERT INTO kc SELECT 'S' || lpad(s::text, 2, '0'), lpad(g::text, 3, '0'), i, s * g * i
            FROM generate_series(1, {_STATES}) s, generate_series(1, {_GEOS}) g, generate_series(1, {_IDS}) i"""
        )
        await conn.execute("CREATE TABLE ku (id UUID PRIMARY KEY, v BIGINT)")
        await conn.execute(f"INSERT INTO ku SELECT gen_random_uuid(), g FROM generate_series(1, {_UUID_ROWS}) g")
        await conn.execute("ANALYZE kc")
        await conn.execute("ANALYZE ku")
        yield _Seeded(conn=conn, yugabyte=await on_yugabyte(conn))
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _keys(conn: asyncpg.Connection, count: int) -> tuple[list[list[Any]], list[str]]:
    """``count`` composite keys and ``count`` UUIDs, spread over their tables."""
    step = _STATES * _GEOS * _IDS // count
    composite: list[list[Any]] = [
        [f"S{(n // (_GEOS * _IDS)) + 1:02d}" for n in range(0, step * count, step)],
        [f"{((n // _IDS) % _GEOS) + 1:03d}" for n in range(0, step * count, step)],
        [(n % _IDS) + 1 for n in range(0, step * count, step)],
    ]
    held = await conn.fetch("SELECT id::text AS id FROM ku WHERE v % $1 = 0", _UUID_ROWS // count)
    return composite, [row["id"] for row in held][:count]


def _arrow(composite: list[list[Any]], ids: list[str]) -> tuple[KeyBatch, KeyBatch]:
    return (
        KeyBatch(
            ("state", "geo", "id"),
            [pa.array(composite[0]), pa.array(composite[1]), pa.array(composite[2], pa.int64())],
            ("text", "text", "bigint"),
        ),
        KeyBatch(("id",), [pa.array(ids)], ("uuid",)),
    )


def _plain(composite: list[list[Any]], ids: list[str]) -> tuple[KeyBatch, KeyBatch]:
    return KeyBatch(("state", "geo", "id"), composite, ("text", "text", "bigint")), KeyBatch(("id",), [ids], ("uuid",))


class _Writes:
    """a connection whose every write is planned (EXPLAIN, never run) before it runs."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn
        self.plans: list[StatementPlan] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.plans.append(await explain_write(self._conn, sql, *args))
        return str(await self._conn.execute(sql, *args))


def _no_seq_scan(plan: Any) -> bool:
    node = plan.plan if isinstance(plan, StatementPlan) else plan
    return node.get("Node Type") != "Seq Scan" and all(_no_seq_scan(child) for child in node.get("Plans", ()))


async def _rows(conn: Any, composite: KeyBatch, ids: KeyBatch) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    by_key = await read_keyed(conn, "kc", keys=composite, columns=["state", "geo", "id", "v"])
    by_id = await read_keyed(conn, "ku", keys=ids, columns=["id", "v"])
    return sorted(tuple(row.values()) for row in by_key), sorted(tuple(row.values()) for row in by_id)


@pytest.mark.parametrize("count", _COUNTS)
async def test_arrow_mode_plans_as_the_plain_lists_plan_and_reads_by_the_key(
    seeded: _Seeded, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if an Arrow-mode statement (each key column one text, split back in SQL) plans other than the same
    statement bound as plain lists (another scan, join or index: its array is a custom plan's constant in both) unless
    it reads by the key's index where they scanned, names other rows, or, on YugabyteDB (the platform's database),
    reads or writes other than through the key's index, at 1,000, 10,000 and 50,000 keys. PostgreSQL may scan a
    200,000-row table for 10,000 UUIDs bound either way: its own choice, the same as before."""
    composite, ids = await _keys(seeded.conn, count)
    plans: dict[bool, list[list[tuple[str, str]]]] = {}
    rows: dict[bool, Any] = {}
    counts: dict[bool, tuple[int, int]] = {}
    for arrow in (True, False):
        monkeypatch.setattr(binding, "_ARROW", arrow)
        keys = _arrow(composite, ids) if arrow else _plain(composite, ids)
        recorder = PlanRecorder(seeded.conn)
        rows[arrow] = await _rows(recorder, *keys)
        writes = _Writes(seeded.conn)
        transaction = seeded.conn.transaction()
        await transaction.start()
        try:
            updated = await update_keyed(writes, "kc", values={"v": Increment(1)}, keys=keys[0], batch=10_000)
            deleted = await delete_keyed(writes, "ku", keys=keys[1], batch=10_000)
        finally:
            await transaction.rollback()
        counts[arrow] = (updated.count, deleted.count)
        plans[arrow] = [_nodes(plan.plan) for plan in (*recorder.plans, *writes.plans)]
        if arrow and seeded.yugabyte:
            for plan in recorder.plans:
                index = "kc_pkey" if '"kc"' in plan.sql else "ku_pkey"
                assert plan_uses_index(plan, index) and _no_seq_scan(plan), (plan.sql[:200], plan.plan)
            for plan in writes.plans:
                assert _no_seq_scan(plan), json.dumps(plan.plan)[:600]
    assert [len(found) for found in rows[True]] == [count, count] and rows[True] == rows[False]
    assert counts[True] == counts[False] == (count, count)
    # the same plan, or, where a composite batch's bounds are made once a statement (an InitPlan the planner cannot
    # read ahead), a read by the key's index where the plain lists' constants planned a scan: never the reverse
    for arrow_plan, plain_plan in zip(plans[True], plans[False], strict=True):
        if arrow_plan != plain_plan:
            nodes = {node for node, _ in arrow_plan}
            assert "Seq Scan" not in nodes and any(index for _, index in arrow_plan), (arrow_plan, plain_plan)


def _nodes(node: Any) -> list[tuple[str, str]]:
    """a plan's nodes, in order: each one's type and the index it reads (its estimates and timings left out). An
    InitPlan is left out: the bounds beside a composite batch's keys, made once a statement in Arrow mode (each a
    ``Result``), read nothing."""
    return [
        (str(node.get("Node Type", "")), str(node.get("Index Name", ""))),
        *(
            child
            for plan in node.get("Plans", ())
            if plan.get("Parent Relationship") != "InitPlan"
            for child in _nodes(plan)
        ),
    ]


@pytest.mark.loop_timing
async def test_arrow_mode_costs_the_serving_process_less_cpu(seeded: _Seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if Arrow mode's 50,000-key read spends more than its :data:`_ARROW_CPU_SHARE` of the CPU the plain lists'
    read spends in this process (binding, the driver's encoding, the answer), composite or UUID-keyed: the speed
    budget Arrow mode keeps."""
    composite, ids = await _keys(seeded.conn, 50_000)
    arrow_keys, plain_keys = _arrow(composite, ids), _plain(composite, ids)
    for at, table, columns in ((0, "kc", ["v"]), (1, "ku", ["v"])):
        monkeypatch.setattr(binding, "_ARROW", True)
        arrow = await best_cpu_seconds(
            lambda: read_keyed(seeded.conn, table, keys=arrow_keys[at], columns=columns, records=True)
        )
        monkeypatch.setattr(binding, "_ARROW", False)
        plain = await best_cpu_seconds(
            lambda: read_keyed(seeded.conn, table, keys=plain_keys[at], columns=columns, records=True)
        )
        assert arrow <= _ARROW_CPU_SHARE[table] * plain, (
            f"{table}: Arrow mode {arrow * 1000:.1f} ms CPU, plain {plain * 1000:.1f} ms"
        )


#: the most seconds a 50,000-key composite read takes on YugabyteDB, best of three, its batches read by their keys alone
#: (measured 2026-11-23, YugabyteDB 2026.1 on a laptop: 0.24-0.29 s; bounded beside the keys, as PostgreSQL's still
#: are, 1.8-2.1 s: keyed/rows.py's render)
_YUGABYTE_50K_SECONDS = 0.8

#: the bound beside a composite batch's keys on its lead column (keyed/rows.py's _tight_values)
_LEAD_BOUND = 't."state" = ANY('


async def test_a_composite_batch_is_bounded_on_postgres_and_read_by_its_keys_alone_on_yugabyte(seeded: _Seeded) -> None:
    """Flips if a composite batch whose values' cross product names at most twice its keys loses the bounds beside
    its keys on PostgreSQL (its prefix reads' index condition), or carries them on YugabyteDB (where they cost ~50 ms
    of YSQL CPU a 1,000 keys), or either names other rows."""
    composite, _ = await _keys(seeded.conn, 1_000)
    keys = KeyBatch(("state", "geo", "id"), composite, ("text", "text", "bigint"))
    recorder = PlanRecorder(seeded.conn)
    rows = await read_keyed(recorder, "kc", keys=keys, columns=["state", "geo", "id", "v"])
    assert len(rows) == 1_000
    assert recorder.plans, "the read sent no statement"
    for plan in recorder.plans:
        assert (_LEAD_BOUND in plan.sql) is not seeded.yugabyte, plan.sql[:400]


@pytest.mark.yugabyte_only
@pytest.mark.loop_timing
async def test_a_50k_composite_read_on_yugabyte_is_within_its_budget(seeded: _Seeded) -> None:
    """Flips if 50,000 composite keys (Arrow mode, the serving path's) take YugabyteDB more than
    :data:`_YUGABYTE_50K_SECONDS` to read, best of three: the bounds beside the keys back (1.8-2.1 s), or any other
    cost a batch's statement gains there."""
    assert seeded.yugabyte, "a yugabyte_only test ran on another engine"
    composite, ids = await _keys(seeded.conn, 50_000)
    keys, _ = _arrow(composite, ids)
    rows = await read_keyed(seeded.conn, "kc", keys=keys, columns=["v"], records=True)
    assert len(rows) == 50_000
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        await read_keyed(seeded.conn, "kc", keys=keys, columns=["v"], records=True)
        best = min(best, time.perf_counter() - started)
    assert best <= _YUGABYTE_50K_SECONDS, f"50,000 composite keys read in {best:.2f} s on YugabyteDB"
