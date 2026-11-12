"""Integration: a keyed aggregate (:func:`aggregate_keyed`) answers what the hand-written GROUP BY answers, and reads
only the rows its prefix, its set of values and its window name, through the index they lead.

A usage table (40,000 events of 200 customers over 30 days) with the two indexes a metrics read needs: (customer_id,
date_created) for a tenant set's window, and (date_created) for every tenant's window. Each statement the helper
sends is captured and planned on PostgreSQL, or YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; each is read
through its index, never the whole table.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.keyed import AnyOf, Count, Range, Selected, Sum, aggregate_keyed
from threetears.core.testing.plans import explain_statement, plan_uses_index

pytestmark = pytest.mark.integration

_ROWS = 40_000
_START = datetime(2026, 10, 1, tzinfo=UTC)
#: a metrics window: most of the table's days, so a tenant set's rows in it are read through its own index, never
#: the window's every tenant
_LOW, _HIGH = _START + timedelta(days=2), _START + timedelta(days=28)

#: a narrow window of every tenant, read through the window's own index
_NARROW_LOW, _NARROW_HIGH = _START + timedelta(days=12), _START + timedelta(days=14)


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"aggregate_{uuid.uuid4().hex[:10]}"
    held = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await held.execute(f'CREATE SCHEMA "{schema}"')
        await held.execute(f'SET search_path TO "{schema}"')
        await held.execute(
            "CREATE TABLE usage (id BIGINT PRIMARY KEY, customer_id UUID NOT NULL, date_created TIMESTAMPTZ NOT NULL, "
            "agent_id TEXT NOT NULL, input_tokens BIGINT NOT NULL, output_tokens BIGINT NOT NULL)"
        )
        await held.execute("CREATE INDEX usage_by_customer_created ON usage (customer_id ASC, date_created ASC)")
        await held.execute("CREATE INDEX usage_by_created ON usage (date_created ASC)")
        customers = [uuid.uuid4() for _ in range(200)]
        await held.execute(
            "INSERT INTO usage SELECT g, ($1::uuid[])[1 + g % 200], $2::timestamptz + make_interval(secs => g * 64.8), "
            "'a' || (g % 7), g % 1000, g % 37 FROM generate_series(1, $3::int) g",
            customers,
            _START,
            _ROWS,
        )
        await held.execute("ANALYZE usage")
        yield held
    finally:
        await held.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await held.close()


class _Recording:
    """the test connection, keeping each statement the helper sends."""

    binds_native_arrays = True

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *params: Any) -> list[Any]:
        self.sent.append((query, params))
        return list(await self._conn.fetch(query, *params))


async def _customers(conn: asyncpg.Connection, count: int) -> list[uuid.UUID]:
    return [row[0] for row in await conn.fetch("SELECT DISTINCT customer_id FROM usage ORDER BY 1 LIMIT $1", count)]


async def test_a_tenant_set_s_window_by_hour_is_the_hand_written_aggregate_through_its_index(
    conn: asyncpg.Connection,
) -> None:
    """Flips if a tenant set's hourly counts and token sums differ from the hand-written GROUP BY, a batch is lost or
    counted twice, or a statement reads past the set's window (not through (customer_id, date_created))."""
    customers = await _customers(conn, 12)
    recording = _Recording(conn)
    rows = await aggregate_keyed(
        recording,
        "usage",
        aggregates={"requests": Count(), "tokens": Sum('t."input_tokens" + t."output_tokens"')},
        by={"customer_id": AnyOf(customers, "uuid")},
        ranges=[Range("date_created", _LOW, _HIGH)],
        group=[Selected("date_trunc('hour', t.\"date_created\")", "bucket")],
        batch=5,
    )
    expected = await conn.fetch(
        "SELECT date_trunc('hour', date_created) AS bucket, count(*) AS requests, "
        "sum(input_tokens + output_tokens) AS tokens FROM usage WHERE customer_id = ANY($1::uuid[]) "
        "AND date_created >= $2 AND date_created < $3 GROUP BY 1",
        customers,
        _LOW,
        _HIGH,
    )
    assert sorted((row["bucket"], row["requests"], row["tokens"]) for row in rows) == sorted(
        (row["bucket"], row["requests"], row["tokens"]) for row in expected
    )
    assert len(recording.sent) == 3
    window = await conn.fetchval(
        "SELECT count(*) FROM usage WHERE customer_id = ANY($1::uuid[]) AND date_created >= $2 AND date_created < $3",
        customers,
        _LOW,
        _HIGH,
    )
    for sql, params in recording.sent:
        plan = await explain_statement(conn, sql, *params)
        assert plan_uses_index(plan, "usage_by_customer_created"), plan.plan
        assert plan.rows_scanned <= window, (plan.rows_scanned, window)


async def test_every_tenant_s_window_is_read_through_the_window_s_index(conn: asyncpg.Connection) -> None:
    """Flips if a window of every tenant (the platform's own metrics) reads the whole table, or its two edges (a
    window split at a rollup) are AND-ed rather than OR-ed."""
    recording = _Recording(conn)
    split = _NARROW_LOW + timedelta(hours=20)
    (counted,) = await aggregate_keyed(
        recording,
        "usage",
        aggregates={"requests": Count()},
        ranges=[
            Range("date_created", _NARROW_LOW, _NARROW_LOW + timedelta(hours=2)),
            Range("date_created", split, _NARROW_HIGH),
        ],
    )
    expected = await conn.fetchval(
        "SELECT count(*) FROM usage WHERE (date_created >= $1 AND date_created < $2) OR "
        "(date_created >= $3 AND date_created < $4)",
        _NARROW_LOW,
        _NARROW_LOW + timedelta(hours=2),
        split,
        _NARROW_HIGH,
    )
    assert counted["requests"] == expected
    ((sql, params),) = recording.sent
    plan = await explain_statement(conn, sql, *params)
    assert plan_uses_index(plan, "usage_by_created"), plan.plan
    assert plan.rows_scanned <= expected, (plan.rows_scanned, expected)


async def test_a_tenant_set_s_distinct_agents_are_each_once(conn: asyncpg.Connection) -> None:
    """Flips if a keyed DISTINCT over batches answers an agent twice, or misses one the hand-written read finds."""
    customers = await _customers(conn, 9)
    rows = await aggregate_keyed(
        conn,
        "usage",
        aggregates={},
        by={"customer_id": AnyOf(customers, "uuid")},
        ranges=[Range("date_created", _LOW, _HIGH)],
        group=["agent_id"],
        batch=2,
    )
    expected = await conn.fetch(
        "SELECT DISTINCT agent_id FROM usage WHERE customer_id = ANY($1::uuid[]) AND date_created >= $2 "
        "AND date_created < $3",
        customers,
        _LOW,
        _HIGH,
    )
    assert sorted(row["agent_id"] for row in rows) == sorted(row["agent_id"] for row in expected)
