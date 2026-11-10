"""integration: each agent-audit index serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. The table is built by the package's own
migration and seeded past the size where the planner would rather scan; the statement the code under test really
writes (run over a store that records statements) is asked whether its plan found its rows through the index by an
index condition (:func:`threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import asyncpg
import pytest

from threetears.agent.audit.migrations import register
from threetears.agent.audit.persist import anonymize_audit_rows, prune_audit_events
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.plans import explain_statement, explain_write, plan_uses_index

pytestmark = pytest.mark.integration


class _Store:
    """The DataStore surface the runner drives, over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        return await self._conn.execute(sql, *params)

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._conn.fetch(sql, *params)]


#: rows seeded per table: past the size where the planner would rather scan than use an index
_ROWS = 5000

#: a value for a NOT NULL column with no default, by its type, from the series row ``g``
_BY_TYPE: dict[str, str] = {
    "uuid": "gen_random_uuid()",
    "text": "'v' || g",
    "character varying": "'v' || g",
    "timestamp with time zone": "now() - g * interval '1 second'",
    "timestamp without time zone": "now() - g * interval '1 second'",
    "integer": "g",
    "bigint": "g",
    "smallint": "1",
    "double precision": "g",
    "real": "g",
    "numeric": "g",
    "boolean": "false",
    "jsonb": "'{}'::jsonb",
    "json": "'{}'::json",
    "ARRAY": "'{}'",
}


async def _seed(
    conn: asyncpg.Connection, table: str, rows: int = _ROWS, *, _params: tuple[Any, ...] = (), **values: str
) -> None:
    """seed ``rows`` rows into ``table``, then ANALYZE it, so the planner prices it at size.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :param table: the table
    :ptype table: str
    :param rows: how many
    :ptype rows: int
    :param _params: the parameters the expressions name ($1, ...)
    :ptype _params: tuple[Any, ...]
    :param values: a SQL expression of the series row ``g`` per column; every other NOT NULL column with no
        default gets one by its type
    :ptype values: str
    :return: nothing
    :rtype: None
    """
    columns = await conn.fetch(
        "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name = $1",
        table,
    )
    names: list[str] = []
    exprs: list[str] = []
    for column in columns:
        name = column["column_name"]
        if name in values:
            names.append(name)
            exprs.append(values[name])
        elif column["is_nullable"] == "NO" and column["column_default"] is None:
            names.append(name)
            exprs.append(_BY_TYPE[column["data_type"]])
    await conn.execute(
        f'INSERT INTO "{table}" ({", ".join(names)}) SELECT {", ".join(exprs)} FROM generate_series(1, {rows}) g',
        *_params,
    )
    await conn.execute(f'ANALYZE "{table}"')


class _Captured:
    """an L3 pool that runs nothing and records each statement asked of it, so a test explains the statement the
    code under test really writes."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any, **_: Any) -> list[Any]:
        self.asked.append((sql, args))
        return []

    async def fetchrow(self, sql: str, *args: Any, **_: Any) -> None:
        self.asked.append((sql, args))

    async def fetchval(self, sql: str, *args: Any, **_: Any) -> int:
        self.asked.append((sql, args))
        return 0

    async def execute(self, sql: str, *args: Any, **_: Any) -> str:
        self.asked.append((sql, args))
        return "UPDATE 0"


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A connection on a fresh schema with audit_events built by the package and seeded: 5000 events over a
    hundred actors, spread over 5000 seconds."""
    schema = f"auditplan_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register(runner)
        await runner.apply_for_platform_schema(_Store(connection))  # type: ignore[arg-type]
        actors = [uuid.uuid4() for _ in range(100)]
        await _seed(connection, "audit_events", actor_user_id="($1::uuid[])[1 + g % 100]", _params=(actors,))
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await connection.close()


async def test_audit_events_time(conn: asyncpg.Connection) -> None:
    """Flips if retention's prune stops finding the expired events through idx_audit_events_time."""
    captured = _Captured()
    await prune_audit_events(captured, older_than=timedelta(minutes=80), batch_size=100)  # type: ignore[arg-type]
    (sql, args), *_ = captured.asked
    plan = await explain_write(conn, sql, *args)
    assert plan_uses_index(plan, "idx_audit_events_time"), plan.plan


async def test_audit_events_actor(conn: asyncpg.Connection) -> None:
    """Flips if an erasure stops finding an actor's events through idx_audit_events_actor."""
    captured = _Captured()
    await anonymize_audit_rows(captured, actor_user_ids=[uuid.uuid4()], batch_size=100)  # type: ignore[arg-type]
    (sql, args), *_ = captured.asked
    plan = await explain_statement(conn, sql, *args)
    assert plan_uses_index(plan, "idx_audit_events_actor"), plan.plan
