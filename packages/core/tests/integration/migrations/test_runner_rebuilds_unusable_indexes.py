"""Integration: a migrate run builds again any index of its schema a failed build left unusable.

On YugabyteDB 2026.1 an online ``CREATE INDEX`` that meets a failure part way, or races other DDL on the database,
leaves its index behind not valid; ``CREATE INDEX IF NOT EXISTS`` takes it as present ever after, so the planner never
reads it and a unique one enforces nothing. The runner repairs every such index of the schema it migrated, once, at
the end of the run (:func:`threetears.core.ddl.index_health.rebuild_unusable_indexes`), so no migration's own index
statement has to.

The leftover is made the way it arises: a ``CREATE UNIQUE INDEX CONCURRENTLY`` over duplicate rows, which both
PostgreSQL and YugabyteDB refuse and leave behind not valid; the duplicate is then resolved, as an operator would.
Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.data.migrations import ConnectionSession, MigrationRunner, MigrationScope, PackageMigrations

pytestmark = pytest.mark.integration

_VALIDITY_SQL = (
    "SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
    "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = current_schema() AND c.relname = $1"
)


@pytest.fixture
async def schema_conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """a connection bound to a schema of its own, dropped after the test."""
    schema = f"heal_{uuid.uuid4().hex[:8]}"
    conn: asyncpg.Connection = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        yield conn
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


def _runner() -> MigrationRunner:
    """a runner with one agent-scope package whose v1 creates the table the test indexes."""
    pkg = PackageMigrations(name="index_heal_it", scope=MigrationScope.AGENT)

    @pkg.version(1)
    async def create(store: Any) -> None:
        await store.execute("CREATE TABLE IF NOT EXISTS u (id INT PRIMARY KEY, c INT)")

    runner = MigrationRunner()
    runner.register(pkg)
    return runner


async def test_a_migrate_run_builds_an_index_left_invalid_again(schema_conn: asyncpg.Connection) -> None:
    """Flips if a migrate run leaves an index of its schema that a failed build left not valid."""
    runner = _runner()
    await runner.apply_for_agent_schema(ConnectionSession(schema_conn))
    await schema_conn.execute("INSERT INTO u VALUES (1, 7), (2, 7)")
    with pytest.raises(asyncpg.PostgresError) as refused:
        await schema_conn.execute("CREATE UNIQUE INDEX CONCURRENTLY ux_u_c ON u (c)")
    # PostgreSQL's unique violation; YugabyteDB 2026.1's internal error for the same refused build
    assert refused.value.sqlstate in ("23505", "XX000")
    assert await schema_conn.fetchval(_VALIDITY_SQL, "ux_u_c") is False
    await schema_conn.execute("DELETE FROM u WHERE id = 2")

    # nothing left to apply: the run still repairs its schema
    assert await runner.apply_for_agent_schema(ConnectionSession(schema_conn)) == 0

    assert await schema_conn.fetchval(_VALIDITY_SQL, "ux_u_c") is True
    with pytest.raises(asyncpg.UniqueViolationError):
        await schema_conn.execute("INSERT INTO u VALUES (3, 7)")
