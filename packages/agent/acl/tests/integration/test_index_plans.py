"""integration: each agent-acl index serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. Each seeds its table past the size where
the planner would rather scan, runs the purpose's own statement through ``EXPLAIN ANALYZE``
(:func:`threetears.core.testing.plans.explain_statement`), and asks the plan whether it found its rows through the
index by an index condition (:func:`~threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.agent.acl import collections as acl
from threetears.agent.acl.migrations import register as register_acl
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.plans import explain_statement, plan_uses_index

pytestmark = pytest.mark.integration


class _Store:
    """The DataStore surface the runner drives, over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        return await self._conn.execute(sql, *params)

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._conn.fetch(sql, *params)]


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A connection on a fresh schema the acl package (and the tools package that ALTERs it) built."""
    schema = f"aclplan_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register_acl(runner)
        register_tools_platform(runner, depends_on=("agent_acl",))
        await runner.apply_for_platform_schema(_Store(connection))  # type: ignore[arg-type]
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await connection.close()


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


async def _seed(conn: asyncpg.Connection, table: str, rows: int = _ROWS, **values: str) -> None:
    """seed ``rows`` rows into ``table``, then ANALYZE it, so the planner prices it at size.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :param table: the table
    :ptype table: str
    :param rows: how many
    :ptype rows: int
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
        f'INSERT INTO "{table}" ({", ".join(names)}) SELECT {", ".join(exprs)} FROM generate_series(1, {rows}) g'
    )
    await conn.execute(f'ANALYZE "{table}"')


async def test_members_reverse(conn: asyncpg.Connection) -> None:
    """Flips if the memberships naming a member stop reading idx_group_members_reverse."""
    await _seed(conn, "groups", 50, row_scope="'platform'", customer_id="NULL", name="'g' || g")
    groups = [row["group_id"] for row in await conn.fetch("SELECT group_id FROM groups")]
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id)"
        " SELECT gen_random_uuid(), ($1::uuid[])[1 + g % 50], CASE WHEN g % 2 = 0 THEN 'user' ELSE 'agent' END,"
        " gen_random_uuid() FROM generate_series(1, 5000) g",
        groups,
    )
    await conn.execute("ANALYZE group_members")
    plan = await explain_statement(conn, acl.MEMBERS_NAMING_SQL, "user", [uuid.uuid4()])
    assert plan_uses_index(plan, "idx_group_members_reverse"), plan.plan


async def test_assignments_group_scope(conn: asyncpg.Connection) -> None:
    """Flips if every assignment a set of groups holds stops reading idx_role_assignments_group_scope."""
    await _seed(conn, "groups", 500, row_scope="'platform'", customer_id="NULL", name="'g' || g")
    await _seed(conn, "roles", 1, name="'r'", description="'r'", permissions="'[]'::jsonb")
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type)"
        " SELECT 'platform', gen_random_uuid(), (SELECT role_id FROM roles), group_id, 'type_customer'"
        " FROM groups, generate_series(1, 10)"
    )
    await conn.execute("ANALYZE role_assignments")
    plan = await explain_statement(conn, acl.ASSIGNMENTS_OF_GROUPS_SQL, [uuid.uuid4()])
    assert plan_uses_index(plan, "idx_role_assignments_group_scope"), plan.plan


async def test_namespaces_name(conn: asyncpg.Connection) -> None:
    """Flips if a namespace by name stops reading idx_namespaces_name."""
    await _seed(
        conn, "namespaces", row_scope="'platform'", customer_id="NULL", namespace_type="'tool'", name="'n' || g"
    )
    plan = await explain_statement(conn, acl.NAMESPACE_BY_NAME_SQL, "n17")
    assert plan_uses_index(plan, "idx_namespaces_name"), plan.plan


async def test_groups_by_name(conn: asyncpg.Connection) -> None:
    """Flips if a group by name, platform or a customer's, stops reading its scope's unique name index."""
    await _seed(conn, "groups", row_scope="'platform'", customer_id="NULL", name="'p' || g")
    await _seed(conn, "groups", row_scope="'customer'", customer_id="gen_random_uuid()", name="'c' || g")
    platform = await explain_statement(conn, acl.GROUP_BY_PLATFORM_NAME_SQL, "p17")
    assert plan_uses_index(platform, "uq_groups_platform_name"), platform.plan
    customer = await explain_statement(conn, acl.GROUP_BY_CUSTOMER_NAME_SQL, uuid.uuid4(), "c17")
    assert plan_uses_index(customer, "uq_groups_customer_name"), customer.plan


async def test_roles_platform_name(conn: asyncpg.Connection) -> None:
    """Flips if a platform role by name (every built-in's lookup) stops reading uq_roles_platform_name."""
    await _seed(conn, "roles", name="'r' || g", description="'r'", permissions="'[]'::jsonb", customer_id="NULL")
    plan = await explain_statement(conn, acl.ROLE_BY_PLATFORM_NAME_SQL, "r17")
    assert plan_uses_index(plan, "uq_roles_platform_name"), plan.plan
