"""the access tables as the acl package's migrations build them, on a schema of a test's own.

The integration tests that read or write ``groups``, ``group_members``, ``roles``, ``role_assignments`` and
``namespaces`` hold them to the shape production runs: agent-acl's migrations (stamped: ``epoch``, ``gone``, the
restore bucket; ``namespaces`` keyed by ``namespace_id``) and agent-tools platform's after them, never a hand-written
copy, which drifted from the stamped tables and left every read of ``gone`` failing. A row a test seeds by hand names
its ``epoch`` (:func:`~threetears.core.ddl.row_epoch.next_epoch`, as a write of its own stamps it): the stamped
tables' epoch is ``NOT NULL`` with no default.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from threetears.agent.acl.migrations import register as register_acl
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.core.collections import init_connection
from threetears.core.data.migrations import MigrationRunner

__all__ = [
    "ACCESS_TABLE_NAMES",
    "access_pool",
    "access_schema",
    "empty_tables",
    "loosen_parents",
    "owner_id_key",
    "set_search_path",
]

#: the access tables, children first
ACCESS_TABLE_NAMES = ("role_assignments", "roles", "group_members", "groups", "namespaces")


class _Store:
    """the migration runner's store over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *args: Any) -> str:
        return str(await self._conn.execute(sql, *args))

    async def query(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in await self._conn.fetch(sql, *args)]


def set_search_path(schema: str) -> str:
    """the statement binding a session to ``schema``.

    :param schema: the schema
    :ptype schema: str
    :return: the statement
    :rtype: str
    """
    return f'SET search_path TO "{schema}"'


@asynccontextmanager
async def access_schema(db_container: str, prefix: str = "acl") -> AsyncIterator[str]:
    """a fresh schema the acl and agent-tools platform migrations built, dropped after.

    :param db_container: the test database
    :ptype db_container: str
    :param prefix: the schema name's prefix
    :ptype prefix: str
    :return: the schema's name
    :rtype: AsyncIterator[str]
    """
    schema = f"{prefix}_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(set_search_path(schema))
        runner = MigrationRunner()
        register_acl(runner)
        register_tools_platform(runner, depends_on=("agent_acl",))
        await runner.apply_for_platform_schema(_Store(conn))
    finally:
        await conn.close()
    try:
        yield schema
    finally:
        conn = await asyncpg.connect(db_container)
        try:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await conn.close()


async def access_pool(db_container: str, schema: str, *, max_size: int = 4) -> asyncpg.Pool:
    """a pool whose sessions are bound to ``schema``, its connections set up as the collections' are.

    :param db_container: the test database
    :ptype db_container: str
    :param schema: the schema
    :ptype schema: str
    :param max_size: the pool's size
    :ptype max_size: int
    :return: the pool
    :rtype: asyncpg.Pool
    """
    pool: asyncpg.Pool = await asyncpg.create_pool(
        db_container,
        min_size=1,
        max_size=max_size,
        init=init_connection,
        server_settings={"search_path": f'"{schema}"'},
    )
    return pool


async def empty_tables(conn: Any, *tables: str) -> None:
    """every row of ``tables`` (the access tables when none are named) deleted: a test sharing a module's schema starts
    from empty tables, as one with a schema of its own does.

    :param conn: a connection or pool on the schema
    :ptype conn: Any
    :param tables: the TRUSTED table names
    :ptype tables: str
    :return: nothing
    :rtype: None
    """
    for table in tables or ACCESS_TABLE_NAMES:
        await conn.execute(f"DELETE FROM {table}")  # noqa: S608 - a test's own table


async def owner_id_key(conn: Any) -> None:
    """the hub's shape, added to the acl package's ``namespaces`` (whose names are not unique and whose owner carries
    no key: acl v001, v011): a name unique among live rows, and the ``owner_namespace_id`` foreign key onto
    ``namespace_id``, where a namespace's owner is a row named by its id (hub v199).

    :param conn: a connection or pool on the schema
    :ptype conn: Any
    :return: nothing
    :rtype: None
    """
    await conn.execute("CREATE UNIQUE INDEX uq_namespaces_name ON namespaces (name) WHERE NOT gone")
    await conn.execute(
        "ALTER TABLE namespaces ADD CONSTRAINT namespaces_owner_namespace_id_fkey"
        " FOREIGN KEY (owner_namespace_id) REFERENCES namespaces (namespace_id)"
    )


#: each foreign key from a membership or a grant onto the row it names, and the check that a member is a user or an
#: agent (acl v001; the hub's platform schema takes a group too)
_PARENT_CONSTRAINTS = (
    ("group_members", "group_members_group_fk"),
    ("group_members", "group_members_member_type_check"),
    ("role_assignments", "role_assignments_group_fk"),
    ("role_assignments", "role_assignments_role_fk"),
    ("role_assignments", "role_assignments_scope_namespace_fk"),
)


async def loosen_parents(conn: Any) -> None:
    """the foreign keys from memberships and grants onto their groups, roles and namespaces dropped, and a member
    allowed to be a group: for the tests that write memberships and grants of rows they never create (a cache's
    eviction, a generation's advance), which the constraints are not what they prove.

    :param conn: a connection or pool on the schema
    :ptype conn: Any
    :return: nothing
    :rtype: None
    """
    for table, constraint in _PARENT_CONSTRAINTS:
        await conn.execute(f"ALTER TABLE {table} DROP CONSTRAINT {constraint}")
