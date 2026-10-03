"""integration: the agent-acl package builds the five rbac tables, and they hold the platform's rules.

On a fresh schema, with ``agent_tools_platform`` registered after it (it
ALTERs ``namespaces``): the build applies, a second build applies nothing,
and each uniqueness the evaluator and the idempotent writers rely on refuses
the row that would break it -- including the grant race, through the
package's own natural-key index rather than one a test wrote.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.agent.acl import RoleAssignmentCollection
from threetears.agent.acl.migrations import register as register_acl
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

pytestmark = pytest.mark.integration


class _Store:
    """The DataStore surface the runner drives, over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        return await self._conn.execute(sql, *params)

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._conn.fetch(sql, *params)]


def _runner() -> MigrationRunner:
    runner = MigrationRunner()
    register_acl(runner)
    register_tools_platform(runner, depends_on=("agent_acl",))
    return runner


@pytest.fixture
async def acl_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """A pool on a fresh schema the two packages have built."""
    schema = f"acl_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        assert await _runner().apply_for_platform_schema(_Store(conn)) == 4  # type: ignore[arg-type]
    finally:
        await conn.close()
    pool: asyncpg.Pool = await asyncpg.create_pool(
        db_container,
        min_size=1,
        max_size=12,
        init=init_connection,
        server_settings={"search_path": f'"{schema}"'},
    )
    try:
        yield pool
    finally:
        await pool.close()
        conn = await asyncpg.connect(db_container)
        try:
            await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
        finally:
            await conn.close()


async def _group(pool: asyncpg.Pool, name: str, customer: uuid.UUID | None) -> uuid.UUID:
    group_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO groups (row_scope, group_id, customer_id, name) VALUES ($1, $2, $3, $4)",
        "platform" if customer is None else "customer",
        group_id,
        customer,
        name,
    )
    return group_id


async def _role(pool: asyncpg.Pool, name: str, customer: uuid.UUID | None = None) -> uuid.UUID:
    role_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO roles (role_id, name, description, permissions, customer_id) VALUES ($1, $2, 'r', '[]', $3)",
        role_id,
        name,
        customer,
    )
    return role_id


async def _namespace(pool: asyncpg.Pool, name: str, **over: Any) -> uuid.UUID:
    row = {"row_scope": "platform", "namespace_type": "tool", "customer_id": None, "schema_name": None, **over}
    namespace_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, schema_name, "
        "owner_namespace, date_created, date_updated) VALUES ($1, $2, $3, $4, $5, $6, $7, now(), now())",
        row["row_scope"],
        namespace_id,
        name,
        row["namespace_type"],
        row["customer_id"],
        row["schema_name"],
        over.get("owner_namespace"),
    )
    return namespace_id


async def test_a_second_build_applies_nothing(acl_pool: asyncpg.Pool) -> None:
    async with acl_pool.acquire() as conn:
        assert await _runner().apply_for_platform_schema(_Store(conn)) == 0  # type: ignore[arg-type]
        columns = {
            r["column_name"]
            for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'namespaces' "
                "AND table_schema = current_schema()"
            )
        }
    assert {"owner_namespace", "tool_eligible", "face_rest"} <= columns


async def test_names_are_unique_within_their_scope(acl_pool: asyncpg.Pool) -> None:
    one, two = uuid.uuid4(), uuid.uuid4()
    await _group(acl_pool, "editors", None)
    await _group(acl_pool, "editors", one)
    await _group(acl_pool, "editors", two)  # another customer's group may share the name
    for customer in (None, one):
        with pytest.raises(asyncpg.UniqueViolationError):
            await _group(acl_pool, "editors", customer)
    await _role(acl_pool, "reader")
    await _role(acl_pool, "reader", one)
    with pytest.raises(asyncpg.UniqueViolationError):
        await _role(acl_pool, "reader")
    # two servers may expose a tool of one name: the row is addressed by its id
    await _namespace(acl_pool, "tools/a")
    await _namespace(acl_pool, "tools/a")


async def test_namespace_rows_keep_the_platform_rules(acl_pool: asyncpg.Pool) -> None:
    customer = uuid.uuid4()
    await _namespace(
        acl_pool, "ws/a", namespace_type="workspace", row_scope="customer", customer_id=customer, schema_name="shared"
    )
    await _namespace(
        acl_pool, "ws/b", namespace_type="workspace", row_scope="customer", customer_id=customer, schema_name="shared"
    )
    await _namespace(acl_pool, "tools/x", schema_name="only_mine")
    with pytest.raises(asyncpg.UniqueViolationError):
        await _namespace(acl_pool, "tools/y", schema_name="only_mine")
    with pytest.raises(asyncpg.CheckViolationError):
        await _namespace(acl_pool, "agents/z", namespace_type="agent")  # a platform row is a platform type
    # an owner is a name, not a row: an agent's own namespace exists only where a registry made it
    await _namespace(acl_pool, "tools/child", owner_namespace="agents.nobody-registered")


async def test_an_unstamped_grant_is_manual(acl_pool: asyncpg.Pool) -> None:
    group_id, role_id = await _group(acl_pool, "g", None), await _role(acl_pool, "r")
    await acl_pool.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type) "
        "VALUES ('platform', $1, $2, $3, 'all')",
        uuid.uuid4(),
        role_id,
        group_id,
    )
    assert await acl_pool.fetchval("SELECT managed_by FROM role_assignments") == "manual"


async def test_concurrent_ensures_of_one_grant_leave_one_row(acl_pool: asyncpg.Pool) -> None:
    """The hub found grants held twice; the package's natural-key index makes the ensure race-safe."""
    group_id, role_id = await _group(acl_pool, "g", None), await _role(acl_pool, "r")
    namespace_id = await _namespace(acl_pool, "tools/raced")

    def _collection() -> RoleAssignmentCollection:
        registry = CollectionRegistry()
        registry.configure(l3_pool=acl_pool, kv_key_scope=f"acl-mig-{uuid.uuid4().hex[:6]}")
        return RoleAssignmentCollection(
            registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        )

    results = await asyncio.gather(
        *(
            _collection().ensure_group_role_assignment(
                group_id=group_id,
                role_id=role_id,
                scope_type="namespace",
                scope_id=namespace_id,
                managed_by="bootstrap",
            )
            for _ in range(12)
        )
    )

    assert await acl_pool.fetchval("SELECT count(*) FROM role_assignments") == 1
    assert len({assignment_id for assignment_id, _ in results}) == 1
    assert sum(1 for _, created in results if created) == 1
