"""integration: a grant records its customer (acl v005), on PostgreSQL and on YugabyteDB.

v005 adds ``role_assignments.customer_id`` to a schema that already holds grants, nullable and unchecked (the expand
step: a pod of the previous release still writes grants without it while a deployment rolls).
``RoleAssignmentCollection.backfill_customers`` fills the older grants a page at a time by key (a namespace grant's
namespace read by its key). The writers fill it from then on: the ensure reads the namespace's customer, ``create()``
derives the rest. A customer's listing will pin the column (the contract step), so a platform grant (NULL) is never
in a customer's page.
"""

from __future__ import annotations

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
        status: str = await self._conn.execute(sql, *params)
        return status

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._conn.fetch(sql, *params)]


def _runner() -> MigrationRunner:
    runner = MigrationRunner()
    register_acl(runner)
    register_tools_platform(runner, depends_on=("agent_acl",))
    return runner


@pytest.fixture
async def schema_url(db_container: str) -> AsyncIterator[tuple[str, str]]:
    """a fresh schema built to acl v004 (agent-tools platform beside it): the shape before v005."""
    schema = f"aclcust_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await _runner().apply_for_platform_schema(_Store(conn), target=4)
    finally:
        await conn.close()
    try:
        yield db_container, schema
    finally:
        conn = await asyncpg.connect(db_container)
        try:
            await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
        finally:
            await conn.close()


@pytest.fixture
async def pool(schema_url: tuple[str, str]) -> AsyncIterator[asyncpg.Pool]:
    url, schema = schema_url
    created: asyncpg.Pool = await asyncpg.create_pool(
        url, min_size=1, max_size=4, init=init_connection, server_settings={"search_path": f'"{schema}"'}
    )
    try:
        yield created
    finally:
        await created.close()


async def _migrate(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        applied: int = await _runner().apply_for_platform_schema(_Store(conn))
    return applied


async def _group(pool: asyncpg.Pool, customer: uuid.UUID | None) -> uuid.UUID:
    group_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO groups (row_scope, group_id, customer_id, name) VALUES ($1, $2, $3, $4)",
        "platform" if customer is None else "customer",
        group_id,
        customer,
        f"g-{group_id.hex}",
    )
    return group_id


async def _role(pool: asyncpg.Pool) -> uuid.UUID:
    role_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO roles (role_id, name, description, permissions) VALUES ($1, $2, 'r', '[]')",
        role_id,
        f"r-{role_id.hex}",
    )
    return role_id


async def _namespace(pool: asyncpg.Pool, customer: uuid.UUID | None) -> uuid.UUID:
    namespace_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, date_created, "
        "date_updated) VALUES ($1, $2, $3, $4, $5, now(), now())",
        "platform" if customer is None else "customer",
        namespace_id,
        f"ns.{namespace_id.hex}",
        "tool" if customer is None else "agent",
        customer,
    )
    return namespace_id


async def _grant(
    pool: asyncpg.Pool, group_id: uuid.UUID, role_id: uuid.UUID, scope_type: str, **scope: Any
) -> uuid.UUID:
    """a grant written by hand without its customer, as one made before v005 is."""
    assignment_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_namespace_id, "
        "scope_namespace_type, scope_customer_id) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        scope.get("row_scope", "customer"),
        assignment_id,
        role_id,
        group_id,
        scope_type,
        scope.get("namespace_id"),
        scope.get("namespace_type"),
        scope.get("customer_id"),
    )
    return assignment_id


def _collection(pool: asyncpg.Pool) -> RoleAssignmentCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope=f"acl-cust-{uuid.uuid4().hex[:6]}")
    return RoleAssignmentCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))


async def _customers(pool: asyncpg.Pool) -> dict[uuid.UUID, uuid.UUID | None]:
    rows = await pool.fetch("SELECT assignment_id, customer_id FROM role_assignments")
    return {row["assignment_id"]: row["customer_id"] for row in rows}


async def test_the_backfill_fills_every_existing_grants_customer_across_pages(pool: asyncpg.Pool) -> None:
    """Flips if v005 changes a grant, or the backfill misses a page or gives a grant another scope's customer (a
    platform grant in a customer's page, or a customer's grant hidden from its admin)."""
    customer = uuid.uuid4()
    group = await _group(pool, customer)
    platform_group = await _group(pool, None)
    customer_namespace = await _namespace(pool, customer)
    platform_namespace = await _namespace(pool, None)
    expected: dict[uuid.UUID, uuid.UUID | None] = {}
    for _ in range(3):  # three of each shape: more than one page of two
        role = await _role(pool)
        expected[await _grant(pool, group, role, "namespace", namespace_id=customer_namespace)] = customer
        expected[await _grant(pool, platform_group, role, "namespace", namespace_id=platform_namespace)] = None
        expected[await _grant(pool, group, role, "type_customer", namespace_type="agent", customer_id=customer)] = (
            customer
        )
        expected[await _grant(pool, platform_group, role, "all", row_scope="platform")] = None

    assert await _migrate(pool) == 1
    assert await _customers(pool) == dict.fromkeys(expected)
    collection = _collection(pool)
    cursor: str | None = None
    while True:
        page = await collection.backfill_customers(limit=2, cursor=cursor)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert await _customers(pool) == expected
    assert await _migrate(pool) == 0


async def test_a_grant_without_its_customer_is_still_written_during_the_rollout(pool: asyncpg.Pool) -> None:
    """Flips if v005 refuses a grant a pod of the previous release writes without the column (the expand step holds
    no CHECK: one would fail that pod's writes until the rollout ends)."""
    await _migrate(pool)
    customer = uuid.uuid4()
    group, role = await _group(pool, customer), await _role(pool)
    typed = await _grant(pool, group, role, "type_customer", namespace_type="agent", customer_id=customer)
    assert (await _customers(pool))[typed] is None


async def test_every_writer_records_the_grants_customer(pool: asyncpg.Pool) -> None:
    """Flips if the ensure (a customer's namespace, a platform namespace, an 'all' grant) or create() + save (a type
    grant) writes a grant without its customer."""
    await _migrate(pool)
    customer = uuid.uuid4()
    group, platform_group, role = await _group(pool, customer), await _group(pool, None), await _role(pool)
    customer_namespace, platform_namespace = await _namespace(pool, customer), await _namespace(pool, None)
    collection = _collection(pool)
    on_customer, _ = await collection.ensure_group_role_assignment(
        group_id=group, role_id=role, scope_type="namespace", scope_id=customer_namespace
    )
    on_platform, _ = await collection.ensure_group_role_assignment(
        group_id=platform_group, role_id=role, scope_type="namespace", scope_id=platform_namespace
    )
    everywhere, _ = await collection.ensure_group_role_assignment(
        group_id=platform_group, role_id=role, scope_type="all", scope_id=None
    )
    typed = collection.create(
        {
            "assignment_id": uuid.uuid4(),
            "role_id": role,
            "group_id": group,
            "scope_type": "type_customer",
            "scope_namespace_type": "agent",
            "scope_customer_id": customer,
        }
    )
    await collection.save_entity(typed)

    assert await _customers(pool) == {
        on_customer: customer,
        on_platform: None,
        everywhere: None,
        typed.assignment_id: customer,
    }


async def test_the_collections_backfill_walks_every_page(pool: asyncpg.Pool) -> None:
    """Flips if RoleAssignmentCollection.backfill_customers stops short or rewrites a grant already right."""
    await _migrate(pool)
    customer = uuid.uuid4()
    group, namespace = await _group(pool, customer), await _namespace(pool, customer)
    granted = [await _grant(pool, group, await _role(pool), "namespace", namespace_id=namespace) for _ in range(5)]
    collection = _collection(pool)
    written: list[uuid.UUID] = []
    cursor: str | None = None
    pages = 0
    while True:
        page = await collection.backfill_customers(limit=2, cursor=cursor)
        written.extend(assignment_id for _, assignment_id, _ in page.written)
        pages += 1
        cursor = page.next_cursor
        if cursor is None:
            break
    assert sorted(written) == sorted(granted)
    assert pages == 3
    assert await _customers(pool) == dict.fromkeys(granted, customer)
    again = await collection.backfill_customers(limit=10)
    assert again.written == ()
