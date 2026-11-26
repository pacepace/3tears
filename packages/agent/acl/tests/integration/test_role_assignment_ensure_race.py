"""``RoleAssignmentCollection.ensure_group_role_assignment`` under concurrency, on a real database.

The lookup and the insert are two statements, and every hub replica runs the ensure at once for
the same manifest -- the aibots hub found grants held twice. Given the natural-key unique index (acl v001's
``uq_role_assignments_namespace_grant``, the hub's v121), the ensure is race-safe: one row, one ``created``,
every caller answered the same id -- including a winner filed under the other partition.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest
import pytest_asyncio

from threetears.agent.acl import RoleAssignmentCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import next_epoch

from .access_schema import access_pool, access_schema, empty_tables

# one schema the acl migrations built, for the module's tests (each starts from empty tables)
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def _module_schema(db_container: str) -> AsyncIterator[str]:
    """a schema of the module's own holding the access tables as the acl migrations build them, natural-key index
    included; built once, as a build is slow on YugabyteDB. Its name.
    """
    async with access_schema(db_container, "ensure_race") as schema:
        yield schema


@pytest_asyncio.fixture(loop_scope="module")
async def indexed_pool(db_container: str, _module_schema: str) -> AsyncIterator[asyncpg.Pool]:
    """a pool on the module's schema, its access tables emptied for this test."""
    pool = await access_pool(db_container, _module_schema, max_size=12)
    try:
        await empty_tables(pool)
        yield pool
    finally:
        await pool.close()


async def _parents(pool: asyncpg.Pool) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """a grant's parents, each a row its foreign key names: a platform group, a platform role and a platform tool
    namespace (no customer, so the grant records none).

    :param pool: the pool
    :ptype pool: asyncpg.Pool
    :return: ``(group_id, role_id, namespace_id)``
    :rtype: tuple[uuid.UUID, uuid.UUID, uuid.UUID]
    """
    group_id, role_id, namespace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await pool.execute(
        "INSERT INTO groups (row_scope, group_id, customer_id, name, epoch) VALUES ('platform', $1, NULL, $2, $3)",
        group_id,
        f"g{group_id.hex}",
        next_epoch(),
    )
    await pool.execute(
        "INSERT INTO roles (role_id, name, description, permissions, epoch) VALUES ($1, $2, '', '{}'::jsonb, $3)",
        role_id,
        f"r{role_id.hex}",
        next_epoch(),
    )
    await pool.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, date_created,"
        " date_updated, epoch) VALUES ('platform', $1, $2, 'tool', NULL, now(), now(), $3)",
        namespace_id,
        f"tools.n{namespace_id.hex}",
        next_epoch(),
    )
    return group_id, role_id, namespace_id


def _collection(pool: asyncpg.Pool) -> RoleAssignmentCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope=f"ensure-race-{uuid.uuid4().hex[:6]}")
    return RoleAssignmentCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))


async def _race(pool: asyncpg.Pool, callers: int) -> tuple[list[tuple[uuid.UUID, bool]], uuid.UUID, uuid.UUID]:
    """``callers`` concurrent ensures of one grant, each through its own collection, as replicas run them."""
    group_id, role_id, namespace_id = await _parents(pool)
    results = await asyncio.gather(
        *(
            _collection(pool).ensure_group_role_assignment(
                group_id=group_id,
                role_id=role_id,
                scope_type="namespace",
                scope_id=namespace_id,
                managed_by="bootstrap",
            )
            for _ in range(callers)
        )
    )
    return list(results), group_id, namespace_id


async def test_concurrent_ensures_leave_one_row_given_the_natural_key_index(indexed_pool: asyncpg.Pool) -> None:
    results, group_id, namespace_id = await _race(indexed_pool, 12)

    rows = await indexed_pool.fetch(
        "SELECT assignment_id FROM role_assignments WHERE group_id = $1 AND scope_namespace_id = $2",
        group_id,
        namespace_id,
    )
    assert len(rows) == 1
    assert {assignment_id for assignment_id, _ in results} == {rows[0]["assignment_id"]}
    assert sum(1 for _, created in results if created) == 1


async def test_a_lost_race_finds_the_winner_in_the_other_partition(indexed_pool: asyncpg.Pool) -> None:
    """the lookup reads one partition; the race's winner is found wherever the grant is filed."""
    group_id, role_id, namespace_id = await _parents(indexed_pool)
    winner = uuid.uuid4()
    await indexed_pool.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, "
        "scope_namespace_id, managed_by, epoch) VALUES ('platform', $1, $2, $3, 'namespace', $4, 'bootstrap', $5)",
        winner,
        role_id,
        group_id,
        namespace_id,
        next_epoch(),
    )

    assignment_id, created = await _collection(indexed_pool).ensure_group_role_assignment(
        group_id=group_id, role_id=role_id, scope_type="namespace", scope_id=namespace_id, managed_by="bootstrap"
    )

    assert (assignment_id, created) == (winner, False)
    assert await indexed_pool.fetchval("SELECT count(*) FROM role_assignments") == 1
