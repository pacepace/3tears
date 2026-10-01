"""``RoleAssignmentCollection.ensure_group_role_assignment`` under concurrency, on a real database.

The lookup and the insert are two statements, and every hub replica runs the ensure at once for
the same manifest -- the aibots hub found grants held twice. Given the natural-key unique index the
deploying application declares (the hub's v121), the ensure is race-safe: one row, one ``created``,
every caller answered the same id -- including a winner filed under the other partition.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.agent.acl import RoleAssignmentCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

pytestmark = pytest.mark.integration

#: the hub's ``role_assignments`` shape, reduced to the columns the ensure reads and writes.
_ROLE_ASSIGNMENTS_DDL = """
CREATE TABLE role_assignments (
    row_scope varchar(8) NOT NULL,
    assignment_id uuid NOT NULL,
    role_id uuid NOT NULL,
    group_id uuid NOT NULL,
    scope_type varchar(16) NOT NULL,
    scope_namespace_id uuid,
    scope_namespace_type varchar(255),
    scope_customer_id uuid,
    scope_namespace_name varchar(255),
    granted_by uuid,
    date_granted timestamptz NOT NULL DEFAULT now(),
    managed_by varchar(32) NOT NULL DEFAULT 'manual',
    source varchar(64),
    CONSTRAINT role_assignments_pkey PRIMARY KEY (row_scope, assignment_id)
)
"""

#: the natural-key index the deploying application declares (the aibots hub's v121).
_NATURAL_KEY_INDEX = """
CREATE UNIQUE INDEX idx_role_assignments_namespace_natural_key
    ON role_assignments (group_id, role_id, scope_namespace_id, managed_by)
 WHERE scope_type = 'namespace'
"""


async def _pool(db_container: str, *, natural_key: bool) -> asyncpg.Pool:
    from threetears.core.collections import init_connection

    pool: asyncpg.Pool = await asyncpg.create_pool(db_container, min_size=1, max_size=12, init=init_connection)
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS role_assignments")
        await conn.execute(_ROLE_ASSIGNMENTS_DDL)
        await conn.execute("CREATE UNIQUE INDEX role_assignments_id_unique ON role_assignments (assignment_id)")
        if natural_key:
            await conn.execute(_NATURAL_KEY_INDEX)
    return pool


@pytest.fixture
async def indexed_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    pool = await _pool(db_container, natural_key=True)
    try:
        yield pool
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS role_assignments")
        await pool.close()


def _collection(pool: asyncpg.Pool) -> RoleAssignmentCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope=f"ensure-race-{uuid.uuid4().hex[:6]}")
    return RoleAssignmentCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))


async def _race(pool: asyncpg.Pool, callers: int) -> tuple[list[tuple[uuid.UUID, bool]], uuid.UUID, uuid.UUID]:
    """``callers`` concurrent ensures of one grant, each through its own collection, as replicas run them."""
    group_id, role_id, namespace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
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
    group_id, role_id, namespace_id, winner = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await indexed_pool.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, "
        "scope_namespace_id, managed_by) VALUES ('platform', $1, $2, $3, 'namespace', $4, 'bootstrap')",
        winner,
        role_id,
        group_id,
        namespace_id,
    )

    assignment_id, created = await _collection(indexed_pool).ensure_group_role_assignment(
        group_id=group_id, role_id=role_id, scope_type="namespace", scope_id=namespace_id, managed_by="bootstrap"
    )

    assert (assignment_id, created) == (winner, False)
    assert await indexed_pool.fetchval("SELECT count(*) FROM role_assignments") == 1
