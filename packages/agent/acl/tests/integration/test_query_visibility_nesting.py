"""the caller-visibility clause reaches through one nesting level, on a real database.

group-in-group membership (depth 2) resolves at authorization time through
the evaluator's parent walk, so the visibility SQL must admit the same rows:
a caller whose group is NESTED inside the granting group sees what the grant
covers, and a caller in an unrelated group still sees nothing. the unit tests
pin the fragment's text; these prove the fragment's SEMANTICS against real
Postgres, both directions, so a future edit that drops the hop -- or widens
it past one membership edge -- fails as a measurement rather than a diff.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.agent.acl import caller_visible_customer_clause

pytestmark = pytest.mark.integration


_DDL = (
    """
    CREATE TABLE role_assignments (
        assignment_id uuid PRIMARY KEY,
        group_id uuid NOT NULL,
        scope_type varchar(16) NOT NULL,
        scope_namespace_id uuid,
        scope_namespace_type varchar(255),
        scope_customer_id uuid
    )
    """,
    """
    CREATE TABLE group_members (
        id uuid PRIMARY KEY,
        group_id uuid NOT NULL,
        member_type varchar(10) NOT NULL,
        member_id uuid NOT NULL,
        customer_id uuid
    )
    """,
    """
    CREATE TABLE namespaces (
        namespace_id uuid PRIMARY KEY,
        customer_id uuid
    )
    """,
    """
    CREATE TABLE listed_things (
        thing_id uuid PRIMARY KEY,
        customer_id uuid
    )
    """,
)


@pytest.fixture
async def pg_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """per-test pool over the four tables the visibility clause touches."""
    from threetears.core.collections import init_connection

    pool: asyncpg.Pool = await asyncpg.create_pool(
        db_container,
        min_size=1,
        max_size=2,
        init=init_connection,
    )
    try:
        async with pool.acquire() as conn:
            for table in ("listed_things", "namespaces", "group_members", "role_assignments"):
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
            for ddl in _DDL:
                await conn.execute(ddl)
        yield pool
    finally:
        await pool.close()


async def _seed_grant(conn: asyncpg.Connection, *, group_id: uuid.UUID, customer_id: uuid.UUID) -> None:
    """seed one type_customer assignment held by ``group_id`` over ``customer_id``."""
    await conn.execute(
        "INSERT INTO role_assignments (assignment_id, group_id, scope_type, scope_customer_id, scope_namespace_type) "
        "VALUES ($1, $2, 'type_customer', $3, 'thing')",
        uuid.uuid4(),
        group_id,
        customer_id,
    )


async def _seed_member(
    conn: asyncpg.Connection, *, group_id: uuid.UUID, member_type: str, member_id: uuid.UUID
) -> None:
    """seed one membership row."""
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id) VALUES ($1, $2, $3, $4)",
        uuid.uuid4(),
        group_id,
        member_type,
        member_id,
    )


async def _visible_count(pool: asyncpg.Pool, user_id: uuid.UUID) -> int:
    """count listed_things rows the caller may see under the clause."""
    fragment, params = caller_visible_customer_clause(
        user_id=user_id,
        customer_id_column="listed_things.customer_id",
        scope_namespace_type="thing",
        param_offset=1,
    )
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT count(*) AS n FROM listed_things WHERE {fragment}", *params)
    return int(row["n"])


async def test_direct_membership_still_admits(pg_pool: asyncpg.Pool) -> None:
    """the pre-nesting behaviour is unchanged: a direct member sees the rows."""
    customer_id = uuid.uuid4()
    user_id = uuid.uuid4()
    group_id = uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO listed_things (thing_id, customer_id) VALUES ($1, $2)", uuid.uuid4(), customer_id
        )
        await _seed_grant(conn, group_id=group_id, customer_id=customer_id)
        await _seed_member(conn, group_id=group_id, member_type="user", member_id=user_id)
    assert await _visible_count(pg_pool, user_id) == 1


async def test_nested_membership_admits_through_one_level(pg_pool: asyncpg.Pool) -> None:
    """a member of a CHILD group sees rows GRANTED to the parent group."""
    customer_id = uuid.uuid4()
    user_id = uuid.uuid4()
    parent_group = uuid.uuid4()
    child_group = uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO listed_things (thing_id, customer_id) VALUES ($1, $2)", uuid.uuid4(), customer_id
        )
        await _seed_grant(conn, group_id=parent_group, customer_id=customer_id)
        await _seed_member(conn, group_id=parent_group, member_type="group", member_id=child_group)
        await _seed_member(conn, group_id=child_group, member_type="user", member_id=user_id)
    assert await _visible_count(pg_pool, user_id) == 1


async def test_an_unnested_group_admits_nothing(pg_pool: asyncpg.Pool) -> None:
    """negative control: membership in an unrelated group reaches no grant."""
    customer_id = uuid.uuid4()
    user_id = uuid.uuid4()
    granted_group = uuid.uuid4()
    unrelated_group = uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO listed_things (thing_id, customer_id) VALUES ($1, $2)", uuid.uuid4(), customer_id
        )
        await _seed_grant(conn, group_id=granted_group, customer_id=customer_id)
        await _seed_member(conn, group_id=unrelated_group, member_type="user", member_id=user_id)
    assert await _visible_count(pg_pool, user_id) == 0


async def test_the_hop_is_one_level_deep_only(pg_pool: asyncpg.Pool) -> None:
    """a two-edge chain does not resolve, matching MAX_GROUP_MEMBERSHIP_DEPTH=2.

    grandchild -> child -> granted parent is DEEPER than the evaluator walks,
    so the visibility clause must not admit it either -- the two surfaces
    stay in agreement at the cap, not just above it.
    """
    customer_id = uuid.uuid4()
    user_id = uuid.uuid4()
    parent_group = uuid.uuid4()
    child_group = uuid.uuid4()
    grandchild_group = uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO listed_things (thing_id, customer_id) VALUES ($1, $2)", uuid.uuid4(), customer_id
        )
        await _seed_grant(conn, group_id=parent_group, customer_id=customer_id)
        await _seed_member(conn, group_id=parent_group, member_type="group", member_id=child_group)
        await _seed_member(conn, group_id=child_group, member_type="group", member_id=grandchild_group)
        await _seed_member(conn, group_id=grandchild_group, member_type="user", member_id=user_id)
    assert await _visible_count(pg_pool, user_id) == 0
