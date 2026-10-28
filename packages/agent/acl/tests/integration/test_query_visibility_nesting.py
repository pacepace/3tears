"""the caller-visibility clause reaches exactly as deep as the evaluator, on a real database.

group-in-group membership (up to MAX_GROUP_MEMBERSHIP_DEPTH groups) resolves at
authorization time through the evaluator's parent walk, so the visibility SQL
must admit the same rows: a caller whose group is NESTED inside the granting
group sees what the grant covers, and a caller in an unrelated group still
sees nothing. the unit tests pin the fragment's text; these prove the
fragment's SEMANTICS against real Postgres, both directions, and prove the
evaluator -- reading the same rows through the production membership loader
-- reaches the same verdict at every chain length up to one past the cap, at
every depth a deployment may configure, so
a future edit that drops a hop or widens past the cap fails as a measurement
rather than a diff.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.agent.acl import (
    MAX_CONFIGURABLE_GROUP_MEMBERSHIP_DEPTH,
    MAX_GROUP_MEMBERSHIP_DEPTH,
    MAX_GROUP_MEMBERSHIP_DEPTH_ENV,
    caller_visible_customer_clause,
)

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


async def _seed_chain(conn: asyncpg.Connection, *, groups: int, user_id: uuid.UUID, customer_id: uuid.UUID) -> None:
    """seed a chain of ``groups`` groups, each a member of the next, the caller in the bottom one.

    the grant sits on the top group only, so the caller reaches it through ``groups - 1``
    group edges.
    """
    chain = [uuid.uuid4() for _ in range(groups)]
    await conn.execute("INSERT INTO listed_things (thing_id, customer_id) VALUES ($1, $2)", uuid.uuid4(), customer_id)
    await _seed_grant(conn, group_id=chain[-1], customer_id=customer_id)
    for child, parent in zip(chain, chain[1:], strict=False):
        await _seed_member(conn, group_id=parent, member_type="group", member_id=child)
    await _seed_member(conn, group_id=chain[0], member_type="user", member_id=user_id)


async def test_the_walk_stops_at_the_cap(pg_pool: asyncpg.Pool) -> None:
    """a chain one group past MAX_GROUP_MEMBERSHIP_DEPTH does not resolve.

    the evaluator walks no deeper than the cap, so the visibility clause must
    not admit the chain either -- the two surfaces stay in agreement at the
    cap, not just below it.
    """
    customer_id = uuid.uuid4()
    user_id = uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await _seed_chain(conn, groups=MAX_GROUP_MEMBERSHIP_DEPTH + 1, user_id=user_id, customer_id=customer_id)
    assert await _visible_count(pg_pool, user_id) == 0


_PROBE = Path(__file__).with_name("depth_agreement_probe.py")


@pytest.mark.parametrize("depth", range(1, MAX_CONFIGURABLE_GROUP_MEMBERSHIP_DEPTH + 1))
async def test_the_evaluator_and_the_clause_agree_at_every_configured_depth(
    pg_pool: asyncpg.Pool, db_container: str, depth: int
) -> None:
    """configured to any depth, the visibility clause admits a chain exactly when the evaluator grants through it.

    the depth is read once at import from ``THREETEARS_ACL_MAX_GROUP_MEMBERSHIP_DEPTH``, the way a
    deployment sets it, so each depth runs in a fresh interpreter (:mod:`depth_agreement_probe`):
    for every chain from one group to one past the depth, the evaluator -- reading the same
    ``group_members`` rows through the production :class:`CollectionMembershipLoader`, its
    batched parent read included -- and the clause must agree, and both must stop exactly at
    the configured depth.
    """
    _ = pg_pool  # the fixture creates the tables the probe seeds
    env = {**os.environ, MAX_GROUP_MEMBERSHIP_DEPTH_ENV: str(depth)}
    probe = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_PROBE),
        db_container,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await probe.communicate()
    assert probe.returncode == 0, err.decode()
    report = json.loads(out.decode().strip().splitlines()[-1])
    assert report["depth"] == depth
    verdicts = {int(groups): verdict for groups, verdict in report["chains"].items()}
    assert set(verdicts) == set(range(1, depth + 2))
    for groups, verdict in verdicts.items():
        expected = groups <= depth
        assert verdict == {"granted": expected, "visible": expected}, (groups, verdict)
