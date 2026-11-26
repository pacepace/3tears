"""the caller-visibility clause reaches exactly as deep as the evaluator, on a real database.

group-in-group membership (up to MAX_GROUP_MEMBERSHIP_DEPTH groups) resolves at
authorization time through the evaluator's parent walk, so the visibility SQL
must admit the same rows: a caller whose group is NESTED inside the granting
group sees what the grant covers, and a caller in an unrelated group still
sees nothing. the unit tests pin the fragment's text; these prove the
fragment's SEMANTICS against real Postgres, both directions, and prove the
evaluator -- reading the same rows through the production membership loader
-- reaches the same verdict at every chain length up to one past the cap, at
every nesting a deployment may configure, so
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
import pytest_asyncio

from threetears.agent.acl import (
    MAX_GROUP_MEMBERSHIP_DEPTH,
    caller_visible_customer_clause,
)
from threetears.agent.acl.types import (
    MAX_CONFIGURABLE_GROUP_NESTING_LEVELS,
    MAX_GROUP_NESTING_LEVELS_ENV,
    MIN_CONFIGURABLE_GROUP_NESTING_LEVELS,
    RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV,
)
from threetears.core.ddl.row_epoch import next_epoch

from .access_schema import access_pool, access_schema, empty_tables

# one schema the acl migrations built, for the module's tests (each starts from empty tables)
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

#: the one role every grant here binds: it carries an action on the listed type, so the role half of
#: the clause admits it and these tests measure the membership walk alone. the probe seeds the same id.
READER_ROLE_ID = uuid.UUID("5f0c2d4e-7a1b-4c3d-9e8f-0a1b2c3d4e5f")


_LISTED_DDL = "CREATE TABLE listed_things (thing_id uuid PRIMARY KEY, customer_id uuid)"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def _module_schema(db_container: str) -> AsyncIterator[str]:
    """a schema of the module's own: the access tables as the acl migrations build them, group-in-group memberships
    allowed (the hub's platform schema takes them; acl v001's check names users and agents only), and the listed
    table. Its name.
    """
    async with access_schema(db_container, "visnest") as schema:
        pool = await access_pool(db_container, schema, max_size=1)
        try:
            await pool.execute("ALTER TABLE group_members DROP CONSTRAINT group_members_member_type_check")
            await pool.execute(_LISTED_DDL)
        finally:
            await pool.close()
        yield schema


@pytest_asyncio.fixture(loop_scope="module")
async def pg_pool(db_container: str, _module_schema: str) -> AsyncIterator[asyncpg.Pool]:
    """per-test pool over the module's schema, its tables emptied, the reader role seeded."""
    pool = await access_pool(db_container, _module_schema, max_size=2)
    try:
        await empty_tables(pool, "listed_things", "role_assignments", "group_members", "groups", "roles", "namespaces")
        await pool.execute(
            "INSERT INTO roles (role_id, name, description, permissions, epoch) VALUES ($1, 'reader', '', $2::text::jsonb, $3)",
            READER_ROLE_ID,
            json.dumps({"thing": ["read"]}),
            next_epoch(),
        )
        yield pool
    finally:
        await pool.close()


async def _seed_group(conn: asyncpg.Connection, group_id: uuid.UUID) -> None:
    """seed a platform group's row, once: a membership's group and a grant's group each name one (their foreign keys)."""
    await conn.execute(
        "INSERT INTO groups (row_scope, group_id, customer_id, name, epoch) VALUES ('platform', $1, NULL, $2, $3)"
        " ON CONFLICT (group_id) DO NOTHING",
        group_id,
        f"g{group_id.hex}",
        next_epoch(),
    )


async def _seed_grant(conn: asyncpg.Connection, *, group_id: uuid.UUID, customer_id: uuid.UUID) -> None:
    """seed one type_customer assignment of the reader role held by ``group_id`` over ``customer_id``, and the group's
    row (a platform group): the clause counts a grant only through its group's row, as the evaluator resolves it."""
    await _seed_group(conn, group_id)
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_customer_id,"
        " scope_namespace_type, epoch) VALUES ('platform', $1, $2, $3, 'type_customer', $4, 'thing', $5)",
        uuid.uuid4(),
        READER_ROLE_ID,
        group_id,
        customer_id,
        next_epoch(),
    )


async def _seed_member(
    conn: asyncpg.Connection, *, group_id: uuid.UUID, member_type: str, member_id: uuid.UUID
) -> None:
    """seed one membership row, its group's row first."""
    await _seed_group(conn, group_id)
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id, epoch) VALUES ($1, $2, $3, $4, $5)",
        uuid.uuid4(),
        group_id,
        member_type,
        member_id,
        next_epoch(),
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


@pytest.mark.parametrize(
    "levels", range(MIN_CONFIGURABLE_GROUP_NESTING_LEVELS, MAX_CONFIGURABLE_GROUP_NESTING_LEVELS + 1)
)
async def test_the_evaluator_and_the_clause_agree_at_every_configured_nesting(
    pg_pool: asyncpg.Pool, db_container: str, _module_schema: str, levels: int
) -> None:
    """configured to any nesting, the visibility clause admits a chain exactly when the evaluator grants through it.

    the nesting is read once at import from ``THREETEARS_ACL_MAX_GROUP_NESTING_LEVELS``, in levels
    (the person plus each group), the way a deployment sets it, so each setting runs in a fresh
    interpreter (:mod:`depth_agreement_probe`): for every chain from one group to one past the
    setting, the evaluator -- reading the same ``group_members`` rows through the production
    :class:`CollectionMembershipLoader`, its batched parent read included -- and the clause must
    agree, and both must stop exactly at ``levels - 1`` groups.
    """
    _ = pg_pool  # the fixture creates the tables the probe seeds
    depth = levels - 1
    env = {key: value for key, value in os.environ.items() if key != RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV}
    env[MAX_GROUP_NESTING_LEVELS_ENV] = str(levels)
    probe = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_PROBE),
        db_container,
        _module_schema,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await probe.communicate()
    assert probe.returncode == 0, err.decode()
    report = json.loads(out.decode().strip().splitlines()[-1])
    assert report["levels"] == levels
    assert report["depth"] == depth
    verdicts = {int(groups): verdict for groups, verdict in report["chains"].items()}
    assert set(verdicts) == set(range(1, depth + 2))
    for groups, verdict in verdicts.items():
        expected = groups <= depth
        assert verdict == {"granted": expected, "visible": expected}, (groups, verdict)
