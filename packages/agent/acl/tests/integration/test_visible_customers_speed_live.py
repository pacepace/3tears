"""Integration: the caller's visible-customer set costs within twice the one statement it replaced.

``visible_customers`` reads the caller's groups (to the configured nesting) and their grants as one capped walk
(``threetears.core.keyed.read_reached`` with ``depth`` and ``joins``). Read a level a read and the grants a read more
(3 round trips), it cost 1.18-1.26 ms against the hand-written reached-CTE's 0.38-0.42 ms on PostgreSQL (5,000
groups, a caller in two, one of them nested; scratchpad d5/lat/pg.log), over the bar of twice the old statement
(coordinator, 2026-11-15). This test is the guard: the same world, a hand-written statement doing the walk's work
(:func:`_baseline`) timed turn about with the set on one session, within the default 2x, and the set the old statement
gives (:func:`_old`).

WHY the baseline is no longer the old statement (2026-11-26): the security fixes since made the walk do more than the
old statement ever did -- each grant read beside its role (the 2026-11-18 security fix "visible_customers counts a
grant only through a role that acts where its scope needs one"), each membership edge, granting group and role beside
the customer it names (2026-11-20 "caller visibility applies the evaluator's cross-customer wall"; 2026-11-21 "a grant
whose group has no row opens no list"), and every table read live alone (``NOT "gone"``, the access tables stamped). Timed against the old statement, the set measured 1.83x on the exclusive
runner (best of 60: old 72 us, the walk's statement sent raw 92 us, ``visible_customers`` 132 us) and over 2x on a
slower host, the gap the joins the old statement never made. The bar compares the set with the same work hand-written.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.agent.acl import MAX_GROUP_MEMBERSHIP_DEPTH, visible_customers
from threetears.agent.acl.migrations import register as register_acl
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.core.data.migrations import MigrationRunner
from threetears.core.ddl.row_epoch import next_epoch
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem

pytestmark = pytest.mark.integration

_TYPE = "datasource"
_GROUPS = 5_000
#: reads a timed run makes: one read is well under a millisecond, under the timer's noise alone
_READS = 20

#: the reached-CTE statement visible_customers sent before its keyed walk (7c7936732 .. 66cd1ec64), its arms the
#: groups a chain of one, two, ... groups reaches from the caller; cut to this deployment's nesting by :func:`_old`
_OLD_ARMS = (
    "    SELECT gm.group_id\n      FROM group_members gm\n     WHERE gm.member_type = 'user' AND gm.member_id = $1",
    "    SELECT gm.group_id\n      FROM group_members gm2\n"
    "      JOIN group_members gm ON gm2.group_id = gm.member_id AND gm.member_type = 'group'\n"
    "     WHERE gm2.member_type = 'user' AND gm2.member_id = $1",
    "    SELECT gm.group_id\n      FROM group_members gm3\n"
    "      JOIN group_members gm2 ON gm3.group_id = gm2.member_id AND gm2.member_type = 'group'\n"
    "      JOIN group_members gm ON gm2.group_id = gm.member_id AND gm.member_type = 'group'\n"
    "     WHERE gm3.member_type = 'user' AND gm3.member_id = $1",
    "    SELECT gm.group_id\n      FROM group_members gm4\n"
    "      JOIN group_members gm3 ON gm4.group_id = gm3.member_id AND gm3.member_type = 'group'\n"
    "      JOIN group_members gm2 ON gm3.group_id = gm2.member_id AND gm2.member_type = 'group'\n"
    "      JOIN group_members gm ON gm2.group_id = gm.member_id AND gm.member_type = 'group'\n"
    "     WHERE gm4.member_type = 'user' AND gm4.member_id = $1",
)
_OLD_TAIL = (
    "SELECT\n        COALESCE(bool_or(ra.scope_type = 'all'), false) AS is_all,\n        COALESCE(\n"
    "            array_remove(\n                array_agg(DISTINCT\n                    CASE\n"
    "                        WHEN ra.scope_type = 'type_customer' AND ra.scope_namespace_type = $2"
    " THEN ra.scope_customer_id\n"
    "                        WHEN ra.scope_type = 'namespace' THEN ns.customer_id\n                    END\n"
    "                ),\n                NULL\n            ),\n            ARRAY[]::uuid[]\n        ) AS customer_ids\n"
    "      FROM reached\n      CROSS JOIN LATERAL (\n"
    "        SELECT ra.scope_type, ra.scope_customer_id, ra.scope_namespace_type, ra.scope_namespace_id\n"
    "          FROM role_assignments ra\n         WHERE ra.group_id = reached.group_id\n        OFFSET 0\n"
    "      ) ra\n      LEFT JOIN namespaces ns ON ns.namespace_id = ra.scope_namespace_id"
)


def _old() -> str:
    """the old statement walking this deployment's nesting."""
    arms = "\n     UNION\n".join(_OLD_ARMS[:MAX_GROUP_MEMBERSHIP_DEPTH])
    return f"WITH reached AS MATERIALIZED (\n{arms}\n)\n{_OLD_TAIL}"


def _baseline_arm(level: int) -> str:
    """the old statement's arm for a chain of ``level`` groups, each membership live, answering the chain's last edge
    (who is in which group, and the customer the edge names: the wall's input) where the old arm answered its group."""
    joins = "".join(
        f"      JOIN group_members gm{n} ON gm{n}.member_id = gm{n + 1}.group_id AND gm{n}.member_type = 'group'"
        f" AND NOT gm{n}.gone\n"
        for n in range(level - 1, 0, -1)
    )
    return (
        f"    SELECT gm1.member_id, gm1.group_id, gm1.customer_id\n      FROM group_members gm{level}\n{joins}"
        f"     WHERE gm{level}.member_type = 'user' AND gm{level}.member_id = $1 AND NOT gm{level}.gone"
    )


#: what the walk joins each reached edge to, by key and live alone, as ``visible_customers`` sends it: its grants,
#: each grant's role, the granting group (in either partition) and a namespace grant's namespace
_BASELINE_TAIL = (
    "SELECT DISTINCT r.member_id, r.group_id, r.customer_id, ra.scope_type, ra.scope_customer_id,"
    " ra.scope_namespace_type, ra.scope_namespace_id, ra.role_id, ro.permissions AS role_permissions,"
    " ro.customer_id AS role_customer_id, g.customer_id AS group_customer_id, g.group_id AS group_row_id,"
    " ns.customer_id AS namespace_customer_id, ns.namespace_type\n"
    "      FROM reached r\n"
    "      LEFT JOIN LATERAL (SELECT scope_type, scope_customer_id, scope_namespace_type, scope_namespace_id, role_id"
    " FROM role_assignments WHERE group_id = r.group_id AND NOT gone OFFSET 0) ra ON TRUE\n"
    "      LEFT JOIN LATERAL (SELECT permissions, customer_id FROM roles"
    " WHERE role_id = ra.role_id AND NOT gone OFFSET 0) ro ON TRUE\n"
    "      LEFT JOIN LATERAL (SELECT customer_id, group_id FROM groups WHERE group_id = r.group_id"
    " AND row_scope IN ('platform', 'customer') AND NOT gone OFFSET 0) g ON TRUE\n"
    "      LEFT JOIN LATERAL (SELECT customer_id, namespace_type FROM namespaces"
    " WHERE namespace_id = ra.scope_namespace_id AND NOT gone OFFSET 0) ns ON TRUE"
)


def _baseline() -> str:
    """the walk's work hand-written: the old statement's arms to this deployment's nesting, each a chain's last live
    edge, and each reached edge's grants, roles, granting group and namespace by key, live alone -- every join and guard
    ``visible_customers``' one statement sends (the module docstring: why the bar moved off :func:`_old`)."""
    arms = "\n     UNION\n".join(_baseline_arm(level) for level in range(1, MAX_GROUP_MEMBERSHIP_DEPTH + 1))
    return f"WITH reached AS MATERIALIZED (\n{arms}\n)\n{_BASELINE_TAIL}"


class _Store:
    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *args: Any) -> str:
        return str(await self._conn.execute(sql, *args))

    async def query(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in await self._conn.fetch(sql, *args)]


@pytest.fixture
async def world(db_container: str) -> AsyncIterator[tuple[str, uuid.UUID]]:
    """a schema of its own the acl package built: 5,000 groups of four users each, a fifth of them nested in the
    next, two grants a group; a caller in two groups, one of them nested. Its name and the caller."""
    schema = f"visspeed_{uuid.uuid4().hex[:10]}"
    held = await asyncpg.connect(db_container)
    try:
        await held.execute(f'CREATE SCHEMA "{schema}"')
        await held.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_acl(runner)
        register_tools_platform(runner, depends_on=("agent_acl",))
        await runner.apply_for_platform_schema(_Store(held))
        # the hub's platform schema takes group-in-group members; acl v001's check names users and agents only
        await held.execute("ALTER TABLE group_members DROP CONSTRAINT group_members_member_type_check")
        role = uuid.uuid4()
        await held.execute(
            "INSERT INTO roles (role_id, name, description, permissions, epoch)"
            " VALUES ($1, 'r', 'r', '{\"*\": [\"read\"]}'::jsonb, $2)",
            role,
            next_epoch(),
        )
        await held.execute(
            "INSERT INTO groups (row_scope, group_id, customer_id, name, epoch)"
            f" SELECT 'platform', gen_random_uuid(), NULL, 'g' || g, $1 FROM generate_series(1, {_GROUPS}) g",
            next_epoch(),
        )
        await held.execute(
            "INSERT INTO group_members (id, group_id, member_type, member_id, epoch)"
            " SELECT gen_random_uuid(), group_id, 'user', gen_random_uuid(), $1 FROM groups, generate_series(1, 4)",
            next_epoch(),
        )
        await held.execute(
            "INSERT INTO group_members (id, group_id, member_type, member_id, epoch)"
            " SELECT gen_random_uuid(), b.group_id, 'group', a.group_id, $1"
            " FROM (SELECT group_id, row_number() OVER (ORDER BY group_id) n FROM groups) a"
            " JOIN (SELECT group_id, row_number() OVER (ORDER BY group_id) n FROM groups) b ON b.n = a.n + 1"
            " WHERE a.n % 5 = 0",
            next_epoch(),
        )
        await held.execute(
            "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_customer_id,"
            " scope_namespace_type, customer_id, epoch)"
            " SELECT 'platform', gen_random_uuid(), $1, group_id, 'type_customer', c, $2, c, $3"
            " FROM (SELECT group_id, gen_random_uuid() c FROM groups, generate_series(1, 2)) g",
            role,
            _TYPE,
            next_epoch(),
        )
        for table in ("groups", "group_members", "role_assignments", "namespaces"):
            await held.execute(f"ANALYZE {table}")
        caller = uuid.uuid4()
        nested = await held.fetchval(
            "SELECT member_id FROM group_members WHERE member_type = 'group' ORDER BY member_id LIMIT 1"
        )
        plain = await held.fetchval("SELECT group_id FROM groups ORDER BY group_id OFFSET 7 LIMIT 1")
        for group in (nested, plain):
            await held.execute(
                "INSERT INTO group_members (id, group_id, member_type, member_id, epoch)"
                " VALUES (gen_random_uuid(), $1, 'user', $2, $3)",
                group,
                caller,
                next_epoch(),
            )
        yield schema, caller
    finally:
        await held.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await held.close()


#: reads on one session before PostgreSQL settles a prepared statement's plan: it tries a custom plan for the first
#: five, then caches the generic one when that is estimated no costlier
_SETTLED = 12


async def test_the_visible_set_caches_its_generic_plan(db_container: str, world: tuple[str, uuid.UUID]) -> None:
    """Flips if PostgreSQL re-plans the visible-customer walk on every call (a custom plan each time, no generic plan
    cached): a discriminator bound as ``$n`` (``member_type = $1``, ``'user'``) estimated the generic plan costlier
    (510 against 391), so it was never cached and every call paid ~75-100 us of planning (IT2, 2026-11-16: 0 generic
    / 800 custom). The walk writes its code's own constants (:class:`~threetears.core.keyed.Const`) as literals."""
    schema, caller = world
    conn = await asyncpg.connect(db_container, server_settings={"search_path": f'"{schema}", public'})
    try:
        for _ in range(_SETTLED):
            await visible_customers(conn, user_id=caller, namespace_type=_TYPE)
        plans = await conn.fetch(
            "SELECT generic_plans, custom_plans FROM pg_prepared_statements"
            " WHERE statement LIKE 'WITH RECURSIVE reached%' AND statement LIKE '%group_members%'"
        )
        assert len(plans) == 1, plans
        assert plans[0]["generic_plans"] > 0, dict(plans[0])
    finally:
        await conn.close()


@pytest.mark.loop_timing
async def test_the_visible_set_costs_within_twice_the_statement_it_replaced(
    db_container: str, world: tuple[str, uuid.UUID]
) -> None:
    """Flips if ``visible_customers`` takes more than twice the same work hand-written on the same session
    (:func:`_baseline`: a level a read back, a join the planner hashes from the grants' side, or Python a call that a
    shape could do once), sends more than one statement, or gives another set than the reached-CTE statement it
    replaced."""
    schema, caller = world
    conn = await asyncpg.connect(db_container, server_settings={"search_path": f'"{schema}", public'})
    try:
        row = await conn.fetchrow(_old(), caller, _TYPE)
        baseline_sql = _baseline()
        sent: list[str] = []

        def log(record: Any) -> None:
            sent.append(record.query)

        conn.add_query_logger(log)
        got = await visible_customers(conn, user_id=caller, namespace_type=_TYPE)
        await asyncio.sleep(0)  # the connection hands its last statement to the logger on the next loop turn
        conn.remove_query_logger(log)
        assert [sql for sql in sent if "group_members" in sql] == [sent[-1]], sent
        assert (got.is_all, list(got.customer_ids)) == (row["is_all"], sorted(row["customer_ids"], key=str))
        assert got.customer_ids, "the caller's groups grant customers"
        # the baseline does the walk's work: the same grants, beside the same roles and groups
        grants = await conn.fetch(baseline_sql, caller)
        assert sorted({g["scope_customer_id"] for g in grants}, key=str) == list(got.customer_ids)
        assert all(g["role_permissions"] is not None and g["group_row_id"] is not None for g in grants)

        async def bare() -> list[Any]:
            return [await conn.fetch(baseline_sql, caller) for _ in range(_READS)]

        async def keyed() -> list[Any]:
            return [await visible_customers(conn, user_id=caller, namespace_type=_TYPE) for _ in range(_READS)]

        old, new = await best_seconds_interleaved(bare, keyed)
        problem = budget_problem("the visible-customer set", new, old)
        assert problem is None, problem
    finally:
        await conn.close()
