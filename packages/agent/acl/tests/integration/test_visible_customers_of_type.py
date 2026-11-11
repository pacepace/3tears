"""the visible-customer set of one namespace type admits exactly what the per-row clause admits for that type.

``caller_visible_customers_query(scope_namespace_type=...)`` resolves the caller's grants to one set a listing of
that type pages by; :func:`caller_visible_customer_clause` is the per-row rule every listing of the type applies.
For each scope arm (all, type_customer on the type, type_customer on another type, namespace, no grant) the rows
the set admits are the rows the clause admits, on a real database. The plan test holds the set's read to the
membership and grant indexes on the acl schema.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.agent.acl import caller_visible_customer_clause, caller_visible_customers_query
from threetears.agent.acl.migrations import register as register_acl
from threetears.agent.acl.query_visibility import _caller_membership_predicate
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.plans import explain_statement, plan_uses_index
from threetears.core.testing.same_results import Seeded, assert_same_results

pytestmark = pytest.mark.integration

_TYPE = "datasource"


class _Store:
    """the migration runner's store over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *args: Any) -> str:
        return str(await self._conn.execute(sql, *args))

    async def query(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in await self._conn.fetch(sql, *args)]


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """a connection on a fresh schema the acl package built, with a table of listed rows one per customer."""
    schema = f"aclvis_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register_acl(runner)
        register_tools_platform(runner, depends_on=("agent_acl",))
        await runner.apply_for_platform_schema(_Store(connection))  # type: ignore[arg-type]
        await connection.execute("CREATE TABLE listed_things (thing_id uuid PRIMARY KEY, customer_id uuid)")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def _grant(conn: asyncpg.Connection, user: uuid.UUID, scope: str, **values: Any) -> None:
    """one group holding the caller, with one grant of ``scope`` (its other columns from ``values``)."""
    group, role = uuid.uuid4(), uuid.uuid4()
    await conn.execute(
        "INSERT INTO groups (row_scope, group_id, customer_id, name) VALUES ('platform', $1, NULL, $2)",
        group,
        f"g{group}",
    )
    await conn.execute(
        "INSERT INTO roles (role_id, customer_id, name, description, permissions) VALUES ($1, NULL, $2, 'r', '[]'::jsonb)",
        role,
        f"r{role}",
    )
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id) VALUES ($1, $2, 'user', $3)",
        uuid.uuid4(),
        group,
        user,
    )
    columns = ", ".join(values)
    marks = ", ".join(f"${index}" for index in range(5, 5 + len(values)))
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type"
        f"{', ' + columns if values else ''}) VALUES ('platform', $1, $2, $3, $4{', ' + marks if values else ''})",
        uuid.uuid4(),
        role,
        group,
        scope,
        *values.values(),
    )


async def _namespace(conn: asyncpg.Connection, customer: uuid.UUID) -> uuid.UUID:
    """a namespace of ``customer``'s."""
    namespace = uuid.uuid4()
    await conn.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, customer_id, namespace_type, name, schema_name,"
        " date_created, date_updated) VALUES ('customer', $1, $2, $3, $4, $5, now(), now())",
        namespace,
        customer,
        _TYPE,
        f"n{namespace.hex}",
        f"s{namespace.hex[:16]}",
    )
    return namespace


async def _admitted(conn: asyncpg.Connection, user: uuid.UUID) -> tuple[set[uuid.UUID], set[uuid.UUID]]:
    """the customers whose rows the typed set admits, and those the per-row clause admits."""
    sql, params = caller_visible_customers_query(user_id=user, scope_namespace_type=_TYPE)
    row = await conn.fetchrow(sql, *params)
    assert row is not None
    held = {r["customer_id"] for r in await conn.fetch("SELECT customer_id FROM listed_things")}
    by_set = held if row["is_all"] else held & set(row["customer_ids"])
    fragment, clause_params = caller_visible_customer_clause(
        user_id=user, customer_id_column="listed_things.customer_id", scope_namespace_type=_TYPE
    )
    by_clause = {
        r["customer_id"]
        for r in await conn.fetch(f"SELECT customer_id FROM listed_things WHERE {fragment}", *clause_params)
    }
    return by_set, by_clause


@pytest.mark.parametrize("arm", ["all", "type_customer_on_the_type", "type_customer_on_another", "namespace", "none"])
async def test_the_typed_set_admits_what_the_clause_admits(conn: asyncpg.Connection, arm: str) -> None:
    """Flips if, for any scope arm, the rows the typed visible-customer set admits are not the rows the per-row
    clause admits for the same type (a grant on another type adding its customer, a namespace grant missing)."""
    customers = [uuid.uuid4() for _ in range(3)]
    for customer in customers:
        await conn.execute("INSERT INTO listed_things VALUES ($1, $2)", uuid.uuid4(), customer)
    user = uuid.uuid4()
    if arm == "all":
        await _grant(conn, user, "all")
    elif arm == "type_customer_on_the_type":
        await _grant(conn, user, "type_customer", scope_customer_id=customers[0], scope_namespace_type=_TYPE)
    elif arm == "type_customer_on_another":
        await _grant(conn, user, "type_customer", scope_customer_id=customers[0], scope_namespace_type="tool")
    elif arm == "namespace":
        namespace = await _namespace(conn, customers[1])
        await _grant(conn, user, "namespace", scope_namespace_id=namespace, scope_namespace_type=_TYPE)
    by_set, by_clause = await _admitted(conn, user)
    assert by_set == by_clause
    expected = {
        "all": set(customers),
        "type_customer_on_the_type": {customers[0]},
        "type_customer_on_another": set(),
        "namespace": {customers[1]},
        "none": set(),
    }[arm]
    assert by_set == expected


async def test_the_typed_set_reads_the_callers_memberships_then_their_groups_grants(
    conn: asyncpg.Connection,
) -> None:
    """Flips if the typed set stops reading the caller's memberships through idx_group_members_reverse or their
    groups' grants through idx_role_assignments_group_scope."""
    await conn.execute(
        "INSERT INTO groups (row_scope, group_id, customer_id, name)"
        " SELECT 'platform', gen_random_uuid(), NULL, 'g' || g FROM generate_series(1, 500) g"
    )
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id)"
        " SELECT gen_random_uuid(), group_id, 'user', gen_random_uuid() FROM groups, generate_series(1, 4)"
    )
    await conn.execute(
        "INSERT INTO roles (role_id, customer_id, name, description, permissions)"
        " VALUES (gen_random_uuid(), NULL, 'r', 'r', '[]'::jsonb)"
    )
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type)"
        " SELECT 'platform', gen_random_uuid(), (SELECT role_id FROM roles), group_id, 'type_customer' FROM groups"
    )
    for table in ("groups", "group_members", "role_assignments"):
        await conn.execute(f"ANALYZE {table}")
    sql, params = caller_visible_customers_query(user_id=uuid.uuid4(), scope_namespace_type=_TYPE)
    plan = await explain_statement(conn, sql, *params)
    assert plan_uses_index(plan, "idx_group_members_reverse"), plan.plan
    assert plan_uses_index(plan, "idx_role_assignments_group_scope"), plan.plan


#: the groups the equivalence seeds hang memberships on, each holding one grant (some a second)
_GROUPS = 26


def _old_customers_query(typed: bool) -> str:
    """the visible-customer set as it was read before the keyed rewrite (7c7936732): every grant joined to its
    group's memberships, the caller matched on each, directly or through the nested hops, at this deployment's depth.
    """
    of_type = " AND ra.scope_namespace_type = $2" if typed else ""
    return f"""SELECT
        COALESCE(bool_or(ra.scope_type = 'all'), false) AS is_all,
        COALESCE(
            array_remove(
                array_agg(DISTINCT
                    CASE
                        WHEN ra.scope_type = 'type_customer'{of_type} THEN ra.scope_customer_id
                        WHEN ra.scope_type = 'namespace' THEN ns.customer_id
                    END
                ),
                NULL
            ),
            ARRAY[]::uuid[]
        ) AS customer_ids
      FROM role_assignments ra
      JOIN group_members gm ON gm.group_id = ra.group_id
      LEFT JOIN namespaces ns ON ns.namespace_id = ra.scope_namespace_id
     WHERE {_caller_membership_predicate(1)}"""


async def _granting_groups(conn: asyncpg.Connection) -> list[uuid.UUID]:
    """``_GROUPS`` groups, group k granting customer k's rows by arm k % 4 (type_customer on the type, a namespace
    of the customer's, type_customer on another type, type_customer on the type again), group 21 also ``all``."""
    role = uuid.uuid4()
    await conn.execute(
        "INSERT INTO roles (role_id, customer_id, name, description, permissions) VALUES ($1, NULL, 'r', 'r', '[]'::jsonb)",
        role,
    )
    groups = [uuid.uuid4() for _ in range(_GROUPS)]
    for k, group in enumerate(groups):
        await conn.execute(
            "INSERT INTO groups (row_scope, group_id, customer_id, name) VALUES ('platform', $1, NULL, $2)",
            group,
            f"g{k}",
        )
        customer = uuid.UUID(int=k + 1)
        grant: dict[str, Any] = {
            0: {"scope_type": "type_customer", "scope_customer_id": customer, "scope_namespace_type": _TYPE},
            1: {"scope_type": "namespace", "scope_namespace_id": await _namespace(conn, customer)},
            2: {"scope_type": "type_customer", "scope_customer_id": customer, "scope_namespace_type": "tool"},
            3: {"scope_type": "type_customer", "scope_customer_id": customer, "scope_namespace_type": _TYPE},
        }[k % 4]
        grants = [grant, {"scope_type": "all"}] if k == 21 else [grant]
        for values in grants:
            columns = ", ".join(values)
            marks = ", ".join(f"${index}" for index in range(4, 4 + len(values)))
            await conn.execute(
                f"INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, {columns})"
                f" VALUES ('platform', $1, $2, $3, {marks})",
                uuid.uuid4(),
                role,
                group,
                *values.values(),
            )
    return groups


async def _allow_group_members(conn: asyncpg.Connection) -> None:
    """let ``group_members`` hold a group: the hub's platform schema takes group-in-group members, acl v001's check
    names users and agents only."""
    await conn.execute("ALTER TABLE group_members DROP CONSTRAINT group_members_member_type_check")


@pytest.mark.parametrize("typed", [False, True])
async def test_the_keyed_set_gives_the_rows_the_joined_set_gave(conn: asyncpg.Connection, typed: bool) -> None:
    """Flips if the keyed read of the caller's groups (their memberships, then each group edge up, then those groups'
    grants) gives another is_all or customer set than the statement it replaced, for any seed: empty, one, many,
    NULLs, ties, a tenant boundary (tenant 1's members are another user), a page boundary. Members by index: the
    caller in group i; group i-1 (the caller's) in group i, a nested hop; group i-1 (itself reached by a hop) in
    group i, two hops, past the default nesting; another user in group i."""
    await _allow_group_members(conn)
    groups = await _granting_groups(conn)
    caller, other = uuid.uuid4(), uuid.uuid4()

    def member(seed: Seeded) -> dict[str, Any]:
        group = groups[seed.index % _GROUPS]
        below = groups[(seed.index - 1) % _GROUPS]
        kind, member_id = {
            0: ("user", caller),
            1: ("group", below),
            2: ("group", below),
            3: ("user", other),
        }[seed.index % 4]
        if seed.tenant == 1:
            kind, member_id = "user", other
        return {
            "id": uuid.uuid5(uuid.NAMESPACE_OID, f"{seed.case}/{seed.index}"),
            "group_id": group,
            "member_type": kind,
            "member_id": member_id,
            "customer_id": None if seed.null else uuid.UUID(int=seed.tie + 1),
        }

    sql, params = caller_visible_customers_query(user_id=caller, scope_namespace_type=_TYPE if typed else None)
    await assert_same_results(
        conn,
        table="group_members",
        row=member,
        old_sql=_old_customers_query(typed),
        params=params,
        helper=lambda c: c.fetch(sql, *params),
    )


async def test_the_set_reaches_a_grant_through_a_nested_group(conn: asyncpg.Connection) -> None:
    """Flips if a caller in group A, A a member of granting group B, does not get B's customer (the nested hop the
    rewrite reads from the caller's groups up)."""
    await _allow_group_members(conn)
    user, customer = uuid.uuid4(), uuid.uuid4()
    await _grant(conn, user, "all")
    await conn.execute("DELETE FROM role_assignments")
    granting, role = uuid.uuid4(), await conn.fetchval("SELECT role_id FROM roles")
    caller_group = await conn.fetchval("SELECT group_id FROM group_members WHERE member_id = $1", user)
    await conn.execute(
        "INSERT INTO groups (row_scope, group_id, customer_id, name) VALUES ('platform', $1, NULL, 'granting')",
        granting,
    )
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id) VALUES ($1, $2, 'group', $3)",
        uuid.uuid4(),
        granting,
        caller_group,
    )
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_customer_id,"
        " scope_namespace_type) VALUES ('platform', $1, $2, $3, 'type_customer', $4, $5)",
        uuid.uuid4(),
        role,
        granting,
        customer,
        _TYPE,
    )
    sql, params = caller_visible_customers_query(user_id=user, scope_namespace_type=_TYPE)
    row = await conn.fetchrow(sql, *params)
    assert row is not None
    assert (row["is_all"], list(row["customer_ids"])) == (False, [customer])
