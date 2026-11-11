"""integration: each agent-acl index serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. Each seeds its table past the size where
the planner would rather scan, runs the purpose's own statement through ``EXPLAIN ANALYZE``
(:func:`threetears.core.testing.plans.explain_statement`), and asks the plan whether it found its rows through the
index by an index condition (:func:`~threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

import asyncpg
import pytest

from threetears.agent.acl import collections as acl
from threetears.agent.acl.migrations import register as register_acl
from threetears.agent.acl.types import ScopeType
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.plans import PlanRecorder, explain_statement, plan_uses_index

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


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A connection on a fresh schema the acl package (and the tools package that ALTERs it) built."""
    schema = f"aclplan_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register_acl(runner)
        register_tools_platform(runner, depends_on=("agent_acl",))
        await runner.apply_for_platform_schema(_Store(connection))
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await connection.close()


#: rows seeded per table: past the size where the planner would rather scan than use an index
_ROWS = 5000

#: a value for a NOT NULL column with no default, by its type, from the series row ``g``
_BY_TYPE: dict[str, str] = {
    "uuid": "gen_random_uuid()",
    "text": "'v' || g",
    "character varying": "'v' || g",
    "timestamp with time zone": "now() - g * interval '1 second'",
    "timestamp without time zone": "now() - g * interval '1 second'",
    "integer": "g",
    "bigint": "g",
    "smallint": "1",
    "double precision": "g",
    "real": "g",
    "numeric": "g",
    "boolean": "false",
    "jsonb": "'{}'::jsonb",
    "json": "'{}'::json",
    "ARRAY": "'{}'",
}


async def _seed(conn: asyncpg.Connection, table: str, rows: int = _ROWS, **values: str) -> None:
    """seed ``rows`` rows into ``table``, then ANALYZE it, so the planner prices it at size.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :param table: the table
    :ptype table: str
    :param rows: how many
    :ptype rows: int
    :param values: a SQL expression of the series row ``g`` per column; every other NOT NULL column with no
        default gets one by its type
    :ptype values: str
    :return: nothing
    :rtype: None
    """
    columns = await conn.fetch(
        "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name = $1",
        table,
    )
    names: list[str] = []
    exprs: list[str] = []
    for column in columns:
        name = column["column_name"]
        if name in values:
            names.append(name)
            exprs.append(values[name])
        elif column["is_nullable"] == "NO" and column["column_default"] is None:
            names.append(name)
            exprs.append(_BY_TYPE[column["data_type"]])
    await conn.execute(
        f'INSERT INTO "{table}" ({", ".join(names)}) SELECT {", ".join(exprs)} FROM generate_series(1, {rows}) g'
    )
    await conn.execute(f'ANALYZE "{table}"')


async def test_members_reverse(conn: asyncpg.Connection) -> None:
    """Flips if the memberships naming a member stop reading idx_group_members_reverse."""
    await _seed(conn, "groups", 50, row_scope="'platform'", customer_id="NULL", name="'g' || g")
    groups = [row["group_id"] for row in await conn.fetch("SELECT group_id FROM groups")]
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id)"
        " SELECT gen_random_uuid(), ($1::uuid[])[1 + g % 50], CASE WHEN g % 2 = 0 THEN 'user' ELSE 'agent' END,"
        " gen_random_uuid() FROM generate_series(1, 5000) g",
        groups,
    )
    await conn.execute("ANALYZE group_members")
    plan = await explain_statement(conn, acl.MEMBERS_NAMING_SQL, "user", [uuid.uuid4()])
    assert plan_uses_index(plan, "idx_group_members_reverse"), plan.plan


async def test_assignments_group_scope(conn: asyncpg.Connection) -> None:
    """Flips if every assignment a set of groups holds stops reading idx_role_assignments_group_scope."""
    await _seed(conn, "groups", 500, row_scope="'platform'", customer_id="NULL", name="'g' || g")
    await _seed(conn, "roles", 1, name="'r'", description="'r'", permissions="'[]'::jsonb")
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type)"
        " SELECT 'platform', gen_random_uuid(), (SELECT role_id FROM roles), group_id, 'type_customer'"
        " FROM groups, generate_series(1, 10)"
    )
    await conn.execute("ANALYZE role_assignments")
    plan = await explain_statement(conn, acl.ASSIGNMENTS_OF_GROUPS_SQL, [uuid.uuid4()])
    assert plan_uses_index(plan, "idx_role_assignments_group_scope"), plan.plan


async def test_namespaces_name(conn: asyncpg.Connection) -> None:
    """Flips if a namespace by name stops reading idx_namespaces_name."""
    await _seed(
        conn, "namespaces", row_scope="'platform'", customer_id="NULL", namespace_type="'tool'", name="'n' || g"
    )
    plan = await explain_statement(conn, acl.NAMESPACE_BY_NAME_SQL, "n17")
    assert plan_uses_index(plan, "idx_namespaces_name"), plan.plan


async def test_groups_by_name(conn: asyncpg.Connection) -> None:
    """Flips if a group by name, platform or a customer's, stops reading its scope's unique name index."""
    await _seed(conn, "groups", row_scope="'platform'", customer_id="NULL", name="'p' || g")
    await _seed(conn, "groups", row_scope="'customer'", customer_id="gen_random_uuid()", name="'c' || g")
    platform = await explain_statement(conn, acl.GROUP_BY_PLATFORM_NAME_SQL, "p17")
    assert plan_uses_index(platform, "uq_groups_platform_name"), platform.plan
    customer = await explain_statement(conn, acl.GROUP_BY_CUSTOMER_NAME_SQL, uuid.uuid4(), "c17")
    assert plan_uses_index(customer, "uq_groups_customer_name"), customer.plan


async def test_roles_platform_name(conn: asyncpg.Connection) -> None:
    """Flips if a platform role by name (every built-in's lookup) stops reading uq_roles_platform_name."""
    await _seed(conn, "roles", name="'r' || g", description="'r'", permissions="'[]'::jsonb", customer_id="NULL")
    plan = await explain_statement(conn, acl.ROLE_BY_PLATFORM_NAME_SQL, "r17")
    assert plan_uses_index(plan, "uq_roles_platform_name"), plan.plan


def _recorded[T](cls: type[T], conn: asyncpg.Connection) -> tuple[T, PlanRecorder]:
    """a collection reading through a plan recorder on ``conn``, and the recorder.

    :param cls: the collection class
    :ptype cls: type[T]
    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :return: the collection and the recorder its reads are planned on
    :rtype: tuple[T, PlanRecorder]
    """
    recorder = PlanRecorder(conn)
    collection = cls(registry=CollectionRegistry(), config=DefaultCoreConfig())  # type: ignore[call-arg]
    collection.l3_pool = recorder  # type: ignore[attr-defined]
    return collection, recorder


def _scans_partial(node: Mapping[str, Any], index: str) -> bool:
    """whether a plan node, or one under it, scans the partial index ``index`` (its predicate the read's bound).

    :param node: a plan node
    :ptype node: Mapping[str, Any]
    :param index: the index
    :ptype index: str
    :return: whether it does
    :rtype: bool
    """
    return node.get("Index Name") == index or any(_scans_partial(child, index) for child in node.get("Plans", ()))


async def _seed_assignments(conn: asyncpg.Connection) -> list[uuid.UUID]:
    """5000 assignments over 500 groups, 50 roles and 500 namespaces, every scope type of acl's own set; the
    namespaces' ids.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :return: the namespaces' ids
    :rtype: list[uuid.UUID]
    """
    await _seed(conn, "groups", 500, row_scope="'platform'", customer_id="NULL", name="'g' || g")
    await _seed(conn, "roles", 50, name="'r' || g", description="'r'", permissions="'[]'::jsonb")
    await _seed(
        conn,
        "namespaces",
        500,
        row_scope="'customer'",
        customer_id="gen_random_uuid()",
        namespace_type="'tool'",
        name="'n' || g",
    )
    namespaces = [row["namespace_id"] for row in await conn.fetch("SELECT namespace_id FROM namespaces")]
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_namespace_id,"
        " scope_namespace_type, scope_customer_id, managed_by)"
        " SELECT 'customer', gen_random_uuid(), (array_agg_r)[1 + s % 50], (array_agg_g)[1 + s % 500],"
        " (ARRAY['namespace', 'type_customer', 'all'])[1 + s % 3],"
        " CASE WHEN s % 3 = 0 THEN ($1::uuid[])[1 + s % 500] END,"
        " CASE WHEN s % 3 = 1 THEN 't' || (s % 20) END,"
        " CASE WHEN s % 3 = 1 THEN gen_random_uuid() END, 'm' || s"
        " FROM (SELECT array_agg(role_id) AS array_agg_r FROM roles) r,"
        " (SELECT array_agg(group_id) AS array_agg_g FROM groups) g, generate_series(1, 5000) s",
        namespaces,
    )
    await conn.execute("ANALYZE role_assignments")
    return namespaces


async def test_members_of_groups(conn: asyncpg.Connection) -> None:
    """Flips if every member of a set of groups stops reading idx_group_members_group_id."""
    await _seed(conn, "groups", 500, row_scope="'platform'", customer_id="NULL", name="'g' || g")
    groups = [row["group_id"] for row in await conn.fetch("SELECT group_id FROM groups")]
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id)"
        " SELECT gen_random_uuid(), ($1::uuid[])[1 + g % 500], 'user', gen_random_uuid()"
        " FROM generate_series(1, 5000) g",
        groups,
    )
    await conn.execute("ANALYZE group_members")
    members, recorder = _recorded(acl.GroupMemberCollection, conn)
    found = await members.members_of_groups([groups[0], groups[1]])
    plan = recorder.plans[0]
    # PostgreSQL reads the group index; YugabyteDB reads the primary key (group_id HASH, id), the rows' own storage,
    # by the same condition. Either way the read finds exactly the groups' rows by group
    keyed = plan_uses_index(plan, "idx_group_members_group_id") or plan_uses_index(plan, "group_members_pkey")
    assert keyed and plan.rows_scanned == len(found) == 20, json.dumps(plan.plan)[:2500]


async def test_assignments_scope_namespace(conn: asyncpg.Connection) -> None:
    """Flips if a namespace's namespace-scope grants stop reading idx_role_assignments_scope_namespace."""
    namespaces = await _seed_assignments(conn)
    assignments, recorder = _recorded(acl.RoleAssignmentCollection, conn)
    await assignments.list_scoped_to_namespace(namespaces[3])
    assert plan_uses_index(recorder.plans[0], "idx_role_assignments_scope_namespace"), recorder.plans[0].plan


async def test_assignments_scope_type_customer(conn: asyncpg.Connection) -> None:
    """Flips if a customer's type_customer grants of one type stop reading idx_role_assignments_scope_type_customer."""
    await _seed_assignments(conn)
    assignments, recorder = _recorded(acl.RoleAssignmentCollection, conn)
    await assignments.list_of_type_customer(uuid.uuid4(), "t1")
    assert plan_uses_index(recorder.plans[0], "idx_role_assignments_scope_type_customer"), recorder.plans[0].plan


async def test_assignments_role(conn: asyncpg.Connection) -> None:
    """Flips if a role's assignments, listed or counted, stop reading idx_role_assignments_role."""
    await _seed_assignments(conn)
    role = await conn.fetchval("SELECT role_id FROM roles LIMIT 1")
    assignments, recorder = _recorded(acl.RoleAssignmentCollection, conn)
    await assignments.list_by_role(role)
    await assignments.count_by_role(role)
    for plan in recorder.plans:
        assert plan_uses_index(plan, "idx_role_assignments_role"), plan.plan


async def test_assignments_scope_type(conn: asyncpg.Connection) -> None:
    """Flips if every grant of one scope type stops reading idx_role_assignments_scope_type."""
    await _seed_assignments(conn)
    # the rare scope a reverse access query weighs against every namespace: a handful of platform grants
    await conn.execute("UPDATE role_assignments SET scope_type = 'type_customer' WHERE scope_type = 'all'")
    await conn.execute(
        "UPDATE role_assignments SET scope_type = 'all', scope_customer_id = NULL, scope_namespace_type = NULL"
        " WHERE assignment_id IN (SELECT assignment_id FROM role_assignments WHERE scope_type = 'type_customer'"
        " LIMIT 5)"
    )
    await conn.execute("ANALYZE role_assignments")
    assignments, recorder = _recorded(acl.RoleAssignmentCollection, conn)
    found = await assignments.list_by_scope_type(ScopeType.ALL)
    plan = recorder.plans[0]
    # PostgreSQL reads the scope-type index; YugabyteDB reads the all grants' own partial unique index (WHERE
    # scope_type = 'all'), which holds them alone. Either way the read finds exactly the scope's rows
    keyed = plan_uses_index(plan, "idx_role_assignments_scope_type") or _scans_partial(
        plan.plan, "uq_role_assignments_all_grant"
    )
    assert keyed and plan.rows_scanned == len(found) == 5, json.dumps(plan.plan)[:2500]


async def test_assignments_group_scope_by_group(conn: asyncpg.Connection) -> None:
    """Flips if a group's assignments, listed or counted, stop reading idx_role_assignments_group_scope."""
    await _seed_assignments(conn)
    group = await conn.fetchval("SELECT group_id FROM groups LIMIT 1")
    assignments, recorder = _recorded(acl.RoleAssignmentCollection, conn)
    await assignments.list_by_group(group)
    await assignments.count_by_group(group)
    for plan in recorder.plans:
        assert plan_uses_index(plan, "idx_role_assignments_group_scope"), plan.plan


async def test_namespaces_type(conn: asyncpg.Connection) -> None:
    """Flips if a type's namespaces stop reading idx_namespaces_type."""
    await _seed(
        conn,
        "namespaces",
        row_scope="'customer'",
        customer_id="gen_random_uuid()",
        namespace_type="(ARRAY['tool', 'workspace', 'agent', 'datasource', 'channel'])[1 + g % 5]",
        name="'n' || g",
    )
    await _seed(
        conn, "namespaces", 3, row_scope="'platform'", customer_id="NULL", namespace_type="'model'", name="'m' || g"
    )
    namespaces, recorder = _recorded(acl.NamespaceCollection, conn)
    assert len(await namespaces.list_by_type("model")) == 3
    plan = recorder.plans[0]
    # the type index, or the tool provider index that leads with the type too (PostgreSQL's pick once both exist):
    # either way exactly the type's rows
    keyed = plan_uses_index(plan, "idx_namespaces_type") or plan_uses_index(plan, "idx_namespaces_tool_provider_pod")
    assert keyed and plan.rows_scanned == 3, json.dumps(plan.plan)[:2500]


async def test_namespace_ids_page_by_key(conn: asyncpg.Connection) -> None:
    """Flips if a page of namespace ids stops reading a key by its partition with the page as its bound (each page
    reads about its own rows, never the rest of the partition), or a walk of the pages misses or repeats an id."""
    await _seed(conn, "namespaces", row_scope="'customer'", customer_id="gen_random_uuid()", name="'n' || g")
    await _seed(
        conn, "namespaces", 7, row_scope="'platform'", customer_id="NULL", namespace_type="'model'", name="'m' || g"
    )
    namespaces, recorder = _recorded(acl.NamespaceCollection, conn)
    walked: list[uuid.UUID] = []
    cursor: str | None = None
    while True:
        page = await namespaces.page_ids(limit=100, cursor=cursor)
        walked.extend(row["namespace_id"] for row in page.rows)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert sorted(walked) == sorted(
        row["namespace_id"] for row in await conn.fetch("SELECT namespace_id FROM namespaces")
    )
    for plan in recorder.plans:
        # the partition's key (row_scope, namespace_id), or the id's own unique key walked in order with the partition
        # as a filter: either way a key in order, under the page's LIMIT
        keyed = plan_uses_index(plan, "namespaces_pkey") or plan_uses_index(plan, "uq_namespaces_namespace_id")
        assert keyed and plan.rows_scanned <= 2 * (100 + 1), json.dumps(plan.plan)[:3000]


async def test_namespaces_tool_provider_pod(conn: asyncpg.Connection) -> None:
    """Flips if a pod's provider nodes, the ownership graph or the node names stop reading
    idx_namespaces_tool_provider_pod."""
    await _seed(
        conn,
        "namespaces",
        row_scope="'customer'",
        customer_id="gen_random_uuid()",
        namespace_type="(ARRAY['tool', 'workspace', 'agent', 'knowledge'])[1 + g % 4]",
        name="'n' || g",
    )
    await _seed(
        conn,
        "namespaces",
        2000,
        row_scope="'platform'",
        customer_id="NULL",
        namespace_type="(ARRAY['tool', 'model'])[1 + g % 2]",
        name="'p' || g",
    )
    await _seed(
        conn,
        "namespaces",
        40,
        row_scope="'platform'",
        customer_id="NULL",
        namespace_type="'tool_provider'",
        name="'tools.provider' || g",
        metadata="jsonb_build_object('tool_pod_id', 'pod-' || (g % 20))",
    )
    namespaces, recorder = _recorded(acl.NamespaceCollection, conn)
    assert len(await namespaces.tool_provider_nodes("pod-3")) == 2
    assert len(await namespaces.tool_provider_ownership()) == 40
    assert len(await namespaces.tool_provider_names()) == 40
    by_pod, *whole = recorder.plans
    assert plan_uses_index(by_pod, "idx_namespaces_tool_provider_pod"), json.dumps(by_pod.plan)[:2500]
    # every node: PostgreSQL reads the pod index by its (namespace_type, row_scope) prefix; YugabyteDB reads the type
    # index with the partition as a storage filter. Either way exactly the nodes' rows
    for plan in whole:
        keyed = plan_uses_index(plan, "idx_namespaces_tool_provider_pod") or plan_uses_index(
            plan, "idx_namespaces_type"
        )
        assert keyed and plan.rows_scanned == 40, json.dumps(plan.plan)[:2500]
