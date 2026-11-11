"""integration: each agent-acl index serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. Each seeds its table past the size where
the planner would rather scan, runs the purpose's own statement through ``EXPLAIN ANALYZE``
(:func:`threetears.core.testing.plans.explain_statement`), and asks the plan whether it found its rows through the
index by an index condition (:func:`~threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest

from threetears.agent.acl import collections as acl
from threetears.agent.acl.migrations import register as register_acl
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.core.data.migrations import MigrationRunner
from threetears.core.collections.scan_cache import ScanCache
from threetears.core.testing.plans import PlanRecorder, explain_statement, explain_write, plan_uses_index

pytestmark = pytest.mark.integration


class _Store:
    """The DataStore surface the runner drives, over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        return await self._conn.execute(sql, *params)

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
        await runner.apply_for_platform_schema(_Store(connection))  # type: ignore[arg-type]
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
    # a group's grant by its natural key, the grant a concurrent ensure won, a group's grants on one scope and
    # their delete: each reads the group's rows through the same index, bound in the seeded row scope and scope type
    # (a scope holding no rows reads as empty through the primary key's row_scope prefix, or a partial index's). On
    # a namespace each may read the namespace grants' own unique index instead, led by the group too.
    group, role = uuid.uuid4(), uuid.uuid4()
    reads = {
        "GRANT_OF_GROUP_SQL": (acl.GRANT_OF_GROUP_SQL, lambda scope, ns: ("platform", group, role, scope, ns)),
        "GRANT_WINNER_OF_GROUP_SQL": (
            acl.GRANT_WINNER_OF_GROUP_SQL,
            lambda scope, ns: (group, role, scope, ns, "manual"),
        ),
        "GRANTS_OF_GROUP_SCOPE_SQL": (
            acl.GRANTS_OF_GROUP_SCOPE_SQL,
            lambda scope, ns: ("platform", group, scope, ns, None),
        ),
        "DELETE_GRANTS_OF_GROUP_SCOPE_SQL": (
            acl.DELETE_GRANTS_OF_GROUP_SCOPE_SQL,
            lambda scope, ns: ("platform", group, scope, ns, "manual"),
        ),
    }
    for name, (statement, params) in reads.items():
        # a write is planned, never run
        explain = explain_write if statement.lstrip().upper().startswith("DELETE") else explain_statement
        async with _rolled_back(conn):
            plan = await explain(conn, statement, *params("type_customer", None))
        assert plan_uses_index(plan, "idx_role_assignments_group_scope"), (name, plan.plan)
        async with _rolled_back(conn):
            plan = await explain(conn, statement, *params("namespace", uuid.uuid4()))
        assert plan_uses_index(plan, "idx_role_assignments_group_scope") or plan_uses_index(
            plan, "uq_role_assignments_namespace_grant"
        ), (f"{name} on a namespace", plan.plan)


async def test_namespaces_name(conn: asyncpg.Connection) -> None:
    """Flips if a namespace by name stops reading idx_namespaces_name."""
    await _seed(
        conn, "namespaces", row_scope="'platform'", customer_id="NULL", namespace_type="'tool'", name="'n' || g"
    )
    plan = await explain_statement(conn, acl.NAMESPACE_BY_NAME_SQL, "n17")
    assert plan_uses_index(plan, "idx_namespaces_name"), plan.plan


async def _seed_namespaces(conn: asyncpg.Connection) -> None:
    """namespaces past the size where the planner would rather scan: customers' agents, workspaces and tools, each
    owned, each naming a schema, plus ownerless platform rows.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :return: nothing
    :rtype: None
    """
    await conn.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, owner_agent_id, customer_id,"
        " schema_name, owner_namespace, date_created, date_updated)"
        " SELECT 'customer', gen_random_uuid(), 'agent.a' || g, (ARRAY['agent', 'workspace', 'tool'])[1 + g % 3],"
        " gen_random_uuid(), ('00000000-0000-0000-0000-' || lpad((g % 500)::text, 12, '0'))::uuid,"
        " 'agent_s' || g, 'agent.o' || (g % 2000), now(), now() FROM generate_series(1, 6000) g"
    )
    await conn.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, date_created, date_updated)"
        " SELECT 'platform', gen_random_uuid(), 'tool.p' || g, 'tool', now(), now() FROM generate_series(1, 3000) g"
    )
    await conn.execute("ANALYZE namespaces")


async def test_namespaces_owner_agent_id(conn: asyncpg.Connection) -> None:
    """Flips if an agent's own namespace, or an owned one by its triple, stops reading idx_namespaces_owner_agent_id."""
    await _seed_namespaces(conn)
    agent = await conn.fetchval("SELECT owner_agent_id FROM namespaces WHERE owner_agent_id IS NOT NULL LIMIT 1")
    plan = await explain_statement(conn, acl.NAMESPACE_OF_AGENT_SQL, agent, "agent")
    assert plan_uses_index(plan, "idx_namespaces_owner_agent_id"), plan.plan
    plan = await explain_statement(conn, acl.NAMESPACE_OF_OWNER_SQL, "customer", "agent", agent, uuid.uuid4())
    assert plan_uses_index(plan, "idx_namespaces_owner_agent_id"), plan.plan


async def test_namespaces_customer_type(conn: asyncpg.Connection) -> None:
    """Flips if a customer's namespaces of a type, or an ownerless one (a customer's or the platform's), stop
    reading idx_namespaces_customer_type."""
    await _seed_namespaces(conn)
    customer = uuid.UUID("00000000-0000-0000-0000-000000000017")
    for statement, params in (
        (acl.NAMESPACES_OF_CUSTOMER_TYPE_SQL, (customer, "tool")),
        (acl.NAMESPACE_IDS_OF_CUSTOMER_TYPE_SQL, (customer, "tool")),
        (acl.OWNERLESS_NAMESPACE_OF_CUSTOMER_SQL, ("customer", customer, "workspace")),
        (acl.OWNERLESS_PLATFORM_NAMESPACE_SQL, ("platform", "shared")),
    ):
        plan = await explain_statement(conn, statement, *params)
        assert plan_uses_index(plan, "idx_namespaces_customer_type"), (statement, plan.plan)


async def test_namespaces_owner_namespace(conn: asyncpg.Connection) -> None:
    """Flips if what a namespace owns stops reading idx_namespaces_owner_namespace."""
    await _seed_namespaces(conn)
    plan = await explain_statement(conn, acl.NAMESPACES_OWNED_BY_SQL, "agent.o17")
    assert plan_uses_index(plan, "idx_namespaces_owner_namespace"), plan.plan


async def test_namespaces_schema_name(conn: asyncpg.Connection) -> None:
    """Flips if whether a schema is still named stops reading idx_namespaces_schema_name."""
    await _seed_namespaces(conn)
    plan = await explain_statement(conn, acl.SCHEMA_IN_USE_SQL, "agent_s17")
    assert plan_uses_index(plan, "idx_namespaces_schema_name"), plan.plan


async def test_namespaces_under_name(conn: asyncpg.Connection) -> None:
    """Flips if the namespaces at or beneath a node stop reading the node by idx_namespaces_name and the names beneath
    it by idx_namespaces_name_c."""
    await _seed_namespaces(conn)
    plan = await explain_statement(conn, acl.NAMESPACE_IDS_UNDER_NAME_SQL, "tool.p17", "tool.p17.", "tool.p17/")
    assert plan_uses_index(plan, "idx_namespaces_name"), plan.plan
    assert plan_uses_index(plan, "idx_namespaces_name_c"), plan.plan


async def test_the_names_beneath_a_node_are_its_subtree_whatever_the_collation(conn: asyncpg.Connection) -> None:
    """Flips if the byte range misses a name beneath the node, or lets a sibling that shares its prefix in -- the
    names a linguistic collation orders between them, ``.`` and ``-`` sorting as ignorable."""
    names = ("a.b", "a.b.c", "a.b.c.d", "a.bc", "a.b-c", "a.b/c", "a.b_c", "ab.c", "a.a.z", "a.c")
    await conn.executemany(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, date_created, date_updated)"
        " VALUES ('platform', gen_random_uuid(), $1, 'tool', now(), now())",
        [(name,) for name in names],
    )
    namespaces = _collection(acl.NamespaceCollection, conn)
    ids = await namespaces.list_ids_under_name("a.b")
    found = {row["name"] for row in await conn.fetch("SELECT name FROM namespaces WHERE namespace_id = ANY($1)", ids)}
    assert found == {"a.b", "a.b.c", "a.b.c.d"}, found
    rows = await conn.fetch(acl.NAMESPACE_IDS_UNDER_NAME_SQL, "a.b", "a.b.", "a.b/")
    assert {row["name"] for row in rows} == {"a.b", "a.b.c", "a.b.c.d"}, "the range alone holds the subtree"


def _collection(cls: type, conn: Any) -> Any:
    """a collection over a stub registry whose L3 pool is ``conn``: no L1, nothing promoted anywhere else.

    :param cls: the collection class
    :ptype cls: type
    :param conn: the connection (or a :class:`~threetears.core.testing.plans.PlanRecorder` over it)
    :ptype conn: Any
    :return: the collection
    :rtype: Any
    """
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.get_l3_pool.return_value = conn
    registry.register.return_value = None
    registry.scan_cache = ScanCache(None)
    config = MagicMock()
    config.collection_flush = "ALWAYS"
    config.collection_flush_tables = ""
    return cls(registry=registry, config=config)


async def test_groups_by_name(conn: asyncpg.Connection) -> None:
    """Flips if a group by name, platform or a customer's, stops reading its scope's unique name index."""
    await _seed(conn, "groups", row_scope="'platform'", customer_id="NULL", name="'p' || g")
    await _seed(conn, "groups", row_scope="'customer'", customer_id="gen_random_uuid()", name="'c' || g")
    platform = await explain_statement(conn, acl.GROUP_BY_PLATFORM_NAME_SQL, "p17")
    assert plan_uses_index(platform, "uq_groups_platform_name"), platform.plan
    customer = await explain_statement(conn, acl.GROUP_BY_CUSTOMER_NAME_SQL, uuid.uuid4(), "c17")
    assert plan_uses_index(customer, "uq_groups_customer_name"), customer.plan
    # a customer's pages of groups by name: GroupCollection.page_by_customer's own statements, first page and after
    owner = await conn.fetchval("SELECT customer_id FROM groups WHERE customer_id IS NOT NULL LIMIT 1")
    await conn.execute(
        "INSERT INTO groups (row_scope, group_id, customer_id, name, date_created, date_updated)"
        " SELECT 'customer', gen_random_uuid(), $1, 'o' || g, now(), now() FROM generate_series(1, 5) g",
        owner,
    )
    recorder = PlanRecorder(conn)
    groups = _collection(acl.GroupCollection, recorder)
    first = await groups.page_by_customer(owner, limit=2)
    assert first.next_cursor is not None
    await groups.page_by_customer(owner, limit=2, cursor=first.next_cursor)
    assert len(recorder.plans) == 2
    for paged in recorder.plans:
        assert plan_uses_index(paged, "uq_groups_customer_name"), paged.plan


async def test_roles_platform_name(conn: asyncpg.Connection) -> None:
    """Flips if a platform role by name (every built-in's lookup) stops reading uq_roles_platform_name."""
    await _seed(conn, "roles", name="'r' || g", description="'r'", permissions="'[]'::jsonb", customer_id="NULL")
    plan = await explain_statement(conn, acl.ROLE_BY_PLATFORM_NAME_SQL, "r17")
    assert plan_uses_index(plan, "uq_roles_platform_name"), plan.plan


@contextlib.asynccontextmanager
async def _rolled_back(conn: asyncpg.Connection) -> AsyncIterator[None]:
    """a transaction rolled back whatever its body did.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :return: the open transaction's body
    :rtype: AsyncIterator[None]
    """
    transaction = conn.transaction()
    await transaction.start()
    try:
        yield
    finally:
        await transaction.rollback()


_NAMESPACE_INSERT = (
    "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, schema_name,"
    " date_created, date_updated) VALUES ('customer', gen_random_uuid(), $1, $2, $3, $4, now(), now())"
)


async def test_namespaces_schema_name_non_workspace_refuses_a_duplicate(conn: asyncpg.Connection) -> None:
    """Flips if two namespaces other than workspaces may name one schema: uq_namespaces_schema_name_non_workspace
    refuses the second, while workspace rows still share their agent's schema."""
    customer = uuid.uuid4()
    async with _rolled_back(conn):
        await conn.execute(_NAMESPACE_INSERT, "a", "agent", customer, "agent_s1")
        with pytest.raises(asyncpg.UniqueViolationError, match="uq_namespaces_schema_name_non_workspace"):
            await conn.execute(_NAMESPACE_INSERT, "a", "agent", customer, "agent_s1")
    for name in ("w1", "w2"):
        await conn.execute(_NAMESPACE_INSERT, name, "workspace", customer, "agent_s2")


async def test_roles_customer_name_refuses_a_duplicate(conn: asyncpg.Connection) -> None:
    """Flips if one customer may hold two roles of one name: uq_roles_customer_name refuses the second."""
    customer = uuid.uuid4()
    async with _rolled_back(conn):
        await conn.execute(
            "INSERT INTO roles (role_id, name, description, permissions, customer_id)"
            " VALUES (gen_random_uuid(), 'Field Manager', 'r', '[]'::jsonb, $1)",
            customer,
        )
        with pytest.raises(asyncpg.UniqueViolationError, match="uq_roles_customer_name"):
            await conn.execute(
                "INSERT INTO roles (role_id, name, description, permissions, customer_id)"
                " VALUES (gen_random_uuid(), 'Field Manager', 'r', '[]'::jsonb, $1)",
                customer,
            )


async def _group_and_role(conn: asyncpg.Connection) -> tuple[uuid.UUID, uuid.UUID]:
    """one platform group and one role, for a grant to name.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :return: the group's id and the role's
    :rtype: tuple[uuid.UUID, uuid.UUID]
    """
    group, role = uuid.uuid4(), uuid.uuid4()
    await conn.execute("INSERT INTO groups (row_scope, group_id, name) VALUES ('platform', $1, 'g')", group)
    await conn.execute(
        "INSERT INTO roles (role_id, name, description, permissions) VALUES ($1, 'r', 'r', '[]'::jsonb)", role
    )
    return group, role


async def test_namespace_grant_refuses_a_duplicate(conn: asyncpg.Connection) -> None:
    """Flips if a group may hold one role on one namespace twice from one provenance:
    uq_role_assignments_namespace_grant refuses the second (ensure_group_role_assignment's ON CONFLICT relies on it)."""
    group, role = await _group_and_role(conn)
    namespace = await conn.fetchval(_NAMESPACE_INSERT + " RETURNING namespace_id", "n", "agent", uuid.uuid4(), None)
    async with _rolled_back(conn):
        await conn.execute(
            "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_namespace_id)"
            " VALUES ('customer', gen_random_uuid(), $1, $2, 'namespace', $3)",
            role,
            group,
            namespace,
        )
        with pytest.raises(asyncpg.UniqueViolationError, match="uq_role_assignments_namespace_grant"):
            await conn.execute(
                "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_namespace_id)"
                " VALUES ('customer', gen_random_uuid(), $1, $2, 'namespace', $3)",
                role,
                group,
                namespace,
            )


async def test_all_grant_refuses_a_duplicate(conn: asyncpg.Connection) -> None:
    """Flips if a group may hold one role over everything twice from one provenance: uq_role_assignments_all_grant
    refuses the second (ensure_group_role_assignment's ON CONFLICT relies on it)."""
    group, role = await _group_and_role(conn)
    async with _rolled_back(conn):
        await conn.execute(
            "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type)"
            " VALUES ('platform', gen_random_uuid(), $1, $2, 'all')",
            role,
            group,
        )
        with pytest.raises(asyncpg.UniqueViolationError, match="uq_role_assignments_all_grant"):
            await conn.execute(
                "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type)"
                " VALUES ('platform', gen_random_uuid(), $1, $2, 'all')",
                role,
                group,
            )
