"""integration: each agent-acl index serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. Each seeds its table past the size where
the planner would rather scan, runs the purpose's own statement through ``EXPLAIN ANALYZE``
(:func:`threetears.core.testing.plans.explain_statement`), and asks the plan whether it found its rows through the
index by an index condition (:func:`~threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest

from threetears.agent.acl import collections as acl
from threetears.agent.acl.migrations import register as register_acl
from threetears.agent.acl import NamespaceEntity
from threetears.agent.acl.types import ScopeType
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.core.collections.delete_cascade import (
    RestrictedDeleteError,
    read_delete_cascade,
    refuse_restricted_delete,
)
from threetears.core.backends.dialect import known_on_yugabyte
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, bucket_epoch_index_name, next_epoch
from threetears.core.data.migrations import MigrationRunner
from threetears.core.keyed import read_by
from threetears.core.collections.scan_cache import ScanCache
from threetears.interfaces.keyed import (
    AnyOf,
    Fragment,
    Sort,
    delete_keyed_statements,
    read_by_statement,
    update_keyed_statements,
)
from threetears.core.testing.captured import CapturedL3
from threetears.core.testing.plans import (
    PlanRecorder,
    StatementPlan,
    explain_statement,
    explain_write,
    plan_uses_index,
)

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
    """seed ``rows`` rows into ``table``, then ANALYZE it, so the planner prices it at size. Every acl table is
    ``ROW_STAMPED`` (v006): the rows are stamped as a write of them leaves them, one fresh epoch the statement
    (:func:`~threetears.core.ddl.row_epoch.next_epoch`), and the stored bucket is left to the database.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :param table: the table
    :ptype table: str
    :param rows: how many
    :ptype rows: int
    :param values: a SQL expression of the series row ``g`` per column; every other NOT NULL column with no
        default gets one by its type, the epoch its stamp; a generated column none
    :ptype values: str
    :return: nothing
    :rtype: None
    """
    columns = await conn.fetch(
        "SELECT column_name, data_type, is_nullable, column_default, is_generated FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name = $1",
        table,
    )
    names: list[str] = []
    exprs: list[str] = []
    for column in columns:
        name = column["column_name"]
        if column["is_generated"] == "ALWAYS":
            continue
        if name == EPOCH_COLUMN and name not in values:
            names.append(name)
            exprs.append(f"{next_epoch()}::bigint")
        elif name in values:
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
        "INSERT INTO group_members (id, group_id, member_type, member_id, epoch)"
        " SELECT gen_random_uuid(), ($1::uuid[])[1 + g % 50], CASE WHEN g % 2 = 0 THEN 'user' ELSE 'agent' END,"
        " gen_random_uuid(), $2 FROM generate_series(1, 5000) g",
        groups,
        next_epoch(),
    )
    await conn.execute("ANALYZE group_members")
    plan = await _planned(conn, acl.GroupMemberCollection, lambda members: members.load_for_user(uuid.uuid4()))
    assert plan_uses_index(plan, "idx_group_members_reverse"), plan.plan


async def test_assignments_group_scope(conn: asyncpg.Connection) -> None:
    """Flips if every assignment a set of groups holds stops reading idx_role_assignments_group_scope."""
    await _seed(conn, "groups", 500, row_scope="'platform'", customer_id="NULL", name="'g' || g")
    await _seed(conn, "roles", 1, name="'r'", description="'r'", permissions="'[]'::jsonb")
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, epoch)"
        " SELECT 'platform', gen_random_uuid(), (SELECT role_id FROM roles), group_id, 'type_customer', $1"
        " FROM groups, generate_series(1, 10)",
        next_epoch(),
    )
    # and namespace grants of some groups on one namespace, so the scope types the reads bind both hold rows
    granted = await conn.fetchval(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, date_created, date_updated, epoch)"
        " VALUES ('platform', gen_random_uuid(), 'tool.granted', 'tool', now(), now(), $1) RETURNING namespace_id",
        next_epoch(),
    )
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_namespace_id,"
        " epoch) SELECT 'platform', gen_random_uuid(), (SELECT role_id FROM roles), group_id, 'namespace', $1, $2"
        " FROM groups LIMIT 100",
        granted,
        next_epoch(),
    )
    await conn.execute("ANALYZE role_assignments")
    plan = await _planned(conn, acl.RoleAssignmentCollection, lambda held: held.load_for_groups([uuid.uuid4()]))
    assert plan_uses_index(plan, "idx_role_assignments_group_scope"), plan.plan
    # a group's grant by its natural key, the grant a concurrent ensure won, a group's grants on one scope and
    # their delete: each reads the group's rows through the same index, bound in the seeded row scope and scope type
    # (a scope holding no rows reads as empty through the primary key's row_scope prefix, or a partial index's). On
    # a namespace each may read the namespace grants' own unique index instead, led by the group too.
    # a group and the role it holds, as a grant names them (the role indexes of v004 would read an unheld role's
    # nothing faster, which says nothing of the group's reads)
    group = await conn.fetchval("SELECT group_id FROM groups ORDER BY group_id LIMIT 1 OFFSET 250")
    role = await conn.fetchval("SELECT role_id FROM roles")
    # a group's grant by its natural key, as the idempotent ensure sends it (its first statement; the method stops
    # there, so what it would write is never sent). On the namespace the seed granted it reads the group's rows by
    # the index (or the namespace grants' own unique index, led by the group too); on 'all', which the seed grants no
    # group, the planner may prove it empty another way, so there it is held to reading no more than its page
    # the ensure reads a namespace grant in the customer row scope: some groups hold one on a customer's namespace
    held_on = await conn.fetchval(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, date_created,"
        " date_updated, epoch) VALUES ('customer', gen_random_uuid(), 'tool.held', 'tool', gen_random_uuid(), now(),"
        " now(), $1) RETURNING namespace_id",
        next_epoch(),
    )
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_namespace_id,"
        " epoch) SELECT 'customer', gen_random_uuid(), $2, group_id, 'namespace', $1, $3 FROM groups",
        held_on,
        role,
        next_epoch(),
    )
    await conn.execute("ANALYZE role_assignments")
    for scope, namespace in (("namespace", held_on), ("all", None)):

        def ensure(
            held: acl.RoleAssignmentCollection, scope: str = scope, namespace: uuid.UUID | None = namespace
        ) -> Awaitable[tuple[uuid.UUID, bool]]:
            return held.ensure_group_role_assignment(group_id=group, role_id=role, scope_type=scope, scope_id=namespace)

        plan = await _first_planned(conn, acl.RoleAssignmentCollection, ensure)
        if scope == "namespace":
            assert plan_uses_index(plan, "idx_role_assignments_group_scope") or plan_uses_index(
                plan, "uq_role_assignments_namespace_grant"
            ), plan.plan
        else:
            assert plan.rows_scanned <= 2, plan
    # the winner read and a group's grants on one scope, read then deleted: the helpers' own statements
    # (read_by_statement / delete_keyed_statements) with the pins each method sends, in both scopes. The namespace
    # is the one every group holds a grant on: the namespace is a plain pin now (``= $n``, where the statements'
    # IS NOT DISTINCT FROM could not seek), so on a namespace holding no grants PostgreSQL rightly reads the
    # namespace's own index (idx_role_assignments_scope_namespace) and finds none
    for scope, namespace, row_scope in (("all", None, "platform"), ("namespace", held_on, "customer")):
        winner = {
            "row_scope": AnyOf(("platform", "customer"), cast=None),
            "group_id": group,
            "role_id": role,
            "scope_type": scope,
            "scope_namespace_id": namespace,
            "managed_by": "manual",
        }
        grants: dict[str, Any] = {
            "row_scope": row_scope,
            "group_id": group,
            "scope_type": scope,
            "scope_namespace_id": namespace,
        }
        statements = {
            "the winner read": (
                explain_statement,
                read_by_statement(
                    "role_assignments",
                    by=winner,
                    columns=("row_scope", "assignment_id", "date_granted"),
                    order=(Sort("date_granted"), Sort("assignment_id")),
                ),
            ),
            "the grants of any provenance": (
                explain_statement,
                read_by_statement("role_assignments", by=grants, columns=("assignment_id",)),
            ),
            "the manual grants": (
                explain_statement,
                read_by_statement(
                    "role_assignments", by={**grants, "managed_by": "manual"}, columns=("assignment_id",)
                ),
            ),
            "their delete": (
                explain_write,
                delete_keyed_statements("role_assignments", by=grants, returning=("assignment_id",))[0],
            ),
        }
        for name, (explain, (sql, params)) in statements.items():
            async with _rolled_back(conn):
                plan = await explain(conn, sql, *params)
            # or the scope's own natural-key unique index, led by the group too (partial on its scope type)
            own = "uq_role_assignments_namespace_grant" if scope == "namespace" else "uq_role_assignments_all_grant"
            assert plan_uses_index(plan, "idx_role_assignments_group_scope") or plan_uses_index(plan, own), (
                f"{name} on scope {scope}",
                _outline(plan.plan),
            )


def _outline(node: Mapping[str, Any]) -> str:
    """a plan's nodes in one line, each with the index it reads: what a failing plan test shows."""
    named = node.get("Index Name") or node.get("Relation Name") or ""
    children = ", ".join(_outline(child) for child in node.get("Plans", ()))
    return f"{node.get('Node Type')}({named})" + (f" > [{children}]" if children else "")


async def test_namespaces_name(conn: asyncpg.Connection) -> None:
    """Flips if a namespace by name stops reading idx_namespaces_name."""
    await _seed(
        conn, "namespaces", row_scope="'platform'", customer_id="NULL", namespace_type="'tool'", name="'n' || g"
    )
    plan = await _planned(conn, acl.NamespaceCollection, lambda namespaces: namespaces.get_by_name("n17"))
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
        " schema_name, owner_namespace, date_created, date_updated, epoch)"
        " SELECT 'customer', gen_random_uuid(), 'agent.a' || g, (ARRAY['agent', 'workspace', 'tool'])[1 + g % 3],"
        " gen_random_uuid(), ('00000000-0000-0000-0000-' || lpad((g % 500)::text, 12, '0'))::uuid,"
        " 'agent_s' || g, 'agent.o' || (g % 2000), now(), now(), $1 FROM generate_series(1, 6000) g",
        next_epoch(),
    )
    await conn.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, date_created, date_updated, epoch)"
        " SELECT 'platform', gen_random_uuid(), 'tool.p' || g, 'tool', now(), now(), $1 FROM generate_series(1, 3000) g",
        next_epoch(),
    )
    await conn.execute("ANALYZE namespaces")


async def test_namespaces_owner_agent_id(conn: asyncpg.Connection) -> None:
    """Flips if an agent's own namespace, or an owned one by its triple, stops reading idx_namespaces_owner_agent_id."""
    await _seed_namespaces(conn)
    agent = await conn.fetchval("SELECT owner_agent_id FROM namespaces WHERE owner_agent_id IS NOT NULL LIMIT 1")
    plan = await _planned(conn, acl.NamespaceCollection, lambda namespaces: namespaces.get_by_agent_id(agent, "agent"))
    assert plan_uses_index(plan, "idx_namespaces_owner_agent_id"), plan.plan
    plan = await _planned(
        conn,
        acl.NamespaceCollection,
        lambda namespaces: namespaces.get_by_owner_and_customer(
            namespace_type="agent", owner_agent_id=agent, customer_id=uuid.uuid4()
        ),
    )
    assert plan_uses_index(plan, "idx_namespaces_owner_agent_id"), plan.plan


async def test_namespaces_tool_flags(conn: asyncpg.Connection) -> None:
    """Flips if the default tool surface's or the skills catalog's flagged tool namespaces stop reading
    idx_namespaces_tool_flags: among 20,000 agents and workspaces, 300 tools, a third of them in the catalog."""
    await conn.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, owner_agent_id, customer_id,"
        " date_created, date_updated, epoch)"
        " SELECT 'customer', gen_random_uuid(), 'agent.f' || g, (ARRAY['agent', 'workspace'])[1 + g % 2],"
        " gen_random_uuid(), ('00000000-0000-0000-0000-' || lpad((g % 500)::text, 12, '0'))::uuid, now(), now(), $1"
        " FROM generate_series(1, 20000) g",
        next_epoch(),
    )
    await conn.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, skill_eligible,"
        " date_created, date_updated, epoch)"
        " SELECT (ARRAY['platform', 'customer'])[1 + g % 2], gen_random_uuid(), 'tool.f' || g, 'tool',"
        " CASE WHEN g % 2 = 1 THEN ('00000000-0000-0000-0000-' || lpad((g % 500)::text, 12, '0'))::uuid END,"
        " g % 3 = 0, now(), now(), $1 FROM generate_series(1, 300) g",
        next_epoch(),
    )
    await conn.execute("ANALYZE namespaces")
    # each flag's listing as the collection sends it, its first statement (the evaluator's decisions come after)
    for flag in ("tool_eligible", "skill_eligible"):

        def listing(namespaces: acl.NamespaceCollection, flag: str = flag) -> Awaitable[list[NamespaceEntity]]:
            read = (
                namespaces.list_tool_namespaces_for_actor
                if flag == "tool_eligible"
                else namespaces.list_skill_eligible_tool_namespaces
            )
            return read(actor_user_id=uuid.uuid4(), actor_agent_id=uuid.uuid4(), cache=None)

        plan = await _first_planned(conn, acl.NamespaceCollection, listing)
        assert plan_uses_index(plan, "idx_namespaces_tool_flags"), (flag, plan.plan)
        # the flagged tools alone (100 or 150 of 300), never the 20,000 agents and workspaces
        assert plan.rows_scanned <= 300, (flag, plan)


async def test_namespaces_customer_type(conn: asyncpg.Connection) -> None:
    """Flips if a customer's namespaces of a type, or an ownerless one (a customer's or the platform's), stop
    reading idx_namespaces_customer_type."""
    await _seed_namespaces(conn)
    customer = uuid.UUID("00000000-0000-0000-0000-000000000017")
    # a customer's namespaces of a type, and their ids: each read as its collection method sends it
    for read in (
        lambda namespaces: namespaces.find_by_type_and_customer(namespace_type="tool", customer_id=customer),
        lambda namespaces: namespaces.list_ids_by_customer_and_type(customer, "tool"),
    ):
        plan = await _planned(conn, acl.NamespaceCollection, read)
        assert plan_uses_index(plan, "idx_namespaces_customer_type"), plan.plan
    # a customer's ownerless namespace of a type, and the platform's, as get_by_owner_and_customer sends them
    for owner_customer, namespace_type in ((customer, "workspace"), (None, "shared")):

        def ownerless(
            namespaces: acl.NamespaceCollection,
            owner_customer: uuid.UUID | None = owner_customer,
            namespace_type: str = namespace_type,
        ) -> Awaitable[NamespaceEntity | None]:
            return namespaces.get_by_owner_and_customer(
                namespace_type=namespace_type, owner_agent_id=None, customer_id=owner_customer
            )

        plan = await _planned(conn, acl.NamespaceCollection, ownerless)
        # the platform's ownerless namespace of a type is keyed by the type's index too (idx_namespaces_type, v004):
        # PostgreSQL reads it there when the type holds no rows, as "shared" does in this seed
        by_type = owner_customer is None and plan_uses_index(plan, "idx_namespaces_type")
        assert plan_uses_index(plan, "idx_namespaces_customer_type") or by_type, (namespace_type, plan.plan)


async def test_namespaces_owner_namespace(conn: asyncpg.Connection) -> None:
    """Flips if what a namespace owns stops reading idx_namespaces_owner_namespace."""
    await _seed_namespaces(conn)
    plan = await _planned(conn, acl.NamespaceCollection, lambda namespaces: namespaces.list_owned_by("agent.o17"))
    assert plan_uses_index(plan, "idx_namespaces_owner_namespace"), plan.plan


async def test_namespaces_schema_name(conn: asyncpg.Connection) -> None:
    """Flips if whether a schema is still named stops reading idx_namespaces_schema_name."""
    await _seed_namespaces(conn)
    plan = await _planned(conn, acl.NamespaceCollection, lambda namespaces: namespaces.schema_in_use("agent_s17"))
    assert plan_uses_index(plan, "idx_namespaces_schema_name"), plan.plan


async def test_namespaces_under_name(conn: asyncpg.Connection) -> None:
    """Flips if the namespaces at or beneath a node stop reading the node by idx_namespaces_name and the names beneath
    it by idx_namespaces_name_c."""
    await _seed_namespaces(conn)
    plan = await _planned(conn, acl.NamespaceCollection, lambda namespaces: namespaces.list_ids_under_name("tool.p17"))
    assert plan_uses_index(plan, "idx_namespaces_name"), plan.plan
    assert plan_uses_index(plan, "idx_namespaces_name_c"), plan.plan


async def test_the_names_beneath_a_node_are_its_subtree_whatever_the_collation(conn: asyncpg.Connection) -> None:
    """Flips if the byte range misses a name beneath the node, or lets a sibling that shares its prefix in -- the
    names a linguistic collation orders between them, ``.`` and ``-`` sorting as ignorable."""
    names = ("a.b", "a.b.c", "a.b.c.d", "a.bc", "a.b-c", "a.b/c", "a.b_c", "ab.c", "a.a.z", "a.c")
    await conn.executemany(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, date_created, date_updated, epoch)"
        " VALUES ('platform', gen_random_uuid(), $1, 'tool', now(), now(), $2)",
        [(name, next_epoch()) for name in names],
    )
    namespaces = _collection(acl.NamespaceCollection, conn)
    ids = await namespaces.list_ids_under_name("a.b")
    found = {row["name"] for row in await conn.fetch("SELECT name FROM namespaces WHERE namespace_id = ANY($1)", ids)}
    assert found == {"a.b", "a.b.c", "a.b.c.d"}, found
    sent = CapturedL3(real=conn)
    await _collection(acl.NamespaceCollection, sent).list_ids_under_name("a.b")
    ((sql, args),) = sent.asked
    rows = await conn.fetch(sql, *args)
    assert {row["name"] for row in rows} == {"a.b", "a.b.c", "a.b.c.d"}, "the range alone holds the subtree"


async def _planned(conn: asyncpg.Connection, cls: type, read: Callable[[Any], Awaitable[Any]]) -> StatementPlan:
    """the plan of the one statement a collection method sends: the method run over a recording connection.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :param cls: the collection class
    :ptype cls: type
    :param read: the method's call, given the collection
    :ptype read: Callable[[Any], Awaitable[Any]]
    :return: the statement's plan
    :rtype: StatementPlan
    """
    recorder = PlanRecorder(conn)
    await read(_collection(cls, recorder))
    assert len(recorder.plans) == 1, [plan.sql for plan in recorder.plans]
    return recorder.plans[0]


class _FirstPlanned(Exception):  # noqa: N818 - a control-flow signal carrying the plan, not an error
    """raised by :class:`_PlanFirst` with the plan of the first statement a method sends."""

    def __init__(self, plan: StatementPlan) -> None:
        super().__init__("planned")
        self.plan = plan


# parity-with: asyncpg.Connection (fetch / fetchrow / fetchval / execute -- the seams a collection method drives)
class _PlanFirst:
    """a connection that plans the first statement asked of it (``EXPLAIN ANALYZE``, rolled back) and stops the
    method there: what the method would write after it is never sent."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def answers_from_yugabyte(self) -> bool:
        """whether its connection is YugabyteDB, as a PlanRecorder answers it: a keyed read's index hint asks this,
        never a ``SELECT version()`` this double would plan as the method's first statement."""
        return await known_on_yugabyte(self._conn)

    async def _plan(self, sql: str, *args: Any) -> Any:
        raise _FirstPlanned(await explain_statement(self._conn, sql, *args))

    fetch = fetchrow = fetchval = execute = _plan


async def _first_planned(conn: asyncpg.Connection, cls: type, read: Callable[[Any], Awaitable[Any]]) -> StatementPlan:
    """the plan of the first statement a collection method sends (a read before the write it decides on); the method
    stops there, so nothing it would write after is sent.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :param cls: the collection class
    :ptype cls: type
    :param read: the method's call, given the collection
    :ptype read: Callable[[Any], Awaitable[Any]]
    :return: the first statement's plan
    :rtype: StatementPlan
    """
    try:
        await read(_collection(cls, _PlanFirst(conn)))
    except _FirstPlanned as planned:
        return planned.plan
    raise AssertionError("the method sent no statement")


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
    platform = await _planned(conn, acl.GroupCollection, lambda groups: groups.get_by_name("p17", None))
    assert plan_uses_index(platform, "uq_groups_platform_name"), platform.plan
    customer = await _planned(conn, acl.GroupCollection, lambda groups: groups.get_by_name("c17", uuid.uuid4()))
    assert plan_uses_index(customer, "uq_groups_customer_name"), customer.plan
    # a customer's pages of groups by name: GroupCollection.page_by_customer's own statements, first page and after
    owner = await conn.fetchval("SELECT customer_id FROM groups WHERE customer_id IS NOT NULL LIMIT 1")
    await conn.execute(
        "INSERT INTO groups (row_scope, group_id, customer_id, name, date_created, date_updated, epoch)"
        " SELECT 'customer', gen_random_uuid(), $1, 'o' || g, now(), now(), $2 FROM generate_series(1, 5) g",
        owner,
        next_epoch(),
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
    plan = await _planned(conn, acl.RoleCollection, lambda roles: roles.find_platform_by_name("r17"))
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
    " date_created, date_updated, epoch) VALUES ('customer', gen_random_uuid(), $1, $2, $3, $4, now(), now(), $5)"
)


async def test_namespaces_schema_name_non_workspace_refuses_a_duplicate(conn: asyncpg.Connection) -> None:
    """Flips if two namespaces other than workspaces may name one schema: uq_namespaces_schema_name_non_workspace
    refuses the second, while workspace rows still share their agent's schema."""
    customer = uuid.uuid4()
    async with _rolled_back(conn):
        await conn.execute(_NAMESPACE_INSERT, "a", "agent", customer, "agent_s1", next_epoch())
        with pytest.raises(asyncpg.UniqueViolationError, match="uq_namespaces_schema_name_non_workspace"):
            await conn.execute(_NAMESPACE_INSERT, "a", "agent", customer, "agent_s1", next_epoch())
    for name in ("w1", "w2"):
        await conn.execute(_NAMESPACE_INSERT, name, "workspace", customer, "agent_s2", next_epoch())


async def test_roles_customer_name_refuses_a_duplicate(conn: asyncpg.Connection) -> None:
    """Flips if one customer may hold two roles of one name: uq_roles_customer_name refuses the second."""
    customer = uuid.uuid4()
    async with _rolled_back(conn):
        await conn.execute(
            "INSERT INTO roles (role_id, name, description, permissions, customer_id, epoch)"
            " VALUES (gen_random_uuid(), 'Field Manager', 'r', '[]'::jsonb, $1, $2)",
            customer,
            next_epoch(),
        )
        with pytest.raises(asyncpg.UniqueViolationError, match="uq_roles_customer_name"):
            await conn.execute(
                "INSERT INTO roles (role_id, name, description, permissions, customer_id, epoch)"
                " VALUES (gen_random_uuid(), 'Field Manager', 'r', '[]'::jsonb, $1, $2)",
                customer,
                next_epoch(),
            )


async def _group_and_role(conn: asyncpg.Connection) -> tuple[uuid.UUID, uuid.UUID]:
    """one platform group and one role, for a grant to name.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :return: the group's id and the role's
    :rtype: tuple[uuid.UUID, uuid.UUID]
    """
    group, role = uuid.uuid4(), uuid.uuid4()
    await conn.execute(
        "INSERT INTO groups (row_scope, group_id, name, epoch) VALUES ('platform', $1, 'g', $2)", group, next_epoch()
    )
    await conn.execute(
        "INSERT INTO roles (role_id, name, description, permissions, epoch) VALUES ($1, 'r', 'r', '[]'::jsonb, $2)",
        role,
        next_epoch(),
    )
    return group, role


async def test_namespace_grant_refuses_a_duplicate(conn: asyncpg.Connection) -> None:
    """Flips if a group may hold one role on one namespace twice from one provenance:
    uq_role_assignments_namespace_grant refuses the second (ensure_group_role_assignment's ON CONFLICT relies on it)."""
    group, role = await _group_and_role(conn)
    namespace = await conn.fetchval(
        _NAMESPACE_INSERT + " RETURNING namespace_id", "n", "agent", uuid.uuid4(), None, next_epoch()
    )
    async with _rolled_back(conn):
        await conn.execute(
            "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_namespace_id,"
            " epoch) VALUES ('customer', gen_random_uuid(), $1, $2, 'namespace', $3, $4)",
            role,
            group,
            namespace,
            next_epoch(),
        )
        with pytest.raises(asyncpg.UniqueViolationError, match="uq_role_assignments_namespace_grant"):
            await conn.execute(
                "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type,"
                " scope_namespace_id, epoch) VALUES ('customer', gen_random_uuid(), $1, $2, 'namespace', $3, $4)",
                role,
                group,
                namespace,
                next_epoch(),
            )


async def test_all_grant_refuses_a_duplicate(conn: asyncpg.Connection) -> None:
    """Flips if a group may hold one role over everything twice from one provenance: uq_role_assignments_all_grant
    refuses the second (ensure_group_role_assignment's ON CONFLICT relies on it)."""
    group, role = await _group_and_role(conn)
    async with _rolled_back(conn):
        await conn.execute(
            "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, epoch)"
            " VALUES ('platform', gen_random_uuid(), $1, $2, 'all', $3)",
            role,
            group,
            next_epoch(),
        )
        with pytest.raises(asyncpg.UniqueViolationError, match="uq_role_assignments_all_grant"):
            await conn.execute(
                "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, epoch)"
                " VALUES ('platform', gen_random_uuid(), $1, $2, 'all', $3)",
                role,
                group,
                next_epoch(),
            )


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
        " scope_namespace_type, scope_customer_id, customer_id, managed_by, epoch)"
        " SELECT 'customer', gen_random_uuid(), (array_agg_r)[1 + s % 50], (array_agg_g)[1 + s % 500],"
        " (ARRAY['namespace', 'type_customer', 'all'])[1 + s % 3],"
        " CASE WHEN s % 3 = 0 THEN ($1::uuid[])[1 + s % 500] END,"
        " CASE WHEN s % 3 = 1 THEN 't' || (s % 20) END,"
        # a type grant's customer (acl v005) is its scope's
        " CASE WHEN s % 3 = 1 THEN md5(s::text)::uuid END, CASE WHEN s % 3 = 1 THEN md5(s::text)::uuid END,"
        " 'm' || s, $2"
        " FROM (SELECT array_agg(role_id) AS array_agg_r FROM roles) r,"
        " (SELECT array_agg(group_id) AS array_agg_g FROM groups) g, generate_series(1, 5000) s",
        namespaces,
        next_epoch(),
    )
    await conn.execute("ANALYZE role_assignments")
    return namespaces


async def test_members_of_groups(conn: asyncpg.Connection) -> None:
    """Flips if every member of a set of groups, or a group's member count, stops reading idx_group_members_group_id."""
    await _seed(conn, "groups", 500, row_scope="'platform'", customer_id="NULL", name="'g' || g")
    groups = [row["group_id"] for row in await conn.fetch("SELECT group_id FROM groups")]
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id, epoch)"
        " SELECT gen_random_uuid(), ($1::uuid[])[1 + g % 500], 'user', gen_random_uuid(), $2"
        " FROM generate_series(1, 5000) g",
        groups,
        next_epoch(),
    )
    await conn.execute("ANALYZE group_members")
    members, recorder = _recorded(acl.GroupMemberCollection, conn)
    found = await members.members_of_groups([groups[0], groups[1]])
    plan = recorder.plans[0]
    # PostgreSQL reads the group index, or since agent_acl v010 the live-only group_members_live_by_group (a live
    # read states its NOT gone, and that index holds only the live rows asked for); YugabyteDB reads the primary key
    # (group_id HASH, id), the rows' own storage, by the same condition. Either way the read finds exactly the groups'
    # rows by group
    keyed = any(
        plan_uses_index(plan, index)
        for index in ("idx_group_members_group_id", "group_members_live_by_group", "group_members_pkey")
    )
    assert keyed and plan.rows_scanned == len(found) == 20, json.dumps(plan.plan)[:2500]
    # a group's member count reads the same index (or, on YugabyteDB, the primary key) by the group alone
    assert await members.count_by_group(groups[0]) == 10
    counted = recorder.plans[-1]
    keyed = any(
        plan_uses_index(counted, index)
        for index in ("idx_group_members_group_id", "group_members_live_by_group", "group_members_pkey")
    )
    assert keyed and counted.rows_scanned == 10, json.dumps(counted.plan)[:2500]


async def test_assignments_scope_namespace(conn: asyncpg.Connection) -> None:
    """Flips if a namespace's namespace-scope grants, or the grants a namespace's delete is about to cascade
    (read_delete_cascade, DeleteCascade on NamespaceCollection), stop reading idx_role_assignments_scope_namespace."""
    namespaces = await _seed_assignments(conn)
    assignments, recorder = _recorded(acl.RoleAssignmentCollection, conn)
    await assignments.list_scoped_to_namespace(namespaces[3])
    assert plan_uses_index(recorder.plans[0], "idx_role_assignments_scope_namespace"), recorder.plans[0].plan
    held, cascading = _recorded(acl.NamespaceCollection, conn)
    found = await read_delete_cascade(held, [namespaces[3]])
    plan = cascading.plans[0]
    granted = await conn.fetchval("SELECT count(*) FROM role_assignments WHERE scope_namespace_id = $1", namespaces[3])
    assert granted and [len(rows.keys) for rows in found] == [granted], found
    assert plan_uses_index(plan, "idx_role_assignments_scope_namespace"), json.dumps(plan.plan)[:2500]
    assert plan.rows_scanned == granted, json.dumps(plan.plan)[:2500]


async def test_assignments_scope_namespace_set(conn: asyncpg.Connection) -> None:
    """Flips if the grants on a set of namespaces in both partitions, the namespace index's guard stated (the hub's
    outside grants: ``scope_namespace_id = ANY(...)``, ``scope_type = 'namespace'``), stop reading
    idx_role_assignments_scope_namespace, or read more rows than the namespaces' grants. On YugabyteDB the read is
    hinted: binding no set, it was hinted to the scope-type index, every namespace grant read and filtered."""
    namespaces = await _seed_assignments(conn)
    named = namespaces[3:6]
    _, recorder = _recorded(acl.RoleAssignmentCollection, conn)
    found = await read_by(
        recorder,
        "role_assignments",
        by={
            "scope_namespace_id": AnyOf(named, cast="uuid"),
            "scope_type": "namespace",
            "row_scope": AnyOf(("platform", "customer"), cast=None),
        },
        where=[Fragment('t."scope_namespace_id" IS NOT NULL')],
        columns=("row_scope", "assignment_id", "scope_namespace_id", "group_id", "role_id"),
    )
    granted = await conn.fetchval(
        "SELECT count(*) FROM role_assignments WHERE scope_namespace_id = ANY($1::uuid[])", named
    )
    (plan,) = recorder.plans
    assert granted and len(found) == granted, (granted, len(found))
    assert plan_uses_index(plan, "idx_role_assignments_scope_namespace"), json.dumps(plan.plan)[:2500]
    assert plan.rows_scanned == granted, json.dumps(plan.plan)[:2500]


async def test_assignments_scope_type_customer(conn: asyncpg.Connection) -> None:
    """Flips if a customer's type_customer grants of one type stop reading idx_role_assignments_scope_type_customer."""
    await _seed_assignments(conn)
    assignments, recorder = _recorded(acl.RoleAssignmentCollection, conn)
    await assignments.list_of_type_customer(uuid.uuid4(), "t1")
    assert plan_uses_index(recorder.plans[0], "idx_role_assignments_scope_type_customer"), recorder.plans[0].plan


async def test_assignments_role(conn: asyncpg.Connection) -> None:
    """Flips if a role's assignments, listed or counted, stop reading role_assignments_page_by_role_scope_type (acl
    v009, the hub's v163 index)."""
    await _seed_assignments(conn)
    role = await conn.fetchval("SELECT role_id FROM roles LIMIT 1")
    assignments, recorder = _recorded(acl.RoleAssignmentCollection, conn)
    await assignments.list_by_role(role)
    await assignments.count_by_role(role)
    for plan in recorder.plans:
        assert plan_uses_index(plan, "role_assignments_page_by_role_scope_type"), plan.plan


async def test_assignments_scope_type(conn: asyncpg.Connection) -> None:
    """Flips if every grant of one scope type stops reading role_assignments_page_by_scope_type (acl v009, the hub's
    v163 index)."""
    await _seed_assignments(conn)
    # the rare scope a reverse access query weighs against every namespace: a handful of platform grants
    await conn.execute("UPDATE role_assignments SET scope_type = 'type_customer' WHERE scope_type = 'all'")
    await conn.execute(
        # an 'all' grant holds no customer (acl v005's CHECK)
        "UPDATE role_assignments SET scope_type = 'all', scope_customer_id = NULL, scope_namespace_type = NULL,"
        " customer_id = NULL WHERE assignment_id IN (SELECT assignment_id FROM role_assignments WHERE scope_type = 'type_customer'"
        " LIMIT 5)"
    )
    await conn.execute("ANALYZE role_assignments")
    assignments, recorder = _recorded(acl.RoleAssignmentCollection, conn)
    found = await assignments.list_by_scope_type(ScopeType.ALL)
    plan = recorder.plans[0]
    # PostgreSQL reads the scope-type index; YugabyteDB reads the all grants' own partial unique index (WHERE
    # scope_type = 'all'), which holds them alone. Either way the read finds exactly the scope's rows
    keyed = plan_uses_index(plan, "role_assignments_page_by_scope_type") or _scans_partial(
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
        # the scope's index (row_scope, namespace_id) (agent_acl v007), or the id's own key walked in order with the
        # scope as a filter: either way a key in order, under the page's LIMIT
        keyed = (
            plan_uses_index(plan, "idx_namespaces_scope_id")
            or plan_uses_index(plan, "namespaces_pkey")
            or plan_uses_index(plan, "uq_namespaces_namespace_id")
        )
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


async def test_restricted_delete_count(conn: asyncpg.Connection) -> None:
    """Flips if a role's delete, counting the live grants that still name it (role_assignments.role_id ON DELETE
    RESTRICT, run by 3tears for a gone mark: refuse_restricted_delete), stops reading
    role_assignments_page_by_role_scope_type."""
    await _seed_assignments(conn)
    role = await conn.fetchval("SELECT role_id FROM roles ORDER BY role_id LIMIT 1")
    roles, recorder = _recorded(acl.RoleCollection, conn)
    with pytest.raises(RestrictedDeleteError):
        await refuse_restricted_delete(roles, [role])
    assert recorder.plans, "the count sent no statement"
    for plan in recorder.plans:
        assert plan_uses_index(plan, "role_assignments_page_by_role_scope_type"), _outline(plan.plan)


async def test_revive_gone_grant(conn: asyncpg.Connection) -> None:
    """Flips if the revive of a gone grant (RoleAssignmentCollection._revive_gone_grant: one keyed UPDATE by the grant's
    whole natural key, in either row scope, of a gone row only) stops finding its row through an index of its pins:
    idx_role_assignments_group_scope or its scope's natural-key unique index, each led by the group, or a namespace
    grant's idx_role_assignments_scope_namespace (YugabyteDB's choice: a keyed write carries no hint)."""
    await _seed_assignments(conn)
    # some grants revoked (marked gone in place), so the revive's gone row is among them
    await conn.execute(
        "UPDATE role_assignments SET gone = true WHERE assignment_id IN (SELECT assignment_id FROM role_assignments"
        " ORDER BY assignment_id LIMIT 500)"
    )
    await conn.execute("ANALYZE role_assignments")
    for scope in ("namespace", "all"):
        held = await conn.fetchrow(
            "SELECT group_id, role_id, scope_namespace_id, managed_by FROM role_assignments WHERE scope_type = $1"
            " ORDER BY assignment_id LIMIT 1",
            scope,
        )
        assert held is not None
        sql, params = update_keyed_statements(
            "role_assignments",
            values={"gone": False, "date_granted": None},
            by={
                "row_scope": AnyOf(("platform", "customer"), cast=None),
                "group_id": held["group_id"],
                "role_id": held["role_id"],
                "scope_type": scope,
                "scope_namespace_id": held["scope_namespace_id"],
                "managed_by": held["managed_by"],
            },
            where=[Fragment('"gone"')],
            returning=("assignment_id", "row_scope"),
            with_gone=True,
        )[0]
        async with _rolled_back(conn):
            plan = await explain_write(conn, sql, *params)
        keyed = ["idx_role_assignments_group_scope"]
        if scope == "namespace":
            keyed += ["uq_role_assignments_namespace_grant", "idx_role_assignments_scope_namespace"]
        else:
            keyed += ["uq_role_assignments_all_grant"]
        assert any(plan_uses_index(plan, index) for index in keyed), (scope, _outline(plan.plan))


async def test_role_assignment_indexes_match_the_migrations(conn: asyncpg.Connection) -> None:
    """Flips if an index RoleAssignmentCollection's schema declares differs from the one the acl migrations create
    (its columns in order, its uniqueness, its predicate), or the migrations create one on role_assignments the schema
    does not declare. A keyed read on YugabyteDB names the declared index that serves it: a declaration that is not
    the live index names one that does not hold the read's rows, and one left out is never named.

    Each declaration is created on a scratch copy of the table, so the database spells its predicate as it spells the
    live one's; the primary key and the stamped table's restore index (its row_epoch declaration's) are not
    declared as indexes."""
    schema = acl.RoleAssignmentCollection.schema
    live_sql = (
        "SELECT c.relname AS name, i.indisunique AS is_unique, pg_get_expr(i.indpred, i.indrelid) AS predicate,"
        " array(SELECT a.attname::text FROM unnest(i.indkey::int2[]) WITH ORDINALITY AS k(attnum, n)"
        " JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum ORDER BY k.n) AS columns"
        " FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid JOIN pg_class t ON t.oid = i.indrelid"
        " WHERE t.relname = $1 AND t.relnamespace = $2::regnamespace"
    )

    async def indexes(table: str, namespace: str) -> dict[str, tuple[bool, tuple[str, ...], str | None]]:
        return {
            row["name"]: (row["is_unique"], tuple(row["columns"]), row["predicate"])
            for row in await conn.fetch(live_sql, table, namespace)
        }

    current = await conn.fetchval("SELECT current_schema()")
    live = await indexes("role_assignments", current)
    async with _rolled_back(conn):
        await conn.execute("CREATE TEMP TABLE declared_probe (LIKE role_assignments)")
        temp = await conn.fetchval("SELECT nspname FROM pg_namespace WHERE oid = pg_my_temp_schema()")
        for at, index in enumerate(schema.indexes):
            unique = "UNIQUE " if index.unique else ""
            where = "" if index.where is None else f" WHERE {index.where}"
            await conn.execute(
                f"CREATE {unique}INDEX declared_probe_{at} ON declared_probe ({', '.join(index.columns)}){where}"
            )
        probed = await indexes("declared_probe", temp)
    for at, index in enumerate(schema.indexes):
        assert index.name in live, f"{index.name} is declared but no migration creates it"
        assert live[index.name] == probed[f"declared_probe_{at}"], (index.name, live[index.name])
    constraints = {
        row["conname"]: tuple(row["columns"])
        for row in await conn.fetch(
            "SELECT con.conname, array(SELECT a.attname::text FROM unnest(con.conkey) WITH ORDINALITY AS k(attnum, n)"
            " JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum ORDER BY k.n) AS columns"
            " FROM pg_constraint con WHERE con.conrelid = 'role_assignments'::regclass AND con.contype = 'u'"
        )
    }
    assert constraints == {unique.name: tuple(unique.columns) for unique in schema.unique_constraints}, constraints
    declared = {index.name for index in schema.indexes} | set(constraints)
    undeclared = set(live) - declared - {"role_assignments_pkey", bucket_epoch_index_name("role_assignments")}
    assert not undeclared, (
        f"the migrations create {sorted(undeclared)} on role_assignments, which the schema does not declare"
    )


#: the rows a live page keeps: each page's plan reads about its own rows, never the gone ones before them
_LIVE_PAGE = 50


def _within_its_page(plan: StatementPlan) -> bool:
    """whether a page's plan read about the page's own rows (its LIMIT, and the one row past it that says another page
    follows), never the gone rows the primary key holds before them.

    :param plan: the page's plan
    :ptype plan: StatementPlan
    :return: whether it did
    :rtype: bool
    """
    return plan.rows_scanned <= 2 * (_LIVE_PAGE + 1)


async def test_group_members_live_page(conn: asyncpg.Connection) -> None:
    """Flips if a group's page of members (GroupMemberCollection.page_by_group) or a group delete's cascade read
    (GroupCollection.read_cascade) stops reading group_members_live_by_group (agent_acl v010): through the primary key
    each walks the group's gone memberships, the first 2,000 of its key order here, before its first live one."""
    await _seed(conn, "groups", 2, row_scope="'platform'", customer_id="NULL", name="'g' || g")
    groups = [row["group_id"] for row in await conn.fetch("SELECT group_id FROM groups ORDER BY group_id")]
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id, epoch)"
        " SELECT gen_random_uuid(), ($1::uuid[])[1 + g % 2], 'user', gen_random_uuid(), $2"
        " FROM generate_series(1, 10000) g",
        groups,
        next_epoch(),
    )
    await conn.execute(
        "UPDATE group_members SET gone = true WHERE id IN (SELECT id FROM group_members WHERE group_id = $1"
        " ORDER BY id LIMIT 2000)",
        groups[0],
    )
    await conn.execute("ANALYZE group_members")
    members, recorder = _recorded(acl.GroupMemberCollection, conn)
    page = await members.page_by_group(groups[0], limit=_LIVE_PAGE)
    assert len(page.entities) == _LIVE_PAGE
    plan = recorder.plans[0]
    assert plan_uses_index(plan, "group_members_live_by_group") and _within_its_page(plan), json.dumps(plan.plan)[:2500]
    held, cascading = _recorded(acl.GroupCollection, conn)
    cascade = await held.read_cascade(groups[0])
    assert len(cascade.members) == 3000, len(cascade.members)
    assert plan_uses_index(cascading.plans[0], "group_members_live_by_group"), json.dumps(cascading.plans[0].plan)[
        :2500
    ]


async def test_groups_live_page(conn: asyncpg.Connection) -> None:
    """Flips if a page of every group (GroupCollection.page_all) stops reading groups_live_by_id (agent_acl v010):
    through the primary key it walks the gone groups, the first 2,000 by id here, before its first live one."""
    await _seed(conn, "groups", row_scope="'platform'", customer_id="NULL", name="'g' || g")
    await conn.execute(
        "UPDATE groups SET gone = true WHERE group_id IN (SELECT group_id FROM groups ORDER BY group_id LIMIT 2000)"
    )
    await conn.execute("ANALYZE groups")
    groups, recorder = _recorded(acl.GroupCollection, conn)
    page = await groups.page_all(limit=_LIVE_PAGE)
    assert len(page.entities) == _LIVE_PAGE
    plan = recorder.plans[0]
    assert plan_uses_index(plan, "groups_live_by_id") and _within_its_page(plan), json.dumps(plan.plan)[:2500]


async def test_role_assignments_live_page(conn: asyncpg.Connection) -> None:
    """Flips if the customer backfill's page of grants (_customer_backfill_plan, run by
    RoleAssignmentCollection.backfill_customers) stops reading role_assignments_live_by_id (agent_acl v010): through
    the primary key it walks the revoked grants, the first 2,000 by id here, before its first live one."""
    await _seed_assignments(conn)
    await conn.execute(
        "UPDATE role_assignments SET gone = true WHERE assignment_id IN (SELECT assignment_id FROM role_assignments"
        " ORDER BY assignment_id LIMIT 2000)"
    )
    await conn.execute("ANALYZE role_assignments")
    assignments, recorder = _recorded(acl.RoleAssignmentCollection, conn)
    await assignments.backfill_customers(limit=_LIVE_PAGE)
    plan = recorder.plans[0]
    assert plan_uses_index(plan, "role_assignments_live_by_id") and _within_its_page(plan), json.dumps(plan.plan)[:2500]
