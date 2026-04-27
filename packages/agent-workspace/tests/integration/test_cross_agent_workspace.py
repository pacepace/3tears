"""integration test for WS-ACL-05 cross-agent workspace access.

rbac-task-01 Phase 3 rewired the authorize helper to delegate to the
unified evaluator in :mod:`threetears.agent.acl`. this test spins up
a real PostgreSQL via testcontainers, materializes the rbac tables
(``platform.namespaces`` + ``groups`` + ``group_members`` +
``roles`` + ``role_assignments``) + the agent-schema workspace
tables, seeds one customer with two agents (A owns the workspace, B
gets a ``WorkspaceEditor`` assignment via a singleton agent-group),
and exercises the workspace tool stack end-to-end:

- agent B's ``fs_read`` on a file in A's workspace succeeds
- agent B's ``fs_write`` succeeds (``WorkspaceEditor`` grants write)
- a third agent C under a different customer is categorically denied
  on any operation (cross-customer short-circuit)
- a user U2 within A's customer but without a group membership is
  denied on every tool (same-customer no-grant deny)

the ``AclCache`` contract the tool layer depends on is exercised via
the real :class:`threetears.agent.acl.AclCache` wired with SQL-backed
:class:`MembershipLoader` + :class:`GrantLoader` stubs resolving
directly against the rbac tables; the cache's three-layer TTL
behavior is covered by its own unit suite, this test is the
end-to-end wiring check.

journal envelope assertion: the brief calls for the phase-5 sweep to
assert ``actor_user_id=B's user``, ``calling_agent_id=B``,
``owner_agent_id=A`` surface on the journal row envelopes. since the
journal rows today carry only ``actor_id`` + ``correlation_id``, the
additional identity fields ride on the audit envelope which phase-6
(audit-task-01) will wire into the hub consumer. here we assert the
journal's ``actor_id`` is B's agent-id and the row lands in A's
schema — the signal the compliance report needs to answer "who did
what to whose data".
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest

from threetears.agent.acl import (
    AclCache,
    Group as AclGroup,
    GroupMembership,
    MemberType,
    Namespace as AclNamespace,
    Role,
    RoleAssignment,
    ScopeType,
)
from threetears.agent.tools.call_scope import ToolCallScope, enter_call_scope
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.workspace.authorize import (
    WorkspaceAccessDenied,
)


# rbac-task-01 v016 seeds ``WorkspaceEditor`` with this deterministic UUID
# (via uuid5 over NAMESPACE_DNS + "threetears.roles.WorkspaceEditor"). the
# integration test seeds the same role row so the evaluator sees its
# ``{"workspace": ["read", "write"]}`` permissions when resolving
# assignments from this test's singleton groups.
_WORKSPACE_EDITOR_ROLE_ID = UUID("f3684adb-bc37-5e79-90b9-af9c814bb29e")

# canonical testcontainer harness from threetears.core; provides
# the ``db_container`` fixture this file's ``pg_url`` alias wraps.


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# testcontainer + schema setup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def db_image() -> str:
    """pin pgvector/pg16; this suite exercises the ``vector`` codec path."""
    return "pgvector/pgvector:pg16"


@pytest.fixture(scope="module")
def pg_url(db_container: str) -> str:
    """alias for the canonical ``db_container`` fixture from
    :mod:`threetears.core.testing.fixtures` (test-harness-task-01).

    :param db_container: canonical session-scoped DB URL
    :ptype db_container: str
    :return: asyncpg-compatible URL
    :rtype: str
    """
    return db_container


def _schema_name(agent_id: UUID) -> str:
    """agent schema name from UUID hex.

    :param agent_id: agent UUID
    :ptype agent_id: UUID
    :return: ``agent_<hex>`` schema name
    :rtype: str
    """
    return f"agent_{agent_id.hex}"


async def _build_platform_schema(conn: asyncpg.Connection) -> None:
    """create the unified rbac tables + seed ``WorkspaceEditor``.

    rbac-task-01 Phase 3 replaced ``namespace_grants`` with ``groups``
    / ``group_members`` / ``roles`` / ``role_assignments``. the
    minimal DDL here mirrors the v016 migration without the invariant
    checks so the test seeds directly into the new shape.

    :param conn: asyncpg connection
    :ptype conn: asyncpg.Connection
    """
    await conn.execute("CREATE SCHEMA IF NOT EXISTS platform")
    await conn.execute(
        """
        CREATE TABLE platform.namespaces (
            id            UUID PRIMARY KEY,
            name          TEXT NOT NULL,
            namespace_type TEXT NOT NULL,
            owner_agent_id UUID NOT NULL,
            schema_name   TEXT,
            customer_id   UUID,
            metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
            date_created  TIMESTAMPTZ NOT NULL,
            date_updated  TIMESTAMPTZ NOT NULL
        )
        """,
    )
    await conn.execute(
        """
        CREATE TABLE platform.groups (
            id UUID PRIMARY KEY,
            customer_id UUID,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            date_created TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            date_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    )
    await conn.execute(
        """
        CREATE TABLE platform.group_members (
            id UUID PRIMARY KEY,
            group_id UUID NOT NULL REFERENCES platform.groups(id) ON DELETE CASCADE,
            member_type VARCHAR(10) NOT NULL
                CHECK (member_type IN ('user', 'agent')),
            member_id UUID NOT NULL,
            customer_id UUID,
            date_added TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    )
    await conn.execute(
        """
        CREATE TABLE platform.roles (
            id UUID PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT,
            permissions JSONB NOT NULL,
            is_builtin BOOLEAN NOT NULL DEFAULT FALSE,
            date_created TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            date_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    )
    await conn.execute(
        """
        CREATE TABLE platform.role_assignments (
            id UUID PRIMARY KEY,
            role_id UUID NOT NULL REFERENCES platform.roles(id),
            group_id UUID NOT NULL REFERENCES platform.groups(id) ON DELETE CASCADE,
            scope_type VARCHAR(16) NOT NULL
                CHECK (scope_type IN ('namespace', 'type_customer', 'all')),
            scope_namespace_id UUID REFERENCES platform.namespaces(id) ON DELETE CASCADE,
            scope_namespace_type VARCHAR(255),
            scope_customer_id UUID,
            granted_by UUID,
            date_granted TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    )
    # seed the WorkspaceEditor role with the production-matching id +
    # permissions so the evaluator resolves correctly. namespace-task-01
    # phase 7 added the path-glob-bearing ``read_file_matching:<glob>``
    # / ``write_file_matching:<glob>`` action types on the ``workspace``
    # bucket; per-file gates require those alongside the workspace-level
    # ``read`` / ``write``.
    await conn.execute(
        """
        INSERT INTO platform.roles (id, name, description, permissions, is_builtin)
        VALUES ($1, 'WorkspaceEditor',
                'Read and write on workspaces and workspace files.',
                '{"workspace": ["read", "write", "read_file_matching:**/*", "write_file_matching:**/*"], "workspace_file": ["read", "write"]}'::jsonb,
                TRUE)
        """,
        _WORKSPACE_EDITOR_ROLE_ID,
    )


async def _build_agent_schema(conn: asyncpg.Connection, agent_id: UUID) -> None:
    """
    materialize the agent-schema workspace tables.

    :param conn: asyncpg connection
    :ptype conn: asyncpg.Connection
    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    """
    schema = _schema_name(agent_id)
    await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    await conn.execute(
        f"""
        CREATE TABLE "{schema}".workspaces (
            id              UUID PRIMARY KEY,
            agent_id        UUID NOT NULL,
            name            TEXT NOT NULL,
            description     TEXT,
            template_name   TEXT,
            created_by      UUID NOT NULL,
            current_version INTEGER NOT NULL DEFAULT 0,
            date_created    TIMESTAMPTZ NOT NULL,
            date_updated    TIMESTAMPTZ NOT NULL,
            date_deleted    TIMESTAMPTZ
        )
        """,
    )
    await conn.execute(
        f"""
        CREATE TABLE "{schema}".workspace_files (
            id              UUID PRIMARY KEY,
            workspace_id    UUID NOT NULL REFERENCES "{schema}".workspaces(id)
                               ON DELETE CASCADE,
            relative_path   TEXT NOT NULL,
            content         BYTEA NOT NULL,
            sha256          TEXT NOT NULL,
            version         INTEGER NOT NULL,
            date_updated    TIMESTAMPTZ NOT NULL,
            UNIQUE (workspace_id, relative_path)
        )
        """,
    )
    await conn.execute(
        f"""
        CREATE TABLE "{schema}".workspace_file_versions (
            id              UUID PRIMARY KEY,
            workspace_id    UUID NOT NULL REFERENCES "{schema}".workspaces(id)
                               ON DELETE CASCADE,
            relative_path   TEXT NOT NULL,
            version         INTEGER NOT NULL,
            content         BYTEA NOT NULL,
            sha256          TEXT NOT NULL,
            action          TEXT NOT NULL,
            label           TEXT,
            actor_id        UUID NOT NULL,
            correlation_id  UUID NOT NULL,
            date_created    TIMESTAMPTZ NOT NULL,
            UNIQUE (workspace_id, relative_path, version)
        )
        """,
    )


# ---------------------------------------------------------------------------
# ACL cache shim
# ---------------------------------------------------------------------------


class _SqlMembershipLoader:
    """SQL-backed :class:`MembershipLoader` for the integration test.

    queries ``platform.group_members`` and wraps rows as shared
    :class:`GroupMembership` dataclasses the evaluator consumes.

    :param pool: asyncpg pool bound to the test database
    :ptype pool: asyncpg.Pool
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def load_for_user(
        self, user_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return every membership row naming ``user_id`` as a user member.

        :param user_id: user UUID
        :ptype user_id: UUID
        :return: tuple of memberships (possibly empty)
        :rtype: tuple[GroupMembership, ...]
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT group_id, member_type, member_id, customer_id
                  FROM platform.group_members
                 WHERE member_type = 'user' AND member_id = $1
                """,
                user_id,
            )
        return tuple(
            GroupMembership(
                group_id=row["group_id"],
                member_type=MemberType(row["member_type"]),
                member_id=row["member_id"],
                customer_id=row["customer_id"],
            )
            for row in rows
        )

    async def load_for_agent(
        self, agent_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return every membership row naming ``agent_id`` as an agent member.

        :param agent_id: agent UUID
        :ptype agent_id: UUID
        :return: tuple of memberships (possibly empty)
        :rtype: tuple[GroupMembership, ...]
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT group_id, member_type, member_id, customer_id
                  FROM platform.group_members
                 WHERE member_type = 'agent' AND member_id = $1
                """,
                agent_id,
            )
        return tuple(
            GroupMembership(
                group_id=row["group_id"],
                member_type=MemberType(row["member_type"]),
                member_id=row["member_id"],
                customer_id=row["customer_id"],
            )
            for row in rows
        )


class _SqlGrantLoader:
    """SQL-backed :class:`GrantLoader` for the integration test.

    queries ``platform.role_assignments``, ``platform.roles``, and
    ``platform.groups`` and wraps rows as shared dataclasses the
    evaluator consumes. mirrors the production
    :class:`aibots.hub.broker.acl.HubGrantLoader` without the TTL
    cache.

    :param pool: asyncpg pool bound to the test database
    :ptype pool: asyncpg.Pool
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace: AclNamespace,
    ) -> tuple[RoleAssignment, ...]:
        """return assignments held by ``group_ids`` that could cover ``namespace``.

        :param group_ids: groups to inspect
        :ptype group_ids: tuple[UUID, ...]
        :param namespace: namespace under evaluation
        :ptype namespace: AclNamespace
        :return: tuple of assignments
        :rtype: tuple[RoleAssignment, ...]
        """
        if not group_ids:
            return ()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role_id, group_id, scope_type,
                       scope_namespace_id, scope_namespace_type, scope_customer_id
                  FROM platform.role_assignments
                 WHERE group_id = ANY($1::uuid[])
                   AND (
                          (scope_type = 'namespace'
                            AND scope_namespace_id = $2)
                       OR (scope_type = 'type_customer'
                            AND scope_namespace_type = $3
                            AND scope_customer_id = $4)
                       OR  scope_type = 'all'
                       )
                """,
                list(group_ids),
                namespace.id,
                namespace.namespace_type,
                namespace.customer_id,
            )
        return tuple(
            RoleAssignment(
                id=row["id"],
                role_id=row["role_id"],
                group_id=row["group_id"],
                scope_type=ScopeType(row["scope_type"]),
                scope_namespace_id=row["scope_namespace_id"],
                scope_namespace_type=row["scope_namespace_type"],
                scope_customer_id=row["scope_customer_id"],
            )
            for row in rows
        )

    async def load_roles(
        self, role_ids: tuple[UUID, ...],
    ) -> dict[UUID, Role]:
        """resolve role ids into :class:`Role` rows with coerced permissions.

        :param role_ids: role UUIDs to resolve
        :ptype role_ids: tuple[UUID, ...]
        :return: mapping role_id -> Role for ids that exist
        :rtype: dict[UUID, Role]
        """
        if not role_ids:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, permissions, is_builtin
                  FROM platform.roles
                 WHERE id = ANY($1::uuid[])
                """,
                list(role_ids),
            )
        result: dict[UUID, Role] = {}
        for row in rows:
            raw = row["permissions"]
            parsed: dict[str, Any] = {}
            if isinstance(raw, dict):
                parsed = raw
            elif isinstance(raw, str) and raw:
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    parsed = loaded
            permissions: dict[str, frozenset[str]] = {}
            for resource_type, actions in parsed.items():
                if isinstance(actions, list):
                    permissions[resource_type] = frozenset(
                        str(a) for a in actions
                    )
            result[row["id"]] = Role(
                id=row["id"],
                name=row["name"],
                permissions=permissions,
                is_built_in=bool(row["is_builtin"]),
            )
        return result

    async def load_groups(
        self, group_ids: tuple[UUID, ...],
    ) -> dict[UUID, object]:
        """resolve group ids into :class:`AclGroup` rows.

        :param group_ids: group UUIDs to resolve
        :ptype group_ids: tuple[UUID, ...]
        :return: mapping group_id -> :class:`AclGroup`
        :rtype: dict[UUID, object]
        """
        if not group_ids:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, customer_id
                  FROM platform.groups
                 WHERE id = ANY($1::uuid[])
                """,
                list(group_ids),
            )
        result: dict[UUID, object] = {}
        for row in rows:
            result[row["id"]] = AclGroup(
                id=row["id"], name=row["name"], customer_id=row["customer_id"],
            )
        return result


def _build_sql_backed_acl_cache(pool: asyncpg.Pool) -> AclCache:
    """build a real :class:`AclCache` backed by direct-SQL loaders.

    production wiring uses the same :class:`AclCache` fronted by
    :class:`HubMembershipLoader` / :class:`HubGrantLoader` over
    Collections; this integration test hits the rbac platform tables
    directly via :class:`_SqlMembershipLoader` /
    :class:`_SqlGrantLoader` so the tool -> authorize -> evaluator
    surface is exercised without the broker.

    :param pool: asyncpg pool bound to the test database
    :ptype pool: asyncpg.Pool
    :return: cache instance with loaders wired
    :rtype: AclCache
    """
    return AclCache(
        membership_loader=_SqlMembershipLoader(pool),
        grant_loader=_SqlGrantLoader(pool),
        ttl_seconds=60,
    )


# ---------------------------------------------------------------------------
# WorkspaceCollection stub shaped for the tools
# ---------------------------------------------------------------------------


class _SqlWorkspaceCollection:
    """tiny collection querying a specific agent schema directly.

    the tools call ``find_by_agent_and_name`` / ``find_by_id_and_agent``.
    this test stub issues SQL against a fixed ``search_path`` and
    returns :class:`_SqlWorkspace` entities that satisfy the
    ``WorkspaceLike`` protocol consumed by authorize + the tools.
    """

    def __init__(self, pool: asyncpg.Pool, schema: str) -> None:
        """capture the pool + target schema.

        :param pool: asyncpg pool
        :ptype pool: asyncpg.Pool
        :param schema: agent schema name (``agent_<hex>``)
        :ptype schema: str
        """
        self._pool = pool
        self._schema = schema

    async def find_by_agent_and_name(
        self,
        agent_id: UUID,
        name: str,
    ) -> "_SqlWorkspace | None":
        """SELECT workspace by agent + name within the bound schema.

        :param agent_id: owning agent
        :ptype agent_id: UUID
        :param name: workspace name
        :ptype name: str
        :return: workspace entity or None
        :rtype: _SqlWorkspace | None
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT * FROM "{self._schema}".workspaces '
                "WHERE agent_id = $1 AND name = $2",
                agent_id,
                name,
            )
        if row is None:
            return None
        return _SqlWorkspace(dict(row))

    async def find_by_id_and_agent(
        self,
        workspace_id: UUID,
        agent_id: UUID,
    ) -> "_SqlWorkspace | None":
        """SELECT workspace by id + agent within the bound schema.

        :param workspace_id: workspace UUID
        :ptype workspace_id: UUID
        :param agent_id: owning agent
        :ptype agent_id: UUID
        :return: workspace entity or None
        :rtype: _SqlWorkspace | None
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT * FROM "{self._schema}".workspaces '
                "WHERE id = $1 AND agent_id = $2",
                workspace_id,
                agent_id,
            )
        if row is None:
            return None
        return _SqlWorkspace(dict(row))


class _SqlWorkspace:
    """thin read-only row wrapper exposing the WorkspaceLike surface."""

    def __init__(self, data: dict[str, Any]) -> None:
        """capture row dict.

        :param data: asyncpg row dict
        :ptype data: dict[str, Any]
        """
        self._data = data

    @property
    def id(self) -> UUID:
        """workspace uuid."""
        return self._data["id"]

    @property
    def name(self) -> str:
        """human name."""
        return self._data["name"]

    @property
    def agent_id(self) -> UUID:
        """owning-agent uuid."""
        return self._data["agent_id"]

    @property
    def owner_agent_id(self) -> UUID:
        """owner alias."""
        return self._data["agent_id"]

    @property
    def created_by_user_id(self) -> UUID:
        """creator user alias."""
        return self._data["created_by"]

    @property
    def namespace_name(self) -> str:
        """canonical namespace key."""
        return f"workspace.{self._data['id']}"

    @property
    def customer_id(self) -> UUID | None:
        """customer uuid — stamped by enrich helper at authorize time."""
        return self._data.get("customer_id")

    @customer_id.setter
    def customer_id(self, value: UUID | None) -> None:
        """set customer_id on the row dict.

        :param value: customer UUID
        :ptype value: UUID | None
        """
        self._data["customer_id"] = value

    @property
    def date_deleted(self) -> datetime | None:
        """soft-delete marker (None = live)."""
        return self._data.get("date_deleted")


class _SqlWorkspaceFileCollection:
    """file collection over the bound agent schema."""

    def __init__(self, pool: asyncpg.Pool, schema: str) -> None:
        """capture pool + schema.

        :param pool: asyncpg pool
        :ptype pool: asyncpg.Pool
        :param schema: agent schema
        :ptype schema: str
        """
        self._pool = pool
        self._schema = schema

    async def find_by_workspace_and_relative_path(
        self,
        workspace_id: UUID,
        relative_path: str,
    ) -> "_SqlWorkspaceFile | None":
        """SELECT a head-state file.

        :param workspace_id: workspace id
        :ptype workspace_id: UUID
        :param relative_path: relative path
        :ptype relative_path: str
        :return: file entity or None
        :rtype: _SqlWorkspaceFile | None
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT * FROM "{self._schema}".workspace_files '
                "WHERE workspace_id = $1 AND relative_path = $2",
                workspace_id,
                relative_path,
            )
        if row is None:
            return None
        return _SqlWorkspaceFile(dict(row))


class _SqlWorkspaceFile:
    """thin read-only row wrapper for file head-state."""

    def __init__(self, data: dict[str, Any]) -> None:
        """capture row.

        :param data: asyncpg row dict
        :ptype data: dict[str, Any]
        """
        self._data = data

    @property
    def relative_path(self) -> str:
        """workspace-relative path."""
        return self._data["relative_path"]

    @property
    def content(self) -> bytes:
        """raw bytes."""
        return bytes(self._data["content"])

    @property
    def sha256(self) -> str:
        """sha256 hex."""
        return self._data["sha256"]

    @property
    def version(self) -> int:
        """version."""
        return self._data["version"]

    @property
    def date_updated(self) -> datetime:
        """last update timestamp."""
        return self._data["date_updated"]


# ---------------------------------------------------------------------------
# schema-bound pool wrapper: the tools expect a pool whose default
# search_path points at the OWNER's schema. we wrap asyncpg.Pool with
# a small facade that sets search_path on every acquire().
# ---------------------------------------------------------------------------


class _SchemaBoundPool:
    """asyncpg pool facade that binds ``search_path`` on acquire.

    the production path routes through the L3 proxy; in this test we
    bypass NATS and go straight to asyncpg. each acquire() sets
    search_path to ``"<owner_schema>", platform`` so SELECTs / INSERTs
    on ``workspaces`` + ``workspace_files`` land in the owner's
    schema, and references to ``platform.namespaces`` resolve.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        schema: str,
        namespaces: dict[str, str] | None = None,
    ) -> None:
        """capture the pool + schema + namespace -> schema mapping.

        :param pool: underlying asyncpg pool
        :ptype pool: asyncpg.Pool
        :param schema: owner agent schema name bound by default on
            each acquire (used for non-tx reads)
        :ptype schema: str
        :param namespaces: optional mapping of workspace namespace
            name -> schema name used to resolve
            ``conn.transaction(namespace=...)`` to the right target
            schema. absent mappings fall back to ``schema``.
        :ptype namespaces: dict[str, str] | None
        """
        self._pool = pool
        self._schema = schema
        self._namespaces = namespaces or {}

    @property
    def raw_pool(self) -> asyncpg.Pool:
        """expose the underlying asyncpg pool for the tx acquire helper.

        :return: underlying pool
        :rtype: asyncpg.Pool
        """
        return self._pool

    def resolve_namespace_schema(self, namespace: str) -> str:
        """look up the target schema for a workspace namespace.

        :param namespace: canonical namespace name
        :ptype namespace: str
        :return: schema name to bind into search_path
        :rtype: str
        """
        return self._namespaces.get(namespace, self._schema)

    async def fetchrow(
        self,
        query: str,
        *args: Any,
        namespace: str | None = None,
    ) -> Any:
        """delegate to pool with schema-bound search_path.

        :param query: SQL
        :ptype query: str
        :param args: params
        :ptype args: Any
        :param namespace: ignored in test
        :ptype namespace: str | None
        :return: row or None
        :rtype: Any
        """
        del namespace
        async with self._pool.acquire() as conn:
            await conn.execute(
                f'SET search_path TO "{self._schema}", platform',
            )
            return await conn.fetchrow(query, *args)

    async def fetch(
        self,
        query: str,
        *args: Any,
        namespace: str | None = None,
    ) -> Any:
        """delegate to pool with schema-bound search_path.

        :param query: SQL
        :ptype query: str
        :param args: params
        :ptype args: Any
        :param namespace: ignored in test
        :ptype namespace: str | None
        :return: rows
        :rtype: Any
        """
        del namespace
        async with self._pool.acquire() as conn:
            await conn.execute(
                f'SET search_path TO "{self._schema}", platform',
            )
            return await conn.fetch(query, *args)

    async def execute(
        self,
        query: str,
        *args: Any,
        namespace: str | None = None,
    ) -> Any:
        """delegate to pool with schema-bound search_path.

        :param query: SQL
        :ptype query: str
        :param args: params
        :ptype args: Any
        :param namespace: ignored in test
        :ptype namespace: str | None
        :return: status tag
        :rtype: Any
        """
        del namespace
        async with self._pool.acquire() as conn:
            await conn.execute(
                f'SET search_path TO "{self._schema}", platform',
            )
            return await conn.execute(query, *args)

    def acquire(self) -> Any:
        """delegate to pool; caller sets search_path inside.

        WS-ACL-06: the returned acquire CM yields a
        :class:`_NamespaceTxWrapper` rather than the raw asyncpg
        connection so the tools' ``conn.transaction(namespace=...)``
        call dispatches to the schema-rebinding CM.

        :return: acquire context manager
        :rtype: Any
        """
        return _SchemaBoundAcquire(self, self._schema)


class _NamespaceTxWrapper:
    """wraps an asyncpg connection to honor ``.transaction(namespace=...)``.

    the production :class:`_ProxyConnection` rebinds search_path to
    the namespace's schema at tx.begin time. this test runs against
    real asyncpg (no proxy), so we rebind search_path here at
    ``transaction()`` entry when the caller supplies a workspace
    namespace. absent a namespace the connection's already-bound
    search_path (set at acquire time) is preserved.
    """

    def __init__(self, conn: asyncpg.Connection, pool: Any) -> None:
        """capture the underlying connection.

        :param conn: asyncpg connection
        :ptype conn: asyncpg.Connection
        :param pool: owning schema-bound pool (for namespace -> schema lookup)
        :ptype pool: Any
        """
        self._conn = conn
        self._pool = pool

    def transaction(self, namespace: str | None = None) -> Any:
        """return a transaction CM, optionally rebinding search_path.

        :param namespace: workspace namespace name or None
        :ptype namespace: str | None
        :return: asyncpg transaction context manager
        :rtype: Any
        """
        if namespace is not None:
            target_schema = self._pool.resolve_namespace_schema(namespace)
            # set search_path synchronously before tx.start; must be
            # wrapped in a thin async CM because SET must await.
            return _NamespaceTxCM(self._conn, target_schema)
        return self._conn.transaction()

    def __getattr__(self, name: str) -> Any:
        """delegate everything else to the underlying connection.

        :param name: attribute name
        :ptype name: str
        :return: delegated attribute
        :rtype: Any
        """
        return getattr(self._conn, name)


class _NamespaceTxCM:
    """rebinds search_path on enter, then opens a real asyncpg tx."""

    def __init__(self, conn: asyncpg.Connection, schema: str) -> None:
        """capture conn + target schema.

        :param conn: connection
        :ptype conn: asyncpg.Connection
        :param schema: target schema to bind
        :ptype schema: str
        """
        self._conn = conn
        self._schema = schema
        self._tx: Any = None

    async def __aenter__(self) -> Any:
        """rebind search_path then start the tx.

        :return: transaction object
        :rtype: Any
        """
        await self._conn.execute(
            f'SET search_path TO "{self._schema}", platform',
        )
        self._tx = self._conn.transaction()
        return await self._tx.__aenter__()

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """commit / rollback via the underlying asyncpg transaction.

        :param exc_type: exception type or None
        :ptype exc_type: Any
        :param exc_val: value or None
        :ptype exc_val: Any
        :param exc_tb: traceback or None
        :ptype exc_tb: Any
        """
        if self._tx is not None:
            await self._tx.__aexit__(exc_type, exc_val, exc_tb)


class _SchemaBoundAcquire:
    """async CM binding search_path on a freshly-acquired connection."""

    def __init__(self, pool: Any, schema: str) -> None:
        """capture pool + schema.

        :param pool: pool (:class:`_SchemaBoundPool`)
        :ptype pool: Any
        :param schema: schema
        :ptype schema: str
        """
        self._pool = pool
        self._schema = schema
        self._conn: asyncpg.Connection | None = None
        self._cm: Any = None

    async def __aenter__(self) -> _NamespaceTxWrapper:
        """acquire + SET search_path.

        :return: connection wrapped to honor ``transaction(namespace=...)``
        :rtype: _NamespaceTxWrapper
        """
        self._cm = self._pool.raw_pool.acquire()
        self._conn = await self._cm.__aenter__()
        await self._conn.execute(
            f'SET search_path TO "{self._schema}", platform',
        )
        return _NamespaceTxWrapper(self._conn, self._pool)

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """release connection.

        :param exc_type: type or None
        :ptype exc_type: Any
        :param exc_val: value or None
        :ptype exc_val: Any
        :param exc_tb: traceback or None
        :ptype exc_tb: Any
        """
        if self._cm is not None:
            await self._cm.__aexit__(exc_type, exc_val, exc_tb)


# ---------------------------------------------------------------------------
# seed helpers
# ---------------------------------------------------------------------------


async def _seed_namespace_row(
    conn: asyncpg.Connection,
    *,
    workspace_id: UUID,
    owner_agent_id: UUID,
    customer_id: UUID,
) -> None:
    """insert the paired platform.namespaces row for a workspace.

    :param conn: connection
    :ptype conn: asyncpg.Connection
    :param workspace_id: workspace UUID (shared with namespace row)
    :ptype workspace_id: UUID
    :param owner_agent_id: owner agent
    :ptype owner_agent_id: UUID
    :param customer_id: customer UUID
    :ptype customer_id: UUID
    """
    now = datetime.now(UTC)
    await conn.execute(
        """
        INSERT INTO platform.namespaces (
            id, name, namespace_type, owner_agent_id, schema_name,
            customer_id, metadata, date_created, date_updated
        ) VALUES ($1, $2, 'workspace', $3, $4, $5, '{}'::jsonb, $6, $7)
        """,
        workspace_id,
        f"workspace.{workspace_id}",
        owner_agent_id,
        _schema_name(owner_agent_id),
        customer_id,
        now,
        now,
    )


async def _grant_via_singleton_group(
    conn: asyncpg.Connection,
    *,
    namespace_id: UUID,
    principal_type: str,
    principal_id: UUID,
    customer_id: UUID,
    role_id: UUID = _WORKSPACE_EDITOR_ROLE_ID,
) -> tuple[UUID, UUID]:
    """seed a singleton group + member + namespace-scope role assignment.

    mirrors v016 backfill semantics: one singleton group per
    principal (``agent:<uuid>`` or ``user:<uuid>``), one member row,
    one ``role_assignments`` row scoped to a specific namespace.

    :param conn: connection
    :ptype conn: asyncpg.Connection
    :param namespace_id: namespace UUID to scope the assignment to
    :ptype namespace_id: UUID
    :param principal_type: ``"user"`` or ``"agent"``
    :ptype principal_type: str
    :param principal_id: principal UUID to add as the group's member
    :ptype principal_id: UUID
    :param customer_id: customer UUID stamped on the group + member
    :ptype customer_id: UUID
    :param role_id: role UUID the assignment binds; defaults to the
        seeded ``WorkspaceEditor``
    :ptype role_id: UUID
    :return: tuple of ``(group_id, assignment_id)``
    :rtype: tuple[UUID, UUID]
    """
    group_id = uuid4()
    member_id = uuid4()
    assignment_id = uuid4()
    group_name = f"{principal_type}:{principal_id}"
    now = datetime.now(UTC)
    await conn.execute(
        """
        INSERT INTO platform.groups (
            id, customer_id, name, description, date_created, date_updated
        ) VALUES ($1, $2, $3, $4, $5, $6)
        """,
        group_id, customer_id, group_name,
        f"test singleton group for {group_name}", now, now,
    )
    await conn.execute(
        """
        INSERT INTO platform.group_members (
            id, group_id, member_type, member_id, customer_id, date_added
        ) VALUES ($1, $2, $3, $4, $5, $6)
        """,
        member_id, group_id, principal_type, principal_id, customer_id, now,
    )
    await conn.execute(
        """
        INSERT INTO platform.role_assignments (
            id, role_id, group_id, scope_type,
            scope_namespace_id, scope_namespace_type, scope_customer_id,
            granted_by, date_granted
        )
        VALUES ($1, $2, $3, 'namespace', $4, NULL, NULL, NULL, $5)
        """,
        assignment_id, role_id, group_id, namespace_id, now,
    )
    return group_id, assignment_id


async def _seed_workspace(
    conn: asyncpg.Connection,
    *,
    workspace_id: UUID,
    agent_id: UUID,
    user_id: UUID,
    name: str,
) -> None:
    """insert a row into agent schema's workspaces + seed one file.

    :param conn: connection
    :ptype conn: asyncpg.Connection
    :param workspace_id: workspace UUID
    :ptype workspace_id: UUID
    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param user_id: creator user UUID
    :ptype user_id: UUID
    :param name: workspace name
    :ptype name: str
    """
    now = datetime.now(UTC)
    schema = _schema_name(agent_id)
    await conn.execute(
        f"""
        INSERT INTO "{schema}".workspaces (
            id, agent_id, name, description, template_name, created_by,
            current_version, date_created, date_updated
        ) VALUES ($1, $2, $3, NULL, NULL, $4, 1, $5, $6)
        """,
        workspace_id,
        agent_id,
        name,
        user_id,
        now,
        now,
    )
    import hashlib

    content = b"initial content"
    sha = hashlib.sha256(content).hexdigest()
    file_id = uuid4()
    version_id = uuid4()
    await conn.execute(
        f"""
        INSERT INTO "{schema}".workspace_files (
            id, workspace_id, relative_path, content, sha256, version, date_updated
        ) VALUES ($1, $2, $3, $4, $5, 1, $6)
        """,
        file_id,
        workspace_id,
        "hello.txt",
        content,
        sha,
        now,
    )
    await conn.execute(
        f"""
        INSERT INTO "{schema}".workspace_file_versions (
            id, workspace_id, relative_path, version, content, sha256,
            action, label, actor_id, correlation_id, date_created
        ) VALUES ($1, $2, $3, 1, $4, $5, 'create', NULL, $6, $7, $8)
        """,
        version_id,
        workspace_id,
        "hello.txt",
        content,
        sha,
        agent_id,
        uuid4(),
        now,
    )


# ---------------------------------------------------------------------------
# test module
# ---------------------------------------------------------------------------


class _CrossAgentWorkspaceCollection:
    """workspace collection that resolves by ``(workspace_name)`` alone.

    production tools receive the workspace name via the discovery
    subject (phase 5a), which returns the workspace id + owner_agent_id
    together. this test stub emulates the shape the tool needs
    *after* discovery: ``find_by_agent_and_name(caller_agent, name)``
    returns the workspace owned by whichever agent actually created it,
    so cross-agent access patterns exercise the authorize hot path
    without requiring full discovery wiring in this integration run.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        schema: str,
        expected_workspace_id: UUID,
        owner_agent_id: UUID,
    ) -> None:
        """capture pool + schema + expected workspace.

        :param pool: asyncpg pool
        :ptype pool: asyncpg.Pool
        :param schema: owner agent schema
        :ptype schema: str
        :param expected_workspace_id: the workspace we expect calls
            to resolve to (matches the seed)
        :ptype expected_workspace_id: UUID
        :param owner_agent_id: the owner agent; used in SELECT
            so the collection returns the workspace regardless of
            the calling agent's ID
        :ptype owner_agent_id: UUID
        """
        self._pool = pool
        self._schema = schema
        self._ws_id = expected_workspace_id
        self._owner = owner_agent_id

    async def find_by_agent_and_name(
        self,
        agent_id: UUID,
        name: str,
    ) -> "_SqlWorkspace | None":
        """look up by owner_agent + name.

        :param agent_id: caller agent id (ignored; owner id is used)
        :ptype agent_id: UUID
        :param name: workspace name
        :ptype name: str
        :return: workspace or None
        :rtype: _SqlWorkspace | None
        """
        del agent_id
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT * FROM "{self._schema}".workspaces '
                "WHERE agent_id = $1 AND name = $2",
                self._owner,
                name,
            )
        if row is None:
            return None
        return _SqlWorkspace(dict(row))

    async def find_by_id_and_agent(
        self,
        workspace_id: UUID,
        agent_id: UUID,
    ) -> "_SqlWorkspace | None":
        """look up by workspace_id + owner.

        :param workspace_id: workspace uuid
        :ptype workspace_id: UUID
        :param agent_id: caller agent (ignored)
        :ptype agent_id: UUID
        :return: workspace or None
        :rtype: _SqlWorkspace | None
        """
        del agent_id
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT * FROM "{self._schema}".workspaces '
                "WHERE id = $1 AND agent_id = $2",
                workspace_id,
                self._owner,
            )
        if row is None:
            return None
        return _SqlWorkspace(dict(row))


@pytest.mark.asyncio
async def test_cross_agent_grantee_can_read_and_write(pg_url: str) -> None:
    """agent B with a grant on A's workspace can fs_read + fs_write.

    asserts WS-ACL-05 + WS-ACL-06 end-to-end: B resolves A's
    workspace via a post-discovery-shaped collection stub, the
    authorize helper consults platform.namespace_grants and allows,
    reads/writes land in A's schema. the journal row carries
    ``actor_id = B`` so compliance can answer "who did what to
    whose data".
    """
    from threetears.agent.workspace.tools.fs_read import FsReadTool
    from threetears.agent.workspace.tools.fs_write import FsWriteTool

    from threetears.core.collections import init_connection

    # --- infra ---
    pool = await asyncpg.create_pool(
        pg_url, min_size=1, max_size=4, init=init_connection,
    )
    assert pool is not None
    try:
        async with pool.acquire() as conn:
            await _build_platform_schema(conn)

        # --- identities ---
        customer_id = uuid4()
        agent_a = uuid4()
        agent_b = uuid4()
        agent_c = uuid4()  # different customer
        user_a = uuid4()
        user_b = uuid4()
        user_u2 = uuid4()  # same customer as A, no grant

        async with pool.acquire() as conn:
            await _build_agent_schema(conn, agent_a)
            await _build_agent_schema(conn, agent_b)
            await _build_agent_schema(conn, agent_c)

        # --- workspace + grant seed ---
        workspace_id = uuid4()
        async with pool.acquire() as conn:
            await _seed_workspace(
                conn,
                workspace_id=workspace_id,
                agent_id=agent_a,
                user_id=user_a,
                name="ws",
            )
            await _seed_namespace_row(
                conn,
                workspace_id=workspace_id,
                owner_agent_id=agent_a,
                customer_id=customer_id,
            )
            # rbac-task-01 Phase 3: the unified evaluator intersects the
            # user and agent sides. to model the old "agent-wide write
            # grant lets B's user call through", both agent B and user B
            # need group memberships that lead to an assignment on the
            # namespace. we seed two singleton groups (agent:<B> +
            # user:<B>), each with its own WorkspaceEditor assignment on
            # the workspace namespace. this matches the production shape
            # where a grant visible to user_b via agent_b requires user_b
            # to be in a group with an assignment covering the namespace.
            await _grant_via_singleton_group(
                conn,
                namespace_id=workspace_id,
                principal_type="agent",
                principal_id=agent_b,
                customer_id=customer_id,
            )
            await _grant_via_singleton_group(
                conn,
                namespace_id=workspace_id,
                principal_type="user",
                principal_id=user_b,
                customer_id=customer_id,
            )

        # --- build tools ---
        acl_cache = _build_sql_backed_acl_cache(pool)
        owner_schema = _schema_name(agent_a)
        # WS-ACL-06: expose the workspace namespace -> owner schema
        # mapping so grantee-side _write_file_atomic calls that pass
        # ``namespace=workspace.namespace_name`` on the transaction
        # route to the OWNER's schema via _NamespaceTxWrapper.
        schema_pool = _SchemaBoundPool(
            pool,
            owner_schema,
            namespaces={f"workspace.{workspace_id}": owner_schema},
        )
        # the workspace collection is bound to the OWNER's schema;
        # for cross-agent calls the grantee resolves the workspace by
        # a name+owner-id pair (the production discovery flow returns
        # owner_agent_id from the discovery subject). here we emulate
        # that: for cross-agent access tests we look up by
        # workspace_id + owner's agent_id. for owner-path tests we
        # look up by caller's agent_id.
        ws_coll = _CrossAgentWorkspaceCollection(
            pool,
            owner_schema,
            expected_workspace_id=workspace_id,
            owner_agent_id=agent_a,
        )
        file_coll = _SqlWorkspaceFileCollection(pool, owner_schema)

        # permissive sandbox + null context (no pin needed, we pass
        # workspace= kwarg explicitly). namespace-task-01 phase 7
        # replaced the legacy ``enforce`` / ``check_relative_key``
        # surface with ``validate_syntax``; the noop still accepts
        # every path.
        class _NoopSandbox:
            def validate_syntax(self, key: str) -> None:
                del key

        class _NullContext:
            pass

        read_tool = FsReadTool(
            workspace_collection=ws_coll,  # type: ignore[arg-type]
            workspace_file_collection=file_coll,  # type: ignore[arg-type]
            sandbox=_NoopSandbox(),  # type: ignore[arg-type]
            context_provider=lambda: _NullContext(),
            agent_id=agent_b,
            db_pool=schema_pool,
            acl_cache=acl_cache,
        )

        # --- fs_read: agent B, user B, same customer -> ALLOW ---
        scope_b = ToolCallScope(
            context=CallContext(
                agent_id=agent_b,
                user_id=user_b,
                customer_id=customer_id,
                correlation_id=uuid4(),
            ),
        )
        async with enter_call_scope(scope_b):
            result = await read_tool.execute(
                relative_path="hello.txt",
                workspace="ws",
            )
        assert result.success is True, result.error
        assert result.content == "initial content"

        # --- fs_write: B under write-grant -> ALLOW, journal row in A's schema ---
        write_tool = FsWriteTool(
            workspace_collection=ws_coll,  # type: ignore[arg-type]
            workspace_file_collection=file_coll,  # type: ignore[arg-type]
            workspace_file_version_collection=None,  # type: ignore[arg-type]
            sandbox=_NoopSandbox(),  # type: ignore[arg-type]
            context_provider=lambda: _NullContext(),
            agent_id=agent_b,
            db_pool=schema_pool,
            acl_cache=acl_cache,
        )
        async with enter_call_scope(scope_b):
            result = await write_tool.execute(
                relative_path="hello.txt",
                content="new by B",
                workspace="ws",
            )
        assert result.success is True, result.error

        # assertion: new journal row in A's schema, actor_id == agent_b
        async with pool.acquire() as conn:
            await conn.execute(
                f'SET search_path TO "{owner_schema}", platform',
            )
            latest = await conn.fetchrow(
                "SELECT actor_id, action, version FROM workspace_file_versions "
                "WHERE workspace_id = $1 AND relative_path = $2 "
                "ORDER BY version DESC LIMIT 1",
                workspace_id,
                "hello.txt",
            )
        assert latest is not None
        assert latest["actor_id"] == agent_b, (
            f"journal actor_id should be B ({agent_b}) but was {latest['actor_id']}"
        )
        assert latest["action"] == "update"

        # --- cross-customer C: categorically denied ---
        different_customer = uuid4()
        scope_c = ToolCallScope(
            context=CallContext(
                agent_id=agent_c,
                user_id=uuid4(),
                customer_id=different_customer,
                correlation_id=uuid4(),
            ),
        )
        c_read_tool = FsReadTool(
            workspace_collection=ws_coll,  # type: ignore[arg-type]
            workspace_file_collection=file_coll,  # type: ignore[arg-type]
            sandbox=_NoopSandbox(),  # type: ignore[arg-type]
            context_provider=lambda: _NullContext(),
            agent_id=agent_c,
            db_pool=schema_pool,
            acl_cache=acl_cache,
        )
        async with enter_call_scope(scope_c):
            c_result = await c_read_tool.execute(
                relative_path="hello.txt",
                workspace="ws",
            )
        assert c_result.success is False, (
            "cross-customer caller should be denied categorically"
        )
        assert c_result.error is not None

        # --- same-customer U2 without grant: denied ---
        scope_u2 = ToolCallScope(
            context=CallContext(
                agent_id=agent_a,  # same as owner agent
                user_id=user_u2,  # but a different user with no user-scoped grant
                customer_id=customer_id,
                correlation_id=uuid4(),
            ),
        )
        # owner-path short-circuit fires only when BOTH agent AND user
        # match. here agent matches but user does not, so grant check
        # runs. since U2 has no grant, the call is denied.
        u2_read_tool = FsReadTool(
            workspace_collection=ws_coll,  # type: ignore[arg-type]
            workspace_file_collection=file_coll,  # type: ignore[arg-type]
            sandbox=_NoopSandbox(),  # type: ignore[arg-type]
            context_provider=lambda: _NullContext(),
            agent_id=agent_a,
            db_pool=schema_pool,
            acl_cache=acl_cache,
        )
        async with enter_call_scope(scope_u2):
            u2_result = await u2_read_tool.execute(
                relative_path="hello.txt",
                workspace="ws",
            )
        # agent_a's default schema would resolve workspace locally;
        # the non-owner-user path falls through to cache and is
        # denied because no grant exists for (ws, agent_a, user_u2).
        # the AclCache also has no agent-wide grant for agent_a (the
        # owner path assumes agent+user match). this is the user-
        # scoping rule in effect.
        assert u2_result.success is False, (
            "same-customer user without grant should be denied"
        )
    finally:
        await pool.close()


# kept live so imports don't get DCE-d on static checkers.
_ = json
_ = WorkspaceAccessDenied
_ = AclCache
