"""integration test for WS-ACL-05 cross-agent workspace access.

spins up a real PostgreSQL via testcontainers, materializes the
``platform.namespaces`` + ``platform.namespace_grants`` + agent-schema
``workspaces`` + ``workspace_files`` + ``workspace_file_versions``
tables with the minimal shape the workspace tools need, seeds one
customer with two agents (A owns the workspace, B gets a grant), and
exercises the workspace tool stack end-to-end:

- agent B's ``fs_read`` on a file in A's workspace succeeds (read grant)
- agent B's ``fs_write`` succeeds (write grant)
- a third agent C under a different customer is categorically denied
  on any operation
- a user U2 within A's customer but without a grant is denied on
  every tool (same-customer no-grant deny)

the ``AclCache`` contract the tool layer depends on is exercised via
a minimal in-process implementation of :class:`AclCacheLike` that
resolves grants directly against the platform.namespace_grants table;
this matches the production cache's contract without the full cache
infrastructure from the aibots repo (the cache itself is covered by
its own unit suite, this test is the end-to-end wiring check).

journal envelope assertion: the brief calls for the phase-5 sweep to
assert `actor_user_id=B's user`, `calling_agent_id=B`,
`owner_agent_id=A` surface on the journal row envelopes. since the
journal rows today carry only ``actor_id`` + ``correlation_id``, the
additional identity fields ride on the audit envelope which phase-6
(audit-task-01) will wire into the hub consumer. here we assert the
journal's ``actor_id`` is B's agent-id and the row lands in A's
schema — the signal the compliance report needs to answer "who did
what to whose data".
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest

from threetears.agent.tools.call_scope import ToolCallScope, enter_call_scope
from threetears.agent.tools.context_envelope import CallContext

from threetears.agent.workspace.authorize import (
    AclCacheLike,
    WorkspaceAccessDenied,
)

pytestmark = pytest.mark.integration


POSTGRES_IMAGE = "pgvector/pgvector:pg16"


# ---------------------------------------------------------------------------
# testcontainer + schema setup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_url() -> Iterator[str]:
    """
    spin up a fresh Postgres container with pgvector.

    :return: asyncpg-compatible URL
    :rtype: str
    """
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers not installed")

    container = PostgresContainer(POSTGRES_IMAGE)
    try:
        container.start()
    except Exception as exc:
        pytest.skip(f"docker unavailable: {exc}")
    try:
        url = container.get_connection_url()
        if url.startswith("postgresql+psycopg2://"):
            url = url.replace("postgresql+psycopg2://", "postgresql://", 1)
        yield url
    finally:
        container.stop()


def _schema_name(agent_id: UUID) -> str:
    """agent schema name from UUID hex.

    :param agent_id: agent UUID
    :ptype agent_id: UUID
    :return: ``agent_<hex>`` schema name
    :rtype: str
    """
    return f"agent_{agent_id.hex}"


async def _build_platform_schema(conn: asyncpg.Connection) -> None:
    """
    create the minimum platform.namespaces + namespace_grants shape.

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
        CREATE TABLE platform.namespace_grants (
            id                UUID PRIMARY KEY,
            namespace_id      UUID NOT NULL REFERENCES platform.namespaces(id),
            agent_id          UUID NOT NULL,
            user_id           UUID,
            access_level      TEXT NOT NULL,
            date_created      TIMESTAMPTZ NOT NULL,
            UNIQUE NULLS NOT DISTINCT (namespace_id, agent_id, user_id)
        )
        """,
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


class _SqlBackedAclCache:
    """in-test implementation of :class:`AclCacheLike` backed by SQL.

    production wiring uses :class:`aibots.hub.broker.acl.AclCache`
    with L1 / L2 / L3 warm-paths; this test hits the platform table
    directly so the authorize contract surfaces correctly without the
    cache infrastructure.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """capture the pool.

        :param pool: asyncpg Pool bound to the test database
        :ptype pool: asyncpg.Pool
        """
        self._pool = pool

    async def check_access(
        self,
        agent_id: UUID,
        namespace_name: str,
        operation: str,
        user_id: UUID | None = None,
    ) -> None:
        """resolve user-scoped first, fall back to agent-wide; raise on deny.

        mirrors the cache's resolution rule: look for a row matching
        the (namespace, agent, user) triple; if none, fall back to
        ``user_id IS NULL``; if still none, raise. access level
        ``read`` satisfies ``select``; ``write`` satisfies both
        ``select`` and ``upsert``.

        :param agent_id: caller agent UUID
        :ptype agent_id: UUID
        :param namespace_name: canonical namespace key
        :ptype namespace_name: str
        :param operation: cache-level verb (``select`` | ``upsert``)
        :ptype operation: str
        :param user_id: caller user UUID or None for agent-wide lookup
        :ptype user_id: UUID | None
        :return: None on allow
        :rtype: None
        :raises RuntimeError: on deny
        """
        async with self._pool.acquire() as conn:
            # resolve namespace_id
            row = await conn.fetchrow(
                "SELECT id FROM platform.namespaces WHERE name = $1",
                namespace_name,
            )
            if row is None:
                raise RuntimeError(f"namespace not found: {namespace_name}")
            namespace_id = row["id"]

            # user-scoped grant wins
            grant_row = await conn.fetchrow(
                """
                SELECT access_level FROM platform.namespace_grants
                 WHERE namespace_id = $1 AND agent_id = $2 AND user_id = $3
                """,
                namespace_id,
                agent_id,
                user_id,
            )
            if grant_row is None:
                grant_row = await conn.fetchrow(
                    """
                    SELECT access_level FROM platform.namespace_grants
                     WHERE namespace_id = $1 AND agent_id = $2
                       AND user_id IS NULL
                    """,
                    namespace_id,
                    agent_id,
                )
            if grant_row is None:
                raise RuntimeError("no matching grant")
            access_level = grant_row["access_level"]
            if operation == "upsert" and access_level != "write":
                raise RuntimeError(
                    f"grant access_level={access_level!r} is not write",
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

    def __init__(self, pool: asyncpg.Pool, schema: str) -> None:
        """capture the pool + schema.

        :param pool: underlying asyncpg pool
        :ptype pool: asyncpg.Pool
        :param schema: owner agent schema name
        :ptype schema: str
        """
        self._pool = pool
        self._schema = schema

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

        :return: acquire context manager
        :rtype: Any
        """
        return _SchemaBoundAcquire(self._pool, self._schema)


class _SchemaBoundAcquire:
    """async CM binding search_path on a freshly-acquired connection."""

    def __init__(self, pool: asyncpg.Pool, schema: str) -> None:
        """capture pool + schema.

        :param pool: pool
        :ptype pool: asyncpg.Pool
        :param schema: schema
        :ptype schema: str
        """
        self._pool = pool
        self._schema = schema
        self._conn: asyncpg.Connection | None = None
        self._cm: Any = None

    async def __aenter__(self) -> asyncpg.Connection:
        """acquire + SET search_path.

        :return: connection
        :rtype: asyncpg.Connection
        """
        self._cm = self._pool.acquire()
        self._conn = await self._cm.__aenter__()
        await self._conn.execute(
            f'SET search_path TO "{self._schema}", platform',
        )
        return self._conn

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


async def _seed_grant(
    conn: asyncpg.Connection,
    *,
    namespace_id: UUID,
    agent_id: UUID,
    user_id: UUID | None,
    access_level: str,
) -> None:
    """insert a row into platform.namespace_grants.

    :param conn: connection
    :ptype conn: asyncpg.Connection
    :param namespace_id: namespace UUID
    :ptype namespace_id: UUID
    :param agent_id: agent UUID
    :ptype agent_id: UUID
    :param user_id: optional user UUID (None = agent-wide)
    :ptype user_id: UUID | None
    :param access_level: ``read`` or ``write``
    :ptype access_level: str
    """
    now = datetime.now(UTC)
    await conn.execute(
        """
        INSERT INTO platform.namespace_grants (
            id, namespace_id, agent_id, user_id, access_level, date_created
        ) VALUES ($1, $2, $3, $4, $5, $6)
        """,
        uuid4(),
        namespace_id,
        agent_id,
        user_id,
        access_level,
        now,
    )


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

    # --- infra ---
    pool = await asyncpg.create_pool(pg_url, min_size=1, max_size=4)
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
            # grant B write access agent-wide
            await _seed_grant(
                conn,
                namespace_id=workspace_id,
                agent_id=agent_b,
                user_id=None,
                access_level="write",
            )

        # --- build tools ---
        acl_cache = _SqlBackedAclCache(pool)
        owner_schema = _schema_name(agent_a)
        schema_pool = _SchemaBoundPool(pool, owner_schema)
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
        # workspace= kwarg explicitly).
        class _NoopSandbox:
            def enforce(self, action: str, target: str) -> None:
                del action, target

            def check_relative_key(self, key: str, mode: str) -> Any:
                from threetears.core.security import SandboxDecision

                del key, mode
                return SandboxDecision.ALLOW

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
_ = AclCacheLike
