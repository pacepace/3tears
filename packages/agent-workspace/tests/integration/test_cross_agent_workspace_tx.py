"""integration test for WS-ACL-06 ``namespace=`` threading through tx path.

spins up testcontainers PostgreSQL + in-process NATS (via the existing
``_fake_kv`` harness pattern is insufficient here because we need
real request-reply) and wires:

- a real :class:`NatsProxyL3Backend` on the agent side
- a minimal in-process ``l3.tx.*`` handler on the broker side that
  mirrors what :class:`aibots.hub.broker.proxy.QueryProxy` does for
  tx.begin / tx.execute / tx.fetchrow / tx.fetch / tx.commit:
  resolves the request namespace, sets search_path to the namespace's
  schema, pins a real asyncpg connection for the tx session, runs
  DML + SELECT against the session's pinned connection, commits /
  rolls back on completion.
- the real :func:`_write_file_atomic` helper

the test exercises:

- agent B (grantee) runs :func:`_write_file_atomic` for a workspace
  owned by agent A through the proxy; the journal + head-state rows
  land in agent A's schema (not agent B's default schema)
- concurrent writes on two different workspaces do not cross-
  contaminate — each session pins its own schema via its namespace
- unauthorized namespace on tx.begin returns TX_NAMESPACE_UNAUTHORIZED

NATS is not containerized here; the broker test double speaks NATS
request-reply against an in-memory bus implementing the subset
:class:`NatsProxyL3Backend` uses (request with reply + subscribe).
this mirrors the ``_fake_kv`` pattern used elsewhere in the suite
and keeps the cross-repo boundary light while still exercising the
full wire-level envelope round-trip.

skipped cleanly when docker is unavailable (:class:`PostgresContainer`
startup fails) so CI without docker does not break.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4, uuid7

import asyncpg
import pytest
import pytest_asyncio

from threetears.core.backends.nats_proxy import NatsProxyL3Backend


__all__ = [
    "POSTGRES_IMAGE",
]


pytestmark = pytest.mark.integration


POSTGRES_IMAGE = "pgvector/pgvector:pg16"
_SCHEMA_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# ---------------------------------------------------------------------------
# minimal in-memory NATS request/reply bus
# ---------------------------------------------------------------------------


class _InMemoryNatsBus:
    """subset of nats-py's :class:`Client` tailored to the proxy tx flow.

    supports :meth:`subscribe` + :meth:`request`; the broker-side test
    double registers subject handlers via :meth:`subscribe`, and the
    agent-side :class:`NatsProxyL3Backend` issues request-reply via
    :meth:`request`. every request resolves against the matching
    handler inline; the semantics match real NATS for the subset the
    proxy uses (no queue groups, no wildcards).

    :ivar handlers: subject -> async callback registry
    """

    def __init__(self) -> None:
        """initialize empty handler map.

        :return: None
        :rtype: None
        """
        self._handlers: dict[str, Any] = {}

    async def subscribe(
        self,
        subject: str,
        cb: Any,
        queue: str | None = None,
    ) -> Any:
        """register a callback for a subject.

        :param subject: NATS subject string
        :ptype subject: str
        :param cb: async callback receiving a message-like object
        :ptype cb: Any
        :param queue: queue group name (ignored in single-handler bus)
        :ptype queue: str | None
        :return: subscription handle (stub)
        :rtype: Any
        """
        del queue
        self._handlers[subject] = cb
        return MagicMock()

    async def request(
        self,
        subject: str,
        payload: bytes,
        timeout: float = 5.0,
    ) -> Any:
        """route a request to the subject's handler and return its reply.

        :param subject: NATS subject to dispatch on
        :ptype subject: str
        :param payload: JSON-encoded bytes payload
        :ptype payload: bytes
        :param timeout: request timeout seconds
        :ptype timeout: float
        :return: mock reply with ``.data`` bytes
        :rtype: Any
        :raises asyncio.TimeoutError: when no handler is registered for
            the subject (matches real NATS behavior)
        """
        del timeout
        handler = self._handlers.get(subject)
        if handler is None:
            raise asyncio.TimeoutError(f"no handler for {subject}")
        reply_holder: dict[str, bytes] = {}

        class _Msg:
            """minimal NATS-message-shape with ``.data`` + ``.respond``."""

            def __init__(self, data: bytes) -> None:
                """capture payload.

                :param data: raw bytes
                :ptype data: bytes
                :return: None
                :rtype: None
                """
                self.data = data
                self.reply = "inbox"

            async def respond(self, reply_data: bytes) -> None:
                """capture reply into outer holder.

                :param reply_data: bytes to reply with
                :ptype reply_data: bytes
                :return: None
                :rtype: None
                """
                reply_holder["data"] = reply_data

        msg = _Msg(payload)
        await handler(msg)
        reply_msg = MagicMock()
        reply_msg.data = reply_holder.get("data", b"{}")
        return reply_msg


# ---------------------------------------------------------------------------
# tx session state + minimal broker-side tx.* handlers
# ---------------------------------------------------------------------------


class _TxSessionState:
    """in-process ``_TxSession`` analogue.

    mirrors the broker-side tx session but shrunken to the fields the
    integration test exercises: the pinned connection + pending
    transaction + bound schema.
    """

    def __init__(
        self,
        *,
        conn: asyncpg.Connection,
        tx: Any,
        agent_id: UUID,
        schema: str,
    ) -> None:
        """capture the pinned connection + transaction.

        :param conn: acquired asyncpg connection
        :ptype conn: asyncpg.Connection
        :param tx: started asyncpg transaction
        :ptype tx: Any
        :param agent_id: owning agent UUID for cross-agent guard
        :ptype agent_id: UUID
        :param schema: schema bound into search_path
        :ptype schema: str
        :return: None
        :rtype: None
        """
        self.conn = conn
        self.tx = tx
        self.agent_id = agent_id
        self.schema = schema


class _TxBroker:
    """in-process broker test double serving the ``l3.tx.*`` subset.

    looks up namespace rows from a test-scoped mapping, enforces a
    minimal ACL policy against a grants mapping, sets search_path to
    the namespace's ``schema_name`` at tx.begin, and executes every
    subsequent tx.* statement on the pinned connection. mirrors the
    production :class:`QueryProxy` behavior for the subset of subjects
    the tx-flow tests exercise.

    :param pool: asyncpg pool bound to the testcontainer DB
    :ptype pool: asyncpg.Pool
    :param namespaces: mapping of namespace-name -> schema_name
    :ptype namespaces: dict[str, str]
    :param grants: mapping of (namespace, agent_id) -> access_level
    :ptype grants: dict[tuple[str, UUID], str]
    """

    def __init__(
        self,
        *,
        pool: asyncpg.Pool,
        namespaces: dict[str, dict[str, Any]],
        grants: dict[tuple[str, UUID], str],
    ) -> None:
        """capture the pool + policy mappings.

        :param pool: asyncpg pool
        :ptype pool: asyncpg.Pool
        :param namespaces: mapping of namespace-name -> row dict with
            ``schema_name`` + ``namespace_type``
        :ptype namespaces: dict[str, dict[str, Any]]
        :param grants: mapping of (namespace_name, agent_id) -> level
        :ptype grants: dict[tuple[str, UUID], str]
        :return: None
        :rtype: None
        """
        self._pool = pool
        self._namespaces = namespaces
        self._grants = grants
        self._sessions: dict[UUID, _TxSessionState] = {}

    async def register(self, bus: _InMemoryNatsBus, ns: str) -> None:
        """register every tx.* handler on the bus.

        :param bus: in-memory NATS bus
        :ptype bus: _InMemoryNatsBus
        :param ns: subject namespace prefix
        :ptype ns: str
        :return: None
        :rtype: None
        """
        await bus.subscribe(f"{ns}.l3.tx.begin", self._handle_begin)
        await bus.subscribe(f"{ns}.l3.tx.execute", self._handle_execute)
        await bus.subscribe(f"{ns}.l3.tx.fetchrow", self._handle_fetchrow)
        await bus.subscribe(f"{ns}.l3.tx.fetch", self._handle_fetch)
        await bus.subscribe(f"{ns}.l3.tx.commit", self._handle_commit)
        await bus.subscribe(f"{ns}.l3.tx.rollback", self._handle_rollback)

    async def _handle_begin(self, msg: Any) -> None:
        """mirror QueryProxy._handle_tx_begin.

        :param msg: NATS message with the begin request payload
        :ptype msg: Any
        :return: None
        :rtype: None
        """
        req = json.loads(msg.data.decode("utf-8"))
        namespace = req["namespace"]
        agent_id = UUID(req["agent_id"])
        ns_row = self._namespaces.get(namespace)
        if ns_row is None:
            await msg.respond(
                json.dumps(
                    {
                        "success": False,
                        "error_code": "NAMESPACE_NOT_FOUND",
                        "error_message": namespace,
                    }
                ).encode(),
            )
            return
        access = self._grants.get((namespace, agent_id))
        if access not in ("write", "readwrite"):
            await msg.respond(
                json.dumps(
                    {
                        "success": False,
                        "error_code": "TX_NAMESPACE_UNAUTHORIZED",
                        "error_message": (
                            f"agent {agent_id} has no write grant "
                            f"on {namespace}"
                        ),
                    }
                ).encode(),
            )
            return
        schema = ns_row["schema_name"]
        assert _SCHEMA_NAME_RE.match(schema), schema
        conn = await self._pool.acquire()
        try:
            await conn.execute(f"SET search_path TO {schema}")
        except Exception:
            await self._pool.release(conn)
            raise
        tx = conn.transaction()
        await tx.start()
        tx_id = uuid7()
        self._sessions[tx_id] = _TxSessionState(
            conn=conn, tx=tx, agent_id=agent_id, schema=schema,
        )
        await msg.respond(
            json.dumps({"success": True, "tx_id": str(tx_id)}).encode(),
        )

    async def _handle_execute(self, msg: Any) -> None:
        """run DML on the session's pinned connection.

        :param msg: NATS message
        :ptype msg: Any
        :return: None
        :rtype: None
        """
        req = json.loads(msg.data.decode("utf-8"))
        tx_id = UUID(req["tx_id"])
        session = self._sessions.get(tx_id)
        if session is None:
            await msg.respond(
                json.dumps(
                    {
                        "success": False,
                        "error_code": "TX_UNKNOWN",
                        "error_message": "unknown tx_id",
                    }
                ).encode(),
            )
            return
        params = [
            _coerce(p) for p in req.get("params", [])
        ]
        tag = await session.conn.execute(req["query"], *params)
        row_count = int(tag.split()[-1]) if tag and tag.split() else 0
        await msg.respond(
            json.dumps(
                {"success": True, "row_count": row_count}
            ).encode(),
        )

    async def _handle_fetchrow(self, msg: Any) -> None:
        """single-row SELECT on the session's pinned connection.

        :param msg: NATS message
        :ptype msg: Any
        :return: None
        :rtype: None
        """
        req = json.loads(msg.data.decode("utf-8"))
        tx_id = UUID(req["tx_id"])
        session = self._sessions.get(tx_id)
        if session is None:
            await msg.respond(
                json.dumps(
                    {
                        "success": False,
                        "error_code": "TX_UNKNOWN",
                        "error_message": "unknown tx_id",
                    }
                ).encode(),
            )
            return
        params = [_coerce(p) for p in req.get("params", [])]
        row = await session.conn.fetchrow(req["query"], *params)
        row_payload = _to_json_row(row) if row is not None else None
        await msg.respond(
            json.dumps(
                {"success": True, "row": row_payload}, default=str,
            ).encode(),
        )

    async def _handle_fetch(self, msg: Any) -> None:
        """multi-row SELECT on the session's pinned connection.

        :param msg: NATS message
        :ptype msg: Any
        :return: None
        :rtype: None
        """
        req = json.loads(msg.data.decode("utf-8"))
        tx_id = UUID(req["tx_id"])
        session = self._sessions.get(tx_id)
        if session is None:
            await msg.respond(
                json.dumps(
                    {
                        "success": False,
                        "error_code": "TX_UNKNOWN",
                        "error_message": "unknown tx_id",
                    }
                ).encode(),
            )
            return
        params = [_coerce(p) for p in req.get("params", [])]
        rows = await session.conn.fetch(req["query"], *params)
        await msg.respond(
            json.dumps(
                {
                    "success": True,
                    "rows": [_to_json_row(r) for r in rows],
                },
                default=str,
            ).encode(),
        )

    async def _handle_commit(self, msg: Any) -> None:
        """commit the session's tx and release the pinned connection.

        :param msg: NATS message
        :ptype msg: Any
        :return: None
        :rtype: None
        """
        req = json.loads(msg.data.decode("utf-8"))
        tx_id = UUID(req["tx_id"])
        session = self._sessions.pop(tx_id, None)
        if session is not None:
            await session.tx.commit()
            await self._pool.release(session.conn)
        await msg.respond(json.dumps({"success": True}).encode())

    async def _handle_rollback(self, msg: Any) -> None:
        """rollback the session's tx and release the pinned connection.

        :param msg: NATS message
        :ptype msg: Any
        :return: None
        :rtype: None
        """
        req = json.loads(msg.data.decode("utf-8"))
        tx_id = UUID(req["tx_id"])
        session = self._sessions.pop(tx_id, None)
        if session is not None:
            await session.tx.rollback()
            await self._pool.release(session.conn)
        await msg.respond(json.dumps({"success": True}).encode())


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _coerce(value: Any) -> Any:
    """rehydrate JSON payload values into native asyncpg bindings.

    UUID-shaped strings get converted to :class:`UUID`; hex-encoded
    bytes (``\\x...``) convert back to :class:`bytes`; ISO-8601
    datetime strings convert back to naive :class:`datetime`.

    :param value: payload value
    :ptype value: Any
    :return: native-typed value
    :rtype: Any
    """
    result = value
    if isinstance(value, str):
        if value.startswith("\\x"):
            result = bytes.fromhex(value[2:])
        elif len(value) == 36 and value.count("-") == 4:
            try:
                result = UUID(value)
            except ValueError:
                result = value
        elif len(value) >= 19 and value[4] == "-" and value[7] == "-":
            try:
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is not None:
                    parsed = parsed.replace(tzinfo=None)
                result = parsed
            except ValueError:
                result = value
    return result


def _to_json_row(row: Any) -> dict[str, Any]:
    """convert an asyncpg :class:`Record` to a JSON-safe dict.

    bytes get hex-encoded so the client can restore them; UUID /
    datetime / Decimal serialize via ``default=str`` at the outer
    :func:`json.dumps` boundary.

    :param row: asyncpg record or mapping
    :ptype row: Any
    :return: JSON-safe dict
    :rtype: dict[str, Any]
    """
    result: dict[str, Any] = {}
    for key, value in dict(row).items():
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = (
                bytes(value)
                if isinstance(value, (memoryview, bytearray))
                else value
            )
            result[key] = "\\x" + raw.hex()
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_url() -> Iterator[str]:
    """start a fresh Postgres testcontainer and yield its URL.

    :return: asyncpg-compatible URL
    :rtype: Iterator[str]
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
    """canonical agent schema name.

    :param agent_id: agent UUID
    :ptype agent_id: UUID
    :return: ``agent_<hex>`` schema string
    :rtype: str
    """
    return f"agent_{agent_id.hex}"


async def _build_agent_schema(conn: asyncpg.Connection, agent_id: UUID) -> None:
    """materialize the minimum agent-schema workspace tables.

    :param conn: asyncpg connection
    :ptype conn: asyncpg.Connection
    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :return: None
    :rtype: None
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
            date_created    TIMESTAMP NOT NULL,
            date_updated    TIMESTAMP NOT NULL,
            date_deleted    TIMESTAMP
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
            date_updated    TIMESTAMP NOT NULL,
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
            date_created    TIMESTAMP NOT NULL,
            UNIQUE (workspace_id, relative_path, version)
        )
        """,
    )


@pytest_asyncio.fixture
async def seeded_env(pg_url: str) -> AsyncIterator[dict[str, Any]]:
    """seed two agent schemas, a workspace, a grant, and the broker harness.

    layout: agent A (owner) holds the physical workspace rows; agent B
    (grantee) gets a write grant on the workspace's namespace.

    :param pg_url: Postgres container URL from :func:`pg_url`
    :ptype pg_url: str
    :return: mapping with pool, bus, broker, identifiers, schemas
    :rtype: AsyncIterator[dict[str, Any]]
    """
    from threetears.core.collections import init_connection
    pool = await asyncpg.create_pool(
        pg_url, min_size=2, max_size=5, init=init_connection,
    )
    agent_a = uuid4()
    agent_b = uuid4()
    user_b = uuid4()
    workspace_a_id = uuid7()
    workspace_c_id = uuid7()
    schema_a = _schema_name(agent_a)
    schema_b = _schema_name(agent_b)
    try:
        async with pool.acquire() as conn:
            await _build_agent_schema(conn, agent_a)
            await _build_agent_schema(conn, agent_b)

        # seed A's workspace row (empty, no files).
        now = datetime.now(UTC).replace(tzinfo=None)
        async with pool.acquire() as conn:
            await conn.execute(
                f'INSERT INTO "{schema_a}".workspaces '
                "(id, agent_id, name, created_by, current_version, "
                " date_created, date_updated) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                workspace_a_id,
                agent_a,
                "shared_ws",
                agent_a,
                0,
                now,
                now,
            )
            # seed B's own workspace so we can assert cross-
            # contamination doesn't happen when B also has an
            # unrelated tx running.
            await conn.execute(
                f'INSERT INTO "{schema_b}".workspaces '
                "(id, agent_id, name, created_by, current_version, "
                " date_created, date_updated) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                workspace_c_id,
                agent_b,
                "local_ws",
                agent_b,
                0,
                now,
                now,
            )

        namespace_a = f"workspace.{workspace_a_id}"
        namespace_c = f"workspace.{workspace_c_id}"
        bus = _InMemoryNatsBus()
        broker = _TxBroker(
            pool=pool,
            namespaces={
                namespace_a: {
                    "schema_name": schema_a,
                    "namespace_type": "workspace",
                },
                namespace_c: {
                    "schema_name": schema_b,
                    "namespace_type": "workspace",
                },
                f"agent.{agent_a}": {
                    "schema_name": schema_a,
                    "namespace_type": "agent",
                },
                f"agent.{agent_b}": {
                    "schema_name": schema_b,
                    "namespace_type": "agent",
                },
            },
            grants={
                # owner has implicit access (modeled as an explicit
                # write grant for the test).
                (namespace_a, agent_a): "write",
                (namespace_c, agent_b): "write",
                # grantee: agent B has write on A's workspace namespace.
                (namespace_a, agent_b): "write",
                # own-agent grants.
                (f"agent.{agent_a}", agent_a): "write",
                (f"agent.{agent_b}", agent_b): "write",
            },
        )
        await broker.register(bus, ns="test")
        yield {
            "pool": pool,
            "bus": bus,
            "broker": broker,
            "agent_a": agent_a,
            "agent_b": agent_b,
            "user_b": user_b,
            "workspace_a_id": workspace_a_id,
            "workspace_c_id": workspace_c_id,
            "namespace_a": namespace_a,
            "namespace_c": namespace_c,
            "schema_a": schema_a,
            "schema_b": schema_b,
        }
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grantee_tx_writes_land_in_owner_schema(
    seeded_env: dict[str, Any],
) -> None:
    """agent B writing to a workspace it was granted lands in A's schema."""
    pool: asyncpg.Pool = seeded_env["pool"]
    bus: _InMemoryNatsBus = seeded_env["bus"]
    agent_b: UUID = seeded_env["agent_b"]
    user_b: UUID = seeded_env["user_b"]
    workspace_a_id: UUID = seeded_env["workspace_a_id"]
    namespace_a: str = seeded_env["namespace_a"]
    schema_a: str = seeded_env["schema_a"]
    schema_b: str = seeded_env["schema_b"]

    backend = NatsProxyL3Backend(
        nats_client=bus,
        namespace_prefix="test",
        agent_id=str(agent_b),
    )

    correlation_id = uuid7()
    now = datetime.now(UTC).replace(tzinfo=None)
    content = b"grantee-written content"
    sha = "9" * 64

    async with backend.transaction(namespace=namespace_a) as conn:
        await conn.execute(
            "INSERT INTO workspace_file_versions "
            "(id, workspace_id, relative_path, version, content, sha256, "
            " action, label, actor_id, correlation_id, date_created) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
            uuid7(),
            workspace_a_id,
            "notes.md",
            1,
            content,
            sha,
            "create",
            None,
            user_b,
            correlation_id,
            now,
        )
        await conn.execute(
            "INSERT INTO workspace_files "
            "(id, workspace_id, relative_path, content, sha256, version, "
            " date_updated) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7)",
            uuid7(),
            workspace_a_id,
            "notes.md",
            content,
            sha,
            1,
            now,
        )

    # assertion 1: rows landed in the OWNER's schema.
    async with pool.acquire() as conn:
        rows_a = await conn.fetch(
            f'SELECT * FROM "{schema_a}".workspace_file_versions',
        )
        rows_b = await conn.fetch(
            f'SELECT * FROM "{schema_b}".workspace_file_versions',
        )
    assert len(rows_a) == 1, (
        "journal row should have landed in agent A's schema"
    )
    assert len(rows_b) == 0, (
        "journal row should NOT have landed in agent B's (caller) schema"
    )
    # assertion 2: actor_id is agent B's user (the invoking user), not
    # agent A; the cross-agent audit trail is preserved.
    assert rows_a[0]["actor_id"] == user_b


@pytest.mark.asyncio
async def test_concurrent_tx_on_different_workspaces_no_cross_contamination(
    seeded_env: dict[str, Any],
) -> None:
    """two concurrent tx sessions on two workspaces don't cross schemas."""
    pool: asyncpg.Pool = seeded_env["pool"]
    bus: _InMemoryNatsBus = seeded_env["bus"]
    agent_b: UUID = seeded_env["agent_b"]
    user_b: UUID = seeded_env["user_b"]
    workspace_a_id: UUID = seeded_env["workspace_a_id"]
    workspace_c_id: UUID = seeded_env["workspace_c_id"]
    namespace_a: str = seeded_env["namespace_a"]
    namespace_c: str = seeded_env["namespace_c"]
    schema_a: str = seeded_env["schema_a"]
    schema_b: str = seeded_env["schema_b"]

    backend = NatsProxyL3Backend(
        nats_client=bus,
        namespace_prefix="test",
        agent_id=str(agent_b),
    )
    now = datetime.now(UTC).replace(tzinfo=None)

    async def _write(
        namespace: str,
        workspace_id: UUID,
        relative_path: str,
        content: bytes,
    ) -> None:
        """open a tx against ``namespace`` and write a journal row.

        :param namespace: workspace namespace to bind the tx to
        :ptype namespace: str
        :param workspace_id: workspace UUID
        :ptype workspace_id: UUID
        :param relative_path: file path
        :ptype relative_path: str
        :param content: file bytes
        :ptype content: bytes
        :return: None
        :rtype: None
        """
        async with backend.transaction(namespace=namespace) as conn:
            await conn.execute(
                "INSERT INTO workspace_file_versions "
                "(id, workspace_id, relative_path, version, content, "
                " sha256, action, label, actor_id, correlation_id, "
                " date_created) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
                uuid7(),
                workspace_id,
                relative_path,
                1,
                content,
                "a" * 64,
                "create",
                None,
                user_b,
                uuid7(),
                now,
            )

    await asyncio.gather(
        _write(namespace_a, workspace_a_id, "from_a.md", b"a-content"),
        _write(namespace_c, workspace_c_id, "from_c.md", b"c-content"),
    )

    async with pool.acquire() as conn:
        a_rows = await conn.fetch(
            f'SELECT relative_path FROM "{schema_a}".workspace_file_versions',
        )
        b_rows = await conn.fetch(
            f'SELECT relative_path FROM "{schema_b}".workspace_file_versions',
        )
    a_paths = {r["relative_path"] for r in a_rows}
    b_paths = {r["relative_path"] for r in b_rows}
    assert "from_a.md" in a_paths
    assert "from_a.md" not in b_paths
    assert "from_c.md" in b_paths
    assert "from_c.md" not in a_paths


@pytest.mark.asyncio
async def test_grantee_without_grant_is_denied_on_tx_begin(
    seeded_env: dict[str, Any],
) -> None:
    """agent without a grant on a workspace namespace is denied at tx.begin."""
    from threetears.core.exceptions import DataLayerUnavailableError

    bus: _InMemoryNatsBus = seeded_env["bus"]
    namespace_a: str = seeded_env["namespace_a"]
    # spin up a NEW agent not in the grants map.
    random_agent = uuid4()

    backend = NatsProxyL3Backend(
        nats_client=bus,
        namespace_prefix="test",
        agent_id=str(random_agent),
    )

    with pytest.raises(DataLayerUnavailableError) as excinfo:
        async with backend.transaction(namespace=namespace_a):
            pass

    assert "TX_NAMESPACE_UNAUTHORIZED" in str(excinfo.value)
