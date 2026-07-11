"""integration-test fixtures for agent-intention.

The bare testcontainer setup (``PostgresContainer`` lifecycle +
docker-skip + asyncpg URL normalisation) lives in
:mod:`threetears.core.testing.fixtures` as the canonical ``db_container``
fixture, pulled in by the root ``conftest.py``'s ``pytest_plugins`` line.
This module pins the ``pgvector/pgvector:pg16`` image (agent-intention
exercises the ``vector`` extension) and provides a per-test ``pg_schema``
fixture that creates a fresh schema per test and drops it on teardown.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from sqlalchemy import Column, DateTime, Float, MetaData, String, Table, Text


@pytest.fixture(scope="session")
def db_image() -> str:
    """override the canonical ``db_image`` to pin pgvector/pg16.

    agent-intention exercises the ``vector`` extension; the canonical
    default ``postgres:16`` does not ship pgvector. overriding here at
    session scope means every test in the package picks up the pgvector
    image without per-test indirect parametrize boilerplate.

    :return: docker image reference
    :rtype: str
    """
    return "pgvector/pgvector:pg16"


@pytest.fixture(scope="module")
def pg_url(db_container: str) -> str:
    """alias for :func:`threetears.core.testing.fixtures.db_container`.

    :param db_container: canonical session-scoped DB URL
    :ptype db_container: str
    :return: asyncpg-compatible PostgreSQL connection URL
    :rtype: str
    """
    return db_container


class AsyncpgStore:
    """DataStore-shape wrapper over an asyncpg connection.

    exposes :meth:`execute` and :meth:`query` matching what the migration
    runner expects.

    :param conn: asyncpg connection with search_path pre-set
    :ptype conn: asyncpg.Connection
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        """initialize wrapper.

        :param conn: asyncpg connection with search_path set
        :ptype conn: asyncpg.Connection
        """
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """execute SQL via the underlying connection.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: asyncpg status tag
        :rtype: str
        """
        result: str = await self._conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """fetch rows as a list of dicts.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        rows = await self._conn.fetch(sql, *params)
        result = [dict(r) for r in rows]
        return result


@pytest.fixture
async def pg_schema(pg_url: str) -> AsyncIterator[tuple[str, str]]:
    """create a fresh schema per test and yield ``(pg_url, schema_name)``.

    the schema is dropped on teardown so each test gets a clean slate.
    also installs the ``vector`` extension at the database level (shared
    across schemas).

    :return: tuple of (pg url, fresh schema name)
    :rtype: tuple[str, str]
    """
    schema = f"int_it_{id(object())}".lower().replace("-", "_")
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    finally:
        await conn.close()
    yield (pg_url, schema)
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Shared three-tier collection-stack builders (used by the collection + tools
# integration modules). Public (no leading underscore) so both modules import
# them without tripping the underscore-access enforcement.
# ---------------------------------------------------------------------------


def build_runner() -> Any:
    """register conversations + agent-intention migrations on a runner.

    agent-intention declares ``depends_on=("conversations",)`` so the
    conversations package has to be registered too.

    :return: runner with both packages registered
    :rtype: MigrationRunner
    """
    from threetears.agent.intention.migrations import register as register_intention
    from threetears.conversations.migrations import register as register_conversations
    from threetears.core.data.migrations import MigrationRunner

    runner = MigrationRunner()
    register_conversations(runner)
    register_intention(runner)
    return runner


async def runner_apply(conn: asyncpg.Connection, store: AsyncpgStore) -> int:
    """apply the migration chain via a fresh runner; return migrations applied.

    :param conn: connection (search_path already set)
    :ptype conn: asyncpg.Connection
    :param store: DataStore wrapper over ``conn``
    :ptype store: AsyncpgStore
    :return: number of migrations applied
    :rtype: int
    """
    _ = conn
    runner = build_runner()
    count: int = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    return count


async def apply_migrations(url: str, schema: str) -> None:
    """apply the migration chain into ``schema``.

    :param url: pg connection url
    :ptype url: str
    :param schema: fresh schema name
    :ptype schema: str
    :return: nothing
    :rtype: None
    """
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        store = AsyncpgStore(conn)
        await runner_apply(conn, store)
    finally:
        await conn.close()


def l1_metadata() -> MetaData:
    """build an L1 mirror of the ``intentions`` table.

    Mirrors the composite-pk shape (``agent_id``, ``intention_id``) so
    SQLite addresses rows the same way L3 does. pgvector / enum columns
    map to text-ish L1 types (the L1 cache is a key-addressed row store,
    not a query engine); ``salience`` is a float.

    :return: metadata carrying the mirrored table
    :rtype: MetaData
    """
    meta = MetaData()
    Table(
        "intentions",
        meta,
        Column("agent_id", String(255), primary_key=True),
        Column("intention_id", String(255), primary_key=True),
        Column("customer_id", String(255)),
        Column("user_id", String(255)),
        Column("status", String(50)),
        Column("content", Text),
        Column("embedding", Text),
        Column("salience", Float),
        Column("last_decayed_at", DateTime),
        Column("last_surfaced_at", DateTime),
        Column("source_memory_id", String(255)),
        Column("source_conversation_id", String(255)),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return meta


async def make_pool(url: str, schema: str) -> asyncpg.Pool:
    """asyncpg pool with search_path + vector codec bound to the test schema.

    :param url: pg connection url
    :ptype url: str
    :param schema: schema to bind the search_path to
    :ptype schema: str
    :return: initialised pool
    :rtype: asyncpg.Pool
    """
    from threetears.core.collections import init_connection

    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    return pool


class InMemoryKvBucket:
    """typed-wrapper KV bucket stand-in matching ``NatsKvBucket``."""

    def __init__(self) -> None:
        self.kv: dict[str, bytes] = {}

    async def get(self, *, key: str) -> bytes | None:
        return self.kv.get(key)

    async def put(self, *, key: str, value: bytes) -> int:
        self.kv[key] = value
        return len(self.kv)

    async def delete(self, *, key: str, revision: int | None = None) -> bool:  # noqa: ARG002
        existed = key in self.kv
        self.kv.pop(key, None)
        return existed or revision is None


class InMemoryNatsBus:
    """typed-wrapper NATS stand-in with KV bucket + typed pub/sub."""

    def __init__(self) -> None:
        self._bucket = InMemoryKvBucket()
        self._subs: dict[str, list[tuple[Any, Any]]] = {}

    @property
    def kv(self) -> dict[str, bytes]:
        return self._bucket.kv

    async def kv_bucket(
        self,
        *,
        name: str,  # noqa: ARG002
        ttl: Any = None,  # noqa: ARG002
        storage: str = "file",  # noqa: ARG002
        create_if_missing: bool = True,  # noqa: ARG002
        history: int = 1,  # noqa: ARG002
    ) -> InMemoryKvBucket:
        return self._bucket

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:  # noqa: ARG002
        subject_str = str(subject)
        for cb, message_type in self._subs.get(subject_str, []):
            payload = message.model_dump_json()
            decoded = message_type.model_validate_json(payload)
            await cb(decoded)

    async def subscribe_typed(
        self,
        *,
        subject: Any,
        cb: Any,
        message_type: Any,
        queue: Any = None,  # noqa: ARG002
        max_in_flight: Any = None,  # noqa: ARG002
        deadletter_on_failure: bool = True,  # noqa: ARG002
    ) -> None:
        subject_str = str(subject)
        self._subs.setdefault(subject_str, []).append((cb, message_type))


def build_collection(pool: asyncpg.Pool, nats: InMemoryNatsBus) -> tuple[Any, Any]:
    """construct an (L1, L2, L3) intention collection stack.

    :param pool: L3 asyncpg pool
    :ptype pool: asyncpg.Pool
    :param nats: in-memory NATS stand-in (L2)
    :ptype nats: InMemoryNatsBus
    :return: tuple of (IntentionsCollection, SQLiteBackend)
    :rtype: tuple[Any, Any]
    """
    from threetears.agent.intention.collections import IntentionsCollection
    from threetears.core.cache.sqlite import SQLiteBackend
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig

    l1 = SQLiteBackend(db_name=f"intentions_{uuid.uuid4().hex[:8]}")
    l1.initialize(l1_metadata())
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1, l2_client=nats, l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    coll = IntentionsCollection(registry=reg, config=cfg, nats_client=nats)
    return coll, l1
