"""integration-test fixtures for agent-identity.

The bare testcontainer setup (``PostgresContainer`` lifecycle + docker-skip
+ asyncpg URL normalisation) lives in
:mod:`threetears.core.testing.fixtures` as the canonical ``db_container``
fixture, pulled in by the root ``conftest.py``'s ``pytest_plugins`` line.
agent-identity needs no pgvector (no embedding column), so this module does
not pin an image; it provides a per-test ``pg_schema`` fixture and the
collection-stack builders.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table, Text


@pytest.fixture(scope="module")
def pg_url(db_container: str) -> str:
    """alias for the canonical session-scoped DB URL."""
    return db_container


class AsyncpgStore:
    """DataStore-shape wrapper over an asyncpg connection (execute + query)."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        result: str = await self._conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        rows = await self._conn.fetch(sql, *params)
        return [dict(r) for r in rows]


@pytest.fixture
async def pg_schema(pg_url: str) -> AsyncIterator[tuple[str, str]]:
    """create a fresh schema per test; yield ``(pg_url, schema)`` and drop on teardown."""
    schema = f"id_it_{id(object())}".lower().replace("-", "_")
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    finally:
        await conn.close()
    yield (pg_url, schema)
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()


def build_runner() -> Any:
    """register agent-identity migrations on a fresh runner."""
    from threetears.agent.identity.migrations import register as register_identity
    from threetears.core.data.migrations import MigrationRunner

    runner = MigrationRunner()
    register_identity(runner)
    return runner


async def apply_migrations(url: str, schema: str) -> None:
    """apply the identity migration chain into ``schema``."""
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        store = AsyncpgStore(conn)
        runner = build_runner()
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await conn.close()


def l1_metadata() -> MetaData:
    """L1 mirror of identity_versions (key-addressed store; text-ish types)."""
    meta = MetaData()
    Table(
        "identity_versions",
        meta,
        Column("agent_id", String(255), primary_key=True),
        Column("version_id", String(255), primary_key=True),
        Column("customer_id", String(255)),
        Column("user_id", String(255)),
        Column("block_key", String(50)),
        Column("content", Text),
        Column("rationale", Text),
        Column("content_hash", String(255)),
        Column("parent_version_id", String(255)),
        Column("status", String(50)),
        Column("proposer_agent_id", String(255)),
        Column("consenter_user_id", String(255)),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return meta


async def make_pool(url: str, schema: str) -> asyncpg.Pool:
    """asyncpg pool with search_path bound to the test schema."""
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
    )
    return pool


class InMemoryKvBucket:
    """KV bucket stand-in matching ``NatsKvBucket``."""

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
    """NATS stand-in with a KV bucket (L2)."""

    def __init__(self) -> None:
        self._bucket = InMemoryKvBucket()

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
        return None


def build_collection(pool: asyncpg.Pool, nats: InMemoryNatsBus) -> tuple[Any, Any]:
    """construct an (L1, L2, L3) identity-versions collection stack."""
    from threetears.agent.identity.collections import IdentityVersionsCollection
    from threetears.core.cache.sqlite import SQLiteBackend
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig

    l1 = SQLiteBackend(db_name=f"identity_{uuid.uuid4().hex[:8]}")
    l1.initialize(l1_metadata())
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1, l2_client=nats, l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    coll = IdentityVersionsCollection(registry=reg, config=cfg, nats_client=nats)
    return coll, l1
