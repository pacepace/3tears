"""integration: agent-intention v001 migration + three-tier collection CRUD.

Exercises against a real pgvector/pg16 container:

- the v001 migration chain applies cleanly (table + enum + indexes),
  replay is a no-op, and the ``intention_status`` enum carries exactly
  the four lifecycle values;
- the ``status`` server default (``open``) applies to a raw INSERT that
  omits it;
- ``IntentionsCollection`` save + get round-trips through L1 / L2 / L3;
- ``find_by_user`` isolates by ``user_id`` (the required boundary) and
  ranks by salience -- one metallm user's wants never surface for
  another sharing the same ``agent_id``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    Text,
)

from threetears.agent.intention.collections import IntentionsCollection
from threetears.agent.intention.migrations import register as register_intention
from threetears.agent.intention.types import INTENTION_STATUS_VALUES
from threetears.conversations.migrations import register as register_conversations
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _build_runner() -> MigrationRunner:
    """register conversations + agent-intention migrations on a runner.

    agent-intention declares ``depends_on=("conversations",)`` so the
    conversations package has to be registered too.

    :return: runner with both packages registered
    :rtype: MigrationRunner
    """
    runner = MigrationRunner()
    register_conversations(runner)
    register_intention(runner)
    return runner


async def _apply(url: str, schema: str) -> None:
    """apply the migration chain into ``schema``."""
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        store = AsyncpgStore(conn)
        await runner_apply(conn, store)
    finally:
        await conn.close()


async def runner_apply(conn: asyncpg.Connection, store: AsyncpgStore) -> int:
    """apply the chain via a fresh runner; return migrations applied.

    :param conn: connection (search_path already set)
    :ptype conn: asyncpg.Connection
    :param store: DataStore wrapper over ``conn``
    :ptype store: AsyncpgStore
    :return: number of migrations applied
    :rtype: int
    """
    runner = _build_runner()
    count: int = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    return count


def _l1_metadata() -> MetaData:
    """build an L1 mirror of the ``intentions`` table.

    Mirrors the composite-pk shape (``agent_id``, ``intention_id``) so
    SQLite addresses rows the same way L3 does. pgvector / enum columns
    map to text-ish L1 types (the L1 cache is a key-addressed row store,
    not a query engine); ``salience`` is a float.
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


async def _make_pool(url: str, schema: str) -> asyncpg.Pool:
    """asyncpg pool with search_path pre-bound to the test schema."""
    from threetears.core.collections import init_connection

    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    return pool


class _InMemoryKvBucket:
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


class _InMemoryNatsBus:
    """typed-wrapper NATS stand-in with KV bucket + typed pub/sub."""

    def __init__(self) -> None:
        self._bucket = _InMemoryKvBucket()
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
    ) -> _InMemoryKvBucket:
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


def _build_collection(
    pool: asyncpg.Pool,
    nats: _InMemoryNatsBus,
) -> tuple[IntentionsCollection, SQLiteBackend]:
    """construct an (L1, L2, L3) intention collection stack."""
    l1 = SQLiteBackend(db_name=f"intentions_{uuid.uuid4().hex[:8]}")
    l1.initialize(_l1_metadata())
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1, l2_client=nats, l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    coll = IntentionsCollection(registry=reg, config=cfg, nats_client=nats)
    return coll, l1


def _make_row(agent_id: uuid.UUID, user_id: uuid.UUID, *, salience: float, content: str) -> dict[str, Any]:
    """an open-status intention write dict for the collection create path."""
    now = datetime.now(UTC)
    return {
        "intention_id": uuid.uuid4(),
        "agent_id": agent_id,
        "customer_id": uuid.uuid4(),
        "user_id": user_id,
        "status": "open",
        "content": content,
        "salience": salience,
        "date_created": now,
        "date_updated": now,
    }


class TestMigrationChain:
    """v001 applies cleanly, replays as a no-op, and builds the enum."""

    async def test_chain_applies_replays_and_builds_schema(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            count = await runner_apply(conn, store)
            assert count > 0

            # table + expected columns present
            cols = {
                r["column_name"]
                for r in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = $1 AND table_name = 'intentions'",
                    schema,
                )
            }
            for expected in (
                "intention_id",
                "agent_id",
                "customer_id",
                "user_id",
                "status",
                "content",
                "embedding",
                "salience",
                "last_decayed_at",
                "last_surfaced_at",
                "source_memory_id",
                "source_conversation_id",
                "date_created",
                "date_updated",
            ):
                assert expected in cols, f"missing column {expected!r}"

            # customer_id / user_id nullable scope grains; salience NOT NULL.
            nullability = {
                r["column_name"]: r["is_nullable"]
                for r in await conn.fetch(
                    "SELECT column_name, is_nullable FROM information_schema.columns "
                    "WHERE table_schema = $1 AND table_name = 'intentions'",
                    schema,
                )
            }
            assert nullability["customer_id"] == "YES"
            assert nullability["user_id"] == "YES"
            assert nullability["salience"] == "NO"

            # all three indexes present
            index_names = {
                r["indexname"]
                for r in await conn.fetch(
                    "SELECT indexname FROM pg_indexes WHERE schemaname = $1 AND tablename = 'intentions'",
                    schema,
                )
            }
            assert "idx_intentions_open_ranked" in index_names
            assert "idx_intentions_last_surfaced" in index_names
            assert "ix_intentions_embedding_hnsw" in index_names

            # re-apply is a no-op
            count2 = await runner_apply(conn, store)
            assert count2 == 0
        finally:
            await conn.close()

    async def test_intention_status_enum_values(self, pg_schema: tuple[str, str]) -> None:
        """The live PG enum carries exactly the four lifecycle values, in order."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner_apply(conn, store)

            labels = [
                r["enumlabel"]
                for r in await conn.fetch(
                    "SELECT e.enumlabel FROM pg_enum e "
                    "JOIN pg_type t ON t.oid = e.enumtypid "
                    "JOIN pg_namespace n ON n.oid = t.typnamespace "
                    "WHERE t.typname = 'intention_status' AND n.nspname = $1 "
                    "ORDER BY e.enumsortorder",
                    schema,
                )
            ]
            assert tuple(labels) == INTENTION_STATUS_VALUES
        finally:
            await conn.close()

    async def test_status_server_default_applies(self, pg_schema: tuple[str, str]) -> None:
        """A raw INSERT omitting status takes the ``open`` server default."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner_apply(conn, store)

            now = datetime.now(UTC)
            agent_id = uuid.uuid4()
            intention_id = uuid.uuid4()
            await conn.execute(
                "INSERT INTO intentions (intention_id, agent_id, content, date_created) VALUES ($1, $2, $3, $4)",
                intention_id,
                agent_id,
                "a want with no explicit status",
                now,
            )
            row = await conn.fetchrow(
                "SELECT status, salience FROM intentions WHERE agent_id = $1 AND intention_id = $2",
                agent_id,
                intention_id,
            )
            assert row is not None
            assert row["status"] == "open"
            assert float(row["salience"]) == 0.5  # salience server default too
        finally:
            await conn.close()


class TestIntentionsCollectionThreeTier:
    """save + get round-trips through the tiers; find_by_user isolates."""

    async def test_save_and_get_round_trip(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await _apply(url, schema)
        pool = await _make_pool(url, schema)
        try:
            nats = _InMemoryNatsBus()
            coll, _l1 = _build_collection(pool, nats)

            agent_id = uuid.uuid4()
            user_id = uuid.uuid4()
            row = _make_row(agent_id, user_id, salience=0.7, content="check the wake threads")
            entity = coll.create(row)
            await coll.save_entity(entity)

            # L3 row present with the mutable fields
            async with pool.acquire() as conn:
                db = await conn.fetchrow(
                    "SELECT status, content, salience FROM intentions WHERE agent_id = $1 AND intention_id = $2",
                    agent_id,
                    row["intention_id"],
                )
            assert db is not None
            assert db["status"] == "open"
            assert db["content"] == "check the wake threads"
            assert float(db["salience"]) == 0.7

            # get resolves by the composite pk tuple
            fetched = await coll.get((agent_id, row["intention_id"]))
            assert fetched is not None
            assert fetched.content == "check the wake threads"
            assert fetched.status == "open"
            assert fetched.user_id == user_id
        finally:
            await pool.close()

    async def test_find_by_user_isolates_and_ranks(self, pg_schema: tuple[str, str]) -> None:
        """find_by_user returns only the caller's wants, salience-ranked."""
        url, schema = pg_schema
        await _apply(url, schema)
        pool = await _make_pool(url, schema)
        try:
            nats = _InMemoryNatsBus()
            coll, _l1 = _build_collection(pool, nats)

            agent_id = uuid.uuid4()  # shared across users (metallm reality)
            user_a = uuid.uuid4()
            user_b = uuid.uuid4()

            a_low = _make_row(agent_id, user_a, salience=0.3, content="A: low")
            a_high = _make_row(agent_id, user_a, salience=0.9, content="A: high")
            b_only = _make_row(agent_id, user_b, salience=0.8, content="B: only")
            for row in (a_low, a_high, b_only):
                await coll.save_entity(coll.create(row))

            a_wants = await coll.find_by_user(user_a, agent_id=agent_id)
            # isolation: B's want never surfaces for A
            assert [w.content for w in a_wants] == ["A: high", "A: low"]
            assert all(w.user_id == user_a for w in a_wants)

            b_wants = await coll.find_by_user(user_b, agent_id=agent_id)
            assert [w.content for w in b_wants] == ["B: only"]
        finally:
            await pool.close()
