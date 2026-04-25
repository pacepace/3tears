"""integration tests for composite-pk BaseCollection end-to-end.

scenarios:

1. ``FakeRefCollection`` persists to a real postgres-backed composite-pk
   table, populates L1 (SQLite) on save, retrieves via get with the
   tuple id.
2. cold-start simulation: a second Collection instance sharing the same
   L3 but a fresh L1 resolves a row by tuple id through the L3-miss
   path, populating its own L1.
3. cross-pod invalidation: pod A writes, pod B's L1 is evicted via the
   ``threetears.cache.invalidate`` stream carrying ``ids: [k1, k2]``.

the suite is guarded by ``@pytest.mark.integration`` and skips if
docker is unavailable.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import Any

import asyncpg
import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity


pytestmark = pytest.mark.integration

POSTGRES_IMAGE = "pgvector/pgvector:pg16"


@pytest.fixture(scope="module")
def pg_url() -> Iterator[str]:
    """spin up a postgres container and yield an asyncpg URL."""
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


@pytest.fixture
async def pg_pool(pg_url: str) -> AsyncIterator[asyncpg.Pool]:
    """per-test pool with a fresh table for isolation."""
    pool: asyncpg.Pool = await asyncpg.create_pool(pg_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                DROP TABLE IF EXISTS fake_refs;
                CREATE TABLE fake_refs (
                    conversation_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    score INTEGER,
                    note TEXT,
                    date_created TIMESTAMP,
                    date_updated TIMESTAMP,
                    PRIMARY KEY (conversation_id, item_id)
                )
                """
            )
        yield pool
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS fake_refs")
        await pool.close()


def _metadata() -> MetaData:
    md = MetaData()
    Table(
        "fake_refs",
        md,
        Column("conversation_id", String(255), primary_key=True),
        Column("item_id", String(255), primary_key=True),
        Column("score", Integer),
        Column("note", String(255)),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return md


class FakeRefEntity(BaseEntity):
    """composite-pk entity with tuple ``_id``."""

    primary_key_field = "conversation_id"

    def __init__(self, data: dict[str, Any], is_new: bool = True, collection: Any = None) -> None:
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(self, "_id", (data["conversation_id"], data["item_id"]))


class FakeRefCollection(BaseCollection[FakeRefEntity]):
    """composite-pk collection against a real postgres table."""

    primary_key_column: str | tuple[str, ...] = ("conversation_id", "item_id")

    @property
    def table_name(self) -> str:
        return "fake_refs"

    @property
    def entity_class(self) -> type[FakeRefEntity]:
        return FakeRefEntity

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        key = self.normalize_pk(entity_id)
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM fake_refs WHERE conversation_id = $1 AND item_id = $2",
            key[0],
            key[1],
        )
        if row is None:
            return None
        return dict(row)

    async def save_to_postgres(self, data: dict[str, Any], original_timestamp: datetime | None = None) -> int:
        status = await self.l3_pool.execute(
            """
            INSERT INTO fake_refs
                (conversation_id, item_id, score, note, date_created, date_updated)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (conversation_id, item_id) DO UPDATE SET
                score = EXCLUDED.score,
                note = EXCLUDED.note,
                date_updated = EXCLUDED.date_updated
            """,
            data["conversation_id"],
            data["item_id"],
            data.get("score"),
            data.get("note"),
            data.get("date_created"),
            data.get("date_updated"),
        )
        # asyncpg execute returns e.g. "INSERT 0 1"; treat any non-empty status as 1 row affected
        return 1 if status else 0

    async def delete_from_postgres(self, entity_id: Any) -> None:
        key = self.normalize_pk(entity_id)
        await self.l3_pool.execute(
            "DELETE FROM fake_refs WHERE conversation_id = $1 AND item_id = $2",
            key[0],
            key[1],
        )

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        return json.loads(data)


class _InMemoryNatsBus:
    """nats mock with pub/sub + KV store for cross-pod simulation."""

    def __init__(self) -> None:
        self.kv: dict[str, bytes] = {}
        self._subs: dict[str, list[Any]] = {}

    def bucket_name(self, suffix: str) -> str:
        return f"it-{suffix}"

    async def get(self, bucket: str, key: str) -> bytes | None:
        return self.kv.get(key)

    async def put(self, bucket: str, key: str, value: bytes) -> bool:
        self.kv[key] = value
        return True

    async def delete(self, bucket: str, key: str) -> bool:
        self.kv.pop(key, None)
        return True

    async def publish(self, subject: str, data: bytes) -> bool:
        for cb in self._subs.get(subject, []):
            try:
                await cb(data)
            except Exception:
                pass
        return True

    async def subscribe(
        self,
        subject: str,
        callback: Any | None = None,
        *,
        cb: Any | None = None,
    ) -> None:
        chosen = cb if cb is not None else callback
        self._subs.setdefault(subject, []).append(chosen)


def _build_pod(
    pool: asyncpg.Pool,
    nats: _InMemoryNatsBus,
    cfg: DefaultCoreConfig,
) -> tuple[FakeRefCollection, CollectionRegistry, SQLiteBackend]:
    """construct a per-pod (L1, L2, L3) stack."""
    l1 = SQLiteBackend(db_name=f"pod_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1, l2_client=nats, l3_pool=pool)
    coll = FakeRefCollection(reg, cfg, nats_client=nats)
    return coll, reg, l1


@pytest.fixture
def cfg_always() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


class TestCompositePkThreeTier:
    async def test_save_populates_all_tiers(
        self, pg_pool: asyncpg.Pool, cfg_always: DefaultCoreConfig
    ) -> None:
        """save via composite-pk collection persists to L3, L1, L2."""
        nats = _InMemoryNatsBus()
        coll, _reg, l1 = _build_pod(pg_pool, nats, cfg_always)
        try:
            entity = coll.create(
                {
                    "conversation_id": "conv-i1",
                    "item_id": "item-i1",
                    "score": 7,
                    "note": "integration",
                }
            )
            await coll.save_entity(entity)

            async with pg_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT score, note FROM fake_refs WHERE conversation_id = $1 AND item_id = $2",
                    "conv-i1",
                    "item-i1",
                )
            assert row is not None
            assert row["score"] == 7

            l1_row = coll.get_row_sync(("conv-i1", "item-i1"))
            assert l1_row is not None

            assert "fake_refs.conv-i1:item-i1" in nats.kv
        finally:
            l1.reset()

    async def test_cold_start_l3_pull_through(
        self, pg_pool: asyncpg.Pool, cfg_always: DefaultCoreConfig
    ) -> None:
        """pod restart: fresh L1 + existing L3 row resolves via get."""
        nats = _InMemoryNatsBus()
        # pod 1 writes
        coll_a, _reg_a, l1_a = _build_pod(pg_pool, nats, cfg_always)
        entity = coll_a.create(
            {
                "conversation_id": "conv-cold",
                "item_id": "item-cold",
                "score": 42,
                "note": "seed",
            }
        )
        try:
            await coll_a.save_entity(entity)
        finally:
            l1_a.reset()

        # clear L2 to force L3 pull-through
        nats.kv.clear()

        # pod 2 starts fresh; same L3
        coll_b, _reg_b, l1_b = _build_pod(pg_pool, nats, cfg_always)
        try:
            loaded = await coll_b.get(("conv-cold", "item-cold"))
            assert loaded is not None
            assert loaded.score == 42
            # now cached in pod 2's L1
            row = coll_b.get_row_sync(("conv-cold", "item-cold"))
            assert row is not None
        finally:
            l1_b.reset()

    async def test_cross_pod_invalidation_composite(
        self, pg_pool: asyncpg.Pool, cfg_always: DefaultCoreConfig
    ) -> None:
        """pod A save emits ``ids`` envelope; pod B evicts composite L1 row."""
        nats = _InMemoryNatsBus()
        coll_a, _reg_a, l1_a = _build_pod(pg_pool, nats, cfg_always)
        coll_b, reg_b, l1_b = _build_pod(pg_pool, nats, cfg_always)
        try:
            await reg_b.start_invalidation_listener(nats)

            # seed a row from pod A
            entity = coll_a.create(
                {
                    "conversation_id": "conv-xp",
                    "item_id": "item-xp",
                    "score": 1,
                    "note": "v1",
                }
            )
            await coll_a.save_entity(entity)

            # pod B loads the row — populates its own L1
            loaded_b = await coll_b.get(("conv-xp", "item-xp"))
            assert loaded_b is not None
            before = coll_b.get_row_sync(("conv-xp", "item-xp"))
            assert before is not None

            # pod A updates -> publishes ids: ["conv-xp", "item-xp"]
            updated = coll_a.create(
                {
                    "conversation_id": "conv-xp",
                    "item_id": "item-xp",
                    "score": 999,
                    "note": "v2",
                }
            )
            # mark as existing so save_entity takes the UPSERT path
            object.__setattr__(updated, "_is_new", False)
            updated.original_date_updated = before.get("date_updated")
            await coll_a.save_entity(updated)

            # pod B's L1 MUST have been evicted by the invalidation signal
            after = coll_b.get_row_sync(("conv-xp", "item-xp"))
            assert after is None
        finally:
            l1_a.reset()
            l1_b.reset()
