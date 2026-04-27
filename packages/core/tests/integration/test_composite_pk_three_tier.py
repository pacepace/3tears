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
from collections.abc import AsyncIterator
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


# canonical testcontainer harness -- single ``pytest_plugins`` entry
# pulls in ``db_container`` / ``db_image`` from
# :mod:`threetears.core.testing.fixtures` (test-harness-task-01).

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def db_image() -> str:
    """pin pgvector/pg16; this suite exercises the ``vector`` codec path."""
    return "pgvector/pgvector:pg16"


@pytest.fixture(scope="module")
def pg_url(db_container: str) -> str:
    """alias for :func:`threetears.core.testing.fixtures.db_container`.

    legacy name retained so existing fixture wiring (``pg_pool`` etc.)
    keeps working without touching every test body.
    """
    return db_container


@pytest.fixture
async def pg_pool(pg_url: str) -> AsyncIterator[asyncpg.Pool]:
    """per-test pool with a fresh table for isolation.

    binds the canonical 3tears jsonb text codec via
    :func:`threetears.core.collections.init_connection`; required for
    ``$N::jsonb`` cast paths to receive Python dicts (the codec
    encodes once -- ``_encode_jsonb`` is a typed pass-through).
    """
    from threetears.core.collections import init_connection
    pool: asyncpg.Pool = await asyncpg.create_pool(
        pg_url, min_size=1, max_size=4, init=init_connection,
    )
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


class _InMemoryKvBucket:
    """typed-wrapper KV bucket stand-in for cross-pod simulation."""

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
    """typed-wrapper NATS stand-in: KV bucket + typed pub/sub."""

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
        deadletter_on_error: bool = True,  # noqa: ARG002
    ) -> None:
        subject_str = str(subject)
        self._subs.setdefault(subject_str, []).append((cb, message_type))


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

            assert "fake_refs.conv-i1_item-i1" in nats.kv
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
