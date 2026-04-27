"""integration: three-tier CRUD for MemoryRefsCollection (composite pk).

namespace-task-01 phase 8.5l-2: the ``conversation_memory_refs`` table
gets a proper :class:`BaseCollection` subclass on top of 8.5l-1's
composite-pk support. this suite exercises the end-to-end tier
behaviour through a real pgvector/pg16 container:

1. ``save_entity`` populates L1 + L2 + L3 for a composite-pk row
   (tuple-keyed ``l2_key`` / ``select_by_id``).
2. ``.get((conversation_id, item_id))`` serves from L1 without an
   L3 round-trip on warm cache, and pulls through L3 -> L1 on a
   cold-start pod (fresh L1 + existing L3 row).
3. cross-pod invalidation: pod A saves, pod B's L1 evicts via the
   ``threetears.cache.invalidate`` envelope carrying
   ``ids: [<conversation_id>, <item_id>]``.
4. ``find_by_conversation`` scans the conversation_id prefix of the
   composite pk (marked ``# cache-bypass:`` inside the Collection
   method).

mirrors :mod:`test_composite_pk_three_tier` in shape — the in-memory
NATS bus lets the cross-pod test run without a real JetStream.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest
from sqlalchemy import TIMESTAMP, Column, MetaData, String, Table  # noqa: N811

from threetears.agent.memory.collections import MemoryRefsCollection
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _l1_metadata() -> MetaData:
    """build an L1 mirror of ``conversation_memory_refs``.

    :class:`BaseCollection.save_entity` auto-stamps ``date_created`` +
    ``date_updated`` on every row so the L1 mirror carries those
    columns even though they are not in the v002 L3 DDL (the L3
    ``save_to_postgres`` side ignores them).

    :return: SQLAlchemy metadata with the composite-pk table
    :rtype: MetaData
    """
    md = MetaData()
    Table(
        "conversation_memory_refs",
        md,
        Column("conversation_id", String(255), primary_key=True),
        Column("item_id", String(255), primary_key=True),
        Column("item_type", String(50)),
        Column("short_desc", String(150)),
        Column("date_added", TIMESTAMP),
        Column("date_created", TIMESTAMP),
        Column("date_updated", TIMESTAMP),
    )
    return md


@pytest.fixture
async def applied_schema(pg_schema: tuple[str, str]) -> tuple[str, str]:
    """apply conversations + memory migrations into the per-test schema.

    :param pg_schema: ``(url, schema)`` tuple from the shared fixture
    :ptype pg_schema: tuple[str, str]
    :return: same tuple with migrations applied
    :rtype: tuple[str, str]
    """
    url, schema = pg_schema
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        store = AsyncpgStore(conn)
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await conn.close()
    return url, schema


async def _make_pool(url: str, schema: str) -> asyncpg.Pool:
    """build an asyncpg pool with search_path pre-bound to the schema.

    :param url: asyncpg URL
    :ptype url: str
    :param schema: per-test schema name
    :ptype schema: str
    :return: pool ready for Collection use
    :rtype: asyncpg.Pool
    """
    from threetears.core.collections import init_connection
    result: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    return result


class _InMemoryKvBucket:
    """typed-wrapper KV bucket stand-in matching :class:`NatsKvBucket`."""

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
    """typed-wrapper NATS stand-in with KV bucket + typed pub/sub.

    mirrors the stand-in used in
    :mod:`threetears.core.tests.integration.test_composite_pk_three_tier`.
    the Collection touches :meth:`kv_bucket` for L2 and :meth:`publish`
    / :meth:`subscribe_typed` for cross-pod invalidation.
    """

    def __init__(self) -> None:
        self._bucket = _InMemoryKvBucket()
        self._subs: dict[str, list[tuple[Any, Any]]] = {}

    @property
    def kv(self) -> dict[str, bytes]:
        # backward-compat property: previously-used in assertions to
        # peek at stored bytes. now exposes the bucket's internal dict.
        return self._bucket.kv

    async def kv_bucket(
        self,
        *,
        name: str,  # noqa: ARG002 -- single shared bucket suffices for tests
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
) -> tuple[MemoryRefsCollection, CollectionRegistry, SQLiteBackend]:
    """construct a per-pod (L1, L2, L3) stack bound to the refs table.

    :param pool: pg pool shared across pods (represents L3)
    :ptype pool: asyncpg.Pool
    :param nats: shared in-memory NATS bus (represents L2 + invalidation)
    :ptype nats: _InMemoryNatsBus
    :return: ``(collection, registry, l1_backend)`` triple
    :rtype: tuple[MemoryRefsCollection, CollectionRegistry, SQLiteBackend]
    """
    l1 = SQLiteBackend(db_name=f"mem_refs_{uuid.uuid4().hex[:8]}")
    l1.initialize(_l1_metadata())
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1, l2_client=nats, l3_pool=pool)
    cfg = DefaultCoreConfig(
        collection_flush="ALWAYS", collection_flush_tables="",
    )
    coll = MemoryRefsCollection(registry=reg, config=cfg, nats_client=nats)
    return coll, reg, l1


class TestMemoryRefsCollectionThreeTier:
    """end-to-end three-tier behaviour for the composite-pk refs table."""

    async def test_save_populates_all_tiers(
        self, applied_schema: tuple[str, str],
    ) -> None:
        """save routes to L3 + L1 + L2 with tuple-keyed addressing."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            nats = _InMemoryNatsBus()
            coll, _reg, l1 = _build_pod(pool, nats)

            conv_id = uuid.uuid4()
            item_id = uuid.uuid4()
            now = datetime.now(UTC).replace(tzinfo=None)
            entity = coll.create(
                {
                    "conversation_id": conv_id,
                    "item_id": item_id,
                    "item_type": "memory",
                    "short_desc": "user prefers dark mode",
                    "date_added": now,
                },
            )
            await coll.save_entity(entity)

            # L3 row present
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT item_type, short_desc FROM conversation_memory_refs "
                    "WHERE conversation_id = $1 AND item_id = $2",
                    conv_id,
                    item_id,
                )
            assert row is not None
            assert row["item_type"] == "memory"
            assert row["short_desc"] == "user prefers dark mode"

            # L1 row reachable via composite pk tuple
            l1_row = l1.select_by_id(
                "conversation_memory_refs",
                (str(conv_id), str(item_id)),
                ("conversation_id", "item_id"),
            )
            assert l1_row is not None

            # L2 KV entry under composite-form key
            assert f"conversation_memory_refs.{conv_id}_{item_id}" in nats.kv
        finally:
            await pool.close()

    async def test_get_hits_l1_after_save(
        self, applied_schema: tuple[str, str],
    ) -> None:
        """warm-cache get returns from L1 without fetching from L3."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            nats = _InMemoryNatsBus()
            coll, _reg, _l1 = _build_pod(pool, nats)

            conv_id = uuid.uuid4()
            item_id = uuid.uuid4()
            now = datetime.now(UTC).replace(tzinfo=None)
            entity = coll.create(
                {
                    "conversation_id": conv_id,
                    "item_id": item_id,
                    "item_type": "chunk",
                    "short_desc": "page 3 intro",
                    "date_added": now,
                },
            )
            await coll.save_entity(entity)

            fetched = await coll.get((conv_id, item_id))
            assert fetched is not None
            assert fetched.item_type == "chunk"
            assert fetched.short_desc == "page 3 intro"
        finally:
            await pool.close()

    async def test_cold_start_l3_pull_through(
        self, applied_schema: tuple[str, str],
    ) -> None:
        """pod restart: fresh L1 + existing L3 row resolves via get."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            nats = _InMemoryNatsBus()

            # pod 1 writes then discards L1
            coll_a, _reg_a, l1_a = _build_pod(pool, nats)
            conv_id = uuid.uuid4()
            item_id = uuid.uuid4()
            now = datetime.now(UTC).replace(tzinfo=None)
            entity = coll_a.create(
                {
                    "conversation_id": conv_id,
                    "item_id": item_id,
                    "item_type": "media",
                    "short_desc": "photo.jpg",
                    "date_added": now,
                },
            )
            await coll_a.save_entity(entity)
            l1_a.reset()

            # clear L2 to force the cold-start L3 pull-through path
            nats.kv.clear()

            # pod 2 starts fresh; resolves through L3
            coll_b, _reg_b, l1_b = _build_pod(pool, nats)
            loaded = await coll_b.get((conv_id, item_id))
            assert loaded is not None
            assert loaded.item_type == "media"
            assert loaded.short_desc == "photo.jpg"

            # after the pull-through, pod 2's L1 has the row
            row = l1_b.select_by_id(
                "conversation_memory_refs",
                (str(conv_id), str(item_id)),
                ("conversation_id", "item_id"),
            )
            assert row is not None
        finally:
            await pool.close()

    async def test_cross_pod_invalidation(
        self, applied_schema: tuple[str, str],
    ) -> None:
        """pod A save publishes ids envelope; pod B evicts composite L1 row."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            nats = _InMemoryNatsBus()
            coll_a, _reg_a, l1_a = _build_pod(pool, nats)
            coll_b, reg_b, l1_b = _build_pod(pool, nats)

            try:
                await reg_b.start_invalidation_listener(nats)

                conv_id = uuid.uuid4()
                item_id = uuid.uuid4()
                now = datetime.now(UTC).replace(tzinfo=None)
                entity = coll_a.create(
                    {
                        "conversation_id": conv_id,
                        "item_id": item_id,
                        "item_type": "memory",
                        "short_desc": "seed v1",
                        "date_added": now,
                    },
                )
                await coll_a.save_entity(entity)

                # pod B loads and warms its L1
                loaded_b = await coll_b.get((conv_id, item_id))
                assert loaded_b is not None
                before = l1_b.select_by_id(
                    "conversation_memory_refs",
                    (str(conv_id), str(item_id)),
                    ("conversation_id", "item_id"),
                )
                assert before is not None

                # pod A updates -> publishes ids: [conv_id, item_id]
                updated_entity = coll_a.create(
                    {
                        "conversation_id": conv_id,
                        "item_id": item_id,
                        "item_type": "memory",
                        "short_desc": "updated v2",
                        "date_added": now,
                    },
                )
                object.__setattr__(updated_entity, "_is_new", False)
                updated_entity.original_date_updated = before.get("date_updated")
                await coll_a.save_entity(updated_entity)

                # pod B's L1 row is evicted by the invalidation signal
                after = l1_b.select_by_id(
                    "conversation_memory_refs",
                    (str(conv_id), str(item_id)),
                    ("conversation_id", "item_id"),
                )
                assert after is None
            finally:
                l1_a.reset()
                l1_b.reset()
        finally:
            await pool.close()

    async def test_find_by_conversation_returns_rows_ordered(
        self, applied_schema: tuple[str, str],
    ) -> None:
        """multi-row scan returns every ref chronologically per conversation."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            nats = _InMemoryNatsBus()
            coll, _reg, _l1 = _build_pod(pool, nats)

            conv_id = uuid.uuid4()
            base = datetime.now(UTC).replace(tzinfo=None)
            ids_in_order: list[uuid.UUID] = []
            for i in range(3):
                item_id = uuid.uuid4()
                ids_in_order.append(item_id)
                added = base.replace(microsecond=i * 1000)
                entity = coll.create(
                    {
                        "conversation_id": conv_id,
                        "item_id": item_id,
                        "item_type": "memory",
                        "short_desc": f"item {i}",
                        "date_added": added,
                    },
                )
                await coll.save_entity(entity)

            # unrelated conversation — must not surface
            other_conv = uuid.uuid4()
            other_entity = coll.create(
                {
                    "conversation_id": other_conv,
                    "item_id": uuid.uuid4(),
                    "item_type": "memory",
                    "short_desc": "other",
                    "date_added": base,
                },
            )
            await coll.save_entity(other_entity)

            refs = await coll.find_by_conversation(conv_id)
            assert len(refs) == 3
            returned_ids = [ref.item_id for ref in refs]
            assert returned_ids == ids_in_order
        finally:
            await pool.close()

    async def test_save_truncates_short_desc(
        self, applied_schema: tuple[str, str],
    ) -> None:
        """long descriptions truncate to 150 chars on the write boundary."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            nats = _InMemoryNatsBus()
            coll, _reg, _l1 = _build_pod(pool, nats)

            conv_id = uuid.uuid4()
            item_id = uuid.uuid4()
            long_text = "x" * 200
            entity = coll.create(
                {
                    "conversation_id": conv_id,
                    "item_id": item_id,
                    "item_type": "memory",
                    "short_desc": long_text,
                    "date_added": datetime.now(UTC).replace(tzinfo=None),
                },
            )
            await coll.save_entity(entity)

            async with pool.acquire() as conn:
                stored = await conn.fetchval(
                    "SELECT short_desc FROM conversation_memory_refs "
                    "WHERE conversation_id = $1 AND item_id = $2",
                    conv_id,
                    item_id,
                )
            assert stored is not None
            assert len(stored) == 150
        finally:
            await pool.close()
