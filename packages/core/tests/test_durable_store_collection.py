"""DurableStoreCollection: a NON-SQL DurableStore drives the full three-tier CRUD.

The capability proof for collections-task-06: a collection backed by a structured,
SQL-free ``DurableStore`` (here an in-memory dict; the exact shape scriob's ``GitL3Backend``
takes) runs the standard ``save_entity`` → evict → ``get`` (pull-through from L3) →
``delete`` lifecycle through the same L1/L2 machinery as a SQL-backed collection. No SQL.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table
from unittest.mock import AsyncMock

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.durable_store import DurableStoreCollection
from threetears.core.collections.flush import WriteBuffer, flush_pending
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity


def _make_metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "scenes",
        metadata,
        Column("id", String(255), primary_key=True),
        Column("text", String(4096)),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return metadata


class _SceneEntity(BaseEntity):
    primary_key_field = "id"


class _InMemoryDurableStore:
    """A non-SQL DurableStore — the GitL3Backend shape, minus the git I/O."""

    def __init__(self) -> None:
        self.tables: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}

    @staticmethod
    def _key(pk: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(pk[k] for k in sorted(pk))

    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        row = self.tables.get(table, {}).get(self._key(pk))
        return dict(row) if row is not None else None

    async def upsert(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Sequence[str],
        on_conflict: str = "update",
        cas: datetime | None = None,
        conn: Any = None,
    ) -> int:
        t = self.tables.setdefault(table, {})
        key = tuple(row[c] for c in sorted(pk))
        if cas is not None and key in t and t[key].get("date_updated") != cas:
            return 0
        t[key] = dict(row)
        return 1

    async def delete(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> None:
        self.tables.get(table, {}).pop(self._key(pk), None)

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        return list(self.tables.get(table, {}).values())


class _SceneCollection(DurableStoreCollection[_SceneEntity]):
    primary_key_column = "id"

    @property
    def table_name(self) -> str:
        return "scenes"

    @property
    def entity_class(self) -> type[_SceneEntity]:
        return _SceneEntity


def _make_nats_mock() -> AsyncMock:
    store: dict[str, bytes] = {}

    async def _get(*, key: str) -> bytes | None:
        return store.get(key)

    async def _put(*, key: str, value: bytes) -> int:
        store[key] = value
        return len(store)

    async def _delete(*, key: str, revision: int | None = None) -> bool:  # noqa: ARG001
        existed = key in store
        store.pop(key, None)
        return existed or revision is None

    bucket = AsyncMock()
    bucket.get = AsyncMock(side_effect=_get)
    bucket.put = AsyncMock(side_effect=_put)
    bucket.delete = AsyncMock(side_effect=_delete)
    nats = AsyncMock()
    nats.kv_bucket = AsyncMock(return_value=bucket)
    nats.publish = AsyncMock()
    nats.store = store
    return nats


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_durable_{uuid.uuid4().hex[:8]}")
    b.initialize(_make_metadata())
    yield b
    from threetears.core._bridge import drain, shutdown

    drain()
    shutdown()


#: the principal scope this module's registries wire. every L2 key leads with one.
_TEST_SCOPE = "test-principal"


@pytest.fixture()
def registry(l1_backend: SQLiteBackend) -> CollectionRegistry:
    r = CollectionRegistry()
    r.configure(l1_backend=l1_backend, kv_key_scope=_TEST_SCOPE)
    return r


@pytest.mark.asyncio
async def test_durable_store_collection_full_three_tier_roundtrip(registry: CollectionRegistry) -> None:
    nats = _make_nats_mock()
    store = _InMemoryDurableStore()
    coll = _SceneCollection(registry, DefaultCoreConfig(), store, nats_client=nats)

    # save → the non-SQL DurableStore.upsert persists it (no SQL constructed anywhere)
    await coll.save_entity(_SceneEntity({"id": "scn-1", "text": "alpha"}, is_new=True))
    assert store.tables["scenes"][("scn-1",)]["text"] == "alpha"

    # evict L1 + L2 → next read MUST pull through fetch_from_store → DurableStore.fetch_one
    await coll.invalidate_cache("scn-1")
    assert coll.get_row_sync("scn-1") is None

    got = await coll.get("scn-1")
    assert got is not None and got.text == "alpha"
    assert coll.get_row_sync("scn-1") is not None  # promoted back to L1
    assert f"{_TEST_SCOPE}.scenes.scn-1" in nats.store  # promoted to L2

    # delete → DurableStore.delete removes it from the non-SQL durable tier
    await coll.delete("scn-1")
    assert ("scn-1",) not in store.tables["scenes"]
    assert await coll.get("scn-1") is None


@pytest.mark.asyncio
async def test_l2_resolves_from_registry_when_nats_client_not_passed(l1_backend: SQLiteBackend) -> None:
    """B1 regression: a DurableStoreCollection built WITHOUT an explicit nats_client must still
    pick up L2 (cross-pod invalidation) from the registry — not silently run single-pod L1-only.

    BaseCollection treats an explicit ``None`` as "L2 disabled"; only the
    ``NATS_CLIENT_FROM_REGISTRY`` sentinel triggers registry resolution. The collection's
    default must be the sentinel, so the git-backed multi-pod use case keeps L2.
    """
    nats = _make_nats_mock()
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1_backend, l2_client=nats, kv_key_scope=_TEST_SCOPE)
    # NB: nats_client is NOT passed — it must default to the registry sentinel.
    coll = _SceneCollection(reg, DefaultCoreConfig(), _InMemoryDurableStore())

    await coll.save_entity(_SceneEntity({"id": "scn-2", "text": "beta"}, is_new=True))
    assert f"{_TEST_SCOPE}.scenes.scn-2" in nats.store  # L2 written ⇒ cross-pod path is live, not disabled


class _TransactionalDurableStore(_InMemoryDurableStore):
    """A DurableStore whose ``transaction()`` batches writes, like a BigQuery backend.

    ``_InMemoryDurableStore`` writes through on every call, so it cannot tell whether the
    collection honoured a transaction handle. This one only commits what it was handed
    inside the ``async with``, which makes a dropped handle observable: the batch stays
    empty and the rows land individually instead.
    """

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[tuple[str, dict[str, Any]]]] = []
        self.direct_writes: list[tuple[str, dict[str, Any]]] = []

    async def upsert(  # type: ignore[override]
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Sequence[str],
        on_conflict: str = "update",
        cas: datetime | None = None,
        conn: Any = None,
    ) -> int:
        if conn is not None:
            conn.append((table, dict(row)))
            return 1
        self.direct_writes.append((table, dict(row)))
        return await super().upsert(table, row, pk=pk, on_conflict=on_conflict, cas=cas)

    @asynccontextmanager
    async def transaction(self, namespace: str | None = None) -> AsyncIterator[list[tuple[str, dict[str, Any]]]]:
        """Collect writes bound to the yielded handle and commit them together on exit."""
        batch: list[tuple[str, dict[str, Any]]] = []
        yield batch
        for table, row in batch:
            t = self.tables.setdefault(table, {})
            t[(row["id"],)] = dict(row)
        self.batches.append(batch)


@pytest.mark.asyncio
async def test_save_to_store_forwards_the_transaction_handle(registry: CollectionRegistry) -> None:
    """``save_to_store`` must pass ``conn`` through to ``DurableStore.upsert``.

    ``flush_pending``'s atomic-batch path opens ``backend.transaction()`` and threads the
    handle down through ``persist_to_store``. Dropping it here means each write executes
    on its own, so a backend implementing ``transaction()`` as a batch never batches.
    """
    store = _TransactionalDurableStore()
    coll = _SceneCollection(registry, DefaultCoreConfig(), store, nats_client=None)

    async with store.transaction() as conn:
        await coll.save_to_store({"id": "scn-tx", "text": "gamma"}, conn=conn)
        assert not store.direct_writes, "the write bypassed the transaction handle"

    assert store.tables["scenes"][("scn-tx",)]["text"] == "gamma"


@pytest.mark.asyncio
async def test_flush_batches_through_a_transactional_durable_store(l1_backend: SQLiteBackend) -> None:
    """A buffered flush over a transaction-capable ``DurableStore`` commits as ONE batch.

    The regression this guards is silent: ``flush_pending`` degrades to the per-entity loop
    by design, so a batch that never batches looks exactly like a batch that worked. Only
    the write-grouping distinguishes them.
    """
    store = _TransactionalDurableStore()
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1_backend, l3_pool=store)
    coll = _SceneCollection(reg, DefaultCoreConfig(), store, nats_client=None)

    buffer = WriteBuffer()
    for index in range(4):
        await buffer.add(coll.table_name, f"scn-{index}", {"id": f"scn-{index}", "text": f"v{index}"})

    flushed = await flush_pending(buffer, reg)

    assert flushed == 4
    assert not store.direct_writes, (
        f"{len(store.direct_writes)} write(s) bypassed the batch; the transaction handle "
        f"was dropped and the flush degraded to one durable write per entity"
    )
    assert len(store.batches) == 1
    assert len(store.batches[0]) == 4
