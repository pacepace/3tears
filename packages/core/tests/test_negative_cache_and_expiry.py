"""BaseCollection negative caching and row expiry.

The contract this pins:

- a negative-caching collection records a full miss in L2, so repeated lookups of a key nobody
  wrote reach L3 once, across replicas, until the marker ages out;
- a marker can never mask a write: a reader's marker that races a writer's put loses, and a
  writer's put that follows a marker replaces it;
- a negative-caching collection's write paths raise when their L2 write fails, because a marker
  left behind would keep reporting the key absent;
- a row whose declared expiry has passed is absent at L1, L2 and L3, and an expired row can be
  superseded by a later write;
- a collection that opts into neither behaves exactly as before.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.nats import KvError

_SCOPE = "test-principal"
_TABLE = "denylist_entries"


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("reason", String(255)),
        Column("expires_at", DateTime(timezone=True)),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _Entry(BaseEntity):
    primary_key_field = "id"


class _Store:
    """the shared L3 every replica reads, counting how often it is asked."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.fetches = 0


class _DenylistCollection(BaseCollection[_Entry]):
    """a three-tier collection with no opt-ins; subclasses turn them on."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"expires_at", "date_created", "date_updated"})

    def __init__(self, registry: CollectionRegistry, config: DefaultCoreConfig, store: _Store) -> None:
        self._store = store
        super().__init__(registry, config)

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Entry]:
        return _Entry

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        self._store.fetches += 1
        row = self._store.rows.get(str(entity_id))
        return dict(row) if row is not None else None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        self._store.rows[str(data["id"])] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self._store.rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class _NegativeCaching(_DenylistCollection):
    negative_cache_max_age: ClassVar[timedelta | None] = timedelta(seconds=60)


class _Expiring(_DenylistCollection):
    expires_at_column: ClassVar[str | None] = "expires_at"


class _NegativeCachingAndExpiring(_DenylistCollection):
    negative_cache_max_age: ClassVar[timedelta | None] = timedelta(seconds=60)
    expires_at_column: ClassVar[str | None] = "expires_at"


class _HookedBucket(FakeKvBucket):
    """a real fake bucket that runs a hook before a create, to force a reader/writer interleaving."""

    def __init__(self, bucket_name: str) -> None:
        super().__init__(bucket_name)
        self.before_create: Callable[[], Awaitable[None]] | None = None
        self.fail_put = False

    async def create(self, *, key: str, value: bytes) -> int | None:
        hook, self.before_create = self.before_create, None
        if hook is not None:
            await hook()
        return await super().create(key=key, value=value)

    async def put(self, *, key: str, value: bytes) -> int:
        if self.fail_put:
            raise KvError("simulated L2 write failure")
        return await super().put(key=key, value=value)


class _HookedNats(FakeNatsClient):
    """one shared hooked bucket for every replica, as the real collections bucket is."""

    def __init__(self) -> None:
        super().__init__()
        self.bucket = _HookedBucket("collections")
        self.published: list[Any] = []

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        """record the invalidation broadcast every write sends; no replica here subscribes."""
        self.published.append(message)

    async def kv_bucket(
        self,
        *,
        name: str,
        ttl: object | None = None,
        storage: str = "memory",
        create_if_missing: bool = True,
        history: int = 1,
    ) -> _HookedBucket:
        return self.bucket


def _replica(
    cls: type[_DenylistCollection], nats: _HookedNats | None, store: _Store, *, with_l3: bool = True
) -> _DenylistCollection:
    """one replica: its own L1, the shared L2 and L3."""
    l1 = SQLiteBackend(db_name=f"negcache_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=l1,
        l2_client=nats,
        l3_pool=object() if with_l3 else None,  # type: ignore[arg-type]
        kv_key_scope=_SCOPE if nats is not None else None,
    )
    return cls(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), store)


async def _write(collection: _DenylistCollection, entity_id: str, *, expires_at: datetime | None = None) -> None:
    entity = collection.create({"id": entity_id, "reason": "revoked", "expires_at": expires_at})
    await collection.save_entity(entity)


class TestNegativeCaching:
    @pytest.mark.asyncio
    async def test_a_key_nobody_wrote_reaches_l3_once_across_replicas(self) -> None:
        nats, store = _HookedNats(), _Store()
        a, b = _replica(_NegativeCaching, nats, store), _replica(_NegativeCaching, nats, store)
        assert await a.get("never-revoked") is None
        assert await a.get("never-revoked") is None
        assert await b.get("never-revoked") is None
        assert store.fetches == 1

    @pytest.mark.asyncio
    async def test_without_opting_in_every_miss_reaches_l3(self) -> None:
        nats, store = _HookedNats(), _Store()
        plain = _replica(_DenylistCollection, nats, store)
        for _ in range(3):
            assert await plain.get("never-revoked") is None
        assert store.fetches == 3

    @pytest.mark.asyncio
    async def test_without_an_l3_pool_no_marker_is_written(self) -> None:
        nats, store = _HookedNats(), _Store()
        l2_only = _replica(_NegativeCaching, nats, store, with_l3=False)
        assert await l2_only.get("k") is None
        assert await nats.bucket.get(key=f"{_SCOPE}.{_TABLE}.k") is None

    @pytest.mark.asyncio
    async def test_a_write_after_the_marker_replaces_it_for_every_replica(self) -> None:
        nats, store = _HookedNats(), _Store()
        reader, writer, cold = (_replica(_NegativeCaching, nats, store) for _ in range(3))
        assert await reader.get("tok") is None  # marker written
        await _write(writer, "tok")
        fetched = await cold.get("tok")
        assert fetched is not None
        assert fetched.to_dict()["reason"] == "revoked"

    @pytest.mark.asyncio
    async def test_a_readers_marker_that_races_a_writers_put_loses(self) -> None:
        # the reader has missed L2 and L3 and is about to record the key absent; the writer lands
        # first. the reader's create must fail, and the revocation must stand.
        nats, store = _HookedNats(), _Store()
        reader, writer, cold = (_replica(_NegativeCaching, nats, store) for _ in range(3))

        async def writer_lands_first() -> None:
            await _write(writer, "tok")

        nats.bucket.before_create = writer_lands_first
        assert await reader.get("tok") is None  # its L3 read predated the write
        fetched = await cold.get("tok")
        assert fetched is not None, "a marker masked a write that landed before it"

    @pytest.mark.asyncio
    async def test_an_aged_out_marker_asks_l3_again_and_is_refreshed(self) -> None:
        nats, store = _HookedNats(), _Store()
        a = _replica(_NegativeCaching, nats, store)
        assert await a.get("k") is None
        key = f"{_SCOPE}.{_TABLE}.k"
        stale = b"\x00threetears.collections.absent\x00" + (datetime.now(UTC) - timedelta(hours=1)).isoformat().encode()
        entry = await nats.bucket.get_entry(key=key)
        assert entry is not None
        assert await nats.bucket.update(key=key, value=stale, revision=entry[1]) is not None

        b = _replica(_NegativeCaching, nats, store)
        assert await b.get("k") is None
        assert store.fetches == 2  # the aged marker did not answer
        refreshed = await nats.bucket.get(key=key)
        assert refreshed is not None and refreshed != stale

    @pytest.mark.asyncio
    async def test_a_failed_l2_write_raises_on_a_negative_caching_collection(self) -> None:
        nats, store = _HookedNats(), _Store()
        writer = _replica(_NegativeCaching, nats, store)
        nats.bucket.fail_put = True
        with pytest.raises(KvError, match="absent-marker"):
            await _write(writer, "tok")
        assert "tok" in store.rows  # the L3 write landed; the caller is told to retry

    @pytest.mark.asyncio
    async def test_a_failed_l2_write_still_degrades_without_opting_in(self) -> None:
        nats, store = _HookedNats(), _Store()
        writer = _replica(_DenylistCollection, nats, store)
        nats.bucket.fail_put = True
        await _write(writer, "tok")
        assert "tok" in store.rows


class TestRowExpiry:
    @pytest.mark.asyncio
    async def test_an_expired_row_is_absent_to_the_reads_that_answer_existence(self) -> None:
        nats, store = _HookedNats(), _Store()
        coll = _replica(_Expiring, nats, store)
        await _write(coll, "old", expires_at=datetime.now(UTC) - timedelta(seconds=1))
        assert await coll.get("old") is None
        assert await coll.ensure("old") is None
        with pytest.raises(KeyError):
            coll["old"]

    @pytest.mark.asyncio
    async def test_an_entity_held_past_its_expiry_can_still_be_written(self) -> None:
        # reporting reads serve the entity's own internals; hiding its row there would turn this
        # save into "L1 cache miss in to_dict()" rather than a write.
        nats, store = _HookedNats(), _Store()
        coll = _replica(_Expiring, nats, store)
        entity = coll.create({"id": "held", "reason": "r", "expires_at": datetime.now(UTC) - timedelta(seconds=1)})
        await coll.save_entity(entity)
        assert store.rows["held"]["reason"] == "r"

    @pytest.mark.asyncio
    async def test_an_expired_row_is_absent_from_l2_and_l3(self) -> None:
        nats, store = _HookedNats(), _Store()
        writer = _replica(_Expiring, nats, store)
        await _write(writer, "old", expires_at=datetime.now(UTC) - timedelta(seconds=1))
        cold = _replica(_Expiring, nats, store)  # empty L1: must go through L2, then L3
        assert await cold.get("old") is None
        assert store.fetches == 1  # the expired L2 row sent it to L3, which is expired too

    @pytest.mark.asyncio
    async def test_a_row_expiring_later_and_a_row_that_never_expires_are_served(self) -> None:
        nats, store = _HookedNats(), _Store()
        coll = _replica(_Expiring, nats, store)
        await _write(coll, "live", expires_at=datetime.now(UTC) + timedelta(hours=1))
        await _write(coll, "forever", expires_at=None)
        cold = _replica(_Expiring, nats, store)
        assert await cold.get("live") is not None
        assert await cold.get("forever") is not None

    @pytest.mark.asyncio
    async def test_an_expired_row_becomes_a_marker_that_a_later_write_supersedes(self) -> None:
        nats, store = _HookedNats(), _Store()
        writer = _replica(_NegativeCachingAndExpiring, nats, store)
        await _write(writer, "tok", expires_at=datetime.now(UTC) - timedelta(seconds=1))
        reader = _replica(_NegativeCachingAndExpiring, nats, store)
        assert await reader.get("tok") is None  # expired row replaced by an absent-marker
        assert await reader.get("tok") is None
        assert store.fetches == 1

        await _write(writer, "tok", expires_at=datetime.now(UTC) + timedelta(hours=1))
        cold = _replica(_NegativeCachingAndExpiring, nats, store)
        assert await cold.get("tok") is not None, "an expired row's marker masked a later write"
