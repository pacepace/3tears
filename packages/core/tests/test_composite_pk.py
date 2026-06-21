"""unit tests for composite-PK support in BaseCollection + SQLiteBackend.

phase 8.5l-1 foundation: BaseCollection accepts
``primary_key_column = ("col_a", "col_b")`` for composite-pk tables,
normalizes caller-supplied ids via :meth:`BaseCollection.normalize_pk`,
and routes through L1 (:class:`SQLiteBackend`) + L2 (NATS KV) + the
invalidation wire envelope uniformly. these tests pin:

1. single-pk collections continue to work unchanged (scalar shape).
2. composite-pk collections persist + retrieve + delete via tuple ids.
3. L1 cache keys respect composite pk (SQLite ``WHERE a = ? AND b = ?``).
4. L2 cache keys join composite values with ``":"`` (joined-string).
5. invalidation wire envelope carries ``ids`` as an array whose
   length matches the target collection's pk column count.
6. cross-pod L1 eviction via invalidation works for composite-pk
   collections.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import (
    CacheInvalidationMessage,
    CollectionRegistry,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.nats import Subjects


# ---------------------------------------------------------------------------
# SQLiteBackend direct tests — composite-pk DDL + upsert + select + delete
# ---------------------------------------------------------------------------


def _composite_metadata() -> MetaData:
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
    Table(
        "single_pk_table",
        md,
        Column("id", String(255), primary_key=True),
        Column("name", String(255)),
        Column("date_updated", DateTime),
    )
    return md


@pytest.fixture()
def backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_comp_{uuid.uuid4().hex[:8]}")
    b.initialize(_composite_metadata())
    yield b
    b.reset()


class TestSQLiteCompositeUpsert:
    """verify composite-pk upsert honours ON CONFLICT on pk tuple."""

    def test_insert_then_update_same_composite_key(self, backend: SQLiteBackend) -> None:
        """second upsert on same (conv_id, item_id) updates non-pk columns."""
        pk = ("conv-A", "item-1")
        backend.upsert(
            "fake_refs",
            {"conversation_id": "conv-A", "item_id": "item-1", "score": 10, "note": "first"},
            primary_key=("conversation_id", "item_id"),
        )
        backend.upsert(
            "fake_refs",
            {"conversation_id": "conv-A", "item_id": "item-1", "score": 99, "note": "second"},
            primary_key=("conversation_id", "item_id"),
        )
        row = backend.select_by_id("fake_refs", pk, ("conversation_id", "item_id"))
        assert row is not None
        assert row["score"] == 99
        assert row["note"] == "second"

    def test_different_composite_keys_coexist(self, backend: SQLiteBackend) -> None:
        """distinct (conv, item) pairs do not collide on insert."""
        pk_cols = ("conversation_id", "item_id")
        backend.upsert(
            "fake_refs",
            {"conversation_id": "conv-A", "item_id": "item-1", "score": 1, "note": "n1"},
            primary_key=pk_cols,
        )
        backend.upsert(
            "fake_refs",
            {"conversation_id": "conv-A", "item_id": "item-2", "score": 2, "note": "n2"},
            primary_key=pk_cols,
        )
        backend.upsert(
            "fake_refs",
            {"conversation_id": "conv-B", "item_id": "item-1", "score": 3, "note": "n3"},
            primary_key=pk_cols,
        )
        r1 = backend.select_by_id("fake_refs", ("conv-A", "item-1"), pk_cols)
        r2 = backend.select_by_id("fake_refs", ("conv-A", "item-2"), pk_cols)
        r3 = backend.select_by_id("fake_refs", ("conv-B", "item-1"), pk_cols)
        assert r1 is not None and r1["score"] == 1
        assert r2 is not None and r2["score"] == 2
        assert r3 is not None and r3["score"] == 3


class TestSQLiteCompositeSelect:
    """verify composite-pk select emits ``WHERE a = ? AND b = ?``."""

    def test_select_hit(self, backend: SQLiteBackend) -> None:
        pk_cols = ("conversation_id", "item_id")
        backend.upsert(
            "fake_refs",
            {"conversation_id": "conv-X", "item_id": "item-9", "score": 42, "note": "hit"},
            primary_key=pk_cols,
        )
        row = backend.select_by_id("fake_refs", ("conv-X", "item-9"), pk_cols)
        assert row is not None
        assert row["score"] == 42

    def test_select_miss_returns_none(self, backend: SQLiteBackend) -> None:
        pk_cols = ("conversation_id", "item_id")
        row = backend.select_by_id("fake_refs", ("missing", "gone"), pk_cols)
        assert row is None

    def test_partial_match_on_one_column_is_miss(self, backend: SQLiteBackend) -> None:
        """row with matching conversation_id but different item_id does not match."""
        pk_cols = ("conversation_id", "item_id")
        backend.upsert(
            "fake_refs",
            {"conversation_id": "conv-X", "item_id": "item-9", "score": 1, "note": "a"},
            primary_key=pk_cols,
        )
        row = backend.select_by_id("fake_refs", ("conv-X", "other-item"), pk_cols)
        assert row is None


class TestSQLiteCompositeDelete:
    """verify composite-pk delete targets one row only."""

    def test_delete_hits_only_target(self, backend: SQLiteBackend) -> None:
        pk_cols = ("conversation_id", "item_id")
        backend.upsert(
            "fake_refs",
            {"conversation_id": "conv-A", "item_id": "item-1", "score": 1, "note": "a"},
            primary_key=pk_cols,
        )
        backend.upsert(
            "fake_refs",
            {"conversation_id": "conv-A", "item_id": "item-2", "score": 2, "note": "b"},
            primary_key=pk_cols,
        )
        backend.delete_by_id("fake_refs", ("conv-A", "item-1"), pk_cols)
        assert backend.select_by_id("fake_refs", ("conv-A", "item-1"), pk_cols) is None
        # sibling row survives
        r = backend.select_by_id("fake_refs", ("conv-A", "item-2"), pk_cols)
        assert r is not None
        assert r["score"] == 2


class TestSQLiteCompositeSelectBatch:
    """verify composite-pk select_batch builds a disjunction of per-pk predicates."""

    def test_batch_returns_matching_rows(self, backend: SQLiteBackend) -> None:
        pk_cols = ("conversation_id", "item_id")
        for idx in range(4):
            backend.upsert(
                "fake_refs",
                {
                    "conversation_id": f"conv-{idx}",
                    "item_id": f"item-{idx}",
                    "score": idx,
                    "note": f"n{idx}",
                },
                primary_key=pk_cols,
            )
        rows = backend.select_batch(
            "fake_refs",
            [("conv-0", "item-0"), ("conv-2", "item-2")],
            pk_cols,
        )
        assert len(rows) == 2
        scores = sorted(r["score"] for r in rows)
        assert scores == [0, 2]


class TestSQLiteSinglePkBackwardCompat:
    """verify single-pk shape continues to work unchanged."""

    def test_single_pk_scalar_roundtrip(self, backend: SQLiteBackend) -> None:
        backend.upsert("single_pk_table", {"id": "e1", "name": "Alice"})
        row = backend.select_by_id("single_pk_table", "e1")
        assert row is not None
        assert row["name"] == "Alice"
        backend.delete_by_id("single_pk_table", "e1")
        assert backend.select_by_id("single_pk_table", "e1") is None

    def test_single_pk_tuple_input_is_accepted(self, backend: SQLiteBackend) -> None:
        """scalar pk caller may also pass a 1-tuple; backend accepts both."""
        backend.upsert("single_pk_table", {"id": "e2", "name": "Bob"}, primary_key="id")
        row_via_scalar = backend.select_by_id("single_pk_table", "e2", "id")
        row_via_tuple = backend.select_by_id("single_pk_table", ("e2",), ("id",))
        assert row_via_scalar is not None
        assert row_via_tuple is not None
        assert row_via_scalar == row_via_tuple


class TestSQLiteArityMismatch:
    """pk-shape validation surfaces clear errors."""

    def test_too_few_values_raises(self, backend: SQLiteBackend) -> None:
        with pytest.raises(ValueError, match="arity mismatch"):
            backend.select_by_id("fake_refs", "conv-only", ("conversation_id", "item_id"))

    def test_too_many_values_raises(self, backend: SQLiteBackend) -> None:
        with pytest.raises(ValueError, match="arity mismatch"):
            backend.select_by_id("single_pk_table", ("a", "b"), "id")


# ---------------------------------------------------------------------------
# BaseCollection composite-pk tests
# ---------------------------------------------------------------------------


class FakeRefEntity(BaseEntity):
    """composite-pk entity — identity is ``(conversation_id, item_id)``.

    overrides ``__init__`` to build ``_id`` as a tuple of the two pk
    values so collection-level id operations receive the tuple
    uniformly. subclasses for future composite-pk collections follow
    the same shape.
    """

    primary_key_field = "conversation_id"  # unused — __init__ overrides _id

    def __init__(self, data: dict[str, Any], is_new: bool = True, collection: Any = None) -> None:
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(self, "_id", (data["conversation_id"], data["item_id"]))


class FakeRefCollection(BaseCollection[FakeRefEntity]):
    """composite-pk collection — pk = ``(conversation_id, item_id)``.

    in-memory l3_rows is keyed by tuple pk so the ``_fetch``/
    ``_save``/``_delete`` implementations exercise the composite
    contract end-to-end.
    """

    primary_key_column: str | tuple[str, ...] = ("conversation_id", "item_id")

    def __init__(
        self,
        registry: CollectionRegistry,
        config: DefaultCoreConfig,
        nats_client: Any = None,
        l3_rows: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    ) -> None:
        self._l3_rows: dict[tuple[Any, ...], dict[str, Any]] = l3_rows if l3_rows is not None else {}
        super().__init__(registry, config, nats_client, write_buffer=None)

    @property
    def table_name(self) -> str:
        return "fake_refs"

    @property
    def entity_class(self) -> type[FakeRefEntity]:
        return FakeRefEntity

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        key = self.normalize_pk(entity_id)
        return self._l3_rows.get(key)

    async def save_to_store(self, data: dict[str, Any], original_timestamp: datetime | None = None) -> int:
        key = (data["conversation_id"], data["item_id"])
        self._l3_rows[key] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        key = self.normalize_pk(entity_id)
        self._l3_rows.pop(key, None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        return json.loads(data)


def _nats_mock() -> AsyncMock:
    """typed-wrapper NATS mock with in-memory KV bucket."""
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
    nats.subscribe_typed = AsyncMock()
    nats.store = store
    nats.bucket = bucket
    return nats


@pytest.fixture()
def composite_l1() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_comp_coll_{uuid.uuid4().hex[:8]}")
    b.initialize(_composite_metadata())
    yield b
    from threetears.core._bridge import drain, shutdown

    drain()
    shutdown()
    b.reset()


@pytest.fixture()
def composite_registry(composite_l1: SQLiteBackend) -> CollectionRegistry:
    reg = CollectionRegistry()
    reg.configure(l1_backend=composite_l1)
    return reg


@pytest.fixture()
def always_cfg() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


class TestNormalizePk:
    """verify BaseCollection.normalize_pk contract."""

    def test_scalar_wraps_into_1_tuple_on_single_pk(
        self, composite_registry: CollectionRegistry, always_cfg: DefaultCoreConfig
    ) -> None:
        """scalar input is wrapped in a 1-tuple when pk column is single."""

        class SingleColl(BaseCollection[BaseEntity]):
            primary_key_column = "id"

            @property
            def table_name(self) -> str:
                return "single_pk_table"

            @property
            def entity_class(self) -> type[BaseEntity]:
                return BaseEntity

            async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
                return None

            async def save_to_store(self, data: dict[str, Any], original_timestamp: datetime | None = None) -> int:
                return 1

            async def delete_from_store(self, entity_id: Any) -> None:
                return None

            def serialize(self, data: dict[str, Any]) -> bytes:
                return b""

            def deserialize(self, data: bytes) -> dict[str, Any]:
                return {}

        coll = SingleColl(composite_registry, always_cfg)
        assert coll.primary_key_columns == ("id",)
        assert coll.normalize_pk("e1") == ("e1",)
        assert coll.normalize_pk(("e1",)) == ("e1",)

    def test_composite_requires_tuple(
        self, composite_registry: CollectionRegistry, always_cfg: DefaultCoreConfig
    ) -> None:
        """composite-pk collection rejects scalar input with clear error."""
        coll = FakeRefCollection(composite_registry, always_cfg, nats_client=_nats_mock())
        assert coll.primary_key_columns == ("conversation_id", "item_id")
        with pytest.raises(ValueError, match="arity mismatch"):
            coll.normalize_pk("conv-A")

    def test_composite_tuple_passes_through(
        self, composite_registry: CollectionRegistry, always_cfg: DefaultCoreConfig
    ) -> None:
        coll = FakeRefCollection(composite_registry, always_cfg, nats_client=_nats_mock())
        assert coll.normalize_pk(("conv-A", "item-1")) == ("conv-A", "item-1")


class TestL2Key:
    """verify composite pk joins L2 key parts with ``":"``."""

    def test_single_pk_key_has_no_colon(
        self, composite_registry: CollectionRegistry, always_cfg: DefaultCoreConfig
    ) -> None:
        class SingleColl(BaseCollection[BaseEntity]):
            primary_key_column = "id"

            @property
            def table_name(self) -> str:
                return "single_pk_table"

            @property
            def entity_class(self) -> type[BaseEntity]:
                return BaseEntity

            async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
                return None

            async def save_to_store(self, data: dict[str, Any], original_timestamp: datetime | None = None) -> int:
                return 1

            async def delete_from_store(self, entity_id: Any) -> None:
                return None

            def serialize(self, data: dict[str, Any]) -> bytes:
                return b""

            def deserialize(self, data: bytes) -> dict[str, Any]:
                return {}

        coll = SingleColl(composite_registry, always_cfg)
        assert coll.l2_key("e1") == "single_pk_table.e1"
        # tuple input also valid for single-pk
        assert coll.l2_key(("e1",)) == "single_pk_table.e1"

    def test_composite_key_joins_with_underscore(
        self, composite_registry: CollectionRegistry, always_cfg: DefaultCoreConfig
    ) -> None:
        """composite-pk keys join components with ``_``.

        the JetStream KV grammar (``^[-/_=.a-zA-Z0-9]+$`` per
        ``nats-server`` ``kv.go``) rejects ``:`` with
        ``nats: JetStream.InvalidKeyError`` -- using ``:`` here was
        the cause of the every-conversation-write KV warning storm.
        """
        coll = FakeRefCollection(composite_registry, always_cfg, nats_client=_nats_mock())
        assert coll.l2_key(("conv-A", "item-1")) == "fake_refs.conv-A_item-1"

    def test_composite_key_passes_jetstream_kv_grammar(
        self, composite_registry: CollectionRegistry, always_cfg: DefaultCoreConfig
    ) -> None:
        """composite-pk keys with realistic UUID + tuple shapes pass
        the JetStream KV ``valid-key`` regex.

        regression for the conversations.<agent_uuid>:<conv_uuid>
        key shape that tripped ``nats: JetStream.InvalidKeyError`` on
        every chat dispatch.
        """
        import re
        from uuid import uuid4

        # exact regex copied from nats-server cmd/nats-server/kv.go
        jetstream_kv_valid_key = re.compile(r"^[-/_=.a-zA-Z0-9]+$")

        coll = FakeRefCollection(composite_registry, always_cfg, nats_client=_nats_mock())

        # UUID-pair PK (conversations / memories / media / etc.)
        agent_id = uuid4()
        conv_id = uuid4()
        key_uuid_pair = coll.l2_key((str(agent_id), str(conv_id)))
        assert jetstream_kv_valid_key.match(key_uuid_pair), f"l2_key {key_uuid_pair!r} violates JetStream KV grammar"

        # slug-pair PK (existing FakeRef fixture)
        key_slug_pair = coll.l2_key(("conv-A", "item-1"))
        assert jetstream_kv_valid_key.match(key_slug_pair), f"l2_key {key_slug_pair!r} violates JetStream KV grammar"


class TestCollectionOps:
    """end-to-end collection ops on composite-pk tables."""

    @pytest.mark.asyncio
    async def test_save_then_get_roundtrip(
        self, composite_registry: CollectionRegistry, always_cfg: DefaultCoreConfig
    ) -> None:
        """save_entity persists to L3, populates L1, roundtrips via get."""
        nats = _nats_mock()
        l3_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
        coll = FakeRefCollection(composite_registry, always_cfg, nats_client=nats, l3_rows=l3_rows)

        entity = coll.create({"conversation_id": "conv-A", "item_id": "item-1", "score": 10, "note": "hello"})
        await coll.save_entity(entity)

        # L3 keyed by composite tuple
        assert ("conv-A", "item-1") in l3_rows
        # L1 has the row
        l1_row = coll.get_row_sync(("conv-A", "item-1"))
        assert l1_row is not None
        assert l1_row["score"] == 10
        # L2 key joins composite pk components with ``_`` so the key
        # passes the JetStream KV grammar (``:`` is rejected with
        # ``InvalidKeyError`` -- see test_composite_key_passes_jetstream_kv_grammar).
        assert "fake_refs.conv-A_item-1" in nats.store

        # subsequent get hits L1 (no L2 call)
        nats.get.reset_mock()
        loaded = await coll.get(("conv-A", "item-1"))
        assert loaded is not None
        assert loaded.score == 10
        nats.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_l1_miss_l3_hit_promotes(
        self, composite_registry: CollectionRegistry, always_cfg: DefaultCoreConfig
    ) -> None:
        """cold L1 -> L3 hit via get() with tuple id promotes to L1+L2."""
        nats = _nats_mock()
        l3_rows: dict[tuple[Any, ...], dict[str, Any]] = {
            ("conv-B", "item-9"): {
                "conversation_id": "conv-B",
                "item_id": "item-9",
                "score": 7,
                "note": "from_l3",
            }
        }
        coll = FakeRefCollection(composite_registry, always_cfg, nats_client=nats, l3_rows=l3_rows)

        loaded = await coll.get(("conv-B", "item-9"))
        assert loaded is not None
        assert loaded.score == 7
        # now in L1
        row = coll.get_row_sync(("conv-B", "item-9"))
        assert row is not None

    @pytest.mark.asyncio
    async def test_delete_removes_from_all_tiers(
        self, composite_registry: CollectionRegistry, always_cfg: DefaultCoreConfig
    ) -> None:
        nats = _nats_mock()
        l3_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
        coll = FakeRefCollection(composite_registry, always_cfg, nats_client=nats, l3_rows=l3_rows)

        entity = coll.create({"conversation_id": "conv-A", "item_id": "item-1", "score": 10, "note": "x"})
        await coll.save_entity(entity)

        ok = await coll.delete(("conv-A", "item-1"))
        assert ok is True
        # L3 gone
        assert ("conv-A", "item-1") not in l3_rows
        # L1 gone
        row = coll.get_row_sync(("conv-A", "item-1"))
        assert row is None
        # L2 gone
        assert "fake_refs.conv-A:item-1" not in nats.store


# ---------------------------------------------------------------------------
# Invalidation wire format tests (``ids`` field)
# ---------------------------------------------------------------------------


class TestInvalidationWireFormat:
    """verify the publish/subscribe envelope carries ``ids`` (plural)."""

    @pytest.mark.asyncio
    async def test_publish_emits_ids_field_single_pk(self) -> None:
        """single-pk publisher emits a typed envelope with length-1 ``ids``."""
        captured: list[CacheInvalidationMessage] = []
        nats = AsyncMock()

        async def _publish(*, subject: Any, message: CacheInvalidationMessage, reply_to: Any = None) -> None:  # noqa: ARG001
            captured.append(message)

        nats.publish = AsyncMock(side_effect=_publish)

        reg = CollectionRegistry()
        await reg.publish_invalidation(nats, "some_table", "the-id")

        assert len(captured) == 1
        assert captured[0].table == "some_table"
        assert captured[0].ids == ["the-id"]

    @pytest.mark.asyncio
    async def test_publish_emits_ids_field_composite_pk(self) -> None:
        """composite-pk publisher emits a typed envelope matching pk arity."""
        captured: list[CacheInvalidationMessage] = []
        nats = AsyncMock()

        async def _publish(*, subject: Any, message: CacheInvalidationMessage, reply_to: Any = None) -> None:  # noqa: ARG001
            captured.append(message)

        nats.publish = AsyncMock(side_effect=_publish)

        reg = CollectionRegistry()
        await reg.publish_invalidation(nats, "fake_refs", ("conv-A", "item-1"))

        assert len(captured) == 1
        assert captured[0].table == "fake_refs"
        assert captured[0].ids == ["conv-A", "item-1"]

    @pytest.mark.asyncio
    async def test_subscriber_evicts_composite_row(
        self, composite_registry: CollectionRegistry, always_cfg: DefaultCoreConfig
    ) -> None:
        """subscriber receives a typed envelope, decodes tuple, evicts L1."""
        nats = _nats_mock()
        l3_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
        coll = FakeRefCollection(composite_registry, always_cfg, nats_client=nats, l3_rows=l3_rows)

        subscribers: list[Any] = []

        async def _subscribe_typed(*, subject: Any, cb: Any, message_type: Any, **kwargs: Any) -> None:  # noqa: ARG001
            subscribers.append(cb)

        nats.subscribe_typed = AsyncMock(side_effect=_subscribe_typed)

        # seed L1 with a composite-pk row
        coll.write_to_cache_sync(
            {"conversation_id": "conv-A", "item_id": "item-1", "score": 99, "note": "x"},
        )
        before = coll.get_row_sync(("conv-A", "item-1"))
        assert before is not None

        await composite_registry.start_invalidation_listener(nats)
        assert len(subscribers) == 1

        # dispatch a typed invalidation envelope
        await subscribers[0](
            CacheInvalidationMessage(table="fake_refs", ids=["conv-A", "item-1"]),
        )

        after = coll.get_row_sync(("conv-A", "item-1"))
        assert after is None

    @pytest.mark.asyncio
    async def test_subscriber_rejects_mismatched_arity(
        self, composite_registry: CollectionRegistry, always_cfg: DefaultCoreConfig
    ) -> None:
        """``ids`` array whose length disagrees with pk arity is ignored."""
        nats = _nats_mock()
        l3_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
        coll = FakeRefCollection(composite_registry, always_cfg, nats_client=nats, l3_rows=l3_rows)

        subscribers: list[Any] = []

        async def _subscribe_typed(*, subject: Any, cb: Any, message_type: Any, **kwargs: Any) -> None:  # noqa: ARG001
            subscribers.append(cb)

        nats.subscribe_typed = AsyncMock(side_effect=_subscribe_typed)

        coll.write_to_cache_sync(
            {"conversation_id": "conv-A", "item_id": "item-1", "score": 1, "note": "x"},
        )
        await composite_registry.start_invalidation_listener(nats)

        # typed envelope but with single id for a composite-pk collection -- ignored
        await subscribers[0](
            CacheInvalidationMessage(table="fake_refs", ids=["conv-A"]),
        )

        # row still in L1
        row = coll.get_row_sync(("conv-A", "item-1"))
        assert row is not None

    # NOTE: malformed-payload rejection is the typed wrapper's
    # responsibility (``model_validate_json`` returns a
    # :class:`pydantic.ValidationError` long before the message
    # reaches the registry callback). that contract is tested
    # directly against the wrapper in 3tears/packages/nats/tests.
