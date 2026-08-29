"""tests for create_dynamic_collection and DataStore wiring.

covers the framework-level findings from the integration-guide review
(PR #84):

- issue #85: ``fetch_from_store`` must return a plain ``dict`` even
  when the L3 pool yields ``asyncpg.Record`` rows -- iterating a Record
  yields values, not keys, which silently breaks
  ``SQLiteBackend.upsert`` during L3 -> L1 re-promotion.
- L2 wiring: collections resolve their NATS client from
  ``CollectionRegistry.get_l2_client`` when no constructor argument is
  supplied, so ``registry.configure(l2_client=...)`` is an effective
  wiring path and ``DataStore.create_table`` collections get L2.
- vector columns: dynamic collections render ``$N::vector`` casts on
  the write path and coerce pgvector text back to ``list[float]`` on
  the read path.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.exceptions import L2ScopeNotConfiguredError
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.collection_factory import create_dynamic_collection
from threetears.core.data.schema import ColumnDef, TableDef
from threetears.core.data.store import DataStore


# parity-with: asyncpg.Record
class FakeRecord:
    """mapping-shaped row that iterates VALUES, like ``asyncpg.Record``.

    ``dict(record)`` works (``keys()`` + ``__getitem__``), but plain
    iteration yields values -- the exact property that broke
    ``SQLiteBackend.upsert``'s ``[c for c in data if c in schema]``.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    def keys(self) -> Any:
        return self._data.keys()

    def values(self) -> Any:
        return self._data.values()

    def items(self) -> Any:
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Any:
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)


# parity-with: asyncpg.Pool
# parity-exempt: the L3 seam is duck-typed (async fetch/fetchrow/execute per BaseCollection.l3_pool docs); the fake implements exactly that consumed subset, and asyncpg.Pool's connection-management surface (acquire/release/copy_*/sizing) is outside the collection contract
class FakeAsyncpgPool:
    """duck-typed L3 pool whose ``fetch`` yields ``FakeRecord`` rows.

    mirrors the asyncpg surface the collections consume (``fetch`` /
    ``fetchrow`` / ``execute``). rows are keyed by the first query
    parameter (the pk value) so single-entity SELECTs resolve.
    """

    def __init__(self, rows: dict[str, dict[str, Any]] | None = None) -> None:
        self.rows = rows or {}
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[FakeRecord]:
        self.executed.append((sql, args))
        if args:
            row = self.rows.get(str(args[0]))
            return [FakeRecord(row)] if row is not None else []
        return [FakeRecord(row) for row in self.rows.values()]

    async def fetchrow(self, sql: str, *args: Any) -> FakeRecord | None:
        rows = await self.fetch(sql, *args)
        return rows[0] if rows else None

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append((sql, args))
        return "INSERT 0 1"


def _widgets_table() -> TableDef:
    return TableDef(
        name="widgets",
        columns=[
            ColumnDef(name="id", column_type="text", primary_key=True),
            ColumnDef(name="name", column_type="text", nullable=False),
            ColumnDef(name="score", column_type="integer"),
        ],
    )


def _make_l1() -> SQLiteBackend:
    return SQLiteBackend(db_name=f"test_factory_{uuid.uuid4().hex[:8]}")


class TestFetchReturnsDict:
    """issue #85: Record rows must be converted to dicts at the L3 border."""

    async def test_fetch_from_store_returns_plain_dict(self) -> None:
        pool = FakeAsyncpgPool(rows={"w1": {"id": "w1", "name": "sprocket", "score": 42}})
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        collection = create_dynamic_collection(
            table_def=_widgets_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
        )

        result = await collection.fetch_from_store("w1")

        assert type(result) is dict
        assert result == {"id": "w1", "name": "sprocket", "score": 42}

    async def test_l3_miss_repromotes_record_row_into_l1(self) -> None:
        """the end-to-end #85 repro: L1 miss -> L3 Record -> L1 upsert."""
        pool = FakeAsyncpgPool(rows={"w1": {"id": "w1", "name": "sprocket", "score": 42}})
        l1 = _make_l1()
        registry = CollectionRegistry()
        registry.configure(l1_backend=l1, l3_pool=pool)
        collection = create_dynamic_collection(
            table_def=_widgets_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
        )
        try:
            entity = await collection.get("w1")

            assert entity is not None
            cached = l1.select_by_id("widgets", "w1", "id")
            assert cached is not None
            assert cached["name"] == "sprocket"
            assert cached["score"] == 42
        finally:
            l1.reset()

    async def test_datastore_query_returns_dicts(self) -> None:
        pool = FakeAsyncpgPool(rows={"w1": {"id": "w1", "name": "sprocket", "score": 42}})
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        store = DataStore(uuid.uuid4(), registry, DefaultCoreConfig(collection_flush="ALWAYS"))

        rows = await store.query("SELECT * FROM widgets WHERE id = $1", "w1")

        assert rows == [{"id": "w1", "name": "sprocket", "score": 42}]
        assert all(type(row) is dict for row in rows)


class TestL2RegistryFallback:
    """collections resolve L2 from the registry when no arg is supplied."""

    def test_factory_resolves_l2_from_registry(self) -> None:
        l2_client = object()
        registry = CollectionRegistry()
        registry.configure(l3_pool=FakeAsyncpgPool(), l2_client=l2_client, kv_key_scope="hub")

        collection = create_dynamic_collection(
            table_def=_widgets_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
        )

        assert collection._nats_client is l2_client

    def test_explicit_none_disables_l2(self) -> None:
        registry = CollectionRegistry()
        registry.configure(l3_pool=FakeAsyncpgPool(), l2_client=object(), kv_key_scope="hub")

        collection = create_dynamic_collection(
            table_def=_widgets_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
            nats_client=None,
        )

        assert collection._nats_client is None

    def test_explicit_client_wins_over_registry(self) -> None:
        constructor_client = object()
        registry = CollectionRegistry()
        registry.configure(l3_pool=FakeAsyncpgPool(), l2_client=object(), kv_key_scope="hub")

        collection = create_dynamic_collection(
            table_def=_widgets_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
            nats_client=constructor_client,
        )

        assert collection._nats_client is constructor_client

    def test_bind_table_refuses_an_l2_client_with_no_registry_scope(self) -> None:
        """the third L2-wiring path was the ungated one.

        ``configure`` and ``BaseCollection(nats_client=)`` are both covered -- the first
        by its own raise, the second by ``l2_key``'s backstop. A registry wired ONLY
        through ``bind_table`` had neither, so it reached production unscoped and failed
        from ``l2_key`` in a request path rather than at wiring.
        """
        registry = CollectionRegistry()
        registry.configure(l3_pool=FakeAsyncpgPool())

        with pytest.raises(L2ScopeNotConfiguredError) as excinfo:
            registry.bind_table("widgets", l2_client=object())

        assert "widgets" in str(excinfo.value)
        assert "kv_key_scope" in str(excinfo.value)

    def test_bind_table_without_an_l2_client_needs_no_scope(self) -> None:
        """an L1/L3-only override touches no keyspace, so the guard must not fire."""
        registry = CollectionRegistry()
        registry.configure(l3_pool=FakeAsyncpgPool())

        registry.bind_table("widgets", l3_pool=FakeAsyncpgPool())

        assert registry.get_l3_pool("widgets") is not None

    def test_bind_table_l2_override_wins_over_default(self) -> None:
        default_client = object()
        table_client = object()
        registry = CollectionRegistry()
        registry.configure(l3_pool=FakeAsyncpgPool(), l2_client=default_client, kv_key_scope="hub")
        registry.bind_table("widgets", l2_client=table_client)

        collection = create_dynamic_collection(
            table_def=_widgets_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
        )

        assert collection._nats_client is table_client

    async def test_datastore_create_table_threads_registry_l2(self) -> None:
        """closes the §13/2 gap: DataStore collections get L2 via the registry."""
        l2_client = object()
        registry = CollectionRegistry()
        registry.configure(l3_pool=FakeAsyncpgPool(), l2_client=l2_client, kv_key_scope="hub")
        store = DataStore(uuid.uuid4(), registry, DefaultCoreConfig(collection_flush="ALWAYS"))

        collection = await store.create_table(_widgets_table())

        assert collection._nats_client is l2_client


def _embeddings_table() -> TableDef:
    return TableDef(
        name="embeddings",
        columns=[
            ColumnDef(name="id", column_type="text", primary_key=True),
            ColumnDef(name="embedding", column_type="vector", vector_dim=3),
        ],
    )


class TestVectorColumns:
    """vector columns work through the dynamic-collection path."""

    async def test_save_casts_vector_and_binds_bracketed_text(self) -> None:
        pool = FakeAsyncpgPool()
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        collection = create_dynamic_collection(
            table_def=_embeddings_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
        )

        await collection.save_to_store({"id": "e1", "embedding": [0.1, 0.2, 0.3]})

        sql, args = pool.executed[-1]
        assert "::text::public.vector" in sql
        bound = args[1]
        assert isinstance(bound, str)
        assert bound.startswith("[") and bound.endswith("]")

    async def test_fetch_coerces_vector_text_to_list(self) -> None:
        pool = FakeAsyncpgPool(rows={"e1": {"id": "e1", "embedding": "[0.1,0.2,0.3]"}})
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        collection = create_dynamic_collection(
            table_def=_embeddings_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
        )

        result = await collection.fetch_from_store("e1")

        assert result is not None
        assert result["embedding"] == [0.1, 0.2, 0.3]

    async def test_vector_round_trips_through_l1(self) -> None:
        pool = FakeAsyncpgPool(rows={"e1": {"id": "e1", "embedding": "[0.1,0.2,0.3]"}})
        l1 = _make_l1()
        registry = CollectionRegistry()
        registry.configure(l1_backend=l1, l3_pool=pool)
        collection = create_dynamic_collection(
            table_def=_embeddings_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
        )
        try:
            entity = await collection.get("e1")

            assert entity is not None
            cached = l1.select_by_id("embeddings", "e1", "id")
            assert cached is not None
            assert cached["embedding"] == [0.1, 0.2, 0.3]
        finally:
            l1.reset()

    def test_serialize_deserialize_vector_for_l2(self) -> None:
        registry = CollectionRegistry()
        registry.configure(l3_pool=FakeAsyncpgPool())
        collection = create_dynamic_collection(
            table_def=_embeddings_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
        )

        payload = collection.serialize({"id": "e1", "embedding": [0.1, 0.2, 0.3]})
        restored = collection.deserialize(payload)

        assert restored["embedding"] == [0.1, 0.2, 0.3]


def _events_table() -> TableDef:
    return TableDef(
        name="events",
        columns=[
            ColumnDef(name="id", column_type="text", primary_key=True),
            ColumnDef(name="date_created", column_type="timestamptz"),
        ],
    )


class TestTimestamptzColumns:
    """timestamptz columns work through the dynamic-collection path."""

    def test_timestamptz_deserializes_to_datetime(self) -> None:
        """an ISO string round-trips through L2 back to a datetime object.

        proves the ``timestamptz`` -> ``datetime`` field-type entry: the L2
        deserialize path reads it as a datetime, not a bare string.
        """
        from datetime import datetime

        registry = CollectionRegistry()
        registry.configure(l3_pool=FakeAsyncpgPool())
        collection = create_dynamic_collection(
            table_def=_events_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
        )

        aware = datetime.fromisoformat("2026-07-24T20:24:04+00:00")
        payload = collection.serialize({"id": "e1", "date_created": aware})
        restored = collection.deserialize(payload)

        assert isinstance(restored["date_created"], datetime)
        assert restored["date_created"] == aware


class _TaggingPool(FakeAsyncpgPool):
    """A pool whose ``execute`` returns a caller-chosen asyncpg status tag."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self._tag = tag

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append((sql, args))
        return self._tag


class TestDynamicSaveReportsTheRealRowcount:
    """A dynamic collection's ``save_to_store`` reports rows affected, not tag truthiness.

    ``BaseCollection.save_entity`` reads this count and raises when it is 0, which is the
    only thing standing between a write the store declined and a caller told it landed.
    Reading the status tag for truthiness answers 1 for every tag a backend can return,
    ``"INSERT 0 0"`` included -- which is what the NATS L3 proxy composes whenever the
    broker's reply carries no ``row_count``, so a zero-row write was reported as a
    persisted one. (A dynamic collection's own SQL is an unconditional ``ON CONFLICT
    DO UPDATE``, so on a direct pool it always affects one row; the proxy is the exposed
    case.)
    """

    @pytest.mark.parametrize(
        ("tag", "expected"),
        [
            ("INSERT 0 1", 1),
            ("INSERT 0 0", 0),
            ("UPDATE 3", 3),
            ("UPDATE 0", 0),
            ("", 0),
        ],
    )
    async def test_rowcount_comes_from_the_status_tag(self, tag: str, expected: int) -> None:
        registry = CollectionRegistry()
        registry.configure(l3_pool=_TaggingPool(tag))
        collection = create_dynamic_collection(
            table_def=_widgets_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
        )

        affected = await collection.save_to_store({"id": "w1", "name": "widget", "score": 1})

        assert affected == expected

    async def test_no_l3_pool_still_reports_nothing_written(self) -> None:
        """the wiring gap keeps its existing answer -- ``save_entity`` raises on it."""
        registry = CollectionRegistry()
        collection = create_dynamic_collection(
            table_def=_widgets_table(),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
        )

        affected = await collection.save_to_store({"id": "w1", "name": "widget", "score": 1})

        assert affected == 0
