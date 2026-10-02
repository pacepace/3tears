"""a write the buffer kept in L1 reaches L3 with the types it was written with.

The L1-backed :class:`~threetears.core.collections.flush.WriteBuffer` stores each pending row as
JSON and reads it back on every drain -- the same process's next flush and, after a crash, the
next process's. Read back with a plain ``json.loads`` a UUID, an instant, a Decimal or bytes came
back as the string the encoder wrote, and the L3 write bound those strings: asyncpg refuses a
string for ``TIMESTAMPTZ`` and ``BYTEA``, so the write failed through its retry budget and was
dropped. The flush now rehydrates each row through the owning collection's own decode
(:meth:`~threetears.core.collections.base.BaseCollection.decode_row`), whatever kind of collection
owns it: a schema-backed one types every declared column, and every collection -- a durable-store
one, a dynamic one, a hand-written subclass -- gets its declared instants back aware.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, ClassVar

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.durable_store import DurableStoreCollection
from threetears.core.collections.flush import WriteBuffer, flush_pending
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.collection_factory import create_dynamic_collection
from threetears.core.data.schema import ColumnDef, TableDef
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import CorruptCacheEntry

_ROW_ID = uuid.UUID("01926f3a-0000-7000-8000-000000000001")
_WHEN = datetime(2026, 10, 1, 12, 30, tzinfo=UTC)


class _Entity(BaseEntity):
    primary_key_field = "id"


class _TypedRows(SchemaBackedCollection[_Entity]):
    primary_key_column: str = "id"
    schema = TableSchema(
        name="typed_rows",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("date_created", DATETIMETZ_TYPE),
            Column("cost", NUMERIC_TYPE, precision=10, scale=4),
            Column("blob", BYTES_TYPE, nullable=True),
            Column("doc", JSONB_TYPE, nullable=True),
        ],
    )

    @property
    def table_name(self) -> str:
        """return table name."""
        return "typed_rows"

    @property
    def entity_class(self) -> type[_Entity]:
        """return entity class."""
        return _Entity


class _RecordingPool:
    """an asyncpg-shaped pool recording each ``execute``; it has no ``transaction``, so the flush
    takes the per-entity path."""

    def __init__(self) -> None:
        self.args: list[tuple[Any, ...]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.args.append(args)
        return "INSERT 0 1"


def _wired() -> tuple[CollectionRegistry, _RecordingPool]:
    pool = _RecordingPool()
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    collection = _TypedRows(
        registry=registry,
        config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        nats_client=None,
    )
    registry.register(collection)
    return registry, pool


def _bound(pool: _RecordingPool) -> list[Any]:
    (args,) = pool.args
    return list(args)


@pytest.mark.asyncio
async def test_a_write_recovered_after_a_restart_binds_its_typed_values() -> None:
    db_name = f"wb_typed_{uuid.uuid4().hex}"
    row = {"id": _ROW_ID, "date_created": _WHEN, "cost": Decimal("2.50"), "blob": b"\x00\x01", "doc": {"k": 1}}
    await WriteBuffer(l1_backend=SQLiteBackend(db_name=db_name)).add("typed_rows", _ROW_ID, row)
    registry, pool = _wired()

    # the process that buffered it is gone; a new one drains the same durable store
    flushed = await flush_pending(WriteBuffer(l1_backend=SQLiteBackend(db_name=db_name)), registry)

    assert flushed == 1
    bound = _bound(pool)
    assert _ROW_ID in bound
    assert _WHEN in bound
    assert Decimal("2.50") in bound
    assert b"\x00\x01" in bound
    assert not any(isinstance(value, str) and value in {str(_ROW_ID), "2.50"} for value in bound)


@pytest.mark.asyncio
async def test_the_same_process_flush_binds_typed_values_too() -> None:
    """an L1-backed buffer reads its own rows back from L1 on every drain, not only after a crash."""
    buffer = WriteBuffer(l1_backend=SQLiteBackend(db_name=f"wb_typed_{uuid.uuid4().hex}"))
    await buffer.add("typed_rows", _ROW_ID, {"id": _ROW_ID, "date_created": _WHEN, "cost": Decimal("1")})
    registry, pool = _wired()

    await flush_pending(buffer, registry)

    bound = _bound(pool)
    assert _ROW_ID in bound
    assert _WHEN in bound


@pytest.mark.asyncio
async def test_a_row_buffered_in_a_legacy_spelling_still_flushes_as_the_same_instant() -> None:
    """rows buffered before the one stored form -- ``str(dt)``, no fraction, naive -- read as UTC."""
    db_name = f"wb_typed_{uuid.uuid4().hex}"
    l1 = SQLiteBackend(db_name=db_name)
    WriteBuffer(l1_backend=l1)  # creates the buffer table
    for index, legacy in enumerate(("2026-10-01 12:30:00+00:00", "2026-10-01T12:30:00", "2026-10-01T12:30:00Z")):
        row_id = uuid.UUID(int=index + 1)
        l1.upsert(
            "write_buffer",
            {
                "key": f"typed_rows:{row_id}",
                "table_name": "typed_rows",
                "entity_id": str(row_id),
                "data": json.dumps({"id": str(row_id), "date_created": legacy, "cost": "3"}),
                "retries": 0,
                "date_updated": _WHEN.isoformat(),
            },
            primary_key="key",
        )
    registry, pool = _wired()

    flushed = await flush_pending(WriteBuffer(l1_backend=SQLiteBackend(db_name=db_name)), registry)

    assert flushed == 3
    for args in pool.args:
        assert _WHEN in args


@pytest.mark.asyncio
async def test_a_table_with_no_registered_collection_still_drains() -> None:
    """with no schema to rehydrate by, the row is handed on as parsed -- and skipped as before."""
    buffer = WriteBuffer(l1_backend=SQLiteBackend(db_name=f"wb_typed_{uuid.uuid4().hex}"))
    await buffer.add("unregistered", "u1", {"id": "u1"})

    assert await flush_pending(buffer, CollectionRegistry()) == 0


class _RecordingDurableStore:
    """a non-SQL durable store recording every row it is asked to upsert; no ``transaction``."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        return None

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
        self.rows.append(dict(row))
        return 1

    async def delete(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> None:
        return None

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        return []


class _Scenes(DurableStoreCollection[_Entity]):
    primary_key_column = "id"
    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created"})

    @property
    def table_name(self) -> str:
        """return table name."""
        return "scenes"

    @property
    def entity_class(self) -> type[_Entity]:
        """return entity class."""
        return _Entity


class _HandWritten(BaseCollection[_Entity]):
    """a collection written by hand against the base: its own codec, its own store."""

    primary_key_column = "id"
    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created"})

    def __init__(self, registry: CollectionRegistry) -> None:
        super().__init__(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), None)
        self.stored: list[dict[str, Any]] = []

    @property
    def table_name(self) -> str:
        """return table name."""
        return "hand_written"

    @property
    def entity_class(self) -> type[_Entity]:
        """return entity class."""
        return _Entity

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        return None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        self.stored.append(dict(data))
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        decoded: dict[str, Any] = json.loads(data)
        return decoded


async def _recovered(table_name: str, row: dict[str, Any], registry: CollectionRegistry) -> int:
    """buffer ``row`` in one process's durable buffer, then flush it from a new process's.

    :param table_name: the row's table
    :ptype table_name: str
    :param row: the row as written
    :ptype row: dict[str, Any]
    :param registry: the new process's registry
    :ptype registry: CollectionRegistry
    :return: rows flushed
    :rtype: int
    """
    db_name = f"wb_typed_{uuid.uuid4().hex}"
    await WriteBuffer(l1_backend=SQLiteBackend(db_name=db_name)).add(table_name, row["id"], row)
    return await flush_pending(WriteBuffer(l1_backend=SQLiteBackend(db_name=db_name)), registry)


@pytest.mark.asyncio
async def test_a_durable_store_collection_recovers_its_declared_instants_aware() -> None:
    """a structured durable store receives the instant it was written with, not the text the buffer kept."""
    registry = CollectionRegistry()
    store = _RecordingDurableStore()
    registry.register(
        _Scenes(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), store, None)
    )

    assert await _recovered("scenes", {"id": "s1", "date_created": _WHEN}, registry) == 1

    (stored,) = store.rows
    assert stored["date_created"] == _WHEN
    assert stored["date_created"].tzinfo is not None


@pytest.mark.asyncio
async def test_a_dynamic_collection_recovers_a_legacy_naive_instant_as_utc() -> None:
    """a ``timestamptz`` row buffered naive before the one stored form flushes as the UTC instant.

    the dynamic codec parses it to a naive datetime, which the L3 write refuses (the driver would
    read it as the host's local time); the decode stamps it UTC, as every L2 read of a legacy
    naive value does, so it lands instead of being dropped after its retry budget.
    """
    pool = _RecordingPool()
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    registry.register(
        create_dynamic_collection(
            table_def=TableDef(
                name="events",
                columns=[
                    ColumnDef(name="id", column_type="text", primary_key=True),
                    ColumnDef(name="date_happened", column_type="timestamptz"),
                ],
            ),
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
            nats_client=None,
        )
    )
    db_name = f"wb_typed_{uuid.uuid4().hex}"
    l1 = SQLiteBackend(db_name=db_name)
    WriteBuffer(l1_backend=l1)  # creates the buffer table
    l1.upsert(
        "write_buffer",
        {
            "key": "events:e1",
            "table_name": "events",
            "entity_id": "e1",
            "data": json.dumps({"id": "e1", "date_happened": "2026-10-01T12:30:00"}),
            "retries": 0,
            "date_updated": _WHEN.isoformat(),
        },
        primary_key="key",
    )

    assert await flush_pending(WriteBuffer(l1_backend=SQLiteBackend(db_name=db_name)), registry) == 1

    assert _bound(pool) == ["e1", _WHEN]


@pytest.mark.asyncio
async def test_a_hand_written_collection_recovers_its_declared_instants_aware() -> None:
    """a subclass with its own plain-JSON codec still gets its declared instants back typed."""
    registry = CollectionRegistry()
    collection = _HandWritten(registry)
    registry.register(collection)

    assert await _recovered("hand_written", {"id": "h1", "date_created": _WHEN, "note": "kept"}, registry) == 1

    (stored,) = collection.stored
    assert stored == {"id": "h1", "date_created": _WHEN, "note": "kept"}


def test_decode_row_is_the_collections_codec_plus_its_instants() -> None:
    """``decode_row`` is ``deserialize`` with every declared instant rehydrated aware -- legacy naive read as UTC."""
    collection = _HandWritten(CollectionRegistry())

    decoded = collection.decode_row(
        json.dumps({"id": "h1", "date_created": "2026-10-01T12:30:00", "other": "2026-10-01T12:30:00"}).encode()
    )

    assert decoded == {"id": "h1", "date_created": _WHEN, "other": "2026-10-01T12:30:00"}


def test_decode_row_refuses_an_instant_that_will_not_parse() -> None:
    """an unreadable declared instant is a corrupt entry, named by table and column, never a string passed on."""
    collection = _HandWritten(CollectionRegistry())

    with pytest.raises(CorruptCacheEntry):
        collection.decode_row(json.dumps({"id": "h1", "date_created": "not a time"}).encode())


@pytest.mark.asyncio
async def test_a_buffered_row_whose_instant_will_not_parse_still_drains() -> None:
    """one unreadable row is handed on as parsed JSON and logged; it never stops the drain."""
    registry = CollectionRegistry()
    collection = _HandWritten(registry)
    registry.register(collection)
    db_name = f"wb_typed_{uuid.uuid4().hex}"
    l1 = SQLiteBackend(db_name=db_name)
    WriteBuffer(l1_backend=l1)  # creates the buffer table
    l1.upsert(
        "write_buffer",
        {
            "key": "hand_written:h1",
            "table_name": "hand_written",
            "entity_id": "h1",
            "data": json.dumps({"id": "h1", "date_created": "not a time"}),
            "retries": 0,
            "date_updated": _WHEN.isoformat(),
        },
        primary_key="key",
    )

    pending = await WriteBuffer(l1_backend=SQLiteBackend(db_name=db_name)).drain(
        decode=lambda table, text: registry.get_collection(table).decode_row(text.encode("utf-8"))
    )

    assert [write.data for write in pending] == [{"id": "h1", "date_created": "not a time"}]
