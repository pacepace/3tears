"""a write the buffer kept in L1 reaches L3 with the types it was written with.

The L1-backed :class:`~threetears.core.collections.flush.WriteBuffer` stores each pending row as
JSON and reads it back on every drain -- the same process's next flush and, after a crash, the
next process's. Read back with a plain ``json.loads`` a UUID, an instant, a Decimal or bytes came
back as the string the encoder wrote, and the L3 write bound those strings: asyncpg refuses a
string for ``TIMESTAMPTZ`` and ``BYTEA``, so the write failed through its retry budget and was
dropped. The flush now rehydrates each row through the owning collection's schema.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
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
from threetears.core.entities.base import BaseEntity

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
