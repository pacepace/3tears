"""a collection that writes its own SQL over a declared ``TableSchema`` caches exactly what L3 holds.

What :class:`DeclaredTableCollection` gives a host, each pinned here through a real SQLite L1:

- its L1 table is generated from the schema and created on whatever L1 the registry hands it, so a
  host declares nothing;
- its key and instants are read off the schema, and a key that disagrees is refused;
- a ``NUMERIC`` column reads back from L1 as the ``Decimal`` L3 gives, at the column's scale;
- a write that leaves a column to the database caches the row read back from L3, not the row sent;
- a natural-key upsert that lands on a row stored under another key evicts that row and caches
  nothing under the key it sent.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, ClassVar

import pytest
from sqlalchemy import MetaData

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import DeclaredTableCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity

_SCHEMA = TableSchema(
    name="price_rows",
    primary_key="code",
    columns=[
        Column("code", STRING_TYPE),
        Column("sku", STRING_TYPE),
        Column("price", NUMERIC_TYPE, precision=12, scale=4),
        Column("tier", INT_TYPE, server_default="3"),
        Column("date_created", DATETIMETZ_TYPE),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)


class _PriceRow(BaseEntity):
    primary_key_field = "code"


class _PriceTable:
    """``price_rows`` in memory: an upsert on ``sku`` that fills ``tier``'s default and returns the stored key.

    An asyncpg-pool-shaped double: the raw pool has no production protocol (``l3_pool`` is typed
    ``Any`` by design), so there is nothing to declare parity against.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        row = self.rows.get(args[0])
        return None if row is None else dict(row)

    async def fetchval(self, sql: str, code: str, sku: str, price: Any, now: datetime) -> str:
        held = next((r for r in self.rows.values() if r["sku"] == sku), None)
        if held is None:
            self.rows[code] = {
                "code": code,
                "sku": sku,
                "price": Decimal(price).quantize(Decimal("0.0001")),
                "tier": 3,
                "date_created": now,
                "date_updated": now,
            }
            return code
        held.update({"price": Decimal(price).quantize(Decimal("0.0001")), "date_updated": now})
        return str(held["code"])

    async def execute(self, sql: str, *args: Any) -> str:
        return "DELETE 1"

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return []


class _Prices(DeclaredTableCollection[_PriceRow]):
    """a collection over ``price_rows`` writing its own natural-key upsert."""

    schema: ClassVar[TableSchema] = _SCHEMA

    @property
    def entity_class(self) -> type[_PriceRow]:
        return _PriceRow

    def serialize(self, data: dict[str, Any]) -> bytes:
        raise NotImplementedError("no L2 in these tests")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        raise NotImplementedError("no L2 in these tests")

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        assert self.l3_pool is not None
        row: dict[str, Any] | None = await self.l3_pool.fetchrow("SELECT * FROM price_rows WHERE code = $1", entity_id)
        return row

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        assert self.l3_pool is not None
        stored = await self.l3_pool.fetchval(
            "INSERT ... ON CONFLICT (sku) DO UPDATE ... RETURNING code",
            data["code"],
            data["sku"],
            data["price"],
            data["date_updated"],
        )
        await self.settle_stored_key(data["code"], stored, conn=conn)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        return None


@pytest.fixture()
def l1() -> Iterator[SQLiteBackend]:
    """an L1 that has been told about no table."""
    backend = SQLiteBackend(db_name=f"declared_table_{uuid.uuid4().hex[:8]}")
    backend.initialize(MetaData())
    yield backend
    backend.reset()


def _prices(l1: SQLiteBackend) -> tuple[_Prices, _PriceTable]:
    table = _PriceTable()
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=table)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return _Prices(registry=registry, config=config, nats_client=None), table


def _row(code: str, sku: str, price: str) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {"code": code, "sku": sku, "price": Decimal(price), "date_created": now, "date_updated": now}


class TestTheSchemaDeclaresTheTable:
    """the key, the instants, the name and the L1 table all come from the schema."""

    def test_key_instants_and_name(self, l1: SQLiteBackend) -> None:
        prices, _ = _prices(l1)
        assert _Prices.primary_key_column == "code"
        assert _Prices.datetime_columns >= {"date_created", "date_updated"}
        assert prices.table_name == "price_rows"

    def test_the_l1_table_is_created_on_the_l1_it_is_given(self, l1: SQLiteBackend) -> None:
        assert not l1.has_table("price_rows")
        _prices(l1)
        assert set(l1.column_types("price_rows")) == {c.name for c in _SCHEMA.columns}

    def test_a_key_the_schema_does_not_declare_is_refused(self) -> None:
        with pytest.raises(TypeError, match="primary key is 'code'"):

            class _Wrong(_Prices):
                schema: ClassVar[TableSchema] = _SCHEMA
                primary_key_column = "sku"


class TestTheRowCachedIsTheRowStored:
    """every tier caches what L3 holds after the write."""

    async def test_a_server_default_is_read_back_and_a_numeric_stays_decimal(self, l1: SQLiteBackend) -> None:
        prices, _ = _prices(l1)
        await prices.save_entity(prices.create(_row("a", "sku-1", "1.5")))
        cached = prices.get_row_sync("a")
        assert cached is not None
        assert cached["tier"] == 3, "the column the database filled was not read back"
        assert isinstance(cached["price"], Decimal) and str(cached["price"]) == "1.5000"
        entity = await prices.get("a")
        assert entity is not None and entity.price * Decimal(2) == Decimal("3.0000")

    async def test_an_upsert_landing_under_another_key_evicts_it_and_caches_nothing(self, l1: SQLiteBackend) -> None:
        prices, table = _prices(l1)
        await prices.save_entity(prices.create(_row("a", "sku-1", "1.5")))
        assert prices.get_row_sync("a") is not None

        await prices.save_entity(prices.create(_row("b", "sku-1", "2.25")))

        assert set(table.rows) == {"a"}
        assert prices.get_row_sync("a") is None
        assert prices.get_row_sync("b") is None
        reread = await prices.get("a")
        assert reread is not None and reread.price == Decimal("2.2500")
