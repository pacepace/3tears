"""a declared collection with no field-type map and no column list writes and reads every type through L3, L1 and L2.

Its upsert is the one :meth:`DeclaredTableCollection.write_row` builds from the schema, and its L2
codec rehydrates by the types the schema declares. Written through a real Postgres table and a real
SQLite L1, a row of a ``Decimal``, an aware instant, UUIDs, JSON, a bool, bytes and integers comes
back the same from L3, from L1 and through the L2 codec, on the insert and on an update; and the
migrated table matches the schema (:func:`live_tier_parity_findings`).

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, ClassVar

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import DeclaredTableCollection
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.base import NO_L2
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    BOOL_TYPE,
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.tier_parity import live_tier_parity_findings

pytestmark = pytest.mark.integration

_DDL = """
CREATE TABLE every_type_rows (
    id uuid PRIMARY KEY,
    code text NOT NULL,
    price numeric(12, 4) NOT NULL,
    owner_id uuid,
    doc jsonb,
    active boolean NOT NULL DEFAULT false,
    blob bytea,
    small integer,
    big bigint,
    seen_at timestamptz,
    date_created timestamptz NOT NULL,
    date_updated timestamptz NOT NULL
)
"""


class _Row(BaseEntity):
    primary_key_field = "id"


class _Every(DeclaredTableCollection[_Row]):
    """``every_type_rows``: its own read, the generated upsert, no map and no list."""

    schema: ClassVar[TableSchema] = TableSchema(
        name="every_type_rows",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("code", STRING_TYPE),
            Column("price", NUMERIC_TYPE, precision=12, scale=4),
            Column("owner_id", UUID_TYPE, nullable=True),
            Column("doc", JSONB_TYPE, nullable=True),
            Column("active", BOOL_TYPE, server_default="false"),
            Column("blob", BYTES_TYPE, nullable=True),
            Column("small", INT_TYPE, nullable=True),
            Column("big", BIGINT_TYPE, nullable=True),
            Column("seen_at", DATETIMETZ_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
    )

    @property
    def entity_class(self) -> type[_Row]:
        return _Row

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        assert self.l3_pool is not None
        row = await self.l3_pool.fetchrow("SELECT * FROM every_type_rows WHERE id = $1", entity_id)
        return None if row is None else dict(row)

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        return await self.write_row(data, conn=conn)

    async def delete_from_store(self, entity_id: Any) -> None:
        assert self.l3_pool is not None
        await self.l3_pool.execute("DELETE FROM every_type_rows WHERE id = $1", entity_id)


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """a pool set up as a host's is, on a fresh schema holding ``every_type_rows``.

    :param db_container: the Postgres container's connection URL
    :ptype db_container: str
    :return: the pool
    :rtype: AsyncIterator[asyncpg.Pool]
    """
    own = f"dt_{uuid.uuid4().hex[:8]}"
    setup = await asyncpg.connect(db_container)
    await setup.execute(f"CREATE SCHEMA {own}")
    await setup.execute(f"SET search_path TO {own}")
    await setup.execute(_DDL)
    created = await asyncpg.create_pool(
        db_container, min_size=1, max_size=4, server_settings={"search_path": own}, init=init_connection
    )
    assert created is not None
    try:
        yield created
    finally:
        await created.close()
        await setup.execute(f"DROP SCHEMA {own} CASCADE")
        await setup.close()


def _collection(pool: asyncpg.Pool) -> tuple[_Every, SQLiteBackend]:
    l1 = SQLiteBackend(db_name=f"declared_live_{uuid.uuid4().hex[:8]}")
    l1.initialize(MetaData())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return _Every(registry=registry, config=config, nats_client=NO_L2), l1


def _row() -> dict[str, Any]:
    now = datetime(2026, 10, 24, 12, 34, 56, 789012, tzinfo=UTC)
    return {
        "id": uuid.uuid4(),
        "code": "a",
        "price": Decimal("-12345678.1234"),
        "owner_id": uuid.uuid4(),
        "doc": {"nested": {"n": 1, "l": [1, "two", None]}, "b": True},
        "active": True,
        "blob": b"\x00\xffbytes",
        "small": -2147483647,
        "big": 9007199254740993,
        "seen_at": now,
        "date_created": now,
        "date_updated": now,
    }


_TYPED = ("price", "owner_id", "doc", "active", "blob", "small", "big", "seen_at")


def _assert_every_tier_holds(coll: _Every, stored: dict[str, Any] | None, sent: dict[str, Any]) -> None:
    assert stored is not None
    cached = coll.get_row_sync(sent["id"])
    decoded = coll.decode_row(coll.serialize(stored))
    for name in _TYPED:
        for tier, row in (("L3", stored), ("L1", cached), ("L2", decoded)):
            assert row is not None, tier
            assert row[name] == sent[name], f"{tier} {name}: {row[name]!r} != {sent[name]!r}"
            assert isinstance(row[name], type(sent[name])), f"{tier} {name}: {type(row[name]).__name__}"
    assert stored["seen_at"].tzinfo is not None


async def test_every_type_is_written_and_read_back_from_l3_l1_and_l2_and_the_table_is_the_schema(
    pool: asyncpg.Pool,
) -> None:
    coll, l1 = _collection(pool)
    try:
        sent = _row()
        await coll.save_entity(coll.create(dict(sent)))
        inserted = await coll.fetch_from_store(sent["id"])
        _assert_every_tier_holds(coll, inserted, sent)
        assert inserted is not None

        changed = {**sent, "price": Decimal("0.0001"), "doc": [1, {"two": 2}], "active": False, "blob": b"\x01"}
        entity = await coll.get(sent["id"])
        assert entity is not None
        for name in ("price", "doc", "active", "blob"):
            setattr(entity, name, changed[name])
        await coll.save_entity(entity)
        stored = await coll.fetch_from_store(sent["id"])
        _assert_every_tier_holds(coll, stored, changed)
        # the immutable column is kept by the update the generated upsert makes
        assert stored is not None and stored["date_created"] == inserted["date_created"]
        async with pool.acquire() as conn:
            assert await live_tier_parity_findings(conn, [_Every]) == []
    finally:
        l1.reset()
