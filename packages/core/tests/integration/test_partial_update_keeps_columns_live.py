"""Integration: an update through a SchemaBackedCollection rewrites only the columns its row names.

The defect (v0.78.0): the generated ``DO UPDATE SET`` and the fenced ``UPDATE`` wrote ``NULL`` to
every nullable column without a server default that the saved row left out, so a partial save
wiped stored values it never meant to touch. The hub hit it re-saving ``platform.datasources``
rows and carried the stored values itself (``HubCapabilitySourceCollection.save_entity``).

What this pins, against real Postgres and a real NATS KV bucket:

- a partial save, through the upsert and through the fenced UPDATE, keeps the column it leaves
  out in L3, and L1 and L2 hold the stored value, read back, never a guessed one;
- a save naming the column as ``None`` still clears it, on every tier;
- an insert still takes ``NULL`` for a nullable column it leaves out and its server default for
  a defaulted one.

Uses the session-scoped ``db_container`` and ``nats_container`` fixtures; a checkout without
docker skips cleanly.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_TABLE = "geo_sources"


class _SourceRow(BaseEntity):
    primary_key_field = "id"


class _Sources(SchemaBackedCollection[_SourceRow]):
    """the shape of the hub's datasources: nullable columns a partial save leaves out."""

    primary_key_column = "id"
    schema = TableSchema(
        name=_TABLE,
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("name", STRING_TYPE),
            Column("geo", STRING_TYPE, nullable=True),
            Column("rank", INT_TYPE, server_default="5"),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_SourceRow]:
        return _SourceRow


@dataclass
class _Tiers:
    pool: asyncpg.Pool
    nats: NatsClient
    sources: _Sources

    async def l3(self, source_id: uuid.UUID) -> asyncpg.Record:
        row = await self.pool.fetchrow(f"SELECT * FROM {_TABLE} WHERE id = $1", source_id)  # noqa: S608
        assert row is not None
        return row

    def l1(self, source_id: uuid.UUID) -> dict[str, Any]:
        row = self.sources.get_row_sync(source_id)
        assert row is not None, "L1 holds no row"
        return row

    async def l2(self, source_id: uuid.UUID) -> dict[str, Any]:
        bucket = await self.nats.kv_bucket(name="collections")
        raw = await bucket.get(key=self.sources.l2_key(source_id))
        assert raw is not None, "L2 holds no row"
        return dict(json.loads(raw))


@pytest.fixture
async def tiers(db_container: str, nats_container: str) -> AsyncIterator[_Tiers]:
    """one collection over a fresh schema, its own L1 and a real bucket.

    :param db_container: the Postgres container's connection URL
    :ptype db_container: str
    :param nats_container: the NATS container's URL
    :ptype nats_container: str
    :return: the pool, the NATS client and the collection
    :rtype: AsyncIterator[_Tiers]
    """
    schema = f"partial_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    namespace = f"partial{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    try:
        await pool.execute(
            f"""
            CREATE TABLE {_TABLE} (
                id UUID PRIMARY KEY,
                name TEXT NOT NULL,
                geo TEXT,
                rank INTEGER NOT NULL DEFAULT 5,
                date_created TIMESTAMPTZ NOT NULL,
                date_updated TIMESTAMPTZ
            )
            """
        )
        async with await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="partial-update"
        ) as nats:
            l1 = SQLiteBackend(db_name=f"partial_{uuid.uuid4().hex[:8]}")
            l1.initialize(_Sources.schema.to_sqlalchemy_table(MetaData()).metadata)
            registry = CollectionRegistry()
            registry.configure(
                l1_backend=l1, l2_client=nats, l3_pool=SqlL3Backend(pool), kv_key_scope=f"p-{uuid.uuid4().hex[:6]}"
            )
            config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
            yield _Tiers(pool, nats, _Sources(registry, config))
    finally:
        await pool.close()


async def _seed(tiers: _Tiers) -> uuid.UUID:
    """store a source whose nullable ``geo`` is set, and drop it from the caches.

    :param tiers: the tiers
    :ptype tiers: _Tiers
    :return: the source's id
    :rtype: uuid.UUID
    """
    source_id = uuid.uuid4()
    first = {"id": source_id, "name": "first", "geo": "county", "rank": 2, "date_updated": None}
    await tiers.sources.save_entity(tiers.sources.create(first))
    assert (await tiers.l3(source_id))["geo"] == "county"
    return source_id


async def test_a_partial_upsert_keeps_the_column_it_leaves_out_on_every_tier(tiers: _Tiers) -> None:
    source_id = await _seed(tiers)
    await tiers.sources.invalidate_cache(source_id)
    # a fresh handle naming only some columns: the generated INSERT ... ON CONFLICT DO UPDATE
    await tiers.sources.save_entity(tiers.sources.create({"id": source_id, "name": "second"}))

    stored = await tiers.l3(source_id)
    assert (stored["name"], stored["geo"], stored["rank"]) == ("second", "county", 2), "the update wiped geo"
    assert tiers.l1(source_id)["geo"] == "county", "L1 holds a guessed geo"
    assert (await tiers.l2(source_id))["geo"] == "county", "L2 holds a guessed geo"


async def test_a_partial_fenced_update_keeps_the_column_it_leaves_out_on_every_tier(tiers: _Tiers) -> None:
    source_id = await _seed(tiers)
    loaded = await tiers.sources.get(source_id)
    assert loaded is not None and loaded.original_date_updated is not None, "the save would not be fenced"
    # a loaded handle holding only some columns, as one read through a narrower query does
    loaded.hold_row({"id": source_id, "name": "first", "date_updated": loaded.date_updated})
    loaded.name = "fenced"
    await tiers.sources.save_entity(loaded)

    stored = await tiers.l3(source_id)
    assert (stored["name"], stored["geo"]) == ("fenced", "county"), "the fenced update wiped geo"
    assert tiers.l1(source_id)["geo"] == "county", "L1 holds a guessed geo"
    assert (await tiers.l2(source_id))["geo"] == "county", "L2 holds a guessed geo"
    assert loaded.geo == "county", "the saving handle holds a guessed geo"


async def test_a_save_naming_the_column_as_none_clears_it_on_every_tier(tiers: _Tiers) -> None:
    source_id = await _seed(tiers)
    await tiers.sources.invalidate_cache(source_id)
    await tiers.sources.save_entity(tiers.sources.create({"id": source_id, "name": "cleared", "geo": None}))

    stored = await tiers.l3(source_id)
    assert (stored["name"], stored["geo"]) == ("cleared", None), "an explicit None did not clear geo"
    assert tiers.l1(source_id)["geo"] is None
    assert (await tiers.l2(source_id))["geo"] is None


async def test_a_fenced_save_naming_the_column_as_none_clears_it(tiers: _Tiers) -> None:
    source_id = await _seed(tiers)
    loaded = await tiers.sources.get(source_id)
    assert loaded is not None
    loaded.geo = None
    await tiers.sources.save_entity(loaded)

    assert (await tiers.l3(source_id))["geo"] is None, "an explicit None did not clear geo"
    assert tiers.l1(source_id)["geo"] is None
    assert (await tiers.l2(source_id))["geo"] is None


async def test_an_insert_takes_null_and_the_server_default_for_the_columns_it_leaves_out(tiers: _Tiers) -> None:
    source_id = uuid.uuid4()
    await tiers.sources.save_entity(tiers.sources.create({"id": source_id, "name": "new"}))

    stored = await tiers.l3(source_id)
    assert (stored["name"], stored["geo"], stored["rank"]) == ("new", None, 5)
    assert stored["date_created"] is not None
    cached = tiers.l1(source_id)
    assert (cached["geo"], cached["rank"]) == (None, 5)
    l2 = await tiers.l2(source_id)
    assert (l2["geo"], l2["rank"]) == (None, 5)
