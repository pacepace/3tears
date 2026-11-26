"""Integration: a real NATS bucket's entry of another shape is ignored and refilled from Postgres.

Every collection L2 entry carries its shape tag (``threetears.core.collections.l2_shape``). What this pins, against real
Postgres and a real NATS KV bucket:

- an entry the collection wrote is served from L2 (a row changed behind it in L3 is not seen until L2 is told);
- an entry with no tag, as every release before 0.95.0 wrote it, is ignored: the read answers L3's row and the bucket
  then holds the collection's own tagged entry.

Uses the session-scoped ``db_container`` and ``nats_container`` fixtures; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_TABLE = "shape_sources"


class _SourceRow(BaseEntity):
    primary_key_field = "id"


class _Sources(SchemaBackedCollection[_SourceRow]):
    primary_key_column = "id"
    schema = TableSchema(
        name=_TABLE,
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("name", STRING_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        l1_eviction=TEST_L1_TABLE,
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

    async def put_raw(self, source_id: uuid.UUID, value: bytes) -> None:
        bucket = await self.nats.kv_bucket(name="collections")
        await bucket.put(key=self.sources.l2_key(source_id), value=value)

    async def get_raw(self, source_id: uuid.UUID) -> bytes | None:
        bucket = await self.nats.kv_bucket(name="collections")
        raw: bytes | None = await bucket.get(key=self.sources.l2_key(source_id))
        return raw


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
    schema = f"shape_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    namespace = f"shape{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    try:
        await pool.execute(
            f"""
            CREATE TABLE {_TABLE} (
                id UUID PRIMARY KEY,
                name TEXT NOT NULL,
                date_created TIMESTAMPTZ NOT NULL,
                date_updated TIMESTAMPTZ
            )
            """
        )
        async with await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="l2-shape-tag"
        ) as nats:
            l1 = SQLiteBackend(db_name=f"shape_{uuid.uuid4().hex[:8]}")
            l1.initialize(_Sources.schema.to_sqlalchemy_table(MetaData()).metadata)
            registry = CollectionRegistry()
            registry.configure(
                l1_backend=l1, l2_client=nats, l3_pool=SqlL3Backend(pool), kv_key_scope=f"p-{uuid.uuid4().hex[:6]}"
            )
            config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
            yield _Tiers(pool, nats, _Sources(registry, config))
    finally:
        await pool.close()


async def _saved(tiers: _Tiers) -> uuid.UUID:
    source_id = uuid.uuid4()
    await tiers.sources.save_entity(tiers.sources.create({"id": source_id, "name": "saved", "date_updated": None}))
    tiers.sources.evict_from_cache_sync(source_id)
    return source_id


async def test_an_entry_the_collection_wrote_is_served_from_l2(tiers: _Tiers) -> None:
    source_id = await _saved(tiers)
    await tiers.pool.execute(f"UPDATE {_TABLE} SET name = 'behind' WHERE id = $1", source_id)  # noqa: S608

    entity = await tiers.sources.get(source_id)

    assert entity is not None and entity.to_dict()["name"] == "saved", "an entry of the collection's shape was not used"


async def test_an_untagged_entry_is_ignored_and_refilled_from_l3(tiers: _Tiers) -> None:
    source_id = await _saved(tiers)
    older = {"id": str(source_id), "name": "an older release", "date_created": None, "date_updated": None}
    await tiers.put_raw(source_id, json.dumps(older).encode("utf-8"))

    entity = await tiers.sources.get(source_id)

    assert entity is not None and entity.to_dict()["name"] == "saved", "an untagged L2 entry was served"
    raw = await tiers.get_raw(source_id)
    assert raw is not None
    refilled = tiers.sources.decode_l2_entry(raw)
    assert refilled is not None and refilled["name"] == "saved", "the untagged entry was not replaced from L3"
