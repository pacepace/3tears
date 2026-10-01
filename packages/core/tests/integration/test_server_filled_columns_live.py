"""Integration: a row the database completes reads whole through the Collection, on every replica.

What a fake cannot prove: that the generated INSERT really drops an omitted ``server_default``
column and Postgres fills it in, that the fenced UPDATE really keeps the stored value of a
``NOT NULL`` default column sent as ``NULL``, and that the row read back -- not the row sent --
is what a real NATS KV bucket serves to a second replica.

The defect this pins was found in the hub's data-space ledger in the pre-PR live validation,
2026-09-30: a first request inserted five of the row's columns, every tier cached those five, and
``GET /admin/v1/data-spaces/{agent}`` answered 500 on the missing ``target_version``.

Uses the session-scoped ``db_container`` and ``nats_container`` fixtures; a checkout without
docker skips cleanly.
"""

from __future__ import annotations

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

_TABLE = "space_versions"


class _SpaceRow(BaseEntity):
    primary_key_field = "space_id"


class _SpaceVersions(SchemaBackedCollection[_SpaceRow]):
    """the data-space ledger's shape: versions and limits taken from server defaults."""

    primary_key_column = "space_id"
    schema = TableSchema(
        name=_TABLE,
        primary_key="space_id",
        columns=[
            Column("space_id", UUID_TYPE),
            Column("applied_table_count", INT_TYPE),
            Column("target_version", INT_TYPE, server_default="0"),
            Column("applied_version", INT_TYPE, server_default="0"),
            Column("max_tables", INT_TYPE, server_default="50"),
            Column("note", STRING_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_SpaceRow]:
        return _SpaceRow


@dataclass
class _Replicas:
    pool: asyncpg.Pool
    writer: _SpaceVersions
    reader: _SpaceVersions


def _replica(nats: NatsClient, pool: asyncpg.Pool, scope: str) -> _SpaceVersions:
    l1 = SQLiteBackend(db_name=f"server_filled_{uuid.uuid4().hex[:8]}")
    l1.initialize(_SpaceVersions.schema.to_sqlalchemy_table(MetaData()).metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=SqlL3Backend(pool), kv_key_scope=scope)
    return _SpaceVersions(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))


@pytest.fixture
async def replicas(db_container: str, nats_container: str) -> AsyncIterator[_Replicas]:
    """two replicas over one fresh schema and one real bucket, each with its own L1.

    :param db_container: the Postgres container's connection URL
    :ptype db_container: str
    :param nats_container: the NATS container's URL
    :ptype nats_container: str
    :return: the pool and both replicas
    :rtype: AsyncIterator[_Replicas]
    """
    schema = f"filled_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    namespace = f"filled{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    try:
        await pool.execute(
            f"""
            CREATE TABLE {_TABLE} (
                space_id UUID PRIMARY KEY,
                applied_table_count INTEGER NOT NULL,
                target_version INTEGER NOT NULL DEFAULT 0,
                applied_version INTEGER NOT NULL DEFAULT 0,
                max_tables INTEGER NOT NULL DEFAULT 50,
                note TEXT,
                date_created TIMESTAMPTZ NOT NULL,
                date_updated TIMESTAMPTZ
            )
            """
        )
        async with await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="server-filled"
        ) as nats:
            scope = f"filled-{uuid.uuid4().hex[:6]}"
            yield _Replicas(pool, _replica(nats, pool, scope), _replica(nats, pool, scope))
    finally:
        await pool.close()


async def test_a_first_write_relying_on_server_defaults_reads_whole_on_every_replica(replicas: _Replicas) -> None:
    space_id = uuid.uuid4()
    entity = replicas.writer.create({"space_id": space_id, "applied_table_count": 0})
    await replicas.writer.save_entity(entity)

    stored = await replicas.pool.fetchrow(f"SELECT * FROM {_TABLE} WHERE space_id = $1", space_id)
    assert stored is not None and (stored["target_version"], stored["max_tables"]) == (0, 50)

    assert (entity.target_version, entity.max_tables) == (0, 50), "the saving handle holds the row as sent"
    on_writer = await replicas.writer.get(space_id)
    assert on_writer is not None
    assert (on_writer.target_version, on_writer.applied_version, on_writer.max_tables) == (0, 0, 50)
    # the reader's L1 is empty, so this read is served from the real bucket the writer filled
    on_reader = await replicas.reader.get(space_id)
    assert on_reader is not None
    assert (on_reader.target_version, on_reader.applied_version, on_reader.max_tables) == (0, 0, 50), (
        "a second replica was served the row as sent"
    )
    assert on_reader.note is None


async def test_a_fenced_update_sending_null_for_a_defaulted_column_caches_the_stored_value(
    replicas: _Replicas,
) -> None:
    space_id = uuid.uuid4()
    # date_updated named, so the stored row carries a fence and the next save is the fenced UPDATE
    first = {"space_id": space_id, "applied_table_count": 0, "target_version": 3, "date_updated": None}
    await replicas.writer.save_entity(replicas.writer.create(first))
    loaded = await replicas.writer.get(space_id)
    assert loaded is not None and loaded.target_version == 3
    assert loaded.original_date_updated is not None, "the next save would not be the fenced UPDATE"
    # the fenced UPDATE skips a NOT NULL server-default column sent as NULL and keeps the stored 3
    loaded.target_version = None
    loaded.applied_table_count = 7
    await replicas.writer.save_entity(loaded)

    stored = await replicas.pool.fetchrow(f"SELECT * FROM {_TABLE} WHERE space_id = $1", space_id)
    assert stored is not None and (stored["target_version"], stored["applied_table_count"]) == (3, 7)
    on_reader = await replicas.reader.get(space_id)
    assert on_reader is not None
    assert (on_reader.target_version, on_reader.applied_table_count) == (3, 7), "a replica was served the NULL sent"
    on_writer = await replicas.writer.get(space_id)
    assert on_writer is not None and on_writer.target_version == 3
