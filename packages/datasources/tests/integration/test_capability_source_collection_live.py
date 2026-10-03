"""live proof that a capability source's ``require_documented_tables`` persists through the Collection.

The defect this pins was found 2026-10-03: the hub's ``PATCH /admin/v1/datasources/{id}``
sets ``require_documented_tables`` on the entity and saves it through
:class:`CapabilitySourceCollection`, and the flag never persisted. Hub migration v076 added the
column, but the Collection's ``TableSchema`` did not declare it, so the schema-driven upsert left
it out of every statement while the hub's response echoed the value sent.

What a fake cannot prove: that the generated INSERT and fenced UPDATE really write the column to
Postgres, and that a replica with an empty L1 reads it back from L3 as written.

The table below is ``platform.datasources`` as the hub's migrations leave it (v001, v037, v042,
v046, v053, v076). The hub-owned columns this Collection deliberately does not declare (``spec``,
``face_*``, ``geo``) are part of it so the tests also show a Collection save leaves them alone.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

import asyncpg
import pytest
from sqlalchemy import MetaData
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.datasources.collections import CapabilitySourceCollection
from threetears.datasources.entities import CapabilitySourceEntity

pytestmark = pytest.mark.integration

_DATASOURCES_DDL = """
CREATE TABLE datasources (
    id uuid NOT NULL PRIMARY KEY,
    name character varying(255) NOT NULL,
    datasource_type character varying(50),
    connection_config text,
    allowed_schemas jsonb DEFAULT '[]'::jsonb NOT NULL,
    status character varying(20) NOT NULL,
    date_created timestamp with time zone NOT NULL,
    date_updated timestamp with time zone NOT NULL,
    access_mode character varying(20) DEFAULT 'read'::character varying NOT NULL,
    customer_id uuid,
    owner_agent_id uuid,
    schema_name character varying(255),
    visibility text DEFAULT 'private'::text NOT NULL,
    origin_datasource_id uuid,
    kind character varying(50) NOT NULL DEFAULT 'datasource',
    ingress_agent_id uuid,
    spec text,
    face_api boolean NOT NULL DEFAULT FALSE,
    face_mcp boolean NOT NULL DEFAULT FALSE,
    face_platform_tool boolean NOT NULL DEFAULT TRUE,
    knowledge_required boolean NOT NULL DEFAULT FALSE,
    geo jsonb,
    require_documented_tables boolean DEFAULT FALSE NOT NULL
)
"""


@dataclass
class _Sources:
    """the pool and a factory for replicas that share it."""

    pool: asyncpg.Pool

    def replica(self) -> CapabilitySourceCollection:
        """a Collection with its own empty L1, over the shared L3 and no L2.

        a fresh replica's read of a row it never wrote is served from L3, which is what makes it
        the read that proves what was persisted.

        :return: the Collection
        :rtype: CapabilitySourceCollection
        """
        l1 = SQLiteBackend(db_name=f"capsrc_{uuid.uuid4().hex[:8]}")
        l1.initialize(CapabilitySourceCollection.schema.to_sqlalchemy_table(MetaData()).metadata)
        registry = CollectionRegistry()
        registry.configure(l1_backend=l1, l3_pool=SqlL3Backend(self.pool))
        return CapabilitySourceCollection(
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        )


@pytest.fixture
async def sources(db_container: str) -> AsyncIterator[_Sources]:
    """a fresh schema holding the hub's ``datasources`` table.

    :param db_container: the Postgres container's connection URL
    :ptype db_container: str
    :return: the pool over the schema
    :rtype: AsyncIterator[_Sources]
    """
    schema = f"capsrc_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(
        db_container, min_size=1, max_size=4, server_settings={"search_path": schema}, init=init_connection
    )
    assert pool is not None
    try:
        await pool.execute(_DATASOURCES_DDL)
        yield _Sources(pool)
    finally:
        await pool.close()
        admin = await asyncpg.connect(db_container)
        try:
            await admin.execute(f"DROP SCHEMA {schema} CASCADE")
        finally:
            await admin.close()


def _new_row(**overrides: object) -> dict[str, object]:
    """a datasource-kind row as the hub's create endpoint writes it.

    :param overrides: fields to set on top of the defaults
    :ptype overrides: object
    :return: the row
    :rtype: dict[str, object]
    """
    row: dict[str, object] = {
        "id": uuid.uuid4(),
        "name": f"ds-{uuid.uuid4().hex[:6]}",
        "customer_id": uuid.uuid4(),
        "kind": "datasource",
        "datasource_type": "postgres",
        "connection_config": "encrypted-blob",
        "allowed_schemas": ["public"],
        "access_mode": "read",
        "status": "active",
        "visibility": "private",
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    row.update(overrides)
    return row


async def _stored(pool: asyncpg.Pool, source_id: uuid.UUID) -> asyncpg.Record:
    """the row as Postgres holds it.

    :param pool: the pool over the test schema
    :ptype pool: asyncpg.Pool
    :param source_id: the datasource id
    :ptype source_id: uuid.UUID
    :return: the stored row
    :rtype: asyncpg.Record
    """
    row = await pool.fetchrow("SELECT * FROM datasources WHERE id = $1", source_id)
    assert row is not None
    return row


async def _fresh_read(sources: _Sources, source_id: uuid.UUID) -> CapabilitySourceEntity:
    """read the row on a replica that never saw it, so L3 serves it.

    :param sources: the shared pool
    :ptype sources: _Sources
    :param source_id: the datasource id
    :ptype source_id: uuid.UUID
    :return: the entity as read
    :rtype: CapabilitySourceEntity
    """
    entity = await sources.replica().get(source_id)
    assert entity is not None
    return entity


@pytest.mark.parametrize("value", [True, False])
async def test_a_created_source_persists_require_documented_tables_as_written(sources: _Sources, value: bool) -> None:
    writer = sources.replica()
    source = writer.create(_new_row(require_documented_tables=value))
    await writer.save_entity(source)

    assert (await _stored(sources.pool, source.id))["require_documented_tables"] is value
    assert (await _fresh_read(sources, source.id)).require_documented_tables is value
    on_writer = await writer.get(source.id)
    assert on_writer is not None and on_writer.require_documented_tables is value


async def test_a_created_source_that_omits_the_flag_takes_the_column_default(sources: _Sources) -> None:
    writer = sources.replica()
    source = writer.create(_new_row())
    await writer.save_entity(source)

    assert (await _stored(sources.pool, source.id))["require_documented_tables"] is False
    assert (await _fresh_read(sources, source.id)).require_documented_tables is False


@pytest.mark.parametrize("value", [True, False])
async def test_an_update_of_an_existing_source_persists_require_documented_tables(
    sources: _Sources, value: bool
) -> None:
    """the hub PATCH path: load the row, set the flag, save it."""
    writer = sources.replica()
    created = writer.create(_new_row(require_documented_tables=not value))
    await writer.save_entity(created)

    patcher = sources.replica()
    loaded = await patcher.find_by_id(created.id)
    assert loaded is not None and loaded.require_documented_tables is (not value)
    loaded.require_documented_tables = value
    await patcher.save_entity(loaded)

    assert (await _stored(sources.pool, created.id))["require_documented_tables"] is value
    assert (await _fresh_read(sources, created.id)).require_documented_tables is value


async def test_a_collection_save_leaves_the_hub_owned_columns_alone(sources: _Sources) -> None:
    """``spec``, ``face_*`` and ``geo`` are written by the hub directly; a Collection save keeps them."""
    writer = sources.replica()
    created = writer.create(_new_row())
    await writer.save_entity(created)
    await sources.pool.execute(
        "UPDATE datasources SET spec = $2, face_api = TRUE, face_platform_tool = FALSE, geo = $3 WHERE id = $1",
        created.id,
        "openapi: 3.1.0",
        {"layers": []},
    )

    patcher = sources.replica()
    loaded = await patcher.find_by_id(created.id)
    assert loaded is not None
    loaded.require_documented_tables = True
    await patcher.save_entity(loaded)

    stored = await _stored(sources.pool, created.id)
    assert stored["require_documented_tables"] is True
    assert (stored["spec"], stored["face_api"], stored["face_platform_tool"], stored["geo"]) == (
        "openapi: 3.1.0",
        True,
        False,
        {"layers": []},
    )
