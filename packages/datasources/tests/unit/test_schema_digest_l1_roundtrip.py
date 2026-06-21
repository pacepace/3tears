"""real-L1 (SQLite) round-trip for the schema-priming digest collection.

this is the test the adversarial review flagged as MISSING: the unit tests
mocked the collection, so the CRITICAL by-pk break (the collection
inherited ``primary_key_column = "id"`` while the table is keyed on
``datasource_id``) was invisible -- a green suite over a broken feature.

these tests stand up a REAL :class:`SQLiteBackend` L1 (no Docker, runs in
CI) initialized with the digest table and drive
``save_entity`` + ``get(datasource_id)`` through it. before the fix,
``get`` emits ``SELECT ... WHERE id = ?`` against a table with no ``id``
column and raises ``sqlite3.OperationalError: no such column: id`` -- so
this test would have caught the regression. the agent's hot read path is
exactly this by-pk L1 lookup.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.datasources.collections import DataSourceSchemaDigestCollection


def _digest_metadata() -> MetaData:
    """the digest table schema, mirroring the v029 DDL + AGENT_L1_METADATA.

    keyed on ``datasource_id`` (NOT ``id``) -- the exact shape that makes
    the inherited-``id`` default break, and the shape the agent pod's L1
    mirror carries.
    """
    metadata = MetaData()
    Table(
        "datasource_schema_digests",
        metadata,
        Column("datasource_id", UUID, primary_key=True),
        Column("customer_id", UUID, nullable=True),
        Column("tables", JSONB, nullable=True),
        Column("source_fingerprint", Text, nullable=True),
        Column("date_created", TIMESTAMP, nullable=False),
        Column("date_updated", TIMESTAMP, nullable=False),
    )
    return metadata


@pytest.fixture()
def l1_backend() -> Iterator[SQLiteBackend]:
    backend = SQLiteBackend(db_name=f"test_digest_{uuid.uuid4().hex[:8]}")
    backend.initialize(_digest_metadata())
    yield backend
    from threetears.core._bridge import drain, shutdown

    drain()
    shutdown()
    backend.reset()


@pytest.fixture()
def registry(l1_backend: SQLiteBackend) -> CollectionRegistry:
    # a real SQLite L1 + a thin L3 stub. the L3 stub only lets
    # ``save_entity`` past its INSERT-rowcount guard (the digest's
    # ``save_to_store`` returns 0 with no pool, which reads as an INSERT
    # failure); the by-pk WRITE + READ this test exercises run against the
    # REAL SQLite L1 -- that is the surface where the inherited-"id" PK bug
    # lived (ON CONFLICT (id) on write, WHERE id=? on read).
    l3_pool = MagicMock()
    l3_pool.execute = AsyncMock(return_value="INSERT 0 1")
    l3_pool.fetchrow = AsyncMock(return_value=None)
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1_backend, l3_pool=l3_pool)
    return reg


@pytest.fixture()
def config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


def _digest_data(datasource_id: Any, customer_id: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "datasource_id": datasource_id,
        "customer_id": customer_id,
        "tables": [
            {
                "schema": "reporting_prod",
                "table": "report_geofacts_joined_data",
                "description": "joined geo facts",
                "columns": [
                    {
                        "name": "metric_name",
                        "type": "character varying",
                        "description": "the EAV metric label",
                    },
                ],
            },
        ],
        "source_fingerprint": "deadbeef",
        "date_created": now,
        "date_updated": now,
    }


class TestDigestL1RoundTrip:
    """save_entity + get(datasource_id) round-trips through a real L1."""

    @pytest.mark.asyncio
    async def test_save_then_get_by_datasource_id(
        self,
        registry: CollectionRegistry,
        config: DefaultCoreConfig,
    ) -> None:
        # this is the exact hot path the agent uses: write a digest, then
        # read it BY PRIMARY KEY (datasource_id). with the inherited-"id"
        # default this get() raised "no such column: id".
        collection = DataSourceSchemaDigestCollection(
            registry=registry,
            config=config,
        )
        datasource_id = uuid.uuid4()
        customer_id = uuid.uuid4()

        entity = collection.create(_digest_data(datasource_id, customer_id))
        await collection.save_entity(entity)

        result = await collection.get(datasource_id)
        assert result is not None
        assert result.id == datasource_id
        # the structured projection survives the L1 round-trip as a list.
        tables = result.tables
        assert isinstance(tables, list)
        assert tables[0]["table"] == "report_geofacts_joined_data"
        assert tables[0]["columns"][0]["name"] == "metric_name"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(
        self,
        registry: CollectionRegistry,
        config: DefaultCoreConfig,
    ) -> None:
        collection = DataSourceSchemaDigestCollection(
            registry=registry,
            config=config,
        )
        assert await collection.get(uuid.uuid4()) is None
