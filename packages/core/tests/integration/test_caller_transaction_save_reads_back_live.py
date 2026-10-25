"""Integration: a save inside a caller's transaction hands back the row L3 holds, not the row as sent.

The defect (v0.78.0): a partial update keeps the stored value of every column it leaves out, and
a synchronous save reads the row back so the caches and the handle hold it. A save joining a
caller's transaction (``save_entity(entity, conn=conn)``) did not: its handle held the row as sent,
so reading a column the row left out raised ``AttributeError``. The hub hit it re-saving partial
``platform.datasources`` rows in one transaction.

What this pins, against real Postgres and a real NATS KV bucket:

- the handle holds the stored value of the column the row leaves out, read over the caller's
  connection, so it sees an uncommitted write that came before the save in the same transaction;
- no tier takes the row while the transaction is open, and once it commits a read takes the
  committed row into L1 and L2;
- a rolled-back transaction leaves nothing cached, and L3 as it was.

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
from threetears.core.collections.caller_transaction import CallerTransaction
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

_TABLE = "txn_sources"


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


class _RolledBackError(Exception):
    """raised inside the transaction to roll it back."""


@dataclass
class _Tiers:
    pool: asyncpg.Pool
    nats: NatsClient
    sources: _Sources

    async def l3(self, source_id: uuid.UUID) -> asyncpg.Record:
        row = await self.pool.fetchrow(f"SELECT * FROM {_TABLE} WHERE id = $1", source_id)  # noqa: S608
        assert row is not None
        return row

    def l1(self, source_id: uuid.UUID) -> dict[str, Any] | None:
        return self.sources.get_row_sync(source_id)

    async def l2(self, source_id: uuid.UUID) -> dict[str, Any] | None:
        bucket = await self.nats.kv_bucket(name="collections")
        raw = await bucket.get(key=self.sources.l2_key(source_id))
        return None if raw is None else dict(json.loads(raw))


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
    schema = f"txnread_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    namespace = f"txnread{uuid.uuid4().hex[:6]}"
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
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="txn-read-back"
        ) as nats:
            l1 = SQLiteBackend(db_name=f"txnread_{uuid.uuid4().hex[:8]}")
            l1.initialize(_Sources.schema.to_sqlalchemy_table(MetaData()).metadata)
            registry = CollectionRegistry()
            registry.configure(
                l1_backend=l1, l2_client=nats, l3_pool=SqlL3Backend(pool), kv_key_scope=f"t-{uuid.uuid4().hex[:6]}"
            )
            config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
            yield _Tiers(pool, nats, _Sources(registry, config))
    finally:
        await pool.close()


async def _seed(tiers: _Tiers) -> uuid.UUID:
    """store a source whose nullable ``geo`` is set; L1 and L2 hold it, as a served row is held.

    :param tiers: the tiers
    :ptype tiers: _Tiers
    :return: the source's id
    :rtype: uuid.UUID
    """
    source_id = uuid.uuid4()
    first = {"id": source_id, "name": "first", "geo": "county", "rank": 2, "date_updated": None}
    await tiers.sources.save_entity(tiers.sources.create(first))
    assert (await tiers.l2(source_id) or {}).get("geo") == "county", "the seed is not cached"
    return source_id


async def test_a_partial_save_in_a_caller_transaction_hands_back_the_stored_row(tiers: _Tiers) -> None:
    source_id = await _seed(tiers)
    async with tiers.pool.acquire() as conn, CallerTransaction(conn):
        # an earlier statement of the same transaction, not yet committed: only a read on this
        # connection sees it
        await conn.execute(f"UPDATE {_TABLE} SET geo = 'state' WHERE id = $1", source_id)  # noqa: S608
        resaved = tiers.sources.create({"id": source_id, "name": "second"})
        await tiers.sources.save_entity(resaved, conn=conn)

        assert resaved.name == "second"
        assert resaved.geo == "state", "the handle does not hold the stored geo"
        assert resaved.rank == 2, "the handle does not hold the stored rank"
        assert resaved.date_created is not None
        assert tiers.l1(source_id) is None, "L1 took a row the transaction has not committed"
        assert (await tiers.l2(source_id) or {}).get("name") != "second", "L2 took an uncommitted row"

    stored = await tiers.l3(source_id)
    assert (stored["name"], stored["geo"], stored["rank"]) == ("second", "state", 2)
    assert resaved.geo == "state", "the handle lost the stored geo at commit"
    # the commit settled every tier: nothing stale is left, and a read takes the committed row
    assert tiers.l1(source_id) is None, "L1 holds a row from before the commit"
    assert await tiers.l2(source_id) is None, "L2 holds a row from before the commit"
    loaded = await tiers.sources.get(source_id)
    assert loaded is not None and (loaded.name, loaded.geo) == ("second", "state")
    cached = tiers.l1(source_id)
    assert cached is not None and (cached["name"], cached["geo"], cached["rank"]) == ("second", "state", 2)
    l2 = await tiers.l2(source_id)
    assert l2 is not None and (l2["name"], l2["geo"], l2["rank"]) == ("second", "state", 2)


async def test_a_rolled_back_save_caches_nothing(tiers: _Tiers) -> None:
    source_id = await _seed(tiers)
    resaved = tiers.sources.create({"id": source_id, "name": "second"})
    with pytest.raises(_RolledBackError):
        async with tiers.pool.acquire() as conn, CallerTransaction(conn):
            await tiers.sources.save_entity(resaved, conn=conn)
            assert resaved.geo == "county", "the handle does not hold the stored geo"
            raise _RolledBackError

    stored = await tiers.l3(source_id)
    assert (stored["name"], stored["geo"]) == ("first", "county"), "the rollback did not undo the save"
    assert tiers.l1(source_id) is None, "L1 holds a row after the rollback"
    assert await tiers.l2(source_id) is None, "L2 holds a row after the rollback"
    loaded = await tiers.sources.get(source_id)
    assert loaded is not None and loaded.name == "first"
    l2 = await tiers.l2(source_id)
    assert l2 is not None and l2["name"] == "first", "L2 took a rolled-back row"
