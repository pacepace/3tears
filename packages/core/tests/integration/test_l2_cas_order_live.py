"""Integration: the compare-and-swap order fence against a real database and a real broker.

What a fake cannot prove:

- the generated ordered upsert is valid SQL and means what it says: it lands over an older order
  and over a row holding none, and leaves a newer-or-equal one alone;
- coordination v002 adds the order columns to a table v001 created before they existed, backfills
  its rows to the floor, and is a no-op when replayed;
- a real bucket deleted and recreated reports a later creation time and restarts its revisions,
  and the fence admits the first write after it.

Uses the session-scoped ``db_container`` and ``nats_container`` fixtures; a checkout without
docker skips cleanly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.l2_order import L2_ORDER_FLOOR, L2Order, l2_order_of, with_l2_order
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination.migrations.v001_create_coordination_tables import create_coordination_tables
from threetears.core.coordination.migrations.v002_add_l2_order_columns import add_l2_order_columns
from threetears.core.coordination.tables import CoordinationRedemptionsCollection, coordination_collection
from threetears.core.coordination.windowed_counter import WindowedCounter
from threetears.core.coordination.tables import CoordinationCountersCollection
from threetears.core.data.store import DataStore
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_EPOCH = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


async def _schema_pool(db_url: str) -> asyncpg.Pool:
    """a pool bound to a fresh, empty schema.

    :param db_url: the container's connection URL
    :ptype db_url: str
    :return: the pool
    :rtype: asyncpg.Pool
    """
    schema = f"order_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_url)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    return pool


def _store(pool: asyncpg.Pool) -> DataStore:
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(pool))
    return DataStore(uuid.uuid4(), registry)


def _redemption(order: L2Order | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "purpose": "p",
        "key": "k",
        "expires_at": None,
        "date_created": _EPOCH,
        "date_updated": _EPOCH,
        "l2_epoch": None,
        "l2_revision": None,
    }
    return row if order is None else with_l2_order(row, order)


async def test_the_ordered_upsert_lands_only_over_an_older_order(db_container: str) -> None:
    pool = await _schema_pool(db_container)
    try:
        await create_coordination_tables(_store(pool))
        backend = SqlL3Backend(pool)
        backend.register_schema("coordination_redemptions", CoordinationRedemptionsCollection.schema)
        table = "coordination_redemptions"

        async def stored() -> L2Order | None:
            row = await pool.fetchrow("SELECT l2_epoch, l2_revision FROM coordination_redemptions")
            assert row is not None
            return l2_order_of(dict(row))

        # a row some other path wrote carries no order, and orders below every swap.
        assert await backend.upsert(table, _redemption(None), pk=["purpose", "key"]) == 1
        assert await backend.upsert_ordered(table, _redemption(L2Order(_EPOCH, 5))) == 1
        assert await stored() == L2Order(_EPOCH, 5)
        assert await backend.upsert_ordered(table, _redemption(L2Order(_EPOCH, 4))) == 0, "an older order landed"
        assert await backend.upsert_ordered(table, _redemption(L2Order(_EPOCH, 5))) == 0, "an equal order landed"
        assert await stored() == L2Order(_EPOCH, 5)
        assert await backend.upsert_ordered(table, _redemption(L2Order(_EPOCH, 6))) == 1
        recreated = L2Order(_EPOCH + timedelta(seconds=1), 1)
        assert await backend.upsert_ordered(table, _redemption(recreated)) == 1, (
            "a recreated bucket's write was refused"
        )
        assert await stored() == recreated
    finally:
        await pool.close()


async def test_v002_adds_and_backfills_the_order_on_a_table_created_without_it(db_container: str) -> None:
    pool = await _schema_pool(db_container)
    try:
        # the shape v001 rendered before the order columns existed.
        await pool.execute(
            "CREATE TABLE coordination_counters (purpose TEXT NOT NULL, key TEXT NOT NULL, expires_at TIMESTAMPTZ, "
            "date_created TIMESTAMPTZ NOT NULL, date_updated TIMESTAMPTZ, count INTEGER NOT NULL, "
            "window_start TIMESTAMPTZ NOT NULL, PRIMARY KEY (purpose, key))"
        )
        for name in ("coordination_claims", "coordination_redemptions"):
            await pool.execute(
                f"CREATE TABLE {name} (purpose TEXT NOT NULL, key TEXT NOT NULL, PRIMARY KEY (purpose, key))"
            )
        await pool.execute(
            "INSERT INTO coordination_counters (purpose, key, date_created, count, window_start) "
            "VALUES ('login', 'ip-1', now(), 4, now())"
        )
        await add_l2_order_columns(_store(pool))
        await add_l2_order_columns(_store(pool))  # a replay on recovery changes nothing
        row = await pool.fetchrow("SELECT count, l2_epoch, l2_revision FROM coordination_counters")
        assert row is not None
        assert row["count"] == 4
        assert l2_order_of(dict(row)) == L2_ORDER_FLOOR, "an existing row was not backfilled to the floor"
        kinds = {
            r["column_name"]: r["data_type"]
            for r in await pool.fetch(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'coordination_claims' AND table_schema = current_schema()"
            )
        }
        assert kinds["l2_epoch"] == "timestamp with time zone"
        assert kinds["l2_revision"] == "bigint"
    finally:
        await pool.close()


async def test_a_recreated_bucket_restarts_its_revisions_and_is_admitted(
    db_container: str, nats_container: str
) -> None:
    namespace = f"order{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    pool = await _schema_pool(db_container)
    try:
        await create_coordination_tables(_store(pool))
        async with await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="order"
        ) as nats:

            def _registry() -> CollectionRegistry:
                registry = CollectionRegistry()
                registry.configure(
                    l1_backend=SQLiteBackend(db_name=f"order_{uuid.uuid4().hex[:8]}"),
                    l2_client=nats,
                    l3_pool=SqlL3Backend(pool),
                    kv_key_scope="live-order",
                )
                return registry

            first = _registry()
            counter = WindowedCounter(first, purpose="login", window_seconds=600)
            for _ in range(3):
                await counter.record_attempt("ip-1")
            await coordination_collection(first, CoordinationCountersCollection).aclose()
            row = await pool.fetchrow("SELECT count, l2_epoch, l2_revision FROM coordination_counters")
            assert row is not None and row["count"] == 3
            before = l2_order_of(dict(row))
            assert before is not None

            js = nats.jetstream_context()
            await js.delete_stream(f"KV_{namespace}-collections")
            second = _registry()
            assert await WindowedCounter(second, purpose="login", window_seconds=600).record_attempt("ip-1") == 4
            await coordination_collection(second, CoordinationCountersCollection).aclose()
            row = await pool.fetchrow("SELECT count, l2_epoch, l2_revision FROM coordination_counters")
            assert row is not None
            after = l2_order_of(dict(row))
            assert row["count"] == 4, "the first write after the recreation was refused"
            assert after is not None and after.epoch > before.epoch, "the recreated bucket kept its creation time"
            assert after.revision < before.revision, "the recreated bucket kept counting revisions"
    finally:
        await pool.close()
