"""Integration: a windowed counter against a real broker and a real database.

What a fake cannot prove:

- the migration's DDL is accepted by Postgres, and the table it creates is the one the collection
  reads and writes (types, composite key, indexes);
- a count that reached L3 survives a broker wipe -- the whole reason the counter left file-backed
  KV -- while an unflushed one does not, which is the write-behind trade stated exactly;
- the expiry sweep's DELETE is valid SQL against that table and removes only closed windows.

Uses the session-scoped ``db_container`` and ``nats_container`` fixtures; a checkout without
docker skips cleanly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination.migrations.v001_create_coordination_tables import create_coordination_tables
from threetears.core.coordination.idempotency import IdempotencyKeyStore
from threetears.core.coordination.tables import (
    CoordinationClaimsCollection,
    CoordinationCountersCollection,
    coordination_collection,
)
from threetears.core.coordination.windowed_counter import WindowedCounter
from threetears.core.data.store import DataStore
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_SCOPE = "live-counter"


async def _migrated_pool(db_url: str) -> asyncpg.Pool:
    """a fresh schema carrying the coordination tables, and a pool bound to it.

    :param db_url: the container's connection URL
    :ptype db_url: str
    :return: the pool
    :rtype: asyncpg.Pool
    """
    schema = f"coord_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_url)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    migration_registry = CollectionRegistry()
    migration_registry.configure(l3_pool=SqlL3Backend(pool))
    await create_coordination_tables(DataStore(uuid.uuid4(), migration_registry))
    return pool


def _registry(nats: NatsClient | None, pool: asyncpg.Pool) -> CollectionRegistry:
    """one replica's registry: its own L1, the shared broker, the shared database."""
    l1 = SQLiteBackend(db_name=f"live_counter_{uuid.uuid4().hex[:8]}")
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=l1,
        l2_client=nats,
        l3_pool=SqlL3Backend(pool),
        kv_key_scope=_SCOPE if nats is not None else None,
    )
    return registry


def _counters(registry: CollectionRegistry) -> CoordinationCountersCollection:
    return coordination_collection(registry, CoordinationCountersCollection)


async def test_a_flushed_count_survives_a_broker_wipe(db_container: str, nats_container: str) -> None:
    namespace = f"coord{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    pool = await _migrated_pool(db_container)
    try:
        async with await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="counter"
        ) as nats:
            registry = _registry(nats, pool)
            counter = WindowedCounter(registry, purpose="login", window_seconds=60)
            assert await counter.record_attempt("ip-1") == 1
            assert await counter.record_attempt("ip-1") == 2
            assert await pool.fetchval("SELECT count(*) FROM coordination_counters") == 0, (
                "a write-behind counter reached L3 before any flush"
            )

            await _counters(registry).aclose()  # the flush interval, forced
            row = await pool.fetchrow("SELECT purpose, key, count, window_start, expires_at FROM coordination_counters")
            assert row is not None
            assert row["purpose"] == "login"
            assert row["count"] == 2
            assert row["expires_at"] - row["window_start"] == timedelta(seconds=60)

            # the wipe: delete the collections bucket the way a broker restart on ephemeral
            # storage does, then count again on a replica that never saw the old value.
            js = nats.jetstream_context()
            await js.delete_stream(f"KV_{namespace}-collections")
            fresh = WindowedCounter(_registry(nats, pool), purpose="login", window_seconds=60)
            assert await fresh.record_attempt("ip-1") == 3, "the wipe reset a throttle that had reached L3"
    finally:
        await pool.close()


async def test_a_flushed_claim_survives_a_broker_wipe(db_container: str, nats_container: str) -> None:
    namespace = f"coord{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    pool = await _migrated_pool(db_container)
    try:
        async with await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="claims"
        ) as nats:
            registry = _registry(nats, pool)
            store = IdempotencyKeyStore(registry, purpose="exports")
            assert (await store.claim("job-1", metadata=b"body-hash")).status == "claimed"
            await store.complete("job-1", result=b"receipt")
            claims = coordination_collection(registry, CoordinationClaimsCollection)
            await claims.aclose()  # the flush interval, forced

            js = nats.jetstream_context()
            await js.delete_stream(f"KV_{namespace}-collections")
            fresh = IdempotencyKeyStore(_registry(nats, pool), purpose="exports")
            outcome = await fresh.claim("job-1")
            assert outcome.status == "exists", "a wipe let a completed operation run a second time"
            assert outcome.record.result == b"receipt"
            assert outcome.record.metadata == b"body-hash"
    finally:
        await pool.close()


async def test_the_sweep_removes_only_closed_windows(db_container: str, nats_container: str) -> None:
    namespace = f"coord{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    pool = await _migrated_pool(db_container)
    try:
        async with await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="sweeper"
        ) as nats:
            registry = _registry(nats, pool)
            live = WindowedCounter(registry, purpose="live", window_seconds=3600)
            closing = WindowedCounter(registry, purpose="closing", window_seconds=1)
            await live.record_attempt("a")
            await closing.record_attempt("b")
            await _counters(registry).aclose()
            assert await pool.fetchval("SELECT count(*) FROM coordination_counters") == 2

            # the second row's window has closed by this cutoff; the first has not.
            deleted = await _counters(registry).sweep_expired(now=datetime.now(UTC) + timedelta(seconds=5))
            assert deleted == 1, "the sweep removed the wrong number of rows"
            purposes = [r["purpose"] for r in await pool.fetch("SELECT purpose FROM coordination_counters")]
            assert purposes == ["live"]
    finally:
        await pool.close()
