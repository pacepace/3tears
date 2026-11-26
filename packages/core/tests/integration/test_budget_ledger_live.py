"""Integration: the budget ledger against a real broker and a real database (the key pool's KP-L2).

What a fake cannot prove:

- v004's DDL is accepted by Postgres, and the tables it creates are the ones the collections read and
  write: money as text, the maps as JSON text, the order columns the persist is fenced on;
- spend reached L3 before each call returned, so a ledger built after the collections bucket is
  deleted -- what a broker restart on memory storage does -- continues from it;
- two replicas share one ledger: each sees the other's accounts and records;
- concurrent reservations from two replicas never together pass the budget.

**Two replicas, not two processes.** Each replica is its own registry, with its own L1, its own NATS
connection and its own database pool, sharing nothing in this process but the event loop. The budget
holds because of the broker's revision compare-and-swap and the database's order fence, and neither
of those can tell two connections in one process from two processes; a second interpreter would add
only a second event loop, and the setup cost of a subprocess harness the suite has no other use for.

Uses the session-scoped ``db_container`` and ``nats_container`` fixtures.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination import BudgetLedger
from threetears.core.coordination.migrations.v004_create_ledger_tables import create_ledger_tables
from threetears.core.data.store import DataStore
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_SCOPE = "live-ledger"
_EXPIRES = datetime.now(UTC) + timedelta(hours=1)


async def _migrated_pool(db_url: str) -> asyncpg.Pool:
    """a fresh schema carrying the ledger tables, as v004 creates them, and a pool bound to it.

    :param db_url: the container's connection URL
    :ptype db_url: str
    :return: the pool
    :rtype: asyncpg.Pool
    """
    schema = f"ledger_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_url)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    migration_registry = CollectionRegistry()
    migration_registry.configure(l3_pool=SqlL3Backend(pool))
    await create_ledger_tables(DataStore(uuid.uuid4(), migration_registry))
    return pool


def _replica(nats: NatsClient, pool: asyncpg.Pool) -> BudgetLedger:
    """one replica's ledger: its own L1, the given broker connection, the shared database."""
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=SQLiteBackend(db_name=f"live_ledger_{uuid.uuid4().hex[:8]}"),
        l2_client=nats,
        l3_pool=SqlL3Backend(pool),
        kv_key_scope=_SCOPE,
    )
    return BudgetLedger(registry, purpose="pool")


async def _connect(nats_url: str, namespace: str, name: str) -> NatsClient:
    return await NatsClient.connect(nats_url=nats_url, nats_subject_namespace=namespace, client_name=name)


async def test_spend_survives_a_new_ledger_after_the_bucket_is_wiped(db_container: str, nats_container: str) -> None:
    namespace = f"ledger{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    pool = await _migrated_pool(db_container)
    try:
        async with await _connect(nats_container, namespace, "ledger") as nats:
            ledger = _replica(nats, pool)
            assert await ledger.reserve("acct", "r1", Decimal("6.00"), Decimal("10.00"), _EXPIRES)
            await ledger.settle("acct", "r1", Decimal("7.25"))
            assert await ledger.reserve("acct", "r2", Decimal("1.50"), Decimal("10.00"), _EXPIRES)
            assert await ledger.compare_and_set_record("active", 0, {"key": "a", "budget": "10.00"}) == (
                True,
                1,
                {"key": "a", "budget": "10.00"},
            )

            # in L3 already, with no flush: the tables are synchronous
            row = await pool.fetchrow(
                "SELECT spent, reserved, open_reservations, orphaned, expires_at, l2_epoch, l2_revision "
                "FROM coordination_ledger_accounts WHERE purpose = 'pool' AND key = 'acct'"
            )
            assert row is not None
            assert (row["spent"], row["reserved"]) == ("7.25", "1.50")
            assert json.loads(row["open_reservations"]) == {
                "r2": {"amount": "1.50", "expires_at": _EXPIRES.isoformat()}
            }
            assert json.loads(row["orphaned"]) == {}
            assert row["expires_at"] is None
            assert row["l2_epoch"] is not None and row["l2_revision"] is not None, "a persist carried no order"
            assert await pool.fetchval("SELECT version FROM coordination_documents WHERE key = 'active'") == 1

            js = nats.jetstream_context()
            await js.delete_stream(f"KV_{namespace}-collections")
            fresh = _replica(nats, pool)

            snap = (await fresh.snapshot(["acct"]))["acct"]
            assert (snap["spent"], snap["reserved"]) == (Decimal("7.25"), Decimal("1.50")), "the wipe lost spend"
            assert await fresh.reserve("acct", "r3", Decimal("1.26"), Decimal("10.00"), _EXPIRES) is False
            await fresh.settle("acct", "r2", Decimal("1.00"))
            assert (await fresh.snapshot(["acct"]))["acct"]["spent"] == Decimal("8.25")
            assert await fresh.read_record("active") == (1, {"key": "a", "budget": "10.00"})
            assert await fresh.compare_and_set_record("active", 0, {"key": "b"}) == (
                False,
                1,
                {"key": "a", "budget": "10.00"},
            ), "a wipe reset a record's version"
    finally:
        await pool.close()


async def test_two_replicas_share_accounts_and_records(db_container: str, nats_container: str) -> None:
    namespace = f"ledger{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    pool = await _migrated_pool(db_container)
    try:
        async with (
            await _connect(nats_container, namespace, "ledger-a") as nats_a,
            await _connect(nats_container, namespace, "ledger-b") as nats_b,
        ):
            first, second = _replica(nats_a, pool), _replica(nats_b, pool)

            assert await first.reserve("acct", "r1", Decimal(4), Decimal(10), _EXPIRES)
            await second.settle("acct", "r1", Decimal(3))
            assert (await first.snapshot(["acct"]))["acct"]["spent"] == Decimal(3)

            assert await second.reserve("acct", "r2", Decimal(7), Decimal(10), _EXPIRES)
            assert await first.reserve("acct", "r3", Decimal("0.01"), Decimal(10), _EXPIRES) is False

            swept_at = _EXPIRES + timedelta(seconds=1)
            assert await first.sweep_expired(["acct"], swept_at) == {"acct": ["r2"]}
            assert (await second.snapshot(["acct"]))["acct"]["orphaned"] == {"r2": Decimal(7)}
            await second.release("acct", "r2")
            assert (await first.snapshot(["acct"]))["acct"]["spent"] == Decimal(3)

            assert (await first.compare_and_set_record("active", 0, {"key": "a"}))[0]
            assert await second.read_record("active") == (1, {"key": "a"})
            assert await second.compare_and_set_record("active", 0, {"key": "b"}) == (False, 1, {"key": "a"})
    finally:
        await pool.close()


async def test_concurrent_reservations_from_two_replicas_never_exceed_the_budget(
    db_container: str, nats_container: str
) -> None:
    namespace = f"ledger{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    pool = await _migrated_pool(db_container)
    try:
        async with (
            await _connect(nats_container, namespace, "ledger-a") as nats_a,
            await _connect(nats_container, namespace, "ledger-b") as nats_b,
        ):
            first, second = _replica(nats_a, pool), _replica(nats_b, pool)
            results = await asyncio.gather(
                *(
                    (first if i % 2 else second).reserve("acct", f"r{i}", Decimal(1), Decimal(7), _EXPIRES)
                    for i in range(20)
                )
            )
            assert sum(results) == 7, f"admitted {sum(results)} of 20 against a budget that fits 7"

            row = await pool.fetchrow(
                "SELECT reserved, open_reservations FROM coordination_ledger_accounts WHERE key = 'acct'"
            )
            assert row is not None
            assert row["reserved"] == "7"
            admitted = {f"r{i}" for i, ok in enumerate(results) if ok}
            assert set(json.loads(row["open_reservations"])) == admitted, "L3 holds a different set than was admitted"
    finally:
        await pool.close()
