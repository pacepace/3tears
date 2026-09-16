#!/usr/bin/env python
"""Measure what the durable coordination primitives cost, against the KV path they replaced.

``docs/design-durable-coordination.md``'s "Hot-path cost" section makes claims about the steady
state: a revocation check that is a marker hit, a counter increment that pays only the NATS round
trips it already paid, and L3 seen once per key per write generation. Those are claims about
latency, and prose latency rots and cannot be re-checked. This can be re-run.

Each operation is measured twice on the SAME broker and the SAME machine:

``kv``
    the shape these primitives had before -- a bare NATS KV bucket, create-if-absent or a
    compare-and-swap read-modify-write. This is the number the new path has to be compared
    against, not an absolute budget.
``collection``
    the primitive as it ships: L1, L2 and L3 through a collection.

What to read from the output: the per-operation medians and the p95, and for the revocation check
the L3 query count -- the point of negative caching is that the count stays at one per generation
however many checks run.

Run against a live broker and database::

    uv run python scripts/measure-coordination-latency.py \\
        --nats nats://localhost:4222 --db postgresql://user:pass@localhost/postgres

With no arguments it starts throwaway containers through the same testcontainer fixtures the
integration suite uses, which is slower to start and identical to measure.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination.migrations.v001_create_coordination_tables import create_coordination_tables
from threetears.core.coordination.revocation import RevocationGuard, hashed_denylist_key
from threetears.core.coordination.tables import CoordinationCountersCollection, coordination_collection
from threetears.core.coordination.windowed_counter import WindowedCounter
from threetears.core.data.store import DataStore
from threetears.epoch import EpochGenerationSource
from threetears.nats import NatsClient, set_default_namespace

_ROUNDS = 200
_SCOPE = "measure"


class _CountingBackend(SqlL3Backend):
    """the L3 backend, counting the queries a primitive actually sends."""

    def __init__(self, pool: Any) -> None:
        super().__init__(pool)
        self.queries = 0

    async def fetch_one(self, table: str, pk: Any, *, conn: Any = None) -> Any:
        self.queries += 1
        return await super().fetch_one(table, pk, conn=conn)


async def _timed(operation: Any, rounds: int = _ROUNDS) -> tuple[float, float]:
    """run ``operation(n)`` ``rounds`` times, returning median and p95 milliseconds.

    :param operation: an async callable taking the round number
    :ptype operation: Any
    :param rounds: how many times to run it
    :ptype rounds: int
    :return: median and p95 in milliseconds
    :rtype: tuple[float, float]
    """
    samples: list[float] = []
    for n in range(rounds):
        started = time.perf_counter()
        await operation(n)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return (statistics.median(samples), samples[int(len(samples) * 0.95)])


async def _migrated_pool(db_url: str) -> asyncpg.Pool:
    """a schema with the coordination tables, and a pool bound to it."""
    schema = f"measure_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_url)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_url, min_size=2, max_size=8, server_settings={"search_path": schema})
    assert pool is not None
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(pool))
    await create_coordination_tables(DataStore(uuid.uuid4(), registry))
    return pool


async def _measure(nats_url: str, db_url: str) -> None:
    """measure every hot operation on both paths and print the table."""
    namespace = f"measure{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    pool = await _migrated_pool(db_url)
    try:
        async with await NatsClient.connect(
            nats_url=nats_url, nats_subject_namespace=namespace, client_name="measure"
        ) as nats:
            backend = _CountingBackend(pool)
            l1 = SQLiteBackend(db_name=f"measure_{uuid.uuid4().hex[:8]}")
            registry = CollectionRegistry()
            registry.configure(l1_backend=l1, l2_client=nats, l3_pool=backend, kv_key_scope=_SCOPE)
            registry.set_generation_source(EpochGenerationSource(nats))

            rows: list[tuple[str, str, float, float, str]] = []

            # --- counter increment ------------------------------------------------------
            bucket = await nats.kv_bucket(name="measure_counter", ttl=timedelta(minutes=5))

            async def _kv_increment(n: int) -> None:
                key = f"ip-{n % 20}"
                entry = await bucket.get_entry(key=key)
                if entry is None:
                    await bucket.create(key=key, value=b'{"count": 1}')
                    return
                _value, revision = entry
                await bucket.update(key=key, value=b'{"count": 2}', revision=revision)

            median, p95 = await _timed(_kv_increment)
            rows.append(("counter increment", "kv", median, p95, ""))

            counter = WindowedCounter(registry, purpose="measure", window_seconds=300)
            median, p95 = await _timed(lambda n: counter.record_attempt(f"ip-{n % 20}"))
            rows.append(("counter increment", "collection", median, p95, ""))

            # --- revocation check, the key nobody revoked --------------------------------
            revocations = await nats.kv_bucket(name="measure_revocations", ttl=timedelta(minutes=5))
            # hashed, as the guard hashes: a raw "sub:<id>" is not a legal KV key, which is
            # one reason both paths hash before storing.
            median, p95 = await _timed(lambda n: revocations.get(key=hashed_denylist_key(f"sub:{n % 50}")))
            rows.append(("revocation check (absent)", "kv", median, p95, ""))

            guard = RevocationGuard(registry, purpose="measure", ttl_seconds=3600)
            before = backend.queries
            median, p95 = await _timed(
                lambda n: guard.is_revoked_before(f"sub:{n % 50}", moment=datetime.now(UTC)),
            )
            rows.append(
                ("revocation check (absent)", "collection", median, p95, f"{backend.queries - before} L3 reads")
            )

            # --- revocation check, a key that IS revoked --------------------------------
            await guard.record_revocation("sub:revoked", revoked_at=datetime.now(UTC))
            before = backend.queries
            median, p95 = await _timed(
                lambda n: guard.is_revoked_before("sub:revoked", moment=datetime.now(UTC) - timedelta(hours=1)),
            )
            rows.append(
                ("revocation check (present)", "collection", median, p95, f"{backend.queries - before} L3 reads")
            )

            await coordination_collection(registry, CoordinationCountersCollection).aclose()

            width = max(len(name) for name, *_ in rows)
            print(f"\n{'operation'.ljust(width)}  {'path':<11} {'median':>9} {'p95':>9}  notes")
            print("-" * (width + 45))
            for name, path, med, p95_value, note in rows:
                print(f"{name.ljust(width)}  {path:<11} {med:>8.3f}ms {p95_value:>8.3f}ms  {note}")
            print(f"\n{_ROUNDS} rounds per row, one broker, one database, same machine.\n")
    finally:
        await pool.close()


def main() -> None:
    """parse arguments and run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nats", help="nats:// URL of a live broker; omit to start a container")
    parser.add_argument("--db", help="postgresql:// URL of a live database; omit to start a container")
    args = parser.parse_args()

    if args.nats and args.db:
        asyncio.run(_measure(args.nats, args.db))
        return

    from testcontainers.nats import NatsContainer  # noqa: PLC0415 - only needed without live URLs
    from testcontainers.postgres import PostgresContainer  # noqa: PLC0415

    # jetstream=True, not the container default: every bucket these primitives open is a
    # JetStream KV, so a plain server refuses the first connect.
    with NatsContainer(jetstream=True) as nats, PostgresContainer("postgres:17-alpine") as postgres:
        url = postgres.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
        asyncio.run(_measure(args.nats or nats.nats_uri(), args.db or url))


if __name__ == "__main__":
    main()
