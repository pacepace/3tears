#!/usr/bin/env python
"""Measure what declaring a write generation (an epoch) costs a collection's ``save_entity``.

The question this answers: can every collection declare ``write_generation = WRITE_GENERATION``,
or is the per-write cost too high? A declared table pays, on every committed write, one
compare-and-swap on the table's key in the epoch bucket (``EpochGenerationSource.advance``), one
put of the epoch's change record (``advance_with_change``), and -- where the registry names a
write-marks schema, as the hub's does -- a trigger insert in the write's own transaction plus a
read and a delete of the marks the advance covered. Every writer of a table advances the SAME
key, so concurrent writers contend on it: a lost compare-and-swap is retried after up to 20 ms of
jitter, which is where the tail comes from.

Three variants, each over its own table, all on the SAME broker, database and machine:

``undeclared``
    the default today (``WRITE_GENERATION_UNDECLARED``): no advance on a save.
``declared``
    ``WRITE_GENERATION``: an advance with its change record on every save, no write marks.
``declared+marks``
    ``WRITE_GENERATION`` with the write-mark trigger installed and the registry naming the
    marks schema: the hub's production shape.

Each writer is its own process with its own NATS connection, database pool, L1 and registry, as
concurrent writers are separate pods; they write distinct rows of one table, so they contend
only where the platform makes them contend. A pod writing through the hub's L3 broker pays the
broker's advance inside its request round trip; this measures that advance directly.

What to read from the output: p50 and p99 per variant at each writer count, the aggregate
throughput, and the compare-and-swap losses per write -- the declared rows over the undeclared
row at the same writer count are the cost.

Run against a live broker and database::

    uv run python scripts/measure-epoch-write-cost.py \\
        --nats nats://localhost:4222 --db postgresql://user:pass@localhost/postgres

With no URLs it starts throwaway NATS (JetStream) and Postgres containers. Not part of any test
lane: it is run by hand, and its numbers go where the decision is recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import statistics
import time
import uuid
from dataclasses import dataclass
from multiprocessing.synchronize import Barrier
from typing import Any

import asyncpg
from sqlalchemy import MetaData

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import WRITE_GENERATION, CollectionRegistry
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, TableSchema, collection_for_schema
from threetears.core.collections.schema_backed import Column as SchemaColumn
from threetears.core.collections.write_marks import install_write_mark_triggers, write_marks_ddl
from threetears.core.config import DefaultCoreConfig
from threetears.epoch import EpochGenerationSource
from threetears.nats import NatsClient, set_default_namespace

_VARIANTS = ("undeclared", "declared", "declared+marks")
_WARMUP = 5
_PAYLOAD = "x" * 200


def _schema_for(variant: str) -> TableSchema:
    """the table one variant writes; each variant has its own, since one table has one declaration.

    :param variant: one of :data:`_VARIANTS`
    :ptype variant: str
    :return: the table's schema
    :rtype: TableSchema
    """
    return TableSchema(
        name=f"bench_{variant.replace('+', '_')}",
        primary_key="id",
        columns=[
            SchemaColumn("id", STRING_TYPE),
            SchemaColumn("n", BIGINT_TYPE, nullable=True),
            SchemaColumn("payload", STRING_TYPE, nullable=True),
        ],
        on_conflict="update",
    )


def _collection_class(variant: str) -> type[Any]:
    """the collection class one variant writes through.

    :param variant: one of :data:`_VARIANTS`
    :ptype variant: str
    :return: the class, declared or not
    :rtype: type[Any]
    """
    base = collection_for_schema(_schema_for(variant))
    if variant == "undeclared":
        return base
    return type(f"Bench_{variant.replace('+', '_')}", (base,), {"write_generation": WRITE_GENERATION})


class _Tally:
    """compare-and-swap outcomes on the epoch bucket, counted in one writer process."""

    def __init__(self) -> None:
        self.losses = 0
        self.wins = 0


class _CountingBucket:
    """the epoch bucket, counting each compare-and-swap it refuses; everything else passes through."""

    def __init__(self, inner: Any, tally: _Tally) -> None:
        self._inner = inner
        self._tally = tally

    async def create(self, *, key: str, value: bytes, **kwargs: Any) -> Any:
        written = await self._inner.create(key=key, value=value, **kwargs)
        self._count(written)
        return written

    async def update(self, *, key: str, value: bytes, revision: int, **kwargs: Any) -> Any:
        written = await self._inner.update(key=key, value=value, revision=revision, **kwargs)
        self._count(written)
        return written

    def _count(self, written: Any) -> None:
        if written is None:
            self._tally.losses += 1
        else:
            self._tally.wins += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _CountingSource(EpochGenerationSource):
    """the shipped epoch source, over a bucket that counts lost compare-and-swaps. Measurement only."""

    def __init__(self, nats_client: Any, tally: _Tally) -> None:
        super().__init__(nats_client)
        self._tally = tally

    async def _bucket(self) -> Any:
        return _CountingBucket(await super()._bucket(), self._tally)


@dataclass(frozen=True)
class _Job:
    """what one writer process is told to do."""

    variant: str
    writer: int
    rounds: int
    nats_url: str
    db_url: str
    db_schema: str
    namespace: str


@dataclass(frozen=True)
class _Result:
    """what one writer process measured."""

    samples_ms: list[float]
    started: float
    finished: float
    cas_losses: int
    cas_wins: int
    failures: int


async def _write(job: _Job, barrier: Barrier) -> _Result:
    """warm up, wait for every writer, then save ``job.rounds`` new rows and time each save.

    :param job: this writer's job
    :ptype job: _Job
    :param barrier: released once every writer is warmed up
    :ptype barrier: Barrier
    :return: the timings and counts
    :rtype: _Result
    """
    set_default_namespace(job.namespace)
    pool = await asyncpg.create_pool(job.db_url, min_size=1, max_size=2, server_settings={"search_path": job.db_schema})
    assert pool is not None
    tally = _Tally()
    try:
        async with await NatsClient.connect(
            nats_url=job.nats_url, nats_subject_namespace=job.namespace, client_name=f"bench-{job.writer}"
        ) as nats:
            metadata = MetaData()
            _schema_for(job.variant).to_sqlalchemy_table(metadata)
            l1 = SQLiteBackend(db_name=f"bench_{uuid.uuid4().hex[:8]}")
            l1.initialize(metadata)
            registry = CollectionRegistry()
            registry.configure(l1_backend=l1, l2_client=nats, l3_pool=SqlL3Backend(pool), kv_key_scope="bench")
            registry.set_generation_source(_CountingSource(nats, tally))
            if job.variant == "declared+marks":
                registry.set_write_marks_schema(job.db_schema)
            rows = _collection_class(job.variant)(registry, DefaultCoreConfig())

            for n in range(_WARMUP):
                await rows.save_entity(rows.create({"id": f"warm-{job.writer}-{n}", "n": n, "payload": _PAYLOAD}))
            tally.losses = tally.wins = 0
            await asyncio.to_thread(barrier.wait)

            samples: list[float] = []
            failures = 0
            started = time.perf_counter()
            for n in range(job.rounds):
                entity = rows.create({"id": f"w{job.writer}-{n}", "n": n, "payload": _PAYLOAD})
                began = time.perf_counter()
                try:
                    await rows.save_entity(entity)
                # a save whose advance ran out of compare-and-swap rounds raises after its row committed;
                # that is a result to count, not a crash
                # prawduct:allow prawduct/broad-except -- counted as a failed save in the table this prints
                except Exception:  # noqa: BLE001
                    failures += 1
                samples.append((time.perf_counter() - began) * 1000)
            finished = time.perf_counter()
            return _Result(samples, started, finished, tally.losses, tally.wins, failures)
    finally:
        await pool.close()


def _writer_main(job: _Job, barrier: Barrier, results: Any) -> None:
    """one writer process: run its job and put the result on the queue.

    :param job: this writer's job
    :ptype job: _Job
    :param barrier: shared start line
    :ptype barrier: Barrier
    :param results: queue the result is put on
    :ptype results: Any
    :return: nothing
    :rtype: None
    """
    results.put(asyncio.run(_write(job, barrier)))


async def _prepare(db_url: str) -> str:
    """create a throwaway schema with every variant's table, the marks queue and the trigger on one table.

    :param db_url: the database
    :ptype db_url: str
    :return: the schema's name
    :rtype: str
    """
    db_schema = f"bench_{uuid.uuid4().hex[:8]}"
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=2)
    assert pool is not None
    try:
        await pool.execute(f"CREATE SCHEMA {db_schema}")
        for variant in _VARIANTS:
            name = _schema_for(variant).name
            await pool.execute(f"CREATE TABLE {db_schema}.{name} (id TEXT PRIMARY KEY, n BIGINT, payload TEXT)")
        for statement in write_marks_ddl(db_schema):
            await pool.execute(statement)
        await install_write_mark_triggers(pool, marks_schema=db_schema, tables=[_schema_for("declared+marks").name])
    finally:
        await pool.close()
    return db_schema


def _run(variant: str, writers: int, rounds: int, nats_url: str, db_url: str, db_schema: str) -> dict[str, Any]:
    """run ``writers`` writer processes on one variant and pool what they measured.

    :return: one row of the table
    :rtype: dict[str, Any]
    """
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(writers)
    results = ctx.Queue()
    namespace = f"bench{uuid.uuid4().hex[:6]}"
    processes = [
        ctx.Process(
            target=_writer_main,
            args=(_Job(variant, w, rounds, nats_url, db_url, db_schema, namespace), barrier, results),
        )
        for w in range(writers)
    ]
    for process in processes:
        process.start()
    collected: list[_Result] = [results.get() for _ in processes]
    for process in processes:
        process.join()
    samples = sorted(s for r in collected for s in r.samples_ms)
    wall = max(r.finished for r in collected) - min(r.started for r in collected)
    advances = sum(r.cas_wins for r in collected)
    losses = sum(r.cas_losses for r in collected)
    return {
        "variant": variant,
        "writers": writers,
        "writes": len(samples),
        "p50": statistics.median(samples),
        "p99": samples[min(len(samples) - 1, int(len(samples) * 0.99))],
        "per_sec": len(samples) / wall if wall > 0 else 0.0,
        "losses_per_write": losses / len(samples),
        "advances": advances,
        "failures": sum(r.failures for r in collected),
    }


def _measure(nats_url: str, db_url: str, writer_counts: list[int], rounds: int) -> None:
    """run every variant at every writer count and print the comparison table.

    :return: nothing
    :rtype: None
    """
    db_schema = asyncio.run(_prepare(db_url))
    rows: list[dict[str, Any]] = []
    for writers in writer_counts:
        for variant in _VARIANTS:
            rows.append(_run(variant, writers, rounds, nats_url, db_url, db_schema))
            row = rows[-1]
            print(f"  done: {variant} x{writers} p50={row['p50']:.2f}ms p99={row['p99']:.2f}ms", flush=True)

    print(
        "\n| writers | variant | writes | p50 ms | p99 ms | writes/s | CAS losses/write | advances | failed |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    for r in rows:
        print(
            f"| {r['writers']} | {r['variant']} | {r['writes']} | {r['p50']:.2f} | {r['p99']:.2f} | "
            f"{r['per_sec']:.0f} | {r['losses_per_write']:.3f} | {r['advances']} | {r['failures']} |"
        )
    print(f"\n{rounds} saves per writer after {_WARMUP} warm-up saves; one broker, one database, same machine.\n")


def main() -> None:
    """parse arguments and run the measurement, starting containers when no URLs are given.

    :return: nothing
    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nats", help="nats:// URL of a live JetStream broker; omit to start a container")
    parser.add_argument("--db", help="postgresql:// URL of a live database; omit to start a container")
    parser.add_argument("--writers", type=int, nargs="+", default=[1, 8, 32], help="concurrent writer counts")
    parser.add_argument("--rounds", type=int, default=200, help="saves per writer")
    args = parser.parse_args()

    if args.nats and args.db:
        _measure(args.nats, args.db, args.writers, args.rounds)
        return

    from testcontainers.community.postgres import PostgresContainer  # noqa: PLC0415 - only needed without live URLs

    from threetears.core.testing.fixtures import nats_server, nats_server_uri  # noqa: PLC0415

    with (
        nats_server(config="jetstream {\n  store_dir: /tmp/js\n}\n") as nats,
        PostgresContainer("postgres:17-alpine").with_command("postgres -c max_connections=200") as postgres,
    ):
        url = postgres.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
        _measure(args.nats or nats_server_uri(nats), args.db or url, args.writers, args.rounds)


if __name__ == "__main__":
    main()
