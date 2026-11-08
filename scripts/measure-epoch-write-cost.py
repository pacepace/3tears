#!/usr/bin/env python
"""Measure what declaring a write generation (an epoch) costs a collection's ``save_entity``.

The question this answers: can every collection declare ``write_generation = WRITE_GENERATION``,
or is the per-write cost too high? A declared table pays, on every committed write, one append of
the epoch's change record to the change log (``EpochGenerationSource.advance_with_change``: the
stream's sequence is the epoch, so concurrent writers never contend for a counter), and -- where
the registry names a write-marks schema, as the hub's does -- a trigger insert in the write's own
transaction plus a read and a delete of the marks the advance covered.

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
throughput, and the append's own p50/p99 with its failures -- the declared rows over the undeclared
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
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.synchronize import Barrier
from typing import Any

import asyncpg
from sqlalchemy import MetaData

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import WRITE_GENERATION, CollectionRegistry
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, TableSchema, collection_for_schema
from threetears.core.collections.schema_backed import Column as SchemaColumn
from threetears.core.collections.write_marks import install_write_mark_triggers, write_marks_ddl
from threetears.core.config import DefaultCoreConfig
from threetears.core.exceptions import GenerationUnavailableError
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
    """the epoch appends one writer process made, timed through the source's public advance."""

    def __init__(self) -> None:
        self.samples_ms: list[float] = []
        self.failures = 0


class _CountingSource(EpochGenerationSource):
    """the shipped epoch source, timing each advance it makes. Measurement only."""

    def __init__(self, nats_client: Any, tally: _Tally) -> None:
        super().__init__(nats_client)
        self._tally = tally

    async def advance_with_change(
        self, table_name: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False
    ) -> str:
        began = time.perf_counter()
        try:
            return await super().advance_with_change(table_name, rows, keys_unknown=keys_unknown)
        except GenerationUnavailableError:
            self._tally.failures += 1
            raise
        finally:
            self._tally.samples_ms.append((time.perf_counter() - began) * 1000)


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
    advance_ms: list[float]
    advance_failures: int
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
            tally.samples_ms.clear()
            tally.failures = 0
            await asyncio.to_thread(barrier.wait)

            samples: list[float] = []
            failures = 0
            started = time.perf_counter()
            for n in range(job.rounds):
                entity = rows.create({"id": f"w{job.writer}-{n}", "n": n, "payload": _PAYLOAD})
                began = time.perf_counter()
                try:
                    await rows.save_entity(entity)
                # a save whose epoch could not be appended raises after its row committed; that is a
                # result to count, not a crash
                # prawduct:allow prawduct/broad-except -- counted as a failed save in the table this prints
                except Exception:  # noqa: BLE001
                    failures += 1
                samples.append((time.perf_counter() - began) * 1000)
            finished = time.perf_counter()
            return _Result(samples, started, finished, list(tally.samples_ms), tally.failures, failures)
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
    advances = sorted(a for r in collected for a in r.advance_ms)
    return {
        "variant": variant,
        "writers": writers,
        "writes": len(samples),
        "p50": statistics.median(samples),
        "p99": samples[min(len(samples) - 1, int(len(samples) * 0.99))],
        "per_sec": len(samples) / wall if wall > 0 else 0.0,
        "advance_p50": statistics.median(advances) if advances else 0.0,
        "advance_p99": advances[min(len(advances) - 1, int(len(advances) * 0.99))] if advances else 0.0,
        "advances": len(advances),
        "advance_failures": sum(r.advance_failures for r in collected),
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
        "\n| writers | variant | writes | p50 ms | p99 ms | writes/s | append p50 ms | append p99 ms | advances "
        "| append failed | failed |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in rows:
        print(
            f"| {r['writers']} | {r['variant']} | {r['writes']} | {r['p50']:.2f} | {r['p99']:.2f} | "
            f"{r['per_sec']:.0f} | {r['advance_p50']:.2f} | {r['advance_p99']:.2f} | {r['advances']} | "
            f"{r['advance_failures']} | {r['failures']} |"
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
