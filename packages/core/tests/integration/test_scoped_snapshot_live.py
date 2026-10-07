"""Integration: a scoped snapshot -- whole tables held in L2 as one columnar chunk per scope, loaded into DuckDB.

What a fake cannot prove, so every test runs against a real nats-server, a real Postgres L3 and a real
DuckDB:

- the first replica, finding nothing in L2, builds the chunks from L3 and says so in its status;
- a second replica loads every table from the chunks alone, without reading L3;
- a scope published by one replica reaches the other through its pointer watch, and only that scope
  is applied, in one DuckDB transaction a reader never sees half of;
- objects of epochs no longer served are retired;
- when NATS loses the snapshot, a replica rebuilds it from L3, and the status names that path;
- a scope L3 moved without a publish (a writer that died after its commit) is caught up from L3.

Uses the session-scoped ``nats_container`` and ``db_container`` fixtures; a checkout without docker
skips cleanly.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.duckdb import DuckDBBackend
from threetears.core.collections.complete_copy import Unsettled
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.collections.scoped_snapshot import (
    ScopedSnapshot,
    SnapshotPhase,
    SnapshotSource,
    SnapshotTable,
)
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_RESULTS = TableSchema(
    name="results",
    primary_key=("race", "county"),
    columns=[
        Column("race", STRING_TYPE),
        Column("county", STRING_TYPE),
        Column("state", STRING_TYPE),
        Column("votes", BIGINT_TYPE, nullable=True),
    ],
    on_conflict="update",
)
_COUNTIES = TableSchema(
    name="counties",
    primary_key="county",
    columns=[Column("county", STRING_TYPE), Column("state", STRING_TYPE), Column("total", BIGINT_TYPE)],
    on_conflict="update",
)
_TABLES = (
    SnapshotTable(name="results", scope_column="state", key=("race", "county")),
    SnapshotTable(name="counties", scope_column="state", key=("county",)),
)
_STATES = ("CA", "DE", "TX")
_WAIT = 30.0


def _new_backend() -> DuckDBBackend:
    metadata = MetaData()
    _RESULTS.to_sqlalchemy_table(metadata)
    _COUNTIES.to_sqlalchemy_table(metadata)
    backend = DuckDBBackend()
    backend.initialize(metadata)
    return backend


class _CountingL3:
    """the L3 backend, counting the statements a snapshot sends it."""

    def __init__(self, inner: SqlL3Backend) -> None:
        self._inner = inner
        self.statements = 0

    async def fetch(self, sql: str, *args: Any) -> Any:
        self.statements += 1
        return await self._inner.fetch(sql, *args)

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self.statements += 1
        return await self._inner.fetchrow(sql, *args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.statements += 1
        return await self._inner.fetchval(sql, *args)


@dataclass
class _Platform:
    """one L3 schema, one hub client that declares the buckets, and the per-scope epochs L3 records."""

    pool: asyncpg.Pool
    hub: NatsClient
    namespace: str
    nats_url: str
    epochs: dict[str, int] = field(default_factory=dict)
    retired: list[str] = field(default_factory=list)
    writing: bool = False
    clients: list[NatsClient] = field(default_factory=list)

    async def current_epochs(self) -> Mapping[str, int]:
        return dict(self.epochs)

    async def settled(self) -> int | Unsettled:
        return Unsettled("a write is in progress") if self.writing else sum(self.epochs.values())

    async def declare(self) -> None:
        await self.hub.ensure_object_store(name="pod-objects", max_bytes=64 * 1024 * 1024)
        await self.hub.ensure_kv_bucket(name="pod-pointers", owns_bucket=True)

    async def retire(self, names: list[str]) -> None:
        store = await self.hub.object_store(name="pod-objects")
        for name in names:
            await store.delete(name)
        self.retired.extend(names)

    async def replica(self, *, l3: _CountingL3 | None = None) -> tuple[ScopedSnapshot, _CountingL3]:
        client = await NatsClient.connect(
            nats_url=self.nats_url, nats_subject_namespace=self.namespace, client_name=f"pod-{len(self.clients)}"
        )
        self.clients.append(client)
        counting = l3 or _CountingL3(SqlL3Backend(self.pool))
        snapshot = ScopedSnapshot(
            name="enr",
            tables=_TABLES,
            backend=_new_backend(),
            store=await client.object_store(name="pod-objects"),
            pointers=await client.kv_bucket(name="pod-pointers", create_if_missing=False),
            l3=counting,
            epochs=self.current_epochs,
            settled=self.settled,
            ensure_buckets=self.declare,
            retire=self.retire,
            pointer_watch_heartbeat=timedelta(seconds=0.5),
        )
        return snapshot, counting


@pytest.fixture
async def platform(db_container: str, nats_container: str) -> AsyncIterator[_Platform]:
    schema = f"snap_{uuid.uuid4().hex[:8]}"
    namespace = f"snap{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    hub = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="hub")
    platform = _Platform(pool=pool, hub=hub, namespace=namespace, nats_url=nats_container)
    try:
        await pool.execute(
            "CREATE TABLE results (race TEXT, county TEXT, state TEXT, votes BIGINT, PRIMARY KEY (race, county))"
        )
        await pool.execute("CREATE TABLE counties (county TEXT PRIMARY KEY, state TEXT NOT NULL, total BIGINT)")
        for state in _STATES:
            for index in range(1500 if state == "TX" else 40):
                county = f"{state}-{index:04d}"
                await pool.execute("INSERT INTO counties VALUES ($1, $2, $3)", county, state, 1000)
                await pool.execute("INSERT INTO results VALUES ($1, $2, $3, $4)", f"{state}-gov", county, state, 1000)
            platform.epochs[state] = 1
        await platform.declare()
        yield platform
    finally:
        for client in platform.clients:
            await client.shutdown(drain_timeout=timedelta(seconds=1))
        await hub.shutdown(drain_timeout=timedelta(seconds=1))
        await pool.close()


async def _rows(pool: asyncpg.Pool, table: str, state: str) -> list[dict[str, Any]]:
    return [dict(r) for r in await pool.fetch(f"SELECT * FROM {table} WHERE state = $1", state)]  # noqa: S608


def _count(snapshot: ScopedSnapshot, table: str, state: str | None = None) -> int:
    with snapshot.read() as cursor:
        if state is None:
            return int(cursor.execute(f"SELECT count(*) FROM {table}").fetchone()[0])  # noqa: S608
        return int(cursor.execute(f"SELECT count(*) FROM {table} WHERE state = ?", [state]).fetchone()[0])  # noqa: S608


async def test_the_first_replica_builds_from_l3_and_the_next_loads_from_l2_alone(platform: _Platform) -> None:
    first, _ = await platform.replica()
    await first.start()
    await first.wait_ready(timeout=_WAIT)
    status = first.status()
    assert status.phase is SnapshotPhase.READY
    assert status.source is SnapshotSource.L3, "nothing was in L2, so the first copy had to come from L3"
    assert status.scopes_done == status.scopes_total == len(_STATES)
    assert _count(first, "results") == 1580 and _count(first, "counties") == 1580

    second, l3 = await platform.replica()
    await second.start()
    await second.wait_ready(timeout=_WAIT)
    assert second.status().source is SnapshotSource.L2
    assert l3.statements == 0, "a replica loading from L2 must not read L3"
    assert _count(second, "results", "TX") == 1500 and _count(second, "counties") == 1580
    assert second.status().rows == {"results": 1580, "counties": 1580}
    assert second.status().timings["load"] > 0
    await first.stop()
    await second.stop()


async def test_a_published_scope_reaches_the_other_replica_alone_and_whole(platform: _Platform) -> None:
    writer, _ = await platform.replica()
    await writer.start()
    await writer.wait_ready(timeout=_WAIT)
    reader, _ = await platform.replica()
    await reader.start()
    await reader.wait_ready(timeout=_WAIT)

    # a reader keeps checking one invariant that spans both tables: every county of a state has its
    # result. a scope applied in two steps would break it between them.
    stop = threading.Event()
    torn: list[tuple[int, int]] = []
    checks = [0]

    def keep_reading() -> None:
        while not stop.is_set():
            with reader.read() as cursor:
                counties = cursor.execute("SELECT count(*) FROM counties WHERE state = 'TX'").fetchone()[0]
                results = cursor.execute("SELECT count(*) FROM results WHERE state = 'TX'").fetchone()[0]
            checks[0] += 1
            if counties != results:
                torn.append((counties, results))

    thread = threading.Thread(target=keep_reading)
    thread.start()
    try:
        # the writer's change to TX: 300 counties gone, the rest re-counted
        async with platform.pool.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM results WHERE state = 'TX' AND county >= 'TX-1200'")
            await conn.execute("DELETE FROM counties WHERE state = 'TX' AND county >= 'TX-1200'")
            await conn.execute("UPDATE results SET votes = 7 WHERE state = 'TX'")
        platform.epochs["TX"] = 2
        rows = {table.name: await _rows(platform.pool, table.name, "TX") for table in _TABLES}
        started = asyncio.get_running_loop().time()
        await writer.publish("TX", 2, rows)
        while reader.status().last_change is None or reader.status().last_change.epoch != 2:
            assert asyncio.get_running_loop().time() - started < 10, "the reader never applied TX"
            await asyncio.sleep(0.02)
    finally:
        stop.set()
        thread.join()
    assert torn == [], f"a reader saw half of TX's update: {torn[:5]}"
    assert checks[0] > 10
    with reader.read() as cursor:
        assert cursor.execute("SELECT DISTINCT votes FROM results WHERE state = 'TX'").fetchall() == [(7,)]
    assert _count(reader, "results", "CA") == 40, "another scope was touched"
    assert reader.status().last_change is not None and reader.status().last_change.scope == "TX"
    assert any("/TX/1/" in name for name in platform.retired), "the superseded epoch was not retired"
    assert not any("/TX/2/" in name or "/CA/" in name for name in platform.retired)
    await writer.stop()
    await reader.stop()


async def test_a_snapshot_nats_lost_is_rebuilt_from_l3(platform: _Platform) -> None:
    replica, l3 = await platform.replica()
    await replica.start()
    await replica.wait_ready(timeout=_WAIT)
    before = l3.statements

    # what a NATS restart does to memory storage: both buckets gone; the hub declares them again
    js = platform.hub.jetstream_context()
    await js.delete_stream(f"OBJ_{platform.namespace}-pod-objects")
    await js.delete_stream(f"KV_{platform.namespace}-pod-pointers")
    await platform.hub.reconnect()

    deadline = asyncio.get_running_loop().time() + _WAIT
    seen_rebuild = False
    while True:
        status = replica.status()
        seen_rebuild = seen_rebuild or status.phase is SnapshotPhase.REBUILDING_FROM_L3
        if status.phase is SnapshotPhase.READY and l3.statements > before:
            break
        assert asyncio.get_running_loop().time() < deadline, f"never rebuilt: {status}"
        await asyncio.sleep(0.05)
    assert replica.status().source is SnapshotSource.L3
    assert _count(replica, "results") == 1580, "the local copy kept serving through the loss"

    fresh, fresh_l3 = await platform.replica()
    await fresh.start()
    await fresh.wait_ready(timeout=_WAIT)
    assert fresh.status().source is SnapshotSource.L2
    assert fresh_l3.statements == 0
    assert seen_rebuild or SnapshotPhase.REBUILDING_FROM_L3 in replica.status().history
    await replica.stop()
    await fresh.stop()


async def test_a_scope_l3_moved_without_a_publish_is_caught_up(platform: _Platform) -> None:
    replica, _ = await platform.replica()
    await replica.start()
    await replica.wait_ready(timeout=_WAIT)
    # a writer committed DE and died before publishing it
    await platform.pool.execute("UPDATE results SET votes = 42 WHERE state = 'DE'")
    platform.epochs["DE"] = 5
    caught = await replica.catch_up_from_l3()
    assert caught == ["DE"]
    with replica.read() as cursor:
        assert cursor.execute("SELECT DISTINCT votes FROM results WHERE state = 'DE'").fetchall() == [(42,)]
    other, _ = await platform.replica()
    await other.start()
    await other.wait_ready(timeout=_WAIT)
    with other.read() as cursor:
        assert cursor.execute("SELECT DISTINCT votes FROM results WHERE state = 'DE'").fetchall() == [(42,)]
    assert await replica.catch_up_from_l3() == [], "nothing is behind any more"
    await replica.stop()
    await other.stop()


async def test_a_rebuild_waits_while_a_write_is_in_progress(platform: _Platform) -> None:
    platform.writing = True
    replica, _ = await platform.replica()
    await replica.start()
    await asyncio.sleep(1.0)
    assert replica.status().phase is not SnapshotPhase.READY
    assert "write" in replica.status().detail
    platform.writing = False
    await replica.wait_ready(timeout=_WAIT)
    assert _count(replica, "results") == 1580
    await replica.stop()
