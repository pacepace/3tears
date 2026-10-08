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
import json
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
    """the L3 backend, counting the statements a snapshot sends it, each slowed by ``delay``."""

    def __init__(self, inner: SqlL3Backend, *, delay: float = 0.0) -> None:
        self._inner = inner
        self.statements = 0
        self.delay = delay

    async def fetch(self, sql: str, *args: Any) -> Any:
        self.statements += 1
        if self.delay:
            await asyncio.sleep(self.delay)
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
    retire_batches: list[int] = field(default_factory=list)
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
        self.retire_batches.append(len(names))

    async def replica(
        self,
        *,
        l3: _CountingL3 | None = None,
        backend: DuckDBBackend | None = None,
        tables: tuple[SnapshotTable, ...] = _TABLES,
        wrap_store: Any = None,
        wrap_pointers: Any = None,
        **options: Any,
    ) -> tuple[ScopedSnapshot, _CountingL3]:
        client = await NatsClient.connect(
            nats_url=self.nats_url, nats_subject_namespace=self.namespace, client_name=f"pod-{len(self.clients)}"
        )
        self.clients.append(client)
        counting = l3 or _CountingL3(SqlL3Backend(self.pool))
        snapshot = ScopedSnapshot(
            name="enr",
            tables=tables,
            backend=backend or _new_backend(),
            store=(wrap_store or (lambda s: s))(await client.object_store(name="pod-objects")),
            pointers=(wrap_pointers or (lambda b: b))(
                await client.kv_bucket(name="pod-pointers", create_if_missing=False)
            ),
            l3=counting,
            epochs=self.current_epochs,
            settled=self.settled,
            ensure_buckets=self.declare,
            retire=self.retire,
            pointer_watch_heartbeat=timedelta(seconds=0.5),
            **options,
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


async def _until(check: Any, *, what: str, timeout: float = _WAIT) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not check():
        assert asyncio.get_running_loop().time() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.05)


async def test_a_pointer_never_moves_to_a_lower_epoch(platform: _Platform) -> None:
    writer, _ = await platform.replica()
    await writer.start()
    await writer.wait_ready(timeout=_WAIT)
    reader, _ = await platform.replica()
    await reader.start()
    await reader.wait_ready(timeout=_WAIT)
    fresh = {t.name: await _rows(platform.pool, t.name, "DE") for t in _TABLES}
    await writer.publish("DE", 3, fresh)
    await _until(
        lambda: reader.status().last_change is not None and reader.status().last_change.epoch == 3, what="DE@3"
    )
    # a stale writer, still holding epoch 2, publishes rows that are not the scope's any more
    stale = {name: [dict(r, votes=-1) if name == "results" else r for r in rows] for name, rows in fresh.items()}
    stale_writer, _ = await platform.replica()
    await stale_writer.start()
    await stale_writer.wait_ready(timeout=_WAIT)
    await stale_writer.publish("DE", 2, stale)
    await asyncio.sleep(1.0)
    for replica in (writer, reader, stale_writer):
        with replica.read() as cursor:
            votes = cursor.execute("SELECT DISTINCT votes FROM results WHERE state = 'DE'").fetchall()
        assert votes == [(1000,)], "a stale publish moved a scope backwards"
    for replica in (writer, reader, stale_writer):
        await replica.stop()


async def test_concurrent_publishes_of_new_scopes_both_land_in_the_index(platform: _Platform) -> None:
    one, _ = await platform.replica()
    await one.start()
    await one.wait_ready(timeout=_WAIT)
    two, _ = await platform.replica()
    await two.start()
    await two.wait_ready(timeout=_WAIT)
    for state in ("WY", "VT"):
        await platform.pool.execute("INSERT INTO counties VALUES ($1, $2, 5)", f"{state}-0001", state)
        await platform.pool.execute(
            "INSERT INTO results VALUES ($1, $2, $3, 5)", f"{state}-gov", f"{state}-0001", state
        )
    rows = {s: {t.name: await _rows(platform.pool, t.name, s) for t in _TABLES} for s in ("WY", "VT")}
    await asyncio.gather(one.publish("WY", 1, rows["WY"]), two.publish("VT", 1, rows["VT"]))
    third, _ = await platform.replica()
    await third.start()
    await third.wait_ready(timeout=_WAIT)
    assert _count(third, "results", "WY") == 1 and _count(third, "results", "VT") == 1
    for replica in (one, two, third):
        await replica.stop()


async def test_a_schema_change_names_new_chunks(platform: _Platform) -> None:
    old, _ = await platform.replica()
    await old.start()
    await old.wait_ready(timeout=_WAIT)
    await old.stop()
    narrow_meta = MetaData()
    _RESULTS.to_sqlalchemy_table(narrow_meta)
    TableSchema(
        name="counties",
        primary_key="county",
        columns=[Column("county", STRING_TYPE), Column("state", STRING_TYPE)],
        on_conflict="update",
    ).to_sqlalchemy_table(narrow_meta)
    narrow = DuckDBBackend()
    narrow.initialize(narrow_meta)
    new, _ = await platform.replica(backend=narrow)
    await new.start()
    await new.wait_ready(timeout=_WAIT)
    assert new.status().source is SnapshotSource.L3, "chunks of other columns were loaded"
    names = [info.name for info in await (await platform.hub.object_store(name="pod-objects")).list_objects()]
    counties = {n for n in names if "/counties." in n}
    digests = {n.rsplit(".", 1)[1] for n in counties}
    assert narrow.schema_digest("counties") in digests
    assert _count(new, "counties") == 1580
    await new.stop()


async def test_a_null_scope_is_refused(platform: _Platform) -> None:
    replica, _ = await platform.replica()
    with pytest.raises(ValueError):
        await replica.publish(None, 1, {t.name: [] for t in _TABLES})  # type: ignore[arg-type]
    await platform.pool.execute("INSERT INTO counties VALUES ('X-1', 'XX', 1)")
    await platform.pool.execute("INSERT INTO results VALUES ('x', 'X-1', NULL, 1)")
    await replica.start()
    await _until(lambda: replica.status().phase is SnapshotPhase.FAILED, what="the refusal")
    assert "no scope" in replica.status().detail and "results" in replica.status().detail
    await replica.stop()


async def test_two_replicas_starting_together_read_l3_once(platform: _Platform) -> None:
    slow_one = _CountingL3(SqlL3Backend(platform.pool), delay=0.2)
    slow_two = _CountingL3(SqlL3Backend(platform.pool), delay=0.2)
    one, _ = await platform.replica(l3=slow_one, claim_ttl=timedelta(seconds=1))
    two, _ = await platform.replica(l3=slow_two, claim_ttl=timedelta(seconds=1))
    await asyncio.gather(one.start(), two.start())
    await asyncio.gather(one.wait_ready(timeout=60), two.wait_ready(timeout=60))
    # the claim outlived its one-second lifetime because its holder renewed it
    assert sorted([slow_one.statements > 0, slow_two.statements > 0]) == [False, True], (
        slow_one.statements,
        slow_two.statements,
    )
    assert {one.status().source, two.status().source} == {SnapshotSource.L2, SnapshotSource.L3}
    await one.stop()
    await two.stop()


class _HeldStore:
    """the Object Store, holding the first read of a chunk until ``release`` is set."""

    def __init__(self, inner: Any, *, hold: str) -> None:
        self._inner = inner
        self._hold = hold
        self.release = asyncio.Event()
        self.held = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def get(self, name: str) -> bytes:
        if self._hold in name and not self.release.is_set():
            self.held.set()
            await self.release.wait()
        return await self._inner.get(name)


async def test_a_chunk_retired_under_a_read_is_read_at_the_current_pointer(platform: _Platform) -> None:
    writer, _ = await platform.replica()
    await writer.start()
    await writer.wait_ready(timeout=_WAIT)
    holders: list[_HeldStore] = []

    def wrap(store: Any) -> _HeldStore:
        holders.append(_HeldStore(store, hold="/TX/1/"))
        return holders[0]

    reader, _ = await platform.replica(wrap_store=wrap)
    await reader.start()
    await asyncio.wait_for(holders[0].held.wait(), _WAIT)
    await platform.pool.execute("UPDATE results SET votes = 9 WHERE state = 'TX'")
    platform.epochs["TX"] = 2
    await writer.publish("TX", 2, {t.name: await _rows(platform.pool, t.name, "TX") for t in _TABLES})
    holders[0].release.set()
    await reader.wait_ready(timeout=_WAIT)
    with reader.read() as cursor:
        assert cursor.execute("SELECT DISTINCT votes FROM results WHERE state = 'TX'").fetchall() == [(9,)]
    await writer.stop()
    await reader.stop()


async def test_a_scope_gone_from_l3_leaves_every_replica(platform: _Platform) -> None:
    writer, _ = await platform.replica()
    await writer.start()
    await writer.wait_ready(timeout=_WAIT)
    reader, _ = await platform.replica()
    await reader.start()
    await reader.wait_ready(timeout=_WAIT)
    served = await _de_chunks(platform)
    await platform.pool.execute("DELETE FROM results WHERE state = 'DE'")
    await platform.pool.execute("DELETE FROM counties WHERE state = 'DE'")
    await writer.catch_up_from_l3()
    await _until(lambda: _count(reader, "results", "DE") == 0, what="DE to leave the reader")
    assert served and served <= set(platform.retired), "the removal left the chunks its pointer served"
    assert not await _de_chunks(platform), "a removed scope's chunks stayed in the bounded store"
    assert _count(reader, "results", "CA") == 40
    # a removal is a change the reader applies, not a lost snapshot: it stays READY and reads no L3
    await _until(lambda: reader.status().phase is SnapshotPhase.READY, what="the reader to be READY", timeout=5)
    assert SnapshotPhase.REBUILDING_FROM_L3 not in reader.status().history, reader.status().history
    assert SnapshotPhase.WAITING not in reader.status().history, reader.status().history
    await writer.stop()
    await reader.stop()


async def test_retirement_asks_in_bounded_batches(platform: _Platform) -> None:
    replica, _ = await platform.replica(retire_batch=2)
    await replica.start()
    await replica.wait_ready(timeout=_WAIT)
    await replica.publish("TX", 7, {t.name: await _rows(platform.pool, t.name, "TX") for t in _TABLES})
    await replica.publish("CA", 7, {t.name: await _rows(platform.pool, t.name, "CA") for t in _TABLES})
    assert platform.retire_batches and max(platform.retire_batches) <= 2
    await replica.stop()


async def test_a_watch_that_ends_says_so(platform: _Platform) -> None:
    replica, _ = await platform.replica()
    await replica.start()
    await replica.wait_ready(timeout=_WAIT)
    await platform.clients[-1].shutdown(drain_timeout=timedelta(seconds=1))
    await _until(lambda: replica.status().phase is SnapshotPhase.FAILED, what="the watch's end")
    assert "watch" in replica.status().detail

    await replica.stop()


class _HeldPuts:
    """the Object Store, holding every chunk write until ``release`` is set: a slowed publish loop."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.release = asyncio.Event()
        self.held = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def put(self, name: str, data: bytes) -> Any:
        if not self.release.is_set():
            self.held.set()
            await self.release.wait()
        return await self._inner.put(name, data)


async def _add_scope(platform: _Platform, state: str, epoch: int) -> dict[str, list[dict[str, Any]]]:
    """commit a new scope in L3, as a writer does before it publishes, and return its rows."""
    await platform.pool.execute("INSERT INTO counties VALUES ($1, $2, 5)", f"{state}-0001", state)
    await platform.pool.execute("INSERT INTO results VALUES ($1, $2, $3, 5)", f"{state}-gov", f"{state}-0001", state)
    platform.epochs[state] = epoch
    return {t.name: await _rows(platform.pool, t.name, state) for t in _TABLES}


async def test_a_scope_published_during_a_rebuild_survives_it_on_every_replica(platform: _Platform) -> None:
    held: list[_HeldPuts] = []

    def wrap(store: Any) -> _HeldPuts:
        held.append(_HeldPuts(store))
        return held[0]

    rebuilder, _ = await platform.replica(wrap_store=wrap)
    await rebuilder.start()
    # the rebuild read L3 (CA, DE, TX) and settled; its publish loop is held at its first chunk
    await asyncio.wait_for(held[0].held.wait(), _WAIT)
    # a replica starting meanwhile waits on the rebuild's claim
    waiting, _ = await platform.replica()
    await waiting.start()
    # a writer commits a new scope and publishes it while the loop is held
    writer, _ = await platform.replica()
    await writer.publish("WY", 2, await _add_scope(platform, "WY", 2))
    held[0].release.set()
    await rebuilder.wait_ready(timeout=_WAIT)
    await waiting.wait_ready(timeout=_WAIT)
    fresh, _ = await platform.replica()
    await fresh.start()
    await fresh.wait_ready(timeout=_WAIT)
    # long enough for a removal the rebuild made to reach every replica through its watch
    await asyncio.sleep(1.5)
    for replica in (rebuilder, waiting, fresh):
        assert _count(replica, "results", "WY") == 1, "a scope published during the rebuild was erased"
        assert _count(replica, "results") == 1581
    pointers = await platform.hub.kv_bucket(name="pod-pointers", create_if_missing=False)
    assert await pointers.get(key="enr.s.WY") is not None, "the scope's pointer was deleted"
    for replica in (rebuilder, waiting, fresh):
        await replica.stop()


async def test_a_cold_rebuild_is_ready_only_with_the_scope_a_writer_moved_past_it(platform: _Platform) -> None:
    held: list[_HeldPuts] = []

    def wrap(store: Any) -> _HeldPuts:
        held.append(_HeldPuts(store))
        return held[0]

    rebuilder, _ = await platform.replica(wrap_store=wrap)
    await rebuilder.start()
    # the rebuild read TX at epoch 1; its publish loop is held at its first chunk (CA)
    await asyncio.wait_for(held[0].held.wait(), _WAIT)
    await platform.pool.execute("UPDATE results SET votes = 9 WHERE state = 'TX'")
    platform.epochs["TX"] = 2
    writer, _ = await platform.replica()
    await writer.publish("TX", 2, {t.name: await _rows(platform.pool, t.name, "TX") for t in _TABLES})
    held[0].release.set()
    await rebuilder.wait_ready(timeout=_WAIT)
    # at once, before any later pass could apply TX from the watch
    assert _count(rebuilder, "results", "TX") == 1500, "ready without the scope the rebuild refused as stale"
    with rebuilder.read() as cursor:
        assert cursor.execute("SELECT DISTINCT votes FROM results WHERE state = 'TX'").fetchall() == [(9,)]
    await rebuilder.stop()


async def test_a_scope_published_before_any_rebuild_is_not_loaded_as_the_whole_snapshot(platform: _Platform) -> None:
    # NATS holds nothing; a writer publishes one scope before any replica rebuilt the snapshot
    writer, _ = await platform.replica()
    await writer.publish("DE", 2, {t.name: await _rows(platform.pool, t.name, "DE") for t in _TABLES})
    replica, _ = await platform.replica()
    await replica.start()
    await replica.wait_ready(timeout=_WAIT)
    assert replica.status().source is SnapshotSource.L3, "an index naming one scope was loaded as the snapshot"
    assert _count(replica, "results") == 1580
    await replica.stop()


class _HeldL3(_CountingL3):
    """the L3 backend, once armed holding the answer of its ``hold_at``-th scope listing until ``release``.

    The answer is read before the hold, so what it says is from before anything done while held.
    """

    def __init__(self, inner: SqlL3Backend, *, hold_at: int) -> None:
        super().__init__(inner)
        self._hold_at = hold_at
        self._listings: int | None = None
        self.held = asyncio.Event()
        self.release = asyncio.Event()

    def arm(self) -> None:
        self._listings = 0

    async def fetch(self, sql: str, *args: Any) -> Any:
        result = await super().fetch(sql, *args)
        if self._listings is not None and "DISTINCT" in sql:
            self._listings += 1
            if self._listings == self._hold_at:
                self.held.set()
                await self.release.wait()
        return result


class _HeldPointers:
    """the pointer bucket, once armed holding after it deletes ``key``, then before it next reads ``key``."""

    def __init__(self, inner: Any, *, key: str) -> None:
        self._inner = inner
        self._key = key
        self._armed = False
        self.deleted = asyncio.Event()
        self.release_delete = asyncio.Event()
        self.reading = asyncio.Event()
        self.release_read = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def arm(self) -> None:
        self._armed = True

    async def delete(self, *, key: str, revision: int | None = None) -> bool:
        result = await self._inner.delete(key=key, revision=revision)
        if self._armed and key == self._key and not self.deleted.is_set():
            self.deleted.set()
            await self.release_delete.wait()
        return bool(result)

    async def get(self, *, key: str) -> bytes | None:
        if self._armed and key == self._key and self.deleted.is_set() and not self.reading.is_set():
            self.reading.set()
            await self.release_read.wait()
        result: bytes | None = await self._inner.get(key=key)
        return result


class _HeldAfterPut:
    """the Object Store, holding once after it wrote a chunk whose name holds ``hold``."""

    def __init__(self, inner: Any, *, hold: str) -> None:
        self._inner = inner
        self._hold = hold
        self.held = asyncio.Event()
        self.release = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def put(self, name: str, data: bytes) -> Any:
        result = await self._inner.put(name, data)
        if self._hold in name and not self.held.is_set():
            self.held.set()
            await self.release.wait()
        return result


async def _delete_de(platform: _Platform) -> dict[str, list[dict[str, Any]]]:
    """take DE out of L3, returning its rows to put back."""
    rows = {t.name: await _rows(platform.pool, t.name, "DE") for t in _TABLES}
    await platform.pool.execute("DELETE FROM results WHERE state = 'DE'")
    await platform.pool.execute("DELETE FROM counties WHERE state = 'DE'")
    return rows


async def _restore_de(platform: _Platform, rows: Mapping[str, list[dict[str, Any]]], epoch: int) -> None:
    """commit DE in L3 again at ``epoch``, as a writer recreating it does before it publishes."""
    for row in rows["counties"]:
        await platform.pool.execute(
            "INSERT INTO counties VALUES ($1, $2, $3)", row["county"], row["state"], row["total"]
        )
    for row in rows["results"]:
        await platform.pool.execute(
            "INSERT INTO results VALUES ($1, $2, $3, $4)", row["race"], row["county"], row["state"], row["votes"]
        )
    platform.epochs["DE"] = epoch


async def _assert_de_stands_at(platform: _Platform, epoch: int, replicas: list[ScopedSnapshot]) -> None:
    """DE's pointer at ``epoch``, the index naming DE, its rows on every replica and a fresh one, its chunks kept."""
    pointers = await platform.hub.kv_bucket(name="pod-pointers", create_if_missing=False)
    raw = await pointers.get(key="enr.s.DE")
    assert raw is not None, "DE's pointer was deleted after a writer recreated it"
    assert json.loads(raw)["epoch"] == epoch
    index = await pointers.get(key="enr.index")
    assert index is not None and "DE" in json.loads(index)["scopes"], "the index lost DE"
    assert not any(f"/DE/{epoch}/" in name for name in platform.retired), "DE's recreated chunks were retired"
    fresh, _ = await platform.replica()
    await fresh.start()
    await fresh.wait_ready(timeout=_WAIT)
    for replica in [*replicas, fresh]:
        await _until(lambda r=replica: _count(r, "results", "DE") == 40, what="DE's rows on every replica")
    await fresh.stop()


async def test_a_removal_keeps_a_scope_whose_pointer_moved_after_it_read_l3(platform: _Platform) -> None:
    held = _HeldL3(SqlL3Backend(platform.pool), hold_at=4)
    remover, _ = await platform.replica(l3=held)
    await remover.start()
    await remover.wait_ready(timeout=_WAIT)
    reader, _ = await platform.replica()
    await reader.start()
    await reader.wait_ready(timeout=_WAIT)
    rows = await _delete_de(platform)
    held.arm()
    removal = asyncio.create_task(remover.catch_up_from_l3())
    # the removal read L3 again (no DE) and holds before it deletes DE's pointer
    await asyncio.wait_for(held.held.wait(), _WAIT)
    await _restore_de(platform, rows, 5)
    writer, _ = await platform.replica()
    await writer.publish("DE", 5, {t.name: await _rows(platform.pool, t.name, "DE") for t in _TABLES})
    held.release.set()
    await removal
    await _assert_de_stands_at(platform, 5, [remover, reader])
    for replica in (remover, reader):
        await replica.stop()


async def test_a_removal_retires_only_the_epochs_its_deleted_pointer_named(platform: _Platform) -> None:
    held = _HeldL3(SqlL3Backend(platform.pool), hold_at=4)
    remover, _ = await platform.replica(l3=held)
    await remover.start()
    await remover.wait_ready(timeout=_WAIT)
    reader, _ = await platform.replica()
    await reader.start()
    await reader.wait_ready(timeout=_WAIT)
    rows = await _delete_de(platform)
    held.arm()
    removal = asyncio.create_task(remover.catch_up_from_l3())
    await asyncio.wait_for(held.held.wait(), _WAIT)
    # a writer recreating DE has written its chunks at epoch 5 and not yet moved the pointer
    await _restore_de(platform, rows, 5)
    stores: list[_HeldAfterPut] = []

    def wrap(store: Any) -> _HeldAfterPut:
        stores.append(_HeldAfterPut(store, hold="/DE/5/counties."))
        return stores[0]

    writer, _ = await platform.replica(wrap_store=wrap)
    publish = asyncio.create_task(
        writer.publish("DE", 5, {t.name: await _rows(platform.pool, t.name, "DE") for t in _TABLES})
    )
    await asyncio.wait_for(stores[0].held.wait(), _WAIT)
    # the removal deletes the pointer it read (epoch 1) and finishes
    held.release.set()
    await removal
    stores[0].release.set()
    await publish
    await _assert_de_stands_at(platform, 5, [remover, reader])
    left = await _de_chunks(platform)
    assert left and all("/DE/5/" in name for name in left), f"the removal left epoch 1, or took epoch 5: {left}"
    assert any("/DE/1/" in name for name in platform.retired), "the removal retired nothing it served"
    for replica in (remover, reader):
        await replica.stop()


async def test_a_scope_recreated_between_its_pointer_delete_and_its_index_removal_stays(platform: _Platform) -> None:
    wrapped: list[_HeldPointers] = []

    def wrap(bucket: Any) -> _HeldPointers:
        wrapped.append(_HeldPointers(bucket, key="enr.s.DE"))
        return wrapped[0]

    remover, _ = await platform.replica(wrap_pointers=wrap)
    await remover.start()
    await remover.wait_ready(timeout=_WAIT)
    reader, _ = await platform.replica()
    await reader.start()
    await reader.wait_ready(timeout=_WAIT)
    rows = await _delete_de(platform)
    held = wrapped[0]
    held.arm()
    removal = asyncio.create_task(remover.catch_up_from_l3())
    # DE's pointer is deleted; the index still names DE
    await asyncio.wait_for(held.deleted.wait(), _WAIT)
    await _restore_de(platform, rows, 5)
    writer, _ = await platform.replica()
    await writer.publish("DE", 5, {t.name: await _rows(platform.pool, t.name, "DE") for t in _TABLES})
    held.release_delete.set()
    # the removal took DE out of the index and holds before it looks for a pointer standing again
    await asyncio.wait_for(held.reading.wait(), _WAIT)
    pointers = await platform.hub.kv_bucket(name="pod-pointers", create_if_missing=False)
    index = await pointers.get(key="enr.index")
    assert index is not None and "DE" not in json.loads(index)["scopes"]
    # long enough for the reader to apply that index: DE's pointer stands, so the reader keeps DE
    await asyncio.sleep(1.5)
    assert _count(reader, "results", "DE") == 40, "the reader dropped a scope whose pointer stands"
    held.release_read.set()
    await removal
    await _assert_de_stands_at(platform, 5, [remover, reader])
    for replica in (remover, reader):
        await replica.stop()


async def test_a_replica_that_waited_on_a_rebuild_is_ready_again(platform: _Platform) -> None:
    replica, l3 = await platform.replica()
    await replica.start()
    await replica.wait_ready(timeout=_WAIT)
    statements = l3.statements
    pointers = await platform.hub.kv_bucket(name="pod-pointers", create_if_missing=False)
    whole = await pointers.get_entry(key="enr.index")
    assert whole is not None
    # another replica holds the rebuild claim, and the index says it names only scopes published
    # since NATS lost the snapshot: the ready replica must wait on that rebuild
    claim = await pointers.create(key="enr.rebuild", value=b"another-replica")
    assert claim is not None
    partial = json.dumps(json.loads(whole[0]) | {"partial": True}).encode("utf-8")
    revision = await pointers.update(key="enr.index", value=partial, revision=whole[1])
    assert revision is not None
    await _until(lambda: replica.status().phase is SnapshotPhase.WAITING, what="the replica to wait")
    # that rebuild finishes: the index is whole again and the claim is released. the replica's next
    # pass applies what the pointers say, and that pass did not wait
    assert await pointers.update(key="enr.index", value=whole[0], revision=revision) is not None
    assert await pointers.delete(key="enr.rebuild", revision=claim)
    await _until(lambda: replica.status().phase is SnapshotPhase.READY, what="the replica READY again")
    assert l3.statements == statements, "the replica rebuilt for itself rather than apply the pointers"
    assert _count(replica, "results") == 1580
    await replica.stop()


async def test_a_staged_scope_shows_nowhere_until_its_pointer_moves_and_then_everywhere(platform: _Platform) -> None:
    writer, _ = await platform.replica()
    await writer.start()
    await writer.wait_ready(timeout=_WAIT)
    reader, _ = await platform.replica()
    await reader.start()
    await reader.wait_ready(timeout=_WAIT)
    changes: list[int] = []
    reader.on_change(lambda: changes.append(1))

    # the writer changes only results in DE; counties are carried from DE's current chunk
    await platform.pool.execute("UPDATE results SET votes = 9 WHERE state = 'DE'")
    staged = await writer.stage("DE", 2, {"results": await _rows(platform.pool, "results", "DE")})
    await asyncio.sleep(0.5)
    for replica in (writer, reader):
        with replica.read() as cursor:
            shown = cursor.execute("SELECT DISTINCT votes FROM results WHERE state = 'DE'").fetchall()
        assert shown == [(1000,)], "a staged chunk was shown before its pointer moved"

    platform.epochs["DE"] = 2
    moved, skipped = await writer.publish_staged([staged], carry_at={"DE": 1})

    assert (moved, skipped) == (["DE"], [])
    for replica in (writer, reader):
        await _until(
            lambda r=replica: r.status().last_change is not None and r.status().last_change.epoch == 2,
            what="DE applied",
        )
        with replica.read() as cursor:
            assert cursor.execute("SELECT DISTINCT votes FROM results WHERE state = 'DE'").fetchall() == [(9,)]
        assert _count(replica, "counties", "DE") == 40, "the carried table lost its rows"
    assert changes, "the reader's change listener was not called"
    assert not any("/DE/1/counties" in name for name in platform.retired), "a carried chunk was retired"
    assert any("/DE/1/results" in name for name in platform.retired), "the superseded chunk was kept"
    fresh, l3 = await platform.replica()
    await fresh.start()
    await fresh.wait_ready(timeout=_WAIT)
    assert l3.statements == 0 and _count(fresh, "counties", "DE") == 40
    for replica in (writer, reader, fresh):
        await replica.stop()


async def test_a_staged_scope_whose_pointer_moved_elsewhere_is_left_for_the_catch_up(platform: _Platform) -> None:
    writer, _ = await platform.replica()
    await writer.start()
    await writer.wait_ready(timeout=_WAIT)
    # a write that died after its commit moved CA to 3 in L3, and CA's pointer still says 1
    await platform.pool.execute("UPDATE counties SET total = 5 WHERE state = 'CA'")
    await platform.pool.execute("UPDATE results SET votes = 6 WHERE state = 'CA'")
    staged = await writer.stage("CA", 4, {"results": await _rows(platform.pool, "results", "CA")})
    platform.epochs["CA"] = 4

    moved, skipped = await writer.publish_staged([staged], carry_at={"CA": 3})

    assert (moved, skipped) == ([], ["CA"]), "a chunk was carried from an epoch the writer did not see"
    assert await writer.catch_up_from_l3() == ["CA"]
    with writer.read() as cursor:
        assert cursor.execute("SELECT DISTINCT total FROM counties WHERE state = 'CA'").fetchall() == [(5,)]
    await writer.stop()


async def test_a_whole_write_into_an_empty_snapshot_is_loaded_from_l2_with_no_rebuild(platform: _Platform) -> None:
    async with platform.pool.acquire() as conn:
        await conn.execute("DELETE FROM results")
        await conn.execute("DELETE FROM counties")
    platform.epochs.clear()
    # the very first load: nothing in NATS, a write in progress, every replica waiting on it
    platform.writing = True
    writer, writer_l3 = await platform.replica()
    await writer.start()
    waiting, l3 = await platform.replica()
    await waiting.start()
    await _until(lambda: "write" in waiting.status().detail, what="the replica waiting on the write")

    # both tables of DE, results alone of TX (TX has no counties yet)
    async with platform.pool.acquire() as conn:
        await conn.execute("INSERT INTO counties VALUES ('DE-1', 'DE', 3)")
        await conn.execute("INSERT INTO results VALUES ('DE-gov', 'DE-1', 'DE', 3), ('TX-gov', 'TX-1', 'TX', 4)")
    staged = [
        await writer.stage("DE", 1, {t.name: await _rows(platform.pool, t.name, "DE") for t in _TABLES}),
        await writer.stage("TX", 1, {"results": await _rows(platform.pool, "results", "TX")}),
    ]
    async with writer.holding_rebuilds() as held:
        assert held
        platform.epochs.update({"DE": 1, "TX": 1})
        platform.writing = False  # the commit
        await asyncio.sleep(0.5)  # the waiting replicas look again while the claim is held
        moved, skipped = await writer.publish_staged(staged, carry_at={}, whole=True)

    assert (sorted(moved), skipped) == (["DE", "TX"], [])
    for replica in (writer, waiting):
        await replica.wait_ready(timeout=_WAIT)
        assert replica.status().source is SnapshotSource.L2, "a replica rebuilt from L3"
        assert _count(replica, "results") == 2 and _count(replica, "counties", "TX") == 0
    assert l3.statements == 0 and writer_l3.statements == 0, "a replica read L3"
    await writer.stop()
    await waiting.stop()


async def test_a_writers_staged_chunks_survive_a_rebuilds_sweep_and_apply_everywhere(platform: _Platform) -> None:
    """a write stages its chunks long before it moves their pointer; a rebuild that sweeps chunks no
    pointer names in the meantime must not take them, or the pointer would move onto nothing."""
    writer, writer_l3 = await platform.replica()
    await writer.start()
    await writer.wait_ready(timeout=_WAIT)

    # the write: DE's results change in L3 and are staged at the write's version, its pointer unmoved
    await platform.pool.execute("UPDATE results SET votes = 9 WHERE state = 'DE'")
    staged = await writer.stage("DE", 2, {"results": await _rows(platform.pool, "results", "DE")})

    # meanwhile NATS loses the pointers (not the chunks): the writer rebuilds every scope and sweeps
    before = writer_l3.statements
    js = platform.hub.jetstream_context()
    await js.delete_stream(f"KV_{platform.namespace}-pod-pointers")
    await platform.hub.reconnect()
    await _until(
        lambda: writer_l3.statements > before and writer.status().phase is SnapshotPhase.READY,
        what="the rebuild and its sweep",
    )
    objects = await (await platform.hub.object_store(name="pod-objects")).list_objects(prefix="enr/DE/2/")
    assert {info.name for info in objects} == set(staged.objects.values()), "the sweep took a staged chunk"

    reader, _ = await platform.replica()
    await reader.start()
    await reader.wait_ready(timeout=_WAIT)
    platform.epochs["DE"] = 2  # the commit
    moved, skipped = await writer.publish_staged([staged], carry_at={"DE": 1})

    assert (moved, skipped) == (["DE"], [])
    for replica in (writer, reader):
        await _until(
            lambda r=replica: (
                r.status().last_change is not None
                and r.status().last_change.scope == "DE"
                and r.status().last_change.epoch == 2
            ),
            what="DE applied at epoch 2",
        )
        assert replica.status().behind == {}
        with replica.read() as cursor:
            assert cursor.execute("SELECT DISTINCT votes FROM results WHERE state = 'DE'").fetchall() == [(9,)]
    await writer.stop()
    await reader.stop()


async def _de_chunks(platform: _Platform) -> set[str]:
    """every chunk of DE in the store."""
    store = await platform.hub.object_store(name="pod-objects")
    return {info.name for info in await store.list_objects(prefix="enr/DE/")}


def _new_backend_with_note() -> DuckDBBackend:
    """the next code version's L1: counties gains a column."""
    metadata = MetaData()
    _RESULTS.to_sqlalchemy_table(metadata)
    TableSchema(
        name="counties",
        primary_key="county",
        columns=[
            Column("county", STRING_TYPE),
            Column("state", STRING_TYPE),
            Column("total", BIGINT_TYPE),
            Column("note", STRING_TYPE, nullable=True),
        ],
        on_conflict="update",
    ).to_sqlalchemy_table(metadata)
    backend = DuckDBBackend()
    backend.initialize(metadata)
    return backend


async def _pointer_revisions(platform: _Platform) -> dict[str, int]:
    pointers = await platform.hub.kv_bucket(name="pod-pointers", create_if_missing=False)
    revisions = {}
    for state in _STATES:
        entry = await pointers.get_entry(key=f"enr.s.{state}")
        assert entry is not None
        revisions[state] = entry[1]
    return revisions


async def test_two_code_versions_with_other_columns_never_load_or_repoint_each_others_chunks(
    platform: _Platform,
) -> None:
    """a rolling deploy of a column change: the old version and the new one run together over the
    same pointers. Neither ever holds the other's columns, neither repoints the other's scopes on a
    rebuild, and a write either one publishes reaches the other, which rebuilds that scope from L3."""
    await platform.pool.execute("ALTER TABLE counties ADD COLUMN note TEXT")
    await platform.pool.execute("UPDATE counties SET note = 'from L3'")
    old, _ = await platform.replica()
    await old.start()
    await old.wait_ready(timeout=_WAIT)
    before = await _pointer_revisions(platform)

    new, new_l3 = await platform.replica(backend=_new_backend_with_note())
    await new.start()
    await new.wait_ready(timeout=_WAIT)
    await asyncio.sleep(1.0)  # several passes of both workers

    assert new.status().source is SnapshotSource.L3, "the new version loaded the old one's chunks"
    assert new_l3.statements > 0
    with new.read() as cursor:
        assert cursor.execute("SELECT DISTINCT note FROM counties").fetchall() == [("from L3",)]
    assert "other columns" in new.status().detail
    assert await _pointer_revisions(platform) == before, "a rebuild repointed the other version's scopes"
    assert old.status().phase is SnapshotPhase.READY and "note" not in str(old.backend.column_types("counties"))

    # the old version writes DE at epoch 2: the new one cannot load those chunks and rebuilds DE from L3
    await platform.pool.execute("UPDATE results SET votes = 11 WHERE state = 'DE'")
    platform.epochs["DE"] = 2
    await old.publish("DE", 2, {t.name: await _rows(platform.pool, t.name, "DE") for t in _TABLES})

    def new_has_de() -> bool:
        with new.read() as cursor:
            return cursor.execute("SELECT DISTINCT votes FROM results WHERE state = 'DE'").fetchall() == [(11,)]

    await _until(new_has_de, what="the new version to rebuild DE from L3")
    with new.read() as cursor:
        assert cursor.execute("SELECT DISTINCT note FROM counties WHERE state = 'DE'").fetchall() == [("from L3",)]
    with old.read() as cursor:
        assert cursor.execute("SELECT DISTINCT votes FROM results WHERE state = 'DE'").fetchall() == [(11,)]
    await old.stop()
    await new.stop()
