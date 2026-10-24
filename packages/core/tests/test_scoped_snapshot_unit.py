"""the scoped snapshot's wiring and bookkeeping, over in-memory fakes of its buckets: no NATS, no L3.

What the live tests (``integration/test_scoped_snapshot_live.py``) cannot pin cheaply: that a tool
pod's snapshot is wired to the hub's asks, that the status never queries DuckDB, that a rebuild
claim survives a renewal that cannot reach NATS, and that a scope whose chunks cannot be read is
shown, retried at the recheck and not in a spin.
"""

from __future__ import annotations

import ast
import contextlib
import asyncio
import json
import linecache
import logging
import sys
import threading
import time
from types import FrameType
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

pytest.importorskip("pyarrow")

from sqlalchemy import MetaData  # noqa: E402

from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.collections.complete_copy import Unsettled  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema  # noqa: E402
from threetears.core.collections import scoped_snapshot as scoped_snapshot_module  # noqa: E402
from threetears.core.collections.scoped_snapshot import (  # noqa: E402
    ScopedSnapshot,
    SnapshotPhase,
    SnapshotSource,
    SnapshotTable,
    encode_chunk,
    open_tool_pod_snapshot,
)
from threetears.nats import ObjectNotFoundError, Subject, set_default_namespace  # noqa: E402
from threetears.nats.kv_watch import KvKeyUpdate  # noqa: E402
from threetears.nats.object_store import ObjectInfo  # noqa: E402

_RESULTS = TableSchema(
    name="results",
    primary_key=("county",),
    columns=[Column("county", STRING_TYPE), Column("state", STRING_TYPE), Column("votes", BIGINT_TYPE, nullable=True)],
    on_conflict="update",
)
_TABLES = (SnapshotTable(name="results", scope_column="state", key=("county",)),)


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("3tears")


def _backend() -> DuckDBBackend:
    metadata = MetaData()
    _RESULTS.to_sqlalchemy_table(metadata)
    backend = DuckDBBackend()
    backend.initialize(metadata)
    return backend


class _Pointers:
    """a pointer bucket in memory: CAS by revision, and a prefix watch that delivers every change to every watcher."""

    def __init__(self) -> None:
        self.entries: dict[str, tuple[bytes, int]] = {}
        self._revision = 0
        # one queue per watch, so replicas sharing the bucket each see every change
        self._queues: list[asyncio.Queue[KvKeyUpdate | None | Exception]] = []
        self.update_fails: Exception | None = None
        # the per-key TTL each key was last written with
        self.ttls: dict[str, timedelta | None] = {}
        # keys whose writer is dead: nothing it sends reaches the bucket any more
        self.dead_keys: set[str] = set()

    def end_watch(self) -> None:
        """end the pointer watch, as a lost connection does, while the bucket's reads and writes still answer."""
        self.broadcast(ConnectionError("the watch's consumer is gone"))

    def broadcast(self, update: KvKeyUpdate | Exception) -> None:
        """deliver one update, or the end of the watch, to every watcher."""
        for queue in self._queues:
            queue.put_nowait(update)

    def put_now(self, key: str, value: bytes) -> int:
        """write a key outright, as another replica's publish would."""
        return self._put(key, value)

    def _put(self, key: str, value: bytes) -> int:
        self._revision += 1
        self.entries[key] = (value, self._revision)
        self.broadcast(KvKeyUpdate(key=key, value=value, revision=self._revision))
        return self._revision

    async def get(self, *, key: str) -> bytes | None:
        await asyncio.sleep(0)  # a read yields, as a real one does, so what the loop shows between reads is seen
        entry = self.entries.get(key)
        return None if entry is None else entry[0]

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        await asyncio.sleep(0)
        return self.entries.get(key)

    async def create(self, *, key: str, value: bytes, ttl: timedelta | None = None) -> int | None:
        if key in self.dead_keys:
            raise ConnectionError(f"the writer of {key} is dead")
        if key in self.entries:
            return None
        self.ttls[key] = ttl
        return self._put(key, value)

    async def update(self, *, key: str, value: bytes, revision: int, ttl: timedelta | None = None) -> int | None:
        if self.update_fails is not None:
            raise self.update_fails
        if key in self.dead_keys:
            raise ConnectionError(f"the writer of {key} is dead")
        entry = self.entries.get(key)
        if entry is None or entry[1] != revision:
            return None
        self.ttls[key] = ttl
        return self._put(key, value)

    async def delete(self, *, key: str, revision: int | None = None) -> bool:
        entry = self.entries.get(key)
        if entry is None or (revision is not None and entry[1] != revision):
            return False
        del self.entries[key]
        self._revision += 1
        self.broadcast(KvKeyUpdate(key=key, value=None, revision=self._revision))
        return True

    async def list_keys(self, *, prefix: str = "") -> list[str]:
        await asyncio.sleep(0)
        return [key for key in self.entries if key.startswith(prefix)]

    async def watch_prefix(self, *, prefix: str, heartbeat: timedelta) -> AsyncIterator[KvKeyUpdate | None]:
        queue: asyncio.Queue[KvKeyUpdate | None | Exception] = asyncio.Queue()
        self._queues.append(queue)  # before the replay: nothing written in between is missed
        for key, (value, revision) in list(self.entries.items()):
            yield KvKeyUpdate(key=key, value=value, revision=revision)
        yield None
        while True:
            update = await queue.get()
            if isinstance(update, Exception):
                raise update
            if update is not None and update.key.startswith(prefix):
                yield update


class _Store:
    """an Object Store in memory, counting reads."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.gets = 0
        self.listings = 0

    async def put(self, name: str, data: bytes) -> None:
        self.objects[name] = data

    async def get(self, name: str) -> bytes:
        await asyncio.sleep(0)  # a read yields, as a real one does, so a spin is counted, not hung
        self.gets += 1
        if name not in self.objects:
            raise ObjectNotFoundError(f"{name} is not in the store", bucket="objects", name=name)
        return self.objects[name]

    async def list_objects(self, *, prefix: str = "") -> list[ObjectInfo]:
        self.listings += 1
        return [
            ObjectInfo(name=n, size=len(d), chunks=1, digest="", nuid="", mtime=datetime.now(UTC))
            for n, d in self.objects.items()
            if n.startswith(prefix)
        ]

    async def info(self, name: str) -> ObjectInfo | None:
        data = self.objects.get(name)
        return (
            None
            if data is None
            else ObjectInfo(name=name, size=len(data), chunks=1, digest="", nuid="", mtime=datetime.now(UTC))
        )


class _L3:
    """L3 for the snapshot's tables: a fixed set of rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.statements = 0
        self.epochs: dict[str, int] = {}
        self.writing = False

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.statements += 1
        if "DISTINCT" in sql:
            return [{"scope": s} for s in sorted({r["state"] for r in self.rows})]
        state = args[0] if args else None
        return [r for r in self.rows if state is None or r["state"] == state]

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        return None


def _snapshot(**options: Any) -> tuple[ScopedSnapshot, _Pointers, _Store, _L3]:
    pointers, store = options.pop("pointers", None) or _Pointers(), options.pop("store", None) or _Store()
    l3 = options.pop("l3", None) or _L3(
        [{"county": "c1", "state": "TX", "votes": 1}, {"county": "c2", "state": "DE", "votes": 2}]
    )

    l3.epochs = {"TX": 1, "DE": 1}

    async def epochs() -> dict[str, int]:
        return dict(l3.epochs)

    l3.writing = False

    async def settled() -> Any:
        return Unsettled("a write is in progress") if l3.writing else dict(l3.epochs)

    snapshot = ScopedSnapshot(
        name="enr",
        tables=options.pop("tables", _TABLES),
        backend=options.pop("backend", None) or _backend(),
        store=store,  # type: ignore[arg-type]
        pointers=pointers,  # type: ignore[arg-type]
        l3=l3,
        epochs=epochs,
        settled=settled,
        recheck=timedelta(seconds=0.05),
        **options,
    )
    return snapshot, pointers, store, l3


async def _until(check: Any, *, what: str, timeout: float = 5.0) -> None:
    """wait for ``check()``, looking every 10 ms, for ``timeout`` seconds of the loop's own running.

    A poll counts at most 0.1 s however long it took, so a pause of the whole process is not a wait
    that ran out: a full garbage collection on a loaded runner stops every thread for seconds, past
    this budget, while the snapshot's tasks make no progress either.
    """
    spent, last = 0.0, time.monotonic()
    while not check():
        assert spent < timeout, f"timed out waiting for {what}"
        await asyncio.sleep(0.01)
        now = time.monotonic()
        spent, last = spent + min(now - last, 0.1), now


class _NoQueries(DuckDBBackend):
    """a DuckDB L1 whose ad-hoc queries fail: a query from the status would wait on a load's lock."""

    def execute_query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        raise AssertionError(f"the status queried DuckDB: {sql}")


async def test_the_status_counts_rows_from_memory_and_never_queries_duckdb() -> None:
    metadata = MetaData()
    _RESULTS.to_sqlalchemy_table(metadata)
    backend = _NoQueries()
    backend.initialize(metadata)
    snapshot, _, _, _ = _snapshot(backend=backend)
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)

    status = snapshot.status()

    assert status.rows == {"results": 2}
    await snapshot.stop()


def _point_at_a_missing_chunk(pointers: _Pointers, scope: str = "TX") -> None:
    """move a scope's pointer to epoch 2, naming a chunk that is not in the store."""
    held = json.loads(pointers.entries[f"enr.s.{scope}"][0])
    held["epoch"], held["tables"]["results"]["object"] = 2, f"enr/{scope}/2/results.gone"
    pointers.put_now(f"enr.s.{scope}", json.dumps(held).encode())


async def _reads_in(store: _Store, seconds: float) -> int:
    gets = store.gets
    await asyncio.sleep(seconds)
    return store.gets - gets


async def test_a_scope_whose_chunks_are_gone_is_shown_behind_and_retried_at_the_recheck_not_in_a_spin() -> None:
    snapshot, pointers, store, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    pointers.put_now("enr.rebuild", b"another replica")  # it holds the claim: no rebuild here can help

    _point_at_a_missing_chunk(pointers)

    await _until(
        lambda: "TX" in snapshot.status().behind and snapshot.status().phase is SnapshotPhase.READY,
        what="TX shown behind, the replica still ready",
    )
    assert "behind" in snapshot.status().detail
    assert "enr/TX/2/results.gone" in snapshot.status().behind["TX"], "the status does not say why TX is behind"
    # a retry per recheck (0.05 s): a handful of reads, not hundreds
    reads = await _reads_in(store, 0.25)
    assert reads < 30, f"{reads} reads in 0.25 s: the failed scope is retried in a spin"
    await snapshot.stop()


async def test_a_scope_whose_chunks_keep_failing_is_rebuilt_from_l3_and_applied() -> None:
    snapshot, pointers, _, l3 = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    l3.epochs["TX"] = 2  # the write that moved TX's pointer committed in L3

    _point_at_a_missing_chunk(pointers)

    await _until(lambda: "TX" in snapshot.status().behind, what="TX shown behind")
    await _until(lambda: snapshot.status().behind == {}, what="TX rebuilt from L3 and applied")
    await _until(lambda: snapshot.status().detail == "ready", what="the rebuild's step ended")
    await snapshot.stop()


async def test_a_rebuild_that_keeps_failing_takes_its_claim_at_the_recheck_not_in_a_spin() -> None:
    snapshot, pointers, store, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)

    # L3 never reaches the pointer's epoch, so every rebuild of TX fails, and each one takes and
    # lets go of the claim: those writes share the watched prefix and are no news of the data
    _point_at_a_missing_chunk(pointers)

    await _until(lambda: "TX" in snapshot.status().behind, what="TX shown behind")
    reads = await _reads_in(store, 0.25)
    assert reads < 30, f"{reads} reads in 0.25 s: the claim's own writes woke the worker into a spin"
    await snapshot.stop()


async def test_a_claim_whose_renewal_cannot_reach_nats_says_so_and_keeps_trying(
    caplog: pytest.LogCaptureFixture,
) -> None:
    snapshot, pointers, _, _ = _snapshot(claim_ttl=timedelta(seconds=0.9))
    pointers.update_fails = ConnectionError("no route to NATS")
    caplog.set_level(logging.WARNING)

    async with snapshot.holding_rebuilds() as held:
        assert held
        await asyncio.sleep(0.75)

    assert "renewing the snapshot rebuild claim" in caplog.text
    assert "enr.rebuild" not in pointers.entries, "the claim was not released"


class _BindingClient:
    """the hub's two asks answered as scripted, and the buckets bound in memory."""

    namespace = "3tears"

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.pointers = _Pointers()
        self.store = _Store()
        self.store.objects["enr/TX/0/results.old"] = b"stale"

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        self.sent.append(subject.path)
        body = json.loads(payload)
        if subject.path.endswith("declare"):
            reply: dict[str, Any] = {
                "success": True,
                "bucket": "3tears-tool_pod-x-objects",
                "pointers_bucket": "3tears-tool_pod-x-pointers",
                "max_bytes": 1024,
            }
        else:
            for name in body["names"]:
                self.store.objects.pop(name, None)
            reply = {"success": True, "retired": len(body["names"]), "absent": 0, "orphan_chunks": 0}
        return json.dumps({"correlation_id": body["correlation_id"], **reply}).encode()

    async def object_store(self, *, name: str, prefix_namespace: bool = True) -> _Store:
        return self.store

    async def kv_bucket(self, *, name: str, create_if_missing: bool = True) -> _Pointers:
        return self.pointers


async def _epochs() -> dict[str, int]:
    return {"TX": 1}


async def test_a_tool_pods_snapshot_is_started_and_retires_through_the_hub() -> None:
    client = _BindingClient()

    snapshot = await open_tool_pod_snapshot(
        client,  # type: ignore[arg-type]
        identity_token=lambda: "token",
        name="enr",
        tables=_TABLES,
        backend=_backend(),
        l3=_L3([{"county": "c1", "state": "TX", "votes": 1}]),
        epochs=_epochs,
        recheck=timedelta(seconds=0.05),
    )
    await snapshot.wait_ready(timeout=5)
    await snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 5}]})

    assert client.sent[0] == "3tears.hub.object_store.declare"
    assert "3tears.hub.object_store.retire" in client.sent, "a superseded chunk was not retired through the hub"
    assert "enr/TX/0/results.old" not in client.store.objects
    await snapshot.stop()


#: any value for an option the tool pod's snapshot wires itself: it is refused before it is used
_refused_option: Any = object()


@pytest.mark.parametrize("option", ["retire", "ensure_buckets", "store", "pointers"])
async def test_a_tool_pods_snapshot_refuses_an_option_it_wires_itself(option: str) -> None:
    client = _BindingClient()

    with pytest.raises(ValueError, match=option):
        await open_tool_pod_snapshot(
            client,  # type: ignore[arg-type]
            identity_token=lambda: "token",
            name="enr",
            tables=_TABLES,
            backend=_backend(),
            l3=_L3([]),
            epochs=_epochs,
            **{option: _refused_option},
        )
    assert client.sent == [], "the hub was asked before the options were checked"


async def test_a_catch_up_on_a_replica_whose_watch_ended_leaves_it_failed() -> None:
    """the watch ends while the buckets still answer (a consumer lost, the client fine): a catch-up
    that really rebuilds a scope must not make the replica, which no longer applies changes, look ready."""
    snapshot, pointers, _, l3 = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    # the first rebuild declares the copy ready before it lets go of its claim
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")
    pointers.end_watch()
    await _until(lambda: snapshot.status().phase is SnapshotPhase.FAILED, what="the watch's end")
    l3.rows[0]["votes"] = 7
    l3.epochs["TX"] = 2

    caught = await snapshot.catch_up_from_l3()

    assert caught == ["TX"], "the catch-up did not rebuild the scope L3 moved"
    assert snapshot.status().phase is SnapshotPhase.FAILED, "a catch-up hid a replica that no longer applies changes"
    assert "watch" in snapshot.status().detail
    await snapshot.stop()


async def test_a_catch_up_that_waits_on_a_write_leaves_the_replica_waiting_not_ready() -> None:
    snapshot, pointers, _, l3 = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    # the first rebuild declares the copy ready before it lets go of its claim
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")
    l3.epochs["TX"] = 2
    l3.writing = True

    caught = await snapshot.catch_up_from_l3()

    assert caught == []
    assert snapshot.status().phase is SnapshotPhase.WAITING, snapshot.status()
    assert "write in progress" in snapshot.status().detail
    l3.writing = False
    assert await snapshot.catch_up_from_l3() == ["TX"]
    assert snapshot.status().phase is SnapshotPhase.READY
    await snapshot.stop()


async def _staged_carry(snapshot: ScopedSnapshot) -> Any:
    """a stage of TX that names no table, so publishing it carries every chunk from epoch 1."""
    return await snapshot.stage("TX", 2, {})


async def test_a_carry_whose_pointer_moved_after_it_was_judged_is_left_for_the_catch_up() -> None:
    snapshot, pointers, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    staged = await _staged_carry(snapshot)
    real_update = pointers.update
    raced = [False]

    async def racing_update(*, key: str, value: bytes, revision: int, ttl: timedelta | None = None) -> int | None:
        if key == "enr.s.TX" and not raced[0]:
            # another writer moves TX between the carry's judgement and its compare-and-set
            raced[0] = True
            pointers.put_now(key, pointers.entries[key][0])
        return await real_update(key=key, value=value, revision=revision, ttl=ttl)

    pointers.update = racing_update  # type: ignore[method-assign]

    moved, skipped = await snapshot.publish_staged([staged], carry_at={"TX": 1})

    assert (moved, skipped) == ([], ["TX"]), "a carry judged against a moved pointer was published"
    await snapshot.stop()


async def test_a_carry_whose_chunk_is_gone_is_left_for_the_catch_up() -> None:
    snapshot, pointers, store, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    staged = await _staged_carry(snapshot)
    carried = json.loads(pointers.entries["enr.s.TX"][0])["tables"]["results"]["object"]
    del store.objects[carried]

    moved, skipped = await snapshot.publish_staged([staged], carry_at={"TX": 1})

    assert (moved, skipped) == ([], ["TX"]), "a pointer was moved onto a carried chunk that is gone"
    await snapshot.stop()


async def test_a_stuck_scope_whose_l3_epoch_is_below_its_pointer_stays_behind_on_every_look() -> None:
    """L3 never reaches the pointer's epoch: every rebuild of TX publishes nothing (a stale epoch) and
    its pointer's chunk still cannot be read, so TX is behind at every look across many rechecks and
    rebuilds, its reason named, never reported current in between."""
    snapshot, pointers, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")
    takes = [0]
    real_create = pointers.create

    async def counting(*, key: str, value: bytes, ttl: timedelta | None = None) -> int | None:
        takes[0] += key == "enr.rebuild"
        return await real_create(key=key, value=value, ttl=ttl)

    pointers.create = counting  # type: ignore[method-assign]
    _point_at_a_missing_chunk(pointers)
    await _until(lambda: "TX" in snapshot.status().behind, what="TX shown behind")

    looks = []
    while takes[0] < 2:  # rebuilds of TX ran, each taking and letting go of the claim
        looks.append(snapshot.status())
        await asyncio.sleep(0.01)
    for _ in range(20):  # and several rechecks after
        looks.append(snapshot.status())
        await asyncio.sleep(0.01)

    assert all("TX" in look.behind and "TX" in look.detail for look in looks), [
        (look.phase, sorted(look.behind)) for look in looks if "TX" not in look.behind
    ][:3]
    await snapshot.stop()


_TX_VOTES = "SELECT votes FROM results WHERE state = 'TX'"


def _read_versioned_now(snapshot: ScopedSnapshot) -> tuple[dict[str, int], dict[str, str], Any]:
    """a versioned read's epochs, behind set and TX's rows; blocking, so run on a worker thread."""
    with snapshot.read_versioned() as read:
        return dict(read.epochs), dict(read.behind), read.cursor.execute(_TX_VOTES).fetchall()


async def _versioned(snapshot: ScopedSnapshot) -> tuple[dict[str, int], dict[str, str], Any]:
    return await asyncio.to_thread(_read_versioned_now, snapshot)


async def _versioned_around(
    snapshot: ScopedSnapshot, during: Callable[[], Any]
) -> tuple[dict[str, int], dict[str, str], Any]:
    """a versioned read opened on a worker thread, held open while the loop runs ``during``, then queried."""
    opened, go = threading.Event(), threading.Event()

    def read() -> tuple[dict[str, int], dict[str, str], Any]:
        with snapshot.read_versioned() as versioned:
            opened.set()
            go.wait(10)
            return dict(versioned.epochs), dict(versioned.behind), versioned.cursor.execute(_TX_VOTES).fetchall()

    reading = asyncio.ensure_future(asyncio.to_thread(read))
    await asyncio.to_thread(opened.wait, 5)
    await during()
    go.set()
    return await reading


async def _versioned_in_time(snapshot: ScopedSnapshot, seconds: float = 3.0) -> tuple[dict[str, int], Any]:
    """a versioned read on a daemon thread that must finish in ``seconds``: one waiting on a swap
    lock nobody will release fails the test rather than hanging the suite."""
    out: list[tuple[dict[str, int], Any]] = []

    def read() -> None:
        epochs, _, rows = _read_versioned_now(snapshot)
        out.append((epochs, rows))

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    await _until(
        lambda: not thread.is_alive(), what="a versioned read to finish; the swap lock was left held", timeout=seconds
    )
    assert out, "the versioned read failed on its thread"
    return out[0]


def _slow_replace(
    snapshot: ScopedSnapshot, *, fail: bool = False
) -> tuple[threading.Event, threading.Event, threading.Event]:
    """hold the snapshot's next L1 commits until released; ``fail`` makes each raise after the hold.

    :return: (inside the commit, release it, the commit has returned or raised)
    """
    inside, release, returned = threading.Event(), threading.Event(), threading.Event()
    backend = snapshot.backend
    replace = backend.replace_partitions

    def held(replacements: Any) -> int:
        inside.set()
        release.wait(5)
        try:
            if fail:
                raise RuntimeError("the L1 refused the commit")
            return replace(replacements)
        finally:
            returned.set()

    backend.replace_partitions = held  # type: ignore[method-assign]
    return inside, release, returned


async def test_a_versioned_read_on_the_event_loop_is_refused_at_once() -> None:
    """it blocks while a swap commits: on the loop it would freeze every other task, so it fails fast."""
    snapshot, _, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    with pytest.raises(RuntimeError, match="worker thread"), snapshot.read_versioned():
        pass
    await snapshot.stop()


async def test_a_swap_whose_awaiter_is_cancelled_still_lets_versioned_reads_go() -> None:
    snapshot, _, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    inside, release, returned = _slow_replace(snapshot)
    publishing = asyncio.ensure_future(
        snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 20}]})
    )
    await asyncio.to_thread(inside.wait, 5)
    publishing.cancel()
    release.set()
    await asyncio.to_thread(returned.wait, 5)
    epochs, rows = await _versioned_in_time(snapshot)
    assert (epochs["TX"], rows) == (2, [(20,)]), "the committed rows and their epoch must move together"
    await snapshot.stop()


async def test_a_swap_whose_commit_raises_lets_versioned_reads_go_and_keeps_the_old_epoch() -> None:
    snapshot, _, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    inside, release, returned = _slow_replace(snapshot, fail=True)
    publishing = asyncio.ensure_future(
        snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 20}]})
    )
    await asyncio.to_thread(inside.wait, 5)
    release.set()
    await asyncio.to_thread(returned.wait, 5)
    with pytest.raises(RuntimeError, match="refused the commit"):
        await publishing
    epochs, rows = await _versioned_in_time(snapshot)
    assert (epochs["TX"], rows) == (1, [(1,)])
    await snapshot.stop()


async def test_a_swap_whose_commit_task_is_cancelled_still_lets_versioned_reads_go() -> None:
    """every task cancelled while the worker commits, the commit's own included (as a loop being torn
    down cancels them): the worker still finishes, and nothing is left holding the lock."""
    snapshot, _, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    inside, release, returned = _slow_replace(snapshot)
    publishing = asyncio.ensure_future(
        snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 20}]})
    )
    await asyncio.to_thread(inside.wait, 5)
    # every task but this one, as a loop being torn down cancels them: whichever runs the commit
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
    await asyncio.sleep(0)
    release.set()
    await asyncio.to_thread(returned.wait, 5)
    await asyncio.gather(publishing, return_exceptions=True)
    epochs, rows = await _versioned_in_time(snapshot)
    assert (epochs["TX"], rows) == (2, [(20,)]), "the committed rows and their epoch must move together"
    await snapshot.stop()


async def test_a_versioned_read_names_the_epoch_of_every_scope_it_reads() -> None:
    snapshot, _, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)

    epochs, _, votes = await _versioned(snapshot)
    assert (epochs, votes) == ({"TX": 1, "DE": 1}, [(1,)])

    await snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 20}]})
    epochs, _, votes = await _versioned(snapshot)
    assert (epochs, votes) == ({"TX": 2, "DE": 1}, [(20,)])
    await snapshot.stop()


async def test_a_versioned_read_never_pairs_a_scopes_rows_with_another_epoch() -> None:
    """a reader on another thread, while the scope moves epoch after epoch: rows and epoch always agree.

    A scope at epoch ``n`` holds ``votes = 10 * n`` here, so a read whose rows are one epoch and whose
    epochs say another is seen at once; the reads before and after each swap both land.
    """
    snapshot, _, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 20}]})
    stop = False
    mismatches: list[tuple[int, int]] = []
    seen: set[int] = set()

    def reader() -> None:
        while not stop:
            with snapshot.read_versioned() as read:
                (votes,) = read.cursor.execute("SELECT votes FROM results WHERE state = 'TX'").fetchone()
                epoch = read.epochs["TX"]
            seen.add(epoch)
            if votes != 10 * epoch:
                mismatches.append((epoch, votes))

    reading = asyncio.ensure_future(asyncio.to_thread(reader))
    for epoch in range(3, 30):
        await snapshot.publish("TX", epoch, {"results": [{"county": "c1", "state": "TX", "votes": 10 * epoch}]})
        await asyncio.sleep(0.005)
    stop = True
    await reading
    await snapshot.stop()

    assert mismatches == []
    assert len(seen) > 5, f"the reader saw only epochs {sorted(seen)}; the test raced nothing"


async def test_a_versioned_read_opened_during_a_long_swap_waits_for_it_and_reads_its_result() -> None:
    """deterministic: a swap held open (as a large load is) never makes the read fail; it waits, then reads it."""
    snapshot, _, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    inside, release = threading.Event(), threading.Event()
    backend = snapshot.backend
    replace = backend.replace_partitions

    def slow_replace(replacements: Any) -> int:
        inside.set()
        release.wait(5)  # a swap far longer than any spin budget
        return replace(replacements)

    backend.replace_partitions = slow_replace  # type: ignore[method-assign]
    publishing = asyncio.ensure_future(
        snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 20}]})
    )
    await asyncio.to_thread(inside.wait, 5)

    def read() -> tuple[dict[str, int], Any]:
        with snapshot.read_versioned() as versioned:
            rows = versioned.cursor.execute("SELECT votes FROM results WHERE state = 'TX'").fetchall()
            return dict(versioned.epochs), rows

    reading = asyncio.ensure_future(asyncio.to_thread(read))
    await asyncio.sleep(0.5)  # longer than the old 200-attempt budget
    assert not reading.done(), "the read did not wait for the swap in progress"
    release.set()
    epochs, rows = await reading
    await publishing
    assert (epochs["TX"], rows) == (2, [(20,)])
    await snapshot.stop()


async def test_a_dropped_scope_leaves_the_versioned_reads_epochs() -> None:
    """a scope L3 no longer holds is dropped by the catch-up, in the same step as its epoch."""
    snapshot, pointers, _, l3 = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the start's rebuild claim released")
    l3.rows = [r for r in l3.rows if r["state"] != "TX"]
    del l3.epochs["TX"]
    await snapshot.catch_up_from_l3()
    epochs, _, votes = await _versioned(snapshot)
    assert (epochs, votes) == ({"DE": 1}, [])
    await snapshot.stop()


async def test_a_versioned_read_keeps_the_state_its_epochs_name_through_a_later_swap() -> None:
    """deterministic: a swap committed after the read opened, before its first query, is not seen."""
    snapshot, _, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    epochs, _, votes = await _versioned_around(
        snapshot, lambda: snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 20}]})
    )
    assert (epochs["TX"], votes) == (1, [(1,)])
    await snapshot.stop()


async def test_a_stuck_scope_whose_later_pointer_carries_other_columns_stays_behind_on_every_look() -> None:
    """another code version wrote TX at epoch 3 under other columns, with a chunk that reads fine, and
    L3's epochs still say 2: each rebuild of TX publishes nothing (a stale epoch) and the pointer's
    chunks may not be loaded here, so TX is behind at every look across many rebuilds, never
    reported current in between while the L1 holds epoch 1."""
    snapshot, pointers, store, l3 = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")
    l3.epochs["TX"] = 2
    takes = [0]
    current: list[Any] = []  # the looks that found TX not behind; the others are not kept
    looking = False
    real_create, real_delete = pointers.create, pointers.delete

    def look() -> None:
        status = snapshot.status()
        if "TX" not in status.behind:
            current.append(status)

    # a look on each side of every take and every release of the claim, inside the loop's turn that runs them
    async def counting(*, key: str, value: bytes, ttl: timedelta | None = None) -> int | None:
        takes[0] += key == "enr.rebuild"
        if looking and key == "enr.rebuild":
            look()
        taken = await real_create(key=key, value=value, ttl=ttl)
        if looking and key == "enr.rebuild":
            look()
        return taken

    async def releasing(*, key: str, revision: int | None = None) -> bool:
        if looking and key == "enr.rebuild":
            look()
        released = await real_delete(key=key, revision=revision)
        if looking and key == "enr.rebuild":
            look()
        return released

    pointers.create = counting  # type: ignore[method-assign]
    pointers.delete = releasing  # type: ignore[method-assign]
    tx = json.loads(pointers.entries["enr.s.TX"][0])
    store.objects["enr/TX/3/results.other"] = encode_chunk(_tx_arrow(7).drop_columns(["votes"]))
    foreign = {
        **tx,
        "epoch": 3,
        "tables": {"results": {"object": "enr/TX/3/results.other", "rows": 1}},
        "schema": {"results": "the other version's columns"},
    }
    pointers.put_now("enr.s.TX", json.dumps(foreign).encode())
    await _until(lambda: "TX" in snapshot.status().behind, what="TX shown behind")

    # and a look at every turn of the loop, so between every step of every rebuild, while the test
    # waits on the rebuilds themselves rather than on a budget a loaded runner's pauses can use up
    async def look_at_every_turn() -> None:
        while looking:
            look()
            await asyncio.sleep(0)

    first, looking = takes[0], True
    watching = asyncio.create_task(look_at_every_turn())
    try:
        await _until(lambda: takes[0] >= first + 3, what="TX rebuilt again and again")
    finally:
        looking = False
        await watching

    assert current == [], [(look.phase, sorted(look.behind)) for look in current][:3]
    assert snapshot.applied_epoch("TX") == 1
    await snapshot.stop()


async def test_a_scope_removed_before_it_was_ever_applied_leaves_nothing_behind() -> None:
    snapshot, pointers, _, l3 = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    # a new scope NV is published with a chunk this replica cannot read
    tx = json.loads(pointers.entries["enr.s.TX"][0])
    nv = {**tx, "scope": "NV", "epoch": 3, "tables": {"results": {"object": "enr/NV/3/results.gone", "rows": 1}}}
    index = json.loads(pointers.entries["enr.index"][0])
    pointers.put_now("enr.s.NV", json.dumps(nv).encode())
    pointers.put_now("enr.index", json.dumps({**index, "scopes": [*index["scopes"], "NV"]}).encode())
    await _until(lambda: "NV" in snapshot.status().behind, what="NV shown behind")

    # and removed: its pointer goes, then it leaves the index
    await pointers.delete(key="enr.s.NV")
    pointers.put_now("enr.index", json.dumps(index).encode())

    await _until(lambda: "NV" not in snapshot.status().behind, what="NV to leave behind")
    reads = l3.statements
    await asyncio.sleep(0.3)  # several rechecks
    assert snapshot.status().behind == {} and "behind" not in snapshot.status().detail
    assert l3.statements == reads, "a removed scope kept being rebuilt from L3"
    await snapshot.stop()


async def test_a_read_carries_the_behind_set_of_the_data_it_reads() -> None:
    """a scope applied while a read is open is still behind for that read, whose data is the old
    epoch's; a read opened after sees it current and the new data."""
    snapshot, pointers, store, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")
    pointers.put_now("enr.rebuild", b"another replica")  # no rebuild here: only the chunk arriving helps
    _point_at_a_missing_chunk(pointers)
    await _until(lambda: "TX" in snapshot.status().behind, what="TX behind")

    with snapshot.read_with_behind() as (cursor, behind):
        store.objects["enr/TX/2/results.gone"] = _tx_chunk(5)
        await _until(lambda: snapshot.applied_epoch("TX") == 2, what="TX applied at epoch 2")
        assert snapshot.applied_epochs()["TX"] == 2
        assert "TX" in behind, "the read's behind set changed under it"
        assert cursor.execute("SELECT votes FROM results WHERE state = 'TX'").fetchall() == [(1,)]
    with snapshot.read_with_behind() as (cursor, behind):
        assert "TX" not in behind
        assert cursor.execute("SELECT votes FROM results WHERE state = 'TX'").fetchall() == [(5,)]
    await snapshot.stop()


async def test_a_versioned_read_carries_the_behind_set_of_the_data_it_reads() -> None:
    """as a plain read: a scope applied while a versioned read is open is still behind for that read,
    whose data and epoch are the old ones; a versioned read opened after sees it current."""
    snapshot, pointers, store, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")
    pointers.put_now("enr.rebuild", b"another replica")
    _point_at_a_missing_chunk(pointers)
    await _until(lambda: "TX" in snapshot.status().behind, what="TX behind")

    async def tx_arrives() -> None:
        store.objects["enr/TX/2/results.gone"] = _tx_chunk(5)
        await _until(lambda: snapshot.applied_epoch("TX") == 2, what="TX applied at epoch 2")

    epochs, behind, votes = await _versioned_around(snapshot, tx_arrives)
    assert "TX" in behind, "the read's behind set changed under it"
    assert (epochs["TX"], votes) == (1, [(1,)])
    epochs, behind, votes = await _versioned(snapshot)
    assert "TX" not in behind
    assert (epochs["TX"], votes) == (2, [(5,)])
    await snapshot.stop()


# ----------------------------------------------------------------------
# readers on other threads: the loop is the one writer of what they read
# ----------------------------------------------------------------------


def _tx_arrow(votes: int) -> Any:
    """TX's one row as Arrow, typed by the L1's own columns (its export, so the test needs no pyarrow)."""
    return _backend().export_rows(
        "results",
        "state",
        "TX",
        [{"county": "c1", "state": "TX", "votes": votes}],
        primary_key=("county",),
        order_by=("county",),
    )


def _tx_chunk(votes: int) -> bytes:
    return encode_chunk(_tx_arrow(votes))


async def _ready_with_tx_and_de_behind() -> tuple[ScopedSnapshot, _Pointers, _Store]:
    """a ready copy holding TX and DE at epoch 1, both behind on an epoch 2 whose chunks are not in the
    store, and no rebuild here (another replica holds the claim): only a chunk arriving moves them."""
    snapshot, pointers, store, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")
    pointers.put_now("enr.rebuild", b"another replica")
    _point_at_a_missing_chunk(pointers, "TX")
    _point_at_a_missing_chunk(pointers, "DE")
    await _until(lambda: set(snapshot.status().behind) == {"TX", "DE"}, what="TX and DE behind")
    return snapshot, pointers, store


class _PauseAtFirstLoopStep:
    """pause a worker thread inside the snapshot's own code mid-iteration, deterministically.

    A line event that repeats the line before it, on a line with a ``for``, is a loop going round
    again: in a comprehension it comes after the first item and before the next, so the thread is
    paused while it iterates. Code with no loop runs through unpaused.
    """

    def __init__(self) -> None:
        self.paused = threading.Event()
        self.resume = threading.Event()
        self._fired = False

    def _local(self) -> Callable[[FrameType, str, Any], Any]:
        last: list[int | None] = [None]

        def trace(frame: FrameType, event: str, arg: Any) -> Any:
            if event == "line" and not self._fired:
                looping = " for " in linecache.getline(frame.f_code.co_filename, frame.f_lineno)
                if frame.f_lineno == last[0] and looping:
                    self._fired = True
                    self.paused.set()
                    self.resume.wait(10)
                last[0] = frame.f_lineno
            return trace

        return trace

    def _call(self, frame: FrameType, event: str, arg: Any) -> Any:
        mine = event == "call" and frame.f_code.co_filename == scoped_snapshot_module.__file__
        return self._local() if mine and not self._fired else None

    def run[T](self, read: Callable[[], T]) -> T:
        sys.settrace(self._call)
        try:
            return read()
        finally:
            sys.settrace(None)


def _rows_and_behind(status: Any) -> tuple[dict[str, int], set[str]]:
    return dict(status.rows), set(status.behind)


def _read_behind(snapshot: ScopedSnapshot) -> set[str]:
    with snapshot.read_with_behind() as (_, behind):
        return set(behind)


def _read_versioned_epochs_and_behind(snapshot: ScopedSnapshot) -> tuple[dict[str, int], set[str]]:
    with snapshot.read_versioned() as read:
        return dict(read.epochs), set(read.behind)


_READERS: dict[str, Callable[[ScopedSnapshot], Any]] = {
    "applied_epochs": lambda snapshot: snapshot.applied_epochs(),
    "read_versioned": _read_versioned_epochs_and_behind,
    "status": lambda snapshot: _rows_and_behind(snapshot.status()),
    "read_with_behind": _read_behind,
}
_BEFORE: dict[str, Any] = {
    "applied_epochs": {"TX": 1, "DE": 1},
    "read_versioned": ({"TX": 1, "DE": 1}, {"TX", "DE"}),
    "status": ({"results": 2}, {"TX", "DE"}),
    "read_with_behind": {"TX", "DE"},
}


@pytest.mark.parametrize("reader", sorted(_READERS))
async def test_a_reader_on_a_worker_thread_paused_mid_iteration_is_untouched_by_the_loop_s_changes(
    reader: str,
) -> None:
    """a worker thread is paused inside an accessor while the loop adds a scope (NV, published here)
    and brings TX current (its chunk arrives): what the reader took is the state from before, and it
    finishes without 'dictionary changed size during iteration'."""
    snapshot, _, store = await _ready_with_tx_and_de_behind()
    pause = _PauseAtFirstLoopStep()
    read = asyncio.create_task(asyncio.to_thread(pause.run, lambda: _READERS[reader](snapshot)))
    await _until(lambda: pause.paused.is_set() or read.done(), what="the reader paused or done")

    async def change() -> None:
        await snapshot.publish("NV", 1, {"results": [{"county": "c9", "state": "NV", "votes": 9}]})
        store.objects["enr/TX/2/results.gone"] = _tx_chunk(5)
        await _until(lambda: snapshot.applied_epoch("TX") == 2, what="TX applied at epoch 2")

    try:
        # bounded: a change that waited for the paused reader would otherwise wait out its pause
        await asyncio.wait_for(change(), 3)
    finally:
        pause.resume.set()
    assert snapshot.applied_epochs() == {"TX": 2, "DE": 1, "NV": 1}

    assert await read == _BEFORE[reader]
    assert pause.paused.is_set(), "the reader never iterated: nothing was interleaved"
    await snapshot.stop()


class _HeldRows(list[dict[str, Any]]):
    """a scope's rows whose reading is held until released: a publish of them stops inside its write,
    after the replacement's DELETE and before its INSERT and COMMIT, on the write's worker thread."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__(rows)
        self.entered = threading.Event()
        self.release = threading.Event()

    def __iter__(self) -> Any:
        self.entered.set()
        self.release.wait(10)
        return super().__iter__()


async def test_a_read_during_a_long_write_neither_waits_for_it_nor_is_told_its_scope_is_current() -> None:
    """while a publish bringing TX current is held inside its DuckDB write (TX's rows deleted, the new
    ones not yet inserted, nothing committed), a read opened on the event loop opens at once (the loop
    is not frozen behind the write), reads TX's old rows and is told TX is behind; once the write
    commits, a new read has the new rows and TX current."""
    snapshot, pointers, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")
    pointers.put_now("enr.rebuild", b"another replica")
    _point_at_a_missing_chunk(pointers)
    await _until(lambda: "TX" in snapshot.status().behind, what="TX behind")
    rows = _HeldRows([{"county": "c1", "state": "TX", "votes": 5}])
    publishing = asyncio.create_task(snapshot.publish("TX", 2, {"results": rows}))
    await _until(rows.entered.is_set, what="the write bringing TX current to be inside its transaction")
    # if the read waited for the write, this lets it go after 3 s so the test fails rather than hangs
    threading.Timer(3.0, rows.release.set).start()

    started = time.perf_counter()
    with snapshot.read_with_behind() as (cursor, behind):
        opened = time.perf_counter() - started
        assert "TX" in behind
        assert cursor.execute("SELECT votes FROM results WHERE state = 'TX'").fetchall() == [(1,)]
    assert opened < 0.5, f"the read waited {opened:.1f} s on the event loop for a write in progress"
    assert "TX" in snapshot.status().behind

    rows.release.set()
    await publishing
    assert snapshot.applied_epoch("TX") == 2
    with snapshot.read_with_behind() as (cursor, behind):
        assert "TX" not in behind
        assert cursor.execute("SELECT votes FROM results WHERE state = 'TX'").fetchall() == [(5,)]
    await snapshot.stop()


class _GatedStore(_Store):
    """a store whose next read of a missing chunk, and whose writes of one scope's epoch, each wait for a release."""

    def __init__(self) -> None:
        super().__init__()
        self.missing_read = asyncio.Event()
        self.release_missing = asyncio.Event()
        self.hold_missing = False
        self.writing = asyncio.Event()
        self.release_writes = asyncio.Event()
        self.hold_prefix: str | None = None

    async def get(self, name: str) -> bytes:
        if self.hold_missing and name not in self.objects:
            self.hold_missing = False
            self.missing_read.set()
            await self.release_missing.wait()
        return await super().get(name)

    async def put(self, name: str, data: bytes) -> None:
        if self.hold_prefix is not None and name.startswith(self.hold_prefix):
            self.writing.set()
            await self.release_writes.wait()
        await super().put(name, data)


async def test_a_failed_read_of_a_pointer_a_publish_moved_on_from_does_not_put_the_scope_behind_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """a recheck reads TX's chunks at a pointer naming a missing chunk while a publish brings TX current: the publish
    commits locally (taking TX off ``behind``), then writes its chunks and moves the pointer, all under the L1's lock.
    The read fails in that window. Recorded outside the lock, the failure put TX back on ``behind`` after the publish
    took it off, and nothing took it off again: TX stayed behind though it was current (CI, 2026-10-18)."""
    caplog.set_level(logging.WARNING, logger="threetears.core.collections.scoped_snapshot")
    store = _GatedStore()
    snapshot, pointers, _, _ = _snapshot(store=store)
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")
    pointers.put_now("enr.rebuild", b"another replica")
    _point_at_a_missing_chunk(pointers)
    await _until(lambda: "TX" in snapshot.status().behind, what="TX behind")

    store.hold_missing = True
    await _until(store.missing_read.is_set, what="a recheck reading the missing chunk")
    store.hold_prefix = "enr/TX/2/"
    publishing = asyncio.create_task(
        snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 5}]})
    )
    await _until(store.writing.is_set, what="the publish writing its chunks, its local commit done")
    assert "TX" not in snapshot.status().behind, "the publish's local commit took TX off behind"

    failures = caplog.text.count("could not be applied")
    store.release_missing.set()
    await _until(lambda: caplog.text.count("could not be applied") > failures, what="the recheck's read failing")
    store.release_writes.set()
    await publishing

    assert snapshot.applied_epoch("TX") == 2
    await _until(lambda: "TX" not in snapshot.status().behind, what="TX current", timeout=2.0)
    with snapshot.read_with_behind() as (cursor, behind):
        assert "TX" not in behind
        assert cursor.execute("SELECT votes FROM results WHERE state = 'TX'").fetchall() == [(5,)]
    await snapshot.stop()


class _HeldPointerCreate(_Pointers):
    """a pointer bucket whose first write of one new key waits for a release."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        self.creating = asyncio.Event()
        self.release = asyncio.Event()

    async def create(self, *, key: str, value: bytes, ttl: timedelta | None = None) -> int | None:
        if key == self.key and not self.release.is_set():
            self.creating.set()
            await self.release.wait()
        return await super().create(key=key, value=value, ttl=ttl)


async def test_a_recheck_during_a_publish_of_a_new_scope_does_not_drop_it() -> None:
    """a publish of a new scope (NV) holds the L1's lock from its local commit, which lists NV as applied, until its
    pointer moves. A recheck in that window saw NV applied, in no index it had read and with no pointer seen, and judged
    it gone; it dropped NV once the publish let the lock go, so NV, published here, was not held (CI, 2026-10-17). The
    publish is held at its pointer's first write for ten recheck intervals, so a recheck runs in the window, and
    once a later recheck applied NV again from its pointer it was held again: the loss is the gap between."""
    pointers = _HeldPointerCreate("enr.s.NV")
    snapshot, _, _, _ = _snapshot(pointers=pointers)
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    # TX behind on a chunk that is not there keeps the recheck passing; another replica holds the rebuild claim
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")
    pointers.put_now("enr.rebuild", b"another replica")
    _point_at_a_missing_chunk(pointers, "TX")
    await _until(lambda: "TX" in snapshot.status().behind, what="TX behind")
    publishing = asyncio.create_task(
        snapshot.publish("NV", 1, {"results": [{"county": "c9", "state": "NV", "votes": 9}]})
    )
    await _until(pointers.creating.is_set, what="the publish moving NV's pointer, its local commit done")
    await asyncio.sleep(0.5)  # ten recheck intervals: the window the drop needs, not a wait for work
    pointers.release.set()
    await publishing

    # the drop landed once the lock was free and a later recheck applied NV again from its pointer, so what a
    # reader lost is a window: NV must be held at every look from the publish's return
    missing = 0
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        missing += snapshot.applied_epochs().get("NV") != 1
        await asyncio.sleep(0.005)
    assert missing == 0, f"NV, published here, was not held at {missing} looks after its publish returned"
    await snapshot.stop()


# the state readers on any thread take, which the event loop alone writes, by rebinding
_SHARED = frozenset({"_applied", "_behind", "_rows", "_progress"})
# set once at construction, and safe to use from any thread
_SET_AT_START = frozenset({"_backend", "_ready", "_name", "_tables", "_append", "_schema", "_versioned"})
# public, synchronous, and called on the event loop only, said so in its docstring
_LOOP_ONLY = {"on_change": "registers a listener; the listener list is the loop's, read by _notify there"}
_IN_PLACE = frozenset(
    {"append", "extend", "insert", "pop", "popitem", "remove", "clear", "update", "setdefault", "add", "discard"}
)
_REBINDERS = frozenset({"_updated", "replace", "_Progress"})


def _module_tree() -> ast.Module:
    with open(scoped_snapshot_module.__file__, encoding="utf-8") as source:
        return ast.parse(source.read())


def _class_def(name: str) -> ast.ClassDef:
    found = [n for n in _module_tree().body if isinstance(n, ast.ClassDef) and n.name == name]
    assert found, f"{name} is not in the module"
    return found[0]


def _self_attr(node: ast.AST) -> str | None:
    """``X`` when ``node`` is ``self.X``."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _shared_root(node: ast.AST) -> str | None:
    """the shared attribute ``node`` reaches into (``self.X[k]``, ``self.X.y``), not ``self.X`` itself."""
    inner = node.value if isinstance(node, (ast.Attribute, ast.Subscript)) else None
    while inner is not None:
        attr = _self_attr(inner)
        if attr is not None:
            return attr if attr in _SHARED else None
        inner = inner.value if isinstance(inner, (ast.Attribute, ast.Subscript)) else None
    return None


def test_the_state_other_threads_read_is_changed_only_by_rebinding_an_immutable_value() -> None:
    """no code changes the shared state in place: every write builds a new read-only value and
    rebinds the attribute, so a reader on another thread never sees a value it holds change."""
    wrong: list[str] = []
    for node in ast.walk(_class_def("ScopedSnapshot")):
        targets: list[ast.AST] = []
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        line = getattr(node, "lineno", 0)
        for target in targets:
            if _shared_root(target) is not None or (isinstance(node, ast.AugAssign) and _self_attr(target) in _SHARED):
                wrong.append(f"line {line}: changes {ast.unparse(target)} in place")
            elif _self_attr(target) in _SHARED and isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                built_by = value.func.id if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) else None
                if built_by not in _REBINDERS:
                    wrong.append(f"line {line}: binds {ast.unparse(target)} to a value not built read-only")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _IN_PLACE:
            if _self_attr(node.func.value) in _SHARED or _shared_root(node.func.value) is not None:
                wrong.append(f"line {node.lineno}: {ast.unparse(node.func)} changes shared state in place")
    progress = _class_def("_Progress")
    frozen = any(
        isinstance(d, ast.Call) and any(k.arg == "frozen" and getattr(k.value, "value", False) for k in d.keywords)
        for d in progress.decorator_list
    )
    if not frozen:
        wrong.append("_Progress is not a frozen dataclass")
    for item in progress.body:
        if isinstance(item, ast.AnnAssign) and ast.unparse(item.annotation).split("[")[0] in {"dict", "list", "set"}:
            wrong.append(f"_Progress.{ast.unparse(item.target)} is a mutable {ast.unparse(item.annotation)}")
    assert not wrong, "\n".join(wrong)


def test_what_any_thread_may_call_reads_only_the_shared_state_each_once_and_runs_nothing_of_the_snapshot_off_the_loop() -> (
    None
):
    """every public synchronous method (callable from a worker thread) reads only the shared state
    and what is set at construction, each shared attribute once, so it sees one state of each; and
    nothing the snapshot hands to a worker thread is one of its own methods, which could reach state
    the loop is changing."""
    snapshot_class = _class_def("ScopedSnapshot")
    methods = {n.name: n for n in snapshot_class.body if isinstance(n, ast.FunctionDef)}
    wrong: list[str] = []
    for name, method in methods.items():
        if name.startswith("_") or name in _LOOP_ONLY:
            continue
        reads: dict[str, int] = {}
        for node in ast.walk(method):
            attr = _self_attr(node)
            if attr is None:
                continue
            if attr in methods:
                wrong.append(f"{name} calls self.{attr}: check what that reads, or inline it")
            elif attr not in _SHARED | _SET_AT_START:
                wrong.append(f"{name} reads self.{attr}, which the loop changes in place")
            reads[attr] = reads.get(attr, 0) + 1
        wrong.extend(
            f"{name} reads self.{a} {n} times: it may see two states"
            for a, n in reads.items()
            if a in _SHARED and n > 1
        )
    for node in ast.walk(snapshot_class):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if _self_attr(target) in _SET_AT_START:
                    owner = next(m for m in methods.values() if any(n is node for n in ast.walk(m)))
                    if owner.name != "__init__":
                        wrong.append(
                            f"{owner.name} rebinds self.{_self_attr(target)}, which is set once at construction"
                        )
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "asyncio.to_thread" and node.args:
            work = node.args[0]
            # what a worker may run: the backend's methods, and the versioned L1's (which owns its state)
            owner = _self_attr(work.value) if isinstance(work, ast.Attribute) else None
            if isinstance(work, ast.Lambda) or _self_attr(work) is not None or owner not in (None, *_OFF_LOOP):
                wrong.append(
                    f"line {node.lineno}: to_thread runs {ast.unparse(work)}, the snapshot's own code, off the loop"
                )
    assert not wrong, "\n".join(wrong)


# what the snapshot may hand a worker thread: the backend, and the versioned L1, whose state is its own
_OFF_LOOP = frozenset({"_backend", "_versioned"})
# the versioned L1's own state: its backend and lock, set once, and the epochs, rebound under the lock
_VERSIONED_SET_AT_START = frozenset({"_backend", "_lock"})
_VERSIONED_REBOUND = frozenset({"_held"})


def test_the_versioned_l1_touches_only_its_own_state_and_rebinds_its_epochs_under_its_lock() -> None:
    """it runs on worker threads: it reads nothing of the snapshot, sets its backend and lock once,
    reads its epochs once per method, and only rebinds them (never in place), inside ``with self._lock``."""
    versioned = _class_def("_VersionedL1")
    methods = {n.name: n for n in versioned.body if isinstance(n, ast.FunctionDef)}
    wrong: list[str] = []
    for name, method in methods.items():
        reads: dict[str, int] = {}
        locked = {
            id(inner)
            for node in ast.walk(method)
            if isinstance(node, ast.With) and any(ast.unparse(item.context_expr) == "self._lock" for item in node.items)
            for inner in ast.walk(node)
        }
        for node in ast.walk(method):
            attr = _self_attr(node)
            if attr is None:
                continue
            if attr not in _VERSIONED_SET_AT_START | _VERSIONED_REBOUND:
                wrong.append(f"_VersionedL1.{name} reaches self.{attr}, which is not the versioned L1's own state")
            if attr in _VERSIONED_REBOUND and isinstance(node.ctx, ast.Load):
                reads[attr] = reads.get(attr, 0) + 1
                if name != "__init__" and id(node) not in locked:
                    wrong.append(f"_VersionedL1.{name} reads self.{attr} outside the lock")
        wrong.extend(f"_VersionedL1.{name} reads self.{a} {n} times" for a, n in reads.items() if n > 1)
        for node in ast.walk(method):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    attr = _self_attr(target)
                    if attr in _VERSIONED_SET_AT_START and name != "__init__":
                        wrong.append(f"_VersionedL1.{name} rebinds self.{attr}, which is set once at construction")
                    if attr in _VERSIONED_REBOUND:
                        value = node.value
                        built = (
                            value.func.id if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) else None
                        )
                        if isinstance(node, ast.AugAssign) or built not in _REBINDERS:
                            wrong.append(f"_VersionedL1.{name} binds self.{attr} to a value not built read-only")
                        if name != "__init__" and id(node) not in locked:
                            wrong.append(f"_VersionedL1.{name} rebinds self.{attr} outside the lock")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _IN_PLACE:
                if _self_attr(node.func.value) in _VERSIONED_REBOUND:
                    wrong.append(f"_VersionedL1.{name}: {ast.unparse(node.func)} changes the epochs in place")
    assert not wrong, "\n".join(wrong)


def test_the_loop_only_public_methods_say_so() -> None:
    methods = {n.name: n for n in _class_def("ScopedSnapshot").body if isinstance(n, ast.FunctionDef)}
    for name in _LOOP_ONLY:
        assert "Call this on the event loop" in " ".join((ast.get_docstring(methods[name]) or "").split()), (
            f"{name} does not say it is loop-only"
        )


async def test_a_pointer_of_other_columns_found_on_the_re_read_after_a_retired_chunk_is_never_loaded() -> None:
    """a rolling deploy: this replica is applying its own version's epoch 2 of TX when the other
    version writes epoch 3 under other columns and retires epoch 2's chunk, before the watch delivers
    epoch 3. The re-read after the missing chunk finds epoch 3, which supersedes what it was fetching;
    its chunks hold another column set, so TX stays behind at epoch 1 and the L1 never holds them."""
    snapshot, pointers, store, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")
    pointers.put_now("enr.rebuild", b"another replica")  # no rebuild here: the other version's write stands
    tx = json.loads(pointers.entries["enr.s.TX"][0])
    store.objects["enr/TX/3/results"] = encode_chunk(_tx_arrow(1).drop_columns(["votes"]))  # other columns
    foreign = {
        **tx,
        "epoch": 3,
        "tables": {"results": {"object": "enr/TX/3/results", "rows": 1}},
        "schema": {"results": "the other version's columns"},
    }

    _point_at_a_missing_chunk(pointers)  # the watch delivers this replica's own epoch 2 ...
    # ... and, before this replica reads it, the other version's epoch 3 is in KV, not yet delivered
    pointers.entries["enr.s.TX"] = (json.dumps(foreign).encode(), pointers.entries["enr.s.TX"][1])

    await _until(lambda: "TX" in snapshot.status().behind, what="TX shown behind")
    assert "other columns" in snapshot.status().behind["TX"]
    await asyncio.sleep(0.3)  # several rechecks, each re-reading epoch 3
    assert snapshot.applied_epoch("TX") == 1, "the L1 took a pointer whose chunks hold other columns"
    with snapshot.read() as cursor:
        assert cursor.execute("SELECT votes FROM results WHERE state = 'TX'").fetchall() == [(1,)]
    await snapshot.stop()


async def test_a_carry_checks_its_chunks_one_by_one_without_listing_the_bucket() -> None:
    snapshot, _, store, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    staged = await _staged_carry(snapshot)
    listings = store.listings

    moved, skipped = await snapshot.publish_staged([staged], carry_at={"TX": 1})

    assert (moved, skipped) == (["TX"], [])
    assert store.listings == listings, "the carry listed the bucket to find chunks it names"
    await snapshot.stop()


async def test_a_listener_told_of_a_publish_made_here_reads_its_new_epoch() -> None:
    snapshot, _, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    told: list[int | None] = []
    snapshot.on_change(lambda: told.append(snapshot.applied_epoch("TX")))

    await snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 5}]})

    assert told and told[-1] == 2, f"the listener read epoch {told} after the publish committed epoch 2"
    await snapshot.stop()


async def test_a_publish_overtaken_by_a_racing_writer_records_the_change_this_replica_made() -> None:
    """another writer moves TX to epoch 5 while this replica writes its epoch 3 chunks: the last
    change recorded is this replica's epoch 3, the one its L1 holds, not the racing writer's."""
    snapshot, pointers, store, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    real_put = store.put
    raced = [False]

    async def racing_put(name: str, data: bytes) -> None:
        await real_put(name, data)
        if name.startswith("enr/TX/3/") and not raced[0]:
            raced[0] = True
            later = json.loads(pointers.entries["enr.s.TX"][0])
            later["epoch"], later["tables"]["results"]["object"] = 5, "enr/TX/5/results.elsewhere"
            pointers.put_now("enr.s.TX", json.dumps(later).encode())

    store.put = racing_put  # type: ignore[method-assign]

    await snapshot.publish("TX", 3, {"results": [{"county": "c1", "state": "TX", "votes": 5}]})

    change = snapshot.status().last_change
    assert raced[0] and change is not None
    assert (change.scope, change.epoch) == ("TX", 3)
    assert snapshot.applied_epoch("TX") == 3
    await snapshot.stop()


class _HeldReplacements(DuckDBBackend):
    """a DuckDB L1 whose next replacement waits, before it starts, until released."""

    def __init__(self) -> None:
        super().__init__()
        self.hold_next = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def replace_partitions(self, replacements: Any) -> int:
        if self.hold_next:
            self.hold_next = False
            self.entered.set()
            self.release.wait(10)
        written: int = super().replace_partitions(replacements)
        return written


async def test_a_scope_being_dropped_is_no_longer_listed_as_held_while_the_drop_commits() -> None:
    """applied_epochs() promises a read opened after it holds every scope it lists: a scope whose
    drop is committing may already be gone, so it is not listed from the moment the drop starts."""
    metadata = MetaData()
    _RESULTS.to_sqlalchemy_table(metadata)
    backend = _HeldReplacements()
    backend.initialize(metadata)
    snapshot, pointers, _, _ = _snapshot(backend=backend)
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await snapshot.publish("NV", 1, {"results": [{"county": "c9", "state": "NV", "votes": 9}]})
    assert "NV" in snapshot.applied_epochs()
    threading.Timer(5.0, backend.release.set).start()  # never hangs the test
    backend.hold_next = True

    await pointers.delete(key="enr.s.NV")
    index = json.loads(pointers.entries["enr.index"][0])
    pointers.put_now("enr.index", json.dumps({**index, "scopes": ["DE", "TX"]}).encode())
    await _until(backend.entered.is_set, what="NV's drop to start")

    assert "NV" not in snapshot.applied_epochs(), "a scope whose rows are being dropped is still listed"
    backend.release.set()

    def nv_rows() -> int:
        with snapshot.read() as cursor:
            return int(cursor.execute("SELECT count(*) FROM results WHERE state = 'NV'").fetchone()[0])

    await _until(lambda: nv_rows() == 0, what="the drop to commit")
    await snapshot.stop()


class _BoundedStore(_Store):
    """an Object Store with a byte bound, refusing a write past it as JetStream does."""

    def __init__(self, max_bytes: int) -> None:
        super().__init__()
        self.max_bytes = max_bytes
        self.fail_put: Callable[[str], bool] = lambda name: False
        self.retired: list[str] = []

    async def put(self, name: str, data: bytes) -> None:
        from threetears.nats import ObjectStoreError, ObjectStoreFullError

        if self.fail_put(name):
            raise ObjectStoreError(f"writing object {name!r} failed")
        if sum(len(d) for d in self.objects.values()) + len(data) > self.max_bytes:
            raise ObjectStoreFullError(f"objects is full: writing {name!r}", bucket="objects", name=name)
        self.objects[name] = data

    async def list_objects(self, *, prefix: str = "") -> list[ObjectInfo]:
        # every chunk as written long ago: the rule must not care how old a chunk is
        long_ago = datetime(2000, 1, 1, tzinfo=UTC)
        return [
            ObjectInfo(name=n, size=len(d), chunks=1, digest="", nuid="", mtime=long_ago)
            for n, d in self.objects.items()
            if n.startswith(prefix)
        ]


def _retiring_snapshot(max_bytes: int = 10**9) -> tuple[ScopedSnapshot, _Pointers, _BoundedStore, _L3]:
    """a snapshot whose retire deletes from a bounded store, as the declarer does, and records what it took."""
    store = _BoundedStore(max_bytes)

    async def retire(names: list[str]) -> None:
        for name in names:
            store.retired.append(name)
            store.objects.pop(name, None)

    snapshot, pointers, _, l3 = _snapshot(store=store, retire=retire)
    return snapshot, pointers, store, l3


async def _started(snapshot: ScopedSnapshot, pointers: _Pointers) -> None:
    """started, ready, and its first rebuild's claim let go, so a catch-up here takes it."""
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)
    await _until(lambda: "enr.rebuild" not in pointers.entries, what="the first rebuild's claim released")


def _epochs_held(store: _Store, scope: str) -> set[int]:
    return {int(name.split("/")[2]) for name in store.objects if name.startswith(f"enr/{scope}/")}


def _served(pointers: _Pointers) -> set[str]:
    """every chunk a pointer names."""
    served: set[str] = set()
    for key, (value, _) in pointers.entries.items():
        if key.startswith("enr.s."):
            served |= {table["object"] for table in json.loads(value)["tables"].values()}
    return served


async def _fill_and_sweep(snapshot: ScopedSnapshot, store: _BoundedStore, l3: _L3) -> None:
    """fill the store to its bound, then publish: the chunk write is refused full and sweeps the store."""
    from threetears.nats import ObjectStoreFullError

    store.max_bytes = sum(len(d) for d in store.objects.values())
    l3.epochs["DE"] = l3.epochs.get("DE", 1) + 1
    # NOSILENT: a store the sweep could not free refuses the write again; the sweep is what is tested
    with contextlib.suppress(ObjectStoreFullError):
        await snapshot.catch_up_from_l3()


def _writer_died(pointers: _Pointers) -> None:
    """the writer holding every write claim now is dead: nothing it sends lands any more, so its
    claims lapse (their per-key TTL runs out) and are never renewed or taken back."""
    for key in [k for k in pointers.entries if k.startswith("enr.w.")]:
        pointers.dead_keys.add(key)
        del pointers.entries[key]


def _bucket_lost_the_claims(pointers: _Pointers) -> None:
    """the bucket lost every write claim (NATS lost it) while their writers still live and renew."""
    for key in [k for k in pointers.entries if k.startswith("enr.w.")]:
        del pointers.entries[key]


_TX_ROWS = {"results": [{"county": "c1", "state": "TX", "votes": 5}]}


async def test_a_catch_up_that_fails_part_way_still_retires_the_old_chunks_of_the_scopes_it_moved() -> None:
    snapshot, pointers, store, l3 = _retiring_snapshot()
    await _started(snapshot, pointers)
    l3.epochs = {"DE": 2, "TX": 2}
    store.fail_put = lambda name: name.startswith("enr/TX/2/")  # DE publishes first, TX's write fails

    with pytest.raises(Exception, match="failed"):
        await snapshot.catch_up_from_l3()

    assert _epochs_held(store, "DE") == {2}, "DE's pointer moved to 2 but its epoch-1 chunks were kept"
    assert _epochs_held(store, "TX") == {1}
    await snapshot.stop()


async def test_a_full_store_retires_only_what_no_pointer_serves_and_no_write_claims() -> None:
    snapshot, pointers, store, l3 = _retiring_snapshot()
    await _started(snapshot, pointers)
    served = _served(pointers)
    staged = await snapshot.stage("TX", 3, _TX_ROWS)  # a live write's stage, its claim held
    held = sum(len(d) for d in store.objects.values())
    store.objects["enr/TX/0/results.superseded"] = b"x" * 4096
    store.objects["enr/DE/0/results.superseded"] = b"x" * 4096
    store.max_bytes = held + 8192 + 64
    l3.epochs = {"DE": 2, "TX": 1}

    caught = await snapshot.catch_up_from_l3()

    assert caught == ["DE"]
    assert set(store.retired) & (served - {n for n in served if n.startswith("enr/DE/")}) == set(), (
        "the full-store sweep took a chunk a pointer served"
    )
    assert not set(store.retired) & set(staged.objects.values()), "the sweep took a live write's stage"
    assert {"enr/TX/0/results.superseded", "enr/DE/0/results.superseded"} <= set(store.retired)
    await snapshot.stop()


async def test_a_live_writes_stages_survive_a_full_store_sweep_however_old_they_are() -> None:
    snapshot, pointers, store, l3 = _retiring_snapshot()
    await _started(snapshot, pointers)
    staged = [
        await snapshot.stage("TX", 3, _TX_ROWS),
        # a scope with no pointer yet: nothing but the write's claim speaks for its stage
        await snapshot.stage("CA", 3, {"results": [{"county": "c9", "state": "CA", "votes": 1}]}),
    ]

    await _fill_and_sweep(snapshot, store, l3)

    for scope in staged:
        assert set(scope.objects.values()) <= set(store.objects), f"a live write's stage of {scope.scope} was swept"
    await snapshot.stop()


async def test_an_abandoned_writes_stages_are_swept_once_its_claim_is_gone() -> None:
    snapshot, pointers, store, l3 = _retiring_snapshot()
    await _started(snapshot, pointers)
    staged = await snapshot.stage("TX", 3, _TX_ROWS)

    _writer_died(pointers)
    await _fill_and_sweep(snapshot, store, l3)

    assert not set(staged.objects.values()) & set(store.objects), "a dead write's stage outlived its claim"
    assert _served(pointers) <= set(store.objects)
    await snapshot.stop()


async def test_a_write_that_never_commits_gives_its_stages_back() -> None:
    snapshot, pointers, store, _ = _retiring_snapshot()
    await _started(snapshot, pointers)
    staged = await snapshot.stage("TX", 2, _TX_ROWS)
    assert _epochs_held(store, "TX") == {1, 2}

    await snapshot.discard_staged([staged])

    assert _epochs_held(store, "TX") == {1}, "an abandoned write's stage was kept"
    assert not any(k.startswith("enr.w.") for k in pointers.entries), "the write claim was not let go"
    await snapshot.stop()


async def test_discarding_a_dead_writes_epoch_takes_its_stages_and_keeps_what_pointers_name_or_claims_hold() -> None:
    snapshot, pointers, store, _ = _retiring_snapshot()
    await _started(snapshot, pointers)
    await snapshot.stage("TX", 3, _TX_ROWS)
    await snapshot.stage("DE", 3, {"results": [{"county": "c2", "state": "DE", "votes": 6}]})
    served = _served(pointers)

    await snapshot.discard_epoch(3)
    assert {n for n in store.objects if "/3/" in n}, "a live write's claimed epoch was discarded"

    _writer_died(pointers)
    await snapshot.discard_epoch(3)
    await snapshot.discard_epoch(1)  # an epoch the pointers serve: nothing of it goes

    assert set(store.objects) == served
    await snapshot.stop()


async def test_a_pointer_never_moves_onto_a_staged_chunk_that_is_gone() -> None:
    snapshot, pointers, store, l3 = _retiring_snapshot()
    await _started(snapshot, pointers)
    before = pointers.entries["enr.s.TX"][0]
    staged = await snapshot.stage("TX", 2, _TX_ROWS)
    for name in staged.objects.values():
        del store.objects[name]
    l3.epochs["TX"] = 2

    moved, skipped = await snapshot.publish_staged([staged], carry_at={"TX": 1})

    assert (moved, skipped) == ([], ["TX"])
    assert pointers.entries["enr.s.TX"][0] == before, "a pointer was moved onto a missing chunk"
    await snapshot.stop()


def _write_claims(pointers: _Pointers) -> list[str]:
    return [key for key in pointers.entries if key.startswith("enr.w.")]


async def test_every_write_is_its_own_claim_in_the_grammar_the_purge_reads_and_lapses_with_its_writer() -> None:
    from threetears.nats.object_store_requests import is_write_claim_key

    snapshot, pointers, _, _ = _retiring_snapshot()
    await _started(snapshot, pointers)
    texas = await snapshot.stage("TX", 3, _TX_ROWS)
    delaware = await snapshot.stage("DE", 3, {"results": [{"county": "c2", "state": "DE", "votes": 6}]})

    claims = _write_claims(pointers)
    assert len(claims) == 2, f"two writes at one epoch hold one claim each: {claims}"
    assert all(is_write_claim_key(key) for key in claims), f"the hub's purge cannot see these claims: {claims}"
    assert all(pointers.ttls[key] is not None for key in claims), "a claim without a per-key TTL outlives its writer"

    await snapshot.discard_staged([texas])
    assert len(_write_claims(pointers)) == 1, "letting one write go released another write's claim at its epoch"
    await snapshot.discard_staged([delaware])
    assert _write_claims(pointers) == []
    await snapshot.stop()


async def test_a_write_claim_lost_while_its_writer_lives_is_taken_again() -> None:
    snapshot, pointers, _, _ = _snapshot(claim_ttl=timedelta(seconds=1))
    await _started(snapshot, pointers)
    staged = await snapshot.stage("TX", 3, _TX_ROWS)

    _bucket_lost_the_claims(pointers)
    await _until(lambda: bool(_write_claims(pointers)), what="the lost write claim taken again", timeout=3)

    await snapshot.discard_staged([staged])
    assert _write_claims(pointers) == []
    await snapshot.stop()


async def test_a_dead_writers_claims_stay_gone_past_its_renewals() -> None:
    snapshot, pointers, _, _ = _snapshot(claim_ttl=timedelta(seconds=1))
    await _started(snapshot, pointers)
    await snapshot.stage("TX", 3, _TX_ROWS)

    _writer_died(pointers)
    await asyncio.sleep(0.8)  # past two of its renewals

    assert _write_claims(pointers) == [], "a dead writer's claim came back"
    await snapshot.stop()


async def test_publishing_stages_lets_their_claims_go() -> None:
    snapshot, pointers, _, l3 = _retiring_snapshot()
    await _started(snapshot, pointers)
    staged = await snapshot.stage("TX", 2, _TX_ROWS)
    l3.epochs["TX"] = 2

    moved, _ = await snapshot.publish_staged([staged], carry_at={"TX": 1})

    assert moved == ["TX"]
    assert _write_claims(pointers) == [], "a published stage's claim was kept"
    await snapshot.stop()


async def test_a_publish_of_stages_that_fails_still_lets_their_claims_go() -> None:
    snapshot, pointers, _, l3 = _retiring_snapshot()
    await _started(snapshot, pointers)
    staged = await snapshot.stage("TX", 2, _TX_ROWS)
    l3.epochs["TX"] = 2
    pointers.update_fails = ConnectionError("no route to NATS")  # the pointer cannot move

    with pytest.raises(ConnectionError):
        await snapshot.publish_staged([staged], carry_at={"TX": 1})

    pointers.update_fails = None
    assert _write_claims(pointers) == [], "a failed publish left its stages' claims renewing"
    await snapshot.stop()


async def test_stopping_ends_the_claim_of_a_stage_its_writer_dropped() -> None:
    snapshot, pointers, _, _ = _retiring_snapshot()
    await _started(snapshot, pointers)
    await snapshot.stage("TX", 2, _TX_ROWS)  # dropped: never published, never discarded
    assert _write_claims(pointers)

    await snapshot.stop()

    assert _write_claims(pointers) == [], "a dropped stage's claim outlived its snapshot"


def _another_writer_took(pointers: _Pointers, key: str) -> None:
    """another holder's entry is on ``key``: the claim held there is lost, and is not taken back."""
    now = datetime.now(UTC)
    pointers.put_now(
        key,
        json.dumps(
            {
                "holder": "another-writer",
                "expires_at": (now + timedelta(minutes=5)).isoformat(),
                "acquired_at": now.isoformat(),
            }
        ).encode(),
    )


async def test_a_stage_whose_claim_was_lost_moves_no_pointer() -> None:
    snapshot, pointers, _, l3 = _snapshot(claim_ttl=timedelta(seconds=1))
    await _started(snapshot, pointers)
    before = pointers.entries["enr.s.TX"][0]
    staged = await snapshot.stage("TX", 2, _TX_ROWS)
    (claim_key,) = _write_claims(pointers)
    _another_writer_took(pointers, claim_key)
    assert staged.claim is not None
    await _until(staged.claim.lost.is_set, what="the stage's claim lost", timeout=3)
    l3.epochs["TX"] = 2

    moved, skipped = await snapshot.publish_staged([staged], carry_at={"TX": 1})

    assert (moved, skipped) == ([], ["TX"]), "a stage whose claim was lost was published"
    assert pointers.entries["enr.s.TX"][0] == before
    await snapshot.stop()


_NV_ROWS = {"results": [{"county": "c9", "state": "NV", "votes": 4}]}


def _indexed(pointers: _Pointers) -> set[str]:
    return set(json.loads(pointers.entries["enr.index"][0])["scopes"])


async def test_a_claim_lost_just_before_a_later_scopes_move_skips_it_and_keeps_the_scope_already_moved() -> None:
    """two stages: the second's claim is lost while its chunks are checked, after the first moved."""
    snapshot, pointers, store, l3 = _snapshot(claim_ttl=timedelta(seconds=1))
    await _started(snapshot, pointers)
    before = pointers.entries["enr.s.TX"][0]
    nevada = await snapshot.stage("NV", 2, _NV_ROWS)
    texas = await snapshot.stage("TX", 2, _TX_ROWS)
    assert texas.claim is not None
    l3.epochs.update({"NV": 2, "TX": 2})
    real_info = store.info
    taken: list[str] = []

    async def info_while_texas_claim_is_taken(name: str) -> Any:
        if name.startswith("enr/TX/") and not taken:
            assert texas.claim is not None
            taken.append(name)
            _another_writer_took(pointers, texas.claim.key)
            await _until(texas.claim.lost.is_set, what="the second stage's claim lost", timeout=3)
        return await real_info(name)

    store.info = info_while_texas_claim_is_taken  # type: ignore[method-assign]

    moved, skipped = await snapshot.publish_staged([nevada, texas], carry_at={"NV": 0, "TX": 1})

    assert taken, "the second stage's chunks were never checked"
    assert (moved, skipped) == (["NV"], ["TX"]), "a claim lost before the move was raised, not skipped"
    assert "NV" in _indexed(pointers), "a scope whose pointer moved is not in the index"
    assert pointers.entries["enr.s.TX"][0] == before, "a pointer moved after its write claim was lost"
    await snapshot.stop()


async def test_a_publish_that_raises_after_one_scope_moved_still_indexes_it_and_lets_every_claim_go() -> None:
    snapshot, pointers, store, l3 = _retiring_snapshot()
    await _started(snapshot, pointers)
    nevada = await snapshot.stage("NV", 2, _NV_ROWS)
    texas = await snapshot.stage("TX", 2, _TX_ROWS)
    l3.epochs.update({"NV": 2, "TX": 2})
    real_update = pointers.update

    async def texas_pointer_unreachable(*, key: str, value: bytes, revision: int, ttl: timedelta | None = None) -> Any:
        if key == "enr.s.TX":
            raise ConnectionError("no route to NATS")
        return await real_update(key=key, value=value, revision=revision, ttl=ttl)

    pointers.update = texas_pointer_unreachable  # type: ignore[method-assign]

    with pytest.raises(ConnectionError):
        await snapshot.publish_staged([nevada, texas], carry_at={"NV": 0, "TX": 1})

    assert "enr.s.NV" in pointers.entries, "the first scope's pointer did not move"
    assert "NV" in _indexed(pointers), "a scope whose pointer moved before the raise is not in the index"
    assert _write_claims(pointers) == [], "a publish that raised left claims renewing"
    texas_chunks = set(texas.objects.values())
    assert texas_chunks <= set(store.retired), "the stage that never moved was not retired"
    assert not set(nevada.objects.values()) & set(store.retired), "a chunk the moved scope serves was retired"
    await snapshot.stop()


async def test_a_publish_whose_claim_is_lost_mid_write_moves_no_pointer() -> None:
    snapshot, pointers, store, _ = _snapshot(claim_ttl=timedelta(seconds=1))
    await _started(snapshot, pointers)
    before = pointers.entries["enr.s.TX"][0]
    real_put = store.put

    async def put_while_the_claim_is_taken(name: str, data: bytes) -> None:
        for key in _write_claims(pointers):
            _another_writer_took(pointers, key)
        await asyncio.sleep(0.8)  # past the claim's next renewal, which finds another holder
        await real_put(name, data)

    store.put = put_while_the_claim_is_taken  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="was lost"):
        await snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 9}]})

    assert pointers.entries["enr.s.TX"][0] == before, "a pointer moved after its write claim was lost"
    await snapshot.stop()


async def test_a_chunk_swept_while_its_claim_lapsed_unseen_moves_no_pointer() -> None:
    """a replica cut off from NATS past its claim's lifetime still reads its claim held; a sweep on
    another replica took what it wrote, so the pointer must not move onto it."""
    snapshot, pointers, store, _ = _snapshot()
    await _started(snapshot, pointers)
    before = pointers.entries["enr.s.TX"][0]
    real_put = store.put

    async def put_then_swept(name: str, data: bytes) -> None:
        await real_put(name, data)
        del store.objects[name]  # the sweep, while this replica's claim had lapsed

    store.put = put_then_swept  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="no pointer is moved onto a missing chunk"):
        await snapshot.publish("TX", 2, {"results": [{"county": "c1", "state": "TX", "votes": 9}]})

    assert pointers.entries["enr.s.TX"][0] == before, "a pointer moved onto a chunk that is gone"
    await snapshot.stop()


async def test_stopping_ends_a_rebuild_claim_still_held() -> None:
    snapshot, pointers, _, _ = _snapshot()
    await _started(snapshot, pointers)
    async with snapshot.holding_rebuilds() as held:
        assert held and "enr.rebuild" in pointers.entries
        await snapshot.stop()
        assert "enr.rebuild" not in pointers.entries, "a stopped snapshot still holds the rebuild claim"


async def test_a_lost_rebuild_claim_is_said_and_never_deleted_from_its_new_holder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    snapshot, pointers, _, _ = _snapshot(claim_ttl=timedelta(seconds=1))
    await _started(snapshot, pointers)
    caplog.set_level(logging.WARNING)
    async with snapshot.holding_rebuilds() as held:
        assert held
        _another_writer_took(pointers, "enr.rebuild")
        await asyncio.sleep(0.8)  # past its next renewal, which finds another holder

    assert "the rebuild claim was lost" in caplog.text
    assert b"another-writer" in pointers.entries["enr.rebuild"][0], "the new holder's claim was deleted"
    await snapshot.stop()


# ----------------------------------------------------------------------
# pieces: a table's base chunk and the rows added after it
# ----------------------------------------------------------------------


def _results_chunk(rows: list[dict[str, Any]], scope: str = "TX") -> bytes:
    """a scope's results rows as a chunk, typed by the L1's own columns."""
    return encode_chunk(
        _backend().export_rows("results", "state", scope, rows, primary_key=("county",), order_by=("county",))
    )


def _county(county: str, votes: int, state: str = "TX") -> dict[str, Any]:
    return {"county": county, "state": state, "votes": votes}


def _with_pieces(
    pointers: _Pointers,
    store: _Store,
    *,
    epoch: int,
    pieces: list[tuple[int, list[dict[str, Any]]]],
    mark: str = "+pieces",
) -> dict[str, Any]:
    """move TX's pointer to ``epoch``, its base kept and ``pieces`` (epoch, rows) after it, as a writer of pieces would."""
    held = json.loads(pointers.entries["enr.s.TX"][0])
    digest = held["schema"]["results"]
    listed = []
    for at, rows in pieces:
        name = f"enr.p/TX/{at}/results.{digest}"
        store.objects[name] = _results_chunk(rows)
        listed.append({"object": name, "rows": len(rows), "epoch": at, "bytes": len(store.objects[name])})
    held["epoch"] = epoch
    held["tables"]["results"]["pieces"] = listed
    held["schema"]["results"] = f"{digest}{mark}"
    pointers.put_now("enr.s.TX", json.dumps(held).encode())
    return held


def _tx_counties(snapshot: ScopedSnapshot) -> list[tuple[str, int]]:
    with snapshot.read() as cursor:
        return cursor.execute("SELECT county, votes FROM results WHERE state = 'TX' ORDER BY county").fetchall()


async def test_a_pointer_with_no_pieces_is_written_in_the_shape_a_replica_that_predates_pieces_reads() -> None:
    snapshot, pointers, _, _ = _snapshot()
    await snapshot.start()
    await snapshot.wait_ready(timeout=5)

    written = json.loads(pointers.entries["enr.s.TX"][0])

    assert set(written) == {"scope", "epoch", "tables", "schema", "at"}
    assert set(written["tables"]["results"]) == {"object", "rows"}
    assert written["schema"]["results"] == snapshot.backend.schema_digest("results")
    await snapshot.stop()


async def test_a_cold_replica_loads_a_tables_base_and_every_piece_after_it() -> None:
    first, pointers, store, _ = _snapshot()
    await first.start()
    await first.wait_ready(timeout=5)
    await first.stop()
    _with_pieces(pointers, store, epoch=3, pieces=[(2, [_county("c3", 3)]), (3, [_county("c4", 4), _county("c5", 5)])])

    second, _, _, l3 = _snapshot(pointers=pointers, store=store)
    statements = l3.statements
    await second.start()
    await second.wait_ready(timeout=5)

    assert second.status().source is SnapshotSource.L2
    assert l3.statements == statements, "a cold replica read L3 although NATS held base and pieces"
    assert _tx_counties(second) == [("c1", 1), ("c3", 3), ("c4", 4), ("c5", 5)]
    assert second.applied_epoch("TX") == 3
    assert second.status().rows == {"results": 5}
    await second.stop()


async def test_a_pointer_marked_by_a_format_this_replica_does_not_know_is_served_from_l3_and_left_standing() -> None:
    first, pointers, store, _ = _snapshot()
    await first.start()
    await first.wait_ready(timeout=5)
    await first.stop()
    # a later format: as a replica that predates pieces sees a pointer with them, its columns are not ours
    _with_pieces(pointers, store, epoch=1, pieces=[(1, [_county("c3", 3)])], mark="+pieces-v9")
    standing = pointers.entries["enr.s.TX"][0]

    second, _, _, _ = _snapshot(pointers=pointers, store=store)
    await second.start()
    await second.wait_ready(timeout=5)

    assert second.status().source is SnapshotSource.L3
    assert _tx_counties(second) == [("c1", 1)], "a pointer of another format was loaded"
    assert pointers.entries["enr.s.TX"][0] == standing, "the other format's pointer was repointed at its epoch"
    await second.stop()


async def test_a_full_store_sweep_keeps_every_piece_a_pointer_serves_and_takes_one_it_no_longer_does() -> None:
    snapshot, pointers, store, l3 = _retiring_snapshot()
    await _started(snapshot, pointers)
    held = _with_pieces(pointers, store, epoch=3, pieces=[(2, [_county("c3", 3)]), (3, [_county("c4", 4)])])
    await _until(lambda: snapshot.applied_epoch("TX") == 3, what="TX's pieces applied")
    folded = f"enr.p/TX/1/results.{snapshot.backend.schema_digest('results')}"
    store.objects[folded] = _results_chunk([_county("c2", 2)])

    await _fill_and_sweep(snapshot, store, l3)

    served = [held["tables"]["results"]["object"], *(p["object"] for p in held["tables"]["results"]["pieces"])]
    assert set(served) <= set(store.objects), "the sweep took a base or a piece a pointer serves"
    assert folded in store.retired, "a piece no pointer names outlived the sweep"
    await snapshot.stop()


# ----------------------------------------------------------------------
# append-only tables: a write publishes the rows it added; a replica appends them
# ----------------------------------------------------------------------

_APPEND = (SnapshotTable(name="results", scope_column="state", key=("county",), append_only=True),)


class _HeldWatch(_Pointers):
    """a pointer bucket whose watch can lose or reorder what it delivers, while KV itself stays right."""

    def __init__(self) -> None:
        super().__init__()
        self.holding = False
        self.held: list[KvKeyUpdate] = []

    def _put(self, key: str, value: bytes) -> int:
        if not self.holding:
            return super()._put(key, value)
        self._revision += 1
        self.entries[key] = (value, self._revision)
        self.held.append(KvKeyUpdate(key=key, value=value, revision=self._revision))
        return self._revision

    def deliver(self, updates: list[KvKeyUpdate]) -> None:
        """deliver these held updates, in this order; the rest are lost."""
        self.holding, self.held = False, []
        for update in updates:
            self.broadcast(update)

    def held_pointers(self, scope: str = "TX") -> list[KvKeyUpdate]:
        return [u for u in self.held if u.key == f"enr.s.{scope}"]


async def _append_pair(
    pointers: _Pointers | None = None, **options: Any
) -> tuple[ScopedSnapshot, ScopedSnapshot, _Pointers, _Store, _L3]:
    """a writer and a second replica over the same buckets, both ready on the first rebuild's TX c1 and DE c2.

    The store returned is the second replica's own view of the objects, so its reads are counted
    apart from the writer's (which applies its own pointers from its watch too).
    """
    writer, pointers, store, l3 = _snapshot(tables=_APPEND, pointers=pointers, **options)
    await _started(writer, pointers)
    view = _Store()
    view.objects = store.objects
    replica, _, _, _ = _snapshot(tables=_APPEND, pointers=pointers, store=view, l3=l3)  # one L3, as in life
    await replica.start()
    await replica.wait_ready(timeout=5)
    return writer, replica, pointers, view, l3


async def _append_staged(writer: ScopedSnapshot, epoch: int, rows: list[dict[str, Any]], carry_at: int) -> None:
    staged = await writer.stage("TX", epoch, {}, added={"results": rows})
    moved, skipped = await writer.publish_staged([staged], carry_at={"TX": carry_at})
    assert (moved, skipped) == (["TX"], [])


async def test_a_staged_append_moves_the_pointer_and_a_second_replica_appends_exactly_the_new_rows() -> None:
    writer, replica, pointers, store, _ = await _append_pair()
    base = json.loads(pointers.entries["enr.s.TX"][0])["tables"]["results"]["object"]
    gets = store.gets

    await _append_staged(writer, 2, [_county("c3", 3), _county("c4", 4)], carry_at=1)

    await _until(lambda: replica.applied_epoch("TX") == 2, what="the second replica applied epoch 2")
    assert _tx_counties(replica) == [("c1", 1), ("c3", 3), ("c4", 4)]
    assert store.gets - gets == 1, "the replica fetched more than the one piece the write added"
    assert replica.status().rows == {"results": 4}
    assert replica.status().last_change is not None and replica.status().last_change.rows == 3
    written = json.loads(pointers.entries["enr.s.TX"][0])
    entry = written["tables"]["results"]
    assert entry["object"] == base and base.startswith("enr/"), "the base moved, or left the old replicas' prefix"
    assert [p["object"].split("/")[0] for p in entry["pieces"]] == ["enr.p"]
    assert entry["pieces"][0]["rows"] == 2 and entry["pieces"][0]["epoch"] == 2
    # a replica that predates pieces compares this with its own digest, and so never loads the scope
    assert written["schema"]["results"] != writer.backend.schema_digest("results")
    await writer.stop()
    await replica.stop()


async def test_a_publish_of_added_rows_appends_them_here_and_on_every_replica() -> None:
    writer, replica, pointers, store, _ = await _append_pair()
    gets = store.gets

    assert await writer.publish("TX", 2, {}, added={"results": [_county("c3", 3)]}, after=1)
    assert await writer.publish("TX", 3, {}, added={"results": [_county("c4", 4)]}, after=2)

    assert _tx_counties(writer) == [("c1", 1), ("c3", 3), ("c4", 4)]
    await _until(lambda: replica.applied_epoch("TX") == 3, what="the second replica applied epoch 3")
    assert _tx_counties(replica) == [("c1", 1), ("c3", 3), ("c4", 4)]
    assert store.gets - gets <= 2, "the replica fetched more than the two pieces"
    assert len(json.loads(pointers.entries["enr.s.TX"][0])["tables"]["results"]["pieces"]) == 2
    await writer.stop()
    await replica.stop()


async def test_a_cold_replica_after_appends_loads_the_base_and_every_piece() -> None:
    writer, replica, pointers, store, l3 = await _append_pair()
    await replica.stop()
    await _append_staged(writer, 2, [_county("c3", 3)], carry_at=1)
    await _append_staged(writer, 3, [_county("c4", 4)], carry_at=2)

    cold, _, _, cold_l3 = _snapshot(tables=_APPEND, pointers=pointers, store=store)
    await cold.start()
    await cold.wait_ready(timeout=5)

    assert cold.status().source is SnapshotSource.L2
    assert _tx_counties(cold) == [("c1", 1), ("c3", 3), ("c4", 4)]
    await writer.stop()
    await cold.stop()


async def test_a_replica_that_lost_pointer_events_fetches_every_piece_after_what_it_holds() -> None:
    writer, replica, pointers, store, _ = await _append_pair(pointers=_HeldWatch())
    assert isinstance(pointers, _HeldWatch)
    pointers.holding = True
    await _append_staged(writer, 2, [_county("c3", 3)], carry_at=1)
    await _append_staged(writer, 3, [_county("c4", 4)], carry_at=2)
    gets = store.gets

    pointers.deliver(pointers.held_pointers()[-1:])  # epoch 2's event is lost

    await _until(lambda: replica.applied_epoch("TX") == 3, what="the replica applied epoch 3")
    assert _tx_counties(replica) == [("c1", 1), ("c3", 3), ("c4", 4)]
    assert store.gets - gets == 2, "the replica did not fetch exactly the two pieces after what it held"
    await writer.stop()
    await replica.stop()


async def test_pointer_events_delivered_out_of_order_converge_on_the_latest() -> None:
    writer, replica, pointers, _, _ = await _append_pair(pointers=_HeldWatch())
    assert isinstance(pointers, _HeldWatch)
    pointers.holding = True
    await _append_staged(writer, 2, [_county("c3", 3)], carry_at=1)
    await _append_staged(writer, 3, [_county("c4", 4)], carry_at=2)

    pointers.deliver(list(reversed(pointers.held_pointers())))

    await _until(lambda: replica.applied_epoch("TX") == 3, what="the replica applied epoch 3")
    await asyncio.sleep(0.2)  # the older event has been seen too, and changed nothing
    assert replica.applied_epoch("TX") == 3
    assert _tx_counties(replica) == [("c1", 1), ("c3", 3), ("c4", 4)]
    await writer.stop()
    await replica.stop()


async def test_a_rebuild_resets_an_append_only_table_to_one_base_and_retires_its_pieces() -> None:
    store = _BoundedStore(10**9)

    async def retire(names: list[str]) -> None:
        for name in names:
            store.retired.append(name)
            store.objects.pop(name, None)

    writer, replica, pointers, _, l3 = await _append_pair(store=store, retire=retire)
    await _append_staged(writer, 2, [_county("c3", 3)], carry_at=1)
    await _until(lambda: replica.applied_epoch("TX") == 2, what="epoch 2 applied")
    pieces = [p["object"] for p in json.loads(pointers.entries["enr.s.TX"][0])["tables"]["results"]["pieces"]]
    l3.rows = [*l3.rows, _county("c3", 3), _county("c5", 5)]
    l3.epochs["TX"] = 4  # a write committed in L3 and died before it published

    assert await writer.catch_up_from_l3() == ["TX"]

    written = json.loads(pointers.entries["enr.s.TX"][0])
    assert set(written["tables"]["results"]) == {"object", "rows", "compacted_at"}, "a rebuild left pieces"
    assert written["schema"]["results"] == writer.backend.schema_digest("results")
    assert set(pieces) <= set(store.retired), "the pieces the rebuild superseded were kept"
    await _until(lambda: replica.applied_epoch("TX") == 4, what="the rebuilt base applied")
    assert _tx_counties(replica) == [("c1", 1), ("c3", 3), ("c5", 5)]
    await writer.stop()
    await replica.stop()


async def test_a_staged_piece_off_the_epoch_its_writer_saw_moves_nothing_and_is_left_for_the_catch_up() -> None:
    writer, replica, pointers, _, _ = await _append_pair()
    standing = pointers.entries["enr.s.TX"][0]
    staged = await writer.stage("TX", 3, {}, added={"results": [_county("c3", 3)]})

    moved, skipped = await writer.publish_staged([staged], carry_at={"TX": 2})  # the pointer is at 1

    assert (moved, skipped) == ([], ["TX"])
    assert pointers.entries["enr.s.TX"][0] == standing
    await writer.stop()
    await replica.stop()


async def test_a_publish_on_a_chain_this_replica_does_not_hold_moves_nothing() -> None:
    writer, replica, pointers, store, _ = await _append_pair()
    pointers.put_now("enr.rebuild", b"another replica")  # nothing here may rebuild TX meanwhile
    foreign = json.loads(pointers.entries["enr.s.TX"][0])
    foreign["epoch"], foreign["schema"]["results"] = 2, "the other version's columns"
    pointers.put_now("enr.s.TX", json.dumps(foreign).encode())
    standing = pointers.entries["enr.s.TX"][0]

    landed = await writer.publish("TX", 3, {}, added={"results": [_county("c3", 3)]}, after=1)

    assert landed is False
    assert pointers.entries["enr.s.TX"][0] == standing, "a piece was appended to a chain the writer does not hold"
    assert not [name for name in store.objects if name.startswith("enr.p/")]
    assert "TX" in writer.status().behind and "rebuilt from L3" in writer.status().behind["TX"]
    assert writer.applied_epoch("TX") == 1
    await writer.stop()
    await replica.stop()


async def test_a_piece_holding_a_row_already_held_is_refused_and_the_scope_rebuilt_from_l3() -> None:
    writer, replica, pointers, store, l3 = await _append_pair()
    l3.epochs["TX"] = 2

    # a writer that broke the promise, past this library's own refusal: c1 is held already
    _with_pieces(pointers, store, epoch=2, pieces=[(2, [_county("c1", 9)])])

    # the epoch is applied before the rebuild hands the phase back, so both are waited for:
    # asserting READY the moment the epoch moved failed on a loaded runner, mid-handback
    await _until(
        lambda: replica.applied_epoch("TX") == 2 and replica.status().phase is SnapshotPhase.READY,
        what="TX rebuilt from L3 after its piece was refused, and the replica ready again",
    )
    assert _tx_counties(replica) == [("c1", 1)], "the refused piece's row reached the L1"
    await writer.stop()
    await replica.stop()


async def test_added_rows_already_held_are_refused_before_anything_is_written() -> None:
    writer, replica, pointers, store, _ = await _append_pair()
    standing, objects = pointers.entries["enr.s.TX"][0], set(store.objects)

    # a caller giving every row of the append-only table as added rows, c1 among them
    every = [_county("c1", 1), _county("c3", 3)]
    with pytest.raises(ValueError, match="held already"):
        await writer.stage("TX", 2, {}, added={"results": every})
    with pytest.raises(ValueError, match="held already"):
        await writer.publish("TX", 2, {}, added={"results": every}, after=1)

    assert pointers.entries["enr.s.TX"][0] == standing
    assert set(store.objects) == objects, "a refused write left a chunk behind"
    assert _tx_counties(writer) == [("c1", 1)]
    assert not [k for k in pointers.entries if k.startswith("enr.w.")], "a refused write kept a claim"
    await writer.stop()
    await replica.stop()


@pytest.mark.parametrize(
    ("rows", "added", "after", "match"),
    [
        ({}, None, None, "gives no rows for"),
        ({"results": []}, {"results": []}, 1, "both rows and added"),
        ({}, {"results": [_county("c3", 3)]}, None, "after="),
        ({"results": [_county("c1", 1)]}, None, 1, "adds none"),
    ],
)
async def test_a_publish_names_each_table_once_and_added_rows_with_the_epoch_they_follow(
    rows: Any, added: Any, after: Any, match: str
) -> None:
    writer, pointers, _, _ = _snapshot(tables=_APPEND)
    await _started(writer, pointers)

    with pytest.raises(ValueError, match=match):
        await writer.publish("TX", 2, rows, added=added, after=after)
    await writer.stop()


async def test_added_rows_for_a_table_not_declared_append_only_are_refused() -> None:
    writer, pointers, _, _ = _snapshot()
    await _started(writer, pointers)

    with pytest.raises(ValueError, match="not append-only"):
        await writer.stage("TX", 2, {}, added={"results": [_county("c3", 3)]})
    await writer.stop()


_TOTALS = TableSchema(
    name="totals",
    primary_key=("state",),
    columns=[Column("state", STRING_TYPE), Column("total", BIGINT_TYPE, nullable=True)],
    on_conflict="update",
)


def _two_table_backend() -> DuckDBBackend:
    metadata = MetaData()
    _RESULTS.to_sqlalchemy_table(metadata)
    _TOTALS.to_sqlalchemy_table(metadata)
    backend = DuckDBBackend()
    backend.initialize(metadata)
    return backend


class _TwoTableL3(_L3):
    """L3 holding the append-only results and a whole-chunk totals table."""

    def __init__(self) -> None:
        super().__init__([_county("c1", 1), _county("c2", 2, state="DE")])
        self.totals = [{"state": "TX", "total": 1}, {"state": "DE", "total": 2}]

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        if '"totals"' not in sql:
            return await super().fetch(sql, *args)
        self.statements += 1
        if "DISTINCT" in sql:
            return [{"scope": t["state"]} for t in self.totals]
        return [t for t in self.totals if not args or t["state"] == args[0]]


async def test_a_table_not_declared_append_only_keeps_one_whole_chunk_beside_an_appended_one() -> None:
    tables = (*_APPEND, SnapshotTable(name="totals", scope_column="state", key=("state",)))
    pointers, store = _Pointers(), _Store()
    replicas = []
    for _ in range(2):
        snapshot, _, _, _ = _snapshot(
            tables=tables, pointers=pointers, store=store, backend=_two_table_backend(), l3=_TwoTableL3()
        )
        replicas.append(snapshot)
    writer, replica = replicas
    await _started(writer, pointers)
    await replica.start()
    await replica.wait_ready(timeout=5)

    landed = await writer.publish(
        "TX", 2, {"totals": [{"state": "TX", "total": 4}]}, added={"results": [_county("c3", 3)]}, after=1
    )
    assert landed

    await _until(lambda: replica.applied_epoch("TX") == 2, what="the second replica applied epoch 2")
    entry = json.loads(pointers.entries["enr.s.TX"][0])
    assert set(entry["tables"]["totals"]) == {"object", "rows"}, "the whole-chunk table took pieces"
    assert entry["tables"]["totals"]["object"].startswith("enr/TX/2/totals.")
    assert entry["schema"]["totals"] == writer.backend.schema_digest("totals")
    assert len(entry["tables"]["results"]["pieces"]) == 1
    with replica.read() as cursor:
        assert cursor.execute("SELECT state, total FROM totals ORDER BY state").fetchall() == [("DE", 2), ("TX", 4)]
    assert _tx_counties(replica) == [("c1", 1), ("c3", 3)]
    await writer.stop()
    await replica.stop()


# ----------------------------------------------------------------------
# compaction: past a threshold a writer folds a scope's pieces into its base
# ----------------------------------------------------------------------


async def _retiring_append_pair(
    **options: Any,
) -> tuple[ScopedSnapshot, ScopedSnapshot, _HeldWatch, _Store, _BoundedStore]:
    store = _BoundedStore(10**9)

    async def retire(names: list[str]) -> None:
        for name in names:
            store.retired.append(name)
            store.objects.pop(name, None)

    writer, replica, pointers, view, _ = await _append_pair(
        pointers=_HeldWatch(), store=store, retire=retire, **options
    )
    assert isinstance(pointers, _HeldWatch)
    return writer, replica, pointers, view, store


def _tx_entry(pointers: _Pointers) -> dict[str, Any]:
    return dict(json.loads(pointers.entries["enr.s.TX"][0])["tables"]["results"])


async def test_past_the_threshold_a_writer_folds_the_pieces_into_a_base_and_retires_them() -> None:
    writer, replica, pointers, view, store = await _retiring_append_pair(compact_pieces=3)
    await _append_staged(writer, 2, [_county("c3", 3)], carry_at=1)
    await _append_staged(writer, 3, [_county("c4", 4)], carry_at=2)
    assert len(_tx_entry(pointers)["pieces"]) == 2, "it folded below the threshold"
    await _until(lambda: replica.applied_epoch("TX") == 3, what="epoch 3 applied")
    pieces = [p["object"] for p in _tx_entry(pointers)["pieces"]]

    pointers.holding = True
    await _append_staged(writer, 4, [_county("c5", 5)], carry_at=3)
    unfolded, folded_event = pointers.held_pointers()
    folded = _tx_entry(pointers)

    assert set(folded) == {"object", "rows", "compacted_at"}, "the third piece did not fold the chain"
    assert folded["object"] == f"enr/TX/4/results.{writer.backend.schema_digest('results')}"
    assert folded["rows"] == 4
    assert {*pieces, f"enr.p/TX/4/results.{writer.backend.schema_digest('results')}"} <= set(store.retired)
    assert folded["object"] in store.objects
    # a replica sees the scope at epoch 4 with its pieces, then the fold at the same epoch
    pointers.deliver([unfolded])
    await _until(lambda: replica.applied_epoch("TX") == 4, what="epoch 4 applied")
    gets = view.gets
    pointers.deliver([folded_event])
    await _until(lambda: replica.status().appended["results"]["TX"].pieces == 0, what="the fold adopted")
    assert view.gets == gets, "a replica holding the scope at the folded epoch fetched the fold"
    assert _tx_counties(replica) == [("c1", 1), ("c3", 3), ("c4", 4), ("c5", 5)]
    await writer.stop()
    await replica.stop()


async def test_a_cold_replica_after_a_fold_loads_one_base() -> None:
    writer, replica, pointers, _, store = await _retiring_append_pair(compact_pieces=2)
    await replica.stop()
    await _append_staged(writer, 2, [_county("c3", 3)], carry_at=1)
    await _append_staged(writer, 3, [_county("c4", 4)], carry_at=2)
    view = _Store()
    view.objects = store.objects

    cold, _, _, _ = _snapshot(tables=_APPEND, pointers=pointers, store=view)
    await cold.start()
    await cold.wait_ready(timeout=5)

    assert view.gets == 2, "a cold replica read more than one base for each of TX and DE"
    assert _tx_counties(cold) == [("c1", 1), ("c3", 3), ("c4", 4)]
    await writer.stop()
    await cold.stop()


async def test_a_fold_that_loses_its_race_moves_nothing_and_leaves_nothing_it_wrote_served() -> None:
    writer, replica, pointers, _, store = await _retiring_append_pair(compact_pieces=2)
    real_put = store.put
    racer: list[bytes] = []

    async def racing_put(name: str, data: bytes) -> None:
        await real_put(name, data)
        if name.startswith("enr/TX/3/") and not racer:
            # another writer appends at epoch 4 while the fold writes its base at epoch 3
            later = json.loads(pointers.entries["enr.s.TX"][0])
            later["epoch"] = 4
            pointers.put_now("enr.s.TX", json.dumps(later).encode())
            racer.append(pointers.entries["enr.s.TX"][0])

    store.put = racing_put  # type: ignore[method-assign]

    await _append_staged(writer, 2, [_county("c3", 3)], carry_at=1)
    await _append_staged(writer, 3, [_county("c4", 4)], carry_at=2)

    assert racer and pointers.entries["enr.s.TX"][0] == racer[0], "the fold moved a pointer that moved under it"
    assert f"enr/TX/3/results.{writer.backend.schema_digest('results')}" in store.retired
    await writer.stop()
    await replica.stop()


async def test_the_status_says_each_append_only_scopes_pieces_and_last_compaction() -> None:
    writer, replica, _, _, _ = await _retiring_append_pair(compact_pieces=3)
    rebuilt = writer.status().appended["results"]["TX"]
    assert rebuilt.pieces == 0 and rebuilt.compacted_at is not None, "a rebuild's base is not a compaction"

    await _append_staged(writer, 2, [_county("c3", 3)], carry_at=1)
    await _append_staged(writer, 3, [_county("c4", 4)], carry_at=2)
    await _until(lambda: replica.applied_epoch("TX") == 3, what="epoch 3 applied")
    appended = replica.status().appended["results"]
    assert appended["TX"].pieces == 2 and appended["TX"].compacted_at == rebuilt.compacted_at
    assert appended["DE"].pieces == 0

    await _append_staged(writer, 4, [_county("c5", 5)], carry_at=3)
    await _until(lambda: writer.status().appended["results"]["TX"].pieces == 0, what="the fold applied here")
    folded_at = writer.status().appended["results"]["TX"].compacted_at
    assert folded_at is not None and rebuilt.compacted_at is not None and folded_at > rebuilt.compacted_at
    assert _snapshot()[0].status().appended == {}, "a snapshot with no append-only table reports pieces"
    await writer.stop()
    await replica.stop()


# ----------------------------------------------------------------------
# a publish of added rows that cannot land: nothing is lost, every replica is repaired from L3
# ----------------------------------------------------------------------


async def test_a_later_write_published_before_an_earlier_one_loses_neither_writes_rows() -> None:
    """write B commits TX at epoch 2 (adding c3), write A at epoch 3 (adding c4, after B); A publishes
    first. A's rows follow epoch 2, not the chain at 1, so A's publish lands nowhere and leaves TX for
    the rebuild from L3; B's publish then lands or loses to that rebuild. Every replica ends with both writes."""
    writer_b, writer_a, pointers, view, l3 = await _append_pair()
    reader_view = _Store()
    reader_view.objects = view.objects
    reader, _, _, _ = _snapshot(tables=_APPEND, pointers=pointers, store=reader_view, l3=l3)
    await reader.start()
    await reader.wait_ready(timeout=5)
    l3.rows = [*l3.rows, _county("c3", 3), _county("c4", 4)]
    l3.epochs["TX"] = 3

    a_landed = await writer_a.publish("TX", 3, {}, added={"results": [_county("c4", 4)]}, after=2)
    b_landed = await writer_b.publish("TX", 2, {}, added={"results": [_county("c3", 3)]}, after=1)

    assert a_landed is False, "a piece landed on a chain it was not computed after"
    for replica in (writer_a, writer_b, reader):
        await _until(
            lambda r=replica: r.applied_epoch("TX") == 3 and _tx_counties(r) == [("c1", 1), ("c3", 3), ("c4", 4)],
            what="every replica holding both writes' rows at epoch 3",
        )
        assert replica.status().behind == {}
    # whether B's own publish lands is a race, not the property: A's refusal hands TX to the rebuild from
    # L3, which may move the pointer to 3 while B writes, and then B's compare-and-set loses and B leaves
    # TX to L3 as well. The property is the state every replica ends in, asserted above.
    del b_landed
    for replica in (writer_a, writer_b, reader):
        await replica.stop()


async def test_a_publish_that_loses_its_pointer_race_records_nothing_here_says_so_and_is_repaired_from_l3() -> None:
    writer, replica, pointers, view, l3 = await _append_pair(pointers=_HeldWatch())
    assert isinstance(pointers, _HeldWatch)
    real_put = view.put
    raced: list[bool] = []

    async def racing_put(name: str, data: bytes) -> None:
        await real_put(name, data)
        if name.startswith("enr.p/TX/2/") and not raced:
            raced.append(True)
            # another replica's rebuild from L3 republishes TX whole at epoch 2 while this piece is written
            moved = json.loads(pointers.entries["enr.s.TX"][0])
            whole = "enr/TX/2/results.rebuilt"
            view.objects[whole] = _results_chunk([_county("c1", 1), _county("c3", 3)])
            moved["epoch"], moved["tables"]["results"] = 2, {"object": whole, "rows": 2}
            pointers.put_now("enr.s.TX", json.dumps(moved).encode())

    view.put = racing_put  # type: ignore[method-assign]
    pointers.put_now("enr.rebuild", b"another replica")  # no rebuild here while the signal is read
    l3.rows = [*l3.rows, _county("c3", 3)]
    l3.epochs["TX"] = 2
    before = replica.status().last_change
    pointers.holding = True  # the racing pointer's event has not reached this replica yet

    landed = await replica.publish("TX", 2, {}, added={"results": [_county("c3", 3)]}, after=1)

    assert raced and landed is False
    assert replica.applied_epoch("TX") == 1, "the replica recorded an epoch the shared pointer never took"
    assert replica.status().last_change == before
    assert _tx_counties(replica) == [("c1", 1)], "the L1 took rows the shared pointer does not name"
    assert "TX" in replica.status().behind and "rebuilt from L3" in replica.status().behind["TX"]
    await asyncio.sleep(0.2)  # several passes: the scope stays shown behind until something lands it
    assert "TX" in replica.status().behind
    del pointers.entries["enr.rebuild"]
    pointers.deliver(pointers.held)
    for snapshot in (replica, writer):
        await _until(
            lambda s=snapshot: s.status().behind == {} and _tx_counties(s) == [("c1", 1), ("c3", 3)],
            what="TX holding the lost write's rows",
        )
    await writer.stop()
    await replica.stop()


async def test_a_whole_publish_in_a_snapshot_with_an_append_only_table_keeps_the_retried_move() -> None:
    writer, replica, pointers, view, _ = await _append_pair()

    landed = await writer.publish("TX", 2, {"results": [_county("c1", 1), _county("c3", 3)]})

    assert landed is True
    entry = _tx_entry(pointers)
    assert "pieces" not in entry and entry["object"].startswith("enr/TX/2/"), "a whole publish wrote a piece"
    await _until(lambda: _tx_counties(replica) == [("c1", 1), ("c3", 3)], what="the whole publish applied")
    await writer.stop()
    await replica.stop()


# ----------------------------------------------------------------------
# bounded memory: a rebuild and a load hold a few scopes at a time
# ----------------------------------------------------------------------

#: rows a keyset page keeps (the read's default: one under the L3 rail's cap)
_PAGE = 999


class _PagedL3(_L3):
    """L3 holding ``sizes[state]`` rows per state, answering each keyset page as the rail would, and
    counting the rows it has handed over."""

    def __init__(self, sizes: dict[str, int]) -> None:
        super().__init__([])
        self.sizes = dict(sizes)
        self.served = 0
        # how many statements began each state's read (a read's first page, no cursor)
        self.reads: dict[str, int] = dict.fromkeys(sizes, 0)
        self.failing: str | None = None
        self.on_read: Callable[[str], None] | None = None
        self.alive: Callable[[], int] | None = None
        self.peak_alive = 0
        self.events: list[str] = []

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        self.statements += 1
        if "DISTINCT" in sql:
            return [{"scope": s} for s in sorted(self.sizes)]
        state = args[0]
        limit = int(sql.rsplit("LIMIT", 1)[1])
        start = 0 if len(args) == 1 else int(args[1].rsplit("-", 1)[1]) + 1
        if start == 0:
            self.reads[state] += 1
            self.events.append(f"read {state}")
            if self.on_read is not None:
                self.on_read(state)
            if state == self.failing:
                raise ConnectionError(f"L3 is unreachable reading {state}")
        page = [
            {"county": f"{state}-{i:06d}", "state": state, "votes": i}
            for i in range(start, min(self.sizes[state], start + limit))
        ]
        self.served += len(page)
        if self.alive is not None:
            self.peak_alive = max(self.peak_alive, self.alive())
        return page


class _CountingBackend(DuckDBBackend):
    """a DuckDB L1 that counts the rows committed into it and records each scope's commit."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.committed = 0
        self.events = events

    def replace_partitions(self, replacements: Any) -> int:
        written = super().replace_partitions(replacements)
        self.committed += written
        self.events.extend(f"commit {r.value}" for r in replacements)
        return written


def _paged(
    sizes: dict[str, int], **options: Any
) -> tuple[ScopedSnapshot, _Pointers, _Store, _PagedL3, _CountingBackend]:
    l3 = _PagedL3(sizes)
    metadata = MetaData()
    _RESULTS.to_sqlalchemy_table(metadata)
    backend = _CountingBackend(l3.events)
    backend.initialize(metadata)
    snapshot, pointers, store, _ = _snapshot(l3=l3, backend=backend, **options)
    l3.epochs = dict.fromkeys(sizes, 1)
    # every row L3 handed over that this L1 has not taken yet: what the rebuild holds
    l3.alive = lambda: l3.served - backend.committed
    return snapshot, pointers, store, l3, backend


_SIZES = {"DE": 2500, "NV": 2500, "TX": 2500}


async def test_a_rebuild_holds_one_scope_and_a_page_at_a_time_not_every_scope() -> None:
    snapshot, _, _, l3, backend = _paged(_SIZES, l3_concurrency=1)

    await snapshot.start()
    await snapshot.wait_ready(timeout=10)

    assert backend.committed == sum(_SIZES.values())
    assert l3.peak_alive <= max(_SIZES.values()) + _PAGE + 1, (
        f"the rebuild held {l3.peak_alive} rows at once; one scope is {max(_SIZES.values())}"
    )
    with snapshot.read() as cursor:
        assert cursor.execute("SELECT state, count(*) FROM results GROUP BY state ORDER BY state").fetchall() == [
            (state, rows) for state, rows in sorted(_SIZES.items())
        ]
    await snapshot.stop()


async def test_a_rebuild_publishes_each_scope_before_it_reads_past_the_scopes_it_holds() -> None:
    snapshot, pointers, _, l3, _ = _paged(_SIZES, l3_concurrency=2)
    published_when_read: dict[str, set[str]] = {}
    l3.on_read = lambda state: published_when_read.setdefault(
        state, {key for key in pointers.entries if key.startswith("enr.s.")}
    )

    await snapshot.start()
    await snapshot.wait_ready(timeout=10)

    # two at once: DE and NV are read together, and TX only once DE is published
    assert published_when_read["TX"] >= {"enr.s.DE"}, published_when_read
    assert "enr.s.NV" not in published_when_read["NV"]
    await snapshot.stop()


async def test_a_rebuild_that_dies_part_way_names_no_snapshot_complete_and_serves_none() -> None:
    snapshot, pointers, store, l3, _ = _paged(_SIZES, l3_concurrency=1)
    l3.failing = "NV"

    await snapshot.start()
    await _until(lambda: "failed" in snapshot.status().detail, what="the rebuild failed at NV")

    assert "enr.s.DE" in pointers.entries, "a scope read before the failure was not published"
    assert "enr.s.TX" not in pointers.entries
    assert "enr.index" not in pointers.entries, "a rebuild that died part way wrote an index"
    assert not snapshot.is_ready
    # a replica starting meanwhile finds no snapshot to load: what DE's pointer names is not the whole
    other, _, _, _ = _snapshot(l3=l3, pointers=pointers, store=store)
    l3.epochs = dict.fromkeys(_SIZES, 1)
    await other.start()
    await asyncio.sleep(0.3)
    assert not other.is_ready and other.status().source is None, "a partial rebuild was loaded as the snapshot"
    await other.stop()

    l3.failing = None
    await snapshot.wait_ready(timeout=10)
    assert _indexed(pointers) == set(_SIZES)
    await snapshot.stop()


async def test_a_write_committed_during_a_rebuild_rereads_only_the_scopes_not_yet_published_at_their_epoch() -> None:
    snapshot, pointers, _, l3, _ = _paged(_SIZES, l3_concurrency=1)

    def a_write_to_tx_commits(state: str) -> None:
        if state == "NV" and l3.reads["NV"] == 1:
            l3.epochs["TX"] = 2

    l3.on_read = a_write_to_tx_commits

    await snapshot.start()
    await snapshot.wait_ready(timeout=10)

    assert l3.reads == {"DE": 1, "NV": 2, "TX": 1}, "a scope published at its epoch was read again, or one was not"
    assert json.loads(pointers.entries["enr.s.TX"][0])["epoch"] == 2
    assert _indexed(pointers) == set(_SIZES)
    await snapshot.stop()


async def test_a_cold_load_commits_each_scope_before_it_fetches_past_the_scopes_it_holds() -> None:
    builder, pointers, store, l3, _ = _paged(_SIZES)
    await builder.start()
    await builder.wait_ready(timeout=10)
    await builder.stop()
    statements = l3.statements
    events: list[str] = []

    class _Recording(_Store):
        async def get(self, name: str) -> bytes:
            events.append(f"fetch {name.split('/')[1]}")
            return await super().get(name)

    recording = _Recording()
    recording.objects = store.objects
    metadata = MetaData()
    _RESULTS.to_sqlalchemy_table(metadata)
    backend = _CountingBackend(events)
    backend.initialize(metadata)
    cold, _, _, _ = _snapshot(l3=l3, pointers=pointers, store=recording, backend=backend, l2_concurrency=1)
    l3.epochs = dict.fromkeys(_SIZES, 1)

    await cold.start()
    await cold.wait_ready(timeout=10)

    assert cold.status().source is SnapshotSource.L2
    assert l3.statements == statements, "a cold load read L3"
    assert events == [f"{step} {state}" for state in sorted(_SIZES) for step in ("fetch", "commit")], events
    assert backend.committed == sum(_SIZES.values())
    await cold.stop()


class _StuckL3(_PagedL3):
    """L3 whose read of ``stuck`` never answers."""

    def __init__(self, sizes: dict[str, int], stuck: str) -> None:
        super().__init__(sizes)
        self.stuck = stuck

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        if args and args[0] == self.stuck:
            self.reads[self.stuck] += 1
            await asyncio.Event().wait()
        return await super().fetch(sql, *args)


async def test_stopping_a_rebuild_whose_abandoned_read_failed_stops_it() -> None:
    l3 = _StuckL3(_SIZES, stuck="DE")
    l3.failing = "NV"
    snapshot, _, _, _ = _snapshot(l3=l3, l3_concurrency=2)
    l3.epochs = dict.fromkeys(_SIZES, 1)

    await snapshot.start()
    # DE is the scope published first, and never read; NV, read beside it, failed meanwhile
    await _until(lambda: l3.reads["DE"] == 1 and l3.reads["NV"] == 1, what="DE and NV read")
    await asyncio.sleep(0.05)

    # the failed read must not stand in for the cancellation: a worker told it failed carries on
    await asyncio.wait_for(snapshot.stop(), timeout=5)


async def test_a_rebuild_and_a_cold_load_count_every_scope_in_the_status() -> None:
    builder, pointers, store, l3, _ = _paged(_SIZES)
    await builder.start()
    await builder.wait_ready(timeout=10)
    built = builder.status()
    await builder.stop()
    cold, _, _, _ = _snapshot(l3=l3, pointers=pointers, store=store)
    l3.epochs = dict.fromkeys(_SIZES, 1)

    await cold.start()
    await cold.wait_ready(timeout=10)
    loaded = cold.status()

    assert built.source is SnapshotSource.L3 and loaded.source is SnapshotSource.L2
    assert built.scopes_total == built.scopes_done == len(_SIZES)
    assert loaded.scopes_total == loaded.scopes_done == len(_SIZES)
    assert set(built.timings) == {"read_l3", "publish", "total"}
    assert set(loaded.timings) == {"fetch", "load", "total"}
    await cold.stop()


async def test_a_write_that_outruns_a_rebuild_with_nothing_to_read_leaves_no_index_of_nothing() -> None:
    snapshot, pointers, _, l3, _ = _paged({})
    written = False
    listed = l3.fetch

    async def a_write_commits_once_l3_is_listed(sql: str, *args: Any) -> list[dict[str, Any]]:
        nonlocal written
        answer = await listed(sql, *args)
        if "DISTINCT" in sql and not written:
            # the write lands DE in L3 just after the rebuild listed L3's scopes, and commits
            written = True
            l3.sizes["DE"] = 10
            l3.reads["DE"] = 0
            l3.epochs["DE"] = 1
        return answer

    l3.fetch = a_write_commits_once_l3_is_listed  # type: ignore[method-assign]

    await snapshot.start()
    await snapshot.wait_ready(timeout=10)

    assert _indexed(pointers) == {"DE"}, "a rebuild a write outran wrote its index of nothing"
    with snapshot.read() as cursor:
        assert cursor.execute("SELECT count(*) FROM results WHERE state = 'DE'").fetchone() == (10,)
    await snapshot.stop()
