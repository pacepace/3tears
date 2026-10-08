"""the scoped snapshot's wiring and bookkeeping, over in-memory fakes of its buckets: no NATS, no L3.

What the live tests (``integration/test_scoped_snapshot_live.py``) cannot pin cheaply: that a tool
pod's snapshot is wired to the hub's asks, that the status never queries DuckDB, that a rebuild
claim survives a renewal that cannot reach NATS, and that a scope whose chunks cannot be read is
shown, retried at the recheck and not in a spin.
"""

from __future__ import annotations

import ast
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
    """a pointer bucket in memory: CAS by revision, and a prefix watch that delivers every change."""

    def __init__(self) -> None:
        self.entries: dict[str, tuple[bytes, int]] = {}
        self._revision = 0
        self._queue: asyncio.Queue[KvKeyUpdate | None | Exception] = asyncio.Queue()
        self.update_fails: Exception | None = None

    def end_watch(self) -> None:
        """end the pointer watch, as a lost connection does, while the bucket's reads and writes still answer."""
        self._queue.put_nowait(ConnectionError("the watch's consumer is gone"))

    def put_now(self, key: str, value: bytes) -> int:
        """write a key outright, as another replica's publish would."""
        return self._put(key, value)

    def _put(self, key: str, value: bytes) -> int:
        self._revision += 1
        self.entries[key] = (value, self._revision)
        self._queue.put_nowait(KvKeyUpdate(key=key, value=value, revision=self._revision))
        return self._revision

    async def get(self, *, key: str) -> bytes | None:
        await asyncio.sleep(0)  # a read yields, as a real one does, so what the loop shows between reads is seen
        entry = self.entries.get(key)
        return None if entry is None else entry[0]

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        await asyncio.sleep(0)
        return self.entries.get(key)

    async def create(self, *, key: str, value: bytes, ttl: timedelta | None = None) -> int | None:
        return None if key in self.entries else self._put(key, value)

    async def update(self, *, key: str, value: bytes, revision: int, ttl: timedelta | None = None) -> int | None:
        if self.update_fails is not None:
            raise self.update_fails
        entry = self.entries.get(key)
        return self._put(key, value) if entry is not None and entry[1] == revision else None

    async def delete(self, *, key: str, revision: int | None = None) -> bool:
        entry = self.entries.get(key)
        if entry is None or (revision is not None and entry[1] != revision):
            return False
        del self.entries[key]
        self._revision += 1
        self._queue.put_nowait(KvKeyUpdate(key=key, value=None, revision=self._revision))
        return True

    async def watch_prefix(self, *, prefix: str, heartbeat: timedelta) -> AsyncIterator[KvKeyUpdate | None]:
        for key, (value, revision) in list(self.entries.items()):
            yield KvKeyUpdate(key=key, value=value, revision=revision)
        yield None
        while True:
            update = await self._queue.get()
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
    pointers, store = _Pointers(), _Store()
    l3 = _L3([{"county": "c1", "state": "TX", "votes": 1}, {"county": "c2", "state": "DE", "votes": 2}])

    l3.epochs = {"TX": 1, "DE": 1}

    async def epochs() -> dict[str, int]:
        return dict(l3.epochs)

    l3.writing = False

    async def settled() -> Any:
        return Unsettled("a write is in progress") if l3.writing else dict(l3.epochs)

    snapshot = ScopedSnapshot(
        name="enr",
        tables=_TABLES,
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
    deadline = asyncio.get_running_loop().time() + timeout
    while not check():
        assert asyncio.get_running_loop().time() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.01)


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

    await _until(lambda: "TX" in snapshot.status().behind, what="TX shown behind")
    assert snapshot.status().phase is SnapshotPhase.READY and "behind" in snapshot.status().detail
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
    deadline = time.monotonic() + seconds
    while thread.is_alive() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert out, f"a versioned read did not finish in {seconds} s: the swap lock was left held"
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
    real_create = pointers.create

    async def counting(*, key: str, value: bytes, ttl: timedelta | None = None) -> int | None:
        takes[0] += key == "enr.rebuild"
        return await real_create(key=key, value=value, ttl=ttl)

    pointers.create = counting  # type: ignore[method-assign]
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

    looks = []
    first = takes[0]
    deadline = asyncio.get_running_loop().time() + 5
    while takes[0] < first + 3 and asyncio.get_running_loop().time() < deadline:
        # rebuilds of TX ran, each taking and letting go of the claim; looked at between every step
        looks.append(snapshot.status())
        await asyncio.sleep(0)
    assert takes[0] >= first + 3, "TX was not rebuilt again and again"

    assert all("TX" in look.behind for look in looks), [
        (look.phase, sorted(look.behind)) for look in looks if "TX" not in look.behind
    ][:3]
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


# the state readers on any thread take, which the event loop alone writes, by rebinding
_SHARED = frozenset({"_applied", "_behind", "_rows", "_progress"})
# set once at construction, and safe to use from any thread
_SET_AT_START = frozenset({"_backend", "_ready", "_name", "_tables", "_schema", "_versioned"})
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
            if isinstance(work, ast.Lambda) or (_self_attr(work) is not None):
                wrong.append(
                    f"line {node.lineno}: to_thread runs {ast.unparse(work)}, the snapshot's own code, off the loop"
                )
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
