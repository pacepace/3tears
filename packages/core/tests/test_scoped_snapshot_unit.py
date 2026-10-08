"""the scoped snapshot's wiring and bookkeeping, over in-memory fakes of its buckets: no NATS, no L3.

What the live tests (``integration/test_scoped_snapshot_live.py``) cannot pin cheaply: that a tool
pod's snapshot is wired to the hub's asks, that the status never queries DuckDB, that a rebuild
claim survives a renewal that cannot reach NATS, and that a scope whose chunks cannot be read is
shown, retried at the recheck and not in a spin.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

pytest.importorskip("pyarrow")

from sqlalchemy import MetaData  # noqa: E402

from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.collections.complete_copy import Unsettled  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema  # noqa: E402
from threetears.core.collections.scoped_snapshot import (  # noqa: E402
    ScopedSnapshot,
    SnapshotPhase,
    SnapshotTable,
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
        entry = self.entries.get(key)
        return None if entry is None else entry[0]

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
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

    async def put(self, name: str, data: bytes) -> None:
        self.objects[name] = data

    async def get(self, name: str) -> bytes:
        await asyncio.sleep(0)  # a read yields, as a real one does, so a spin is counted, not hung
        self.gets += 1
        if name not in self.objects:
            raise ObjectNotFoundError(f"{name} is not in the store", bucket="objects", name=name)
        return self.objects[name]

    async def list_objects(self, *, prefix: str = "") -> list[ObjectInfo]:
        return [
            ObjectInfo(name=n, size=len(d), chunks=1, digest="", nuid="", mtime=datetime.now(UTC))
            for n, d in self.objects.items()
            if n.startswith(prefix)
        ]


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


def _point_at_a_missing_chunk(pointers: _Pointers) -> None:
    """move TX's pointer to epoch 2, naming a chunk that is not in the store."""
    held = json.loads(pointers.entries["enr.s.TX"][0])
    held["epoch"], held["tables"]["results"]["object"] = 2, "enr/TX/2/results.gone"
    pointers.put_now("enr.s.TX", json.dumps(held).encode())


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
            **{option: object()},
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
