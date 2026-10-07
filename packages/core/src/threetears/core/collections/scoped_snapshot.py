"""A scoped snapshot: tables a pod needs whole, held in L2 as one columnar chunk per scope, loaded into DuckDB.

**Why.** A pod that answers analytic queries over whole tables (:mod:`complete_copy`) must hold every
row in its DuckDB L1. Reading the tables from L3 a thousand rows a statement takes minutes, and doing
it again after every change, or in every new replica, is the cost this capability removes. The
tables are partitioned by a scope column (a state, say); each scope's rows of each table are kept in
NATS -- the L2 tier -- as one compact Arrow IPC chunk (zstd) in the pod's Object Store, named by the
scope and its EPOCH (the per-scope version :class:`~threetears.core.collections.scope_epochs.ScopeEpochs`
keeps), and one small KV pointer per scope names the scope's current epoch and its chunks.

**Loading.** A starting replica watches the pointers (``watch_prefix``), fetches every current chunk
in parallel and loads them into its DuckDB L1 through :class:`~threetears.core.cache.duckdb.DuckDBBackend`:
seconds, with no L3 read. Each chunk's digest (the Object Store's) and row count (the pointer's) are
checked; the pointers' index names every scope, so a missing one is seen, not skipped.

**Changes.** A writer that commits a scope's change calls :meth:`ScopedSnapshot.publish` with the
scope's complete new rows: they replace the scope in its own L1, the chunks are written under the
new epoch, and then the pointer moves. Every other replica's watch sees the pointer, fetches only that
scope's chunks and replaces the scope in ONE DuckDB transaction, so a reader -- holding
:meth:`ScopedSnapshot.read` for a request -- sees the scope as it was or as it is, never half of it.
Nothing is rebuilt whole and nothing polls.

**L3 stays the truth.** NATS buckets are memory-backed, so a NATS restart loses the snapshot. A
replica that finds the pointers gone asks for the buckets again (``ensure_buckets``), takes a claim
so one replica does the work, and rebuilds every scope from L3 -- the slow, correct path -- while its
own L1 keeps answering from what it last held; its status says so. :meth:`ScopedSnapshot.catch_up_from_l3`
republishes any scope whose L3 epoch is ahead of its pointer (a writer that died between its commit
and its publish). A rebuild reads L3 only while no write is in progress and keeps what it read only
when no write committed meanwhile (``settled``, the writer's seqlock).

**Retirement.** Chunks of an epoch a pointer no longer names are deleted -- by the bucket's declarer,
since a delete is a purge no pod holds (``retire``). A reader that lost the race to a retired chunk
reads the scope's pointer again and fetches its current chunks.

**Transparent.** :meth:`ScopedSnapshot.status` says what it is doing (loading from L2, rebuilding from
L3, waiting, ready), how many scopes are done, each table's rows, how long each step took, and the
last change applied.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import aclosing, contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final

from threetears.observe import get_logger

from threetears.core.cache.base import quote_identifier
from threetears.core.cache.duckdb import DuckDBBackend, PartitionReplacement
from threetears.core.collections.complete_copy import Unsettled, read_l3_rows

__all__ = [
    "ScopeChange",
    "ScopedSnapshot",
    "SnapshotPhase",
    "SnapshotSource",
    "SnapshotStatus",
    "SnapshotTable",
    "decode_chunk",
    "encode_chunk",
    "scope_token",
]

log = get_logger(__name__)

#: the snapshot name's grammar: one KV subject token and one object-name segment
_NAME_GRAMMAR: Final = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")

#: how long a rebuild's claim lasts unless renewed by its completion; a replica that dies holding it
#: frees it after this long
_DEFAULT_CLAIM_TTL: Final = timedelta(minutes=10)

#: how often a replica waiting on another's rebuild, or on a write in progress, looks again
_DEFAULT_RECHECK: Final = timedelta(seconds=2)

#: how many scope reads a rebuild runs on L3 at once
_DEFAULT_L3_CONCURRENCY: Final = 4

#: how many chunk reads a load runs on L2 at once
_DEFAULT_L2_CONCURRENCY: Final = 16

#: how many times a rebuild reads again after a write committed while it read
_REBUILD_ATTEMPTS: Final = 3


class SnapshotPhase(StrEnum):
    """what a snapshot is doing.

    :cvar STARTING: watching the pointers, not yet caught up with them
    :cvar LOADING_FROM_L2: fetching and loading every current chunk
    :cvar REBUILDING_FROM_L3: reading scopes from L3 and publishing their chunks
    :cvar WAITING: waiting for another replica's rebuild, or for a write in progress to commit
    :cvar READY: every scope loaded; changes are applied as they are published
    :cvar FAILED: the last attempt failed; it is tried again
    """

    STARTING = "starting"
    LOADING_FROM_L2 = "loading_from_l2"
    REBUILDING_FROM_L3 = "rebuilding_from_l3"
    WAITING = "waiting"
    READY = "ready"
    FAILED = "failed"


class SnapshotSource(StrEnum):
    """where the copy a replica serves came from.

    :cvar L2: the chunks in NATS
    :cvar L3: the tables in L3, read because L2 did not hold the snapshot
    """

    L2 = "l2"
    L3 = "l3"


@dataclass(frozen=True)
class SnapshotTable:
    """one table of the snapshot.

    :ivar name: the table, as the DuckDB backend and L3 name it (a TRUSTED identifier)
    :ivar scope_column: the column whose value is a row's scope
    :ivar key: the table's key, ordering a chunk's rows and paging its L3 reads
    """

    name: str
    scope_column: str
    key: tuple[str, ...]


@dataclass(frozen=True)
class ScopeChange:
    """the last scope change a replica applied.

    :ivar scope: the scope
    :ivar epoch: its new epoch
    :ivar rows: the rows it now holds across every table
    :ivar seconds: from seeing the pointer to the commit of the swap
    :ivar applied_at: when the swap committed
    """

    scope: str | None
    epoch: int
    rows: int
    seconds: float
    applied_at: datetime


@dataclass(frozen=True)
class SnapshotStatus:
    """what a snapshot is doing and has done, for a status tool to show.

    :ivar phase: the current phase
    :ivar source: where the copy served came from, once one is loaded
    :ivar detail: a short plain statement of the current step
    :ivar scopes_total: the scopes the current step covers
    :ivar scopes_done: the scopes it has finished
    :ivar rows: each table's rows in the L1
    :ivar timings: seconds per step of the last load or rebuild
    :ivar history: every phase entered, in order, without repeats in a row
    :ivar last_change: the last scope change applied, if any
    :ivar ready_at: when the copy first became ready
    """

    phase: SnapshotPhase
    source: SnapshotSource | None
    detail: str
    scopes_total: int
    scopes_done: int
    rows: Mapping[str, int]
    timings: Mapping[str, float]
    history: tuple[SnapshotPhase, ...]
    last_change: ScopeChange | None
    ready_at: datetime | None


@dataclass(frozen=True)
class _Pointer:
    """what a scope's pointer says: its epoch, and its chunk for each table."""

    scope: str | None
    epoch: int
    objects: Mapping[str, str]
    rows: Mapping[str, int]
    schema: Mapping[str, str]

    def encode(self) -> bytes:
        """the pointer as the KV entry's value.

        :return: its JSON
        :rtype: bytes
        """
        return json.dumps(
            {
                "scope": self.scope,
                "epoch": self.epoch,
                "tables": {t: {"object": self.objects[t], "rows": self.rows[t]} for t in self.objects},
                "schema": dict(self.schema),
            },
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def decode(cls, raw: bytes) -> _Pointer:
        """a pointer from its KV entry's value.

        :param raw: the value
        :ptype raw: bytes
        :return: the pointer
        :rtype: _Pointer
        :raises KeyError: when a field is missing
        :raises ValueError: when it is not JSON
        """
        body = json.loads(raw)
        tables = body["tables"]
        return cls(
            scope=body["scope"],
            epoch=int(body["epoch"]),
            objects={t: str(v["object"]) for t, v in tables.items()},
            rows={t: int(v["rows"]) for t, v in tables.items()},
            schema={t: str(v) for t, v in body["schema"].items()},
        )


def scope_token(scope: str | None) -> str:
    """a scope as one KV key token and one object-name segment, reversibly.

    Letters, digits, ``-`` and ``_`` stand for themselves; every other byte is ``=`` and two hex
    digits; a null scope is ``=`` alone.

    :param scope: the scope value
    :ptype scope: str | None
    :return: the token
    :rtype: str
    """
    if scope is None:
        return "="
    return "".join(
        ch if ch in _NAME_GRAMMAR else "".join(f"={byte:02X}" for byte in ch.encode("utf-8")) for ch in scope
    )


def encode_chunk(table: Any) -> bytes:
    """an Arrow table as a chunk: one Arrow IPC stream, zstd-compressed.

    :param table: the rows
    :ptype table: pyarrow.Table
    :return: the chunk's bytes
    :rtype: bytes
    """
    import pyarrow as pa  # noqa: PLC0415 -- the snapshot extra
    import pyarrow.ipc as ipc  # noqa: PLC0415

    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema, options=ipc.IpcWriteOptions(compression="zstd")) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue().to_pybytes())


def decode_chunk(data: bytes) -> Any:
    """a chunk's rows as an Arrow table.

    :param data: the chunk's bytes
    :ptype data: bytes
    :return: the rows
    :rtype: pyarrow.Table
    """
    import pyarrow.ipc as ipc  # noqa: PLC0415 -- the snapshot extra

    return ipc.open_stream(data).read_all()


class _Lost(Exception):
    """the snapshot in L2 is incomplete or unreadable; L3 must rebuild it."""


@dataclass
class _Progress:
    """the mutable half of the status."""

    phase: SnapshotPhase = SnapshotPhase.STARTING
    source: SnapshotSource | None = None
    detail: str = "watching the pointers"
    scopes_total: int = 0
    scopes_done: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    history: list[SnapshotPhase] = field(default_factory=lambda: [SnapshotPhase.STARTING])
    last_change: ScopeChange | None = None
    ready_at: datetime | None = None


class ScopedSnapshot:
    """tables a pod needs whole, held in L2 as one chunk per scope and per table, loaded into DuckDB.

    :param name: the snapshot's name, keying its pointers and objects; letters, digits, ``-``, ``_``
    :ptype name: str
    :param tables: the tables, each with its scope column and key
    :ptype tables: Sequence[SnapshotTable]
    :param backend: the DuckDB L1 holding every table, initialized; the snapshot owns its contents
    :ptype backend: DuckDBBackend
    :param store: the pod's Object Store (``NatsClient.object_store``)
    :ptype store: NatsObjectStore
    :param pointers: the pod's pointer bucket (``NatsClient.kv_bucket``), allowing per-entry TTLs
    :ptype pointers: NatsKvBucket
    :param l3: the L3 backend the tables live in (``fetch``, ``fetchrow``)
    :ptype l3: Any
    :param epochs: each scope's current epoch as L3 records it; a scope it does not name is epoch 0
    :ptype epochs: Callable[[], Awaitable[Mapping[str, int]]]
    :param settled: the writer's seqlock (``ScopeEpochs.settled``): a stamp naming the last committed
        write, or :class:`Unsettled` while one is in progress; ``None`` when nothing writes
    :ptype settled: Callable[[], Awaitable[Any]] | None
    :param ensure_buckets: asks the buckets' declarer to declare them again, after NATS lost them
    :ptype ensure_buckets: Callable[[], Awaitable[None]] | None
    :param retire: asks the declarer to delete objects no pointer names any more; ``None`` keeps them
    :ptype retire: Callable[[list[str]], Awaitable[Any]] | None
    :param l2_concurrency: chunk reads at once
    :ptype l2_concurrency: int
    :param l3_concurrency: scope reads on L3 at once during a rebuild
    :ptype l3_concurrency: int
    :param claim_ttl: how long a rebuild's claim lasts if its holder dies
    :ptype claim_ttl: timedelta
    :param recheck: how often a waiting replica looks again
    :ptype recheck: timedelta
    :param pointer_watch_heartbeat: the pointer watch's heartbeat; a lost consumer is replaced after three
    :ptype pointer_watch_heartbeat: timedelta
    :raises ValueError: when the name is outside its grammar or names no table
    """

    def __init__(
        self,
        *,
        name: str,
        tables: Sequence[SnapshotTable],
        backend: DuckDBBackend,
        store: Any,
        pointers: Any,
        l3: Any,
        epochs: Callable[[], Awaitable[Mapping[str, int]]],
        settled: Callable[[], Awaitable[Any]] | None = None,
        ensure_buckets: Callable[[], Awaitable[None]] | None = None,
        retire: Callable[[list[str]], Awaitable[Any]] | None = None,
        l2_concurrency: int = _DEFAULT_L2_CONCURRENCY,
        l3_concurrency: int = _DEFAULT_L3_CONCURRENCY,
        claim_ttl: timedelta = _DEFAULT_CLAIM_TTL,
        recheck: timedelta = _DEFAULT_RECHECK,
        pointer_watch_heartbeat: timedelta = timedelta(seconds=5),
    ) -> None:
        if not name or any(ch not in _NAME_GRAMMAR for ch in name):
            raise ValueError(f"snapshot name {name!r} must be letters, digits, '-' and '_'")
        if not tables:
            raise ValueError("a snapshot holds at least one table")
        self._name = name
        self._tables = tuple(tables)
        self._backend = backend
        self._store = store
        self._pointers_bucket = pointers
        self._l3 = l3
        self._epochs = epochs
        self._settled = settled
        self._ensure_buckets = ensure_buckets
        self._retire = retire
        self._l2_concurrency = l2_concurrency
        self._l3_concurrency = l3_concurrency
        self._claim_ttl = claim_ttl
        self._recheck = recheck
        self._heartbeat = pointer_watch_heartbeat
        self._schema = {t.name: backend.schema_digest(t.name) for t in self._tables}
        self._replica = uuid.uuid7().hex
        # what the pointer bucket says, as the watch last delivered it
        self._seen: dict[str | None, _Pointer] = {}
        self._index: frozenset[str | None] | None = None
        self._caught_up = False
        self._ever_indexed = False
        # what the L1 holds
        self._applied: dict[str | None, _Pointer] = {}
        self._local = asyncio.Lock()
        self._changed = asyncio.Event()
        self._ready = asyncio.Event()
        self._progress = _Progress()
        self._tasks: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------
    # keys and names
    # ------------------------------------------------------------------

    @property
    def _index_key(self) -> str:
        return f"{self._name}.index"

    @property
    def _claim_key(self) -> str:
        return f"{self._name}.rebuild"

    def _pointer_key(self, scope: str | None) -> str:
        return f"{self._name}.s.{scope_token(scope)}"

    def _object_name(self, scope: str | None, epoch: int, table: str) -> str:
        return f"{self._name}/{scope_token(scope)}/{epoch}/{table}"

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def _enter(self, phase: SnapshotPhase, detail: str, *, total: int | None = None) -> None:
        """record a phase and a plain statement of what it is doing.

        :param phase: the phase
        :ptype phase: SnapshotPhase
        :param detail: what it is doing
        :ptype detail: str
        :param total: the scopes the step covers, resetting the count done
        :ptype total: int | None
        :return: nothing
        :rtype: None
        """
        progress = self._progress
        if progress.phase is not phase or progress.detail != detail:
            log.info("scoped snapshot %s: %s", self._name, detail, extra={"extra_data": {"phase": phase.value}})
        progress.phase = phase
        progress.detail = detail
        if progress.history[-1] is not phase:
            progress.history.append(phase)
        if total is not None:
            progress.scopes_total = total
            progress.scopes_done = 0

    def status(self) -> SnapshotStatus:
        """what the snapshot is doing and has done.

        :return: the status
        :rtype: SnapshotStatus
        """
        progress = self._progress
        rows: dict[str, int] = {}
        if self._backend.is_initialized():
            for table in self._tables:
                result = self._backend.execute_query(f"SELECT count(*) AS n FROM {quote_identifier(table.name)}")  # noqa: S608
                rows[table.name] = int(result[0]["n"])
        return SnapshotStatus(
            phase=progress.phase,
            source=progress.source,
            detail=progress.detail,
            scopes_total=progress.scopes_total,
            scopes_done=progress.scopes_done,
            rows=rows,
            timings=dict(progress.timings),
            history=tuple(progress.history),
            last_change=progress.last_change,
            ready_at=progress.ready_at,
        )

    @property
    def backend(self) -> DuckDBBackend:
        """the DuckDB L1 holding the tables.

        :return: the backend
        :rtype: DuckDBBackend
        """
        return self._backend

    @property
    def is_ready(self) -> bool:
        """whether every scope has been loaded.

        :return: whether the copy is ready
        :rtype: bool
        """
        return self._ready.is_set()

    @contextmanager
    def read(self) -> Iterator[Any]:
        """a cursor reading one state of every table for the whole block; hold it for one request.

        :return: the cursor
        :rtype: Iterator[duckdb.DuckDBPyConnection]
        """
        with self._backend.read_snapshot() as cursor:
            yield cursor

    async def wait_ready(self, *, timeout: float) -> None:
        """wait until every scope has been loaded.

        :param timeout: seconds to wait
        :ptype timeout: float
        :return: nothing
        :rtype: None
        :raises TimeoutError: when the copy is not ready in time
        """
        await asyncio.wait_for(self._ready.wait(), timeout)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """start watching the pointers and loading; returns at once.

        :return: nothing
        :rtype: None
        """
        self._tasks = [
            asyncio.create_task(self._watch(), name=f"scoped-snapshot-watch:{self._name}"),
            asyncio.create_task(self._work(), name=f"scoped-snapshot-work:{self._name}"),
        ]

    async def stop(self) -> None:
        """stop watching and loading.

        :return: nothing
        :rtype: None
        """
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            # NOSILENT: the cancellation this method just asked for
            with suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    async def _watch(self) -> None:
        """follow the pointer bucket and wake the worker on every change.

        :return: nothing
        :rtype: None
        """
        from threetears.nats import KvError  # noqa: PLC0415

        try:
            await self._follow(f"{self._name}.")
        except KvError as exc:
            # the watch outlives every NATS outage but a closed client; say so rather than go quiet
            log.error("scoped snapshot %s: the pointer watch ended: %s", self._name, exc)
            self._enter(SnapshotPhase.FAILED, f"the pointer watch ended ({exc}); changes are no longer applied")

    async def _follow(self, prefix: str) -> None:
        """consume the pointer watch, recording what it delivers.

        :param prefix: the snapshot's key prefix
        :ptype prefix: str
        :return: nothing
        :rtype: None
        :raises KvError: when the NATS connection is closed
        """
        async with aclosing(self._pointers_bucket.watch_prefix(prefix=prefix, heartbeat=self._heartbeat)) as watch:
            async for update in watch:
                if update is None:
                    self._caught_up = True
                elif update.key == self._index_key:
                    self._index = None if update.deleted else frozenset(json.loads(update.value)["scopes"])
                    self._ever_indexed = self._ever_indexed or self._index is not None
                elif update.key.startswith(f"{prefix}s."):
                    try:
                        pointer = None if update.deleted else _Pointer.decode(update.value)
                    except (ValueError, KeyError, TypeError) as exc:
                        log.error("scoped snapshot %s: pointer %s does not decode: %s", self._name, update.key, exc)
                        continue
                    if pointer is not None:
                        self._seen[pointer.scope] = pointer
                    else:
                        for scope in [s for s in self._seen if self._pointer_key(s) == update.key]:
                            del self._seen[scope]
                self._changed.set()

    async def _work(self) -> None:
        """bring the L1 to what the pointers say, each time they change or a wait is due.

        :return: nothing
        :rtype: None
        """
        while True:
            try:
                await asyncio.wait_for(self._changed.wait(), self._recheck.total_seconds())
            except TimeoutError:
                pass  # NOSILENT: the recheck interval passed with no change; a pass runs anyway
            self._changed.clear()
            if not self._caught_up:
                continue
            try:
                await self._sync()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- the worker outlives any one failed attempt; logged and stated in the status, tried again at the next recheck
                log.exception("scoped snapshot %s: sync failed", self._name)
                self._enter(SnapshotPhase.FAILED, f"failed ({type(exc).__name__}: {exc}); trying again")

    async def _sync(self) -> None:
        """one pass: load the snapshot, rebuild it, or apply the scopes that moved.

        :return: nothing
        :rtype: None
        """
        index = self._index
        complete = index is not None and all(scope in self._seen for scope in index)
        mismatched = complete and any(self._seen[scope].schema != self._schema for scope in index or ())
        if not self._ready.is_set():
            loaded = False
            if complete and not mismatched:
                try:
                    await self._load({scope: self._seen[scope] for scope in index or ()}, cold=True)
                    loaded = True
                except _Lost as exc:
                    log.warning("scoped snapshot %s: the snapshot in L2 is not loadable: %s", self._name, exc)
            if not loaded:
                await self._rebuild(reason=self._why_rebuild(index, complete, mismatched))
        elif not complete or mismatched:
            await self._rebuild(reason=self._why_rebuild(index, complete, mismatched))
        else:
            await self._apply_moved(index or frozenset())

    def _why_rebuild(self, index: frozenset[str | None] | None, complete: bool, mismatched: bool) -> str:
        """a plain reason for rebuilding from L3.

        :return: the reason
        :rtype: str
        """
        if index is None and self._ever_indexed:
            reason = "NATS lost the snapshot"
        elif index is None:
            reason = "no snapshot in NATS yet"
        elif not complete:
            reason = "NATS lost part of the snapshot"
        elif mismatched:
            reason = "the snapshot in NATS was written for other table columns"
        else:
            reason = "the snapshot in NATS could not be read"
        return reason

    # ------------------------------------------------------------------
    # loading from L2
    # ------------------------------------------------------------------

    async def _fetch(self, pointer: _Pointer) -> list[PartitionReplacement]:
        """every table's chunk of one scope, decoded and checked; re-read once if a chunk was retired.

        :param pointer: the scope's pointer
        :ptype pointer: _Pointer
        :return: the scope's replacements
        :rtype: list[PartitionReplacement]
        :raises _Lost: when a chunk is gone or does not hold what its pointer says
        """
        from threetears.nats import ObjectNotFoundError, ObjectStoreError  # noqa: PLC0415

        replacements: list[PartitionReplacement] = []
        for table in self._tables:
            name = pointer.objects.get(table.name)
            if name is None:
                raise _Lost(f"scope {pointer.scope!r} names no chunk for {table.name}")
            try:
                data = await self._store.get(name)
            except ObjectNotFoundError as exc:
                raise _Lost(f"chunk {name} is gone") from exc
            except ObjectStoreError as exc:
                raise _Lost(f"chunk {name} could not be read: {exc}") from exc
            arrow = await asyncio.to_thread(decode_chunk, data)
            if arrow.num_rows != pointer.rows.get(table.name):
                raise _Lost(
                    f"chunk {name} holds {arrow.num_rows} rows where its pointer says {pointer.rows.get(table.name)}"
                )
            replacements.append(
                PartitionReplacement(table=table.name, column=table.scope_column, value=pointer.scope, arrow=arrow)
            )
        return replacements

    async def _fetch_current(self, scope: str | None, pointer: _Pointer) -> tuple[_Pointer, list[PartitionReplacement]]:
        """a scope's chunks, reading its pointer again once when a chunk was retired under the read.

        :return: the pointer the chunks belong to, and the replacements
        :rtype: tuple[_Pointer, list[PartitionReplacement]]
        :raises _Lost: when the chunks cannot be read even at the current pointer
        """
        current = pointer
        try:
            replacements = await self._fetch(pointer)
        except _Lost:
            raw = await self._pointers_bucket.get(key=self._pointer_key(scope))
            if raw is None or _Pointer.decode(raw).epoch == pointer.epoch:
                raise
            current = _Pointer.decode(raw)
            replacements = await self._fetch(current)
        return current, replacements

    async def _load(self, pointers: Mapping[str | None, _Pointer], *, cold: bool) -> None:
        """fetch the scopes' chunks in parallel and replace those scopes in one transaction.

        :param pointers: the scopes to load and their pointers
        :ptype pointers: Mapping[str | None, _Pointer]
        :param cold: whether this is the first load (every scope, and the copy becomes ready)
        :ptype cold: bool
        :return: nothing
        :rtype: None
        :raises _Lost: when a chunk is gone or wrong
        """
        started = time.perf_counter()
        if cold:
            self._enter(SnapshotPhase.LOADING_FROM_L2, "loading from NATS", total=len(pointers))
        semaphore = asyncio.Semaphore(self._l2_concurrency)

        async def one(scope: str | None, pointer: _Pointer) -> tuple[_Pointer, list[PartitionReplacement]]:
            async with semaphore:
                result = await self._fetch_current(scope, pointer)
            if cold:
                self._progress.scopes_done += 1
            return result

        fetched = await asyncio.gather(*(one(scope, pointer) for scope, pointer in pointers.items()))
        fetched_at = time.perf_counter()
        replacements = [replacement for _, scope_replacements in fetched for replacement in scope_replacements]
        async with self._local:
            await asyncio.to_thread(self._backend.replace_partitions, replacements)
            for pointer, _ in fetched:
                self._applied[pointer.scope] = pointer
        done = time.perf_counter()
        if cold:
            self._progress.timings.update(
                {
                    "fetch": round(fetched_at - started, 3),
                    "load": round(done - fetched_at, 3),
                    "total": round(done - started, 3),
                }
            )
            self._become_ready(SnapshotSource.L2)
        log.info(
            "scoped snapshot %s: scopes loaded from NATS",
            self._name,
            extra={
                "extra_data": {
                    "scopes": len(pointers),
                    "rows": sum(r.arrow.num_rows for r in replacements if r.arrow is not None),
                    "fetch_seconds": round(fetched_at - started, 3),
                    "load_seconds": round(done - fetched_at, 3),
                }
            },
        )

    def _become_ready(self, source: SnapshotSource) -> None:
        """record that every scope is loaded, and from where.

        :param source: where the copy came from
        :ptype source: SnapshotSource
        :return: nothing
        :rtype: None
        """
        self._progress.source = source
        if self._progress.ready_at is None:
            self._progress.ready_at = datetime.now(UTC)
        self._enter(SnapshotPhase.READY, "ready")
        self._ready.set()

    async def _apply_moved(self, index: frozenset[str | None]) -> None:
        """apply each scope whose pointer moved past what the L1 holds; drop scopes the index left.

        :param index: the scopes the snapshot holds
        :ptype index: frozenset[str | None]
        :return: nothing
        :rtype: None
        """
        moved = {
            scope: self._seen[scope]
            for scope in index
            if scope not in self._applied or self._seen[scope].epoch > self._applied[scope].epoch
        }
        gone = [scope for scope in self._applied if scope not in index]
        if gone:
            async with self._local:
                await asyncio.to_thread(
                    self._backend.replace_partitions,
                    [
                        PartitionReplacement(table=t.name, column=t.scope_column, value=scope, rows=[])
                        for scope in gone
                        for t in self._tables
                    ],
                )
                for scope in gone:
                    del self._applied[scope]
        for scope, pointer in moved.items():
            started = time.perf_counter()
            try:
                applied_pointer, replacements = await self._fetch_current(scope, pointer)
            except _Lost as exc:
                log.warning("scoped snapshot %s: scope %r could not be applied: %s", self._name, scope, exc)
                self._changed.set()
                continue
            async with self._local:
                held = self._applied.get(scope)
                if held is not None and held.epoch >= applied_pointer.epoch:
                    continue
                await asyncio.to_thread(self._backend.replace_partitions, replacements)
                self._applied[scope] = applied_pointer
            self._record_change(scope, applied_pointer, replacements, started)

    def _record_change(
        self, scope: str | None, pointer: _Pointer, replacements: Sequence[PartitionReplacement], started: float
    ) -> None:
        """record and log one applied scope change.

        :return: nothing
        :rtype: None
        """
        rows = sum(r.arrow.num_rows if r.arrow is not None else len(r.rows or ()) for r in replacements)
        change = ScopeChange(
            scope=scope,
            epoch=pointer.epoch,
            rows=rows,
            seconds=round(time.perf_counter() - started, 3),
            applied_at=datetime.now(UTC),
        )
        self._progress.last_change = change
        log.info(
            "scoped snapshot %s: scope applied",
            self._name,
            extra={"extra_data": {"scope": scope, "epoch": pointer.epoch, "rows": rows, "seconds": change.seconds}},
        )

    # ------------------------------------------------------------------
    # publishing
    # ------------------------------------------------------------------

    async def publish(self, scope: str | None, epoch: int, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        """make ``rows`` the scope's content at ``epoch``: in this L1, in its chunks, then its pointer.

        Called by the writer once the scope's change is committed in L3, with every row the scope now
        holds in each table. The other replicas apply it when they see the pointer move. Chunks of the
        scope's older epochs are retired.

        :param scope: the scope
        :ptype scope: str | None
        :param epoch: its epoch after the change
        :ptype epoch: int
        :param rows: table name -> every row of the scope in that table
        :ptype rows: Mapping[str, Sequence[Mapping[str, Any]]]
        :return: nothing
        :rtype: None
        :raises ValueError: when ``rows`` does not name every table of the snapshot
        """
        missing = [t.name for t in self._tables if t.name not in rows]
        if missing:
            raise ValueError(f"a publish of scope {scope!r} gives no rows for {missing}")
        started = time.perf_counter()
        async with self._local:
            pointer = await self._publish_locked(scope, epoch, rows)
        replacements = [
            PartitionReplacement(table=t.name, column=t.scope_column, value=scope, rows=rows[t.name])
            for t in self._tables
        ]
        self._record_change(scope, pointer, replacements, started)
        await self._ensure_indexed({scope})
        await self._retire_older(scope, epoch)

    async def _publish_locked(
        self, scope: str | None, epoch: int, rows: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> _Pointer:
        """the body of :meth:`publish`, under the L1 lock: replace locally, write chunks, move the pointer.

        :return: the pointer written
        :rtype: _Pointer
        """
        from threetears.nats import ObjectExistsError  # noqa: PLC0415

        await asyncio.to_thread(
            self._backend.replace_partitions,
            [
                PartitionReplacement(
                    table=t.name, column=t.scope_column, value=scope, rows=rows[t.name], primary_key=t.key
                )
                for t in self._tables
            ],
        )
        objects: dict[str, str] = {}
        counts: dict[str, int] = {}
        for table in self._tables:
            arrow = await asyncio.to_thread(
                self._backend.export_partition, table.name, table.scope_column, scope, order_by=table.key
            )
            data = await asyncio.to_thread(encode_chunk, arrow)
            name = self._object_name(scope, epoch, table.name)
            # NOSILENT: a replica rebuilding the same scope at the same epoch wrote the same rows first
            with suppress(ObjectExistsError):
                await self._store.put(name, data)
            objects[table.name] = name
            counts[table.name] = arrow.num_rows
        pointer = _Pointer(scope=scope, epoch=epoch, objects=objects, rows=counts, schema=self._schema)
        await self._pointers_bucket.put(key=self._pointer_key(scope), value=pointer.encode())
        self._applied[scope] = pointer
        self._seen[scope] = pointer
        return pointer

    async def _ensure_indexed(self, scopes: set[str | None]) -> None:
        """add scopes to the index, after their pointers exist.

        :param scopes: scopes now published
        :ptype scopes: set[str | None]
        :return: nothing
        :rtype: None
        """
        current = set(self._index or ())
        if scopes <= current:
            return
        await self._write_index(current | scopes)

    async def _write_index(self, scopes: set[str | None]) -> None:
        """write the index naming every scope the snapshot holds.

        :param scopes: the scopes
        :ptype scopes: set[str | None]
        :return: nothing
        :rtype: None
        """
        ordered = sorted(scopes, key=lambda s: (s is None, s or ""))
        await self._pointers_bucket.put(key=self._index_key, value=json.dumps({"scopes": ordered}).encode("utf-8"))
        self._index = frozenset(scopes)
        self._ever_indexed = True

    async def _retire_older(self, scope: str | None, epoch: int) -> None:
        """ask the declarer to delete the scope's chunks of epochs before ``epoch``; never raises.

        :return: nothing
        :rtype: None
        """
        if self._retire is None:
            return
        prefix = f"{self._name}/{scope_token(scope)}/"
        try:
            names = [
                info.name
                for info in await self._store.list_objects(prefix=prefix)
                if int(info.name[len(prefix) :].split("/", 1)[0]) < epoch
            ]
            if names:
                await self._retire(names)
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- retirement is housekeeping; a failure leaves old chunks for the next retire and must not fail a publish that already succeeded
            log.warning("scoped snapshot %s: retiring scope %r's old chunks failed: %s", self._name, scope, exc)

    async def _retire_unreferenced(self) -> None:
        """ask the declarer to delete every chunk no current pointer names; never raises.

        :return: nothing
        :rtype: None
        """
        if self._retire is None:
            return
        try:
            named = {name for pointer in self._seen.values() for name in pointer.objects.values()}
            stale = [
                info.name for info in await self._store.list_objects(prefix=f"{self._name}/") if info.name not in named
            ]
            if stale:
                await self._retire(stale)
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- housekeeping, as in _retire_older
            log.warning("scoped snapshot %s: retiring unreferenced chunks failed: %s", self._name, exc)

    # ------------------------------------------------------------------
    # rebuilding from L3
    # ------------------------------------------------------------------

    async def _l3_scopes(self) -> set[str | None]:
        """every scope any table holds in L3.

        :return: the scopes
        :rtype: set[str | None]
        """
        scopes: set[str | None] = set()
        for table in self._tables:
            rows = await self._l3.fetch(
                f"SELECT DISTINCT {quote_identifier(table.scope_column)} AS scope FROM {quote_identifier(table.name)}"  # noqa: S608
            )
            scopes.update(row["scope"] for row in rows)
        return scopes

    async def _read_scope(self, scope: str | None) -> dict[str, list[dict[str, Any]]]:
        """every row of one scope in each table, from L3.

        :return: table name -> rows
        :rtype: dict[str, list[dict[str, Any]]]
        :raises ValueError: for the null scope, which an equality read cannot name
        """
        if scope is None:
            raise ValueError("rows with no scope cannot be read by scope from L3")
        result: dict[str, list[dict[str, Any]]] = {}
        for table in self._tables:
            columns = tuple(self._backend.column_types(table.name))
            result[table.name] = await read_l3_rows(
                self._l3, table.name, columns, table.key, where={table.scope_column: scope}
            )
        return result

    async def _claim(self) -> bool:
        """take the rebuild claim, so one replica rebuilds while the rest wait.

        :return: whether this replica holds it
        :rtype: bool
        """
        mine = self._replica.encode("utf-8")
        revision = await self._pointers_bucket.create(key=self._claim_key, value=mine, ttl=self._claim_ttl)
        held = mine if revision is not None else await self._pointers_bucket.get(key=self._claim_key)
        return bool(held == mine)

    async def _release(self) -> None:
        """drop the rebuild claim.

        :return: nothing
        :rtype: None
        """
        try:
            await self._pointers_bucket.delete(key=self._claim_key)
        except (
            Exception
        ) as exc:  # prawduct:allow prawduct/broad-except -- a claim left behind expires on its own TTL; logged
            log.warning(
                "scoped snapshot %s: releasing the rebuild claim failed; it expires on its own: %s", self._name, exc
            )

    async def _rebuild(self, *, reason: str) -> None:
        """rebuild every scope's chunks from L3 under the claim, or wait for whoever holds it.

        :param reason: why, for the status
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        if self._ensure_buckets is not None and self._index is None and self._ever_indexed:
            await self._ensure_buckets()
        if not await self._claim():
            self._enter(SnapshotPhase.WAITING, f"{reason}; another replica is rebuilding it from L3")
            return
        try:
            caught = await self._rebuild_scopes(None, reason=reason)
        finally:
            await self._release()
        if caught is not None:
            await self._retire_unreferenced()

    async def _rebuild_scopes(self, only: set[str | None] | None, *, reason: str) -> list[str | None] | None:
        """read scopes from L3 while no write moves them, publish them, then write the index.

        :param only: the scopes to rebuild; every scope L3 holds when ``None``
        :ptype only: set[str | None] | None
        :param reason: why, for the status
        :ptype reason: str
        :return: the scopes published, or ``None`` when a write in progress made it wait
        :rtype: list[str | None] | None
        """
        result: list[str | None] | None = None
        attempts = 0
        while result is None and attempts < _REBUILD_ATTEMPTS:
            attempts += 1
            before = await self._settled() if self._settled is not None else None
            if isinstance(before, Unsettled):
                self._enter(SnapshotPhase.WAITING, f"{reason}; waiting for a write in progress ({before.reason})")
                break
            result = await self._read_and_publish(only, reason=reason, before=before)
        if result is None and attempts >= _REBUILD_ATTEMPTS:
            self._enter(SnapshotPhase.WAITING, f"{reason}; writes kept committing while it read; trying again")
        return result

    async def _read_and_publish(
        self, only: set[str | None] | None, *, reason: str, before: Any
    ) -> list[str | None] | None:
        """one rebuild attempt: read the scopes from L3, and publish them if no write committed meanwhile.

        :param only: the scopes to rebuild; every scope L3 holds when ``None``
        :ptype only: set[str | None] | None
        :param reason: why, for the status
        :ptype reason: str
        :param before: the writer's stamp before the read
        :ptype before: Any
        :return: the scopes published, or ``None`` when a write committed during the read
        :rtype: list[str | None] | None
        """
        started = time.perf_counter()
        l3_scopes = await self._l3_scopes()
        scopes = sorted(l3_scopes if only is None else (only & l3_scopes), key=lambda s: (s is None, s or ""))
        epochs = await self._epochs()
        self._enter(SnapshotPhase.REBUILDING_FROM_L3, f"rebuilding from L3: {reason}", total=len(scopes))
        semaphore = asyncio.Semaphore(self._l3_concurrency)
        read: dict[str | None, dict[str, list[dict[str, Any]]]] = {}

        async def one(scope: str | None) -> None:
            async with semaphore:
                read[scope] = await self._read_scope(scope)
            self._progress.scopes_done += 1

        await asyncio.gather(*(one(scope) for scope in scopes))
        read_at = time.perf_counter()
        after = await self._settled() if self._settled is not None else None
        published: list[str | None] | None = None
        if after != before:
            log.info("scoped snapshot %s: a write committed during the rebuild; reading again", self._name)
        else:
            for scope in scopes:
                await self.publish_without_retire(scope, int(epochs.get(scope or "", 0)), read[scope])
            gone = set() if only is not None else {s for s in (self._index or ()) if s not in l3_scopes}
            await self._write_index((set(self._index or ()) - gone) | set(scopes))
            done = time.perf_counter()
            self._progress.timings.update(
                {
                    "read_l3": round(read_at - started, 3),
                    "publish": round(done - read_at, 3),
                    "total": round(done - started, 3),
                }
            )
            if not self._ready.is_set() or only is None:
                self._become_ready(SnapshotSource.L3)
            published = list(scopes)
        return published

    async def publish_without_retire(
        self, scope: str | None, epoch: int, rows: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> None:
        """:meth:`publish` without retiring older chunks; a rebuild retires them all at its end.

        :param scope: the scope
        :ptype scope: str | None
        :param epoch: its epoch
        :ptype epoch: int
        :param rows: table name -> every row of the scope
        :ptype rows: Mapping[str, Sequence[Mapping[str, Any]]]
        :return: nothing
        :rtype: None
        """
        async with self._local:
            await self._publish_locked(scope, epoch, rows)

    async def catch_up_from_l3(self) -> list[str | None]:
        """republish every scope whose epoch in L3 is ahead of its pointer, read from L3.

        For a writer that committed and died before it published, and for scopes L3 holds that the
        snapshot does not.

        :return: the scopes republished
        :rtype: list[str | None]
        """
        epochs = await self._epochs()
        l3_scopes = await self._l3_scopes()
        behind = {
            scope
            for scope in l3_scopes
            if scope not in self._seen or int(epochs.get(scope or "", 0)) > self._seen[scope].epoch
        }
        caught: list[str | None] = []
        if behind and not await self._claim():
            log.info(
                "scoped snapshot %s: %d scopes behind L3; another replica is catching them up", self._name, len(behind)
            )
        elif behind:
            try:
                caught = await self._rebuild_scopes(behind, reason=f"{len(behind)} scopes behind L3") or []
            finally:
                await self._release()
            for scope in caught:
                await self._retire_older(scope, self._seen[scope].epoch)
            if self._ready.is_set():
                self._enter(SnapshotPhase.READY, "ready")
        return caught
