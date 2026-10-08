"""A scoped snapshot: tables a pod needs whole, held in L2 as one columnar chunk per scope, loaded into DuckDB.

**Why.** A pod that answers analytic queries over whole tables (:mod:`complete_copy`) must hold every
row in its DuckDB L1. Reading the tables from L3 a thousand rows a statement takes minutes, and doing
it again after every change, or in every new replica, is the cost this capability removes. The
tables are partitioned by a scope column (a state, say); each scope's rows of each table are kept in
NATS -- the L2 tier -- as one compact Arrow IPC chunk (zstd) in the pod's Object Store, named by the
scope, its EPOCH (the per-scope version :class:`~threetears.core.collections.scope_epochs.ScopeEpochs`
keeps) and the table's column digest, and one small KV pointer per scope names the scope's current
epoch and its chunk for each table. Design and decisions: ``docs/design-scoped-snapshot.md``.

**Every row has a scope.** The scope column of every table is never null: a publish of no scope is
refused, and a rebuild that finds rows with no scope in L3 fails, naming the table, rather than
leave them out of a copy that claims to be whole.

**Loading.** A starting replica watches the pointers (``watch_prefix``), fetches every current chunk
in parallel and loads them into its DuckDB L1 through :class:`~threetears.core.cache.duckdb.DuckDBBackend`:
seconds, with no L3 read. Each chunk's digest (the Object Store's) and row count (the pointer's) are
checked; the index names every scope, so a missing one is seen, not skipped.

**Changes.** A writer that commits a scope's change calls :meth:`ScopedSnapshot.publish` with the
scope's complete new rows: they replace the scope in its own L1, the chunks are written under the
new epoch, and then the pointer moves -- by compare-and-set, and never to a lower epoch, so a stale
writer moves nothing. Every other replica's watch sees the pointer, fetches only that scope's chunks
and replaces the scope in ONE DuckDB transaction, so a reader -- holding :meth:`ScopedSnapshot.read`
for a request -- sees the scope as it was or as it is, never half of it. No replica's L1 ever goes
back to a lower epoch of a scope. Nothing is rebuilt whole and nothing polls.

**A writer that stages.** A writer that knows its change's epoch before it commits (the write's
version, :mod:`~threetears.core.collections.scope_epochs`) writes each scope's chunks as soon as
the scope is written to L3 (:meth:`ScopedSnapshot.stage`, nothing shown, nothing held but the
compressed chunks), and after its commit moves every pointer (:meth:`ScopedSnapshot.publish_staged`).
A table it did not change keeps the scope's current chunk, but only from the epoch the writer saw
before its write; any other scope is left to :meth:`ScopedSnapshot.catch_up_from_l3`. Holding the
rebuild claim around the commit and the publish (:meth:`ScopedSnapshot.holding_rebuilds`) keeps a
replica waiting on the write from reading every scope from L3 in the moment between them.
:meth:`ScopedSnapshot.on_change` calls back after every commit to the L1, for a reader that caches
what it computed from it.

**L3 stays the truth.** NATS buckets are memory-backed, so a NATS restart loses the snapshot. A
replica that finds the pointers gone asks for the buckets again (``ensure_buckets``), takes a claim
-- renewed while it works, released only if still its own -- so one replica does the work, and
rebuilds every scope from L3, the slow and correct path, while its own L1 keeps answering from what
it last held; its status says so. :meth:`ScopedSnapshot.catch_up_from_l3` republishes any scope
whose L3 epoch is ahead of its pointer (a writer that died between its commit and its publish) and
drops scopes L3 no longer holds. A rebuild reads L3 only while no write is in progress and keeps what
it read only when no write committed meanwhile (``settled``, the writer's seqlock).

**Retirement.** Chunks are deleted by the bucket's declarer, since a delete is a purge no pod holds
(``retire``), in batches the declarer accepts, and only as one rule allows (``_deletable``), judged
against the scope's pointer as KV holds it: never a chunk the pointer names or one above its epoch
(a writer's stage), always a superseded one below it, and an unnamed one at its epoch or of a scope
with no pointer only once it is older than ``stray_age``. A reader that lost the race to a retired
chunk reads the scope's pointer again and fetches its current chunks; a scope whose chunks still
cannot be read is shown behind, retried at the recheck, and rebuilt from L3 after three failures.

**Transparent.** :meth:`ScopedSnapshot.status` says what it is doing (loading from L2, rebuilding from
L3, waiting, ready, failed), how many scopes are done, each table's rows, how long each step took,
and the last change applied.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import aclosing, asynccontextmanager, contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from threetears.nats.object_store_requests import MAX_RETIRED_OBJECTS
from threetears.observe import get_logger

from threetears.core.cache.base import quote_identifier
from threetears.core.cache.duckdb import DuckDBBackend, PartitionReplacement
from threetears.core.collections.complete_copy import Unsettled, read_l3_rows

if TYPE_CHECKING:
    from threetears.nats.client import NatsClient
    from threetears.nats.kv import NatsKvBucket
    from threetears.nats.object_store import NatsObjectStore, ObjectInfo

__all__ = [
    "ScopeChange",
    "ScopedSnapshot",
    "SnapshotPhase",
    "SnapshotSource",
    "SnapshotStatus",
    "SnapshotTable",
    "StagedScope",
    "decode_chunk",
    "encode_chunk",
    "open_tool_pod_snapshot",
    "scope_token",
]

log = get_logger(__name__)

#: the snapshot name's grammar: one KV subject token and one object-name segment
_NAME_GRAMMAR: Final = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")

#: how long a rebuild's claim lasts unless its holder renews it; a holder that dies frees it after this long
_DEFAULT_CLAIM_TTL: Final = timedelta(minutes=2)

#: how often a replica waiting on another's rebuild, or on a write in progress, looks again
_DEFAULT_RECHECK: Final = timedelta(seconds=2)

#: how many scope reads a rebuild runs on L3 at once
_DEFAULT_L3_CONCURRENCY: Final = 4

#: how many chunk reads a load runs on L2 at once
_DEFAULT_L2_CONCURRENCY: Final = 16

#: how many times a rebuild reads again after a write committed while it read
_REBUILD_ATTEMPTS: Final = 3

#: how many times a compare-and-set on a pointer or the index is tried against a racing writer
_CAS_ATTEMPTS: Final = 16

#: how many phases the status keeps in its history
_HISTORY: Final = 20

#: how many passes in a row a scope's chunks may fail to apply before it is rebuilt from L3
_BEHIND_REBUILD_AFTER: Final = 3

#: how old a chunk no pointer can vouch for must be before it may be deleted: one a writer staged for a
#: scope with no pointer yet, or a rebuild wrote at its scope's epoch under other columns, is that young
_DEFAULT_STRAY_AGE: Final = timedelta(hours=1)

#: the options :func:`open_tool_pod_snapshot` wires itself, refused when a caller gives them
_POD_WIRED_OPTIONS: Final = frozenset({"store", "pointers", "ensure_buckets", "retire"})

#: what ``_completed`` answers for a stage that may move by the usual compare-and-set
_ANY_ENTRY: Final = object()


class SnapshotPhase(StrEnum):
    """what a snapshot is doing.

    :cvar STARTING: watching the pointers, not yet caught up with them
    :cvar LOADING_FROM_L2: fetching and loading every current chunk
    :cvar REBUILDING_FROM_L3: reading scopes from L3 and publishing their chunks
    :cvar WAITING: waiting for another replica's rebuild, or for a write in progress to commit
    :cvar READY: every scope loaded; changes are applied as they are published
    :cvar FAILED: the last attempt failed, or the pointer watch ended; the detail says which
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
    :cvar L3: the tables in L3, read because L2 did not hold a usable snapshot
    """

    L2 = "l2"
    L3 = "l3"


@dataclass(frozen=True)
class SnapshotTable:
    """one table of the snapshot.

    :ivar name: the table, as the DuckDB backend and L3 name it (a TRUSTED identifier)
    :ivar scope_column: the column whose value is a row's scope; never null
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
    :ivar seconds: from starting the change to the commit of the swap
    :ivar applied_at: when the swap committed
    """

    scope: str
    epoch: int
    rows: int
    seconds: float
    applied_at: datetime


@dataclass(frozen=True)
class StagedScope:
    """a scope's chunks written under a new epoch whose pointer has not moved yet (:meth:`ScopedSnapshot.stage`).

    :ivar scope: the scope
    :ivar epoch: the epoch the chunks are written under
    :ivar objects: table -> the chunk written for it; a table left out keeps its rows
    :ivar rows: table -> the rows its chunk holds
    """

    scope: str
    epoch: int
    objects: Mapping[str, str]
    rows: Mapping[str, int]


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
    :ivar history: the last phases entered, in order, without repeats in a row
    :ivar last_change: the last scope change applied, if any
    :ivar ready_at: when the copy first became ready
    :ivar behind: scope -> why its current chunks could not be applied here; the copy serves an older
        epoch of each until a retry, or a rebuild of the scope from L3, applies it
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
    behind: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _Pointer:
    """what a scope's pointer says: its epoch, and its chunk for each table."""

    scope: str
    epoch: int
    objects: Mapping[str, str]
    rows: Mapping[str, int]
    schema: Mapping[str, str]
    #: when it was written (epoch seconds); None for a pointer written before it was recorded
    published_at: float | None = field(default=None, compare=False)

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
                "at": time.time() if self.published_at is None else self.published_at,
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
            scope=str(body["scope"]),
            epoch=int(body["epoch"]),
            objects={t: str(v["object"]) for t, v in tables.items()},
            rows={t: int(v["rows"]) for t, v in tables.items()},
            schema={t: str(v) for t, v in body["schema"].items()},
            published_at=float(body["at"]) if "at" in body else None,
        )

    def supersedes(self, other: _Pointer | None) -> bool:
        """whether moving a scope from ``other`` to this pointer moves it forward.

        :param other: the pointer held now, if any
        :ptype other: _Pointer | None
        :return: whether this one is newer, or the same epoch with other chunks (a rebuild under
            other columns)
        :rtype: bool
        """
        return (
            other is None or self.epoch > other.epoch or (self.epoch == other.epoch and self.objects != other.objects)
        )


def scope_token(scope: str) -> str:
    """a scope as one KV key token and one object-name segment, reversibly.

    Letters, digits, ``-`` and ``_`` stand for themselves; every other byte is ``=`` and two hex digits.

    :param scope: the scope value
    :ptype scope: str
    :return: the token
    :rtype: str
    """
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


class _Claim:
    """the rebuild claim this replica holds: renewed while held, released only while still its own."""

    def __init__(self, bucket: NatsKvBucket, *, key: str, owner: bytes, ttl: timedelta) -> None:
        self._bucket = bucket
        self._key = key
        self._owner = owner
        self._ttl = ttl
        self._revision: int | None = None
        self._renewal: asyncio.Task[None] | None = None

    async def take(self) -> bool:
        """take the claim when nobody holds it, and keep renewing it.

        :return: whether this replica holds it
        :rtype: bool
        """
        self._revision = await self._bucket.create(key=self._key, value=self._owner, ttl=self._ttl)
        if self._revision is not None:
            self._renewal = asyncio.create_task(self._renew(), name=f"snapshot-claim:{self._key}")
        return self._revision is not None

    async def _renew(self) -> None:
        """renew the claim at a third of its lifetime, by compare-and-set, until it is released or lost.

        :return: nothing
        :rtype: None
        """
        interval = max(self._ttl.total_seconds() / 3, 0.2)
        renewed = time.monotonic()
        while self._revision is not None:
            await asyncio.sleep(interval)
            try:
                revision = await self._bucket.update(
                    key=self._key, value=self._owner, revision=self._revision, ttl=self._ttl
                )
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- a renewal that cannot reach NATS is tried again until the claim's lifetime is spent; then it is lost, said so
                if time.monotonic() - renewed < self._ttl.total_seconds():
                    log.warning("renewing the snapshot rebuild claim %s failed; trying again: %s", self._key, exc)
                    continue
                log.warning("the snapshot rebuild claim %s lapsed while it could not be renewed: %s", self._key, exc)
                revision = None
            if revision is None:
                log.warning("the snapshot rebuild claim %s was lost; another replica may take it", self._key)
            else:
                renewed = time.monotonic()
            self._revision = revision

    async def release(self) -> None:
        """stop renewing and delete the claim, only if it is still this replica's.

        :return: nothing
        :rtype: None
        """
        try:
            if self._renewal is not None:
                self._renewal.cancel()
                # waited, not awaited: awaiting a cancelled task raises its CancelledError, and
                # suppressing that would also swallow a cancellation of THIS task arriving meanwhile
                await asyncio.wait({self._renewal})
        finally:
            # even when this task is being cancelled: a claim left behind makes every other replica
            # wait out its lifetime. read after the renewal stopped, so it is the revision last written
            revision, self._revision = self._revision, None
            if revision is not None:
                try:
                    await self._bucket.delete(key=self._key, revision=revision)
                except (
                    Exception
                ) as exc:  # prawduct:allow prawduct/broad-except -- a claim left behind expires on its own TTL; logged
                    log.warning("releasing snapshot rebuild claim %s failed; it expires on its own: %s", self._key, exc)


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
    :param retire_batch: the most names one retire asks for
    :ptype retire_batch: int
    :param l2_concurrency: chunk reads at once
    :ptype l2_concurrency: int
    :param l3_concurrency: scope reads on L3 at once during a rebuild
    :ptype l3_concurrency: int
    :param claim_ttl: how long a rebuild's claim lasts if its holder dies; renewed while held
    :ptype claim_ttl: timedelta
    :param recheck: how often a waiting replica looks again
    :ptype recheck: timedelta
    :param pointer_watch_heartbeat: the pointer watch's heartbeat; a lost consumer is replaced after three
    :ptype pointer_watch_heartbeat: timedelta
    :param stray_age: how old a chunk no pointer vouches for must be before it is deleted (a stage for
        a scope with no pointer yet, or a same-epoch chunk under other columns); above any write's length
    :ptype stray_age: timedelta
    :raises ValueError: when the name is outside its grammar or names no table
    """

    def __init__(
        self,
        *,
        name: str,
        tables: Sequence[SnapshotTable],
        backend: DuckDBBackend,
        store: NatsObjectStore,
        pointers: NatsKvBucket,
        l3: Any,
        epochs: Callable[[], Awaitable[Mapping[str, int]]],
        settled: Callable[[], Awaitable[Any]] | None = None,
        ensure_buckets: Callable[[], Awaitable[None]] | None = None,
        retire: Callable[[list[str]], Awaitable[Any]] | None = None,
        retire_batch: int = MAX_RETIRED_OBJECTS,
        l2_concurrency: int = _DEFAULT_L2_CONCURRENCY,
        l3_concurrency: int = _DEFAULT_L3_CONCURRENCY,
        claim_ttl: timedelta = _DEFAULT_CLAIM_TTL,
        recheck: timedelta = _DEFAULT_RECHECK,
        pointer_watch_heartbeat: timedelta = timedelta(seconds=5),
        stray_age: timedelta = _DEFAULT_STRAY_AGE,
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
        self._retire_batch = max(1, min(retire_batch, MAX_RETIRED_OBJECTS))
        self._l2_concurrency = l2_concurrency
        self._l3_concurrency = l3_concurrency
        self._claim_ttl = claim_ttl
        self._recheck = recheck
        self._heartbeat = pointer_watch_heartbeat
        self._stray_age = stray_age
        self._schema = {t.name: backend.schema_digest(t.name) for t in self._tables}
        self._replica = uuid.uuid7().hex.encode("utf-8")
        # what the pointer bucket says, as the watch last delivered it
        self._seen: dict[str, _Pointer] = {}
        self._index: frozenset[str] | None = None
        # an index a publish created while no rebuild had written one names only the scopes published
        # since, not every scope; it is not a snapshot to load
        self._index_partial = False
        # scopes the index still names whose pointer the watch saw deleted: a removal deletes the
        # pointer before it takes the scope out of the index, so for a ready copy this is a removal
        # in flight, not a lost snapshot
        self._removing: set[str] = set()
        self._caught_up = False
        self._ever_indexed = False
        self._watch_ended: str | None = None
        # what the L1 holds
        self._applied: dict[str, _Pointer] = {}
        self._local = asyncio.Lock()
        self._changed = asyncio.Event()
        self._ready = asyncio.Event()
        self._progress = _Progress()
        # the facts the phase is derived from (_settle): the step in progress, the waits outstanding
        # per control flow (the worker's passes, a catch-up), whether the worker's last pass failed
        self._activity: tuple[SnapshotPhase, str] | None = None
        self._waiting: dict[str, str] = {}
        self._waited: dict[str, bool] = {}
        self._failure: str | None = None
        # scope -> (how many passes in a row its current chunks could not be applied, why the last failed)
        self._behind: dict[str, tuple[int, str]] = {}
        # table -> scope -> rows the L1 holds, kept as each replacement commits, so the status never
        # queries DuckDB (a query would wait on the lock a load or a swap holds, on the event loop)
        self._rows: dict[str, dict[str, int]] = {t.name: {} for t in self._tables}
        self._tasks: list[asyncio.Task[None]] = []
        self._listeners: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # keys and names: the one pattern for each
    # ------------------------------------------------------------------

    @property
    def _index_key(self) -> str:
        return f"{self._name}.index"

    @property
    def _claim_key(self) -> str:
        return f"{self._name}.rebuild"

    def _pointer_key(self, scope: str) -> str:
        return f"{self._name}.s.{scope_token(scope)}"

    def _object_prefix(self, scope: str) -> str:
        return f"{self._name}/{scope_token(scope)}/"

    def _object_name(self, scope: str, epoch: int, table: str) -> str:
        """``{name}/{scope}/{epoch}/{table}.{column digest}``: a chunk names the columns it holds.

        :return: the object name
        :rtype: str
        """
        return f"{self._object_prefix(scope)}{epoch}/{table}.{self._schema[table]}"

    def _epoch_of(self, scope: str, object_name: str) -> int | None:
        """the epoch an object of ``scope`` was written under, or ``None`` when it is not one of the scope's.

        :return: the epoch
        :rtype: int | None
        """
        prefix = self._object_prefix(scope)
        head = object_name[len(prefix) :].split("/", 1)[0] if object_name.startswith(prefix) else ""
        return int(head) if head.isdigit() else None

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def _settle(self) -> None:
        """derive the phase and its detail from the facts: the one place either is written.

        The pointer watch's end comes first (the replica no longer applies changes, whatever else is
        true); then the step in progress (loading, rebuilding); then any wait outstanding; then a
        failed pass; then ready, with the scopes behind named; else starting.

        :return: nothing
        :rtype: None
        """
        phase: SnapshotPhase
        if self._watch_ended is not None:
            phase, detail = SnapshotPhase.FAILED, self._watch_ended
        elif self._activity is not None:
            phase, detail = self._activity
        elif self._waiting:
            phase, detail = SnapshotPhase.WAITING, next(iter(self._waiting.values()))
        elif self._failure is not None:
            phase, detail = SnapshotPhase.FAILED, self._failure
        elif self._ready.is_set():
            behind = sorted(self._behind)
            phase, detail = SnapshotPhase.READY, f"ready; behind on {behind}" if behind else "ready"
            foreign = sorted(scope for scope, pointer in self._seen.items() if not self._loadable(pointer))
            if foreign:
                detail += f"; {len(foreign)} scopes' pointers carry other columns, so they are served from L3"
        else:
            phase, detail = SnapshotPhase.STARTING, "watching the pointers"
        if self._behind and phase is not SnapshotPhase.READY:
            # whatever else it is doing, a replica serving an older epoch of some scopes says which
            detail = f"{detail}; behind on {sorted(self._behind)}"
        progress = self._progress
        if progress.phase is not phase or progress.detail != detail:
            log.info("scoped snapshot %s: %s", self._name, detail, extra={"extra_data": {"phase": phase.value}})
        progress.phase = phase
        progress.detail = detail
        if progress.history[-1] is not phase:
            progress.history.append(phase)
            del progress.history[:-_HISTORY]

    @contextmanager
    def _doing(self, phase: SnapshotPhase, detail: str, *, total: int) -> Iterator[None]:
        """a step in progress (loading from L2, rebuilding from L3), counted over ``total`` scopes.

        :return: nothing
        :rtype: Iterator[None]
        """
        self._activity = (phase, detail)
        self._progress.scopes_total = total
        self._progress.scopes_done = 0
        self._settle()
        try:
            yield
        finally:
            self._activity = None
            self._settle()

    def _wait(self, flow: str, reason: str) -> None:
        """record that ``flow`` is waiting, and why; it stays waiting until a run of it does not wait.

        :param flow: the control flow (``sync``, the worker's passes; ``catch_up``)
        :ptype flow: str
        :param reason: why
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        self._waiting[flow] = reason
        self._waited[flow] = True
        self._settle()

    def _begin_run(self, flow: str) -> None:
        """start one run of ``flow``: it has not waited yet.

        :return: nothing
        :rtype: None
        """
        self._waited[flow] = False

    def _end_run(self, flow: str) -> None:
        """end one run of ``flow``: a run that did not wait clears the flow's wait.

        :return: nothing
        :rtype: None
        """
        if not self._waited.get(flow, False):
            self._waiting.pop(flow, None)
        self._settle()

    def _count(self, replacements: Iterable[PartitionReplacement]) -> None:
        """record what each replaced scope now holds, after the replacement committed.

        :return: nothing
        :rtype: None
        """
        for r in replacements:
            held = r.arrow.num_rows if r.arrow is not None else len(r.rows or ())
            self._rows[r.table][str(r.value)] = held

    def status(self) -> SnapshotStatus:
        """what the snapshot is doing and has done.

        :return: the status
        :rtype: SnapshotStatus
        """
        progress = self._progress
        return SnapshotStatus(
            phase=progress.phase,
            source=progress.source,
            detail=progress.detail,
            scopes_total=progress.scopes_total,
            scopes_done=progress.scopes_done,
            rows={table: sum(scopes.values()) for table, scopes in self._rows.items()},
            timings=dict(progress.timings),
            history=tuple(progress.history),
            last_change=progress.last_change,
            ready_at=progress.ready_at,
            behind={scope: why for scope, (_, why) in sorted(self._behind.items())},
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

    def on_change(self, listener: Callable[[], None]) -> Callable[[], None]:
        """call ``listener`` after every commit that changes what this L1 holds.

        A load from L2, a scope applied or dropped, a publish made here: each calls every listener
        once its DuckDB transaction has committed, so a reader opened from the listener sees the
        change. Called on the event loop; a listener that raises is logged, never propagated.

        :param listener: called with no arguments
        :ptype listener: Callable[[], None]
        :return: a function that removes the listener
        :rtype: Callable[[], None]
        """
        self._listeners.append(listener)

        def remove() -> None:
            with suppress(ValueError):  # NOSILENT: removing a listener twice is a no-op
                self._listeners.remove(listener)

        return remove

    def _notify(self) -> None:
        """call every change listener; one that raises is logged.

        :return: nothing
        :rtype: None
        """
        for listener in list(self._listeners):
            try:
                listener()
            except Exception:  # prawduct:allow prawduct/broad-except -- a listener's failure must not undo or stop an L1 change already committed; logged
                log.exception("scoped snapshot %s: a change listener failed", self._name)

    @asynccontextmanager
    async def holding_rebuilds(self, *, attempts: int = 10) -> AsyncIterator[bool]:
        """hold the rebuild claim for the block, so no replica starts a rebuild from L3 meanwhile.

        For a writer that is about to commit a change to L3 and publish it here: between its commit
        and its publish, a replica waiting on the write would otherwise read every scope from L3
        (the slow path) only to find the same rows already published. The claim is tried a few times
        (a replica that holds it only to look at the writer's seqlock lets go at once); when another
        replica is really rebuilding, the block runs without it, which is correct, only slower.

        :param attempts: tries at the claim, a tenth of a second apart
        :ptype attempts: int
        :return: whether the claim is held
        :rtype: AsyncIterator[bool]
        """
        claim = self._claim()
        held = False
        for attempt in range(max(1, attempts)):
            if attempt:
                await asyncio.sleep(0.1)
            if await claim.take():
                held = True
                break
        try:
            yield held
        finally:
            if held:
                await claim.release()

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
        if self._tasks:
            # waited, not awaited: see _Claim.release
            await asyncio.wait(self._tasks)
        self._tasks = []

    async def _watch(self) -> None:
        """follow the pointer bucket and wake the worker on every change; say so loudly if it ends.

        :return: nothing
        :rtype: None
        """
        try:
            await self._follow(f"{self._name}.")
            ended = "the pointer watch ended"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the watch is the replica's only news of changes; whatever ends it is logged and stated in the status rather than lost with the task
            log.exception("scoped snapshot %s: the pointer watch ended", self._name)
            ended = f"the pointer watch ended ({type(exc).__name__}: {exc})"
        self._watch_ended = f"{ended}; changes are no longer applied"
        self._settle()

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
                    body = None if update.value is None else json.loads(update.value)
                    self._index = None if body is None else frozenset(body["scopes"])
                    self._index_partial = body is not None and bool(body.get("partial", False))
                    self._ever_indexed = self._ever_indexed or self._index is not None
                    self._removing &= self._index or frozenset()
                elif update.key.startswith(f"{prefix}s."):
                    self._record_pointer(update.key, update.value)
                else:
                    # the rebuild claim shares the prefix: its takes, renewals and releases are no
                    # news of the data, and waking on them would turn a failing pass into a spin
                    continue
                self._changed.set()

    def _record_pointer(self, key: str, value: bytes | None) -> None:
        """record one pointer the watch delivered, or its removal.

        :param key: the pointer's key
        :ptype key: str
        :param value: its value, or ``None`` when it was removed
        :ptype value: bytes | None
        :return: nothing
        :rtype: None
        """
        if value is None:
            for scope in [s for s in self._seen if self._pointer_key(s) == key]:
                del self._seen[scope]
                self._removing.add(scope)
            return
        try:
            pointer = _Pointer.decode(value)
        except (ValueError, KeyError, TypeError) as exc:
            log.error("scoped snapshot %s: pointer %s does not decode: %s", self._name, key, exc)
            return
        self._removing.discard(pointer.scope)
        if pointer.supersedes(self._seen.get(pointer.scope)):
            self._seen[pointer.scope] = pointer

    async def _work(self) -> None:
        """bring the L1 to what the pointers say, each time they change or a wait is due.

        :return: nothing
        :rtype: None
        """
        while self._watch_ended is None:
            try:
                await asyncio.wait_for(self._changed.wait(), self._recheck.total_seconds())
            except TimeoutError:
                pass  # NOSILENT: the recheck interval passed with no change; a pass runs anyway
            self._changed.clear()
            if not self._caught_up or self._watch_ended is not None:
                continue
            try:
                await self._sync()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- the worker outlives any one failed attempt; logged and stated in the status, tried again at the next recheck
                log.exception("scoped snapshot %s: sync failed", self._name)
                self._failure = f"failed ({type(exc).__name__}: {exc}); trying again"
                self._waiting.pop("sync", None)
                self._settle()

    async def _sync(self) -> None:
        """one pass: load the snapshot, rebuild it, or apply the scopes that moved; READY when done.

        :return: nothing
        :rtype: None
        """
        self._begin_run("sync")
        index = self._index
        removing = self._removing if self._ready.is_set() else set()
        complete = (
            index is not None
            and not self._index_partial
            and all(scope in self._seen or scope in removing for scope in index)
        )
        mismatched = complete and any(
            not self._loadable(self._seen[scope]) for scope in index or () if scope in self._seen
        )
        loaded = False
        if not self._ready.is_set() and complete and not mismatched:
            try:
                await self._load({scope: self._seen[scope] for scope in index or ()}, cold=True)
                loaded = True
            except _Lost as exc:
                log.warning("scoped snapshot %s: the snapshot in L2 is not loadable: %s", self._name, exc)
        if loaded:
            pass
        elif not self._ready.is_set() or not complete:
            await self._rebuild(reason=self._why_rebuild(index, complete, mismatched))
        else:
            await self._apply_moved(index or frozenset())
        # a pass that waited stays WAITING; one that did not clears the worker's wait and failure
        self._failure = None
        self._end_run("sync")

    def _why_rebuild(self, index: frozenset[str] | None, complete: bool, mismatched: bool) -> str:
        """a plain reason for rebuilding from L3.

        :return: the reason
        :rtype: str
        """
        if index is None and self._ever_indexed:
            reason = "NATS lost the snapshot"
        elif index is None:
            reason = "no snapshot in NATS yet"
        elif self._index_partial:
            reason = "NATS holds only scopes published since it lost the snapshot"
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

    def _loadable(self, pointer: _Pointer) -> bool:
        """whether this replica can load ``pointer``'s chunks: they hold exactly its own columns.

        :return: True when the pointer's column digests are this replica's
        :rtype: bool
        """
        return dict(pointer.schema) == self._schema

    def _ahead(self, pointer: _Pointer, applied: _Pointer | None) -> bool:
        """whether this L1 should take ``pointer``: chunks it can load, past what it holds.

        The one judgement every apply makes. A pointer of other columns is never ahead: a replica of
        another code version wrote it, and its chunks would load wrong columns here.

        :return: True when the pointer is loadable and moves the scope forward
        :rtype: bool
        """
        return self._loadable(pointer) and pointer.supersedes(applied)

    def _foreign_ahead(self, pointer: _Pointer, applied: _Pointer | None) -> bool:
        """whether a pointer of other columns names a LATER epoch than this L1 holds: a write this
        replica cannot load from its chunks, which it rebuilds from L3 instead.

        :return: True for a later epoch under other columns
        :rtype: bool
        """
        return not self._loadable(pointer) and (applied is None or pointer.epoch > applied.epoch)

    def _mark_foreign(self, scope: str) -> None:
        """record a scope whose later epoch is under other columns: rebuilt from L3 at the next pass.

        :return: nothing
        :rtype: None
        """
        self._behind[scope] = (_BEHIND_REBUILD_AFTER, "its pointer carries other columns; rebuilt from L3")

    async def _fetch(self, pointer: _Pointer) -> list[PartitionReplacement]:
        """every table's chunk of one scope, decoded and checked.

        :param pointer: the scope's pointer
        :ptype pointer: _Pointer
        :return: the scope's replacements
        :rtype: list[PartitionReplacement]
        :raises _Lost: when a chunk is gone or does not hold what its pointer says
        """
        from threetears.nats import ObjectNotFoundError, ObjectStoreError  # noqa: PLC0415

        if not self._loadable(pointer):
            # every load passes here: no path can put another column set's chunks in this L1
            raise _Lost(f"scope {pointer.scope!r}'s pointer carries other columns than this replica's")
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

    async def _fetch_current(self, scope: str, pointer: _Pointer) -> tuple[_Pointer, list[PartitionReplacement]]:
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
            if raw is None or not _Pointer.decode(raw).supersedes(pointer):
                raise
            current = _Pointer.decode(raw)
            replacements = await self._fetch(current)
        return current, replacements

    async def _load(self, pointers: Mapping[str, _Pointer], *, cold: bool) -> None:
        """fetch the scopes' chunks in parallel and replace those scopes in one transaction.

        :param pointers: the scopes to load and their pointers
        :ptype pointers: Mapping[str, _Pointer]
        :param cold: whether this is the first load (every scope, and the copy becomes ready)
        :ptype cold: bool
        :return: nothing
        :rtype: None
        :raises _Lost: when a chunk is gone or wrong; the other reads are cancelled
        """
        with (
            self._doing(SnapshotPhase.LOADING_FROM_L2, "loading from NATS", total=len(pointers))
            if cold
            else nullcontext()
        ):
            await self._load_inner(pointers, cold=cold)

    async def _load_inner(self, pointers: Mapping[str, _Pointer], *, cold: bool) -> None:
        """the body of :meth:`_load`.

        :return: nothing
        :rtype: None
        :raises _Lost: when a chunk is gone or wrong
        """
        started = time.perf_counter()
        semaphore = asyncio.Semaphore(self._l2_concurrency)
        fetched: list[tuple[_Pointer, list[PartitionReplacement]]] = []

        async def one(scope: str, pointer: _Pointer) -> None:
            async with semaphore:
                fetched.append(await self._fetch_current(scope, pointer))
            if cold:
                self._progress.scopes_done += 1

        try:
            async with asyncio.TaskGroup() as group:
                for scope, pointer in pointers.items():
                    group.create_task(one(scope, pointer))
        except* _Lost as lost:
            raise lost.exceptions[0] from None
        fetched_at = time.perf_counter()
        # in scope order, not arrival order: DuckDB scans rows in the order they were written, and a
        # floating-point sum over them depends on that order, so every replica loading the same chunks
        # holds them in the same order and sums them to the same last digit
        fetched.sort(key=lambda item: item[0].scope)
        replacements = [replacement for _, scope_replacements in fetched for replacement in scope_replacements]
        async with self._local:
            fresh = [(p, r) for p, r in fetched if p.supersedes(self._applied.get(p.scope))]
            await asyncio.to_thread(self._backend.replace_partitions, [x for _, rs in fresh for x in rs])
            for pointer, _ in fresh:
                self._applied[pointer.scope] = pointer
                self._behind.pop(pointer.scope, None)
            self._count(x for _, rs in fresh for x in rs)
        self._notify()
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
        self._ready.set()
        self._settle()

    async def _apply_moved(self, index: frozenset[str]) -> None:
        """apply each scope whose pointer moved past what the L1 holds; drop scopes the index left.

        :param index: the scopes the snapshot holds
        :ptype index: frozenset[str]
        :return: nothing
        :rtype: None
        """
        # a scope leaves only once its pointer is gone too: a remover deletes the pointer before it
        # takes the scope out of the index, and puts it back if a writer published it meanwhile, so a
        # scope out of the index with its pointer still standing is in flight, not gone
        gone = [scope for scope in self._applied if scope not in index and scope not in self._seen]
        if gone:
            await self._drop_locally(gone)
        moved = {
            scope: self._seen[scope]
            for scope in index
            if scope in self._seen and self._ahead(self._seen[scope], self._applied.get(scope))
        }
        for scope in index:
            if scope in self._seen and self._foreign_ahead(self._seen[scope], self._applied.get(scope)):
                self._mark_foreign(scope)
        failed = await self._apply_pointers(moved)
        # a scope removed before it was ever applied leaves no `behind` entry to report, or to rebuild
        for scope in [scope for scope in self._behind if scope not in index and scope not in self._seen]:
            del self._behind[scope]
        for scope, why in failed.items():
            self._behind[scope] = (self._behind.get(scope, (0, ""))[0] + 1, why)
        # a scope that failed is tried again at the next recheck, not at once; one that keeps failing
        # is rebuilt from L3, which writes its chunks again
        stuck = {scope for scope, (failures, _) in self._behind.items() if failures >= _BEHIND_REBUILD_AFTER}
        if stuck:
            await self._rebuild_stuck(stuck)
        self._settle()

    async def _rebuild_stuck(self, scopes: set[str]) -> None:
        """rebuild from L3 scopes whose current chunks kept failing to apply, under the claim.

        :param scopes: the scopes
        :ptype scopes: set[str]
        :return: nothing
        :rtype: None
        """
        claim = self._claim()
        if not await claim.take():
            log.info("scoped snapshot %s: another replica holds the claim; %s wait", self._name, sorted(scopes))
            return
        try:
            log.warning(
                "scoped snapshot %s: rebuilding %s from L3: their chunks kept failing", self._name, sorted(scopes)
            )
            rebuilt = await self._rebuild_scopes(
                scopes, reason=f"{len(scopes)} scopes could not be applied", flow="sync"
            )
        finally:
            await claim.release()
        # a rebuild that returns has applied every scope the index names at its pointer (or it raises,
        # leaving them behind), and each scope applied left `behind` as it was applied
        for scope in rebuilt or ():
            self._behind.pop(scope, None)

    async def _apply_pointers(self, pointers: Mapping[str, _Pointer]) -> dict[str, str]:
        """fetch each scope's chunks and replace it in this L1, unless the L1 already holds as new.

        :param pointers: the scopes to apply and their pointers
        :ptype pointers: Mapping[str, _Pointer]
        :return: scope -> why its chunks could not be read, each logged
        :rtype: dict[str, str]
        """
        failed: dict[str, str] = {}
        for scope, pointer in pointers.items():
            started = time.perf_counter()
            try:
                current, replacements = await self._fetch_current(scope, pointer)
            except _Lost as exc:
                log.warning("scoped snapshot %s: scope %r could not be applied: %s", self._name, scope, exc)
                failed[scope] = str(exc)
                continue
            async with self._local:
                if not current.supersedes(self._applied.get(scope)):
                    continue
                await asyncio.to_thread(self._backend.replace_partitions, replacements)
                self._applied[scope] = current
                self._count(replacements)
            self._behind.pop(scope, None)
            self._notify()
            self._record_change(scope, current, replacements, started)
        return failed

    async def _drop_locally(self, scopes: Sequence[str]) -> None:
        """remove scopes from this L1, in one transaction.

        :param scopes: the scopes
        :ptype scopes: Sequence[str]
        :return: nothing
        :rtype: None
        """
        async with self._local:
            await asyncio.to_thread(
                self._backend.replace_partitions,
                [
                    PartitionReplacement(table=t.name, column=t.scope_column, value=scope, rows=[])
                    for scope in scopes
                    for t in self._tables
                ],
            )
            for scope in scopes:
                self._applied.pop(scope, None)
                self._behind.pop(scope, None)
                for table in self._rows.values():
                    table.pop(scope, None)
        self._notify()
        log.info("scoped snapshot %s: scopes dropped", self._name, extra={"extra_data": {"scopes": list(scopes)}})

    def _record_change(
        self, scope: str, pointer: _Pointer, replacements: Sequence[PartitionReplacement], started: float
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

    async def publish(self, scope: str, epoch: int, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        """make ``rows`` the scope's content at ``epoch``: in this L1, in its chunks, then its pointer.

        Called by the writer once the scope's change is committed in L3, with every row the scope now
        holds in each table. The other replicas apply it when they see the pointer move. A publish
        below the scope's current epoch changes nothing anywhere. Chunks of the scope's older epochs
        are retired.

        :param scope: the scope; never ``None``
        :ptype scope: str
        :param epoch: its epoch after the change
        :ptype epoch: int
        :param rows: table name -> every row of the scope in that table
        :ptype rows: Mapping[str, Sequence[Mapping[str, Any]]]
        :return: nothing
        :rtype: None
        :raises ValueError: when the scope is ``None`` or ``rows`` does not name every table
        """
        if scope is None:
            raise ValueError("a scoped snapshot holds no rows without a scope; every row's scope column is set")
        missing = [t.name for t in self._tables if t.name not in rows]
        if missing:
            raise ValueError(f"a publish of scope {scope!r} gives no rows for {missing}")
        started = time.perf_counter()
        async with self._local:
            pointer = await self._publish_locked(scope, epoch, rows)
        if pointer is not None:
            replacements = [
                PartitionReplacement(table=t.name, column=t.scope_column, value=scope, rows=rows[t.name])
                for t in self._tables
            ]
            self._record_change(scope, pointer, replacements, started)
            await self._update_index(add={scope})
            await self._retire_older(scope, epoch)

    async def _publish_locked(
        self, scope: str, epoch: int, rows: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> _Pointer | None:
        """the body of :meth:`publish`, under the L1 lock: refuse a stale epoch, replace locally,
        write the chunks, move the pointer by compare-and-set.

        :return: the pointer written, or ``None`` when the scope is already at a later epoch
        :rtype: _Pointer | None
        """
        held = await self._pointers_bucket.get(key=self._pointer_key(scope))
        current = None if held is None else _Pointer.decode(held)
        applied = self._applied.get(scope)
        stale = (current is not None and current.epoch > epoch) or (applied is not None and applied.epoch > epoch)
        result: _Pointer | None = None
        if stale:
            log.warning(
                "scoped snapshot %s: a publish of scope %r at epoch %d is behind the scope's epoch; nothing moved",
                self._name,
                scope,
                epoch,
            )
            if current is not None:
                # a rebuild applies what it refused here from this pointer before it declares ready;
                # the watch may not have delivered it yet
                self._note_seen(current)
        else:
            local = [
                PartitionReplacement(
                    table=t.name, column=t.scope_column, value=scope, rows=rows[t.name], primary_key=t.key
                )
                for t in self._tables
            ]
            await asyncio.to_thread(self._backend.replace_partitions, local)
            self._count(local)
            self._behind.pop(scope, None)
            self._notify()
            objects: dict[str, str] = {}
            counts: dict[str, int] = {}
            for table in self._tables:
                arrow = await asyncio.to_thread(
                    self._backend.export_partition, table.name, table.scope_column, scope, order_by=table.key
                )
                objects[table.name] = await self._put_chunk(scope, epoch, table, arrow)
                counts[table.name] = arrow.num_rows
            pointer = _Pointer(scope=scope, epoch=epoch, objects=objects, rows=counts, schema=self._schema)
            self._applied[scope] = pointer
            result = await self._move_pointer(pointer)
        return result

    async def _put_chunk(self, scope: str, epoch: int, table: SnapshotTable, arrow: Any) -> str:
        """write one table's chunk of a scope at an epoch; the one place a chunk is written.

        :param scope: the scope
        :ptype scope: str
        :param epoch: the epoch
        :ptype epoch: int
        :param table: the table
        :ptype table: SnapshotTable
        :param arrow: the scope's rows of the table
        :ptype arrow: pyarrow.Table
        :return: the chunk's name
        :rtype: str
        """
        from threetears.nats import ObjectExistsError  # noqa: PLC0415

        data = await asyncio.to_thread(encode_chunk, arrow)
        name = self._object_name(scope, epoch, table.name)
        # NOSILENT: a replica rebuilding the same scope at the same epoch under the same columns wrote the same rows
        with suppress(ObjectExistsError):
            await self._store.put(name, data)
        return name

    async def stage(self, scope: str, epoch: int, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> StagedScope:
        """write a scope's chunks under ``epoch`` without moving its pointer or touching this L1.

        For a writer that knows a change's epoch before it commits (the write's version is the
        epoch, :mod:`~threetears.core.collections.scope_epochs`): it stages each scope as soon as
        its rows are written to L3, keeping only the compressed chunks rather than every row, and
        after its commit moves every pointer at once (:meth:`publish_staged`). Nothing reads a
        staged chunk until its pointer moves, so a write that never commits shows nothing. A staged
        chunk is above its scope's pointer, which no retirement touches (:meth:`_deletable`); if its
        write never commits, the scope's next publish retires it.

        :param scope: the scope; never ``None``
        :ptype scope: str
        :param epoch: the epoch the write will commit
        :ptype epoch: int
        :param rows: table name -> every row the scope holds in that table after the write; a table
            left out is one the write did not change, kept from the scope's current chunk
        :ptype rows: Mapping[str, Sequence[Mapping[str, Any]]]
        :return: the staged chunks
        :rtype: StagedScope
        :raises ValueError: when the scope is ``None`` or a table is not one of the snapshot's
        """
        if scope is None:
            raise ValueError("a scoped snapshot holds no rows without a scope; every row's scope column is set")
        unknown = sorted(set(rows) - {t.name for t in self._tables})
        if unknown:
            raise ValueError(f"a stage of scope {scope!r} names tables the snapshot does not hold: {unknown}")
        objects: dict[str, str] = {}
        counts: dict[str, int] = {}
        for table in self._tables:
            if table.name in rows:
                arrow = await self._export(table, scope, rows[table.name])
                objects[table.name] = await self._put_chunk(scope, epoch, table, arrow)
                counts[table.name] = arrow.num_rows
        return StagedScope(scope=scope, epoch=epoch, objects=objects, rows=counts)

    async def _export(self, table: SnapshotTable, scope: str, rows: Sequence[Mapping[str, Any]]) -> Any:
        """the Arrow table a scope of ``table`` holds with ``rows``, typed by this L1, changing nothing.

        :return: the rows
        :rtype: pyarrow.Table
        """
        return await asyncio.to_thread(
            self._backend.export_rows,
            table.name,
            table.scope_column,
            scope,
            rows,
            primary_key=table.key,
            order_by=table.key,
        )

    async def publish_staged(
        self, staged: Sequence[StagedScope], *, carry_at: Mapping[str, int], whole: bool = False
    ) -> tuple[list[str], list[str]]:
        """move the pointers of staged scopes, once the write that staged them has committed in L3.

        A table a stage left out keeps the scope's current chunk, but only when the scope's pointer
        is at ``carry_at[scope]`` -- the epoch the writer saw before its write, so the carried chunk
        holds what L3 held then. A scope with no pointer and ``carry_at`` 0 is new: its missing
        tables are empty, when the snapshot is known whole (``whole``, or a whole index that does not
        name it). Any other scope is skipped and left to :meth:`catch_up_from_l3`, which republishes
        it from L3 because its L3 epoch is now ahead of its pointer. Every replica applies the moved
        scopes from its pointer watch, this one included.

        :param staged: the scopes :meth:`stage` wrote
        :ptype staged: Sequence[StagedScope]
        :param carry_at: scope -> its epoch before the write (0 for a scope it never had)
        :ptype carry_at: Mapping[str, int]
        :param whole: whether the staged scopes are every scope L3 holds (a write of everything)
        :ptype whole: bool
        :return: the scopes whose pointers moved, and the scopes skipped (left to the catch-up). A
            scope a racing writer already moved past the staged epoch is in neither, and is logged:
            nothing is owed for it
        :rtype: tuple[list[str], list[str]]
        """
        moved: list[str] = []
        skipped: list[str] = []
        superseded: list[str] = []
        for scope in staged:
            completed = await self._completed(scope, int(carry_at.get(scope.scope, 0)), whole=whole)
            if completed is None:
                skipped.append(scope.scope)
                continue
            pointer, observed = completed
            if observed is _ANY_ENTRY:
                held = await self._move_pointer(pointer)
                (moved if held == pointer else superseded).append(scope.scope)
            elif await self._move_exactly(pointer, observed):
                moved.append(scope.scope)
            else:
                # what it carried, or filled empty, was judged against an entry that has moved since
                skipped.append(scope.scope)
        if superseded:
            log.info(
                "scoped snapshot %s: %d staged scopes were already at a later epoch; nothing moved for them",
                self._name,
                len(superseded),
                extra={"extra_data": {"scopes": superseded}},
            )
        if moved:
            await self._update_index(add=moved, whole=whole and not skipped)
        for scope in staged:
            if scope.scope in moved:
                await self._retire_older(scope.scope, scope.epoch)
        if skipped:
            log.warning(
                "scoped snapshot %s: %d staged scopes left for the catch-up from L3",
                self._name,
                len(skipped),
                extra={"extra_data": {"scopes": skipped}},
            )
        self._changed.set()
        return moved, skipped

    async def _completed(
        self, staged: StagedScope, carry_at: int, *, whole: bool
    ) -> tuple[_Pointer, int | None | object] | None:
        """the pointer a staged scope moves to, every table named, and the pointer entry it was judged
        against; ``None`` when it cannot be completed.

        A stage that named every table moves by the usual compare-and-set (``_ANY_ENTRY``). One that
        carried a chunk, or filled a table empty, is right only against the entry it read, so it moves
        only from that entry's revision (``None``: only if there is still no pointer).

        :return: the pointer and the revision it must move from
        :rtype: tuple[_Pointer, int | None | object] | None
        """
        objects = dict(staged.objects)
        counts = dict(staged.rows)
        missing = [t for t in self._tables if t.name not in objects]
        pointer = _Pointer(scope=staged.scope, epoch=staged.epoch, objects=objects, rows=counts, schema=self._schema)
        result: tuple[_Pointer, int | None | object] | None = (pointer, _ANY_ENTRY)
        if missing:
            entry = await self._pointers_bucket.get_entry(key=self._pointer_key(staged.scope))
            current = None if entry is None else _Pointer.decode(entry[0])
            result = (pointer, None if entry is None else entry[1])
            index_whole = self._index is not None and not self._index_partial
            if (
                current is not None
                and current.epoch == carry_at
                and current.schema == self._schema
                and all(t.name in current.objects for t in missing)
            ):
                for table in missing:
                    objects[table.name] = current.objects[table.name]
                    counts[table.name] = current.rows[table.name]
                # the carried chunks must still be there: a sweep judged by the same rule spares them
                if not await self._exist(current.objects[t.name] for t in missing):
                    result = None
            elif (
                current is None
                and carry_at == 0
                and (whole or (index_whole and staged.scope not in (self._index or ())))
            ):
                for table in missing:
                    arrow = await self._export(table, staged.scope, [])
                    objects[table.name] = await self._put_chunk(staged.scope, staged.epoch, table, arrow)
                    counts[table.name] = 0
            else:
                result = None
        return result

    async def _exist(self, names: Iterable[str]) -> bool:
        """whether every named chunk is in the store.

        :return: True when each is
        :rtype: bool
        """
        wanted = set(names)
        prefixes = {name.rsplit("/", 2)[0] + "/" for name in wanted}
        found: set[str] = set()
        for prefix in prefixes:
            found |= {info.name for info in await self._store.list_objects(prefix=prefix)}
        return wanted <= found

    async def _move_exactly(self, pointer: _Pointer, revision: int | None | object) -> bool:
        """move a scope's pointer to ``pointer`` only from the entry at ``revision`` (``None``: from none).

        :return: whether it moved
        :rtype: bool
        """
        key = self._pointer_key(pointer.scope)
        if revision is None:
            moved = await self._pointers_bucket.create(key=key, value=pointer.encode()) is not None
        else:
            assert isinstance(revision, int)  # noqa: S101 -- _completed answers an int, None or _ANY_ENTRY
            moved = await self._pointers_bucket.update(key=key, value=pointer.encode(), revision=revision) is not None
        if moved:
            self._note_seen(pointer)
        return moved

    async def _move_pointer(self, pointer: _Pointer) -> _Pointer:
        """move a scope's pointer to ``pointer`` by compare-and-set, never to a lower epoch.

        :param pointer: the new pointer
        :ptype pointer: _Pointer
        :return: the pointer the scope holds after the move: ``pointer``, or a later one a racing writer set
        :rtype: _Pointer
        :raises RuntimeError: when racing writers kept moving it through every attempt
        """
        key = self._pointer_key(pointer.scope)
        held: _Pointer | None = None
        attempts = 0
        while held is None and attempts < _CAS_ATTEMPTS:
            attempts += 1
            entry = await self._pointers_bucket.get_entry(key=key)
            current = None if entry is None else _Pointer.decode(entry[0])
            if current is not None and (not pointer.supersedes(current) or self._yields_to(current, pointer)):
                held = current
            elif entry is None:
                held = pointer if await self._pointers_bucket.create(key=key, value=pointer.encode()) else None
            else:
                moved = await self._pointers_bucket.update(key=key, value=pointer.encode(), revision=entry[1])
                held = pointer if moved is not None else None
        if held is None:
            raise RuntimeError(f"the pointer of scope {pointer.scope!r} kept moving under {_CAS_ATTEMPTS} attempts")
        self._note_seen(held)
        return held

    def _yields_to(self, current: _Pointer, pointer: _Pointer) -> bool:
        """whether a move to ``pointer`` leaves ``current`` in place: the mixed-version rule.

        While two code versions run together (a rolling deploy of a column change), a rebuild must
        not repoint a scope another version published at the same epoch, or each version would
        repoint the other's scopes on every rebuild. So a same-epoch pointer of other columns stays
        until it is older than the stray age (by then the other version is gone); this replica
        serves its own L1, rebuilt from L3, meanwhile. A later epoch (a write) always moves the
        pointer, to the writer's columns.

        :return: True when ``current`` stays
        :rtype: bool
        """
        fresh = (
            current.published_at is not None and time.time() - current.published_at < self._stray_age.total_seconds()
        )
        return current.epoch == pointer.epoch and dict(current.schema) != dict(pointer.schema) and fresh

    def _note_seen(self, pointer: _Pointer) -> None:
        """record a pointer this replica wrote or read, ahead of the watch's delivery.

        :param pointer: the pointer
        :ptype pointer: _Pointer
        :return: nothing
        :rtype: None
        """
        if pointer.supersedes(self._seen.get(pointer.scope)):
            self._seen[pointer.scope] = pointer

    async def _update_index(self, *, add: Iterable[str] = (), remove: Iterable[str] = (), whole: bool = False) -> None:
        """add and remove scopes in the index by compare-and-set, merging what racing writers added.

        :param add: scopes now published
        :ptype add: Iterable[str]
        :param remove: scopes no longer held
        :ptype remove: Iterable[str]
        :param whole: whether ``add`` is every scope L3 holds (a full rebuild); only that makes an index
            a publish created, naming just the scopes published since, a whole one
        :ptype whole: bool
        :return: nothing
        :rtype: None
        :raises RuntimeError: when racing writers kept moving it through every attempt
        """
        adding, removing = frozenset(add), frozenset(remove)
        settled: frozenset[str] | None = None
        partial = False
        attempts = 0
        while settled is None and attempts < _CAS_ATTEMPTS:
            attempts += 1
            entry = await self._pointers_bucket.get_entry(key=self._index_key)
            body = {} if entry is None else json.loads(entry[0])
            held: frozenset[str] = frozenset(body.get("scopes", ()))
            held_partial = entry is None or bool(body.get("partial", False))
            partial = held_partial and not whole
            wanted = (held | adding) - removing
            value = json.dumps({"scopes": sorted(wanted)} | ({"partial": True} if partial else {})).encode("utf-8")
            if entry is not None and wanted == held and partial == held_partial:
                settled = held
            elif entry is None:
                settled = wanted if await self._pointers_bucket.create(key=self._index_key, value=value) else None
            else:
                moved = await self._pointers_bucket.update(key=self._index_key, value=value, revision=entry[1])
                settled = wanted if moved is not None else None
        if settled is None:
            raise RuntimeError(f"the index of snapshot {self._name!r} kept moving under {_CAS_ATTEMPTS} attempts")
        self._index = settled
        self._index_partial = partial
        self._ever_indexed = True

    async def _retire_names(self, names: list[str]) -> None:
        """ask the declarer to delete objects, in batches it accepts; never raises.

        :param names: the objects
        :ptype names: list[str]
        :return: nothing
        :rtype: None
        """
        if self._retire is None or not names:
            return
        for start in range(0, len(names), self._retire_batch):
            batch = names[start : start + self._retire_batch]
            try:
                await self._retire(batch)
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- retirement is housekeeping; a failure leaves old chunks for the next retire and must not fail a publish that already succeeded
                log.warning(
                    "scoped snapshot %s: retiring %d chunks failed; they stay until the next retire: %s",
                    self._name,
                    len(batch),
                    exc,
                    extra={"extra_data": {"objects": batch}},
                )

    def _deletable(self, info: ObjectInfo, epoch: int | None, reference: _Pointer | None, now: datetime) -> bool:
        """whether one chunk may be deleted: the one rule every retirement goes by.

        Four things decide it, and only the pointer read from KV knows them all -- never this
        replica's view of the pointers, which lags another replica's publish:

        1. a chunk its scope's pointer names is being served: kept;
        2. a chunk at an epoch ABOVE its scope's pointer is a writer's stage whose pointer has not
           moved yet: kept (if that write never commits, the scope's next publish, at a later epoch
           still, finds it below and retires it);
        3. a chunk at an epoch BELOW the pointer, and not named by it, is superseded: deleted;
        4. a chunk at the pointer's own epoch it does not name (a rebuild of that epoch under other
           columns), or of a scope with no pointer at all (a stage for a new scope), is deleted only
           once it is older than the stray age. (A removal empties its scope itself, in
           :meth:`_retire_older`: what the deleted pointer served is no keep-reference.)

        :param info: the chunk, as the store lists it
        :ptype info: ObjectInfo
        :param epoch: its epoch, from its name; ``None`` when the name is not a chunk's
        :ptype epoch: int | None
        :param reference: its scope's pointer, read from KV
        :ptype reference: _Pointer | None
        :param now: the time the ages are judged at
        :ptype now: datetime
        :return: whether it may be deleted
        :rtype: bool
        """
        written = info.mtime if info.mtime is None or info.mtime.tzinfo else info.mtime.replace(tzinfo=UTC)
        aged = written is not None and now - written > self._stray_age
        deletable: bool
        if epoch is None:
            deletable = aged
        elif reference is None:
            deletable = aged
        elif info.name in reference.objects.values() or epoch > reference.epoch:
            deletable = False
        elif epoch < reference.epoch:
            deletable = True
        else:
            deletable = aged
        return deletable

    async def _pointer_now(self, key: str) -> _Pointer | None:
        """a scope's pointer as KV holds it now, by its key.

        :return: the pointer, or ``None`` when there is none
        :rtype: _Pointer | None
        """
        raw = await self._pointers_bucket.get(key=key)
        return None if raw is None else _Pointer.decode(raw)

    async def _retire_older(self, scope: str, epoch: int, *, removed: _Pointer | None = None) -> None:
        """retire the scope's chunks of epochs before ``epoch`` that may go; never raises.

        Judged by :meth:`_deletable` against the scope's pointer in KV. After a removal
        (``removed``, the pointer it deleted) with no pointer standing again, the scope serves
        nothing: every chunk at or below the deleted pointer's epoch goes, the ones it named
        included, since a writer recreating the scope writes at a later epoch. If a recreating
        writer's pointer already stands, it is the judge, as for any scope.

        :param scope: the scope
        :ptype scope: str
        :param epoch: retire only chunks below this epoch
        :ptype epoch: int
        :param removed: the pointer a removal of the scope just deleted
        :ptype removed: _Pointer | None
        :return: nothing
        :rtype: None
        """
        if self._retire is None:
            return
        try:
            infos = await self._store.list_objects(prefix=self._object_prefix(scope))
            reference = await self._pointer_now(self._pointer_key(scope))
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- housekeeping, as in _retire_names
            log.warning("scoped snapshot %s: listing scope %r's chunks failed: %s", self._name, scope, exc)
            return
        # a removed scope with no pointer standing again serves nothing up to the removed epoch
        emptied = removed.epoch if removed is not None and reference is None else None
        now = datetime.now(UTC)
        stale = []
        for info in infos:
            chunk_epoch = self._epoch_of(scope, info.name)
            if chunk_epoch is None or chunk_epoch >= epoch:
                continue
            if (emptied is not None and chunk_epoch <= emptied) or self._deletable(info, chunk_epoch, reference, now):
                stale.append(info.name)
        await self._retire_names(stale)

    async def _retire_unreferenced(self) -> None:
        """retire every chunk :meth:`_deletable` allows, each judged by its scope's pointer in KV; never raises.

        :return: nothing
        :rtype: None
        """
        if self._retire is None:
            return
        try:
            infos = await self._store.list_objects(prefix=f"{self._name}/")
            by_token: dict[str, list[ObjectInfo]] = {}
            for info in infos:
                by_token.setdefault(info.name[len(self._name) + 1 :].split("/", 1)[0], []).append(info)
            pointers = {token: await self._pointer_now(f"{self._name}.s.{token}") for token in by_token}
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- housekeeping, as in _retire_names
            log.warning("scoped snapshot %s: listing chunks failed: %s", self._name, exc)
            return
        now = datetime.now(UTC)
        stale = []
        for token, scope_infos in by_token.items():
            for info in scope_infos:
                head = info.name[len(self._name) + 1 + len(token) + 1 :].split("/", 1)[0]
                epoch = int(head) if head.isdigit() else None
                if self._deletable(info, epoch, pointers[token], now):
                    stale.append(info.name)
        await self._retire_names(stale)

    # ------------------------------------------------------------------
    # rebuilding from L3
    # ------------------------------------------------------------------

    async def _l3_scopes(self) -> set[str]:
        """every scope any table holds in L3.

        :return: the scopes
        :rtype: set[str]
        :raises ValueError: when a table holds rows with no scope
        """
        scopes: set[str] = set()
        for table in self._tables:
            rows = await self._l3.fetch(
                f"SELECT DISTINCT {quote_identifier(table.scope_column)} AS scope FROM {quote_identifier(table.name)}"  # noqa: S608
            )
            found = {row["scope"] for row in rows}
            if None in found:
                raise ValueError(
                    f"{table.name} holds rows with no scope ({table.scope_column} is null); a scoped snapshot "
                    f"cannot hold them, and a copy without them would not be whole"
                )
            scopes.update(str(scope) for scope in found)
        return scopes

    async def _read_scope(self, scope: str) -> dict[str, list[dict[str, Any]]]:
        """every row of one scope in each table, from L3.

        :return: table name -> rows
        :rtype: dict[str, list[dict[str, Any]]]
        """
        result: dict[str, list[dict[str, Any]]] = {}
        for table in self._tables:
            columns = tuple(self._backend.column_types(table.name))
            result[table.name] = await read_l3_rows(
                self._l3, table.name, columns, table.key, where={table.scope_column: scope}
            )
        return result

    def _claim(self) -> _Claim:
        """a fresh claim on the rebuild.

        :return: the claim
        :rtype: _Claim
        """
        return _Claim(self._pointers_bucket, key=self._claim_key, owner=self._replica, ttl=self._claim_ttl)

    async def _rebuild(self, *, reason: str) -> None:
        """rebuild every scope's chunks from L3 under the claim, or wait for whoever holds it.

        :param reason: why, for the status
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        if self._ensure_buckets is not None and self._index is None and self._ever_indexed:
            await self._ensure_buckets()
        claim = self._claim()
        if await claim.take():
            try:
                caught = await self._rebuild_scopes(None, reason=reason, flow="sync")
            finally:
                await claim.release()
            if caught is not None:
                await self._retire_unreferenced()
        else:
            self._wait("sync", f"{reason}; another replica is rebuilding or publishing it")

    async def _rebuild_scopes(self, only: set[str] | None, *, reason: str, flow: str) -> list[str] | None:
        """read scopes from L3 while no write moves them, publish them, then update the index.

        :param only: the scopes to rebuild; every scope L3 holds when ``None``
        :ptype only: set[str] | None
        :param reason: why, for the status
        :ptype reason: str
        :param flow: the control flow it runs in, whose wait it records (``sync``, ``catch_up``)
        :ptype flow: str
        :return: the scopes published, or ``None`` when a write in progress made it wait
        :rtype: list[str] | None
        """
        result: list[str] | None = None
        attempts = 0
        while result is None and attempts < _REBUILD_ATTEMPTS:
            attempts += 1
            before = await self._settled() if self._settled is not None else None
            if isinstance(before, Unsettled):
                self._wait(flow, f"{reason}; waiting for a write in progress ({before.reason})")
                break
            result = await self._read_and_publish(only, reason=reason, before=before)
        if result is None and attempts >= _REBUILD_ATTEMPTS:
            self._wait(flow, f"{reason}; writes kept committing while it read; trying again")
        return result

    async def _read_and_publish(self, only: set[str] | None, *, reason: str, before: Any) -> list[str] | None:
        """one rebuild attempt: read the scopes from L3, and publish them if no write committed meanwhile.

        :param only: the scopes to rebuild; every scope L3 holds when ``None``
        :ptype only: set[str] | None
        :param reason: why, for the status
        :ptype reason: str
        :param before: the writer's stamp before the read
        :ptype before: Any
        :return: the scopes published, or ``None`` when a write committed during the read
        :rtype: list[str] | None
        :raises ValueError: when a table holds rows with no scope
        """
        started = time.perf_counter()
        l3_scopes = await self._l3_scopes()
        scopes = sorted(l3_scopes if only is None else (only & l3_scopes))
        epochs = await self._epochs()
        with self._doing(SnapshotPhase.REBUILDING_FROM_L3, f"rebuilding from L3: {reason}", total=len(scopes)):
            return await self._read_and_publish_scopes(only, scopes, l3_scopes, epochs, before=before, started=started)

    async def _read_and_publish_scopes(
        self,
        only: set[str] | None,
        scopes: list[str],
        l3_scopes: set[str],
        epochs: Mapping[str, int],
        *,
        before: Any,
        started: float,
    ) -> list[str] | None:
        """the body of :meth:`_read_and_publish`, inside its step.

        :return: the scopes published, or ``None`` when a write committed during the read
        :rtype: list[str] | None
        """
        semaphore = asyncio.Semaphore(self._l3_concurrency)
        read: dict[str, dict[str, list[dict[str, Any]]]] = {}

        async def one(scope: str) -> None:
            async with semaphore:
                read[scope] = await self._read_scope(scope)
            self._progress.scopes_done += 1

        # a TaskGroup: one failed read cancels the rest rather than leave them reading for nothing
        async with asyncio.TaskGroup() as group:
            for scope in scopes:
                group.create_task(one(scope))
        read_at = time.perf_counter()
        after = await self._settled() if self._settled is not None else None
        published: list[str] | None = None
        if after != before:
            log.info("scoped snapshot %s: a write committed during the rebuild; reading again", self._name)
        else:
            for scope in scopes:
                async with self._local:
                    await self._publish_locked(scope, int(epochs.get(scope, 0)), read[scope])
            await self._update_index(add=scopes, whole=only is None)
            gone = set() if only is not None else set(self._index or ()) - l3_scopes
            if gone:
                await self._remove_scopes(gone)
            await self._apply_published()
            done = time.perf_counter()
            self._progress.timings.update(
                {
                    "read_l3": round(read_at - started, 3),
                    "publish": round(done - read_at, 3),
                    "total": round(done - started, 3),
                }
            )
            # only a rebuild of every scope makes a copy ready; a rebuild of some scopes repairs them
            if only is None:
                self._become_ready(SnapshotSource.L3)
            published = list(scopes)
        return published

    async def catch_up_from_l3(self) -> list[str]:
        """bring the snapshot to L3: republish scopes whose epoch in L3 is ahead of their pointer, drop scopes L3 left.

        For a writer that committed and died before it published, for scopes L3 holds that the
        snapshot does not, and for scopes L3 no longer holds. Runs under the rebuild claim. A
        catch-up that has to wait on a write in progress leaves the status
        WAITING until a later call does not; nothing here calls it again, so its caller schedules
        the next (the ENR pod's refresh calls it after every publish).

        :return: the scopes republished
        :rtype: list[str]
        :raises ValueError: when a table holds rows with no scope
        """
        self._begin_run("catch_up")
        try:
            caught = await self._catch_up()
        finally:
            self._end_run("catch_up")
        return caught

    async def _catch_up(self) -> list[str]:
        """the body of :meth:`catch_up_from_l3`.

        :return: the scopes republished
        :rtype: list[str]
        """
        epochs = await self._epochs()
        l3_scopes = await self._l3_scopes()
        behind = {
            scope
            for scope in l3_scopes
            if scope not in self._seen or int(epochs.get(scope, 0)) > self._seen[scope].epoch
        }
        gone = set(self._index or ()) - l3_scopes
        caught: list[str] = []
        claim = self._claim()
        if (behind or gone) and not await claim.take():
            log.info("scoped snapshot %s: another replica holds the claim; it catches up", self._name)
        elif behind or gone:
            try:
                if behind:
                    caught = (
                        await self._rebuild_scopes(behind, reason=f"{len(behind)} scopes behind L3", flow="catch_up")
                        or []
                    )
                if gone:
                    await self._remove_scopes(gone)
            finally:
                await claim.release()
            for scope in caught:
                # the watch may have delivered a removal meanwhile, or a stale publish noted nothing
                seen = self._seen.get(scope)
                if seen is not None:
                    await self._retire_older(scope, seen.epoch)
        return caught

    async def _apply_published(self) -> None:
        """apply every scope the index names whose pointer is ahead of this L1, before a rebuild declares ready.

        A rebuild refuses to publish a scope a writer already moved past the epoch it read, and a
        writer may publish a scope the rebuild never read; either way the L1 lacks it, and a copy
        declared ready must hold every scope.

        :return: nothing
        :rtype: None
        :raises _Lost: when a scope's chunks cannot be read; the copy is not declared ready
        """
        pointers: dict[str, _Pointer] = {}
        for scope in sorted(self._index or ()):
            pointer = self._seen.get(scope)
            if pointer is None:
                raw = await self._pointers_bucket.get(key=self._pointer_key(scope))
                pointer = None if raw is None else _Pointer.decode(raw)
            if pointer is not None and self._ahead(pointer, self._applied.get(scope)):
                pointers[scope] = pointer
            elif pointer is not None and self._foreign_ahead(pointer, self._applied.get(scope)):
                self._mark_foreign(scope)
        failed = await self._apply_pointers(pointers)
        if failed:
            raise _Lost(f"published scopes {sorted(failed)} could not be read; not ready without them")

    async def _remove_scopes(self, candidates: set[str]) -> None:
        """remove the scopes L3 holds no more, judged now: their pointers, then the index, this L1, their chunks.

        ``candidates`` were judged gone from a read of L3 that a writer may have outrun since. Each
        scope's pointer is read first and L3 after, so a scope L3 still holds is kept, and a writer
        that commits after that read moves the pointer, which the compare-and-set delete then refuses.
        A writer that recreates a scope after its pointer was deleted finds it still in the index, so
        the scope goes back in when its pointer is found standing again.

        :param candidates: the scopes judged gone
        :ptype candidates: set[str]
        :return: nothing
        :rtype: None
        :raises ValueError: when a table holds rows with no scope
        """
        held = {scope: await self._pointers_bucket.get_entry(key=self._pointer_key(scope)) for scope in candidates}
        l3_now = await self._l3_scopes()
        removed: dict[str, _Pointer | None] = {}
        for scope in sorted(candidates - l3_now):
            entry = held[scope]
            if entry is None:
                removed[scope] = None
            elif await self._pointers_bucket.delete(key=self._pointer_key(scope), revision=entry[1]):
                removed[scope] = _Pointer.decode(entry[0])
            else:
                log.info("scoped snapshot %s: scope %r moved while it was removed; kept", self._name, scope)
        if not removed:
            return
        await self._update_index(remove=removed)
        standing = {
            scope for scope in removed if await self._pointers_bucket.get(key=self._pointer_key(scope)) is not None
        }
        if standing:
            await self._update_index(add=standing)
        gone = {scope: pointer for scope, pointer in removed.items() if scope not in standing}
        for scope in gone:
            self._seen.pop(scope, None)
        await self._drop_locally(sorted(gone))
        for scope, pointer in gone.items():
            # every chunk up to the deleted pointer's epoch, the ones it served included: epochs only
            # move forward, so chunks a writer recreating the scope wrote meanwhile are at a later
            # epoch and stay. a scope that had no pointer judged nothing; its strays age out of the
            # next rebuild's unreferenced sweep
            if pointer is not None:
                await self._retire_older(scope, pointer.epoch + 1, removed=pointer)


async def open_tool_pod_snapshot(
    nats_client: NatsClient,
    *,
    identity_token: Callable[[], str],
    name: str,
    tables: Sequence[SnapshotTable],
    backend: DuckDBBackend,
    l3: Any,
    epochs: Callable[[], Awaitable[Mapping[str, int]]],
    settled: Callable[[], Awaitable[Any]] | None = None,
    **options: Any,
) -> ScopedSnapshot:
    """a tool pod's scoped snapshot over its OWN Object Store and pointer bucket, started.

    The hub declares both buckets when the pod asks (its registry row must opt in, ``aibots tool-pod
    set-object-store POD --on``), declares them again when NATS lost them (``ensure_buckets``), and
    deletes the chunks the snapshot retires: a pod holds no management verb on either
    (:mod:`threetears.nats.object_store_requests`).

    :param nats_client: the pod's connected NATS client
    :ptype nats_client: NatsClient
    :param identity_token: the pod's CURRENT hub identity token, read at each ask of the hub
    :ptype identity_token: Callable[[], str]
    :param name: the snapshot's name
    :ptype name: str
    :param tables: its tables
    :ptype tables: Sequence[SnapshotTable]
    :param backend: the DuckDB L1 holding every table, initialized
    :ptype backend: DuckDBBackend
    :param l3: the L3 backend the tables live in
    :ptype l3: Any
    :param epochs: each scope's epoch as L3 records it
    :ptype epochs: Callable[[], Awaitable[Mapping[str, int]]]
    :param settled: the writer's seqlock
    :ptype settled: Callable[[], Awaitable[Any]] | None
    :param options: any other :class:`ScopedSnapshot` option; not ``store``, ``pointers``,
        ``ensure_buckets`` or ``retire``, which this wires
    :ptype options: Any
    :return: the snapshot, watching its pointers and loading
    :rtype: ScopedSnapshot
    :raises ObjectStoreRequestError: when the hub refuses or does not answer the declare
    :raises ValueError: when ``options`` names what this wires; nothing is asked of the hub
    """
    from threetears.nats.object_store_requests import bind_pod_object_store  # noqa: PLC0415

    wired = sorted(set(options) & _POD_WIRED_OPTIONS)
    if wired:
        raise ValueError(
            f"open_tool_pod_snapshot wires {wired} to the pod's own buckets and the hub itself; "
            "build a ScopedSnapshot directly to give them"
        )
    buckets = await bind_pod_object_store(nats_client, identity_token=identity_token)

    async def retire(names: list[str]) -> None:
        await buckets.retire(names)

    snapshot = ScopedSnapshot(
        name=name,
        tables=tables,
        backend=backend,
        store=buckets.store,
        pointers=buckets.pointers,
        l3=l3,
        epochs=epochs,
        settled=settled,
        ensure_buckets=buckets.declare,
        retire=retire,
        **options,
    )
    await snapshot.start()
    return snapshot
