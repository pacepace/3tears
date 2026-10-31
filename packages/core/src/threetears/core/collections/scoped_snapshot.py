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

**Loading.** A starting replica watches the pointers (``watch_prefix``), fetches the current chunks,
several scopes at once, and loads each scope into its DuckDB L1 through
:class:`~threetears.core.cache.duckdb.DuckDBBackend` as soon as it and the scopes before it are in, in
scope order: seconds, with no L3 read, holding a few scopes' chunks at a time, never every scope's.
Each chunk's digest (the Object Store's) and row count (the pointer's) are checked; the index names
every scope, so a missing one is seen, not skipped.

**Changes.** A writer that commits a scope's change calls :meth:`ScopedSnapshot.publish` with the
scope's complete new rows: they replace the scope in its own L1, the chunks are written under the
new epoch, and then the pointer moves -- by compare-and-set, and never to a lower epoch, so a stale
writer moves nothing. Every other replica's watch sees the pointer, fetches only that scope's chunks
and replaces the scope in ONE DuckDB transaction, so a reader -- holding :meth:`ScopedSnapshot.read`
for a request -- sees the scope as it was or as it is, never half of it. No replica's L1 ever goes
back to a lower epoch of a scope. Nothing is rebuilt whole and nothing polls.

**Tables that only grow.** A table declared append-only (:attr:`SnapshotTable.append_only`) may be
published by the rows each write ADDED (``added=``, with the epoch they follow): a scope's pointer
names, for it, a base chunk and the pieces after it, and a replica fetches and appends only the
pieces after what it holds, in the same one swap. Past a threshold the pieces fold into a new base. A pointer with pieces is marked so a replica
that predates them takes it for another code version's columns and serves the scope from L3; one
with none is written as before (``docs/design-scoped-snapshot.md``, "Append-only tables").

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
-- renewed while it works, let go once it makes no progress for ``rebuild_stall``, released only if
still its own -- so one replica does the work, and
rebuilds every scope from L3, the slow and correct path, while its own L1 keeps answering from what
it last held; its status says so. A rebuild reads each scope a page at a time, types each page as
the L1 keeps it, and publishes the scope before it reads past a few more: it holds a few scopes' rows,
never every scope's. :meth:`ScopedSnapshot.catch_up_from_l3` republishes any scope
whose L3 epoch is ahead of its pointer (a writer that died between its commit and its publish) and
drops scopes L3 no longer holds. A rebuild reads L3 only while no write is in progress and keeps what
it read only when no write committed meanwhile (``settled``, the writer's seqlock).

**Retirement.** Chunks are deleted by the bucket's declarer, since a delete is a purge no pod holds
(``retire``), in batches the declarer accepts, and only as one rule allows (``_Sweeper.deletable``),
judged by state as KV holds it when the sweep runs, never by age: a chunk its scope's pointer names
is served, and one at an epoch a live write claims is being written or waits for its pointer
(``{name}.w.{epoch}.{writer}``, one per write, held by the writer from before its first chunk until
a pointer names them, and renewed while it lives: a writer that dies stops renewing, and its claim
lapses);
every other chunk serves nothing and goes. A scope's older chunks go as soon as its pointer moves,
whichever path moved it, so a run that fails part way leaves nothing superseded behind for the
scopes it did move. A write that will not commit gives its stages back
(:meth:`ScopedSnapshot.discard_staged`); a writer taking over a dead one may name its epoch
(:meth:`ScopedSnapshot.discard_epoch`). A store found full is swept by the same rule and the write
tried once more: it recovers when what fills it is chunks no pointer serves and no live write
claims; pieces of a torn write wait for the declarer's orphan purge. No pointer is ever moved onto a
chunk that is not there: a staged publish and a publish that found its chunk already written check
it first, and leave the scope to the catch-up rather than move. A reader that lost the race to a retired
chunk reads the scope's pointer again and fetches its current chunks; a scope whose chunks still
cannot be read is shown behind, retried at the recheck, and rebuilt from L3 after three failures.

**Transparent.** :meth:`ScopedSnapshot.status` says what it is doing (loading from L2, rebuilding from
L3, waiting, ready, failed), how many scopes are done, each table's rows, how long each step took,
and the last change applied.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from collections.abc import Set as AbstractSet
from contextlib import (
    AbstractContextManager,
    ExitStack,
    aclosing,
    asynccontextmanager,
    contextmanager,
    nullcontext,
    suppress,
)
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import islice
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from threetears.nats import ObjectExistsError, ObjectNotFoundError, ObjectStoreError, ObjectStoreFullError
from threetears.nats.object_store_requests import MAX_RETIRED_OBJECTS, WRITE_CLAIM_SEGMENT, bind_pod_object_store
from threetears.observe import contain, get_logger

from threetears.core.sql_fragments import quote_identifier
from threetears.core.backends.arrow_reply import ReplyNotArrowError
from threetears.core.backends.protocol import ArrowL3Reader
from threetears.core.cache.duckdb import DuckDBBackend, PartitionReplacement
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE, Unsettled, l3_row_pages
from threetears.core.keyset import keyset_arrow_pages
from threetears.core.coordination.lease import HeldLease, KVLease, LeaseUnavailable

if TYPE_CHECKING:
    from threetears.core.backends.protocol import L3Reader
    from threetears.nats.client import NatsClient
    from threetears.nats.kv import NatsKvBucket
    from threetears.nats.object_store import NatsObjectStore

__all__ = [
    "AppendedScope",
    "CatchUpOutcome",
    "CatchUpResult",
    "ScopeLag",
    "ScopeChange",
    "ScopedSnapshot",
    "SnapshotPhase",
    "SnapshotSource",
    "SnapshotStatus",
    "SnapshotTable",
    "StagedPartitions",
    "StagedScope",
    "VersionedRead",
    "decode_chunk",
    "encode_chunk",
    "open_tool_pod_snapshot",
    "scope_token",
]

log = get_logger(__name__)

try:
    import duckdb
    import pyarrow as pa
    import pyarrow.ipc as ipc

    _SNAPSHOT_EXTRA: ImportError | None = None
except ImportError as exc:  # NOSILENT: the snapshot extra is optional; using a snapshot says so
    _SNAPSHOT_EXTRA = exc


def _require_snapshot_extra() -> None:
    """raise, naming the extra, when duckdb or pyarrow could not be imported.

    :return: nothing
    :rtype: None
    :raises ImportError: when the snapshot extra is not installed
    """
    if _SNAPSHOT_EXTRA is not None:
        raise ImportError(f"a scoped snapshot needs duckdb and pyarrow: install 3tears[snapshot] ({_SNAPSHOT_EXTRA})")


#: the snapshot name's grammar: one KV subject token and one object-name segment
_NAME_GRAMMAR: Final = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")

#: how long a rebuild's claim lasts unless its holder renews it; a holder that dies frees it after this long
_DEFAULT_CLAIM_TTL: Final = timedelta(minutes=2)

#: how often a replica waiting on another's rebuild, or on a write in progress, looks again
_DEFAULT_RECHECK: Final = timedelta(seconds=2)

#: how long a holder of the rebuild claim may go without progress -- a page read from L3, a chunk read or
#: written, scopes committed here, a pointer or the index moved -- before it lets the claim go and its run (a
#: rebuild, a catch-up) ends with a TimeoutError; a writer's hold across its commit and publish
#: (:meth:`ScopedSnapshot.holding_rebuilds`) is renewed no longer than this. The claim's renewal runs on its own
#: task, so a holder stuck on one await kept renewing it for as long as its process lived: on cobalt prod
#: (2026-10-31) a catch-up's first read beside a write never answered, and its claim kept every replica's
#: catch-up out all night while L3 ran 30 scopes ahead of the copy. Each step is seconds to a minute or two on the
#: largest scope; ten minutes of none is a run that is stuck
_DEFAULT_REBUILD_STALL: Final = timedelta(minutes=10)

#: the longest a stuck run's watchdog looks away between checks
_STALL_LOOK_SECONDS: Final = 5.0

#: how many scopes a rebuild holds at once: read from L3 beside each other and the one being published.
#: What a rebuild holds is these scopes' rows (typed, as the L1 keeps them) and a page of L3's each,
#: never every scope's. Measured on 3 million rows in 50 scopes (Linux): 4 rebuilt in 24 to 31 s at a
#: 1.6 GB peak, 1 in 28 to 37 s at 1.2 GB; reading every scope before publishing any took 22 s at 4.2 GB
_DEFAULT_L3_CONCURRENCY: Final = 4

#: how many scopes a load holds at once: fetched from L2 beside each other and the one being committed.
#: What a load holds is these scopes' decoded chunks, never every scope's
_DEFAULT_L2_CONCURRENCY: Final = 16

#: the chunks one snapshot reads from the object store at once, whatever the scopes it fetches ahead: each read
#: is the server pushing the chunk's messages to this connection, and a load fetching 16 scopes ahead had more
#: pending than the server holds for a client (MaxPending, 64 MiB), so the server dropped the pod as a slow consumer
#: (cobalt dev, 2026-10-29)
_DEFAULT_CHUNK_READS: Final = 4

#: the moved scopes a publish compacts at once
_SWEEPS_AT_ONCE: Final = 8

#: how many scopes' fetched chunks a replica holds at once to commit them as one move; a burst of more is
#: committed in batches of this many
_DEFAULT_APPLY_BATCH: Final = 64

#: how many times the scopes that moved while a batch fetched join it before it is committed: enough to fold a
#: burst's stragglers in, few enough that a scope a writer keeps moving never holds the batch back
_APPLY_JOINS: Final = 2


def _raise_first(results: Sequence[object], *, log: Any, what: str) -> None:
    """raise the first failure of a gather that ran every task to its end, recording every later one.

    Every failure either reaches the caller (the first) or is recorded where an operator sees it (the rest).

    :param results: what ``asyncio.gather(..., return_exceptions=True)`` returned
    :ptype results: Sequence[object]
    :param log: where the later failures are recorded
    :ptype log: Any
    :param what: what failed, for the record
    :ptype what: str
    :return: nothing
    :rtype: None
    :raises BaseException: the first failure, when there is one
    """
    failures = [result for result in results if isinstance(result, BaseException)]
    for later in failures[1:]:
        if isinstance(later, Exception):
            contain(later, log=log, what=f"{what} (beside an earlier failure, which is raised): {later}")
    if failures:
        raise failures[0]


#: how many times a rebuild reads again after a write committed while it read
_REBUILD_ATTEMPTS: Final = 3

#: how often a rebuild reading a scope from L3 asks whether a write has begun since it started, by default: a write
#: that began (a reset's clear, which empties the tables) ends the read within this, rather than after the scope's
#: last page -- minutes for a large scope, while the write waits on the rows the read holds open
_WRITE_CHECK: Final = timedelta(seconds=5)

#: how many times a compare-and-set on a pointer or the index is tried against a racing writer
_CAS_ATTEMPTS: Final = 16

#: how many phases the status keeps in its history
_HISTORY: Final = 20

#: how many passes in a row a scope's chunks may fail to apply before it is rebuilt from L3
_BEHIND_REBUILD_AFTER: Final = 3

#: how many pieces an append-only table's scope gathers before a writer folds them into its base: a
#: cold replica fetches the base and every piece, so this bounds its reads per scope
_DEFAULT_COMPACT_PIECES: Final = 32

#: how many bytes of pieces an append-only table's scope gathers before a writer folds them
_DEFAULT_COMPACT_BYTES: Final = 4 * 1024 * 1024


#: what a pointer's schema entry carries after a table's column digest when the table has pieces. A
#: replica that predates pieces compares the entry with its own digest before loading anything, so it
#: takes the scope for another code version's and serves it from L3 rather than misread the pointer
_PIECES_MARK: Final = "+pieces"

#: what begins any mark on a schema entry; a column digest is hex, so it never holds one
_MARK_SEPARATOR: Final = "+"

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


class CatchUpResult(StrEnum):
    """how one catch-up from L3 (:meth:`ScopedSnapshot.catch_up`) ended.

    :cvar CAUGHT_UP: it ran to its end without waiting: every scope behind L3 when it looked was republished
        (none, when none was behind), and every scope L3 left removed
    :cvar CLAIM_HELD: another replica (or another run here) holds the rebuild claim; nothing was done, and the
        holder catches up
    :cvar WAITED: a write in progress kept it from some scope; what it could read beside the write is republished,
        and a later catch-up does the rest
    :cvar STALLED: it made no progress for ``rebuild_stall``, let the claim go and stopped
    :cvar FAILED: it raised; the error says what
    """

    CAUGHT_UP = "caught_up"
    CLAIM_HELD = "claim_held"
    WAITED = "waited"
    STALLED = "stalled"
    FAILED = "failed"


@dataclass(frozen=True)
class ScopeLag:
    """one scope whose epoch in L3 is ahead of the pointer this copy has seen: the catch-up's own test.

    Not :attr:`SnapshotStatus.behind`, which names scopes whose published chunks could not be applied here.

    :ivar l3_epoch: the scope's epoch as L3 records it
    :ivar pointer_epoch: the epoch of the scope's pointer as this replica last saw it; ``None`` for a scope with
        no pointer yet
    """

    l3_epoch: int
    pointer_epoch: int | None


@dataclass(frozen=True)
class CatchUpOutcome:
    """what one catch-up from L3 did, and what it left behind.

    :ivar result: how it ended
    :ivar caught: the scopes it republished, in order
    :ivar behind: scope -> its lag, for every scope still behind L3 when it ended, against the pointers as this
        replica sees them then: L3's epochs read again after a catch-up that ran to its end or waited (scopes a write
        moved meanwhile included), else L3's epochs as it found them
    :ivar holder: with ``CLAIM_HELD``, the replica holding the claim (its snapshot's replica token), when the
        claim still named one as it looked
    :ivar held_since: with ``CLAIM_HELD``, when that holder took the claim
    :ivar reason: with ``WAITED``, the write it waited on; with ``STALLED`` or ``FAILED``, the error
    :ivar finished_at: when it ended
    """

    result: CatchUpResult
    caught: tuple[str, ...]
    behind: Mapping[str, ScopeLag]
    finished_at: datetime
    holder: str | None = None
    held_since: datetime | None = None
    reason: str | None = None


@dataclass(frozen=True)
class SnapshotTable:
    """one table of the snapshot.

    :ivar name: the table, as the DuckDB backend and L3 name it (a TRUSTED identifier)
    :ivar scope_column: the column whose value is a row's scope; never null
    :ivar key: the table's key, ordering a chunk's rows and paging its L3 reads
    :ivar append_only: rows are only ever added: a row once held never changes or goes away. A write
        may then give :meth:`ScopedSnapshot.stage` and :meth:`ScopedSnapshot.publish` only the rows it
        ADDED to the scope (their ``added=``), kept as a piece after the scope's base, and a replica
        appends only the pieces after what it holds (``docs/design-scoped-snapshot.md``)
    """

    name: str
    scope_column: str
    key: tuple[str, ...]
    append_only: bool = False


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
class AppendedScope:
    """what one scope of an append-only table holds in this replica: its base and the pieces after it.

    :ivar pieces: the pieces after its base
    :ivar compacted_at: when its base was last formed whole (a fold of its pieces, a rebuild, its
        first rows); ``None`` for a base written before that was recorded
    """

    pieces: int
    compacted_at: datetime | None


@dataclass(frozen=True)
class StagedScope:
    """a scope's chunks written under a new epoch whose pointer has not moved yet (:meth:`ScopedSnapshot.stage`).

    :ivar scope: the scope
    :ivar epoch: the epoch the chunks are written under
    :ivar objects: table -> the chunk written for it; a table left out keeps its rows. For an
        append-only table, the piece holding the rows the write added
    :ivar rows: table -> the rows its chunk holds
    :ivar sizes: table -> its chunk's bytes
    :ivar pieces: the tables whose chunk is a piece of the rows the write added (:meth:`ScopedSnapshot.stage`'s
        ``added``), appended to the chain rather than written whole
    :ivar claim: the write claim keeping the chunks from a sweep, held from before the first of them
        was written until :meth:`ScopedSnapshot.publish_staged` or :meth:`ScopedSnapshot.discard_staged`
        lets it go; ``None`` for a stage that holds none
    """

    scope: str
    epoch: int
    objects: Mapping[str, str]
    rows: Mapping[str, int]
    claim: HeldLease | None = field(default=None, repr=False, compare=False)
    sizes: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    pieces: frozenset[str] = frozenset()


@dataclass(frozen=True)
class VersionedRead:
    """a read of the copy, and the epoch of every scope it reads (:meth:`ScopedSnapshot.read_versioned`).

    :ivar cursor: the read's cursor: one state of every table, held still for the block
    :ivar epochs: scope -> the epoch whose rows the cursor reads, exactly
    :ivar behind: scope -> why it is behind, as :meth:`ScopedSnapshot.read_with_behind` takes it:
        every scope the cursor's data is behind on, and perhaps one brought current as it opened
    """

    cursor: Any
    epochs: Mapping[str, int]
    behind: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


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
    :ivar appended: append-only table -> scope -> its pieces and its last compaction, as this replica
        holds it
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
    appended: Mapping[str, Mapping[str, AppendedScope]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Piece:
    """one piece of an append-only table's scope: the rows one write added, in its own chunk.

    :ivar name: the chunk's object name
    :ivar rows: the rows it holds
    :ivar epoch: the epoch it was written at
    :ivar size: its bytes, for the compaction threshold
    """

    name: str
    rows: int
    epoch: int
    size: int

    def encode(self) -> dict[str, Any]:
        """the piece as its pointer entry holds it.

        :return: its fields
        :rtype: dict[str, Any]
        """
        return {"object": self.name, "rows": self.rows, "epoch": self.epoch, "bytes": self.size}

    @classmethod
    def decode(cls, body: Mapping[str, Any]) -> _Piece:
        """a piece from its pointer entry.

        :param body: the entry
        :ptype body: Mapping[str, Any]
        :return: the piece
        :rtype: _Piece
        :raises KeyError: when a field is missing
        """
        return cls(name=str(body["object"]), rows=int(body["rows"]), epoch=int(body["epoch"]), size=int(body["bytes"]))


@dataclass(frozen=True)
class _Write:
    """one table's rows a write puts in a chunk: whole, or a piece after the scope's chain.

    :ivar arrow: the rows, as the L1 types them
    :ivar piece: whether they are appended to the chain as a piece rather than written whole
    """

    arrow: Any
    piece: bool = False


@dataclass(frozen=True)
class _Pointer:
    """what a scope's pointer says: its epoch, and for each table its base chunk and any pieces after it.

    A table's rows in the scope are its base chunk (``objects``, ``rows``) followed by its pieces in
    order (``pieces``, present only for a table that has some): the format ``docs/design-scoped-snapshot.md``
    records. A pointer with no pieces is written exactly as a replica that predates pieces writes and
    reads one.
    """

    scope: str
    epoch: int
    objects: Mapping[str, str]
    rows: Mapping[str, int]
    schema: Mapping[str, str]
    #: when it was written (epoch seconds); None for a pointer written before it was recorded
    published_at: float | None = field(default=None, compare=False)
    #: table -> its pieces after the base, in order; a table with none is not named
    pieces: Mapping[str, tuple[_Piece, ...]] = field(default_factory=lambda: MappingProxyType({}))
    #: table -> when an append-only table's base was last formed whole (epoch seconds), if recorded
    compacted_at: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}), compare=False)

    def chain(self, table: str) -> tuple[str | None, tuple[_Piece, ...]]:
        """a table's base chunk and its pieces after it.

        :param table: the table
        :ptype table: str
        :return: the base's name (``None`` when the pointer names none) and the pieces
        :rtype: tuple[str | None, tuple[_Piece, ...]]
        """
        return self.objects.get(table), tuple(self.pieces.get(table, ()))

    def total_rows(self, table: str) -> int:
        """a table's rows in the scope: its base and every piece.

        :param table: the table
        :ptype table: str
        :return: the rows
        :rtype: int
        """
        return self.rows.get(table, 0) + sum(piece.rows for piece in self.pieces.get(table, ()))

    def names(self) -> list[str]:
        """every chunk the pointer serves: each table's base and its pieces.

        :return: the names
        :rtype: list[str]
        """
        return [*self.objects.values(), *(piece.name for pieces in self.pieces.values() for piece in pieces)]

    def encode(self) -> bytes:
        """the pointer as the KV entry's value.

        :return: its JSON
        :rtype: bytes
        """
        tables: dict[str, dict[str, Any]] = {}
        schema: dict[str, str] = {}
        for t in self.objects:
            entry: dict[str, Any] = {"object": self.objects[t], "rows": self.rows[t]}
            pieces = self.pieces.get(t, ())
            if pieces:
                entry["pieces"] = [piece.encode() for piece in pieces]
            if t in self.compacted_at:
                entry["compacted_at"] = self.compacted_at[t]
            tables[t] = entry
        for t, digest in self.schema.items():
            schema[t] = f"{digest}{_PIECES_MARK}" if self.pieces.get(t) else digest
        return json.dumps(
            {
                "scope": self.scope,
                "epoch": self.epoch,
                "tables": tables,
                "schema": schema,
                "at": time.time() if self.published_at is None else self.published_at,
            },
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def decode(cls, raw: bytes) -> _Pointer:
        """a pointer from its KV entry's value.

        A table's schema entry ends ``+pieces`` exactly when it lists pieces; the mark is taken off,
        so the schema holds column digests alone. Any other mark is left on and what it lists is not
        read, so the pointer reads as another code version's columns and nothing of it is loaded here.

        :param raw: the value
        :ptype raw: bytes
        :return: the pointer
        :rtype: _Pointer
        :raises KeyError: when a field is missing
        :raises ValueError: when it is not JSON, or a table's pieces and its schema mark disagree
        """
        body = json.loads(raw)
        tables = body["tables"]
        schema: dict[str, str] = {}
        pieces: dict[str, tuple[_Piece, ...]] = {}
        for t, entry in body["schema"].items():
            digest = str(entry)
            listed = tables.get(t, {}).get("pieces")
            if digest.endswith(_PIECES_MARK):
                if not listed:
                    raise ValueError(f"table {t}: its schema entry {digest!r} is marked for pieces it does not list")
                pieces[t] = tuple(_Piece.decode(piece) for piece in listed)
                digest = digest.removesuffix(_PIECES_MARK)
            elif listed and _MARK_SEPARATOR not in digest:
                raise ValueError(f"table {t}: it lists pieces its schema entry {digest!r} is not marked for")
            # any other mark is a later format's: left on, so the entry is no column digest of ours
            schema[t] = digest
        return cls(
            scope=str(body["scope"]),
            epoch=int(body["epoch"]),
            objects={t: str(v["object"]) for t, v in tables.items()},
            rows={t: int(v["rows"]) for t, v in tables.items()},
            schema=schema,
            published_at=float(body["at"]) if "at" in body else None,
            pieces=pieces,
            compacted_at={t: float(v["compacted_at"]) for t, v in tables.items() if "compacted_at" in v},
        )

    def supersedes(self, other: _Pointer | None) -> bool:
        """whether moving a scope from ``other`` to this pointer moves it forward.

        :param other: the pointer held now, if any
        :ptype other: _Pointer | None
        :return: whether this one is newer, or the same epoch with other chunks (a rebuild under
            other columns, or a compaction)
        :rtype: bool
        """
        return (
            other is None
            or self.epoch > other.epoch
            or (self.epoch == other.epoch and (self.objects != other.objects or self.pieces != other.pieces))
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
    _require_snapshot_extra()
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
    _require_snapshot_extra()
    return ipc.open_stream(data).read_all()


#: the L1 type codes whose values the rail answers as text, as ``fetch`` hands them to a dict row
_RAIL_TEXT_CODES: Final = frozenset({"VARCHAR", "VARCHAR_UUID", "VARCHAR_DATETIME"})


def _rail_types(codes: Mapping[str, str]) -> dict[str, Any] | None:
    """each column's Arrow type as the L3 rail's reply carries it, from this L1's type codes; None when a column has
    none (``fetch`` answers it otherwise than its JSON: a JSON, array, vector or bytes column, or a ``date_`` column
    that is not a timestamp), and the table is read as dict rows.

    The same rule as :func:`~threetears.core.collections.complete_copy.rail_arrow_types`, read off the L1's codes:
    text, a UUID, a ``NUMERIC`` and a timestamp are text; a ``date_`` timestamp an instant in UTC; an integer
    ``int64``; a float ``float64``; a boolean ``bool``.

    :param codes: column name -> its type code (:meth:`~threetears.core.cache.duckdb.DuckDBBackend.column_types`)
    :ptype codes: Mapping[str, str]
    :return: column name -> its Arrow type, in order, or None
    :rtype: dict[str, pyarrow.DataType] | None
    """
    types: dict[str, Any] | None = {}
    for name, code in codes.items():
        if types is None:
            break
        if name.startswith("date_"):
            if code == "VARCHAR_DATETIME":
                types[name] = pa.timestamp("us", tz="UTC")
            else:
                types = None
        elif code in _RAIL_TEXT_CODES or code.startswith("DECIMAL("):
            types[name] = pa.string()
        elif code == "BIGINT":
            types[name] = pa.int64()
        elif code == "DOUBLE":
            types[name] = pa.float64()
        elif code == "BOOLEAN":
            types[name] = pa.bool_()
        else:
            types = None
    return types


def _concatenated(parts: Sequence[Any]) -> Any:
    """Arrow tables of one schema as one, in order: a base and the pieces after it.

    :param parts: the tables, at least one
    :ptype parts: Sequence[pyarrow.Table]
    :return: their rows, in order
    :rtype: pyarrow.Table
    """
    _require_snapshot_extra()
    return parts[0] if len(parts) == 1 else pa.concat_tables(parts)


async def _in_order[T, R](
    items: Iterable[T],
    produce: Callable[[T], Coroutine[Any, Any, R]],
    consume: Callable[[T, R], Awaitable[bool]],
    *,
    ahead: int,
) -> bool:
    """produce each item's result, ``ahead`` at once, and consume the results in the items' order, one at a time.

    At most ``ahead`` results are alive at once -- the one being consumed among them, the next ones
    produced meanwhile -- never every item's: the bound a load from L2 and a rebuild from L3 hold their
    memory to. A result is let go once consumed, before the next item is started.
    In the items' order, so every replica commits the same scopes in the same order (DuckDB scans rows
    in the order they were written).

    :param items: the items, in the order their results are consumed
    :ptype items: Iterable[T]
    :param produce: one item's result (a read)
    :ptype produce: Callable[[T], Coroutine[Any, Any, R]]
    :param consume: takes one item's result (a commit); answers whether to go on
    :ptype consume: Callable[[T, R], Awaitable[bool]]
    :param ahead: how many results are produced at once; at least one
    :ptype ahead: int
    :return: whether every item was consumed; ``False`` once ``consume`` said stop, the rest not produced
    :rtype: bool
    :raises BaseException: what a ``produce`` or ``consume`` raised, or the caller's cancellation;
        whatever was still being produced is cancelled and waited for first
    """
    upcoming = iter(items)
    pending: deque[tuple[T, asyncio.Task[R]]] = deque()
    whole = True

    def top_up() -> None:
        for item in islice(upcoming, max(1, ahead) - len(pending)):
            pending.append((item, asyncio.create_task(produce(item))))

    try:
        top_up()
        while pending and whole:
            item, task = pending.popleft()
            result = await task
            whole = await consume(item, result)
            del result
            top_up()
    except BaseException:
        # what ends the run (a failure, the caller's cancellation) is what is raised: an abandoned
        # read's own failure is said, never raised in its place, which would turn a cancellation
        # into an error the caller handles and carries on from
        await _abandon([task for _, task in pending], unwinding=True)
        raise
    await _abandon([task for _, task in pending], unwinding=False)
    return whole


async def _bounded_each[T](items: Sequence[T], each: Callable[[T], Awaitable[Any]], *, at_once: int) -> None:
    """``each`` over ``items``, at most ``at_once`` at a time, every one awaited; the first failure raised once all
    have ended.

    :param items: the items
    :ptype items: Sequence[T]
    :param each: what is done to one
    :ptype each: Callable[[T], Awaitable[Any]]
    :param at_once: how many at a time
    :ptype at_once: int
    :return: nothing
    :rtype: None
    """
    gate = asyncio.Semaphore(max(1, at_once))

    async def one(item: T) -> None:
        async with gate:
            await each(item)

    results = await asyncio.gather(*(one(item) for item in items), return_exceptions=True)
    _raise_first(results, log=log, what="a scope's sweep or compaction failed")


async def _abandon(tasks: Sequence[asyncio.Task[Any]], *, unwinding: bool) -> None:
    """cancel ``tasks`` and wait for every one to end.

    :param tasks: tasks whose results are no longer wanted
    :ptype tasks: Sequence[asyncio.Task[Any]]
    :param unwinding: whether another exception is ending the caller: each task's own failure is then
        logged, not raised in its place
    :ptype unwinding: bool
    :return: nothing
    :rtype: None
    :raises BaseException: the first task's own failure (anything but its cancellation), when not
        unwinding; the caller's cancellation while it waits
    """
    for task in tasks:
        task.cancel()
    if tasks:
        # asyncio.wait never raises a task's outcome: a CancelledError here is the caller's own
        await asyncio.wait(tasks)
    failures = [error for task in tasks if not task.cancelled() and (error := task.exception()) is not None]
    if failures and not unwinding:
        raise failures[0]
    for error in failures:
        log.warning("a read abandoned while its run ended had failed: %s: %s", type(error).__name__, error)


@dataclass
class _Headway:
    """the progress of one run under the rebuild claim: when it last moved, and what moved it.

    :ivar at: :func:`time.monotonic` at its last step
    :ivar what: that step, for the log of a run that stops making progress
    """

    at: float
    what: str


#: the run under the rebuild claim the current task belongs to (:meth:`ScopedSnapshot._holding`), inherited by
#: every task it starts (a rebuild's reads ahead); ``None`` elsewhere. So only the run's own steps count as its
#: progress: the pointer watch applying another replica's publishes on the same snapshot does not
_HEADWAY: Final[contextvars.ContextVar[_Headway | None]] = contextvars.ContextVar(
    "threetears_scoped_snapshot_headway", default=None
)


def _headway(what: str) -> None:
    """record a step of the run under the rebuild claim the current task belongs to; nothing outside one.

    :param what: the step
    :ptype what: str
    :return: nothing
    :rtype: None
    """
    headway = _HEADWAY.get()
    if headway is not None:
        headway.at = time.monotonic()
        headway.what = what


class _Stalled(TimeoutError):
    """a run under the rebuild claim made no progress for ``rebuild_stall``; the claim was let go."""


class _CatchUpRecord:
    """what a catch-up in progress has found and done, kept as it goes so a catch-up that raises still says it."""

    def __init__(self) -> None:
        self.behind: dict[str, ScopeLag] = {}
        self.epochs: Mapping[str, int] = {}
        self.scopes: set[str] = set()
        self.caught: list[str] = []
        self.holder: str | None = None
        self.held_since: datetime | None = None
        self.claim_held = False


class _WriteBegan(Exception):
    """a write began while a rebuild read a scope: what it read is not the scope at its epoch, and the read stops."""


class _Lost(Exception):
    """the snapshot in L2 is incomplete or unreadable; L3 must rebuild it."""


def _on_an_event_loop() -> bool:
    """whether this thread is running an event loop.

    :return: True on a thread running an event loop
    :rtype: bool
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False  # asyncio's answer for a thread running no loop: a worker thread
    return True


@runtime_checkable
class StagedPartitions(Protocol):
    """an L1 whose commit is staged apart from its publish: what :class:`_VersionedL1` stages outside its swap lock.

    ``stage_partitions`` does all the work of a commit (a worker process applying the change, say) and makes
    nothing visible to a reader; the callable it yields publishes the staged change in one short step, which
    the swap runs under its lock with the epochs' move. Leaving the block releases whatever the staging holds,
    whether it was published or not (a staging that raises, or a publish never called, leaves the L1 as it was).
    A backend without it commits in ``replace_partitions`` under the swap lock, as before.
    """

    def stage_partitions(
        self, replacements: Sequence[PartitionReplacement]
    ) -> AbstractContextManager[Callable[[], object]]:
        """stage ``replacements``, invisible to readers, and yield the step that publishes them.

        :param replacements: the scopes' new contents
        :ptype replacements: Sequence[PartitionReplacement]
        :return: a context manager yielding the publish step
        :rtype: AbstractContextManager[Callable[[], object]]
        """
        ...


class _VersionedL1:
    """the L1, and the epoch of every scope whose rows it holds, moved together under one lock.

    The one owner of the swap lock, from acquire to release: a swap holds it on the worker thread
    that runs it, across the commit and the epochs' move, and lets it go on every path (a commit
    that raises; an awaiter or the commit's own task cancelled, which a worker thread outlives and
    runs to its end). A versioned read holds it only to take the epochs and pin its read, so the
    read sees exactly the rows the epochs name. Nothing here is the snapshot's loop-owned state:
    the loop never reads or writes the epochs, so a swap may move them off the loop.

    A backend whose commit is long and separable (:class:`StagedPartitions`: a change applied in a
    worker process, then published in one step) is staged OUTSIDE the swap lock and only published
    under it: a read opened while the change is staged reads the previous version at once, rather
    than waiting the whole staging out (cobalt prod, 2026-10-31: every versioned read, and every
    call behind one, waited 8-18 s a change). Swaps run one at a time under their own writer lock,
    so a change is staged over the version the one before it published.

    :param backend: the L1
    :ptype backend: DuckDBBackend
    """

    def __init__(self, backend: DuckDBBackend) -> None:
        self._backend = backend
        self._lock = threading.Lock()
        # one swap at a time, its staging included: held outside the swap lock, so readers never wait on it
        self._writing = threading.Lock()
        self._held: Mapping[str, int] = _updated({})

    def swap(self, replacements: Sequence[PartitionReplacement], put: Mapping[str, int], drop: Sequence[str]) -> None:
        """commit replacements in the L1 and move the epochs with them, as one step to any reader.

        Blocking: run on a worker thread.

        :param replacements: the scopes' new contents
        :ptype replacements: Sequence[PartitionReplacement]
        :param put: scope -> its epoch after the commit
        :ptype put: Mapping[str, int]
        :param drop: the scopes the commit drops
        :ptype drop: Sequence[str]
        :return: nothing
        :rtype: None
        """
        backend = self._backend
        with ExitStack() as stack:
            stack.enter_context(self._writing)
            # the long part (a worker process applying the change) outside the swap lock: readers read the previous
            # version meanwhile; only the publish and the epochs' move are one step under it. A backend that does
            # not stage commits whole under the lock, as before
            commit: Callable[[], object] = (
                stack.enter_context(backend.stage_partitions(replacements))
                if isinstance(backend, StagedPartitions)
                else lambda: backend.replace_partitions(replacements)
            )
            with self._lock:
                commit()
                self._held = _updated(self._held, put, drop=drop)

    @contextmanager
    def open(self, table: str) -> Iterator[tuple[Any, Mapping[str, int]]]:
        """a read pinned to one state of the L1, and the epochs of exactly that state.

        Blocking (it waits for a swap in progress): run on a worker thread.

        :param table: a table of the L1, quoted, which the read's first statement touches
        :ptype table: str
        :return: the cursor and the epochs
        :rtype: Iterator[tuple[duckdb.DuckDBPyConnection, Mapping[str, int]]]
        """
        with ExitStack() as stack:
            with self._lock:
                epochs = self._held
                cursor = stack.enter_context(self._backend.read_snapshot())
                # DuckDB fixes a read's state at its first statement, not at BEGIN: pin it while no
                # swap can commit, so the state read is the one the epochs name
                cursor.execute(f"SELECT 1 FROM {table} LIMIT 0").fetchall()
            yield cursor, epochs


def _updated[K, V](mapping: Mapping[K, V], put: Mapping[K, V] | None = None, drop: Iterable[K] = ()) -> Mapping[K, V]:
    """a new read-only mapping: ``mapping`` with ``put`` set and ``drop`` removed; ``mapping`` is untouched.

    The one way the snapshot changes a mapping a reader on another thread may hold (see
    :class:`ScopedSnapshot`): it builds the next value and rebinds the attribute, so a reader
    iterating the value it took never sees it change.

    :return: the new mapping
    :rtype: Mapping[K, V]
    """
    dropped = set(drop)
    merged = {key: value for key, value in mapping.items() if key not in dropped}
    merged.update(put or {})
    return MappingProxyType(merged)


@dataclass(frozen=True)
class _Progress:
    """the derived half of the status, one immutable value rebound on the event loop at each change."""

    phase: SnapshotPhase = SnapshotPhase.STARTING
    source: SnapshotSource | None = None
    detail: str = "watching the pointers"
    scopes_total: int = 0
    scopes_done: int = 0
    timings: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    history: tuple[SnapshotPhase, ...] = (SnapshotPhase.STARTING,)
    last_change: ScopeChange | None = None
    ready_at: datetime | None = None


def _lease_ttl(ttl: timedelta) -> timedelta:
    """a claim's lifetime as a lease and a NATS per-key TTL take it: whole seconds, at least one.

    :param ttl: the lifetime asked for
    :ptype ttl: timedelta
    :return: it, rounded up
    :rtype: timedelta
    """
    return timedelta(seconds=max(1, math.ceil(ttl.total_seconds())))


class _RebuildClaim:
    """one take of the rebuild claim: so one replica at a time rebuilds from L3.

    A :class:`~threetears.core.coordination.lease.KVLease` hold on ``{name}.rebuild``, its entry
    carrying a NATS per-key TTL and renewed at a third of it, so a replica that dies frees it. It is
    contended, so it is never taken back once lost, and it is released only while still this
    replica's. A loss is surfaced: :meth:`release` says whether it was held to the end.

    :param lease: the snapshot's lease on its pointer bucket
    :ptype lease: KVLease
    :param key: the claim's key
    :ptype key: str
    :param ttl: how long it outlives its holder's last renewal
    :ptype ttl: timedelta
    :param name: the snapshot, for the log
    :ptype name: str
    """

    def __init__(self, lease: KVLease, *, key: str, ttl: timedelta, name: str) -> None:
        self._lease = lease
        self._key = key
        self._ttl = _lease_ttl(ttl)
        self._name = name
        self._held: HeldLease | None = None

    async def take(self, *, max_hold: timedelta | None = None) -> bool:
        """take the claim when nobody holds it, and keep renewing it.

        :param max_hold: the longest it is renewed, for a holder that cannot tell its progress (a writer's
            block); past it the renewal stops and the TTL hands the claim on. ``None`` renews it until released:
            a run whose progress is watched (:meth:`ScopedSnapshot._holding`) lets it go itself once it stalls
        :ptype max_hold: timedelta | None
        :return: whether this replica holds it
        :rtype: bool
        """
        try:
            self._held = await self._lease.hold(
                self._key,
                ttl=self._ttl,
                renew_every=self._ttl / 3,
                name="the snapshot rebuild claim",
                renew_failure_level=logging.WARNING,
                log_extra={"snapshot": self._name},
                max_hold=max_hold,
            )
        except LeaseUnavailable:
            return False
        return True

    async def release(self) -> bool:
        """stop renewing and delete the claim, only if it is still this replica's. Never raises.

        :return: whether it was held to the end; ``False`` when it was lost meanwhile (another
            replica may have rebuilt alongside) or never taken
        :rtype: bool
        """
        held, self._held = self._held, None
        if held is None:
            return False
        kept = not held.lost.is_set()
        await held.release()
        if not kept:
            log.warning(
                "scoped snapshot %s: the rebuild claim was lost while this replica held it; another replica "
                "may have rebuilt alongside it",
                self._name,
            )
        return kept


@dataclass(frozen=True)
class _Layout:
    """every key and object name of one snapshot, and how each is read back: the one place each
    grammar lives, so a writer and the sweep that judges what it wrote can never disagree.

    Keys: ``{name}.index``, ``{name}.rebuild`` (the claim), ``{name}.s.{token}`` (a scope's
    pointer) and ``{name}.w.{epoch}.{writer}`` (one write's claim on an epoch). Objects: a whole
    chunk or base, ``{name}/{token}/{epoch}/{table}.{column digest}``; a piece of an append-only
    table, ``{name}.p/{token}/{epoch}/{table}.{column digest}``. ``token`` is :func:`scope_token` of
    the scope. Pieces sit under their own prefix because a replica that predates them sweeps every
    object under ``{name}/`` its pointer's base does not name; a snapshot's name holds no ``.``, so
    no other snapshot's prefix is ``{name}.p/``.
    """

    name: str
    schema: Mapping[str, str]

    @property
    def watch_prefix(self) -> str:
        """the prefix of every key the snapshot keeps."""
        return f"{self.name}."

    @property
    def index_key(self) -> str:
        """the index's key."""
        return f"{self.name}.index"

    @property
    def claim_key(self) -> str:
        """the rebuild claim's key."""
        return f"{self.name}.rebuild"

    @property
    def write_claims_prefix(self) -> str:
        """the prefix of every write claim's key; the declarer's orphan sweep reads the same grammar
        (:func:`~threetears.nats.object_store_requests.is_write_claim_key`)."""
        return f"{self.name}.{WRITE_CLAIM_SEGMENT}."

    def write_claim_key(self, epoch: int, owner: str) -> str:
        """the key of one write's claim on an epoch it writes chunks at; ``owner`` is one subject token."""
        return f"{self.write_claims_prefix}{epoch}.{owner}"

    def epoch_of_write_claim(self, key: str) -> int | None:
        """the epoch a write claim's key names; ``None`` for any other key."""
        head = key[len(self.write_claims_prefix) :].split(".", 1)[0] if key.startswith(self.write_claims_prefix) else ""
        return int(head) if head.isdigit() else None

    def pointer_key(self, scope: str) -> str:
        """a scope's pointer key."""
        return self.pointer_key_of_token(scope_token(scope))

    def pointer_key_of_token(self, token: str) -> str:
        """the pointer key of a scope by its token."""
        return f"{self.name}.s.{token}"

    def is_pointer_key(self, key: str) -> bool:
        """whether a key is a scope's pointer."""
        return key.startswith(f"{self.name}.s.")

    @property
    def objects_prefix(self) -> str:
        """the prefix of every whole chunk and base the snapshot writes."""
        return f"{self.name}/"

    @property
    def pieces_prefix(self) -> str:
        """the prefix of every piece the snapshot writes."""
        return f"{self.name}.p/"

    @property
    def all_prefixes(self) -> tuple[str, str]:
        """the prefixes of everything the snapshot writes: chunks and bases, then pieces."""
        return self.objects_prefix, self.pieces_prefix

    def object_prefix(self, scope: str) -> str:
        """the prefix of a scope's chunks and bases."""
        return self.object_prefix_of_token(scope_token(scope))

    def object_prefix_of_token(self, token: str) -> str:
        """the prefix of a scope's chunks and bases, by its token."""
        return f"{self.name}/{token}/"

    def scope_prefixes(self, scope: str) -> tuple[str, str]:
        """the prefixes of everything written for a scope: its chunks and bases, then its pieces."""
        token = scope_token(scope)
        return self.object_prefix_of_token(token), f"{self.pieces_prefix}{token}/"

    def object_name(self, scope: str, epoch: int, table: str) -> str:
        """a chunk's name: it names the scope, the epoch, the table and the columns it holds."""
        return f"{self.object_prefix(scope)}{epoch}/{table}.{self.schema[table]}"

    def piece_name(self, scope: str, epoch: int, table: str) -> str:
        """a piece's name: as a chunk's, under the pieces' prefix."""
        return f"{self.pieces_prefix}{scope_token(scope)}/{epoch}/{table}.{self.schema[table]}"

    def token_of(self, object_name: str) -> str | None:
        """the scope token an object of this snapshot is under; ``None`` for any other name."""
        rest = next((object_name[len(p) :] for p in self.all_prefixes if object_name.startswith(p)), "")
        token = rest.split("/", 1)[0] if "/" in rest else ""
        return token or None

    def epoch_in_token(self, token: str, object_name: str) -> int | None:
        """the epoch an object under ``token`` was written at; ``None`` when it is not a chunk or piece of it."""
        prefixes = (self.object_prefix_of_token(token), f"{self.pieces_prefix}{token}/")
        rest = next((object_name[len(p) :] for p in prefixes if object_name.startswith(p)), "")
        head = rest.split("/", 1)[0]
        return int(head) if head.isdigit() else None

    def epoch_of(self, scope: str, object_name: str) -> int | None:
        """the epoch an object of ``scope`` was written at; ``None`` when it is not one of the scope's."""
        return self.epoch_in_token(scope_token(scope), object_name)


class _WriteClaims:
    """claims on the epochs this replica writes chunks at, each held in KV while its write runs and
    until a pointer names what it wrote (or the write gives it back): the state the sweep spares
    them by.

    Every hold is its own lease (:class:`~threetears.core.coordination.lease.HeldLease`), which its
    caller owns and releases: one KV entry ``{name}.w.{epoch}.{writer}``, ``writer`` naming this
    replica and the one write, so two writes at one epoch hold two entries and neither lets go of the
    other's. The entry carries a NATS per-key TTL and is renewed at a third of it, so a replica that
    dies stops renewing and its entries LAPSE -- the key goes, which is all the hub's orphan purge
    and the sweep read: a lost lease, a fact about the writer, never an age of the data. An entry
    found gone while its writer still writes (NATS lost the bucket, or an outage outlasted it) is
    taken again, said so.

    :param bucket: the pointer bucket, allowing per-entry TTLs
    :ptype bucket: NatsKvBucket
    :param layout: the snapshot's keys
    :ptype layout: _Layout
    :param owner: this replica, one subject token
    :ptype owner: str
    :param ttl: how long a claim outlives its writer's last renewal; whole seconds, rounded up
    :ptype ttl: timedelta
    """

    def __init__(self, bucket: NatsKvBucket, layout: _Layout, *, owner: str, ttl: timedelta) -> None:
        self._layout = layout
        self._owner = owner
        self._lease = KVLease(None, bucket=bucket, pod_id=owner, expire_entries=True)
        self._ttl = _lease_ttl(ttl)

    async def hold(self, epoch: int) -> HeldLease:
        """claim ``epoch`` for one write, before its first chunk is written; the caller releases it.

        :param epoch: the epoch
        :ptype epoch: int
        :return: the claim, renewing until released
        :rtype: HeldLease
        :raises RuntimeError: when the claim cannot be taken: writing chunks no claim protects would
            let a sweep take them
        """
        key = self._layout.write_claim_key(epoch, f"{self._owner}-{uuid.uuid7().hex}")
        try:
            return await self._lease.hold(
                key,
                ttl=self._ttl,
                renew_every=self._ttl / 3,
                retake=True,
                log_extra={"snapshot": self._layout.name, "epoch": epoch},
            )
        except LeaseUnavailable as exc:
            raise RuntimeError(f"the write claim {key} could not be taken; no chunk is written without it") from exc

    async def close(self) -> None:
        """end every claim handed out that its holder has not released (a stage dropped, a write
        that never reached its release): each is deleted while still this replica's. Never raises.

        :return: nothing
        :rtype: None
        """
        await self._lease.close()


def _still_held(claim: HeldLease | None, what: str) -> None:
    """refuse to go on writing past a lost write claim: a sweep, or the hub's purge, may take what
    it protected.

    :param claim: the claim, or ``None`` for a write that holds none
    :ptype claim: HeldLease | None
    :param what: what was about to happen, for the error
    :ptype what: str
    :return: nothing
    :rtype: None
    :raises RuntimeError: when the claim was lost
    """
    if claim is not None and not claim.held:
        raise RuntimeError(f"the write claim {claim.key} was lost; {what} is refused")


class _Sweeper:
    """retires the chunks the one deletion rule (:meth:`deletable`) allows, through the declarer's
    retire, in batches it accepts. Every deletion path goes through the rule, judged against the
    pointers and the write claims as KV holds them when the sweep runs.

    Apart from the replica so the rule and the sweeps can be read, and changed, on their own: they
    need the layout, the store, the pointer bucket and the retire, and nothing the replica holds.
    """

    def __init__(
        self,
        *,
        name: str,
        layout: _Layout,
        store: NatsObjectStore,
        pointers: NatsKvBucket,
        retire: Callable[[list[str]], Awaitable[Any]] | None,
        retire_batch: int,
    ) -> None:
        self._name = name
        self._layout = layout
        self._store = store
        self._pointers_bucket = pointers
        self._retire = retire
        self._retire_batch = retire_batch

    @property
    def retires(self) -> bool:
        """whether this replica can retire anything (a retire is wired).

        :return: True when it can
        :rtype: bool
        """
        return self._retire is not None

    async def retire_names(self, names: list[str]) -> int:
        """ask the declarer to delete objects, in batches it accepts; never raises.

        :param names: the objects
        :ptype names: list[str]
        :return: how many it asked to delete
        :rtype: int
        """
        asked = 0
        if self._retire is None or not names:
            return asked
        for start in range(0, len(names), self._retire_batch):
            batch = names[start : start + self._retire_batch]
            try:
                await self._retire(batch)
                asked += len(batch)
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- retirement is housekeeping; a failure leaves old chunks for the next retire and must not fail a publish that already succeeded
                contain(
                    exc,
                    log=log,
                    what=(
                        f"scoped snapshot {self._name}: retiring {len(batch)} chunks failed; "
                        f"they stay until the next retire: {exc}"
                    ),
                    extra={"objects": batch},
                )
        return asked

    @staticmethod
    def deletable(name: str, epoch: int | None, served: AbstractSet[str], claimed: frozenset[int]) -> bool:
        """whether one chunk may be deleted: the one rule every retirement goes by.

        Judged by state, never by age:

        1. a chunk its scope's pointer names, as a base or a piece, is being served: kept;
        2. a chunk at an epoch a live write claims is being written, or waits for its pointer to move
           (a writer's stage, a publish or rebuild between its chunk and its pointer): kept;
        3. any other chunk serves nothing and nothing will point at it: deleted. That is a superseded
           chunk, a chunk of a removed scope, and the stage of a write that died or gave it back.

        :param name: the chunk's name
        :ptype name: str
        :param epoch: its epoch, from its name; ``None`` when the name is not a chunk's
        :ptype epoch: int | None
        :param served: every chunk its scope's pointer, read from KV, names (:meth:`_Pointer.names`); empty when
            the scope has none
        :ptype served: AbstractSet[str]
        :param claimed: the epochs live write claims hold, read from KV
        :ptype claimed: frozenset[int]
        :return: whether it may be deleted
        :rtype: bool
        """
        return name not in served and (epoch is None or epoch not in claimed)

    async def pointer_now(self, key: str) -> _Pointer | None:
        """a scope's pointer as KV holds it now, by its key.

        :return: the pointer, or ``None`` when there is none
        :rtype: _Pointer | None
        """
        raw = await self._pointers_bucket.get(key=key)
        return None if raw is None else _Pointer.decode(raw)

    async def claimed_now(self) -> frozenset[int]:
        """the epochs live write claims hold, as KV holds them now (a lapsed claim is gone).

        :return: the epochs
        :rtype: frozenset[int]
        """
        keys = await self._pointers_bucket.list_keys(prefix=self._layout.write_claims_prefix)
        epochs = (self._layout.epoch_of_write_claim(key) for key in keys)
        return frozenset(epoch for epoch in epochs if epoch is not None)

    async def _sweep(self, names: Sequence[str], *, what: str) -> int:
        """retire every chunk of ``names`` the rule allows, judged against KV read after the listing.

        The claims and pointers are read AFTER the chunks were listed: a chunk listed was written
        under a claim taken before it, so the claim is there to be read. Never raises; nothing is
        deleted when KV cannot be read.

        :param names: the chunks listed
        :ptype names: Sequence[str]
        :param what: what is swept, for the log
        :ptype what: str
        :return: how many were retired
        :rtype: int
        """
        retired = 0
        if self._retire is None or not names:
            return retired
        try:
            claimed = await self.claimed_now()
            by_token: dict[str, list[str]] = {}
            for name in names:
                token = self._layout.token_of(name)
                if token is not None:
                    by_token.setdefault(token, []).append(name)
            pointers = {token: await self.pointer_now(self._layout.pointer_key_of_token(token)) for token in by_token}
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- housekeeping, as in retire_names; nothing is deleted on a judgement that could not be made
            contain(
                exc,
                log=log,
                what=f"scoped snapshot {self._name}: reading the pointers and write claims for {what} failed: {exc}",
            )
            return retired
        # each pointer's chunks made a set once: asked per name, the list was built again and searched for every
        # chunk listed, on the event loop
        served = {token: frozenset(() if p is None else p.names()) for token, p in pointers.items()}
        stale = [
            name
            for token, scope_names in by_token.items()
            for name in scope_names
            if self.deletable(name, self._layout.epoch_in_token(token, name), served[token], claimed)
        ]
        retired = await self.retire_names(stale)
        if stale:
            log.info(
                "scoped snapshot %s: %s: %d chunks no pointer serves and no write claims retired",
                self._name,
                what,
                retired,
            )
        return retired

    async def _listed(self, prefixes: Sequence[str], what: str) -> list[str]:
        """the names of the objects under each of ``prefixes``; none (logged) when they cannot be listed.

        :return: the names
        :rtype: list[str]
        """
        names: list[str] = []
        try:
            # one listing filtered here: each listing reads every name the bucket holds, whatever the prefix
            names = [name for name in await self._store.list_names() if name.startswith(tuple(prefixes))]
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- housekeeping, as in retire_names; nothing listed is nothing deleted
            contain(exc, log=log, what=f"scoped snapshot {self._name}: listing chunks for {what} failed: {exc}")
            names = []
        return names

    async def retire_older(self, scope: str, epoch: int) -> int:
        """retire the scope's chunks of epochs before ``epoch`` the rule allows; never raises.

        :param scope: the scope
        :ptype scope: str
        :param epoch: retire only chunks below this epoch
        :ptype epoch: int
        :return: how many were retired
        :rtype: int
        """
        if self._retire is None:
            return 0
        names = await self._listed(self._layout.scope_prefixes(scope), f"scope {scope!r}")
        older = [n for n in names if (e := self._layout.epoch_of(scope, n)) is not None and e < epoch]
        return await self._sweep(older, what=f"scope {scope!r} below epoch {epoch}")

    async def retire_older_many(self, epochs: Mapping[str, int]) -> int:
        """retire, for each scope, its chunks of epochs before its epoch in ``epochs``; never raises.

        One listing and one sweep for every scope, where :meth:`retire_older` per scope lists the
        bucket and asks the declarer once each.

        :param epochs: scope -> retire only its chunks below this epoch
        :ptype epochs: Mapping[str, int]
        :return: how many were retired
        :rtype: int
        """
        if self._retire is None or not epochs:
            return 0
        prefixes = [prefix for scope in epochs for prefix in self._layout.scope_prefixes(scope)]
        names = await self._listed(prefixes, f"{len(epochs)} scopes")
        # each name read once, by its own scope's token: asked of every scope in turn, every name was parsed once per
        # scope, the scope's token made again each time -- 50-60 ms on the event loop a publish on cobalt prod
        below = {scope_token(scope): epoch for scope, epoch in epochs.items()}
        older = [
            n
            for n in names
            if (t := self._layout.token_of(n)) is not None
            and t in below
            and (e := self._layout.epoch_in_token(t, n)) is not None
            and e < below[t]
        ]
        return await self._sweep(older, what=f"{len(epochs)} scopes below their epochs")

    async def retire_names_if_unserved(self, names: Sequence[str], *, what: str) -> int:
        """retire the named chunks the rule allows (a write that gave them back); never raises.

        :param names: the chunks
        :ptype names: Sequence[str]
        :param what: what they are, for the log
        :ptype what: str
        :return: how many were retired
        :rtype: int
        """
        return await self._sweep(names, what=what)

    async def retire_epoch(self, epoch: int) -> int:
        """retire every chunk at ``epoch`` the rule allows; never raises.

        :param epoch: the epoch
        :ptype epoch: int
        :return: how many were retired
        :rtype: int
        """
        if self._retire is None:
            return 0
        names = await self._listed(self._layout.all_prefixes, f"epoch {epoch}")
        at = [
            n
            for n in names
            if (t := self._layout.token_of(n)) is not None and self._layout.epoch_in_token(t, n) == epoch
        ]
        return await self._sweep(at, what=f"epoch {epoch}")

    async def retire_unreferenced(self) -> int:
        """retire every chunk the rule allows, across the store; never raises.

        :return: how many were retired
        :rtype: int
        """
        if self._retire is None:
            return 0
        return await self._sweep(
            await self._listed(self._layout.all_prefixes, "the whole store"), what="the whole store"
        )


@dataclass(frozen=True)
class _Fetched:
    """one scope's chunks fetched for a batch: what the L1 held when the fetch began, and what it brings.

    :ivar held: the pointer the L1 held the scope at, as the fetch judged against
    :ivar current: the pointer the chunks belong to
    :ivar replacements: the scope's new contents
    :ivar started: when the fetch began, for the change's seconds
    """

    held: _Pointer | None
    current: _Pointer
    replacements: list[PartitionReplacement]
    started: float


class ScopedSnapshot:
    """tables a pod needs whole, held in L2 as chunks per scope and table, loaded into DuckDB.

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
    :ptype l3: L3Reader
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
    :param l2_concurrency: scopes a load holds at once: fetched from L2 beside the one it commits
    :ptype l2_concurrency: int
    :param apply_batch: scopes whose moved chunks a replica holds at once to commit them in one swap, so a
        burst of pointers reaches readers as one move; a larger burst is committed in batches of this many
    :ptype apply_batch: int
    :param chunk_reads: the chunks the snapshot reads from the object store at once, whatever the scopes a load
        fetches ahead: each read is the server pushing the chunk's messages to this connection, and too many pending
        has the server drop the pod as a slow consumer
    :ptype chunk_reads: int
    :param l3_concurrency: scopes a rebuild holds at once: read from L3 beside the one it publishes
    :ptype l3_concurrency: int
    :param write_check: how often a rebuild reading a scope asks whether a write has begun, which stops the read
        (:data:`_WRITE_CHECK`)
    :ptype write_check: timedelta
    :param claim_ttl: how long a rebuild's claim, or a write claim, lasts once its holder stops renewing
        it (it died); renewed while held
    :ptype claim_ttl: timedelta
    :param rebuild_stall: how long a holder of the rebuild claim may go without progress before it lets the
        claim go and its rebuild or catch-up ends with a TimeoutError; the longest a writer's
        :meth:`holding_rebuilds` renews it (:data:`_DEFAULT_REBUILD_STALL`)
    :ptype rebuild_stall: timedelta
    :param recheck: how often a waiting replica looks again
    :ptype recheck: timedelta
    :param pointer_watch_heartbeat: the pointer watch's heartbeat; a lost consumer is replaced after three
    :ptype pointer_watch_heartbeat: timedelta
    :param compact_pieces: how many pieces an append-only table's scope gathers before the writer that
        moved it folds them into its base
    :ptype compact_pieces: int
    :param compact_bytes: how many bytes of pieces it gathers before that
    :ptype compact_bytes: int
    :raises ValueError: when the name is outside its grammar or names no table, or ``rebuild_stall`` is not positive
    """

    def __init__(
        self,
        *,
        name: str,
        tables: Sequence[SnapshotTable],
        backend: DuckDBBackend,
        store: NatsObjectStore,
        pointers: NatsKvBucket,
        l3: L3Reader,
        epochs: Callable[[], Awaitable[Mapping[str, int]]],
        settled: Callable[[], Awaitable[Any]] | None = None,
        ensure_buckets: Callable[[], Awaitable[None]] | None = None,
        retire: Callable[[list[str]], Awaitable[Any]] | None = None,
        retire_batch: int = MAX_RETIRED_OBJECTS,
        l2_concurrency: int = _DEFAULT_L2_CONCURRENCY,
        apply_batch: int = _DEFAULT_APPLY_BATCH,
        chunk_reads: int = _DEFAULT_CHUNK_READS,
        l3_concurrency: int = _DEFAULT_L3_CONCURRENCY,
        write_check: timedelta = _WRITE_CHECK,
        claim_ttl: timedelta = _DEFAULT_CLAIM_TTL,
        rebuild_stall: timedelta = _DEFAULT_REBUILD_STALL,
        recheck: timedelta = _DEFAULT_RECHECK,
        pointer_watch_heartbeat: timedelta = timedelta(seconds=5),
        compact_pieces: int = _DEFAULT_COMPACT_PIECES,
        compact_bytes: int = _DEFAULT_COMPACT_BYTES,
    ) -> None:
        _require_snapshot_extra()
        if not name or any(ch not in _NAME_GRAMMAR for ch in name):
            raise ValueError(f"snapshot name {name!r} must be letters, digits, '-' and '_'")
        if not tables:
            raise ValueError("a snapshot holds at least one table")
        if rebuild_stall <= timedelta(0):
            raise ValueError(f"rebuild_stall must be positive; got {rebuild_stall}")
        self._name = name
        self._tables = tuple(tables)
        self._append = frozenset(t.name for t in self._tables if t.append_only)
        self._backend = backend
        self._store = store
        self._pointers_bucket = pointers
        self._l3 = l3
        self._epochs = epochs
        self._settled = settled
        self._ensure_buckets = ensure_buckets
        self._l2_concurrency = l2_concurrency
        #: the chunk reads in flight, across every scope fetched ahead
        self._chunk_reads = asyncio.Semaphore(max(1, chunk_reads))
        self._apply_batch = max(1, apply_batch)
        self._l3_concurrency = l3_concurrency
        self._write_check = write_check
        self._claim_ttl = claim_ttl
        self._rebuild_stall = rebuild_stall
        self._recheck = recheck
        self._heartbeat = pointer_watch_heartbeat
        self._compact_pieces = max(1, compact_pieces)
        self._compact_bytes = max(1, compact_bytes)
        self._schema = {t.name: backend.schema_digest(t.name) for t in self._tables}
        self._layout = _Layout(name, self._schema)
        self._sweeper = _Sweeper(
            name=name,
            layout=self._layout,
            store=store,
            pointers=pointers,
            retire=retire,
            retire_batch=max(1, min(retire_batch, MAX_RETIRED_OBJECTS)),
        )
        self._replica = uuid.uuid7().hex.encode("utf-8")
        self._write_claims = _WriteClaims(pointers, self._layout, owner=self._replica.decode("utf-8"), ttl=claim_ttl)
        self._rebuild_lease = KVLease(None, bucket=pointers, pod_id=self._replica.decode("utf-8"), expire_entries=True)
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
        # What the L1 holds. This, `_behind`, `_rows` and `_progress` are the state a reader on any
        # thread may take (status, read_with_behind, applied_epoch(s)). The event loop is their one
        # writer, and it never changes a value in place: it builds the next one (_updated, replace)
        # and rebinds the attribute, so whatever a reader took stays as it was while it iterates.
        # Code run on a worker thread touches the backend and the versioned L1 (_VersionedL1, which
        # owns its own lock and epochs), and nothing else here.
        self._applied: Mapping[str, _Pointer] = _updated({})
        # the L1's commits, each with the epoch of every scope's rows moved in the same step, so a
        # versioned read on any thread pairs its rows with exactly their epochs (read_versioned).
        # it owns its lock and its epochs; this snapshot's commits run through it, off the loop
        self._versioned = _VersionedL1(self._backend)
        self._local = asyncio.Lock()
        # scope -> the epoch a rebuild from L3 (a full one or a catch-up) published it at, from this L1, under
        # this replica's columns. Kept across the rebuild's waits: a rebuild a write in progress stopped goes
        # on from what it published, and does not read every scope again from the first, which a loader
        # writing with short gaps between its writes kept from ever finishing. Emptied when NATS lost the
        # snapshot, whose chunks and pointers went with it. Only the event loop reads or writes it.
        self._rebuilt: dict[str, int] = {}
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
        self._behind: Mapping[str, tuple[int, str]] = _updated({})
        # table -> scope -> rows the L1 holds, kept as each replacement commits, so the status never
        # queries DuckDB (a query would wait on the lock a load or a swap holds, on the event loop)
        self._rows: Mapping[str, Mapping[str, int]] = _updated({t.name: _updated({}) for t in self._tables})
        self._tasks: list[asyncio.Task[None]] = []
        self._listeners: list[Callable[[], None]] = []

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
        history = progress.history if progress.history[-1] is phase else (*progress.history, phase)[-_HISTORY:]
        self._progress = replace(progress, phase=phase, detail=detail, history=history)

    @contextmanager
    def _doing(self, phase: SnapshotPhase, detail: str, *, total: int) -> Iterator[None]:
        """a step in progress (loading from L2, rebuilding from L3), counted over ``total`` scopes.

        :return: nothing
        :rtype: Iterator[None]
        """
        self._activity = (phase, detail)
        self._progress = replace(self._progress, scopes_total=total, scopes_done=0)
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
        rows = dict(self._rows)
        for r in replacements:
            held = r.arrow.num_rows if r.arrow is not None else len(r.rows or ())
            if r.append:
                held += rows[r.table].get(str(r.value), 0)
            rows[r.table] = _updated(rows[r.table], {str(r.value): held})
        self._rows = _updated(rows)

    def status(self) -> SnapshotStatus:
        """what the snapshot is doing and has done.

        :return: the status
        :rtype: SnapshotStatus
        """
        # each taken once: the loop rebinds them, never changes them, so this is one state of each
        progress, rows, behind, applied = self._progress, self._rows, self._behind, self._applied
        return SnapshotStatus(
            phase=progress.phase,
            source=progress.source,
            detail=progress.detail,
            scopes_total=progress.scopes_total,
            scopes_done=progress.scopes_done,
            rows={table: sum(scopes.values()) for table, scopes in rows.items()},
            timings=dict(progress.timings),
            history=tuple(progress.history),
            last_change=progress.last_change,
            ready_at=progress.ready_at,
            behind={scope: why for scope, (_, why) in sorted(behind.items())},
            appended={
                table: {
                    scope: AppendedScope(
                        pieces=len(pointer.pieces.get(table, ())),
                        compacted_at=(
                            datetime.fromtimestamp(pointer.compacted_at[table], UTC)
                            if table in pointer.compacted_at
                            else None
                        ),
                    )
                    for scope, pointer in sorted(applied.items())
                }
                for table in sorted(self._append)
            },
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

    async def _commit(
        self,
        replacements: Sequence[PartitionReplacement],
        current: Sequence[str],
        *,
        epochs: Mapping[str, int | None],
        count: bool = True,
    ) -> None:
        """commit replacements in this L1 on a worker thread, then, back on the event loop, take the
        scopes they bring current off ``behind`` and count what each scope holds.

        Only the versioned L1 (:class:`_VersionedL1`: the backend's commit and the epochs' move,
        under its own lock) runs off the loop. ``behind`` changes after the commit, never before,
        so a reader that takes ``behind`` before it opens its read (:meth:`read_with_behind`) is
        never told a scope is current when its read holds the older rows.

        :param replacements: the scopes' new contents
        :ptype replacements: Sequence[PartitionReplacement]
        :param current: the scopes they bring current
        :ptype current: Sequence[str]
        :param epochs: scope -> its epoch after the commit, ``None`` for a scope dropped; moved with the
            rows in one step, for :meth:`read_versioned`
        :ptype epochs: Mapping[str, int | None]
        :param count: whether to count what each scope now holds (a drop forgets them instead)
        :ptype count: bool
        :return: nothing
        :rtype: None
        """
        put = {scope: epoch for scope, epoch in epochs.items() if epoch is not None}
        dropped = [scope for scope, epoch in epochs.items() if epoch is None]
        # the worker takes and lets go of the swap lock itself, whatever happens to this task
        await asyncio.to_thread(self._versioned.swap, replacements, put, dropped)
        _headway(f"committed {len(replacements)} replacements here")
        self._behind = _updated(self._behind, drop=current)
        if count:
            self._count(replacements)

    @contextmanager
    def read_with_behind(self) -> Iterator[tuple[Any, Mapping[str, str]]]:
        """a read of one state of every table, and the scopes that state is behind on, taken together.

        The behind set is taken before the read opens, and a scope leaves it only once its new rows
        have committed, so it names every scope the cursor's data is behind on; it may also name one
        brought current just as the read opened (over-reporting is the safe side). Safe on any
        thread, and it never waits on a write in progress.

        :return: the cursor, and scope -> why it is behind
        :rtype: Iterator[tuple[duckdb.DuckDBPyConnection, Mapping[str, str]]]
        """
        # taken before the read opens, and a scope leaves it only after its commit: so it names every
        # scope the read is behind on, and may still name one brought current in between
        behind = self._behind
        with self._backend.read_snapshot() as cursor:
            yield cursor, MappingProxyType({scope: why for scope, (_, why) in behind.items()})

    @property
    def behind(self) -> Mapping[str, str]:
        """the scopes this replica's L1 is behind on now, and why, as a read opened now reports them
        (:meth:`read_versioned`): for a reader of a copy published from an earlier read (a shared-memory
        version, :mod:`~threetears.core.collections.shared_snapshot`) to see that what it holds no longer
        says what the L1 does. Safe on any thread.

        :return: scope -> why it is behind
        :rtype: Mapping[str, str]
        """
        behind = self._behind
        return MappingProxyType({scope: why for scope, (_, why) in behind.items()})

    def applied_epoch(self, scope: str) -> int | None:
        """the epoch of ``scope`` this replica's L1 holds; None when it holds none.

        :param scope: the scope
        :ptype scope: str
        :return: the epoch
        :rtype: int | None
        """
        held = self._applied.get(scope)
        return None if held is None else held.epoch

    def applied_epochs(self) -> dict[str, int]:
        """every scope this replica's L1 holds, and its epoch. A scope is listed only once its
        replacement has committed, so a read opened after this call sees at least these epochs.

        :return: scope -> epoch
        :rtype: dict[str, int]
        """
        applied = self._applied  # taken once: the loop rebinds it, never changes it
        return {scope: pointer.epoch for scope, pointer in applied.items()}

    @contextmanager
    def read(self) -> Iterator[Any]:
        """a cursor reading one state of every table for the whole block; hold it for one request.

        :return: the cursor
        :rtype: Iterator[duckdb.DuckDBPyConnection]
        """
        with self._backend.read_snapshot() as cursor:
            yield cursor

    @contextmanager
    def read_versioned(self) -> Iterator[VersionedRead]:
        """a read of the copy, as :meth:`read`, and the epoch of every scope whose rows it reads.

        For an answer published under the version of the data it was computed from: the epochs are
        those of the rows the cursor sees, never one swap before or after, so an answer labelled
        with them is the answer at those epochs. A read opened while a swap is in progress waits
        for the swap and its epochs' move, and reads their result; it never fails for a swap however
        long. Blocking: call it from a worker thread; on the event loop it raises at once.

        :return: the read, its epochs and the scopes it is behind on
        :rtype: Iterator[VersionedRead]
        :raises RuntimeError: when called on a thread running an event loop
        """
        if _on_an_event_loop():
            raise RuntimeError(
                "read_versioned blocks while a swap commits; call it from a worker thread "
                "(asyncio.to_thread), never on the event loop"
            )
        table = quote_identifier(self._tables[0].name)
        # taken before the read opens, and a scope leaves it only after its commit (as
        # read_with_behind): it names every scope the read is behind on
        behind = self._behind
        with self._versioned.open(table) as (cursor, epochs):
            yield VersionedRead(
                cursor=cursor,
                epochs=epochs,
                behind=MappingProxyType({scope: why for scope, (_, why) in behind.items()}),
            )

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
        """call ``listener`` after every commit that changes what this L1 holds, and once when the copy becomes ready.

        A load from L2, a scope applied or dropped, a publish made here: each calls every listener
        once its DuckDB transaction has committed, so a reader opened from the listener sees the
        change. The copy becoming ready (:attr:`is_ready` turning true) calls each once more, so a
        listener that skips a change while the copy is not ready yet is told when it is. Called on the
        event loop; a listener that raises is logged, never propagated. Call this on the event loop
        too: the listeners are the loop's.

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
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- a listener's failure must not undo or stop an L1 change already committed; logged
                contain(exc, log=log, what=f"scoped snapshot {self._name}: a change listener failed", level="error")

    @asynccontextmanager
    async def holding_rebuilds(self, *, attempts: int = 10) -> AsyncIterator[bool]:
        """hold the rebuild claim for the block, so no replica starts a rebuild from L3 meanwhile.

        For a writer that is about to commit a change to L3 and publish it here: between its commit
        and its publish, a replica waiting on the write would otherwise read every scope from L3
        (the slow path) only to find the same rows already published. The claim is tried a few times
        (a replica that holds it only to look at the writer's seqlock lets go at once); when another
        replica is really rebuilding, the block runs without it, which is correct, only slower. It is
        renewed for at most ``rebuild_stall``: a block stuck past that runs on without it, so a writer
        that never gets through its commit does not keep every replica's rebuild out.

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
            if await claim.take(max_hold=self._rebuild_stall):
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
            # waited, not awaited: awaiting a cancelled task raises its CancelledError, and suppressing
            # that would also swallow a cancellation of THIS task arriving meanwhile
            await asyncio.wait(self._tasks)
        self._tasks = []
        # every claim this replica handed out and nobody released (a stage its writer dropped) ends
        # here, rather than renewing for the life of the process
        await self._write_claims.close()
        await self._rebuild_lease.close()

    async def _watch(self) -> None:
        """follow the pointer bucket and wake the worker on every change; say so loudly if it ends.

        :return: nothing
        :rtype: None
        """
        try:
            await self._follow(self._layout.watch_prefix)
            ended = "the pointer watch ended"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the watch is the replica's only news of changes; whatever ends it is logged and stated in the status rather than lost with the task
            contain(exc, log=log, what=f"scoped snapshot {self._name}: the pointer watch ended", level="error")
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
        # sender-posture: own-consumer -- a pointer counts only with the watch consumer's own ack reply
        async with aclosing(self._pointers_bucket.watch_prefix(prefix=prefix, heartbeat=self._heartbeat)) as watch:
            async for update in watch:
                if update is None:
                    self._caught_up = True
                elif update.key == self._layout.index_key:
                    body = None if update.value is None else json.loads(update.value)
                    self._index = None if body is None else frozenset(body["scopes"])
                    self._index_partial = body is not None and bool(body.get("partial", False))
                    self._ever_indexed = self._ever_indexed or self._index is not None
                    self._removing &= self._index or frozenset()
                elif self._layout.is_pointer_key(update.key):
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
            for scope in [s for s in self._seen if self._layout.pointer_key(s) == key]:
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
                contain(exc, log=log, what=f"scoped snapshot {self._name}: sync failed", level="error")
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
                # chunks a pointer names are gone: what a rebuild here published is no longer known to stand
                self._rebuilt.clear()
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
        self._behind = _updated(
            self._behind, {scope: (_BEHIND_REBUILD_AFTER, "its pointer carries other columns; rebuilt from L3")}
        )

    async def _fetch(self, pointer: _Pointer, held: _Pointer | None = None) -> list[PartitionReplacement]:
        """every table's rows of one scope, decoded and checked; of a chain, only what ``held`` lacks.

        A table whose rows are a chain (an append-only table, or one whose pointer lists pieces) is
        judged against what this L1 holds of it (``held``, the pointer it last applied):

        - the same chain: nothing to fetch;
        - what it holds is a leading run of the new chain (the same base, its pieces the first of
          the new list): only the pieces after them, appended;
        - the same epoch, the chain folded otherwise (a compaction): nothing to fetch, since an
          append-only table's rows at one epoch are one set;
        - anything else: the base and every piece, replacing the scope.

        Any other table is replaced whole, its chunk fetched, as before pieces existed.

        :param pointer: the scope's pointer
        :ptype pointer: _Pointer
        :param held: the pointer this L1 holds the scope at, judged against; ``None`` fetches everything
        :ptype held: _Pointer | None
        :return: the scope's replacements, an appended table's marked ``append``
        :rtype: list[PartitionReplacement]
        :raises _Lost: when a chunk is gone or does not hold what its pointer says
        """
        if not self._loadable(pointer):
            # every load passes here: no path can put another column set's chunks in this L1
            raise _Lost(f"scope {pointer.scope!r}'s pointer carries other columns than this replica's")
        relative = held is not None and self._loadable(held)
        replacements: list[PartitionReplacement] = []
        for table in self._tables:
            name = pointer.objects.get(table.name)
            if name is None:
                raise _Lost(f"scope {pointer.scope!r} names no chunk for {table.name}")
            base, pieces = pointer.chain(table.name)
            chained = table.append_only or bool(pieces) or bool(held is not None and held.pieces.get(table.name))
            fetch: tuple[_Piece, ...] | None = None  # the pieces to append; None for the whole chain
            if relative and chained and held is not None:
                held_base, held_pieces = held.chain(table.name)
                if (held_base, held_pieces) == (base, pieces) or held.epoch == pointer.epoch:
                    continue
                if held_base == base and pieces[: len(held_pieces)] == held_pieces:
                    fetch = pieces[len(held_pieces) :]
            if fetch is not None:
                parts = [await self._read_chunk(piece.name, piece.rows) for piece in fetch]
            else:
                parts = [await self._read_chunk(name, pointer.rows.get(table.name))]
                parts += [await self._read_chunk(piece.name, piece.rows) for piece in pieces]
            replacements.append(
                PartitionReplacement(
                    table=table.name,
                    column=table.scope_column,
                    value=pointer.scope,
                    arrow=_concatenated(parts),
                    append=fetch is not None,
                )
            )
        return replacements

    async def _read_chunk(self, name: str, rows: int | None) -> Any:
        """one chunk's rows, checked against the count its pointer gives.

        :param name: the chunk
        :ptype name: str
        :param rows: the rows its pointer says it holds
        :ptype rows: int | None
        :return: the rows
        :rtype: pyarrow.Table
        :raises _Lost: when it is gone, unreadable, or holds another count
        """
        try:
            async with self._chunk_reads:
                data = await self._store.get(name)
        except ObjectNotFoundError as exc:
            raise _Lost(f"chunk {name} is gone") from exc
        except ObjectStoreError as exc:
            raise _Lost(f"chunk {name} could not be read: {exc}") from exc
        arrow = await asyncio.to_thread(decode_chunk, data)
        if arrow.num_rows != rows:
            raise _Lost(f"chunk {name} holds {arrow.num_rows} rows where its pointer says {rows}")
        _headway(f"read chunk {name}")
        return arrow

    async def _fetch_current(
        self, scope: str, pointer: _Pointer, held: _Pointer | None = None
    ) -> tuple[_Pointer, list[PartitionReplacement]]:
        """a scope's chunks, reading its pointer again once when a chunk was retired under the read.

        :param held: the pointer this L1 holds the scope at, as :meth:`_fetch` judges against
        :ptype held: _Pointer | None
        :return: the pointer the chunks belong to, and the replacements
        :rtype: tuple[_Pointer, list[PartitionReplacement]]
        :raises _Lost: when the chunks cannot be read even at the current pointer
        """
        current = pointer
        try:
            replacements = await self._fetch(pointer, held)
        except _Lost:
            raw = await self._pointers_bucket.get(key=self._layout.pointer_key(scope))
            if raw is None or not _Pointer.decode(raw).supersedes(pointer):
                raise
            current = _Pointer.decode(raw)
            replacements = await self._fetch(current, held)
        return current, replacements

    async def _commit_fetched(
        self, replacements: Sequence[PartitionReplacement], current: Sequence[str], *, epochs: Mapping[str, int]
    ) -> None:
        """commit what was fetched from chunks; a commit the L1 refuses is a scope that cannot be applied.

        A piece holding a row the L1 already holds (a writer that broke the append-only promise) is
        refused by the table's key, and the swap rolls back. Raised as :class:`_Lost`, so the scope is
        shown behind and, failing again, rebuilt from L3 like any scope whose chunks cannot be applied,
        rather than failing every pass of the worker.

        :raises _Lost: when the L1 refuses the rows
        """
        try:
            await self._commit(replacements, current, epochs=epochs)
        except duckdb.ConstraintException as exc:
            raise _Lost(f"the L1 refused the rows of {sorted(current)}: {exc}") from exc

    async def _load(self, pointers: Mapping[str, _Pointer], *, cold: bool) -> None:
        """fetch the scopes' chunks, several at once, and replace each scope in its own transaction, in scope order.

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

        A scope is committed as soon as it and every scope before it are fetched, and let go once
        committed: what a load holds at once is the decoded chunks of ``l2_concurrency`` scopes, the
        one committing among them, never every scope's. In scope order, not arrival order: DuckDB scans rows
        in the order they were written, and a floating-point sum over them depends on that order, so
        every replica loading the same chunks holds them in the same order and sums them to the same
        last digit. A load that fails part way keeps the scopes it committed, each whole at its
        pointer's epoch; the copy is not ready until every scope is in.

        :return: nothing
        :rtype: None
        :raises _Lost: when a chunk is gone or wrong
        """
        started = time.perf_counter()
        spent = {"fetch": 0.0, "load": 0.0}
        loaded = 0

        async def fetch(scope: str) -> tuple[_Pointer, list[PartitionReplacement]]:
            began = time.perf_counter()
            fetched = await self._fetch_current(scope, pointers[scope])
            spent["fetch"] += time.perf_counter() - began
            return fetched

        async def commit(scope: str, fetched: tuple[_Pointer, list[PartitionReplacement]]) -> bool:
            nonlocal loaded
            began = time.perf_counter()
            current, replacements = fetched
            async with self._local:
                fresh = current.supersedes(self._applied.get(scope))
                if fresh:
                    await self._commit_fetched(replacements, [scope], epochs={scope: current.epoch})
                    self._applied = _updated(self._applied, {scope: current})
            if fresh:
                self._notify()
            loaded += sum(r.arrow.num_rows for r in replacements if r.arrow is not None)
            spent["load"] += time.perf_counter() - began
            if cold:
                self._progress = replace(self._progress, scopes_done=self._progress.scopes_done + 1)
            return True

        await _in_order(sorted(pointers), fetch, commit, ahead=self._l2_concurrency)
        done = time.perf_counter()
        if cold:
            self._progress = replace(
                self._progress,
                timings=_updated(
                    self._progress.timings,
                    {
                        # the seconds the scopes' fetches took, and their commits, summed over the scopes;
                        # the fetches run beside each other and beside the commits, so these overlap
                        "fetch": round(spent["fetch"], 3),
                        "load": round(spent["load"], 3),
                        "total": round(done - started, 3),
                    },
                ),
            )
            self._become_ready(SnapshotSource.L2)
        log.info(
            "scoped snapshot %s: scopes loaded from NATS",
            self._name,
            extra={
                "extra_data": {
                    "scopes": len(pointers),
                    "rows": loaded,
                    "fetch_seconds": round(spent["fetch"], 3),
                    "load_seconds": round(spent["load"], 3),
                    "total_seconds": round(done - started, 3),
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
        self._progress = replace(self._progress, source=source, ready_at=self._progress.ready_at or datetime.now(UTC))
        becoming = not self._ready.is_set()
        self._ready.set()
        # a flow that has not waited in the run it is in waits no longer: a reader woken by wait_ready sees READY, not
        # the wait of a run before, which _end_run would clear only once this run has let its claim go
        for flow, waited in self._waited.items():
            if not waited:
                self._waiting.pop(flow, None)
        self._settle()
        if becoming:
            # a listener told of a commit while the copy was not ready yet skipped it; with no later write it
            # would never hear again, and a consumer built on the copy would never build
            self._notify()

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
            # judged again under the L1's lock: a publish of a new scope holds it from listing the scope as
            # applied until its pointer is seen, and a scope judged gone in between was dropped once the lock
            # was free, lost to readers until a later recheck applied it again
            await self._drop_locally(gone, unless_seen=True)
        moved = {
            scope: self._seen[scope]
            for scope in index
            if scope in self._seen and self._ahead(self._seen[scope], self._applied.get(scope))
        }
        for scope in index:
            if scope in self._seen and self._foreign_ahead(self._seen[scope], self._applied.get(scope)):
                self._mark_foreign(scope)

        def still_moved() -> dict[str, _Pointer]:
            return {
                scope: self._seen[scope]
                for scope in self._index or ()
                if scope in self._seen and self._ahead(self._seen[scope], self._applied.get(scope))
            }

        await self._apply_pointers(moved, mark_behind=True, more=still_moved)
        # a scope removed before it was ever applied leaves no `behind` entry to report, or to rebuild
        self._behind = _updated(
            self._behind, drop=[scope for scope in self._behind if scope not in index and scope not in self._seen]
        )
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
            rebuilt = await self._holding(
                claim,
                lambda: self._rebuild_scopes(scopes, reason=f"{len(scopes)} scopes could not be applied", flow="sync"),
                what="a rebuild of scopes that could not be applied",
            )
        finally:
            await claim.release()
        # a scope the rebuild brought current left `behind` at its commit (_commit); one it could not
        # (L3 below its pointer, or a pointer of other columns it may not load) stays behind
        log.info("scoped snapshot %s: rebuilt %s from L3", self._name, sorted(rebuilt or ()))

    async def _apply_pointers(
        self,
        pointers: Mapping[str, _Pointer],
        *,
        mark_behind: bool = False,
        more: Callable[[], Mapping[str, _Pointer]] | None = None,
    ) -> dict[str, str]:
        """fetch the scopes' chunks side by side and replace them in this L1 in one swap, unless it already holds as new.

        A burst -- a writer moving many scopes' pointers at once -- reaches readers as one move: the chunks
        are fetched ``l2_concurrency`` at once, the scopes whose pointers moved meanwhile (``more``) are
        fetched too, and what was fetched is committed in one transaction, with one change notice. At most
        ``apply_batch`` scopes' chunks are held at once; a larger burst is committed in batches of that
        many. A batch the L1 refuses is committed scope by scope, so one scope that cannot be applied does
        not hold the others back.

        A scope whose chunks cannot be read has failed, and with ``mark_behind`` goes on ``behind``, one failure
        more, unless a write moved it on while they were read. That is judged under the L1's lock, which a publish holds from its local commit (which takes
        the scope off ``behind``) until its pointer moves: judged outside it, a failure landing in that window put
        the scope back on ``behind`` after the publish took it off, and it stayed there though it was current.

        :param pointers: the scopes to apply and their pointers
        :ptype pointers: Mapping[str, _Pointer]
        :param mark_behind: whether a failure goes on ``behind`` (a recheck's), or is only returned (a rebuild's)
        :ptype mark_behind: bool
        :param more: the scopes to apply as they stand now, asked again once a batch is fetched, so pointers that
            moved while it was fetched join it; ``None`` applies ``pointers`` alone
        :ptype more: Callable[[], Mapping[str, _Pointer]] | None
        :return: scope -> why its chunks could not be read, for each one no write moved on from, each logged
        :rtype: dict[str, str]
        """
        failed: dict[str, str] = {}
        pending = dict(pointers)
        # the pointer each scope was fetched at in this call: a scope is fetched again only once it moves past it
        fetched_at: dict[str, _Pointer] = {}
        joins = 0
        while pending:
            batch: dict[str, _Fetched] = {}
            while pending and len(batch) < self._apply_batch:
                wave = dict(islice(sorted(pending.items()), self._apply_batch - len(batch)))
                for scope, pointer in wave.items():
                    del pending[scope]
                    fetched_at[scope] = pointer
                batch.update(await self._fetch_wave(wave, failed, mark_behind=mark_behind))
                if more is not None and not pending and joins < _APPLY_JOINS:
                    # the scopes that moved while the wave fetched join it, a bounded number of times: a scope a
                    # writer keeps moving would otherwise hold the batch uncommitted for as long as it moves
                    joins += 1
                    pending = self._moved_since(more(), fetched_at)
            await self._commit_batch(batch, failed, mark_behind=mark_behind)
        if more is not None and self._moved_since(more(), fetched_at):
            # what moved after the last join is the next pass's, at once
            self._changed.set()
        return failed

    @staticmethod
    def _moved_since(current: Mapping[str, _Pointer], fetched_at: Mapping[str, _Pointer]) -> dict[str, _Pointer]:
        """the scopes not fetched yet, or moved past the pointer they were fetched at.

        :return: scope -> its pointer now
        :rtype: dict[str, _Pointer]
        """
        return {
            scope: pointer
            for scope, pointer in current.items()
            if scope not in fetched_at or pointer.supersedes(fetched_at[scope])
        }

    async def _fetch_wave(
        self, pointers: Mapping[str, _Pointer], failed: dict[str, str], *, mark_behind: bool
    ) -> dict[str, _Fetched]:
        """fetch the scopes' chunks side by side, ``l2_concurrency`` at once; record each that cannot be read.

        :param pointers: the scopes and their pointers
        :ptype pointers: Mapping[str, _Pointer]
        :param failed: where a scope that cannot be read is recorded, with why
        :ptype failed: dict[str, str]
        :param mark_behind: whether such a scope goes on ``behind``
        :ptype mark_behind: bool
        :return: scope -> what was fetched, for each scope that was
        :rtype: dict[str, _Fetched]
        """
        gate = asyncio.Semaphore(max(1, self._l2_concurrency))

        async def fetch(scope: str, pointer: _Pointer) -> _Fetched | None:
            started = time.perf_counter()
            held = self._applied.get(scope)
            try:
                async with gate:
                    current, replacements = await self._fetch_current(scope, pointer, held)
            except _Lost as exc:
                await self._note_unread(scope, pointer, exc, failed, mark_behind=mark_behind)
                return None
            return _Fetched(held=held, current=current, replacements=replacements, started=started)

        results = await asyncio.gather(
            *(fetch(scope, pointer) for scope, pointer in pointers.items()), return_exceptions=True
        )
        _raise_first(results, log=log, what=f"scoped snapshot {self._name}: a scope's fetch failed")
        return {scope: got for scope, got in zip(pointers, results, strict=True) if isinstance(got, _Fetched)}

    async def _note_unread(
        self, scope: str, pointer: _Pointer, exc: _Lost, failed: dict[str, str], *, mark_behind: bool
    ) -> None:
        """record a scope whose chunks could not be read or applied, unless a write moved it on meanwhile.

        :return: nothing
        :rtype: None
        """
        log.warning("scoped snapshot %s: scope %r could not be applied: %s", self._name, scope, exc)
        async with self._local:
            seen = self._seen.get(scope)
            if (seen is not None and seen.supersedes(pointer)) or not self._ahead(pointer, self._applied.get(scope)):
                # a write moved the scope past the pointer that failed; the next recheck judges the new one
                return
            if mark_behind:
                self._behind = _updated(self._behind, {scope: (self._behind.get(scope, (0, ""))[0] + 1, str(exc))})
        failed[scope] = str(exc)

    async def _commit_batch(self, batch: Mapping[str, _Fetched], failed: dict[str, str], *, mark_behind: bool) -> None:
        """commit what a batch fetched in one swap, the scopes in order; scope by scope when the L1 refuses it.

        :param batch: scope -> what was fetched
        :ptype batch: Mapping[str, _Fetched]
        :param failed: where a scope the L1 refuses is recorded, with why, as one whose chunks cannot be read is
        :ptype failed: dict[str, str]
        :param mark_behind: whether such a scope goes on ``behind``
        :ptype mark_behind: bool
        :return: nothing
        :rtype: None
        """
        if not batch:
            return
        applied: dict[str, _Fetched] = {}
        async with self._local:
            fresh = {}
            for scope in sorted(batch):
                got = batch[scope]
                if self._applied.get(scope) is not got.held:
                    # what the fetch was judged against moved meanwhile (a publish here): the next
                    # pass judges the scope again from what the L1 holds now
                    self._changed.set()
                elif got.current.supersedes(got.held):
                    fresh[scope] = got
            try:
                await self._commit_fetched(
                    [replacement for got in fresh.values() for replacement in got.replacements],
                    list(fresh),
                    epochs={scope: got.current.epoch for scope, got in fresh.items()},
                )
                applied = fresh
            except _Lost as exc:
                # NOSILENT: logged here; each scope is committed on its own below, and one refused alone is recorded
                log.warning(
                    "scoped snapshot %s: the L1 refused a batch of %d scopes; committing them one by one: %s",
                    self._name,
                    len(fresh),
                    exc,
                    extra={"extra_data": {"scopes": list(fresh)}},
                )
                applied = {}
            else:
                self._applied = _updated(self._applied, {scope: got.current for scope, got in applied.items()})
        if fresh and not applied:
            for scope, got in fresh.items():
                await self._commit_one(scope, got, failed, mark_behind=mark_behind)
            return
        if applied:
            self._notify()
            for scope, got in applied.items():
                self._record_change(scope, got.current, got.started)

    async def _commit_one(self, scope: str, got: _Fetched, failed: dict[str, str], *, mark_behind: bool) -> None:
        """commit one scope's fetched chunks, as a batch the L1 refused is committed.

        :param failed: where the scope is recorded when the L1 refuses it
        :ptype failed: dict[str, str]
        :param mark_behind: whether it then goes on ``behind``
        :ptype mark_behind: bool
        :return: nothing
        :rtype: None
        """
        try:
            async with self._local:
                if self._applied.get(scope) is not got.held or not got.current.supersedes(got.held):
                    return
                await self._commit_fetched(got.replacements, [scope], epochs={scope: got.current.epoch})
                self._applied = _updated(self._applied, {scope: got.current})
        except _Lost as exc:
            await self._note_unread(scope, got.current, exc, failed, mark_behind=mark_behind)
            return
        self._notify()
        self._record_change(scope, got.current, got.started)

    async def _drop_locally(self, scopes: Sequence[str], *, unless_seen: bool = False) -> None:
        """remove scopes from this L1, in one transaction.

        :param scopes: the scopes
        :ptype scopes: Sequence[str]
        :param unless_seen: keep a scope whose pointer has been seen by the time the L1's lock is taken
        :ptype unless_seen: bool
        :return: nothing
        :rtype: None
        """
        async with self._local:
            if unless_seen:
                scopes = [scope for scope in scopes if scope not in self._seen]
                if not scopes:
                    return
            # forgotten BEFORE the drop commits: a scope listed by applied_epochs() is one a read opened
            # after holds, and a dropped scope's rows may already be gone (under-listing is the safe side)
            self._applied = _updated(self._applied, drop=scopes)
            self._rows = _updated({table: _updated(held, drop=scopes) for table, held in self._rows.items()})
            await self._commit(
                [
                    PartitionReplacement(table=t.name, column=t.scope_column, value=scope, rows=[])
                    for scope in scopes
                    for t in self._tables
                ],
                list(scopes),
                epochs=dict.fromkeys(scopes),
                count=False,
            )
        self._notify()
        log.info("scoped snapshot %s: scopes dropped", self._name, extra={"extra_data": {"scopes": list(scopes)}})

    def _record_change(self, scope: str, pointer: _Pointer, started: float) -> None:
        """record and log one applied scope change, once its rows are counted.

        :return: nothing
        :rtype: None
        """
        rows = sum(held.get(scope, 0) for held in self._rows.values())
        change = ScopeChange(
            scope=scope,
            epoch=pointer.epoch,
            rows=rows,
            seconds=round(time.perf_counter() - started, 3),
            applied_at=datetime.now(UTC),
        )
        self._progress = replace(self._progress, last_change=change)
        log.info(
            "scoped snapshot %s: scope applied",
            self._name,
            extra={"extra_data": {"scope": scope, "epoch": pointer.epoch, "rows": rows, "seconds": change.seconds}},
        )

    # ------------------------------------------------------------------
    # publishing
    # ------------------------------------------------------------------

    async def publish(
        self,
        scope: str,
        epoch: int,
        rows: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        added: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        after: int | None = None,
    ) -> bool:
        """make the scope's content at ``epoch`` this L1's, its chunks', and then its pointer's.

        Called by the writer once the scope's change is committed in L3. Every table is named once:
        in ``rows`` with every row the scope now holds in it, or, for an append-only table, in
        ``added`` with only the rows the change ADDED. The other replicas apply it when they see the
        pointer move. A publish below the scope's current epoch changes nothing anywhere. Chunks of
        the scope's older epochs are retired.

        **Whole rows only** (no ``added``): every table is written whole and the pointer moves by the
        usual compare-and-set, retried against racing writers and never to a lower epoch. This is the
        same for a snapshot with append-only tables as for one without; an append-only table given
        whole rows gets a fresh base.

        **Added rows.** They are a piece after the chain the scope's pointer holds at ``after``, the
        epoch the writer saw before its change, so the pointer moves only from exactly that entry
        (the whole tables of the same publish move with it, as one pointer). The publish first
        applies the scope's pointer here if the watch lagged. When the pointer is not at ``after``
        on the chain this L1 holds, or a racing writer moves it first, nothing moves, nothing here
        changes, and the scope is shown behind (:meth:`status`) and rebuilt from L3, which holds
        the change, at the next pass. :meth:`catch_up_from_l3` does the same for a writer that
        calls it.

        :param scope: the scope; never ``None``
        :ptype scope: str
        :param epoch: its epoch after the change
        :ptype epoch: int
        :param rows: table name -> every row of the scope in that table
        :ptype rows: Mapping[str, Sequence[Mapping[str, Any]]]
        :param added: append-only table name -> the rows the change added to the scope
        :ptype added: Mapping[str, Sequence[Mapping[str, Any]]] | None
        :param after: the scope's epoch before the change, which the added rows follow (0 for a scope
            new to the snapshot); required with ``added``
        :ptype after: int | None
        :return: whether the scope's pointer took this publish; ``False`` when it was stale, or left
            for the rebuild from L3 (a racing writer's whole publish that moved it past this one also
            reads ``False``)
        :rtype: bool
        :raises ValueError: when the scope is ``None``; when a table is named in neither or both of
            ``rows`` and ``added``, ``added`` names a table not declared append-only, or ``after`` is
            missing with it; or when an added row's key is one the scope already holds here (rows are
            only ever added: a caller giving every row of an append-only table is told so)
        """
        if scope is None:
            raise ValueError("a scoped snapshot holds no rows without a scope; every row's scope column is set")
        added = dict(added or {})
        self._check_named(scope, rows, added, every=True)
        if added and after is None:
            raise ValueError(f"a publish of scope {scope!r} with added rows names the epoch they follow (after=)")
        if not added and after is not None:
            raise ValueError(f"after= names the epoch added rows follow; the publish of scope {scope!r} adds none")
        started = time.perf_counter()
        if added:
            await self._catch_up_scope(scope)
        async with self._local:
            if added:
                pointer, landed = await self._publish_added_locked(scope, epoch, rows, added, int(after or 0))
            else:
                pointer, landed = await self._publish_locked(scope, epoch, self._whole(scope, rows))
        if pointer is not None:
            self._record_change(scope, pointer, started)
            await self._update_index(add={scope})
            await self._sweeper.retire_older(scope, epoch)
            if self._append:
                await self._compact(scope)
        return landed

    def _check_named(
        self,
        scope: str,
        rows: Mapping[str, Any],
        added: Mapping[str, Any],
        *,
        every: bool,
    ) -> None:
        """refuse tables a write cannot name: unknown, named twice, added to but not append-only, or missing.

        :param every: whether every table must be named (a publish) or any may be left out (a stage)
        :ptype every: bool
        :return: nothing
        :rtype: None
        :raises ValueError: naming what is wrong
        """
        names = {t.name for t in self._tables}
        unknown = sorted((set(rows) | set(added)) - names)
        if unknown:
            raise ValueError(f"a write of scope {scope!r} names tables the snapshot does not hold: {unknown}")
        twice = sorted(set(rows) & set(added))
        if twice:
            raise ValueError(f"a write of scope {scope!r} names {twice} in both rows and added")
        whole = sorted(set(added) - self._append)
        if whole:
            raise ValueError(
                f"a write of scope {scope!r} gives added rows for {whole}, which are not append-only: give every row"
            )
        missing = sorted(names - set(rows) - set(added)) if every else []
        if missing:
            raise ValueError(f"a publish of scope {scope!r} gives no rows for {missing}")

    async def _refuse_held(self, scope: str, added: Mapping[str, Any]) -> None:
        """refuse added rows whose keys the scope already holds here: they are no addition.

        :param added: append-only table -> its added rows, as this L1 types them
        :ptype added: Mapping[str, pyarrow.Table]
        :return: nothing
        :rtype: None
        :raises ValueError: naming the table and how many rows were held already
        """
        for table in self._tables:
            if table.name not in added:
                continue
            held = await asyncio.to_thread(
                self._backend.held_keys, table.name, table.scope_column, scope, added[table.name], key=table.key
            )
            if held:
                raise ValueError(
                    f"{held} of the rows added to scope {scope!r} of {table.name} are held already; an append-only "
                    "table is given only the rows a write added (every row goes in rows=, as a fresh base)"
                )

    async def _catch_up_scope(self, scope: str) -> None:
        """apply a scope's pointer here if it is ahead of what this L1 holds: a watch that lagged.

        :param scope: the scope
        :ptype scope: str
        :return: nothing
        :rtype: None
        """
        raw = await self._pointers_bucket.get(key=self._layout.pointer_key(scope))
        current = None if raw is None else _Pointer.decode(raw)
        if current is not None and self._ahead(current, self._applied.get(scope)):
            self._note_seen(current)
            await self._apply_pointers({scope: current})

    def _leave_for_l3(self, scope: str, why: str) -> None:
        """show a scope behind, so the next pass rebuilds it from L3: a change committed there that no
        pointer here can carry.

        :param scope: the scope
        :ptype scope: str
        :param why: why, for the status and the log
        :ptype why: str
        :return: nothing
        :rtype: None
        """
        log.warning("scoped snapshot %s: scope %r: %s; it is rebuilt from L3", self._name, scope, why)
        self._behind = _updated(self._behind, {scope: (_BEHIND_REBUILD_AFTER, f"{why}; rebuilt from L3")})
        self._changed.set()

    async def _publish_added_locked(
        self,
        scope: str,
        epoch: int,
        rows: Mapping[str, Sequence[Mapping[str, Any]]],
        added: Mapping[str, Sequence[Mapping[str, Any]]],
        after: int,
    ) -> tuple[_Pointer | None, bool]:
        """:meth:`publish` with added rows, under the L1 lock.

        The chunks are written first and the pointer moved from exactly the entry at ``after``; only
        once it has moved does this L1 take the change, so ``applied_epoch`` and the last change only
        ever name an epoch the scope's pointer took.

        :return: the pointer written and landed (``None`` when nothing moved), and whether it landed
        :rtype: tuple[_Pointer | None, bool]
        :raises ValueError: when an added row is held already
        """
        entry = await self._pointers_bucket.get_entry(key=self._layout.pointer_key(scope))
        current = None if entry is None else _Pointer.decode(entry[0])
        applied = self._applied.get(scope)
        if (current is not None and current.epoch >= epoch) or (applied is not None and applied.epoch >= epoch):
            log.warning(
                "scoped snapshot %s: a publish of scope %r at epoch %d is not past the scope's epoch; nothing moved",
                self._name,
                scope,
                epoch,
            )
            if current is not None:
                self._note_seen(current)
            return None, False
        new = current is None and applied is None and after == 0
        on_chain = (
            current is not None
            and applied is not None
            and self._loadable(current)
            and current.epoch == after == applied.epoch
            and (current.objects, current.pieces) == (applied.objects, applied.pieces)
        )
        if not (new or on_chain):
            self._leave_for_l3(scope, f"its added rows follow epoch {after}, which is not the chain its pointer holds")
            return None, False
        typed = {
            t.name: await self._export(t, scope, added[t.name] if t.name in added else rows[t.name])
            for t in self._tables
        }
        await self._refuse_held(scope, {name: arrow for name, arrow in typed.items() if name in added})
        # a new scope's added rows are all its rows: its first piece is its base
        writes = {t.name: _Write(typed[t.name], piece=t.name in added and current is not None) for t in self._tables}
        pointer, held = await self._write_scope(
            scope, epoch, writes, base=current, move_from=None if entry is None else entry[1]
        )
        if held is None:
            await self._sweeper.retire_names_if_unserved(
                [name for name in pointer.names() if current is None or name not in current.names()],
                what=f"a publish of {scope!r} that lost its race",
            )
            self._leave_for_l3(scope, f"a writer moved its pointer while epoch {epoch}'s added rows were written")
            return None, False
        local = [
            PartitionReplacement(
                table=t.name, column=t.scope_column, value=scope, arrow=typed[t.name], append=t.name in added
            )
            for t in self._tables
        ]
        try:
            await self._commit_fetched(local, [scope], epochs={scope: epoch})
        except _Lost as exc:
            # the pointer took the change; this L1 applies it from the pointer, as any replica does
            log.warning("scoped snapshot %s: scope %r: %s; applied from its pointer instead", self._name, scope, exc)
            self._changed.set()
            return pointer, True
        self._applied = _updated(self._applied, {scope: pointer})
        self._notify()
        return pointer, True

    def _whole(self, scope: str, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[PartitionReplacement]:
        """the replacements making ``scope`` hold ``rows`` in every table.

        :param scope: the scope
        :ptype scope: str
        :param rows: table name -> every row of the scope in that table
        :ptype rows: Mapping[str, Sequence[Mapping[str, Any]]]
        :return: one replacement per table
        :rtype: list[PartitionReplacement]
        """
        return [
            PartitionReplacement(table=t.name, column=t.scope_column, value=scope, rows=rows[t.name], primary_key=t.key)
            for t in self._tables
        ]

    async def _publish_locked(
        self, scope: str, epoch: int, local: Sequence[PartitionReplacement]
    ) -> tuple[_Pointer | None, bool]:
        """the body of a whole :meth:`publish` (and of a rebuild's), under the L1 lock: refuse a stale
        epoch, replace locally, write the chunks, move the pointer by compare-and-set.

        :param local: the scope's whole content, one replacement per table: rows a writer gave, or a
            rebuild's Arrow tables read from L3
        :ptype local: Sequence[PartitionReplacement]
        :return: the pointer written (``None`` when the scope is already at a later epoch), and whether
            the scope's pointer took it
        :rtype: tuple[_Pointer | None, bool]
        """
        held = await self._pointers_bucket.get(key=self._layout.pointer_key(scope))
        current = None if held is None else _Pointer.decode(held)
        applied = self._applied.get(scope)
        stale = (current is not None and current.epoch > epoch) or (applied is not None and applied.epoch > epoch)
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
            return None, False
        await self._commit(local, [scope], epochs={scope: epoch})

        def written(pointer: _Pointer) -> None:
            # the change recorded is this replica's own: a racing writer's later pointer is not what
            # this L1 holds
            self._applied = _updated(self._applied, {scope: pointer})

        try:
            writes = {
                t.name: _Write(
                    await asyncio.to_thread(
                        self._backend.export_partition, t.name, t.scope_column, scope, order_by=t.key
                    )
                )
                for t in self._tables
            }
            pointer, moved = await self._write_scope(
                scope, epoch, writes, base=None, move_from=_ANY_ENTRY, written=written
            )
        finally:
            # after `_applied` moves, as every other commit path does, so a listener reading
            # applied_epoch sees the epoch of what it reads; on a failed chunk write, still told
            self._notify()
        return pointer, moved == pointer

    async def _write_scope(
        self,
        scope: str,
        epoch: int,
        writes: Mapping[str, _Write],
        *,
        base: _Pointer | None,
        move_from: int | None | object,
        written: Callable[[_Pointer], None] | None = None,
    ) -> tuple[_Pointer, _Pointer | None]:
        """write a scope's chunks at ``epoch`` and move its pointer onto them: the one publish sequence.

        Every write (a publish, a rebuild, a fold) runs it: a write claim on the epoch, held from before
        the first chunk until the pointer names them, so no sweep takes one in between; each table's
        chunk written whole, or appended to ``base``'s chain as a piece, or carried from ``base``
        when ``writes`` leaves it out; every chunk the pointer names checked to be there; and the
        pointer moved, by the retried compare-and-set (``move_from`` ``_ANY_ENTRY``) or only from
        exactly the entry at revision ``move_from`` (``None``: only if there is none).

        :param writes: table -> its rows, and whether they are a piece after ``base``'s chain
        :ptype writes: Mapping[str, _Write]
        :param base: the pointer the pieces follow and the left-out tables are carried from
        :ptype base: _Pointer | None
        :param move_from: how the pointer moves
        :ptype move_from: int | None | object
        :param written: told the pointer once every chunk is written, before it moves
        :ptype written: Callable[[_Pointer], None] | None
        :return: the pointer written, and the one the scope holds after: it, a racing writer's later
            one (the retried move), or ``None`` (an exact move that lost)
        :rtype: tuple[_Pointer, _Pointer | None]
        :raises RuntimeError: when the claim was lost, or a chunk it names is gone
        """
        claim = await self._write_claims.hold(epoch)
        try:
            objects: dict[str, str] = {}
            counts: dict[str, int] = {}
            pieces: dict[str, tuple[_Piece, ...]] = {}
            compacted: dict[str, float] = {}
            reused: list[str] = []
            for table in self._tables:
                write = writes.get(table.name)
                if write is None or write.piece:
                    if base is None:
                        raise RuntimeError(f"scope {scope!r}: {table.name} has no chain to carry or append to")
                    objects[table.name], counts[table.name] = base.objects[table.name], base.rows[table.name]
                    pieces[table.name] = base.pieces.get(table.name, ())
                    if table.name in base.compacted_at:
                        compacted[table.name] = base.compacted_at[table.name]
                if write is None:
                    continue
                _still_held(claim, f"writing scope {scope!r}'s chunk of {table.name}")
                name, size = await self._put_chunk(scope, epoch, table, write.arrow, reused=reused, piece=write.piece)
                if write.piece:
                    pieces[table.name] = (*pieces[table.name], _Piece(name, write.arrow.num_rows, epoch, size))
                else:
                    objects[table.name], counts[table.name] = name, write.arrow.num_rows
                    if table.append_only:
                        # a whole write forms an append-only table's base afresh
                        compacted[table.name] = time.time()
            pointer = _Pointer(
                scope=scope,
                epoch=epoch,
                objects=objects,
                rows=counts,
                schema=self._schema,
                pieces={table: chain for table, chain in pieces.items() if chain},
                compacted_at=compacted,
            )
            if written is not None:
                written(pointer)
            # every chunk the pointer will name must still be there: one found already written was
            # judged by a sweep that may have read the claims before this one was taken, and one
            # written here may have been swept while this replica was cut off from NATS past its
            # claim's lifetime (a lapse it cannot see until it reconnects)
            if not await self._exist(pointer.names()):
                raise RuntimeError(
                    f"scope {scope!r}: a chunk of epoch {epoch} was deleted before its pointer moved "
                    f"({sorted(pointer.names())}, reused: {reused}); no pointer is moved onto a missing chunk"
                )
            _still_held(claim, f"moving scope {scope!r}'s pointer to epoch {epoch}")
            if move_from is _ANY_ENTRY:
                held: _Pointer | None = await self._move_pointer(pointer)
            else:
                held = pointer if await self._move_exactly(pointer, move_from) else None
        finally:
            await claim.release()
        return pointer, held

    async def _put_chunk(
        self,
        scope: str,
        epoch: int,
        table: SnapshotTable,
        arrow: Any,
        *,
        reused: list[str] | None = None,
        piece: bool = False,
    ) -> tuple[str, int]:
        """write one table's chunk of a scope at an epoch; the one place a chunk is written.

        The caller holds a write claim on ``epoch`` (:meth:`_WriteClaims.hold`). A store refused as full
        is swept by the one rule and the write tried once more; what that frees is what no pointer
        serves and no live write claims. A torn write's pieces (an object's chunks land before its
        metadata) are not listed and wait for the declarer's orphan purge, so a store full of them
        recovers only once that purge has run.

        :param scope: the scope
        :ptype scope: str
        :param epoch: the epoch
        :ptype epoch: int
        :param table: the table
        :ptype table: SnapshotTable
        :param arrow: the scope's rows of the table
        :ptype arrow: pyarrow.Table
        :param reused: where a chunk found already written is recorded, for the caller to check it is
            still there before a pointer names it
        :ptype reused: list[str] | None
        :param piece: whether it is a piece of an append-only table (named under the pieces' prefix)
        :ptype piece: bool
        :return: the chunk's name and its bytes
        :rtype: tuple[str, int]
        """
        data = await asyncio.to_thread(encode_chunk, arrow)
        name = (self._layout.piece_name if piece else self._layout.object_name)(scope, epoch, table.name)
        try:
            try:
                await self._store.put(name, data)
            except ObjectStoreFullError:
                retired = await self._sweeper.retire_unreferenced()
                log.warning(
                    "scoped snapshot %s: the object store is full writing %r; %s, trying once more",
                    self._name,
                    name,
                    "this replica retires nothing (no retire wired)"
                    if self._sweeper.retires is False
                    else f"retired {retired} chunks no pointer serves and no write claims",
                )
                await self._store.put(name, data)
        except ObjectExistsError:
            # a replica rebuilding the same scope at the same epoch under the same columns wrote the same rows
            if reused is not None:
                reused.append(name)
        _headway(f"wrote chunk {name}")
        return name, len(data)

    async def stage(
        self,
        scope: str,
        epoch: int,
        rows: Mapping[str, Sequence[Mapping[str, Any]] | Any],
        *,
        added: Mapping[str, Sequence[Mapping[str, Any]] | Any] | None = None,
    ) -> StagedScope:
        """write a scope's chunks under ``epoch`` without moving its pointer or touching this L1.

        A table's rows are dict rows, or one Arrow table (an export read straight into Arrow): either is
        typed by this L1 alike (:meth:`~threetears.core.cache.duckdb.DuckDBBackend.storage_arrow`), refused
        alike, and staged as the same chunk bytes.

        For a writer that knows a change's epoch before it commits (the write's version is the
        epoch, :mod:`~threetears.core.collections.scope_epochs`): it stages each scope as soon as
        its rows are written to L3, keeping only the compressed chunks rather than every row, and
        after its commit moves every pointer at once (:meth:`publish_staged`). Nothing reads a
        staged chunk until its pointer moves, so a write that never commits shows nothing. Each stage
        holds a write claim on its epoch until :meth:`publish_staged` takes it, so no sweep touches it
        meanwhile (:meth:`_Sweeper.deletable`). A write that will not commit gives it back
        (:meth:`discard_staged`); a write whose writer died loses its claim with its writer, and its
        stages go at the next sweep (:meth:`discard_epoch`, a publish of the scope, a full store).

        :param scope: the scope; never ``None``
        :ptype scope: str
        :param epoch: the epoch the write will commit
        :ptype epoch: int
        :param rows: table name -> every row the scope holds in that table after the write, as dict rows
            or one ``pyarrow.Table``; a table left out of both ``rows`` and ``added`` is one the write did
            not change, kept from the scope's current chunk
        :ptype rows: Mapping[str, Sequence[Mapping[str, Any]] | pyarrow.Table]
        :param added: append-only table name -> the rows the write ADDED to the scope, as dict rows or one
            ``pyarrow.Table``, staged as a piece that :meth:`publish_staged` appends to the chain the
            pointer holds at ``carry_at``
        :ptype added: Mapping[str, Sequence[Mapping[str, Any]] | pyarrow.Table] | None
        :return: the staged chunks
        :rtype: StagedScope
        :raises ValueError: when the scope is ``None``; a table is not one of the snapshot's, is named
            in both, or is given added rows but is not append-only; or an added row's key is one this
            replica already holds for the scope (rows are only ever added: a caller giving every row of
            an append-only table as added rows is told so, before anything is written)
        """
        if scope is None:
            raise ValueError("a scoped snapshot holds no rows without a scope; every row's scope column is set")
        added = dict(added or {})
        self._check_named(scope, rows, added, every=False)
        typed = {
            t.name: await self._export(t, scope, added[t.name] if t.name in added else rows[t.name])
            for t in self._tables
            if t.name in rows or t.name in added
        }
        await self._refuse_held(scope, {name: arrow for name, arrow in typed.items() if name in added})
        objects: dict[str, str] = {}
        counts: dict[str, int] = {}
        sizes: dict[str, int] = {}
        # held until publish_staged moves (or skips) this stage, or discard_staged gives it back
        claim = await self._write_claims.hold(epoch)
        try:
            for table in self._tables:
                if table.name in typed:
                    arrow = typed[table.name]
                    _still_held(claim, f"staging scope {scope!r}'s chunk of {table.name}")
                    objects[table.name], sizes[table.name] = await self._put_chunk(
                        scope, epoch, table, arrow, piece=table.name in added
                    )
                    counts[table.name] = arrow.num_rows
        except BaseException:
            await claim.release()
            await self._sweeper.retire_names_if_unserved(list(objects.values()), what=f"a failed stage of {scope!r}")
            raise
        return StagedScope(
            scope=scope,
            epoch=epoch,
            objects=objects,
            rows=counts,
            claim=claim,
            sizes=sizes,
            pieces=frozenset(added),
        )

    async def stage_from_l3(self, scope: str, epoch: int) -> StagedScope:
        """stage every table of a scope whole as L3 holds it now, under ``epoch``: :meth:`stage` for a scope the
        writer cannot build from this L1.

        For the writer of the write ``epoch`` names, once its rows for the scope are in L3: one write at a time
        holds the epochs (:mod:`~threetears.core.collections.scope_epochs`), so the scope's rows in L3 are then
        exactly its rows at ``epoch``. A writer that builds a scope's chunks from this replica's copy can do so only
        while the copy holds the scope at the epoch before the write; a scope the copy is behind on is staged here
        instead, so it moves with the write rather than waiting for a catch-up that writes back to back keep from
        running. Every table is staged whole, so the stage carries nothing from the scope's current chunks.

        :param scope: the scope
        :ptype scope: str
        :param epoch: the epoch the write will commit
        :ptype epoch: int
        :return: the staged chunks
        :rtype: StagedScope
        """
        read = await self._read_scope(scope)
        rows = {part.table: part.arrow if part.arrow is not None else [] for part in read}
        return await self.stage(scope, epoch, rows)

    async def _export(self, table: SnapshotTable, scope: str, rows: Sequence[Mapping[str, Any]] | Any) -> Any:
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

    async def discard_staged(self, staged: Sequence[StagedScope]) -> None:
        """give back what a write that will never commit staged: its claims released, its chunks retired.

        For the writer whose write failed or was abandoned before its commit. The chunks go by the
        one rule, so one another write claims at the same epoch, or a pointer names, stays. Never
        raises: a retire that fails leaves them unclaimed, for the next sweep.

        :param staged: the scopes the write staged
        :ptype staged: Sequence[StagedScope]
        :return: nothing
        :rtype: None
        """
        for scope in staged:
            if scope.claim is not None:
                await scope.claim.release()
        names = [name for scope in staged for name in scope.objects.values()]
        await self._sweeper.retire_names_if_unserved(names, what="a discarded write's stages")

    async def discard_epoch(self, epoch: int) -> None:
        """retire every chunk of a write that died before it committed, by its epoch.

        For a writer taking over from one that died mid-write: the dead write's version (its epoch)
        never commits, so nothing will ever serve what it staged. By the one rule: chunks a pointer
        names, or a live write still claims, are kept. Never raises.

        :param epoch: the dead write's epoch
        :ptype epoch: int
        :return: nothing
        :rtype: None
        """
        await self._sweeper.retire_epoch(epoch)

    async def publish_staged(
        self, staged: Sequence[StagedScope], *, carry_at: Mapping[str, int], whole: bool = False
    ) -> tuple[list[str], list[str]]:
        """move the pointers of staged scopes, once the write that staged them has committed in L3.

        A table a stage left out keeps the scope's current chunk, but only when the scope's pointer
        is at ``carry_at[scope]`` -- the epoch the writer saw before its write, so the carried chunk
        holds what L3 held then. A scope with no pointer and ``carry_at`` 0 is new: its missing
        tables are empty, when the snapshot is known whole (``whole``, or a whole index that does not
        name it). A scope whose pointer is elsewhere below the stage's epoch (behind the writer's, a
        stage of an earlier write skipped) has the tables it left out, and those it appended to, read
        whole from L3, only while L3 holds it exactly at the stage's epoch (:meth:`_filled_from_l3`).
        Any other scope is skipped and left to :meth:`catch_up_from_l3`, which republishes it from L3
        because its L3 epoch is now ahead of its pointer. Every replica applies the moved scopes from
        its pointer watch, this one included.

        A stage whose write claim is lost, checked before the scope is completed and again the moment
        before its pointer moves, is skipped and left to the catch-up. Every way out, a raise part way
        through included, indexes the scopes whose pointers already moved, lets every claim go and
        retires what the moved scopes no longer serve, so no moved scope is left out of the index.

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
        finished = False
        try:
            try:
                # every scope is completed, checked and moved side by side, ``l2_concurrency`` at once: each
                # is several store round trips, and one after another they held a burst's last pointer back
                # by all of them. Each records its outcome the moment it has one, so a pointer that moved is
                # indexed below on every way out; every scope runs to its end even when another raises, and
                # the first failure is raised after
                gate = asyncio.Semaphore(max(1, self._l2_concurrency))
                outcomes = {"moved": moved, "skipped": skipped, "superseded": superseded}

                async def publish_one(scope: StagedScope) -> None:
                    async with gate:
                        outcome = await self._publish_one_staged(scope, int(carry_at.get(scope.scope, 0)), whole=whole)
                    outcomes[outcome].append(scope.scope)

                results = await asyncio.gather(*(publish_one(scope) for scope in staged), return_exceptions=True)
                _raise_first(results, log=log, what=f"scoped snapshot {self._name}: a staged scope's publish failed")
                # in the order the scopes were staged, as one after another they were
                order = {scope.scope: index for index, scope in enumerate(staged)}
                for listed in (moved, skipped, superseded):
                    listed.sort(key=order.__getitem__)
                finished = True
                if superseded:
                    log.info(
                        "scoped snapshot %s: %d staged scopes were already at a later epoch; nothing moved for them",
                        self._name,
                        len(superseded),
                        extra={"extra_data": {"scopes": superseded}},
                    )
            finally:
                # every way out, raised or returned: a scope whose pointer moved is indexed, so the
                # replicas serve it (an index is whole only from a write of everything that finished
                # with nothing skipped)
                if moved:
                    await self._update_index(add=moved, whole=whole and finished and not skipped)
        finally:
            # the stages are named by their pointers now, or will never be by this call: the claims are
            # let go on every way out, and a skipped or superseded stage serves nothing and goes by the
            # rule (which judges the pointers as KV holds them, so a stage whose move landed unseen is
            # spared); the scopes that moved retire their older epochs
            for scope in staged:
                if scope.claim is not None:
                    await scope.claim.release()
            unserved = [name for scope in staged if scope.scope not in moved for name in scope.objects.values()]
            await self._sweeper.retire_names_if_unserved(unserved, what="stages left unpublished")
            await self._sweeper.retire_older_many(
                {scope.scope: scope.epoch for scope in staged if scope.scope in moved}
            )
            self._changed.set()
        if self._append:
            await _bounded_each(moved, self._compact, at_once=_SWEEPS_AT_ONCE)
        if skipped:
            log.warning(
                "scoped snapshot %s: %d staged scopes left for the catch-up from L3",
                self._name,
                len(skipped),
                extra={"extra_data": {"scopes": skipped}},
            )
        return moved, skipped

    async def _publish_one_staged(self, scope: StagedScope, carry_at: int, *, whole: bool) -> str:
        """complete, check and move one staged scope's pointer: what :meth:`publish_staged` does for each.

        :param scope: the stage
        :ptype scope: StagedScope
        :param carry_at: its epoch before the write
        :ptype carry_at: int
        :param whole: whether the staged scopes are every scope L3 holds
        :ptype whole: bool
        :return: ``"moved"``, ``"skipped"`` (left to the catch-up) or ``"superseded"`` (a racing writer is ahead)
        :rtype: str
        """
        if self._claim_lost(scope):
            return "skipped"
        completed = await self._completed(scope, carry_at, whole=whole)
        if completed is None:
            return "skipped"
        pointer, observed = completed
        if not await self._exist(pointer.names()):
            # never a pointer onto a missing chunk: the scope is left to the catch-up, which
            # republishes it from L3 (it stages it again)
            log.error(
                "scoped snapshot %s: a chunk staged for scope %r at epoch %d is gone; its pointer is "
                "not moved, and the catch-up republishes it from L3",
                self._name,
                scope.scope,
                scope.epoch,
            )
            return "skipped"
        # the claim may have lapsed during the round trips above: checked again, the moment
        # before the move, and a lost one skipped like any other
        if self._claim_lost(scope):
            return "skipped"
        if observed is _ANY_ENTRY:
            held = await self._move_pointer(pointer)
            return "moved" if held == pointer else "superseded"
        if await self._move_exactly(pointer, observed):
            return "moved"
        # what it carried, or filled empty, was judged against an entry that has moved since
        return "skipped"

    async def _compact(self, scope: str) -> bool:
        """fold a scope's pieces into a new base where they are past the threshold; never raises.

        Each table whose pieces number ``compact_pieces`` or reach ``compact_bytes`` gets a new base at
        the pointer's epoch: its base and pieces concatenated in order, not re-sorted, so a replica
        loading it holds its rows in the order one that appended them does. Written under a write
        claim on that epoch, and the pointer moves by compare-and-set from exactly the entry folded:
        a write that moved the scope meanwhile wins, and the next writer folds. The folded pieces go
        by the sweeper's one rule. A replica holding the scope at that epoch adopts the folded
        pointer without fetching it.

        :param scope: the scope a write just moved
        :ptype scope: str
        :return: whether its pointer was folded
        :rtype: bool
        """
        folded = False
        written: list[str] = []
        try:
            entry = await self._pointers_bucket.get_entry(key=self._layout.pointer_key(scope))
            current = None if entry is None else _Pointer.decode(entry[0])
            due = (
                []
                if current is None or not self._loadable(current)
                else [t for t in self._tables if self._due(current.pieces.get(t.name, ()))]
            )
            if current is None or entry is None or not due:
                return folded
            writes: dict[str, _Write] = {}
            for table in due:
                parts = [await self._read_chunk(current.objects[table.name], current.rows[table.name])]
                parts += [await self._read_chunk(p.name, p.rows) for p in current.pieces[table.name]]
                writes[table.name] = _Write(_concatenated(parts))
            base, held = await self._write_scope(scope, current.epoch, writes, base=current, move_from=entry[1])
            written = [base.objects[t.name] for t in due]
            folded = held is not None
            if folded:
                await self._sweeper.retire_older(scope, current.epoch + 1)
                log.info(
                    "scoped snapshot %s: scope %r's pieces folded into its base",
                    self._name,
                    scope,
                    extra={"extra_data": {"scope": scope, "epoch": current.epoch, "tables": [t.name for t in due]}},
                )
            else:
                await self._sweeper.retire_names_if_unserved(written, what=f"a fold of {scope!r} that lost its race")
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- a fold is housekeeping: the scope is served from its base and pieces until the next writer folds it, so a failure must not fail the publish that already moved it
            contain(exc, log=log, what=f"scoped snapshot {self._name}: folding scope {scope!r}'s pieces failed: {exc}")
        return folded

    def _due(self, pieces: Sequence[_Piece]) -> bool:
        """whether a table's pieces in one scope are past the threshold, so they fold into its base.

        :param pieces: the pieces
        :ptype pieces: Sequence[_Piece]
        :return: True when there are ``compact_pieces`` of them, or they reach ``compact_bytes``
        :rtype: bool
        """
        return bool(pieces) and (
            len(pieces) >= self._compact_pieces or sum(p.size for p in pieces) >= self._compact_bytes
        )

    def _claim_lost(self, scope: StagedScope) -> bool:
        """whether a stage's write claim is lost, logged when it is: its pointer must not move.

        :param scope: the stage
        :ptype scope: StagedScope
        :return: True when the claim was lost
        :rtype: bool
        """
        if scope.claim is None or scope.claim.held:
            return False
        # what it staged may already be gone to a sweep: never written to, never pointed at
        log.error(
            "scoped snapshot %s: the write claim of scope %r at epoch %d was lost; its pointer is not "
            "moved, and the catch-up republishes it from L3",
            self._name,
            scope.scope,
            scope.epoch,
        )
        return True

    async def _completed(
        self, staged: StagedScope, carry_at: int, *, whole: bool
    ) -> tuple[_Pointer, int | None | object] | None:
        """the pointer a staged scope moves to, every table named, and the pointer entry it was judged
        against; ``None`` when it cannot be completed.

        A stage that named every table with whole chunks moves by the usual compare-and-set
        (``_ANY_ENTRY``). One that carried a chunk, filled a table empty, or appended a piece is right
        only against the entry it read, so it moves only from that entry's revision (``None``: only if
        there is still no pointer). A piece is appended to the chain the pointer holds at ``carry_at``,
        the epoch the writer saw, the rows it added being the rows past that chain; a scope new to the
        snapshot takes its first piece as its base.

        :return: the pointer and the revision it must move from
        :rtype: tuple[_Pointer, int | None | object] | None
        """
        appended = [t for t in self._tables if t.name in staged.pieces]
        missing = [t for t in self._tables if t.name not in staged.objects]
        # filled as tables are completed: the pointer holds these very dicts
        objects = {t: name for t, name in staged.objects.items() if t not in staged.pieces}
        counts = {t: rows for t, rows in staged.rows.items() if t not in staged.pieces}
        pieces: dict[str, tuple[_Piece, ...]] = {}
        compacted: dict[str, float] = {}
        pointer = _Pointer(
            scope=staged.scope,
            epoch=staged.epoch,
            objects=objects,
            rows=counts,
            schema=self._schema,
            pieces=pieces,
            compacted_at=compacted,
        )
        result: tuple[_Pointer, int | None | object] | None = (pointer, _ANY_ENTRY)
        if missing or appended:
            entry = await self._pointers_bucket.get_entry(key=self._layout.pointer_key(staged.scope))
            current = None if entry is None else _Pointer.decode(entry[0])
            result = (pointer, None if entry is None else entry[1])
            index_whole = self._index is not None and not self._index_partial
            if (
                current is not None
                and current.epoch == carry_at
                # a piece follows the chain it was written after: at a later epoch, never the same one
                and (staged.epoch > carry_at or not appended)
                and current.schema == self._schema
                and all(t.name in current.objects for t in (*missing, *appended))
            ):
                carried: list[str] = []
                for table in (*missing, *appended):
                    # a table is carried with its chain: its base, and its pieces after it
                    base, prior = current.chain(table.name)
                    objects[table.name] = current.objects[table.name]
                    counts[table.name] = current.rows[table.name]
                    if table in appended:
                        added = _Piece(
                            name=staged.objects[table.name],
                            rows=staged.rows[table.name],
                            epoch=staged.epoch,
                            size=staged.sizes.get(table.name, 0),
                        )
                        pieces[table.name] = (*prior, added)
                    elif prior:
                        pieces[table.name] = prior
                    if table.name in current.compacted_at:
                        compacted[table.name] = current.compacted_at[table.name]
                    carried += [objects[table.name], *(piece.name for piece in prior)]
                # the carried chunks must still be there: a sweep judged by the same rule spares them
                if not await self._exist(carried):
                    result = None
            elif (
                current is None
                and carry_at == 0
                and (whole or (index_whole and staged.scope not in (self._index or ())))
            ):
                for table in missing:
                    arrow = await self._export(table, staged.scope, [])
                    objects[table.name], _ = await self._put_chunk(staged.scope, staged.epoch, table, arrow)
                    counts[table.name] = 0
                for table in appended:
                    # a new scope's added rows are all its rows: its first piece is its base
                    objects[table.name] = staged.objects[table.name]
                    counts[table.name] = staged.rows[table.name]
                    compacted[table.name] = time.time()
            elif current is None or current.epoch < staged.epoch:
                # nothing can be carried (the pointer is not at the epoch the writer saw -- behind it, a stage of an
                # earlier write skipped -- or holds other columns, or there is none): left to the catch-up, the scope
                # stayed behind for as long as writes ran back to back, every later partial write skipped the same
                # way (cobalt prod, 2026-10-31). Never onto a pointer at the stage's epoch or past it
                filled = await self._filled_from_l3(staged, [*missing, *appended], objects, counts, compacted)
                if filled:
                    pieces.clear()
                result = result if filled else None
            else:
                result = None
        return result

    async def _filled_from_l3(
        self,
        staged: StagedScope,
        tables: Sequence[SnapshotTable],
        objects: dict[str, str],
        counts: dict[str, int],
        compacted: dict[str, float],
    ) -> bool:
        """complete a stage whose pointer cannot carry: the tables it left out, and those it appended to, read whole
        from L3 and staged under its epoch, only while L3 holds the scope exactly at that epoch.

        L3 holds a scope at the stage's epoch when its epoch is the stage's and no write in progress has touched it,
        both before and after the read: a write's rows there would have marked it pending, and a commit would have
        moved its epoch. Then the tables read are the scope's at that epoch, as the tables the stage wrote are, and
        the pointer holds every table at the one epoch it names. Anything else answers False, and the stage is left
        to the catch-up as before.

        :param staged: the stage, its write committed
        :ptype staged: StagedScope
        :param tables: the tables to read whole from L3
        :ptype tables: Sequence[SnapshotTable]
        :param objects: the pointer's chunks, filled here
        :ptype objects: dict[str, str]
        :param counts: the pointer's row counts, filled here
        :ptype counts: dict[str, int]
        :param compacted: when each table's base was last made whole, filled here
        :ptype compacted: dict[str, float]
        :return: whether every table was filled
        :rtype: bool
        """
        if self._settled is None or (staged.claim is not None and not staged.claim.held):
            return False

        async def at_epoch() -> Any:
            """the writer's record when L3 holds the scope at the stage's epoch now; None when it does not."""
            settled = await self._settled() if self._settled is not None else None
            if isinstance(settled, Unsettled) and (settled.epochs is None or staged.scope in settled.pending):
                return None
            held = int((await self._epochs()).get(staged.scope, 0))
            return settled if held == staged.epoch else None

        before = await at_epoch()
        if before is None:
            return False
        read = await self._read_scope(staged.scope, tables=tables)
        after = await at_epoch()
        if after is None or (not isinstance(before, Unsettled) and after != before):
            log.info(
                "scoped snapshot %s: scope %r was written while its untouched tables were read from L3; left for "
                "the catch-up",
                self._name,
                staged.scope,
            )
            return False
        by_name = {table.name: table for table in tables}
        for part in read:
            table = by_name[part.table]
            arrow = part.arrow if part.arrow is not None else await self._export(table, staged.scope, [])
            _still_held(staged.claim, f"filling scope {staged.scope!r}'s {table.name} from L3")
            objects[table.name], _ = await self._put_chunk(staged.scope, staged.epoch, table, arrow)
            counts[table.name] = arrow.num_rows
            compacted[table.name] = time.time()
        log.info(
            "scoped snapshot %s: scope %r's pointer was not at the epoch its writer saw; the tables its stage left "
            "out were read from L3 at the stage's epoch",
            self._name,
            staged.scope,
            extra={"extra_data": {"scope": staged.scope, "epoch": staged.epoch, "tables": sorted(by_name)}},
        )
        return True

    async def _exist(self, names: Iterable[str]) -> bool:
        """whether every named chunk is in the store.

        :return: True when each is
        :rtype: bool
        """
        # one direct read per name, not a listing of the bucket
        missing = [name for name in names if await self._store.info(name) is None]
        return not missing

    async def _move_exactly(self, pointer: _Pointer, revision: int | None | object) -> bool:
        """move a scope's pointer to ``pointer`` only from the entry at ``revision`` (``None``: from none).

        :return: whether it moved
        :rtype: bool
        """
        key = self._layout.pointer_key(pointer.scope)
        if revision is None:
            moved = await self._pointers_bucket.create(key=key, value=pointer.encode()) is not None
        else:
            assert isinstance(revision, int)  # noqa: S101 -- _completed answers an int, None or _ANY_ENTRY
            moved = await self._pointers_bucket.update(key=key, value=pointer.encode(), revision=revision) is not None
        if moved:
            self._note_seen(pointer)
            _headway(f"moved scope {pointer.scope!r}'s pointer")
        return moved

    async def _move_pointer(self, pointer: _Pointer) -> _Pointer:
        """move a scope's pointer to ``pointer`` by compare-and-set, never to a lower epoch.

        A same-epoch pointer of other columns is moved too: a rebuild repoints a scope another code
        version published at the epoch L3 still records, to this replica's columns. Leaving it in place
        until a write moved the scope kept every scope no write touched after a column change on the old
        columns, so every replica rebuilt the whole snapshot from L3 at every start. While two versions
        run together, a replica that starts repoints the other's same-epoch scopes; a ready replica of the
        other version never takes a same-epoch pointer of other columns, so it keeps serving its own L1
        and is not made to rebuild, and the repointing ends with the deploy.

        :param pointer: the new pointer
        :ptype pointer: _Pointer
        :return: the pointer the scope holds after the move: ``pointer``, or a later one a racing writer set
        :rtype: _Pointer
        :raises RuntimeError: when racing writers kept moving it through every attempt
        """
        key = self._layout.pointer_key(pointer.scope)
        held: _Pointer | None = None
        attempts = 0
        while held is None and attempts < _CAS_ATTEMPTS:
            attempts += 1
            entry = await self._pointers_bucket.get_entry(key=key)
            current = None if entry is None else _Pointer.decode(entry[0])
            if current is not None and not pointer.supersedes(current):
                held = current
            elif entry is None:
                held = pointer if await self._pointers_bucket.create(key=key, value=pointer.encode()) else None
            else:
                moved = await self._pointers_bucket.update(key=key, value=pointer.encode(), revision=entry[1])
                held = pointer if moved is not None else None
        if held is None:
            raise RuntimeError(f"the pointer of scope {pointer.scope!r} kept moving under {_CAS_ATTEMPTS} attempts")
        self._note_seen(held)
        _headway(f"moved scope {pointer.scope!r}'s pointer")
        return held

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
            entry = await self._pointers_bucket.get_entry(key=self._layout.index_key)
            body = {} if entry is None else json.loads(entry[0])
            held: frozenset[str] = frozenset(body.get("scopes", ()))
            held_partial = entry is None or bool(body.get("partial", False))
            partial = held_partial and not whole
            wanted = (held | adding) - removing
            value = json.dumps({"scopes": sorted(wanted)} | ({"partial": True} if partial else {})).encode("utf-8")
            if entry is not None and wanted == held and partial == held_partial:
                settled = held
            elif entry is None:
                settled = (
                    wanted if await self._pointers_bucket.create(key=self._layout.index_key, value=value) else None
                )
            else:
                moved = await self._pointers_bucket.update(key=self._layout.index_key, value=value, revision=entry[1])
                settled = wanted if moved is not None else None
        if settled is None:
            raise RuntimeError(f"the index of snapshot {self._name!r} kept moving under {_CAS_ATTEMPTS} attempts")
        self._index = settled
        self._index_partial = partial
        self._ever_indexed = True
        _headway("moved the index")

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
            _headway(f"listed the scopes of {table.name} in L3")
        return scopes

    async def _read_scope(
        self,
        scope: str,
        moved: Callable[[], Awaitable[bool]] | None = None,
        tables: Sequence[SnapshotTable] | None = None,
    ) -> list[PartitionReplacement]:
        """every row of one scope in each table, from L3, as the replacements that make this L1 hold them.

        Read a page at a time, and each page typed by this L1 as it arrives (the backend's
        ``export_rows``, through which a backend that keeps less than L3 holds -- a retention -- lets go
        of what it would not keep), so what is held is the scope's rows as the L1 keeps them and one
        page of L3's, never the scope's raw rows. The pages are in key order, so their Arrow tables,
        concatenated, are the scope's.

        :param scope: the scope
        :ptype scope: str
        :param moved: asked between pages: whether a write has begun since the read started, which stops it
        :ptype moved: Callable[[], Awaitable[bool]] | None
        :param tables: the tables to read; every table when ``None``
        :ptype tables: Sequence[SnapshotTable] | None
        :return: one replacement per table, its rows an Arrow table (``None`` for none)
        :rtype: list[PartitionReplacement]
        :raises _WriteBegan: when ``moved`` says a write began
        """
        result: list[PartitionReplacement] = []
        for table in self._tables if tables is None else tables:
            parts: list[Any] | None = None
            types = _rail_types(self._backend.column_types(table.name)) if isinstance(self._l3, ArrowL3Reader) else None
            if types is not None:
                try:
                    parts = await self._read_table(table, scope, types, moved)
                except ReplyNotArrowError as exc:
                    # NOSILENT: a value the rail answers otherwise than its column's Arrow type: read as dict rows
                    log.info(
                        "a scope's rows are read from L3 as dict rows: a value is not its column's Arrow type",
                        extra={"extra_data": {"table": table.name, "scope": scope, "error": str(exc)}},
                    )
            if parts is None:
                parts = await self._read_table(table, scope, None, moved)
            result.append(
                PartitionReplacement(
                    table=table.name,
                    column=table.scope_column,
                    value=scope,
                    arrow=_concatenated(parts) if parts else None,
                    primary_key=table.key,
                )
            )
        return result

    async def _read_table(
        self,
        table: SnapshotTable,
        scope: str,
        types: Mapping[str, Any] | None,
        moved: Callable[[], Awaitable[bool]] | None = None,
    ) -> list[Any]:
        """one table's rows of a scope from L3, each page typed by this L1 as it arrives (:meth:`_read_scope`).

        With ``types`` the rail's pages are read straight into Arrow (``fetch_arrow``: no row is a Python object
        here) and typed by the L1 alike (:meth:`~threetears.core.cache.duckdb.DuckDBBackend.storage_arrow`); without,
        they are dict rows.

        :param table: the table
        :ptype table: SnapshotTable
        :param scope: the scope
        :ptype scope: str
        :param types: each column's Arrow type as the rail answers it (:func:`_rail_types`), or None for dict rows
        :ptype types: Mapping[str, pyarrow.DataType] | None
        :param moved: asked between pages: whether a write has begun since the read started
        :ptype moved: Callable[[], Awaitable[bool]] | None
        :return: the scope's rows as the L1 keeps them, a part a page
        :rtype: list[pyarrow.Table]
        :raises ReplyNotArrowError: with ``types``, when a value is not its column's Arrow type
        :raises _WriteBegan: when ``moved`` says a write began
        """
        columns = tuple(self._backend.column_types(table.name))
        where = {table.scope_column: scope}
        pages: AsyncGenerator[Any, None]
        if types is None:
            pages = l3_row_pages(self._l3, table.name, columns, table.key, where=where)
        else:
            select = f"SELECT {', '.join(quote_identifier(c) for c in columns)} FROM {quote_identifier(table.name)}"  # noqa: S608
            pages = keyset_arrow_pages(
                self._l3, select, table.key, types=types, where=where, page_size=DEFAULT_PAGE_SIZE
            )
        parts: list[Any] = []
        async with aclosing(pages):
            async for page in pages:
                if len(page):
                    parts.append(await self._export(table, scope, page))
                _headway(f"read a page of {table.name} in scope {scope!r} from L3")
                if moved is not None and await moved():
                    raise _WriteBegan(f"a write began while {table.name} of scope {scope!r} was read")
        return parts

    def _claim(self) -> _RebuildClaim:
        """a fresh claim on the rebuild.

        :return: the claim
        :rtype: _RebuildClaim
        """
        return _RebuildClaim(self._rebuild_lease, key=self._layout.claim_key, ttl=self._claim_ttl, name=self._name)

    async def _holding[T](self, claim: _RebuildClaim, work: Callable[[], Coroutine[Any, Any, T]], *, what: str) -> T:
        """run ``work`` under the claim this replica holds, letting the claim go and stopping it once it stalls.

        The claim's renewal runs on its own task, so it says nothing of the work: a run stuck on one await (an
        L3 read that never answers, a thread that never comes) renewed its claim for as long as its process lived,
        and every other replica waited on it. So the run's own steps (:func:`_headway`: pages read, chunks read
        and written, scopes committed, pointers moved) are watched, and once none has come for ``rebuild_stall``
        the claim is let go first -- whatever the run is stuck in, no replica waits on it any longer -- and then
        the run is cancelled.

        :param claim: the claim, taken
        :ptype claim: _RebuildClaim
        :param work: the run
        :ptype work: Callable[[], Coroutine[Any, Any, T]]
        :param what: the run, for the log
        :ptype what: str
        :return: what the run answered
        :rtype: T
        :raises TimeoutError: when it made no progress for ``rebuild_stall``; the claim is let go
        :raises BaseException: what the run raised
        """
        headway = _Headway(time.monotonic(), "took the claim")

        async def watched() -> T:
            # in this task's own context, which every task the run starts inherits; nothing else sees it
            _HEADWAY.set(headway)
            return await work()

        task = asyncio.create_task(watched(), name=f"scoped-snapshot-{self._name}-under-claim")
        stall = self._rebuild_stall.total_seconds()
        look = min(max(stall / 4, 0.01), _STALL_LOOK_SECONDS)
        try:
            while not task.done():
                await asyncio.wait([task], timeout=look)
                idle = time.monotonic() - headway.at
                if task.done() or idle < stall:
                    continue
                message = (
                    f"scoped snapshot {self._name}: {what} made no progress for {idle:.0f} s (its last step: "
                    f"{headway.what}); it lets the rebuild claim go and stops"
                )
                log.error(message, extra={"extra_data": {"snapshot": self._name, "last_step": headway.what}})
                await claim.release()
                await self._stop_run(task, stall)
                raise _Stalled(message)
        except asyncio.CancelledError:
            await self._stop_run(task, stall)
            raise
        return task.result()

    async def _stop_run(self, task: asyncio.Task[Any], within: float) -> None:
        """cancel a run under the claim and wait, up to ``within`` seconds, for it to end; its own failure is logged.

        :param task: the run
        :ptype task: asyncio.Task[Any]
        :param within: how long to wait for it
        :ptype within: float
        :return: nothing
        :rtype: None
        """
        task.cancel()
        done, _ = await asyncio.wait([task], timeout=within)
        if not done:
            log.error(
                "scoped snapshot %s: a run under the rebuild claim did not end within %.0f s of being stopped; "
                "it ends on its own, the claim already let go",
                self._name,
                within,
            )
        elif not task.cancelled() and (error := task.exception()) is not None:
            log.warning("scoped snapshot %s: a stopped run had failed: %s: %s", self._name, type(error).__name__, error)

    async def _rebuild(self, *, reason: str) -> None:
        """rebuild every scope's chunks from L3 under the claim, or wait for whoever holds it.

        :param reason: why, for the status
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        # the pointers were lost with their bucket, and every write claim with them: until the
        # writers take theirs again a sweep could not tell a live write's stage from a dead one's
        lost = self._index is None and self._ever_indexed
        if lost:
            # what a rebuild published went with the bucket: none of it stands in for a scope now
            self._rebuilt.clear()
        if self._ensure_buckets is not None and lost:
            await self._ensure_buckets()
        claim = self._claim()
        if await claim.take():
            kept = False
            try:
                caught = await self._holding(
                    claim, lambda: self._rebuild_scopes(None, reason=reason, flow="sync"), what="a rebuild from L3"
                )
            finally:
                kept = await claim.release()
            # a rebuild that lost its claim meanwhile may have run beside another: it leaves the sweep
            # to a rebuild that held the claim throughout
            if caught is not None and not lost and kept:
                await self._sweeper.retire_unreferenced()
        else:
            self._wait("sync", f"{reason}; another replica is rebuilding or publishing it")

    def _rebuilt_at(self, scope: str, epoch: int, done: Mapping[str, int]) -> bool:
        """whether a rebuild here already published ``scope`` at ``epoch``, and this L1 still holds it there.

        :param scope: the scope
        :ptype scope: str
        :param epoch: the epoch L3 records for it now
        :ptype epoch: int
        :param done: scope -> the epoch a rebuild published it at
        :ptype done: Mapping[str, int]
        :return: True when it need not be read again: L3, this L1 and the pointer all name that epoch
        :rtype: bool
        """
        applied = self._applied.get(scope)
        seen = self._seen.get(scope)
        # its pointer must still name it there, under these columns: a pointer NATS lost, or one another
        # version repointed since, is published again
        return (
            done.get(scope) == epoch
            and applied is not None
            and applied.epoch == epoch
            and seen is not None
            and seen.epoch == epoch
            and self._loadable(seen)
        )

    async def _rebuild_scopes(self, only: set[str] | None, *, reason: str, flow: str) -> list[str] | None:
        """read scopes from L3 while no write moves them, publish them, then update the index.

        An attempt that a committed write stopped is tried again, up to :data:`_REBUILD_ATTEMPTS`
        times; a scope an earlier attempt already published at the epoch L3 still records is not
        read again. That holds across calls too: what any earlier rebuild here published
        (:attr:`_rebuilt`) is kept while this replica waits on a write in progress, so each gap between
        a loader's writes adds scopes rather than beginning again from the first.

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
        done = self._rebuilt
        attempts = 0
        while result is None and attempts < _REBUILD_ATTEMPTS:
            attempts += 1
            before = await self._settled() if self._settled is not None else None
            if isinstance(before, Unsettled):
                if only is not None and before.epochs is not None:
                    # a write that records what it touched: the scopes it has not touched are read while it runs
                    result = await self._publish_untouched(only, before, reason=reason, done=done)
                if result is None or set(result) != only:
                    self._wait(flow, f"{reason}; waiting for a write in progress ({before.reason})")
                break
            result = await self._read_and_publish(only, reason=reason, before=before, done=done)
        if result is None and attempts >= _REBUILD_ATTEMPTS:
            self._wait(flow, f"{reason}; writes kept committing while it read; trying again")
        return result

    async def _publish_untouched(
        self, only: set[str], before: Unsettled, *, reason: str, done: dict[str, int]
    ) -> list[str]:
        """publish the scopes of ``only`` a write in progress has not touched, each read while it runs.

        A write marks a scope pending in the transaction of its first rows there, and clears the mark only
        when it commits, moving the scope's epoch in the same transaction. So a scope that is not pending,
        and still not pending at the same epoch once its rows have been read, was read whole at that epoch:
        rows the write committed there during the read would have left it pending, and a commit would have
        moved its epoch. Anything else is left for a catch-up once the write has committed.

        :param only: the scopes to catch up
        :ptype only: set[str]
        :param before: the writer's record before any scope was read, naming every scope's epoch and the pending ones
        :ptype before: Unsettled
        :param reason: why, for the status
        :ptype reason: str
        :param done: scope -> the epoch it was published at; this adds to it
        :ptype done: dict[str, int]
        :return: the scopes published
        :rtype: list[str]
        """
        epochs = before.epochs or {}
        l3_scopes = await self._l3_scopes()
        scopes = sorted(scope for scope in only & l3_scopes if scope not in before.pending)
        published: list[str] = []
        retire: dict[str, int] = {}
        with self._doing(SnapshotPhase.REBUILDING_FROM_L3, f"catching up beside a write: {reason}", total=len(scopes)):
            for scope in scopes:
                epoch = int(epochs.get(scope, 0))
                rows = await self._read_scope(scope)
                after = await self._settled() if self._settled is not None else None
                if isinstance(after, Unsettled):
                    if after.epochs is None:
                        break
                    still = scope not in after.pending and int(after.epochs.get(scope, 0)) == epoch
                else:
                    # the write committed: the scope is as read only when its epoch did not move
                    still = int((await self._epochs()).get(scope, 0)) == epoch
                if not still:
                    log.info(
                        "scoped snapshot %s: scope %r was written while it was read beside a write; left for later",
                        self._name,
                        scope,
                    )
                    continue
                async with self._local:
                    moved, _ = await self._publish_locked(scope, epoch, rows)
                if moved is not None:
                    retire[scope] = epoch
                done[scope] = epoch
                published.append(scope)
                self._progress = replace(self._progress, scopes_done=self._progress.scopes_done + 1)
                # as the rebuild says each scope it lands: a catch-up of many scopes beside a write is watched by it
                log.info(
                    "scoped snapshot %s: caught up scope %r from L3 beside a write (%d of %d)",
                    self._name,
                    scope,
                    self._progress.scopes_done,
                    self._progress.scopes_total,
                    extra={"extra_data": {"scope": scope, "epoch": epoch}},
                )
        if published:
            await self._update_index(add=published)
        await self._sweeper.retire_older_many(retire)
        return published

    async def _read_and_publish(
        self, only: set[str] | None, *, reason: str, before: Any, done: dict[str, int]
    ) -> list[str] | None:
        """one rebuild attempt: read the scopes from L3, publishing each while no write has committed since ``before``.

        :param only: the scopes to rebuild; every scope L3 holds when ``None``
        :ptype only: set[str] | None
        :param reason: why, for the status
        :ptype reason: str
        :param before: the writer's stamp before the attempt, taken before the epochs are read
        :ptype before: Any
        :param done: scope -> the epoch an earlier attempt published it at; this attempt adds to it
        :ptype done: dict[str, int]
        :return: the scopes published, or ``None`` when a write committed during the attempt
        :rtype: list[str] | None
        :raises ValueError: when a table holds rows with no scope
        """
        started = time.perf_counter()
        l3_scopes = await self._l3_scopes()
        scopes = sorted(l3_scopes if only is None else (only & l3_scopes))
        epochs = await self._epochs()
        with self._doing(SnapshotPhase.REBUILDING_FROM_L3, f"rebuilding from L3: {reason}", total=len(scopes)):
            return await self._read_and_publish_scopes(
                only, scopes, l3_scopes, epochs, before=before, started=started, done=done
            )

    async def _read_and_publish_scopes(
        self,
        only: set[str] | None,
        scopes: list[str],
        l3_scopes: set[str],
        epochs: Mapping[str, int],
        *,
        before: Any,
        started: float,
        done: dict[str, int],
    ) -> list[str] | None:
        """the body of :meth:`_read_and_publish`, inside its step.

        Scope by scope, in scope order: each is read from L3 (:meth:`_read_scope`), ``l3_concurrency``
        ahead of the one being published, and published -- this L1, its chunks, its pointer -- then let
        go. So what a rebuild holds at once is a few scopes' rows as the L1 keeps them, never every
        scope's: the rows of every scope at once were more than a pod's memory.

        A scope is published only when the writer's stamp, read after the scope's rows, is still
        ``before``: no write committed since the epochs were read, so the rows are exactly the scope's
        at its epoch, every table at the same one. Once a write has committed nothing more is published
        and the attempt answers ``None``; the scopes it published stay published (each whole, at its
        epoch, as any publish), and the next attempt reads only the scopes not already published at the
        epoch L3 then records. The index is written whole, and the copy declared ready, only once every
        scope is published and the stamp is still ``before`` after the last: a rebuild that dies part
        way, or that a write outran, leaves no index naming it complete.

        :return: the scopes published, or ``None`` when a write committed during the attempt
        :rtype: list[str] | None
        """
        spent = {"read_l3": 0.0, "publish": 0.0}
        read_seconds: dict[str, float] = {}
        pending = [scope for scope in scopes if not self._rebuilt_at(scope, int(epochs.get(scope, 0)), done)]
        self._progress = replace(self._progress, scopes_done=len(scopes) - len(pending))
        asked = [time.monotonic()]

        async def moved() -> bool:
            # a write that began (a reset's clear empties the tables being read) stops the read: asked between pages,
            # at most every few seconds, so a scope's read stops within that rather than after its last page
            if self._settled is None or time.monotonic() - asked[0] < self._write_check.total_seconds():
                return False
            asked[0] = time.monotonic()
            return bool(await self._settled() != before)

        async def read(scope: str) -> list[PartitionReplacement]:
            began = time.perf_counter()
            rows = await self._read_scope(scope, moved)
            read_seconds[scope] = time.perf_counter() - began
            spent["read_l3"] += read_seconds[scope]
            return rows

        async def publish(scope: str, rows: list[PartitionReplacement]) -> bool:
            after = await self._settled() if self._settled is not None else None
            if after != before:
                log.info(
                    "scoped snapshot %s: a write committed during the rebuild, before scope %r; reading again",
                    self._name,
                    scope,
                )
                return False
            began = time.perf_counter()
            epoch = int(epochs.get(scope, 0))
            async with self._local:
                moved, _ = await self._publish_locked(scope, epoch, rows)
            if moved is not None:
                # at once, scope by scope: a rebuild that fails at a later scope (a full store)
                # still leaves nothing superseded behind for the scopes it did move
                await self._sweeper.retire_older(scope, epoch)
            done[scope] = epoch
            published_seconds = time.perf_counter() - began
            spent["publish"] += published_seconds
            self._progress = replace(self._progress, scopes_done=self._progress.scopes_done + 1)
            # the line an operator watches a rebuild of hours by: each scope as it lands, its rows and its seconds
            log.info(
                "scoped snapshot %s: rebuilt scope %r from L3 (%d of %d)",
                self._name,
                scope,
                self._progress.scopes_done,
                self._progress.scopes_total,
                extra={
                    "extra_data": {
                        "scope": scope,
                        "epoch": epoch,
                        "rows": sum(0 if r.arrow is None else int(r.arrow.num_rows) for r in rows),
                        "read_seconds": round(read_seconds.get(scope, 0.0), 2),
                        "publish_seconds": round(published_seconds, 2),
                        "done": self._progress.scopes_done,
                        "total": self._progress.scopes_total,
                    }
                },
            )
            return True

        published: list[str] | None = None
        try:
            whole = await _in_order(pending, read, publish, ahead=self._l3_concurrency)
        except _WriteBegan as stopped:
            # NOSILENT: logged; the next attempt finds the write in progress and waits for it, keeping what was published
            log.info("scoped snapshot %s: %s; the rebuild stops reading and waits for it", self._name, stopped)
            return None
        # once more after the last scope: the index is written whole only when no write committed during
        # the attempt, even one that left no scope to read (L3 read empty, or every scope published)
        after = await self._settled() if self._settled is not None and whole else before
        if whole and after != before:
            log.info("scoped snapshot %s: a write committed during the rebuild; reading again", self._name)
        if whole and after == before:
            await self._update_index(add=scopes, whole=only is None)
            gone = set() if only is not None else set(self._index or ()) - l3_scopes
            if gone:
                await self._remove_scopes(gone)
            await self._apply_published()
            done_at = time.perf_counter()
            self._progress = replace(
                self._progress,
                timings=_updated(
                    self._progress.timings,
                    {
                        # the seconds the scopes' reads took, and their publishes, summed over the scopes;
                        # the reads run beside each other and beside the publishes, so these overlap
                        "read_l3": round(spent["read_l3"], 3),
                        "publish": round(spent["publish"], 3),
                        "total": round(done_at - started, 3),
                    },
                ),
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
        the next (the ENR pod's refresh calls it after every publish). :meth:`catch_up` does the same and
        says how it ended.

        :return: the scopes republished
        :rtype: list[str]
        :raises ValueError: when a table holds rows with no scope
        :raises TimeoutError: when it made no progress for ``rebuild_stall`` (the claim is let go)
        """
        record = _CatchUpRecord()
        self._begin_run("catch_up")
        try:
            await self._catch_up(record)
        finally:
            self._end_run("catch_up")
        return list(record.caught)

    async def catch_up(self) -> CatchUpOutcome:
        """:meth:`catch_up_from_l3`, answering how it ended rather than only what it republished; never raises.

        A caller that must tell "another replica holds the claim" from "waited on a write" from "nothing was behind"
        reads :attr:`CatchUpOutcome.result`, not an empty list; :attr:`CatchUpOutcome.behind` names every scope
        still behind L3 by L3's epoch against the pointer this copy has seen. A failure is logged here and answered
        as ``FAILED``, a stall as ``STALLED``, each with its error.

        :return: the outcome
        :rtype: CatchUpOutcome
        """
        record = _CatchUpRecord()
        result = CatchUpResult.CAUGHT_UP
        reason: str | None = None
        ran = False
        self._begin_run("catch_up")
        try:
            try:
                ran = await self._catch_up(record)
            except _Stalled as exc:
                # NOSILENT: logged at error where the stall was found; answered as STALLED with its message
                result, reason = CatchUpResult.STALLED, str(exc)
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- a catch-up's failure is the caller's outcome: logged here and answered as FAILED with its error, never dropped
                contain(exc, log=log, what=f"scoped snapshot {self._name}: the catch-up from L3 failed")
                result, reason = CatchUpResult.FAILED, f"{type(exc).__name__}: {exc}"
            waited = self._waited.get("catch_up", False)
            wait = self._waiting.get("catch_up")
        finally:
            self._end_run("catch_up")
        if record.claim_held:
            result = CatchUpResult.CLAIM_HELD
        elif result is CatchUpResult.CAUGHT_UP and waited:
            result, reason = CatchUpResult.WAITED, wait
        # against the pointers as seen now, which every scope it republished has moved: what it did before a stall
        # or a failure counts too
        behind = self._lagging(record.epochs, record.scopes)
        if ran and result in (CatchUpResult.CAUGHT_UP, CatchUpResult.WAITED):
            # L3 read again: a write may have moved scopes while it ran
            behind = self._lagging(await self._epochs(), record.scopes)
        return CatchUpOutcome(
            result=result,
            caught=tuple(record.caught),
            behind=behind,
            finished_at=datetime.now(UTC),
            holder=record.holder,
            held_since=record.held_since,
            reason=reason,
        )

    def _lagging(self, epochs: Mapping[str, int], scopes: Iterable[str]) -> dict[str, ScopeLag]:
        """the scopes whose epoch in L3 is ahead of the pointer this replica has seen.

        :param epochs: each scope's epoch in L3
        :ptype epochs: Mapping[str, int]
        :param scopes: the scopes L3 holds
        :ptype scopes: Iterable[str]
        :return: scope -> its lag
        :rtype: dict[str, ScopeLag]
        """
        lagging: dict[str, ScopeLag] = {}
        for scope in sorted(scopes):
            seen = self._seen.get(scope)
            l3_epoch = int(epochs.get(scope, 0))
            if seen is None or l3_epoch > seen.epoch:
                lagging[scope] = ScopeLag(l3_epoch=l3_epoch, pointer_epoch=None if seen is None else seen.epoch)
        return lagging

    async def _catch_up(self, record: _CatchUpRecord) -> bool:
        """the body of :meth:`catch_up_from_l3` and :meth:`catch_up`, recording into ``record`` as it goes.

        :param record: where what it finds and does is kept
        :ptype record: _CatchUpRecord
        :return: whether it ran under the claim
        :rtype: bool
        """
        epochs = await self._epochs()
        l3_scopes = await self._l3_scopes()
        record.epochs, record.scopes = epochs, set(l3_scopes)
        record.behind = self._lagging(epochs, l3_scopes)
        behind = set(record.behind)
        gone = set(self._index or ()) - l3_scopes
        if not behind and not gone:
            return False
        claim = self._claim()
        if not await claim.take():
            holding = await self._rebuild_lease.held(self._layout.claim_key)
            record.claim_held = True
            if holding is not None:
                record.holder, record.held_since = holding.holder, holding.acquired_at
            log.info(
                "scoped snapshot %s: another replica holds the claim; it catches up",
                self._name,
                extra={
                    "extra_data": {
                        "holder": record.holder,
                        "held_since": None if record.held_since is None else record.held_since.isoformat(),
                        "behind": len(behind),
                    }
                },
            )
            return False
        try:
            await self._holding(
                claim, lambda: self._catch_up_under_claim(behind, gone, record), what="a catch-up from L3"
            )
        finally:
            await claim.release()
        return True

    async def _catch_up_under_claim(self, behind: set[str], gone: set[str], record: _CatchUpRecord) -> None:
        """what a catch-up does under the claim: rebuild the scopes behind L3, remove the scopes L3 left.

        :param behind: the scopes whose epoch in L3 is ahead of their pointer
        :ptype behind: set[str]
        :param gone: the scopes the index names and L3 holds no more
        :ptype gone: set[str]
        :param record: where the scopes republished are kept
        :ptype record: _CatchUpRecord
        :return: nothing
        :rtype: None
        """
        if behind:
            record.caught = (
                await self._rebuild_scopes(behind, reason=f"{len(behind)} scopes behind L3", flow="catch_up") or []
            )
        if gone:
            await self._remove_scopes(gone)

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
                raw = await self._pointers_bucket.get(key=self._layout.pointer_key(scope))
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
        held = {
            scope: await self._pointers_bucket.get_entry(key=self._layout.pointer_key(scope)) for scope in candidates
        }
        l3_now = await self._l3_scopes()
        removed: dict[str, _Pointer | None] = {}
        for scope in sorted(candidates - l3_now):
            entry = held[scope]
            if entry is None:
                removed[scope] = None
            elif await self._pointers_bucket.delete(key=self._layout.pointer_key(scope), revision=entry[1]):
                removed[scope] = _Pointer.decode(entry[0])
            else:
                log.info("scoped snapshot %s: scope %r moved while it was removed; kept", self._name, scope)
        if not removed:
            return
        await self._update_index(remove=removed)
        standing = {
            scope
            for scope in removed
            if await self._pointers_bucket.get(key=self._layout.pointer_key(scope)) is not None
        }
        if standing:
            await self._update_index(add=standing)
        gone = {scope: pointer for scope, pointer in removed.items() if scope not in standing}
        for scope in gone:
            self._seen.pop(scope, None)
            self._rebuilt.pop(scope, None)
        await self._drop_locally(sorted(gone))
        for scope, pointer in gone.items():
            # every chunk up to the deleted pointer's epoch, by the one rule: with the pointer gone
            # nothing names them, and a writer recreating the scope writes at a later epoch under its
            # own claim. a scope that had no pointer goes at the next sweep
            if pointer is not None:
                await self._sweeper.retire_older(scope, pointer.epoch + 1)


async def open_tool_pod_snapshot(
    nats_client: NatsClient,
    *,
    identity_token: Callable[[], str],
    name: str,
    tables: Sequence[SnapshotTable],
    backend: DuckDBBackend,
    l3: L3Reader,
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
    :ptype l3: L3Reader
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
