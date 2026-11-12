"""Epochs per scope, kept in L3: the version of the data each part of it was last changed at.

**Why per scope.** A result is cached under its epoch (a CDN URL ``.../v{epoch}.json``, a derived
value keyed on ``(view, scope, epoch)``), so a new epoch is a new key and nothing is ever purged
or invalidated: old keys age out. One epoch for all the data would change every key on every write
and empty every cache; an epoch per scope (a race, a state) changes only the keys of the scopes a
write touched, and every other scope keeps its keys and its cached copies.

**Why in L3.** The value escapes the process: it is in URLs browsers and the CDN hold. A counter
that reset would hand out ``v1`` again for different data, at an address caches still hold. So the
epochs are rows of an L3 table the owner declares (:func:`scope_epochs_schema`), moved in the
writer's own transaction, and they survive anything a broker or a pod restart does.

**The version is the epoch.** Each write takes the next version number (:meth:`ScopeEpochs.begin`)
and, when it commits, moves every scope it changed to that number (:meth:`ScopeEpochs.commit`). A
scope's epoch is therefore the version of the last write that changed it: it only moves forward,
it may move by more than one, and the epoch it replaced is recorded, never inferred, for a reader
that must still serve the one before. A write's rows may carry the same number (``source_version``)
so a reader can tell which write left each row.

**A seqlock for readers.** The table holds one more row, :data:`WHOLE`, whose epoch is the last
committed version and whose ``writing`` names a write begun and not committed. A writer commits
``begin`` before its first data write and ``commit`` in the transaction of its last, so a reader
that finds no write in progress and the same version before and after reading every data table
(:meth:`ScopeEpochs.settled`, the ``settled`` a
:class:`~threetears.core.collections.copy_follower.CopyFollower` takes) read one state of all of them.
A write that never commits leaves ``writing`` set: readers keep the state they have until the next
write commits, rather than reading rows a partial write left. The next write takes a number above
it, so no number a partial write may have stamped rows with is used twice.

**Order: write, commit, then the epochs.** The epochs move in the same transaction as the commit
of the last data write (or after it), never before, so a reader that learns of a new epoch finds
its data already committed.

**Telling other replicas.** The epochs' collection broadcasts each committed change as any
collection does (when it has a bus); :meth:`ScopeEpochs.on_change` calls back on a peer's
broadcast as on this process's own write. A replica that missed the broadcast learns the same by
reading :meth:`ScopeEpochs.snapshot`, which is the truth either way.

**One writer at a time, fenced in the database.** A writer holds the write's lease (a
:class:`~threetears.core.coordination.coalesced_run.CoalescedRun`), but a lease can lapse under a
writer that has not noticed. So the version is also a fencing token: ``begin`` takes the next number
under the row's lock, never waiting for it: a ``begin`` racing another, or one behind a data
transaction still open, is refused at once (:class:`EpochRecordBusyError`), so two writes never
share a number. The latest ``begin`` supersedes any write still in progress. ``commit`` moves
anything only while its version is still the write in progress, checked in the same statement that
clears it, inside the caller's transaction: a writer whose lease lapsed, and was followed by another,
is refused and its transaction moves nothing.

**A hung writer's hold is bounded.** Each data transaction holds the row's share lock (``touch``)
until it ends, so a writer that hangs with one open keeps every later ``begin`` refused until that
transaction ends. Through the L3 broker it ends within the broker's idle bound: a transaction session
idle past ``tx_idle_timeout_seconds`` (60 s on the hub) is rolled back by its sweep (every 10 s), and
each statement is bounded by the session's statement timeout. On a direct Postgres connection pass
``idle_timeout`` (sized to the write's lease): every transaction ``begin`` and ``touch`` join sets
``idle_in_transaction_session_timeout`` for itself, so the server ends a stalled one. The broker
refuses ``SET``, so leave it unset there.

**A write whose writer is gone is abandoned.** Given the lease every writer holds while it writes
(:class:`WriterLease`), :meth:`ScopeEpochs.settled` abandons a write in progress whose lease is free:
its writer let go (its run ended, a failure included) or died and the lease lapsed. A live writer,
however slow, renews its lease and is never abandoned; nor is a write with a data transaction still
open, which holds the version row (the abandon takes it without waiting, as ``begin`` does). The abandon
is a commit of what the write left: every scope it touched, and the data's version, move to its number
(so the rows it committed are named by an epoch of their own, never by the one before), and ``writing``
clears. A writer that comes back is refused by the fence, as one a later write superseded. Without a
lease nothing is abandoned, and readers keep what they hold until the next write commits.

**A reader reads only what changed.** :meth:`ScopeEpochs.snapshot` keeps the snapshot it last read and, after its
first, reads only the rows of a change: :data:`WHOLE`, every scope still pending, and every scope whose epoch passed
the version it last read. That is every row a write can have changed since: a touch marks a scope pending, and a
commit (an abandon included) moves each scope it settles to its own version, which is above every version committed
before it began. :data:`WHOLE` is read first, in the read's first statement, so a commit landing while a read pages
moves its scopes past the version that read keeps, and the next read takes them. A record it cannot trust (the
writer's record recreated or gone, a version or an epoch gone back, a scope held as pending neither pending nor moved)
is read whole again, counted (``scope_epochs_whole_reads_total``). A scope's row deleted outright, or an epoch set back
by hand below the version held, is not seen: nothing here does either. Whatever the read, the answer is the one a whole
read gives.

**Scopes a write touched are recorded as it writes.** :meth:`ScopeEpochs.touch`, in the transaction
of each data write, marks the scopes that write changed as pending. ``commit`` moves every pending
scope, the ones an abandoned write left included: rows a write committed before it died are then
covered by the next write's epochs, even when that write does not change those scopes itself.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import asyncpg
from threetears.observe import contain, get_logger

from threetears.core.keyed import KeyBatch, Lock, WholeReads, read_keyed, read_row, read_value
from threetears.core.sql_fragments import quote_identifier
from threetears.core.declared import whole_read
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE, Unsettled, read_l3_rows
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
    collection_for_schema,
)
from threetears.core.coordination.lease import KVLease
from threetears.core.entities.base import BaseEntity
from threetears.core.keyset import read_keyset_pages

__all__ = [
    "SCOPE_EPOCHS_TABLE",
    "STALLED_WRITE_SECONDS",
    "WHOLE",
    "EpochRecordBusyError",
    "EpochSnapshot",
    "ScopeEpochs",
    "WriterLease",
    "scope_epochs_collection",
    "scope_epochs_schema",
]

log = get_logger(__name__)

#: the table's default name
SCOPE_EPOCHS_TABLE: Final = "scope_epochs"

#: the row holding the data's version and any write in progress; not a scope a caller may name
WHOLE: Final = "*"

#: the columns a snapshot reads
_COLUMNS: Final = ("scope", "epoch", "previous_epoch", "writing", "pending", "date_created", "date_updated")

#: why a snapshot reads the epochs table whole, counted by table and why (``scope_epochs_whole_reads_total``). Only
#: ``first`` (an instance's first read) is expected; every other is a record a read of its changes could not be trusted
#: to follow. Each is a cold start (coordinator ruling 2026-11-12): the table emptied or replaced, or restored behind
#: the version held
_WHOLE_READS: Final = WholeReads(
    "scope_epochs_whole_reads_total",
    subject="whole reads of a scope epochs table",
    whys={
        "first": "the first read of an instance: every scope is read once",
        "reset": "the writer's record was recreated or is gone (the table emptied or replaced)",
        "version_backwards": "the data's version went back (the table restored from behind)",
    },
    expected={"first"},
)

#: why a change names scopes the rows held cannot be merged with, each read again by key, never the table whole
#: (coordinator ruling 2026-11-12): a scope's epoch went back, or a scope held as pending is neither pending nor moved
#: (its mark cleared without a commit moving it)
_REREAD_BY_KEY: Final = frozenset({"epoch_backwards", "pending_lost"})

#: scopes one statement names when touching
_TOUCH_BATCH: Final = 500

#: a write in progress longer than this has most likely died: readers say so at warning, not info
STALLED_WRITE_SECONDS: Final = 600


class EpochRecordBusyError(RuntimeError):
    """the version row is locked by another transaction, so no write may begin now.

    Held by a data transaction still open (the fence of :meth:`ScopeEpochs.touch`), or by another
    ``begin`` racing this one. Waiting on a hung data transaction would stall this write as long as it
    hangs, so ``begin`` refuses at once instead; the next request to write begins once the holder has
    ended (a hung one within the bound the module describes).
    """


def scope_epochs_schema(name: str = SCOPE_EPOCHS_TABLE) -> TableSchema:
    """the epochs table, for the owner to declare among its tables.

    :param name: the table's name
    :ptype name: str
    :return: the schema, keyed on ``scope``
    :rtype: TableSchema
    """
    return TableSchema(
        name=name,
        primary_key="scope",
        columns=[
            Column("scope", STRING_TYPE),
            Column("epoch", BIGINT_TYPE),
            Column("previous_epoch", BIGINT_TYPE, nullable=True),
            Column("writing", BIGINT_TYPE, nullable=True),
            Column("pending", BIGINT_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        on_conflict="update",
    )


class _ScopeEpoch(BaseEntity):
    """one scope's epoch."""

    primary_key_field = "scope"


def scope_epochs_collection(name: str = SCOPE_EPOCHS_TABLE) -> type[SchemaBackedCollection[Any]]:
    """a collection class over the epochs table; build it once and keep it.

    :param name: the table's name
    :ptype name: str
    :return: the collection class
    :rtype: type[SchemaBackedCollection[Any]]
    """
    return collection_for_schema(scope_epochs_schema(name), entity_class=_ScopeEpoch)


@dataclass(frozen=True)
class WriterLease:
    """the lease every writer of the epochs holds from before its ``begin`` until its write ends.

    Its being free is what tells a write in progress whose writer is gone from a slow one: the holder
    renews it while it writes, and lets it go (or stops renewing, and it lapses) once it does not.

    :ivar lease: the lease factory the writers hold it through
    :ivar key: the lease's name
    """

    lease: KVLease
    key: str


@dataclass(frozen=True)
class EpochSnapshot:
    """every scope's epoch, and the data's version, as read at one moment.

    :ivar version: the last committed write's version; 0 before any
    :ivar writing: the version of a write begun and not committed, or None
    :ivar epochs: each scope's epoch, by scope (:data:`WHOLE` left out)
    :ivar previous: the epoch each scope's last move replaced (None after its first move), by scope;
        :data:`WHOLE`'s is the version before the current one
    :ivar writing_since: when the write in progress began; None when none is
    :ivar pending: the scopes a write has touched and no commit has moved since (:meth:`ScopeEpochs.touch`)
    """

    version: int
    writing: int | None
    epochs: Mapping[str, int]
    previous: Mapping[str, int | None]
    writing_since: datetime | None = None
    pending: frozenset[str] = frozenset()

    def epoch(self, scope: str) -> int:
        """one scope's epoch; 0 for a scope no write has changed.

        :param scope: the scope
        :ptype scope: str
        :return: the epoch
        :rtype: int
        """
        return self.epochs.get(scope, 0)


class ScopeEpochs:
    """per-scope epochs over the owner's epochs table.

    :param collection: the epochs table's collection (:func:`scope_epochs_collection`); give it the
        pod's bus so its commits reach the other replicas
    :ptype collection: SchemaBackedCollection
    :param page_size: rows per L3 page when reading every scope
    :ptype page_size: int
    :param idle_timeout: on a direct Postgres connection, how long a transaction ``begin`` or
        ``touch`` joined may sit idle before the server ends it (``idle_in_transaction_session_timeout``),
        sized to the write's lease; None sets nothing (through the L3 broker, whose own idle sweep
        bounds it and which refuses ``SET``)
    :ptype idle_timeout: timedelta | None
    :param writer: the lease every writer holds while it writes; a write in progress whose lease is
        free is abandoned (the module's "A write whose writer is gone is abandoned"). None abandons nothing
    :ptype writer: WriterLease | None
    :raises ValueError: when ``idle_timeout`` is not positive
    """

    def __init__(
        self,
        collection: SchemaBackedCollection[Any],
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        idle_timeout: timedelta | None = None,
        writer: WriterLease | None = None,
    ) -> None:
        if idle_timeout is not None and idle_timeout <= timedelta(0):
            raise ValueError(f"idle_timeout must be positive; got {idle_timeout}")
        self._collection = collection
        self._page_size = page_size
        self._writer = writer
        self._table = quote_identifier(collection.table_name)
        #: the table's key (:func:`scope_epochs_schema`): a scope's row, and the record's, are read by it
        self._key_column = quote_identifier("scope")
        self._idle_ms = None if idle_timeout is None else max(1, int(idle_timeout.total_seconds() * 1000))
        # the snapshot last read, the rows it was built from (by scope), and the version a read of the changes reads past
        self._held: EpochSnapshot | None = None
        self._rows: dict[str, dict[str, Any]] = {}
        self._since = 0
        self._reading = asyncio.Lock()

    @property
    def table_name(self) -> str:
        """the epochs table's name: every replica hears its row messages when a write begins or commits."""
        return self._collection.table_name

    async def _bound_idle(self, conn: Any) -> None:
        """bound how long the caller's transaction may sit idle, when an idle timeout was given.

        :param conn: the caller's connection, its transaction open
        :ptype conn: Any
        :return: nothing
        :rtype: None
        """
        if self._idle_ms is not None:
            # an int this class computed; SET takes no parameters
            await conn.execute(f"SET LOCAL idle_in_transaction_session_timeout = {int(self._idle_ms)}")

    async def snapshot(self) -> EpochSnapshot:
        """every scope's epoch and the data's version, read from L3: whole at the first read, then only what changed.

        The module's "A reader reads only what changed": the answer is the one a whole read gives.

        :return: the snapshot
        :rtype: EpochSnapshot
        """
        async with self._reading:
            held = self._held
            why = "first"
            if held is not None:
                changed = await self._read_changed()
                distrust = self._distrust(held, changed)
                if distrust is None:
                    return self._merge(held, changed)
                if distrust in _REREAD_BY_KEY:
                    return await self._reread(held, changed)
                why = distrust
            return await whole_read(
                self._read_whole(why),
                kind="cold_start",
                reason="every scope's epoch is read once into a reader holding none (its first read), and again only when "
                "the table was emptied, replaced or restored behind the version held: no row held answers for it",
                ruling="Pace 2026-11-12: cold start fills read whole; coordinator 2026-11-12: a reset and a restore are one",
                proof="packages/core/tests/integration/test_scope_epochs_incremental.py::"
                "test_read_whole_is_reached_only_at_a_first_read_a_reset_or_a_restore",
                speed="packages/core/tests/integration/test_scope_epochs_incremental.py::test_read_whole_speed_at_real_size",
            )

    async def _read_whole(self, why: str) -> EpochSnapshot:
        """read every row, keep it, and answer its snapshot; counted by why.

        The version the next read reads past is read first, on its own: a commit landing while the rows page then
        moves its scopes past it, and the next read takes them.

        :param why: why the table is read whole (:data:`_WHOLE_READS`)
        :ptype why: str
        :return: the snapshot
        :rtype: EpochSnapshot
        """
        _WHOLE_READS.count(why, table=self._collection.table_name)
        l3 = self._collection.required_l3_pool
        record = await read_value(l3, self.table_name, by={"scope": WHOLE}, column="epoch")
        rows = await whole_read(
            read_l3_rows(l3, self._collection.table_name, _COLUMNS, ("scope",), page_size=self._page_size),
            kind="cold_start",
            reason="every scope's epoch is read once into a reader holding none (its first read), and again only when "
            "the table was emptied, replaced or restored behind the version held: no row held answers for it",
            ruling="Pace 2026-11-12: cold start fills read whole; coordinator 2026-11-12: a reset and a restore are one",
            proof="packages/core/tests/integration/test_scope_epochs_incremental.py::"
            "test_read_whole_is_reached_only_at_a_first_read_a_reset_or_a_restore",
            speed="packages/core/tests/integration/test_scope_epochs_incremental.py::test_read_whole_speed_at_real_size",
        )
        self._rows = {str(row["scope"]): row for row in rows}
        self._since = int(record) if record is not None else 0
        built = _Building()
        for row in rows:
            built.apply(row)
        self._held = built.snapshot()
        return self._held

    async def _read_changed(self) -> list[dict[str, Any]]:
        """the rows of what changed since the last read: :data:`WHOLE`, every pending scope, and every scope moved past it.

        :data:`WHOLE` first (``later`` 0, every scope 1), so it is in the first statement whatever the collation sorts
        first. One statement but for a change of more scopes than a page.

        :return: the rows, :data:`WHOLE`'s first
        :rtype: list[dict[str, Any]]
        """
        columns = ", ".join(quote_identifier(column) for column in _COLUMNS)
        scope, pending, epoch = (quote_identifier(column) for column in ("scope", "pending", "epoch"))
        # the version is an int this class read; the record's name a constant
        select = (
            f"SELECT {quote_identifier('later')}, {columns} FROM ("  # noqa: S608 - the owner's table name
            f"SELECT CASE WHEN {scope} = '{WHOLE}' THEN 0 ELSE 1 END AS {quote_identifier('later')}, {columns} "
            f"FROM {self._table} WHERE {scope} = '{WHOLE}' OR {pending} IS NOT NULL OR {epoch} > {int(self._since)}"
            f") AS {quote_identifier('changed')}"
        )
        read = await read_keyset_pages(
            self._collection.required_l3_pool, select, ("later", "scope"), page_size=self._page_size
        )
        return [{column: row[column] for column in _COLUMNS} for row in read.rows]

    def _distrust(self, held: EpochSnapshot, changed: list[dict[str, Any]]) -> str | None:
        """why the rows of a change cannot be merged into the rows held, or None when they can.

        :param held: the snapshot held
        :ptype held: EpochSnapshot
        :param changed: the rows :meth:`_read_changed` read
        :ptype changed: list[dict[str, Any]]
        :return: the why (:data:`_WHOLE_READS`, or :data:`_REREAD_BY_KEY`), or None
        :rtype: str | None
        """
        record = next((row for row in changed if str(row["scope"]) == WHOLE), None)
        kept = self._rows.get(WHOLE)
        if kept is not None and (record is None or _instant(record["date_created"]) != _instant(kept["date_created"])):
            return "reset"
        if kept is not None and record is not None and int(record["epoch"]) < int(kept["epoch"]):
            return "version_backwards"
        for row in changed:
            before = self._rows.get(str(row["scope"]))
            if before is not None and int(row["epoch"]) < int(before["epoch"]):
                return "epoch_backwards"
        read = {str(row["scope"]) for row in changed}
        if not held.pending <= read:
            return "pending_lost"
        return None

    async def _reread(self, held: EpochSnapshot, changed: list[dict[str, Any]]) -> EpochSnapshot:
        """the snapshot when a change names scopes the rows held cannot be merged with: an epoch gone back, or a pending
        mark cleared without its epoch moving past the last read. Those scopes alone are read again, by key, and the
        snapshot built from the rows held, the change's and theirs: what a whole read answers, read by key.

        :param held: the snapshot held
        :ptype held: EpochSnapshot
        :param changed: the rows :meth:`_read_changed` read
        :ptype changed: list[dict[str, Any]]
        :return: the snapshot
        :rtype: EpochSnapshot
        """
        read = {str(row["scope"]) for row in changed}
        backwards = {
            str(row["scope"])
            for row in changed
            if (before := self._rows.get(str(row["scope"]))) is not None and int(row["epoch"]) < int(before["epoch"])
        }
        named = sorted(backwards | (held.pending - read))
        again = await read_keyed(
            self._collection.required_l3_pool,
            self.table_name,
            keys=KeyBatch(("scope",), [(scope,) for scope in named]),
            columns=_COLUMNS,
        )
        found = {str(row["scope"]): dict(row) for row in again}
        for row in changed:
            self._rows[str(row["scope"])] = row
        for scope in named:
            if scope in found:
                self._rows[scope] = found[scope]
            else:
                self._rows.pop(scope, None)
        record = self._rows.get(WHOLE)
        self._since = int(record["epoch"]) if record is not None else 0
        built = _Building()
        for row in self._rows.values():
            built.apply(row)
        self._held = built.snapshot()
        return self._held

    def _merge(self, held: EpochSnapshot, changed: list[dict[str, Any]]) -> EpochSnapshot:
        """merge the rows of a change into the rows held, and answer the snapshot; the one held when nothing moved.

        :param held: the snapshot held
        :ptype held: EpochSnapshot
        :param changed: the rows :meth:`_read_changed` read, trusted
        :ptype changed: list[dict[str, Any]]
        :return: the snapshot
        :rtype: EpochSnapshot
        """
        record = next((row for row in changed if str(row["scope"]) == WHOLE), None)
        self._since = int(record["epoch"]) if record is not None else 0
        moved = [row for row in changed if self._rows.get(str(row["scope"])) != row]
        if not moved:
            return held
        built = _Building.of(held)
        for row in moved:
            self._rows[str(row["scope"])] = row
            built.apply(row)
        self._held = built.snapshot()
        return self._held

    async def settled(self) -> EpochSnapshot | Unsettled:
        """the snapshot, or :class:`Unsettled` (which write, how long it has run, what it touched) while one is in progress.

        The :class:`Unsettled` names every scope's epoch and the scopes the write has touched, so a reader may
        read a scope the write has not touched while it runs (a touch commits with the write's first rows there).

        A write in progress whose writer is gone (its :class:`WriterLease` free) is abandoned first, and the
        snapshot read again.

        :return: the snapshot when no write is in progress
        :rtype: EpochSnapshot | Unsettled
        """
        snapshot = await self.snapshot()
        if snapshot.writing is not None and await self._writer_gone() and await self._abandon(snapshot.writing):
            snapshot = await self.snapshot()
        settled: EpochSnapshot | Unsettled = snapshot
        if snapshot.writing is not None:
            since = snapshot.writing_since
            seconds = None if since is None else round((datetime.now(UTC) - since).total_seconds(), 1)
            settled = Unsettled(
                f"write {snapshot.writing} in progress for {seconds} s",
                epochs=snapshot.epochs,
                pending=snapshot.pending,
                version=snapshot.version,
            )
            stalled = seconds is not None and seconds > STALLED_WRITE_SECONDS
            # a write that began long ago and never committed is one that died: say so, with its age
            log.log(
                logging.WARNING if stalled else logging.INFO,
                "a write is in progress; readers keep what they hold"
                + (
                    "; it has run so long it has most likely died, and the next write supersedes it"
                    if stalled and self._writer is None
                    else "; it has run long, and its writer still holds its lease"
                    if stalled
                    else ""
                ),
                extra={
                    "extra_data": {
                        "table": self._collection.table_name,
                        "version": snapshot.writing,
                        "seconds": seconds,
                    }
                },
            )
        return settled

    async def _writer_gone(self) -> bool:
        """whether the writers' lease is free, so no writer writes now; False without a lease, or when it cannot be read.

        :return: True only when the lease was read and is free
        :rtype: bool
        """
        if self._writer is None:
            return False
        try:
            return await self._writer.lease.held(self._writer.key) is None
        except Exception as exc:  # a lease that cannot be read proves nothing about its writer: the write stands
            contain(exc, log=log, what="the writers' lease could not be read; the write in progress stands")
            return False

    async def _abandon(self, version: int) -> bool:
        """abandon write ``version``, whose writer is gone: commit what it left, in a transaction of its own.

        The version row is locked without waiting: a data transaction still open (of this write, or a ``begin``
        racing) holds it, and then the writer is not provably gone and nothing is done. Under the lock the write must
        still be the one in progress; a later ``begin`` superseded it otherwise.

        :param version: the write in progress
        :ptype version: int
        :return: True when it was abandoned
        :rtype: bool
        """
        abandoned = False
        try:
            async with self._collection.required_l3_pool.acquire() as conn, CallerTransaction(conn):
                row = await read_row(
                    conn,
                    self.table_name,
                    by={"scope": WHOLE},
                    columns=("writing",),
                    records=True,
                    lock=Lock.UPDATE_NOWAIT,
                )
                if row is not None and row["writing"] is not None and int(row["writing"]) == version:
                    log.warning(
                        "a write in progress is abandoned: its writer's lease is free, so no writer writes it; the "
                        "scopes it touched move to its version, and its writer, should it come back, is refused",
                        extra={"extra_data": {"table": self._collection.table_name, "version": version}},
                    )
                    await self.commit(version, (), conn=conn)
                    abandoned = True
        except asyncpg.exceptions.LockNotAvailableError:
            # NOSILENT: logged at info; a data transaction still open holds the row, so its writer may yet write
            log.info(
                "a write whose writer's lease is free still has a transaction open; it is not abandoned",
                extra={"extra_data": {"table": self._collection.table_name, "version": version}},
            )
        return abandoned

    async def begin(self, *, conn: Any) -> int:
        """take the next version for a write, and record the write as in progress.

        Under the row's lock, taken without waiting: a ``begin`` racing another, or behind a data
        transaction still open, is refused at once, so two writes never share a number. The latest
        supersedes any write still in progress (only it may commit). The number is above both
        the last committed version and any write begun before, so no number a partial write may have
        stamped rows with is used again. Commit the caller's transaction before the write's first
        data row: until then a reader cannot tell the write has begun.

        :param conn: the caller's connection, its transaction opened by ``CallerTransaction``
        :ptype conn: Any
        :return: the write's version
        :rtype: int
        :raises EpochRecordBusyError: when a data transaction of an earlier write, or a racing
            ``begin``, holds the version row (refused at once, never waited on)
        """
        transaction = CallerTransaction.join(conn, writer="ScopeEpochs.begin")
        await self._bound_idle(conn)
        try:
            # the record's lock, taken without waiting: a data transaction left open by a stale writer
            # holds it, and waiting on it would stall this write for as long as that one hangs
            await read_row(
                conn, self.table_name, by={"scope": WHOLE}, columns=("writing",), records=True, lock=Lock.UPDATE_NOWAIT
            )
        except asyncpg.exceptions.LockNotAvailableError as exc:
            # the one type a refused NOWAIT arrives as, on a direct pool and through the L3 broker alike
            raise EpochRecordBusyError(
                f"{self._collection.table_name}: a data transaction of an earlier write, or a racing begin, "
                "holds the version row; no write may begin until it ends"
            ) from exc
        row = await conn.fetchrow(
            f"INSERT INTO {self._table} (scope, epoch, writing, date_created, date_updated) "  # noqa: S608 - the owner's table name
            "VALUES ($1, 0, 1, now(), now()) "
            "ON CONFLICT (scope) DO UPDATE SET "
            f"writing = GREATEST({self._table}.epoch, COALESCE({self._table}.writing, 0)) + 1, date_updated = now() "
            "RETURNING writing",
            WHOLE,
        )
        transaction.enroll(self._collection, WHOLE)
        version = int(row["writing"])
        log.info("write begun", extra={"extra_data": {"table": self._collection.table_name, "version": version}})
        return version

    async def touch(self, version: int, scopes: Iterable[str], *, conn: Any) -> None:
        """record, in the transaction of a data write, that the write changed ``scopes``.

        Fenced as :meth:`commit` is: in the same transaction, the write must still be the one in
        progress, or this raises and the caller's data transaction rolls back with it. So a writer
        whose lease lapsed, and was followed by another that committed, cannot land rows under a
        scope's epoch that will never move for them. The fence holds the version row's share lock
        until the data transaction ends, so no later write can begin, and this write cannot commit, in
        between; the write's other data transactions share it and run alongside.

        The next commit moves every touched scope, whichever write touched it: a write that dies
        after committing some of its rows leaves them covered by the next write's epochs.

        :param version: the write's version
        :ptype version: int
        :param scopes: the scopes the data write changed
        :ptype scopes: Iterable[str]
        :param conn: the data write's connection, its transaction opened by ``CallerTransaction``
        :ptype conn: Any
        :return: nothing
        :rtype: None
        :raises ValueError: when ``version`` is no longer the write in progress, or a scope is
            :data:`WHOLE`
        """
        named = _named(scopes)
        CallerTransaction.join(conn, writer="ScopeEpochs.touch")
        await self._bound_idle(conn)
        # the row's share lock, held to the end of the data transaction: it conflicts with begin's
        # FOR UPDATE NOWAIT and commit's UPDATE, not with this write's other data transactions
        fenced = await read_row(
            conn,
            self.table_name,
            by={"scope": WHOLE, "writing": version},
            columns=("writing",),
            records=True,
            lock=Lock.SHARE,
        )
        if fenced is None:
            raise ValueError(
                f"write {version} is no longer in progress (a later write began); its data write is refused"
            )
        await self._mark_pending(version, named, conn=conn)

    async def _mark_pending(self, version: int, scopes: list[str], *, conn: Any) -> None:
        """mark ``scopes`` pending on the caller's transaction, a batch of statements at a time.

        :param version: the write's version
        :ptype version: int
        :param scopes: the scopes, distinct, :data:`WHOLE` excluded
        :ptype scopes: list[str]
        :param conn: the caller's connection
        :ptype conn: Any
        :return: nothing
        :rtype: None
        """
        transaction = CallerTransaction.join(conn, writer="ScopeEpochs.touch")
        for start in range(0, len(scopes), _TOUCH_BATCH):
            batch = scopes[start : start + _TOUCH_BATCH]
            values = ", ".join(f"(${index + 2}, 0, $1, now(), now())" for index in range(len(batch)))
            await conn.execute(
                f"INSERT INTO {self._table} (scope, epoch, pending, date_created, date_updated) "  # noqa: S608 - the owner's table name
                f"VALUES {values} "
                f"ON CONFLICT (scope) DO UPDATE SET pending = GREATEST(COALESCE({self._table}.pending, 0), $1)",
                version,
                *batch,
            )
        for scope in scopes:
            transaction.enroll(self._collection, scope)

    async def commit(self, version: int, scopes: Iterable[str], *, conn: Any) -> None:
        """record the write as committed and move every scope it changed, or touched before, to its version.

        In the caller's transaction, with the write's last data write or after it commits; never
        before. Refused, and nothing moves (raise inside the transaction to roll it back), when
        ``version`` is no longer the write in progress or a scope to move is already at or past it.

        :param version: the write's version, as :meth:`begin` answered it
        :ptype version: int
        :param scopes: scopes the write changed and did not :meth:`touch`
        :ptype scopes: Iterable[str]
        :param conn: the caller's connection, its transaction opened by ``CallerTransaction``
        :ptype conn: Any
        :return: nothing
        :rtype: None
        :raises ValueError: when ``version`` is not the write in progress, a scope is :data:`WHOLE`,
            or a scope to move is already at or past ``version``
        """
        transaction = CallerTransaction.join(conn, writer="ScopeEpochs.commit")
        named = _named(scopes)
        # the fence: clears the marker only while this write is still the one in progress
        fenced = await conn.fetchrow(
            f"UPDATE {self._table} SET previous_epoch = epoch, epoch = $1, writing = NULL, "  # noqa: S608 - the owner's table name
            f"date_updated = now() WHERE {self._key_column} = $2 AND writing = $1 RETURNING epoch",
            version,
            WHOLE,
        )
        if fenced is None:
            raise ValueError(
                f"commit of version {version} refused: it is not the write in progress (a later write "
                "began, or this one never did); nothing moved"
            )
        transaction.enroll(self._collection, WHOLE)
        await self._mark_pending(version, named, conn=conn)
        behind = await conn.fetch(
            f"SELECT scope FROM {self._table} WHERE pending IS NOT NULL AND scope <> $2 AND epoch >= $1 "  # noqa: S608 - the owner's table name
            "ORDER BY scope LIMIT 5",
            version,
            WHOLE,
        )
        if behind:
            raise ValueError(
                f"commit of version {version} refused: {[str(r['scope']) for r in behind]} are already at or past it"
            )
        # the scopes it moves, read first in pages under the rail's row cap: an UPDATE ... RETURNING of them answers one
        # row a scope, which a write of more scopes than the cap (cobalt prod moved 531 in one write, 2026-11-07)
        # would have cut without saying so, and the scopes past the cut would never be enrolled
        moved: list[str] = []
        page: list[Any] = [None] * DEFAULT_PAGE_SIZE
        while len(page) == DEFAULT_PAGE_SIZE:
            after = "" if not moved else "AND scope > $2 "
            page = await conn.fetch(
                f"SELECT scope FROM {self._table} WHERE pending IS NOT NULL AND scope <> $1 {after}"  # noqa: S608 - the owner's table name
                f"ORDER BY scope LIMIT {DEFAULT_PAGE_SIZE}",
                WHOLE,
                *((moved[-1],) if moved else ()),
            )
            moved += [str(row["scope"]) for row in page]
        # only the scopes read: whatever else is pending by the UPDATE stays pending for the next commit, never moved
        # unenrolled (the fence keeps a touch out until this transaction ends; this holds the commit to what it read)
        await conn.execute(
            f"UPDATE {self._table} SET previous_epoch = NULLIF(epoch, 0), epoch = $1, pending = NULL, "  # noqa: S608 - the owner's table name
            "date_updated = now() WHERE pending IS NOT NULL AND scope <> $2 AND scope = ANY($3::text[])",
            version,
            WHOLE,
            moved,
        )
        for scope in moved:
            transaction.enroll(self._collection, scope)
        log.info(
            "write committed and its scopes' epochs moved",
            extra={"extra_data": {"table": self._collection.table_name, "version": version, "scopes": len(moved)}},
        )

    def on_change(self, listener: Callable[[], None]) -> Callable[[], None]:
        """call ``listener`` whenever an epoch changes: this process's commit, or a peer's broadcast.

        It runs synchronously inside the change and must not raise; schedule work, do not do it.

        :param listener: called with no arguments
        :ptype listener: Callable[[], None]
        :return: a call that removes the listener
        :rtype: Callable[[], None]
        """
        return self._collection.add_l1_change_listener(lambda _scope: listener())


class _Building:
    """a snapshot being built from the rows read: every row of a whole read, or a held snapshot and the rows of a change.

    One way to read a row for both, so a read of the changes answers what a whole read would.
    """

    def __init__(self) -> None:
        self.version = 0
        self.writing: int | None = None
        self.writing_since: datetime | None = None
        self.epochs: dict[str, int] = {}
        self.previous: dict[str, int | None] = {}
        self.pending: set[str] = set()

    @classmethod
    def of(cls, held: EpochSnapshot) -> _Building:
        """a building that starts from ``held``.

        :param held: the snapshot held
        :ptype held: EpochSnapshot
        :return: the building
        :rtype: _Building
        """
        built = cls()
        built.version, built.writing, built.writing_since = held.version, held.writing, held.writing_since
        built.epochs, built.previous, built.pending = dict(held.epochs), dict(held.previous), set(held.pending)
        return built

    def apply(self, row: Mapping[str, Any]) -> None:
        """take one row as read, replacing what the building held of its scope.

        :param row: the row
        :ptype row: Mapping[str, Any]
        :return: nothing
        :rtype: None
        """
        scope = str(row["scope"])
        previous = None if row["previous_epoch"] is None else int(row["previous_epoch"])
        if scope == WHOLE:
            self.version = int(row["epoch"])
            self.writing = None if row["writing"] is None else int(row["writing"])
            self.writing_since = None if self.writing is None else _instant(row["date_updated"])
            self.previous[scope] = previous
            return
        if row["pending"] is not None:
            self.pending.add(scope)
        else:
            self.pending.discard(scope)
        if int(row["epoch"]) > 0:
            self.epochs[scope] = int(row["epoch"])
            self.previous[scope] = previous
        else:
            # a scope only touched has not moved yet: it has no epoch to name
            self.epochs.pop(scope, None)
            self.previous.pop(scope, None)

    def snapshot(self) -> EpochSnapshot:
        """the snapshot built.

        :return: the snapshot
        :rtype: EpochSnapshot
        """
        return EpochSnapshot(
            version=self.version,
            writing=self.writing,
            epochs=self.epochs,
            previous=self.previous,
            writing_since=self.writing_since,
            pending=frozenset(self.pending),
        )


def _instant(value: Any) -> datetime:
    """a timestamp as read: a datetime from Postgres, ISO text over the L3 rail.

    :param value: the value read
    :ptype value: Any
    :return: the instant
    :rtype: datetime
    """
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def _named(scopes: Iterable[str]) -> list[str]:
    """the scopes, each once, in order; :data:`WHOLE` refused.

    :param scopes: the scopes
    :ptype scopes: Iterable[str]
    :return: the sorted distinct scopes
    :rtype: list[str]
    :raises ValueError: when :data:`WHOLE` is among them
    """
    named = sorted(set(scopes))
    if WHOLE in named:
        raise ValueError(f"{WHOLE!r} is reserved for the data's version; it is not a scope")
    return named
