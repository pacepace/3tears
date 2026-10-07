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
(:meth:`ScopeEpochs.settled`, the ``settled`` of
:class:`~threetears.core.collections.complete_copy.BufferedCopies`) read one state of all of them.
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
in one statement under the row's lock (two writes begun together take two numbers), and the latest
``begin`` supersedes any write still in progress. ``commit`` moves anything only while its version is
still the write in progress, checked in the same statement that clears it, inside the caller's
transaction: a writer whose lease lapsed, and was followed by another, is refused and its
transaction moves nothing.

**Scopes a write touched are recorded as it writes.** :meth:`ScopeEpochs.touch`, in the transaction
of each data write, marks the scopes that write changed as pending. ``commit`` moves every pending
scope, the ones an abandoned write left included: rows a write committed before it died are then
covered by the next write's epochs, even when that write does not change those scopes itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from threetears.observe import get_logger

from threetears.core.cache.base import quote_identifier
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE, read_l3_rows
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
    collection_for_schema,
)
from threetears.core.entities.base import BaseEntity

__all__ = [
    "SCOPE_EPOCHS_TABLE",
    "STALLED_WRITE_SECONDS",
    "WHOLE",
    "EpochSnapshot",
    "ScopeEpochs",
    "scope_epochs_collection",
    "scope_epochs_schema",
]

log = get_logger(__name__)

#: the table's default name
SCOPE_EPOCHS_TABLE: Final = "scope_epochs"

#: the row holding the data's version and any write in progress; not a scope a caller may name
WHOLE: Final = "*"

#: the columns a snapshot reads
_COLUMNS: Final = ("scope", "epoch", "previous_epoch", "writing", "date_updated")

#: scopes one statement names when touching
_TOUCH_BATCH: Final = 500

#: a write in progress longer than this has most likely died: readers say so at warning, not info
STALLED_WRITE_SECONDS: Final = 600


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
class EpochSnapshot:
    """every scope's epoch, and the data's version, as read at one moment.

    :ivar version: the last committed write's version; 0 before any
    :ivar writing: the version of a write begun and not committed, or None
    :ivar epochs: each scope's epoch, by scope (:data:`WHOLE` left out)
    :ivar previous: the epoch each scope's last move replaced (None after its first move), by scope;
        :data:`WHOLE`'s is the version before the current one
    :ivar writing_since: when the write in progress began; None when none is
    """

    version: int
    writing: int | None
    epochs: Mapping[str, int]
    previous: Mapping[str, int | None]
    writing_since: datetime | None = None

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
    """

    def __init__(self, collection: SchemaBackedCollection[Any], *, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        self._collection = collection
        self._page_size = page_size
        self._table = quote_identifier(collection.table_name)
        self._why_unsettled = "no write was in progress when last read"

    async def snapshot(self) -> EpochSnapshot:
        """every scope's epoch and the data's version, read from L3.

        :return: the snapshot
        :rtype: EpochSnapshot
        """
        rows = await read_l3_rows(
            self._collection.required_l3_pool,
            self._collection.table_name,
            _COLUMNS,
            ("scope",),
            page_size=self._page_size,
        )
        version = 0
        writing: int | None = None
        writing_since: datetime | None = None
        epochs: dict[str, int] = {}
        previous: dict[str, int | None] = {}
        for row in rows:
            scope = str(row["scope"])
            if scope == WHOLE:
                version = int(row["epoch"])
                writing = None if row["writing"] is None else int(row["writing"])
                writing_since = None if writing is None else _instant(row["date_updated"])
                previous[scope] = None if row["previous_epoch"] is None else int(row["previous_epoch"])
            elif int(row["epoch"]) > 0:
                # a scope only touched has not moved yet: it has no epoch to name
                epochs[scope] = int(row["epoch"])
                previous[scope] = None if row["previous_epoch"] is None else int(row["previous_epoch"])
        return EpochSnapshot(
            version=version, writing=writing, epochs=epochs, previous=previous, writing_since=writing_since
        )

    async def settled(self) -> EpochSnapshot | None:
        """the snapshot, or None while a write is in progress.

        :return: the snapshot when no write is in progress
        :rtype: EpochSnapshot | None
        """
        snapshot = await self.snapshot()
        settled: EpochSnapshot | None = snapshot
        if snapshot.writing is not None:
            settled = None
            since = snapshot.writing_since
            seconds = None if since is None else round((datetime.now(UTC) - since).total_seconds(), 1)
            self._why_unsettled = f"write {snapshot.writing} in progress for {seconds} s"
            stalled = seconds is not None and seconds > STALLED_WRITE_SECONDS
            # a write that began long ago and never committed is one that died: say so, with its age
            log.log(
                logging.WARNING if stalled else logging.INFO,
                "a write is in progress; readers keep what they hold"
                + ("; it has run so long it has most likely died, and the next write supersedes it" if stalled else ""),
                extra={
                    "extra_data": {
                        "table": self._collection.table_name,
                        "version": snapshot.writing,
                        "seconds": seconds,
                    }
                },
            )
        return settled

    def why_unsettled(self) -> str:
        """what the last :meth:`settled` that answered None found: which write, and how long it has run.

        For a reader refused by a write in progress to say why (``BufferedCopies``' ``why_unsettled``).

        :return: the description
        :rtype: str
        """
        return self._why_unsettled

    async def begin(self, *, conn: Any) -> int:
        """take the next version for a write, and record the write as in progress.

        One statement under the row's lock, so two writes begun together take two numbers; the
        latest supersedes any write still in progress (only it may commit). The number is above both
        the last committed version and any write begun before, so no number a partial write may have
        stamped rows with is used again. Commit the caller's transaction before the write's first
        data row: until then a reader cannot tell the write has begun.

        :param conn: the caller's connection, its transaction opened by ``CallerTransaction``
        :ptype conn: Any
        :return: the write's version
        :rtype: int
        """
        transaction = CallerTransaction.join(conn, writer="ScopeEpochs.begin")
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
        scope's epoch that will never move for them. The fence holds the version row's lock until the
        data transaction ends, so no later write can begin in between.

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
        # the row's lock, held to the end of the data transaction; nothing about the row changes
        fenced = await conn.fetchrow(
            f"UPDATE {self._table} SET writing = writing WHERE scope = $2 AND writing = $1 RETURNING writing",  # noqa: S608 - the owner's table name
            version,
            WHOLE,
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
            "date_updated = now() WHERE scope = $2 AND writing = $1 RETURNING epoch",
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
        moved = await conn.fetch(
            f"UPDATE {self._table} SET previous_epoch = NULLIF(epoch, 0), epoch = $1, pending = NULL, "  # noqa: S608 - the owner's table name
            "date_updated = now() WHERE pending IS NOT NULL AND scope <> $2 RETURNING scope",
            version,
            WHOLE,
        )
        for row in moved:
            transaction.enroll(self._collection, str(row["scope"]))
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
