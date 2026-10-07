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

**One writer at a time.** ``begin`` and ``commit`` assume the caller holds the write's lease (a
:class:`~threetears.core.coordination.coalesced_run.CoalescedRun`); a commit for a write that is not
the one in progress is refused.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from threetears.observe import get_logger

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
_COLUMNS: Final = ("scope", "epoch", "previous_epoch", "writing")


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
    """

    version: int
    writing: int | None
    epochs: Mapping[str, int]
    previous: Mapping[str, int | None]

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
        epochs: dict[str, int] = {}
        previous: dict[str, int | None] = {}
        for row in rows:
            scope = str(row["scope"])
            previous[scope] = None if row["previous_epoch"] is None else int(row["previous_epoch"])
            if scope == WHOLE:
                version = int(row["epoch"])
                writing = None if row["writing"] is None else int(row["writing"])
            else:
                epochs[scope] = int(row["epoch"])
        return EpochSnapshot(version=version, writing=writing, epochs=epochs, previous=previous)

    async def settled(self) -> EpochSnapshot | None:
        """the snapshot, or None while a write is in progress.

        :return: the snapshot when no write is in progress
        :rtype: EpochSnapshot | None
        """
        snapshot = await self.snapshot()
        return None if snapshot.writing is not None else snapshot

    async def begin(self, *, conn: Any) -> int:
        """take the next version for a write, and record the write as in progress.

        Commit the caller's transaction before the write's first data row: until then a reader
        cannot tell the write has begun.

        :param conn: the caller's connection, its transaction opened by ``CallerTransaction``
        :ptype conn: Any
        :return: the write's version
        :rtype: int
        """
        current = await self.snapshot()
        version = max(current.version, current.writing or 0) + 1
        await self._collection.save_rows(
            [
                {
                    "scope": WHOLE,
                    "epoch": current.version,
                    "previous_epoch": current.previous.get(WHOLE),
                    "writing": version,
                }
            ],
            conn=conn,
        )
        log.info("write begun", extra={"extra_data": {"table": self._collection.table_name, "version": version}})
        return version

    async def commit(self, version: int, scopes: Iterable[str], *, conn: Any) -> None:
        """record the write as committed and move every scope it changed to its version.

        In the transaction of the write's last data write, or after it commits; never before.

        :param version: the write's version, as :meth:`begin` answered it
        :ptype version: int
        :param scopes: every scope whose data the write changed
        :ptype scopes: Iterable[str]
        :param conn: the caller's connection, its transaction opened by ``CallerTransaction``
        :ptype conn: Any
        :return: nothing
        :rtype: None
        :raises ValueError: when ``version`` is not the write in progress, a scope is
            :data:`WHOLE`, or a scope's epoch is already at or past ``version``
        """
        moved = sorted(set(scopes))
        if WHOLE in moved:
            raise ValueError(f"{WHOLE!r} is reserved for the data's version; it is not a scope")
        current = await self.snapshot()
        if current.writing != version:
            raise ValueError(
                f"commit of version {version} refused: the write in progress is {current.writing}; "
                "only the write that began may commit"
            )
        behind = [scope for scope in moved if current.epoch(scope) >= version]
        if behind:
            raise ValueError(f"commit of version {version} refused: {behind[:5]} are already at or past it")
        rows = [{"scope": WHOLE, "epoch": version, "previous_epoch": current.version, "writing": None}]
        rows += [
            {"scope": scope, "epoch": version, "previous_epoch": current.epochs.get(scope), "writing": None}
            for scope in moved
        ]
        await self._collection.save_rows(rows, conn=conn)
        log.info(
            "write committed and its scopes' epochs moved",
            extra={
                "extra_data": {
                    "table": self._collection.table_name,
                    "version": version,
                    "scopes": len(moved),
                }
            },
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
