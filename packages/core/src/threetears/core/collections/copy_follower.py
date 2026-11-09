"""A complete copy of collection tables in DuckDB, kept current by the keys each epoch changed.

**Why.** A pod that answers analytic queries over whole tables holds every row of them in a DuckDB L1
(:mod:`~threetears.core.collections.complete_copy`). :class:`~threetears.core.collections.complete_copy.CompleteCopy`
is voided by any change and warmed whole again, so a change costs the size of the table. A copy kept by
:class:`CopyFollower` costs what changed: it hears the keys of every committed write of its tables through the
registry's standard path (each row broadcast, and each epoch's change record a catch-up applies:
:meth:`~threetears.core.collections.registry.CollectionRegistry.register_derived_cache`), reads exactly those keys'
rows from L3, and swaps them in. Nothing here is a side system: the keys come from the epoch change log, the rows
from L3, and the copy is the registry's L1 backend's own tables.

**One state of every table.** A change is applied only at a settled stamp of the writer's seqlock (``settled``, as
:class:`~threetears.core.collections.complete_copy.BufferedCopies` takes it; on the reports pod ``ScopeEpochs``): the
stamp is read, every table is caught up to its generation read AFTER it, the keys heard are read from L3 and staged
beside the copy (:meth:`~threetears.core.cache.duckdb.DuckDBBackend.stage_rows`), the stamp is read again, and only
when it has not moved is everything staged applied in ONE DuckDB transaction
(:meth:`~threetears.core.cache.duckdb.DuckDBBackend.apply_staged`) and the copy labelled with the stamp. A stamp that
moved, or a write in progress, keeps every key pending for the next pass and the copy as it was: stale, never mixed.
The label is exact when the writer's seqlock commit comes after the generation advances of the data it covers (a
writer whose last data transaction commits the stamp itself races that transaction's advance).

**Bounded.** What this process holds of a change is a page of rows at a time: each page is staged as it is read.
The keys pending are the change's. A first warm reads each table a page at a time, by an order an index serves
(:attr:`FollowedTable.order`).

**Never a drop, never a rebuild.** An epoch the log cannot give (lost, aged out, keys unknown) leaves the registry
behind on the table (``CollectionRegistry.behind_on``): the copy keeps the stamp it has and says it is behind until
the log is put back (``ChangeNoteKeeper``) and the catch-up applies it. A record that names the whole table (a
truncate, a file load) reads that table again beside the copy and swaps it in whole, in the same one transaction as
everything else.

**Start.** :meth:`CopyFollower.start` registers with the registry and takes each table's mark: from a checkpoint
(:mod:`~threetears.core.collections.copy_checkpoint`), whose marks the registry adopts so the catch-up walks the log
from them, or, when there is none, from the head, after which every table is read whole once (the first start only).
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from decimal import Decimal
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Protocol

from threetears.observe import contain, get_logger

from threetears.core.cache.duckdb import STAGE_PREFIX, StagedChange
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE, Unsettled, rail_arrow_types
from threetears.core.collections.registry import CacheInvalidationMessage
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    NUMERIC_TYPE,
    UUID_TYPE,
    Column,
    TableSchema,
)
from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.backends.broker_generation import GENERATION_BY_TRIGGER
from threetears.core.keyset import keyset_arrow_pages
from threetears.core.sql_fragments import quote_identifier

if TYPE_CHECKING:
    from threetears.core.cache.duckdb import DuckDBBackend
    from threetears.core.collections.copy_checkpoint import CopyCheckpoints
    from threetears.core.collections.epoch_change import EpochChangeReader
    from threetears.core.collections.registry import CollectionRegistry

try:
    import pyarrow as pa
except ImportError:  # NOSILENT: the snapshot extra is optional; a follower says so when built
    pa = None

__all__ = [
    "CopyFollower",
    "HubCapabilityMissingError",
    "FollowedTable",
    "FollowerBackend",
    "FollowerPhase",
    "FollowerStatus",
    "require_generation_by_trigger",
]

log = get_logger(__name__)

#: how many applied changes' stamps are kept for readers whose snapshot is older than the newest
_STAMPS_KEPT: Final = 256

#: the SQL type a key's text is cast to before it is compared with its column, by the column's type tag
_KEY_CASTS: Final[Mapping[str, str]] = {
    UUID_TYPE: "uuid",
    INT_TYPE: "bigint",
    BIGINT_TYPE: "bigint",
    BOOL_TYPE: "boolean",
    NUMERIC_TYPE: "numeric",
    DATETIMETZ_TYPE: "timestamptz",
}


class HubCapabilityMissingError(RuntimeError):
    """the hub's broker lacks a capability a copy follower needs: the copy would be silently stale, so it never serves.

    :ivar capability: the capability's name
    """

    def __init__(self, capability: str) -> None:
        self.capability = capability
        super().__init__(
            f"the hub does not announce {capability!r}: it advances no generation for a pod's own tables switched on "
            f"by their trigger, so a copy following them would never hear a change. Deploy the hub that announces it "
            f"before this pod; the pod stays not ready until then"
        )


async def require_generation_by_trigger(l3: Any) -> None:
    """refuse, before a copy follower serves, a hub whose broker does not advance trigger-switched tables.

    An L3 reached through the hub's broker (anything with ``hub_capabilities``) must announce
    :data:`~threetears.core.backends.broker_generation.GENERATION_BY_TRIGGER`. An L3 reached directly (a SQL pool, a
    test's) has no broker between: its writers advance their own generations, and nothing is asked.

    :param l3: the pod's L3
    :ptype l3: Any
    :return: nothing
    :rtype: None
    :raises HubCapabilityMissingError: when the hub does not announce it
    """
    probe = getattr(l3, "hub_capabilities", None)
    if probe is None:
        return
    if GENERATION_BY_TRIGGER not in await probe():
        raise HubCapabilityMissingError(GENERATION_BY_TRIGGER)


def _typed(kind: str, text: str) -> Any:
    """a value given as text (a record's partition) as the L3 rail binds its column's type.

    :param kind: the column's type tag
    :ptype kind: str
    :param text: the value
    :ptype text: str
    :return: the value
    :rtype: Any
    """
    if kind in (INT_TYPE, BIGINT_TYPE):
        return int(text)
    if kind == BOOL_TYPE:
        return text.lower() in ("true", "t", "1")
    if kind == NUMERIC_TYPE:
        return Decimal(text)
    if kind == DATETIMETZ_TYPE:
        return datetime.fromisoformat(text)
    return text


@dataclass(frozen=True, slots=True)
class FollowedTable:
    """one table a :class:`CopyFollower` keeps whole.

    :ivar name: the table, as L3 and the copy both name it
    :ivar key: its primary key's columns, in declared order: the order a change record names a key's values in
    :ivar columns: the columns the copy holds, each read from L3 (``name`` and ``column_type``); every key column is one
    :ivar order: a unique column order an index of L3 serves, for the first warm's pages; the key when empty
    :ivar partition_order: a unique column order an index led by a load's constant columns serves, for the pages of a
        load's rows (a run's, a state's at a version); the key when empty
    """

    name: str
    key: tuple[str, ...]
    columns: tuple[Column, ...]
    order: tuple[str, ...] = ()
    partition_order: tuple[str, ...] = ()
    #: every column L3 holds and its type tag, the copy's or not: a load's partition may name a column the copy does
    #: not hold (a version stamp), which the read of its rows still filters on
    l3_kinds: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        held = {column.name for column in self.columns}
        missing = [column for column in self.key if column not in held]
        if missing:
            raise ValueError(f"{self.name}: the copy must hold every key column; {missing} is not among its columns")

    @classmethod
    def of(
        cls, schema: TableSchema, *, columns: Sequence[str] | None = None, order: Sequence[str] = ()
    ) -> FollowedTable:
        """a table as its schema declares it.

        :param schema: the table's schema
        :ptype schema: TableSchema
        :param columns: the columns the copy holds, by name; every column when None
        :ptype columns: Sequence[str] | None
        :param order: a unique column order an index serves, for the first warm
        :ptype order: Sequence[str]
        :return: the table
        :rtype: FollowedTable
        """
        key = schema.primary_key if isinstance(schema.primary_key, tuple) else (schema.primary_key,)
        wanted = None if columns is None else set(columns)
        held = tuple(c for c in schema.columns if wanted is None or c.name in wanted)
        kinds = tuple((c.name, c.column_type) for c in schema.columns)
        return cls(name=schema.name, key=tuple(key), columns=held, order=tuple(order), l3_kinds=kinds)

    @property
    def read_order(self) -> tuple[str, ...]:
        """the columns the first warm pages on.

        :return: :attr:`order`, or the key
        :rtype: tuple[str, ...]
        """
        return self.order or self.key

    def kind(self, column: str) -> str:
        """a column's type tag.

        :param column: a column the copy holds, or one its L3 table holds
        :ptype column: str
        :return: its type tag
        :rtype: str
        :raises ValueError: when neither names it
        """
        for held in self.columns:
            if held.name == column:
                return held.column_type
        for name, kind in self.l3_kinds:
            if name == column:
                return kind
        raise ValueError(f"{self.name}: neither the copy nor its schema names {column!r}")

    def key_cast(self, column: str) -> str:
        """the SQL type a key column's text is cast to.

        :param column: a key column
        :ptype column: str
        :return: the type
        :rtype: str
        """
        kind = next(c.column_type for c in self.columns if c.name == column)
        return _KEY_CASTS.get(kind, "text")


@dataclass(frozen=True, slots=True)
class _KeyedRead:
    """the statement that reads some keys' rows of one table from L3: the keys bound as one text array per key
    column, joined to the table on its key.

    On PostgreSQL and YugabyteDB the join is answered by the primary key, a lookup a key (``EXPLAIN`` shows an
    index scan of the key, never a scan of the table): its cost is the keys', whatever the table holds.

    :ivar sql: the statement
    :ivar types: each column's Arrow type, as the rail answers it
    """

    sql: str
    types: Mapping[str, Any]

    @classmethod
    def of(cls, table: FollowedTable) -> _KeyedRead:
        """the read of ``table``'s rows by key.

        :param table: the table
        :ptype table: FollowedTable
        :return: the read
        :rtype: _KeyedRead
        """
        types = rail_arrow_types(table.columns)
        selected = ", ".join(f"t.{quote_identifier(name)}" for name in types)
        keys = ", ".join(f"unnest(${i + 1}::text[]) AS k{i}" for i in range(len(table.key)))
        joined = " AND ".join(
            f"t.{quote_identifier(column)} = CAST(k.k{i} AS {table.key_cast(column)})"
            for i, column in enumerate(table.key)
        )
        sql = f"SELECT {selected} FROM {quote_identifier(table.name)} AS t JOIN (SELECT {keys}) AS k ON {joined}"  # noqa: S608
        return cls(sql=sql, types=types)


class FollowerPhase(StrEnum):
    """what a follower is doing."""

    STARTING = "starting"
    WARMING = "warming"
    FOLLOWING = "following"
    WAITING = "waiting"
    BEHIND = "behind"
    STOPPED = "stopped"


@dataclass(slots=True)
class FollowerStatus:
    """what a follower says of itself.

    :ivar phase: what it is doing
    :ivar stamp: the writer's stamp the copy holds every table at; None until it holds one
    :ivar behind: the tables whose log the registry is behind on, with why: its reason
        (:data:`~threetears.core.collections.registry.BEHIND_REASONS`), as ``collection_epoch_behind_total`` labels it
    :ivar pending: keys heard and not yet applied, by table
    :ivar rows: each table's rows, as last counted
    :ivar applied: changes applied since start
    :ivar last_apply_seconds: how long the last change took, from the first stamp read to the commit
    :ivar detail: a line for an operator
    """

    phase: FollowerPhase = FollowerPhase.STARTING
    stamp: Any = None
    behind: dict[str, str] = field(default_factory=dict)
    pending: dict[str, int] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    applied: int = 0
    last_apply_seconds: float | None = None
    detail: str = ""


class FollowerBackend(Protocol):
    """what a :class:`CopyFollower` keeps its copy in: a backend that stages a change beside the copy and applies
    everything staged in one transaction (:class:`~threetears.core.cache.duckdb.DuckDBBackend` is one; a pod's own
    backend that publishes the copy to its worker processes is another). Each method is called on a thread.
    """

    def stage_rows(self, stage: str, table: str, rows: Any, key: Sequence[str]) -> int:
        """stage some rows of ``table`` under ``stage``; the rows staged."""
        ...

    def stage_keys(
        self, stage: str, table: str, key: Sequence[str], keys: Any, *, rows_stage: str | None = None
    ) -> int:
        """stage the keys read under ``stage`` (a key not among ``rows_stage``'s rows is deleted); the keys staged."""
        ...

    def drop_stage(self, *stages: str) -> None:
        """drop these stages."""
        ...

    def staged_stages(self) -> list[str]:
        """every stage held."""
        ...

    def apply_staged(self, changes: Sequence[StagedChange], *, mark: int | None = None) -> int:
        """apply every change staged in one transaction, labelled with ``mark``; the rows applied."""
        ...

    def read_snapshot(self) -> AbstractContextManager[Any]:
        """a cursor over one applied state of the copy, for as long as the block runs."""
        ...

    def staged_mark(self, cursor: Any) -> int | None:
        """the mark of the change the state ``cursor`` reads was applied with; None before any."""
        ...


class CopyFollower:
    """keeps tables whole in a DuckDB backend, current to the writer's settled stamp, by the keys each epoch changed.

    :param registry: the registry whose epoch catch-up names the keys; each table is followed there
    :ptype registry: CollectionRegistry
    :param tables: the tables, each created on ``backend`` already
    :ptype tables: Sequence[FollowedTable]
    :param backend: the copy
    :ptype backend: FollowerBackend
    :param l3: L3, the rail's proxy or anything with an asyncpg-shaped ``fetch``
    :ptype l3: Any
    :param settled: the writer's seqlock: a stamp, or :class:`Unsettled` while a write is in progress
    :ptype settled: Callable[[], Awaitable[Any]]
    :param generation: a table's current write generation, or None when it has none
    :ptype generation: Callable[[str], Awaitable[str | None]]
    :param changes: reads the epoch change log; None leaves every unheard epoch behind
    :ptype changes: EpochChangeReader | None
    :param page_size: rows a statement reads; under the rail's cap
    :ptype page_size: int
    :param checkpoints: where the copy's checkpoints are kept (:mod:`~threetears.core.collections.copy_checkpoint`);
        None keeps none, and every start reads L3 whole
    :ptype checkpoints: CopyCheckpoints | None
    """

    def __init__(
        self,
        registry: CollectionRegistry,
        tables: Sequence[FollowedTable],
        backend: FollowerBackend,
        l3: Any,
        *,
        settled: Callable[[], Awaitable[Any]],
        generation: Callable[[str], Awaitable[str | None]],
        changes: EpochChangeReader | None,
        page_size: int = DEFAULT_PAGE_SIZE,
        checkpoints: CopyCheckpoints | None = None,
    ) -> None:
        if pa is None:
            raise ImportError("a copy follower needs pyarrow: install 3tears[snapshot]")
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        self._registry = registry
        self._tables: dict[str, FollowedTable] = {table.name: table for table in tables}
        self._reads = {table.name: _KeyedRead.of(table) for table in tables}
        self._backend = backend
        self._l3 = l3
        self._settled = settled
        self._generation = generation
        self._changes = changes
        self._page_size = page_size
        #: keys heard and not yet applied, by table, each the key's values as text
        self._pending: dict[str, set[tuple[str, ...]]] = {name: set() for name in self._tables}
        #: tables a whole-table change named, read again whole at the next apply
        self._whole: set[str] = set()
        #: the loads' partitions heard and not yet applied, by table: each the columns its rows hold and their values
        self._partitions: dict[str, list[dict[str, str]]] = {name: [] for name in self._tables}
        #: the loads' partitions staged and not yet applied, by table, put back to be read again if a pass fails
        self._partitions_staged: dict[str, list[dict[str, str]]] = {}
        self._lock = asyncio.Lock()
        self._poked = asyncio.Event()
        self._removers: list[Callable[[], None]] = []
        self._status = FollowerStatus()
        self._listeners: list[Callable[[Any], None]] = []
        #: rows applied since :meth:`take_rows_since_mark` was last called (the checkpoint's trigger)
        self._rows_since_mark = 0
        #: what is staged beside the copy and not yet applied, by table, and the keys each stage was read for
        self._staged: dict[str, StagedChange] = {}
        self._staged_keys: dict[str, set[tuple[str, ...]]] = {}
        #: the mark of the last change applied, and the stamp of each recent one (a reader's snapshot names its mark)
        self._mark = 0
        self._stamps: dict[int, tuple[Any, Mapping[str, str | None]]] = {}
        #: each table's mark when the keys being staged were taken: what the next change applied holds
        self._staging_marks: dict[str, str | None] = {}
        self._checkpoints = checkpoints
        self._checkpointing: asyncio.Task[Any] | None = None

    # --- what the registry tells it -------------------------------------------------------------

    def _on_row(self, table: str) -> Callable[[CacheInvalidationMessage], None]:
        def heard(message: CacheInvalidationMessage) -> None:
            self._pending[table].add(tuple(message.ids))
            self._poked.set()

        return heard

    def _on_dropped(self, table: str) -> Callable[[], None]:
        def dropped() -> None:
            self._whole.add(table)
            self._pending[table].clear()
            self._poked.set()

        return dropped

    def _on_partition(self, table: str) -> Callable[[Mapping[str, str]], None]:
        def loaded(partition: Mapping[str, str]) -> None:
            self._partitions[table].append(dict(partition))
            self._poked.set()

        return loaded

    def _on_caught_up(self) -> None:
        self._poked.set()

    # --- start -----------------------------------------------------------------------------------

    def register(self) -> None:
        """hear every table's changes through the registry; idempotent.

        The owner follows each table in the registry and schedules its catch-up pass (``CollectionRegistry
        .follow_generation`` and ``threetears.epoch.generation_catchup_tick``), as for any table it follows: the pass
        applies a missed epoch's rows, which reach this follower as heard keys, and pokes it.

        :return: nothing
        :rtype: None
        :raises ValueError: when a table is not followed in the registry
        """
        if self._removers:
            return
        unfollowed = sorted(name for name in self._tables if not self._registry.generation_marks.follows(name))
        if unfollowed:
            raise ValueError(
                f"follow {unfollowed} in the registry and schedule its catch-up pass before a copy follower hears them"
            )
        for name in self._tables:
            self._removers.append(
                self._registry.register_derived_cache(
                    name,
                    on_row=self._on_row(name),
                    on_table_dropped=self._on_dropped(name),
                    on_caught_up=self._on_caught_up,
                    on_partition=self._on_partition(name),
                )
            )

    def adopt_marks(self, marks: Mapping[str, str | None]) -> None:
        """take each table's mark from a checkpoint: the catch-up walks the log from there.

        Call after :meth:`register` and before the first :meth:`catch_up`, once the checkpoint's rows are the copy's.

        :param marks: each table's generation the checkpoint holds it at
        :ptype marks: Mapping[str, str | None]
        :return: nothing
        :rtype: None
        :raises ValueError: when a table has no mark
        """
        missing = sorted(set(self._tables) - set(marks))
        if missing:
            raise ValueError(f"a checkpoint names no mark for {missing}")
        for name in self._tables:
            self._registry.generation_marks.adopt_generation(name, marks[name])

    async def start(self) -> bool:
        """register, and fill the copy: from the newest checkpoint, whose marks the registry adopts so the next
        :meth:`catch_up` walks the log from them; or, when there is none, by reading every table whole (:meth:`warm`).

        :return: whether it started from a checkpoint
        :rtype: bool
        """
        self.register()
        staged = None if self._checkpoints is None else await self._checkpoints.stage()
        if staged is None:
            await self.warm()
            return False
        stamp, marks, changes, rows = staged
        self._mark += 1
        await asyncio.to_thread(self._backend.apply_staged, changes, mark=self._mark)
        self._stamps[self._mark] = (stamp, dict(marks))
        self.adopt_marks(marks)
        self._status.stamp = stamp
        self._say(FollowerPhase.FOLLOWING, f"loaded a checkpoint of {rows} rows; catching up from its marks")
        self._poked.set()
        return True

    async def warm(self) -> None:
        """the first start: take each table's mark at the head, then read every table whole beside the copy and swap
        it in, in one transaction. A change committed while it reads is heard (the marks were taken first) and applied
        by the next :meth:`catch_up`; the copy is labelled by that catch-up, not here.

        :return: nothing
        :rtype: None
        """
        self._status.phase = FollowerPhase.WARMING
        for name in self._tables:
            token = await self._generation(name)
            await self._registry.catch_up_generation(name, token, changes=self._changes)
        self._whole.update(self._tables)
        self._poked.set()

    def marks(self) -> dict[str, str | None]:
        """each table's mark in the registry now: what a checkpoint of the copy at its stamp records.

        :return: table -> its generation mark
        :rtype: dict[str, str | None]
        """
        return {name: self._registry.generation_marks.recorded(name) for name in self._tables}

    # --- the apply -------------------------------------------------------------------------------

    async def catch_up(self) -> bool:
        """apply every key heard, at one settled stamp; label the copy with it.

        What a pass reads is staged beside the copy and kept there until a pass ends at a stamp that held still while
        it read: a pass the stamp moved under keeps its stages, and the next pass reads only the keys heard since
        (a key named again leaves the stage and is read again), so a long read (a table read whole) is never thrown
        away for the writes it ran beside.

        :return: whether the copy is now current to a settled stamp (False: a write is in progress, the stamp moved
            while it read, or the registry is behind on a table; the copy is as it was)
        :rtype: bool
        """
        async with self._lock:
            began = time.monotonic()
            self._poked.clear()
            first = await self._settled()
            if isinstance(first, Unsettled):
                self._say(FollowerPhase.WAITING, f"a write is in progress: {first.reason}")
                return False
            behind: dict[str, str] = {}
            for name in self._tables:
                token = await self._generation(name)
                await self._registry.catch_up_generation(name, token, changes=self._changes)
                reason = self._registry.behind_reason(name)
                if reason is not None:
                    # the registry's reason (BEHIND_REASONS): the status and collection_epoch_behind_total say the same
                    behind[name] = reason
            self._status.behind = behind
            if behind:
                self._say(FollowerPhase.BEHIND, f"behind on {sorted(behind)}; serving the copy at {self._status.stamp}")
                return False
            pending = {name: keys for name, keys in self._pending.items() if keys}
            whole = set(self._whole)
            partitions = {name: list(parts) for name, parts in self._partitions.items() if parts}
            if not pending and not whole and not partitions and not self._staged and first == self._status.stamp:
                self._say(FollowerPhase.FOLLOWING, "current")
                return True
            # taken with the keys, no await between: every key of an epoch at or below these marks is among them
            self._staging_marks = self.marks()
            self._pending = {name: set() for name in self._tables}
            self._whole = set()
            self._partitions = {name: [] for name in self._tables}
            try:
                await self._stage(pending, whole)
                await self._stage_partitions(partitions, whole)
            except BaseException:
                await self._forget_stages()
                for name, keys in pending.items():
                    self._pending[name] |= keys
                for name, parts in partitions.items():
                    self._partitions[name].extend(parts)
                self._whole |= whole
                raise
            second = await self._settled()
            if second != first:
                self._say(FollowerPhase.WAITING, "the stamp moved while the change was read; the next pass tops it up")
                return False
            self._mark += 1
            mark = self._mark
            try:
                rows = await asyncio.to_thread(self._backend.apply_staged, list(self._staged.values()), mark=mark)
            except BaseException:
                await self._forget_stages()
                raise
            self._stamps[mark] = (first, dict(self._staging_marks))
            while len(self._stamps) > _STAMPS_KEPT:
                del self._stamps[min(self._stamps)]
            self._staged = {}
            self._staged_keys = {}
            self._partitions_staged = {}
            self._rows_since_mark += rows
            self._status.stamp = first
            self._status.applied += 1
            self._status.last_apply_seconds = time.monotonic() - began
            self._say(FollowerPhase.FOLLOWING, "current")
            for listener in tuple(self._listeners):
                listener(first)
            self._checkpoint_when_due(rows)
            return True

    def _checkpoint_when_due(self, rows: int) -> None:
        """count a change's rows toward the next checkpoint, and write one beside the follower when it is due.

        :param rows: the rows the change applied
        :ptype rows: int
        :return: nothing
        :rtype: None
        """
        if self._checkpoints is None:
            return
        self._checkpoints.applied(rows)
        if not self._checkpoints.due() or (self._checkpointing is not None and not self._checkpointing.done()):
            return
        self._checkpointing = asyncio.create_task(self._write_checkpoint())

    async def _write_checkpoint(self) -> None:
        assert self._checkpoints is not None  # noqa: S101 -- narrowed by the caller
        try:
            await self._checkpoints.write(self.reading_marked)
        except asyncio.CancelledError, KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - a checkpoint is a start's shortcut; the copy is unaffected
            contain(
                exc,
                log=log,
                what="a copy checkpoint could not be written; the next change that makes one due tries again",
            )

    async def checkpointed(self) -> None:
        """wait for a checkpoint being written, if one is.

        :return: nothing
        :rtype: None
        """
        if self._checkpointing is not None:
            await asyncio.gather(self._checkpointing, return_exceptions=True)

    async def _forget_stages(self) -> None:
        """drop every stage and put what they held back to be read again.

        :return: nothing
        :rtype: None
        """
        for name, change in self._staged.items():
            if change.whole:
                self._whole.add(name)
            else:
                self._pending[name] |= self._staged_keys.get(name, set())
        for name, parts in self._partitions_staged.items():
            self._partitions[name].extend(parts)
        self._staged = {}
        self._staged_keys = {}
        self._partitions_staged = {}
        stages = await asyncio.to_thread(self._backend.staged_stages)
        if stages:
            await asyncio.to_thread(self._backend.drop_stage, *stages)

    async def _stage(self, pending: Mapping[str, set[tuple[str, ...]]], whole: set[str]) -> None:
        """stage each table's change beside the copy, a page at a time, onto what earlier passes staged.

        :param pending: the keys heard since, by table
        :ptype pending: Mapping[str, set[tuple[str, ...]]]
        :param whole: the tables to read whole
        :ptype whole: set[str]
        :return: nothing
        :rtype: None
        """
        if not self._staged:
            # a stage no change holds is one an earlier process or a failed pass left: never applied
            left = await asyncio.to_thread(self._backend.staged_stages)
            if left:
                await asyncio.to_thread(self._backend.drop_stage, *left)
        for name, table in self._tables.items():
            rows_stage = f"{STAGE_PREFIX}{name}__rows"
            keys_stage = f"{STAGE_PREFIX}{name}__keys"
            if name in whole:
                await asyncio.to_thread(self._backend.drop_stage, rows_stage, keys_stage)
                self._staged.pop(name, None)
                self._staged_keys.pop(name, None)
                await self._stage_whole(table, rows_stage)
                self._staged[name] = StagedChange(table=name, key=table.key, rows=rows_stage, whole=True)
                continue
            keys = pending.get(name)
            if not keys:
                continue
            await self._stage_keyed(table, sorted(keys), keys_stage, rows_stage)
            held = self._staged.get(name)
            self._staged[name] = (
                StagedChange(table=name, key=table.key, keys=keys_stage, rows=rows_stage, whole=True)
                if held is not None and held.whole
                else StagedChange(table=name, key=table.key, keys=keys_stage, rows=rows_stage)
            )
            self._staged_keys.setdefault(name, set()).update(keys)

    async def _stage_partitions(self, partitions: Mapping[str, list[dict[str, str]]], whole: set[str]) -> None:
        """stage each load's rows beside the copy as keyed changes: the rows whose columns hold the load's values, read
        from L3 a page at a time by those columns (an index led by them, :attr:`FollowedTable.partition_order`).

        A load from files only inserts (a key L3 already holds fails it), so its reach is exactly the rows it read
        back: each is staged by its key, onto the table's keyed stage, as a key a write named would be. A partition
        may name a column the copy does not hold (a version stamp): L3 filters on it, and the copy needs none.

        :param partitions: each table's loads' partitions heard since
        :ptype partitions: Mapping[str, list[dict[str, str]]]
        :param whole: the tables read whole this pass, which cover their partitions
        :ptype whole: set[str]
        :return: nothing
        :rtype: None
        """
        for name, parts in partitions.items():
            held = self._staged.get(name)
            if name in whole or (held is not None and held.whole):
                continue
            table = self._tables[name]
            keys_stage = f"{STAGE_PREFIX}{name}__keys"
            rows_stage = f"{STAGE_PREFIX}{name}__rows"
            types = rail_arrow_types(table.columns)
            selected = ", ".join(quote_identifier(column) for column in types)
            for partition in parts:
                pages = keyset_arrow_pages(
                    self._l3,
                    f"SELECT {selected} FROM {quote_identifier(name)}",  # noqa: S608 - trusted identifiers
                    table.partition_order or table.key,
                    types=types,
                    where={column: _typed(table.kind(column), value) for column, value in sorted(partition.items())},
                    page_size=self._page_size,
                )
                async for page in pages:
                    if page.num_rows == 0:
                        continue
                    await asyncio.to_thread(
                        self._backend.stage_keys,
                        keys_stage,
                        name,
                        table.key,
                        page.select(list(table.key)),
                        rows_stage=rows_stage,
                    )
                    await asyncio.to_thread(self._backend.stage_rows, rows_stage, name, page, table.key)
                self._partitions_staged.setdefault(name, []).append(dict(partition))
            if name in self._partitions_staged:
                self._staged[name] = StagedChange(table=name, key=table.key, keys=keys_stage, rows=rows_stage)

    async def _stage_whole(self, table: FollowedTable, rows_stage: str) -> None:
        types = rail_arrow_types(table.columns)
        selected = ", ".join(quote_identifier(name) for name in types)
        pages = keyset_arrow_pages(
            self._l3,
            f"SELECT {selected} FROM {quote_identifier(table.name)}",  # noqa: S608 - trusted identifiers
            table.read_order,
            types=types,
            page_size=self._page_size,
        )
        async for page in pages:
            await asyncio.to_thread(self._backend.stage_rows, rows_stage, table.name, page, table.key)

    async def _stage_keyed(
        self, table: FollowedTable, keys: Sequence[tuple[str, ...]], keys_stage: str, rows_stage: str
    ) -> None:
        read = self._reads[table.name]
        for start in range(0, len(keys), self._page_size):
            chunk = keys[start : start + self._page_size]
            columns = [[key[i] for key in chunk] for i in range(len(table.key))]
            arrow_keys = pa.table(
                {column: pa.array(values, pa.string()) for column, values in zip(table.key, columns, strict=True)}
            )
            await asyncio.to_thread(
                self._backend.stage_keys, keys_stage, table.name, table.key, arrow_keys, rows_stage=rows_stage
            )
            rows = await l3_fetch_arrow(self._l3, read.sql, *columns, types=read.types)
            await asyncio.to_thread(self._backend.stage_rows, rows_stage, table.name, rows, table.key)

    # --- running ---------------------------------------------------------------------------------

    async def run(self) -> None:
        """apply each change as it is heard, until cancelled: a pass whenever the registry hears a key, catches up, or
        :meth:`poke` is called (the writer's stamp moved). Never on a timer.

        :return: nothing
        :rtype: None
        """
        while True:
            await self._poked.wait()
            try:
                await self.catch_up()
            except asyncio.CancelledError, KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - one failed pass leaves the copy as it was; the next tries again
                self._say(FollowerPhase.WAITING, f"a pass failed: {type(exc).__name__}: {exc}")
                contain(
                    exc, log=log, what="a copy follower's pass failed; the copy is as it was and the keys stay pending"
                )

    def poke(self) -> None:
        """ask for a pass: the writer's stamp moved.

        :return: nothing
        :rtype: None
        """
        self._poked.set()

    def on_change(self, listener: Callable[[Any], None]) -> Callable[[], None]:
        """call ``listener`` with the stamp after every change applied.

        :param listener: called with the stamp, on the loop; must not raise
        :ptype listener: Callable[[Any], None]
        :return: a call that removes it
        :rtype: Callable[[], None]
        """
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def stop(self) -> None:
        """stop hearing the registry.

        :return: nothing
        :rtype: None
        """
        for remove in self._removers:
            remove()
        self._removers = []
        if self._checkpointing is not None:
            self._checkpointing.cancel()
        self._status.phase = FollowerPhase.STOPPED

    # --- reading ---------------------------------------------------------------------------------

    @property
    def stamp(self) -> Any:
        """the stamp the copy holds every table at; None until the first change is applied.

        :return: the stamp
        :rtype: Any
        """
        return self._status.stamp

    @contextmanager
    def reading(self) -> Iterator[tuple[Any, Any]]:
        """a cursor reading one state of every table, and the stamp it holds them at.

        The stamp is read from the snapshot itself (the change's mark, committed with it), so it is exactly the
        change the cursor reads, whatever is applied meanwhile.

        :return: ``(cursor, stamp)``; the stamp is None before the first change
        :rtype: Iterator[tuple[duckdb.DuckDBPyConnection, Any]]
        """
        with self.reading_marked() as (cursor, stamp, _marks):
            yield cursor, stamp

    @contextmanager
    def reading_marked(self) -> Iterator[tuple[Any, Any, Mapping[str, str | None]]]:
        """:meth:`reading`, with each table's epoch mark the read's change holds: what a checkpoint of it records.

        :return: ``(cursor, stamp, marks)``; the stamp None and the marks empty before the first change
        :rtype: Iterator[tuple[duckdb.DuckDBPyConnection, Any, Mapping[str, str | None]]]
        """
        with self._backend.read_snapshot() as cursor:
            mark = self._backend.staged_mark(cursor)
            stamp, marks = self._stamps.get(mark, (None, {})) if mark is not None else (None, {})
            yield cursor, stamp, marks

    def take_rows_since_mark(self) -> int:
        """the rows applied since the last call, and start counting again: what a checkpoint's trigger weighs.

        :return: the rows
        :rtype: int
        """
        rows, self._rows_since_mark = self._rows_since_mark, 0
        return rows

    def status(self) -> FollowerStatus:
        """what the follower is doing, with each table's pending keys counted now.

        :return: the status
        :rtype: FollowerStatus
        """
        self._status.pending = {name: len(keys) for name, keys in self._pending.items() if keys}
        return self._status

    def _say(self, phase: FollowerPhase, detail: str) -> None:
        if (self._status.phase, self._status.detail) != (phase, detail):
            log.info("copy follower", extra={"extra_data": {"phase": phase.value, "detail": detail}})
        self._status.phase = phase
        self._status.detail = detail


if TYPE_CHECKING:

    def _duckdb_is_a_follower_backend(backend: DuckDBBackend) -> FollowerBackend:
        """mypy's proof that the DuckDB L1 is a copy follower's backend: it fails here if they part."""
        return backend
