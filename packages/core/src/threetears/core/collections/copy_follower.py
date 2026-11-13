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
from datetime import UTC, datetime
from decimal import Decimal
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Final, Protocol, TYPE_CHECKING

from threetears.interfaces.keyed.rows import DeclaredKey, KeyColumns
from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter

from threetears.core.cache.duckdb import STAGE_PREFIX, StagedChange
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE, Unsettled, rail_arrow_types
from threetears.core.collections.registry import CacheInvalidationMessage
from threetears.core.collections.scope_epochs import ScopeEpochs
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    NUMERIC_TYPE,
    Column,
    TableSchema,
    sql_cast,
)
from threetears.core.backends.broker_generation import GENERATION_BY_TRIGGER
from threetears.core.declared import whole_read
from threetears.core.keyed import KeyBatch, WholeReads, read_keyed_arrow
from threetears.core.keyset import keyset_arrow_pages
from threetears.core.sql_fragments import quote_identifier, unnest_key_join

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
    "GATED_WHOLE_READ_WHYS",
    "WHOLE_READ_WHYS",
    "CopyFollower",
    "HubCapabilityMissingError",
    "FollowedTable",
    "FollowerBackend",
    "FollowerPhase",
    "FollowerStatus",
    "PartialStamp",
    "require_generation_by_trigger",
]

log = get_logger(__name__)

#: why a follower reads a table whole from L3, counted by table and why (``copy_follower_whole_reads_total``). Only
#: ``cold_start`` (a first start with no checkpoint) and ``reset`` (an operator's reset of that table) are expected;
#: every other is a change some writer did not record by key or key prefix
_WHOLE_READS: Final = WholeReads(
    "copy_follower_whole_reads_total",
    subject="tables a copy follower read whole from L3",
    whys={
        "cold_start": "a first start with no checkpoint: every table is read once",
        "reset": "an operator's reset of the table (SchemaBackedCollection.reset): that table's cold start",
        "schema_change": "a table dropped or recreated (CopyFollower.schema_changed): only that table is read again",
        # TODO(3tears 0.96.0 contract step, dated 2026-11-09): remove once no 0.94-or-older writer is left to record one
        "legacy_whole_record": "an epoch record naming the whole table, from a writer older than 0.95.0 mid-roll",
    },
    expected={"cold_start", "reset"},
)

# TODO(3tears 0.96.0 contract step, dated 2026-11-10): the two names below are _WHOLE_READS.whys and .gated, kept one
# release for their importers (reports' replay and prove scripts)
WHOLE_READ_WHYS: Final[Mapping[str, str]] = _WHOLE_READS.whys

#: the whys a replay or a soak requires to stay at zero: every whole read but a true cold start and an operator's reset
GATED_WHOLE_READ_WHYS: Final = _WHOLE_READS.gated

#: behind reasons the change notes have already failed to answer this pass: asking them again would only wait again
_NOTES_CANNOT: Final = frozenset({"notes_unavailable", "notes_not_covered", "notes_whole"})

#: the most values of a key prefix's last column one partition read names
_PREFIX_BATCH: Final = 1000


def _warms() -> Any:
    return counter(
        "copy_follower_warms_total",
        description="copy follower warms from L3 (every table, or one table again), by why; zero but the cold start",
        label_names=("why",),
    )


#: how many applied changes' stamps are kept for readers whose snapshot is older than the newest
_STAMPS_KEPT: Final = 256


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


def _text_keys(rows: Any, key: Sequence[str]) -> list[tuple[str, ...]]:
    """each row's key, its values as text in key order (as a change record names a key).

    :param rows: rows holding the key columns
    :ptype rows: pyarrow.Table
    :param key: the key columns
    :ptype key: Sequence[str]
    :return: the keys
    :rtype: list[tuple[str, ...]]
    """
    return [tuple(str(v) for v in values) for values in zip(*(rows.column(c).to_pylist() for c in key), strict=True)]


def _key_prefix(table: FollowedTable, partition: Mapping[str, str]) -> bool:
    """whether a partition names the leading columns of the table's key: the one shape an index reads exactly (a key
    range), never the table.

    :param table: the table
    :ptype table: FollowedTable
    :param partition: the partition's columns and values
    :ptype partition: Mapping[str, str]
    :return: whether its columns are a key prefix
    :rtype: bool
    """
    return bool(partition) and set(partition) == set(table.key[: len(partition)])


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


def _prefix_batches(
    key: Annotated[Sequence[str], DeclaredKey], parts: Sequence[Mapping[str, str]]
) -> list[tuple[tuple[str, ...], tuple[str, ...], list[str]]]:
    """partitions grouped into reads: those of one shape (their columns, in key order, and the values of all but the
    last) are one read, every value of the last column bound at once, at most :data:`_PREFIX_BATCH` a read.

    :param key: the table's key, whose order a prefix's columns take
    :ptype key: Sequence[str]
    :param parts: the partitions, each a key prefix or a held scope's column
    :ptype parts: Sequence[Mapping[str, str]]
    :return: ``(columns, leading values, the last column's values)`` per read
    :rtype: list[tuple[tuple[str, ...], tuple[str, ...], list[str]]]
    """
    grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, None]] = {}
    for part in parts:
        columns = tuple(sorted(part, key=lambda c: key.index(c) if c in key else len(key)))
        lead = tuple(part[c] for c in columns[:-1])
        grouped.setdefault((columns, lead), {})[part[columns[-1]]] = None
    return [
        (columns, lead, list(values)[start : start + _PREFIX_BATCH])
        for (columns, lead), values in grouped.items()
        for start in range(0, len(values), _PREFIX_BATCH)
    ]


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
    key: KeyColumns
    columns: tuple[Column, ...]
    order: tuple[str, ...] = ()
    partition_order: tuple[str, ...] = ()
    #: every column L3 holds and its type tag, the copy's or not: a load's partition may name a column the copy does
    #: not hold (a version stamp), which the read of its rows still filters on
    l3_kinds: tuple[tuple[str, str], ...] = ()
    #: the scope each row belongs to, as the writer's seqlock names scopes: ``(column, prefix)``, a row's scope being
    #: ``prefix + str(row[column])`` (``("join_state_code", "state:")``). A write touches the scope of every row it
    #: writes, so while a write is in progress the follower still applies every scope it does not hold. None: the
    #: table's changes apply only at a settled stamp (a write in progress holds all of them back)
    scope: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        held = {column.name for column in self.columns}
        missing = [column for column in self.key if column not in held]
        if missing:
            raise ValueError(f"{self.name}: the copy must hold every key column; {missing} is not among its columns")
        if self.scope is not None and self.scope[0] not in held:
            raise ValueError(f"{self.name}: the copy must hold its scope column {self.scope[0]!r}")
        # a keyset read pages past the last row's values of its order: an order that is not unique skips every row
        # sharing the cursor's values at each page boundary, so the copy would hold fewer rows than L3 under L3's
        # epochs. Unique here means it holds every key column
        for field, order in (("order", self.order), ("partition_order", self.partition_order)):
            short = [column for column in self.key if order and column not in order]
            if short:
                raise ValueError(
                    f"{self.name}: {field} {list(order)} is not unique: a keyset read by it skips rows at page "
                    f"boundaries; it must hold every key column, missing {short}"
                )

    def scope_of(self, value: Any) -> str:
        """the scope a row whose scope column holds ``value`` belongs to.

        :param value: the scope column's value
        :ptype value: Any
        :return: the scope, as the writer's seqlock names it
        :rtype: str
        """
        assert self.scope is not None  # noqa: S101 -- asked only of a scoped table
        return f"{self.scope[1]}{value}"

    @property
    def scope_in_key(self) -> int | None:
        """where the scope column sits in the key, or None when the key does not hold it (or the table has no scope).

        :return: its index
        :rtype: int | None
        """
        if self.scope is None or self.scope[0] not in self.key:
            return None
        return self.key.index(self.scope[0])

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
        return sql_cast(kind)


@dataclass(frozen=True, slots=True)
class _KeyedRead:
    """how some keys' rows of one table are read from L3: :func:`~threetears.core.keyed.read_keyed_arrow` over the
    table's key, each key column's values bound as one text array and joined to the table on its key.

    On PostgreSQL and YugabyteDB the join is answered by the primary key, a lookup a key (``EXPLAIN`` shows an
    index scan of the key, never a scan of the table): its cost is the keys', whatever the table holds.

    :ivar casts: each key column's SQL type, its text cast back to
    :ivar types: each column's Arrow type, as the rail answers it
    """

    casts: tuple[str, ...]
    types: Mapping[str, Any]

    @classmethod
    def of(cls, table: FollowedTable) -> _KeyedRead:
        """the read of ``table``'s rows by key.

        :param table: the table
        :ptype table: FollowedTable
        :return: the read
        :rtype: _KeyedRead
        """
        return cls(casts=tuple(table.key_cast(column) for column in table.key), types=rail_arrow_types(table.columns))


def _stamp_table_of(
    stamp_table: str | None,
    settled: Callable[[], Awaitable[Any]],
    tables: Sequence[FollowedTable],
) -> str | None:
    """the table the writer's stamp is kept in, whose row messages wake the follower: the one named, else derived from
    ``settled`` when it is a ``ScopeEpochs.settled``; None for a follower none of whose tables has a scope, whose keys
    no write can hold.

    :raises ValueError: when a table has a scope and the stamp table is neither named nor derivable
    """
    if stamp_table is not None:
        return stamp_table
    epochs = getattr(settled, "__self__", None)
    if isinstance(epochs, ScopeEpochs):
        return epochs.table_name
    scoped = sorted(table.name for table in tables if table.scope is not None)
    if not scoped:
        return None
    raise ValueError(
        f"{scoped} have a scope a write may hold: pass stamp_table=, the table the writer's stamp is kept in "
        "(ScopeEpochs' own, ScopeEpochs.table_name), or settled=ScopeEpochs.settled to derive it; without it a "
        "follower holding a write's keys never hears another replica commit it"
    )


@dataclass(frozen=True, slots=True)
class PartialStamp:
    """what a copy holds while a write is in progress: every scope the write does not hold at its committed epoch, and
    each scope it holds at the epoch the copy last applied it at.

    A reader of a held scope reads its last settled version; a reader of any other reads the newest committed. The
    stamp the follower labels a change with when its ``settled`` answered :class:`Unsettled` (with each scope's epoch
    and the scopes the write touched).

    :ivar epochs: each scope's epoch as the copy holds it
    :ivar held: the scopes the write in progress holds, each at an older epoch than L3's rows may be
    :ivar version: the last committed write's version, when the writer's record names it
    """

    epochs: Mapping[str, int]
    held: frozenset[str]
    version: int | None = None

    def epoch(self, scope: str) -> int:
        """one scope's epoch as the copy holds it; 0 for a scope it holds at none.

        :param scope: the scope
        :ptype scope: str
        :return: the epoch
        :rtype: int
        """
        return self.epochs.get(scope, 0)


def _scoped(stamp: Any) -> tuple[Mapping[str, int], frozenset[str]] | None:
    """a writer's record as each scope's epoch and the scopes a write in progress holds; None for a stamp that names
    no scope (the follower then applies only at a settled stamp, compared whole).

    :param stamp: what ``settled`` answered, or a label
    :ptype stamp: Any
    :return: the epochs and the held scopes
    :rtype: tuple[Mapping[str, int], frozenset[str]] | None
    """
    if isinstance(stamp, PartialStamp):
        return stamp.epochs, stamp.held
    epochs = getattr(stamp, "epochs", None)
    if not isinstance(epochs, Mapping):
        return None
    held = frozenset(getattr(stamp, "pending", frozenset())) if isinstance(stamp, Unsettled) else frozenset()
    return epochs, held


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
    :param stamp_table: the table the writer's stamp is kept in (``ScopeEpochs``' own), whose row messages the
        registry hears from every replica: each one asks a pass, so keys a write held are applied once another
        replica's commit moves its scopes, with nothing else heard. None derives it from ``settled`` when that is a
        ``ScopeEpochs.settled``, and hears none when no table has a scope (no write can hold its keys)
    :ptype stamp_table: str | None
    :raises ValueError: when a table has a scope and ``stamp_table`` is neither given nor derivable
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
        stamp_table: str | None = None,
    ) -> None:
        if pa is None:
            raise ImportError("a copy follower needs pyarrow: install 3tears[snapshot]")
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        self._stamp_table = _stamp_table_of(stamp_table, settled, tables)
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
        #: tables to read again whole at the next apply, and why (:data:`WHOLE_READ_WHYS`): a cold start, a schema
        #: change, or an older writer's whole-table record mid-roll; a second drop before the read is the same read
        self._whole: dict[str, str] = {}
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
        #: the keys each stage of a table with no scope replaces (keyed and partition changes alike): what a
        #: checkpoint patches the table's one part by
        self._staged_touched: dict[str, set[tuple[str, ...]]] = {}
        #: the tables staged whole, and why: put back to be read again only when their stage is forgotten
        self._staged_whole: dict[str, str] = {}
        #: the scopes each table's stage holds rows of (None: rows of no scope, a table without one)
        self._staged_scopes: dict[str, set[str | None]] = {}
        #: the mark of the last change applied, and the stamp of each recent one (a reader's snapshot names its mark)
        self._mark = 0
        self._stamps: dict[int, tuple[Any, Mapping[str, str | None]]] = {}
        #: each table's mark when the keys being staged were taken: what the next change applied holds
        self._staging_marks: dict[str, str | None] = {}
        #: each table's mark the copy holds every key of, at or below it: what a checkpoint of the copy records. A table
        #: whose keys a write in progress holds (left pending) keeps the mark it was last current at, so a start from
        #: that checkpoint walks the log from there and hears those keys again
        self._current_marks: dict[str, str | None] = {}
        self._checkpoints = checkpoints
        if checkpoints is not None:
            # every other table the checkpoint holds is derived by the follower's owner: loaded with these, never warmed
            checkpoints.follow(self._tables)
        #: stages of forgotten tables, dropped before the next is staged; the table a pass reads now
        self._dropping: set[str] = set()
        self._reading: str | None = None
        self._checkpointing: asyncio.Task[Any] | None = None
        #: when each table was last current in the registry (its mark caught up, nothing behind): a table the registry
        #: falls behind on is resolved from the change notes since then (:meth:`CollectionRegistry
        #: .resolve_behind_from_notes`), never by a warm
        self._current_at: dict[str, datetime] = {}

    # --- what the registry tells it -------------------------------------------------------------

    def _on_row(self, table: str) -> Callable[[CacheInvalidationMessage], None]:
        def heard(message: CacheInvalidationMessage) -> None:
            self._pending[table].add(tuple(message.ids))
            self._poked.set()

        return heard

    def _on_dropped(self, table: str) -> Callable[[], None]:
        def dropped() -> None:
            # every writer of 0.95.0 records keys or a partition; a whole-table record is an older writer mid-roll
            log.warning(
                "an epoch record names a whole table: a writer older than 3tears 0.95.0 recorded neither keys nor a "
                "partition; this one table is read again whole, and counted",
                extra={"extra_data": {"table": table, "why": "legacy_whole_record", "writer": "3tears < 0.95.0"}},
            )
            self._read_whole(table, "legacy_whole_record")

        return dropped

    def _read_whole(self, table: str, why: str) -> None:
        """read ``table`` again whole at the next apply; drops heard before that read are one read.

        :param table: the table
        :ptype table: str
        :param why: a :data:`WHOLE_READ_WHYS` key
        :ptype why: str
        :return: nothing
        :rtype: None
        """
        self._whole.setdefault(table, why)
        self._pending[table].clear()
        self._partitions[table].clear()
        self._poked.set()

    def schema_changed(self, table: str) -> None:
        """a table was dropped or recreated (a migration): read it again whole, once, beside the copy. The one change
        a follower reads a table whole for, counted (``copy_follower_whole_reads_total{why="schema_change"}``).

        :param table: the table
        :ptype table: str
        :return: nothing
        :rtype: None
        :raises KeyError: when the follower does not keep it
        """
        if table not in self._tables:
            raise KeyError(table)
        log.warning(
            "a followed table's schema changed; it alone is read again whole", extra={"extra_data": {"table": table}}
        )
        self._read_whole(table, "schema_change")

    def _on_partition(self, table: str) -> Callable[[Mapping[str, str]], None]:
        def loaded(partition: Mapping[str, str]) -> None:
            if not _key_prefix(self._tables[table], partition):
                # an older writer's partition on columns no key leads: no index reads it exactly, so it is the table's
                self._on_dropped(table)()
                return
            if table in self._whole:
                return  # the whole read to come covers it
            if dict(partition) not in self._partitions[table]:
                self._partitions[table].append(dict(partition))
            self._poked.set()

        return loaded

    def _on_reset(self, table: str) -> Callable[[], None]:
        def reset() -> None:
            log.info(
                "a followed table was reset by an operator; it alone is read again whole, as on a first start",
                extra={"extra_data": {"table": table, "why": "reset"}},
            )
            self._whole[table] = "reset"
            self._read_whole(table, "reset")

        return reset

    def _on_caught_up(self) -> None:
        self._poked.set()

    def _on_stamp_moved(self, message: CacheInvalidationMessage) -> None:
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
                    on_table_reset=self._on_reset(name),
                )
            )
        if self._stamp_table is not None:
            # the writer's stamp moved (a write begun, a commit on any replica): a pass reads it; nothing of the stamp
            # table itself is read or kept
            self._removers.append(
                self._registry.register_derived_cache(
                    self._stamp_table, on_row=self._on_stamp_moved, on_table_dropped=self.poke
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
        """register, and fill the copy: each table from the checkpoint that holds it (the newest, else the previous),
        whose marks the registry adopts so the next :meth:`catch_up` walks the log from them. A table no checkpoint
        holds at its schema is read whole, alone (why ``cold_start`` or ``schema_change``); with no checkpoint at
        all, every table is (:meth:`warm`).

        A table the checkpoint holds that this follower does not follow (one its owner derives from the followed
        ones) is never warmed here: when the checkpoint cannot give it, its owner derives it again (``settle_all``),
        which a start not wholly from a checkpoint asks for.

        :return: whether the start was wholly from a checkpoint (every table it holds loaded); False asks the owner to
            derive its derived tables again
        :rtype: bool
        """
        self.register()
        if self._checkpoints is None:
            await self.warm()
            return False
        staged = await self._checkpoints.stage()
        if not staged.loaded:
            await self._warm_tables({name: staged.missing.get(name, "cold_start") for name in self._tables})
            return False
        followed_missing = {name: why for name, why in staged.missing.items() if name in self._tables}
        self._mark += 1
        await asyncio.to_thread(self._backend.apply_staged, staged.changes, mark=self._mark)
        # the loaded tables' marks alone: a table still to be read whole has no mark a checkpoint may record
        self._stamps[self._mark] = (staged.stamp, dict(staged.marks))
        self._current_marks = dict(staged.marks)
        for name, mark in staged.marks.items():
            self._registry.generation_marks.adopt_generation(name, mark)
        self._status.stamp = staged.stamp
        if followed_missing:
            await self._warm_tables(followed_missing)
        self._say(
            FollowerPhase.FOLLOWING,
            f"loaded a checkpoint of {staged.rows} rows ({len(followed_missing)} tables read whole, "
            f"{len(staged.missing) - len(followed_missing)} derived tables to derive again); catching up",
        )
        self._poked.set()
        return not staged.missing

    async def _warm_tables(self, whys: Mapping[str, str]) -> None:
        """take these tables' marks at the head and read each whole at the next apply, counted by why.

        :param whys: each table and why (:data:`WHOLE_READ_WHYS`)
        :ptype whys: Mapping[str, str]
        :return: nothing
        :rtype: None
        """
        self._status.phase = FollowerPhase.WARMING
        for why in sorted(set(whys.values())):
            _warms().labels(why=why).inc()
        for name in whys:
            token = await self._generation(name)
            await self._registry.catch_up_generation(name, token, changes=self._changes)
        for name, why in whys.items():
            self._read_whole(name, why)

    async def warm(self) -> None:
        """the first start: take each table's mark at the head, then read every table whole beside the copy and swap
        it in, in one transaction. A change committed while it reads is heard (the marks were taken first) and applied
        by the next :meth:`catch_up`; the copy is labelled by that catch-up, not here.

        :return: nothing
        :rtype: None
        """
        await self._warm_tables(dict.fromkeys(self._tables, "cold_start"))

    def marks(self) -> dict[str, str | None]:
        """each table's mark in the registry now: what a checkpoint of the copy at its stamp records.

        :return: table -> its generation mark
        :rtype: dict[str, str | None]
        """
        return {name: self._registry.generation_marks.recorded(name) for name in self._tables}

    # --- the apply -------------------------------------------------------------------------------

    async def catch_up(self) -> bool:
        """apply every key heard at one stamp of the writer's seqlock; label the copy with it.

        At a settled stamp every key heard is applied. While a write is in progress, and the writer names each scope's
        epoch and the scopes the write holds (:class:`Unsettled`), every change of a scope the write does not hold is
        applied and the copy labelled with a :class:`PartialStamp`: a held scope stays at the epoch the copy last
        applied it at, its keys pending until the write commits.

        What a pass reads is staged beside the copy and kept there until a pass ends at a stamp that held still for
        every scope it staged: a pass the stamp moved under keeps its stages, and the next pass reads only the keys
        heard since (a key named again leaves the stage and is read again). A stage holding a scope a write has since
        taken is forgotten (its keys read again once the write commits). A pass whose apply fails keeps every stage
        and tries only the apply again.

        :return: whether the copy is now labelled with this pass's stamp (False: a write in progress whose record
            names no scope, the stamp moved while it read, or the registry is behind on a table; the copy is as it was)
        :rtype: bool
        """
        async with self._lock:
            began = time.monotonic()
            self._poked.clear()
            first = await self._settled()
            scoped = _scoped(first)
            writing = isinstance(first, Unsettled)
            if writing and scoped is None:
                self._say(FollowerPhase.WAITING, f"a write is in progress: {first.reason}")
                return False
            held = scoped[1] if scoped is not None else frozenset()
            # taken before any generation is read: every epoch at or below a mark read after it is in the copy's reach
            now = datetime.now(UTC)
            behind: dict[str, str] = {}
            for name in self._tables:
                token = await self._generation(name)
                await self._registry.catch_up_generation(name, token, changes=self._changes)
                reason = self._registry.behind_reason(name)
                if (
                    reason is not None
                    and reason not in _NOTES_CANNOT
                    and token is not None
                    and name in self._current_at
                ):
                    # never a warm: the keys the change notes name since the table was last current are read instead
                    resolved = await self._registry.resolve_behind_from_notes(name, self._current_at[name], token)
                    reason = resolved if resolved is not None else self._registry.behind_reason(name)
                if reason is not None:
                    # the registry's reason (BEHIND_REASONS): the status and collection_epoch_behind_total say the same
                    behind[name] = reason
                else:
                    self._current_at[name] = now
            self._status.behind = behind
            if behind:
                self._say(FollowerPhase.BEHIND, f"behind on {sorted(behind)}; serving the copy at {self._status.stamp}")
                return False
            # a stage holding a scope the write now holds (or a table of no scope, while any write is) is read again
            await self._forget_tables(
                [name for name, scopes in self._staged_scopes.items() if writing and self._clashes(scopes, held)]
            )
            label = self._label(first, held)
            nothing = not any(self._pending.values()) and not self._whole and not any(self._partitions.values())
            if nothing and not self._staged and label == self._status.stamp:
                self._say(FollowerPhase.FOLLOWING, "current" if not writing else "current to every scope not held")
                return True
            # taken with the keys, no await between: every key of an epoch at or below these marks is among them
            self._staging_marks = self.marks()
            pending, whole, partitions = self._take(held, writing=writing)
            try:
                await self._stage(pending, whole, held)
                await self._stage_partitions(partitions, held)
            except BaseException:
                # the table being read when it failed is read again; every table staged before it keeps its stage
                await self._unstaged(pending, whole, partitions)
                raise
            second = await self._settled()
            if not self._still(first, second):
                self._say(FollowerPhase.WAITING, "the stamp moved while the change was read; the next pass tops it up")
                return False
            self._mark += 1
            mark = self._mark
            # a failed apply keeps every stage: the next pass tries only the apply again, never a read
            rows = await asyncio.to_thread(self._backend.apply_staged, list(self._staged.values()), mark=mark)
            # a table is current at the marks taken for this pass only when nothing of it is left to read: no key left
            # pending (a scope the write holds, a table of no scope while any write is), no whole read, no partition
            for name in self._tables:
                if not self._pending[name] and name not in self._whole and not self._partitions[name]:
                    self._current_marks[name] = self._staging_marks.get(name)
            self._stamps[mark] = (label, dict(self._current_marks))
            # what this apply touched, for the checkpoint's incremental parts: a table read whole is every scope
            touched: dict[str, frozenset[str] | None] = {
                name: None
                if name in self._staged_whole
                else frozenset(scope for scope in self._staged_scopes.get(name, ()) if scope is not None)
                for name in self._staged
            }
            # why each table read whole was: the situation a checkpoint writing it whole again is in
            whole_whys = {name: why for name, why in self._staged_whole.items() if name in touched}
            # a table with no scope: the keys it replaced, which a checkpoint patches its one part by
            touched_keys = {
                name: frozenset(self._staged_touched.get(name, set()))
                for name, scopes in touched.items()
                if scopes is not None and self._tables[name].scope is None
            }
            while len(self._stamps) > _STAMPS_KEPT:
                del self._stamps[min(self._stamps)]
            self._staged = {}
            self._staged_keys = {}
            self._staged_touched = {}
            self._staged_whole = {}
            self._staged_scopes = {}
            self._partitions_staged = {}
            self._rows_since_mark += rows
            self._status.stamp = label
            self._status.applied += 1
            self._status.last_apply_seconds = time.monotonic() - began
            self._say(
                FollowerPhase.FOLLOWING,
                "current" if not writing else f"current to every scope not held; {len(held)} held by the write",
            )
            for listener in tuple(self._listeners):
                listener(label)
            self._checkpoint_when_due(rows, touched, touched_keys, whole_whys)
            return True

    def _checkpoint_when_due(
        self,
        rows: int,
        touched: Mapping[str, frozenset[str] | None],
        keys: Mapping[str, frozenset[tuple[str, ...]]],
        whys: Mapping[str, str],
    ) -> None:
        """count a change's rows toward the next checkpoint, and write one beside the follower when it is due.

        :param rows: the rows the change applied
        :ptype rows: int
        :param touched: each table the change applied and the scopes it touched; None for a table read whole
        :ptype touched: Mapping[str, frozenset[str] | None]
        :param keys: each table with no scope the change applied (not whole), and the keys it replaced
        :ptype keys: Mapping[str, frozenset[tuple[str, ...]]]
        :param whys: each table read whole, onto why (``cold_start``, ``reset``, ``schema_change``,
            ``legacy_whole_record``)
        :ptype whys: Mapping[str, str]
        :return: nothing
        :rtype: None
        """
        if self._checkpoints is None:
            return
        self._checkpoints.applied(rows, touched, keys, whys)
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

    @staticmethod
    def _clashes(scopes: set[str | None], held: frozenset[str]) -> bool:
        """whether a stage of these scopes may not be applied while a write holds ``held``.

        :param scopes: the scopes a stage holds rows of (None: a table of no scope)
        :ptype scopes: set[str | None]
        :param held: the scopes the write holds
        :ptype held: frozenset[str]
        :return: whether it clashes
        :rtype: bool
        """
        return None in scopes or bool(scopes & held)

    def _label(self, first: Any, held: frozenset[str]) -> Any:
        """the stamp the copy holds once this pass applies: ``first`` itself at a settled stamp; while a write is in
        progress, a :class:`PartialStamp` keeping each held scope at the epoch the copy holds it at now.

        :param first: what ``settled`` answered first
        :ptype first: Any
        :param held: the scopes the write holds
        :ptype held: frozenset[str]
        :return: the label
        :rtype: Any
        """
        if not isinstance(first, Unsettled):
            return first
        epochs = dict(first.epochs or {})
        before = _scoped(self._status.stamp)
        kept = before[0] if before is not None else {}
        for scope in held:
            epochs.pop(scope, None)
            if scope in kept:
                epochs[scope] = kept[scope]
        return PartialStamp(epochs=epochs, held=held, version=first.version)

    def _take(
        self, held: frozenset[str], *, writing: bool
    ) -> tuple[dict[str, set[tuple[str, ...]]], dict[str, str], dict[str, list[dict[str, str]]]]:
        """take what was heard for this pass, leaving pending what a write in progress holds.

        A key whose scope its key names is left when the write holds that scope; a key whose scope only its row names
        is taken, and left after its row is read (:meth:`_stage_keyed`). A table of no scope is left whole while any
        write is in progress. A table to read whole is taken only at a settled stamp, or on the first start (nothing to
        keep: its held scopes are read once the write commits).

        :param held: the scopes the write holds
        :ptype held: frozenset[str]
        :param writing: whether a write is in progress
        :ptype writing: bool
        :return: the keys, the tables read whole and why, and the partitions taken
        :rtype: tuple[dict[str, set[tuple[str, ...]]], dict[str, str], dict[str, list[dict[str, str]]]]
        """
        pending: dict[str, set[tuple[str, ...]]] = {}
        partitions: dict[str, list[dict[str, str]]] = {}
        whole: dict[str, str] = {}
        for name, table in self._tables.items():
            keys = self._pending[name]
            if writing and table.scope is None:
                continue
            if name in self._whole:
                why = self._whole[name]
                if writing and why not in ("cold_start", "reset"):
                    continue
                whole[name] = self._whole.pop(name)
                self._pending[name] = set()
                self._partitions[name] = []
                continue
            at = table.scope_in_key
            if held and at is not None:
                taken = {key for key in keys if table.scope_of(key[at]) not in held}
            else:
                taken = set(keys)
            if taken:
                pending[name] = taken
                self._pending[name] = keys - taken
            parts = self._partitions[name]
            column = table.scope[0] if table.scope is not None else None
            if held:
                # a partition naming its scope is judged by it; one whose scope is in its rows only is judged as it is
                # read (:meth:`_stage_partitions`), and put back if a held scope turns up
                go = [p for p in parts if column not in p or table.scope_of(p[column]) not in held]
            else:
                go = list(parts)
            if go:
                partitions[name] = go
                self._partitions[name] = [p for p in parts if p not in go]
        return pending, whole, partitions

    def _still(self, first: Any, second: Any) -> bool:
        """whether every stage may be applied at ``first``: the stamp held still for each scope staged.

        A stamp that names no scope must be the same whole. Otherwise each staged scope must be untouched by a write
        in progress and at the same epoch in both, and a table of no scope needs the whole record unmoved. A stage
        holding a scope a write took meanwhile is forgotten here (its keys read again once the write commits); one
        whose scope only moved is kept, its keys heard again topping it up.

        :param first: what ``settled`` answered before the pass read
        :ptype first: Any
        :param second: what it answered after
        :ptype second: Any
        :return: whether to apply
        :rtype: bool
        """
        a, b = _scoped(first), _scoped(second)
        if a is None or b is None:
            return bool(second == first)
        (epochs_a, _), (epochs_b, held_b) = a, b
        moved = {scope for scope in set(epochs_a) | set(epochs_b) if epochs_a.get(scope) != epochs_b.get(scope)}
        taken = [name for name, scopes in self._staged_scopes.items() if self._clashes(scopes, held_b)]
        if taken and isinstance(second, Unsettled):
            self._forget_tables_now(taken)
        whole_moved = moved or (isinstance(second, Unsettled) != isinstance(first, Unsettled))
        for scopes in self._staged_scopes.values():
            if scopes & moved or (None in scopes and whole_moved):
                return False
        return not (taken and isinstance(second, Unsettled))

    def _forget_tables_now(self, names: Sequence[str]) -> None:
        """put back what these tables' stages hold, to be read again; their stages are dropped by the next pass.

        :param names: the tables
        :ptype names: Sequence[str]
        :return: nothing
        :rtype: None
        """
        for name in names:
            self._staged.pop(name, None)
            self._staged_scopes.pop(name, None)
            self._pending[name] |= self._staged_keys.pop(name, set())
            self._staged_touched.pop(name, None)
            self._partitions[name].extend(self._partitions_staged.pop(name, []))
            why = self._staged_whole.pop(name, None)
            if why is not None:
                self._whole.setdefault(name, why)
            self._dropping.update((f"{STAGE_PREFIX}{name}__rows", f"{STAGE_PREFIX}{name}__keys"))

    async def _forget_tables(self, names: Sequence[str]) -> None:
        """:meth:`_forget_tables_now`, and drop the stages of every table forgotten so far.

        :param names: the tables
        :ptype names: Sequence[str]
        :return: nothing
        :rtype: None
        """
        self._forget_tables_now(names)
        if self._dropping:
            dropping, self._dropping = sorted(self._dropping), set()
            await asyncio.to_thread(self._backend.drop_stage, *dropping)

    async def _unstaged(
        self,
        pending: Mapping[str, set[tuple[str, ...]]],
        whole: Mapping[str, str],
        partitions: Mapping[str, list[dict[str, str]]],
    ) -> None:
        """after a pass failed while it read: put back what it had not staged, and forget the table it was reading.

        :param pending: the keys not yet staged (each table's entry leaves once its stage is complete)
        :ptype pending: Mapping[str, set[tuple[str, ...]]]
        :param whole: the tables not yet read whole
        :ptype whole: Mapping[str, str]
        :param partitions: the partitions not yet staged
        :ptype partitions: Mapping[str, list[dict[str, str]]]
        :return: nothing
        :rtype: None
        """
        for name, keys in pending.items():
            self._pending[name] |= keys
        for name, why in whole.items():
            self._whole.setdefault(name, why)
        for name, parts in partitions.items():
            self._partitions[name].extend(parts)
        reading, self._reading = self._reading, None
        await self._forget_tables([reading] if reading is not None else [])

    async def _stage(
        self, pending: dict[str, set[tuple[str, ...]]], whole: dict[str, str], held: frozenset[str]
    ) -> None:
        """stage each table's change beside the copy, a page at a time, onto what earlier passes staged. Each table's
        entry leaves ``pending`` or ``whole`` once its stage is complete.

        :param pending: the keys taken, by table
        :ptype pending: dict[str, set[tuple[str, ...]]]
        :param whole: the tables to read whole, and why
        :ptype whole: dict[str, str]
        :param held: the scopes a write in progress holds
        :ptype held: frozenset[str]
        :return: nothing
        :rtype: None
        """
        if not self._staged:
            # a stage no change holds is one an earlier process or a failed pass left: never applied
            left = await asyncio.to_thread(self._backend.staged_stages)
            if left:
                await asyncio.to_thread(self._backend.drop_stage, *left)
            self._dropping = set()
        await self._forget_tables([])
        for name, table in self._tables.items():
            rows_stage = f"{STAGE_PREFIX}{name}__rows"
            keys_stage = f"{STAGE_PREFIX}{name}__keys"
            if name in whole:
                self._reading = name
                await self._forget_tables([name])
                self._pending[name] = set()  # a whole read covers every key heard before it
                self._whole.pop(name, None)
                self._partitions[name] = []
                why = whole[name]
                # the reason is known here: each is read by its own function, and declared as that situation
                if why == "schema_change":
                    scopes = await whole_read(
                        self._stage_schema_change(table, rows_stage, held),
                        kind="schema_change",
                        reason="a followed table was dropped or recreated by a migration: that table alone is read again",
                        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade",
                        proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
                        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
                    )
                elif why == "legacy_whole_record":
                    scopes = await whole_read(
                        self._stage_legacy_whole_record(table, rows_stage, held),
                        kind="rolling_upgrade",
                        reason="a writer older than 0.95.0 recorded the whole table mid-roll: that table is read again",
                        until="0.96.0",
                        ruling="Pace 2026-11-12: rolling_upgrade until 0.96.0 (the older writers' whole record is read for one release)",
                        proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
                        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
                    )
                else:
                    scopes = await whole_read(
                        self._stage_cold_start(table, rows_stage, why, held),
                        kind="cold_start",
                        reason="the copy holds nothing valid for the table: a first start with no checkpoint, or an operator's reset",
                        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade",
                        proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
                        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
                    )
                self._staged[name] = StagedChange(table=name, key=table.key, rows=rows_stage, whole=True)
                self._staged_whole[name] = why
                self._staged_scopes[name] = scopes
                del whole[name]
                self._reading = None
                continue
            keys = pending.get(name)
            if not keys:
                pending.pop(name, None)
                continue
            self._reading = name
            staged, scopes = await self._stage_keyed(table, sorted(keys), keys_stage, rows_stage, held)
            if staged:
                was = self._staged.get(name)
                self._staged[name] = StagedChange(
                    table=name, key=table.key, keys=keys_stage, rows=rows_stage, whole=was is not None and was.whole
                )
                self._staged_keys.setdefault(name, set()).update(staged)
                if table.scope is None:
                    self._staged_touched.setdefault(name, set()).update(staged)
                self._staged_scopes.setdefault(name, set()).update(scopes)
            del pending[name]
            self._reading = None

    def _scopes_of_rows(self, table: FollowedTable, rows: Any) -> list[str | None]:
        """each row's scope, in order (None for a table of no scope).

        :param table: the table
        :ptype table: FollowedTable
        :param rows: rows holding its scope column
        :ptype rows: pyarrow.Table
        :return: the scopes
        :rtype: list[str | None]
        """
        if table.scope is None:
            return [None] * rows.num_rows
        return [table.scope_of(value) for value in rows.column(table.scope[0]).to_pylist()]

    @staticmethod
    def _copy_columns(table: FollowedTable) -> str:
        """the copy's own key and scope columns of a table, selected over ``t``.

        :param table: the table
        :ptype table: FollowedTable
        :return: the select list
        :rtype: str
        """
        columns = list(table.key) + (
            [table.scope[0]] if table.scope is not None and table.scope[0] not in table.key else []
        )
        return ", ".join(f"t.{quote_identifier(c)}" for c in columns)

    def _copy_rows_of_keys(
        self, table: FollowedTable, columns: Annotated[Sequence[str], DeclaredKey], values: Sequence[Sequence[str]]
    ) -> Any:
        """the copy's own key and scope columns of the rows some keys, or key prefixes, name: a local read joined on
        a leading part of the key (the whole key, or a load's prefix), never L3.

        :param table: the table
        :ptype table: FollowedTable
        :param columns: the key columns named, a leading part of the key in key order
        :ptype columns: Sequence[str]
        :param values: each named column's values, as text, one list a column and one entry a key
        :ptype values: Sequence[Sequence[str]]
        :return: the rows
        :rtype: pyarrow.Table
        """
        selected = self._copy_columns(table)
        keys, pk_join = unnest_key_join(columns, [table.key_cast(column) for column in columns])
        with self._backend.read_snapshot() as cursor:
            return cursor.execute(
                f"SELECT {selected} FROM {quote_identifier(table.name)} AS t JOIN (SELECT {keys}) AS k ON {pk_join}",  # noqa: S608 - trusted identifiers
                [list(column) for column in values],
            ).fetch_arrow_table()

    def _copy_rows_of_prefixes(
        self, table: FollowedTable, columns: Sequence[str], lead: Sequence[str], last: Sequence[str]
    ) -> Any:
        """the copy's own key and scope columns of some key prefixes' rows (one prefix shape: its leading columns'
        values, and the last column's values), read by that prefix of the key (:meth:`_copy_rows_of_keys`).

        :param table: the table
        :ptype table: FollowedTable
        :param columns: the prefix's key columns, in key order
        :ptype columns: Sequence[str]
        :param lead: the values of all but the last, as text
        :ptype lead: Sequence[str]
        :param last: the last column's values, as text
        :ptype last: Sequence[str]
        :return: the rows
        :rtype: pyarrow.Table
        """
        distinct = sorted(set(last))
        return self._copy_rows_of_keys(table, columns, [*([value] * len(distinct) for value in lead), distinct])

    async def _copy_scopes(
        self, table: FollowedTable, keys: Sequence[tuple[str, ...]]
    ) -> dict[tuple[str, ...], set[str]]:
        """the scope of each key's row the copy holds: a local read, never L3.

        :param table: a table whose scope is in its rows only
        :ptype table: FollowedTable
        :param keys: the keys, each its values as text
        :ptype keys: Sequence[tuple[str, ...]]
        :return: each held key's scopes
        :rtype: dict[tuple[str, ...], set[str]]
        """
        columns = [[key[i] for key in keys] for i in range(len(table.key))]
        mine = await asyncio.to_thread(self._copy_rows_of_keys, table, table.key, columns)
        of: dict[tuple[str, ...], set[str]] = {}
        if mine.num_rows:
            copy_keys = zip(*(mine.column(c).to_pylist() for c in table.key), strict=True)
            for key, scope in zip(copy_keys, self._scopes_of_rows(table, mine), strict=True):
                of.setdefault(tuple(str(v) for v in key), set()).add(str(scope))
        return of

    async def _stage_keyed(
        self,
        table: FollowedTable,
        keys: Sequence[tuple[str, ...]],
        keys_stage: str,
        rows_stage: str,
        held: frozenset[str],
    ) -> tuple[set[tuple[str, ...]], set[str | None]]:
        """stage some keys' rows, read from L3 by key a page at a time; a key whose row (in L3, or in the copy) is of a
        scope a write holds is left pending, unstaged.

        :return: the keys staged, and the scopes their rows are of
        :rtype: tuple[set[tuple[str, ...]], set[str | None]]
        """
        read = self._reads[table.name]
        at = table.scope_in_key
        staged: set[tuple[str, ...]] = set()
        scopes: set[str | None] = set()
        for start in range(0, len(keys), self._page_size):
            chunk = keys[start : start + self._page_size]
            of: dict[tuple[str, ...], set[str]] = {}
            if table.scope is not None and at is None:
                # the scope is in the row only: first the copy's own rows (a local read), so a key whose row is of a
                # scope a write holds waits for it without an L3 read, and is read once, after the write commits
                of = await self._copy_scopes(table, chunk)
                if held:
                    late = {key for key in chunk if of.get(tuple(key), set()) & held}
                    if late:
                        self._pending[table.name] |= late
                        chunk = [key for key in chunk if key not in late]
                        if not chunk:
                            continue
            (rows,) = await read_keyed_arrow(
                self._l3,
                table.name,
                keys=KeyBatch(table.key, chunk, read.casts),
                types=read.types,
                batch=len(chunk),
            )
            kept = list(chunk)
            if table.scope is None:
                scopes.add(None)
            elif at is not None:
                scopes.update(table.scope_of(key[at]) for key in chunk)
            else:
                # the row L3 holds now beside the one the copy holds (a delete's, a move's): a key whose new row is of a
                # held scope (a row new to the copy, or moved into it) waits too
                l3_keys = (
                    list(zip(*(rows.column(c).to_pylist() for c in table.key), strict=True)) if rows.num_rows else []
                )
                for key, scope in zip(l3_keys, self._scopes_of_rows(table, rows), strict=True):
                    of.setdefault(tuple(str(v) for v in key), set()).add(str(scope))
                if held:
                    left = {key for key in chunk if of.get(tuple(key), set()) & held}
                    if left:
                        self._pending[table.name] |= left
                        kept = [key for key in chunk if key not in left]
                        mask = [tuple(str(v) for v in key) not in left for key in l3_keys]
                        rows = rows.filter(pa.array(mask, pa.bool_())) if rows.num_rows else rows
                for key in kept:
                    scopes.update(of.get(tuple(key), set()))
            if not kept:
                continue
            arrow_keys = pa.table(
                {column: pa.array([key[i] for key in kept], pa.string()) for i, column in enumerate(table.key)}
            )
            await asyncio.to_thread(
                self._backend.stage_keys, keys_stage, table.name, table.key, arrow_keys, rows_stage=rows_stage
            )
            await asyncio.to_thread(self._backend.stage_rows, rows_stage, table.name, rows, table.key)
            staged.update(kept)
        return staged, scopes

    async def _stage_partitions(self, partitions: dict[str, list[dict[str, str]]], held: frozenset[str]) -> None:
        """stage each partition's change beside the copy as keyed changes: the partition is replaced. The copy's own
        rows of it leave (a local read) and L3's rows of it go in, read a page at a time as a key range: the
        partitions of one shape are batched, every value of the prefix's last column bound as one array beside its
        leading columns' values, and paged by the key. A load from files and a clear of a key prefix are alike.

        While a write holds some scopes, a partition whose rows (the copy's, or L3's) turn out to be of a held scope
        is put back with everything the table staged, to be read once the write commits.

        :param partitions: each table's partitions taken; each table's entry leaves once staged
        :ptype partitions: dict[str, list[dict[str, str]]]
        :param held: the scopes a write in progress holds
        :ptype held: frozenset[str]
        :return: nothing
        :rtype: None
        """
        for name in list(partitions):
            parts = partitions[name]
            table = self._tables[name]
            held_stage = self._staged.get(name)
            if held_stage is not None and held_stage.whole and name in self._staged_whole:
                # a whole read beside covers them
                del partitions[name]
                continue
            self._reading = name
            keys_stage = f"{STAGE_PREFIX}{name}__keys"
            rows_stage = f"{STAGE_PREFIX}{name}__rows"
            types = rail_arrow_types(table.columns)
            selected = ", ".join(quote_identifier(column) for column in types)
            scopes: set[str | None] = set()
            #: a table with no scope: the keys its partitions replace, for the checkpoint's patch
            replaced: set[tuple[str, ...]] = set()
            clashed = False
            #: prefixes the copy holds a row of a held scope in: they wait for the write, the rest are read now
            waiting: list[dict[str, str]] = []
            for columns, lead, batch in _prefix_batches(table.key, parts):
                last = columns[-1]
                mine = await asyncio.to_thread(self._copy_rows_of_prefixes, table, columns, lead, batch)
                of = self._scopes_of_rows(table, mine) if mine.num_rows else []
                if held and set(of) & held:
                    values = [str(v) for v in mine.column(last).to_pylist()]
                    late = {value for value, scope in zip(values, of, strict=True) if scope in held}
                    waiting += [
                        dict(p)
                        for p in parts
                        if tuple(sorted(p)) == tuple(sorted(columns))
                        and tuple(p[c] for c in columns[:-1]) == lead
                        and p[last] in late
                    ]
                    batch = [value for value in batch if value not in late]
                    mine = mine.filter(pa.array([value not in late for value in values], pa.bool_()))
                    of = [scope for value, scope in zip(values, of, strict=True) if value not in late]
                    if not batch:
                        continue
                scopes.update(of)
                if mine.num_rows and table.scope is None:
                    replaced.update(_text_keys(mine, table.key))
                if mine.num_rows:
                    await asyncio.to_thread(
                        self._backend.stage_keys,
                        keys_stage,
                        name,
                        table.key,
                        mine.select(list(table.key)),
                        rows_stage=rows_stage,
                    )
                pages = keyset_arrow_pages(
                    self._l3,
                    f"SELECT {selected} FROM {quote_identifier(name)}",  # noqa: S608 - trusted identifiers
                    table.key,
                    types=types,
                    where={c: _typed(table.kind(c), v) for c, v in zip(columns[:-1], lead, strict=True)} or None,
                    where_any={
                        last: (
                            f"{sql_cast(table.kind(last))}[]",
                            [_typed(table.kind(last), v) for v in batch],
                        )
                    },
                    page_size=self._page_size,
                )
                async for page in pages:
                    if page.num_rows == 0:
                        continue
                    of = self._scopes_of_rows(table, page)
                    if held and set(of) & held:
                        clashed = True
                        break
                    scopes.update(of)
                    if table.scope is None:
                        replaced.update(_text_keys(page, table.key))
                    await asyncio.to_thread(
                        self._backend.stage_keys,
                        keys_stage,
                        name,
                        table.key,
                        page.select(list(table.key)),
                        rows_stage=rows_stage,
                    )
                    await asyncio.to_thread(self._backend.stage_rows, rows_stage, name, page, table.key)
                if clashed:
                    break
            if clashed:
                # a row of a scope the write holds: everything this table staged waits for the write to commit
                self._partitions[name].extend(parts)
                self._forget_tables_now([name])
                del partitions[name]
                self._reading = None
                continue
            self._partitions[name].extend(waiting)
            self._staged_scopes.setdefault(name, set()).update(scopes)
            if replaced:
                self._staged_touched.setdefault(name, set()).update(replaced)
            self._partitions_staged.setdefault(name, []).extend(dict(p) for p in parts if dict(p) not in waiting)
            was = self._staged.get(name)
            self._staged[name] = StagedChange(
                table=name, key=table.key, keys=keys_stage, rows=rows_stage, whole=was is not None and was.whole
            )
            del partitions[name]
            self._reading = None

    @staticmethod
    def _whole_select(table: FollowedTable) -> tuple[str, dict[str, Any], str]:
        """a table's whole read beside the copy: its select of every column, the columns' arrow types, and its lead
        column's array cast (declared, so hashed on YugabyteDB: no declaration names a range-sharded lead).

        :param table: the table
        :ptype table: FollowedTable
        :return: the select, the types and the lead array cast
        :rtype: tuple[str, dict[str, Any], str]
        """
        types = rail_arrow_types(table.columns)
        selected = ", ".join(quote_identifier(column) for column in types)
        select = f"SELECT {selected} FROM {quote_identifier(table.name)}"  # noqa: S608 - trusted identifiers
        return select, types, f"{sql_cast(table.kind(table.read_order[0]))}[]"

    async def _stage_cold_start(
        self, table: FollowedTable, rows_stage: str, why: str, held: frozenset[str]
    ) -> set[str | None]:
        """read a table whole beside the copy because the copy holds nothing valid for it: a first start with no checkpoint
        (``cold_start``) or an operator's reset of the table (``reset``).

        :return: the scopes of the rows staged
        :rtype: set[str | None]
        """
        select, types, lead = self._whole_select(table)
        pages = whole_read(
            keyset_arrow_pages(
                self._l3, select, table.read_order, types=types, page_size=self._page_size, lead_array=lead
            ),
            kind="cold_start",
            reason="the copy holds nothing valid for the table: a first start with no checkpoint, or an operator's reset",
            ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade",
            proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
            speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
        )
        return await self._stage_whole(table, rows_stage, why, held, pages)

    async def _stage_schema_change(
        self, table: FollowedTable, rows_stage: str, held: frozenset[str]
    ) -> set[str | None]:
        """read a table whole beside the copy because a migration dropped or recreated it (``schema_change``).

        :return: the scopes of the rows staged
        :rtype: set[str | None]
        """
        select, types, lead = self._whole_select(table)
        pages = whole_read(
            keyset_arrow_pages(
                self._l3, select, table.read_order, types=types, page_size=self._page_size, lead_array=lead
            ),
            kind="schema_change",
            reason="a followed table was dropped or recreated by a migration: that table alone is read again",
            ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade",
            proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
            speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
        )
        return await self._stage_whole(table, rows_stage, "schema_change", held, pages)

    async def _stage_legacy_whole_record(
        self, table: FollowedTable, rows_stage: str, held: frozenset[str]
    ) -> set[str | None]:
        """read a table whole beside the copy because a writer older than 0.95.0 recorded the whole table mid-roll
        (``legacy_whole_record``; gone by 0.96.0).

        :return: the scopes of the rows staged
        :rtype: set[str | None]
        """
        select, types, lead = self._whole_select(table)
        pages = whole_read(
            keyset_arrow_pages(
                self._l3, select, table.read_order, types=types, page_size=self._page_size, lead_array=lead
            ),
            kind="rolling_upgrade",
            reason="a writer older than 0.95.0 recorded the whole table mid-roll: that table is read again",
            until="0.96.0",
            ruling="Pace 2026-11-12: rolling_upgrade until 0.96.0 (the older writers' whole record is read for one release)",
            proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
            speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
        )
        return await self._stage_whole(table, rows_stage, "legacy_whole_record", held, pages)

    async def _stage_whole(
        self,
        table: FollowedTable,
        rows_stage: str,
        why: str,
        held: frozenset[str],
        pages: AsyncIterator[Any],
    ) -> set[str | None]:
        """stage a table's whole read (each situation's own function declares it) beside the copy, counted by why. While a write holds
        some scopes (a first start beside a long write) their rows are left out and each is read as a partition once
        the write commits.

        :param pages: the table's keyset pages, read whole
        :ptype pages: AsyncIterator[Any]
        :return: the scopes of the rows staged
        :rtype: set[str | None]
        """
        _WHOLE_READS.count(why, table=table.name)
        if why != "cold_start":
            _warms().labels(why=why).inc()
        scopes: set[str | None] = set()
        await asyncio.to_thread(self._backend.stage_rows, rows_stage, table.name, pa.table({}), table.key)
        async for page in pages:
            of = self._scopes_of_rows(table, page)
            if held and table.scope is not None:
                mask = [scope not in held for scope in of]
                page = page.filter(pa.array(mask, pa.bool_()))
                of = [scope for scope, keep in zip(of, mask, strict=True) if keep]
            scopes.update(of)
            await asyncio.to_thread(self._backend.stage_rows, rows_stage, table.name, page, table.key)
        if held and table.scope is not None:
            column, prefix = table.scope
            for scope in held:
                if scope.startswith(prefix):
                    self._partitions[table.name].append({column: scope[len(prefix) :]})
        return scopes or ({None} if table.scope is None else set())

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
