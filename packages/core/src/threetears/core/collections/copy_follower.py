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

**From delta files, for a table that opts in** (:attr:`FollowedTable.delta_files`). Each key heard keeps the epoch
it was heard at; the epoch's record names the commit's delta file (``EpochChange.delta_file``), read off the shared
volume (``THREETEARS_DELTA_FILES_DIR``), and the key's row is the row the copy holds with each of its epochs' changes
applied in order, merged as L3 merged them (:func:`~threetears.core.collections.delta_files.apply_delta_row`): no
statement reaches L3, however many pods follow (``design-pod-delta-files.md``, decisions 3, 4 and 19). The scope rules
are the same: a key left out of a change keeps its epochs and is staged again from its files. A key whose file cannot
be had (a lost volume, an older writer's record) is read from L3 by key as for any table, counted by why
(``copy_follower_delta_file_fallback_keys_total``). A commit too big to list its rows (a keys-unknown record) that
names its file is applied whole from it: the registry names every key the file holds, stamped with the epoch, and
each is staged from the file as above, so neither the change notes nor L3 is read for it and nothing a restarted copy
needs expires (decision 21). A commit of several records names one file in each; a key is applied from it once.

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
A table that opts in to delta files starts instead from the full on the shared volume
(:mod:`~threetears.core.collections.volume_fulls`): with none published the follower waits for it, its status saying so,
and never reads that table whole from L3.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from decimal import Decimal
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Final, Protocol, TYPE_CHECKING

from threetears.interfaces.keyed.rows import DeclaredKey, KeyColumns
from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter, gauge

from threetears.core.cache.duckdb import STAGE_PREFIX, StagedChange, WholeKind
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE, Unsettled, rail_arrow_types
from threetears.core.collections.delta_files import (
    DeltaChange,
    DeltaFile,
    apply_delta_row,
    delta_files_root,
    read_delta_file,
)
from threetears.core.collections.epoch_change import key_text
from threetears.core.collections.generation import split_generation_token
from threetears.core.collections.registry import CacheInvalidationMessage
from threetears.core.collections.scope_epochs import ScopeEpochs
from threetears.core.collections.volume_fulls import (
    FullAbsent,
    FullManifest,
    StagedFull,
    VolumeFulls,
    current_full_manifest,
    full_table_floor,
)
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


#: failed passes, by what failed: ``stages_lost`` (the apply worker died and its stages with it: the change is read
#: again) or ``pass`` (any other failure: the stages are kept and the next pass tries again)
_PASS_FAILURES = counter(
    "copy_follower_pass_failures_total",
    description="copy follower passes that failed, by what failed; the copy keeps serving its last change",
    label_names=("what",),
)

#: when the copy stopped taking changes: unix time of the first failed pass since the last good one, 0 while current
_BEHIND_SINCE = gauge(
    "copy_follower_behind_since_timestamp_seconds",
    description="unix time of the first failed copy follower pass since the last change applied, 0 while passes "
    "succeed; alert when it is non-zero for several minutes (the copy serves an old change)",
)

#: consecutive failed passes after which the follower says FAILING (it keeps trying, at the longest wait)
FAILING_AFTER: Final = 5

#: the wait before a failed pass is tried again: doubled each failure from the first, up to the second (seconds)
RETRY_WAITS: Final = (1.0, 60.0)


#: passes that only relabelled the copy: nothing heard or staged, the writer's stamp moved. Such a pass relabels the
#: change the copy already holds in place and asks the backend for no apply (see ``CopyFollower._relabel``)
_RELABELS = counter(
    "copy_follower_relabels_total",
    description="copy follower passes that only relabelled the copy with a new stamp (nothing to apply): no apply, "
    "no settle, no publish",
)


#: keys of a table that opts in to delta files read from L3 by key instead of from their files, by table and why:
#: ``file_missing`` (the record names a file the volume does not hold: a lost volume, or none mounted),
#: ``key_not_in_file`` (the file of an epoch that named the key holds no change of it), ``record_missing`` (the log
#: holds no record of the epoch the key was heard at), ``no_file_named`` (the record names no file: a writer before
#: delta files, or a write before the table opted in) and ``no_epoch`` (the key was heard with no epoch that orders it:
#: a change note's, a publisher's that names none). Every one is an L3 read the files were meant to save
_DELTA_FILE_FALLBACKS = counter(
    "copy_follower_delta_file_fallback_keys_total",
    description="keys of a table that opts in to delta files read from L3 by key instead of from their files, by "
    "table and why; zero while every epoch's file is on the volume",
    label_names=("table", "why"),
)

#: the fallbacks that mean something is lost or wrong (a file, a record, a row a file should hold): logged at WARNING.
#: The others (an older writer mid-roll, a change note's keys) are expected and logged at INFO, counted alike
_DELTA_FILE_LOST: Final = frozenset({"file_missing", "key_not_in_file", "record_missing"})

#: the most file names one fallback's log line lists
_FILES_NAMED: Final = 10


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


def _rail_python(value: Any, kind: Any) -> Any:
    """a value the copy holds as the L3 rail answers its column (``rail_arrow_types``): text where the rail answers
    text (a ``NUMERIC``'s digits, an instant in its key spelling, a UUID), an instant where it answers an instant (the
    copy may keep one as its ISO text), else as it is.

    Why: a row merged from a delta file takes the file's values, typed as the rail answers them, beside the copy's own
    of every column its write left out, and the stage takes one type a column, as it takes an L3 read's.

    :param value: the copy's value
    :ptype value: Any
    :param kind: the column's Arrow type as the rail answers it
    :ptype kind: pyarrow.DataType
    :return: the value as the rail answers it
    :rtype: Any
    """
    if value is None:
        return value
    if pa.types.is_timestamp(kind) and isinstance(value, str):
        return datetime.fromisoformat(value)
    if not pa.types.is_string(kind) or isinstance(value, str):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    return key_text(value)


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
    #: the table opts in to delta files (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_files`):
    #: each epoch's changes are applied from the file its record names, read off the shared volume, merged onto the
    #: row the copy holds, never read from L3 (``design-pod-delta-files.md``, decisions 3, 4 and 19). A key whose
    #: file cannot be had is read from L3 by key, counted (``copy_follower_delta_file_fallback_keys_total``)
    delta_files: bool = False

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
        for named, order in (("order", self.order), ("partition_order", self.partition_order)):
            short = [column for column in self.key if order and column not in order]
            if short:
                raise ValueError(
                    f"{self.name}: {named} {list(order)} is not unique: a keyset read by it skips rows at page "
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
        return cls(
            name=schema.name,
            key=tuple(key),
            columns=held,
            order=tuple(order),
            l3_kinds=kinds,
            delta_files=schema.delta_files,
        )

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
    FAILING = "failing"
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
    :ivar relabelled: passes since start that only relabelled the copy (nothing to apply, a new stamp): no apply
    :ivar last_apply_seconds: how long the last change took, from the first stamp read to the commit
    :ivar failures: passes failed in a row since the last good one; at :data:`FAILING_AFTER` the phase is FAILING
    :ivar behind_since: when the first of those failed (None while passes succeed): the copy serves the change it held
        then
    :ivar waiting_for_full: each opted-in table reset or recreated while the follower ran, waiting for the writer to
        publish its full on the delta-files volume (never read from L3), onto why it is still waiting; every other
        table keeps being followed meanwhile
    :ivar detail: a line for an operator
    """

    phase: FollowerPhase = FollowerPhase.STARTING
    stamp: Any = None
    behind: dict[str, str] = field(default_factory=dict)
    pending: dict[str, int] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    applied: int = 0
    relabelled: int = 0
    last_apply_seconds: float | None = None
    failures: int = 0
    behind_since: datetime | None = None
    waiting_for_full: dict[str, str] = field(default_factory=dict)
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _FullAfter:
    """the full an opted-in table reset or recreated while the follower ran must be reloaded from: one that holds the
    table at or past the reset (decisions 12 and 14 of the delta-files design: the writer rebuilds the table from L3
    once and publishes it; a copy never reads it from L3). A full from before the reset holds the table the reset
    replaced, so it is never taken.

    :ivar why: ``reset`` or ``schema_change``
    :ivar token: the reset's generation token when the record named it; else the newest generation the copy had heard
        of the table, which the reset came after (a reset in this process is told before its advance is written);
        None when the copy had heard none
    :ivar inclusive: whether ``token`` is the reset's own (a full at it holds the reset table) or one it came after
    """

    why: str
    token: str | None
    inclusive: bool

    def short_of(self, table: str, manifest: FullManifest | None) -> str | None:
        """why ``manifest`` cannot be reloaded from; None when it can.

        :param table: the table
        :ptype table: str
        :param manifest: the current manifest, None when none is published
        :ptype manifest: FullManifest | None
        :return: why not, for the status
        :rtype: str | None
        :raises ValueError: when the table's versions in it are not generation tokens of one incarnation
        """
        if manifest is None:
            return "no full on the volume; the writer publishes one"
        entry = manifest.tables.get(table)
        if entry is None:
            return f"full {manifest.name} does not hold the table; the writer publishes one that does"
        if self.token is None:
            return None
        floor = full_table_floor(table, entry)
        held, after = split_generation_token(floor), split_generation_token(self.token)
        if held is None or after is None or held[0] != after[0]:
            # a count under one incarnation is never compared with one under another: the reset's own full is awaited
            return f"full {manifest.name} holds the table at {floor}, of another incarnation than the {self.why} at {self.token}"
        if held[1] > after[1] or (self.inclusive and held[1] == after[1]):
            return None
        return f"full {manifest.name} holds the table at {floor}, from before the {self.why} at {self.token}"


class _StagesLostError(LookupError):
    """a pass's stages were lost with the worker holding them: forgotten, and the change is read again."""


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

    def unstage_keys(self, stages: Sequence[str], table: str, key: Sequence[str], keys: Any) -> int:
        """take these keys out of these stages of ``table`` (their keys, their rows), the rest of each kept; the keys
        given."""
        ...

    def drop_stage(self, *stages: str) -> None:
        """drop these stages."""
        ...

    def staged_stages(self) -> list[str]:
        """every stage held."""
        ...

    def apply_staged(self, changes: Sequence[StagedChange], *, mark: int | None = None) -> int:
        """apply every change staged in one transaction, labelled with ``mark``; the rows applied. Raises
        ``LookupError`` when the stages are gone (the worker holding them died; nothing applied): the follower then
        reads the change again. Any other failure keeps the stages for the apply to be tried again."""
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
    :param retry_waits: after a failed pass, the first wait before :meth:`run` tries again and the longest it doubles to
        (seconds)
    :ptype retry_waits: tuple[float, float]
    :param fulls: the full on the delta-files volume (:mod:`~threetears.core.collections.volume_fulls`) for the tables
        that opt in to delta files: a start loads them from it, waiting for one when none is published, and never
        reads them whole from L3; ``checkpoints`` then holds only the other tables. None: every table starts as before
    :ptype fulls: VolumeFulls | None
    :raises ValueError: when a table has a scope and ``stamp_table`` is neither given nor derivable, or a table is both
        in ``fulls`` and in ``checkpoints``, or ``fulls`` holds a table the follower does not follow
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
        retry_waits: tuple[float, float] = RETRY_WAITS,
        fulls: VolumeFulls | None = None,
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
        self._retry_waits = retry_waits
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
        #: the scopes each staged whole read left out because a write held them (their partitions queued, read once
        #: the write commits), with the rows left out of each: a partition of one is never dropped as covered by
        #: that read (:meth:`_stage_partitions`), and each step that reads or keeps them says so
        self._whole_left_out: dict[str, dict[str, int]] = {}
        #: the scopes each table's stage holds rows of (None: rows of no scope, a table without one)
        self._staged_scopes: dict[str, set[str | None]] = {}
        #: the scopes of each key a keyed stage was read for, by table: a scope the stamp moved under while a pass
        #: read (or a write took) leaves the stage by these keys alone, and the rest of the stage is applied
        #: (:meth:`_still`). Kept for keyed stages only: a whole read's or a partition's stage is not split
        self._staged_key_scopes: dict[str, dict[tuple[str, ...], frozenset[str | None]]] = {}
        #: the scopes the last :meth:`_still` left out of the change (their keys put back pending): the label keeps
        #: each at the epoch the copy holds it at, as it keeps a held one (:meth:`_label`)
        self._left_scopes: frozenset[str] = frozenset()
        #: the keys of each table the last :meth:`_still` left out of the change, for :meth:`_leave_out`
        self._leaving: dict[str, set[tuple[str, ...]]] = {}
        #: the mark of the last change applied, and the stamp of each recent one (a reader's snapshot names its mark)
        self._mark = 0
        #: the mark of the change the copy holds now (the last apply that succeeded); None before the first. A pass
        #: that only relabels the copy relabels this one in place (:meth:`_relabel`)
        self._applied_mark: int | None = None
        self._stamps: dict[int, tuple[Any, Mapping[str, str | None]]] = {}
        #: each table's mark when the keys being staged were taken: what the next change applied holds
        self._staging_marks: dict[str, str | None] = {}
        #: each table's mark the copy holds every key of, at or below it: what a checkpoint of the copy records. A table
        #: whose keys a write in progress holds (left pending) keeps the mark it was last current at, so a start from
        #: that checkpoint walks the log from there and hears those keys again
        self._current_marks: dict[str, str | None] = {}
        self._checkpoints = checkpoints
        self._fulls = fulls
        if fulls is not None:
            # an opted-in table's full is on the volume alone: the NATS checkpoint store keeps every other table
            unknown = sorted(set(fulls.tables) - set(self._tables))
            if unknown:
                raise ValueError(f"the volume's fulls name {unknown}, which this follower does not follow")
            both = sorted(set(fulls.tables) & set(checkpoints.table_names if checkpoints is not None else ()))
            if both:
                raise ValueError(f"{both} opt in to delta files: their full is on the volume, never in a checkpoint")
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
        #: each table whose keys the registry named from the change notes (an epoch recorded keys unknown, or a
        #: resolve of being behind), onto where the oldest such notes bracket begins: handed to the checkpoint once an
        #: apply holds the table current past it, so a checkpoint past that epoch is due before the notes can go
        self._noted: dict[str, datetime] = {}
        #: asks for a checkpoint when an epoch applied from the notes makes one due with no change to apply then
        self._checkpoint_wake: asyncio.TimerHandle | None = None
        #: for each table that opts in to delta files, the epochs each key was heard at and not yet applied (None: heard
        #: with no epoch). Kept until an apply holds them, so a key left out of a change, forgotten with its stage or
        #: named again is staged again from the copy's row and every one of its epochs' files, in order, never from L3
        self._epochs_of: dict[str, dict[tuple[str, ...], set[str | None]]] = {
            table.name: {} for table in tables if table.delta_files
        }
        #: the epochs each staged key of such a table was staged from: what an apply of the stage holds
        self._staged_epochs: dict[str, dict[tuple[str, ...], frozenset[str | None]]] = {}
        #: each opted-in table reset or recreated while the follower runs, onto the full it waits for: it is reloaded
        #: from the volume alone, never read from L3 (:meth:`_wait_for_reload`); every other table is followed meanwhile
        self._full_after: dict[str, _FullAfter] = {}
        #: what is heard of each such table while it waits (its keys with their epochs, and its partitions): changes
        #: to the table the reset left, applied over the full once it is reloaded, never over the table it replaced
        self._heard_waiting: dict[str, tuple[dict[tuple[str, ...], set[str | None]], list[dict[str, str]]]] = {}
        #: the task looking for each such table's full
        self._reloads: dict[str, asyncio.Task[None]] = {}

    # --- what the registry tells it -------------------------------------------------------------

    def _on_row(self, table: str) -> Callable[[CacheInvalidationMessage], None]:
        epochs = self._epochs_of.get(table)

        def heard(message: CacheInvalidationMessage) -> None:
            key = tuple(message.ids)
            waiting = self._heard_waiting.get(table)
            if waiting is not None:
                # a change to the table a reset left: applied over its full once reloaded (:meth:`_reload_from_volume`)
                waiting[0].setdefault(key, set()).add(message.generation)
                return
            self._pending[table].add(key)
            if epochs is not None:
                # the epoch names the record, and the record the file the key's change is applied from
                epochs.setdefault(key, set()).add(message.generation)
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
        self._epochs_of.get(table, {}).clear()
        self._partitions[table].clear()
        self._poked.set()

    def schema_changed(self, table: str, token: str | None = None) -> None:
        """a table was dropped or recreated (a migration): read it again whole, once, beside the copy. The one change
        a follower reads a table whole for, counted (``copy_follower_whole_reads_total{why="schema_change"}``). A table
        that opts in to delta files and starts from the volume is never read from L3: it waits for the writer's
        rebuild of it (:meth:`_wait_for_reload`).

        :param table: the table
        :ptype table: str
        :param token: the table's generation token once recreated, when known: an opted-in table's full is taken only
            at or past it
        :ptype token: str | None
        :return: nothing
        :rtype: None
        :raises KeyError: when the follower does not keep it
        """
        if table not in self._tables:
            raise KeyError(table)
        if self._fulls is not None and table in self._fulls.tables:
            self._wait_for_reload(table, "schema_change", token)
            return
        log.warning(
            "a followed table's schema changed; it alone is read again whole", extra={"extra_data": {"table": table}}
        )
        self._read_whole(table, "schema_change")

    def _on_partition(self, table: str) -> Callable[[Mapping[str, str]], None]:
        def loaded(partition: Mapping[str, str]) -> None:
            waiting = self._heard_waiting.get(table)
            if waiting is not None:
                # a load into the table a reset left: read once its full is reloaded
                if dict(partition) not in waiting[1]:
                    waiting[1].append(dict(partition))
                return
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

    def _on_reset(self, table: str) -> Callable[[str | None], None]:
        def reset(token: str | None) -> None:
            if self._fulls is not None and table in self._fulls.tables:
                self._wait_for_reload(table, "reset", token)
                return
            log.info(
                "a followed table was reset by an operator; it alone is read again whole, as on a first start",
                extra={"extra_data": {"table": table, "why": "reset"}},
            )
            self._whole[table] = "reset"
            self._read_whole(table, "reset")

        return reset

    def _wait_for_reload(self, table: str, why: str, token: str | None) -> None:
        """an opted-in table was reset or recreated: never read it from L3 (decisions 12 and 14 of the delta-files
        design: the writer rebuilds it from L3 once and publishes its full). Forget everything the copy heard or staged
        of the table the reset replaced, hold what is heard of it from now on, say in the status that it waits, and look
        for a full holding it at or past the reset (:meth:`_reload_when_published`). The copy keeps serving the table as
        it held it, and keeps following every other table. A second reset while it waits raises the bar to the newer.

        :param table: the table
        :ptype table: str
        :param why: ``reset`` or ``schema_change``
        :ptype why: str
        :param token: the reset's generation token, when the record named it
        :ptype token: str | None
        :return: nothing
        :rtype: None
        """
        after = (
            _FullAfter(why=why, token=token, inclusive=True)
            if token is not None
            else _FullAfter(why=why, token=self._registry.generation_marks.highest_heard(table), inclusive=False)
        )
        log.warning(
            "a followed table that opts in to delta files was reset or recreated: it is reloaded from the writer's "
            "full on the volume once published, never read from L3; the copy serves it as it held it meanwhile",
            extra={"extra_data": {"table": table, "why": why, "token": token, "full_after": after.token}},
        )
        self._full_after[table] = after
        # a stage of the table from before the reset holds the table it replaced: never applied
        self._forget_tables_now([table])
        self._whole.pop(table, None)
        self._pending[table].clear()
        self._epochs_of.get(table, {}).clear()
        self._partitions[table].clear()
        self._heard_waiting[table] = ({}, [])
        self._status.waiting_for_full[table] = f"{why}: looking for the writer's full of the table"
        task = self._reloads.get(table)
        if task is None or task.done():
            self._reloads[table] = asyncio.get_running_loop().create_task(
                self._reload_when_published(table), name=f"copy follower reload of {table}"
            )

    async def _reload_when_published(self, table: str) -> None:
        """look for a full holding ``table`` at or past its reset every ``poll_seconds`` until one is published, then
        reload the table from it; the status says why each look found none. Nothing reads L3 meanwhile.

        :param table: the table
        :ptype table: str
        :return: nothing
        :rtype: None
        """
        fulls = self._fulls
        if fulls is None:
            return  # only a follower starting from the volume waits for a full
        while table in self._full_after and self._status.phase is not FollowerPhase.STOPPED:
            short: str | None
            try:
                manifest = await asyncio.to_thread(current_full_manifest, fulls.root)
                short = self._full_after[table].short_of(table, manifest)
                if short is None:
                    if await self._reload_from_volume(table, fulls, manifest):
                        return
                    short = "the table was reset again while its full loaded; looking again"
            except FileNotFoundError as exc:
                short = f"a part of the current full is not on the volume ({exc.filename}); looking again"
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one failed look leaves the copy as it was; the next looks again
                contain(exc, log=log, what="a copy follower could not reload a reset table from the volume's full")
                short = f"the reload failed ({type(exc).__name__}: {exc}); looking again"
            if table in self._full_after:
                self._status.waiting_for_full[table] = short
            await asyncio.sleep(fulls.poll_seconds)

    async def _reload_from_volume(self, table: str, fulls: VolumeFulls, manifest: FullManifest | None) -> bool:
        """reload ``table`` alone from ``manifest`` in its own change, under the pass lock: its parts staged and applied
        in place of the table the reset replaced, its mark taken as the full's oldest version (the next pass walks the
        log from there, applying each later epoch from its file), and what was heard of it while it waited put back to
        be applied over it.

        :param table: the table
        :ptype table: str
        :param fulls: the volume's fulls
        :ptype fulls: VolumeFulls
        :param manifest: the manifest found
        :ptype manifest: FullManifest | None
        :return: whether it was reloaded (False: reset again since the manifest was judged)
        :rtype: bool
        :raises FileNotFoundError: when a part the manifest names is gone (a newer full swept it)
        """
        async with self._lock:
            after = self._full_after.get(table)
            if after is None or after.short_of(table, manifest) is not None:
                return False
            await self._forget_tables([])  # the stage the reset forgot, among any others
            found = await VolumeFulls(fulls.root, [fulls.tables[table]], poll_seconds=fulls.poll_seconds).stage(
                self._backend, manifest=manifest
            )
            if isinstance(found, FullAbsent):
                return False
            self._mark += 1
            mark = self._mark
            rows = await asyncio.to_thread(self._backend.apply_staged, found.changes, mark=mark)
            self._applied_mark = mark
            floor = found.marks[table]
            self._registry.generation_marks.adopt_generation(table, floor)
            self._current_marks[table] = floor
            label = self._status.stamp
            self._stamps[mark] = (label, dict(self._current_marks))
            while len(self._stamps) > _STAMPS_KEPT:
                del self._stamps[min(self._stamps)]
            keys, partitions = self._heard_waiting.pop(table)
            del self._full_after[table]
            self._status.waiting_for_full.pop(table, None)
            self._pending[table] |= set(keys)
            epochs = self._epochs_of.setdefault(table, {})
            for key, of in keys.items():
                epochs.setdefault(key, set()).update(of)
            for partition in partitions:
                if partition not in self._partitions[table]:
                    self._partitions[table].append(partition)
            self._rows_since_mark += rows
            self._status.applied += 1
            log.info(
                "a reset table was reloaded from the writer's full on the volume",
                extra={
                    "extra_data": {
                        "table": table,
                        "why": after.why,
                        "manifest": found.manifest,
                        "rows": found.rows,
                        "mark": floor,
                        "heard_while_waiting": len(keys),
                    }
                },
            )
            if label is not None:
                for listener in tuple(self._listeners):
                    listener(label)
        self._poked.set()
        return True

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
        if self._fulls is not None:
            return await self._start_from_volume(self._fulls)
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
        self._applied_mark = self._mark
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

    async def _start_from_volume(self, fulls: VolumeFulls) -> bool:
        """:meth:`start` for a follower whose opted-in tables start from the full on the delta-files volume: wait for a
        full holding every one of them (the status says so, and nothing reads L3 meanwhile), stage it, and stage every
        other table from the checkpoint as before; apply both in one transaction, take each table's mark (an opted-in
        table's oldest part version), and read whole only the other tables no checkpoint holds.

        The copy is not labelled with a stamp here: the full's parts are each at their own version, so the copy holds
        one state of every table only once the first catch-up applies what came after each.

        :param fulls: the volume's fulls
        :ptype fulls: VolumeFulls
        :return: False: the copy's derived tables are derived again, since none matches parts at their own versions (and
            False too when the follower was stopped while it waited)
        :rtype: bool
        """
        full = await self._wait_for_full(fulls)
        if full is None:
            return False
        staged = await self._checkpoints.stage() if self._checkpoints is not None else None
        checkpointed = staged is not None and staged.loaded
        marks: dict[str, str | None] = dict(full.marks)
        changes = list(full.changes)
        if staged is not None and checkpointed:
            marks.update(staged.marks)
            changes.extend(staged.changes)
        missing = {
            name: (staged.missing.get(name, "cold_start") if staged is not None else "cold_start")
            for name in self._tables
            if name not in marks
        }
        self._mark += 1
        await asyncio.to_thread(self._backend.apply_staged, changes, mark=self._mark)
        self._applied_mark = self._mark
        self._stamps[self._mark] = (None, dict(marks))
        self._current_marks = dict(marks)
        for name, mark in marks.items():
            self._registry.generation_marks.adopt_generation(name, mark)
        if missing:
            await self._warm_tables(missing)
        self._say(
            FollowerPhase.FOLLOWING,
            f"loaded full {full.manifest} from the volume ({full.rows} rows, {len(full.marks)} tables) and "
            f"{len(marks) - len(full.marks)} tables from a checkpoint ({len(missing)} read whole); catching up",
        )
        self._poked.set()
        return False

    async def _wait_for_full(self, fulls: VolumeFulls) -> StagedFull | None:
        """the current full staged, once the writer has published one holding every opted-in table: until then the
        status says the follower is waiting for the full, and it looks again every ``poll_seconds``. A copy never reads
        an opted-in table whole from L3 (decision 14 of the delta-files design: the writer rebuilds the full from L3
        once). A part the manifest names that is gone (a newer full was published and the old one swept) is looked
        for again the same way.

        :param fulls: the volume's fulls
        :ptype fulls: VolumeFulls
        :return: what was staged; None when the follower was stopped while it waited
        :rtype: StagedFull | None
        """
        while self._status.phase is not FollowerPhase.STOPPED:
            try:
                found = await fulls.stage(self._backend)
            except FileNotFoundError as exc:
                found = FullAbsent(f"a part of the current full is not on the volume ({exc.filename}); looking again")
            if isinstance(found, StagedFull):
                return found
            self._say(FollowerPhase.WAITING, f"waiting for the full: {found.reason}")
            await asyncio.sleep(fulls.poll_seconds)
        return None

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
        """one pass (:meth:`_catch_up`), counted: a failed pass raises as it did, and is counted in the status
        (``failures``, ``behind_since``, FAILING after :data:`FAILING_AFTER` in a row) and in
        ``copy_follower_pass_failures_total`` / ``copy_follower_behind_since_timestamp_seconds``; a good one clears
        them. The copy keeps serving its last change throughout.

        :return: as :meth:`_catch_up`
        :rtype: bool
        """
        try:
            done = await self._catch_up()
        except asyncio.CancelledError, KeyboardInterrupt:
            raise
        except _StagesLostError as exc:
            self._failed("stages_lost", exc)
            raise
        except Exception as exc:
            self._failed("pass", exc)
            raise
        if self._status.failures:
            log.info(
                "copy follower recovered",
                extra={"extra_data": {"failures": self._status.failures, "since": str(self._status.behind_since)}},
            )
        self._status.failures = 0
        self._status.behind_since = None
        _BEHIND_SINCE.set(0)
        return done

    def _failed(self, what: str, exc: BaseException) -> None:
        """count a failed pass: its counter, the status's run of failures and when it began, FAILING at the bound.

        :param what: ``stages_lost`` or ``pass``
        :ptype what: str
        :param exc: what it raised
        :ptype exc: BaseException
        :return: nothing
        :rtype: None
        """
        _PASS_FAILURES.labels(what=what).inc()
        self._status.failures += 1
        if self._status.behind_since is None:
            self._status.behind_since = datetime.now(UTC)
            _BEHIND_SINCE.set(self._status.behind_since.timestamp())
        line = f"a pass failed ({self._status.failures} in a row): {type(exc).__name__}: {exc}"
        if self._status.failures >= FAILING_AFTER:
            self._say(
                FollowerPhase.FAILING,
                f"{line}; serving the copy at {self._status.stamp} since {self._status.behind_since.isoformat()}",
            )
        else:
            self._say(FollowerPhase.WAITING, line)

    async def _catch_up(self) -> bool:
        """apply every key heard at one stamp of the writer's seqlock; label the copy with it.

        At a settled stamp every key heard is applied. While a write is in progress, and the writer names each scope's
        epoch and the scopes the write holds (:class:`Unsettled`), every change of a scope the write does not hold is
        applied and the copy labelled with a :class:`PartialStamp`: a held scope stays at the epoch the copy last
        applied it at, its keys pending until the write commits.

        What a pass reads is staged beside the copy and kept there until a pass ends at a stamp that held still for
        every scope it staged: a pass the stamp moved under keeps its stages, and the next pass reads only the keys
        heard since (a key named again leaves the stage and is read again). A stage holding a scope a write has since
        taken is forgotten (its keys read again once the write commits). A pass whose apply fails keeps every stage
        and tries only the apply again -- unless the backend says the stages are gone (``LookupError``: the worker
        holding them died): then every stage is forgotten and its keys, partitions and whole reads are put back, so the
        next pass reads the change again (never more than the change).

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
                noted = self._registry.take_notes_resolved(name)
                if noted is not None:
                    self._noted[name] = min(self._noted.get(name, noted), noted)
                if reason is not None:
                    # the registry's reason (BEHIND_REASONS): the status and collection_epoch_behind_total say the same
                    behind[name] = reason
                else:
                    self._current_at[name] = now
            self._status.behind = behind
            if behind:
                self._say(FollowerPhase.BEHIND, f"behind on {sorted(behind)}; serving the copy at {self._status.stamp}")
                return False
            # a stage holding a scope the write now holds (or a table of no scope, while any write is) is read again;
            # a keyed stage loses only that scope's keys, once this pass has read (_still)
            await self._forget_tables(
                [
                    name
                    for name, scopes in self._staged_scopes.items()
                    if writing and self._clashes(scopes, held) and (None in scopes or not self._splittable(name))
                ]
            )
            label = self._label(first, held)
            nothing = not any(self._pending.values()) and not self._whole and not any(self._partitions.values())
            if nothing and not self._staged and label == self._status.stamp:
                self._say(FollowerPhase.FOLLOWING, "current" if not writing else "current to every scope not held")
                return True
            if nothing and not self._staged and self._applied_mark in self._stamps:
                return await self._relabel(first, label, writing=writing, held=held)
            # taken with the keys, no await between: every key of an epoch at or below these marks is among them
            self._staging_marks = self.marks()
            pending, whole, partitions = self._take(held, writing=writing)
            try:
                await self._stage(pending, whole, held)
                await self._stage_partitions(partitions, held)
            except LookupError as exc:
                # the worker holding the stages died while this pass staged: what it had not staged is put back, and
                # every stage (this pass's and earlier ones') is forgotten, to be read again
                await self._unstaged(pending, whole, partitions)
                self._stages_lost()
                raise _StagesLostError(f"the stages were lost while staging: {exc}") from exc
            except BaseException:
                # the table being read when it failed is read again; every table staged before it keeps its stage
                await self._unstaged(pending, whole, partitions)
                raise
            second = await self._settled()
            if not self._still(first, second):
                log.info(
                    "copy follower kept its stages for the next pass: the stamp moved while the change was read",
                    extra={
                        "extra_data": {
                            "staged_whole": dict(sorted(self._staged_whole.items())),
                            "staged_tables": sorted(self._staged),
                            "partitions_waiting": {n: len(p) for n, p in sorted(self._partitions.items()) if p},
                            "held": sorted(held),
                        }
                    },
                )
                self._say(FollowerPhase.WAITING, "the stamp moved while the change was read; the next pass tops it up")
                return False
            try:
                await self._leave_out()
            except LookupError as exc:
                self._stages_lost()
                raise _StagesLostError(f"the stages were lost leaving scopes out of the change: {exc}") from exc
            label = self._label(first, held, left=self._left_scopes)
            self._mark += 1
            mark = self._mark
            # a failed apply keeps every stage: the next pass tries only the apply again, never a read -- unless the
            # stages died with the worker holding them: then the change is read again
            try:
                rows = await asyncio.to_thread(self._backend.apply_staged, list(self._staged.values()), mark=mark)
            except LookupError as exc:
                self._stages_lost()
                raise _StagesLostError(f"the stages were lost applying the change: {exc}") from exc
            self._applied_mark = mark
            self._epochs_applied()
            if self._staged_whole or any(self._partitions_staged.values()):
                log.info(
                    "copy follower applied a change",
                    extra={
                        "extra_data": {
                            "mark": mark,
                            "rows": rows,
                            "applied_whole": {
                                n: {"why": w, "left_out": sorted(self._whole_left_out.get(n, {}))}
                                for n, w in sorted(self._staged_whole.items())
                            },
                            "partitions_applied": {n: len(p) for n, p in sorted(self._partitions_staged.items()) if p},
                            "partitions_waiting": {n: len(p) for n, p in sorted(self._partitions.items()) if p},
                            "held": sorted(held),
                        }
                    },
                )
            # a table is current at the marks taken for this pass only when nothing of it is left to read: no key left
            # pending (a scope the write holds, a table of no scope while any write is), no whole read, no partition
            current = [
                name
                for name in self._tables
                if not self._pending[name]
                and name not in self._whole
                and not self._partitions[name]
                and name not in self._full_after
            ]
            for name in current:
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
            self._whole_left_out = {}
            self._staged_scopes = {}
            self._staged_key_scopes = {}
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
            self._checkpoint_when_due(rows, touched, touched_keys, whole_whys, self._noted_held(current))
            return True

    async def _relabel(self, first: Any, label: Any, *, writing: bool, held: frozenset[str]) -> bool:
        """a pass with nothing heard, staged, or to read whole, and a new stamp: the change the copy holds is relabelled
        in place. Its rows are those of the new stamp already, so the backend is asked for nothing: no apply, so on a
        pod's backend no worker round trip, no settle of an empty change, no publish of unchanged rows.

        Why in place: a reader learns its stamp from the mark its snapshot holds (:meth:`reading_marked`), so the new
        label goes on the mark the copy holds now, with each table's mark now (every table is current: nothing is left
        pending). A reader that opened before still reads the same rows; it reads them under the label it read first.
        Everything a labelling pass did besides the apply is kept: the stamp read again and the pass taken only when it
        held still (as :meth:`_still` judges a pass with nothing staged), the listeners told (a reader's epochs moved),
        the checkpoint counted (no rows; it may be due by time or a left-out table).

        Measured (E1, 2026-11-15, S/e1/measure_relabel.py, log S/e1/measure.log): every such pass before this sent
        the backend an empty change. Reports' pod relabels the copy around each write in progress that holds no state
        it is behind on (a PartialStamp in, the settled record out). On reports' shared copy at prod's size (9.52M
        lsf_projections rows, test_follower_change_cost's seed), followed as the pod follows it (settle in the apply
        worker), one empty apply took 181 ms median (144-241 ms, 10 runs; machine load 4.4 at start) for nothing
        changed; followed with no settle, 3.6 ms. Now such a pass costs one stamp read and a dict assignment.
        Counted by ``copy_follower_relabels_total``.

        :param first: what ``settled`` answered first
        :ptype first: Any
        :param label: the stamp the copy is labelled with
        :ptype label: Any
        :param writing: whether a write is in progress
        :ptype writing: bool
        :param held: the scopes the write holds
        :ptype held: frozenset[str]
        :return: whether the copy is now labelled with the new stamp (False: the stamp moved while it was read)
        :rtype: bool
        """
        marks = self.marks()
        second = await self._settled()
        if not self._still(first, second):
            self._say(FollowerPhase.WAITING, "the stamp moved while the change was read; the next pass tops it up")
            return False
        assert self._applied_mark is not None  # noqa: S101 -- narrowed by the caller (a mark in self._stamps)
        self._current_marks.update({name: marks.get(name) for name in self._tables})
        # one assignment: a reader on another thread reads the old entry or the new, never half of one
        self._stamps[self._applied_mark] = (label, dict(self._current_marks))
        self._status.stamp = label
        self._status.relabelled += 1
        _RELABELS.inc()
        self._say(
            FollowerPhase.FOLLOWING,
            "current" if not writing else f"current to every scope not held; {len(held)} held by the write",
        )
        for listener in tuple(self._listeners):
            listener(label)
        self._checkpoint_when_due(0, {}, {}, {}, self._noted_held(self._tables))
        return True

    def _noted_held(self, current: Iterable[str]) -> datetime | None:
        """take the notes times of the tables the copy now holds current: their marks, which a checkpoint records, are
        past every epoch named from the notes so far. A table left pending keeps its own until it is current.

        :param current: the tables whose marks this change moved
        :ptype current: Iterable[str]
        :return: where the oldest of their notes brackets begins; None when none was named from the notes
        :rtype: datetime | None
        """
        held = [self._noted.pop(name) for name in current if name in self._noted]
        return min(held) if held else None

    def _checkpoint_when_due(
        self,
        rows: int,
        touched: Mapping[str, frozenset[str] | None],
        keys: Mapping[str, frozenset[tuple[str, ...]]],
        whys: Mapping[str, str],
        noted: datetime | None = None,
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
        :param noted: where the oldest notes bracket begins of the epochs this change holds that were named from the
            change notes; None when none was
        :ptype noted: datetime | None
        :return: nothing
        :rtype: None
        """
        if self._checkpoints is None:
            return
        self._checkpoints.applied(rows, touched, keys, whys, noted=noted)
        self._checkpoint_if_due()

    def _checkpoint_if_due(self) -> None:
        """write a checkpoint beside the follower when one is due and none is being written; otherwise, when an epoch
        applied from the change notes will make one due, ask again then: after a large commit nothing else may be
        applied before its notes can no longer be read (:meth:`CopyCheckpoints.noted_due_in`).

        :return: nothing
        :rtype: None
        """
        assert self._checkpoints is not None  # noqa: S101 -- narrowed by the callers
        if self._checkpointing is not None and not self._checkpointing.done():
            return  # the write's end asks again
        if self._checkpoint_wake is not None:
            self._checkpoint_wake.cancel()
            self._checkpoint_wake = None
        if self._checkpoints.due():
            self._checkpointing = asyncio.create_task(self._write_checkpoint())
            return
        wait = self._checkpoints.noted_due_in()
        if wait is not None:
            self._checkpoint_wake = asyncio.get_running_loop().call_later(wait, self._checkpoint_if_due)

    async def _write_checkpoint(self) -> None:
        assert self._checkpoints is not None  # noqa: S101 -- narrowed by the caller
        try:
            written = await self._checkpoints.write(self.reading_marked)
        except asyncio.CancelledError, KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - a checkpoint is a start's shortcut; the copy is unaffected
            contain(
                exc,
                log=log,
                what="a copy checkpoint could not be written; the next change that makes one due tries again",
            )
            return
        if written is not None:
            # a change applied while it was written stays counted: asked again once this write is done, so an epoch
            # applied from the notes meanwhile is not left waiting for a change that may not come
            asyncio.get_running_loop().call_soon(self._checkpoint_if_due)

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

    def _label(self, first: Any, held: frozenset[str], *, left: frozenset[str] = frozenset()) -> Any:
        """the stamp the copy holds once this pass applies: ``first`` itself at a settled stamp with nothing left out;
        otherwise a :class:`PartialStamp` keeping each held scope, and each scope left out of the change
        (:meth:`_still`), at the epoch the copy holds it at now.

        :param first: what ``settled`` answered first
        :ptype first: Any
        :param held: the scopes the write holds
        :ptype held: frozenset[str]
        :param left: the scopes left out of this change, their keys read again by the next pass
        :ptype left: frozenset[str]
        :return: the label
        :rtype: Any
        """
        if not isinstance(first, Unsettled) and not left:
            return first
        scoped = _scoped(first)
        epochs = dict(scoped[0]) if scoped is not None else dict(first.epochs or {})
        before = _scoped(self._status.stamp)
        kept = before[0] if before is not None else {}
        older = held | left
        for scope in older:
            epochs.pop(scope, None)
            if scope in kept:
                epochs[scope] = kept[scope]
        return PartialStamp(epochs=epochs, held=older, version=getattr(first, "version", None))

    def _splittable(self, name: str) -> bool:
        """whether a table's stage may lose some keys and keep the rest: a keyed stage, never a whole read's or a
        partition's.

        :param name: the table
        :ptype name: str
        :return: whether it may
        :rtype: bool
        """
        return (
            bool(self._staged_key_scopes.get(name))
            and name not in self._staged_whole
            and not self._partitions_staged.get(name)
        )

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
        """whether a change is left to apply at ``first``: every stage, less the keys of each scope the stamp did not
        hold still for.

        A stamp that names no scope must be the same whole. Otherwise a staged scope that moved between the two reads,
        or that a write in progress now holds, is left out of this change: a keyed stage loses those scopes' keys alone
        (put back pending, read again by the next pass; :meth:`_leave_out` takes them out of the stage), the rest of it
        is applied, and the label keeps each scope left out at the epoch the copy holds it at (:meth:`_label`). A stage
        that cannot be split by key (a whole read's, a partition's), or that every key of would leave, is forgotten
        whole. A table of no scope needs the whole record unmoved and no write in progress: otherwise nothing is
        applied, and a write holding it forgets its stage (its keys read again once the write commits).

        Why per scope: a pass that read for minutes while an LSF load committed a state every ~13 s found some staged
        scope moved on every pass, and kept every stage for the next; the pages froze for the rest of the night
        (reports' burst replay, 2026-11-23: 21 passes in a row kept their stages, none applied after 10:42).

        :param first: what ``settled`` answered before the pass read
        :ptype first: Any
        :param second: what it answered after
        :ptype second: Any
        :return: whether to apply
        :rtype: bool
        """
        self._leaving = {}
        self._left_scopes = frozenset()
        a, b = _scoped(first), _scoped(second)
        if a is None or b is None:
            return bool(second == first)
        (epochs_a, _), (epochs_b, held_b) = a, b
        unsettled = isinstance(second, Unsettled)
        moved = {scope for scope in set(epochs_a) | set(epochs_b) if epochs_a.get(scope) != epochs_b.get(scope)}
        whole_moved = bool(moved) or (unsettled != isinstance(first, Unsettled))
        blocked = moved | (held_b if unsettled else frozenset())
        forget: list[str] = []
        left: set[str] = set()
        whole_rule = False
        for name, scopes in self._staged_scopes.items():
            if None in scopes:
                whole_rule = whole_rule or unsettled or whole_moved
                continue
            hit = {scope for scope in scopes if scope in blocked}
            if not hit:
                continue
            key_scopes = self._staged_key_scopes.get(name)
            if not key_scopes or not self._splittable(name):
                # a whole read's or a partition's stage is never split: the stamp is judged for it as a whole
                whole_rule = True
                continue
            left.update(scope for scope in hit if scope is not None)
            leaving = {key for key, of in key_scopes.items() if of & hit}
            if leaving >= self._staged_keys.get(name, set()):
                forget.append(name)
            elif leaving:
                self._leaving[name] = leaving
        if whole_rule:
            # nothing applies this pass: a stage holding a scope a write took is forgotten (its keys read again once
            # the write commits); one whose scope only moved is kept, its keys heard again topping it up
            self._leaving = {}
            if unsettled:
                self._forget_tables_now(
                    [name for name, scopes in self._staged_scopes.items() if self._clashes(scopes, held_b)]
                )
            return False
        self._forget_tables_now(forget)
        self._left_scopes = frozenset(left)
        # a pass whose every stage was forgotten has nothing to apply; one that staged nothing (a relabel) may
        return not forget or bool(self._staged)

    async def _leave_out(self) -> None:
        """take out of each stage the keys :meth:`_still` left out of the change, and put them back pending: the next
        pass reads them again, at a stamp that holds their scope still.

        :return: nothing
        :rtype: None
        """
        leaving, self._leaving = self._leaving, {}
        for name, keys in leaving.items():
            change = self._staged.get(name)
            if change is None:
                continue
            table = self._tables[name]
            ordered = sorted(keys)
            arrow_keys = pa.table(
                {column: pa.array([key[i] for key in ordered], pa.string()) for i, column in enumerate(table.key)}
            )
            stages = [stage for stage in (change.keys, change.rows) if stage is not None]
            await asyncio.to_thread(self._backend.unstage_keys, stages, name, table.key, arrow_keys)
            self._pending[name] |= keys
            self._staged_keys[name] -= keys
            key_scopes = self._staged_key_scopes[name]
            staged_epochs = self._staged_epochs.get(name, {})
            for key in keys:
                key_scopes.pop(key, None)
                # its epochs stay heard: the next pass stages it again from its files, never from L3
                staged_epochs.pop(key, None)
            self._staged_scopes[name] = {scope for of in key_scopes.values() for scope in of}
        if leaving:
            log.info(
                "copy follower left scopes out of a change: the stamp moved under them while the change was read",
                extra={
                    "extra_data": {
                        "left_keys": {name: len(keys) for name, keys in sorted(leaving.items())},
                        "left_scopes": len(self._left_scopes),
                    }
                },
            )

    def _epochs_applied(self) -> None:
        """after an apply: each staged key's epochs it was staged from are held by the copy, so they are no longer
        heard. An epoch heard for the key since it was staged stays, to be applied by the next pass.

        :return: nothing
        :rtype: None
        """
        staged, self._staged_epochs = self._staged_epochs, {}
        for name, of in staged.items():
            heard = self._epochs_of[name]
            for key, epochs in of.items():
                left = heard.get(key)
                if left is None:
                    continue
                left -= epochs
                if not left:
                    del heard[key]

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
            self._staged_key_scopes.pop(name, None)
            # its keys' epochs stay heard: each key is staged again from its files
            self._staged_epochs.pop(name, None)
            self._pending[name] |= self._staged_keys.pop(name, set())
            self._staged_touched.pop(name, None)
            self._partitions[name].extend(self._partitions_staged.pop(name, []))
            why = self._staged_whole.pop(name, None)
            if why is not None:
                self._whole.setdefault(name, why)
                self._whole_left_out.pop(name, None)
            self._dropping.update((f"{STAGE_PREFIX}{name}__rows", f"{STAGE_PREFIX}{name}__keys"))

    def _stages_lost(self) -> None:
        """the backend lost every stage (the worker holding them died): forget them all, putting back what each held
        to be read again; the next pass drops whatever names are left and reads the change again.

        :return: nothing
        :rtype: None
        """
        self._forget_tables_now(sorted(self._staged))

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
                self._epochs_of.get(name, {}).clear()
                self._staged_epochs.pop(name, None)
                self._whole.pop(name, None)
                self._partitions[name] = []
                why = whole[name]
                # the reason is known here: each is read by its own function, and declared as that situation
                kind: WholeKind
                if why == "schema_change":
                    kind = "schema_change"
                    scopes = await whole_read(
                        self._stage_schema_change(table, rows_stage, held),
                        kind="schema_change",
                        reason="a followed table was dropped or recreated by a migration: that table alone is read again",
                        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade",
                        proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
                        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
                    )
                elif why == "legacy_whole_record":
                    kind = "rolling_upgrade"
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
                    kind = "cold_start"
                    scopes = await whole_read(
                        self._stage_cold_start(table, rows_stage, why, held),
                        kind="cold_start",
                        reason="the copy holds nothing valid for the table: a first start with no checkpoint, or an operator's reset",
                        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade",
                        proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
                        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
                    )
                # the apply empties the copy's table under the same declared kind as this read (apply_staged)
                self._staged[name] = StagedChange(
                    table=name, key=table.key, rows=rows_stage, whole=True, whole_kind=kind
                )
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
            staged, scopes, key_scopes = await self._stage_keyed(table, sorted(keys), keys_stage, rows_stage, held)
            if staged:
                was = self._staged.get(name)
                self._staged[name] = StagedChange(
                    table=name,
                    key=table.key,
                    keys=keys_stage,
                    rows=rows_stage,
                    whole=was is not None and was.whole,
                    whole_kind=None if was is None else was.whole_kind,
                )
                self._staged_keys.setdefault(name, set()).update(staged)
                if table.scope is None:
                    self._staged_touched.setdefault(name, set()).update(staged)
                self._staged_scopes.setdefault(name, set()).update(scopes)
                self._staged_key_scopes.setdefault(name, {}).update(key_scopes)
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
        self,
        table: FollowedTable,
        columns: Annotated[Sequence[str], DeclaredKey],
        values: Sequence[Sequence[str]],
        *,
        selected: str | None = None,
    ) -> Any:
        """the copy's own key and scope columns of the rows some keys, or key prefixes, name: a local read joined on
        a leading part of the key (the whole key, or a load's prefix), never L3.

        :param table: the table
        :ptype table: FollowedTable
        :param columns: the key columns named, a leading part of the key in key order
        :ptype columns: Sequence[str]
        :param values: each named column's values, as text, one list a column and one entry a key
        :ptype values: Sequence[Sequence[str]]
        :param selected: the select list over ``t``; None selects the key and scope columns
        :ptype selected: str | None
        :return: the rows
        :rtype: pyarrow.Table
        """
        selected = self._copy_columns(table) if selected is None else selected
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
    ) -> tuple[set[tuple[str, ...]], set[str | None], dict[tuple[str, ...], frozenset[str | None]]]:
        """stage some keys' rows, read from L3 by key a page at a time -- or, for a table that opts in to delta files,
        merged from their epochs' files onto the copy's rows (:meth:`_rows_from_files`), L3 read only for a key whose
        file cannot be had; a key whose row (in L3 or its file, or in the copy) is of a scope a write holds is left
        pending, unstaged.

        :return: the keys staged, the scopes their rows are of, and each staged key's scopes
        :rtype: tuple[set[tuple[str, ...]], set[str | None], dict[tuple[str, ...], frozenset[str | None]]]
        """
        at = table.scope_in_key
        #: for a table that opts in to delta files: each epoch's file, read once this pass; the keys read from L3
        files: dict[str, tuple[dict[tuple[str, ...], list[DeltaChange]] | str, str | None]] = {}
        fallbacks: dict[str, dict[str, Any]] = {}
        epochs: dict[tuple[str, ...], frozenset[str | None]] = {}
        staged: set[tuple[str, ...]] = set()
        scopes: set[str | None] = set()
        key_scopes: dict[tuple[str, ...], frozenset[str | None]] = {}
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
            if table.delta_files:
                rows, epochs = await self._rows_from_files(table, chunk, files, fallbacks)
            else:
                (rows,) = await self._rows_from_l3(table, chunk)
            kept = list(chunk)
            if table.scope is None:
                scopes.add(None)
                key_scopes.update((tuple(key), frozenset({None})) for key in chunk)
            elif at is not None:
                for key in chunk:
                    in_key = table.scope_of(key[at])
                    scopes.add(in_key)
                    key_scopes[tuple(key)] = frozenset({in_key})
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
                    of_key = of.get(tuple(key), set())
                    scopes.update(of_key)
                    key_scopes[tuple(key)] = frozenset(of_key)
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
            if table.delta_files:
                self._staged_epochs.setdefault(table.name, {}).update((tuple(key), epochs[key]) for key in kept)
        if fallbacks:
            self._fell_back(table, fallbacks)
        return staged, scopes, {key: key_scopes[key] for key in staged if key in key_scopes}

    async def _rows_from_l3(self, table: FollowedTable, keys: Sequence[tuple[str, ...]]) -> list[Any]:
        """some keys' rows as L3 holds them now, read by key in one statement.

        :param table: the table
        :ptype table: FollowedTable
        :param keys: the keys, each its values as text
        :ptype keys: Sequence[tuple[str, ...]]
        :return: one Arrow table of the rows that exist, typed as the rail answers them
        :rtype: list[pyarrow.Table]
        """
        read = self._reads[table.name]
        rows: list[Any] = await read_keyed_arrow(
            self._l3,
            table.name,
            keys=KeyBatch(table.key, keys, read.casts),
            types=read.types,
            batch=len(keys),
        )
        return rows

    async def _delta_file_of(
        self,
        table: FollowedTable,
        epoch: str,
        files: Mapping[str, tuple[dict[tuple[str, ...], list[DeltaChange]] | str, str | None]],
    ) -> tuple[dict[tuple[str, ...], list[DeltaChange]] | str, str | None]:
        """the changes of one epoch's delta file, by key: the file its record names, read off the shared volume, or
        taken from another epoch of this pass that named the same file (each of a commit's records names its one
        file), so a file is read once a pass.

        :param table: a table that opts in to delta files
        :ptype table: FollowedTable
        :param epoch: the epoch, as its advance wrote it
        :ptype epoch: str
        :param files: the files read this pass, by epoch
        :ptype files: Mapping[str, tuple[dict[tuple[str, ...], list[DeltaChange]] | str, str | None]]
        :return: each key's changes in the order written, or why the file cannot be had (a
            :data:`_DELTA_FILE_FALLBACKS` why); and the file's name when the record names one
        :rtype: tuple[dict[tuple[str, ...], list[DeltaChange]] | str, str | None]
        :raises GenerationUnavailableError: when the log cannot be read (the pass fails and is tried again)
        """
        record = None if self._changes is None else await self._changes.change(table.name, epoch)
        if record is None:
            return "record_missing", None
        if record.delta_file is None:
            return "no_file_named", None
        for read, named in files.values():
            if named == record.delta_file and not isinstance(read, str):
                return read, named
        root = delta_files_root()
        if root is None:
            return "file_missing", record.delta_file
        try:
            delta: DeltaFile = await asyncio.to_thread(read_delta_file, root / record.delta_file)
        except FileNotFoundError:
            # NOSILENT: a lost volume; the caller reads the file's keys from L3 by key, counted and logged at WARNING
            return "file_missing", record.delta_file
        of: dict[tuple[str, ...], list[DeltaChange]] = {}
        for change in delta.changes:
            of.setdefault(tuple(key_text(change.row[column]) for column in table.key), []).append(change)
        return of, record.delta_file

    def _held_rows(
        self, table: FollowedTable, keys: Sequence[tuple[str, ...]]
    ) -> dict[tuple[str, ...], dict[str, Any]]:
        """the rows the copy holds of some keys, each as the L3 rail answers it: a local read, never L3.

        Why the copy's: an upsert keeps every column its write left out, and the row's ``date_created``, from the row
        L3 held, which is the row the copy holds before the change (``design-pod-delta-files.md``, decision 19).

        :param table: the table
        :ptype table: FollowedTable
        :param keys: the keys, each its values as text
        :ptype keys: Sequence[tuple[str, ...]]
        :return: each held key's row, keyed by column, typed as the rail answers it
        :rtype: dict[tuple[str, ...], dict[str, Any]]
        """
        types = self._reads[table.name].types
        selected = ", ".join(f"t.{quote_identifier(column.name)}" for column in table.columns)
        columns = [[key[i] for key in keys] for i in range(len(table.key))]
        mine = self._copy_rows_of_keys(table, table.key, columns, selected=selected)
        held: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in mine.to_pylist():
            typed = {name: _rail_python(value, types[name]) for name, value in row.items()}
            held[tuple(key_text(row[column]) for column in table.key)] = typed
        return held

    async def _rows_from_files(
        self,
        table: FollowedTable,
        keys: Sequence[tuple[str, ...]],
        files: dict[str, tuple[dict[tuple[str, ...], list[DeltaChange]] | str, str | None]],
        fallbacks: dict[str, dict[str, Any]],
    ) -> tuple[Any, dict[tuple[str, ...], frozenset[str | None]]]:
        """some keys' rows of a table that opts in to delta files, from their epochs' files: each key's row the copy
        holds, with every change of it in each epoch it was heard at applied in order, as L3 merged them
        (:func:`~threetears.core.collections.delta_files.apply_delta_row`). A key some file of which cannot be had is
        read from L3 by key instead, and counted in ``fallbacks``.

        :param table: the table
        :ptype table: FollowedTable
        :param keys: the keys, each its values as text
        :ptype keys: Sequence[tuple[str, ...]]
        :param files: the files read this pass, by epoch (each read once a pass); filled here
        :ptype files: dict[str, tuple[dict[tuple[str, ...], list[DeltaChange]] | str, str | None]]
        :param fallbacks: the keys read from L3 this pass, by why: their count and the files named; filled here
        :ptype fallbacks: dict[str, dict[str, Any]]
        :return: the rows that exist after the change, typed as the rail answers them; and the epochs each key was
            staged from (an apply holds those)
        :rtype: tuple[pyarrow.Table, dict[tuple[str, ...], frozenset[str | None]]]
        """
        heard = self._epochs_of[table.name]
        epochs = {key: frozenset(heard.get(key, ())) for key in keys}
        ordered: dict[tuple[str, ...], list[str]] = {}
        from_l3: list[tuple[str, ...]] = []
        for key in keys:
            why, named, order = await self._epochs_in_order(table, key, epochs[key], files)
            if why is None:
                ordered[key] = order
                continue
            from_l3.append(key)
            fell = fallbacks.setdefault(why, {"keys": 0, "files": set()})
            fell["keys"] += 1
            if named is not None:
                fell["files"].add(named)
        types = self._reads[table.name].types
        names = [column.name for column in table.columns]
        held = await asyncio.to_thread(self._held_rows, table, list(ordered)) if ordered else {}
        merged: list[dict[str, Any]] = []
        for key, order in ordered.items():
            row: dict[str, Any] | None = held.get(key)
            for epoch in order:
                of = files[epoch][0]
                assert isinstance(of, dict)  # noqa: S101 -- an epoch whose file could not be had sends its key to L3
                for change in of[key]:
                    after = apply_delta_row(row, change)
                    row = None if after is None else {name: after.get(name) for name in names}
            if row is not None:
                merged.append(row)
        rows = pa.Table.from_pylist(merged, schema=pa.schema(list(types.items())))
        if from_l3:
            (read,) = await self._rows_from_l3(table, from_l3)
            rows = pa.concat_tables([rows, read.select(rows.column_names).cast(rows.schema)])
        return rows, epochs

    async def _epochs_in_order(
        self,
        table: FollowedTable,
        key: tuple[str, ...],
        epochs: frozenset[str | None],
        files: dict[str, tuple[dict[tuple[str, ...], list[DeltaChange]] | str, str | None]],
    ) -> tuple[str | None, str | None, list[str]]:
        """the epochs a key's change is applied from, oldest first, each one's file read (once a pass); or why the
        key is read from L3 instead.

        :param table: the table
        :ptype table: FollowedTable
        :param key: the key
        :ptype key: tuple[str, ...]
        :param epochs: the epochs it was heard at (None: one with no epoch)
        :ptype epochs: frozenset[str | None]
        :param files: the files read this pass, by epoch
        :ptype files: dict[str, tuple[dict[tuple[str, ...], list[DeltaChange]] | str, str | None]]
        :return: None and the epochs in order, one a delta file (a commit of several records is applied once); or
            why, and the file named when one is
        :rtype: tuple[str | None, str | None, list[str]]
        """
        parts = [split_generation_token(epoch) if epoch is not None else None for epoch in epochs]
        if not parts or None in parts or len({part[0] for part in parts if part is not None}) != 1:
            # heard with no epoch, or under two incarnations of the log, whose counts do not order them
            return "no_epoch", None, []
        order = [f"{incarnation}:{count}" for incarnation, count in sorted(part for part in parts if part is not None)]
        applied: list[str] = []
        named_files: set[str | None] = set()
        for epoch in order:
            if epoch not in files:
                files[epoch] = await self._delta_file_of(table, epoch, files)
            of, named = files[epoch]
            if isinstance(of, str):
                return of, named, []
            if key not in of:
                # the record named this key at this epoch and its file holds no change of it: never guessed
                return "key_not_in_file", named, []
            if named in named_files:
                # another record of the same commit: its file holds the key's every change of the commit, applied
                # once at the commit's first record; applying it again would replay the commit over itself
                continue
            named_files.add(named)
            applied.append(epoch)
        return None, None, applied

    def _fell_back(self, table: FollowedTable, fallbacks: Mapping[str, Mapping[str, Any]]) -> None:
        """count and say the keys of a table that opts in to delta files a pass read from L3 instead of their files.

        :param table: the table
        :ptype table: FollowedTable
        :param fallbacks: by why, the keys' count and the files named
        :ptype fallbacks: Mapping[str, Mapping[str, Any]]
        :return: nothing
        :rtype: None
        """
        for why, fell in sorted(fallbacks.items()):
            _DELTA_FILE_FALLBACKS.labels(table=table.name, why=why).inc(fell["keys"])
            extra = {
                "extra_data": {
                    "table": table.name,
                    "why": why,
                    "keys": fell["keys"],
                    "files": sorted(fell["files"])[:_FILES_NAMED],
                }
            }
            if why in _DELTA_FILE_LOST:
                log.warning("copy follower read keys from L3 by key: their delta files could not be had", extra=extra)
            else:
                log.info("copy follower read keys from L3 by key: their epochs name no delta file", extra=extra)

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
                # a whole read beside covers the partitions of the scopes it read; never those it left out for a write
                # (queued by :meth:`_stage_whole`, read once the write commits), nor one whose scope its key does not
                # name. Dropped as covered, a left-out scope's rows were missing from the copy for as long as it ran:
                # the whole stage, kept across a pass whose stamp moved, empties the table on apply without them
                # (2026-11-22 replay: IA, SC, SD, TN and TX held none of their L3 rows)
                left_out = self._whole_left_out.get(name, {})
                column = table.scope[0] if table.scope is not None else None
                prefix = table.scope[1] if table.scope is not None else ""
                covered = [
                    p for p in parts if column is not None and column in p and f"{prefix}{p[column]}" not in left_out
                ]
                kept = [p for p in parts if p not in covered]
                log.info(
                    "copy follower dropped the partitions a staged whole read covers",
                    extra={
                        "extra_data": {
                            "table": name,
                            "dropped": sorted(f"{prefix}{p[column]}" for p in covered if column is not None),
                            "kept_to_read": len(kept),
                            "whole_why": self._staged_whole.get(name),
                            "whole_left_out": sorted(left_out),
                            "held": sorted(held),
                        }
                    },
                )
                if not kept:
                    del partitions[name]
                    continue
                # the rest are read as any partition is, into the same stage: the whole read's apply empties the
                # table and puts back its rows and theirs
                parts = kept
                partitions[name] = kept
            self._reading = name
            keys_stage = f"{STAGE_PREFIX}{name}__keys"
            rows_stage = f"{STAGE_PREFIX}{name}__rows"
            types = rail_arrow_types(table.columns)
            selected = ", ".join(quote_identifier(column) for column in types)
            scopes: set[str | None] = set()
            #: a table with no scope: the keys its partitions replace, for the checkpoint's patch
            replaced: set[tuple[str, ...]] = set()
            clashed = False
            #: whether this pass handed the backend a page of keys, of rows: a stage is made by its first page, so a
            #: partition that staged none must not name it (the apply failed on the missing table, every pass after)
            keys_made = rows_made = False
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
                    keys_made = True
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
                    keys_made = rows_made = True
                if clashed:
                    break
            if clashed:
                # a row of a scope the write holds: everything this table staged waits for the write to commit
                log.info(
                    "copy follower put partitions back: a row of a held scope turned up",
                    extra={"extra_data": {"table": name, "partitions": len(parts), "held": sorted(held)}},
                )
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
            # each stage named only once a page made it, here or by an earlier pass's change of the table: partitions
            # neither the copy nor L3 holds a row of change nothing, and stage nothing
            keys_named = keys_stage if keys_made or (was is not None and was.keys is not None) else None
            rows_named = rows_stage if rows_made or (was is not None and was.rows is not None) else None
            if was is not None or keys_named is not None or rows_named is not None:
                self._staged[name] = StagedChange(
                    table=name,
                    key=table.key,
                    keys=keys_named,
                    rows=rows_named,
                    whole=was is not None and was.whole,
                    whole_kind=None if was is None else was.whole_kind,
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
        left_out: dict[str, int] = {}
        staged_rows = 0
        await asyncio.to_thread(self._backend.stage_rows, rows_stage, table.name, pa.table({}), table.key)
        async for page in pages:
            of = self._scopes_of_rows(table, page)
            if held and table.scope is not None:
                for scope in of:
                    if scope is not None and scope in held:
                        left_out[scope] = left_out.get(scope, 0) + 1
                mask = [scope not in held for scope in of]
                page = page.filter(pa.array(mask, pa.bool_()))
                of = [scope for scope, keep in zip(of, mask, strict=True) if keep]
            scopes.update(of)
            staged_rows += page.num_rows
            await asyncio.to_thread(self._backend.stage_rows, rows_stage, table.name, page, table.key)
        queued: list[str] = []
        if held and table.scope is not None:
            column, prefix = table.scope
            for scope in held:
                if scope.startswith(prefix):
                    self._partitions[table.name].append({column: scope[len(prefix) :]})
                    queued.append(scope)
                    left_out.setdefault(scope, 0)
        self._whole_left_out[table.name] = left_out
        log.info(
            "copy follower staged a whole read",
            extra={
                "extra_data": {
                    "table": table.name,
                    "why": why,
                    "rows_staged": staged_rows,
                    "scopes_staged": len(scopes),
                    "held": sorted(held),
                    "left_out_rows": dict(sorted(left_out.items())),
                    "queued_partitions": sorted(queued),
                }
            },
        )
        return scopes or ({None} if table.scope is None else set())

    # --- running ---------------------------------------------------------------------------------

    async def run(self) -> None:
        """apply each change as it is heard, until cancelled: a pass whenever the registry hears a key, catches up, or
        :meth:`poke` is called (the writer's stamp moved). Never on a timer while passes succeed; after a failed pass,
        the next is tried after a wait doubling from the first of ``retry_waits`` to the longest (or sooner when
        poked), FAILING in the status after :data:`FAILING_AFTER` in a row, until one succeeds.

        :return: nothing
        :rtype: None
        """
        while True:
            if self._status.failures:
                # a failed pass is tried again after a wait (doubling to the longest), or sooner when poked: the keys
                # it left pending are never left waiting for a change that may not come
                first, longest = self._retry_waits
                wait = min(longest, first * 2 ** (self._status.failures - 1))
                try:
                    await asyncio.wait_for(self._poked.wait(), timeout=wait)
                except TimeoutError:  # NOSILENT: the wait ran out with no poke: that is the retry's cue, not a failure
                    pass
            else:
                await self._poked.wait()
            try:
                await self.catch_up()
            except asyncio.CancelledError, KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - one failed pass leaves the copy as it was; the next tries again
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
        if self._checkpoint_wake is not None:
            self._checkpoint_wake.cancel()
            self._checkpoint_wake = None
        for task in self._reloads.values():
            task.cancel()
        self._reloads = {}
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
