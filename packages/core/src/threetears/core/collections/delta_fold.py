"""the writer's fold: each state's new part of a full, from its old part and the delta files since.

**Why.** A table that opts in to delta files (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_files`)
saves every commit's rows as a file a pod copy applies (:mod:`~threetears.core.collections.delta_files`), and a copy
starts from the full on the same volume (:mod:`~threetears.core.collections.volume_fulls`). Nothing expires on a timer,
so without a fold a restarted copy would apply every delta since the full was first written, and the volume would hold
every one. The writing load folds them: a change is kept until a newer full covers it (``design-pod-delta-files.md``,
decisions 2, 8, 9, 10, 13, 15, 16, 19 and "How it works", the writer's step 3).

**When a state folds** (decision 10). :meth:`DeltaFolds.committed` is told each commit's delta files after the commit.
A delta file holds every state its commit wrote (decision 6), so each file's bytes are shared among its states by their
rows. Once a state's bytes since its last fold reach half its part's size on the volume, the state folds; a state with
no part yet folds at its first delta.

**How a state folds** (decisions 9 and 19). The old part's rows, then each delta's rows of the state in commit order,
merged as L3 merged them (:func:`~threetears.core.collections.delta_files.apply_delta_row`: a column the write left out
keeps its value, ``date_created`` keeps the held row's, a delete marker deletes), the newest per key. A delta that
clears key prefixes (a load that replaced them, a key-prefix delete; decision 23) first takes every row of the state
under each out, then its own rows apply: a state a prefix names by its scope column shares the file, and every state
of the table does for a prefix that names none. A history table
(AP snapshots, LSF runs) has a key unique to each run or upload, so the same rule appends its rows. Then the table's
keep rule (decision 13, :class:`~threetears.core.collections.delta_keep.DeltaKeepRule`), the one its copy keeps by: a
table that declares none keeps every row.

**A delta as the part** (decision 16). When the last delta holds only the state and, after the keep rule, the folded
rows are exactly its own (a race's newest run replacing the last), the manifest names that delta file as the part and
no row is copied. Files never change once written, and :func:`~threetears.core.collections.volume_fulls.
deletable_full_files` never lists a delta a manifest in use names; the fold keeps such a delta in its list until none
does, then deletes it.

**Published by term** (decision 15). The new parts, then a manifest naming them and every part kept, at the writer's
term, through :func:`~threetears.core.collections.volume_fulls.publish_full_manifest`. The term is the version of the
writer's own write (:meth:`~threetears.core.collections.scope_epochs.ScopeEpochs.begin`): the fencing version taken
under the write lease, durable in L3 and only ever higher for a later writer, never the NATS KV's revision, which a
memory-only bucket resets. A writing load takes its lease and then begins, so it is the first ``begin`` after the
lease; a writer that lost its lease publishes an older term than the next writer's, which no reader takes, or is
refused (:class:`~threetears.core.collections.volume_fulls.StaleTermError`) and changes nothing.

**Deleted after the publish.** Each delta the new full covers (every state it held is folded past it), every part
and manifest no current or previous manifest names, and every orphan delta file (saved before a commit that never
happened, :func:`~threetears.core.collections.delta_files.sweep_orphan_delta_files`) of the tables folded, saved since
the fold started and before the writer's oldest work in flight: a file older than the fold may be a commit it never
heard of, so it is left.

**Never on the commit path.** :meth:`DeltaFolds.committed` only notes the files and starts a round in the background,
off the commit; each round runs in a worker process (:class:`~threetears.core.process_workers.ProcessWorkers`), since
reading, merging and writing parts is CPU, and hands back only names, versions and counts. A round that fails is
logged and counted (``delta_fold_rounds_total{outcome="failed"}``) and its files are kept for the round the next commit
starts; the write it followed has committed already and never fails for it.

**The rebuild from L3** (decisions 12, 14 and 15). A table the current full does not hold is never folded: a full made
from deltas alone would lack every row written before them. The writer rebuilds it instead, once: :meth:`DeltaFolds.
start`, called at the writer's start under its write lease, finds each folded table the current full lacks (no full
at all: a lost volume, or a new environment; or a full without the table: one newly opted in) and reads it from L3
whole, the one whole read a writer makes, declared a cold start; :meth:`DeltaFolds.start_fresh` marks a table reset or
recreated by a migration (its files no longer valid) for the same, read declared as its situation. The table's
generation is read first, then its rows: a commit landing during the read is in the rows and its delta is applied
again over them by every copy, which ends where L3 does (decision 19's merge, applied in order, leaves a row as its
last write left it). The rows are written beside the full as they are read (I/O, the serving process's), then a
worker process splits them by state, applies the keep rule, writes the parts and publishes the manifest at the
writer's term, before any fold of the table: :meth:`DeltaFolds.committed`'s rounds run a rebuild owed first, and a
table owed one is not folded meanwhile. Then its old files go: every delta of the table saved before the rebuild's
read began (and before the writer's work in flight) that neither manifest in use names, and its old parts once the
previous manifest no longer names them. A table with no generation yet (never written) is rebuilt before the round
its first commit starts. A writer whose term is no longer the highest reads nothing and publishes nothing; one whose
lease is taken during its read is refused at the publish, deletes only its own new files, and changes nothing a
reader sees. A copy never reads L3 whole: it waits for the manifest (:class:`~threetears.core.collections.
volume_fulls.VolumeFulls`).

**A restarted writer re-learns its unfolded deltas** (decisions 22 and 24). The fold's memory of each state's deltas
since its last fold dies with its process. :meth:`DeltaFolds.start` learns it again from the NATS epoch log, never
from a listing of the volume and never from L3: the current full says which version each part is current to, and
every commit since has a record in the log naming its delta file. Each table's records after its oldest part's version
are walked (``changes_after``), each file named is noted as a commit is, and a round drops each state's share a part
is already current to (folded before the restart); a file left with no share is deleted with the round's covered
files. So folding resumes where it stopped, at the same half-size point.

**A lost log starts the table fresh.** A table whose generation is of a new incarnation (its store replaced: the
memory-only log lost) has parts current to versions no record of the new log orders, and a walk that misses a record
cannot say which deltas the parts lack. Either way only L3 can say what the table holds: the table starts fresh
(``why="log_lost"``), rebuilt from L3 once under the writer's term, declared a cold start (decision 14: only L3 is
left), logged and counted; its deltas of the old log are dropped with the rebuild. A round that finds a commit of
another incarnation than the table's full marks it the same way rather than folding across it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from threetears.observe import get_logger
from threetears.observe.metrics import counter

from threetears.core.collections.delta_files import (
    _DELETED_COLUMN,
    _PREFIXES_METADATA,
    _WRITTEN_COLUMN,
    _WRITTEN_METADATA,
    DeltaChange,
    apply_delta_row,
    sweep_orphan_delta_files,
    under_prefix,
)
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE
from threetears.core.collections.delta_keep import DeltaKeepRule
from threetears.core.collections.epoch_change import EpochChangeReader
from threetears.core.collections.generation import GenerationSource, split_generation_token
from threetears.core.collections.schema_backed import Column, TableSchema, rail_arrow_types, sql_cast
from threetears.core.declared import whole_read
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.keyset import keyset_arrow_pages
from threetears.core.sql_fragments import quote_identifier
from threetears.core.collections.volume_fulls import (
    FullManifest,
    FullPart,
    FullTable,
    StaleTermError,
    current_full_manifest,
    deletable_full_files,
    publish_full_manifest,
    write_full_part,
)
from threetears.core.process_workers import ProcessWorkers

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as pa_ipc

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a fold refuses without it, saying so
    _PYARROW = False

__all__ = ["DeltaFolds", "DeltaScopeRule", "FoldedTable"]

log = get_logger(__name__)

#: a round's outcome: ``folded`` (a full published), ``nothing_due`` (no state reached half its part), ``no_full`` (no
#: full on the volume holds the table yet: the writer's rebuild publishes one), ``stale`` (a later writer's term is
#: current: this writer's lease was taken), ``failed`` (an error; the round's files are kept for the next), and
#: ``no_generation`` (a commit's file came with no generation to version it by: it is not folded)
_ROUNDS = counter(
    "delta_fold_rounds_total",
    description="writer fold rounds over tables that opt in to delta files, by outcome; failed rounds keep their files "
    "for the round the next commit starts",
    label_names=("outcome",),
)

#: states folded, by table and how: ``written`` (a new part file) or ``reused`` (a delta file named as the part)
_STATES = counter(
    "delta_fold_states_total",
    description="states folded into a new part of a full, by table and how (written: a new part file; reused: the "
    "delta file named as the part, no row copied)",
    label_names=("table", "how"),
)

#: rebuilds of a table's full from L3, by table, why (``cold_start``: no full holds it; ``reset``; ``schema_change``;
#: ``log_lost``: the epoch log that ordered its deltas was replaced, a new incarnation, or lost a record)
#: and outcome: ``rebuilt`` (published), ``stale`` (a later writer's term is current: nothing read or published),
#: ``no_generation`` (the table has none yet: rebuilt before its first commit's round), ``failed`` (retried before
#: the next round)
_REBUILDS = counter(
    "delta_fold_rebuilds_total",
    description="writer rebuilds of a table's full from L3 (its one whole read), by table, why and outcome",
    label_names=("table", "why", "outcome"),
)

#: delta files a restarted writer re-learned from the epoch log (:meth:`DeltaFolds.start`), by table
_RELEARNED = counter(
    "delta_fold_relearned_files_total",
    description="delta files a restarted writer re-learned from the NATS epoch log at its start, to fold, by table",
    label_names=("table",),
)

#: why a table starts fresh: it was reset, or recreated by a migration (:meth:`DeltaFolds.start_fresh`), or the epoch
#: log that ordered its deltas was lost (a new incarnation of its generation, or a record missing)
_FreshWhy = Literal["reset", "schema_change", "log_lost"]

#: where a rebuild's rows are written as they are read, under the fulls' directory: one file a table, removed once its
#: parts are published (or the rebuild fails)
_REBUILD_DIR: Final = ".fulls/rebuild"

#: a state folds once the bytes of its deltas since its last fold reach this share of its part's size (decision 10:
#: half), as a numerator over :data:`_DUE_OF`, so the test is whole numbers: ``bytes * _DUE_OF >= part * _DUE_AT``
_DUE_AT: Final = 1
_DUE_OF: Final = 2

#: the start of every file the fulls write themselves (``volume_fulls``): a part named outside it is a delta file
_FULLS_PREFIX: Final = ".fulls/"

#: one (delta file, state) share a round carries: table, scope, file, generation token, the share's bytes
_Share = tuple[str, str | None, str, str, int]


class DeltaScopeRule:
    """the state a key column's value names, for a table keyed without its state (AP results, keyed by the race,
    ``office_key``); subclass it and override :meth:`state_of`.

    A class, not a function, as :class:`~threetears.core.collections.delta_keep.DeltaKeepRule` is: a round runs in a
    worker process, which is handed plain data and classes (by name), never a function or an object. Declared at
    module level, so the worker can import it.
    """

    @classmethod
    def state_of(cls, value: Any) -> Any:
        """the state the key column's ``value`` names, as its scope's suffix (``VA`` for the scope ``state:VA``).

        Runs in the fold's worker process. Pure: the same value in, the same state out, for every row and every delete
        marker of the key.

        :param value: the key column's value
        :ptype value: Any
        :return: the state
        :rtype: Any
        """
        return value


@dataclass(frozen=True, slots=True)
class FoldedTable:
    """one table the writer folds, as its schema declares it.

    :ivar name: the table
    :ivar key: its key columns, in key order
    :ivar columns: its columns, in the order its delta files and parts hold them
    :ivar scope: the scope each row belongs to, ``(column, prefix)`` as the writer's seqlock names it (a row's state is
        ``prefix + str(row[column])``, or ``prefix + str(scope_from.state_of(row[column]))``); one part a state. The column is
        a key column, so a delete marker (its key alone) still names its state. None: one part for the whole table
    :ivar scope_from: for a table keyed without its state (AP results, keyed by the race, ``office_key``), the rule
        from the scope column's value to the state (:class:`DeltaScopeRule`): ``scope=("office_key", "state:")`` with
        the race's state. None: the column's value is the state
    :ivar keep: the rows a part keeps (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_keep`); None
        keeps every row
    :ivar kinds: each column's type tag, ``(column, column_type)`` in :attr:`columns`' order: how the writer's rebuild
        reads the table from L3 (:meth:`DeltaFolds.start`); empty for a table never rebuilt
    """

    name: str
    key: tuple[str, ...]
    columns: tuple[str, ...]
    scope: tuple[str, str] | None = None
    keep: type[DeltaKeepRule] | None = None
    kinds: tuple[tuple[str, str], ...] = ()
    scope_from: type[DeltaScopeRule] | None = None

    def __post_init__(self) -> None:
        missing = [column for column in self.key if column not in self.columns]
        if missing:
            raise ValueError(f"{self.name}: key columns {missing} are not among its columns")
        if self.scope is not None and self.scope[0] not in self.key:
            # a delete marker carries the key alone: a scope derived from no key column leaves a delete's state
            # unknown, so no one state's part could be folded past it
            raise ValueError(
                f"{self.name}: its scope column {self.scope[0]!r} must be a key column, so each delete marker names "
                f"its state (a state outside the key is derived from a key column with scope_from)"
            )
        if self.scope_from is not None:
            if self.scope is None:
                raise ValueError(f"{self.name}: scope_from derives a state from the scope column; name scope too")
            if not (isinstance(self.scope_from, type) and issubclass(self.scope_from, DeltaScopeRule)):
                raise ValueError(f"{self.name}: scope_from is a DeltaScopeRule subclass")
            if "<" in self.scope_from.__qualname__:
                # a round runs in a worker process, which imports the class by its module and name
                raise ValueError(f"{self.name}: scope_from must be declared at module level, not in a function")

    @classmethod
    def of(
        cls,
        schema: TableSchema,
        *,
        scope: tuple[str, str] | None = None,
        scope_from: type[DeltaScopeRule] | None = None,
    ) -> FoldedTable:
        """a table as its schema declares it: its key, its columns as its delta files type them, its keep rule.

        :param schema: the table's schema; it must opt in to delta files
        :ptype schema: TableSchema
        :param scope: the scope each row belongs to, ``(column, prefix)``, as its copy follows it
            (:attr:`~threetears.core.collections.copy_follower.FollowedTable.scope`); None folds the table whole
        :ptype scope: tuple[str, str] | None
        :param scope_from: the rule from the scope column's value to the state, for a table keyed without its state
            (:attr:`scope_from`); None: the column's value is the state
        :ptype scope_from: type[DeltaScopeRule] | None
        :return: the table
        :rtype: FoldedTable
        :raises ValueError: when the table does not opt in to delta files
        """
        if not schema.delta_files:
            raise ValueError(f"{schema.name}: only a table that opts in to delta files (delta_files=True) is folded")
        return cls(
            name=schema.name,
            key=tuple(schema.pk_columns),
            columns=tuple(rail_arrow_types(schema.columns)),
            scope=scope,
            keep=schema.delta_keep,
            kinds=tuple((column.name, column.column_type) for column in schema.columns),
            scope_from=scope_from,
        )

    def plain(self) -> tuple[Any, ...]:
        """the table as a worker process is handed it: plain data, and its keep rule's class by name.

        :return: ``(name, key, columns, scope column, scope prefix, keep, scope_from)``
        :rtype: tuple
        """
        column, prefix = self.scope if self.scope is not None else (None, None)
        return (self.name, self.key, self.columns, column, prefix, self.keep, self.scope_from)


class DeltaFolds:
    """the writing load's fold of the tables that opt in to delta files: call :meth:`committed` after each commit.

    One a writer, made once and kept: it remembers each state's deltas since its last fold. Only the writer holding
    the write lease makes commits, so only it folds; no other lock.

    :param root: the delta files' root (``delta_files_root()``), the shared volume's mount
    :ptype root: Path
    :param tables: the tables the writer folds
    :ptype tables: Sequence[FoldedTable]
    :param workers: the worker processes the rounds run in; None makes one worker of its own, shut by :meth:`close`
    :ptype workers: ProcessWorkers | None
    """

    def __init__(self, root: Path, tables: Sequence[FoldedTable], *, workers: ProcessWorkers | None = None) -> None:
        if not _PYARROW:
            raise ImportError("a fold needs pyarrow: install 3tears[snapshot]")
        self.root = root
        self.tables: Final = {table.name: table for table in tables}
        self._plain_tables = tuple(table.plain() for table in tables)
        self._owns_workers = workers is None
        self._workers = workers if workers is not None else ProcessWorkers(workers=1, name="delta folds")
        #: delta files older than this the fold never heard named, so its orphan sweep leaves them
        self._since = time.time()
        #: (table, file, token) of each commit not yet handed to a round that finished, in commit order
        self._queued: list[tuple[str, str, str]] = []
        #: the moment the newest queued commit was told, or its writer's oldest work in flight: nothing saved after
        #: it is swept as an orphan
        self._before = self._since
        self._term: int | None = None
        #: each (delta file, state) share not folded yet, the deltas named as parts, the covered deltas not deleted
        self._pending: tuple[_Share, ...] = ()
        self._reused: tuple[str, ...] = ()
        self._doomed: tuple[str, ...] = ()
        self._round: asyncio.Task[None] | None = None
        self._unfolded_warned: set[str] = set()
        #: how the writer reads L3 for a rebuild, from :meth:`start`: None until the writer's start calls it
        self._reader: _Reader | None = None
        #: tables reset or recreated (:meth:`start_fresh`), by why: rebuilt before any round folds them
        self._fresh: dict[str, _FreshWhy] = {}
        #: whether a rebuild may be owed: the next round looks before it folds
        self._recheck = False
        #: whether a table was marked fresh since the last rebuild began
        self._marked = False
        #: each table a rebuild published, in order
        self._rebuilt: list[str] = []

    @property
    def unfolded(self) -> frozenset[str]:
        """the delta files told (or re-learned) and not folded yet, for a status line and the tests.

        :return: the files, under the root
        :rtype: frozenset[str]
        """
        return frozenset(file for _, file, _ in self._queued) | frozenset(share[2] for share in self._pending)

    def committed(
        self,
        files: Mapping[str, str],
        generations: Mapping[str, str],
        *,
        version: int,
        in_flight_since: float | None = None,
    ) -> None:
        """note a commit's delta files and start a round of folds in the background; never waits, never raises for
        a fold. Call it after the commit, on the event loop.

        :param files: table -> its delta file, as the commit's transaction names them
            (:attr:`~threetears.core.collections.caller_transaction.CallerTransaction.delta_files`)
        :ptype files: Mapping[str, str]
        :param generations: table -> the generation token the commit's advance wrote: the version a part folded past
            the file is current to
        :ptype generations: Mapping[str, str]
        :param version: the write's version (:meth:`~threetears.core.collections.scope_epochs.ScopeEpochs.begin`),
            taken under the write lease: the writer's term (module docstring)
        :ptype version: int
        :param in_flight_since: when the writer's oldest transaction still open began, seconds since the epoch, for a
            writer with transactions open beside this commit; None when none is (the orphan sweep leaves every file
            saved after it)
        :ptype in_flight_since: float | None
        :return: nothing
        :rtype: None
        """
        now = time.time()
        for table, file in files.items():
            if table not in self.tables:
                if table not in self._unfolded_warned:
                    self._unfolded_warned.add(table)
                    log.warning(
                        "a commit named a delta file of a table the writer does not fold; its files are never folded "
                        "or deleted",
                        extra={"extra_data": {"table": table, "file": file}},
                    )
                continue
            token = generations.get(table)
            if token is None or split_generation_token(token) is None:
                _ROUNDS.labels(outcome="no_generation").inc()
                log.error(
                    "a commit's delta file came with no generation to version a part by; it is not folded",
                    extra={"extra_data": {"table": table, "file": file, "generation": token}},
                )
                continue
            self._queued.append((table, file, token))
        self._term = version if self._term is None else max(self._term, version)
        self._before = now if in_flight_since is None else min(now, in_flight_since)
        if self._queued:
            self._run_rounds()

    async def start(
        self,
        l3: Any,
        generations: GenerationSource,
        *,
        changes: EpochChangeReader,
        version: int,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> tuple[str, ...]:
        """the writer's start: re-learn from the epoch log the delta files not yet folded (module docstring, "A
        restarted writer re-learns"), then rebuild from L3, once, each folded table the current full on the volume
        lacks (no full at all: a lost volume or a new environment; or one without the table) or whose log was lost,
        and publish them at ``version`` (module docstring, "The rebuild from L3"); then fold each state already due.
        Call it after taking the write lease and beginning the first write, before that write's commit is told to
        :meth:`committed`. Kept for the rebuilds a later round owes (:meth:`start_fresh`, a table with no generation
        yet).

        :param l3: the writer's L3, read whole once for each table rebuilt (the rail's proxy, or a pool)
        :ptype l3: Any
        :param generations: the writer's generation source: each table's generation is read before its rows, the
            version its parts are current to
        :ptype generations: GenerationSource
        :param changes: the NATS epoch log, as a copy reads it: each table's records since its full's parts name the
            delta files not folded yet (decision 24); never L3, never a listing of the volume
        :ptype changes: EpochChangeReader
        :param version: the write's version (:meth:`~threetears.core.collections.scope_epochs.ScopeEpochs.begin`),
            taken under the write lease: the writer's term
        :ptype version: int
        :param page_size: rows a page of the read holds
        :ptype page_size: int
        :return: the tables rebuilt and published, in order; empty when the full held every one (or this writer's
            term is no longer the highest, or a read failed: logged and counted, retried before the next round)
        :rtype: tuple[str, ...]
        :raises ValueError: when ``page_size`` is under one, or a table names no column types (``FoldedTable.of``)
        :raises GenerationUnavailableError: when the epoch log cannot be read: a fold without the deltas it would
            not know of would publish parts lacking their rows, so the writer's start fails and is tried again
        """
        if page_size < 1:
            raise ValueError("page_size must be at least one")
        for table in self.tables.values():
            _table_types(table)  # refuses a table it could not read
        self._reader = _Reader(l3=l3, generations=generations, page_size=page_size)
        self._term = version if self._term is None else max(self._term, version)
        self._recheck = True
        since = len(self._rebuilt)
        await self.wait()
        await self._relearn(generations, changes)
        self._run_rounds()
        await self.wait()
        return tuple(self._rebuilt[since:])

    async def _relearn(self, generations: GenerationSource, changes: EpochChangeReader) -> None:
        """queue, as commits told, each delta file the epoch log names since each table's oldest part (module
        docstring, "A restarted writer re-learns"); a table whose log was lost is owed a fresh start instead.

        :param generations: the writer's generation source: how far each table's log reaches
        :ptype generations: GenerationSource
        :param changes: the epoch log
        :ptype changes: EpochChangeReader
        :return: nothing
        :rtype: None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        current = await asyncio.to_thread(current_full_manifest, self.root)
        if current is None or self._term is None or current.term > self._term:
            return  # no full to continue (the rebuild publishes one), or a later writer's
        told = {file for _, file, _ in self._queued} | {share[2] for share in self._pending}
        for name in sorted(self.tables):
            entry = current.tables.get(name)
            if entry is None or name in self._fresh:
                continue  # rebuilt from L3 before any round: the rebuild's parts are current to L3, no delta owed
            try:
                now = await generations.current(name)
            except (
                GenerationUnavailableError
            ):  # NOSILENT: never written since its store was made; the round of its first commit looks
                continue
            held, at = split_generation_token(entry.version), split_generation_token(now)
            if held is None or at is None:
                continue
            if held[0] != at[0]:
                self._owe_fresh(name, "log_lost", detail=f"the full is of incarnation {held[0]}, the log of {at[0]}")
                continue
            low = min(
                (
                    parsed[1]
                    for parsed in (split_generation_token(part.version) for part in entry.parts)
                    if parsed is not None and parsed[0] == held[0]
                ),
                default=held[1],
            )
            if at[1] <= low:
                continue
            walk = await changes.changes_after(name, f"{held[0]}:{low}", now)
            unnamed = [change.token for change in walk.changes if change.delta_file is None]
            if not walk.complete or unnamed:
                self._owe_fresh(
                    name,
                    "log_lost",
                    detail=f"first missing {walk.first_missing}, records naming no file {unnamed[:5]}",
                )
                continue
            files: dict[str, str] = {}
            for change in walk.changes:
                assert change.delta_file is not None  # noqa: S101 -- refused above
                files[change.delta_file] = change.token  # a commit of several records: its newest
            present = await asyncio.to_thread(_present, self.root, list(files))
            # a file gone was covered by a published full and deleted (nothing else deletes a committed delta)
            learned = [(name, file, files[file]) for file in present if file not in told]
            self._queued.extend(learned)
            _RELEARNED.labels(table=name).inc(len(learned))
            log.info(
                "a restarted writer re-learned its unfolded delta files from the epoch log",
                extra={
                    "extra_data": {
                        "table": name,
                        "since": f"{held[0]}:{low}",
                        "upto": now,
                        "records": len(walk.changes),
                        "files": len(learned),
                        "gone": len(files) - len(present),
                    }
                },
            )

    def start_fresh(self, table: str, *, why: _FreshWhy) -> None:
        """start ``table`` fresh: it was reset (``SchemaBackedCollection.reset``) or dropped and recreated by a
        migration, so its delta files and parts no longer hold it. Rebuilt from L3 under the writer's term before any
        round folds it again, its old files dropped (decision 12). Call it once the reset or the migration has
        committed; never waits.

        :param table: the table
        :ptype table: str
        :param why: ``"reset"`` or ``"schema_change"``
        :ptype why: str
        :return: nothing
        :rtype: None
        :raises KeyError: when the writer does not fold the table
        :raises ValueError: for another why
        """
        if table not in self.tables:
            raise KeyError(table)
        if why not in ("reset", "schema_change"):
            raise ValueError(f"a table starts fresh after a reset or a schema change, not {why!r}")
        self._owe_fresh(table, why)

    def _owe_fresh(self, table: str, why: _FreshWhy, *, detail: str = "") -> None:
        """mark ``table`` owed a rebuild from L3 before any round folds it again, its old files dropped.

        :param table: the table
        :ptype table: str
        :param why: why it starts fresh
        :ptype why: str
        :param detail: what was found, for the log line
        :ptype detail: str
        :return: nothing
        :rtype: None
        """
        log.warning(
            "a folded table starts fresh: it is rebuilt from L3 and its old files dropped",
            extra={"extra_data": {"table": table, "why": why, "detail": detail}},
        )
        self._fresh[table] = why
        self._recheck = True
        self._marked = True
        if self._reader is not None:
            self._run_rounds()

    def _run_rounds(self) -> None:
        """start the background rounds unless they are running.

        :return: nothing
        :rtype: None
        """
        if self._round is None or self._round.done():
            self._round = asyncio.get_running_loop().create_task(self._rounds(), name="delta folds")

    async def wait(self) -> None:
        """wait for the round in progress, and any it starts, to end.

        :return: nothing
        :rtype: None
        """
        while self._round is not None and not self._round.done():
            await asyncio.shield(self._round)

    async def close(self) -> None:
        """wait for the round in progress, then shut the worker this fold made.

        :return: nothing
        :rtype: None
        """
        await self.wait()
        if self._owns_workers:
            await self._workers.close()

    async def _rounds(self) -> None:
        """run rounds while commits are queued, each after any rebuild owed; a failed round keeps its files and waits
        for the next commit.

        :return: nothing
        :rtype: None
        """
        while True:
            if self._reader is not None and self._recheck:
                self._recheck = False
                self._marked = False
                await self._fresh_fulls(self._reader)
            if not self._queued:
                if self._marked and self._reader is not None:
                    continue  # a table marked fresh during that rebuild is rebuilt now, not at the next commit
                return
            taken, self._queued = self._queued, []
            term = self._term
            assert term is not None  # noqa: S101 -- set by the commit that queued the files
            try:
                answer = await self._workers.run(
                    fold_round,
                    str(self.root),
                    term,
                    self._plain_tables,
                    tuple(taken),
                    self._pending,
                    self._reused,
                    self._doomed,
                    self._since,
                    self._before,
                    tuple(sorted(self._fresh)),
                )
            except BaseException as error:  # prawduct:allow prawduct/broad-except -- a fold never fails a write: logged, counted, retried at the next commit
                self._queued = taken + self._queued
                if isinstance(error, asyncio.CancelledError):
                    raise
                _ROUNDS.labels(outcome="failed").inc()
                log.error(
                    "a delta fold round failed; its files are kept for the round the next commit starts",
                    extra={"extra_data": {"files": len(taken), "error": repr(error)}},
                    exc_info=error,
                )
                return
            self._took(answer)

    def _took(self, answer: tuple[Any, ...]) -> None:
        """keep what a round answered, and count and log it.

        :param answer: :func:`fold_round`'s answer
        :ptype answer: tuple
        :return: nothing
        :rtype: None
        """
        pending, reused, doomed, outcome, folded, deleted, swept, manifest, pid, renewed = answer
        self._pending, self._reused, self._doomed = pending, reused, doomed
        _ROUNDS.labels(outcome=outcome).inc()
        for table in renewed:
            # a commit of another incarnation than the table's full: never folded across, rebuilt from L3 instead
            if table not in self._fresh:
                self._owe_fresh(table, "log_lost", detail="a commit's generation is of another incarnation")
        if outcome == "no_full":
            # no full on the volume (lost since the writer's start): rebuilt before the next round
            self._recheck = True
        for table, _scope, how, _rows in folded:
            _STATES.labels(table=table, how=how).inc()
        if outcome == "stale":
            log.warning(
                "a later writer's full is current: this writer's lease was taken, so its folds publish nothing",
                extra={"extra_data": {"term": self._term, "manifest": manifest}},
            )
        elif folded:
            log.info(
                "delta files folded into a new full",
                extra={
                    "extra_data": {
                        "manifest": manifest,
                        "states": len(folded),
                        "reused": sum(1 for _, _, how, _ in folded if how == "reused"),
                        "deleted": deleted,
                        "swept": swept,
                        "pending": len(pending),
                        "worker": pid,
                    }
                },
            )

    async def _fresh_fulls(self, reader: _Reader) -> None:
        """rebuild each table owed one (no full holds it, or :meth:`start_fresh` marked it), then drop what the
        rebuild covers from the folds' queue. Never raises but on cancellation: a failure is logged, counted and
        retried before the next round.

        :param reader: how L3 is read
        :ptype reader: _Reader
        :return: nothing
        :rtype: None
        """
        term = self._term
        assert term is not None  # noqa: S101 -- set by start() before any round looks
        current = await asyncio.to_thread(current_full_manifest, self.root)
        owed: dict[str, str] = {
            name: "cold_start" for name in self.tables if current is None or name not in current.tables
        }
        owed.update(self._fresh)
        if not owed:
            return
        if current is not None and current.term > term:
            for name, why in owed.items():
                _REBUILDS.labels(table=name, why=why, outcome="stale").inc()
            log.warning(
                "a later writer's full is current: this writer's lease was taken, so it rebuilds nothing",
                extra={"extra_data": {"term": term, "manifest": current.name, "tables": sorted(owed)}},
            )
            return
        staged: list[tuple[str, str, str, int]] = []
        #: each table's files told before its generation was read: covered by its rebuild whatever their incarnation
        known: dict[str, set[str]] = {}
        cutoff = self._before
        # a rebuild's rows left by a writer that died mid-read: only the writer holding the lease writes there
        await asyncio.to_thread(_cleared, self.root / _REBUILD_DIR)
        try:
            for name, why in sorted(owed.items()):
                began = time.time()
                known[name] = {file for table_name, file, _ in self._queued if table_name == name} | {
                    share[2] for share in self._pending if share[0] == name
                }
                try:
                    # the generation before the rows: a commit during the read is in the rows and applied again over
                    # them (module docstring), never missed
                    token = await reader.generations.current(name)
                except GenerationUnavailableError as error:
                    # never written: its first commit makes one, and the round that commit starts looks again
                    self._recheck = True
                    _REBUILDS.labels(table=name, why=why, outcome="no_generation").inc()
                    log.warning(
                        "a table to rebuild has no generation yet: rebuilt before the round its first commit starts",
                        extra={"extra_data": {"table": name, "why": why, "error": str(error)}},
                    )
                    continue
                table = self.tables[name]
                # the one whole read a writer makes, declared as its situation (each function holds its own)
                if why == "schema_change":
                    pages = whole_read(
                        _schema_change_pages(reader, table),
                        kind="schema_change",
                        reason="a folded table was dropped or recreated by a migration: the writer reads that table alone once and publishes its "
                        "full",
                        ruling="Pace 2026-11-12: reset, schema change, rolling upgrade; Pace 2026-11-23: a schema change starts the table fresh, "
                        "one read of it from L3 (design-pod-delta-files decision 12)",
                        proof="packages/core/tests/unit/collections/test_delta_fold.py::test_the_writer_reads_l3_whole_only_for_a_table_owed_a_rebuild",
                        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_rebuild_speed_at_real_size",
                    )
                else:
                    pages = whole_read(
                        _cold_start_pages(reader, table),
                        kind="cold_start",
                        reason="no full on the delta-files volume holds the table (a lost volume, a new environment), it was reset, or the epoch log "
                        "ordering its deltas was lost: the writer reads it once and publishes its full; copies wait for that, never reading L3 whole",
                        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; Pace 2026-11-23: with only L3 left the writer "
                        "rebuilds the full, once (design-pod-delta-files decisions 12 and 14)",
                        proof="packages/core/tests/unit/collections/test_delta_fold.py::test_the_writer_reads_l3_whole_only_for_a_table_owed_a_rebuild",
                        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_rebuild_speed_at_real_size",
                    )
                file, rows = await self._written_beside(table, pages)
                staged.append((name, file, token, rows))
                cutoff = min(cutoff, began)
            if not staged:
                return
            answer = await self._workers.run(
                rebuild_round,
                str(self.root),
                term,
                tuple(self.tables[name].plain() for name, _, _, _ in staged),
                tuple(staged),
                cutoff,
            )
        except BaseException as error:  # prawduct:allow prawduct/broad-except -- a rebuild never fails a write: logged, counted, retried before the next round
            self._recheck = True
            await asyncio.to_thread(_unlink_all, self.root, [file for _, file, _, _ in staged])
            if isinstance(error, asyncio.CancelledError):
                raise
            for name, why in owed.items():
                _REBUILDS.labels(table=name, why=why, outcome="failed").inc()
            log.error(
                "a rebuild of the full from L3 failed; it is tried again before the next round",
                extra={"extra_data": {"tables": sorted(owed), "error": repr(error)}},
                exc_info=error,
            )
            return
        outcome, manifest, rebuilt, deleted = answer
        for name, token, rows in rebuilt:
            why = owed[name]
            _REBUILDS.labels(table=name, why=why, outcome=outcome).inc()
            if outcome != "rebuilt":
                continue
            if self._fresh.get(name) == why:
                self._fresh.pop(name)
            self._rebuilt.append(name)
            self._drop_covered(name, token, known[name])
            log.info(
                "a table's full rebuilt from L3 and published",
                extra={"extra_data": {"table": name, "why": why, "rows": rows, "generation": token}},
            )
        if outcome == "rebuilt":
            log.info(
                "rebuilt full published", extra={"extra_data": {"manifest": manifest, "deleted": deleted, "term": term}}
            )
        else:
            log.warning(
                "a later writer's full is current: this writer's rebuild publishes nothing",
                extra={"extra_data": {"term": term, "manifest": manifest}},
            )

    async def _written_beside(self, table: FoldedTable, pages: AsyncIterator[Any]) -> tuple[str, int]:
        """write a table's rows beside the full as its read from L3 hands them, a page at a time on a thread (Arrow
        writes with the interpreter lock released), for the worker process to split.

        :param table: the table
        :ptype table: FoldedTable
        :param pages: its pages, read whole
        :ptype pages: AsyncIterator[pyarrow.Table]
        :return: the file the rows were written to, under the root, and how many
        :rtype: tuple[str, int]
        """
        types = _table_types(table)
        file = f"{_REBUILD_DIR}/{table.name}/{uuid.uuid4().hex}.arrow"
        sink = await asyncio.to_thread(_RowsFile, self.root / file, pa.schema(list(types.items())))
        try:
            async for page in pages:
                await asyncio.to_thread(sink.write, page)
        finally:
            await asyncio.to_thread(sink.close)
        return file, sink.rows

    def _drop_covered(self, table: str, token: str, known: set[str]) -> None:
        """drop from the folds' queue each of ``table``'s commits its rebuild at ``token`` holds (of that incarnation,
        at or below its count; of another, one told before the generation was read: a commit of the lost log): a copy
        applies only the deltas newer than a part, and a fold of an older one would undo a later write. Their files
        are deleted with the next round's.

        :param table: the table rebuilt
        :ptype table: str
        :param token: the generation its parts are current to
        :ptype token: str
        :param known: the table's files told before that generation was read
        :ptype known: set[str]
        :return: nothing
        :rtype: None
        """
        held = split_generation_token(token)

        def covered(name: str, file: str, share_token: str) -> bool:
            if name != table or held is None:
                return False
            parsed = split_generation_token(share_token)
            if parsed is not None and parsed[0] != held[0]:
                return file in known
            return parsed is not None and parsed[1] <= held[1]

        dropped = {file for name, file, share_token in self._queued if covered(name, file, share_token)}
        dropped |= {share[2] for share in self._pending if covered(share[0], share[2], share[3])}
        self._queued = [item for item in self._queued if not covered(*item)]
        self._pending = tuple(share for share in self._pending if not covered(share[0], share[2], share[3]))
        left = {item[1] for item in self._queued} | {share[2] for share in self._pending}
        self._doomed = tuple(sorted(set(self._doomed) | (dropped - left)))


@dataclass(frozen=True, slots=True)
class _Reader:
    """how the writer reads L3 for a rebuild (:meth:`DeltaFolds.start`)."""

    l3: Any
    generations: GenerationSource
    page_size: int


def _table_types(table: FoldedTable) -> dict[str, Any]:
    """a table's columns' Arrow types as the L3 rail answers them, in order.

    :param table: the table
    :ptype table: FoldedTable
    :return: column -> its Arrow type
    :rtype: dict[str, Any]
    :raises ValueError: when the table names no column types (``FoldedTable.kinds``)
    """
    if not table.kinds:
        raise ValueError(f"{table.name}: a rebuild reads the table by its column types; make it FoldedTable.of")
    return rail_arrow_types([Column(name, kind) for name, kind in table.kinds])


def _lead_array(table: FoldedTable) -> str:
    """the SQL array cast of a table's lead key column, for its keyset read (a declared lead is hashed on YugabyteDB).

    :param table: the table
    :ptype table: FoldedTable
    :return: the cast
    :rtype: str
    """
    return f"{sql_cast(dict(table.kinds)[table.key[0]])}[]"


def _cold_start_pages(reader: _Reader, table: FoldedTable) -> AsyncIterator[Any]:
    """a table's whole read for its rebuild: no full on the volume holds it (a lost volume, a new environment, a table
    newly opted in), or an operator reset it.

    :param reader: how L3 is read
    :ptype reader: _Reader
    :param table: the table
    :ptype table: FoldedTable
    :return: its keyset pages, read whole
    :rtype: AsyncIterator[pyarrow.Table]
    """
    types = _table_types(table)
    selected = ", ".join(quote_identifier(column) for column in types)
    return whole_read(
        keyset_arrow_pages(
            reader.l3,
            f"SELECT {selected} FROM {quote_identifier(table.name)}",  # noqa: S608 - trusted identifiers
            table.key,
            types=types,
            page_size=reader.page_size,
            lead_array=_lead_array(table),
        ),
        kind="cold_start",
        reason="no full on the delta-files volume holds the table (a lost volume, a new environment), it was reset, or the epoch log "
        "ordering its deltas was lost: the writer reads it once and publishes its full; copies wait for that, never reading L3 whole",
        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; Pace 2026-11-23: with only L3 left the writer "
        "rebuilds the full, once (design-pod-delta-files decisions 12 and 14)",
        proof="packages/core/tests/unit/collections/test_delta_fold.py::test_the_writer_reads_l3_whole_only_for_a_table_owed_a_rebuild",
        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_rebuild_speed_at_real_size",
    )


def _schema_change_pages(reader: _Reader, table: FoldedTable) -> AsyncIterator[Any]:
    """a table's whole read for its rebuild: a migration dropped or recreated it.

    :param reader: how L3 is read
    :ptype reader: _Reader
    :param table: the table
    :ptype table: FoldedTable
    :return: its keyset pages, read whole
    :rtype: AsyncIterator[pyarrow.Table]
    """
    types = _table_types(table)
    selected = ", ".join(quote_identifier(column) for column in types)
    return whole_read(
        keyset_arrow_pages(
            reader.l3,
            f"SELECT {selected} FROM {quote_identifier(table.name)}",  # noqa: S608 - trusted identifiers
            table.key,
            types=types,
            page_size=reader.page_size,
            lead_array=_lead_array(table),
        ),
        kind="schema_change",
        reason="a folded table was dropped or recreated by a migration: the writer reads that table alone once and publishes its "
        "full",
        ruling="Pace 2026-11-12: reset, schema change, rolling upgrade; Pace 2026-11-23: a schema change starts the table fresh, "
        "one read of it from L3 (design-pod-delta-files decision 12)",
        proof="packages/core/tests/unit/collections/test_delta_fold.py::test_the_writer_reads_l3_whole_only_for_a_table_owed_a_rebuild",
        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_rebuild_speed_at_real_size",
    )


class _RowsFile:
    """an Arrow IPC file a rebuild's rows are written to as they are read; blocking, each call on a thread.

    :param path: the file
    :ptype path: Path
    :param schema: the rows' schema
    :ptype schema: pyarrow.Schema
    """

    def __init__(self, path: Path, schema: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.schema = schema
        self.rows = 0
        self._sink = pa.OSFile(str(path), "wb")
        self._writer = pa_ipc.new_file(self._sink, schema)

    def write(self, page: Any) -> None:
        self._writer.write_table(page.cast(self.schema))
        self.rows += page.num_rows

    def close(self) -> None:
        try:
            self._writer.close()
        finally:
            self._sink.close()


def _unlink_all(root: Path, files: Sequence[str]) -> None:
    """remove each of ``files`` under ``root``, those already gone included; blocking.

    :param root: the root
    :ptype root: Path
    :param files: the files, relative to it
    :ptype files: Sequence[str]
    :return: nothing
    :rtype: None
    """
    for file in files:
        (root / file).unlink(missing_ok=True)


def _present(root: Path, files: Sequence[str]) -> list[str]:
    """each of ``files`` that is on the volume, in order; blocking. Each is opened by its name, never listed.

    :param root: the delta files' root
    :ptype root: Path
    :param files: the files, relative to it
    :ptype files: Sequence[str]
    :return: those present
    :rtype: list[str]
    """
    return [file for file in files if (root / file).is_file()]


def _cleared(directory: Path) -> None:
    """remove every file under each table's directory of ``directory``; blocking.

    :param directory: the rebuilds' directory
    :ptype directory: Path
    :return: nothing
    :rtype: None
    """
    if not directory.is_dir():
        return
    for table in directory.iterdir():
        if table.is_dir():
            for path in table.iterdir():
                path.unlink(missing_ok=True)


def rebuild_round(
    root: str,
    term: int,
    tables: tuple[tuple[Any, ...], ...],
    staged: tuple[tuple[str, str, str, int], ...],
    cutoff: float,
) -> tuple[Any, ...]:
    """a rebuild, in a worker process: each table's rows read from L3 split into its parts by state, the keep rule
    applied, the parts written and the manifest published at the writer's term; then the tables' old files dropped.

    Plain data in and out (:mod:`~threetears.core.process_workers`): names, versions and counts, never rows.

    :param root: the delta files' root
    :ptype root: str
    :param term: the writer's term
    :ptype term: int
    :param tables: each rebuilt table, :meth:`FoldedTable.plain`
    :ptype tables: tuple
    :param staged: each table's ``(table, file of its rows, generation, rows)``, the generation read before its rows
    :ptype staged: tuple
    :param cutoff: when the first table's read began, or the writer's oldest work in flight if older: each of the
        tables' delta files saved before it that no manifest in use names is dropped
    :ptype cutoff: float
    :return: ``(outcome, manifest, rebuilt, deleted)``: ``"rebuilt"`` or ``"stale"``, the manifest published (or
        the current one), each table as ``(table, generation, rows)``, and the files deleted
    :rtype: tuple
    """
    base = Path(root)
    config = {plain[0]: _Folding.of(plain) for plain in tables}
    rebuilt = tuple((name, token, rows) for name, _, token, rows in staged)
    written: list[str] = []
    try:
        current = current_full_manifest(base)
        if current is not None and current.term > term:
            return ("stale", current.name, rebuilt, 0)
        tables_now = dict(current.tables) if current is not None else {}
        for name, file, token, _ in staged:
            parts = _parts_of_rows(base, config[name], _read_ipc(base / file), token)
            written.extend(part.file for part in parts)
            tables_now[name] = FullTable(version=token, parts=parts)
        try:
            published = publish_full_manifest(base, term=term, tables=tables_now)
        except StaleTermError:
            # a later writer published since this one looked: its parts are this writer's alone, named by none
            _unlink_all(base, written)
            stale = current_full_manifest(base)
            return ("stale", stale.name if stale is not None else "", rebuilt, 0)
    except BaseException:
        _unlink_all(base, written)
        raise
    finally:
        _unlink_all(base, [file for _, file, _, _ in staged])
    kept = published.files()
    if published.previous is not None and (base / ".fulls" / "manifests" / published.previous).is_file():
        kept |= FullManifest.decode((base / ".fulls" / "manifests" / published.previous).read_bytes()).files()
    old: list[str] = []
    for name in config:
        directory = base / name
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            file = f"{name}/{path.name}"
            if path.is_file() and file not in kept and path.stat().st_mtime < cutoff:
                old.append(file)
    deleted = 0
    for file in deletable_full_files(base, covered_deltas=sorted(old)):
        try:
            (base / file).unlink(missing_ok=True)
        except OSError as error:  # NOSILENT: logged; the file is listed again by the next round
            log.warning(
                "a file the rebuilt full no longer names could not be deleted",
                extra={"extra_data": {"file": file, "error": repr(error)}},
            )
            continue
        deleted += 1
    return ("rebuilt", published.name, rebuilt, deleted)


def _parts_of_rows(root: Path, table: _Folding, rows: Any, token: str) -> tuple[FullPart, ...]:
    """a table's parts from its rows read whole: one a state (one for a table with no scope), each kept by the
    table's keep rule; a state the rule keeps no row of has no part.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the table
    :ptype table: _Folding
    :param rows: every row L3 held
    :ptype rows: pyarrow.Table
    :param token: the generation the rows are current to
    :ptype token: str
    :return: the parts, written
    :rtype: tuple[FullPart, ...]
    """
    rows = rows.select(list(table.columns)).replace_schema_metadata(None)
    states: list[tuple[str | None, Any]] = []
    if table.scope_column is None:
        states.append((None, rows))
    else:
        values = rows.column(table.scope_column)
        # several values may name one state (a table keyed by the race: each race's rows are its state's), so the
        # values are grouped by the state they name: one part a state
        by_state: dict[str, list[Any]] = {}
        for value in pc.unique(values).to_pylist():
            by_state.setdefault(table.scope_of(value), []).append(value)
        for state, named in by_state.items():
            states.append((state, rows.filter(pc.is_in(values, value_set=pa.array(named, values.type)))))
    parts: list[FullPart] = []
    for scope, held in sorted(states, key=lambda state: str(state[0])):
        kept = table.keep.keep(held) if table.keep is not None else held
        if kept.num_rows:
            parts.append(write_full_part(root, table.name, scope, kept.select(list(table.columns)), token))
    return tuple(parts)


def fold_round(
    root: str,
    term: int,
    tables: tuple[tuple[Any, ...], ...],
    committed: tuple[tuple[str, str, str], ...],
    pending: tuple[_Share, ...],
    reused: tuple[str, ...],
    doomed: tuple[str, ...],
    since: float,
    before: float,
    fresh: tuple[str, ...] = (),
) -> tuple[Any, ...]:
    """one round of folds, in a worker process: note the commits' files, fold each state due, publish, delete.

    Plain data in and out (:mod:`~threetears.core.process_workers`): names, versions and counts, never rows.

    :param root: the delta files' root
    :ptype root: str
    :param term: the writer's term
    :ptype term: int
    :param tables: each folded table, :meth:`FoldedTable.plain`
    :ptype tables: tuple
    :param committed: each new commit's (table, delta file, generation token), in commit order
    :ptype committed: tuple
    :param pending: each (delta file, state) share not folded yet, from the round before
    :ptype pending: tuple
    :param reused: the delta files named as parts, kept until no manifest in use names them
    :ptype reused: tuple[str, ...]
    :param doomed: covered delta files a round before could not delete
    :ptype doomed: tuple[str, ...]
    :param since: when the fold started: older files are left by the orphan sweep
    :ptype since: float
    :param before: the writer's oldest work in flight: newer files are left by the orphan sweep
    :ptype before: float
    :param fresh: tables owed a rebuild (reset or recreated): their files are kept, never folded onto the old parts
    :ptype fresh: tuple[str, ...]
    :return: ``(pending, reused, doomed, outcome, folded, deleted, swept, manifest, pid, renewed)``: the state for
        the next round, the round's outcome (:data:`_ROUNDS`), each state folded as ``(table, scope, how, rows)``, the
        files deleted and swept, the manifest published (or the current one), this worker's process id, and each
        table a commit of another incarnation than its full's was told of (its log lost: rebuilt, never folded)
    :rtype: tuple
    """
    base = Path(root)
    config = {plain[0]: _Folding.of(plain) for plain in tables}
    current = current_full_manifest(base)
    shares = list(pending)
    for table, file, token in committed:
        # a cleared prefix reaches the states the full holds of the table, each counted at its part's size
        held = current.tables.get(table) if current is not None else None
        sizes = {part.scope: (base / part.file).stat().st_size for part in held.parts} if held is not None else {}
        shares.extend(_shares(base, config[table], file, token, sizes))
    renewed: tuple[str, ...] = ()

    def answer(
        outcome: str, manifest: str, *, folded: Sequence[Any] = (), deleted: int = 0, swept: int = 0
    ) -> tuple[Any, ...]:
        return (tuple(shares), reused, doomed, outcome, tuple(folded), deleted, swept, manifest, os.getpid(), renewed)

    if current is None:
        return answer("no_full", "")
    if current.term > term:
        return answer("stale", current.name)
    # a share a part is already current to was folded before the writer restarted (its files re-learned from the
    # epoch log, decision 24): dropped, and a file left with no share deleted with the round's covered files
    named = {share[2] for share in shares}
    shares = [share for share in shares if not _folded_in(current, share)]
    doomed = tuple(sorted(set(doomed) | (named - {share[2] for share in shares} - current.files())))
    renewed = tuple(
        sorted(
            {
                share[0]
                for share in shares
                if share[0] in current.tables
                and share[0] not in fresh
                and not _one_incarnation(current.tables[share[0]].version, share[3])
            }
        )
    )
    # a delta a full names as a part is kept until no manifest in use names it, then deleted: a writer that started
    # after the publish that named it learns it here, from the manifest
    reused = tuple(sorted(set(reused) | {file for file in current.files() if not file.startswith(_FULLS_PREFIX)}))
    by_state: dict[tuple[str, str | None], list[_Share]] = {}
    for share in shares:
        if share[0] in current.tables and share[0] not in fresh and share[0] not in renewed:
            by_state.setdefault((share[0], share[1]), []).append(share)
    tables_now = dict(current.tables)
    folded: list[tuple[str, str | None, str, int]] = []
    folded_shares: set[_Share] = set()
    named_parts: list[str] = []
    for (table, scope), state_shares in by_state.items():
        entry = tables_now[table]
        part = next((p for p in entry.parts if p.scope == scope), None)
        held_bytes = (base / part.file).stat().st_size if part is not None else 0
        if sum(share[4] for share in state_shares) * _DUE_OF < held_bytes * _DUE_AT:
            continue
        new_part, how = _fold_state(base, config[table], scope, part, state_shares)
        if how == "reused":
            named_parts.append(new_part.file)
        tables_now[table] = FullTable(
            version=_newest(table, [entry.version, new_part.version]),
            parts=tuple(p for p in entry.parts if p.scope != scope) + (new_part,),
        )
        folded.append((table, scope, how, new_part.rows))
        folded_shares.update(state_shares)
    if not folded:
        return answer("nothing_due", current.name)
    try:
        published = publish_full_manifest(base, term=term, tables=tables_now)
    except StaleTermError:
        # another writer published a later term since this round read the current full: its parts are orphans
        # the current writer's next round deletes
        return answer("stale", current.name)
    shares = [share for share in shares if share not in folded_shares]
    left = {share[2] for share in shares}
    covered = {share[2] for share in folded_shares} - left
    kept_reused = set(reused) | set(named_parts)
    listed = deletable_full_files(base, covered_deltas=sorted(covered | set(doomed) | kept_reused))
    deleted: set[str] = set()
    for file in listed:
        try:
            (base / file).unlink(missing_ok=True)
        except OSError as error:  # NOSILENT: logged; the file is listed again by the next round
            log.warning(
                "a file the full no longer names could not be deleted",
                extra={"extra_data": {"file": file, "error": repr(error)}},
            )
            continue
        deleted.add(file)
    reused = tuple(sorted(kept_reused - deleted))
    doomed = tuple(sorted((covered | set(doomed)) - deleted - kept_reused))
    named = left | set(reused) | set(doomed) | published.files()
    swept = sweep_orphan_delta_files(base, named=named, before=before, since=since, tables=list(config))
    return answer("folded", published.name, folded=folded, deleted=len(deleted), swept=len(swept))


def _one_incarnation(token: str, other: str) -> bool:
    """whether two generation tokens are of one incarnation of the table's log, so their counts order them.

    :param token: one token
    :ptype token: str
    :param other: the other
    :ptype other: str
    :return: whether both parse and their incarnations match
    :rtype: bool
    """
    first, second = split_generation_token(token), split_generation_token(other)
    return first is not None and second is not None and first[0] == second[0]


def _folded_in(current: FullManifest, share: _Share) -> bool:
    """whether the current full already holds a share: its state's part (or, for a state with no part, the table,
    which then holds no row of the state) is current to the share's generation or a later one of its incarnation.

    :param current: the current full
    :ptype current: FullManifest
    :param share: the share
    :ptype share: _Share
    :return: whether it was folded before
    :rtype: bool
    """
    entry = current.tables.get(share[0])
    if entry is None:
        return False
    part = next((p for p in entry.parts if p.scope == share[1]), None)
    held = split_generation_token(part.version if part is not None else entry.version)
    mine = split_generation_token(share[3])
    return held is not None and mine is not None and held[0] == mine[0] and mine[1] <= held[1]


@dataclass(frozen=True, slots=True)
class _Folding:
    """a folded table as a round reads it from :meth:`FoldedTable.plain`."""

    name: str
    key: tuple[str, ...]
    columns: tuple[str, ...]
    scope_column: str | None
    scope_prefix: str | None
    keep: type[DeltaKeepRule] | None
    scope_from: type[DeltaScopeRule] | None = None

    @classmethod
    def of(cls, plain: tuple[Any, ...]) -> _Folding:
        name, key, columns, column, prefix, keep, scope_from = plain
        return cls(name, tuple(key), tuple(columns), column, prefix, keep, scope_from)

    def scope_of(self, value: Any) -> str:
        """a row's state from its scope column's value (through :attr:`FoldedTable.scope_from` when the column names
        the state only by way of another key, a race), as the writer's seqlock names it."""
        return f"{self.scope_prefix}{value if self.scope_from is None else self.scope_from.state_of(value)}"


def _read_ipc(path: Path) -> Any:
    """an Arrow IPC file's table, read whole.

    :param path: the file
    :ptype path: Path
    :return: its table
    :rtype: pyarrow.Table
    """
    with pa.memory_map(str(path)) as source:
        return pa_ipc.open_file(source).read_all()


def _shares(
    root: Path, table: _Folding, file: str, token: str, sizes: Mapping[str | None, int] | None = None
) -> list[_Share]:
    """a delta file's bytes shared among the states its rows belong to, by rows, and every state a prefix it clears
    reaches (decision 23): the prefix's state when it names the scope column, else every state the full holds.

    A cleared state's share is its part's size: the clear may take every row of the part out in a file of a few
    hundred bytes, so the state folds at once (decision 10's half is reached) rather than keeping a part of rows L3
    no longer holds until other writes add up.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the file's table
    :ptype table: _Folding
    :param file: the file, under the root
    :ptype file: str
    :param token: the generation its commit wrote
    :ptype token: str
    :param sizes: each state the current full holds of the table, and its part's bytes
    :ptype sizes: Mapping[str | None, int] | None
    :return: one share a state
    :rtype: list
    """
    size = (root / file).stat().st_size
    if table.scope_column is None:
        return [(table.name, None, file, token, max(1, size))]
    rows = _read_ipc(root / file)
    counts = Counter(table.scope_of(value) for value in rows.column(table.scope_column).to_pylist())
    total = max(1, rows.num_rows)
    shared = {scope: max(1, size * count // total) for scope, count in counts.items()}
    held = dict(sizes or {})
    for prefix in _cleared_prefixes(rows):
        reached = [table.scope_of(prefix[table.scope_column])] if table.scope_column in prefix else list(held)
        for scope in reached:
            if scope is not None:
                shared[scope] = max(shared.get(scope, 1), held.get(scope, 1))
    return [(table.name, scope, file, token, share) for scope, share in shared.items()]


def _cleared_prefixes(delta: Any) -> list[dict[str, str]]:
    """the key prefixes a delta file clears before its rows, as it names them (decision 23).

    :param delta: the file's table, its schema's metadata read with it
    :ptype delta: pyarrow.Table
    :return: the prefixes
    :rtype: list[dict[str, str]]
    """
    named = (delta.schema.metadata or {}).get(_PREFIXES_METADATA)
    return [dict(prefix) for prefix in json.loads(named)] if named else []


@dataclass(frozen=True, slots=True)
class _Cleared:
    """a step of a fold: every row under the prefixes leaves, before the delta's own changes apply."""

    prefixes: tuple[dict[str, str], ...]


def _held_rows(path: Path, columns: Sequence[str]) -> Any:
    """a part's rows: its table's columns, a delta file named as the part read without its delete markers.

    :param path: the part's file
    :ptype path: Path
    :param columns: the table's columns
    :ptype columns: Sequence[str]
    :return: its rows
    :rtype: pyarrow.Table
    """
    rows = _read_ipc(path)
    if _DELETED_COLUMN in rows.column_names:
        rows = rows.filter(pa.array([not deleted for deleted in rows.column(_DELETED_COLUMN).to_pylist()], pa.bool_()))
    return rows.select(list(columns)).replace_schema_metadata(None)


def _fold_state(
    root: Path, table: _Folding, scope: str | None, part: FullPart | None, shares: Sequence[_Share]
) -> tuple[FullPart, str]:
    """one state's new part: the old part, each delta's rows of the state applied in order, then the keep rule;
    the last delta named as the part when it is exactly what the fold keeps (decision 16), else a new part file.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the table
    :ptype table: _Folding
    :param scope: the state; None for a table folded whole
    :ptype scope: str | None
    :param part: the state's part in the current full; None when it has none
    :ptype part: FullPart | None
    :param shares: the state's shares since its last fold, in commit order
    :ptype shares: Sequence
    :return: the new part, and ``"reused"`` or ``"written"``
    :rtype: tuple[FullPart, str]
    :raises ValueError: when a delta's generation is of another incarnation than the part's (a reset: the rebuild's)
    """
    schema: Any = None
    changes: list[DeltaChange | _Cleared] = []
    last_rows: list[dict[str, Any]] = []
    last_only_state = False
    for index, (_, _, file, _, _) in enumerate(shares):
        delta = _read_ipc(root / file)
        if schema is None:
            schema = pa.schema([delta.schema.field(name).with_nullable(True) for name in table.columns])
        sets = [frozenset(named) for named in json.loads(delta.schema.metadata[_WRITTEN_METADATA])]
        mine = delta
        if scope is not None:
            assert table.scope_column is not None  # noqa: S101 -- a scoped share is a scoped table's
            mask = [table.scope_of(value) == scope for value in delta.column(table.scope_column).to_pylist()]
            mine = delta.filter(pa.array(mask, pa.bool_()))
        is_last = index == len(shares) - 1
        if is_last:
            last_only_state = mine.num_rows == delta.num_rows
            last_rows = []
        cleared = _cleared_prefixes(delta)
        if cleared:
            changes.append(_Cleared(tuple(cleared)))
        for row in mine.to_pylist():
            deleted, written = row.pop(_DELETED_COLUMN), row.pop(_WRITTEN_COLUMN)
            change = DeltaChange(
                row={name: row[name] for name in table.columns}, deleted=deleted, written=sets[written]
            )
            changes.append(change)
            if is_last and not deleted:
                last_rows.append(change.row)
    held = schema.empty_table() if part is None else _held_rows(root / part.file, table.columns).cast(schema)
    version = shares[-1][3]
    if part is not None:
        # refused across a reset: the part and the delta must be of one incarnation
        _newest(table.name, [part.version, version])
    folded = _applied(table, schema, held, changes)
    if table.keep is not None:
        folded = table.keep.keep(folded)
    if last_only_state and _same_rows(table, folded, last_rows):
        last_file = shares[-1][2]
        return FullPart(scope=scope, file=last_file, version=version, rows=len(last_rows)), "reused"
    return write_full_part(root, table.name, scope, folded.select(list(table.columns)), version), "written"


def _applied(table: _Folding, schema: Any, held: Any, changes: Sequence[DeltaChange | _Cleared]) -> Any:
    """the held rows with every change applied in order, merged as L3 merged it (:func:`apply_delta_row`), and every
    row under a cleared prefix taken out where the clear falls among them.

    Rows no change names and no cleared prefix holds stay as Arrow; only the rest are merged a row at a time.

    :param table: the table
    :ptype table: _Folding
    :param schema: the part's schema
    :ptype schema: pyarrow.Schema
    :param held: the old part's rows
    :ptype held: pyarrow.Table
    :param changes: the deltas' changes of the state, and their clears, in order
    :ptype changes: Sequence[DeltaChange | _Cleared]
    :return: the folded rows
    :rtype: pyarrow.Table
    """

    def key_of(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row[name] for name in table.key)

    def under(key: tuple[Any, ...], prefixes: Sequence[Mapping[str, str]]) -> bool:
        return under_prefix(dict(zip(table.key, key, strict=True)), prefixes)

    touched = {key_of(change.row) for change in changes if isinstance(change, DeltaChange)}
    every_cleared = [prefix for change in changes if isinstance(change, _Cleared) for prefix in change.prefixes]
    held_keys = list(zip(*(held.column(name).to_pylist() for name in table.key), strict=True))
    hit = [key in touched or (bool(every_cleared) and under(key, every_cleared)) for key in held_keys]
    merged: dict[tuple[Any, ...], dict[str, Any]] = {
        key_of(row): row for row in held.filter(pa.array(hit, pa.bool_())).to_pylist()
    }
    for change in changes:
        if isinstance(change, _Cleared):
            merged = {key: row for key, row in merged.items() if not under(key, change.prefixes)}
            continue
        key = key_of(change.row)
        after = apply_delta_row(merged.get(key), change)
        if after is None:
            merged.pop(key, None)
        else:
            merged[key] = after
    untouched = held.filter(pa.array([not h for h in hit], pa.bool_()))
    return pa.concat_tables([untouched, pa.Table.from_pylist(list(merged.values()), schema=schema)])


def _same_rows(table: _Folding, folded: Any, rows: Sequence[Mapping[str, Any]]) -> bool:
    """whether the folded rows are exactly ``rows`` (a delta's rows written), one a key.

    :param table: the table
    :ptype table: _Folding
    :param folded: the rows a part would hold
    :ptype folded: pyarrow.Table
    :param rows: the last delta's rows written
    :ptype rows: Sequence[Mapping[str, Any]]
    :return: whether they are the same rows
    :rtype: bool
    """
    if folded.num_rows != len(rows):
        return False
    by_key = {tuple(row[name] for name in table.key): dict(row) for row in rows}
    if len(by_key) != len(rows):
        # a key written twice in one commit: the file holds both rows, a part may hold only one
        return False
    for row in folded.select(list(table.columns)).to_pylist():
        if by_key.get(tuple(row[name] for name in table.key)) != row:
            return False
    return True


def _newest(table: str, tokens: Sequence[str]) -> str:
    """the newest of a table's generation tokens.

    :param table: the table, for the message
    :ptype table: str
    :param tokens: its tokens
    :ptype tokens: Sequence[str]
    :return: the newest
    :rtype: str
    :raises ValueError: when one is not a token, or two are of different incarnations (a reset: the table starts
        fresh from the writer's rebuild, never from a fold)
    """
    split = []
    for token in tokens:
        parsed = split_generation_token(token)
        if parsed is None:
            raise ValueError(f"{table}: {token!r} is not a generation token")
        split.append((parsed, token))
    if len({incarnation for (incarnation, _), _ in split}) != 1:
        raise ValueError(
            f"{table}: generations of incarnations {sorted({i for (i, _), _ in split})}: a reset starts the table "
            f"fresh from the writer's rebuild, never from a fold"
        )
    return max(split)[1]
