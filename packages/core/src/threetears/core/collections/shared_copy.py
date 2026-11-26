"""The copy's apply in a worker process: a change to a copied table rewrites the parts it lands in, and a new shared
Arrow version names every other part's file as the version before it held it.

**Why** (``design-copy-worker-backend.md``). 3tears' in-process copy (``cache.duckdb.DuckDBBackend``) applies each
change on the serving process's threads: CPU beside the event loop. Here the apply runs in one worker process of a
:class:`~threetears.core.shared_arrow.SharedArrowPool` (named ``copy-apply``): the change's pages cross to it
through the store's hand-off (``SharedArrowStore.hand``, Arrow IPC in shared memory, never pickled), it maps the
version it applies to read-only, and it writes each part it rewrites into a result slot in pieces
(``reserve_pieces``). The serving process does IO alone: it hands the pages over and publishes the result on the
version it was applied to (``SharedArrowStore.publish`` with ``base=``), each rewritten part's file linked from the
result, never copied, and every other part's file kept as the base's own. A change costs its parts, not its table's.

**The parts** follow :mod:`~threetears.core.collections.copy_parts`: a table split by its declared scope above the size
floor is a part per scope value; a per-run table (``PartRule.run_column``) a part per scope value and run, in tiers
(:func:`~threetears.core.collections.copy_parts.tier_part`), merged by :meth:`SharedCopyBackend.merge` as a publish of
its own (never between a change and the pages that show it, its merge job); every other table one part
(:data:`~threetears.core.collections.copy_parts.WHOLE_PART`). A table whose held parts are not the shape its rule
names now (it crossed the floor) is rewritten whole in the new shape. A split table keyed without its scope column is
found through its locator (:class:`~threetears.core.collections.copy_parts.Locator`), published in the version beside
it as ``__located__<table>`` (:data:`_LOCATED_PREFIX`), so the worker reads it with the parts and the serving process
holds no copy of it.

**Few, large files a version** (decision 2b). A split table's scope values under ``copy_parts.PACK_BELOW_BYTES``
share packs (``copy_parts.pack_name``) filled to ``copy_parts.PACK_TARGET_BYTES``, named in the version beside it as
``__packed__<table>`` (:data:`_PACKED_PREFIX`: each packed scope value's part name and its pack). A row of a packed
scope lands in its pack, whatever its run; every other row as above. A whole change cuts the table afresh, packed; a
change rewrites the packs and parts it lands in alone, a pack grown past ``PACK_TARGET_BYTES + PACK_BELOW_BYTES``
re-cut as it is rewritten; and the small parts a change leaves (a new small scope, a pack emptied) are packed by
:meth:`SharedCopyBackend.merge`, a publish of its own, like a tier merge. A version holding more files than its bound
(:data:`_MAX_VERSION_FILES`) makes a merge due, which then packs every scope under ``PACK_TARGET_BYTES``.

A whole change of a per-run table (a cold start, a reload) lands in the tiers its :class:`MergePolicy`'s merges would
leave, never a part a run (:func:`_merged_tiers`). A part's name is its scope value's (``copy_parts.part_name``):
escaped, or hashed, for a value the store would refuse as a name.

**A change** is pages in the order they were staged (:class:`ChangePage`): a rows page upserts its rows, a keys page
deletes its keys, and a key's last page wins (a key named in a keys page after its rows page is gone, a rows page after
it is back; a page naming a key twice keeps its last row): the semantics of the staging tables this was moved from
(reports' ``follower_parts.Stages``), settled in Arrow (:func:`~threetears.core.arrow_keys.last_rows`). A leave page
(:data:`LEAVE`) names keys the change leaves as the copy holds them: every earlier page's naming of them is void, a
later page's stands (a follower's ``unstage_keys``, ``copy_stages``). A page may join several pages of a change
(``ChangePage.places``: each run of its rows and its place among them): the worker slices it back into its places
before settling, so a change staged as thousands of small pages crosses as a few. A table named in ``whole`` is
replaced by the change's rows: every part it held is dropped or rewritten.

**A part's rows** are its held rows but those whose keys the change names, plus the rows the change writes to it, in
key order, so a row keeps its place and a column the change leaves alone keeps its column token (``shared_arrow.
stamped``); a change replacing held rows one for one writes each in its row's place, the part never sorted. All of it
in Arrow's kernels and NumPy over the row numbers, never a Python object a row or a key: Python objects here are one a
table, a part or a page.

**An instant in a key.** A table's key columns named in ``instants`` hold an instant as text in one spelling
(``arrow_keys.stored_instants``, in Arrow), of the rows and the keys alike: a key a change record names
(``2026-11-03T23:59:00.000000+00:00``) finds the row the rail wrote (``2026-11-03T23:59:00Z``).

**A key held as text in one spelling.** A table's key columns named in ``canonical`` (a ``NUMERIC`` or a UUID the
copy holds as the rail's text) are kept in their type's one spelling (``arrow_keys.canonical_texts``: ``1.50`` is
``1.5``, a UUID lower-cased), of the rows and the keys alike, once as each change is applied: a lookup
(``copy_queries.CopyQueries.lookup``) then spells only the keys it is asked, never the held column.

**One change at a time, in order.** :class:`SharedCopyBackend` applies the change it is given; its callers give
changes through :class:`~threetears.core.collections.copy_lane.ApplyLane`, which orders them, bounds them, schedules
the merges and replaces a dead worker.

**Product behaviour stays behind a seam** (reports' settle and retention, R-1 of ``design-copy-worker-backend.md``).
:class:`ApplyHooks` (``derive``) runs in the worker after the change is applied. It sees what the change did and the
version with it applied (:class:`AppliedTables`: the rows upserted, the keys deleted and the rows they named as held,
the scopes touched, and :meth:`AppliedTables.table`, a read of any table in place, never a DuckDB connection in the
serving process), and returns (:class:`DerivedPages`) changes to tables of its own (a product's derived tables)
applied into the same result, so they land in the change's publish, and rows to let go (:class:`LetGoRows`, by key or by any of a
table's columns: a whole run, inside a pack too), published as a version of their own right after the change's in the
same apply turn. :class:`WholeHooks` adds ``derive_whole``: after a start that read L3 whole,
:meth:`SharedCopyBackend.defer_hooks_until_whole` applies changes with no ``derive`` until
:meth:`SharedCopyBackend.settle_whole` derives every scope at once, its let-go in the same publish.
:class:`MergePolicy` picks a per-run table's tier merge (:class:`ByCount` by default; reports merges its LSF runs by
their retention's ages), seeing each run's time and model from the table :class:`RunTimes` names. :class:`ColumnHooks`
(``compute``) runs in the worker before a change is written: it computes columns of a table's rows the copy holds and
L3 does not (reports' snapshots' ``most_recent_ind``, the newest upload of each row), over the change's rows and the
version's held rows (:meth:`AppliedTables.table`, mapped), and gives back the change's rows with them computed and any
held row whose value moved, written with the change. All are module-level objects, pickled to the worker by name.

**Parts follow the declared scope as tables grow.** Given ``declared`` (``copy_parts.PartDeclaration``), the copy sizes
its tables after each change's publish and holds each by the rule its size names (``copy_parts.part_rules``): a table
crossing ``SPLIT_FLOOR_BYTES`` is cut by its scope from the next change to it, with no table list written by hand.

**Moved from reports** (``aibots_reports.pod.follower_parts`` and ``shared_backend``): the part rewrite (``_part_rows``,
``_in_place``, ``_in_key_order``), tiers and their merge by count, ``held_parts``, ``file_name``, the change's publish
on its base with its mark (:data:`MARK_TABLE`), the let-go drop (``drop_let_go``, now :class:`LetGoRows`), the whole
settle (``settle_whole``, ``defer_settles_until_whole``) and the run times a merge sees (``_run_times``). Not moved:
reports' settle and retention policy themselves (its hooks) and its DuckDB staging.

**The copy's health** (moved from reports' ``enr_shared_copy_*`` metrics with the apply, R-1). Every failed apply,
reload publish and whole settle is counted by why (``shared_copy_apply_failures_total{why}``: ``died``, the apply
worker died under it, ``BrokenProcessPool``, and :class:`~threetears.core.collections.copy_lane.ApplyLane` gives the
change once more to a fresh one; ``failed``, anything else), the copy serving its last good version meanwhile. After
each, failed or not, the store's room is set (``shared_copy_store_room_bytes``: its bound less what it holds), from the
store's own tallies (``SharedArrowStore.status``), never a directory walk, so the event loop sets it at no cost.

**What the copy lets go** (moved from reports' ``CountedCopyBackend``). Every row the hooks let go, of an apply, a
reload publish or a whole settle, is counted by table (``shared_copy_let_go_rows_total{table}``), and
:attr:`SharedCopyBackend.let_go` holds the same counts since the copy started, for a product's status: each
``Applied.let_go`` added up on the event loop as it publishes, no row read.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Collection, Coroutine, Iterator, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

from threetears.core.arrow_keys import (
    CANONICAL_CASTS,
    canonical_texts,
    distinct_keys,
    key_codes,
    last_rows,
    rows_named,
    stored_instants,
)
from threetears.core.collections.copy_pieces import (
    APPLY_PIECE_BYTES,
    APPLY_PIECE_MINIMUM_BYTES,
    apply_piece_bytes,
    reload_piece_bytes,
)
from threetears.core.collections.copy_parts import (
    PACK_BELOW_BYTES,
    PACK_TARGET_BYTES,
    WHOLE_PART,
    Locator,
    PartDeclaration,
    PartRule,
    is_pack,
    pack_name,
    part_name,
    parts_of_keys,
    tier_of,
    tier_part,
)
from threetears.core.frozen_state import EMPTY, frozen
from threetears.core.shared_arrow import (
    JobContext,
    JobResult,
    MappedSegment,
    Parts,
    PiecedSlot,
    SegmentSlot,
    SharedArrowPool,
    SharedArrowStore,
    VersionFile,
    merged_rows,
    stamped,
    version_files,
)
from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter, gauge

__all__ = [
    "INDEX_PREFIXES",
    "KEYS",
    "LEAVE",
    "MARK_TABLE",
    "ROWS",
    "RUNS_SCHEMA",
    "Applied",
    "AppliedTables",
    "ApplyPlan",
    "ApplyHooks",
    "ByCount",
    "ChangePage",
    "ColumnHooks",
    "ComputedRows",
    "DerivedPages",
    "LetGoRows",
    "LetGone",
    "MergePolicy",
    "Merged",
    "ReloadScope",
    "ReloadStale",
    "Reloaded",
    "RunTimes",
    "SharedCopyBackend",
    "StagedReload",
    "WholeHooks",
    "apply_job",
    "file_name",
    "held_parts",
]

log = get_logger(__name__)

#: the copy's applies, reload publishes and whole settles that failed, by why: ``died`` (the apply worker died under it,
#: ``BrokenProcessPool``: the apply lane gives the change once more to a fresh one) or ``failed`` (anything else). The
#: copy serves its last good version meanwhile. A product's dashboard reads it (reports' copy health row)
_APPLY_FAILURES = counter(
    "shared_copy_apply_failures_total",
    description="shared copy changes (applies, reload publishes, whole settles) that failed, by why (died: the apply "
    "worker died; failed: anything else); the copy serves its last good version",
    label_names=("why",),
)

#: the store's room: its bound less what it holds, from the store's own tallies (``SharedArrowStore.status``, never a
#: directory walk), set after every apply, reload publish and whole settle, failed or not
_STORE_ROOM = gauge(
    "shared_copy_store_room_bytes",
    description="bytes the shared copy's store has left: its bound less what it holds, set after each change",
)

#: the rows the copy's hooks let go (:class:`DerivedPages`'s ``let_go``), by table, of every apply, reload publish and
#: whole settle that published them (``Applied.let_go``). A product's dashboard reads it (reports' retention row)
_LET_GO_ROWS = counter(
    "shared_copy_let_go_rows_total",
    description="rows the shared copy's hooks let go (retention), by table, of each apply, reload publish and whole "
    "settle that published them",
    label_names=("table",),
)

#: a page of rows to upsert (:class:`ChangePage`)
ROWS: Final = "rows"
#: a page of keys to delete (:class:`ChangePage`)
KEYS: Final = "keys"
#: a page of keys the change leaves as the copy holds them: what an earlier page of the change said of each is void
#: (:class:`ChangePage`). Why: a follower takes keys back out of a change it staged when their scope moved while it
#: read (``copy_follower.CopyFollower._leave_out``); settled here, in the apply worker, so the serving process never
#: filters the staged pages
LEAVE: Final = "leave"

#: between a table and its part in a version's file name, and the files' suffix: ``{table}@{part}.arrow``
#: (``shared_arrow.version_files``' contract)
_PART_SEPARATOR: Final = "@"
_TABLE_SUFFIX: Final = ".arrow"

#: a split table's locator is published beside it as this prefix and its name, one part (:data:`WHOLE_PART`)
_LOCATED_PREFIX: Final = "__located__"

#: a split table's packs index is published beside it as this prefix and its name, one part (:data:`WHOLE_PART`): each
#: packed scope value's part name and its pack (``copy_parts.pack_name``)
_PACKED_PREFIX: Final = "__packed__"

#: the prefixes of the tables a version holds beside the copy's own (a locator, a packs index): no query sees them
INDEX_PREFIXES: Final = (_LOCATED_PREFIX, _PACKED_PREFIX)

#: the most files a version of the copy holds before a merge is due to pack its small parts (every table's parts, its
#: schema file, its locator and packs index, the mark). A version over it makes a merge due (:meth:`SharedCopyBackend.
#: merge`) that packs every scope value under ``copy_parts.PACK_TARGET_BYTES``, not only those under
#: ``PACK_BELOW_BYTES``, until it is under again or nothing is left to pack (said at WARNING).
#:
#: **Why 512** (decision 2b; ``research-shared-arrow-free.md``). Letting a version go frees every array of its files,
#: and glibc consolidated the small frees in one ``free()`` holding the GIL: 35-37 ms for the replay's 2,240 files
#: (~200 arrays each), 12.9 ms for 500, 2.6 ms for 100, growing with the arrays, never the bytes. Packed, reports'
#: small scopes no longer add a file each, so a version is its tables' schema files and big parts: a quarter of the
#: replay's count at most, and a fifth of its stall. The let-go lateness test on the Linux pool is the proof at size
#: (``test_shared_arrow_let_go_loop``).
_MAX_VERSION_FILES: Final = 512

#: the one-row table each change's mark is published as (``mark``, int64): a reader of a version reads the mark of
#: the rows it maps, never a newer one. Moved from reports (``follower_parts.MARK_TABLE``)
MARK_TABLE: Final = "__staged_mark__"

#: how many neighbouring parts of one level :class:`ByCount` merges into one of the next. Moved from reports
#: (``follower_parts.FAN_OUT``): a scope of N runs holds about ``FAN_OUT * log10(N)`` parts and each row is rewritten
#: once a level
_FAN_OUT: Final = 10

#: an apply worker holding more private memory than this after a change is replaced before the next, so what a
#: change's scratch copies allocated goes back to the system. Judged by private (anonymous) memory
#: (``shared_arrow.private_bytes``), never resident memory, which counts the store's pages it maps: judged so, reports'
#: apply worker was retired 227 times on cobalt prod (2026-11-01). Moved from reports
#: (``shared_backend.APPLY_WORKER_PRIVATE_BYTES``)
_APPLY_PRIVATE_BYTES: Final = 2 << 30

#: the name the apply worker gives itself (``/proc/<pid>/comm``, 15 bytes at most): what an operator's fault aims at
_APPLY_PROCESS_NAME: Final = "copy-apply"

#: a published change's version: ``c-<n>``
_VERSION: Final = re.compile(r"^c-(?P<n>\d+)$")

#: a part's row's place among the rows of a change: its rows page's row, or none for a keys page's key
_ROW: Final = "__row"

#: whether a key's event is a leave page's (:data:`LEAVE`): a key whose last event is one is left as held
_LEFT: Final = "__left"

#: a scope value no key's locator names (a key the copy never held): its part is none
_NO_SCOPE: Final = "\x00none"


@dataclass(frozen=True, slots=True)
class ChangePage:
    """one page of a change, in the order staged.

    :ivar kind: :data:`ROWS` (upsert its rows), :data:`KEYS` (delete its keys) or :data:`LEAVE` (leave its keys as
        held: the change's earlier pages of them are void)
    :ivar table: the table
    :ivar rows: its rows (every column of the table) or its keys (the key's columns, as text or typed)
    :ivar places: for a page joining several of its change's pages, each run of its rows, in row order: its place and
        its rows. Neighbouring pages with places are one run of the change, applied in place order (a place given
        once, rising across changes: ``copy_stages``); empty for a page applied where it stands

    **Why places.** Each page crosses to the apply worker as a file of its own, and a follower stages a big change a
    page of keys and a page of rows at a time, the two kinds alternating: a commit of 25,000 keys in pages of 50 was
    some 1,900 files, past the process's open files. Joined by kind with their places, they cross as a few, and the
    worker slices them back (zero-copy) into the order they were staged in.
    """

    kind: str
    table: str
    rows: pa.Table
    places: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in (ROWS, KEYS, LEAVE):
            raise ValueError(f"a change's page is {ROWS!r}, {KEYS!r} or {LEAVE!r}, not {self.kind!r}")
        if self.places and sum(rows for _place, rows in self.places) != self.rows.num_rows:
            raise ValueError(
                f"a page's places name {sum(r for _p, r in self.places)} rows; it holds {self.rows.num_rows}"
            )


@dataclass(frozen=True, slots=True)
class AppliedTables:
    """what a change did, as :class:`ApplyHooks` sees it in the worker, and the version with the change applied, read
    in place (:meth:`table`): a pinned handle of the worker's own, never a DuckDB connection in the serving process.

    :ivar root: the store's directory
    :ivar version: the version the change applied to; None for the first
    :ivar rows: each changed table's rows the change upserted (a key once, its last row)
    :ivar deleted: each changed table's keys the change deleted
    :ivar parts: each changed table's parts as the change leaves them, by name (held parts mapped, rewritten parts as
        written); a part the change dropped is absent
    :ivar reloaded: the tables a reload's publish replaced (:meth:`SharedCopyBackend.publish_reload`), whole or by
        scope: their parts are here, their reloaded rows are not in ``rows`` (only the changes applied over them)
    :ivar deleted_rows: each changed table's rows the change deleted and wrote not again, as the version held them
        (every column): a table replaced whole names none (it is read whole, :meth:`table`)
    :ivar whole: the tables the change replaced whole (a whole change's, a whole reload's): read them from
        :meth:`table`, their ``rows`` are only the change's
    :ivar touched: the part names of the scope values the change wrote, deleted or reloaded, of every split table
        (``copy_parts.part_name``: a plain value is its own name); None for every scope (a whole settle,
        :meth:`SharedCopyBackend.settle_whole`)
    :ivar files: the version's files with the reloads in place, before the change's own writes (read by :meth:`table`)
    :ivar rules: each table's rule
    :ivar schemas: each table's columns
    """

    root: str
    version: str | None
    rows: Mapping[str, pa.Table]
    deleted: Mapping[str, pa.Table]
    parts: Mapping[str, Mapping[str, pa.Table]]
    reloaded: frozenset[str] = frozenset()
    deleted_rows: Mapping[str, pa.Table] = EMPTY
    whole: frozenset[str] = frozenset()
    touched: frozenset[str] | None = frozenset()
    files: Mapping[str, Any] = EMPTY
    rules: Mapping[str, PartRule] = EMPTY
    schemas: Mapping[str, pa.Schema] = EMPTY

    def table(self, table: str, scopes: Collection[str] | None = None) -> pa.Table:
        """a table's rows as the version holds them with the change applied: its parts as the change leaves them,
        mapped (never read into this process's memory but for the parts the change wrote).

        :param table: the table
        :ptype table: str
        :param scopes: the scope values' part names whose rows are read (:attr:`touched`, say) of a split table; a
            table held as one part by its rule is read whole. None for every row
        :ptype scopes: Collection[str] | None
        :return: the rows, in the table's schema
        :rtype: pa.Table
        :raises KeyError: for a table the copy does not hold
        """
        schema = self.schemas[table]
        rule = self.rules[table]
        if table in self.parts:
            held: Mapping[str, Any] = self.parts[table]
        else:
            held = _parts_of(self.files, table)
        wanted = None if scopes is None or not rule.split else frozenset(scopes)
        chunks: list[pa.Table] = []
        for part, source in sorted(held.items()):
            # a pack holds many scopes, and so does a split table's one part (it crossed the floor and is re-cut by its
            # next change): only the asked scopes' rows of either
            many = is_pack(part) or part == WHOLE_PART
            if wanted is not None and not many and _scope_part(part) not in wanted:
                continue
            rows = source if isinstance(source, pa.Table) else _map(source, schema)
            if wanted is not None and many:
                rows = rows.filter(pc.is_in(parts_of_keys(rule, rows), value_set=pa.array(sorted(wanted), pa.string())))
            chunks.append(rows.select(schema.names).cast(schema))
        return pa.concat_tables(chunks) if chunks else schema.empty_table()


@dataclass(frozen=True, slots=True)
class LetGoRows:
    """rows a product's hook lets go of a table (a retention letting runs go): every held row equal to one of
    :attr:`rows` on each of its columns (a null equal to a null) is deleted.

    The columns are any of the table's: its whole key (rows let go by key), its scope column and run column (a run's
    rows, a whole tier), or more (a run's rows of one group). A split table's parts are found by the scope column (and
    the run column, a per-run table's) when the rows hold them, through its locator when they hold that, else every
    part is matched. A part is rewritten without the matched rows, other rows of it kept (a pack's other scopes, a
    tier's other runs); a part every row of which is let go is dropped.

    :ivar table: the table
    :ivar rows: the columns rows are matched on, a row a match
    """

    table: str
    rows: pa.Table


@dataclass(frozen=True, slots=True)
class DerivedPages:
    """what :class:`ApplyHooks` derives from a change.

    :ivar pages: pages to tables the change did not carry (a product's derived tables), applied into the change's own
        result, so they land in its publish; a page to a table the change carried is dropped (its rows stand)
    :ivar whole: the derived tables the pages replace whole (every row of them not in the pages goes)
    :ivar let_go: rows let go (:class:`LetGoRows`) of any table, the change's own among them, published as a version
        of their own right after the change's (a change's pages never wait on its let-go), under the same apply turn:
        no change or merge lands between. A whole settle's are in its one publish (:meth:`SharedCopyBackend.
        settle_whole`)
    """

    pages: Sequence[ChangePage] = ()
    whole: frozenset[str] = frozenset()
    let_go: Sequence[LetGoRows] = ()


class ApplyHooks(Protocol):
    """a product's behaviour on each change, run in the apply worker (a module-level object, pickled by name).

    ``derive`` returns pages of further changes to tables the change did not carry (a product's derived tables),
    applied into the same result, so they land in the change's own publish, and rows to let go after it
    (:class:`DerivedPages`); a plain sequence of pages is a :class:`DerivedPages` of those pages alone. A page to a
    table the change carried is dropped: the change's rows of it stand.
    """

    def derive(self, applied: AppliedTables) -> Sequence[ChangePage] | DerivedPages:
        """the derived tables' pages, and what is let go.

        :param applied: what the change did, and the version with it applied (:meth:`AppliedTables.table`)
        :ptype applied: AppliedTables
        :return: pages to apply after it, or :class:`DerivedPages`
        :rtype: Sequence[ChangePage] | DerivedPages
        """
        ...


@runtime_checkable
class WholeHooks(ApplyHooks, Protocol):
    """hooks that also derive from the whole copy at once (:meth:`SharedCopyBackend.settle_whole`): the first settle
    after a start that read L3 whole, every change before it applied with no ``derive``
    (:meth:`SharedCopyBackend.defer_hooks_until_whole`)."""

    def derive_whole(self, applied: AppliedTables) -> DerivedPages:
        """the derived tables from every scope of the copy.

        :param applied: the current version read whole (:meth:`AppliedTables.table`; ``touched`` None, no rows)
        :ptype applied: AppliedTables
        :return: the derived tables' pages, those replaced whole, and what is let go (in the same publish)
        :rtype: DerivedPages
        """
        ...


@dataclass(frozen=True, slots=True)
class ComputedRows:
    """what :class:`ColumnHooks` computes of one table's change.

    :ivar rows: the change's upserted rows (``AppliedTables.rows[table]``) one for one, in their order, with the
        columns computed: every column of the table
    :ivar held: rows the version holds whose computed value the change moves (another row of the same group came, or
        went), with the value computed again: written with the change, as rows it upserts. None for none. A held row
        naming a key the change writes or deletes is refused (the change's own row stands)
    """

    rows: pa.Table
    held: pa.Table | None = None


class ColumnHooks(Protocol):
    """a product's columns computed over a change's rows before the change is written, in the apply worker (a
    module-level object, pickled by name): columns the copy holds and L3 does not, so a row read from L3 lacks them
    (reports' snapshots' ``most_recent_ind``: each row's upload ranked newest first among its group's, over the rows
    the change brings and those the copy holds).

    Run on every change to a table it names (:attr:`computed`): a change's, a reload's as it stages and as it publishes,
    a derived table's, a whole settle's. A rows page may lack the computed columns (they are null until computed).
    """

    @property
    def computed(self) -> Mapping[str, Collection[str]]:
        """each table whose columns are computed, and those columns (never a key column)."""
        ...

    def compute(self, table: str, change: AppliedTables) -> ComputedRows:
        """the columns of one table's change computed.

        ``change`` is the change settled, before it is written: ``rows[table]`` its upserted rows (a key once, its
        last row), ``deleted[table]`` its deleted keys, ``deleted_rows[table]`` the rows those keys name as the version
        holds them (absent for none), ``touched`` the part names of the scope values it writes or deletes (a split
        table's), ``whole`` the table when the change replaces it whole (its held rows go: compute over ``rows``
        alone). ``table(name, scopes=)`` reads the version as it is before the change (``parts`` is empty), mapped:
        the held rows of the groups the change touches, read by its touched scopes.

        :param table: the table
        :ptype table: str
        :param change: the change, and the version before it (:meth:`AppliedTables.table`)
        :ptype change: AppliedTables
        :return: the change's rows computed, and the held rows whose value moved
        :rtype: ComputedRows
        """
        ...


#: a merge policy's view of a scope's held runs (:class:`RunTimes`): each run once, by run, with its newest time
#: (seconds since the epoch) and the model it was made by (null for none)
RUNS_SCHEMA: Final = pa.schema(
    [
        pa.field("run", pa.int64(), nullable=False),
        pa.field("at", pa.float64(), nullable=False),
        pa.field("model", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class RunTimes:
    """where a copy's runs' times and models are held, so a :class:`MergePolicy` sees them: a table of the copy (a
    product's held runs), a row a (scope value, run) and its time, maybe more than one (the newest wins).

    :ivar table: the table
    :ivar scope_column: its column of the scope value (as the per-run tables' scope column holds it)
    :ivar run_column: its column of the run (an integer)
    :ivar time_column: its column of the run's time: a timestamp, or an instant as text in any ISO spelling
        (``arrow_keys.stored_instants``; one with no offset is read as UTC)
    :ivar model_column: its column of the run's model; None for none
    """

    table: str
    scope_column: str
    run_column: str
    time_column: str
    model_column: str | None = None


class MergePolicy(Protocol):
    """which of a scope's per-run parts merge next (a module-level object, pickled by name)."""

    def due(
        self, rule: PartRule, parts: Sequence[tuple[int, int, int, str]], runs: pa.Table | None
    ) -> tuple[int, Sequence[str]] | None:
        """the level merged from and its parts, neighbours of one level in name order; None when none is due.

        :param rule: the table's rule
        :ptype rule: PartRule
        :param parts: one scope's parts: each one's first run, last run, level and name, by first run
        :ptype parts: Sequence[tuple[int, int, int, str]]
        :param runs: the scope's held runs (:data:`RUNS_SCHEMA`: each run's newest time and its model, by run), as the
            version's :class:`RunTimes` table holds them; empty when it holds none of the scope's; None when the copy
            names no such table
        :ptype runs: pa.Table | None
        :return: the level and the group, or None
        :rtype: tuple[int, Sequence[str]] | None
        """
        ...


@dataclass(frozen=True, slots=True)
class ByCount:
    """merge the first ``fan_out`` neighbouring parts of one level into one of the next; the scope's newest run is
    always a part of its own (a change to the newest run rewrites that run alone). Moved from reports
    (``follower_parts._counted_due``).

    :ivar fan_out: the parts a merge takes
    """

    fan_out: int = _FAN_OUT

    def due(
        self, rule: PartRule, parts: Sequence[tuple[int, int, int, str]], runs: pa.Table | None = None
    ) -> tuple[int, Sequence[str]] | None:
        """the first ``fan_out`` neighbours of one level, the newest run aside.

        :param rule: the table's rule (not read)
        :ptype rule: PartRule
        :param parts: one scope's parts, by first run
        :ptype parts: Sequence[tuple[int, int, int, str]]
        :param runs: the scope's held runs (not read: merged by count)
        :ptype runs: pa.Table | None
        :return: the level and the group, or None
        :rtype: tuple[int, Sequence[str]] | None
        """
        group: list[str] = []
        level = -1
        for _first, _last, part_level, part in parts[:-1]:
            if part_level != level:
                group, level = [], part_level
            group.append(part)
            if len(group) == self.fan_out:
                return level, tuple(group)
        return None


@dataclass(frozen=True, slots=True)
class Applied:
    """what one change published.

    :ivar version: the version published
    :ivar parts: each table's parts written (True) or dropped (False); every other part is the base's own file
    :ivar rows: the rows upserted
    :ivar written_bytes: the bytes of the parts written
    :ivar worker: the process the apply ran in
    :ivar files: the files the version holds
    :ivar merge_due: whether a merge is due after it: small parts to pack, or more files than the bound
    :ivar let_go: what the hooks let go of it, published right after it (:class:`DerivedPages`'s ``let_go``); None for
        nothing let go, or a let-go that failed (said at ERROR: the copy keeps those rows until a later settle lets
        them go again)
    :ivar rules_changed: the tables whose rule the copy's declaration changed after this publish (they crossed the
        split floor, :class:`~threetears.core.collections.copy_parts.PartDeclaration`): the next change to each cuts
        it in its new shape
    """

    version: str
    parts: Mapping[str, Mapping[str, bool]]
    rows: int
    written_bytes: int
    worker: int
    files: int = 0
    merge_due: bool = False
    let_go: LetGone | None = None
    rules_changed: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class LetGone:
    """what a change's hooks let go (:class:`DerivedPages`'s ``let_go``), published as a version of its own.

    :ivar version: the version published; None when nothing held matched
    :ivar parts: each table's parts written (True) or dropped (False)
    :ivar rows: each table's rows the copy lost
    """

    version: str | None
    parts: Mapping[str, Mapping[str, bool]]
    rows: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class Merged:
    """one tier merge published.

    :ivar version: the version published
    :ivar table: the table
    :ivar into: the merged part (a tier merge's), or the first part written (a packing's)
    :ivar parts: the parts merged into it, dropped
    :ivar written: every part written: ``into`` alone for a tier merge, the packs and parts a packing wrote
    """

    version: str
    table: str
    into: str
    parts: tuple[str, ...]
    written: tuple[str, ...] = ()


class ReloadStale(LookupError):
    """a per-scope reload staged under a table's rule that is not its rule now (the table crossed the size floor):
    nothing is published, and the reload is read again."""


@dataclass(frozen=True, slots=True)
class ReloadScope:
    """the scope values a per-scope reload replaces of its table: their rows, and nothing else of it.

    :ivar column: the table's declared scope column (its rule's ``scope_column`` when the table is split)
    :ivar values: the values, as text
    """

    column: str
    values: frozenset[str]


@dataclass(frozen=True, slots=True)
class Reloaded:
    """one table a reload staged (:meth:`SharedCopyBackend.stage_reload`), as its publish is told it: the files the
    reload worker wrote, never their rows.

    :ivar table: the table
    :ivar parts: each part staged and its file in the store (a claimed result's, held until the publish)
    :ivar located: its locator's file (:data:`_LOCATED_PREFIX`); None for none
    :ivar scope: the scope values it replaces; None for the whole table
    :ivar rule: the rule it was cut by
    :ivar packed: its packs index's file (:data:`_PACKED_PREFIX`), a whole reload's; None for none
    """

    table: str
    parts: tuple[tuple[str, str], ...]
    located: str | None
    scope: ReloadScope | None
    rule: PartRule
    packed: str | None = None


@dataclass(frozen=True, slots=True)
class ApplyPlan:
    """what :func:`apply_job` is told beside the pages it is handed: names and schemas, never rows.

    :ivar root: the store's directory
    :ivar schemas: each table and its columns (pairs: a mapping proxy does not pickle)
    :ivar rules: each table and its parts
    :ivar pages: each handed page's kind and table, in order: page ``n`` is handed as ``p<n>``
    :ivar whole: the tables the change replaces whole
    :ivar hooks: the product's hooks; None for none
    :ivar merges: the tier merges a per-run table's whole change lands in (:class:`MergePolicy`)
    :ivar instants: each table and its key columns holding an instant as text, kept in one spelling
        (``arrow_keys.stored_instants``)
    :ivar reloads: the tables a reload staged, put in place of the version's before the pages apply over them
        (:meth:`SharedCopyBackend.publish_reload`); none for a change
    :ivar pack_below: a split table's scope value under this many bytes is packed (``copy_parts.PACK_BELOW_BYTES``);
        0 packs none
    :ivar pack_target: a pack is filled to this many bytes (``copy_parts.PACK_TARGET_BYTES``)
    :ivar unpacked: tables cut without packs (a per-scope reload's staging: its parts swap in by scope)
    :ivar canonical: each table and its key columns held as text in one spelling, with their SQL types
        (``arrow_keys.canonical_texts``)
    :ivar run_times: where the runs' times the merges see are held (:class:`RunTimes`); None for none
    :ivar columns: the product's computed columns (:class:`ColumnHooks`); None for none
    :ivar places: each handed page's places (:attr:`ChangePage.places`), in order; empty when none has any
    """

    root: str
    schemas: tuple[tuple[str, pa.Schema], ...]
    rules: tuple[tuple[str, PartRule], ...]
    pages: tuple[tuple[str, str], ...]
    whole: frozenset[str]
    hooks: ApplyHooks | None
    merges: MergePolicy = ByCount()
    instants: tuple[tuple[str, frozenset[str]], ...] = ()
    reloads: tuple[Reloaded, ...] = ()
    pack_below: int = PACK_BELOW_BYTES
    pack_target: int = PACK_TARGET_BYTES
    unpacked: frozenset[str] = frozenset()
    canonical: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    run_times: RunTimes | None = None
    columns: ColumnHooks | None = None
    places: tuple[tuple[tuple[int, int], ...], ...] = ()


@dataclass(frozen=True, slots=True)
class _Outcome:
    """what the apply worker gives back: each part written (its result table's name) or dropped (None), and each part
    a reload staged that the publish links as it is (its file), and whether a packing of its small parts is due."""

    parts: tuple[tuple[str, str, str | None], ...]
    rows: int
    written_bytes: int
    worker: int
    staged: tuple[tuple[str, str, str], ...] = ()
    pack_due: bool = False
    #: each table's rows the hooks let go, held in the result as this name (never published as a part): what the
    #: let-go publish after the change matches (:func:`_let_go_job`)
    let_go: tuple[tuple[str, str], ...] = ()
    #: each table's rows the copy lost to a let-go applied in this job (a whole settle's)
    lost: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _MergePlan:
    root: str
    rules: tuple[tuple[str, PartRule], ...]
    policy: MergePolicy
    schemas: tuple[tuple[str, pa.Schema], ...] = ()
    pack_below: int = PACK_BELOW_BYTES
    pack_target: int = PACK_TARGET_BYTES
    max_files: int = _MAX_VERSION_FILES
    run_times: RunTimes | None = None


@dataclass(frozen=True, slots=True)
class _LetGoPlan:
    """what :func:`_let_go_job` is told: the tables, and each table's let-go rows' file (the change's result's, held
    open until this job is done)."""

    root: str
    schemas: tuple[tuple[str, pa.Schema], ...]
    rules: tuple[tuple[str, PartRule], ...]
    instants: tuple[tuple[str, frozenset[str]], ...]
    canonical: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    let_go: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _MergeOutcome:
    """one merge the merge job wrote: a tier merge's one part, or a packing's packs and parts.

    :ivar table: the table
    :ivar written: each part written and its result's name
    :ivar dropped: the parts it replaces
    :ivar index: the packs index's result name; None for the index as it was
    :ivar index_dropped: the packs index is dropped (nothing of the table is packed now)
    """

    table: str
    written: tuple[tuple[str, str], ...]
    dropped: tuple[str, ...]
    index: str | None = None
    index_dropped: bool = False


@dataclass(frozen=True, slots=True)
class _Tables:
    """a job's tables, read-only, made in the worker from its plan's pairs."""

    schemas: Mapping[str, pa.Schema]
    rules: Mapping[str, PartRule]
    whole: frozenset[str]
    merges: MergePolicy = ByCount()
    instants: Mapping[str, frozenset[str]] = EMPTY
    pack_below: int = PACK_BELOW_BYTES
    pack_target: int = PACK_TARGET_BYTES
    unpacked: frozenset[str] = frozenset()
    canonical: Mapping[str, Mapping[str, str]] = EMPTY
    #: each scope's held runs (:data:`RUNS_SCHEMA`) by its part name, as the merges see them; None until read, or for no
    #: :class:`RunTimes`
    times: Mapping[str, pa.Table] | None = None
    #: where the runs' times are held; None for none
    run_times: RunTimes | None = None
    #: the product's computed columns; None for none
    columns: ColumnHooks | None = None

    def unread(self, table: str) -> Collection[str]:
        """``table``'s columns computed in the worker (:class:`ColumnHooks`), which a rows page may lack."""
        return () if self.columns is None else self.columns.computed.get(table, ())

    def packing(self, table: str) -> bool:
        """whether ``table``'s small scopes are packed as it is cut."""
        return self.rules[table].split and self.pack_below > 0 and table not in self.unpacked

    @property
    def merging(self) -> _Merges:
        """the merges a whole change's runs land in, and the runs' times they see."""
        return _Merges(self.merges, self.times)


@dataclass(frozen=True, slots=True)
class _Merges:
    """a merge policy and the runs' times it sees, by scope part name (None for no :class:`RunTimes`)."""

    policy: MergePolicy
    times: Mapping[str, pa.Table] | None = None

    def due(
        self, rule: PartRule, scope: str, parts: Sequence[tuple[int, int, int, str]]
    ) -> tuple[int, Sequence[str]] | None:
        """the policy's merge due among a scope's parts, with its runs' times."""
        runs = None if self.times is None else self.times.get(scope, RUNS_SCHEMA.empty_table())
        return self.policy.due(rule, parts, runs)


@dataclass(frozen=True, slots=True)
class _Settled:
    """a table's change settled: its rows upserted, its keys deleted, and every key it names (both)."""

    rows: pa.Table
    deleted: pa.Table
    named: pa.Table


def file_name(table: str, part: str) -> str:
    """the name a version gives a part's file (a key of ``shared_arrow.version_files``).

    :param table: the table
    :ptype table: str
    :param part: the part
    :ptype part: str
    :return: ``<table>@<part>.arrow``
    :rtype: str
    """
    return f"{table}{_PART_SEPARATOR}{part}{_TABLE_SUFFIX}"


def held_parts(root: Path | str, version: str, table: str) -> Mapping[str, Path]:
    """the parts of ``table`` a version holds, with their files (hold the version while they are read).

    :param root: the store's directory
    :ptype root: Path | str
    :param version: the version
    :ptype version: str
    :param table: the table
    :ptype table: str
    :return: part -> its file, read-only
    :rtype: Mapping[str, Path]
    """
    return _parts_of(version_files(root, version), table)


def _parts_of(files: Mapping[str, Any], table: str) -> Mapping[str, Path]:
    """the parts of ``table`` among a version's files (``version_files``' answer), the schema file aside."""
    prefix = f"{table}{_PART_SEPARATOR}"
    return frozen(
        (name[len(prefix) : -len(_TABLE_SUFFIX)], held.path)
        for name, held in files.items()
        if name.startswith(prefix) and name.endswith(_TABLE_SUFFIX) and len(name) > len(prefix) + len(_TABLE_SUFFIX)
    )


def _map(path: Path, schema: pa.Schema | None = None) -> pa.Table:
    """a part's file, mapped (its pages the store's, shared, never read into this process's memory); with ``schema``,
    under its metadata (a file's own carries its column tokens, ``shared_arrow.stamped``), so it joins the change's
    rows."""
    with pa.memory_map(str(path), "r") as source:
        rows = ipc.open_file(source).read_all()
    if schema is not None and not rows.schema.equals(schema, check_metadata=True):
        rows = rows.replace_schema_metadata(schema.metadata)
    return rows


# -- the change, settled -------------------------------------------------------------------------------------------


def _respelled(page: pa.Table, instants: Collection[str], canonical: Mapping[str, str] = EMPTY) -> pa.Table:
    """a page with each of its instant key columns in its stored spelling (``arrow_keys.stored_instants``) and each
    canonical key column in its type's one spelling (``arrow_keys.canonical_texts``)."""
    for column in sorted(instants):
        if column in page.column_names:
            page = page.set_column(page.column_names.index(column), column, stored_instants(page[column]))
    for column, cast in sorted(canonical.items()):
        if column in page.column_names:
            page = page.set_column(page.column_names.index(column), column, canonical_texts(page[column], cast))
    return page


def _settled(
    schema: pa.Schema,
    key: Sequence[str],
    pages: Sequence[tuple[str, pa.Table]],
    instants: Collection[str] = frozenset(),
    canonical: Mapping[str, str] = EMPTY,
    unread: Collection[str] = (),
) -> _Settled:
    """a table's pages, in order, as the rows they leave upserted and the keys deleted: each key's last page wins, and
    a key whose last page is a leave page (:data:`LEAVE`) is neither: the change leaves it as held.

    An instant in the key (``instants``) is kept in one spelling, of the rows and of the keys alike, so a key a change
    record names (``key_text``: ``2026-11-03T23:59:00.000000+00:00``) finds the row the rail wrote
    (``2026-11-03T23:59:00Z``).

    :param schema: the table's columns
    :ptype schema: pa.Schema
    :param key: its key
    :ptype key: Sequence[str]
    :param pages: each page's kind and rows, in order
    :ptype pages: Sequence[tuple[str, pa.Table]]
    :param instants: its key columns holding an instant as text
    :ptype instants: Collection[str]
    :param canonical: its key columns held as text in one spelling, and their SQL types
    :ptype canonical: Mapping[str, str]
    :param unread: its columns computed in the worker (:class:`ColumnHooks`): a rows page lacking one holds it null
    :ptype unread: Collection[str]
    :return: the change settled
    :rtype: _Settled
    """
    key_schema = pa.schema([schema.field(column) for column in key])
    rows: list[pa.Table] = []
    events: list[pa.Table] = []
    offset = 0
    for kind, page in pages:
        if not page.num_rows:
            continue
        page = _respelled(page, instants, canonical)
        keys = page.select(list(key)).cast(key_schema)
        if kind == ROWS:
            for column in unread:
                if column not in page.column_names:
                    page = page.append_column(column, pa.nulls(page.num_rows, schema.field(column).type))
            rows.append(page.select(schema.names).cast(schema))
            at = pa.array(np.arange(offset, offset + page.num_rows, dtype=np.int64))
            offset += page.num_rows
        else:
            at = pa.nulls(page.num_rows, pa.int64())
        left = pa.array(np.full(page.num_rows, kind == LEAVE, dtype=np.bool_))
        events.append(keys.append_column(_ROW, at).append_column(_LEFT, left))
    if not events:
        empty = key_schema.empty_table()
        return _Settled(schema.empty_table(), empty, empty)
    happened = pa.concat_tables(events).combine_chunks()
    final = happened.take(pa.array(last_rows(happened, key)))
    final = final.filter(pc.invert(final[_LEFT])).drop_columns([_LEFT])
    upserted = pc.is_valid(final[_ROW])
    written = pa.concat_tables(rows) if rows else schema.empty_table()
    return _Settled(
        rows=written.take(final[_ROW].filter(upserted)),
        deleted=final.filter(pc.invert(upserted)).select(list(key)),
        named=final.select(list(key)),
    )


# -- each row's and each key's part --------------------------------------------------------------------------------


def _tiers(held: Collection[str]) -> Mapping[str, tuple[tuple[int, int, str], ...]]:
    """each scope value's held tiers: first run, last run and name, by first run."""
    by_scope: dict[str, list[tuple[int, int, str]]] = {}
    for part in held:
        tier = tier_of(part)
        if tier is not None:
            by_scope.setdefault(tier[0], []).append((tier[2], tier[3], part))
    return frozen((scope, tuple(sorted(parts))) for scope, parts in by_scope.items())


def _merged_tiers(rule: PartRule, policy: _Merges, scope: str, runs: Sequence[int]) -> Mapping[str, str]:
    """each level-0 part of a scope's runs and the part the tier merges leave it in: ``policy``'s merges, in turn,
    until none is due, as :meth:`SharedCopyBackend.merge` would make them one publish at a time.

    **Why** (reports' ``follower_parts._whole_tiers``). A whole change of a per-run table (a cold start, a reload) cut
    a part a run wrote the full start's 68 runs as 10,880 parts, each a file written, linked and mapped, and 365 merges
    after, one publish each, each holding the copy's next change (reports' full start measured 2026-11-17). Routed here, the change
    lands in the parts the merges would leave, and none is due after it.

    :param rule: the table's rule
    :ptype rule: PartRule
    :param policy: the merges, and the runs' times they see
    :ptype policy: _Merges
    :param scope: the scope value's part name
    :ptype scope: str
    :param runs: its runs, ascending, each once
    :ptype runs: Sequence[int]
    :return: level-0 part -> the part it lands in (absent: its own)
    :rtype: Mapping[str, str]
    :raises ValueError: for a policy naming a group of fewer than two parts (it would never be done)
    """
    parts = [(run, run, 0, tier_part(scope, 0, run, run)) for run in runs]
    into: dict[str, str] = {}
    while (due := policy.due(rule, scope, parts)) is not None:
        level, group = due
        if len(group) < 2:
            raise ValueError(f"{rule.table}: a merge takes two parts at least, not {list(group)}")
        tiers = [tier for tier in (tier_of(name) for name in group) if tier is not None]
        if len(tiers) != len(group):
            raise ValueError(f"{rule.table}: a merge takes tiers alone, not {list(group)}")
        merged = tier_part(scope, level + 1, tiers[0][2], tiers[-1][3])
        gone = set(group)
        for part in [part for part, to in into.items() if to in gone]:
            into[part] = merged
        into.update((part, merged) for part, tier in zip(group, tiers, strict=True) if tier[1] == 0)
        parts = sorted(
            [*(held for held in parts if held[3] not in gone), (tiers[0][2], tiers[-1][3], level + 1, merged)]
        )
    return frozen(into)


def _run_parts(
    scopes: pa.Array, runs: pa.Array, held: Collection[str], merged: tuple[PartRule, _Merges] | None = None
) -> pa.Array:
    """each (scope value, run)'s part: the held tier covering the run, else the run's own level-0 part; null for a
    null scope (a key the copy never held). Named once a distinct pair, never a row.

    :param scopes: each row's scope value as text (null for none)
    :ptype scopes: pa.Array
    :param runs: each row's run
    :ptype runs: pa.Array
    :param held: the table's held parts
    :ptype held: Collection[str]
    :param merged: for a whole change (nothing held), the rule and the merges whose parts its runs land in
        (:func:`_merged_tiers`); None for level 0
    :ptype merged: tuple[PartRule, _Merges] | None
    :return: each row's part
    :rtype: pa.Array
    """
    pairs = pa.table({"scope": pc.fill_null(scopes, _NO_SCOPE), "run": pc.cast(runs, pa.int64())})
    distinct = distinct_keys(pairs, ["scope", "run"])
    tiers = _tiers(held)
    scope_runs = list(zip(distinct["scope"].to_pylist(), distinct["run"].to_pylist(), strict=True))
    routed: Mapping[str, str] = EMPTY
    if merged is not None:
        by_scope: dict[str, list[int]] = {}
        for scope, run in scope_runs:
            if scope != _NO_SCOPE:
                by_scope.setdefault(scope, []).append(run)
        routed = frozen(
            (part, into)
            for scope, scope_of in sorted(by_scope.items())
            for part, into in _merged_tiers(merged[0], merged[1], scope, sorted(set(scope_of))).items()
        )
    names: list[str | None] = []
    for scope, run in scope_runs:
        covering = [name for first, last, name in tiers.get(scope, ()) if first <= run <= last]
        own = tier_part(scope, 0, run, run)
        names.append(None if scope == _NO_SCOPE else covering[0] if covering else routed.get(own, own))
    distinct_codes, row_codes = key_codes([distinct, pairs], ["scope", "run"])
    at = pc.index_in(pa.array(row_codes), value_set=pa.array(distinct_codes))
    named: pa.Array = pa.array(names, pa.string()).take(at)
    return named


#: a packs index's columns (:data:`_PACKED_PREFIX`): each packed scope value's part name and its pack, a locator's shape
_INDEX: Final = Locator().located.schema
_INDEX_SCOPE: Final = _INDEX.names[0]
_INDEX_PACK: Final = _INDEX.names[1]


def _index_of(files: Mapping[str, Any], table: str) -> pa.Table:
    """a split table's packs index as a version holds it; empty for none."""
    held = _parts_of(files, f"{_PACKED_PREFIX}{table}")
    return _map(held[WHOLE_PART]).cast(_INDEX) if WHOLE_PART in held else _INDEX.empty_table()


def _routed(names: pa.Array, scopes: pa.Array, index: pa.Table | None) -> pa.Array:
    """each row's part with a packed scope's rows in its pack (``index``), whatever their run; every other's as
    ``names`` names it. One hash lookup, no Python object a row."""
    if index is None or not index.num_rows:
        return names
    at = pc.index_in(scopes, value_set=index[_INDEX_SCOPE].combine_chunks())
    packs = index[_INDEX_PACK].combine_chunks().take(at)
    routed: pa.Array = pc.if_else(pc.is_valid(packs), packs, names)
    return routed


def _row_parts(
    rule: PartRule,
    rows: pa.Table,
    held: Collection[str],
    merges: _Merges | None = None,
    index: pa.Table | None = None,
) -> pa.Array:
    """each row's part, by its own columns (a split table's rows hold its scope column); with ``merges``, a whole
    change's runs in the tiers those merges leave (:func:`_merged_tiers`); a packed scope's in its pack (``index``)."""
    if not rule.split:
        return pa.repeat(pa.scalar(WHOLE_PART, pa.string()), rows.num_rows)
    scopes = parts_of_keys(rule, rows)
    names = scopes
    if rule.run_column is not None:
        merged = None if merges is None else (rule, merges)
        names = _run_parts(scopes, rows[rule.run_column].combine_chunks(), held, merged)
    return _routed(names, scopes, index)


def _key_parts(
    rule: PartRule, keys: pa.Table, held: Collection[str], locator: Locator, index: pa.Table | None = None
) -> pa.Array:
    """each key's held part: by its scope column, or its locator value's part as the version held it, a packed scope's
    in its pack (``index``); null for none."""
    if not rule.split:
        return pa.repeat(pa.scalar(WHOLE_PART, pa.string()), keys.num_rows)
    scopes = parts_of_keys(rule, keys, locator)
    names = scopes
    if rule.run_column is not None:
        names = _run_parts(scopes, keys[rule.run_column].combine_chunks(), held)
    return _routed(names, scopes, index)


def _cut(
    rule: PartRule,
    rows: pa.Table,
    held: Collection[str],
    merges: _Merges | None = None,
    index: pa.Table | None = None,
) -> Mapping[str, pa.Table]:
    """rows by part, each part's rows a slice in their order in ``rows``: one stable sort of each row's part's name
    and one run-end encoding (a scope's part, or its tier for a per-run table); with ``merges`` (a whole
    change), the tiers those merges leave; with ``index``, a packed scope's rows in its pack."""
    if not rows.num_rows:
        return EMPTY
    names = _row_parts(rule, rows, held, merges, index)
    order = pc.sort_indices(names)
    ordered = rows.take(order)
    runs = pc.run_end_encode(names.take(order))
    ends = runs.run_ends.to_numpy()
    values = runs.values
    cut: list[tuple[str, pa.Table]] = []
    start = 0
    for at, end in enumerate(ends):
        cut.append((str(values[at].as_py()), ordered.slice(start, int(end) - start)))
        start = int(end)
    return frozen(cut)


def _packed_cut(rule: PartRule, rows: pa.Table, tables: _Tables) -> tuple[Mapping[str, pa.Table], pa.Table]:
    """a split table's rows cut afresh with its small scopes packed: each scope value under ``tables.pack_below``
    bytes (all its runs) filled into a pack, in their part names' order, until the pack holds ``tables.pack_target``;
    every other scope cut as a whole change's (by tier, the merges' tiers, for a per-run table). A pack of one scope is
    that scope's own part.

    One stable sort of the scope's part names and one run-end encoding, each scope a slice; Python objects one a scope.

    :param rule: the table's rule, split
    :ptype rule: PartRule
    :param rows: the rows
    :ptype rows: pa.Table
    :param tables: the job's tables (the pack bounds, the merges)
    :ptype tables: _Tables
    :return: the parts by name, and the packs index (each packed scope's part name and its pack)
    :rtype: tuple[Mapping[str, pa.Table], pa.Table]
    """
    if not rows.num_rows:
        return EMPTY, _INDEX.empty_table()
    scopes = parts_of_keys(rule, rows)
    order = pc.sort_indices(scopes)
    ordered = rows.take(order)
    runs = pc.run_end_encode(scopes.take(order))
    ends = runs.run_ends.to_numpy()
    values = runs.values
    own: list[pa.Table] = []
    packs: list[list[tuple[str, pa.Table]]] = []
    filling: list[tuple[str, pa.Table]] = []
    filled = 0
    start = 0
    for at, end in enumerate(ends):
        scope_rows = ordered.slice(start, int(end) - start)
        start = int(end)
        size = scope_rows.nbytes
        if size >= tables.pack_below:
            own.append(scope_rows)
            continue
        filling.append((str(values[at].as_py()), scope_rows))
        filled += size
        if filled >= tables.pack_target:
            packs.append(filling)
            filling, filled = [], 0
    if filling:
        packs.append(filling)
    parts: list[tuple[str, pa.Table]] = []
    packed_scopes: list[str] = []
    packed_into: list[str] = []
    for pack in packs:
        if len(pack) == 1:
            own.append(pack[0][1])
            continue
        name = pack_name(pack[0][0])
        parts.append((name, pa.concat_tables([scope_rows for _, scope_rows in pack])))
        packed_scopes.extend(scope for scope, _ in pack)
        packed_into.extend([name] * len(pack))
    if own:
        parts.extend(_cut(rule, pa.concat_tables(own), (), tables.merging).items())
    index = pa.Table.from_arrays(
        [pa.array(packed_scopes, pa.string()), pa.array(packed_into, pa.string())], schema=_INDEX
    )
    return frozen(sorted(parts)), index


def _misshapen(rule: PartRule, held: Collection[str]) -> bool:
    """whether a table's held parts are not the shape its rule names now (it crossed the size floor, say). A split
    table's packs are its shape at every rule but one part."""
    if not held:
        return False
    if not rule.split:
        return set(held) != {WHOLE_PART}
    own = [part for part in held if not is_pack(part)]
    tiered = [tier_of(part) is not None for part in own]
    return WHOLE_PART in held or (bool(own) and any(tiered) != (rule.run_column is not None)) or len(set(tiered)) > 1


def _packing_due(sizes: Mapping[str, int], below: int, target: int) -> tuple[int, int, tuple[str, ...]]:
    """what packing a split table's small parts would leave: the files they are now, the files they would be, and the
    parts. A small unit is a scope value whose own parts hold under ``below`` bytes, or a pack under ``target``; they
    are filled into packs in name order as :func:`_packed_cut` fills them (a group of one stays as it is).

    :param sizes: each part of the table and its file's bytes
    :ptype sizes: Mapping[str, int]
    :param below: a scope under this is small
    :ptype below: int
    :param target: a pack is filled to this
    :ptype target: int
    :return: the files now, the files after, the parts packed
    :rtype: tuple[int, int, tuple[str, ...]]
    """
    by_scope: dict[str, list[str]] = {}
    units: list[tuple[str, int, tuple[str, ...]]] = []
    for part in sorted(sizes):
        if part == WHOLE_PART:
            continue
        if is_pack(part):
            if sizes[part] < target:
                units.append((part[len(pack_name("")) :], sizes[part], (part,)))
            continue
        by_scope.setdefault(_scope_part(part), []).append(part)
    for scope, scope_parts in by_scope.items():
        size = sum(sizes[part] for part in scope_parts)
        if size < below:
            units.append((scope, size, tuple(scope_parts)))
    units.sort()
    now = sum(len(parts) for _, _, parts in units)
    after = 0
    group: list[tuple[str, ...]] = []
    filled = 0
    for _, size, parts in units:
        group.append(parts)
        filled += size
        if filled >= target:
            after += 1 if len(group) > 1 else len(group[0])
            group, filled = [], 0
    if group:
        after += 1 if len(group) > 1 else len(group[0])
    return now, after, tuple(part for _, _, parts in units for part in parts)


# -- a part's rows (moved from reports' follower_parts) ------------------------------------------------------------


def _ascending(rows: pa.Table, key: Sequence[str]) -> bool:
    """whether rows are in key order already, each key once: a sort leaves them as they are. A pass over the key
    columns; a null in the key is never in order here (the sort places it)."""
    count = rows.num_rows
    before: Any = None
    for column in reversed(key):
        values = rows.column(column)
        this, after = values.slice(0, count - 1), values.slice(1)
        less = pc.less(this, after)
        before = less if before is None else pc.or_(less, pc.and_(pc.equal(this, after), before))
    return bool(pc.all(pc.fill_null(before, False)).as_py())


def _in_key_order(rows: pa.Table, key: Sequence[str]) -> pa.Table:
    """a part's rows in its table's key order, then by every other column, as every write of the part puts them.

    **Why** (reports' replay diagnostics, 2026-11-12 and 13): a rewrite keeps a column's token only position by position
    (``shared_arrow.stamped``). A part rewritten as its kept rows then the change's rows moved every column of it,
    the static ones too, and every query reading them ran again on each vote-only change; in key order a row keeps
    its place. A key naming many rows breaks its tie by the other columns, so the same rows written again keep their
    order.
    """
    if not key or rows.num_rows < 2 or _ascending(rows, key):
        return rows
    rest = [
        field.name
        for field in rows.schema
        if field.name not in key and not pa.types.is_nested(field.type) and not pa.types.is_dictionary(field.type)
    ]
    return rows.sort_by([(column, "ascending") for column in (*key, *rest)])


def _keys_once(ordered: pa.Table, key: Sequence[str]) -> bool:
    """whether rows in key order hold each key once."""
    count = ordered.num_rows
    same: Any = None
    if count > 1:
        for column in key:
            values = ordered.column(column)
            equal = pc.equal(values.slice(1), values.slice(0, count - 1))
            same = equal if same is None else pc.and_(same, equal)
    return same is None or not pc.any(same).as_py()


def _without(held: pa.Table, rows: np.ndarray) -> pa.Table:
    """a part's rows but some, in the part's order."""
    kept = np.ones(held.num_rows, dtype=bool)
    kept[rows] = False
    return held.filter(pa.array(kept))


def _in_place(held: pa.Table, hits: np.ndarray, written: pa.Table, key: Sequence[str]) -> pa.Table | None:
    """a part's rows with each written row in the place of the held row of its key, when the change replaces held
    rows one for one; None otherwise (the caller sorts the part).

    **Why** (reports' apply profile, 2026-11-17, a laptop): sorted again, 28k-row parts cost 8 ms of a 44 ms 382-row change; a column the
    change leaves as held is the held column itself, never copied (copying the part cost ~1.6 ms a 28k-row part).
    """
    found = None
    if len(hits) == written.num_rows:
        ordered = written.take(pc.sort_indices(written, [(column, "ascending") for column in key])).combine_chunks()
        at = held.take(pa.array(hits))
        if _keys_once(ordered, key) and all(at.column(c).equals(ordered.column(c)) for c in key):
            index = np.arange(held.num_rows)
            index[hits] = held.num_rows + np.arange(len(hits))
            taken = pa.array(index)
            found = pa.Table.from_arrays(
                [
                    values
                    if at.column(i).equals(ordered.column(i))
                    else pa.concat_arrays([*values.chunks, *ordered.column(i).chunks]).take(taken)
                    for i, values in enumerate(held.columns)
                ],
                schema=held.schema,
            )
    return found


def _part_rows(
    schema: pa.Schema, held: pa.Table | None, written: pa.Table | None, key: Sequence[str], named: pa.Table | None
) -> pa.Table | None:
    """a part's rows with a change applied: its held rows but those whose keys the change names, and the rows it
    writes, in key order; written in place when the change replaces held rows one for one (:func:`_in_place`).

    :param schema: the table's columns
    :ptype schema: pa.Schema
    :param held: the part's rows as the version holds them; None for a part the change makes new
    :ptype held: pa.Table | None
    :param written: the rows the change writes to it; None for none
    :ptype written: pa.Table | None
    :param key: the table's key
    :ptype key: Sequence[str]
    :param named: the keys the change names; None for none
    :ptype named: pa.Table | None
    :return: the rows; None for a held part the change leaves as it is
    :rtype: pa.Table | None
    """
    hits = None
    if held is not None and named is not None and held.num_rows and named.num_rows:
        found = rows_named(held, named, key)
        hits = found if len(found) else None
    rows: pa.Table | None = None
    if held is not None and hits is not None and written is not None:
        rows = _in_place(held, hits, written, key)
    if rows is None and (hits is not None or written is not None or held is None):
        pieces: list[pa.Table] = []
        if held is not None:
            pieces.append(held if hits is None else _without(held, hits))
        if written is not None:
            pieces.append(written)
        rows = _in_key_order(pa.concat_tables(pieces), key) if pieces else schema.empty_table()
    return rows


# -- the apply, in the worker --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Result:
    """one job's writes into its result slot, named ``p<n>`` in the order written: local to the job, never shared.

    Each part is written stamped with its column tokens against ``base``'s same part (``shared_arrow.stamped``): a
    column the write leaves as the version held it keeps its token, every other column a fresh one.

    **Why** (reports R-1, 2026-11-25): reports' page memo (``build_memo``) runs again only the statements reading a
    column whose token moved. Written unstamped, a part held no tokens and printed as its file, so every vote-only
    change ran every statement reading the part's static columns too.
    """

    step: Any
    parts: list[tuple[str, str, str | None]]
    #: the version's files the job's writes are made from (``version_files``), each part stamped against its own
    base: Mapping[str, Any] = EMPTY
    written_bytes: int = 0
    #: each part written and its file's bytes, the last write's
    sizes: dict[tuple[str, str], int] = field(default_factory=dict)
    #: the rows held that are no part (:meth:`hold`)
    held: int = 0

    def write(self, table: str, part: str, rows: pa.Table) -> None:
        name = f"p{len(self.parts)}"
        # one record batch a file: a reader maps each batch as arrays of its own, and letting a version go frees
        # every array (``_MAX_VERSION_FILES``' why); a pack's scopes, concatenated, were a batch a scope
        held = self.base.get(file_name(table, part))
        wrote = self.step({name: stamped(rows.combine_chunks(), None if held is None else held.path)})
        self.written_bytes += wrote
        self.sizes[(table, part)] = wrote
        self.parts.append((table, part, name))

    def drop(self, table: str, part: str) -> None:
        self.parts.append((table, part, None))

    def hold(self, rows: pa.Table) -> str:
        """write rows into the result that are no part of the version (a let-go's rows, for the job after); their
        name in the result."""
        name = f"g{self.held}"
        self.held += 1
        self.written_bytes += self.step({name: rows.combine_chunks()})
        return name


@dataclass(frozen=True, slots=True)
class _TableApplied:
    """one table's change as the hooks see it: its parts written (as written), the rows it deleted as they were held,
    and the scope part names it touched."""

    parts: Mapping[str, pa.Table]
    deleted_rows: pa.Table
    touched: frozenset[str]


def _touched(rule: PartRule, rows: pa.Table, keys: pa.Table | None, locator: Locator) -> frozenset[str]:
    """the scope part names a split table's change writes (``rows``) or names (``keys``); none for a table held as one
    part. One unique of each, never a Python object a row."""
    if not rule.split:
        return frozenset()
    found: set[str] = set()
    if rows.num_rows:
        found.update(pc.unique(parts_of_keys(rule, rows).drop_null()).to_pylist())
    if keys is not None and keys.num_rows:
        found.update(pc.unique(parts_of_keys(rule, keys, locator).drop_null()).to_pylist())
    return frozenset(found)


def _deleted_of(held: pa.Table, deleted: pa.Table, key: Sequence[str]) -> pa.Table | None:
    """the held rows a change's deleted keys name (one hash match, by row number); None for none."""
    if not deleted.num_rows or not held.num_rows:
        return None
    found = rows_named(held, deleted, key)
    return held.take(pa.array(found)) if len(found) else None


def _apply_table(
    tables: _Tables,
    table: str,
    change: _Settled,
    files: Mapping[str, Any],
    result: _Result,
    locators: Mapping[str, Locator] = EMPTY,
) -> _TableApplied:
    """one table's change, its parts written into the result; its parts as the change leaves them, the rows it deleted
    as they were held and the scopes it touched, for the hooks. ``locators`` names a table's locator in place of its
    file's (a per-scope reload's, over the version's)."""
    rule, schema = tables.rules[table], tables.schemas[table]
    key = rule.key
    held = _parts_of(files, table)
    located = _parts_of(files, f"{_LOCATED_PREFIX}{table}")
    packed = f"{_PACKED_PREFIX}{table}"
    index = _index_of(files, table) if rule.split else _INDEX.empty_table()
    locator = _locator_of(files, table, locators)
    named = change.named if change.named.num_rows else None
    written = change.rows if change.rows.num_rows else None
    after: dict[str, pa.Table] = {}
    gone: list[pa.Table] = []
    touched = _touched(rule, change.rows, named, locator)
    if table in tables.whole or _misshapen(rule, held):
        # the table in its rule's shape now: its rows (the change's alone, for a whole change) cut afresh, every part
        # held before dropped or rewritten
        before = None
        if held and table not in tables.whole:
            before = pa.concat_tables([_map(path, schema) for _, path in sorted(held.items())])
            if (found := _deleted_of(before, change.deleted, key)) is not None:
                gone.append(found)
        rows = change.rows if before is None else (_part_rows(schema, before, written, key, named) or before)
        # a per-run table's runs land in the tiers the merges would leave them in, never a part a run; small scopes
        # land in packs
        if tables.packing(table):
            cut, fresh_index = _packed_cut(rule, rows, tables)
        else:
            cut, fresh_index = _cut(rule, rows, (), tables.merging), _INDEX.empty_table()
        for part, part_rows in cut.items():
            after[part] = _in_key_order(part_rows, key)
            result.write(table, part, after[part])
        for part in sorted(set(held) - set(cut)):
            result.drop(table, part)
        _write_index(result, packed, fresh_index)
        located_rows = rows if rule.locator is not None and rule.split else None
        fresh = Locator() if located_rows is None else Locator().extended(rule, located_rows)
        if fresh.located.num_rows:
            result.write(f"{_LOCATED_PREFIX}{table}", WHOLE_PART, fresh.located)
        elif located:
            result.drop(f"{_LOCATED_PREFIX}{table}", WHOLE_PART)
        return _TableApplied(frozen(after), _concat(schema, gone), touched)
    by_part = _cut(rule, change.rows, held, index=index)
    #: the index pairs of packs this change made new or cut again
    recut: list[pa.Table] = []
    if tables.packing(table):
        by_part, fresh = _new_scopes_packed(rule, by_part, held, tables)
        if fresh.num_rows:
            recut.append(fresh)
    targets = set(by_part)
    if named is not None:
        named_parts = _key_parts(rule, named, held, locator, index)
        targets |= set(pc.unique(named_parts.drop_null()).to_pylist()) & set(held)
    #: the packs this change emptied or cut again
    repacked: list[str] = []
    for part in sorted(targets):
        held_rows = _map(held[part], schema) if part in held else None
        if held_rows is not None and (found := _deleted_of(held_rows, change.deleted, key)) is not None:
            gone.append(found)
        rows = _part_rows(schema, held_rows, by_part.get(part), key, named)
        if rows is None:
            continue
        if not rows.num_rows:
            if part in held:
                result.drop(table, part)
            if is_pack(part):
                repacked.append(part)
            continue
        if is_pack(part) and tables.packing(table) and rows.nbytes >= tables.pack_target + tables.pack_below:
            # a pack outgrown: cut again as it is rewritten anyway, its big scopes to parts of their own
            cut, cut_index = _packed_cut(rule, rows, tables)
            for name, part_rows in cut.items():
                after[name] = _in_key_order(part_rows, key)
                result.write(table, name, after[name])
            if part not in cut:
                result.drop(table, part)
            repacked.append(part)
            recut.append(cut_index)
            continue
        result.write(table, part, rows)
        after[part] = rows
    if repacked or recut:
        kept = index.filter(pc.invert(pc.is_in(index[_INDEX_PACK], value_set=pa.array(repacked, pa.string()))))
        _write_index(result, packed, pa.concat_tables([kept, *recut]))
    if rule.locator is not None and rule.split and written is not None:
        result.write(f"{_LOCATED_PREFIX}{table}", WHOLE_PART, locator.extended(rule, change.rows).located)
    return _TableApplied(frozen(after), _concat(schema, gone), touched)


def _concat(schema: pa.Schema, tables: Sequence[pa.Table]) -> pa.Table:
    """tables of one schema together; empty for none."""
    return pa.concat_tables(tables) if tables else schema.empty_table()


def _new_scopes_packed(
    rule: PartRule, by_part: Mapping[str, pa.Table], held: Collection[str], tables: _Tables
) -> tuple[Mapping[str, pa.Table], pa.Table]:
    """a change's rows by part with the scope values it brings new (held in no pack and no part of their own) cut as
    :func:`_packed_cut` cuts them: its small new scopes packed together in its own publish, never a file each, and no
    held part read for it. A single new scope keeps its own part (a later merge packs it with others).

    :return: the rows by part, and the new packs' index pairs
    :rtype: tuple[Mapping[str, pa.Table], pa.Table]
    """
    scopes = {_scope_part(part) for part in held if not is_pack(part)}
    new = sorted(part for part in by_part if part not in held and not is_pack(part) and _scope_part(part) not in scopes)
    if len({_scope_part(part) for part in new}) < 2:
        return by_part, _INDEX.empty_table()
    cut, index = _packed_cut(rule, pa.concat_tables([by_part[part] for part in new]), tables)
    if not index.num_rows:
        return by_part, index
    kept = [(part, rows) for part, rows in by_part.items() if part not in set(new)]
    return frozen(sorted([*kept, *cut.items()])), index


def _write_index(result: _Result, packed: str, index: pa.Table) -> None:
    """a table's packs index written, or dropped when it packs nothing (a drop of an index never held is left out)."""
    if index.num_rows:
        result.write(packed, WHOLE_PART, index.combine_chunks())
    else:
        result.drop(packed, WHOLE_PART)


def _sizes_after(files: Mapping[str, Any], table: str, result: _Result) -> Mapping[str, int]:
    """each part of ``table`` and its file's bytes as the job leaves it: the version's, then the job's writes and drops."""
    sizes = {part: files[file_name(table, part)].nbytes for part in _parts_of(files, table)}
    for written, part, name in result.parts:
        if written != table:
            continue
        if name is None:
            sizes.pop(part, None)
        else:
            sizes[part] = result.sizes[(table, part)]
    return frozen(sizes.items())


def _apply_pages(
    tables: _Tables,
    files: Mapping[str, Any],
    pages: Sequence[tuple[str, str, pa.Table]],
    result: _Result,
    locators: Mapping[str, Locator] = EMPTY,
    *,
    root: str = "",
    version: str | None = None,
) -> tuple[Mapping[str, _Settled], Mapping[str, _TableApplied]]:
    """each table's pages settled, its computed columns computed (:class:`ColumnHooks`), and applied into the result.
    ``root`` and ``version`` name the store and the version ``files`` are, as the column hooks see them."""
    by_table: dict[str, list[tuple[str, pa.Table]]] = {}
    for kind, table, rows in pages:
        if table not in tables.rules:
            raise ValueError(f"a change to {table!r}, a table the copy does not hold")
        by_table.setdefault(table, []).append((kind, rows))
    for table in tables.whole:
        by_table.setdefault(table, [])
    settled = frozen(
        (
            table,
            _settled(
                tables.schemas[table],
                tables.rules[table].key,
                table_pages,
                tables.instants.get(table, frozenset()),
                tables.canonical.get(table, EMPTY),
                tables.unread(table),
            ),
        )
        for table, table_pages in sorted(by_table.items())
    )
    if tables.columns is not None:
        settled = _computing(tables, tables.columns, files, settled, locators, root, version)
    if tables.run_times is not None and tables.times is None and _routes_whole_runs(tables, files, settled):
        tables = replace(tables, times=_change_run_times(tables, files, settled))
    after = frozen(
        (table, _apply_table(tables, table, change, files, result, locators)) for table, change in settled.items()
    )
    return settled, after


def _locator_of(files: Mapping[str, Any], table: str, locators: Mapping[str, Locator]) -> Locator:
    """a table's locator: ``locators``' (a per-scope reload's), else the version's file's, else none."""
    if table in locators:
        return locators[table]
    located = _parts_of(files, f"{_LOCATED_PREFIX}{table}")
    return Locator(_map(located[WHOLE_PART])) if WHOLE_PART in located else Locator()


def _held_named(tables: _Tables, table: str, keys: pa.Table, files: Mapping[str, Any], locator: Locator) -> pa.Table:
    """the rows the version holds of ``keys``: only the parts the keys' parts name mapped, each matched by one hash
    match (:func:`_deleted_of`); empty for none."""
    rule, schema = tables.rules[table], tables.schemas[table]
    held = _parts_of(files, table)
    if not keys.num_rows or not held:
        return schema.empty_table()
    index = _index_of(files, table) if rule.split else _INDEX.empty_table()
    named = _key_parts(rule, keys, held, locator, index)
    found: list[pa.Table] = []
    for part in sorted(set(pc.unique(named.drop_null()).to_pylist()) & set(held)):
        if (rows := _deleted_of(_map(held[part], schema), keys, rule.key)) is not None:
            found.append(rows)
    return _concat(schema, found)


def _computing(
    tables: _Tables,
    hooks: ColumnHooks,
    files: Mapping[str, Any],
    settled: Mapping[str, _Settled],
    locators: Mapping[str, Locator],
    root: str,
    version: str | None,
) -> Mapping[str, _Settled]:
    """each settled change to a table the hooks compute columns of, with them computed (:class:`ColumnHooks`): the
    change's rows as the hooks give them back, and the held rows whose value moved upserted with them."""
    computed = hooks.computed
    done: list[tuple[str, _Settled]] = []
    for table, change in settled.items():
        if table not in computed or not change.named.num_rows:
            done.append((table, change))
            continue
        rule, schema = tables.rules[table], tables.schemas[table]
        whole = table in tables.whole
        locator = _locator_of(files, table, locators)
        named = change.named if change.named.num_rows else None
        gone = schema.empty_table() if whole else _held_named(tables, table, change.deleted, files, locator)
        view = AppliedTables(
            root=root,
            version=version,
            rows=frozen([(table, change.rows)]),
            deleted=frozen([(table, change.deleted)]),
            parts=EMPTY,
            deleted_rows=frozen([(table, gone)] if gone.num_rows else []),
            whole=frozenset([table]) if whole else frozenset(),
            touched=_touched(rule, change.rows, named, locator),
            files=files,
            rules=tables.rules,
            schemas=tables.schemas,
        )
        done.append((table, _with_computed(table, schema, rule.key, change, hooks.compute(table, view))))
    return frozen(done)


def _with_computed(
    table: str, schema: pa.Schema, key: Sequence[str], change: _Settled, computed: ComputedRows
) -> _Settled:
    """a settled change with its rows as the column hooks computed them, and the held rows they computed again
    upserted beside them.

    :raises ValueError: for rows that are not the change's one for one in its order, or held rows naming a key the
        change writes or deletes
    """
    rows = computed.rows.select(schema.names).cast(schema)
    if rows.num_rows != change.rows.num_rows or not rows.select(list(key)).equals(change.rows.select(list(key))):
        raise ValueError(f"{table}'s computed columns must give back the change's rows one for one, in their order")
    held = computed.held
    if held is None or not held.num_rows:
        return replace(change, rows=rows)
    held = held.select(schema.names).cast(schema)
    held = held.take(pa.array(last_rows(held, key)))
    if len(rows_named(held, change.named, key)):
        raise ValueError(f"{table}'s computed columns give back held rows of keys the change writes or deletes")
    return _Settled(
        rows=pa.concat_tables([rows, held]),
        deleted=change.deleted,
        named=pa.concat_tables([change.named, held.select(list(key))]),
    )


def _routes_whole_runs(tables: _Tables, files: Mapping[str, Any], settled: Mapping[str, _Settled]) -> bool:
    """whether a change cuts a per-run table afresh (whole, or in a new shape), so its runs land in the tiers the
    merges would leave (:func:`_merged_tiers`), which see the runs' times."""
    return any(
        rule.split
        and rule.run_column is not None
        and (table in tables.whole or _misshapen(rule, _parts_of(files, table)))
        for table, rule in ((table, tables.rules[table]) for table in settled)
    )


def _change_run_times(
    tables: _Tables, files: Mapping[str, Any], settled: Mapping[str, _Settled]
) -> Mapping[str, pa.Table]:
    """the runs' times a change's whole cut sees: the change's own rows of the runs table when it replaces that table
    whole (a cold start carries it), else the version's."""
    spec = tables.run_times
    assert spec is not None  # noqa: S101 -- read only with a RunTimes
    if spec.table in settled and spec.table in tables.whole:
        return _runs_by_scope(settled[spec.table].rows, spec)
    return _run_times(files, spec) or EMPTY


# -- the runs' times a merge policy sees ---------------------------------------------------------------------------


def _run_times(files: Mapping[str, Any], spec: RunTimes | None) -> Mapping[str, pa.Table] | None:
    """each scope's held runs as a version's :class:`RunTimes` table holds them, by the scope's part name; None for no
    such table named. Its parts mapped, the columns read in Arrow."""
    if spec is None:
        return None
    held = _parts_of(files, spec.table)
    if not held:
        return EMPTY
    columns = [spec.scope_column, spec.run_column, spec.time_column]
    if spec.model_column is not None:
        columns.append(spec.model_column)
    return _runs_by_scope(pa.concat_tables([_map(path).select(columns) for _, path in sorted(held.items())]), spec)


def _runs_by_scope(rows: pa.Table, spec: RunTimes) -> Mapping[str, pa.Table]:
    """rows of a :class:`RunTimes` table as each scope's runs (:data:`RUNS_SCHEMA`): a run once, its newest time and
    that row's model, by run. One sort and one run-end encoding; a Python object a scope, never a run or a row."""
    if not rows.num_rows:
        return EMPTY
    scoping = PartRule(spec.table, (spec.scope_column,), scope_column=spec.scope_column)
    model = (
        pc.cast(rows[spec.model_column], pa.string())
        if spec.model_column is not None
        else pa.nulls(rows.num_rows, pa.string())
    )
    runs = pa.table(
        {
            "scope": parts_of_keys(scoping, rows),
            "run": pc.cast(rows[spec.run_column], pa.int64()),
            "at": _seconds(rows[spec.time_column]),
            "model": model,
        }
    )
    runs = runs.filter(pc.and_(pc.and_(pc.is_valid(runs["scope"]), pc.is_valid(runs["run"])), pc.is_valid(runs["at"])))
    if not runs.num_rows:
        return EMPTY
    ordered = runs.sort_by([("scope", "ascending"), ("run", "ascending"), ("at", "ascending")])
    newest = ordered.take(pa.array(last_rows(ordered, ["scope", "run"]))).combine_chunks()
    encoded = pc.run_end_encode(newest["scope"].combine_chunks())
    ends = encoded.run_ends.to_numpy()
    values = encoded.values
    found: list[tuple[str, pa.Table]] = []
    start = 0
    for at, end in enumerate(ends):
        scope_runs = newest.slice(start, int(end) - start).select(RUNS_SCHEMA.names).cast(RUNS_SCHEMA)
        found.append((str(values[at].as_py()), scope_runs))
        start = int(end)
    return frozen(found)


#: an instant as :func:`~threetears.core.arrow_keys.stored_instants` spells it: zoned in UTC, or naive (read as UTC)
_STORED_ZONED_RE2: Final = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$"
_STORED_NAIVE_RE2: Final = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}$"


def _seconds(values: pa.ChunkedArray | pa.Array) -> pa.Array:
    """a column of instants as seconds since the epoch (float64): a timestamp (a naive one read as UTC), a number of
    seconds, or text in any ISO spelling (``arrow_keys.stored_instants``; one with no offset read as UTC, one that is
    no instant null). In Arrow's kernels, never a Python object a value."""
    column = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    kind = column.type
    if pa.types.is_timestamp(kind):
        stamps = pc.cast(column, pa.timestamp("us", tz=kind.tz))
    elif pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_decimal(kind):
        numbers: pa.Array = pc.cast(column, pa.float64())
        return numbers
    else:
        text = stored_instants(column)
        naive = pc.fill_null(pc.match_substring_regex(text, _STORED_NAIVE_RE2), False)
        text = pc.if_else(naive, pc.binary_join_element_wise(text, "+00:00", ""), text)
        zoned = pc.fill_null(pc.match_substring_regex(text, _STORED_ZONED_RE2), False)
        text = pc.if_else(zoned, text, pa.nulls(len(text), pa.string()))
        stamps = pc.cast(text, pa.timestamp("us", tz="UTC"))
    micros = pc.cast(pc.cast(stamps, pa.timestamp("us")), pa.int64())
    seconds: pa.Array = pc.divide(pc.cast(micros, pa.float64()), 1e6)
    return seconds


# -- what a product's hooks let go ---------------------------------------------------------------------------------


def _derived(value: Sequence[ChangePage] | DerivedPages) -> DerivedPages:
    """a hook's answer as :class:`DerivedPages`: a plain sequence of pages is its pages alone."""
    return value if isinstance(value, DerivedPages) else DerivedPages(pages=tuple(value))


def _let_go_by_table(tables: _Tables, let_go: Sequence[LetGoRows]) -> tuple[tuple[str, pa.Table], ...]:
    """a hook's let-go rows, each table's together (one set of columns a table), each a table the copy holds and
    columns of it.

    :raises ValueError: for a table the copy does not hold, a column it does not hold, or one table's rows naming
        different columns
    """
    by_table: dict[str, list[pa.Table]] = {}
    for gone in let_go:
        if gone.table not in tables.rules:
            raise ValueError(f"rows let go of {gone.table!r}, a table the copy does not hold")
        stray = sorted(set(gone.rows.column_names) - set(tables.schemas[gone.table].names))
        if stray or not gone.rows.column_names:
            raise ValueError(f"rows let go of {gone.table} are matched on its columns, not {stray or 'none'}")
        if gone.rows.num_rows:
            by_table.setdefault(gone.table, []).append(gone.rows)
    found: list[tuple[str, pa.Table]] = []
    for table, pieces in sorted(by_table.items()):
        names = pieces[0].column_names
        if any(piece.column_names != names for piece in pieces):
            raise ValueError(f"rows let go of {table} name one set of columns")
        schema = pa.schema([tables.schemas[table].field(column) for column in names])
        found.append((table, pa.concat_tables([piece.cast(schema) for piece in pieces]).combine_chunks()))
    return tuple(found)


def _let_go_parts(
    rule: PartRule, gone: pa.Table, held: Mapping[str, Path], files: Mapping[str, Any], table: str
) -> frozenset[str]:
    """the held parts a table's let-go rows may match (:class:`LetGoRows`): by their scope column (and run, a per-run
    table's), else their locator, a packed scope's pack among them; every part when they name no scope. A split
    table's one part (it crossed the floor and is re-cut by its next change, as :meth:`AppliedTables.table` reads it)
    holds every scope's rows: always among them while it is held.

    Why: a replica rebuilt from L3 holds each table its whole read wrote as one part, and its whole settle's let-go of
    a split table's runs matched no part, keeping every run it let go (reports R-2, 2026-11-25)."""
    if not rule.split:
        return frozenset(held) & {WHOLE_PART}
    return _let_go_split_parts(rule, gone, held, files, table) | (frozenset(held) & {WHOLE_PART})


def _let_go_split_parts(
    rule: PartRule, gone: pa.Table, held: Mapping[str, Path], files: Mapping[str, Any], table: str
) -> frozenset[str]:
    """:func:`_let_go_parts` for a split table's cut parts: by scope column (and run), else locator, a packed scope's
    pack among them; every part when the rows name no scope."""
    columns = set(gone.column_names)
    scopes: pa.Array | None = None
    if rule.scope_column in columns:
        scopes = parts_of_keys(rule, gone)
    elif rule.locator is not None and rule.locator in columns:
        located = _parts_of(files, f"{_LOCATED_PREFIX}{table}")
        locator = Locator(_map(located[WHOLE_PART])) if WHOLE_PART in located else Locator()
        scopes = parts_of_keys(rule, gone.select([rule.locator]), locator)
    if scopes is None:
        return frozenset(held)
    index = _index_of(files, table)
    if rule.run_column is not None and rule.run_column in columns:
        names = _routed(_run_parts(scopes, gone[rule.run_column].combine_chunks(), held), scopes, index)
        return frozenset(pc.unique(names.drop_null()).to_pylist()) & frozenset(held)
    named = frozenset(pc.unique(scopes.drop_null()).to_pylist())
    packs = index.filter(pc.is_in(index[_INDEX_SCOPE], value_set=pa.array(sorted(named), pa.string())))
    own = {part for part in held if not is_pack(part) and _scope_part(part) in named}
    return frozenset(own | set(packs[_INDEX_PACK].to_pylist())) & frozenset(held)


def _let_go_table(tables: _Tables, table: str, gone: pa.Table, files: Mapping[str, Any], result: _Result) -> int:
    """one table's let-go rows let go into the result (:class:`LetGoRows`): each part they may match mapped and
    matched by its columns (``arrow_keys.rows_named``, a null equal to a null), rewritten without the matched rows in
    its order, or dropped when every row of it matched (a pack's index then without it). Returns the rows lost."""
    rule, schema = tables.rules[table], tables.schemas[table]
    held = _parts_of(files, table)
    if not held or not gone.num_rows:
        return 0
    gone = _respelled(gone, tables.instants.get(table, frozenset()), tables.canonical.get(table, EMPTY))
    columns = gone.column_names
    gone = gone.cast(pa.schema([schema.field(column) for column in columns]))
    lost = 0
    emptied: list[str] = []
    for part in sorted(_let_go_parts(rule, gone, held, files, table)):
        rows = _map(held[part], schema)
        hits = rows_named(rows, gone, columns)
        if not len(hits):
            continue
        lost += len(hits)
        if len(hits) == rows.num_rows:
            result.drop(table, part)
            if is_pack(part):
                emptied.append(part)
        else:
            result.write(table, part, _without(rows, hits))
    if emptied:
        index = _index_of(files, table)
        kept = index.filter(pc.invert(pc.is_in(index[_INDEX_PACK], value_set=pa.array(emptied, pa.string()))))
        _write_index(result, f"{_PACKED_PREFIX}{table}", kept)
    return lost


def _let_go(
    tables: _Tables, files: Mapping[str, Any], let_go: Sequence[tuple[str, pa.Table]], result: _Result
) -> tuple[tuple[str, int], ...]:
    """each table's let-go rows let go into the result; each table's rows lost."""
    lost = ((table, _let_go_table(tables, table, gone, files, result)) for table, gone in let_go)
    return tuple((table, count) for table, count in lost if count)


def _tables_of(plan: ApplyPlan) -> _Tables:
    """a job's tables, made in the worker from its plan's pairs."""
    return _Tables(
        schemas=frozen(plan.schemas),
        rules=frozen(plan.rules),
        whole=plan.whole,
        merges=plan.merges,
        instants=frozen(plan.instants),
        pack_below=plan.pack_below,
        pack_target=plan.pack_target,
        unpacked=plan.unpacked,
        canonical=frozen((table, frozen(columns)) for table, columns in plan.canonical),
        run_times=plan.run_times,
        columns=plan.columns,
    )


def _pack_due(tables: _Tables, files: Mapping[str, Any], result: _Result) -> bool:
    """whether a job's writes leave a split table's small parts due a packing."""
    touched = sorted({table for table, _, _ in result.parts if table in tables.rules and tables.packing(table)})
    due = False
    for table in touched:
        now, packed, _ = _packing_due(_sizes_after(files, table, result), tables.pack_below, tables.pack_target)
        due = due or packed < now
    return due


def _let_go_job(context: JobContext, plan: _LetGoPlan) -> _Outcome:
    """what a change's hooks let go (:class:`DerivedPages`'s ``let_go``), let go from the change's published version
    (``context.version``) into the job's result: only the parts the rows may match read, those they match rewritten
    or dropped. The rows are mapped from the change's result, where :func:`apply_job` held them.

    :param context: the change's version and the job's result slot
    :ptype context: JobContext
    :param plan: the tables, and each table's let-go rows' file
    :ptype plan: _LetGoPlan
    :return: each part written or dropped, the rows lost by table
    :rtype: _Outcome
    """
    if context.version is None or not isinstance(context.result, (PiecedSlot, SegmentSlot)):
        raise ValueError("a let-go reads a version and writes into a result slot")
    files = version_files(plan.root, context.version.name)
    tables = _Tables(
        schemas=frozen(plan.schemas),
        rules=frozen(plan.rules),
        whole=frozenset(),
        instants=frozen(plan.instants),
        canonical=frozen((table, frozen(columns)) for table, columns in plan.canonical),
    )
    with context.result.writing() as step:
        result = _Result(step, [], files)
        lost = _let_go(tables, files, [(table, _map(Path(path))) for table, path in plan.let_go], result)
        due = _pack_due(tables, files, result)
    parts, _ = _settled_parts(files, (), result.parts)
    return _Outcome(
        parts=parts, rows=0, written_bytes=result.written_bytes, worker=os.getpid(), pack_due=due, lost=lost
    )


def _settle_whole_job(context: JobContext, plan: ApplyPlan) -> _Outcome:
    """the whole copy settled in the apply worker (:meth:`SharedCopyBackend.settle_whole`): the hooks'
    ``derive_whole`` reads every scope of the current version in place (:class:`AppliedTables` with ``touched``
    None), its derived pages are applied into the job's result, and what it lets go into the same result: one
    publish, nothing reading the derived rows before their let-go.

    :param context: the current version and the job's result slot
    :ptype context: JobContext
    :param plan: the tables and the hooks (:class:`WholeHooks`)
    :ptype plan: ApplyPlan
    :return: each part written or dropped, the derived rows, the rows lost by table
    :rtype: _Outcome
    :raises ValueError: for hooks with no ``derive_whole``, or a whole settle deriving rows of a table it lets go of
    """
    hooks = plan.hooks
    if not isinstance(hooks, WholeHooks):
        raise ValueError("a whole settle needs hooks with derive_whole (WholeHooks)")
    if context.version is None or not isinstance(context.result, (PiecedSlot, SegmentSlot)):
        raise ValueError("a whole settle reads a version and writes into a result slot")
    version = context.version.name
    files = version_files(plan.root, version)
    tables = replace(_tables_of(plan), whole=frozenset())
    with context.result.writing() as step:
        result = _Result(step, [], files)
        applied = AppliedTables(
            root=plan.root,
            version=version,
            rows=EMPTY,
            deleted=EMPTY,
            parts=EMPTY,
            touched=None,
            files=files,
            rules=tables.rules,
            schemas=tables.schemas,
        )
        derived = hooks.derive_whole(applied)
        let_go = _let_go_by_table(tables, derived.let_go)
        written = {page.table for page in derived.pages} | set(derived.whole)
        both = sorted(written & {table for table, _ in let_go})
        if both:
            raise ValueError(f"a whole settle derives rows of tables it lets go of: {both}")
        settled, _ = _apply_pages(
            replace(tables, whole=frozenset(derived.whole)),
            files,
            [(page.kind, page.table, page.rows) for page in derived.pages],
            result,
            root=plan.root,
            version=version,
        )
        lost = _let_go(tables, files, let_go, result)
        due = _pack_due(tables, files, result)
    parts, _ = _settled_parts(files, (), result.parts)
    return _Outcome(
        parts=parts,
        rows=sum(change.rows.num_rows for change in settled.values()),
        written_bytes=result.written_bytes,
        worker=os.getpid(),
        pack_due=due,
        lost=lost,
    )


def _as_staged(
    pages: Sequence[tuple[str, str, pa.Table]], places: Sequence[tuple[tuple[int, int], ...]]
) -> list[tuple[str, str, pa.Table]]:
    """the pages as their change staged them: each run of neighbouring pages with places (:attr:`ChangePage.places`)
    sliced back into its places (zero-copy) and put in place order; every other page where it stands.

    :param pages: each handed page's kind, table and rows, in order
    :ptype pages: Sequence[tuple[str, str, pa.Table]]
    :param places: each page's places, in the same order
    :ptype places: Sequence[tuple[tuple[int, int], ...]]
    :return: the pages, in order
    :rtype: list[tuple[str, str, pa.Table]]
    """
    ordered: list[tuple[str, str, pa.Table]] = []
    run: list[tuple[int, str, str, pa.Table]] = []
    for (kind, table, rows), held in zip(pages, places, strict=True):
        if not held:
            ordered.extend((kind, table, sliced) for _place, kind, table, sliced in sorted(run, key=_place_of))
            run = []
            ordered.append((kind, table, rows))
            continue
        offset = 0
        for place, count in held:
            run.append((place, kind, table, rows.slice(offset, count)))
            offset += count
    ordered.extend((kind, table, sliced) for _place, kind, table, sliced in sorted(run, key=_place_of))
    return ordered


def _place_of(placed: tuple[int, str, str, pa.Table]) -> int:
    return placed[0]


def apply_job(context: JobContext, plan: ApplyPlan) -> _Outcome:
    """a change applied in the apply worker (:class:`SharedCopyBackend`): its pages mapped from the hand-off, the
    version it applies to mapped, and each part it rewrites written into the job's result, then the hooks' derived
    tables into the same result. A reload's publish (``plan.reloads``) first puts each reloaded table's staged parts in
    place of the version's (:func:`_reloaded`), and its pages (the changes applied since the reload was given) apply
    over them.

    :param context: the job's version (None for the first change), its pages (``p<n>``) and its result slot
    :ptype context: JobContext
    :param plan: the tables' schemas and rules, the pages' kinds and tables, the tables replaced whole, the hooks
    :ptype plan: ApplyPlan
    :return: each part written or dropped, the rows upserted, the bytes written, this process
    :rtype: _Outcome
    """
    version = None if context.version is None else context.version.name
    base: Mapping[str, Any] = EMPTY if version is None else version_files(plan.root, version)
    mapped = EMPTY if context.tables is None else context.tables.tables
    pages = _as_staged(
        [(kind, table, mapped[f"p{at}"]) for at, (kind, table) in enumerate(plan.pages)],
        plan.places or ((),) * len(plan.pages),
    )
    tables = _tables_of(plan)
    slot = context.result
    if not isinstance(slot, (PiecedSlot, SegmentSlot)):
        raise ValueError("a change's apply writes into a result slot")
    rows = 0
    let_go: tuple[tuple[str, str], ...] = ()
    with slot.writing() as step:
        # stamped against the version itself, a reloaded table's parts too: a column the reload and the pages over it
        # leave as the version held it keeps the token its readers last saw
        result = _Result(step, [], base)
        view = _reloaded(tables, base, plan.reloads, result)
        files = view.files
        settled, after = _apply_pages(
            tables, files, [*view.pages, *pages], result, view.locators, root=plan.root, version=version
        )
        rows += sum(change.rows.num_rows for change in settled.values())
        if plan.hooks is not None:
            applied = _applied_tables(plan, version, tables, files, settled, after, view, result)
            derived = _derived(plan.hooks.derive(applied))
            pages_derived = [(p.kind, p.table, p.rows) for p in derived.pages if p.table not in settled]
            replaced = frozenset(derived.whole) - frozenset(settled)
            if pages_derived or replaced:
                more, _ = _apply_pages(
                    replace(tables, whole=replaced),
                    files,
                    pages_derived,
                    result,
                    view.locators,
                    root=plan.root,
                    version=version,
                )
                rows += sum(change.rows.num_rows for change in more.values())
            # held in the result for the let-go's own publish after this one, never a part of this version
            let_go = tuple((table, result.hold(gone)) for table, gone in _let_go_by_table(tables, derived.let_go))
        due = _pack_due(tables, files, result)
    parts, staged = _settled_parts(base, view.swaps, result.parts)
    return _Outcome(
        parts=parts,
        rows=rows,
        written_bytes=result.written_bytes,
        worker=os.getpid(),
        staged=staged,
        pack_due=due,
        let_go=let_go,
    )


def _applied_tables(
    plan: ApplyPlan,
    version: str | None,
    tables: _Tables,
    files: Mapping[str, Any],
    settled: Mapping[str, _Settled],
    after: Mapping[str, _TableApplied],
    view: _ReloadView,
    result: _Result,
) -> AppliedTables:
    """what a change did as the hooks see it, and the version with it applied (:class:`AppliedTables`)."""
    left = {**dict.fromkeys(view.tables, EMPTY), **{table: done.parts for table, done in after.items()}}
    touched = set().union(*(done.touched for done in after.values())) if after else set()
    for table, part, _ in view.swaps:
        if table in tables.rules and tables.rules[table].split and not is_pack(part):
            touched.add(_scope_part(part))
    whole = tables.whole | frozenset(reload.table for reload in plan.reloads if reload.scope is None)
    return AppliedTables(
        root=plan.root,
        version=version,
        rows=frozen((table, change.rows) for table, change in settled.items()),
        deleted=frozen((table, change.deleted) for table, change in settled.items()),
        parts=frozen((table, _as_left(_parts_of(files, table), parts, result, table)) for table, parts in left.items()),
        reloaded=view.tables,
        deleted_rows=frozen((table, done.deleted_rows) for table, done in after.items() if done.deleted_rows.num_rows),
        whole=whole,
        touched=frozenset(touched),
        files=files,
        rules=tables.rules,
        schemas=tables.schemas,
    )


def _reload_job(
    context: JobContext, plan: ApplyPlan, scopes: tuple[tuple[str, ReloadScope], ...], check: Any
) -> _Outcome:
    """a reload staged in the reload worker (:meth:`SharedCopyBackend.stage_reload`): each reloaded table's rows cut
    into its parts as a whole change's are (:func:`apply_job` with ``plan.whole``, over no version), written into the
    job's result, which is held until its publish. A per-scope table's rows must all be of its scope values.

    :param context: the job's pages (``p<n>``) and its result slot; no version
    :ptype context: JobContext
    :param plan: the reload's tables, every one of them whole, and no hooks
    :ptype plan: ApplyPlan
    :param scopes: each per-scope table and the values it replaces
    :ptype scopes: tuple[tuple[str, ReloadScope], ...]
    :param check: the product's hooks run over the staged tables in this worker (:class:`ApplyHooks`; what they derive
        is the publish's, never staged); None for none
    :ptype check: ApplyHooks | None
    :return: each part written, the rows, the bytes written, this process
    :rtype: _Outcome
    :raises ValueError: for a per-scope table's row outside its scope values
    """
    mapped = EMPTY if context.tables is None else context.tables.tables
    for table, scope in scopes:
        allowed = pa.array(sorted(scope.values), pa.string())
        for at, (_kind, paged) in enumerate(plan.pages):
            if paged != table:
                continue
            values = pc.cast(mapped[f"p{at}"][scope.column], pa.string())
            outside = pc.sum(pc.invert(pc.fill_null(pc.is_in(values, value_set=allowed), False))).as_py()
            if outside:
                raise ValueError(f"a reload of {table}'s {sorted(scope.values)} holds {outside} rows of other values")
    # a per-scope reload's parts swap in by scope, so they are staged unpacked; its publish packs what lands in a pack
    unpacked = plan.unpacked | frozenset(table for table, _ in scopes)
    return apply_job(context, replace(plan, hooks=None if check is None else _Checked(check), unpacked=unpacked))


@dataclass(frozen=True, slots=True)
class _Checked:
    """a product's hooks run over a reload as it stages: what they derive is dropped (derived tables are the
    publish's, over the version it publishes on)."""

    hooks: Any

    def derive(self, applied: AppliedTables) -> Sequence[ChangePage]:
        self.hooks.derive(applied)
        return ()


@dataclass(frozen=True, slots=True)
class _ReloadView:
    """a version's files with a reload's staged tables in place: what a reload's publish applies its pages over.

    :ivar files: the version's files, each reloaded table's replaced (whole, or its scope's parts)
    :ivar swaps: each part put in (its staged file) or taken out (None) of the version
    :ivar pages: pages applied first: a per-scope reload of a table held as one part, as a change (its held rows of
        those values deleted, the reload's rows upserted)
    :ivar locators: each per-scope reloaded table's locator, the reload's over the version's
    :ivar tables: the tables reloaded
    """

    files: Mapping[str, Any]
    swaps: tuple[tuple[str, str, str | None], ...] = ()
    pages: tuple[tuple[str, str, pa.Table], ...] = ()
    locators: Mapping[str, Locator] = EMPTY
    tables: frozenset[str] = frozenset()


def _scope_part(part: str) -> str:
    """the scope value's part name a part is of: a tier's scope, else the part's own name."""
    tier = tier_of(part)
    return part if tier is None else tier[0]


def _reloaded(tables: _Tables, files: Mapping[str, Any], reloads: Sequence[Reloaded], result: _Result) -> _ReloadView:
    """``files`` with each reloaded table's staged parts in place (:class:`_ReloadView`): a whole reload's every held
    part out and its staged parts in; a per-scope reload of a split table its scope values' parts alike; a per-scope
    reload of a table held as one part a change applied first. A per-scope locator is the reload's over the
    version's, written into the result.

    :raises ReloadStale: for a per-scope reload staged under a rule that is not the table's now
    :raises ValueError: for a per-scope reload naming a scope column that is not its split table's
    """
    if not reloads:
        return _ReloadView(files=files)
    view = dict(files)
    swaps: list[tuple[str, str, str | None]] = []
    pages: list[tuple[str, str, pa.Table]] = []
    locators: list[tuple[str, Locator]] = []
    for reload in reloads:
        table, scope = reload.table, reload.scope
        rule, schema = tables.rules[table], tables.schemas[table]
        held = _parts_of(files, table)
        located_name = f"{_LOCATED_PREFIX}{table}"
        packed_name = f"{_PACKED_PREFIX}{table}"
        if scope is not None and rule != reload.rule:
            raise ReloadStale(
                f"{table} is held by another rule than its reload was staged by; the reload is read again"
            )
        if scope is not None and not rule.split:
            if WHOLE_PART in held:
                rows = _map(held[WHOLE_PART], schema)
                values = pc.cast(rows[scope.column], pa.string())
                gone = rows.filter(pc.is_in(values, value_set=pa.array(sorted(scope.values), pa.string())))
                pages.append((KEYS, table, gone.select(list(rule.key))))
            pages.extend((ROWS, table, _map(Path(path), schema)) for _, path in sorted(reload.parts))
            continue
        if scope is not None and scope.column != rule.scope_column:
            raise ValueError(f"{table} is split by {rule.scope_column!r}, not {scope.column!r}")
        names = None if scope is None else {part_name(value) for value in scope.values}
        stray = sorted(part for part, _ in reload.parts if names is not None and _scope_part(part) not in names)
        if stray:
            raise ValueError(f"a reload of {table} staged parts outside its scope: {stray}")
        # a per-scope reload's scopes held in packs: each pack loses their rows by a keys page, and their staged rows
        # land in it as rows pages (the pack is rewritten, the other scopes it holds kept)
        in_packs: frozenset[str] = frozenset()
        if names is not None:
            index = _index_of(files, table)
            asked = pa.array(sorted(names), pa.string())
            hit = index.filter(pc.is_in(index[_INDEX_SCOPE], value_set=asked))
            in_packs = frozenset(hit[_INDEX_SCOPE].to_pylist())
            for pack in sorted(set(hit[_INDEX_PACK].to_pylist()) & set(held)):
                rows = _map(held[pack], schema)
                gone = rows.filter(pc.is_in(parts_of_keys(rule, rows), value_set=asked))
                pages.append((KEYS, table, gone.select(list(rule.key))))
        for part in sorted(held):
            if names is None or (not is_pack(part) and _scope_part(part) in names):
                del view[file_name(table, part)]
                swaps.append((table, part, None))
        for part, path in reload.parts:
            if _scope_part(part) in in_packs:
                pages.append((ROWS, table, _map(Path(path), schema)))
                continue
            view[file_name(table, part)] = VersionFile(Path(path), Path(path).stat().st_size)
            swaps.append((table, part, path))
        if scope is None:
            # a whole reload's packs index is its own
            if WHOLE_PART in _parts_of(files, packed_name):
                del view[file_name(packed_name, WHOLE_PART)]
                swaps.append((packed_name, WHOLE_PART, None))
            if reload.packed is not None:
                path = reload.packed
                view[file_name(packed_name, WHOLE_PART)] = VersionFile(Path(path), Path(path).stat().st_size)
                swaps.append((packed_name, WHOLE_PART, path))
        held_located = _parts_of(files, located_name)
        if scope is None:
            if WHOLE_PART in held_located:
                del view[file_name(located_name, WHOLE_PART)]
                swaps.append((located_name, WHOLE_PART, None))
            if reload.located is not None:
                path = reload.located
                view[file_name(located_name, WHOLE_PART)] = VersionFile(Path(path), Path(path).stat().st_size)
                swaps.append((located_name, WHOLE_PART, path))
        elif reload.located is not None:
            now = Locator(_map(held_located[WHOLE_PART])) if WHOLE_PART in held_located else Locator()
            merged = now.overlaid(Locator(_map(Path(reload.located))))
            result.write(located_name, WHOLE_PART, merged.located)
            locators.append((table, merged))
    return _ReloadView(
        files=frozen(view.items()),
        swaps=tuple(swaps),
        pages=tuple(pages),
        locators=frozen(locators),
        tables=frozenset(reload.table for reload in reloads),
    )


def _settled_parts(
    base: Mapping[str, Any],
    swaps: Sequence[tuple[str, str, str | None]],
    written: Sequence[tuple[str, str, str | None]],
) -> tuple[tuple[tuple[str, str, str | None], ...], tuple[tuple[str, str, str], ...]]:
    """each part's last word: a reload's swap, then what the job wrote or dropped over it. A part dropped that the
    version never held is left out (a staged part the changes over it emptied).

    :return: the parts written (their result names) or dropped, and the staged parts linked as they are (their files)
    """
    last: dict[tuple[str, str], tuple[bool, str | None]] = {}
    for table, part, path in swaps:
        last[(table, part)] = (True, path)
    for table, part, name in written:
        last[(table, part)] = (False, name)
    parts: list[tuple[str, str, str | None]] = []
    staged: list[tuple[str, str, str]] = []
    for (table, part), (swapped, ref) in last.items():
        if ref is None:
            if file_name(table, part) in base:
                parts.append((table, part, None))
        elif swapped:
            staged.append((table, part, ref))
        else:
            parts.append((table, part, ref))
    return tuple(parts), tuple(staged)


def _as_left(
    held: Mapping[str, Path], written: Mapping[str, pa.Table], result: _Result, table: str
) -> Mapping[str, pa.Table]:
    """a table's parts as a change leaves them: the held ones it did not touch, mapped, and those it wrote."""
    dropped = {part for name, part, out in result.parts if name == table and out is None}
    kept = ((part, _map(path)) for part, path in sorted(held.items()) if part not in written and part not in dropped)
    return frozen([*kept, *sorted(written.items())])


def _level(part: str) -> int:
    """a tier's level (a part :func:`_tiers` found a tier)."""
    tier = tier_of(part)
    return -1 if tier is None else tier[1]


def _merge_job(context: JobContext, plan: _MergePlan) -> _MergeOutcome | None:
    """one merge due, written into the job's result: a tier merge, the lowest level first (the group's rows copied
    once from their mapped files, ``shared_arrow.merged_rows``, as ``m``), else the packing of one table's small parts
    (:func:`_pack_job`). Nothing else is read.

    :param context: the job's version and result slot
    :ptype context: JobContext
    :param plan: the tables' rules, the merge policy and the pack bounds
    :ptype plan: _MergePlan
    :return: what it wrote; None when none is due
    :rtype: _MergeOutcome | None
    """
    if context.version is None or not isinstance(context.result, (PiecedSlot, SegmentSlot)):
        raise ValueError("a merge reads a version and writes into a result slot")
    version = context.version.name
    files = version_files(plan.root, version)
    best: tuple[int, str, tuple[str, ...]] | None = None
    per_run = any(rule.run_column is not None and rule.split for _, rule in plan.rules)
    merges = _Merges(plan.policy, _run_times(files, plan.run_times) if per_run else None)
    for table, rule in sorted(plan.rules):
        if rule.run_column is None or not rule.split:
            continue
        for scope, tiers in sorted(_tiers(_parts_of(files, table)).items()):
            parts = [(first, last, _level(name), name) for first, last, name in tiers]
            due = merges.due(rule, scope, parts)
            if due is not None and (best is None or due[0] < best[0]):
                best = (due[0], table, tuple(due[1]))
    if best is None:
        return _pack_job(context, plan, files)
    level, table, group = best
    ends = [tier for tier in (tier_of(group[0]), tier_of(group[-1])) if tier is not None]
    into = tier_part(ends[0][0], level + 1, ends[0][2], ends[-1][3])
    # stamped fresh as it streams (its rows are never held to compare), so the next change to it keeps tokens
    context.result.write({"m": stamped(merged_rows(plan.root, version, table, group)(), None)})
    return _MergeOutcome(table=table, written=((into, "m"),), dropped=tuple(group))


def _pack_job(context: JobContext, plan: _MergePlan, files: Mapping[str, Any]) -> _MergeOutcome | None:
    """the packing of the split table whose small parts it saves the most files of (:func:`_packing_due`): their rows
    mapped, cut afresh with :func:`_packed_cut`, written as ``m<n>``, and the table's packs index rewritten as ``i``.
    A version over ``plan.max_files`` packs every scope under ``plan.pack_target``, not only those under
    ``plan.pack_below``. Only the small parts are read and written, never the table.

    :param context: the job's version and result slot
    :ptype context: JobContext
    :param plan: the tables, the merge policy and the pack bounds
    :ptype plan: _MergePlan
    :param files: the version's files
    :ptype files: Mapping[str, Any]
    :return: what it wrote; None when no packing saves a file
    :rtype: _MergeOutcome | None
    """
    if plan.pack_below <= 0 or not isinstance(context.result, (PiecedSlot, SegmentSlot)):
        return None
    below = plan.pack_target if len(files) > plan.max_files else plan.pack_below
    rules = frozen(plan.rules)
    best: tuple[int, str, tuple[str, ...]] | None = None
    for table, rule in sorted(rules.items()):
        if not rule.split:
            continue
        sizes = frozen((part, files[file_name(table, part)].nbytes) for part in _parts_of(files, table))
        now, after, small = _packing_due(sizes, below, plan.pack_target)
        if after < now and (best is None or now - after > best[0]):
            best = (now - after, table, small)
    if best is None:
        return None
    _, table, small = best
    rule, schema = rules[table], frozen(plan.schemas)[table]
    held = _parts_of(files, table)
    rows = pa.concat_tables([_map(held[part], schema) for part in small])
    tables = _Tables(
        schemas=frozen(plan.schemas),
        rules=rules,
        whole=frozenset(),
        merges=plan.policy,
        pack_below=below,
        pack_target=plan.pack_target,
        times=_run_times(files, plan.run_times) if rule.run_column is not None else None,
        run_times=plan.run_times,
    )
    cut, cut_index = _packed_cut(rule, rows, tables)
    if len(cut) >= len(small):
        return None
    index = _index_of(files, table)
    dropped_packs = pa.array([part for part in small if is_pack(part)], pa.string())
    kept = index.filter(pc.invert(pc.is_in(index[_INDEX_PACK], value_set=dropped_packs)))
    fresh = pa.concat_tables([kept, cut_index]).combine_chunks()
    written: list[tuple[str, str]] = []
    with context.result.writing() as step:
        for at, (part, part_rows) in enumerate(cut.items()):
            held_file = files.get(file_name(table, part))
            step(
                {
                    f"m{at}": stamped(
                        _in_key_order(part_rows, rule.key).combine_chunks(),
                        None if held_file is None else held_file.path,
                    )
                }
            )
            written.append((part, f"m{at}"))
        if fresh.num_rows:
            step({"i": fresh})
    held_index = WHOLE_PART in _parts_of(files, f"{_PACKED_PREFIX}{table}")
    return _MergeOutcome(
        table=table,
        written=tuple(written),
        dropped=tuple(part for part in small if part not in cut),
        index="i" if fresh.num_rows else None,
        index_dropped=not fresh.num_rows and held_index,
    )


# -- the serving process's side ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StagedReload:
    """a reload staged by the reload worker (:meth:`SharedCopyBackend.reload_run`): its tables' parts written into the
    store, held there until :meth:`SharedCopyBackend.publish_reload` links them or :meth:`close` lets them go.

    :ivar reloads: each table staged, its files
    :ivar rows: the rows staged
    :ivar written_bytes: the bytes written
    :ivar worker: the process that staged it
    :ivar segment: the staged files, held; None for a reload of no rows
    """

    reloads: tuple[Reloaded, ...]
    rows: int
    written_bytes: int
    worker: int
    segment: MappedSegment | None

    async def close(self) -> None:
        """let the staged files go (once published, the version holds them). Harmless twice.

        :return: nothing
        :rtype: None
        """
        if self.segment is not None:
            await asyncio.to_thread(self.segment.close)


async def _staged(
    running: Coroutine[Any, Any, JobResult], rules: Mapping[str, PartRule], scopes: Mapping[str, ReloadScope]
) -> StagedReload:
    """a reload's staging awaited, and its result held as a :class:`StagedReload`."""
    done = await running
    outcome: _Outcome = done.value
    files = EMPTY if done.segment is None else done.segment.files
    staged: dict[str, list[tuple[str, str]]] = {}
    for table, part, name in outcome.parts:
        if name is not None:
            staged.setdefault(table, []).append((part, str(files[name])))
    reloads = tuple(
        Reloaded(
            table=table,
            parts=tuple(sorted(staged.get(table, ()))),
            located=dict(staged.get(f"{_LOCATED_PREFIX}{table}", ())).get(WHOLE_PART),
            scope=scopes.get(table),
            rule=rule,
            packed=dict(staged.get(f"{_PACKED_PREFIX}{table}", ())).get(WHOLE_PART),
        )
        for table, rule in sorted(rules.items())
    )
    return StagedReload(
        reloads=reloads,
        rows=outcome.rows,
        written_bytes=outcome.written_bytes,
        worker=outcome.worker,
        segment=done.segment,
    )


class SharedCopyBackend:
    """a copy's tables in a shared Arrow store, each change applied in the copy's apply worker and published as a new
    version that shares every part the change did not touch.

    Make it as the pod starts, or on a thread: its pool starts the fork server its worker is forked by. One change or
    merge at a time; each publishes one version (``c-<n>``).

    :param store: the store the versions are published in
    :ptype store: SharedArrowStore
    :param schemas: each table's columns
    :ptype schemas: Mapping[str, pa.Schema]
    :param rules: each table's parts (``copy_parts.part_rules``)
    :ptype rules: Mapping[str, PartRule]
    :param hooks: a product's behaviour on each change (:class:`ApplyHooks`); None for none
    :ptype hooks: ApplyHooks | None
    :param merges: which tiers merge (:class:`MergePolicy`)
    :ptype merges: MergePolicy
    :param private_bytes: the apply worker is replaced after a change leaving it more private memory than this
    :ptype private_bytes: int
    :param preload: modules the worker holds from its start (the hooks' among them)
    :ptype preload: Sequence[str]
    :param instants: each table's key columns holding an instant as text (a ``timestamptz`` the copy keeps as its
        text): kept in one spelling, of rows and keys alike (``arrow_keys.stored_instants``), so a key named in any
        ISO spelling finds its row
    :ptype instants: Mapping[str, Collection[str]]
    :param canonical: each table's key columns held as text whose type has one spelling, and that type
        (``arrow_keys.CANONICAL_CASTS``: a ``NUMERIC``, a UUID): kept in it, of rows and keys alike, as each change is
        applied (``copy_reads.canonical_keys`` names them from the followed tables), so a lookup spells only its keys
    :ptype canonical: Mapping[str, Mapping[str, str]]
    :param pack_below: a split table's scope value under this many bytes is packed (``copy_parts.PACK_BELOW_BYTES``);
        0 packs none
    :ptype pack_below: int
    :param pack_target: a pack is filled to this many bytes (``copy_parts.PACK_TARGET_BYTES``)
    :ptype pack_target: int
    :param max_files: a version holding more files than this makes a merge due (:data:`_MAX_VERSION_FILES`)
    :ptype max_files: int
    :param run_times: where the runs' times and models are held (:class:`RunTimes`), so ``merges`` sees them; None
        for none (``MergePolicy.due`` is given None)
    :ptype run_times: RunTimes | None
    :param declared: the tables' declared scopes (``copy_parts.PartDeclaration``): after each change's publish the copy
        sizes its tables and holds each by the rule its size names now (a table crossing ``SPLIT_FLOOR_BYTES`` is cut
        by its scope from the next change to it). None keeps ``rules`` until :meth:`use_rules`
    :ptype declared: PartDeclaration | None
    :param columns: columns computed over each change's rows in the apply worker before it is written
        (:class:`ColumnHooks`); None for none
    :ptype columns: ColumnHooks | None
    :raises ValueError: for a rule of a table with no schema, a schema missing a rule's column, an instant or canonical
        column that is not a text key column of its table, a canonical type with no one spelling, or a computed column
        that is not a non-key column of a table the copy holds
    """

    def __init__(
        self,
        store: SharedArrowStore,
        schemas: Mapping[str, pa.Schema],
        rules: Mapping[str, PartRule],
        *,
        hooks: ApplyHooks | None = None,
        merges: MergePolicy | None = None,
        private_bytes: int = _APPLY_PRIVATE_BYTES,
        preload: Sequence[str] = (),
        instants: Mapping[str, Collection[str]] = EMPTY,
        canonical: Mapping[str, Mapping[str, str]] = EMPTY,
        pack_below: int = PACK_BELOW_BYTES,
        pack_target: int = PACK_TARGET_BYTES,
        max_files: int = _MAX_VERSION_FILES,
        run_times: RunTimes | None = None,
        declared: PartDeclaration | None = None,
        columns: ColumnHooks | None = None,
    ) -> None:
        if pack_below < 0 or pack_target <= 0 or max_files <= 0:
            raise ValueError(f"pack bounds and a file bound are positive: {pack_below}, {pack_target}, {max_files}")
        self._schemas = frozen(schemas)
        self._rules = _checked(self._schemas, rules)
        self._instants = _instant_keys(self._schemas, self._rules, instants)
        self._canonical = _canonical_keys(self._schemas, self._rules, canonical)
        self._pack_below = pack_below
        self._pack_target = pack_target
        self._max_files = max_files
        #: whether the last version published was over ``max_files`` (said once a crossing)
        self._over = False
        self._store = store
        self._hooks = hooks
        self._merges = ByCount() if merges is None else merges
        if run_times is not None:
            spec = self._rules.get(run_times.table)
            named = [run_times.scope_column, run_times.run_column, run_times.time_column, run_times.model_column]
            if spec is None or not {c for c in named if c is not None} <= set(self._schemas[run_times.table].names):
                raise ValueError(
                    f"the runs' times are held in {run_times.table!r}: a table the copy holds, with its columns"
                )
        self._run_times = run_times
        self._declared = declared
        if columns is not None:
            _check_computed(self._schemas, self._rules, columns)
        self._columns = columns
        #: whether a whole settle is due (:meth:`defer_hooks_until_whole`): until it publishes, changes apply with no
        #: ``derive``
        self._whole_due = False
        #: each table's rows the hooks let go since the copy started (:attr:`let_go`): replaced whole on the loop
        self._let_go: Mapping[str, int] = EMPTY
        self._pool = SharedArrowPool(
            store,
            workers=1,
            private_bytes=private_bytes,
            name="copy apply",
            preload=(__name__, *preload),
            process_name=_APPLY_PROCESS_NAME,
        )
        self._applying = asyncio.Lock()
        current = store.current
        found = None if current is None else _VERSION.match(current.name)
        self._next = 0 if found is None else int(found["n"]) + 1

    @property
    def let_go(self) -> Mapping[str, int]:
        """each table's rows the hooks let go since the copy started (``shared_copy_let_go_rows_total``), read-only.

        :return: table -> rows; a table appears once any of its rows was let go
        :rtype: Mapping[str, int]
        """
        return self._let_go

    def _count_let_go(self, gone: LetGone | None) -> None:
        """add what a publish let go to :attr:`let_go` and :data:`_LET_GO_ROWS`. On the event loop: a count a table."""
        if gone is None or not gone.rows:
            return
        for table, rows in gone.rows.items():
            if rows:
                _LET_GO_ROWS.labels(table=table).inc(int(rows))
        self._let_go = frozen(
            (table, self._let_go.get(table, 0) + int(gone.rows.get(table, 0)))
            for table in sorted({*self._let_go, *gone.rows})
        )

    @property
    def rules(self) -> Mapping[str, PartRule]:
        """each table's parts, read-only.

        :return: table -> its rule
        :rtype: Mapping[str, PartRule]
        """
        return self._rules

    @property
    def derives_whole(self) -> bool:
        """whether the hooks derive from the whole copy at once (:class:`WholeHooks`): a whole settle may be deferred
        to (:meth:`defer_hooks_until_whole`).

        :return: True for :class:`WholeHooks`
        :rtype: bool
        """
        return isinstance(self._hooks, WholeHooks)

    @property
    def schemas(self) -> Mapping[str, pa.Schema]:
        """each table's columns, read-only.

        :return: table -> its schema
        :rtype: Mapping[str, pa.Schema]
        """
        return self._schemas

    def use_rules(self, rules: Mapping[str, PartRule]) -> None:
        """hold the tables by these rules from the next change on: a table whose held parts are of another shape is
        rewritten whole in its new one as the change touching it lands.

        :param rules: each table's parts
        :ptype rules: Mapping[str, PartRule]
        :return: nothing
        :rtype: None
        :raises ValueError: as the constructor's
        """
        self._rules = _checked(self._schemas, rules)

    @property
    def store(self) -> SharedArrowStore:
        """the store the versions are published in.

        :return: the store
        :rtype: SharedArrowStore
        """
        return self._store

    def _unknown(self, pages: Sequence[ChangePage], whole: Collection[str]) -> None:
        unknown = sorted(
            {page.table for page in pages if page.table not in self._rules} | (set(whole) - set(self._rules))
        )
        if unknown:
            raise ValueError(f"a change to tables the copy does not hold: {unknown}")

    def _plan(
        self, pages: Sequence[ChangePage], whole: Collection[str], reloads: tuple[Reloaded, ...] = ()
    ) -> ApplyPlan:
        return ApplyPlan(
            root=str(self._store.root),
            schemas=tuple(self._schemas.items()),
            rules=tuple(self._rules.items()),
            pages=tuple((page.kind, page.table) for page in pages),
            places=tuple(page.places for page in pages) if any(page.places for page in pages) else (),
            whole=frozenset(whole),
            # a whole settle due derives every scope at once: a change's own derive before it is work thrown away
            hooks=None if self._whole_due else self._hooks,
            merges=self._merges,
            instants=self._instants,
            reloads=reloads,
            pack_below=self._pack_below,
            pack_target=self._pack_target,
            canonical=self._canonical,
            run_times=self._run_times,
            columns=self._columns,
        )

    async def apply(
        self, pages: Sequence[ChangePage], *, whole: Collection[str] = (), mark: int | None = None
    ) -> Applied:
        """apply a change in the apply worker and publish it: each part it lands in rewritten, every other part the
        version before's own file.

        :param pages: the change's pages, in the order staged
        :ptype pages: Sequence[ChangePage]
        :param whole: the tables the change replaces whole
        :ptype whole: Collection[str]
        :param mark: the change's mark, published beside it (:data:`MARK_TABLE`); None for none
        :ptype mark: int | None
        :return: what it published
        :rtype: Applied
        :raises Exception: what the apply raised; ``BrokenProcessPool`` when its worker died. Nothing is published
        """
        with self._measured():
            self._unknown(pages, whole)
            return await self._applied(pages, whole, mark, ())

    async def publish_reload(
        self,
        staged: StagedReload,
        pages: Sequence[ChangePage] = (),
        *,
        whole: Collection[str] = (),
        mark: int | None = None,
    ) -> Applied:
        """publish a reload the reload worker staged (:meth:`stage_reload`) on the version current now, in the apply
        worker: each reloaded table's staged parts in place of the version's (whole, or its scope values' parts),
        ``pages`` (the changes to those tables applied since the reload was given) applied over them, then the
        hooks, all in one version. A table in ``whole`` (a change since the reload replaced it whole) keeps that
        change's rows, not the reload's. The staged parts are linked, never copied; ``staged`` stays its owner's to
        close.

        :param staged: the reload
        :ptype staged: StagedReload
        :param pages: the changes applied over it, in order
        :ptype pages: Sequence[ChangePage]
        :param whole: the tables those changes replaced whole
        :ptype whole: Collection[str]
        :param mark: the mark published beside it; None keeps the version's
        :ptype mark: int | None
        :return: what it published (its rows: the reload's and the changes')
        :rtype: Applied
        :raises ReloadStale: for a per-scope reload staged under a rule that is not its table's now. Nothing is
            published on any failure
        """
        with self._measured():
            self._unknown(pages, whole)
            reloads = tuple(reload for reload in staged.reloads if reload.table not in set(whole))
            applied = await self._applied(pages, whole, mark, reloads)
        return replace(applied, rows=applied.rows + staged.rows)

    @contextmanager
    def _measured(self) -> Iterator[None]:
        """count a change's failure by why (:data:`_APPLY_FAILURES`) and set the store's room after it, failed or not
        (:data:`_STORE_ROOM`). On the event loop: the room is the store's own tallies, no file read."""
        try:
            yield
        except Exception as exc:
            _APPLY_FAILURES.labels(why="died" if isinstance(exc, BrokenProcessPool) else "failed").inc()
            raise
        finally:
            _STORE_ROOM.set(self._store.max_bytes - int(self._store.status()["held_bytes"]))

    async def _applied(
        self, pages: Sequence[ChangePage], whole: Collection[str], mark: int | None, reloads: tuple[Reloaded, ...]
    ) -> Applied:
        sent = [page for page in pages if page.rows.num_rows]
        let_gone: LetGone | None = None
        async with self._applying:
            current = self._store.current
            base = None if current is None else current.name
            done = await self._pool.run(
                apply_job,
                self._plan(sent, whole, reloads),
                read=False if base is None else base,
                tables={f"p{at}": page.rows for at, page in enumerate(sent)} or None,
                result_pieces=apply_piece_bytes(self._store.max_bytes, sent, self._held_largest(base, sent)),
                result_minimum=APPLY_PIECE_MINIMUM_BYTES,
                result_grow_to=APPLY_PIECE_BYTES,
            )
            outcome: _Outcome = done.value
            version = f"c-{self._next}"
            try:
                files, sizes = await asyncio.to_thread(self._publish_change, version, base, outcome, done.segment, mark)
                self._next += 1
                over = self._said_over(version, files)
                if outcome.let_go and done.segment is not None:
                    # the let-go's rows are files of this change's result: held open until its publish is done
                    let_gone, sizes, let_go_due = await self._let_go_after(version, outcome.let_go, done.segment, sizes)
                    over = over or let_go_due
            finally:
                if done.segment is not None:
                    await asyncio.to_thread(done.segment.close)
            changed = self._regrown(sizes)
        parts: dict[str, dict[str, bool]] = {}
        for table, part, _path in outcome.staged:
            parts.setdefault(table, {})[part] = True
        for table, part, name in outcome.parts:
            parts.setdefault(table, {})[part] = name is not None
        self._count_let_go(let_gone)
        return Applied(
            version=version,
            parts=frozen((table, frozen(done_parts)) for table, done_parts in parts.items()),
            rows=outcome.rows,
            written_bytes=outcome.written_bytes,
            worker=outcome.worker,
            files=files,
            merge_due=outcome.pack_due or over,
            let_go=let_gone,
            rules_changed=changed,
        )

    async def _let_go_after(
        self, version: str, let_go: tuple[tuple[str, str], ...], segment: MappedSegment, sizes: Mapping[str, int]
    ) -> tuple[LetGone | None, Mapping[str, int], bool]:
        """what a change's hooks let go, let go from its version in the apply worker and published right after it,
        under the same apply turn (the caller holds ``_applying``). A failure publishes nothing more and is said at
        ERROR: the change is published, and the copy keeps the rows until a later settle lets them go again.

        :return: what was let go (None on a failure), the tables' sizes as the copy holds them now, and whether a merge
            is due after it (small parts to pack, or more files than the bound)
        """
        files = segment.files
        plan = _LetGoPlan(
            root=str(self._store.root),
            schemas=tuple(self._schemas.items()),
            rules=tuple(self._rules.items()),
            instants=self._instants,
            canonical=self._canonical,
            let_go=tuple((table, str(files[name])) for table, name in let_go),
        )
        try:
            done = await self._pool.run(
                _let_go_job,
                plan,
                read=version,
                # as a merge: a piece of the least, each rewritten file growing its own, never a GiB held for the run
                result_pieces=APPLY_PIECE_MINIMUM_BYTES,
                result_minimum=APPLY_PIECE_MINIMUM_BYTES,
                result_grow_to=APPLY_PIECE_BYTES,
            )
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the change is published; what it let go waits for a later settle
            contain(
                exc,
                log=log,
                what="letting go what a change's settle let go failed; the copy keeps those rows until a later one",
                extra={"version": version, "tables": sorted(table for table, _ in let_go)},
                level="error",
            )
            return None, sizes, False
        outcome: _Outcome = done.value
        published: str | None = None
        over = False
        try:
            if outcome.parts:
                published = f"c-{self._next}"
                count, sizes = await asyncio.to_thread(
                    self._publish_change, published, version, outcome, done.segment, None
                )
                self._next += 1
                over = self._said_over(published, count)
        finally:
            if done.segment is not None:
                await asyncio.to_thread(done.segment.close)
        parts: dict[str, dict[str, bool]] = {}
        for table, part, name in outcome.parts:
            parts.setdefault(table, {})[part] = name is not None
        return (
            LetGone(
                version=published,
                parts=frozen((table, frozen(done_parts)) for table, done_parts in parts.items()),
                rows=frozen(outcome.lost),
            ),
            sizes,
            outcome.pack_due or over,
        )

    def _regrown(self, sizes: Mapping[str, int]) -> frozenset[str]:
        """hold the tables by the rules their sizes name now (``declared``, :class:`~threetears.core.collections.
        copy_parts.PartDeclaration`); the tables whose rule changed. On the event loop: a rule a table, no row read."""
        if self._declared is None or not sizes:
            return frozenset()
        now = self._declared.rules(sizes, held=self._rules)
        changed = frozenset(table for table, rule in now.items() if table in self._rules and self._rules[table] != rule)
        if changed:
            self._rules = _checked(self._schemas, {**self._rules, **{table: now[table] for table in changed}})
            log.info(
                "copied tables crossed the split floor: each is cut in its new shape by the next change to it",
                extra={"extra_data": {"tables": {table: now[table].split for table in sorted(changed)}}},
            )
        return changed

    @property
    def whole_settle_due(self) -> bool:
        """whether a whole settle is due (:meth:`defer_hooks_until_whole`), changes applying with no ``derive`` until
        it publishes.

        :return: True until :meth:`settle_whole` publishes
        :rtype: bool
        """
        return self._whole_due

    def defer_hooks_until_whole(self) -> None:
        """a whole settle follows (a start that read L3 whole, no checkpoint): until :meth:`settle_whole` publishes,
        each change applies with no ``derive``. The whole settle derives every scope from the copy as it then is, so a
        change's own derive before it is work thrown away (reports' first change after such a start is every row of
        the copy, and its settle was a whole settle of its own: 6-7 s at prod's size, measured 2026-11-17).

        :return: nothing
        :rtype: None
        :raises ValueError: for hooks with no ``derive_whole`` (:class:`WholeHooks`)
        """
        if not isinstance(self._hooks, WholeHooks):
            raise ValueError("a whole settle needs hooks with derive_whole (WholeHooks)")
        self._whole_due = True

    async def settle_whole(self) -> Applied | None:
        """settle the whole copy: in the apply worker, over every scope of the current version read in place, the
        hooks' ``derive_whole`` computes the derived tables, published with what it lets go as one version, under the
        apply turn, so no change lands between. Every change after it is derived as it applies.

        :return: what it published; None when the copy holds no version yet
        :rtype: Applied | None
        :raises ValueError: for hooks with no ``derive_whole``, or a settle deriving rows of a table it lets go of
        """
        with self._measured():
            return await self._settled_whole()

    async def _settled_whole(self) -> Applied | None:
        hooks = self._hooks
        if not isinstance(hooks, WholeHooks):
            raise ValueError("a whole settle needs hooks with derive_whole (WholeHooks)")
        async with self._applying:
            current = self._store.current
            if current is None:
                return None
            done = await self._pool.run(
                _settle_whole_job,
                replace(self._plan((), ()), hooks=hooks),
                read=current.name,
                # which derived tables it writes is the hooks' to find: a piece of the least, each file growing its
                # own (:data:`~threetears.core.collections.copy_pieces._APPLY_STORE_SHARE`'s why), never a GiB held for the settle's whole run
                result_pieces=APPLY_PIECE_MINIMUM_BYTES,
                result_minimum=APPLY_PIECE_MINIMUM_BYTES,
                result_grow_to=APPLY_PIECE_BYTES,
            )
            outcome: _Outcome = done.value
            version = f"c-{self._next}"
            try:
                files, sizes = await asyncio.to_thread(
                    self._publish_change, version, current.name, outcome, done.segment, None
                )
            finally:
                if done.segment is not None:
                    await asyncio.to_thread(done.segment.close)
            self._next += 1
            self._whole_due = False
            over = self._said_over(version, files)
            changed = self._regrown(sizes)
        parts: dict[str, dict[str, bool]] = {}
        for table, part, name in outcome.parts:
            parts.setdefault(table, {})[part] = name is not None
        let_gone = LetGone(version=version, parts=EMPTY, rows=frozen(outcome.lost)) if outcome.lost else None
        self._count_let_go(let_gone)
        return Applied(
            version=version,
            parts=frozen((table, frozen(done_parts)) for table, done_parts in parts.items()),
            rows=outcome.rows,
            written_bytes=outcome.written_bytes,
            worker=outcome.worker,
            files=files,
            merge_due=outcome.pack_due or over,
            let_go=let_gone,
            rules_changed=changed,
        )

    def _held_largest(self, base: str | None, pages: Sequence[ChangePage]) -> Mapping[str, int]:
        """each table ``pages`` change and the largest file ``base`` holds of it (a part, its locator, its packs
        index), from the store's manifest: no file read. Empty for no base."""
        if base is None:
            return EMPTY
        return frozen(
            (
                table,
                self._store.largest_file_bytes(base, (table, f"{_LOCATED_PREFIX}{table}", f"{_PACKED_PREFIX}{table}")),
            )
            for table in sorted({page.table for page in pages})
        )

    def _said_over(self, version: str, files: int) -> bool:
        """whether a version holds more files than the bound, said at WARNING as the copy crosses it."""
        over = files > self._max_files
        if over and not self._over:
            log.warning(
                "the copy's version holds more files than its bound: a merge packs its small parts",
                extra={"extra_data": {"version": version, "files": files, "max_files": self._max_files}},
            )
        self._over = over
        return over

    def reload_run(
        self,
        pool: SharedArrowPool,
        pages: Sequence[ChangePage],
        *,
        tables: Collection[str],
        scopes: Mapping[str, ReloadScope] = EMPTY,
        check: ApplyHooks | None = None,
    ) -> Coroutine[Any, Any, StagedReload]:
        """a reload's staging in ``pool``'s worker (the reload lane's, never the apply worker): ``tables``' rows cut
        into parts by their rules now, written into the store and held there until :meth:`publish_reload`.

        Returns the staging to await, holding the pages only until they are handed over: the caller lets its own go
        before awaiting it, so the serving process holds none of a reload's rows while it stages.

        :param pool: the reload lane's pool, over this backend's store
        :ptype pool: SharedArrowPool
        :param pages: the reload's rows pages
        :ptype pages: Sequence[ChangePage]
        :param tables: the tables it reloads (a table with no page is reloaded empty)
        :ptype tables: Collection[str]
        :param scopes: each table reloaded by scope and its values; every other is reloaded whole
        :ptype scopes: Mapping[str, ReloadScope]
        :param check: hooks run in the reload worker over the staged tables (what they derive is dropped)
        :ptype check: ApplyHooks | None
        :return: the staging
        :rtype: Coroutine[Any, Any, StagedReload]
        :raises ValueError: for a table the copy does not hold, a keys page, a page of a table not reloaded, or a
            scope of a table not reloaded or with no scope column
        """
        named = frozenset(tables)
        self._unknown(pages, named)
        wrong = sorted({page.table for page in pages if page.kind != ROWS or page.table not in named})
        if wrong:
            raise ValueError(f"a reload is rows pages of the tables it reloads, not pages of {wrong}")
        for table, scope in scopes.items():
            if table not in named:
                raise ValueError(f"a scope for {table!r}, a table the reload does not reload")
            if scope.column not in self._schemas[table].names:
                raise ValueError(f"{table} has no column {scope.column!r} to reload by")
        sent = [page for page in pages if page.rows.num_rows]
        plan = replace(self._plan(sent, named), hooks=None)
        rules = frozen((table, self._rules[table]) for table in sorted(named))
        running = pool.run(
            _reload_job,
            plan,
            tuple(sorted(scopes.items())),
            check,
            read=False,
            tables={f"p{at}": page.rows for at, page in enumerate(sent)} or None,
            result_pieces=reload_piece_bytes(self._store.max_bytes, sent),
            result_minimum=APPLY_PIECE_MINIMUM_BYTES,
            result_grow_to=APPLY_PIECE_BYTES,
        )
        return _staged(running, rules, frozen(scopes))

    def _publish_change(
        self, version: str, base: str | None, outcome: _Outcome, segment: MappedSegment | None, mark: int | None
    ) -> tuple[int, Mapping[str, int]]:
        """publish a change's result on its base (a thread's: IO alone, each written part linked, never copied; a
        reload's staged parts linked from where it staged them); the files the version holds, and each table's bytes
        (its parts' files') when the copy follows its declaration (``declared``), else none."""
        files = EMPTY if segment is None else segment.files
        by_table: dict[str, dict[str, Path | None]] = {}
        for table, part, path in outcome.staged:
            by_table.setdefault(table, {})[part] = Path(path)
        for table, part, name in outcome.parts:
            by_table.setdefault(table, {})[part] = None if name is None else files[name]
        if base is None:
            for table in self._schemas:
                by_table.setdefault(table, {})
        tables: dict[str, Any] = {
            table: Parts(schema=self._schema_of(table), parts=parts) for table, parts in by_table.items()
        }
        if mark is not None:
            tables[MARK_TABLE] = pa.table({"mark": pa.array([int(mark)], pa.int64())})
        self._store.publish(version, tables, base=base)
        held = version_files(self._store.root, version)
        return len(held), (EMPTY if self._declared is None else _table_bytes(held, self._schemas))

    def _schema_of(self, table: str) -> pa.Schema:
        if table.startswith(_LOCATED_PREFIX):
            return Locator().located.schema
        if table.startswith(_PACKED_PREFIX):
            return _INDEX
        return self._schemas[table]

    async def merge(self) -> Merged | None:
        """one merge due, written in the apply worker and published as a version of its own, never between a change
        and the version showing it: a tier merge (:class:`MergePolicy`), the lowest level first, else the packing of
        one split table's small parts (a scope that came new and small, a pack a change emptied or a version over
        the file bound: ``copy_parts.pack_name``).

        :return: what it merged; None when none is due
        :rtype: Merged | None
        """
        if not any(rule.split for rule in self._rules.values()):
            return None
        async with self._applying:
            current = self._store.current
            if current is None:
                return None
            done = await self._pool.run(
                _merge_job,
                _MergePlan(
                    root=str(self._store.root),
                    rules=tuple(self._rules.items()),
                    policy=self._merges,
                    schemas=tuple(self._schemas.items()),
                    pack_below=self._pack_below,
                    pack_target=self._pack_target,
                    max_files=self._max_files,
                    run_times=self._run_times,
                ),
                read=current.name,
                # which table merges is the worker's to find: a piece of the least, each merged file growing its own
                # to its rows (a pack's target, a tier's parts), never a GiB held for the merge's whole run
                result_pieces=APPLY_PIECE_MINIMUM_BYTES,
                result_minimum=APPLY_PIECE_MINIMUM_BYTES,
                result_grow_to=APPLY_PIECE_BYTES,
            )
            if done.value is None:
                if done.segment is not None:
                    await asyncio.to_thread(done.segment.close)
                return None
            outcome: _MergeOutcome = done.value
            table = outcome.table
            version = f"c-{self._next}"
            segment = done.segment
            if segment is None:
                raise RuntimeError(f"a merge of {table!r} wrote nothing")
            try:
                parts: dict[str, Path | None] = {
                    **dict.fromkeys(outcome.dropped),
                    **{part: segment.files[name] for part, name in outcome.written},
                }
                tables: dict[str, Parts] = {table: Parts(self._schemas[table], parts)}
                if outcome.index is not None or outcome.index_dropped:
                    index = None if outcome.index is None else segment.files[outcome.index]
                    tables[f"{_PACKED_PREFIX}{table}"] = Parts(_INDEX, {WHOLE_PART: index})
                await asyncio.to_thread(self._store.publish, version, tables, base=current.name)
                files = await asyncio.to_thread(lambda: len(version_files(self._store.root, version)))
            finally:
                await asyncio.to_thread(segment.close)
            self._next += 1
            self._said_over(version, files)
        written = tuple(part for part, _ in outcome.written)
        log.debug("copy merge", extra={"extra_data": {"table": table, "written": written, "parts": outcome.dropped}})
        return Merged(version=version, table=table, into=written[0], parts=outcome.dropped, written=written)

    async def close(self) -> None:
        """stop the apply worker.

        :return: nothing
        :rtype: None
        """
        await self._pool.close()


def _table_bytes(files: Mapping[str, VersionFile], schemas: Collection[str]) -> Mapping[str, int]:
    """each table's bytes in a version: its parts' files' (``shared_arrow.version_files``' ``nbytes``), its schema
    file and the copy's own tables (a locator, a packs index, the mark) aside; 0 for a table holding none."""
    sizes = dict.fromkeys(schemas, 0)
    for name, held in files.items():
        table, separator, rest = name.partition(_PART_SEPARATOR)
        if separator and table in sizes and rest != _TABLE_SUFFIX:
            sizes[table] += held.nbytes
    return frozen(sizes.items())


def _instant_keys(
    schemas: Mapping[str, pa.Schema], rules: Mapping[str, PartRule], instants: Mapping[str, Collection[str]]
) -> tuple[tuple[str, frozenset[str]], ...]:
    """each table's instant key columns, as the plan carries them, each a text key column of a table the copy holds."""
    found: list[tuple[str, frozenset[str]]] = []
    for table, columns in sorted(instants.items()):
        if table not in rules:
            raise ValueError(f"instant columns for {table!r}, a table the copy does not hold")
        wrong = sorted(
            column
            for column in columns
            if column not in rules[table].key
            or not (
                pa.types.is_string(schemas[table].field(column).type)
                or pa.types.is_large_string(schemas[table].field(column).type)
            )
        )
        if wrong:
            raise ValueError(f"{table}: an instant kept in one spelling is a text key column, not {wrong}")
        if columns:
            found.append((table, frozenset(columns)))
    return tuple(found)


def _canonical_keys(
    schemas: Mapping[str, pa.Schema], rules: Mapping[str, PartRule], canonical: Mapping[str, Mapping[str, str]]
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """each table's canonical key columns and their types, as the plan carries them: each a text key column of a table
    the copy holds, of a type with one spelling (``arrow_keys.CANONICAL_CASTS``)."""
    found: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for table, columns in sorted(canonical.items()):
        if table not in rules:
            raise ValueError(f"canonical key columns for {table!r}, a table the copy does not hold")
        wrong = sorted(
            column
            for column, cast in columns.items()
            if column not in rules[table].key
            or cast not in CANONICAL_CASTS
            or not (
                pa.types.is_string(schemas[table].field(column).type)
                or pa.types.is_large_string(schemas[table].field(column).type)
            )
        )
        if wrong:
            raise ValueError(
                f"{table}: a key kept in one spelling is a text key column of type {sorted(CANONICAL_CASTS)}, not "
                f"{wrong}"
            )
        if columns:
            found.append((table, tuple(sorted(columns.items()))))
    return tuple(found)


def _check_computed(schemas: Mapping[str, pa.Schema], rules: Mapping[str, PartRule], columns: ColumnHooks) -> None:
    """refuse computed columns that are not non-key columns of a table the copy holds.

    :raises ValueError: for one
    """
    wrong = sorted(
        f"{table}.{column}"
        for table, names in columns.computed.items()
        for column in names
        if table not in rules or column not in schemas[table].names or column in rules[table].key
    )
    if wrong:
        raise ValueError(f"computed columns must be non-key columns of tables the copy holds: {wrong}")


def _checked(schemas: Mapping[str, pa.Schema], rules: Mapping[str, PartRule]) -> Mapping[str, PartRule]:
    """the rules, read-only, each of a table with a schema holding its columns."""
    for table, rule in rules.items():
        if table not in schemas:
            raise ValueError(f"a rule for {table!r}, a table with no schema")
        named = [*rule.key, *(c for c in (rule.scope_column, rule.locator, rule.run_column) if c is not None)]
        missing = sorted(set(named) - set(schemas[table].names))
        if missing:
            raise ValueError(f"{table}: its rule names columns its schema does not hold: {missing}")
    unruled = sorted(set(schemas) - set(rules))
    if unruled:
        raise ValueError(f"tables with no rule: {unruled}")
    return frozen(rules)
