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
it as ``__located__<table>`` (:data:`LOCATED_PREFIX`), so the worker reads it with the parts and the serving process
holds no copy of it.

**Few, large files a version** (decision 2b). A split table's scope values under ``copy_parts.PACK_BELOW_BYTES``
share packs (``copy_parts.pack_name``) filled to ``copy_parts.PACK_TARGET_BYTES``, named in the version beside it as
``__packed__<table>`` (:data:`PACKED_PREFIX`: each packed scope value's part name and its pack). A row of a packed
scope lands in its pack, whatever its run; every other row as above. A whole change cuts the table afresh, packed; a
change rewrites the packs and parts it lands in alone, a pack grown past ``PACK_TARGET_BYTES + PACK_BELOW_BYTES``
re-cut as it is rewritten; and the small parts a change leaves (a new small scope, a pack emptied) are packed by
:meth:`SharedCopyBackend.merge`, a publish of its own, like a tier merge. A version holding more files than its bound
(:data:`MAX_VERSION_FILES`) makes a merge due, which then packs every scope under ``PACK_TARGET_BYTES``.

A whole change of a per-run table (a cold start, a reload) lands in the tiers its :class:`MergePolicy`'s merges would
leave, never a part a run (:func:`_merged_tiers`). A part's name is its scope value's (``copy_parts.part_name``):
escaped, or hashed, for a value the store would refuse as a name.

**A change** is pages in the order they were staged (:class:`ChangePage`): a rows page upserts its rows, a keys page
deletes its keys, and a key's last page wins (a key named in a keys page after its rows page is gone, a rows page after
it is back; a page naming a key twice keeps its last row): the semantics of the staging tables this was moved from
(reports' ``follower_parts.Stages``), settled in Arrow (:func:`~threetears.core.arrow_keys.last_rows`). A table named
in ``whole`` is replaced by the change's rows: every part it held is dropped or rewritten.

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

**Product behaviour stays behind a seam.** :class:`ApplyHooks` (``derive``) runs in the worker after the change is
applied, sees the applied tables (:class:`AppliedTables`), and returns changes to tables of its own (a product's
derived tables: reports' settle) applied into the same result, so they land in the change's publish.
:class:`MergePolicy` picks a per-run table's tier merge (:class:`ByCount` by default; reports merges its LSF runs by
their retention's ages). Both are module-level objects, pickled to the worker by name.

**Moved from reports** (``aibots_reports.pod.follower_parts`` and ``shared_backend``): the part rewrite (``_part_rows``,
``_in_place``, ``_in_key_order``), tiers and their merge by count, ``held_parts``, ``file_name``, the change's publish
on its base with its mark (:data:`MARK_TABLE`). Not moved: reports' settle, retention (``LetGo``, let-go drops), held
runs and its DuckDB staging (the follower's page-by-page staging API is wired to this apply in a later step).
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Collection, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Protocol

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
from threetears.core.collections.copy_parts import (
    PACK_BELOW_BYTES,
    PACK_TARGET_BYTES,
    WHOLE_PART,
    Locator,
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
    version_files,
)
from threetears.observe import get_logger

__all__ = [
    "INDEX_PREFIXES",
    "KEYS",
    "LOCATED_PREFIX",
    "MARK_TABLE",
    "MAX_VERSION_FILES",
    "PACKED_PREFIX",
    "ROWS",
    "Applied",
    "AppliedTables",
    "ApplyPlan",
    "ApplyHooks",
    "ByCount",
    "ChangePage",
    "MergePolicy",
    "Merged",
    "ReloadScope",
    "ReloadStale",
    "Reloaded",
    "SharedCopyBackend",
    "StagedReload",
    "apply_job",
    "file_name",
    "held_parts",
    "apply_piece_bytes",
    "reload_piece_bytes",
]

log = get_logger(__name__)

#: a page of rows to upsert (:class:`ChangePage`)
ROWS: Final = "rows"
#: a page of keys to delete (:class:`ChangePage`)
KEYS: Final = "keys"

#: between a table and its part in a version's file name, and the files' suffix: ``{table}@{part}.arrow``
#: (``shared_arrow.version_files``' contract)
_PART_SEPARATOR: Final = "@"
_TABLE_SUFFIX: Final = ".arrow"

#: a split table's locator is published beside it as this prefix and its name, one part (:data:`WHOLE_PART`)
LOCATED_PREFIX: Final = "__located__"

#: a split table's packs index is published beside it as this prefix and its name, one part (:data:`WHOLE_PART`): each
#: packed scope value's part name and its pack (``copy_parts.pack_name``)
PACKED_PREFIX: Final = "__packed__"

#: the prefixes of the tables a version holds beside the copy's own (a locator, a packs index): no query sees them
INDEX_PREFIXES: Final = (LOCATED_PREFIX, PACKED_PREFIX)

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
MAX_VERSION_FILES: Final = 512

#: the one-row table each change's mark is published as (``mark``, int64): a reader of a version reads the mark of
#: the rows it maps, never a newer one. Moved from reports (``follower_parts.MARK_TABLE``)
MARK_TABLE: Final = "__staged_mark__"

#: how many neighbouring parts of one level :class:`ByCount` merges into one of the next. Moved from reports
#: (``follower_parts.FAN_OUT``): a scope of N runs holds about ``FAN_OUT * log10(N)`` parts and each row is rewritten
#: once a level
_FAN_OUT: Final = 10

#: the most one file of a change's, a merge's or a reload's result may take: the most a piece grows to for a file of
#: its own (``SharedArrowStore.reserve_pieces``' ``grow_to``), and the most one is reserved at. Moved from reports
#: (``shared_backend.SHARED_COPY_RESULT_BYTES``): a fixed 1 GiB slot refused every change of a store with less room.
#: A piece is reserved at what its job is expected to write as one file (:func:`apply_piece_bytes`,
#: :func:`reload_piece_bytes`), never at this: it grows here only for a file that needs it
_APPLY_PIECE_BYTES: Final = 1 << 30

#: the least room a change's piece is reserved at: one part's bound, a piece holding each file whole. Moved from
#: reports (``shared_backend.SHARED_COPY_PIECE_MINIMUM_BYTES``): the largest file a change wrote in its 2026-11-03
#: replay's cold start was a few MiB (a 123 MB table in 53 parts)
_APPLY_PIECE_MINIMUM_BYTES: Final = 16 << 20

#: a reload's result piece is reserved at its largest table's bytes and this share more (an IPC file's padding and
#: metadata, its locator and packs index), at :data:`_APPLY_PIECE_MINIMUM_BYTES` at least
_RELOAD_PIECE_SLACK_DIVISOR: Final = 8

#: a change's result piece is reserved at the largest file it may rewrite and this share more (the IPC file's
#: padding and metadata, a scope's rows cut again), at :data:`_APPLY_PIECE_MINIMUM_BYTES` more at least
#: (:func:`apply_piece_bytes`)
_APPLY_PIECE_SLACK_DIVISOR: Final = 8

#: a change's result piece is reserved at most at this share of the store's ``max_bytes`` (a file that needs more
#: grows its piece, up to :data:`_APPLY_PIECE_BYTES`, while the store has the room).
#:
#: **Why** (``test_copy_lane``'s second-death test, refused "no room for a hand-off in a 1 GiB store", 2026-11-25).
#: An apply reserved a piece of :data:`_APPLY_PIECE_BYTES` (1 GiB), or all the room a smaller store had, for as long
#: as it applied, whatever the change: a change is a few MiB on a busy night (``copy_reload._MAX_REPLAY_BYTES``) and
#: rewrites only the parts it lands in (the largest file the 2026-11-03 replay's cold start wrote was a few MiB, its
#: 123 MB table in 53 parts), so the GiB held was almost all room nobody wrote, and a reload staging or a hand-off
#: beside it found none. Now a piece is reserved at the largest file of the change's tables the version holds (a part,
#: its locator, its packs index) plus the change's own rows of that table and an eighth: the most one rewritten file
#: can hold, unless a hook derives a table or a whole rewrite cuts parts afresh, and then the file grows its piece. It
#: is never past a quarter of the store, beside a reload's quarter (:data:`_RELOAD_STORE_SHARE`): half the store
#: (4 GiB of reports' 8 GiB, ``values.yaml``' ``maxBytes``) stays for versions and hand-offs
_APPLY_STORE_SHARE: Final = 4

#: a reload's result piece takes at most this share of the store's ``max_bytes``.
#:
#: **Why** (inventory ``apply_staged`` / ``reserve_pieces``, 2026-11-25). A reload stages in its own worker for as long
#: as its rows take, holding its open piece's room all the while; at :data:`_APPLY_PIECE_BYTES` (1 GiB) every reload
#: took a GiB, or all the room a smaller store had, and a change's apply beside it found less than its 16 MiB minimum
#: and was refused. A reload's largest file is one part of one table, never more than that table's rows, so its piece
#: is sized to its largest table (most reloads are a state's: a few MiB; the largest table the 2026-11-03 replay's
#: cold start wrote was 123 MB, in 53 parts), and never past a quarter of the store, leaving the applies three
#: quarters (2 GiB of reports' 8 GiB store, ``values.yaml``' ``maxBytes``)
_RELOAD_STORE_SHARE: Final = 4

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

#: a scope value no key's locator names (a key the copy never held): its part is none
_NO_SCOPE: Final = "\x00none"


@dataclass(frozen=True, slots=True)
class ChangePage:
    """one page of a change, in the order staged.

    :ivar kind: :data:`ROWS` (upsert its rows) or :data:`KEYS` (delete its keys)
    :ivar table: the table
    :ivar rows: its rows (every column of the table) or its keys (the key's columns, as text or typed)
    """

    kind: str
    table: str
    rows: pa.Table

    def __post_init__(self) -> None:
        if self.kind not in (ROWS, KEYS):
            raise ValueError(f"a change's page is {ROWS!r} or {KEYS!r}, not {self.kind!r}")


@dataclass(frozen=True, slots=True)
class AppliedTables:
    """what a change did, as :class:`ApplyHooks` sees it in the worker.

    :ivar root: the store's directory
    :ivar version: the version the change applied to; None for the first
    :ivar rows: each changed table's rows the change upserted (a key once, its last row)
    :ivar deleted: each changed table's keys the change deleted
    :ivar parts: each changed table's parts as the change leaves them, by name (held parts mapped, rewritten parts as
        written); a part the change dropped is absent
    :ivar reloaded: the tables a reload's publish replaced (:meth:`SharedCopyBackend.publish_reload`), whole or by
        scope: their parts are here, their reloaded rows are not in ``rows`` (only the changes applied over them)
    """

    root: str
    version: str | None
    rows: Mapping[str, pa.Table]
    deleted: Mapping[str, pa.Table]
    parts: Mapping[str, Mapping[str, pa.Table]]
    reloaded: frozenset[str] = frozenset()


class ApplyHooks(Protocol):
    """a product's behaviour on each change, run in the apply worker (a module-level object, pickled by name).

    ``derive`` returns pages of further changes to tables the change did not carry (a product's derived tables),
    applied into the same result, so they land in the change's own publish. A page to a table the change carried is
    dropped: the change's rows of it stand.
    """

    def derive(self, applied: AppliedTables) -> Sequence[ChangePage]:
        """the derived tables' pages.

        :param applied: what the change did
        :ptype applied: AppliedTables
        :return: pages to apply after it
        :rtype: Sequence[ChangePage]
        """
        ...


class MergePolicy(Protocol):
    """which of a scope's per-run parts merge next (a module-level object, pickled by name)."""

    def due(self, rule: PartRule, parts: Sequence[tuple[int, int, int, str]]) -> tuple[int, Sequence[str]] | None:
        """the level merged from and its parts, neighbours of one level in name order; None when none is due.

        :param rule: the table's rule
        :ptype rule: PartRule
        :param parts: one scope's parts: each one's first run, last run, level and name, by first run
        :ptype parts: Sequence[tuple[int, int, int, str]]
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

    def due(self, rule: PartRule, parts: Sequence[tuple[int, int, int, str]]) -> tuple[int, Sequence[str]] | None:
        """the first ``fan_out`` neighbours of one level, the newest run aside.

        :param rule: the table's rule (not read)
        :ptype rule: PartRule
        :param parts: one scope's parts, by first run
        :ptype parts: Sequence[tuple[int, int, int, str]]
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
    """

    version: str
    parts: Mapping[str, Mapping[str, bool]]
    rows: int
    written_bytes: int
    worker: int
    files: int = 0
    merge_due: bool = False


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
    :ivar located: its locator's file (:data:`LOCATED_PREFIX`); None for none
    :ivar scope: the scope values it replaces; None for the whole table
    :ivar rule: the rule it was cut by
    :ivar packed: its packs index's file (:data:`PACKED_PREFIX`), a whole reload's; None for none
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


@dataclass(frozen=True, slots=True)
class _MergePlan:
    root: str
    rules: tuple[tuple[str, PartRule], ...]
    policy: MergePolicy
    schemas: tuple[tuple[str, pa.Schema], ...] = ()
    pack_below: int = PACK_BELOW_BYTES
    pack_target: int = PACK_TARGET_BYTES
    max_files: int = MAX_VERSION_FILES


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

    def packing(self, table: str) -> bool:
        """whether ``table``'s small scopes are packed as it is cut."""
        return self.rules[table].split and self.pack_below > 0 and table not in self.unpacked


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
) -> _Settled:
    """a table's pages, in order, as the rows they leave upserted and the keys deleted: each key's last page wins.

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
            rows.append(page.select(schema.names).cast(schema))
            at = pa.array(np.arange(offset, offset + page.num_rows, dtype=np.int64))
            offset += page.num_rows
        else:
            at = pa.nulls(page.num_rows, pa.int64())
        events.append(keys.append_column(_ROW, at))
    if not events:
        empty = key_schema.empty_table()
        return _Settled(schema.empty_table(), empty, empty)
    happened = pa.concat_tables(events).combine_chunks()
    final = happened.take(pa.array(last_rows(happened, key)))
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


def _merged_tiers(rule: PartRule, policy: MergePolicy, scope: str, runs: Sequence[int]) -> Mapping[str, str]:
    """each level-0 part of a scope's runs and the part the tier merges leave it in: ``policy``'s merges, in turn,
    until none is due, as :meth:`SharedCopyBackend.merge` would make them one publish at a time.

    **Why** (reports' ``follower_parts._whole_tiers``). A whole change of a per-run table (a cold start, a reload) cut
    a part a run wrote the full start's 68 runs as 10,880 parts, each a file written, linked and mapped, and 365 merges
    after, one publish each, each holding the copy's next change (reports' full start measured 2026-11-17). Routed here, the change
    lands in the parts the merges would leave, and none is due after it.

    :param rule: the table's rule
    :ptype rule: PartRule
    :param policy: the merges
    :ptype policy: MergePolicy
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
    while (due := policy.due(rule, parts)) is not None:
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
    scopes: pa.Array, runs: pa.Array, held: Collection[str], merged: tuple[PartRule, MergePolicy] | None = None
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
    :ptype merged: tuple[PartRule, MergePolicy] | None
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


#: a packs index's columns (:data:`PACKED_PREFIX`): each packed scope value's part name and its pack, a locator's shape
_INDEX: Final = Locator().located.schema
_INDEX_SCOPE: Final = _INDEX.names[0]
_INDEX_PACK: Final = _INDEX.names[1]


def _index_of(files: Mapping[str, Any], table: str) -> pa.Table:
    """a split table's packs index as a version holds it; empty for none."""
    held = _parts_of(files, f"{PACKED_PREFIX}{table}")
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
    merges: MergePolicy | None = None,
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
    merges: MergePolicy | None = None,
    index: pa.Table | None = None,
) -> Mapping[str, pa.Table]:
    """rows by part, each part's rows a slice in their order in ``rows``: one stable sort of each row's part's name
    and one run-end encoding (``copy_parts.split_parts``' cut, by tier for a per-run table); with ``merges`` (a whole
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
        parts.extend(_cut(rule, pa.concat_tables(own), (), tables.merges).items())
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
    """one job's writes into its result slot, named ``p<n>`` in the order written: local to the job, never shared."""

    step: Any
    parts: list[tuple[str, str, str | None]]
    written_bytes: int = 0
    #: each part written and its file's bytes, the last write's
    sizes: dict[tuple[str, str], int] = field(default_factory=dict)

    def write(self, table: str, part: str, rows: pa.Table) -> None:
        name = f"p{len(self.parts)}"
        # one record batch a file: a reader maps each batch as arrays of its own, and letting a version go frees
        # every array (``MAX_VERSION_FILES``' why); a pack's scopes, concatenated, were a batch a scope
        wrote = self.step({name: rows.combine_chunks()})
        self.written_bytes += wrote
        self.sizes[(table, part)] = wrote
        self.parts.append((table, part, name))

    def drop(self, table: str, part: str) -> None:
        self.parts.append((table, part, None))


def _apply_table(
    tables: _Tables,
    table: str,
    change: _Settled,
    files: Mapping[str, Any],
    result: _Result,
    locators: Mapping[str, Locator] = EMPTY,
) -> Mapping[str, pa.Table]:
    """one table's change, its parts written into the result; its parts as the change leaves them, for the hooks.
    ``locators`` names a table's locator in place of its file's (a per-scope reload's, over the version's)."""
    rule, schema = tables.rules[table], tables.schemas[table]
    key = rule.key
    held = _parts_of(files, table)
    located = _parts_of(files, f"{LOCATED_PREFIX}{table}")
    packed = f"{PACKED_PREFIX}{table}"
    index = _index_of(files, table) if rule.split else _INDEX.empty_table()
    if table in locators:
        locator = locators[table]
    else:
        locator = Locator(_map(located[WHOLE_PART])) if WHOLE_PART in located else Locator()
    named = change.named if change.named.num_rows else None
    written = change.rows if change.rows.num_rows else None
    after: dict[str, pa.Table] = {}
    if table in tables.whole or _misshapen(rule, held):
        # the table in its rule's shape now: its rows (the change's alone, for a whole change) cut afresh, every part
        # held before dropped or rewritten
        before = None
        if held and table not in tables.whole:
            before = pa.concat_tables([_map(path, schema) for _, path in sorted(held.items())])
        rows = change.rows if before is None else (_part_rows(schema, before, written, key, named) or before)
        # a per-run table's runs land in the tiers the merges would leave them in, never a part a run; small scopes
        # land in packs
        if tables.packing(table):
            cut, fresh_index = _packed_cut(rule, rows, tables)
        else:
            cut, fresh_index = _cut(rule, rows, (), tables.merges), _INDEX.empty_table()
        for part, part_rows in cut.items():
            after[part] = _in_key_order(part_rows, key)
            result.write(table, part, after[part])
        for part in sorted(set(held) - set(cut)):
            result.drop(table, part)
        _write_index(result, packed, fresh_index)
        located_rows = rows if rule.locator is not None and rule.split else None
        fresh = Locator() if located_rows is None else Locator().extended(rule, located_rows)
        if fresh.located.num_rows:
            result.write(f"{LOCATED_PREFIX}{table}", WHOLE_PART, fresh.located)
        elif located:
            result.drop(f"{LOCATED_PREFIX}{table}", WHOLE_PART)
        return frozen(after)
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
        result.write(f"{LOCATED_PREFIX}{table}", WHOLE_PART, locator.extended(rule, change.rows).located)
    return frozen(after)


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
) -> tuple[Mapping[str, _Settled], Mapping[str, Mapping[str, pa.Table]]]:
    """each table's pages settled and applied into the result."""
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
            ),
        )
        for table, table_pages in sorted(by_table.items())
    )
    after = frozen(
        (table, _apply_table(tables, table, change, files, result, locators)) for table, change in settled.items()
    )
    return settled, after


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
    pages = [(kind, table, mapped[f"p{at}"]) for at, (kind, table) in enumerate(plan.pages)]
    tables = _Tables(
        schemas=frozen(plan.schemas),
        rules=frozen(plan.rules),
        whole=plan.whole,
        merges=plan.merges,
        instants=frozen(plan.instants),
        pack_below=plan.pack_below,
        pack_target=plan.pack_target,
        unpacked=plan.unpacked,
        canonical=frozen((table, frozen(columns)) for table, columns in plan.canonical),
    )
    slot = context.result
    if not isinstance(slot, (PiecedSlot, SegmentSlot)):
        raise ValueError("a change's apply writes into a result slot")
    rows = 0
    with slot.writing() as step:
        result = _Result(step, [])
        view = _reloaded(tables, base, plan.reloads, result)
        files = view.files
        settled, after = _apply_pages(tables, files, [*view.pages, *pages], result, view.locators)
        rows += sum(change.rows.num_rows for change in settled.values())
        if plan.hooks is not None:
            left = {**dict.fromkeys(view.tables, EMPTY), **after}
            applied = AppliedTables(
                root=plan.root,
                version=version,
                rows=frozen((table, change.rows) for table, change in settled.items()),
                deleted=frozen((table, change.deleted) for table, change in settled.items()),
                parts=frozen(
                    (table, _as_left(_parts_of(files, table), parts, result, table)) for table, parts in left.items()
                ),
                reloaded=view.tables,
            )
            derived = [page for page in plan.hooks.derive(applied) if page.table not in settled]
            if derived:
                more, _ = _apply_pages(
                    replace(tables, whole=frozenset()),
                    files,
                    [(p.kind, p.table, p.rows) for p in derived],
                    result,
                    view.locators,
                )
                rows += sum(change.rows.num_rows for change in more.values())
        touched = sorted({table for table, _, _ in result.parts if table in tables.rules and tables.packing(table)})
        due = False
        for table in touched:
            now, packed, _ = _packing_due(_sizes_after(files, table, result), tables.pack_below, tables.pack_target)
            due = due or packed < now
    parts, staged = _settled_parts(base, view.swaps, result.parts)
    return _Outcome(
        parts=parts,
        rows=rows,
        written_bytes=result.written_bytes,
        worker=os.getpid(),
        staged=staged,
        pack_due=due,
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
        located_name = f"{LOCATED_PREFIX}{table}"
        packed_name = f"{PACKED_PREFIX}{table}"
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
    for table, rule in sorted(plan.rules):
        if rule.run_column is None or not rule.split:
            continue
        for _scope, tiers in sorted(_tiers(_parts_of(files, table)).items()):
            parts = [(first, last, _level(name), name) for first, last, name in tiers]
            due = plan.policy.due(rule, parts)
            if due is not None and (best is None or due[0] < best[0]):
                best = (due[0], table, tuple(due[1]))
    if best is None:
        return _pack_job(context, plan, files)
    level, table, group = best
    ends = [tier for tier in (tier_of(group[0]), tier_of(group[-1])) if tier is not None]
    into = tier_part(ends[0][0], level + 1, ends[0][2], ends[-1][3])
    context.result.write({"m": merged_rows(plan.root, version, table, group)()})
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
            step({f"m{at}": _in_key_order(part_rows, rule.key).combine_chunks()})
            written.append((part, f"m{at}"))
        if fresh.num_rows:
            step({"i": fresh})
    held_index = WHOLE_PART in _parts_of(files, f"{PACKED_PREFIX}{table}")
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
            located=dict(staged.get(f"{LOCATED_PREFIX}{table}", ())).get(WHOLE_PART),
            scope=scopes.get(table),
            rule=rule,
            packed=dict(staged.get(f"{PACKED_PREFIX}{table}", ())).get(WHOLE_PART),
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
    :param max_files: a version holding more files than this makes a merge due (:data:`MAX_VERSION_FILES`)
    :ptype max_files: int
    :raises ValueError: for a rule of a table with no schema, a schema missing a rule's column, an instant or canonical
        column that is not a text key column of its table, or a canonical type with no one spelling
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
        max_files: int = MAX_VERSION_FILES,
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
    def rules(self) -> Mapping[str, PartRule]:
        """each table's parts, read-only.

        :return: table -> its rule
        :rtype: Mapping[str, PartRule]
        """
        return self._rules

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
            whole=frozenset(whole),
            hooks=self._hooks,
            merges=self._merges,
            instants=self._instants,
            reloads=reloads,
            pack_below=self._pack_below,
            pack_target=self._pack_target,
            canonical=self._canonical,
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
        self._unknown(pages, whole)
        reloads = tuple(reload for reload in staged.reloads if reload.table not in set(whole))
        applied = await self._applied(pages, whole, mark, reloads)
        return replace(applied, rows=applied.rows + staged.rows)

    async def _applied(
        self, pages: Sequence[ChangePage], whole: Collection[str], mark: int | None, reloads: tuple[Reloaded, ...]
    ) -> Applied:
        sent = [page for page in pages if page.rows.num_rows]
        async with self._applying:
            current = self._store.current
            base = None if current is None else current.name
            done = await self._pool.run(
                apply_job,
                self._plan(sent, whole, reloads),
                read=False if base is None else base,
                tables={f"p{at}": page.rows for at, page in enumerate(sent)} or None,
                result_pieces=apply_piece_bytes(self._store.max_bytes, sent, self._held_largest(base, sent)),
                result_minimum=_APPLY_PIECE_MINIMUM_BYTES,
                result_grow_to=_APPLY_PIECE_BYTES,
            )
            outcome: _Outcome = done.value
            version = f"c-{self._next}"
            try:
                files = await asyncio.to_thread(self._publish_change, version, base, outcome, done.segment, mark)
            finally:
                if done.segment is not None:
                    await asyncio.to_thread(done.segment.close)
            self._next += 1
            over = self._said_over(version, files)
        parts: dict[str, dict[str, bool]] = {}
        for table, part, _path in outcome.staged:
            parts.setdefault(table, {})[part] = True
        for table, part, name in outcome.parts:
            parts.setdefault(table, {})[part] = name is not None
        return Applied(
            version=version,
            parts=frozen((table, frozen(done_parts)) for table, done_parts in parts.items()),
            rows=outcome.rows,
            written_bytes=outcome.written_bytes,
            worker=outcome.worker,
            files=files,
            merge_due=outcome.pack_due or over,
        )

    def _held_largest(self, base: str | None, pages: Sequence[ChangePage]) -> Mapping[str, int]:
        """each table ``pages`` change and the largest file ``base`` holds of it (a part, its locator, its packs
        index), from the store's manifest: no file read. Empty for no base."""
        if base is None:
            return EMPTY
        return frozen(
            (
                table,
                self._store.largest_file_bytes(base, (table, f"{LOCATED_PREFIX}{table}", f"{PACKED_PREFIX}{table}")),
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
            result_minimum=_APPLY_PIECE_MINIMUM_BYTES,
            result_grow_to=_APPLY_PIECE_BYTES,
        )
        return _staged(running, rules, frozen(scopes))

    def _publish_change(
        self, version: str, base: str | None, outcome: _Outcome, segment: MappedSegment | None, mark: int | None
    ) -> int:
        """publish a change's result on its base (a thread's: IO alone, each written part linked, never copied; a
        reload's staged parts linked from where it staged them); the files the version holds."""
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
        return len(version_files(self._store.root, version))

    def _schema_of(self, table: str) -> pa.Schema:
        if table.startswith(LOCATED_PREFIX):
            return Locator().located.schema
        if table.startswith(PACKED_PREFIX):
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
                ),
                read=current.name,
                # which table merges is the worker's to find: a piece of the least, each merged file growing its own
                # to its rows (a pack's target, a tier's parts), never a GiB held for the merge's whole run
                result_pieces=_APPLY_PIECE_MINIMUM_BYTES,
                result_minimum=_APPLY_PIECE_MINIMUM_BYTES,
                result_grow_to=_APPLY_PIECE_BYTES,
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
                    tables[f"{PACKED_PREFIX}{table}"] = Parts(_INDEX, {WHOLE_PART: index})
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


def apply_piece_bytes(max_bytes: int, pages: Sequence[ChangePage], held: Mapping[str, int] = EMPTY) -> int:
    """the bytes a change's result piece is reserved at: over the tables it changes, the largest of a table's largest
    file the version holds and the change's rows of that table, and an eighth more, at least
    ``_APPLY_PIECE_MINIMUM_BYTES`` (16 MiB) more, at most ``_APPLY_PIECE_BYTES`` (1 GiB) and a quarter of the store
    (:data:`_APPLY_STORE_SHARE`'s why). A file that needs more grows its piece. Arrow ``nbytes``, no row read.

    :param max_bytes: the store's bound
    :ptype max_bytes: int
    :param pages: the change's pages
    :ptype pages: Sequence[ChangePage]
    :param held: each changed table's largest file in the version the change applies to; empty for none
    :ptype held: Mapping[str, int]
    :return: bytes
    :rtype: int
    """
    by_table: dict[str, int] = {}
    for page in pages:
        by_table[page.table] = by_table.get(page.table, 0) + page.rows.nbytes
    largest = max((held.get(table, 0) + rows for table, rows in by_table.items()), default=0)
    wanted = largest + largest // _APPLY_PIECE_SLACK_DIVISOR + _APPLY_PIECE_MINIMUM_BYTES
    bound = min(_APPLY_PIECE_BYTES, max(_APPLY_PIECE_MINIMUM_BYTES, max_bytes // _APPLY_STORE_SHARE))
    return max(_APPLY_PIECE_MINIMUM_BYTES, min(wanted, bound))


def reload_piece_bytes(max_bytes: int, pages: Sequence[ChangePage]) -> int:
    """the most one piece of a reload's result may take: its largest table's bytes and an eighth more, at least
    ``_APPLY_PIECE_MINIMUM_BYTES`` (16 MiB), at most ``_APPLY_PIECE_BYTES`` (1 GiB) and a quarter of the store
    (:data:`_RELOAD_STORE_SHARE`'s why). Arrow ``nbytes``, no row read.

    :param max_bytes: the store's bound
    :ptype max_bytes: int
    :param pages: the reload's pages
    :ptype pages: Sequence[ChangePage]
    :return: bytes
    :rtype: int
    """
    by_table: dict[str, int] = {}
    for page in pages:
        by_table[page.table] = by_table.get(page.table, 0) + page.rows.nbytes
    largest = max(by_table.values(), default=0)
    wanted = largest + largest // _RELOAD_PIECE_SLACK_DIVISOR + _APPLY_PIECE_MINIMUM_BYTES
    bound = min(_APPLY_PIECE_BYTES, max(_APPLY_PIECE_MINIMUM_BYTES, max_bytes // _RELOAD_STORE_SHARE))
    return max(_APPLY_PIECE_MINIMUM_BYTES, min(wanted, bound))


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
