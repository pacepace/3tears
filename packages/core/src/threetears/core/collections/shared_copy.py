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
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

from threetears.core.arrow_keys import distinct_keys, key_codes, last_rows, rows_named, stored_instants
from threetears.core.collections.copy_parts import (
    WHOLE_PART,
    Locator,
    PartRule,
    parts_of_keys,
    split_parts,
    tier_of,
    tier_part,
)
from threetears.core.frozen_state import EMPTY, frozen
from threetears.core.shared_arrow import (
    JobContext,
    MappedSegment,
    Parts,
    PiecedSlot,
    SegmentSlot,
    SharedArrowPool,
    SharedArrowStore,
    merged_rows,
    version_files,
)
from threetears.observe import get_logger

__all__ = [
    "KEYS",
    "LOCATED_PREFIX",
    "MARK_TABLE",
    "ROWS",
    "Applied",
    "AppliedTables",
    "ApplyPlan",
    "ApplyHooks",
    "ByCount",
    "ChangePage",
    "MergePolicy",
    "Merged",
    "SharedCopyBackend",
    "apply_job",
    "file_name",
    "held_parts",
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

#: the one-row table each change's mark is published as (``mark``, int64): a reader of a version reads the mark of
#: the rows it maps, never a newer one. Moved from reports (``follower_parts.MARK_TABLE``)
MARK_TABLE: Final = "__staged_mark__"

#: how many neighbouring parts of one level :class:`ByCount` merges into one of the next. Moved from reports
#: (``follower_parts.FAN_OUT``): a scope of N runs holds about ``FAN_OUT * log10(N)`` parts and each row is rewritten
#: once a level
_FAN_OUT: Final = 10

#: the most one piece of a change's result may take (``SharedArrowStore.reserve_pieces``): each piece is reserved as
#: the one before fills, so a store is never asked for this much room at once. Moved from reports
#: (``shared_backend.SHARED_COPY_RESULT_BYTES``): a fixed 1 GiB slot refused every change of a store with less room
_APPLY_PIECE_BYTES: Final = 1 << 30

#: the least room a change's piece is reserved at: one part's bound, a piece holding each file whole. Moved from
#: reports (``shared_backend.SHARED_COPY_PIECE_MINIMUM_BYTES``): the largest file a change wrote in its 2026-11-03
#: replay's cold start was a few MiB (a 123 MB table in 53 parts)
_APPLY_PIECE_MINIMUM_BYTES: Final = 16 << 20

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
    """

    root: str
    version: str | None
    rows: Mapping[str, pa.Table]
    deleted: Mapping[str, pa.Table]
    parts: Mapping[str, Mapping[str, pa.Table]]


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
    """

    version: str
    parts: Mapping[str, Mapping[str, bool]]
    rows: int
    written_bytes: int
    worker: int


@dataclass(frozen=True, slots=True)
class Merged:
    """one tier merge published.

    :ivar version: the version published
    :ivar table: the table
    :ivar into: the merged part
    :ivar parts: the parts merged into it, dropped
    """

    version: str
    table: str
    into: str
    parts: tuple[str, ...]


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
    """

    root: str
    schemas: tuple[tuple[str, pa.Schema], ...]
    rules: tuple[tuple[str, PartRule], ...]
    pages: tuple[tuple[str, str], ...]
    whole: frozenset[str]
    hooks: ApplyHooks | None
    merges: MergePolicy = ByCount()
    instants: tuple[tuple[str, frozenset[str]], ...] = ()


@dataclass(frozen=True, slots=True)
class _Outcome:
    """what the apply worker gives back: each part written (its result table's name) or dropped (None)."""

    parts: tuple[tuple[str, str, str | None], ...]
    rows: int
    written_bytes: int
    worker: int


@dataclass(frozen=True, slots=True)
class _MergePlan:
    root: str
    rules: tuple[tuple[str, PartRule], ...]
    policy: MergePolicy


@dataclass(frozen=True, slots=True)
class _Tables:
    """a job's tables, read-only, made in the worker from its plan's pairs."""

    schemas: Mapping[str, pa.Schema]
    rules: Mapping[str, PartRule]
    whole: frozenset[str]
    merges: MergePolicy = ByCount()
    instants: Mapping[str, frozenset[str]] = EMPTY


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


def _respelled(page: pa.Table, instants: Collection[str]) -> pa.Table:
    """a page with each of its instant key columns in its stored spelling (``arrow_keys.stored_instants``)."""
    for column in sorted(instants):
        if column in page.column_names:
            page = page.set_column(page.column_names.index(column), column, stored_instants(page[column]))
    return page


def _settled(
    schema: pa.Schema,
    key: Sequence[str],
    pages: Sequence[tuple[str, pa.Table]],
    instants: Collection[str] = frozenset(),
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
        page = _respelled(page, instants)
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


def _row_parts(rule: PartRule, rows: pa.Table, held: Collection[str], merges: MergePolicy | None = None) -> pa.Array:
    """each row's part, by its own columns (a split table's rows hold its scope column); with ``merges``, a whole
    change's runs in the tiers those merges leave (:func:`_merged_tiers`)."""
    if not rule.split:
        return pa.repeat(pa.scalar(WHOLE_PART, pa.string()), rows.num_rows)
    scopes = parts_of_keys(rule, rows)
    if rule.run_column is not None:
        merged = None if merges is None else (rule, merges)
        scopes = _run_parts(scopes, rows[rule.run_column].combine_chunks(), held, merged)
    return scopes


def _key_parts(rule: PartRule, keys: pa.Table, held: Collection[str], locator: Locator) -> pa.Array:
    """each key's held part: by its scope column, or its locator value's part as the version held it; null for none."""
    if not rule.split:
        return pa.repeat(pa.scalar(WHOLE_PART, pa.string()), keys.num_rows)
    scopes = parts_of_keys(rule, keys, locator)
    if rule.run_column is not None:
        scopes = _run_parts(scopes, keys[rule.run_column].combine_chunks(), held)
    return scopes


def _cut(
    rule: PartRule, rows: pa.Table, held: Collection[str], merges: MergePolicy | None = None
) -> Mapping[str, pa.Table]:
    """rows by part, each part's rows a slice in their order in ``rows``: ``copy_parts.split_parts``' cut, and for a
    per-run table the same cut (one stable sort of the parts' names, one run-end encoding) by each row's tier; with
    ``merges`` (a whole change), the tiers those merges leave."""
    if not rows.num_rows:
        return EMPTY
    if rule.run_column is None:
        return frozen((part.name, part.rows) for part in split_parts(rule, rows))
    names = _row_parts(rule, rows, held, merges)
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


def _misshapen(rule: PartRule, held: Collection[str]) -> bool:
    """whether a table's held parts are not the shape its rule names now (it crossed the size floor, say)."""
    if not held:
        return False
    if not rule.split:
        return set(held) != {WHOLE_PART}
    tiered = [tier_of(part) is not None for part in held]
    return WHOLE_PART in held or (any(tiered) != (rule.run_column is not None)) or len(set(tiered)) > 1


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

    def write(self, table: str, part: str, rows: pa.Table) -> None:
        name = f"p{len(self.parts)}"
        self.written_bytes += self.step({name: rows})
        self.parts.append((table, part, name))

    def drop(self, table: str, part: str) -> None:
        self.parts.append((table, part, None))


def _apply_table(
    tables: _Tables,
    table: str,
    change: _Settled,
    files: Mapping[str, Any],
    result: _Result,
) -> Mapping[str, pa.Table]:
    """one table's change, its parts written into the result; its parts as the change leaves them, for the hooks."""
    rule, schema = tables.rules[table], tables.schemas[table]
    key = rule.key
    held = _parts_of(files, table)
    located = _parts_of(files, f"{LOCATED_PREFIX}{table}")
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
        # a per-run table's runs land in the tiers the merges would leave them in, never a part a run
        cut = _cut(rule, rows, (), tables.merges)
        for part, part_rows in cut.items():
            after[part] = _in_key_order(part_rows, key)
            result.write(table, part, after[part])
        for part in sorted(set(held) - set(cut)):
            result.drop(table, part)
        located_rows = rows if rule.locator is not None and rule.split else None
        fresh = Locator() if located_rows is None else Locator().extended(rule, located_rows)
        if fresh.located.num_rows:
            result.write(f"{LOCATED_PREFIX}{table}", WHOLE_PART, fresh.located)
        elif located:
            result.drop(f"{LOCATED_PREFIX}{table}", WHOLE_PART)
        return frozen(after)
    by_part = _cut(rule, change.rows, held)
    targets = set(by_part)
    if named is not None:
        named_parts = _key_parts(rule, named, held, locator)
        targets |= set(pc.unique(named_parts.drop_null()).to_pylist()) & set(held)
    for part in sorted(targets):
        held_rows = _map(held[part], schema) if part in held else None
        rows = _part_rows(schema, held_rows, by_part.get(part), key, named)
        if rows is None:
            continue
        if not rows.num_rows:
            if part in held:
                result.drop(table, part)
            continue
        result.write(table, part, rows)
        after[part] = rows
    if rule.locator is not None and rule.split and written is not None:
        result.write(f"{LOCATED_PREFIX}{table}", WHOLE_PART, locator.extended(rule, change.rows).located)
    return frozen(after)


def _apply_pages(
    tables: _Tables,
    files: Mapping[str, Any],
    pages: Sequence[tuple[str, str, pa.Table]],
    result: _Result,
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
                tables.schemas[table], tables.rules[table].key, table_pages, tables.instants.get(table, frozenset())
            ),
        )
        for table, table_pages in sorted(by_table.items())
    )
    after = frozen((table, _apply_table(tables, table, change, files, result)) for table, change in settled.items())
    return settled, after


def apply_job(context: JobContext, plan: ApplyPlan) -> _Outcome:
    """a change applied in the apply worker (:class:`SharedCopyBackend`): its pages mapped from the hand-off, the
    version it applies to mapped, and each part it rewrites written into the job's result, then the hooks' derived
    tables into the same result.

    :param context: the job's version (None for the first change), its pages (``p<n>``) and its result slot
    :ptype context: JobContext
    :param plan: the tables' schemas and rules, the pages' kinds and tables, the tables replaced whole, the hooks
    :ptype plan: ApplyPlan
    :return: each part written or dropped, the rows upserted, the bytes written, this process
    :rtype: _Outcome
    """
    version = None if context.version is None else context.version.name
    files = EMPTY if version is None else version_files(plan.root, version)
    mapped = EMPTY if context.tables is None else context.tables.tables
    pages = [(kind, table, mapped[f"p{at}"]) for at, (kind, table) in enumerate(plan.pages)]
    tables = _Tables(
        schemas=frozen(plan.schemas),
        rules=frozen(plan.rules),
        whole=plan.whole,
        merges=plan.merges,
        instants=frozen(plan.instants),
    )
    slot = context.result
    if not isinstance(slot, (PiecedSlot, SegmentSlot)):
        raise ValueError("a change's apply writes into a result slot")
    rows = 0
    with slot.writing() as step:
        result = _Result(step, [])
        settled, after = _apply_pages(tables, files, pages, result)
        rows += sum(change.rows.num_rows for change in settled.values())
        if plan.hooks is not None:
            applied = AppliedTables(
                root=plan.root,
                version=version,
                rows=frozen((table, change.rows) for table, change in settled.items()),
                deleted=frozen((table, change.deleted) for table, change in settled.items()),
                parts=frozen(
                    (table, _as_left(_parts_of(files, table), parts, result, table)) for table, parts in after.items()
                ),
            )
            derived = [page for page in plan.hooks.derive(applied) if page.table not in settled]
            if derived:
                more, _ = _apply_pages(
                    replace(tables, whole=frozenset()), files, [(p.kind, p.table, p.rows) for p in derived], result
                )
                rows += sum(change.rows.num_rows for change in more.values())
    return _Outcome(parts=tuple(result.parts), rows=rows, written_bytes=result.written_bytes, worker=os.getpid())


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


def _merge_job(context: JobContext, plan: _MergePlan) -> tuple[str, str, tuple[str, ...]] | None:
    """one tier merge due, the lowest level first, written into the job's result as ``m``: the group's rows copied
    once from their mapped files (``shared_arrow.merged_rows``), nothing else read.

    :param context: the job's version and result slot
    :ptype context: JobContext
    :param plan: the tables' rules and the merge policy
    :ptype plan: _MergePlan
    :return: the table, the merged part and the parts it replaces; None when none is due
    :rtype: tuple[str, str, tuple[str, ...]] | None
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
        return None
    level, table, group = best
    ends = [tier for tier in (tier_of(group[0]), tier_of(group[-1])) if tier is not None]
    into = tier_part(ends[0][0], level + 1, ends[0][2], ends[-1][3])
    context.result.write({"m": merged_rows(plan.root, version, table, group)()})
    return table, into, group


# -- the serving process's side ------------------------------------------------------------------------------------


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
    :raises ValueError: for a rule of a table with no schema, a schema missing a rule's column, or an instant column
        that is not a text key column of its table
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
    ) -> None:
        self._schemas = frozen(schemas)
        self._rules = _checked(self._schemas, rules)
        self._instants = _instant_keys(self._schemas, self._rules, instants)
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
        unknown = sorted(
            {page.table for page in pages if page.table not in self._rules} | (set(whole) - set(self._rules))
        )
        if unknown:
            raise ValueError(f"a change to tables the copy does not hold: {unknown}")
        sent = [page for page in pages if page.rows.num_rows]
        async with self._applying:
            current = self._store.current
            base = None if current is None else current.name
            plan = ApplyPlan(
                root=str(self._store.root),
                schemas=tuple(self._schemas.items()),
                rules=tuple(self._rules.items()),
                pages=tuple((page.kind, page.table) for page in sent),
                whole=frozenset(whole),
                hooks=self._hooks,
                merges=self._merges,
                instants=self._instants,
            )
            done = await self._pool.run(
                apply_job,
                plan,
                read=False if base is None else base,
                tables={f"p{at}": page.rows for at, page in enumerate(sent)} or None,
                result_pieces=_APPLY_PIECE_BYTES,
                result_minimum=_APPLY_PIECE_MINIMUM_BYTES,
            )
            outcome: _Outcome = done.value
            version = f"c-{self._next}"
            try:
                await asyncio.to_thread(self._publish_change, version, base, outcome, done.segment, mark)
            finally:
                if done.segment is not None:
                    await asyncio.to_thread(done.segment.close)
            self._next += 1
        parts: dict[str, dict[str, bool]] = {}
        for table, part, name in outcome.parts:
            parts.setdefault(table, {})[part] = name is not None
        return Applied(
            version=version,
            parts=frozen((table, frozen(done_parts)) for table, done_parts in parts.items()),
            rows=outcome.rows,
            written_bytes=outcome.written_bytes,
            worker=outcome.worker,
        )

    def _publish_change(
        self, version: str, base: str | None, outcome: _Outcome, segment: MappedSegment | None, mark: int | None
    ) -> None:
        """publish a change's result on its base (a thread's: IO alone, each written part linked, never copied)."""
        files = EMPTY if segment is None else segment.files
        by_table: dict[str, dict[str, Path | None]] = {}
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

    def _schema_of(self, table: str) -> pa.Schema:
        if table.startswith(LOCATED_PREFIX):
            return Locator().located.schema
        return self._schemas[table]

    async def merge(self) -> Merged | None:
        """one tier merge due (:class:`MergePolicy`), the lowest level first, written in the apply worker and
        published as a version of its own: never between a change and the version showing it.

        :return: what it merged; None when none is due
        :rtype: Merged | None
        """
        if not any(rule.run_column is not None and rule.split for rule in self._rules.values()):
            return None
        async with self._applying:
            current = self._store.current
            if current is None:
                return None
            done = await self._pool.run(
                _merge_job,
                _MergePlan(root=str(self._store.root), rules=tuple(self._rules.items()), policy=self._merges),
                read=current.name,
                result_pieces=_APPLY_PIECE_BYTES,
                result_minimum=_APPLY_PIECE_MINIMUM_BYTES,
            )
            if done.value is None:
                if done.segment is not None:
                    await asyncio.to_thread(done.segment.close)
                return None
            table, into, group = done.value
            version = f"c-{self._next}"
            segment = done.segment
            if segment is None:
                raise RuntimeError(f"a merge of {table!r} into {into!r} wrote nothing")
            try:
                parts: dict[str, Path | None] = {**dict.fromkeys(group), into: segment.files["m"]}
                await asyncio.to_thread(
                    self._store.publish, version, {table: Parts(self._schemas[table], parts)}, base=current.name
                )
            finally:
                await asyncio.to_thread(segment.close)
            self._next += 1
        log.debug("tier merge", extra={"extra_data": {"table": table, "into": into, "parts": len(group)}})
        return Merged(version=version, table=table, into=into, parts=tuple(group))

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
