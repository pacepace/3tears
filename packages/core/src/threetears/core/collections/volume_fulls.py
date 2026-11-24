"""a full on the delta-files volume: per (table, state) part files and a manifest naming them, fenced by term.

**Why.** A table that opts in to delta files (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_files`)
keeps every change on the shared volume, and nothing there expires on a timer, so a pod copy can always catch up
(``design-pod-delta-files.md``). Its starting point, the full, lives on the same volume instead of the pod's NATS
Object Store (decision 7): a copy loads the full, then applies every delta newer than each part. L3 is never read
whole by a copy: with no full on the volume it waits for the writer to publish one (decision 14).

**Parts and a manifest** (decision 8). A full is one Arrow IPC file (zstd inside, decision 18) per (table, state), its
rows the table's columns typed as the L3 rail answers them, plus a manifest naming, per table, each part's file and the
version (the table's generation token, ``{incarnation}:{count}``) the part is current to, and the version the table as
a whole is current to (a state with no part holds no rows at it). A delta file may be named as a part (decision 16):
its delete markers are dropped as it loads, and any column beyond the table's own is left out. Files never change once
written; a new full is new parts for the states that changed plus a new manifest naming every part.

**Fenced by term** (decision 15). Every manifest is named by the writer's term (its write lease's, only ever going up
across handovers) and a sequence within the term: ``full-<term>-<seq>.json``. Readers take the highest term, then the
newest sequence in it. A writer that lost its lease and wakes to publish writes an older term's manifest, which no
reader takes and :func:`deletable_full_files` lists; nothing is overwritten, so a stale writer never replaces the
current writer's work. Where the term comes from is the writer's to settle; this module takes it as given.

**Published by rename** (decision 8). A manifest is written under a temporary name, flushed to disk, renamed to its
own name and the directory flushed: a reader sees the whole manifest or none, never part of one. A name already taken
is refused, never replaced: one writer holds a term (the lease), so nothing else makes a name of it.

**Kept and deleted.** The current manifest and the one it replaced (its ``previous``, which a reader that took it
before the publish may still be loading) are kept with every file they name; :func:`deletable_full_files` lists the
rest for the writer to delete after its own publish. Nothing calls it on a timer. It lists a delta file only when the
writer hands it covered ones, which a fold that keeps its delta files (``retention="keep"``, decision 51) never does.

**Fingerprinted** (decision 32). Each part's manifest entry carries :func:`rows_fingerprint` of its rows: each row's
columns, in name order, made canonical by a column type in Arrow (:data:`_FINGERPRINT_KINDS`, so the rail's types and
a copy's DuckDB types of the same values agree) and rendered as one length-prefixed text in DuckDB, hashed (MD5, 128
bits) and the hashes' two 64-bit halves each summed modulo 2**64, all a column at a time, never a row at a time in
Python: so the order rows are held in never counts and a row held twice does; with the row count and a short hash of
the columns and their kinds, so a copy holding other columns is told apart from one holding other rows, and the
format (:data:`FINGERPRINT_FORMAT`), so a part an older writer fingerprinted another way is never compared. A pod copy
fingerprints its own rows of a state held at the part's version the same way and compares
(:meth:`~threetears.core.collections.copy_follower.CopyFollower.check_drift`); the writer's sample of L3 by key
compares rows by the same per-row hash (:mod:`~threetears.core.collections.delta_drift`). One definition, here, for
both sides.

**Where.** Under the delta files' root (``THREETEARS_DELTA_FILES_DIR``), in ``.fulls/``: ``.fulls/manifests/`` and
``.fulls/parts/<table>/``. A table's delta files are ``<table>/<write>.arrow``; a table is an identifier, which never
begins with a dot, so no table's directory is ``.fulls``, and the delta files' orphan sweep, which looks only at the
files directly under each directory of the root, never reaches a part or a manifest.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

from threetears.observe import get_logger
from threetears.observe.metrics import counter, histogram

from threetears.core.cache.duckdb import STAGE_PREFIX, StagedChange
from threetears.core.collections.delta_files import DeltaFileUnreadableError, read_volume_file, write_volume_file
from threetears.core.collections.generation import split_generation_token
from threetears.core.collections.schema_backed import rail_arrow_types

if TYPE_CHECKING:
    from threetears.core.collections.copy_follower import FollowedTable, FollowerBackend

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as pa_ipc

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a full refuses without it, saying so
    _PYARROW = False

try:
    import duckdb

    _DUCKDB = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a fingerprint needs it (the rows are hashed in DuckDB)
    _DUCKDB = False

__all__ = [
    "FINGERPRINT_CHECK_SECONDS",
    "FINGERPRINT_FORMAT",
    "FullAbsent",
    "FullManifest",
    "FullPart",
    "FullTable",
    "StagedFull",
    "StaleTermError",
    "VolumeFulls",
    "current_full_manifest",
    "deletable_full_files",
    "fingerprint_columns",
    "fingerprint_comparable",
    "fingerprint_ipc",
    "fingerprint_kinds",
    "full_table_floor",
    "other_columns_parts",
    "publish_full_manifest",
    "read_full_part",
    "row_digests",
    "rows_fingerprint",
    "write_full_part",
]

log = get_logger(__name__)

#: each copy's check of one part against the full's fingerprint (decision 32,
#: :meth:`~threetears.core.collections.copy_follower.CopyFollower.check_drift`), its seconds by outcome (counted by
#: outcome in ``copy_follower_drift_checks_total``): the copy's read of its rows and their fingerprint in a worker
#: process, observed once the check ends
FINGERPRINT_CHECK_SECONDS = histogram(
    "volume_fulls_fingerprint_check_seconds",
    description="seconds of a copy's check of one part of the full against its fingerprint (its own rows of the "
    "state read and fingerprinted), by outcome (equal, drift, skipped, not_current, version_unknown, no_fingerprint, "
    "other_columns)",
    label_names=("outcome",),
)

#: a copy's checks of each part it loads from the full (:meth:`VolumeFulls.stage`), by outcome: ``ok`` (its checksum
#: verified and its rows the count its manifest says), ``unreadable`` (damaged: its checksum failed), ``missing`` (not
#: on the volume), ``rows_differ`` (another count of rows than its manifest says). Any but ``ok`` fails the copy's start
_PART_CHECKS = counter(
    "volume_fulls_part_checks_total",
    description="a copy's checks of each part of the full it loads (checksum, then the row count its manifest says), "
    "by outcome: ok, unreadable, missing, rows_differ; any but ok fails the start",
    label_names=("outcome",),
)
#: each such check's seconds (the part read off the volume and verified), by outcome, on the copy's worker thread
_PART_CHECK_SECONDS = histogram(
    "volume_fulls_part_check_seconds",
    description="seconds of a copy's check of one part of the full it loads (read off the volume, checksum verified, "
    "rows counted), by outcome",
    label_names=("outcome",),
)

#: a copy's looks at a full holding one of its tables over other columns than the table's schema (a column added or
#: dropped in place by a newer table list, decision 12): the copy waits for the writer's fresh full, reading no L3
_OTHER_COLUMNS = counter(
    "volume_fulls_other_columns_total",
    description="a copy's looks at a full whose parts of a table were made over other columns than the table's "
    "schema: it waits for the writer's fresh full of the table, reading nothing from L3",
    label_names=("table",),
)

#: the fulls' directory under the delta files' root: a name no table's directory can take (a table is an identifier)
_FULLS: Final = ".fulls"
#: the manifests' directory, under :data:`_FULLS`
_MANIFESTS: Final = "manifests"
#: the parts' directory, under :data:`_FULLS`: one directory a table
_PARTS: Final = "parts"
#: a manifest's name: its term and sequence, zero-padded so a listing reads in order
_MANIFEST_NAME: Final = re.compile(r"^full-(\d{20})-(\d{20})\.json$")
#: the manifest's format; one of another is refused
_FORMAT: Final = 1
#: a part's suffix: the delta files' own, so a delta named as a part reads the same way
_SUFFIX: Final = ".arrow"
#: the suffix of a file being written, before its rename; never a name a manifest holds
_PARTIAL_SUFFIX: Final = ".partial"
#: a table name as it may become a directory
_TABLE_NAME: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: the column a delta file marks its delete markers by (``delta_files._DELETED_COLUMN``): a delta named as a part
#: loads its rows, never its markers
_DELETED_COLUMN: Final = "_3tears_deleted"
#: a part's schema metadata: its table, its state and the version it is current to
_TABLE_METADATA: Final = b"threetears.full.table"
_SCOPE_METADATA: Final = b"threetears.full.scope"
_VERSION_METADATA: Final = b"threetears.full.version"
#: how often a copy with no full to load looks for one again (seconds): one listing of the manifests' directory
_DEFAULT_POLL_SECONDS: Final = 1.0
#: the kinds a fingerprint renders a column's value by, one an Arrow type the L3 rail answers
#: (:func:`~threetears.core.collections.schema_backed.rail_arrow_types`): text, an integer, a boolean, an instant; any
#: other renders as DuckDB's text of it
_FINGERPRINT_KINDS: Final = ("text", "int", "bool", "instant", "other")
#: the fingerprint's format, which every fingerprint opens with (``<format>:``): one of another format (``f1``, the
#: per-row BLAKE2b sum in Python, wrote none) is never compared, so a part an older writer fingerprinted reads as
#: ``no_fingerprint`` on a copy, never as drift. Raise it whenever the definition changes.
FINGERPRINT_FORMAT: Final = "f2"
#: the modulus each half of the rows' hashes is summed under
_HALF_MODULUS: Final = 1 << 64
#: an instant's text as :func:`~threetears.core.cache.base.stored_instant` spells it (``%S`` of a microsecond
#: timestamp carries its six places): the spelling the rail's text of an instant arrives in
_STORED_INSTANT_FORMAT: Final = "%Y-%m-%dT%H:%M:%S+00:00"
#: the rows a batch DuckDB scans a fingerprint's rows in: a batch a scan thread
_SCAN_BATCH_ROWS: Final = 32_768
#: ISO text that names its offset; text that names none is the rail's UTC kept without its zone
_ZONED_TEXT: Final = r"(Z|[+-]\d\d(:?\d\d)?)$"


class StaleTermError(RuntimeError):
    """a writer would publish a full under a term older than the current manifest's: another writer took the lease
    since, so this one's full is refused before it is written (readers would ignore it anyway)."""


def fingerprint_kinds(types: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """each column's fingerprint kind (:data:`_FINGERPRINT_KINDS`), from its Arrow type, in column-name order: what
    :func:`rows_fingerprint` renders a row by. A copy passes the types the L3 rail answers its columns as, whatever its
    own backend keeps them as; a part's writer, its rows' own.

    :param types: column -> its Arrow type
    :ptype types: Mapping[str, pyarrow.DataType]
    :return: ``(column, kind)`` pairs, sorted by column
    :rtype: tuple[tuple[str, str], ...]
    """
    kinds: list[tuple[str, str]] = []
    for name, kind in types.items():
        if pa.types.is_string(kind) or pa.types.is_large_string(kind):
            kinds.append((name, "text"))
        elif pa.types.is_integer(kind):
            kinds.append((name, "int"))
        elif pa.types.is_boolean(kind):
            kinds.append((name, "bool"))
        elif pa.types.is_timestamp(kind):
            kinds.append((name, "instant"))
        else:
            kinds.append((name, "other"))
    return tuple(sorted(kinds))


def _canonical_column(values: Any, kind: str) -> Any:
    """one column as the fingerprint renders it, by its kind, in Arrow (a kernel a column, never a value at a time),
    so the rail's type and the type a copy keeps the same values as come out as one text: text stays text, a
    ``NUMERIC`` a copy keeps as a decimal is its digits at its scale, a timestamp a copy keeps in a text column is
    :func:`~threetears.core.cache.base.stored_instant`'s text of it; an integer is its digits; a boolean ``t`` or
    ``f``; an instant is the digits of its microseconds since the epoch in UTC, whether held as a timestamp or as ISO
    text (a timestamp or text naming no zone is the rail's UTC kept without it). Any other kind is left for DuckDB's
    text of it.

    :param values: the column
    :ptype values: pyarrow.ChunkedArray
    :param kind: its kind
    :ptype kind: str
    :return: the column as text; an ``other`` column as it is
    :rtype: pyarrow.ChunkedArray
    """
    held = values.type
    textual = pa.types.is_string(held) or pa.types.is_large_string(held)
    if kind == "instant":
        if textual:
            zoned = pc.match_substring_regex(values, _ZONED_TEXT)
            values = pc.cast(
                pc.if_else(zoned, values, pc.binary_join_element_wise(values, "+00:00", "")),
                pa.timestamp("us", tz="UTC"),
            )
        elif pa.types.is_timestamp(held):
            # a naive timestamp's storage is its wall clock, which is the rail's UTC: the same integer
            values = pc.cast(values, pa.timestamp("us", tz=held.tz), safe=False)
        return pc.cast(pc.cast(values, pa.int64()), pa.string())
    if kind == "text" and pa.types.is_timestamp(held):
        utc = pc.cast(pc.cast(values, pa.timestamp("us", tz=held.tz), safe=False), pa.timestamp("us", tz="UTC"))
        return pc.strftime(utc, format=_STORED_INSTANT_FORMAT)
    if kind == "int":
        return pc.cast(pc.cast(values, pa.int64()), pa.string())
    if kind == "bool":
        return pc.if_else(pc.cast(values, pa.bool_()), "t", "f")
    if kind == "text":
        # text as it is; a decimal as its digits at its scale (``1.50``, as ``format(Decimal, "f")`` spells it)
        return pc.cast(values, pa.string())
    return values


def _rendered_rows(rows: Any, kinds: Sequence[tuple[str, str]]) -> tuple[Any, str]:
    """the rows made canonical (:func:`_canonical_column`), one column a kind, named by position, cut into batches
    DuckDB scans in parallel, and the SQL of each row's text over them: its columns in ``kinds``' order, each its text
    length-prefixed (``<bytes>:<text>``), a NULL ``-`` (never how a value starts), so the text names one row only.

    :param rows: the rows, holding every column ``kinds`` names
    :ptype rows: pyarrow.Table
    :param kinds: :func:`fingerprint_kinds` of the columns
    :ptype kinds: Sequence[tuple[str, str]]
    :return: the canonical rows and the row text's SQL expression
    :rtype: tuple[pyarrow.Table, str]
    :raises ImportError: without DuckDB
    """
    if not _DUCKDB:
        raise ImportError("a fingerprint hashes rows in DuckDB: install 3tears[snapshot]")
    columns = {f"c{at}": _canonical_column(rows.column(name), kind) for at, (name, kind) in enumerate(kinds)}
    canonical = pa.table(columns) if columns else pa.table({"c": pa.nulls(rows.num_rows)})
    pieces = []
    for at, values in enumerate(columns.values()):
        text = f"c{at}" if pa.types.is_string(values.type) else f"CAST(c{at} AS VARCHAR)"
        pieces.append(f"coalesce(CAST(strlen({text}) AS VARCHAR) || ':' || {text}, '-')")
    # zero-copy slices; one batch would be one scan thread (measured: 337k rows x 15 columns, 210 ms on one thread,
    # 40 ms on ten)
    batched = pa.Table.from_batches(canonical.to_batches(max_chunksize=_SCAN_BATCH_ROWS), canonical.schema)
    return batched, " || ".join(pieces) if pieces else "''"


#: fingerprinting is background work (a copy's drift check every 15 minutes, a fold's part write): one DuckDB thread,
#: never every core the pod sees beside its serving process. Measured on a 337k x 15-column part: 268-274 ms at one
#: thread against 63-67 ms at the default (all cores), well off the event loop either way (2026-11-24)
_FINGERPRINT_THREADS: Final = 1


def row_digests(rows: Any, kinds: Sequence[tuple[str, str]]) -> Any:
    """each row's hash, in the rows' order: the MD5 (hex) of its text (:func:`_rendered_rows`), computed in DuckDB a
    column at a time, never a row at a time in Python. The one per-row rule a full's fingerprint sums and the
    writer's sample compares by. CPU: run it in a worker process (or on a handful of rows).

    :param rows: the rows, holding every column ``kinds`` names
    :ptype rows: pyarrow.Table
    :param kinds: :func:`fingerprint_kinds` of the rows' columns
    :ptype kinds: Sequence[tuple[str, str]]
    :return: each row's hash
    :rtype: pyarrow.ChunkedArray
    """
    canonical, text = _rendered_rows(rows, kinds)
    with duckdb.connect(config={"threads": _FINGERPRINT_THREADS}) as con:
        # a projection over one Arrow scan keeps the rows' order (DuckDB's preserve_insertion_order, on by default)
        return con.from_arrow(canonical).select(f"md5({text}) AS digest").to_arrow_table().column("digest")


def rows_fingerprint(rows: Any, kinds: Sequence[tuple[str, str]] | None = None) -> str:
    """the fingerprint of some rows (module docstring, "Fingerprinted"): ``f2:<columns>-<rows>-<sums>``, the format,
    the columns' and kinds' hash (8 hex digits), the row count, and each row's 128-bit MD5 (:func:`row_digests`) split
    in two 64-bit halves, each half summed over the rows modulo 2**64 (16 hex digits each), in DuckDB. The order the
    rows are held in never changes it; a row held twice, or a column more, does. Summed, not XORed: XOR cancels a row
    held twice, so ``{a, a, b}`` and ``{b, c, c}`` would agree; a sum moves by every row's hash however often it is
    held, and a changed row moves both halves by independent amounts (a miss is a 2**-128 chance). CPU: run it in a
    worker process.

    :param rows: the rows
    :ptype rows: pyarrow.Table
    :param kinds: :func:`fingerprint_kinds` of the columns to fingerprint; None takes the rows' own columns and types
    :ptype kinds: Sequence[tuple[str, str]] | None
    :return: the fingerprint, compared for equality only
    :rtype: str
    """
    if kinds is None:
        kinds = fingerprint_kinds({field.name: field.type for field in rows.schema})
    canonical, text = _rendered_rows(rows, kinds)
    with duckdb.connect(config={"threads": _FINGERPRINT_THREADS}) as con:
        # one MD5 a row, its two halves summed apart (UBIGINT sums are exact HUGEINTs below 2**64 rows)
        summed = (
            con.from_arrow(canonical)
            .select(f"md5_number({text}) AS h")
            .aggregate(f"count(*), sum((h >> 64)::UBIGINT), sum((h & {_HALF_MODULUS - 1}::UHUGEINT)::UBIGINT)")
            .fetchone()
        )
    assert summed is not None  # noqa: S101 -- an aggregate answers one row
    count, upper, lower = summed
    upper, lower = (upper or 0) % _HALF_MODULUS, (lower or 0) % _HALF_MODULUS
    return f"{FINGERPRINT_FORMAT}:{fingerprint_columns(kinds)}-{count}-{upper:016x}{lower:016x}"


def fingerprint_comparable(fingerprint: str | None) -> bool:
    """whether a part's fingerprint is of this format (:data:`FINGERPRINT_FORMAT`), so a copy's own can be compared
    with it; one an older writer wrote (another format, or none) is not.

    :param fingerprint: the part's fingerprint
    :ptype fingerprint: str | None
    :return: whether it is comparable
    :rtype: bool
    """
    return fingerprint is not None and fingerprint.startswith(f"{FINGERPRINT_FORMAT}:")


def fingerprint_columns(kinds: Sequence[tuple[str, str]]) -> str:
    """the hash of some columns and their kinds a fingerprint of their rows opens with: a fingerprint opening with
    another is of other columns (a copy holding fewer than the full), whose rows are not comparable.

    :param kinds: :func:`fingerprint_kinds` of the columns
    :ptype kinds: Sequence[tuple[str, str]]
    :return: 8 hex digits
    :rtype: str
    """
    named = ",".join(f"{name}:{kind}" for name, kind in kinds)
    return hashlib.blake2b(named.encode(), digest_size=4).hexdigest()


@dataclass(frozen=True, slots=True)
class _ColumnKind:
    """a column's name and type tag: all :func:`~threetears.core.collections.schema_backed.rail_arrow_types` reads."""

    name: str
    column_type: str


def _columns_hash(kinds: Sequence[tuple[str, str]]) -> str:
    """the column hash (:func:`fingerprint_columns`) a part of a table made over these columns, typed as the L3 rail
    answers them, is fingerprinted with: what a full's part is compared with to tell a table whose columns changed in
    place (a newer table list adds them, with no reset and no recreate) from one at its schema (decision 12).

    :param kinds: every column of the table and its type tag, ``(column, column_type)``
    :ptype kinds: Sequence[tuple[str, str]]
    :return: 8 hex digits
    :rtype: str
    :raises ValueError: for a column the rail does not answer as one Arrow type
    """
    return fingerprint_columns(fingerprint_kinds(rail_arrow_types([_ColumnKind(name, kind) for name, kind in kinds])))


def other_columns_parts(entry: FullTable, kinds: Sequence[tuple[str, str]]) -> tuple[FullPart, ...]:
    """the parts of a table's full made over other columns than ``kinds``: by each part's fingerprint, whose column
    hash names the columns its rows were written with (decision 32). A part with no fingerprint of this format (an
    older writer's) cannot tell, and is not named.

    :param entry: the table in the full
    :ptype entry: FullTable
    :param kinds: every column of the table now and its type tag, ``(column, column_type)``
    :ptype kinds: Sequence[tuple[str, str]]
    :return: the parts over other columns; empty when every comparable part is over these
    :rtype: tuple[FullPart, ...]
    """
    opening = f"{FINGERPRINT_FORMAT}:{_columns_hash(kinds)}-"
    return tuple(
        part
        for part in entry.parts
        if part.fingerprint is not None
        and fingerprint_comparable(part.fingerprint)
        and not part.fingerprint.startswith(opening)
    )


def fingerprint_ipc(ipc: bytes, kinds: tuple[tuple[str, str], ...]) -> str:
    """:func:`rows_fingerprint` of rows handed over as an Arrow IPC stream: the worker-process job a copy runs it as
    (plain data in and out, :mod:`~threetears.core.process_workers`).

    :param ipc: the rows, an Arrow IPC stream
    :ptype ipc: bytes
    :param kinds: :func:`fingerprint_kinds` of the columns
    :ptype kinds: tuple[tuple[str, str], ...]
    :return: the fingerprint
    :rtype: str
    """
    return rows_fingerprint(pa_ipc.open_stream(ipc).read_all(), kinds)


@dataclass(frozen=True, slots=True)
class FullPart:
    """one (table, state) part of a full.

    :ivar scope: the state, as the writer's seqlock names it (``state:TX``); None for a table with no scope
    :ivar file: the file, relative to the delta files' root: a part's own (``.fulls/parts/<table>/<id>.arrow``) or a
        delta file named as the part (``<table>/<write>.arrow``)
    :ivar version: the table's generation token the part is current to: deltas of this version or older are in it
    :ivar rows: the rows it holds (a delta named as a part: its rows written, delete markers not counted)
    :ivar fingerprint: :func:`rows_fingerprint` of its rows, the table's columns (decision 32); None for a part an
        older writer wrote: a copy cannot compare it. A part carried to a new version unchanged keeps its own
    """

    scope: str | None
    file: str
    version: str
    rows: int
    fingerprint: str | None = None

    def encode(self) -> dict[str, Any]:
        """the part as the manifest's JSON holds it.

        :return: the body
        :rtype: dict[str, Any]
        """
        body: dict[str, Any] = {"scope": self.scope, "file": self.file, "version": self.version, "rows": self.rows}
        if self.fingerprint is not None:
            # left out when unknown: a manifest an older reader takes holds only the fields it knows
            body["fingerprint"] = self.fingerprint
        return body

    @classmethod
    def decode(cls, body: Mapping[str, Any]) -> FullPart:
        """a part from the manifest's JSON, its file checked to stay under the root.

        :param body: the body :meth:`encode` wrote
        :ptype body: Mapping[str, Any]
        :return: the part
        :rtype: FullPart
        """
        scope = body["scope"]
        fingerprint = body.get("fingerprint")
        return cls(
            scope=None if scope is None else str(scope),
            file=_relative_file(str(body["file"])),
            version=str(body["version"]),
            rows=int(body["rows"]),
            fingerprint=None if fingerprint is None else str(fingerprint),
        )


@dataclass(frozen=True, slots=True)
class FullTable:
    """one table in a full.

    :ivar version: the generation token the table as a whole is current to: a state with no part holds no rows at it
    :ivar parts: its parts, one a state
    """

    version: str
    parts: tuple[FullPart, ...] = ()

    def encode(self) -> dict[str, Any]:
        """the table as the manifest's JSON holds it.

        :return: the body
        :rtype: dict[str, Any]
        """
        return {"version": self.version, "parts": [part.encode() for part in self.parts]}

    @classmethod
    def decode(cls, body: Mapping[str, Any]) -> FullTable:
        """a table from the manifest's JSON.

        :param body: the body :meth:`encode` wrote
        :ptype body: Mapping[str, Any]
        :return: the table
        :rtype: FullTable
        """
        return cls(version=str(body["version"]), parts=tuple(FullPart.decode(part) for part in body["parts"]))


@dataclass(frozen=True, slots=True)
class FullManifest:
    """a published full: its term and sequence, the manifest it replaced, and each table's parts.

    :ivar term: the writer's term
    :ivar seq: its sequence within the term, from 1
    :ivar previous: the name of the manifest that was current when this one was published (kept beside it, since a
        reader may still be loading it); None for the first
    :ivar tables: each table's entry
    """

    term: int
    seq: int
    previous: str | None
    tables: Mapping[str, FullTable]

    @property
    def name(self) -> str:
        """the manifest's file name: its term, then its sequence.

        :return: the name
        :rtype: str
        """
        return _manifest_name(self.term, self.seq)

    def files(self) -> set[str]:
        """every file it names, relative to the root.

        :return: the files
        :rtype: set[str]
        """
        return {part.file for entry in self.tables.values() for part in entry.parts}

    def encode(self) -> bytes:
        """the manifest as the JSON bytes its file holds.

        :return: the bytes
        :rtype: bytes
        """
        body = {
            "format": _FORMAT,
            "term": self.term,
            "seq": self.seq,
            "previous": self.previous,
            "tables": {name: entry.encode() for name, entry in sorted(self.tables.items())},
        }
        return json.dumps(body, sort_keys=True).encode()

    @classmethod
    def decode(cls, raw: bytes) -> FullManifest:
        """a manifest read back.

        :raises ValueError: when it is not a manifest of this format
        :raises KeyError: when a field is missing
        """
        body = json.loads(raw)
        if body.get("format") != _FORMAT:
            raise ValueError(f"a full manifest of format {body.get('format')!r}, not {_FORMAT}")
        previous = body.get("previous")
        return cls(
            term=int(body["term"]),
            seq=int(body["seq"]),
            previous=None if previous is None else str(previous),
            tables={str(name): FullTable.decode(entry) for name, entry in body["tables"].items()},
        )


def _manifest_name(term: int, seq: int) -> str:
    return f"full-{term:020d}-{seq:020d}.json"


def _relative_file(file: str) -> str:
    """a manifest's file name, checked to stay under the root.

    :raises ValueError: when it is absolute or climbs out
    """
    path = PurePosixPath(file)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"a full names a file outside its root: {file!r}")
    return file


def _manifests_dir(root: Path) -> Path:
    return root / _FULLS / _MANIFESTS


def _fsync_directory(directory: Path) -> None:
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _write_renamed(final: Path, write: Any) -> None:
    """write ``final`` under a temporary name beside it, flush it, rename it into place and flush the directory, so a
    reader that opens the name sees the whole file or none. Refuses a name already taken: nothing is overwritten.

    :param final: the file
    :ptype final: Path
    :param write: writes the file's bytes to the open sink
    :ptype write: Callable[[BinaryIO], None]
    :raises FileExistsError: when ``final`` exists
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"{final} exists; a full's files are never overwritten")
    partial = final.parent / f".{final.name}.{uuid.uuid7().hex}{_PARTIAL_SUFFIX}"
    renamed = False
    try:
        with partial.open("wb") as sink:
            write(sink)
            sink.flush()
            os.fsync(sink.fileno())
        os.rename(partial, final)
        renamed = True
    finally:
        if not renamed:
            partial.unlink(missing_ok=True)
    _fsync_directory(final.parent)


def write_full_part(root: Path, table: str, scope: str | None, rows: Any, version: str) -> FullPart:
    """save one (table, state) part of a full under ``root``: Arrow IPC with zstd inside, under a new name, written
    and renamed into place. Blocking: run it on a thread.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the table
    :ptype table: str
    :param scope: the state; None for a table with no scope
    :ptype scope: str | None
    :param rows: the part's rows, the table's columns
    :ptype rows: pyarrow.Table
    :param version: the generation token the rows are current to
    :ptype version: str
    :return: the part, to name in a manifest
    :rtype: FullPart
    :raises ValueError: when the table is not an identifier
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a full needs pyarrow: install 3tears[snapshot]")
    if not _TABLE_NAME.match(table):
        raise ValueError(f"a full's table must be an identifier, got {table!r}")
    file = f"{_FULLS}/{_PARTS}/{table}/{uuid.uuid7().hex}{_SUFFIX}"
    metadata = {_TABLE_METADATA: table.encode(), _VERSION_METADATA: version.encode()}
    if scope is not None:
        metadata[_SCOPE_METADATA] = scope.encode()
    held = rows.replace_schema_metadata(metadata)

    # closed by its checksum, as a delta file is: a damaged part is refused, never staged as other rows
    _write_renamed(root / file, lambda sink: write_volume_file(sink, held))
    # its rows' fingerprint, taken here where they are written (the fold's worker process): what a copy compares its
    # own rows of the state with (decision 32)
    return FullPart(scope=scope, file=file, version=version, rows=held.num_rows, fingerprint=rows_fingerprint(rows))


def read_full_part(root: Path, part: FullPart) -> Any:
    """a part's rows off the volume, its checksum verified first: a delta file named as the part read without its
    delete markers. Blocking: run it on a thread.

    :param root: the delta files' root
    :ptype root: Path
    :param part: the part
    :ptype part: FullPart
    :return: its rows, every column its file holds but the markers' flag
    :rtype: pyarrow.Table
    :raises DeltaFileUnreadableError: when the file is damaged (never read as other rows)
    :raises FileNotFoundError: when it is not on the volume
    """
    rows = read_volume_file(root / part.file)
    if _DELETED_COLUMN in rows.column_names:
        # a delta named as the part: its rows written, never its delete markers
        rows = rows.filter(pc.invert(rows.column(_DELETED_COLUMN))).drop_columns([_DELETED_COLUMN])
    return rows


def _listed_manifests(root: Path) -> list[tuple[int, int, str]]:
    """every published manifest under ``root``, as (term, seq, name), oldest first; a temporary one is not listed."""
    directory = _manifests_dir(root)
    if not directory.is_dir():
        return []
    listed = []
    for path in directory.iterdir():
        matched = _MANIFEST_NAME.match(path.name)
        if matched is not None:
            listed.append((int(matched.group(1)), int(matched.group(2)), path.name))
    return sorted(listed)


def _read_manifest(root: Path, name: str) -> FullManifest:
    return FullManifest.decode((_manifests_dir(root) / name).read_bytes())


def current_full_manifest(root: Path) -> FullManifest | None:
    """the manifest readers take: the highest term, then the newest sequence in it; None when none is published.
    Blocking: run it on a thread.

    :param root: the delta files' root
    :ptype root: Path
    :return: the manifest
    :rtype: FullManifest | None
    :raises ValueError: when the current manifest does not read back (never skipped for an older one: an older term's
        may be a stale writer's)
    """
    listed = _listed_manifests(root)
    if not listed:
        return None
    term, seq, name = listed[-1]
    manifest = _read_manifest(root, name)
    if (manifest.term, manifest.seq) != (term, seq):
        raise ValueError(f"full manifest {name} says term {manifest.term} seq {manifest.seq}")
    return manifest


def publish_full_manifest(root: Path, *, term: int, tables: Mapping[str, FullTable]) -> FullManifest:
    """publish a full: a manifest of ``tables`` at ``term``, the next sequence in it, naming the current manifest as
    its previous; written under a temporary name, flushed, renamed into place. Blocking: run it on a thread.

    Every part it names must already be written (:func:`write_full_part`, or a delta file). Only the writer holding the
    write lease publishes, under that lease's term.

    :param root: the delta files' root
    :ptype root: Path
    :param term: the writer's term, from its write lease; only ever higher for a later holder
    :ptype term: int
    :param tables: every table the full holds, each with every part (those it keeps of the previous full, and new ones)
    :ptype tables: Mapping[str, FullTable]
    :return: the manifest published
    :rtype: FullManifest
    :raises StaleTermError: when the current manifest's term is higher: another writer took the lease since
    :raises FileExistsError: when the name is taken (two writers of one term: the lease was not held)
    """
    if term < 0:
        raise ValueError(f"a term is not negative, got {term}")
    for name in tables:
        if not _TABLE_NAME.match(name):
            raise ValueError(f"a full's table must be an identifier, got {name!r}")
    listed = _listed_manifests(root)
    if listed and listed[-1][0] > term:
        raise StaleTermError(
            f"the current full is of term {listed[-1][0]}, above this writer's {term}: another writer took the lease"
        )
    seq = max((s for t, s, _ in listed if t == term), default=0) + 1
    manifest = FullManifest(term=term, seq=seq, previous=listed[-1][2] if listed else None, tables=dict(tables))
    for file in manifest.files():
        _relative_file(file)
    raw = manifest.encode()
    _write_renamed(_manifests_dir(root) / manifest.name, lambda sink: sink.write(raw))
    log.info(
        "full published",
        extra={
            "extra_data": {
                "manifest": manifest.name,
                "previous": manifest.previous,
                "tables": len(manifest.tables),
                "parts": len(manifest.files()),
            }
        },
    )
    return manifest


def deletable_full_files(root: Path, *, covered_deltas: Collection[str] = ()) -> list[str]:
    """the files of the fulls the writer may delete: every manifest but the current one and its previous, every
    part neither names, every temporary file left by a write that died, and each delta in ``covered_deltas`` neither
    names (a delta named as a part is kept, decision 16). Blocking: run it on a thread.

    Call it only from the writer, after its own publish and before it writes the next full's parts: a part written for
    a manifest not yet published is named by none. With no manifest published nothing is listed (a rebuild's parts
    are written before its first manifest).

    :param root: the delta files' root
    :ptype root: Path
    :param covered_deltas: delta files (``<table>/<write>.arrow``) the current full covers, to delete too; empty from a
        fold that keeps its delta files (decision 51), so then no delta file is listed
    :ptype covered_deltas: Collection[str]
    :return: the files, relative to ``root``
    :rtype: list[str]
    """
    current = current_full_manifest(root)
    if current is None:
        return []
    kept_manifests = {current.name}
    kept_files = current.files()
    if current.previous is not None and (_manifests_dir(root) / current.previous).is_file():
        kept_manifests.add(current.previous)
        kept_files |= _read_manifest(root, current.previous).files()
    deletable: list[str] = []
    manifests = _manifests_dir(root)
    for path in sorted(manifests.iterdir()):
        if path.is_file() and path.name not in kept_manifests:
            deletable.append(f"{_FULLS}/{_MANIFESTS}/{path.name}")
    parts = root / _FULLS / _PARTS
    if parts.is_dir():
        for directory in sorted(p for p in parts.iterdir() if p.is_dir()):
            for path in sorted(directory.iterdir()):
                file = f"{_FULLS}/{_PARTS}/{directory.name}/{path.name}"
                if path.is_file() and file not in kept_files:
                    deletable.append(file)
    deletable.extend(sorted(file for file in covered_deltas if file not in kept_files))
    return deletable


def _versions_floor(table: str, versions: Iterable[str]) -> str:
    """the oldest of a table's versions: where a copy of every part of it catches up from.

    :raises ValueError: when a version is not ``{incarnation}:{count}``, or two are of different incarnations (a table
        reset or rebuilt starts fresh: every part is of the new one)
    """
    split = []
    for version in versions:
        parsed = split_generation_token(version)
        if parsed is None:
            raise ValueError(f"{table}: a full's version must be a generation token, got {version!r}")
        split.append((parsed, version))
    incarnations = {incarnation for (incarnation, _), _ in split}
    if len(incarnations) != 1:
        raise ValueError(f"{table}: a full's parts are of incarnations {sorted(incarnations)}, not one")
    return min(split)[1]


def full_table_floor(table: str, entry: FullTable) -> str:
    """the oldest version of a table in a full, of the table and its parts: where a copy loading it catches up from,
    and how a copy waiting after a reset judges whether the full holds the table at or past it.

    :param table: the table, for the error
    :ptype table: str
    :param entry: its entry in the manifest
    :ptype entry: FullTable
    :return: the generation token
    :rtype: str
    :raises ValueError: when a version is not a generation token, or two are of different incarnations
    """
    return _versions_floor(table, [entry.version, *(part.version for part in entry.parts)])


@dataclass(frozen=True, slots=True)
class FullAbsent:
    """no full a copy can load yet: it waits for the writer to publish one, never reading L3 whole.

    :ivar reason: why, for the follower's status
    :ivar lacking: the tables no full holds (no manifest at all: every table), or holds over other columns than their
        schema's: the ones the copy asks the writer for (``CopyFollower``'s wait, ``DeltaRecreateRequester.full``), so
        a writer whose fold never started (no write begun) still rebuilds them. Empty when a part of the full is only
        missing for now (a newer full swept it): nothing to ask for, the next look finds the newer full
    """

    reason: str
    lacking: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StagedFull:
    """what :meth:`VolumeFulls.stage` staged beside the copy from the current full.

    :ivar manifest: the manifest's name
    :ivar versions: each table's version per state: the part's, or the table's for a state with no part (key None
        for a table with no scope, or a table with no part). A delta at or below a state's version is in the copy
    :ivar marks: each table's oldest version, of the table and its parts: where the copy catches up from
    :ivar changes: each table's stage, whole, to apply in one transaction
    :ivar rows: the rows staged
    """

    manifest: str
    versions: Mapping[str, Mapping[str | None, str]]
    marks: Mapping[str, str]
    changes: list[StagedChange]
    rows: int


class VolumeFulls:
    """a copy follower's start from the full on the delta-files volume, for the tables that opt in.

    :param root: the delta files' root (``delta_files_root()``)
    :ptype root: Path
    :param tables: the copy's tables that opt in to delta files; every other table starts as today
    :ptype tables: Sequence[FollowedTable]
    :param poll_seconds: how often a copy with no full to load looks for one again
    :ptype poll_seconds: float
    """

    def __init__(self, root: Path, tables: Sequence[FollowedTable], *, poll_seconds: float = _DEFAULT_POLL_SECONDS):
        if not _PYARROW:
            raise ImportError("a full needs pyarrow: install 3tears[snapshot]")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.root = root
        self.tables: Final = {table.name: table for table in tables}
        self.poll_seconds = poll_seconds

    async def stage(self, backend: FollowerBackend, *, manifest: FullManifest | None = None) -> StagedFull | FullAbsent:
        """stage each table's parts of the current full beside the copy, a part at a time; :class:`FullAbsent` when no
        full is published, or the current one does not hold every table.

        :param backend: the copy
        :ptype backend: FollowerBackend
        :param manifest: the manifest to load, read already; None reads the current one
        :ptype manifest: FullManifest | None
        :return: what was staged, or why nothing was
        :rtype: StagedFull | FullAbsent
        :raises FileNotFoundError: when a part the manifest names is not on the volume
        """
        if manifest is None:
            manifest = await asyncio.to_thread(current_full_manifest, self.root)
        if manifest is None:
            return FullAbsent("no full on the volume yet; the writer publishes one", lacking=tuple(sorted(self.tables)))
        missing = sorted(set(self.tables) - set(manifest.tables))
        if missing:
            return FullAbsent(
                f"full {manifest.name} does not hold {missing} yet; the writer publishes one that does",
                lacking=tuple(missing),
            )
        for name, table in self.tables.items():
            # a table whose columns changed in place (decision 12): its parts hold the old columns, so the copy waits
            # for the writer's fresh full, as for a missing one, never reading L3
            kinds = table.l3_kinds or tuple((column.name, column.column_type) for column in table.columns)
            other = other_columns_parts(manifest.tables[name], kinds)
            if other:
                _OTHER_COLUMNS.labels(table=name).inc()
                return FullAbsent(
                    f"full {manifest.name} holds {name} over other columns than its schema ({len(other)} parts, "
                    f"{other[0].fingerprint}); the writer starts it fresh and publishes one over its columns",
                    lacking=(name,),
                )
        versions: dict[str, dict[str | None, str]] = {}
        marks: dict[str, str] = {}
        changes: list[StagedChange] = []
        rows = 0
        for name, table in self.tables.items():
            entry = manifest.tables[name]
            marks[name] = full_table_floor(name, entry)
            held: dict[str | None, str] = {None: entry.version}
            held.update((part.scope, part.version) for part in entry.parts)
            versions[name] = held
            stage = f"{STAGE_PREFIX}{name}__rows"
            await asyncio.to_thread(backend.drop_stage, stage)
            try:
                await asyncio.to_thread(backend.stage_rows, stage, name, pa.table({}), table.key)
                for part in entry.parts:
                    rows += await asyncio.to_thread(self._stage_part, backend, stage, table, part)
            except BaseException:
                await asyncio.to_thread(backend.drop_stage, *[c.rows for c in changes if c.rows], stage)
                raise
            # a start from the full fills an empty copy: its apply empties the table as a cold start, as a checkpoint's
            changes.append(StagedChange(table=name, key=table.key, rows=stage, whole=True, whole_kind="cold_start"))
        log.info(
            "full staged from the volume",
            extra={"extra_data": {"manifest": manifest.name, "tables": len(changes), "rows": rows}},
        )
        return StagedFull(manifest=manifest.name, versions=versions, marks=marks, changes=changes, rows=rows)

    def _stage_part(self, backend: FollowerBackend, stage: str, table: FollowedTable, part: FullPart) -> int:
        """one part read off the volume and staged, on the caller's worker thread.

        :return: the rows staged
        :raises ValueError: when the part holds other than the rows its manifest says
        """
        # its checksum verified first: a damaged part raises DeltaFileUnreadableError, never stages other rows
        began = time.monotonic()
        outcome = "ok"
        try:
            rows = read_full_part(self.root, part).select([column.name for column in table.columns])
            if rows.num_rows != part.rows:
                outcome = "rows_differ"
                raise ValueError(f"part {part.file} holds {rows.num_rows} rows, its manifest says {part.rows}")
        except DeltaFileUnreadableError:
            outcome = "unreadable"
            raise
        except FileNotFoundError:
            outcome = "missing"
            raise
        finally:
            _PART_CHECKS.labels(outcome=outcome).inc()
            _PART_CHECK_SECONDS.labels(outcome=outcome).observe(time.monotonic() - began)
        return int(backend.stage_rows(stage, table.name, rows, table.key)) if rows.num_rows else 0
