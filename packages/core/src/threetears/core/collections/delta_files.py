"""a delta file: the rows one commit wrote to a table and the keys it deleted, saved before the commit.

**Why.** Election night failed because L3 was on the hot path: every pod copy read each change's rows back from L3,
and a restarted copy whose checkpoint was older than the change notes could never learn which rows changed. A table
that opts in (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_files`) has every commit's rows
saved as a file on a volume every pod mounts, so a copy applies the change from the file instead of reading L3, and
nothing expires on a timer (``design-pod-delta-files.md``, decisions 5, 6, 9, 11 and 12).

**One file per table per commit.** A commit is one epoch of each table it wrote (the broker advances each table once
for it), so the file holds every row the commit wrote to the table, in the order written, and a delete marker
(:data:`_DELETED_COLUMN` true, the key columns set, every other column null but, for a table keyed without its
state, the state the delete named in its ``delta_scope`` column, decision 27) for each key it deleted. The columns are
the table's own, typed as the L3 rail answers them (:func:`~threetears.core.collections.schema_backed.rail_arrow_types`):
the same table a copy reads from L3 today, so applying a file and reading the keys from L3 stage the same rows. The
schema's metadata names the table, the write (a unique id: the commit's version is the broker's, known only after the
commit, so the epoch record names the file rather than the file the version) and the key columns.

**Arrow IPC with zstd inside** (decision 18). An open Apache standard the pod's Arrow copy loads without decoding,
each record batch's buffers compressed with zstd, which keeps a file close to Parquet's size on the volume. Plain IPC
was only ever needed for the browser's per-report files, which these are not.

**Which columns each write set** (decision 19). L3 merges an upsert of a row it already holds: the ``DO UPDATE``
sets only the columns its statement names (:func:`~threetears.core.backends.schema_sql.upsert_set_columns`), so a
column the write left out keeps its held value, and an immutable column (``date_created``) and the key always keep
theirs. The file shows such a column as null (a new row is inserted so), so it also says, for each row, which columns
its write set: :data:`_WRITTEN_COLUMN` is an index into the distinct column sets the file's metadata lists
(:data:`_WRITTEN_METADATA`), one per kind of write. A bulk write's rows share one set and a single row's names its
own, so a file holds a handful of sets however many rows, and an index a row that zstd packs to almost nothing: a
bitmask or a list a row would cost bytes a row (an LSF table has 179 columns) for what a handful of sets says.
Applied as L3 merged each row (:class:`HeldDeltaRows`, :func:`fold_delta_tables`; the rule a row at a time is
:func:`~threetears.core.testing.delta_files.apply_delta_row`, the tests' reference), a copy or a fold ends exactly as L3
does.

**A fixed default is written in; anything the database decides is refused** (decision 20). A column with a
``server_default`` that a write leaves out is left out of its ``INSERT``, so a new row takes the default. When the
default is a fixed value (``'active'``, ``0``, ``false``), the writer puts that value in the file's row
(:attr:`TableDelta.defaults`), as the database fills it; it is not among the columns the write set, so a row L3
already held keeps its value, as the ``DO UPDATE`` that never names the column keeps it. The file never says whether a
row was new: the copy and the fold tell, by whether they hold the key, which is how L3 told (the conflict). A default
the database computes (``now()``, a sequence) or a guarded write (one the database may skip) cannot be known before
the commit, so such a table is refused where it opts in
(:class:`~threetears.core.collections.schema_backed.TableSchema`).

**A delta can name the key slices it clears** (decision 23). A bulk file load (LSF's, the hub's ``load_files``)
replaces every row under a key prefix, and a delete of gone runs (``delete_prefixes``) removes every row under one;
L3 deletes those rows by prefix, and the writer cannot list them without reading L3. So the file names each prefix
(:attr:`TableDelta.prefixes`, the file's :data:`_PREFIXES_METADATA`): the leading key columns and their values as
text (:func:`_prefix_text`). A copy and the fold apply such a file as L3 took it: every row they hold under each prefix
leaves first, then the file's rows and delete markers are applied. Same result as L3, no L3 read, no delete marker a
row.

**Saved before the commit, under a unique name, atomically.** The file is written under a temporary name in its
table's directory, flushed to disk, then renamed to its own name, and the directory flushed: a reader that opens the
name sees the whole file or nothing, never part of one. A crash before the commit leaves a file no committed record
names (an orphan, which :func:`sweep_orphan_delta_files` removes); a crash after it leaves the file in place, so every
committed version has its file.

**A checksum closes every file** (decision 29 depends on it). The IPC bytes are followed by their sha256 and a mark
(:data:`_CHECKSUM_MARK`); every reader of the volume's files (this module's, the fold's, a full's parts) verifies it
before Arrow decodes a byte (:func:`_checked_ipc`), and a file whose bytes do not match is
:class:`DeltaFileUnreadableError`, which a copy waits on and the writer recreates. Why: without it about two in five
single-byte flips still read as a valid file holding other values (measured 2026-11-23, a 2,000-row file, a flip every
23 bytes), which a copy would apply as the change. The hash costs about 4 ms for a 12.7 MB file (3 GB/s, measured
2026-11-23 on the builder's machine), small beside the zstd decode that follows.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from itertools import chain
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Final

from threetears.core.backends.arrow_reply import _rail_value
from threetears.core.collections.epoch_change import key_text
from threetears.core.frozen_state import frozen, with_chunk, with_entry
from threetears.observe import get_logger

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as pa_ipc
    import pyarrow.parquet as pa_parquet

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a delta file refuses without it, saying so
    _PYARROW = False

__all__ = [
    "DeltaChange",
    "DeltaFileRefusedError",
    "DeltaFileUnreadableError",
    "DeltaKeys",
    "DeltaTable",
    "HeldDeltaRows",
    "LoadDelta",
    "OWN_DELTA_SAVES",
    "TableDelta",
    "build_file_table",
    "check_delta_scope",
    "delta_files_root",
    "fold_delta_tables",
    "joined_key_texts",
    "key_text_table",
    "read_delta_file_keys",
    "read_delta_table",
    "read_volume_file",
    "text_key_batch",
    "sweep_orphan_delta_files",
    "under_prefix",
    "write_delta_file",
    "write_table_file",
    "write_volume_file",
]

log = get_logger(__name__)

#: names the directory delta files go in: the shared volume's mount path in a pod. Unset, no delta file can be saved,
#: so a write to a table that opts in is refused rather than committed without its file
_DELTA_FILES_DIR_ENV: Final = "THREETEARS_DELTA_FILES_DIR"

#: the column that marks a row as a deleted key: true for a delete marker, false for a row written
_DELETED_COLUMN: Final = "_3tears_deleted"

#: the column that names which columns a row's write set: an index into the sets :data:`_WRITTEN_METADATA` lists
_WRITTEN_COLUMN: Final = "_3tears_written"

#: the file's schema metadata: the table it holds rows of
_TABLE_METADATA: Final = b"threetears.delta.table"
#: the file's schema metadata: the write that saved it, its unique id (the file's name without its suffix)
_WRITE_METADATA: Final = b"threetears.delta.write"
#: the file's schema metadata: the table's key columns, a JSON list in key order
_KEY_METADATA: Final = b"threetears.delta.key"
#: the file's schema metadata: the distinct sets of columns its writes set, a JSON list of lists of column names,
#: each in the table's column order; :data:`_WRITTEN_COLUMN` indexes it
_WRITTEN_METADATA: Final = b"threetears.delta.written"
#: the file's schema metadata: the key prefixes it clears before its rows apply, a JSON list of objects, each the
#: leading key columns and their values as text (:func:`_prefix_text`); an empty list when it clears none
_PREFIXES_METADATA: Final = b"threetears.delta.prefixes"
#: the file's schema metadata: the format's version
_FORMAT_METADATA: Final = b"threetears.delta.format"
#: the file's schema metadata, on a file the writer wrote again from L3 for a failed load request (ruling C of
#: 2026-11-23): its rows are exactly what L3 holds for the request's keys, so a copy may apply it although its record
#: names it unconfirmed (``EpochChange.delta_file_unconfirmed``). Absent from every other file; a reader built before it
#: ignores it, and the format's version stays: an older reader never reads an unconfirmed record's file at all
_CONFIRMED_METADATA: Final = b"threetears.delta.confirmed"
#: the value of :data:`_CONFIRMED_METADATA`: where the rows were read from
_CONFIRMED_FROM_L3: Final = b"l3"
#: this format's version: rows and delete markers in one table, as :data:`_DELETED_COLUMN` tells them apart, each
#: row naming the columns its write set (:data:`_WRITTEN_COLUMN`), the key prefixes it clears first
#: (:data:`_PREFIXES_METADATA`), zstd inside, the IPC bytes closed by their checksum (:data:`_CHECKSUM_MARK`).
#: Version 1 was plain IPC with no written columns, version 2 named no prefixes, version 3 carried no checksum; none was
#: released
_DELTA_FORMAT: Final = b"4"
#: the codec inside the file's IPC buffers
_COMPRESSION: Final = "zstd"

#: the mark that ends every file on the volume, after the sha256 of the IPC bytes before it (:func:`write_volume_file`)
_CHECKSUM_MARK: Final = b"3tsha256"
#: the trailer's length: the digest, then the mark
_CHECKSUM_TRAILER: Final = 32 + len(_CHECKSUM_MARK)

#: a delta file's suffix
_SUFFIX: Final = ".arrow"
#: the suffix of a file being written, before its rename; never a name a record holds
_PARTIAL_SUFFIX: Final = ".partial"
#: a table name as it may become a directory: an identifier, so no name reaches outside the root
_TABLE_NAME: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: a write id as it may become a file name
_WRITE_ID: Final = re.compile(r"^[A-Za-z0-9_-]+$")


class DeltaFileRefusedError(RuntimeError):
    """a write to a table that opts in to delta files would commit without its file, so it is refused.

    Raised before the commit: the transaction rolls back, and nothing of the write reaches L3.
    """


def check_delta_scope(table: str, scope_column: str | None, scope: Any) -> None:
    """refuse a delete whose markers could not name their state: a table whose key does not name a row's state
    (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_scope`) deletes naming the state, and a table
    whose key names it names none (decision 27).

    :param table: the table
    :ptype table: str
    :param scope_column: the row column its markers carry the state in; None when its key names it
    :ptype scope_column: str | None
    :param scope: the state the delete named; None when it named none
    :ptype scope: Any
    :return: nothing
    :rtype: None
    :raises DeltaFileRefusedError: when the markers carry a state and none was named
    :raises ValueError: when a state was named for a table whose markers carry none
    """
    if scope_column is not None and scope is None:
        raise DeltaFileRefusedError(
            f"{table}: its key does not name a row's state, so each delete names the state of the keys it deletes "
            f"(delta_scope=, the value of {scope_column!r} in their rows): its delete markers carry it, and the fold "
            f"files each under that state's part (design-pod-delta-files.md, decision 27)"
        )
    if scope_column is None and scope is not None:
        raise ValueError(
            f"{table}: a delete named the state {scope!r}, but the table declares no delta_scope (its key names the "
            f"state, or it does not opt in to delta files): no marker would carry it"
        )


class DeltaFileUnreadableError(ValueError):
    """a delta file that is on the volume under its name but cannot be applied as that file: truncated, garbage, a
    format this reader does not know, a damaged zstd buffer, a broken footer or array, or another table's or another
    write's file under the name. Never raised for a file that is not there (``FileNotFoundError``), nor for an error the
    volume itself answered (an ``OSError`` carrying an errno): those are not the file's bytes.

    Why one class: a copy that finds a change's file so waits for the writer to recreate it, reading nothing from L3
    (``design-pod-delta-files.md``, decision 29), and it must tell that apart precisely from a failure it should not
    wait on. Measured (2026-11-23, a 2,000-row file, a byte flipped every 23 bytes, one process a flip): without a full
    validation 15 of 509 reads killed the process (a segfault in ``to_pylist`` over a broken offset), the rest raised
    ``MemoryError``, ``UnicodeDecodeError``, ``KeyError`` or ``OSError``; with ``validate(full=True)`` before any row
    is decoded none crashed, and every failure was ``pyarrow.ArrowInvalid`` or an errno-less ``OSError`` (zstd, the
    footer's flatbuffer, a read past the end). About two in five flips still read as a file holding other values,
    so format 4 closes each file with a checksum of its bytes (:func:`_checked_ipc`), verified before Arrow decodes
    one: a flipped byte anywhere is this error.
    """


def _prefix_text(prefix: Mapping[str, Any]) -> dict[str, str]:
    """a key prefix as a delta file names it: each leading key column's value as a change record names a key's
    (:func:`~threetears.core.collections.epoch_change.key_text`), so a key a copy holds as text matches it as is.

    :param prefix: the leading key columns and their values
    :ptype prefix: Mapping[str, Any]
    :return: the prefix, its values as text
    :rtype: dict[str, str]
    """
    return {column: key_text(value) for column, value in prefix.items()}


def under_prefix(row: Mapping[str, Any], prefixes: Sequence[Mapping[str, str]]) -> bool:
    """whether a row (or a key, keyed by column) lies under one of ``prefixes``: each of a prefix's columns holds its
    value, compared as text (:func:`_prefix_text`).

    :param row: the row's columns, its key columns among them; values typed or as text
    :ptype row: Mapping[str, Any]
    :param prefixes: the prefixes, as a delta file names them
    :ptype prefixes: Sequence[Mapping[str, str]]
    :return: whether it does
    :rtype: bool
    """
    return any(all(key_text(row[column]) == value for column, value in prefix.items()) for prefix in prefixes)


def delta_files_root() -> Path | None:
    """the directory delta files go in: ``THREETEARS_DELTA_FILES_DIR``, or None when it is not set.

    :return: the directory, or None
    :rtype: Path | None
    """
    named = os.environ.get(_DELTA_FILES_DIR_ENV)
    return Path(named) if named else None


def _delta_file_name(table: str, write_id: str) -> str:
    """a delta file's name under the root: ``<table>/<write id>.arrow``, the name an epoch record holds.

    :param table: the table
    :ptype table: str
    :param write_id: the write's unique id
    :ptype write_id: str
    :return: the name, relative to the root
    :rtype: str
    :raises ValueError: when the table is not an identifier or the write id has a character a file name may not
    """
    if not _TABLE_NAME.match(table):
        raise ValueError(f"a delta file's table must be an identifier, got {table!r}")
    if not _WRITE_ID.match(write_id):
        raise ValueError(f"a delta file's write id must be letters, digits, '-' or '_', got {write_id!r}")
    return f"{table}/{write_id}{_SUFFIX}"


#: the columns a write set: one set for every row of a write (a bulk statement's), or one a row (single-row upserts,
#: each naming its own)
Written = Sequence[str] | Sequence[Sequence[str]]


@dataclass(slots=True)
class TableDelta:
    """what one transaction wrote to one table that opts in, in the order written, until its file is saved.

    Each part is kept as the write handed it (rows, an Arrow table, keys deleted, or a load's files' rows), with the
    columns its write set, and typed only when the file is made: dict rows into Arrow where the dicts are
    (:meth:`segments`), everything else in the worker process that builds and saves the file
    (:mod:`~threetears.core.collections.delta_save`).

    :ivar name: the table
    :ivar types: each column's Arrow type, in the table's column order (``rail_arrow_types``)
    :ivar key: the key columns, in key order
    :ivar parts: each write's part, ``("rows", rows, written)``, ``("arrow", table, written)`` or
        ``("deletes", keys, ())``; ``written`` is one set of columns for every row, or a list of one a row. An
        append-only log of immutable chunks, replaced whole (:func:`~threetears.core.frozen_state.with_chunk`): read it
        in order with ``itertools.chain.from_iterable``
    :ivar entries: rows written and keys deleted so far: one a key the transaction enrolled
    :ivar defaults: each column whose database default is a fixed value, and that value as the rail answers it
        (``TableSchema.delta_file_defaults``): a row written without the column was inserted with it, so the file's
        row holds it rather than null
    :ivar prefixes: the key prefixes the write cleared before its rows, as text (:func:`_prefix_text`), each once: a
        key-prefix delete's, or a load's that replaces them (decision 23)
    :ivar scope_column: the row column a delete marker carries its row's state in, for a table whose key does not name
        it (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_scope`, decision 27); None when the key
        names it
    """

    name: str
    types: Mapping[str, Any]
    key: tuple[str, ...]
    parts: tuple[tuple[tuple[Any, ...], ...], ...] = ()
    entries: int = 0
    defaults: Mapping[str, Any] = field(default_factory=dict)
    prefixes: tuple[dict[str, str], ...] = ()
    scope_column: str | None = None

    def clears(self, prefixes: Sequence[Mapping[str, Any]]) -> None:
        """add key prefixes the write cleared: every row under each left L3 before the rows the delta holds went in.

        A prefix cleared after rows were recorded would clear them too, which no file can say (its prefixes apply
        before its rows), so it is refused.

        :param prefixes: each prefix, the leading key columns and their values
        :ptype prefixes: Sequence[Mapping[str, Any]]
        :return: nothing
        :rtype: None
        :raises ValueError: when rows were recorded first, or a prefix names a column that does not lead the key
        """
        if self.parts and prefixes:
            raise ValueError(
                f"{self.name}: key prefixes cleared after rows were written in one delta; a delta file's prefixes "
                f"clear before its rows, so clear them first"
            )
        for prefix in prefixes:
            named = _prefix_text(prefix)
            if not named or set(named) != set(self.key[: len(named)]):
                raise ValueError(f"{self.name}: {dict(prefix)} is not a key prefix of {list(self.key)}")
            if named not in self.prefixes:
                self.prefixes = (*self.prefixes, named)

    def rows(self, rows: Sequence[Mapping[str, Any]], written: Written) -> None:
        """add rows written, each keyed by column, and the columns their write set.

        :param rows: the rows, as written
        :ptype rows: Sequence[Mapping[str, Any]]
        :param written: the columns the write set in a row L3 already held (``upsert_set_columns``): one list for
            every row, or one list a row
        :ptype written: Sequence[str] | Sequence[Sequence[str]]
        :return: nothing
        :rtype: None
        """
        if rows:
            self.parts = with_chunk(self.parts, (("rows", rows, written),))
            self.entries += len(rows)

    def arrow(self, rows: Any, written: Sequence[str]) -> None:
        """add rows written, already an Arrow table, and the columns their write set.

        :param rows: the rows, a column each
        :ptype rows: pyarrow.Table
        :param written: the columns the write set in a row L3 already held, the same for every row
        :ptype written: Sequence[str]
        :return: nothing
        :rtype: None
        """
        if rows.num_rows:
            self.parts = with_chunk(self.parts, (("arrow", rows, written),))
            self.entries += int(rows.num_rows)

    def deletes(self, keys: Sequence[Sequence[Any]] | Any, *, scope: Any = None) -> None:
        """add keys deleted, each its values in key order (or an Arrow table of the key columns, typed as the table's,
        which no key leaves), and, for a table whose key does not name a row's state, the state of them all: each
        marker carries it in :attr:`scope_column`, so the fold files the delete under that state's part (decision 27).

        :param keys: the keys
        :ptype keys: Sequence[Sequence[Any]] | pyarrow.Table
        :param scope: the keys' state, the value of :attr:`scope_column` in each of their rows; None for a table whose
            key names it
        :ptype scope: Any
        :return: nothing
        :rtype: None
        :raises DeltaFileRefusedError: when the table's markers carry a state and none is given: the marker would be
            filed under no state's part, and the deleted row would stay in the full
        :raises ValueError: when a state is given for a table whose markers carry none
        """
        if len(keys):
            check_delta_scope(self.name, self.scope_column, scope)
            self.parts = with_chunk(self.parts, (("deletes", keys, (), scope),))
            self.entries += len(keys)

    def load(self, given: Sequence[Any], counts: Sequence[int], constants: Mapping[str, Any]) -> None:
        """add one request of a load from files: each file's rows the caller gave (:attr:`LoadDelta.rows`: an Arrow
        table or record batch, or a parquet file's path), the rows each file holds, and the load's constants. A copy
        inserts and sets no column of a row already held (it fails the load instead), so its rows name no column
        written. Joined and checked against the counts where the file is made (:func:`_load_delta_rows`).

        :param given: the request's files' rows, one entry a file
        :ptype given: Sequence[Any]
        :param counts: the rows each file holds, in the same order
        :ptype counts: Sequence[int]
        :param constants: column -> the value every row takes for it
        :ptype constants: Mapping[str, Any]
        :return: nothing
        :rtype: None
        """
        if given:
            self.parts = with_chunk(self.parts, (("load", list(given), list(counts), dict(constants)),))
            self.entries += sum(counts)

    def segments(self) -> list[tuple[Any, ...]]:
        """each part as the file is made from it, in the order written: rows a write handed as dicts typed here (the
        one step that must run where the dicts are: Arrow's own conversion, :func:`_rows_table`), every other part as
        it was handed. Blocking: run it on a thread.

        :return: each segment: ``("typed", table, written)``, ``("arrow", table, written)``, ``("deleted", table)``
            or ``("load", given, counts, constants)``
        :rtype: list[tuple]
        :raises ImportError: without pyarrow
        """
        if not _PYARROW:
            raise ImportError("a delta file needs pyarrow: install 3tears[snapshot]")
        segments: list[tuple[Any, ...]] = []
        for kind, part, *rest in chain.from_iterable(self.parts):
            if kind == "rows":
                segments.append(("typed", _rows_table(part, self.types, self.defaults), rest[0]))
            elif kind == "arrow":
                segments.append(("arrow", part, rest[0]))
            elif kind == "load":
                segments.append(("load", part, *rest))
            else:
                # a marker of a table keyed without its state carries the state the delete named (decision 27)
                state = {} if self.scope_column is None else {self.scope_column: rest[1]}
                segments.append(("deleted", _marker_table(part, self.key, state, self.types)))
        return segments

    def table(self, write_id: str, *, confirmed: bool = False) -> Any:
        """every part as one Arrow table, rows and delete markers in the order written, each row naming the columns
        its write set, with the file's metadata. Blocking: run it on a thread.

        :param write_id: the write's unique id
        :ptype write_id: str
        :param confirmed: the rows were read from L3 for a failed load request's keys (:data:`_CONFIRMED_METADATA`)
        :ptype confirmed: bool
        :return: the file's table
        :rtype: pyarrow.Table
        :raises ImportError: without pyarrow
        :raises ValueError: when a write names a column the table does not have
        """
        return build_file_table(
            self.name,
            self.types,
            self.key,
            self.defaults,
            self.prefixes,
            self.segments(),
            write_id,
            confirmed=confirmed,
        )


def _rows_table(rows: Sequence[Mapping[str, Any]], types: Mapping[str, Any], defaults: Mapping[str, Any]) -> Any:
    """rows a write handed as dicts, as the table's columns typed as the rail answers them, each column a write left
    out of a row holding its fixed database default (:attr:`TableDelta.defaults`): Arrow's own conversion with the
    declared schema (no type inferred), each value converted in C, never a Python object made a row. A value Arrow will
    not take as its column's type (a ``Decimal`` or a UUID in a text column, which the rail's JSON carries as its text)
    sends that column through the rail's value-by-value conversion
    (:func:`~threetears.core.backends.arrow_reply._rail_value`).

    Why: the rows were typed a value at a time in Python (and copied a row at a time where a default was left out),
    on the serving process (2026-11-23 audit).

    :param rows: the rows, by column name
    :ptype rows: Sequence[Mapping[str, Any]]
    :param types: column name -> its Arrow type, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :param defaults: column -> its fixed default, as the rail answers it
    :ptype defaults: Mapping[str, Any]
    :return: the rows
    :rtype: pyarrow.Table
    """
    listed = rows if isinstance(rows, list) else list(rows)
    schema = pa.schema([pa.field(name, kind) for name, kind in types.items()])
    try:
        typed = pa.Table.from_pylist(listed, schema=schema)
    except pa.ArrowInvalid, pa.ArrowTypeError, TypeError:  # NOSILENT: a value Arrow will not take as its type
        columns = []
        for name, kind in types.items():
            values = [row.get(name) for row in listed]
            try:
                columns.append(pa.array(values, kind))
            except pa.ArrowInvalid, pa.ArrowTypeError, TypeError:  # NOSILENT: the rail's text of the value
                columns.append(pa.array([_rail_value(value, kind) for value in values], kind))
        typed = pa.Table.from_arrays(columns, schema=schema)
    for name, value in defaults.items():
        if name not in types:
            continue
        # a row that names the column, None included, keeps what it named: the INSERT sent it
        absent = np.fromiter((name not in row for row in listed), dtype=bool, count=len(listed))
        if absent.any():
            at = typed.schema.get_field_index(name)
            filled = pc.if_else(pa.array(absent), pa.scalar(value, types[name]), typed.column(name))
            typed = typed.set_column(at, schema.field(name), filled)
    return typed


def _marker_table(
    keys: Sequence[Sequence[Any]] | Any, key: Sequence[str], state: Mapping[str, Any], types: Mapping[str, Any]
) -> Any:
    """delete markers as the table's columns: the key columns set, the state a table keyed without it carries
    (decision 27), every other column null; a column at a time, never a row made.

    :param keys: the keys deleted, each its values in key order; or an Arrow table of the key columns, taken as it is
    :ptype keys: Sequence[Sequence[Any]] | pyarrow.Table
    :param key: the key columns, in key order
    :ptype key: Sequence[str]
    :param state: the state column and the value every marker carries; empty for a table whose key names it
    :ptype state: Mapping[str, Any]
    :param types: column name -> its Arrow type, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :return: the markers
    :rtype: pyarrow.Table
    """
    count = len(keys)
    columns = []
    for name, kind in types.items():
        if name in key and isinstance(keys, pa.Table):
            columns.append(keys.column(name).combine_chunks().cast(kind))
        elif name in key:
            at = list(key).index(name)
            values = [values[at] for values in keys]
            try:
                columns.append(pa.array(values, kind))
            except pa.ArrowInvalid, pa.ArrowTypeError, TypeError:  # NOSILENT: the rail's text of the value
                columns.append(pa.array([_rail_value(value, kind) for value in values], kind))
        elif name in state:
            columns.append(pa.array([_rail_value(state[name], kind)] * count, kind))
        else:
            columns.append(pa.nulls(count, kind))
    return pa.Table.from_arrays(columns, schema=pa.schema([pa.field(name, kind) for name, kind in types.items()]))


def build_file_table(
    name: str,
    types: Mapping[str, Any],
    key: Sequence[str],
    defaults: Mapping[str, Any],
    prefixes: Sequence[Mapping[str, str]],
    segments: Sequence[tuple[Any, ...]],
    write_id: str,
    *,
    confirmed: bool = False,
) -> Any:
    """a delta file's table from its segments (:meth:`TableDelta.segments`): each typed as the table's columns, its
    rows marked deleted or not and each naming the columns its write set, joined in order, with the file's metadata.
    Blocking; where a write saves its files, in a worker process (:mod:`~threetears.core.collections.delta_save`).

    :param name: the table
    :ptype name: str
    :param types: each column's Arrow type, in the table's column order
    :ptype types: Mapping[str, pyarrow.DataType]
    :param key: the key columns, in key order
    :ptype key: Sequence[str]
    :param defaults: each column's fixed database default (:attr:`TableDelta.defaults`)
    :ptype defaults: Mapping[str, Any]
    :param prefixes: the key prefixes the write cleared, as text
    :ptype prefixes: Sequence[Mapping[str, str]]
    :param segments: the segments, in the order written
    :ptype segments: Sequence[tuple]
    :param write_id: the write's unique id
    :ptype write_id: str
    :param confirmed: mark the file confirmed from L3 (:data:`_CONFIRMED_METADATA`)
    :ptype confirmed: bool
    :return: the file's table
    :rtype: pyarrow.Table
    :raises ImportError: without pyarrow
    :raises ValueError: when a write names a column the table does not have, or a load's rows are not as many as its
        files hold
    """
    if not _PYARROW:
        raise ImportError("a delta file needs pyarrow: install 3tears[snapshot]")
    order = {column: position for position, column in enumerate(types)}
    sets: dict[tuple[str, ...], int] = {}

    def index(columns: Sequence[str]) -> int:
        unknown = [column for column in columns if column not in order]
        if unknown:
            raise ValueError(f"{name}: a write set {unknown[0]!r}, not a column of the table")
        # in the table's column order, so one set written in two orders is one set
        named = tuple(sorted(set(columns), key=order.__getitem__))
        return sets.setdefault(named, len(sets))

    joined_segments = []
    for kind, part, *rest in segments:
        written: Any = ()
        if kind == "typed":
            segment, written = part, rest[0]
        elif kind == "arrow":
            segment, written = _typed(part, types, defaults), rest[0]
        elif kind == "load":
            # a COPY inserts and sets no column of a row already held: none written
            segment = _typed(_load_delta_rows(part, rest[0], rest[1]), types, defaults)
        else:
            segment = part
        if written and not isinstance(written[0], str):
            # single-row upserts, each naming its own columns: one index a row (a handful of distinct sets)
            indices = pa.array([index(columns) for columns in written], pa.int32())
        else:
            indices = pa.array(np.full(segment.num_rows, index(written), dtype=np.int32))
        marker = pa.array(np.full(segment.num_rows, kind == "deleted", dtype=bool))
        segment = segment.append_column(pa.field(_DELETED_COLUMN, pa.bool_(), nullable=False), marker)
        joined_segments.append(segment.append_column(pa.field(_WRITTEN_COLUMN, pa.int32(), nullable=False), indices))
    schema = pa.schema(
        [
            *(pa.field(column, kind) for column, kind in types.items()),
            pa.field(_DELETED_COLUMN, pa.bool_(), False),
            pa.field(_WRITTEN_COLUMN, pa.int32(), False),
        ]
    )
    joined = pa.concat_tables(joined_segments) if joined_segments else schema.empty_table()
    metadata = {
        _TABLE_METADATA: name.encode(),
        _WRITE_METADATA: write_id.encode(),
        _KEY_METADATA: json.dumps(list(key)).encode(),
        _WRITTEN_METADATA: json.dumps([list(named) for named in sets]).encode(),
        _PREFIXES_METADATA: json.dumps(list(prefixes)).encode(),
        _FORMAT_METADATA: _DELTA_FORMAT,
    }
    if confirmed:
        metadata[_CONFIRMED_METADATA] = _CONFIRMED_FROM_L3
    return joined.replace_schema_metadata(metadata)


@dataclass(frozen=True, slots=True)
class DeltaChange:
    """one row of a delta file: a row a write upserted, or a key it deleted.

    :ivar row: the row as the file holds it, keyed by the table's columns (a delete marker's key columns alone set)
    :ivar deleted: the write deleted the key
    :ivar written: the columns the write set in a row L3 already held; every other column kept its held value
    """

    row: dict[str, Any]
    deleted: bool
    written: frozenset[str]


@dataclass(frozen=True, slots=True)
class DeltaTable:
    """a delta file read back as Arrow, never a Python row of it built: what a copy applies a file from
    (:class:`HeldDeltaRows`).

    Why not Python rows (a test's reader: :func:`threetears.core.testing.delta_files.read_delta_file`): a row as a
    dict costs about ten times its Arrow (291 MB for a 200,000-row file of 15 text columns, 2026-11-23 audit), and a
    copy holding every epoch's file that way through a pass ran out of memory before it was CPU bound.

    :ivar table: the table
    :ivar write: the write's unique id
    :ivar key: the table's key columns, in key order
    :ivar rows: every row and delete marker, in the order written: the table's columns as the file holds them (a
        delete marker's key columns alone set), the file's two marker columns left out
    :ivar deleted: per row, whether the write deleted the key
    :ivar written: per row, an index into ``sets``: the columns its write set in a row L3 already held
    :ivar sets: the distinct column sets the rows' writes set
    :ivar prefixes: the key prefixes the write cleared before them, as text (:func:`_prefix_text`)
    :ivar confirmed: the writer wrote it again from L3 for a failed load request's keys (:data:`_CONFIRMED_METADATA`)
    """

    table: str
    write: str
    key: tuple[str, ...]
    rows: Any
    deleted: Any
    written: Any
    sets: tuple[frozenset[str], ...]
    prefixes: tuple[dict[str, str], ...] = ()
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class DeltaKeys:
    """what a delta file reaches, as Arrow: the keys it names, each once in the order first written, and the prefixes
    it clears. Never a Python object a key.

    :ivar key: the table's key columns, in key order
    :ivar keys: one row a key, each key column's values as text (``key_text``), named as the key's columns
    :ivar prefixes: the key prefixes it clears before its rows, as text
    :ivar table: the table the file holds rows of
    :ivar confirmed: the writer wrote it again from L3 for a failed load request's keys (:data:`_CONFIRMED_METADATA`):
        a record naming it unconfirmed may be applied from it
    """

    key: tuple[str, ...]
    keys: Any
    prefixes: tuple[dict[str, str], ...]
    table: str = ""
    confirmed: bool = False


@contextmanager
def _decoding(path: Path) -> Iterator[None]:
    """what decoding a delta file raises for its bytes, as :class:`DeltaFileUnreadableError`; the rest as raised.

    :param path: the file
    :ptype path: Path
    :return: a context the file is decoded in
    :rtype: Iterator[None]
    :raises DeltaFileUnreadableError: when the file's bytes are not a delta file this reader applies
    """
    try:
        yield
    except DeltaFileUnreadableError:
        raise
    except FileNotFoundError:
        raise  # not there: the caller says so, never "unreadable"
    except OSError as exc:
        if exc.errno is not None:
            raise  # the volume's own error (EIO, EACCES): a failed read to try again, not the file's bytes
        # Arrow's decode failures carry no errno: a damaged zstd buffer, a footer that fails verification, a read past
        # the end of a truncated file
        raise DeltaFileUnreadableError(f"{path}: not a readable delta file: {exc}") from exc
    except (pa.ArrowInvalid, KeyError, IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        # ArrowInvalid: not an Arrow file, too small, an array that fails validation. KeyError: metadata a delta file
        # always holds is missing. IndexError: a row names a column set the file does not list. The JSON and its text
        # in the metadata do not decode
        raise DeltaFileUnreadableError(f"{path}: not a readable delta file: {type(exc).__name__}: {exc}") from exc


def _checked_metadata(path: Path, metadata: Mapping[bytes, bytes], table: str | None) -> tuple[str, str]:
    """a delta file's format, table and write as its metadata names them, checked against its path and the table
    expected.

    :param path: the file
    :ptype path: Path
    :param metadata: its schema's metadata
    :ptype metadata: Mapping[bytes, bytes]
    :param table: the table the caller expects its rows of; None checks none
    :ptype table: str | None
    :return: the table and the write it names
    :rtype: tuple[str, str]
    :raises DeltaFileUnreadableError: when the format is not this reader's, or the file is another table's, or another
        write's than its name says
    :raises KeyError: when a name is missing (raised as unreadable by :func:`_decoding`)
    """
    if metadata.get(_FORMAT_METADATA) != _DELTA_FORMAT:
        raise DeltaFileUnreadableError(
            f"{path}: a delta file of format {metadata.get(_FORMAT_METADATA)!r}, not this reader's {_DELTA_FORMAT!r}"
        )
    named, write = metadata[_TABLE_METADATA].decode(), metadata[_WRITE_METADATA].decode()
    if table is not None and named != table:
        raise DeltaFileUnreadableError(f"{path}: a delta file of table {named!r}, not {table!r}")
    if path.suffix == _SUFFIX and write != path.stem:
        # every file is saved under its write's own id (_delta_file_name): another name holds another commit's rows
        raise DeltaFileUnreadableError(f"{path}: the delta file of write {write!r}, not of {path.stem!r}")
    return named, write


class _HashingSink:
    """a binary file Arrow writes IPC bytes to, hashing each as it passes: the checksum costs no second read.

    :param sink: the open file
    :ptype sink: BinaryIO
    """

    def __init__(self, sink: BinaryIO) -> None:
        self._sink = sink
        self._hash = hashlib.sha256()
        self._written = 0
        self.closed = False

    def write(self, data: Any) -> int:
        """hash ``data``, count it and pass it on to the file.

        :param data: the bytes Arrow writes
        :ptype data: bytes-like
        :return: the number of bytes written
        :rtype: int
        """
        self._hash.update(data)
        self._written += len(memoryview(data))
        self._sink.write(data)
        return len(memoryview(data))

    def flush(self) -> None:
        """flush the file."""
        self._sink.flush()

    def tell(self) -> int:
        """the bytes written so far: Arrow asks where it is in the stream.

        :return: the count
        :rtype: int
        """
        return self._written

    def writable(self) -> bool:
        """always writable: Arrow checks before writing.

        :return: True
        :rtype: bool
        """
        return True

    def close(self) -> None:
        """mark the sink closed for Arrow, leaving the caller's file open."""
        self.closed = True  # the caller's file stays open for the trailer

    def digest(self) -> bytes:
        """the sha256 of every byte written: the file's checksum trailer.

        :return: the digest
        :rtype: bytes
        """
        return self._hash.digest()


def write_volume_file(sink: BinaryIO, table: Any) -> None:
    """write ``table`` to ``sink`` as Arrow IPC with zstd inside, then its checksum trailer: the sha256 of every IPC
    byte, then :data:`_CHECKSUM_MARK`. Blocking. Every file on the delta-files volume is written so (module docstring).

    :param sink: the open binary file, at its start
    :ptype sink: BinaryIO
    :param table: the rows
    :ptype table: pyarrow.Table
    :return: nothing
    :rtype: None
    """
    hashing = _HashingSink(sink)
    with pa_ipc.new_file(
        pa.PythonFile(hashing, mode="w"), table.schema, options=pa_ipc.IpcWriteOptions(compression=_COMPRESSION)
    ) as writer:
        writer.write_table(table)
    sink.write(hashing.digest() + _CHECKSUM_MARK)


def _checked_ipc(path: Path) -> Any:
    """a volume file's IPC bytes, its checksum verified before any is decoded. Blocking.

    :param path: the file
    :ptype path: Path
    :return: the IPC bytes, mapped (``pa.memory_map``: the map outlives the file's handle), for ``pa_ipc.open_file``
    :rtype: pyarrow.Buffer
    :raises FileNotFoundError: when it is not there
    :raises DeltaFileUnreadableError: when it carries no trailer, or its bytes do not match the checksum
    """
    with pa.memory_map(str(path)) as source:
        whole = source.read_buffer()
    size = whole.size
    if size < _CHECKSUM_TRAILER or whole.slice(size - len(_CHECKSUM_MARK)).to_pybytes() != _CHECKSUM_MARK:
        raise DeltaFileUnreadableError(f"{path}: no checksum trailer: not a file of this format, or cut short")
    body = whole.slice(0, size - _CHECKSUM_TRAILER)
    if hashlib.sha256(memoryview(body)).digest() != whole.slice(size - _CHECKSUM_TRAILER, 32).to_pybytes():
        raise DeltaFileUnreadableError(f"{path}: its bytes do not match its checksum: the file is damaged")
    return body


def read_volume_file(path: Path, *, columns: Sequence[str] | None = None) -> Any:
    """a volume file's table, its checksum verified first (:func:`_checked_ipc`); Arrow's own decode failures raised
    as Arrow raises them (a caller in :func:`_decoding` gets them as unreadable). Blocking.

    :param path: the file
    :ptype path: Path
    :param columns: only these columns; None reads every one
    :ptype columns: Sequence[str] | None
    :return: its table, its schema's metadata with it
    :rtype: pyarrow.Table
    :raises FileNotFoundError: when it is not there
    :raises DeltaFileUnreadableError: when its checksum does not hold
    """
    body = _checked_ipc(path)
    if columns is None:
        return pa_ipc.open_file(pa.BufferReader(body)).read_all()
    schema = pa_ipc.open_file(pa.BufferReader(body)).schema
    options = pa_ipc.IpcReadOptions(included_fields=[schema.get_field_index(column) for column in columns])
    return pa_ipc.open_file(pa.BufferReader(body), options=options).read_all()


def read_delta_table(path: Path, *, table: str | None = None) -> DeltaTable:
    """read a delta file saved by :func:`write_delta_file` as Arrow. Blocking: run it on a thread.

    Every array is validated in full, and every row's written set checked to be one the file lists, before anything is
    applied from it: a damaged file raises, never crashes the process (:class:`DeltaFileUnreadableError`). No Python
    row of it is built: the file's IPC buffers are its memory (zero-copy off the checked bytes).

    :param path: the file
    :ptype path: Path
    :param table: the table whose file it must be; None checks none
    :ptype table: str | None
    :return: the file's table, write, key, rows and markers
    :rtype: DeltaTable
    :raises FileNotFoundError: when it is not there
    :raises DeltaFileUnreadableError: when it is there but not a delta file of this format, of ``table`` and of the
        write its name says
    :raises OSError: when the volume cannot read it (an errno)
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a delta file needs pyarrow: install 3tears[snapshot]")
    with _decoding(path):
        read = read_volume_file(path)
        read.validate(full=True)
        metadata = read.schema.metadata or {}
        named, write = _checked_metadata(path, metadata, table)
        sets = tuple(frozenset(columns) for columns in json.loads(metadata[_WRITTEN_METADATA]))
        prefixes = tuple(dict(prefix) for prefix in json.loads(metadata[_PREFIXES_METADATA]))
        key = tuple(json.loads(metadata[_KEY_METADATA]))
        written = read.column(_WRITTEN_COLUMN)
        deleted = read.column(_DELETED_COLUMN)
        if read.num_rows:
            low, high = pc.min_max(written).values()
            if low.as_py() < 0 or high.as_py() >= len(sets):
                raise IndexError(f"a row names written set {high.as_py()} of the file's {len(sets)}")
        rows = read.drop_columns([_DELETED_COLUMN, _WRITTEN_COLUMN])
        confirmed = metadata.get(_CONFIRMED_METADATA) == _CONFIRMED_FROM_L3
    return DeltaTable(
        table=named,
        write=write,
        key=key,
        rows=rows,
        deleted=deleted,
        written=written,
        sets=sets,
        prefixes=prefixes,
        confirmed=confirmed,
    )


def read_delta_file_keys(path: Path, *, table: str | None = None) -> DeltaKeys:
    """the keys a delta file names, rows and delete markers alike, each once in the order first written, as Arrow
    text (``key_text``'s spelling), and the key prefixes it clears: its key columns and its metadata alone are read,
    the key columns validated in full before a key is decoded, as the full read (:func:`read_delta_table`) validates every column.
    Blocking: run it on a thread.

    Why: a copy hears a commit too big to list (337,000 keys in the 2026-11-23 audit) and a load's keys from the file;
    a Python tuple a key, then a row message a key, cost the event loop a key at a time and left the collector a
    tuple a key to walk (``design-pod-delta-files.md``, decision 35). Here no key becomes a Python object.

    :param path: the file
    :ptype path: Path
    :param table: the table whose file it must be; None checks none
    :ptype table: str | None
    :return: its key columns, keys and prefixes
    :rtype: DeltaKeys
    :raises FileNotFoundError: when it is not there
    :raises DeltaFileUnreadableError: when it is there but not a delta file of this format, of ``table`` and of the
        write its name says, or its key columns do not decode
    :raises OSError: when the volume cannot read it (an errno)
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a delta file needs pyarrow: install 3tears[snapshot]")
    with _decoding(path):
        body = _checked_ipc(path)
        schema = pa_ipc.open_file(pa.BufferReader(body)).schema
        metadata = schema.metadata or {}
        named, _write = _checked_metadata(path, metadata, table)
        key = tuple(json.loads(metadata[_KEY_METADATA]))
        prefixes = tuple(dict(prefix) for prefix in json.loads(metadata[_PREFIXES_METADATA]))
        options = pa_ipc.IpcReadOptions(included_fields=[schema.get_field_index(column) for column in key])
        keys = pa_ipc.open_file(pa.BufferReader(body), options=options).read_all()
        keys.validate(full=True)
        texts = key_text_table(keys, key)
        confirmed = metadata.get(_CONFIRMED_METADATA) == _CONFIRMED_FROM_L3
    return DeltaKeys(key=key, keys=_first_written(texts, key), prefixes=prefixes, table=named, confirmed=confirmed)


def key_text_table(rows: Any, key: Sequence[str], instants: Collection[str] = ()) -> Any:
    """the key columns of ``rows``, each as text in ``key_text``'s spelling (a change record's, a row message's), named
    as the key's columns, in key order: what a copy holds a key heard as.

    :param rows: rows holding the key columns
    :ptype rows: pyarrow.Table
    :param key: the key columns, in key order
    :ptype key: Sequence[str]
    :param instants: the key columns of a ``timestamptz`` column (``DATETIMETZ_TYPE``): each spelling of an instant
        named in its one canonical text (``key_text(..., instant=True)``), whichever text or type the rows hold it in
    :ptype instants: Collection[str]
    :return: one row a row of ``rows``
    :rtype: pyarrow.Table
    """
    return pa.table({column: _text_column(rows.column(column), instant=column in instants) for column in key})


def joined_key_texts(rows: Any, key: Sequence[str]) -> Any:
    """each row's key as one text, its key columns' texts joined by a control character no key value holds: what keys
    held as text (:func:`key_text_table`) are matched by.

    :param rows: rows holding the key columns as text
    :ptype rows: pyarrow.Table
    :param key: the key columns, in key order
    :ptype key: Sequence[str]
    :return: one text a row
    :rtype: pyarrow.Array
    """
    columns = [rows.column(column).combine_chunks() for column in key]
    joined = _key_texts(columns)
    return joined.combine_chunks() if isinstance(joined, pa.ChunkedArray) else joined


def text_key_batch(keys: Sequence[Sequence[str]]) -> Any:
    """keys given as text, one sequence a key (an epoch record's rows' ``ids``), as one Arrow batch: one row a key, its
    columns named by position (``k0``, ``k1``, ...), each as text. What the registry tells a cache that takes key
    batches (``on_keys``) in place of a row message a key.

    :param keys: the keys, each its values as text in key order
    :ptype keys: Sequence[Sequence[str]]
    :return: the batch
    :rtype: pyarrow.Table
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a key batch needs pyarrow: install 3tears[snapshot]")
    width = len(keys[0]) if keys else 0
    return pa.table({f"k{at}": pa.array([key[at] for key in keys], pa.string()) for at in range(width)})


def _first_written(texts: Any, key: Sequence[str]) -> Any:
    """``texts`` with each key once, at its first row, in the order first written.

    :param texts: key columns as text
    :ptype texts: pyarrow.Table
    :param key: the key columns
    :ptype key: Sequence[str]
    :return: the keys
    :rtype: pyarrow.Table
    """
    if not texts.num_rows:
        return texts
    indices = np.asarray(pc.dictionary_encode(joined_key_texts(texts, key)).indices, dtype=np.int64)
    _unique, first = np.unique(indices, return_index=True)
    return texts.take(pa.array(np.sort(first)))


#: the separator between a key's parts in its one text (:func:`_key_texts`): a control character no key value holds
_KEY_SEPARATOR: Final = "\x1e"


def _text_column(column: Any, *, instant: bool = False) -> Any:
    """a key column's values as :func:`~threetears.core.collections.epoch_change.key_text` names each (a change
    record's, a row message's spelling): text kept, an integer's digits cast in Arrow, any other type one distinct value
    at a time. A ``timestamptz`` column's (``instant``) one distinct value at a time too, text included: the rail's
    ``...Z``, Postgres' ``... +00`` and the record's ``....000000+00:00`` are one instant, named in the record's text.

    :param column: the column
    :ptype column: pyarrow.Array | pyarrow.ChunkedArray
    :param instant: the column is a ``timestamptz`` column's
    :ptype instant: bool
    :return: its texts
    :rtype: pyarrow.Array
    """
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    kind = column.type
    if not instant and (pa.types.is_string(kind) or pa.types.is_large_string(kind) or pa.types.is_integer(kind)):
        return column.cast(pa.string())
    encoded = pc.dictionary_encode(column)
    texts = pa.array(
        [None if value is None else key_text(value, instant=instant) for value in encoded.dictionary.to_pylist()],
        pa.string(),
    )
    return texts.take(encoded.indices)


def _key_texts(columns: Sequence[Any]) -> Any:
    """each row's key as one text, its columns' texts joined (:data:`_KEY_SEPARATOR`): what keys are matched by.

    :param columns: the key columns' texts, in key order
    :ptype columns: Sequence[pyarrow.Array]
    :return: one text a row
    :rtype: pyarrow.Array
    """
    return columns[0] if len(columns) == 1 else pc.binary_join_element_wise(*columns, _KEY_SEPARATOR)


class HeldDeltaRows:
    """some keys' rows as Arrow, each held or not, with delta files' changes applied in order exactly as
    :func:`~threetears.core.testing.delta_files.apply_delta_row` applies one, a column at a time over every key a file changes: never a Python row built.

    Why: a copy applied a file by turning it into Python rows (``to_pylist``, about ten times its Arrow: 291 MB for a
    200,000-row file of 15 text columns) and merging each key's row in a Python loop, every epoch's file held so through
    the pass (2026-11-23 audit, P0). Here a file stays the Arrow it was read as, and the copy holds one at a time.

    The rule, per change (decision 19 of ``design-pod-delta-files.md``): a delete leaves no row; a key not held takes
    the file's row whole (a column the file lacks is null); a held key takes the file's value of each column its write
    set and keeps every other (its ``date_created`` among them). A key under a prefix the file clears is not held
    before the file's own changes of it. A key's several changes in one file apply in the order written.

    :ivar key: the table's key columns, in key order
    :ivar applied: how many changes were applied, each once
    """

    def __init__(
        self, key: Sequence[str], keys: Any, types: Mapping[str, Any], *, instants: Collection[str] = ()
    ) -> None:
        """the keys, none held yet.

        :param key: the table's key columns, in key order
        :ptype key: Sequence[str]
        :param keys: the keys, each its values as text (``key_text``), unique; a key's position is its index here.
            A table of the key columns as text (:func:`key_text_table`), or one tuple a key
        :ptype keys: pyarrow.Table | Sequence[Sequence[str]]
        :param types: each column the rows hold, onto its Arrow type, in order
        :ptype types: Mapping[str, pyarrow.DataType]
        :param instants: the key columns of a ``timestamptz`` column: the keys and the rows' keys are matched by each
            instant's canonical text (:func:`key_text_table`), whichever spelling each holds
        :ptype instants: Collection[str]
        """
        self.key = tuple(key)
        self._instants = frozenset(instants)
        self._types = dict(types)
        if isinstance(keys, pa.Table):
            self._count = keys.num_rows
            texts = [keys.column(column).combine_chunks().cast(pa.string()) for column in self.key]
        else:
            self._count = len(keys)
            texts = [pa.array([values[at] for values in keys], pa.string()) for at in range(len(self.key))]
        texts = [
            _text_column(text, instant=True) if column in self._instants else text
            for column, text in zip(self.key, texts, strict=True)
        ]
        self._key_columns = dict(zip(self.key, texts, strict=True))
        self._texts = _key_texts(texts)
        #: each column's values (Arrow), one value a key: an immutable value replaced whole on change
        self._columns: Mapping[str, Any] = frozen(
            {name: pa.nulls(self._count, kind) for name, kind in self._types.items()}
        )
        self._held = np.zeros(self._count, dtype=bool)
        self.applied = 0

    def _positions(self, rows: Any) -> Any:
        """each row's key's position among the keys; -1 for a key not among them.

        :param rows: rows holding the key columns
        :ptype rows: pyarrow.Table
        :return: one position a row
        :rtype: numpy.ndarray
        """
        found = pc.index_in(
            _key_texts([_text_column(rows.column(name), instant=name in self._instants) for name in self.key]),
            value_set=self._texts,
        )
        return np.asarray(pc.fill_null(found, -1).to_numpy(zero_copy_only=False), dtype=np.int64)

    def _placed(self, column: Any, positions: Any, values: Any) -> Any:
        """``column`` with ``values`` at ``positions``, every other value kept.

        :param column: one value a key
        :ptype column: pyarrow.Array
        :param positions: the keys' positions, each once
        :ptype positions: numpy.ndarray
        :param values: one value a position, of the column's type
        :ptype values: pyarrow.Array
        :return: the column
        :rtype: pyarrow.Array
        """
        index = np.arange(self._count, dtype=np.int64)
        index[positions] = self._count + np.arange(len(positions), dtype=np.int64)
        return pa.concat_arrays([column, values]).take(pa.array(index))

    def _typed(self, column: Any, name: str) -> Any:
        """a column's values as its type here.

        :param column: the values
        :ptype column: pyarrow.Array | pyarrow.ChunkedArray
        :param name: the column
        :ptype name: str
        :return: the values, typed
        :rtype: pyarrow.Array
        """
        if isinstance(column, pa.ChunkedArray):
            column = column.combine_chunks()
        kind = self._types[name]
        return column if column.type == kind else column.cast(kind)

    def hold(self, rows: Any) -> None:
        """the rows held before any change (the copy's own), each at its key; a row of a key not among the keys is
        left out.

        :param rows: the rows, every column of ``types`` typed as it is
        :ptype rows: pyarrow.Table
        :return: nothing
        :rtype: None
        """
        if not rows.num_rows:
            return
        positions = self._positions(rows)
        taken = np.flatnonzero(positions >= 0)
        positions = positions[taken]
        index = pa.array(taken)
        self._columns = frozen(
            {
                name: self._placed(self._columns[name], positions, self._typed(rows.column(name).take(index), name))
                for name in self._types
            }
        )
        self._held[positions] = True

    def _under(self, prefixes: Sequence[Mapping[str, str]]) -> Any:
        """which keys lie under one of ``prefixes`` (:func:`under_prefix`, compared as text).

        :param prefixes: the prefixes, as a delta file names them
        :ptype prefixes: Sequence[Mapping[str, str]]
        :return: one flag a key
        :rtype: numpy.ndarray
        """
        return _under_texts(self._key_columns, self._count, prefixes, self._instants)

    def apply(self, delta: DeltaTable, positions: Any) -> Any:
        """apply one file to the keys at ``positions`` (those that reach it at its epoch): each under a prefix it clears
        is no longer held, then its changes of them apply in the order written.

        :param delta: the file
        :ptype delta: DeltaTable
        :param positions: the keys' positions, each once
        :ptype positions: numpy.ndarray
        :return: the positions of those the file neither changes nor clears (the caller says what that means)
        :rtype: numpy.ndarray
        """
        reach = np.zeros(self._count, dtype=bool)
        reach[positions] = True
        cleared = np.zeros(self._count, dtype=bool)
        if delta.prefixes:
            cleared = self._under(delta.prefixes) & reach
            self._held[cleared] = False
        at = self._positions(delta.rows.select(list(self.key)))
        rows = np.flatnonzero(at >= 0)
        rows = rows[reach[at[rows]]]
        keys = at[rows]
        changed = np.zeros(self._count, dtype=bool)
        changed[keys] = True
        # each key's changes in the order written: every key's first change at once, then every second, ...
        order = np.lexsort((rows, keys))
        rows, keys = rows[order], keys[order]
        if len(keys):
            first = np.ones(len(keys), dtype=bool)
            first[1:] = keys[1:] != keys[:-1]
            starts = np.flatnonzero(first)
            rank = np.arange(len(keys)) - np.repeat(starts, np.diff(np.append(starts, len(keys))))
            for nth in range(int(rank.max()) + 1):
                this = rank == nth
                self._changes(delta, keys[this], rows[this])
        return positions[~(changed[positions] | cleared[positions])]

    def _changes(self, delta: DeltaTable, keys: Any, rows: Any) -> None:
        """apply one change to each of ``keys``: the file's row at the same index of ``rows``.

        :param delta: the file
        :ptype delta: DeltaTable
        :param keys: the keys' positions, each once
        :ptype keys: numpy.ndarray
        :param rows: each one's change, as the file's row index
        :ptype rows: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        index = pa.array(rows)
        deleted = np.asarray(delta.deleted.take(index).to_numpy(zero_copy_only=False), dtype=bool)
        written = np.asarray(delta.written.take(index).to_numpy(zero_copy_only=False), dtype=np.int64)
        held = self._held[keys]
        named = set(delta.rows.column_names)
        for name, kind in self._types.items():
            sets = np.fromiter((name in columns for columns in delta.sets), dtype=bool, count=len(delta.sets))
            # a key not held takes the file's row whole; a held one the columns its write set
            from_file = ~held | sets[written]
            if not from_file.any():
                continue
            values = (
                self._typed(delta.rows.column(name).take(index), name) if name in named else pa.nulls(len(rows), kind)
            )
            if not from_file.all():
                values = pc.if_else(pa.array(from_file), values, self._columns[name].take(pa.array(keys)))
            self._columns = with_entry(self._columns, name, self._placed(self._columns[name], keys, values))
        self._held[keys] = ~deleted
        self.applied += len(keys)

    def rows(self) -> tuple[Any, Any]:
        """the rows held now, in key position order, and each one's position.

        :return: the rows, typed as ``types``; their keys' positions, ascending
        :rtype: tuple[pyarrow.Table, numpy.ndarray]
        """
        positions = np.flatnonzero(self._held)
        index = pa.array(positions)
        schema = pa.schema(list(self._types.items()))
        return pa.table([self._columns[name].take(index) for name in self._types], schema=schema), positions


def _under_texts(
    columns: Mapping[str, Any], count: int, prefixes: Sequence[Mapping[str, str]], instants: Collection[str] = ()
) -> Any:
    """which of ``count`` keys, their key columns as text, lie under one of ``prefixes`` (:func:`under_prefix`).

    :param columns: each key column's texts (``key_text``'s spelling; an instant column's in its canonical text,
        :func:`_text_column` with ``instant``)
    :ptype columns: Mapping[str, pyarrow.Array | pyarrow.ChunkedArray]
    :param count: the keys
    :ptype count: int
    :param prefixes: the prefixes, as a delta file names them
    :ptype prefixes: Sequence[Mapping[str, str]]
    :param instants: the key columns of a ``timestamptz`` column: a prefix's value of one is compared in its instant's
        canonical text (``key_text(..., instant=True)``), as the keys' texts are, whichever spelling the file kept
    :ptype instants: Collection[str]
    :return: one flag a key
    :rtype: numpy.ndarray
    """
    under = np.zeros(count, dtype=bool)
    for prefix in prefixes:
        within = np.ones(count, dtype=bool)
        for column, value in prefix.items():
            # a file's few prefixes, one value each: the one canonical text, never a second spelling of an instant
            named = key_text(value, instant=True) if column in instants else value
            equal = pc.equal(columns[column], pa.scalar(named, pa.string()))
            within &= np.asarray(pc.fill_null(equal, False).to_numpy(zero_copy_only=False), dtype=bool)
        under |= within
    return under


def fold_delta_tables(
    key: Sequence[str],
    held: Any,
    deltas: Sequence[DeltaTable],
    types: Mapping[str, Any],
    *,
    instants: Collection[str] = (),
) -> Any:
    """a part's rows with delta files' changes applied in order, exactly as :func:`~threetears.core.testing.delta_files.apply_delta_row` applies each (a
    prefix a file clears taking out every row under it before the file's own changes): in Arrow, a column at a time.
    The writer's fold of a state (:mod:`~threetears.core.collections.delta_fold`). CPU: run it in a worker process.

    Only the held rows a file changes or clears, and the keys the files add, are merged (:class:`HeldDeltaRows`); every
    other held row is kept as it was read, first, in its order. Why (0.95 audit): the fold turned each changed row into
    a Python dict (``to_pylist``) and merged them in a Python loop, in the worker, at every fold of every state.
    Measured 2026-11-23 (macOS arm64, Python 3.14): a 337,000-row state part and four deltas (95,000 changes, a
    cleared prefix) merged in 0.70 s, now 0.11 s; a 337,000-row, 15-column part and two 60,000-row deltas peaked at
    254 MB of Python, now 7.1 MB (``test_delta_fold_merge``).

    :param key: the table's key columns, in key order
    :ptype key: Sequence[str]
    :param held: the part's rows, typed as ``types``
    :ptype held: pyarrow.Table
    :param deltas: the files' changes of the part's state, in commit order
    :ptype deltas: Sequence[DeltaTable]
    :param types: each column the part holds, onto its Arrow type, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :param instants: the key columns of a ``timestamptz`` column: the part's and the files' keys are matched by each
        instant's one canonical text (:func:`key_text_table`), whichever spelling each holds (the rail's ``...Z``, a
        record's ``...+00:00``)
    :ptype instants: Collection[str]
    :return: the folded rows, typed as ``types``
    :rtype: pyarrow.Table
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a fold needs pyarrow: install 3tears[snapshot]")
    schema = pa.schema(list(types.items()))
    prefixes = [prefix for delta in deltas for prefix in delta.prefixes]
    changed = [key_text_table(delta.rows, key, instants) for delta in deltas if delta.rows.num_rows]
    if not changed and not prefixes:
        return held
    held_keys = key_text_table(held, key, instants)
    hit = np.zeros(held.num_rows, dtype=bool)
    if changed:
        touched = pc.unique(joined_key_texts(pa.concat_tables(changed), key))
        named = pc.is_in(joined_key_texts(held_keys, key), value_set=touched)
        hit |= np.asarray(named.to_numpy(zero_copy_only=False), dtype=bool)
    if prefixes and held.num_rows:
        columns = {column: held_keys.column(column) for column in key}
        hit |= _under_texts(columns, held.num_rows, prefixes, instants)
    if not hit.any() and not changed:
        return held
    kept = pa.array(~hit)
    merged = pa.array(hit)
    keys = _first_written(pa.concat_tables([held_keys.filter(merged), *changed]), key)
    rows = HeldDeltaRows(key, keys, types, instants=instants)
    rows.hold(held.filter(merged))
    everyone = np.arange(keys.num_rows, dtype=np.int64)
    for delta in deltas:
        rows.apply(delta, everyone)
    folded, _ = rows.rows()
    return pa.concat_tables([held.filter(kept).cast(schema), folded])


def _typed(rows: Any, types: Mapping[str, Any], defaults: Mapping[str, Any]) -> Any:
    """rows a write handed as Arrow, as the table's columns typed as the rail answers them: a column already of its
    type kept, one Arrow casts exactly (integers, instants, text) cast, any other made value by value as the rail's
    JSON carries it (a ``Decimal`` or UUID as text); a column the rows lack is its fixed database default (the write's
    ``INSERT`` left it out, so a new row took it), or null.

    :param rows: the rows
    :ptype rows: pyarrow.Table
    :param types: each column's Arrow type, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :param defaults: column -> its fixed default, as the rail answers it
    :ptype defaults: Mapping[str, Any]
    :return: the rows, typed
    :rtype: pyarrow.Table
    """
    columns = []
    for name, kind in types.items():
        if name not in rows.column_names:
            fill = defaults.get(name)
            columns.append(
                pa.nulls(rows.num_rows, kind) if fill is None else pa.repeat(pa.scalar(fill, kind), rows.num_rows)
            )
            continue
        column = rows.column(name)
        given = column.type
        if given == kind:
            columns.append(column)
        elif (
            (pa.types.is_integer(given) and pa.types.is_integer(kind))
            or (pa.types.is_timestamp(given) and pa.types.is_timestamp(kind))
            or (pa.types.is_boolean(given) and pa.types.is_boolean(kind))
            or (pa.types.is_large_string(given) and pa.types.is_string(kind))
        ):
            columns.append(pc.cast(column, kind))
        else:
            # the rail's text of each distinct value (a Decimal, a UUID), then taken a row at a time in Arrow: never a
            # Python object a row
            encoded = pc.dictionary_encode(column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column)
            texts = pa.array([_rail_value(value, kind) for value in encoded.dictionary.to_pylist()], kind)
            columns.append(texts.take(encoded.indices))
    return pa.Table.from_arrays(columns, schema=pa.schema([pa.field(name, kind) for name, kind in types.items()]))


def _fsync_directory(directory: Path) -> None:
    """flush a directory's entries to disk, so a rename in it survives a crash.

    :param directory: the directory
    :ptype directory: Path
    :return: nothing
    :rtype: None
    """
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def write_delta_file(root: Path, delta: TableDelta, write_id: str, *, confirmed: bool = False) -> str:
    """save ``delta`` as its file under ``root``: written under a temporary name, flushed, renamed to its own name,
    the directory flushed. Blocking: where it runs is CPU (the rows typed, joined, compressed and hashed), so a serving
    process's write saves its files in a worker process (:func:`~threetears.core.collections.delta_save.save_in_worker`);
    this is the step itself, for a worker and a test.

    A reader opening the file's name sees the whole file or no file: a write that fails or a process that dies
    before the rename leaves only the temporary name, which no record holds (:func:`sweep_orphan_delta_files`).

    :param root: the delta files' directory (:func:`delta_files_root`)
    :ptype root: Path
    :param delta: what the transaction wrote to the table
    :ptype delta: TableDelta
    :param write_id: the write's unique id
    :ptype write_id: str
    :param confirmed: mark it written again from L3 for a failed load request's keys (:data:`_CONFIRMED_METADATA`)
    :ptype confirmed: bool
    :return: the file's name under ``root`` (:func:`_delta_file_name`)
    :rtype: str
    :raises ValueError: when the table or write id cannot name a file
    :raises OSError: when the file cannot be written
    """
    return write_table_file(root, delta.name, write_id, delta.table(write_id, confirmed=confirmed))


def write_table_file(root: Path, table_name: str, write_id: str, table: Any) -> str:
    """save a delta file's table (:func:`build_file_table`) under its name: a temporary name, flushed, renamed, the
    directory flushed. Blocking.

    :param root: the delta files' directory
    :ptype root: Path
    :param table_name: the table
    :ptype table_name: str
    :param write_id: the write's unique id
    :ptype write_id: str
    :param table: the file's table
    :ptype table: pyarrow.Table
    :return: the file's name under ``root``
    :rtype: str
    :raises ValueError: when the table or write id cannot name a file
    :raises OSError: when the file cannot be written
    """
    name = _delta_file_name(table_name, write_id)
    final = root / name
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.parent / f".{write_id}{_PARTIAL_SUFFIX}"
    renamed = False
    try:
        with partial.open("wb") as sink:
            write_volume_file(sink, table)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(partial, final)
        renamed = True
    finally:
        if not renamed:
            # a write that raised leaves nothing under its own name; a process that died leaves this temporary
            # name for the sweep
            partial.unlink(missing_ok=True)
    _fsync_directory(final.parent)
    log.debug(
        "delta file saved before its commit",
        extra={"extra_data": {"table": table_name, "file": name, "rows": table.num_rows}},
    )
    return name


@dataclass(frozen=True, slots=True)
class LoadDelta:
    """the rows a load from files writes, as the caller read them from the same files, and the key prefixes it replaces:
    what lets a table that opts in to delta files be loaded from files (decision 17).

    The store copies the files itself (on the rail the hub's broker fetches them), so no row passes through the
    writing process; the caller reads the same export files and hands their rows here, and the load saves them as the
    delta file each request's commit names, before the request is sent.

    :ivar rows: one entry a file, in the files' order: an Arrow table or record batch of the file's rows, or the path of
        a parquet file pyarrow reads (the export file itself, fetched); the load's constants are added to each
    :ivar replaces: the key prefixes the load replaces, each the leading key columns and their values: every row L3
        holds under them is deleted first, in a transaction of its own whose delta file names them (decision 23)
    """

    rows: Sequence[Any]
    replaces: Sequence[Mapping[str, Any]] = ()

    @classmethod
    def from_read(
        cls, rows: Any, file_rows: Sequence[int | None], *, replaces: Sequence[Mapping[str, Any]] = ()
    ) -> LoadDelta:
        """the delta of a load whose files the caller read as one table: ``rows`` cut into one table a file by the
        rows each file's manifest counts, in the files' order (the order the load hands them over in).

        One read of a counted export is the common case (a state's files read together, or a part's divisions read in
        turn), and its files' rows are told apart only by their counts: every caller cutting it by hand was the same
        code twice. The cuts are zero-copy slices, so no row is copied until the load saves the delta file on a thread.

        :param rows: the files' rows, read in the files' order, each file's rows together
        :ptype rows: pyarrow.Table
        :param file_rows: the rows each file holds, as its manifest counts them (``LoadFile.rows``), in the same order;
            None for a file the manifest does not count
        :ptype file_rows: Sequence[int | None]
        :param replaces: the key prefixes the load replaces (:attr:`replaces`)
        :ptype replaces: Sequence[Mapping[str, Any]]
        :return: the delta, one table a file
        :rtype: LoadDelta
        :raises ValueError: when a file's count is unknown or negative, or the counts do not add up to the rows read:
            the cut would hand a file rows that are not its own, and the delta file would not be what L3 loaded
        """
        cut: list[Any] = []
        at = 0
        for index, count in enumerate(file_rows):
            if count is None or count < 0:
                raise ValueError(
                    f"a load's file {index} has no row count ({count!r}), so its rows cannot be told apart"
                )
            cut.append(rows.slice(at, count))
            at += count
        if at != rows.num_rows:
            raise ValueError(f"a load's files count {at} rows, where {rows.num_rows} were read")
        return cls(rows=cut, replaces=replaces)


def _load_delta_rows(given: Sequence[Any], counts: Sequence[int], constants: Mapping[str, Any]) -> Any:
    """one request's rows of a load, as one Arrow table: each file's rows the caller gave (:attr:`LoadDelta.rows`),
    each with the load's constants as columns, checked against the rows the file holds. Blocking: run it on a thread.

    :param given: the request's files' rows, one entry a file
    :ptype given: Sequence[Any]
    :param counts: the rows each file holds (``LoadFile.rows``), in the same order
    :ptype counts: Sequence[int]
    :param constants: column -> the value every row takes for it
    :ptype constants: Mapping[str, Any]
    :return: the rows
    :rtype: pyarrow.Table
    :raises ValueError: when a file's rows given are not as many as it holds: the file would not be what L3 loaded
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a delta file needs pyarrow: install 3tears[snapshot]")
    tables = []
    for index, (rows, count) in enumerate(zip(given, counts, strict=True)):
        if isinstance(rows, pa.RecordBatch):
            table = pa.Table.from_batches([rows])
        elif isinstance(rows, pa.Table):
            table = rows
        else:
            table = pa_parquet.read_table(os.fspath(rows))
        if table.num_rows != count:
            raise ValueError(
                f"a load's file {index} holds {count} rows and the rows given for its delta file are {table.num_rows}"
            )
        for column, value in constants.items():
            if column not in table.column_names:
                table = table.append_column(column, pa.repeat(value, table.num_rows))
        tables.append(table)
    return pa.concat_tables(tables, promote_options="default") if tables else pa.table({})


class _OwnDeltaSaves:
    """the delta files this process saved before a commit, from the save until the commit settles, and those whose
    commit did not land: the only files the orphan sweep removes (:func:`sweep_orphan_delta_files`).

    Why names, not times: the sweep once removed every file of a folded table's directory saved (by its modification
    time) between the fold's start and the writer's oldest work in flight. That compared the volume's clock (the time
    an NFS server stamps) with the writer's, so a skew between them could take a file of a commit still in flight; and
    it listed every folded table's directory every round. A process knows the names it saved and how each commit
    ended, so the sweep opens only those: no listing, no clock, and never another process's file.

    Called on the event loop only (each save and each commit's end), so no lock. Bounded: a name leaves when its
    commit lands, or once the sweep has removed it; an orphan the sweep never reaches (a process that folds nothing)
    is let go past :data:`_OWN_ORPHANS_HELD`, its file left as any file of a crash is.
    """

    def __init__(self) -> None:
        #: read-only snapshots, each change a new one: never a set another call changes in place
        self._in_flight: frozenset[str] = frozenset()
        self._orphans: tuple[str, ...] = ()

    def saving(self, tables: Collection[str], write_id: str) -> list[str]:
        """note the files a write is about to save, one a table: in flight until its commit settles.

        :param tables: the tables
        :ptype tables: Collection[str]
        :param write_id: the write's unique id
        :ptype write_id: str
        :return: the files' names under the root
        :rtype: list[str]
        :raises ValueError: when a table or the write id cannot name a file
        """
        names = [_delta_file_name(table, write_id) for table in tables]
        self._in_flight = self._in_flight | frozenset(names)
        return names

    def settled(self, names: Collection[str], *, landed: bool) -> None:
        """the commit of ``names`` ended: landed, they are named by its records and leave; not, they are orphans.

        A commit whose outcome is unknown (its reply lost) is settled as not landed, as the sweep always took it: a
        record naming a file swept so is the lost-file path's (decision 29), which recreates it.

        :param names: the files
        :ptype names: Collection[str]
        :param landed: whether the commit landed
        :ptype landed: bool
        :return: nothing
        :rtype: None
        """
        self._in_flight = self._in_flight - frozenset(names)
        if landed:
            return
        orphans = (*self._orphans, *(name for name in names if name not in self._orphans))
        dropped = orphans[: max(0, len(orphans) - _OWN_ORPHANS_HELD)]
        self._orphans = orphans[len(dropped) :]
        for name in dropped:
            log.warning(
                "an orphan delta file no sweep reached is let go; its file stays on the volume",
                extra={"extra_data": {"file": name, "held": _OWN_ORPHANS_HELD}},
            )

    @property
    def in_flight(self) -> frozenset[str]:
        """the files saved whose commit has not ended: never deleted by anything.

        :return: their names
        :rtype: frozenset[str]
        """
        return self._in_flight

    @property
    def orphans(self) -> tuple[str, ...]:
        """the files saved whose commit did not land, oldest first: the sweep's to remove.

        :return: their names
        :rtype: tuple[str, ...]
        """
        return self._orphans

    def swept(self, names: Collection[str]) -> None:
        """forget orphans the sweep removed (or found gone).

        :param names: the files
        :ptype names: Collection[str]
        :return: nothing
        :rtype: None
        """
        gone = frozenset(names)
        self._orphans = tuple(name for name in self._orphans if name not in gone)


#: the most orphans this process holds for a sweep: each is a commit that failed, and a writer's fold sweeps them at
#: its next round, so only a process that saves files and folds none ever reaches it
_OWN_ORPHANS_HELD: Final = 10_000

#: this process's own saves (:class:`_OwnDeltaSaves`)
OWN_DELTA_SAVES: Final = _OwnDeltaSaves()


def _partial_name(name: str) -> str:
    """the temporary name a file is written under before its rename (:func:`write_delta_file`), under the root.

    :param name: the file's name under the root
    :ptype name: str
    :return: its temporary name
    :rtype: str
    """
    table, _, file = name.partition("/")
    return f"{table}/.{file.removesuffix(_SUFFIX)}{_PARTIAL_SUFFIX}"


def sweep_orphan_delta_files(root: Path, *, orphans: Collection[str], named: Collection[str]) -> list[str]:
    """remove each of ``orphans`` (files this process saved whose commit did not land, :class:`_OwnDeltaSaves`) that
    ``named`` (every file a committed record or a full in use names) does not hold, and the temporary file of each,
    left when its write died before its rename. Blocking.

    Each is opened by its name: no directory is listed and no file's time read, so the sweep cannot take a file of
    work still in flight, this process's or another's.

    :param root: the delta files' directory
    :ptype root: Path
    :param orphans: the files to remove, relative to ``root``
    :ptype orphans: Collection[str]
    :param named: every file name a committed record or a full in use names: never removed
    :ptype named: Collection[str]
    :return: the orphans handled (removed, or already gone), relative to ``root``: forget them
    :rtype: list[str]
    """
    handled: list[str] = []
    removed = 0
    for name in orphans:
        if name in named:
            handled.append(name)  # named after all: no orphan
            continue
        for path in (root / name, root / _partial_name(name)):
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:  # NOSILENT: gone already (never renamed, or removed before): nothing to do
                pass
        handled.append(name)
    if removed:
        log.info(
            "orphan delta files swept",
            extra={"extra_data": {"files": removed, "first": handled[0] if handled else None}},
        )
    return handled
