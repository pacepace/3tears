"""a delta file: the rows one commit wrote to a table and the keys it deleted, saved before the commit.

**Why.** Election night failed because L3 was on the hot path: every pod copy read each change's rows back from L3,
and a restarted copy whose checkpoint was older than the change notes could never learn which rows changed. A table
that opts in (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_files`) has every commit's rows
saved as a file on a volume every pod mounts, so a copy applies the change from the file instead of reading L3, and
nothing expires on a timer (``design-pod-delta-files.md``, decisions 5, 6, 9, 11 and 12).

**One file per table per commit.** A commit is one epoch of each table it wrote (the broker advances each table once
for it), so the file holds every row the commit wrote to the table, in the order written, and a delete marker
(:data:`_DELETED_COLUMN` true, the key columns set, every other column null) for each key it deleted. The columns are
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
:func:`apply_delta_row` merges a row as L3 did, so a copy or a fold that applies the file ends exactly as L3 does.

**A fixed default is written in; anything the database decides is refused** (decision 20). A column with a
``server_default`` that a write leaves out is left out of its ``INSERT``, so a new row takes the default. When the
default is a fixed value (``'active'``, ``0``, ``false``), the writer puts that value in the file's row
(:attr:`TableDelta.defaults`), as the database fills it; it is not among the columns the write set, so a row L3
already held keeps its value, as the ``DO UPDATE`` that never names the column keeps it. The file never says whether a
row was new: the copy and the fold tell, by whether they hold the key, which is how L3 told (the conflict). A default
the database computes (``now()``, a sequence) or a guarded write (one the database may skip) cannot be known before
the commit, so such a table is refused where it opts in
(:class:`~threetears.core.collections.schema_backed.TableSchema`).

**Saved before the commit, under a unique name, atomically.** The file is written under a temporary name in its
table's directory, flushed to disk, then renamed to its own name, and the directory flushed: a reader that opens the
name sees the whole file or nothing, never part of one. A crash before the commit leaves a file no committed record
names (an orphan, which :func:`sweep_orphan_delta_files` removes); a crash after it leaves the file in place, so every
committed version has its file.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from threetears.core.backends.arrow_reply import _rail_arrow_table, _rail_value
from threetears.observe import get_logger

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as pa_ipc

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a delta file refuses without it, saying so
    _PYARROW = False

__all__ = [
    "DeltaChange",
    "DeltaFile",
    "DeltaFileRefusedError",
    "TableDelta",
    "apply_delta_row",
    "delta_files_root",
    "read_delta_file",
    "read_delta_file_keys",
    "sweep_orphan_delta_files",
    "write_delta_file",
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
#: the file's schema metadata: the format's version
_FORMAT_METADATA: Final = b"threetears.delta.format"
#: this format's version: rows and delete markers in one table, as :data:`_DELETED_COLUMN` tells them apart, each
#: row naming the columns its write set (:data:`_WRITTEN_COLUMN`), zstd inside. Version 1 was plain IPC with no
#: written columns, never released
_DELTA_FORMAT: Final = b"2"
#: the codec inside the file's IPC buffers
_COMPRESSION: Final = "zstd"

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

    Each part is kept as the write handed it (rows, an Arrow table, or keys deleted), with the columns its write set,
    and typed only when the file is made (:meth:`table`), on a worker thread.

    :ivar name: the table
    :ivar types: each column's Arrow type, in the table's column order (``rail_arrow_types``)
    :ivar key: the key columns, in key order
    :ivar parts: each write's part, ``("rows", rows, written)``, ``("arrow", table, written)`` or
        ``("deletes", keys, ())``; ``written`` is one set of columns for every row, or a list of one a row
    :ivar entries: rows written and keys deleted so far: one a key the transaction enrolled
    :ivar defaults: each column whose database default is a fixed value, and that value as the rail answers it
        (``TableSchema.delta_file_defaults``): a row written without the column was inserted with it, so the file's
        row holds it rather than null
    """

    name: str
    types: Mapping[str, Any]
    key: tuple[str, ...]
    parts: list[tuple[str, Any, Any]] = field(default_factory=list)
    entries: int = 0
    defaults: Mapping[str, Any] = field(default_factory=dict)

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
            self.parts.append(("rows", rows, written))
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
            self.parts.append(("arrow", rows, written))
            self.entries += int(rows.num_rows)

    def deletes(self, keys: Sequence[Sequence[Any]]) -> None:
        """add keys deleted, each its values in key order.

        :param keys: the keys
        :ptype keys: Sequence[Sequence[Any]]
        :return: nothing
        :rtype: None
        """
        if keys:
            self.parts.append(("deletes", keys, ()))
            self.entries += len(keys)

    def table(self, write_id: str) -> Any:
        """every part as one Arrow table, rows and delete markers in the order written, each row naming the columns
        its write set, with the file's metadata. Blocking: run it on a thread.

        :param write_id: the write's unique id
        :ptype write_id: str
        :return: the file's table
        :rtype: pyarrow.Table
        :raises ImportError: without pyarrow
        :raises ValueError: when a write names a column the table does not have
        """
        if not _PYARROW:
            raise ImportError("a delta file needs pyarrow: install 3tears[snapshot]")
        order = {name: position for position, name in enumerate(self.types)}
        sets: dict[tuple[str, ...], int] = {}

        def index(columns: Sequence[str]) -> int:
            unknown = [name for name in columns if name not in order]
            if unknown:
                raise ValueError(f"{self.name}: a write set {unknown[0]!r}, not a column of the table")
            # in the table's column order, so one set written in two orders is one set
            named = tuple(sorted(set(columns), key=order.__getitem__))
            return sets.setdefault(named, len(sets))

        segments = []
        for kind, part, written in self.parts:
            if kind == "rows":
                segment = _rail_arrow_table(_defaulted(part, self.defaults), self.types)
            elif kind == "arrow":
                segment = _typed(part, self.types, self.defaults)
            else:
                segment = _rail_arrow_table([dict(zip(self.key, key, strict=True)) for key in part], self.types)
            if kind == "rows" and written and not isinstance(written[0], str):
                indices = [index(columns) for columns in written]
            else:
                indices = [index(written)] * segment.num_rows
            marker = pa.array([kind == "deletes"] * segment.num_rows, pa.bool_())
            segment = segment.append_column(pa.field(_DELETED_COLUMN, pa.bool_(), nullable=False), marker)
            segments.append(
                segment.append_column(
                    pa.field(_WRITTEN_COLUMN, pa.int32(), nullable=False), pa.array(indices, pa.int32())
                )
            )
        schema = pa.schema(
            [
                *(pa.field(name, kind) for name, kind in self.types.items()),
                pa.field(_DELETED_COLUMN, pa.bool_(), False),
                pa.field(_WRITTEN_COLUMN, pa.int32(), False),
            ]
        )
        joined = pa.concat_tables(segments) if segments else schema.empty_table()
        return joined.replace_schema_metadata(
            {
                _TABLE_METADATA: self.name.encode(),
                _WRITE_METADATA: write_id.encode(),
                _KEY_METADATA: json.dumps(list(self.key)).encode(),
                _WRITTEN_METADATA: json.dumps([list(named) for named in sets]).encode(),
                _FORMAT_METADATA: _DELTA_FORMAT,
            }
        )


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
class DeltaFile:
    """a delta file read back: the table it holds rows of, the write that saved it, and its changes in order.

    :ivar table: the table
    :ivar write: the write's unique id
    :ivar key: the table's key columns, in key order
    :ivar changes: every row and delete marker, in the order written
    """

    table: str
    write: str
    key: tuple[str, ...]
    changes: list[DeltaChange]


def read_delta_file(path: Path) -> DeltaFile:
    """read a delta file saved by :func:`write_delta_file`. Blocking: run it on a thread.

    :param path: the file
    :ptype path: Path
    :return: the file's table, write, key and changes
    :rtype: DeltaFile
    :raises ValueError: when the file is not of this format
    :raises OSError: when it cannot be read
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a delta file needs pyarrow: install 3tears[snapshot]")
    with pa.memory_map(str(path)) as source:
        table = pa_ipc.open_file(source).read_all()
    metadata = table.schema.metadata or {}
    if metadata.get(_FORMAT_METADATA) != _DELTA_FORMAT:
        raise ValueError(
            f"{path}: a delta file of format {metadata.get(_FORMAT_METADATA)!r}, not this reader's {_DELTA_FORMAT!r}"
        )
    sets = [frozenset(named) for named in json.loads(metadata[_WRITTEN_METADATA])]
    changes = []
    for row in table.to_pylist():
        deleted, written = row.pop(_DELETED_COLUMN), row.pop(_WRITTEN_COLUMN)
        changes.append(DeltaChange(row=row, deleted=deleted, written=sets[written]))
    return DeltaFile(
        table=metadata[_TABLE_METADATA].decode(),
        write=metadata[_WRITE_METADATA].decode(),
        key=tuple(json.loads(metadata[_KEY_METADATA])),
        changes=changes,
    )


def read_delta_file_keys(path: Path) -> tuple[tuple[str, ...], list[tuple[Any, ...]]]:
    """the keys a delta file names, rows and delete markers alike, each once in the order first written: its key
    columns alone are read. Blocking: run it on a thread.

    Why: a commit too big to list its rows in its epoch record names its keys here, and a follower evicts each from
    every cache it holds (``design-pod-delta-files.md``, decision 21); its other columns, the bulk of the file, are
    left unread.

    :param path: the file
    :ptype path: Path
    :return: the table's key columns, in key order; and each key's values as the file holds them
    :rtype: tuple[tuple[str, ...], list[tuple[Any, ...]]]
    :raises ValueError: when the file is not of this format
    :raises OSError: when it cannot be read
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a delta file needs pyarrow: install 3tears[snapshot]")
    with pa.memory_map(str(path)) as source:
        schema = pa_ipc.open_file(source).schema
        metadata = schema.metadata or {}
        if metadata.get(_FORMAT_METADATA) != _DELTA_FORMAT:
            raise ValueError(
                f"{path}: a delta file of format {metadata.get(_FORMAT_METADATA)!r}, not this reader's "
                f"{_DELTA_FORMAT!r}"
            )
        key = tuple(json.loads(metadata[_KEY_METADATA]))
        source.seek(0)
        options = pa_ipc.IpcReadOptions(included_fields=[schema.get_field_index(column) for column in key])
        keys = pa_ipc.open_file(source, options=options).read_all()
    values = zip(*(keys.column(column).to_pylist() for column in key), strict=True)
    return key, list(dict.fromkeys(values))


def apply_delta_row(held: Mapping[str, Any] | None, change: DeltaChange) -> dict[str, Any] | None:
    """the row L3 holds after a delta file's change, from the row it held before: merged as its upsert merged it.

    A delete leaves no row. A row L3 did not hold is the row as the write inserted it, every column as the file shows
    it (a column it left out is its fixed database default, or null). A row L3 held keeps every column the write did not set, its key and its ``date_created``
    (immutable, never set by an upsert) among them, and takes the file's value of each column the write set
    (:func:`~threetears.core.backends.schema_sql.upsert_set_columns`); a write that set none (a table that ignores
    conflicts) leaves it as it was. Pure: the copy and the fold apply a file through this one rule.

    :param held: the row held before the change, keyed by column; None when there was none
    :ptype held: Mapping[str, Any] | None
    :param change: the file's change (:attr:`DeltaFile.changes`)
    :ptype change: DeltaChange
    :return: the row held after it, keyed by column; None when the change deleted it
    :rtype: dict[str, Any] | None
    """
    if change.deleted:
        return None
    if held is None:
        return dict(change.row)
    merged = dict(held)
    for name in change.written:
        merged[name] = change.row[name]
    return merged


def _defaulted(rows: Sequence[Mapping[str, Any]], defaults: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """rows a write handed, each with the fixed database default of every column it leaves out: the value its
    ``INSERT`` (which leaves the column out) gave a new row. A column a row names, ``None`` included, is as named:
    the ``INSERT`` sent it. Only the rows that lack one are copied.

    :param rows: the rows, as written
    :ptype rows: Sequence[Mapping[str, Any]]
    :param defaults: column -> its fixed default, as the rail answers it
    :ptype defaults: Mapping[str, Any]
    :return: the rows, defaulted
    :rtype: Sequence[Mapping[str, Any]]
    """
    if not defaults:
        return rows
    return [
        row if all(name in row for name in defaults) else {**{n: v for n, v in defaults.items() if n not in row}, **row}
        for row in rows
    ]


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
            columns.append(pa.nulls(rows.num_rows, kind) if fill is None else pa.array([fill] * rows.num_rows, kind))
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
            columns.append(pa.array([_rail_value(value, kind) for value in column.to_pylist()], kind))
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


def write_delta_file(root: Path, delta: TableDelta, write_id: str) -> str:
    """save ``delta`` as its file under ``root``: written under a temporary name, flushed, renamed to its own name,
    the directory flushed. Blocking: run it on a thread.

    A reader opening the file's name sees the whole file or no file: a write that fails or a process that dies
    before the rename leaves only the temporary name, which no record holds (:func:`sweep_orphan_delta_files`).

    :param root: the delta files' directory (:func:`delta_files_root`)
    :ptype root: Path
    :param delta: what the transaction wrote to the table
    :ptype delta: TableDelta
    :param write_id: the write's unique id
    :ptype write_id: str
    :return: the file's name under ``root`` (:func:`_delta_file_name`)
    :rtype: str
    :raises ValueError: when the table or write id cannot name a file
    :raises OSError: when the file cannot be written
    """
    name = _delta_file_name(delta.name, write_id)
    final = root / name
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.parent / f".{write_id}{_PARTIAL_SUFFIX}"
    table = delta.table(write_id)
    renamed = False
    try:
        with partial.open("wb") as sink:
            with pa_ipc.new_file(
                sink, table.schema, options=pa_ipc.IpcWriteOptions(compression=_COMPRESSION)
            ) as writer:
                writer.write_table(table)
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
        extra={"extra_data": {"table": delta.name, "file": name, "rows": table.num_rows}},
    )
    return name


def sweep_orphan_delta_files(root: Path, *, named: Collection[str], before: float) -> list[str]:
    """remove every delta file under ``root`` that no committed record names and that was saved before ``before``,
    and every temporary file left by a write that died before its rename.

    A file is saved before its commit, so one no record names is either an orphan (its commit never happened) or a
    write still in flight: ``before`` is the start of the writer's oldest work in flight, and nothing saved since is
    touched. The caller says what is named and when its work began; this removes only what both clear.

    :param root: the delta files' directory
    :ptype root: Path
    :param named: every file name (:func:`_delta_file_name`) a committed record names
    :ptype named: Collection[str]
    :param before: the start of the writer's oldest work in flight, seconds since the epoch (``time.time()``)
    :ptype before: float
    :return: the names removed, relative to ``root``
    :rtype: list[str]
    """
    removed: list[str] = []
    if not root.is_dir():
        return removed
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(directory.iterdir()):
            relative = f"{directory.name}/{path.name}"
            ours = path.name.endswith(_SUFFIX) or path.name.endswith(_PARTIAL_SUFFIX)
            if not ours or relative in named or not path.is_file():
                continue
            if path.stat().st_mtime >= before:
                continue
            path.unlink(missing_ok=True)
            removed.append(relative)
    if removed:
        log.info(
            "orphan delta files swept",
            extra={"extra_data": {"files": len(removed), "first": removed[0]}},
        )
    return removed
