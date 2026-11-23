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

**Plain Arrow IPC, uncompressed.** Not zstd inside: 3tears writes no compressed IPC anywhere today (its IPC streams and
its shared Arrow are plain), the browser's files have to be plain for DuckDB-WASM and Arrow JS, and the files are
small (a typical AP delta a few KB, an LSF state's a slice of about 2 MB; the design's sizes), so one format serves
every reader with no codec to agree on.

**Saved before the commit, under a unique name, atomically.** The file is written under a temporary name in its
table's directory, flushed to disk, then renamed to its own name, and the directory flushed: a reader that opens the
name sees the whole file or nothing, never part of one. A crash before the commit leaves a file no committed record
names (an orphan, which :func:`_sweep_orphan_delta_files` removes); a crash after it leaves the file in place, so every
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
    "DeltaFileRefusedError",
    "TableDelta",
    "delta_files_root",
    "write_delta_file",
]

log = get_logger(__name__)

#: names the directory delta files go in: the shared volume's mount path in a pod. Unset, no delta file can be saved,
#: so a write to a table that opts in is refused rather than committed without its file
_DELTA_FILES_DIR_ENV: Final = "THREETEARS_DELTA_FILES_DIR"

#: the column that marks a row as a deleted key: true for a delete marker, false for a row written
_DELETED_COLUMN: Final = "_3tears_deleted"

#: the file's schema metadata: the table it holds rows of
_TABLE_METADATA: Final = b"threetears.delta.table"
#: the file's schema metadata: the write that saved it, its unique id (the file's name without its suffix)
_WRITE_METADATA: Final = b"threetears.delta.write"
#: the file's schema metadata: the table's key columns, a JSON list in key order
_KEY_METADATA: Final = b"threetears.delta.key"
#: the file's schema metadata: the format's version
_FORMAT_METADATA: Final = b"threetears.delta.format"
#: this format's version: rows and delete markers in one table, as :data:`_DELETED_COLUMN` tells them apart
_DELTA_FORMAT: Final = b"1"

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


@dataclass(slots=True)
class TableDelta:
    """what one transaction wrote to one table that opts in, in the order written, until its file is saved.

    Each part is kept as the write handed it (rows, an Arrow table, or keys deleted) and typed only when the file is
    made (:meth:`table`), on a worker thread.

    :ivar name: the table
    :ivar types: each column's Arrow type, in the table's column order (``rail_arrow_types``)
    :ivar key: the key columns, in key order
    :ivar parts: each write's part, ``("rows", rows)``, ``("arrow", table)`` or ``("deletes", keys)``
    :ivar entries: rows written and keys deleted so far: one a key the transaction enrolled
    """

    name: str
    types: Mapping[str, Any]
    key: tuple[str, ...]
    parts: list[tuple[str, Any]] = field(default_factory=list)
    entries: int = 0

    def rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """add rows written, each keyed by column.

        :param rows: the rows, as written
        :ptype rows: Sequence[Mapping[str, Any]]
        :return: nothing
        :rtype: None
        """
        if rows:
            self.parts.append(("rows", rows))
            self.entries += len(rows)

    def arrow(self, rows: Any) -> None:
        """add rows written, already an Arrow table.

        :param rows: the rows, a column each
        :ptype rows: pyarrow.Table
        :return: nothing
        :rtype: None
        """
        if rows.num_rows:
            self.parts.append(("arrow", rows))
            self.entries += int(rows.num_rows)

    def deletes(self, keys: Sequence[Sequence[Any]]) -> None:
        """add keys deleted, each its values in key order.

        :param keys: the keys
        :ptype keys: Sequence[Sequence[Any]]
        :return: nothing
        :rtype: None
        """
        if keys:
            self.parts.append(("deletes", keys))
            self.entries += len(keys)

    def table(self, write_id: str) -> Any:
        """every part as one Arrow table, rows and delete markers in the order written, with the file's metadata.
        Blocking: run it on a thread.

        :param write_id: the write's unique id
        :ptype write_id: str
        :return: the file's table
        :rtype: pyarrow.Table
        :raises ImportError: without pyarrow
        """
        if not _PYARROW:
            raise ImportError("a delta file needs pyarrow: install 3tears[snapshot]")
        segments = []
        for kind, part in self.parts:
            if kind == "rows":
                segment = _rail_arrow_table(part, self.types)
            elif kind == "arrow":
                segment = _typed(part, self.types)
            else:
                segment = _rail_arrow_table([dict(zip(self.key, key, strict=True)) for key in part], self.types)
            marker = pa.array([kind == "deletes"] * segment.num_rows, pa.bool_())
            segments.append(segment.append_column(pa.field(_DELETED_COLUMN, pa.bool_(), nullable=False), marker))
        schema = pa.schema(
            [*(pa.field(name, kind) for name, kind in self.types.items()), pa.field(_DELETED_COLUMN, pa.bool_(), False)]
        )
        joined = pa.concat_tables(segments) if segments else schema.empty_table()
        return joined.replace_schema_metadata(
            {
                _TABLE_METADATA: self.name.encode(),
                _WRITE_METADATA: write_id.encode(),
                _KEY_METADATA: json.dumps(list(self.key)).encode(),
                _FORMAT_METADATA: _DELTA_FORMAT,
            }
        )


def _typed(rows: Any, types: Mapping[str, Any]) -> Any:
    """rows a write handed as Arrow, as the table's columns typed as the rail answers them: a column already of its
    type kept, one Arrow casts exactly (integers, instants, text) cast, any other made value by value as the rail's
    JSON carries it (a ``Decimal`` or UUID as text); a column the rows lack is null.

    :param rows: the rows
    :ptype rows: pyarrow.Table
    :param types: each column's Arrow type, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :return: the rows, typed
    :rtype: pyarrow.Table
    """
    columns = []
    for name, kind in types.items():
        if name not in rows.column_names:
            columns.append(pa.nulls(rows.num_rows, kind))
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
    before the rename leaves only the temporary name, which no record holds (:func:`_sweep_orphan_delta_files`).

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
            with pa_ipc.new_file(sink, table.schema) as writer:
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


def _sweep_orphan_delta_files(root: Path, *, named: Collection[str], before: float) -> list[str]:
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
