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
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Final

from threetears.core.backends.arrow_reply import _rail_arrow_table, _rail_value
from threetears.core.collections.epoch_change import key_text
from threetears.observe import get_logger

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as pa_ipc
    import pyarrow.parquet as pa_parquet

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a delta file refuses without it, saying so
    _PYARROW = False

__all__ = [
    "DeltaChange",
    "DeltaFile",
    "DeltaFileRefusedError",
    "DeltaFileUnreadableError",
    "DeltaReach",
    "LoadDelta",
    "TableDelta",
    "apply_delta_row",
    "check_delta_scope",
    "delta_files_root",
    "load_delta_rows",
    "read_delta_file",
    "read_delta_file_reach",
    "read_volume_file",
    "sweep_orphan_delta_files",
    "under_prefix",
    "write_delta_file",
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
    :ivar prefixes: the key prefixes the write cleared before its rows, as text (:func:`_prefix_text`), each once: a
        key-prefix delete's, or a load's that replaces them (decision 23)
    :ivar scope_column: the row column a delete marker carries its row's state in, for a table whose key does not name
        it (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_scope`, decision 27); None when the key
        names it
    """

    name: str
    types: Mapping[str, Any]
    key: tuple[str, ...]
    parts: list[tuple[Any, ...]] = field(default_factory=list)
    entries: int = 0
    defaults: Mapping[str, Any] = field(default_factory=dict)
    prefixes: list[dict[str, str]] = field(default_factory=list)
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
                self.prefixes.append(named)

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

    def deletes(self, keys: Sequence[Sequence[Any]], *, scope: Any = None) -> None:
        """add keys deleted, each its values in key order, and, for a table whose key does not name a row's state,
        the state of them all: each marker carries it in :attr:`scope_column`, so the fold files the delete under that
        state's part (decision 27).

        :param keys: the keys
        :ptype keys: Sequence[Sequence[Any]]
        :param scope: the keys' state, the value of :attr:`scope_column` in each of their rows; None for a table whose
            key names it
        :ptype scope: Any
        :return: nothing
        :rtype: None
        :raises DeltaFileRefusedError: when the table's markers carry a state and none is given: the marker would be
            filed under no state's part, and the deleted row would stay in the full
        :raises ValueError: when a state is given for a table whose markers carry none
        """
        if keys:
            check_delta_scope(self.name, self.scope_column, scope)
            self.parts.append(("deletes", keys, (), scope))
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
        for kind, part, written, *scoped in self.parts:
            if kind == "rows":
                segment = _rail_arrow_table(_defaulted(part, self.defaults), self.types)
            elif kind == "arrow":
                segment = _typed(part, self.types, self.defaults)
            else:
                # a marker of a table keyed without its state carries the state the delete named (decision 27)
                state = {} if self.scope_column is None else {self.scope_column: scoped[0]}
                segment = _rail_arrow_table(
                    [{**dict(zip(self.key, key, strict=True)), **state} for key in part], self.types
                )
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
                _PREFIXES_METADATA: json.dumps(self.prefixes).encode(),
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
    :ivar prefixes: the key prefixes the write cleared before them, as text (:func:`_prefix_text`)
    """

    table: str
    write: str
    key: tuple[str, ...]
    changes: list[DeltaChange]
    prefixes: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class DeltaReach:
    """what a delta file reaches, its other columns unread: the keys it names and the prefixes it clears.

    :ivar key: the table's key columns, in key order
    :ivar keys: each key's values as the file holds them, once each in the order first written
    :ivar prefixes: the key prefixes it clears before its rows, as text
    :ivar table: the table the file holds rows of
    """

    key: tuple[str, ...]
    keys: list[tuple[Any, ...]]
    prefixes: tuple[dict[str, str], ...]
    table: str = ""


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
        self._hash.update(data)
        self._written += len(memoryview(data))
        self._sink.write(data)
        return len(memoryview(data))

    def flush(self) -> None:
        self._sink.flush()

    def tell(self) -> int:
        return self._written

    def writable(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True  # the caller's file stays open for the trailer

    def digest(self) -> bytes:
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


def read_delta_file(path: Path, *, table: str | None = None) -> DeltaFile:
    """read a delta file saved by :func:`write_delta_file`. Blocking: run it on a thread.

    Every array is validated in full before a row is decoded: a damaged file raises, never crashes the process
    (:class:`DeltaFileUnreadableError`).

    :param path: the file
    :ptype path: Path
    :param table: the table whose file it must be; None checks none
    :ptype table: str | None
    :return: the file's table, write, key and changes
    :rtype: DeltaFile
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
        sets = [frozenset(columns) for columns in json.loads(metadata[_WRITTEN_METADATA])]
        prefixes = tuple(dict(prefix) for prefix in json.loads(metadata[_PREFIXES_METADATA]))
        key = tuple(json.loads(metadata[_KEY_METADATA]))
        changes = []
        for row in read.to_pylist():
            deleted, written = row.pop(_DELETED_COLUMN), row.pop(_WRITTEN_COLUMN)
            changes.append(DeltaChange(row=row, deleted=deleted, written=sets[written]))
    return DeltaFile(table=named, write=write, key=key, changes=changes, prefixes=prefixes)


def read_delta_file_reach(path: Path, *, table: str | None = None) -> DeltaReach:
    """the keys a delta file names, rows and delete markers alike, each once in the order first written, and the key
    prefixes it clears: its key columns and its metadata alone are read. Blocking: run it on a thread.

    Why: a commit too big to list its rows in its epoch record names its keys here, and a follower evicts each from
    every cache it holds (``design-pod-delta-files.md``, decision 21); its other columns, the bulk of the file, are
    left unread. And a commit that cleared key slices (a load that replaces them, a key-prefix delete) is applied by a follower from
    its file: the prefixes tell it which of its own rows leave, the keys which rows the file brings (decision 23). The
    key columns read are validated in full before a key is decoded, as :func:`read_delta_file` validates every column.

    :param path: the file
    :ptype path: Path
    :param table: the table whose file it must be; None checks none
    :ptype table: str | None
    :return: its key columns, keys and prefixes
    :rtype: DeltaReach
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
        values = zip(*(keys.column(column).to_pylist() for column in key), strict=True)
        held = list(dict.fromkeys(values))
    return DeltaReach(key=key, keys=held, prefixes=prefixes, table=named)


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
        extra={"extra_data": {"table": delta.name, "file": name, "rows": table.num_rows}},
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


def load_delta_rows(given: Sequence[Any], counts: Sequence[int], constants: Mapping[str, Any]) -> Any:
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
                table = table.append_column(column, pa.array([value] * table.num_rows))
        tables.append(table)
    return pa.concat_tables(tables, promote_options="default") if tables else pa.table({})


def sweep_orphan_delta_files(
    root: Path,
    *,
    named: Collection[str],
    before: float,
    since: float | None = None,
    tables: Collection[str] | None = None,
) -> list[str]:
    """remove every delta file under ``root`` that no committed record names and that was saved before ``before``,
    and every temporary file left by a write that died before its rename.

    A file is saved before its commit, so one no record names is either an orphan (its commit never happened) or a
    write still in flight: ``before`` is the start of the writer's oldest work in flight, and nothing saved since is
    touched. The caller says what is named and when its work began; this removes only what both clear.

    A writer knows the commits it made itself, not a predecessor's: ``since`` (its own start) leaves every file saved
    before it, which a commit the writer never heard of may name, and ``tables`` keeps the sweep to the tables whose
    commits it hears, so a table it does not fold never loses a committed file.

    :param root: the delta files' directory
    :ptype root: Path
    :param named: every file name (:func:`_delta_file_name`) a committed record names
    :ptype named: Collection[str]
    :param before: the start of the writer's oldest work in flight, seconds since the epoch (``time.time()``)
    :ptype before: float
    :param since: leave every file saved before it, seconds since the epoch; None sweeps from the first
    :ptype since: float | None
    :param tables: the tables whose directories are swept; None sweeps every one
    :ptype tables: Collection[str] | None
    :return: the names removed, relative to ``root``
    :rtype: list[str]
    """
    removed: list[str] = []
    if not root.is_dir():
        return removed
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if tables is not None and directory.name not in tables:
            continue
        for path in sorted(directory.iterdir()):
            relative = f"{directory.name}/{path.name}"
            ours = path.name.endswith(_SUFFIX) or path.name.endswith(_PARTIAL_SUFFIX)
            if not ours or relative in named or not path.is_file():
                continue
            saved = path.stat().st_mtime
            if saved >= before or (since is not None and saved < since):
                continue
            path.unlink(missing_ok=True)
            removed.append(relative)
    if removed:
        log.info(
            "orphan delta files swept",
            extra={"extra_data": {"files": len(removed), "first": removed[0]}},
        )
    return removed
