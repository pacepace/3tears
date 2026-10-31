"""Shared-memory Arrow between a serving process and its worker processes: one physical copy, no pickle, no disk.

**Why.** CPU work in a serving process belongs in worker processes, never on threads: a thread running Python holds
the interpreter lock and the event loop waits behind it. Two things then cost memory and time. A worker cannot read
the serving process's data (its DuckDB, say), and a copy in each worker multiplies the pod's memory by its workers.
And a worker's answer -- a table of rows -- comes back pickled, copied through a pipe and copied again. Both are
solved by writing Arrow IPC files into a memory-backed directory (``/dev/shm``, or a Kubernetes ``emptyDir`` with
``medium: Memory``: RAM, never a disk -- a pod has no scratch disk and must never spill to one) and memory-mapping
them: every process mapping a file shares its pages, and a mapped Arrow column IS the file's bytes.

**Three hand-offs, one owner.** The serving process owns the directory through one :class:`SharedArrowStore`:

- *Down, versions*: :meth:`SharedArrowStore.publish` writes a data version's tables once; workers map it read-only
  (:func:`read_version`) and run DuckDB over it without copying (:meth:`MappedTables.connect`). Each version has a
  directory of its own, staged under a hidden name and renamed into place whole; the newest is current.
- *Up, results*: :meth:`SharedArrowStore.reserve` gives a job a :class:`SegmentSlot` -- a small value that pickles
  over whatever channel hands the job to its worker -- the worker writes its result tables into it
  (:meth:`SegmentSlot.write`), and the serving process maps them (:meth:`SharedArrowStore.claim`): no row is pickled
  and none copied. The serving process owns every slot from reservation to release: it claims it, or discards it
  when the job failed, was abandoned or its worker died.

- *Down, per job*: :meth:`SharedArrowStore.hand` writes one job's input tables (a change's rows, say) into a hand-off
  of their own and gives back a :class:`HandedTables` -- the hand-off's name, its tables' names and its bytes, which
  pickle in a few hundred bytes however many rows it holds -- and the worker maps them read-only
  (:meth:`HandedTables.mapped`), no row copied. Handing tables through a ``ProcessPoolExecutor`` instead pickles them
  whole in one call on its feeder thread, which never lets the interpreter lock go: the serving process's event loop
  waits behind it (74-165 ms a change on cobalt prod, 2026-10-31). The write is Arrow IPC (pyarrow writes each batch
  with the lock released) and runs on the caller's thread, never the loop. The hand-off is released when the worker
  acknowledges -- its job's future is done, whether it returned, raised or its worker died
  (:meth:`SharedArrowStore.release`) -- and its files go once the worker no longer maps them.

**Nothing leaks, judged by state, never by time.** A reader of a version -- the serving process's lease
(:meth:`SharedArrowStore.lease`) handed with a job, and the worker while it maps the files -- holds a shared
``flock`` on the version's lock file, a worker mapping a hand-off holds one on the hand-off's, and a worker writing a
slot holds an exclusive one on its staging lock. The
operating system drops a lock when its holder closes it or dies. A sweep, after every publish, release, claim and
discard, deletes every version that is not current and that no reader holds, every result whose slot nobody owns, every
hand-off released that no worker maps, and every staging directory whose writer is gone. A new store removes whatever the directory held that no live process
holds (a container restarted in the same pod keeps its ``emptyDir``). There is no timer and no TTL.

**Bounded.** A store is given the bytes it may hold, every version, result and hand-off together (``max_bytes``, set
below the volume's own limit, since a pod whose ``emptyDir`` passes its ``sizeLimit`` is evicted). An open slot counts at the
most it may take (its ``max_bytes``), and a hand-off being written at its own bound; a publish, a reservation or a
hand-off that would pass the bound, after a sweep, is refused
with :class:`SharedArrowFullError` and leaves nothing behind, and a worker's write that would pass its slot's bytes
is refused in the worker.

**Worker processes.** :class:`SharedArrowPool` runs jobs ``job(context, *args)`` in spawned, long-lived workers
(spawned, never forked: a serving process runs threads, whose locks a fork copies mid-hold). Each job may read a
version, leased for the job's whole run, may be handed tables (``tables=``, mapped as ``context.tables``), and may
write a result, claimed back as a :class:`MappedSegment`. A worker
keeps the version it read last mapped between jobs and lets it go after any job that finds a newer one current. A
worker whose private memory (its anonymous pages: the mapped files are shared and not its own) passes the pool's
bound after a job has its pool replaced, so what it allocated goes back to the system.

Guide: ``docs/shared-arrow.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import functools
import hashlib
import multiprocessing
import os
import pickle
import resource
import threading
import uuid
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Final

from threetears.observe import contain, get_logger, remove_tree

from threetears.core.utils.atomic_write import atomic_write_sync

try:
    import duckdb
    import pyarrow as pa

    # DuckDB scans a registered Arrow table through pyarrow.dataset's Scanner, importing the module (and what it
    # imports) at the first scan in the process, 25-70 ms inside a serving thread's read; imported here, at boot
    import pyarrow.dataset  # noqa: F401
    import pyarrow.ipc as ipc

    _SNAPSHOT_EXTRA: ImportError | None = None
except ImportError as exc:  # NOSILENT: the snapshot extra is optional; using shared Arrow says so
    _SNAPSHOT_EXTRA = exc

__all__ = [
    "HandedTables",
    "JobContext",
    "JobResult",
    "MappedSegment",
    "MappedTables",
    "MappedVersion",
    "Parts",
    "SegmentSlot",
    "SharedArrowFullError",
    "SharedArrowPool",
    "SharedArrowReader",
    "SharedArrowStore",
    "SharedLease",
    "SharedRead",
    "SharedVersion",
    "epochs_version",
    "plain_name",
    "query_batches",
    "read_version",
]

log = get_logger(__name__)

#: the store's three parts: published versions, workers' results, and the tables handed to one job each
_VERSIONS_DIR: Final = "versions"
_SEGMENTS_DIR: Final = "segments"
_HANDOFFS_DIR: Final = "handoffs"

#: what a hand-off's bound adds to its tables' bytes when the caller names none: the file's schema and footer, and
#: per record batch its message and each buffer's padding (Arrow IPC pads every buffer to 8 bytes)
_HANDOFF_FILE_BYTES: Final = 64 << 10
_HANDOFF_BATCH_BYTES: Final = 4 << 10
_HANDOFF_COLUMN_BYTES: Final = 512

#: the file in each version's directory its readers hold a shared ``flock`` on, and in each staging directory its
#: writer holds an exclusive one on
_LOCK_FILE: Final = ".lock"

#: the file in the versions directory naming the current version
_CURRENT_FILE: Final = "CURRENT"

#: the prefix of a version or a result being written, and of a directory being removed
_STAGING_PREFIX: Final = ".staging-"
_RETIRING_PREFIX: Final = ".retiring-"

#: each table's file: Arrow IPC, file format (random access, so a reader maps it), uncompressed (so a mapped buffer
#: IS the column, never a decompressed copy of it)
_TABLE_SUFFIX: Final = ".arrow"

#: between a table held in parts and a part's name in its file's name (``{table}@{part}.arrow``): a character no name
#: holds, so a file names its table unambiguously. The part named ``""`` (``{table}@.arrow``) holds the table's schema
#: and no rows, so a table of no parts still has its columns.
_PART_SEPARATOR: Final = "@"

#: what a version's and a table's name may hold: one path segment, never hidden
_NAME_CHARS: Final = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
_NAME_MAX: Final = 128

#: the rows of each record batch :meth:`SharedArrowStore.publish_query` reads from DuckDB at a time
DEFAULT_BATCH_ROWS: Final = 64 * 1024

#: the prefix a mapped table is registered under before its view is made (:meth:`MappedTables.connect`)
_REGISTERED_PREFIX: Final = "__shared_"


def _require_snapshot_extra() -> None:
    """raise, naming the extra, when duckdb or pyarrow could not be imported.

    :return: nothing
    :rtype: None
    :raises ImportError: when the snapshot extra is not installed
    """
    if _SNAPSHOT_EXTRA is not None:
        raise ImportError(f"shared Arrow needs duckdb and pyarrow: install 3tears[snapshot] ({_SNAPSHOT_EXTRA})")


def plain_name(name: str) -> bool:
    """whether ``name`` may name a version, a table or a part as it is: one visible path segment of
    ``[A-Za-z0-9._-]``, at most 128 long, and not reserved.

    :param name: the name
    :ptype name: str
    :return: True when it may
    :rtype: bool
    """
    return (
        0 < len(name) <= _NAME_MAX and not name.startswith(".") and set(name) <= _NAME_CHARS and name != _CURRENT_FILE
    )


def _valid_name(name: str, what: str) -> str:
    """``name`` when it is one visible path segment of :data:`_NAME_CHARS`, at most :data:`_NAME_MAX` long.

    :param name: the name
    :ptype name: str
    :param what: what it names, for the error
    :ptype what: str
    :return: the name
    :rtype: str
    :raises ValueError: when it is not
    """
    if not plain_name(name):
        raise ValueError(
            f"a shared Arrow {what} name must be 1-{_NAME_MAX} of [A-Za-z0-9._-], not hidden, not {_CURRENT_FILE!r}: "
            f"{name!r}"
        )
    return name


def _quoted(identifier: str) -> str:
    """a DuckDB identifier, double-quoted.

    :param identifier: the name
    :ptype identifier: str
    :return: the quoted name
    :rtype: str
    """
    return '"' + identifier.replace('"', '""') + '"'


def epochs_version(epochs: Mapping[str, int]) -> str:
    """a version name for the data at ``epochs`` (scope -> epoch, a scoped snapshot's read's): the same epochs always
    name the same version, and any epoch moving names another.

    :param epochs: scope -> epoch
    :ptype epochs: Mapping[str, int]
    :return: the name
    :rtype: str
    """
    text = "\n".join(f"{scope}\t{int(epoch)}" for scope, epoch in sorted(epochs.items()))
    return "e-" + hashlib.sha256(text.encode()).hexdigest()[:32]


class SharedArrowFullError(RuntimeError):
    """a write or a reservation would pass the bytes it may take (the store's, or its own bound); nothing of it was
    kept."""


def _locked(path: Path, operation: int) -> int | None:
    """an open descriptor on ``path``'s lock file holding ``operation`` (``LOCK_SH`` or ``LOCK_EX``, never waiting),
    or None when another holder's lock refuses it or the directory is gone.

    :param path: the directory
    :ptype path: Path
    :param operation: ``fcntl.LOCK_SH`` or ``fcntl.LOCK_EX``
    :ptype operation: int
    :return: the descriptor, or None
    :rtype: int | None
    """
    try:
        descriptor = os.open(path / _LOCK_FILE, os.O_RDONLY)
    except FileNotFoundError:
        # NOSILENT: the directory is gone (retired, or never written whole): the caller says what that means
        return None
    try:
        fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
    except BlockingIOError:
        # NOSILENT: another holder's lock refuses it, which is the answer the caller asked for
        os.close(descriptor)
        return None
    return descriptor


def _retiring(path: Path) -> Path | None:
    """rename a directory out of its name, so nothing opens it again: the caller removes what this returns, outside
    any lock (removing a version's files takes longer than renaming it).

    :param path: the directory
    :ptype path: Path
    :return: its new name, or None when it was gone already
    :rtype: Path | None
    """
    gone = path.with_name(f"{_RETIRING_PREFIX}{path.name}-{uuid.uuid7().hex[-8:]}")
    try:
        os.rename(path, gone)
    except FileNotFoundError:
        # NOSILENT: already gone, which is what was wanted
        return None
    return gone


def _retire(path: Path, gone: list[Path]) -> None:
    """rename ``path`` out of its name (:func:`_retiring`) and add it to ``gone``, to be removed after the lock.

    :param path: the directory
    :ptype path: Path
    :param gone: what to remove
    :ptype gone: list[Path]
    :return: nothing
    :rtype: None
    """
    renamed = _retiring(path)
    if renamed is not None:
        gone.append(renamed)


def _retire_unless_held(entry: Path, gone: list[Path]) -> bool:
    """retire a directory (into ``gone``) unless a reader holds its shared lock: a version a lease or a worker reads,
    a hand-off a worker maps. A directory with no lock file is no reader's and goes.

    :param entry: the directory
    :ptype entry: Path
    :param gone: what to remove
    :ptype gone: list[Path]
    :return: whether it was retired
    :rtype: bool
    """
    descriptor = _locked(entry, fcntl.LOCK_EX)
    if descriptor is None and (entry / _LOCK_FILE).is_file():
        return False  # a reader holds it
    try:
        _retire(entry, gone)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return True


def _directory_bytes(path: Path, seen: set[tuple[int, int]] | None = None) -> int:
    """the bytes the files directly in a directory hold, each file once however many names it has.

    A version that reuses another's parts holds hard links to the same files, whose pages exist once: ``seen``, shared
    across directories, counts each file (by device and inode) the first time it is met.

    :param path: the directory
    :ptype path: Path
    :param seen: the files counted already, by (device, inode); None to count this directory alone
    :ptype seen: set[tuple[int, int]] | None
    :return: bytes; 0 when it is gone
    :rtype: int
    """
    counted = set() if seen is None else seen
    total = 0
    with contextlib.suppress(FileNotFoundError):  # NOSILENT: a directory removed meanwhile holds nothing
        for entry in path.iterdir():
            with contextlib.suppress(FileNotFoundError):  # NOSILENT: likewise a file
                stat = entry.stat()
                key = (stat.st_dev, stat.st_ino)
                if key not in counted:
                    counted.add(key)
                    total += stat.st_size
    return total


def _table_of(file_name: str) -> str | None:
    """the table a version's file holds (all of it, or one part), or None for a file that is not a table's.

    :param file_name: the file's name
    :ptype file_name: str
    :return: the table's name, or None
    :rtype: str | None
    """
    if not file_name.endswith(_TABLE_SUFFIX) or file_name.startswith("."):
        return None
    return file_name[: -len(_TABLE_SUFFIX)].split(_PART_SEPARATOR, 1)[0]


@dataclass(frozen=True)
class Parts:
    """a table published as named parts (a scope's rows each, say), so a later version can rewrite only the parts that
    changed and share every other one with the version it was published from (:meth:`SharedArrowStore.publish`,
    ``base=``). Readers see one table: its parts' rows, in part-name order, no row copied.

    :ivar schema: the table's columns; every part written must have exactly these
    :ivar parts: part name -> its rows (as a table may be given to :meth:`SharedArrowStore.publish`), the ``Path`` of
        an Arrow IPC file already in the store's directory (a claimed result's, :meth:`MappedSegment.files`) to link
        as the part, or None to drop that part of the base version's
    """

    schema: Any
    parts: Mapping[str, Any]


def _batches(source: Any) -> tuple[Any, Iterator[Any]]:
    """a table's schema and its record batches, from a ``pyarrow.Table``, a ``pyarrow.RecordBatchReader``, or a
    callable taking no argument that returns either (opened only now, as the write reaches the table: a DuckDB
    cursor holds one answer at a time).

    :param source: the table, the reader, or what opens one
    :ptype source: pa.Table | pa.RecordBatchReader | Callable[[], pa.Table | pa.RecordBatchReader]
    :return: the schema and the batches
    :rtype: tuple[pa.Schema, Iterator[pa.RecordBatch]]
    :raises TypeError: for any other source
    """
    if callable(source) and not isinstance(source, (pa.Table, pa.RecordBatchReader)):
        source = source()
    if isinstance(source, pa.Table):
        return source.schema, iter(source.to_batches())
    if isinstance(source, pa.RecordBatchReader):
        return source.schema, iter(source)
    raise TypeError(f"a shared Arrow table is a pyarrow Table or RecordBatchReader, not {type(source).__name__}")


def _write_file(path: Path, schema: Any, batches: Iterator[Any], room: Callable[[], int], *, what: str) -> int:
    """write one uncompressed Arrow IPC file, refusing a batch that would pass ``room`` (asked before each batch: a
    store's room shrinks while a publish writes, as results are reserved beside it).

    :param path: the file
    :ptype path: Path
    :param schema: its columns; a batch with others is refused
    :ptype schema: pa.Schema
    :param batches: its record batches
    :ptype batches: Iterator[pa.RecordBatch]
    :param room: the bytes it may take, now
    :ptype room: Callable[[], int]
    :param what: what it holds, for an error
    :ptype what: str
    :return: the bytes written
    :rtype: int
    :raises SharedArrowFullError: when it would take more than ``room``
    :raises ValueError: when a batch's columns are not ``schema``
    """
    options = ipc.IpcWriteOptions(compression=None)
    with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, schema, options=options) as writer:
        for batch in batches:
            if not batch.schema.equals(schema):
                raise ValueError(f"{what}: a batch's columns are not the table's ({batch.schema} vs {schema})")
            if sink.tell() + batch.nbytes > room():
                raise SharedArrowFullError(f"{what} would pass the bytes it may take")
            writer.write_batch(batch)
    return path.stat().st_size


def _file_schema(path: Path) -> Any:
    """the columns of an Arrow IPC file.

    :param path: the file
    :ptype path: Path
    :return: the schema
    :rtype: pa.Schema
    """
    with pa.memory_map(str(path), "r") as mapped:
        return ipc.open_file(mapped).schema


def _write_tables(directory: Path, tables: Mapping[str, Any], room: Callable[[], int], base: Path | None = None) -> int:
    """write each table as uncompressed Arrow IPC in ``directory``, refusing a batch that would pass ``room``; with a
    ``base`` version, link in every file of it that ``tables`` does not replace.

    A table given whole is one file; one given as :class:`Parts` is a schema file and a file per part, and with a base
    holding it in parts keeps every base part ``Parts`` does not name. A table of the base that ``tables`` does not
    name is kept whole. A linked file is the base's own (a hard link): no byte is written and none counted again.

    :param directory: where
    :ptype directory: Path
    :param tables: name -> a table, a reader of its batches, what opens either, or :class:`Parts`
    :ptype tables: Mapping[str, Any]
    :param room: the bytes every new file together may take, now (asked before each batch)
    :ptype room: Callable[[], int]
    :param base: the version directory to keep what is not replaced from, or None
    :ptype base: Path | None
    :return: the bytes written
    :rtype: int
    :raises SharedArrowFullError: when they would take more than ``room``
    :raises ValueError: when a table or part name is not one plain path segment, a part's columns are not its
        table's, or a table in parts changed its columns from the base's (publish it without a base)
    """
    written = 0

    def left() -> int:
        return room() - written

    for name, source in tables.items():
        _valid_name(name, "table")
        if isinstance(source, Parts):
            # every writer's parts checked here, the worker's slot as the serving process's publish
            for part in source.parts:
                _valid_name(part, "part")
            schema_file = f"{name}{_PART_SEPARATOR}{_TABLE_SUFFIX}"
            written += _write_file(directory / schema_file, source.schema, iter(()), left, what=name)
            kept: list[Path] = []
            if base is not None and (base / schema_file).is_file():
                if not _file_schema(base / schema_file).equals(source.schema):
                    raise ValueError(f"table {name!r} changed its columns from the base version's; publish it whole")
                prefix = f"{name}{_PART_SEPARATOR}"
                kept = [
                    entry
                    for entry in base.iterdir()
                    if entry.name.startswith(prefix)
                    and entry.name != schema_file
                    and entry.name.endswith(_TABLE_SUFFIX)
                    and entry.name[len(prefix) : -len(_TABLE_SUFFIX)] not in source.parts
                ]
            for entry in kept:
                os.link(entry, directory / entry.name)
            for part, rows in source.parts.items():
                if rows is None:
                    continue
                path = directory / f"{name}{_PART_SEPARATOR}{part}{_TABLE_SUFFIX}"
                if isinstance(rows, Path):
                    # a file already in the store (a worker's result): linked, its pages never copied
                    if not _file_schema(rows).equals(source.schema):
                        raise ValueError(f"table {name!r} part {part!r}: its columns are not the table's")
                    os.link(rows, path)
                    continue
                part_schema, batches = _batches(rows)
                if not part_schema.equals(source.schema):
                    raise ValueError(f"table {name!r} part {part!r}: its columns are not the table's")
                written += _write_file(path, source.schema, batches, left, what=f"table {name!r} part {part!r}")
        else:
            schema, batches = _batches(source)
            path = directory / f"{name}{_TABLE_SUFFIX}"
            written += _write_file(path, schema, batches, left, what=f"table {name!r}")
    if base is not None:
        for entry in base.iterdir():
            table = _table_of(entry.name)
            if table is not None and table not in tables:
                os.link(entry, directory / entry.name)
    return written


def _shared_files(path: Path, mapped: Mapping[tuple[int, int], Any]) -> dict[tuple[int, int], Any]:
    """the mapped files (by device and inode) that the directory ``path`` holds too: what a mapping of it may take.

    :param path: a version's directory
    :ptype path: Path
    :param mapped: files mapped already, by (device, inode)
    :ptype mapped: Mapping[tuple[int, int], pa.Table]
    :return: those of them ``path`` holds; empty when it is gone
    :rtype: dict[tuple[int, int], pa.Table]
    """
    try:
        device = path.stat().st_dev
        with os.scandir(path) as listing:
            held = {(device, entry.inode()) for entry in listing}
    except FileNotFoundError:
        # NOSILENT: the version is gone already: nothing to lend its first read
        return {}
    return {inode: table for inode, table in mapped.items() if inode in held}


class MappedTables:
    """Arrow tables memory-mapped read-only from a directory: every process mapping it shares its pages, and no row
    is copied into this one. Close it when done; a table kept past :meth:`close` keeps its pages mapped, never its
    lock or its files' place in the store.

    A version made from another (``base=``) holds most of its files as hard links to the other's: given the other's
    mapped files (``reuse``, its :attr:`by_inode`), a file that is the same file (device and inode) is taken from it as
    it is, never mapped again, so a version's first read maps only what changed. The pages are the same either way: a
    file is never written once it is in place. The caller keeps every file it lends linked or mapped while this is
    made (a lease, a lock, or the mapping itself, which keeps an unlinked file's inode, and so its number, in use).

    :param name: the directory's name (a version's, a result's slot)
    :ptype name: str
    :param path: the directory
    :ptype path: Path
    :param lock: whether to hold a shared lock on the directory while mapped
    :ptype lock: bool
    :param reuse: another mapping's files by (device, inode) (:attr:`by_inode`), taken as they are where they are
        the same files
    :ptype reuse: Mapping[tuple[int, int], pa.Table] | None
    :ivar name: the directory's name
    :ivar tables: table name -> the mapped table
    """

    def __init__(
        self, name: str, path: Path, *, lock: bool, reuse: Mapping[tuple[int, int], Any] | None = None
    ) -> None:
        self.name = name
        self._descriptor: int | None = None
        if lock:
            self._descriptor = _locked(path, fcntl.LOCK_SH)
            if self._descriptor is None:
                raise LookupError(f"shared Arrow {name!r} is gone")
        self._maps: list[Any] = []
        #: each file's table, by the file's device and inode, for a later version's mapping to take (``reuse``)
        self._by_inode: dict[tuple[int, int], Any] = {}
        held = reuse or {}
        pieces: dict[str, list[Any]] = {}
        try:
            # every file of a directory is on its device; a directory entry carries its inode (no stat a file)
            device = path.stat().st_dev
            with os.scandir(path) as listing:
                entries = sorted(listing, key=lambda each: each.name)
            # sorted, so a table's schema file (``{table}@.arrow``) comes before its parts, and parts in name order
            for entry in entries:
                table = _table_of(entry.name)
                if table is not None:
                    inode = (device, entry.inode())
                    piece = held.get(inode)
                    if piece is None:
                        mapped = pa.memory_map(entry.path, "r")
                        self._maps.append(mapped)
                        piece = ipc.open_file(mapped).read_all()
                    self._by_inode[inode] = piece
                    pieces.setdefault(table, []).append(piece)
        except BaseException:
            self._unmap()
            raise
        # a table in parts is its parts' chunks side by side: concatenating tables copies no buffer
        tables = {name: held[0] if len(held) == 1 else pa.concat_tables(held) for name, held in pieces.items()}
        self.tables: Mapping[str, Any] = MappingProxyType(tables)

    @property
    def by_inode(self) -> Mapping[tuple[int, int], Any]:
        """each mapped file's table by the file's device and inode: what a later version's mapping takes (``reuse``).

        :return: (device, inode) -> the file's table; empty once closed
        :rtype: Mapping[tuple[int, int], pa.Table]
        """
        return MappingProxyType(self._by_inode)

    def connect(self, *, schema: str | None = None, config: Mapping[str, Any] | None = None) -> Any:
        """a new in-memory DuckDB connection with every table as a view over its mapped Arrow, no row copied.

        :param schema: the schema the views are made in; None for the connection's default
        :ptype schema: str | None
        :param config: DuckDB settings (``threads``, ``memory_limit``)
        :ptype config: Mapping[str, Any] | None
        :return: the connection; close it when done, before this is closed
        :rtype: duckdb.DuckDBPyConnection
        """
        connection = duckdb.connect(":memory:", config=dict(config or {}))
        if schema is not None:
            connection.execute(f"CREATE SCHEMA IF NOT EXISTS {_quoted(schema)}")
        for name, table in self.tables.items():
            registered = f"{_REGISTERED_PREFIX}{name}"
            connection.register(registered, table)
            qualified = _quoted(name) if schema is None else f"{_quoted(schema)}.{_quoted(name)}"
            connection.execute(f"CREATE VIEW {qualified} AS SELECT * FROM {_quoted(registered)}")
        return connection

    def close(self) -> None:
        """unmap the files and let the lock go.

        :return: nothing
        :rtype: None
        """
        self.tables = MappingProxyType({})
        self._by_inode = {}
        for mapped in self._maps:
            mapped.close()
        self._maps = []
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is not None:
            os.close(descriptor)

    def _unmap(self) -> None:
        """:meth:`MappedTables.close`, for a constructor that failed part way.

        :return: nothing
        :rtype: None
        """
        MappedTables.close(self)

    def __enter__(self) -> MappedTables:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.close()


class MappedVersion(MappedTables):
    """a published version mapped in a worker, holding the version's shared lock while mapped.

    :param root: the store's directory
    :ptype root: Path | str
    :param version: the version
    :ptype version: str
    :param reuse: another version's mapped files by (device, inode), taken where they are the same files
    :ptype reuse: Mapping[tuple[int, int], pa.Table] | None
    :raises LookupError: when the version is gone
    """

    def __init__(self, root: Path | str, version: str, *, reuse: Mapping[tuple[int, int], Any] | None = None) -> None:
        name = _valid_name(version, "version")
        super().__init__(name, Path(root) / _VERSIONS_DIR / name, lock=True, reuse=reuse)


class _KeptVersions:
    """a worker process's mapped version per store directory: the one it read last, kept between reads."""

    def __init__(self) -> None:
        self._kept: dict[str, MappedVersion] = {}
        self._lock = threading.Lock()

    def take(self, root: str, version: str) -> MappedVersion:
        """the kept mapping of ``version``, or a fresh one (any other kept mapping let go first).

        :param root: the store's directory
        :ptype root: str
        :param version: the version
        :ptype version: str
        :return: the mapping, no longer kept until :meth:`give_back`
        :rtype: MappedVersion
        :raises LookupError: when the version is gone
        """
        with self._lock:
            kept = self._kept.pop(root, None)
        if kept is not None and kept.name != version:
            # the new version's files that are the kept one's are taken from it, never mapped again
            try:
                return MappedVersion(root, version, reuse=kept.by_inode)
            finally:
                kept.close()
        return kept if kept is not None else MappedVersion(root, version)

    def give_back(self, root: str, mapped: MappedVersion) -> None:
        """keep ``mapped`` for the next read while it is the current version; let it go otherwise.

        :param root: the store's directory
        :ptype root: str
        :param mapped: the mapping
        :ptype mapped: MappedVersion
        :return: nothing
        :rtype: None
        """
        try:
            current: str | None = (Path(root) / _VERSIONS_DIR / _CURRENT_FILE).read_text().strip()
        except FileNotFoundError:
            # NOSILENT: nothing current (the store restarted): nothing kept is worth holding
            current = None
        with self._lock:
            keep = current == mapped.name and root not in self._kept
            if keep:
                self._kept[root] = mapped
        if not keep:
            mapped.close()


#: the versions this worker process keeps mapped between reads
_KEPT_VERSIONS = _KeptVersions()


@contextlib.contextmanager
def read_version(root: Path | str, version: str) -> Iterator[MappedVersion]:
    """a worker's read of a version: the mapping it kept from its last read when that is the version, else a fresh
    one (the kept one let go). After the block the mapping is kept for the next read unless another version is
    current, so an idle worker never holds a superseded version past its next read.

    The serving process must hold a lease on the version (:meth:`SharedArrowStore.lease`) until the read is done, so
    it is not retired between the hand-off and the mapping. One read at a time per process and store.

    :param root: the store's directory
    :ptype root: Path | str
    :param version: the version
    :ptype version: str
    :return: the mapped version
    :rtype: Iterator[MappedVersion]
    :raises LookupError: when the version is gone
    """
    mapped = _KEPT_VERSIONS.take(str(root), version)
    try:
        yield mapped
    finally:
        _KEPT_VERSIONS.give_back(str(root), mapped)


@dataclass(frozen=True)
class SegmentSlot:
    """a place reserved for one job's result: pickled to the worker, which writes into it once.

    :ivar root: the store's directory
    :ivar slot_id: the slot
    :ivar max_bytes: the most its tables may take
    """

    root: str
    slot_id: str
    max_bytes: int

    def write(self, tables: Mapping[str, Any]) -> int:
        """write the result's tables (in the worker): staged under a lock this process holds while it writes, then
        renamed into the slot whole. A writer that dies leaves a staging directory the store removes.

        :param tables: name -> a ``pyarrow.Table`` or a ``pyarrow.RecordBatchReader``
        :ptype tables: Mapping[str, pa.Table | pa.RecordBatchReader]
        :return: the bytes written
        :rtype: int
        :raises SharedArrowFullError: when they would pass the slot's ``max_bytes``; nothing is kept
        :raises FileExistsError: when the slot was written already
        """
        _require_snapshot_extra()
        segments = Path(self.root) / _SEGMENTS_DIR
        target = segments / self.slot_id
        if target.exists():
            raise FileExistsError(f"slot {self.slot_id} is written once")
        staging = segments / f"{_STAGING_PREFIX}{self.slot_id}-{uuid.uuid7().hex[-8:]}"
        staging.mkdir()
        (staging / _LOCK_FILE).touch()
        descriptor = _locked(staging, fcntl.LOCK_EX)
        try:
            written = _write_tables(staging, tables, lambda: self.max_bytes)
            os.rename(staging, target)
        except BaseException:
            remove_tree(staging, log=log, what="a failed result's staging directory")
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return written


@dataclass(frozen=True)
class HandedTables:
    """one job's input tables, written by the serving process into a hand-off of their own
    (:meth:`SharedArrowStore.hand`): what crosses to the worker in place of the tables -- a few names and numbers,
    however many rows they hold.

    :ivar root: the store's directory
    :ivar handoff_id: the hand-off
    :ivar tables: its tables' names
    :ivar nbytes: the bytes its files hold
    """

    root: str
    handoff_id: str
    tables: tuple[str, ...]
    nbytes: int

    def mapped(self) -> MappedTables:
        """map the tables read-only (in the worker), holding the hand-off's shared lock until closed: its files are
        not removed while it is open, even once the serving process has released it. Close it before the job
        returns; a context manager closes it on exit.

        :return: the mapped tables
        :rtype: MappedTables
        :raises LookupError: when the hand-off is gone (released before the worker mapped it)
        """
        _require_snapshot_extra()
        path = Path(self.root) / _HANDOFFS_DIR / _valid_name(self.handoff_id, "hand-off")
        try:
            return MappedTables(self.handoff_id, path, lock=True)
        except FileNotFoundError as exc:
            # its lock taken as a sweep renamed it away: gone all the same
            raise LookupError(f"shared Arrow hand-off {self.handoff_id!r} is gone") from exc


def _handoff_bound(tables: Mapping[str, Any]) -> int:
    """the most a hand-off of ``tables`` may take: their bytes, and the file's own per table, batch and column.

    :param tables: name -> a ``pyarrow.Table``
    :ptype tables: Mapping[str, Any]
    :return: bytes
    :rtype: int
    :raises ValueError: for a table that is not a ``pyarrow.Table`` (its bytes are not known before it is written)
    """
    total = 0
    for name, table in tables.items():
        if not isinstance(table, pa.Table):
            raise ValueError(f"hand-off table {name!r} is not a pyarrow Table: name its max_bytes")
        batches = max((column.num_chunks for column in table.columns), default=1) or 1
        total += (
            table.nbytes
            + _HANDOFF_FILE_BYTES
            + batches * (_HANDOFF_BATCH_BYTES + _HANDOFF_COLUMN_BYTES * max(1, table.num_columns))
        )
    return total


@dataclass(frozen=True)
class SharedVersion:
    """one version the store holds.

    :ivar name: the version's name
    :ivar path: its directory
    :ivar tables: its tables' names
    :ivar nbytes: the bytes its files hold, counting the files it shares with other versions (:class:`Parts`)
    """

    name: str
    path: Path
    tables: tuple[str, ...]
    nbytes: int


def _described(path: Path) -> SharedVersion:
    """a version's directory as a :class:`SharedVersion`.

    :param path: the directory
    :ptype path: Path
    :return: the version
    :rtype: SharedVersion
    """
    files = [entry for entry in path.iterdir() if _table_of(entry.name) is not None]
    return SharedVersion(
        name=path.name,
        path=path,
        tables=tuple(sorted({name for entry in files if (name := _table_of(entry.name)) is not None})),
        nbytes=sum(entry.stat().st_size for entry in files),
    )


class SharedLease:
    """a version held for a reader: not retired until released. A context manager releases it on exit;
    :meth:`release` again is harmless.

    :ivar version: the version held
    """

    def __init__(self, version: SharedVersion, descriptor: int, on_release: Callable[[], object]) -> None:
        self.version = version
        self._descriptor: int | None = descriptor
        self._on_release = on_release
        self._lock = threading.Lock()

    def release(self) -> None:
        """let the version go; the store retires it now when it is not current and nobody else holds it.

        :return: nothing
        :rtype: None
        """
        with self._lock:
            descriptor, self._descriptor = self._descriptor, None
        if descriptor is not None:
            os.close(descriptor)
            self._on_release()

    def __enter__(self) -> SharedVersion:
        return self.version

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.release()


class MappedSegment(MappedTables):
    """a worker's result, claimed by the serving process and mapped here: its files are the store's until
    :meth:`close`, which removes them (their pages go once nothing maps them).

    :ivar nbytes: the bytes its files hold
    :ivar files: table name -> its Arrow IPC file, for a publish that links it as a part (:class:`Parts`) while
        the segment is open: the version then holds the file after :meth:`close`
    """

    def __init__(self, slot_id: str, path: Path, on_close: Callable[[str], object]) -> None:
        super().__init__(slot_id, path, lock=False)
        self._on_close: Callable[[str], object] | None = on_close
        self.nbytes = _directory_bytes(path)
        self.files: Mapping[str, Path] = MappingProxyType(
            {entry.name[: -len(_TABLE_SUFFIX)]: entry for entry in path.iterdir() if entry.name.endswith(_TABLE_SUFFIX)}
        )

    def close(self) -> None:
        """unmap the result and remove its files. Harmless twice.

        :return: nothing
        :rtype: None
        """
        super().close()
        on_close, self._on_close = self._on_close, None
        if on_close is not None:
            on_close(self.name)


def query_batches(
    cursor: Any, sql: str, params: Sequence[Any] = (), *, batch_rows: int = DEFAULT_BATCH_ROWS
) -> Callable[[], Any]:
    """what opens a query's answer over a DuckDB ``cursor`` as a reader of record batches, only when a write reaches
    it (a cursor holds one answer at a time): a table to give :meth:`SharedArrowStore.publish`.

    :param cursor: the DuckDB cursor
    :ptype cursor: duckdb.DuckDBPyConnection
    :param sql: the query
    :ptype sql: str
    :param params: its parameters
    :ptype params: Sequence[Any]
    :param batch_rows: rows per batch
    :ptype batch_rows: int
    :return: the opener
    :rtype: Callable[[], pa.RecordBatchReader]
    """

    def opened() -> Any:
        answer = cursor.execute(sql, list(params))
        # to_arrow_reader from DuckDB 1.4; fetch_record_batch before it (deprecated since)
        reader = getattr(answer, "to_arrow_reader", None) or answer.fetch_record_batch
        return reader(batch_rows)

    return opened


class SharedArrowStore:
    """the serving process's shared-memory Arrow: the versions it publishes for workers to read, and the results
    workers hand back. One store per directory, in one process.

    Blocking: call :meth:`publish`, :meth:`claim` and :meth:`sweep` from a worker thread, never the event loop.

    :param root: the directory, on a memory-backed filesystem
    :ptype root: Path | str
    :param max_bytes: the most the directory may hold, every version and result together
    :ptype max_bytes: int
    """

    def __init__(self, root: Path | str, *, max_bytes: int) -> None:
        _require_snapshot_extra()
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._root = Path(root)
        self._versions = self._root / _VERSIONS_DIR
        self._segments = self._root / _SEGMENTS_DIR
        self._handoffs = self._root / _HANDOFFS_DIR
        self._versions.mkdir(parents=True, exist_ok=True)
        self._segments.mkdir(parents=True, exist_ok=True)
        self._handoffs.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        #: held while the current version, the slots or the claims are read or moved, and while the directory is
        #: swept: briefly, never while a table is written
        self._state = threading.Lock()
        #: one publish at a time, so the room each is given is exact
        self._publishing = threading.Lock()
        self._current: SharedVersion | None = None
        #: the version staging directory being written now, and the version being published, which a sweep leaves alone
        self._staging: str | None = None
        self._pending: str | None = None
        #: the serving process's readers of this store (:meth:`reader`), told after each publish
        self._readers: weakref.WeakSet[SharedArrowReader] = weakref.WeakSet()
        #: slots reserved and not yet claimed or discarded, with the bytes each may take
        self._slots: dict[str, int] = {}
        #: results claimed and not yet closed
        self._claimed: set[str] = set()
        #: each claimed result's bytes, counted once at its claim, until its files are gone (a publish's room)
        self._claimed_bytes: dict[str, int] = {}
        #: hand-offs being written, with the bytes each may take; and hand-offs written and not yet released
        self._handing: dict[str, int] = {}
        self._handed: set[str] = set()
        #: each written hand-off's bytes, counted once as its write ends, until its files are gone (a publish's room)
        self._handed_bytes: dict[str, int] = {}
        self._handoffs_made = 0
        self._published = 0
        self._retired = 0
        self._refused = 0
        with contextlib.suppress(FileNotFoundError):  # NOSILENT: nothing current is the start's state either way
            (self._versions / _CURRENT_FILE).unlink()
        self.sweep()

    @property
    def published(self) -> int:
        """versions published: a diagnostic, also in :meth:`status`, written only by this object.

        :return: the count
        :rtype: int
        """
        return self._published

    @property
    def retired(self) -> int:
        """versions retired: a diagnostic, also in :meth:`status`, written only by this object.

        :return: the count
        :rtype: int
        """
        return self._retired

    @property
    def refused(self) -> int:
        """publishes and slots refused for room: a diagnostic, also in :meth:`status`, written only by this object.

        :return: the count
        :rtype: int
        """
        return self._refused

    @property
    def root(self) -> Path:
        """the directory.

        :return: the path
        :rtype: Path
        """
        return self._root

    @property
    def max_bytes(self) -> int:
        """the most the directory may hold.

        :return: bytes
        :rtype: int
        """
        return self._max_bytes

    @property
    def current(self) -> SharedVersion | None:
        """the version published last, or None before the first.

        :return: the version
        :rtype: SharedVersion | None
        """
        return self._current

    def versions(self) -> tuple[str, ...]:
        """every version the directory holds now, current and retained, by name.

        :return: the names, sorted
        :rtype: tuple[str, ...]
        """
        return tuple(
            sorted(
                entry.name
                for entry in self._versions.iterdir()
                if entry.is_dir() and not entry.name.startswith(".") and (entry / _LOCK_FILE).is_file()
            )
        )

    def segments(self) -> tuple[str, ...]:
        """every result the directory holds now (written, claimed or not yet), by slot.

        :return: the slots, sorted
        :rtype: tuple[str, ...]
        """
        return tuple(sorted(entry.name for entry in self._segments.iterdir() if not entry.name.startswith(".")))

    def handoffs(self) -> tuple[str, ...]:
        """every hand-off the directory holds now (open, or released and still mapped by its worker), by name.

        :return: the names, sorted
        :rtype: tuple[str, ...]
        """
        return tuple(sorted(entry.name for entry in self._handoffs.iterdir() if not entry.name.startswith(".")))

    def held_bytes(self) -> int:
        """the bytes the directory's files hold now, staged ones included, a file shared by versions once.

        :return: bytes
        :rtype: int
        """
        seen: set[tuple[int, int]] = set()
        return sum(
            _directory_bytes(entry, seen)
            for half in (self._versions, self._segments, self._handoffs)
            for entry in half.iterdir()
            if entry.is_dir()
        )

    def status(self) -> dict[str, Any]:
        """what the store holds, as plain JSON values.

        :return: the values
        :rtype: dict[str, Any]
        """
        with self._state:
            current = None if self._current is None else self._current.name
            slots, claimed = len(self._slots), len(self._claimed)
            handoffs = len(self._handing) + len(self._handed)
        held_handoffs = len(self.handoffs())
        return {
            "current": current,
            "versions": list(self.versions()),
            "slots_open": slots,
            "segments_claimed": claimed,
            "handoffs_open": handoffs,
            # on disk: the open ones, and those released that a worker still maps
            "handoffs_held": held_handoffs,
            "handoffs_made": self._handoffs_made,
            "held_bytes": self.held_bytes(),
            "max_bytes": self._max_bytes,
            "published": self._published,
            "retired": self._retired,
            "refused": self._refused,
        }

    def publish(self, version: str, tables: Mapping[str, Any], *, base: str | None = None) -> SharedVersion:
        """write ``tables`` as ``version`` and make it current; a version already present is made current as it is.

        With ``base``, the new version starts as that version: every table of it not named in ``tables`` is kept, and
        a table given as :class:`Parts` keeps every part of the base's it does not name. What is kept is the base's
        own files, hard-linked: its pages are not copied and not counted twice, so a change to one part costs one
        part's bytes and one part's write.

        :param version: the version's name (:func:`epochs_version`)
        :ptype version: str
        :param tables: name -> a ``pyarrow.Table``, a ``pyarrow.RecordBatchReader`` of its rows, or a callable taking no
            argument that returns either, called only as the write reaches the table (never when the version is
            present already); or :class:`Parts` of those
        :ptype tables: Mapping[str, Any]
        :param base: a version the store holds to keep what ``tables`` does not replace from; None to write every
            table whole
        :ptype base: str | None
        :return: the version
        :rtype: SharedVersion
        :raises SharedArrowFullError: when it would pass ``max_bytes``; nothing of it is kept
        :raises ValueError: for a version, table or part name that is not one plain path segment, or a part whose
            columns are not its table's
        :raises LookupError: when ``base`` is not held
        """
        name = _valid_name(version, "version")
        for table, source in tables.items():
            _valid_name(table, "table")
            if isinstance(source, Parts):
                for part in source.parts:
                    _valid_name(part, "part")
        target = self._versions / name
        with self._publishing, contextlib.ExitStack() as held:
            # named before the write, so no sweep (a lease released on another thread) retires the version between its
            # rename into place and its being made current
            with self._state:
                self._pending = name
            try:
                if not (target / _LOCK_FILE).is_file():
                    self.sweep()
                    # the base is held until the version is current, so no sweep retires it while it is linked
                    base_path = None if base is None else held.enter_context(self.lease(base)).path
                    self._write_version(name, tables, base_path)
                with self._state:
                    published = _described(target)
                    atomic_write_sync(self._versions / _CURRENT_FILE, name, mode=0o644)
                    self._current = published
                    self._published += 1
            finally:
                with self._state:
                    self._pending = None
        self.sweep()
        self._tell_readers()
        log.info(
            "shared Arrow version published",
            extra={"extra_data": {"version": name, "bytes": published.nbytes, "held_bytes": self.held_bytes()}},
        )
        return published

    def publish_query(
        self, version: str, cursor: Any, queries: Mapping[str, str], *, batch_rows: int = DEFAULT_BATCH_ROWS
    ) -> SharedVersion:
        """write each query's answer over a DuckDB ``cursor`` as a table of ``version``, a batch at a time: no table
        is ever held whole in this process.

        :param version: the version's name
        :ptype version: str
        :param cursor: a DuckDB connection or cursor, held still for the publish (a scoped snapshot's read)
        :ptype cursor: duckdb.DuckDBPyConnection
        :param queries: table name -> the SQL whose answer it holds
        :ptype queries: Mapping[str, str]
        :param batch_rows: rows per batch read from DuckDB
        :ptype batch_rows: int
        :return: the version
        :rtype: SharedVersion
        :raises SharedArrowFullError: as :meth:`publish`
        """
        return self.publish(
            version, {table: query_batches(cursor, sql, batch_rows=batch_rows) for table, sql in queries.items()}
        )

    def lease(self, version: str | None = None) -> SharedLease:
        """hold a version for a reader: the current one, or one named that the directory still holds.

        :param version: the version; None for the current one
        :ptype version: str | None
        :return: the lease; release it (or use it as a context manager) when the reader is done
        :rtype: SharedLease
        :raises LookupError: when nothing is current, or the version named is gone
        """
        with self._state:
            if version is None:
                if self._current is None:
                    raise LookupError("no shared Arrow version is published yet")
                held = self._current
            else:
                path = self._versions / _valid_name(version, "version")
                if self._current is not None and self._current.name == version:
                    held = self._current
                elif (path / _LOCK_FILE).is_file():
                    held = _described(path)
                else:
                    raise LookupError(f"shared Arrow version {version!r} is not held")
            # under the state lock, which every retirement takes: the version cannot go between the check and this
            descriptor = _locked(held.path, fcntl.LOCK_SH)
        if descriptor is None:
            raise LookupError(f"shared Arrow version {held.name!r} is being retired")
        return SharedLease(held, descriptor, self.sweep)

    def reader(
        self,
        *,
        schema: str | None = None,
        config: Mapping[str, Any] | None = None,
        prepare: Callable[[Any], Any] | None = None,
    ) -> SharedArrowReader:
        """a reader of the current version for this process (the serving process reading what its workers read), told
        after every publish so it lets a superseded version go as soon as its last read ends.

        :param schema: the schema each read's views are made in; None for the connection's default
        :ptype schema: str | None
        :param config: DuckDB settings for the reader's connection (``threads``, ``memory_limit``)
        :ptype config: Mapping[str, Any] | None
        :param prepare: called once with each version's connection as it is mapped (to make views over its tables,
            say); what it returns is each read's ``prepared``
        :ptype prepare: Callable[[duckdb.DuckDBPyConnection], Any] | None
        :return: the reader; close it when done
        :rtype: SharedArrowReader
        """
        made = SharedArrowReader(self, schema=schema, config=config, prepare=prepare)
        with self._state:
            self._readers.add(made)
        return made

    def _tell_readers(self) -> None:
        """tell every reader a version was published, so each lets go of a superseded one nobody reads.

        :return: nothing
        :rtype: None
        """
        with self._state:
            readers = list(self._readers)
        for each in readers:
            each.refresh()

    def reserve(self, max_bytes: int) -> SegmentSlot:
        """reserve a slot for one worker's result, counted at ``max_bytes`` until it is claimed or discarded.

        :param max_bytes: the most the result's tables may take
        :ptype max_bytes: int
        :return: the slot, to hand to the worker
        :rtype: SegmentSlot
        :raises SharedArrowFullError: when the store has no room for it, after a sweep
        """
        if max_bytes <= 0:
            raise ValueError("a slot's max_bytes must be positive")
        self.sweep()
        slot_id: str | None = None
        with self._state:
            if self._committed_bytes() + max_bytes > self._max_bytes:
                self._refused += 1
            else:
                slot_id = f"s-{uuid.uuid7().hex}"
                self._slots[slot_id] = max_bytes
        if slot_id is None:
            log.warning(
                "shared Arrow result slot refused: it would pass the store's bytes",
                extra={"extra_data": {"slot_bytes": max_bytes, "max_bytes": self._max_bytes}},
            )
            raise SharedArrowFullError(f"no room for a {max_bytes}-byte result in {self._max_bytes} bytes")
        return SegmentSlot(root=str(self._root), slot_id=slot_id, max_bytes=max_bytes)

    def claim(self, slot: SegmentSlot) -> MappedSegment | None:
        """map a slot's result here, the serving process's from now until it is closed.

        :param slot: the slot
        :ptype slot: SegmentSlot
        :return: the result; None when the worker wrote nothing (the slot is closed)
        :rtype: MappedSegment | None
        :raises KeyError: when the slot is not open (claimed or discarded already)
        """
        with self._state:
            if self._slots.pop(slot.slot_id, None) is None:
                raise KeyError(f"slot {slot.slot_id} is not open")
            path = self._segments / slot.slot_id
            written = path.is_dir()
            if written:
                self._claimed.add(slot.slot_id)
                self._claimed_bytes[slot.slot_id] = _directory_bytes(path, set())
        if not written:
            self.sweep()
            return None
        try:
            return MappedSegment(slot.slot_id, path, self._released)
        except BaseException:
            self._released(slot.slot_id)
            raise

    def discard(self, slot: SegmentSlot) -> None:
        """close a slot nobody will claim (its job failed, was abandoned, or its worker died): whatever it holds goes.
        Harmless on a slot already claimed or discarded.

        :param slot: the slot
        :ptype slot: SegmentSlot
        :return: nothing
        :rtype: None
        """
        with self._state:
            self._slots.pop(slot.slot_id, None)
        self.sweep()

    def hand(self, tables: Mapping[str, Any], *, max_bytes: int | None = None) -> HandedTables:
        """write one job's input tables into a hand-off of their own, for a worker to map (:meth:`HandedTables.mapped`):
        what crosses to it is the :class:`HandedTables` alone. Blocking: on a thread, never the event loop (pyarrow
        writes each batch with the interpreter lock released, so the loop runs meanwhile).

        The hand-off counts against the store's ``max_bytes`` at its own bound while it is written, then at what its
        files hold until they are gone. Release it (:meth:`release`) once its worker is done with it, or use
        :meth:`handing`.

        :param tables: name -> a ``pyarrow.Table``, a ``pyarrow.RecordBatchReader`` or what opens either (then name
            ``max_bytes``)
        :ptype tables: Mapping[str, Any]
        :param max_bytes: the most its files may take; None for its tables' bytes and the file's own
        :ptype max_bytes: int | None
        :return: the hand-off, to pass to the worker
        :rtype: HandedTables
        :raises SharedArrowFullError: when the store has no room for it, after a sweep, or its write passes its bound;
            nothing of it is kept
        :raises ValueError: for a table name that is not one plain path segment, or a table that is not a
            ``pyarrow.Table`` with no ``max_bytes``
        """
        for name in tables:
            _valid_name(name, "table")
        bound = _handoff_bound(tables) if max_bytes is None else max_bytes
        if bound <= 0:
            raise ValueError("a hand-off's max_bytes must be positive")
        self.sweep()
        handoff_id = f"h-{uuid.uuid7().hex}"
        with self._state:
            room = self._committed_bytes() + bound <= self._max_bytes
            if room:
                self._handing[handoff_id] = bound
            else:
                self._refused += 1
        if not room:
            log.warning(
                "shared Arrow hand-off refused: it would pass the store's bytes",
                extra={"extra_data": {"handoff_bytes": bound, "max_bytes": self._max_bytes}},
            )
            raise SharedArrowFullError(f"no room for a {bound}-byte hand-off in {self._max_bytes} bytes")
        staging = self._handoffs / f"{_STAGING_PREFIX}{handoff_id}-{uuid.uuid7().hex[-8:]}"
        try:
            staging.mkdir()
            written = _write_tables(staging, tables, lambda: bound)
            (staging / _LOCK_FILE).touch()
            os.rename(staging, self._handoffs / handoff_id)
        except BaseException:
            remove_tree(staging, log=log, what="a failed hand-off's staging directory")
            with self._state:
                self._handing.pop(handoff_id, None)
            raise
        with self._state:
            self._handing.pop(handoff_id, None)
            self._handed.add(handoff_id)
            self._handed_bytes[handoff_id] = written
            self._handoffs_made += 1
        return HandedTables(root=str(self._root), handoff_id=handoff_id, tables=tuple(tables), nbytes=written)

    def release(self, handed: HandedTables) -> None:
        """the worker is done with a hand-off (its job returned or raised, or its worker died): it goes now, or at
        the first sweep after no worker maps it. Harmless twice.

        :param handed: the hand-off
        :ptype handed: HandedTables
        :return: nothing
        :rtype: None
        """
        with self._state:
            self._handed.discard(handed.handoff_id)
        self.sweep()

    @contextlib.contextmanager
    def handing(self, tables: Mapping[str, Any], *, max_bytes: int | None = None) -> Iterator[HandedTables]:
        """:meth:`hand` for a block, released (:meth:`release`) as it ends whichever way: wait in it for the job the
        hand-off is given to, so a worker that died releases it as surely as one that answered.

        :param tables: as :meth:`hand`
        :ptype tables: Mapping[str, Any]
        :param max_bytes: as :meth:`hand`
        :ptype max_bytes: int | None
        :return: the hand-off
        :rtype: Iterator[HandedTables]
        """
        handed = self.hand(tables, max_bytes=max_bytes)
        try:
            yield handed
        finally:
            self.release(handed)

    def sweep(self) -> list[str]:
        """remove every version neither current nor held, every result no open slot or claim owns, every hand-off
        released that no worker maps, and every staging directory whose writer is gone.

        :return: the versions retired
        :rtype: list[str]
        """
        gone: list[Path] = []
        with self._state:
            retired = self._sweep_versions(gone)
            self._sweep_segments(gone)
            self._sweep_handoffs(gone)
            self._retired += len(retired)
        # renamed out of their names under the lock (nothing opens them again); their files go after it
        for path in gone:
            remove_tree(path, log=log, what="a retired shared Arrow directory (a later sweep retries it)")
        if retired:
            log.info("shared Arrow versions retired", extra={"extra_data": {"versions": retired}})
        return retired

    def _released(self, slot_id: str) -> None:
        """a claimed result was closed: its files go.

        :param slot_id: the slot
        :ptype slot_id: str
        :return: nothing
        :rtype: None
        """
        with self._state:
            self._claimed.discard(slot_id)
        self.sweep()
        # counted until the sweep removed its files: a publish's room grows only once they are gone
        with self._state:
            self._claimed_bytes.pop(slot_id, None)

    def _committed_bytes(self) -> int:
        """the bytes the directory may be holding: what its files hold, except an open slot's and a hand-off's being
        written, each counted at its bound however much it has written. Called under ``_state``.

        :return: bytes
        :rtype: int
        """
        seen: set[tuple[int, int]] = set()
        held = sum(_directory_bytes(entry, seen) for entry in self._versions.iterdir() if entry.is_dir())
        held += sum(
            _directory_bytes(entry, seen)
            for entry in self._segments.iterdir()
            if entry.is_dir() and _slot_of(entry.name) not in self._slots
        )
        held += sum(
            _directory_bytes(entry, seen)
            for entry in self._handoffs.iterdir()
            if entry.is_dir() and _slot_of(entry.name) not in self._handing
        )
        return held + self._reserved_bytes()

    def _reserved_bytes(self) -> int:
        """the bytes open slots and hand-offs being written are counted at. Called under ``_state``.

        :return: bytes
        :rtype: int
        """
        return sum(self._slots.values()) + sum(self._handing.values())

    def _sweep_versions(self, gone: list[Path]) -> list[str]:
        """retire every version neither current nor held, and every leftover staging directory: each renamed out of
        its name into ``gone``, for the caller to remove once it lets go of ``_state``. Called under ``_state``.

        :param gone: what to remove
        :ptype gone: list[Path]
        :return: the versions retired
        :rtype: list[str]
        """
        retired: list[str] = []
        current = None if self._current is None else self._current.name
        for entry in list(self._versions.iterdir()):
            name = entry.name
            if not entry.is_dir() or name in (current, self._staging, self._pending):
                continue
            if name.startswith(_RETIRING_PREFIX):
                gone.append(entry)
                continue
            if not _retire_unless_held(entry, gone):
                continue
            if not name.startswith(_STAGING_PREFIX):
                retired.append(name)
        return retired

    def _sweep_segments(self, gone: list[Path]) -> None:
        """retire every result no open slot or claim owns, and every staging directory whose writer is gone, into
        ``gone`` (as :meth:`_sweep_versions`). Called under ``_state``.

        :param gone: what to remove
        :ptype gone: list[Path]
        :return: nothing
        :rtype: None
        """
        for entry in list(self._segments.iterdir()):
            name = entry.name
            if name.startswith(_RETIRING_PREFIX):
                gone.append(entry)
            elif name.startswith(_STAGING_PREFIX):
                # a writer still writing holds its lock; one that died, or whose write failed, does not
                descriptor = _locked(entry, fcntl.LOCK_EX)
                if descriptor is not None:
                    try:
                        _retire(entry, gone)
                    finally:
                        os.close(descriptor)
                elif not (entry / _LOCK_FILE).exists():
                    _retire(entry, gone)
            elif name not in self._slots and name not in self._claimed:
                _retire(entry, gone)

    def _sweep_handoffs(self, gone: list[Path]) -> None:
        """retire every hand-off released that no worker maps, and every staging directory no write of this process
        owns, into ``gone`` (as :meth:`_sweep_versions`); forget the bytes of each that is gone. Called under
        ``_state``.

        :param gone: what to remove
        :ptype gone: list[Path]
        :return: nothing
        :rtype: None
        """
        for entry in list(self._handoffs.iterdir()):
            name = entry.name
            if name.startswith(_RETIRING_PREFIX):
                gone.append(entry)
            elif name.startswith(_STAGING_PREFIX):
                # written by this process on a thread, never a worker: one this process is not writing is a leftover
                if _slot_of(name) not in self._handing:
                    _retire(entry, gone)
            elif name not in self._handed and name not in self._handing:
                # written and renamed into place by a hand() not yet done registering it is in _handing; a worker
                # mapping a released one holds its shared lock; one that finished or died does not
                _retire_unless_held(entry, gone)
        # a hand-off's bytes count in a publish's room until its directory is out of its name (removed just after)
        for name in [each for each in self._handed_bytes if each not in self._handed]:
            if not (self._handoffs / name).exists():
                del self._handed_bytes[name]

    def _write_version(self, name: str, tables: Mapping[str, Any], base: Path | None = None) -> None:
        """write a version under a staging name and rename it into place whole; on any failure keep nothing.

        :param name: the version
        :ptype name: str
        :param tables: its tables
        :ptype tables: Mapping[str, Any]
        :param base: the version directory to keep what ``tables`` does not replace from
        :ptype base: Path | None
        :return: nothing
        :rtype: None
        :raises SharedArrowFullError: when it would pass ``max_bytes``
        """
        staging_name = f"{_STAGING_PREFIX}{name}-{uuid.uuid7().hex[-8:]}"
        staging = self._versions / staging_name
        with self._state:
            self._staging = staging_name
            # what the directory holds but for open slots and hand-offs being written, as this write starts (its
            # staging directory not yet made)
            files = self._committed_bytes() - self._reserved_bytes()
            # the results claimed and hand-offs written before this write are in ``files``; one claimed or written
            # during it counts from then
            claimed_before = set(self._claimed_bytes)
            handed_before = set(self._handed_bytes)

        def room() -> int:
            # asked before every batch: a slot or hand-off reserved while this writes takes its room from this write,
            # never from the bound (each counts what this write has staged so far); a slot claimed or a hand-off
            # written while this writes stops counting at its bound, so its files count from then on instead
            with self._state:
                claimed = sum(size for slot, size in self._claimed_bytes.items() if slot not in claimed_before)
                handed = sum(size for each, size in self._handed_bytes.items() if each not in handed_before)
                return self._max_bytes - files - claimed - handed - self._reserved_bytes()

        try:
            staging.mkdir()
            _write_tables(staging, tables, room, base)
            (staging / _LOCK_FILE).touch()
            os.rename(staging, self._versions / name)
        except SharedArrowFullError:
            with self._state:
                self._refused += 1
            log.warning(
                "shared Arrow version refused: it would pass the store's bytes",
                extra={"extra_data": {"version": name, "max_bytes": self._max_bytes}},
            )
            raise
        finally:
            # renamed into place on success, so only a failed write's staging is still here to remove
            remove_tree(staging, log=log, what="a failed version's staging directory")
            with self._state:
                self._staging = None


@dataclass(frozen=True)
class SharedRead:
    """one read of a version through a :class:`SharedArrowReader`, valid for its block.

    :ivar version: the version read
    :ivar tables: table name -> the mapped table
    :ivar cursor: a DuckDB cursor with every table as a view over its mapped Arrow, no row copied
    :ivar prepared: what the reader's ``prepare`` returned for this version's connection; None without one
    """

    version: str
    tables: Mapping[str, Any]
    cursor: Any
    prepared: Any = None


class _HeldVersion:
    """a version a reader holds: its lease, its mapping, a DuckDB connection with a view per table, and its reads."""

    def __init__(
        self,
        lease: SharedLease,
        schema: str | None,
        config: Mapping[str, Any] | None,
        prepare: Callable[[Any], Any] | None,
        reuse: Mapping[tuple[int, int], Any] | None = None,
    ) -> None:
        self.lease = lease
        self.name = lease.version.name
        try:
            self.mapped = MappedTables(self.name, lease.version.path, lock=False, reuse=reuse)
            try:
                self.connection = self.mapped.connect(schema=schema, config=config)
                try:
                    self.prepared = None if prepare is None else prepare(self.connection)
                except BaseException:
                    self.connection.close()
                    raise
            except BaseException:
                self.mapped.close()
                raise
        except BaseException:
            lease.release()
            raise
        self.reads = 0
        self.superseded = False

    def cursor(self) -> Any:
        """a cursor of the connection with this version's tables registered on it (a registration is the connection's
        own, and the views made by :meth:`MappedTables.connect` name them).

        :return: the cursor; close it when done
        :rtype: duckdb.DuckDBPyConnection
        """
        cursor = self.connection.cursor()
        try:
            for name, table in self.mapped.tables.items():
                cursor.register(f"{_REGISTERED_PREFIX}{name}", table)
        except BaseException:
            cursor.close()
            raise
        return cursor

    def close(self) -> None:
        """close the connection, unmap the files and release the lease (whose release retires the version when it is
        not current and nobody else holds it).

        :return: nothing
        :rtype: None
        """
        try:
            self.connection.close()
            self.mapped.close()
        finally:
            self.lease.release()


class SharedArrowReader:
    """the current version read in place by the process that owns the store: a serving process answering from the
    same pages its workers map, with no copy of its own. Each read takes the version current as it opens and keeps it
    for its block, whatever is published meanwhile; a superseded version is let go when its last read ends (the
    store tells this reader after each publish), so the store holds two versions only while a read of the old one
    is open. Made by :meth:`SharedArrowStore.reader`.

    Blocking: a read that finds a new version maps it (one file per table or part); call :meth:`read` from a worker
    thread, never the event loop.

    :param store: the store
    :ptype store: SharedArrowStore
    :param schema: the schema each read's views are made in; None for the connection's default
    :ptype schema: str | None
    :param config: DuckDB settings for the reader's connection
    :ptype config: Mapping[str, Any] | None
    :param prepare: called once with each version's connection as it is mapped; its answer is each read's
        ``prepared``
    :ptype prepare: Callable[[duckdb.DuckDBPyConnection], Any] | None
    """

    def __init__(
        self,
        store: SharedArrowStore,
        *,
        schema: str | None = None,
        config: Mapping[str, Any] | None = None,
        prepare: Callable[[Any], Any] | None = None,
    ) -> None:
        self._store = store
        self._schema = schema
        self._config = dict(config or {})
        self._prepare = prepare
        self._lock = threading.Lock()
        self._held: _HeldVersion | None = None
        #: the files of the version let go that the current one shares, by (device, inode), for its first read to take
        #: (:class:`MappedTables`): mapped already, so keeping them keeps no page the current version does not hold
        self._lend: Mapping[tuple[int, int], Any] = {}
        self._closed = False
        self._reads = 0
        self._mapped = 0

    @property
    def reads(self) -> int:
        """reads served: a diagnostic, also in :meth:`status`, written only by this object.

        :return: the count
        :rtype: int
        """
        return self._reads

    @property
    def mapped(self) -> int:
        """versions mapped: a diagnostic, also in :meth:`status`, written only by this object.

        :return: the count
        :rtype: int
        """
        return self._mapped

    @contextlib.contextmanager
    def read(self) -> Iterator[SharedRead]:
        """read the current version for the block.

        :return: the read
        :rtype: Iterator[SharedRead]
        :raises LookupError: when nothing is published yet
        :raises RuntimeError: after :meth:`close`
        """
        let_go: _HeldVersion | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("the shared Arrow reader is closed")
            current = self._store.current
            if current is None:
                raise LookupError("no shared Arrow version is published yet")
            held = self._held
            if held is None or held.name != current.name:
                # the version it replaces (or what it left to lend) gives it every file the two share: only what
                # changed is mapped
                reuse = self._lend if held is None else held.mapped.by_inode
                held = _HeldVersion(self._store.lease(), self._schema, self._config, self._prepare, reuse)
                self._lend = {}
                self._mapped += 1
                let_go = self._supersede()
                self._held = held
            held.reads += 1
            self._reads += 1
            try:
                cursor = held.cursor()
            except BaseException:
                held.reads -= 1
                raise
        if let_go is not None:
            let_go.close()
        try:
            yield SharedRead(version=held.name, tables=held.mapped.tables, cursor=cursor, prepared=held.prepared)
        finally:
            cursor.close()
            with self._lock:
                held.reads -= 1
                done = held.superseded and held.reads == 0
            if done:
                held.close()

    def refresh(self) -> None:
        """let go of the version held when another is current now; a read still open keeps it until it ends.

        :return: nothing
        :rtype: None
        """
        with self._lock:
            current = self._store.current
            let_go = None
            if self._held is not None and (current is None or self._held.name != current.name):
                if current is not None:
                    self._lend = _shared_files(current.path, {**self._lend, **self._held.mapped.by_inode})
                let_go = self._supersede()
            elif self._held is None and self._lend and current is not None:
                self._lend = _shared_files(current.path, self._lend)
        if let_go is not None:
            let_go.close()

    def close(self) -> None:
        """let the version go (when its open reads end) and read no more. Harmless twice.

        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._closed = True
            self._lend = {}
            let_go = self._supersede()
        if let_go is not None:
            let_go.close()

    def status(self) -> dict[str, Any]:
        """what the reader holds, as plain JSON values.

        :return: the version held, its open reads, and the reads and mappings so far
        :rtype: dict[str, Any]
        """
        with self._lock:
            held = self._held
            return {
                "version": None if held is None else held.name,
                "open_reads": 0 if held is None else held.reads,
                "reads": self._reads,
                "mapped": self._mapped,
            }

    def _supersede(self) -> _HeldVersion | None:
        """stop handing out the held version; the caller closes what this returns (nothing while a read is open, its
        last read closes it then). Called under ``_lock``.

        :return: the version to close now, or None
        :rtype: _HeldVersion | None
        """
        held, self._held = self._held, None
        if held is None:
            return None
        held.superseded = True
        return held if held.reads == 0 else None

    def __enter__(self) -> SharedArrowReader:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.close()


def _slot_of(name: str) -> str:
    """the owner a result's or a hand-off's directory belongs to (its slot, or its hand-off): its name, or its
    staging name's owner.

    :param name: the directory's name
    :ptype name: str
    :return: the slot or hand-off
    :rtype: str
    """
    if name.startswith(_STAGING_PREFIX):
        return name[len(_STAGING_PREFIX) :].rsplit("-", 1)[0]
    return name


@dataclass(frozen=True)
class JobContext:
    """what a pool's job is handed beside its own arguments.

    :ivar version: the version it reads, mapped; None for a job run with ``read=False``
    :ivar result: the slot it writes its result tables into; None for a job run with no ``result_bytes``
    :ivar tables: the tables it was handed (``tables=``), mapped read-only for the job's run; None for none
    """

    version: MappedVersion | None
    result: SegmentSlot | None
    tables: MappedTables | None = None


@dataclass(frozen=True)
class JobResult:
    """what a pool's job gave back.

    :ivar value: what the job returned (pickled back: keep it small)
    :ivar segment: the result tables it wrote, mapped here and the caller's to close; None when it wrote none or was
        given no slot
    """

    value: Any
    segment: MappedSegment | None


def _private_bytes() -> int:
    """this process's private memory: its anonymous resident pages (Linux ``RssAnon``), never the shared files it
    maps; its peak resident memory where there is no ``/proc`` (a laptop, which counts mapped pages too).

    :return: bytes
    :rtype: int
    """
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        # NOSILENT: no /proc: the peak below
        pass
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if os.uname().sysname == "Darwin" else peak * 1024


def _sendable(error: BaseException) -> BaseException:
    """an error as a worker sends it back: itself when it pickles, else its text with its type named.

    :param error: the error
    :ptype error: BaseException
    :return: what is sent
    :rtype: BaseException
    """
    try:
        pickle.dumps(error)
    except pickle.PicklingError, TypeError, AttributeError:
        # NOSILENT: an error that will not pickle goes back as its text, its type named in it
        return RuntimeError(f"{type(error).__name__}: {error}")
    return error


def _run_job(
    root: str,
    version: str | None,
    slot: SegmentSlot | None,
    job: Callable[..., Any],
    args: tuple[Any, ...],
    handed: HandedTables | None = None,
) -> tuple[Any, BaseException | None, int, int]:
    """a worker's job: ``job(context, *args)``, over the version leased for it, the tables handed to it, and with the
    slot reserved for it.

    :param root: the store's directory
    :ptype root: str
    :param version: the version the serving process leased for the job; None for none
    :ptype version: str | None
    :param slot: the slot reserved for its result; None for none
    :ptype slot: SegmentSlot | None
    :param job: the job, a module function
    :ptype job: Callable[..., Any]
    :param args: its arguments after the context
    :ptype args: tuple[Any, ...]
    :param handed: the tables the serving process handed it; None for none
    :ptype handed: HandedTables | None
    :return: its value; what it raised (None for nothing); this worker's private bytes after; its process id
    :rtype: tuple[Any, BaseException | None, int, int]
    """
    value: Any = None
    error: BaseException | None = None
    try:
        with contextlib.ExitStack() as held:
            tables = None if handed is None else held.enter_context(handed.mapped())
            mapped = None if version is None else held.enter_context(read_version(root, version))
            value = job(JobContext(version=mapped, result=slot, tables=tables), *args)
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- sent back and raised in the serving process, which handles it as the job's own failure
        contain(exc, log=log, what="a shared Arrow job failed; raised in the serving process", level="warning")
        error = _sendable(exc)
    return value, error, _private_bytes(), os.getpid()


def _spawned() -> int:
    """a fresh worker's first task, which makes the pool spawn it before any job: its process id.

    :return: the process id
    :rtype: int
    """
    return os.getpid()


class _Run:
    """one run's lease, hand-off and slot, each released exactly once whichever way the run ends: the lease and the
    hand-off when the worker is done with them (its job returned or raised, or the worker died), the slot claimed by
    the caller or discarded once the worker is done with it (the job failed, the caller stopped waiting, the worker
    died).

    :param store: the store
    :ptype store: SharedArrowStore
    :param lease: the version's lease; None for none
    :ptype lease: SharedLease | None
    """

    def __init__(self, store: SharedArrowStore, lease: SharedLease | None) -> None:
        self._store = store
        self._lease = lease
        self.slot: SegmentSlot | None = None
        self.handed: HandedTables | None = None

    def worker_done(self, _: Future[Any]) -> None:
        """the worker is done with the version and the tables it was handed: the lease and the hand-off go. Run as
        the job's future's callback, on the executor's thread, whichever way the job ended.

        :return: nothing
        :rtype: None
        """
        if self._lease is not None:
            self._lease.release()
        handed = self.handed
        if handed is not None:
            try:
                self._store.release(handed)
            except Exception as exc:  # noqa: BLE001 -- a callback has no caller to raise to; contained, so it is counted
                contain(exc, log=log, what="a shared Arrow hand-off could not be released")

    def abandon(self, future: Future[Any]) -> None:
        """nobody will claim the slot: discarded once the worker is done with it (now, when it already is).

        :param future: the job's future
        :ptype future: Future[Any]
        :return: nothing
        :rtype: None
        """
        slot = self.slot
        if slot is not None:
            future.add_done_callback(lambda _: self._discard(slot))

    def _discard(self, slot: SegmentSlot) -> None:
        """discard ``slot``, logging a failure: it runs as a future's callback, where a raise would reach no caller.

        :param slot: the slot
        :ptype slot: SegmentSlot
        :return: nothing
        :rtype: None
        """
        try:
            self._store.discard(slot)
        except Exception as exc:  # noqa: BLE001 -- a callback has no caller to raise to; contained, so it is counted
            contain(exc, log=log, what="a shared Arrow result could not be discarded")

    def release_all(self) -> None:
        """release what was taken, the job never submitted.

        :return: nothing
        :rtype: None
        """
        if self._lease is not None:
            self._lease.release()
        if self.handed is not None:
            self._store.release(self.handed)
        if self.slot is not None:
            self._store.discard(self.slot)


def _off_loop(work: Callable[..., object], *args: Any, what: str) -> None:
    """run ``work(*args)`` on a thread, never the event loop this is called on, and log what it raises.

    :param work: the blocking work (a discard sweeps the directory; a close removes files)
    :ptype work: Callable[..., object]
    :param args: its arguments
    :ptype args: Any
    :param what: what failed, for the log
    :ptype what: str
    :return: nothing
    :rtype: None
    """
    running = asyncio.get_running_loop().run_in_executor(None, work, *args)
    running.add_done_callback(functools.partial(_said_if_failed, what=what))


def _said_if_failed(done: asyncio.Future[Any], *, what: str) -> None:
    """contain what a thread's work raised, so a failure is logged and counted, not left to a warning at collection.

    :param done: the work's future, done
    :ptype done: asyncio.Future[Any]
    :param what: what failed
    :ptype what: str
    :return: nothing
    :rtype: None
    """
    failed = None if done.cancelled() else done.exception()
    if isinstance(failed, Exception):
        contain(failed, log=log, what=what)


def _abandon_submitted(submitting: asyncio.Future[Any]) -> None:
    """a submission whose caller stopped waiting: once it is done, its slot is discarded when its worker is done.

    :param submitting: the submission, done
    :ptype submitting: asyncio.Future[tuple[_Run, Future[Any]]]
    :return: nothing
    :rtype: None
    """
    if not submitting.cancelled() and submitting.exception() is None:
        run, future = submitting.result()
        # a discard sweeps the directory: on a thread, never the event loop this is called on
        _off_loop(run.abandon, future, what="an abandoned shared Arrow result could not be discarded")


def _close_claimed(claiming: asyncio.Future[Any]) -> None:
    """a claim whose caller stopped waiting: once it is done, the result it mapped is closed (its files go).

    :param claiming: the claim, done
    :ptype claiming: asyncio.Future[MappedSegment | None]
    :return: nothing
    :rtype: None
    """
    if not claiming.cancelled() and claiming.exception() is None:
        segment = claiming.result()
        if segment is not None:
            # closing removes its files: on a thread, never the event loop this is called on
            _off_loop(segment.close, what="an abandoned claimed shared Arrow result could not be closed")


class SharedArrowPool:
    """spawned, long-lived worker processes running jobs over a store's versions and handing results back through it.

    A job is a module function ``job(context: JobContext, *args)`` whose return value pickles back to the serving
    process (keep it small: tables go back through ``context.result``). A worker whose private memory passes
    ``private_bytes`` after a job has its pool replaced: the old pool finishes what it was given and goes, the next
    job starts a fresh one.

    :param store: the store
    :ptype store: SharedArrowStore
    :param workers: the processes, side by side
    :ptype workers: int
    :param private_bytes: a worker holding more private memory than this after a job is replaced
    :ptype private_bytes: int
    :param name: what the pool works, for its log
    :ptype name: str
    :param initializer: run once in each worker as it starts (to size its thread pools to its CPU share, say)
    :ptype initializer: Callable[..., object] | None
    :param initargs: the initializer's arguments
    :ptype initargs: tuple[Any, ...]
    """

    def __init__(
        self,
        store: SharedArrowStore,
        *,
        workers: int,
        private_bytes: int,
        name: str,
        initializer: Callable[..., object] | None = None,
        initargs: tuple[Any, ...] = (),
    ) -> None:
        if workers < 1:
            raise ValueError("a pool needs a worker")
        self._store = store
        self._workers = workers
        self._private_bytes = private_bytes
        self._name = name
        self._initializer = initializer
        self._initargs = initargs
        self._pool: ProcessPoolExecutor | None = None
        self._stale = False
        #: held while a thread picks, makes or shuts a pool: never by the event loop
        self._lock = threading.Lock()
        self._jobs = 0
        self._replaced = 0

    @property
    def jobs(self) -> int:
        """jobs finished: a diagnostic, also in :meth:`status`, written only by this object.

        :return: the count
        :rtype: int
        """
        return self._jobs

    @property
    def replaced(self) -> int:
        """pools replaced: a diagnostic, also in :meth:`status`, written only by this object.

        :return: the count
        :rtype: int
        """
        return self._replaced

    async def run(
        self,
        job: Callable[..., Any],
        *args: Any,
        read: bool | str = True,
        result_bytes: int | None = None,
        tables: Mapping[str, Any] | None = None,
        tables_bytes: int | None = None,
    ) -> JobResult:
        """run ``job(context, *args)`` in a worker.

        :param job: the job, a module function (pickled by name)
        :ptype job: Callable[..., Any]
        :param args: its arguments after the context (pickled)
        :ptype args: Any
        :param read: the version it reads: True for the one current at submission, a name for that one, False for
            none
        :ptype read: bool | str
        :param result_bytes: the most its result tables may take, reserved for it; None for a job that writes none
        :ptype result_bytes: int | None
        :param tables: tables to hand it (:meth:`SharedArrowStore.hand`), mapped read-only as ``context.tables``:
            written on a thread, never pickled; None for none
        :ptype tables: Mapping[str, Any] | None
        :param tables_bytes: the most the handed tables' files may take; None for their bytes and the file's own
        :ptype tables_bytes: int | None
        :return: its value, and its result tables mapped here (the caller closes them)
        :rtype: JobResult
        :raises LookupError: when nothing is published, or the version named is gone
        :raises SharedArrowFullError: when the store has no room for ``result_bytes`` or the handed tables
        :raises Exception: what the job raised; ``BrokenProcessPool`` when its worker died (the pool is replaced)
        """
        # leased, reserved and submitted on a thread: a fresh pool spawns its workers there, and a lease waits on a
        # sweep. A caller that stops waiting meanwhile leaves the thread to finish; what it took is let go then
        submitting = asyncio.ensure_future(
            asyncio.to_thread(self._submit, read, result_bytes, job, args, tables, tables_bytes)
        )
        try:
            run, future = await asyncio.shield(submitting)
        except asyncio.CancelledError:
            submitting.add_done_callback(_abandon_submitted)
            raise
        claimed = False
        try:
            try:
                value, error, private, worker = await asyncio.wrap_future(future)
            except BrokenProcessPool:
                if not self._stale:
                    log.warning(
                        "shared Arrow worker died; its pool is replaced",
                        extra={"extra_data": {"pool": self._name}},
                    )
                self._stale = True
                raise
            self._jobs += 1
            if private > self._private_bytes:
                log.info(
                    "shared Arrow worker past its private memory; its pool is replaced",
                    extra={"extra_data": {"pool": self._name, "worker": worker, "private_bytes": private}},
                )
                self._stale = True
            if error is not None:
                raise error
            segment = None
            if run.slot is not None:
                # the claim maps the result on a thread; a caller that stops waiting meanwhile has it closed there
                claiming = asyncio.ensure_future(asyncio.to_thread(self._store.claim, run.slot))
                try:
                    segment = await asyncio.shield(claiming)
                except asyncio.CancelledError:
                    claiming.add_done_callback(_close_claimed)
                    raise
            claimed = True
            return JobResult(value=value, segment=segment)
        finally:
            if not claimed:
                # a discard sweeps the directory, and runs at once on a job already done: on a thread, never the loop
                _off_loop(run.abandon, future, what="a failed shared Arrow job's result could not be discarded")

    def _submit(
        self,
        read: bool | str,
        result_bytes: int | None,
        job: Callable[..., Any],
        args: tuple[Any, ...],
        tables: Mapping[str, Any] | None = None,
        tables_bytes: int | None = None,
    ) -> tuple[_Run, Future[tuple[Any, BaseException | None, int, int]]]:
        """lease the version, hand the tables over, reserve the slot and submit the job to the current pool, or a
        fresh one; on a thread.

        :param read: as :meth:`run`
        :ptype read: bool | str
        :param result_bytes: as :meth:`run`
        :ptype result_bytes: int | None
        :param job: the job
        :ptype job: Callable[..., Any]
        :param args: its arguments
        :ptype args: tuple[Any, ...]
        :param tables: as :meth:`run`
        :ptype tables: Mapping[str, Any] | None
        :param tables_bytes: as :meth:`run`
        :ptype tables_bytes: int | None
        :return: the run (its lease, hand-off and slot) and the job's future (the lease and the handed tables are released when it is
            done)
        :rtype: tuple[_Run, Future[tuple[Any, BaseException | None, int, int]]]
        """
        lease = None if read is False else self._store.lease(None if read is True else read)
        run = _Run(self._store, lease)
        try:
            if tables is not None:
                run.handed = self._store.hand(tables, max_bytes=tables_bytes)
            if result_bytes is not None:
                run.slot = self._store.reserve(result_bytes)
            with self._lock:
                if self._pool is not None and self._stale:
                    self._pool.shutdown(wait=False)
                    self._pool = None
                    self._replaced += 1
                if self._pool is None:
                    self._pool = ProcessPoolExecutor(
                        max_workers=self._workers,
                        mp_context=multiprocessing.get_context("spawn"),
                        initializer=self._initializer,
                        initargs=self._initargs,
                    )
                    self._stale = False
                    for _ in range(self._workers):
                        self._pool.submit(_spawned)
                future = self._pool.submit(
                    _run_job,
                    str(self._store.root),
                    None if lease is None else lease.version.name,
                    run.slot,
                    job,
                    args,
                    run.handed,
                )
        except BaseException:
            run.release_all()
            raise
        future.add_done_callback(run.worker_done)
        return run, future

    async def close(self) -> None:
        """shut the workers down, waiting for them: what they map and hold goes back to the system.

        :return: nothing
        :rtype: None
        """
        await asyncio.to_thread(self._close)

    def _close(self) -> None:
        """:meth:`close`, on a thread.

        :return: nothing
        :rtype: None
        """
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)
        self._store.sweep()
