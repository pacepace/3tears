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
  (:func:`read_version`) and run DuckDB over it without copying (:meth:`MappedTables.connect`). Each version is a
  directory holding a manifest that names its files, staged under a hidden name and renamed into place whole; the
  newest is current. The files themselves live in the parts directory, one directory per publish holding the files it
  wrote, never written again, and a version made from another (``base=``) names the other's files it keeps: a publish
  writes the files it changes and one manifest, whatever the version holds besides (:func:`version_files`).
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
hand-off released that no worker maps, and every staging directory whose writer is gone. A part file goes when the
last version naming it is retired: a reader holding a version holds every file its manifest names. A new store removes
whatever the directory held that no live process holds (a container restarted in the same pod keeps its
``emptyDir``), part files no version it kept names among it, and a version written before manifests (a directory
holding its files itself) is read, kept while held and removed as any other. There is no timer and no TTL.

**A publish does what it changes.** The sweep a publish leaves -- readers told, superseded versions retired, their
files removed -- runs after it returns, on the store's own thread: one at a time, a publish meanwhile asking for one
more, never a pile (:meth:`SharedArrowStore.settle` waits for it). A publish that finds no room sweeps first itself.

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
import json
import mmap
import multiprocessing
import os
import pickle
import resource
import threading
import uuid
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Final

from threetears.observe import contain, get_logger, remove_tree


# pyarrow alone carries the store, its slots and hand-offs, and the pool: a process that never opens DuckDB (the hub,
# offloading its broker's work) needs no duckdb, and only the calls that open it say so
try:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    _ARROW_EXTRA: ImportError | None = None
except ImportError as exc:  # NOSILENT: the snapshot extra is optional; using shared Arrow says so
    _ARROW_EXTRA = exc

try:
    import duckdb

    # DuckDB scans a registered Arrow table through pyarrow.dataset's Scanner, importing the module (and what it
    # imports) at the first scan in the process, 25-70 ms inside a serving thread's read; imported here, at boot
    import pyarrow.dataset  # noqa: F401

    _DUCKDB_EXTRA: ImportError | None = None
except ImportError as exc:  # NOSILENT: DuckDB over shared Arrow is optional; the calls that open it say so
    _DUCKDB_EXTRA = exc

__all__ = [
    "COLUMN_TOKENS_KEY",
    "FileTokens",
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
    "VersionFile",
    "column_print",
    "epochs_version",
    "file_tokens",
    "merge_parts",
    "merged_rows",
    "plain_name",
    "private_bytes",
    "query_batches",
    "read_version",
    "stamped",
    "stamping",
    "version_files",
]

log = get_logger(__name__)

#: the store's four parts: published versions (each a manifest), the files they name, workers' results, and the tables
#: handed to one job each
_VERSIONS_DIR: Final = "versions"
_PARTS_DIR: Final = "parts"
_SEGMENTS_DIR: Final = "segments"
_HANDOFFS_DIR: Final = "handoffs"

#: the file in each version's directory naming its files: a header line, then one line a file,
#: ``{file name}\t{its directory under the parts directory}\t{its bytes}``
_MANIFEST_FILE: Final = "MANIFEST"
_MANIFEST_HEADER: Final = "shared-arrow manifest 1\n"

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


def _require_arrow() -> None:
    """raise, naming the extra, when pyarrow could not be imported: the store, its slots, hand-offs and pool need it.

    :return: nothing
    :rtype: None
    :raises ImportError: when pyarrow is not installed
    """
    if _ARROW_EXTRA is not None:
        raise ImportError(f"shared Arrow needs pyarrow: install 3tears[snapshot] ({_ARROW_EXTRA})")


def _require_duckdb() -> None:
    """raise, naming the extra, when duckdb or pyarrow could not be imported: the calls that open DuckDB over the
    mapped tables (:meth:`MappedTables.connect`, :meth:`SharedArrowStore.reader`,
    :meth:`SharedArrowStore.publish_query`) need both.

    :return: nothing
    :rtype: None
    :raises ImportError: when the snapshot extra is not installed
    """
    missing = _ARROW_EXTRA or _DUCKDB_EXTRA
    if missing is not None:
        raise ImportError(f"shared Arrow needs duckdb and pyarrow: install 3tears[snapshot] ({missing})")


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
    """a version name for the data at ``epochs`` (scope -> epoch): the same epochs always
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


def _retire(path: Path, gone: list[tuple[Path, Path]]) -> None:
    """rename ``path`` out of its name (:func:`_retiring`) and add it to ``gone`` with the name it had, to be removed
    after the lock.

    :param path: the directory
    :ptype path: Path
    :param gone: what to remove: (the name it had, its name now)
    :ptype gone: list[tuple[Path, Path]]
    :return: nothing
    :rtype: None
    """
    renamed = _retiring(path)
    if renamed is not None:
        gone.append((path, renamed))


def _retire_unless_held(entry: Path, gone: list[tuple[Path, Path]]) -> bool:
    """retire a directory (into ``gone``) unless a reader holds its shared lock: a version a lease or a worker reads,
    a hand-off a worker maps. A directory with no lock file is no reader's and goes.

    :param entry: the directory
    :ptype entry: Path
    :param gone: what to remove (:func:`_retire`)
    :ptype gone: list[tuple[Path, Path]]
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


def _directory_files(path: Path) -> dict[tuple[int, int], int]:
    """the files directly in a directory, by (device, inode), with their bytes: each file once however many names
    it has there.

    :param path: the directory
    :ptype path: Path
    :return: (device, inode) -> bytes; empty when it is gone
    :rtype: dict[tuple[int, int], int]
    """
    files: dict[tuple[int, int], int] = {}
    with contextlib.suppress(FileNotFoundError):  # NOSILENT: a directory removed meanwhile holds nothing
        for entry in path.iterdir():
            with contextlib.suppress(FileNotFoundError):  # NOSILENT: likewise a file
                stat = entry.stat()
                files.setdefault((stat.st_dev, stat.st_ino), stat.st_size)
    return files


def _directory_bytes(path: Path, seen: set[tuple[int, int]] | None = None) -> int:
    """the bytes the files directly in a directory hold, each file once however many names it has.

    A claimed result linked into a version as a part is one file under two names, whose pages exist once: ``seen``,
    shared across directories, counts each file (by device and inode) the first time it is met.

    :param path: the directory
    :ptype path: Path
    :param seen: the files counted already, by (device, inode); None to count this directory alone
    :ptype seen: set[tuple[int, int]] | None
    :return: bytes; 0 when it is gone
    :rtype: int
    """
    counted = set() if seen is None else seen
    total = 0
    for key, size in _directory_files(path).items():
        if key not in counted:
            counted.add(key)
            total += size
    return total


class _HeldFiles:
    """the store's own tally of the files its directories hold, so :meth:`SharedArrowStore.status` reads numbers and
    never the file system: each directory's files recorded as it takes its final name (a version published, the part
    files a publish wrote, a hand-off written, a result claimed, a directory found at the start) and forgotten once it
    is removed; a part file is dropped from its directory's record as it is removed (:meth:`drop`).

    Exact as :meth:`SharedArrowStore.held_bytes` counts: a file is counted once however many directories hold a link
    to it (a claimed result linked as a part), by (device, inode), and its bytes go only when the last directory holding
    it goes. Files never change after their directory takes its name, so what a directory held then is what it holds
    until it is removed, or a part file of it is. An inode met again at another size is a file the system reused the
    number for (the old one removed while its directory's removal was still being recorded): the newer size is the true
    one. Mutated under the store's ``_state``; :attr:`view` is replaced whole on each change, so a reader takes it
    without the lock.

    :ivar view: (bytes held, the versions' names, the hand-offs held), each as of the last change
    """

    def __init__(self, versions: Path, handoffs: Path) -> None:
        self._versions = versions
        self._handoffs = handoffs
        #: directory -> its files, (device, inode) -> bytes
        self._dirs: dict[Path, dict[tuple[int, int], int]] = {}
        #: (device, inode) -> [directories holding it, its bytes]
        self._links: dict[tuple[int, int], list[int]] = {}
        #: the versions and hand-offs recorded under their own names: what :attr:`view` lists, kept as directories are
        #: recorded, renamed and forgotten, so a change never walks every directory recorded (a part directory each
        #: publish whose files a version still names)
        self._listed: set[Path] = set()
        self._bytes = 0
        self.view: tuple[int, tuple[str, ...], int] = (0, (), 0)

    @property
    def bytes(self) -> int:
        """the bytes recorded. Read under the store's ``_state``.

        :return: bytes
        :rtype: int
        """
        return self._bytes

    def __contains__(self, path: Path) -> bool:
        """whether a directory is recorded.

        :param path: the directory
        :ptype path: Path
        :return: whether it is
        :rtype: bool
        """
        return path in self._dirs

    def add(self, path: Path, files: Mapping[tuple[int, int], int]) -> None:
        """record a directory's files (replacing what it was recorded with).

        :param path: the directory
        :ptype path: Path
        :param files: its files (:func:`_directory_files`)
        :ptype files: Mapping[tuple[int, int], int]
        :return: nothing
        :rtype: None
        """
        self._forget(path)
        self._dirs[path] = dict(files)
        for key, size in files.items():
            link = self._links.get(key)
            if link is None:
                self._links[key] = [1, size]
                self._bytes += size
            else:
                link[0] += 1
                if link[1] != size:
                    self._bytes += size - link[1]
                    link[1] = size
        if self._lists(path):
            self._listed.add(path)
        self._viewed()

    def drop(self, path: Path, keys: Sequence[tuple[int, int]]) -> int | None:
        """files of a recorded directory were removed: each goes from its record, its bytes with it when no other
        directory holds it.

        :param path: the directory
        :ptype path: Path
        :param keys: the files removed, by (device, inode)
        :ptype keys: Sequence[tuple[int, int]]
        :return: the files its record holds now; None when it is not recorded
        :rtype: int | None
        """
        files = self._dirs.get(path)
        if files is None:
            return None
        for key in keys:
            if files.pop(key, None) is not None:
                self._unlinked(key)
        self._viewed()
        return len(files)

    def moved(self, old: Path, new: Path) -> None:
        """a directory was renamed (retired out of its name): its files are recorded under its new name.

        :param old: its name before
        :ptype old: Path
        :param new: its name now
        :ptype new: Path
        :return: nothing
        :rtype: None
        """
        files = self._dirs.pop(old, None)
        if files is not None:
            self._dirs[new] = files
            self._listed.discard(old)
            if self._lists(new):
                self._listed.add(new)
            self._viewed()

    def forget(self, path: Path) -> None:
        """a directory was removed: its files go, each whose last directory it was.

        :param path: the directory
        :ptype path: Path
        :return: nothing
        :rtype: None
        """
        if self._forget(path):
            self._viewed()

    def _forget(self, path: Path) -> bool:
        """drop a directory's record, and each file's bytes whose last directory it was; :attr:`view` unchanged.

        :param path: the directory
        :ptype path: Path
        :return: whether it was recorded
        :rtype: bool
        """
        files = self._dirs.pop(path, None)
        if files is None:
            return False
        self._listed.discard(path)
        for key in files:
            self._unlinked(key)
        return True

    def _unlinked(self, key: tuple[int, int]) -> None:
        """one directory's link to a file is gone: the file's bytes go when it was the last.

        :param key: the file, by (device, inode)
        :ptype key: tuple[int, int]
        :return: nothing
        :rtype: None
        """
        link = self._links[key]
        link[0] -= 1
        if link[0] == 0:
            del self._links[key]
            self._bytes -= link[1]

    def _lists(self, path: Path) -> bool:
        """whether :attr:`view` lists a directory: a version or a hand-off under its own name.

        :param path: the directory
        :ptype path: Path
        :return: whether it does
        :rtype: bool
        """
        return path.parent in (self._versions, self._handoffs) and not path.name.startswith(".")

    def _viewed(self) -> None:
        """replace :attr:`view` with what is recorded now.

        :return: nothing
        :rtype: None
        """
        versions = tuple(sorted(path.name for path in self._listed if path.parent == self._versions))
        handoffs = sum(1 for path in self._listed if path.parent == self._handoffs)
        self.view = (self._bytes, versions, handoffs)


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


def _write_file(
    path: Path,
    schema: Any,
    batches: Iterator[Any],
    room: Callable[[], int],
    *,
    what: str,
    reclaim: Callable[[], object] | None = None,
) -> os.stat_result:
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
    :param reclaim: called once when a batch would pass ``room``, before the batch is refused (a sweep: what a publish
        left to the store's thread may not have gone yet); None to refuse at once
    :ptype reclaim: Callable[[], object] | None
    :return: the file written, as ``os.stat`` reads it
    :rtype: os.stat_result
    :raises SharedArrowFullError: when it would take more than ``room``
    :raises ValueError: when a batch's columns are not ``schema``
    """
    options = ipc.IpcWriteOptions(compression=None)
    with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, schema, options=options) as writer:
        for batch in batches:
            if not batch.schema.equals(schema):
                raise ValueError(f"{what}: a batch's columns are not the table's ({batch.schema} vs {schema})")
            if sink.tell() + batch.nbytes > room():
                if reclaim is not None:
                    reclaim()
                if sink.tell() + batch.nbytes > room():
                    raise SharedArrowFullError(f"{what} would pass the bytes it may take")
            writer.write_batch(batch)
    return os.stat(path)


def _file_schema(path: Path) -> Any:
    """the columns of an Arrow IPC file.

    :param path: the file
    :ptype path: Path
    :return: the schema
    :rtype: pa.Schema
    """
    with pa.memory_map(str(path), "r") as mapped:
        return ipc.open_file(mapped).schema


def _write_tables(directory: Path, tables: Mapping[str, Any], room: Callable[[], int]) -> int:
    """write each table as uncompressed Arrow IPC in ``directory`` (a result's or a hand-off's: a directory holding its
    files itself), refusing a batch that would pass ``room``.

    A table given whole is one file; one given as :class:`Parts` is a schema file and a file per part (a part given as
    None is no file).

    :param directory: where
    :ptype directory: Path
    :param tables: name -> a table, a reader of its batches, what opens either, or :class:`Parts`
    :ptype tables: Mapping[str, Any]
    :param room: the bytes every new file together may take, now (asked before each batch)
    :ptype room: Callable[[], int]
    :return: the bytes written
    :rtype: int
    :raises SharedArrowFullError: when they would take more than ``room``
    :raises ValueError: when a table or part name is not one plain path segment, or a part's columns are not its
        table's
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
            written += _write_file(directory / schema_file, source.schema, iter(()), left, what=name).st_size
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
                written += _write_file(path, source.schema, batches, left, what=f"table {name!r} part {part!r}").st_size
        else:
            schema, batches = _batches(source)
            path = directory / f"{name}{_TABLE_SUFFIX}"
            written += _write_file(path, schema, batches, left, what=f"table {name!r}").st_size
    return written


@dataclass(frozen=True)
class _Manifest:
    """a version's files as its ``MANIFEST`` names them, held by the store for every version it holds: per table, each
    file's line (``{file name}\\t{its part directory}\\t{its bytes}\\n``), those lines joined, and the bytes its files
    hold. A version made from another shares every table it leaves alone with it -- the same mappings, never copied --
    so a publish copies only the tables it changes, and a sweep compares only the tables that differ.

    :ivar lines: table -> file name -> its line
    :ivar text: table -> its lines, joined in the order of ``lines``
    :ivar nbytes: table -> the bytes its files hold
    :ivar file_bytes: the bytes of the ``MANIFEST`` file itself
    """

    lines: Mapping[str, Mapping[str, str]]
    text: Mapping[str, str]
    nbytes: Mapping[str, int]
    file_bytes: int = 0

    def joined(self) -> str:
        """the ``MANIFEST`` file's text: the header, then every table's lines.

        :return: the text
        :rtype: str
        """
        return _MANIFEST_HEADER + "".join(self.text.values())


def _line_of(file_name: str, part_dir: str, size: int) -> str:
    """a manifest's line for one file.

    :param file_name: the file's name, as a version names it (``{table}@{part}.arrow``, ``{table}.arrow``)
    :ptype file_name: str
    :param part_dir: its directory under the parts directory
    :ptype part_dir: str
    :param size: its bytes
    :ptype size: int
    :return: the line
    :rtype: str
    """
    return f"{file_name}\t{part_dir}\t{size}\n"


def _line_fields(line: str) -> tuple[str, str, int]:
    """a manifest line's file name, part directory and bytes.

    :param line: the line
    :ptype line: str
    :return: (file name, part directory, bytes)
    :rtype: tuple[str, str, int]
    """
    name, part_dir, size = line.rstrip("\n").split("\t")
    return name, part_dir, int(size)


def _line_rel(line: str) -> str:
    """the file a manifest line names, as a path under the parts directory (``{part dir}/{file name}``).

    :param line: the line
    :ptype line: str
    :return: the path, relative to the parts directory
    :rtype: str
    """
    name, part_dir, _ = _line_fields(line)
    return f"{part_dir}/{name}"


def _parsed_manifest(text: str) -> _Manifest:
    """a ``MANIFEST`` file's text as the store holds it.

    :param text: the text, its header first
    :ptype text: str
    :return: the manifest
    :rtype: _Manifest
    :raises ValueError: for a line naming a file that is no table's
    """
    lines: dict[str, dict[str, str]] = {}
    nbytes: dict[str, int] = {}
    for line in text.splitlines(keepends=True)[1:]:
        name, _, size = _line_fields(line)
        table = _table_of(name)
        if table is None:
            raise ValueError(f"a shared Arrow manifest names a file that is no table's: {name!r}")
        lines.setdefault(table, {})[name] = line
        nbytes[table] = nbytes.get(table, 0) + size
    joined = {table: "".join(held.values()) for table, held in lines.items()}
    return _Manifest(lines=lines, text=joined, nbytes=nbytes, file_bytes=len(text))


def _manifest_text(path: Path) -> str | None:
    """the text of a version directory's ``MANIFEST``, or None for a directory holding its files itself (a result, a
    hand-off, a version written before manifests) or gone.

    :param path: the directory
    :ptype path: Path
    :return: the text, or None
    :rtype: str | None
    :raises ValueError: for a manifest this code does not read
    """
    try:
        text = (path / _MANIFEST_FILE).read_text()
    except FileNotFoundError:
        # NOSILENT: no manifest is a directory holding its files itself (or one gone): the caller lists it
        return None
    if not text.startswith(_MANIFEST_HEADER):
        raise ValueError(f"shared Arrow manifest {path / _MANIFEST_FILE} is not one this code reads")
    return text


def _mapped_files(path: Path) -> list[tuple[str, str, Any]]:
    """the table files of a directory, in name order: each one's name, its path, and what identifies it among mapped
    files (:attr:`MappedTables.by_inode`): its manifest line for a version's file (a line names one file, never
    reused), its (device, inode) for a file the directory holds itself.

    :param path: the directory
    :ptype path: Path
    :return: (file name, path, identity), sorted by name
    :rtype: list[tuple[str, str, Any]]
    """
    text = _manifest_text(path)
    if text is None:
        # every file of a directory is on its device; a directory entry carries its inode (no stat a file)
        device = path.stat().st_dev
        with os.scandir(path) as listing:
            return sorted(
                (entry.name, entry.path, (device, entry.inode()))
                for entry in listing
                if _table_of(entry.name) is not None
            )
    parts = f"{path.parent.parent / _PARTS_DIR}{os.sep}"
    files: list[tuple[str, str, Any]] = []
    for line in text.splitlines(keepends=True)[1:]:
        name, part_dir, _ = _line_fields(line)
        files.append((name, f"{parts}{part_dir}{os.sep}{name}", line))
    # names are unique: the sort never compares further
    files.sort(key=lambda file: file[0])
    return files


@dataclass(frozen=True)
class VersionFile:
    """one file a version holds (:func:`version_files`).

    :ivar path: the file, to map or open read-only; never written, renamed or removed by a reader
    :ivar nbytes: its bytes
    """

    path: Path
    nbytes: int


def _files_of(path: Path) -> dict[str, VersionFile]:
    """the table files of a version's directory (:func:`version_files`), by the name the version gives each.

    :param path: the directory
    :ptype path: Path
    :return: file name -> the file
    :rtype: dict[str, VersionFile]
    """
    text = _manifest_text(path)
    if text is None:
        with os.scandir(path) as listing:
            return {
                entry.name: VersionFile(Path(entry.path), entry.stat().st_size)
                for entry in listing
                if _table_of(entry.name) is not None
            }
    parts = path.parent.parent / _PARTS_DIR
    files: dict[str, VersionFile] = {}
    for line in text.splitlines(keepends=True)[1:]:
        name, part_dir, size = _line_fields(line)
        files[name] = VersionFile(parts / part_dir / name, size)
    return files


def version_files(root: Path | str, version: str) -> dict[str, VersionFile]:
    """every table file a version holds, by the name the version gives it: ``{table}.arrow`` for a table written whole,
    ``{table}@.arrow`` (its columns, no rows) and ``{table}@{part}.arrow`` for one in :class:`Parts`.

    A version names its files in a manifest: each file is where the publish that wrote it put it, under the store's
    parts directory, and a version made from another names the other's files it keeps, so the version's own directory
    holds no table file. Read a version's files through this, never by listing its directory. Hold the version while
    the files are read (a lease, :func:`read_version`, a :class:`MappedVersion`): a file goes once no version held names
    it. A version written before manifests holds its files itself, and is listed.

    :param root: the store's directory
    :ptype root: Path | str
    :param version: the version
    :ptype version: str
    :return: file name -> the file
    :rtype: dict[str, VersionFile]
    :raises FileNotFoundError: when the version is gone
    """
    return _files_of(Path(root) / _VERSIONS_DIR / _valid_name(version, "version"))


#: the schema metadata key a file's column tokens are held under (:func:`stamped`)
COLUMN_TOKENS_KEY: Final = b"threetears.column_tokens"

#: the most files whose tokens a process keeps (:func:`file_tokens`), the least recently read let go first
_TOKENED_FILES: Final = 65_536


@dataclass(frozen=True)
class FileTokens:
    """a file's column tokens (:func:`stamped`): which columns a write moved, known without reading a row.

    Every column holding a token holds exactly the content the token was minted for: a token is minted fresh, or kept
    only from a column equal to it position by position in a file of as many rows. So two reads whose files' (file,
    row token, tokens of the columns read) are equal read the same relation over those columns, in the same order.

    :ivar rows: moves whenever the file's row count moves
    :ivar columns: column -> its token
    """

    rows: str
    columns: Mapping[str, str]


def _fresh_token() -> str:
    """a token equal to no other.

    :return: the token
    :rtype: str
    """
    return uuid.uuid7().hex


def _tokens_of(schema: Any) -> FileTokens | None:
    """the tokens a schema's metadata holds.

    :param schema: the schema
    :ptype schema: pa.Schema
    :return: the tokens; None for none, or for metadata not in the shape :func:`stamped` writes
    :rtype: FileTokens | None
    """
    raw = (schema.metadata or {}).get(COLUMN_TOKENS_KEY)
    held: Any = None
    if raw is not None:
        try:
            held = json.loads(raw)
        except ValueError:
            # NOSILENT: tokens not in the shape written here: the file prints as its identity (column_print)
            held = None
    tokens: FileTokens | None = None
    if isinstance(held, dict) and isinstance(held.get("rows"), str) and isinstance(held.get("columns"), dict):
        tokens = FileTokens(held["rows"], MappingProxyType({str(k): str(v) for k, v in held["columns"].items()}))
    return tokens


def stamped(table: Any, base: Path | None) -> Any:
    """``table`` with its column tokens in its schema's metadata (:data:`COLUMN_TOKENS_KEY`, JSON
    ``{"rows": token, "columns": {column: token}}``): a column takes the token of ``base``'s same-named column only
    when ``base`` holds tokens, as many rows, and that column equal to it (``ChunkedArray.equals``: values and nulls
    position by position, the same type); the row token is kept when the row count is. Every other token is fresh, and
    every token is fresh when ``base`` is None, holds no tokens, or cannot be read. Other metadata is kept.

    The compare is Arrow's own over the memory-mapped base: no Python per value. Run it where the file is written (a
    worker, :func:`stamping`), never on a serving process's loop.

    :param table: the rows as they are written
    :ptype table: pa.Table
    :param base: the file holding the same rows in the version this one is made from; None when there is none
    :ptype base: Path | None
    :return: the rows, stamped
    :rtype: pa.Table
    """
    _require_arrow()
    rows = _fresh_token()
    columns = {name: _fresh_token() for name in table.column_names}
    if base is not None:
        try:
            with pa.memory_map(str(base), "r") as source:
                reader = ipc.open_file(source)
                held = _tokens_of(reader.schema)
                if held is not None:
                    before = reader.read_all()
                    if before.num_rows == table.num_rows:
                        rows = held.rows
                        for name in table.column_names:
                            token = held.columns.get(name)
                            if (
                                token is not None
                                and name in before.column_names
                                and before.column(name).equals(table.column(name))
                            ):
                                columns[name] = token
        except OSError, pa.ArrowInvalid:
            # NOSILENT: a base gone with its version (or not an Arrow file) keeps nothing: every token is fresh
            log.debug(
                "a base file could not be read; its column tokens are fresh", extra={"extra_data": {"base": str(base)}}
            )
    metadata = dict(table.schema.metadata or {})
    metadata[COLUMN_TOKENS_KEY] = json.dumps({"rows": rows, "columns": columns}, sort_keys=True).encode()
    return table.replace_schema_metadata(metadata)


def stamping(
    write: Callable[[Mapping[str, Any]], int], base_of: Callable[[str], Path | None]
) -> Callable[[Mapping[str, Any]], int]:
    """a slot's write step (:meth:`SegmentSlot.writing`) that stamps every table it is given whole (:func:`stamped`,
    against ``base_of(name)``), where the write runs (the worker). Anything else -- a reader, an opener, a file, parts
    -- is written with no tokens: a reader's schema, and what an opener opens, are cleared of any tokens they carry
    from the file they were read from; a ``Path`` (linked, its tokens its own) and :class:`Parts` pass as given.

    :param write: the step
    :ptype write: Callable[[Mapping[str, Any]], int]
    :param base_of: a written name's file in the base version, or None
    :ptype base_of: Callable[[str], Path | None]
    :return: the stamping step
    :rtype: Callable[[Mapping[str, Any]], int]
    """

    def step(tables: Mapping[str, Any]) -> int:
        return write(
            {
                name: stamped(rows, base_of(name)) if isinstance(rows, pa.Table) else _unstamped(rows)
                for name, rows in tables.items()
            }
        )

    return step


def _without_tokens(schema: Any) -> Any:
    """``schema`` without column tokens.

    :param schema: the schema
    :ptype schema: pa.Schema
    :return: the schema, its other metadata kept
    :rtype: pa.Schema
    """
    metadata = {k: v for k, v in (schema.metadata or {}).items() if k != COLUMN_TOKENS_KEY}
    return schema.with_metadata(metadata) if metadata else schema.remove_metadata()


def _unstamped(rows: Any) -> Any:
    """rows given otherwise than as a table, carrying no column tokens (:func:`stamping`).

    :param rows: what a write step was given for one name
    :ptype rows: Any
    :return: the same rows, carrying no tokens
    :rtype: Any
    """
    made: Any = rows
    if isinstance(rows, pa.RecordBatchReader):
        made = pa.RecordBatchReader.from_batches(_without_tokens(rows.schema), rows)
    elif isinstance(rows, pa.Table):
        made = rows.replace_schema_metadata(_without_tokens(rows.schema).metadata)
    elif callable(rows):
        made = functools.partial(_opened_unstamped, rows)
    return made


def _opened_unstamped(opener: Callable[[], Any]) -> Any:
    """what ``opener`` opens, carrying no column tokens.

    :param opener: what opens a table or a reader
    :ptype opener: Callable[[], pa.Table | pa.RecordBatchReader]
    :return: the table or reader
    :rtype: pa.Table | pa.RecordBatchReader
    """
    return _unstamped(opener())


def _file_identity(path: Path) -> tuple[str, int, int, int]:
    """a file's identity: where it is, its inode, size and modification time (a store's files are never written
    again: a new file at the same name is another inode).

    :param path: the file
    :ptype path: Path
    :return: (path, inode, bytes, modified ns)
    :rtype: tuple[str, int, int, int]
    """
    status = os.stat(path, follow_symlinks=False)
    return (str(path), status.st_ino, status.st_size, status.st_mtime_ns)


@functools.lru_cache(maxsize=_TOKENED_FILES)
def _tokens_at(identity: tuple[str, int, int, int]) -> FileTokens | None:
    """the tokens of the file ``identity`` names, read from its footer once per process (:func:`file_tokens`).

    :param identity: the file's identity (:func:`_file_identity`)
    :ptype identity: tuple[str, int, int, int]
    :return: its tokens, or None
    :rtype: FileTokens | None
    """
    with pa.memory_map(identity[0], "r") as source:
        return _tokens_of(ipc.open_file(source).schema)


def file_tokens(path: Path) -> FileTokens | None:
    """the column tokens a file's footer holds (its schema, memory-mapped: no page of rows read); None for none.
    Kept per process by the file's identity (path, inode, bytes, modified), the least recently read of
    65,536 let go first. Blocking (a stat, a footer read): off the loop.

    :param path: the file
    :ptype path: Path
    :return: its tokens, or None
    :rtype: FileTokens | None
    """
    _require_arrow()
    return _tokens_at(_file_identity(path))


def column_print(path: Path, columns: Collection[str]) -> str:
    """the print of ``columns`` of one file: ``"<rows token>|column=token;..."`` (columns sorted; a column the file
    holds no token for prints empty). A file holding no tokens prints as its identity
    (``"file:<path>:<inode>:<bytes>:<modified ns>"``): any write moves it.

    :param path: the file
    :ptype path: Path
    :param columns: the columns read
    :ptype columns: Collection[str]
    :return: the print
    :rtype: str
    """
    tokens = file_tokens(path)
    if tokens is None:
        name, inode, size, modified = _file_identity(path)
        printed = f"file:{name}:{inode}:{size}:{modified}"
    else:
        printed = tokens.rows + "|" + ";".join(f"{c}={tokens.columns.get(c, '')}" for c in sorted(columns))
    return printed


def _write_parts(
    part_dir: Path,
    tables: Mapping[str, Any],
    base: _Manifest | None,
    room: Callable[[], int],
    reclaim: Callable[[], object],
) -> tuple[_Manifest, dict[str, os.stat_result], int]:
    """write a version's new files into ``part_dir`` (made only when a file is written) and make its manifest: the
    files ``tables`` names, and every file of ``base`` they do not replace, named as the base names it.

    A table given whole is one file; one given as :class:`Parts` is a schema file and a file per part, and with a base
    holding it in parts keeps every base part ``Parts`` does not name. A table of the base that ``tables`` does not
    name is kept whole. What is kept is the base's own lines: no file is touched for it, and a table left alone is the
    base's own mapping.

    :param part_dir: the publish's own directory under the parts directory
    :ptype part_dir: Path
    :param tables: name -> a table, a reader of its batches, what opens either, or :class:`Parts`
    :ptype tables: Mapping[str, Any]
    :param base: the manifest of the version to keep what is not replaced from, or None
    :ptype base: _Manifest | None
    :param room: the bytes every new file together may take, now (asked before each batch)
    :ptype room: Callable[[], int]
    :param reclaim: called before a batch past ``room`` is refused (:func:`_write_file`)
    :ptype reclaim: Callable[[], object]
    :return: the manifest, each file written or linked into ``part_dir`` by name, and the bytes written
    :rtype: tuple[_Manifest, dict[str, os.stat_result], int]
    :raises SharedArrowFullError: when they would take more than ``room``
    :raises ValueError: when a part's columns are not its table's, or a table in parts changed its columns from the
        base's (publish it without a base)
    """
    made: dict[str, os.stat_result] = {}
    written = 0

    def left() -> int:
        return room() - written

    def placed(file_name: str) -> Path:
        if not made and not part_dir.is_dir():
            part_dir.mkdir()
        return part_dir / file_name

    def recorded(table_lines: dict[str, str], file_name: str, stat: os.stat_result) -> int:
        # the file's line in its table, replacing the base's: what the table's bytes grow by
        made[file_name] = stat
        before = table_lines.get(file_name)
        table_lines[file_name] = _line_of(file_name, part_dir.name, stat.st_size)
        return stat.st_size - (0 if before is None else _line_fields(before)[2])

    lines: dict[str, Mapping[str, str]] = {} if base is None else dict(base.lines)
    text: dict[str, str] = {} if base is None else dict(base.text)
    nbytes: dict[str, int] = {} if base is None else dict(base.nbytes)
    for name, source in tables.items():
        _valid_name(name, "table")
        table_lines: dict[str, str] = {}
        size = 0
        if isinstance(source, Parts):
            for part in source.parts:
                _valid_name(part, "part")
            schema_file = f"{name}{_PART_SEPARATOR}{_TABLE_SUFFIX}"
            held = None if base is None else base.lines.get(name)
            if held is not None and schema_file in held:
                _, held_dir, _ = _line_fields(held[schema_file])
                if not _file_schema(part_dir.parent / held_dir / schema_file).equals(source.schema):
                    raise ValueError(f"table {name!r} changed its columns from the base version's; publish it whole")
                table_lines = dict(held)
                size = nbytes[name]
            path = placed(schema_file)
            stat = _write_file(path, source.schema, iter(()), left, what=name, reclaim=reclaim)
            written += stat.st_size
            size += recorded(table_lines, schema_file, stat)
            for part, rows in source.parts.items():
                file_name = f"{name}{_PART_SEPARATOR}{part}{_TABLE_SUFFIX}"
                if rows is None:
                    dropped = table_lines.pop(file_name, None)
                    size -= 0 if dropped is None else _line_fields(dropped)[2]
                    continue
                path = placed(file_name)
                if isinstance(rows, Path):
                    # a file already in the store (a worker's result): linked, its pages never copied
                    if not _file_schema(rows).equals(source.schema):
                        raise ValueError(f"table {name!r} part {part!r}: its columns are not the table's")
                    os.link(rows, path)
                    size += recorded(table_lines, file_name, os.stat(path))
                    continue
                part_schema, batches = _batches(rows)
                if not part_schema.equals(source.schema):
                    raise ValueError(f"table {name!r} part {part!r}: its columns are not the table's")
                stat = _write_file(
                    path, source.schema, batches, left, what=f"table {name!r} part {part!r}", reclaim=reclaim
                )
                written += stat.st_size
                size += recorded(table_lines, file_name, stat)
        else:
            schema, batches = _batches(source)
            file_name = f"{name}{_TABLE_SUFFIX}"
            stat = _write_file(placed(file_name), schema, batches, left, what=f"table {name!r}", reclaim=reclaim)
            written += stat.st_size
            size += recorded(table_lines, file_name, stat)
        lines[name] = table_lines
        text[name] = "".join(table_lines.values())
        nbytes[name] = size
    return _Manifest(lines=lines, text=text, nbytes=nbytes), made, written


def _imported(flat: Path, part_dir: Path) -> _Manifest:
    """a version written before manifests (its directory holding its files itself) as a manifest: each of its files
    linked into ``part_dir``, once, so a version made from it names them as any other. Only a store started over such
    a directory, held by a reader that outlived the process before, ever meets one.

    :param flat: the version's directory
    :ptype flat: Path
    :param part_dir: a directory of its own under the parts directory
    :ptype part_dir: Path
    :return: the manifest
    :rtype: _Manifest
    """
    part_dir.mkdir()
    lines: list[str] = []
    with os.scandir(flat) as listing:
        for entry in sorted(listing, key=lambda each: each.name):
            if _table_of(entry.name) is not None:
                os.link(entry.path, part_dir / entry.name)
                lines.append(_line_of(entry.name, part_dir.name, entry.stat().st_size))
    return _parsed_manifest(_MANIFEST_HEADER + "".join(lines))


def _shared_files(path: Path, mapped: Mapping[Any, Any]) -> dict[Any, Any]:
    """the mapped files that the version directory ``path`` holds too (:func:`_mapped_files` identifies each): what a
    mapping of it may take.

    :param path: a version's directory
    :ptype path: Path
    :param mapped: files mapped already, by identity
    :ptype mapped: Mapping[Any, pa.Table]
    :return: those of them ``path`` holds; empty when it is gone
    :rtype: dict[Any, pa.Table]
    """
    try:
        text = _manifest_text(path)
        if text is None:
            device = path.stat().st_dev
            with os.scandir(path) as listing:
                held: set[Any] = {(device, entry.inode()) for entry in listing}
        else:
            held = set(text.splitlines(keepends=True))
    except FileNotFoundError:
        # NOSILENT: the version is gone already: nothing to lend its first read
        return {}
    return {identity: table for identity, table in mapped.items() if identity in held}


class _Pin:
    """a shared lock on a directory, let go when the last mapped file holding it is freed: what keeps a store
    counting a directory's files while any buffer taken from them lives (:class:`MappedTables`, ``pin=True``).

    Each mapped file's finalizer holds this object, so it is freed (and the lock let go) when the mapping holding it
    is closed and the last buffer over each of its files is freed, whatever thread frees it. It takes no lock of its
    own: a finalizer may run inside any code, a store's own locked sections included.

    :param descriptor: the open lock file holding the shared lock
    :ptype descriptor: int
    """

    __slots__ = ("__weakref__",)

    def __init__(self, descriptor: int) -> None:
        weakref.finalize(self, os.close, descriptor)


def _held(_pin: _Pin) -> None:
    """a mapped file's finalizer: it held ``_pin`` while the file was mapped, and lets it go now.

    :param _pin: the pin
    :ptype _pin: _Pin
    :return: nothing
    :rtype: None
    """


def _map_pinned(path: str, pin: _Pin | None) -> Any:
    """one Arrow IPC file mapped read-only through a Python ``mmap``, whose buffers keep the map alive: the map, and
    with it ``pin``, is freed only once no buffer over the file lives (a table, a slice, a cast that shares them).

    :param path: the file
    :ptype path: str
    :param pin: the directory's lock, held while the file is mapped; None for none
    :ptype pin: _Pin | None
    :return: the file's table
    :rtype: pa.Table
    """
    descriptor = os.open(path, os.O_RDONLY)
    try:
        region = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ, trackfd=False)
    finally:
        os.close(descriptor)
    if pin is not None:
        weakref.finalize(region, _held, pin)
    return ipc.open_file(pa.BufferReader(pa.py_buffer(region))).read_all()


class MappedTables:
    """Arrow tables memory-mapped read-only from a directory: every process mapping it shares its pages, and no row
    is copied into this one. Close it when done; a table kept past :meth:`close` keeps its pages mapped, and, mapped
    with ``pin=True``, its directory's shared lock too, so a store keeps counting the files until the last buffer
    over them is freed. Without ``pin`` it keeps neither the lock nor its files' place in the store.

    A version made from another (``base=``) names most of the other's files as its own: given the other's
    mapped files (``reuse``, its :attr:`by_inode`), a file that is the same file (its manifest line, or for a directory
    holding its files itself its device and inode) is taken from it as
    it is, never mapped again, so a version's first read maps only what changed. The pages are the same either way: a
    file is never written once it is in place. The caller keeps every file it lends linked or mapped while this is
    made (a lease, a lock, or the mapping itself, which keeps an unlinked file's inode, and so its number, in use).

    :param name: the directory's name (a version's, a result's slot)
    :ptype name: str
    :param path: the directory
    :ptype path: Path
    :param lock: whether to hold a shared lock on the directory while mapped
    :ptype lock: bool
    :param reuse: another mapping's files by identity (:attr:`by_inode`), taken as they are where they are the same
        files
    :ptype reuse: Mapping[Any, pa.Table] | None
    :param pin: hold the directory's shared lock (when it has a lock file) until :meth:`close` AND the last buffer
        over its files is freed, not only until :meth:`close`; its files are mapped through a Python ``mmap``
        (:func:`_map_pinned`), zero-copy as ever
    :ptype pin: bool
    :ivar name: the directory's name
    :ivar tables: table name -> the mapped table
    """

    def __init__(
        self,
        name: str,
        path: Path,
        *,
        lock: bool,
        reuse: Mapping[Any, Any] | None = None,
        pin: bool = False,
    ) -> None:
        self.name = name
        self._descriptor: int | None = None
        if lock:
            self._descriptor = _locked(path, fcntl.LOCK_SH)
            if self._descriptor is None:
                raise LookupError(f"shared Arrow {name!r} is gone")
        self._maps: list[Any] = []
        #: the pinned directory's lock, freed once this is closed and the last buffer over its files is
        pinned: _Pin | None = None
        if pin:
            descriptor = _locked(path, fcntl.LOCK_SH)
            pinned = None if descriptor is None else _Pin(descriptor)
        #: each file's table, by the file's identity (:func:`_mapped_files`), for a later version's mapping to take
        #: (``reuse``)
        self._by_inode: dict[Any, Any] = {}
        held = reuse or {}
        pieces: dict[str, list[Any]] = {}
        try:
            # sorted, so a table's schema file (``{table}@.arrow``) comes before its parts, and parts in name order
            for name, file, identity in _mapped_files(path):
                table = _table_of(name)
                if table is not None:
                    piece = held.get(identity)
                    if piece is None and pin:
                        piece = _map_pinned(file, pinned)
                    elif piece is None:
                        mapped = pa.memory_map(file, "r")
                        self._maps.append(mapped)
                        piece = ipc.open_file(mapped).read_all()
                    self._by_inode[identity] = piece
                    pieces.setdefault(table, []).append(piece)
        except BaseException:
            self._unmap()
            raise
        # a table in parts is its parts' chunks side by side: concatenating tables copies no buffer
        tables = {name: held[0] if len(held) == 1 else pa.concat_tables(held) for name, held in pieces.items()}
        self.tables: Mapping[str, Any] = MappingProxyType(tables)

    @property
    def by_inode(self) -> Mapping[Any, Any]:
        """each mapped file's table by the file's identity -- a version's file by its manifest line (a line names one
        file, never reused), a file its directory holds itself by (device, inode) -- what a later version's mapping
        takes (``reuse``).

        :return: (device, inode) -> the file's table; empty once closed
        :rtype: Mapping[Any, pa.Table]
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
        :raises ImportError: when duckdb is not installed
        """
        _require_duckdb()
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
    :param reuse: another version's mapped files by identity (:attr:`MappedTables.by_inode`), taken where they are the
        same files
    :ptype reuse: Mapping[Any, pa.Table] | None
    :raises LookupError: when the version is gone
    """

    def __init__(self, root: Path | str, version: str, *, reuse: Mapping[Any, Any] | None = None) -> None:
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


def _table_files(directory: Path, table: str) -> tuple[dict[str, Path], Path | None]:
    """each part of ``table`` a version (or a result) directory holds, by part name, and its schema file, which is not
    a part.

    :param directory: the directory
    :ptype directory: Path
    :param table: the table
    :ptype table: str
    :return: part name -> its file, and the schema file (None when the directory holds the table in no parts)
    :rtype: tuple[dict[str, Path], Path | None]
    """
    prefix = f"{table}{_PART_SEPARATOR}"
    schema_name = f"{prefix}{_TABLE_SUFFIX}"
    files = _files_of(directory)
    parts = {
        name[len(prefix) : -len(_TABLE_SUFFIX)]: held.path
        for name, held in files.items()
        if name.startswith(prefix) and name.endswith(_TABLE_SUFFIX) and name != schema_name
    }
    schema = files.get(schema_name)
    return parts, None if schema is None else schema.path


def merged_rows(root: Path | str, version: str, table: str, names: Sequence[str]) -> Callable[[], Any]:
    """what opens the rows of ``names`` (parts of ``table`` in ``version``) as one reader, in part-name order.

    For :class:`Parts` (a merged part :meth:`SharedArrowStore.publish` writes as it reaches it) and for a worker's
    :meth:`SegmentSlot.write`: the reader maps each part's file and hands on its record batches, so the merge copies
    each byte of the group once, into the new file, and reads nothing else. While the reader is open it holds a
    shared ``flock`` on the version, as any reader of it does, so no sweep retires the version under it.

    :param root: the store's directory
    :ptype root: Path | str
    :param version: the version holding the parts
    :ptype version: str
    :param table: the table held in parts
    :ptype table: str
    :param names: the parts to merge
    :ptype names: Sequence[str]
    :return: a callable taking no argument that returns a ``pyarrow.RecordBatchReader`` of their rows
    :rtype: Callable[[], pa.RecordBatchReader]
    """
    path = Path(root) / _VERSIONS_DIR / _valid_name(version, "version")
    _valid_name(table, "table")
    ordered = sorted(_valid_name(name, "part") for name in names)

    def held() -> int:
        descriptor = _locked(path, fcntl.LOCK_SH)
        if descriptor is None:
            raise LookupError(f"shared Arrow version {version!r} is gone")
        return descriptor

    def opened() -> Any:
        descriptor = held()
        try:
            files, schema_file = _table_files(path, table)
            missing = [name for name in ordered if name not in files]
            if missing or schema_file is None:
                raise LookupError(f"shared Arrow version {version!r} holds no part {missing} of table {table!r}")
            schema = _file_schema(schema_file)
        finally:
            os.close(descriptor)

        def batches() -> Iterator[Any]:
            # held again from the first batch to the last: a reader never read takes no lock and leaves none behind
            again = held()
            try:
                for name in ordered:
                    with pa.memory_map(str(files[name]), "r") as mapped:
                        reader = ipc.open_file(mapped)
                        for index in range(reader.num_record_batches):
                            yield reader.get_batch(index)
            finally:
                os.close(again)

        return pa.RecordBatchReader.from_batches(schema, batches())

    return opened


def merge_parts(store: SharedArrowStore, version: str, table: str, names: Sequence[str], into: str) -> Parts:
    """the :class:`Parts` that merge ``names`` (parts of ``table`` in ``version``) into one part, ``into``.

    Publish them on the version (``store.publish(new, {table: merge_parts(...)}, base=version)``): every part of the
    base outside the group is kept as the base's own file, each part of the group is dropped, and ``into`` is written from the group's
    mapped files (:func:`merged_rows`), so the merge writes the group's bytes once and nothing else. A reader sees the
    same rows in the same order before and after: the group must be a run of neighbours in part-name order, and
    ``into`` must sort into the run's place (``f"{first}.1"`` after the first part's name ``first``, say), so no other
    part falls between. ``into`` may be one of ``names``.

    Run it where the store is owned (the serving process, on a thread: the write is Arrow IPC and releases the
    interpreter lock). A worker process merges into a result instead, through :func:`merged_rows` and
    :meth:`SegmentSlot.write`, and the serving process links the claimed file as the part (:class:`Parts`).

    :param store: the store holding ``version``
    :ptype store: SharedArrowStore
    :param version: the version to merge in
    :ptype version: str
    :param table: the table held in parts
    :ptype table: str
    :param names: the parts to merge, at least two
    :ptype names: Sequence[str]
    :param into: the merged part's name
    :ptype into: str
    :return: the table's parts for :meth:`SharedArrowStore.publish`: ``into`` -> the merged rows, each other name -> None
    :rtype: Parts
    :raises ValueError: for fewer than two parts, a name that is not one plain path segment, a group that is not a run
        of neighbours, or an ``into`` that would sort past another part
    :raises LookupError: when the version is not held, or holds no such part
    """
    group = sorted({_valid_name(name, "part") for name in names})
    _valid_name(into, "part")
    if len(group) < 2:
        raise ValueError(f"a merge takes at least two parts of {table!r}, not {group}")
    with store.lease(version) as held:
        files, schema_file = _table_files(held.path, _valid_name(table, "table"))
        schema = _file_schema(schema_file) if files and schema_file is not None else None
    missing = [name for name in group if name not in files]
    if schema is None or missing:
        raise LookupError(f"shared Arrow version {version!r} holds no part {missing or group} of table {table!r}")
    others = sorted(set(files) - set(group))
    low, high = group[0], group[-1]
    between = [name for name in others if min(low, into) < name < max(high, into)]
    if between:
        raise ValueError(
            f"merging {group} of {table!r} into {into!r} would move parts {between} past merged rows: merge neighbours, "
            f"into a name that sorts in their place"
        )
    return Parts(schema=schema, parts={**dict.fromkeys(group), into: merged_rows(store.root, version, table, group)})


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
        with self._staged() as (staging, target):
            written = _write_tables(staging, tables, lambda: self.max_bytes)
            os.rename(staging, target)
        return written

    @contextlib.contextmanager
    def writing(self) -> Iterator[Callable[[Mapping[str, Any]], int]]:
        """write the result's tables in steps (in the worker): each step's tables written as it is given, so the
        worker holds one step's rows at a time, never the whole result's. Staged as :meth:`write` stages, the slot's
        ``max_bytes`` bounding every step together, and renamed into the slot whole as the block ends; a block that
        raises keeps nothing, and one whose steps wrote no table leaves the slot unwritten (its claim is None).

        :return: a context manager yielding the step: ``write(tables)`` (as :meth:`write` takes them), returning the
            bytes the step wrote
        :rtype: Iterator[Callable[[Mapping[str, Any]], int]]
        :raises SharedArrowFullError: when the steps together would pass the slot's ``max_bytes``; nothing is kept
        :raises ValueError: when a step names a table an earlier step wrote
        :raises FileExistsError: when the slot was written already
        """
        written = 0
        names: set[str] = set()
        with self._staged() as (staging, target):

            def step(tables: Mapping[str, Any]) -> int:
                nonlocal written
                again = sorted(names & set(tables))
                if again:
                    raise ValueError(f"a slot's tables are written once each, and {again} were written already")
                names.update(tables)
                made = _write_tables(staging, tables, lambda: self.max_bytes - written)
                written += made
                return made

            yield step
            if names:
                os.rename(staging, target)
            else:
                remove_tree(staging, log=log, what="an empty result's staging directory")

    @contextlib.contextmanager
    def _staged(self) -> Iterator[tuple[Path, Path]]:
        """a staging directory for the slot's result, held under an exclusive lock while it is written: the caller
        writes into it and renames it to the target; one the caller leaves (it raised) is removed.

        :return: a context manager yielding the staging directory and the slot's directory
        :rtype: Iterator[tuple[Path, Path]]
        :raises FileExistsError: when the slot was written already
        :raises RuntimeError: when its staging directory could not be locked
        """
        _require_arrow()
        segments = Path(self.root) / _SEGMENTS_DIR
        target = segments / self.slot_id
        if target.exists():
            raise FileExistsError(f"slot {self.slot_id} is written once")
        staging = segments / f"{_STAGING_PREFIX}{self.slot_id}-{uuid.uuid7().hex[-8:]}"
        staging.mkdir()
        (staging / _LOCK_FILE).touch()
        descriptor = _locked(staging, fcntl.LOCK_EX)
        if descriptor is None:
            # never written unlocked: a sweep would take it from under the write
            raise RuntimeError(f"slot {self.slot_id}'s staging directory {staging.name} could not be locked")
        try:
            yield staging, target
        except BaseException:
            remove_tree(staging, log=log, what="a failed result's staging directory")
            raise
        finally:
            os.close(descriptor)


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
        _require_arrow()
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
    :meth:`close` and until the last buffer taken from them is freed (``pin``): a table kept past the close (a slice
    of it, a cast sharing its buffers) keeps the files, and the store counts them, so ``max_bytes`` bounds what the
    store's files hold in memory however long a caller keeps a result. The first sweep after both removes them.

    :ivar nbytes: the bytes its files hold
    :ivar files: table name -> its Arrow IPC file, for a publish that links it as a part (:class:`Parts`) while
        the segment is open: the version then holds the file after :meth:`close`
    """

    def __init__(self, slot_id: str, path: Path, on_close: Callable[[str], object]) -> None:
        super().__init__(slot_id, path, lock=False, pin=True)
        self._on_close: Callable[[str], object] | None = on_close
        self.nbytes = _directory_bytes(path)
        self.files: Mapping[str, Path] = MappingProxyType(
            {entry.name[: -len(_TABLE_SUFFIX)]: entry for entry in path.iterdir() if entry.name.endswith(_TABLE_SUFFIX)}
        )

    def close(self) -> None:
        """let the result go: its files are removed at once when no table taken from it is held, else by the first
        sweep after the last is freed. Harmless twice.

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

    Blocking: call :meth:`publish`, :meth:`claim`, :meth:`sweep` and :meth:`settle` from a worker thread, never the
    event loop.

    :param root: the directory, on a memory-backed filesystem
    :ptype root: Path | str
    :param max_bytes: the most the directory may hold, every version and result together
    :ptype max_bytes: int
    """

    def __init__(self, root: Path | str, *, max_bytes: int) -> None:
        _require_arrow()
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._root = Path(root)
        self._versions = self._root / _VERSIONS_DIR
        self._parts = self._root / _PARTS_DIR
        self._segments = self._root / _SEGMENTS_DIR
        self._handoffs = self._root / _HANDOFFS_DIR
        self._versions.mkdir(parents=True, exist_ok=True)
        self._parts.mkdir(parents=True, exist_ok=True)
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
        #: what the directories' files hold, kept by every write and removal so :meth:`status` never walks them
        self._held = _HeldFiles(self._versions, self._handoffs)
        #: the manifest of every version the directory holds under its name, by name: what a publish on it starts
        #: from, and what a sweep compares a retired version's with to find the part files no version names any more
        self._manifests: dict[str, _Manifest] = {}
        #: each part file's (device, inode), by its path under the parts directory (``{part dir}/{file}``): what the
        #: tally drops as the file is removed
        self._part_keys: dict[str, tuple[int, int]] = {}
        #: part files a sweep could not remove (under the parts directory), retried by the next
        self._unremoved: list[str] = []
        #: the sweep a publish leaves, on a thread of its own: due once more, and the thread running it (one at most)
        self._settling = threading.Condition()
        self._settle_due = False
        self._settler: threading.Thread | None = None
        with contextlib.suppress(FileNotFoundError):  # NOSILENT: nothing current is the start's state either way
            (self._versions / _CURRENT_FILE).unlink()
        self._load_manifests()
        self.sweep()
        self._sweep_parts_on_disk()
        self._adopt()

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
            for half in (self._versions, self._parts, self._segments, self._handoffs)
            for entry in half.iterdir()
            if entry.is_dir()
        )

    def status(self) -> dict[str, Any]:
        """what the store holds, as plain JSON values: read from the store's own tallies, never the file system and
        never waiting on a lock, so an event loop may call it (walking the directories took a serving loop 50-56 ms a
        call on cobalt dev, 2026-11-07, and a lock is held by sweeps and room checks that walk them).

        ``versions``, ``handoffs_held`` and ``held_bytes`` are what :meth:`versions`, :meth:`handoffs` and
        :meth:`held_bytes` would find, but for what is still being written: a version's or a hand-off's staged files
        count once it takes its name (a hand-off being written counts in ``handoffs_open``), and a worker's result
        once it is claimed (an open slot counts in ``slots_open``). Each value is read on its own: one taken while
        another thread changes the store may be a step behind another.

        :return: the values
        :rtype: dict[str, Any]
        """
        current = self._current
        held_bytes, versions, held_handoffs = self._held.view
        return {
            "current": None if current is None else current.name,
            "versions": list(versions),
            "slots_open": len(self._slots),
            "segments_claimed": len(self._claimed),
            "handoffs_open": len(self._handing) + len(self._handed),
            # in the directory: the open ones, and those released that a worker still maps
            "handoffs_held": held_handoffs,
            "handoffs_made": self._handoffs_made,
            "held_bytes": held_bytes,
            "max_bytes": self._max_bytes,
            "published": self._published,
            "retired": self._retired,
            "refused": self._refused,
        }

    def publish(self, version: str, tables: Mapping[str, Any], *, base: str | None = None) -> SharedVersion:
        """write ``tables`` as ``version`` and make it current; a version already present is made current as it is.

        With ``base``, the new version starts as that version: every table of it not named in ``tables`` is kept, and
        a table given as :class:`Parts` keeps every part of the base's it does not name. What is kept is the base's
        own files, named in the new version's manifest: no file is touched for it and its pages are counted once, so
        a change to one part costs one part's write and one manifest's, however many parts the version holds.

        It returns once the version is current. What it leaves -- its readers told (:meth:`reader`), every version
        superseded and unheld retired and the files only those named removed -- runs after, on the store's own thread
        (:meth:`settle` waits for it). A write that would pass ``max_bytes`` sweeps at once before it is refused.

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
                    # the base is held until the version is current, so no sweep retires it while it is named; let
                    # go, it leaves its sweep to the store's thread with the rest of the publish's
                    base_path = None if base is None else held.enter_context(self._lease(base, self._settle_soon)).path
                    self._write_version(name, tables, base_path)
                with self._state:
                    published = self._described(target)
                    self._write_current(name)
                    self._current = published
                    self._published += 1
            finally:
                with self._state:
                    self._pending = None
        self._settle_soon()
        log.info(
            "shared Arrow version published",
            extra={"extra_data": {"version": name, "bytes": published.nbytes, "held_bytes": self._held.view[0]}},
        )
        return published

    def publish_query(
        self, version: str, cursor: Any, queries: Mapping[str, str], *, batch_rows: int = DEFAULT_BATCH_ROWS
    ) -> SharedVersion:
        """write each query's answer over a DuckDB ``cursor`` as a table of ``version``, a batch at a time: no table
        is ever held whole in this process.

        :param version: the version's name
        :ptype version: str
        :param cursor: a DuckDB connection or cursor, held still for the publish (a read transaction's cursor)
        :ptype cursor: duckdb.DuckDBPyConnection
        :param queries: table name -> the SQL whose answer it holds
        :ptype queries: Mapping[str, str]
        :param batch_rows: rows per batch read from DuckDB
        :ptype batch_rows: int
        :return: the version
        :rtype: SharedVersion
        :raises SharedArrowFullError: as :meth:`publish`
        :raises ImportError: when duckdb is not installed
        """
        _require_duckdb()
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
        return self._lease(version, self.sweep)

    def _lease(self, version: str | None, on_release: Callable[[], object]) -> SharedLease:
        """:meth:`lease`, calling ``on_release`` as the lease is released.

        :param version: the version; None for the current one
        :ptype version: str | None
        :param on_release: what the release calls (a sweep now, or one left to the store's thread)
        :ptype on_release: Callable[[], object]
        :return: the lease
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
                    held = self._described(path)
                else:
                    raise LookupError(f"shared Arrow version {version!r} is not held")
            # under the state lock, which every retirement takes: the version cannot go between the check and this
            descriptor = _locked(held.path, fcntl.LOCK_SH)
        if descriptor is None:
            raise LookupError(f"shared Arrow version {held.name!r} is being retired")
        return SharedLease(held, descriptor, on_release)

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
        :raises ImportError: when duckdb is not installed
        """
        _require_duckdb()
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
                files = _directory_files(path)
                self._claimed.add(slot.slot_id)
                self._claimed_bytes[slot.slot_id] = sum(files.values())
                self._held.add(path, files)
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
            files = _directory_files(staging)
            os.rename(staging, self._handoffs / handoff_id)
            with self._state:
                # in _handing until here, so no sweep retires it before it is recorded
                self._held.add(self._handoffs / handoff_id, files)
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
        """the worker is done with a hand-off (its job returned or raised, or its worker died): it goes now, or as soon
        as no worker maps it. Harmless twice.

        A worker that still maps it holds its shared lock: a reader finishing, or a worker that died and whose exit
        has not yet closed its descriptors (the pool hears of a death as its pipe closes, which can come first). Then
        a thread of the store's waits on the lock and sweeps once it is free, so the hand-off never waits for some
        later, unrelated sweep to go.

        :param handed: the hand-off
        :ptype handed: HandedTables
        :return: nothing
        :rtype: None
        """
        with self._state:
            self._handed.discard(handed.handoff_id)
        self.sweep()
        entry = self._handoffs / handed.handoff_id
        if entry.exists():
            waiter = threading.Thread(
                target=self._sweep_when_let_go, args=(entry,), name="shared-arrow-handoff-wait", daemon=True
            )
            waiter.start()

    def _sweep_when_let_go(self, entry: Path) -> None:
        """wait until no reader holds ``entry``'s lock, then sweep: a released hand-off's last mapping let go.

        :param entry: the hand-off's directory
        :ptype entry: Path
        :return: nothing
        :rtype: None
        """
        try:
            descriptor = os.open(entry / _LOCK_FILE, os.O_RDONLY)
        except FileNotFoundError:
            # NOSILENT: already swept away
            return
        try:
            # blocks until every shared holder is gone: its readers closed, or its dead worker's exit closed them
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        finally:
            os.close(descriptor)
        try:
            self.sweep()
        except Exception as exc:  # noqa: BLE001 -- a thread has no caller to raise to; contained, so it is counted
            contain(exc, log=log, what="a released shared Arrow hand-off could not be swept once let go")

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
        """remove every version neither current nor held, the part files only those named, every result no open slot
        or claim owns, every hand-off released that no worker maps, and every staging directory whose writer is gone.
        Now, on the caller's thread, beside any sweep a publish left to the store's thread (:meth:`settle`).

        :return: the versions retired
        :rtype: list[str]
        """
        gone: list[tuple[Path, Path]] = []
        with self._state:
            retired = self._sweep_versions(gone)
            # a version's manifest leaves with the version: the files it alone named go with it
            dying = self._freed([self._manifests.pop(name) for name in retired if name in self._manifests])
            dying.extend(self._unremoved)
            self._unremoved = []
            self._sweep_segments(gone)
            self._sweep_handoffs(gone)
            self._retired += len(retired)
            for old, new in gone:
                self._held.moved(old, new)
        # renamed out of their names under the lock (nothing opens them again); their files go after it, and the
        # tally forgets them once they are gone: what a removal left (a later sweep retries it) is counted still.
        # What is left is listed under the lock: two sweeps may remove the same directory, and a listing taken
        # before the other's removal and recorded after its forgetting would count files that are gone
        for _, path in gone:
            remove_tree(path, log=log, what="a retired shared Arrow directory (a later sweep retries it)")
            with self._state:
                left = _directory_files(path)
                if left:
                    self._held.add(path, left)
                else:
                    self._held.forget(path)
        self._remove_parts(dying)
        with self._state:
            # a closed result's bytes count in a publish's room until its files are gone (a sweep after the last
            # table taken from it is freed, for one kept past its close)
            for slot_id in [each for each in self._claimed_bytes if each not in self._claimed]:
                if not (self._segments / slot_id).exists():
                    del self._claimed_bytes[slot_id]
        if retired:
            log.info("shared Arrow versions retired", extra={"extra_data": {"versions": retired}})
        return retired

    def settle(self) -> None:
        """wait until the sweep a publish left to the store's thread has run: its readers told, superseded versions
        retired and their files removed (:meth:`publish`). What the directory holds is then what the store's state
        says it may: a caller that asks right after a publish (a test, a measure) reads that, not the step before.

        :return: nothing
        :rtype: None
        """
        with self._settling:
            while self._settler is not None:
                self._settling.wait()

    def _settle_soon(self) -> None:
        """have the store's thread tell the readers and sweep, after the caller returns: one thread at a time, a
        request while it runs making it run once more, never a pile of them.

        :return: nothing
        :rtype: None
        """
        with self._settling:
            self._settle_due = True
            if self._settler is not None:
                return
            self._settler = threading.Thread(target=self._settle_all, name="shared-arrow-settle", daemon=True)
            settler = self._settler
        try:
            settler.start()
        except RuntimeError as exc:
            # NOSILENT: no thread to be had (the interpreter shutting down): the caller sweeps, said
            contain(
                exc, log=log, what="the shared Arrow store's sweeping thread could not start; swept on the caller's"
            )
            self._settle_all()

    def _settle_all(self) -> None:
        """tell the readers and sweep while a publish asked for it since the last time: the store's sweeping thread.

        :return: nothing
        :rtype: None
        """
        while True:
            with self._settling:
                if not self._settle_due:
                    self._settler = None
                    self._settling.notify_all()
                    return
                self._settle_due = False
            try:
                self._tell_readers()
                self.sweep()
            except Exception as exc:  # noqa: BLE001 -- a thread has no caller to raise to; contained, so it is counted
                contain(exc, log=log, what="a shared Arrow sweep after a publish failed; the next sweep retries it")

    def _write_current(self, name: str) -> None:
        """name ``name`` the current version in the directory, for workers (:func:`read_version`): written under a
        staging name and renamed over the last, so a reader reads one whole name. The directory is memory-backed and
        a new store removes the file at its start, so nothing is synced. Called under ``_state``.

        :param name: the version
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        staged = self._versions / f"{_STAGING_PREFIX}{_CURRENT_FILE}-{uuid.uuid7().hex[-8:]}"
        try:
            staged.write_text(name)
            os.replace(staged, self._versions / _CURRENT_FILE)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):  # NOSILENT: never written, or renamed already
                staged.unlink()
            raise

    def _described(self, path: Path) -> SharedVersion:
        """a version's directory as a :class:`SharedVersion`: from its manifest the store holds, no file read.

        :param path: the directory
        :ptype path: Path
        :return: the version
        :rtype: SharedVersion
        """
        manifest = self._manifests.get(path.name) if path.parent == self._versions else None
        if manifest is None:
            return _described(path)
        return SharedVersion(
            name=path.name,
            path=path,
            tables=tuple(sorted(manifest.lines)),
            nbytes=sum(manifest.nbytes.values()) + manifest.file_bytes,
        )

    def _freed(self, retired: Sequence[_Manifest]) -> list[str]:
        """the part files the manifests of versions just retired name and no version the store holds does: each as
        a path under the parts directory. A table a version holds as the retired one did (the same mapping, left alone
        by every publish between) costs nothing; a table it changed costs a comparison of its lines, in C. Called under
        ``_state``.

        :param retired: the retired versions' manifests, out of :attr:`_manifests`
        :ptype retired: Sequence[_Manifest]
        :return: the files to remove
        :rtype: list[str]
        """
        live = list(self._manifests.values())
        lines: set[str] = set()
        for manifest in retired:
            for table, held in manifest.lines.items():
                others = [each.lines[table] for each in live if table in each.lines]
                if any(other is held for other in others):
                    continue
                candidates = set(held.values())
                for other in others:
                    candidates.difference_update(other.values())
                lines |= candidates
        return [_line_rel(line) for line in lines]

    def _remove_parts(self, files: Sequence[str]) -> None:
        """remove part files no version names (:meth:`_freed`), outside the store's lock, and drop each from the tally;
        a part directory whose last file went goes too. A file that would not go is retried by the next sweep.

        :param files: the files, as paths under the parts directory
        :ptype files: Sequence[str]
        :return: nothing
        :rtype: None
        """
        if not files:
            return
        removed: list[str] = []
        failed: list[str] = []
        for rel in files:
            try:
                os.unlink(self._parts / rel)
            except FileNotFoundError:
                # NOSILENT: gone already, which is what was wanted
                pass
            except OSError as exc:
                contain(exc, log=log, what="a shared Arrow part file could not be removed (the next sweep retries it)")
                failed.append(rel)
                continue
            removed.append(rel)
        emptied: list[Path] = []
        with self._state:
            self._unremoved.extend(failed)
            for rel in removed:
                directory = self._parts / rel.split("/", 1)[0]
                key = self._part_keys.pop(rel, None)
                if key is not None and self._held.drop(directory, [key]) == 0:
                    emptied.append(directory)
        for directory in emptied:
            try:
                os.rmdir(directory)
            except FileNotFoundError:
                # NOSILENT: gone already
                pass
            except OSError as exc:
                contain(exc, log=log, what="an emptied shared Arrow part directory could not be removed")
                continue
            with self._state:
                self._held.forget(directory)

    def _recorded_parts(self, part_dir: Path, made: Mapping[str, os.stat_result]) -> None:
        """record the files a publish wrote or linked into its part directory in the tally. Called under ``_state``.

        :param part_dir: the directory
        :ptype part_dir: Path
        :param made: each file by name, as ``os.stat`` read it
        :ptype made: Mapping[str, os.stat_result]
        :return: nothing
        :rtype: None
        """
        if not made:
            return
        files: dict[tuple[int, int], int] = {}
        for name, stat in made.items():
            key = (stat.st_dev, stat.st_ino)
            files[key] = stat.st_size
            self._part_keys[f"{part_dir.name}/{name}"] = key
        self._held.add(part_dir, files)

    def _load_manifests(self) -> None:
        """read the manifest of every version the directory holds at the start: a version a reader of an earlier
        process still holds is kept, and names the part files that are kept with it.

        :return: nothing
        :rtype: None
        """
        for entry in self._versions.iterdir():
            if not entry.is_dir() or entry.name.startswith(".") or not (entry / _LOCK_FILE).is_file():
                continue
            text = _manifest_text(entry)
            if text is not None:
                self._manifests[entry.name] = _parsed_manifest(text)

    def _sweep_parts_on_disk(self) -> None:
        """at the start, after the first sweep: remove every part file no version kept names (a publish that died
        mid-write, or a version removed whose files a crash left), and every part directory left empty.

        :return: nothing
        :rtype: None
        """
        named = {
            _line_rel(line)
            for manifest in self._manifests.values()
            for held in manifest.lines.values()
            for line in held.values()
        }
        for directory in list(self._parts.iterdir()):
            if not directory.is_dir():
                directory.unlink()
                continue
            left = False
            for entry in list(directory.iterdir()):
                if f"{directory.name}/{entry.name}" in named:
                    left = True
                elif entry.is_dir():
                    remove_tree(entry, log=log, what="a stray directory among shared Arrow part files")
                else:
                    entry.unlink()
            if not left:
                os.rmdir(directory)

    def _adopt(self) -> None:
        """record what the directory holds at the start that no sweep could remove (a version or a hand-off a live
        worker of an earlier process still maps), so the tally starts exact. Staging directories are left out: a
        live writer's, from another process, counts once it takes its name, as at any time.

        :return: nothing
        :rtype: None
        """
        for half in (self._versions, self._segments, self._handoffs):
            for entry in half.iterdir():
                if not entry.is_dir() or entry.name.startswith("."):
                    continue
                if half == self._versions and not (entry / _LOCK_FILE).is_file():
                    continue
                files = _directory_files(entry)
                with self._state:
                    if entry not in self._held:
                        self._held.add(entry, files)
        for directory in self._parts.iterdir():
            with os.scandir(directory) as listing:
                made = {entry.name: entry.stat() for entry in listing}
            with self._state:
                self._recorded_parts(directory, made)

    def _released(self, slot_id: str) -> None:
        """a claimed result was closed: its files go.

        :param slot_id: the slot
        :ptype slot_id: str
        :return: nothing
        :rtype: None
        """
        with self._state:
            self._claimed.discard(slot_id)
        # its bytes are forgotten by the sweep that removes its files: a publish's room grows only once they are gone
        self.sweep()

    def _committed_bytes(self) -> int:
        """the bytes the directory may be holding: what its files hold, except an open slot's and a hand-off's being
        written, each counted at its bound however much it has written. Called under ``_state``.

        What the tally records (every version, part file, claimed result and written hand-off, a file once) is read
        from it, never walked: a version's parts are as many as it holds, and counting them took a stat each, every
        publish. The few directories it does not record are walked: a result written and not claimed, a staging
        directory, one a sweep is removing.

        :return: bytes
        :rtype: int
        """
        held = self._held.bytes
        for half, owned in ((self._versions, ()), (self._segments, self._slots), (self._handoffs, self._handing)):
            for entry in half.iterdir():
                if entry.is_dir() and entry not in self._held and _slot_of(entry.name) not in owned:
                    held += _directory_bytes(entry)
        return held + self._reserved_bytes()

    def _reserved_bytes(self) -> int:
        """the bytes open slots and hand-offs being written are counted at. Called under ``_state``.

        :return: bytes
        :rtype: int
        """
        return sum(self._slots.values()) + sum(self._handing.values())

    def _sweep_versions(self, gone: list[tuple[Path, Path]]) -> list[str]:
        """retire every version neither current nor held, and every leftover staging directory: each renamed out of
        its name into ``gone``, for the caller to remove once it lets go of ``_state``. Called under ``_state``.

        :param gone: what to remove (:func:`_retire`)
        :ptype gone: list[tuple[Path, Path]]
        :return: the versions retired
        :rtype: list[str]
        """
        retired: list[str] = []
        current = None if self._current is None else self._current.name
        for entry in list(self._versions.iterdir()):
            name = entry.name
            if not entry.is_dir() and name.startswith(_STAGING_PREFIX):
                # the current version's name being written (:meth:`_write_current`, under ``_state``): one left is a
                # write that died
                with contextlib.suppress(FileNotFoundError):  # NOSILENT: gone already
                    entry.unlink()
                continue
            if not entry.is_dir() or name in (current, self._staging, self._pending):
                continue
            if name.startswith(_RETIRING_PREFIX):
                gone.append((entry, entry))
                continue
            if not _retire_unless_held(entry, gone):
                continue
            if not name.startswith(_STAGING_PREFIX):
                retired.append(name)
        return retired

    def _sweep_segments(self, gone: list[tuple[Path, Path]]) -> None:
        """retire every result no open slot or claim owns, and every staging directory whose writer is gone, into
        ``gone`` (as :meth:`_sweep_versions`). Called under ``_state``.

        :param gone: what to remove (:func:`_retire`)
        :ptype gone: list[tuple[Path, Path]]
        :return: nothing
        :rtype: None
        """
        for entry in list(self._segments.iterdir()):
            name = entry.name
            if name.startswith(_RETIRING_PREFIX):
                gone.append((entry, entry))
            elif name.startswith(_STAGING_PREFIX):
                if _slot_of(name) in self._slots:
                    # an open slot's: its writer makes it, then its lock file, then locks it, and a sweep between
                    # those steps took it from under the write (the local replay of 2026-11-01). It goes with the
                    # slot's claim or discard, as an open hand-off's staging goes with the hand-off
                    continue
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
                # a result closed while a table taken from it is still held keeps its shared lock (``pin``)
                _retire_unless_held(entry, gone)

    def _sweep_handoffs(self, gone: list[tuple[Path, Path]]) -> None:
        """retire every hand-off released that no worker maps, and every staging directory no write of this process
        owns, into ``gone`` (as :meth:`_sweep_versions`); forget the bytes of each that is gone. Called under
        ``_state``.

        :param gone: what to remove (:func:`_retire`)
        :ptype gone: list[tuple[Path, Path]]
        :return: nothing
        :rtype: None
        """
        for entry in list(self._handoffs.iterdir()):
            name = entry.name
            if name.startswith(_RETIRING_PREFIX):
                gone.append((entry, entry))
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
        """write a version: its new files into a part directory of its own, its manifest under a staging name, renamed
        into place whole; on any failure keep nothing.

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
        part_dir = self._parts / uuid.uuid7().hex
        imported: Path | None = None
        with self._state:
            self._staging = staging_name
        #: what the directory holds but for open slots and hand-offs being written, as this write starts (its own
        #: files not yet made), and the results claimed and hand-offs written by then
        start: list[Any] = []

        def counted() -> None:
            with self._state:
                start[:] = [
                    self._committed_bytes() - self._reserved_bytes(),
                    set(self._claimed_bytes),
                    set(self._handed_bytes),
                ]

        def room() -> int:
            # asked before every batch: a slot or hand-off reserved while this writes takes its room from this write,
            # never from the bound (each counts what this write has staged so far); a slot claimed or a hand-off
            # written while this writes stops counting at its bound, so its files count from then on instead
            with self._state:
                files, claimed_before, handed_before = start
                claimed = sum(size for slot, size in self._claimed_bytes.items() if slot not in claimed_before)
                handed = sum(size for each, size in self._handed_bytes.items() if each not in handed_before)
                return int(self._max_bytes - files - claimed - handed - self._reserved_bytes())

        def reclaim() -> None:
            # what a publish left to the store's thread may not have gone yet: swept now, then counted again
            self._tell_readers()
            self.sweep()
            counted()

        counted()
        done = False
        try:
            staging.mkdir()
            from_base: _Manifest | None = None
            if base is not None:
                from_base = self._manifests.get(base.name)
                if from_base is None:
                    # a version written before manifests: its files named once, from a part directory of their own
                    imported = self._parts / uuid.uuid7().hex
                    from_base = _imported(base, imported)
            manifest, made, written = _write_parts(part_dir, tables, from_base, room, reclaim)
            text = manifest.joined()
            if len(text) > room() - written:
                reclaim()
                if len(text) > room() - written:
                    raise SharedArrowFullError(f"version {name!r}'s manifest would pass the bytes it may take")
            (staging / _MANIFEST_FILE).write_text(text)
            (staging / _LOCK_FILE).touch()
            staged = _directory_files(staging)
            kept = {} if imported is None else self._kept_imported(manifest, imported)
            os.rename(staging, self._versions / name)
            with self._state:
                # the publish's _pending until it is current, so no sweep retires it before it is recorded
                self._manifests[name] = replace(manifest, file_bytes=len(text))
                self._recorded_parts(part_dir, made)
                if imported is not None:
                    self._recorded_parts(imported, kept)
                self._held.add(self._versions / name, staged)
            done = True
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
            if not done:
                remove_tree(part_dir, log=log, what="a failed version's part files")
                if imported is not None:
                    remove_tree(imported, log=log, what="a failed version's part files named from its base")
            with self._state:
                self._staging = None

    @staticmethod
    def _kept_imported(manifest: _Manifest, imported: Path) -> dict[str, os.stat_result]:
        """of the files a version written before manifests was named by (:func:`_imported`), keep those the new
        version names, remove the rest.

        :param manifest: the new version's manifest
        :ptype manifest: _Manifest
        :param imported: the directory its base's files were named from
        :ptype imported: Path
        :return: each file kept, by name, as ``os.stat`` reads it
        :rtype: dict[str, os.stat_result]
        """
        named = {
            name
            for held in manifest.lines.values()
            for name, line in held.items()
            if _line_fields(line)[1] == imported.name
        }
        kept: dict[str, os.stat_result] = {}
        for entry in list(imported.iterdir()):
            if entry.name in named:
                kept[entry.name] = entry.stat()
            else:
                entry.unlink()
        if not kept:
            os.rmdir(imported)
        return kept


@dataclass(frozen=True)
class SharedRead:
    """one read of a version through a :class:`SharedArrowReader`, valid for its block.

    :ivar version: the version read
    :ivar tables: table name -> the mapped table
    :ivar cursor: a DuckDB cursor with every table as a view over its mapped Arrow, no row copied; once the read ends
        it is handed to a later read of the version (one read at a time), so leave nothing on it and change no setting:
        a cursor holding a temporary view or table it was not made with, or whose read raised, is closed instead
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
        reuse: Mapping[Any, Any] | None = None,
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
        #: cursors finished reads handed back, each with its count of temporary objects when it was made: the next read
        #: borrows one instead of registering every table again on a new cursor; closed as the version is let go
        self.free: list[tuple[Any, int]] = []

    def borrow(self) -> tuple[Any, int]:
        """a cursor of the connection with this version's tables registered on it (a registration is the cursor's
        own, and the views made by :meth:`MappedTables.connect` name them): one a finished read handed back, else a new
        one. Called under the reader's lock; the cursor is the borrower's alone until it is handed back.

        :return: the cursor, and its count of temporary objects as made (:func:`_temporary_objects`)
        :rtype: tuple[duckdb.DuckDBPyConnection, int]
        """
        if self.free:
            return self.free.pop()
        cursor = self.connection.cursor()
        try:
            for name, table in self.mapped.tables.items():
                cursor.register(f"{_REGISTERED_PREFIX}{name}", table)
            made = _temporary_objects(cursor)
        except BaseException:
            cursor.close()
            raise
        return cursor, made

    def close(self) -> None:
        """close the free cursors and the connection, unmap the files and release the lease (whose release retires the
        version when it is not current and nobody else holds it).

        :return: nothing
        :rtype: None
        """
        try:
            free, self.free = self.free, []
            for cursor, _ in free:
                cursor.close()
            self.connection.close()
            self.mapped.close()
        finally:
            self.lease.release()


#: a cursor's temporary views and tables: a read's leftovers, which would shadow a mapped table for the next read
_TEMPORARY_VIEWS: Final = "SELECT count(*) FROM duckdb_views() WHERE temporary"
_TEMPORARY_TABLES: Final = "SELECT count(*) FROM duckdb_tables() WHERE temporary"


def _temporary_objects(cursor: Any) -> int:
    """how many temporary views and tables a cursor sees (its registrations among them).

    :param cursor: the cursor
    :ptype cursor: duckdb.DuckDBPyConnection
    :return: the count
    :rtype: int
    """
    views = cursor.execute(_TEMPORARY_VIEWS).fetchone()
    tables = cursor.execute(_TEMPORARY_TABLES).fetchone()
    return int(views[0]) + int(tables[0])


def _clean_cursor(cursor: Any, made: int) -> bool:
    """whether a read left its cursor as it was made (no temporary view or table added or dropped), so the next read
    may borrow it.

    :param cursor: the cursor
    :ptype cursor: duckdb.DuckDBPyConnection
    :param made: its count of temporary objects as made
    :ptype made: int
    :return: whether it is clean; False when it cannot be read
    :rtype: bool
    """
    try:
        return _temporary_objects(cursor) == made
    except duckdb.Error:
        return False


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
        #: the files of the version let go that the current one shares, by identity, for its first read to take
        #: (:class:`MappedTables`): mapped already, so keeping them keeps no page the current version does not hold
        self._lend: Mapping[Any, Any] = {}
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
                cursor, made = held.borrow()
            except BaseException:
                held.reads -= 1
                raise
        if let_go is not None:
            let_go.close()
        clean = False
        try:
            yield SharedRead(version=held.name, tables=held.mapped.tables, cursor=cursor, prepared=held.prepared)
            clean = True
        finally:
            # handed back only as it was made: a temporary view or table left on it would shadow a mapped table for
            # the next read, and a read that raised may have left anything
            clean = clean and _clean_cursor(cursor, made)
            with self._lock:
                held.reads -= 1
                if clean:
                    held.free.append((cursor, made))
                done = held.superseded and held.reads == 0
            if not clean:
                cursor.close()
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


def private_bytes() -> int:
    """this process's private memory: its anonymous resident pages (Linux ``RssAnon``), never the shared files it
    maps; its peak resident memory where there is no ``/proc`` (a laptop, which counts mapped pages too). What a
    worker over a store is judged by (:class:`SharedArrowPool`'s ``private_bytes``): its resident memory counts every
    shared page it maps, the store's files among them, which are not its own and which replacing it frees nothing of.

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
    return value, error, private_bytes(), os.getpid()


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
