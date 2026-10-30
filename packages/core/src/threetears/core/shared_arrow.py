"""Shared-memory Arrow between a serving process and its worker processes: one physical copy, no pickle, no disk.

**Why.** CPU work in a serving process belongs in worker processes, never on threads: a thread running Python holds
the interpreter lock and the event loop waits behind it. Two things then cost memory and time. A worker cannot read
the serving process's data (its DuckDB, say), and a copy in each worker multiplies the pod's memory by its workers.
And a worker's answer -- a table of rows -- comes back pickled, copied through a pipe and copied again. Both are
solved by writing Arrow IPC files into a memory-backed directory (``/dev/shm``, or a Kubernetes ``emptyDir`` with
``medium: Memory``: RAM, never a disk -- a pod has no scratch disk and must never spill to one) and memory-mapping
them: every process mapping a file shares its pages, and a mapped Arrow column IS the file's bytes.

**Two directions, one owner.** The serving process owns the directory through one :class:`SharedArrowStore`:

- *Down, versions*: :meth:`SharedArrowStore.publish` writes a data version's tables once; workers map it read-only
  (:func:`read_version`) and run DuckDB over it without copying (:meth:`MappedTables.connect`). Each version has a
  directory of its own, staged under a hidden name and renamed into place whole; the newest is current.
- *Up, results*: :meth:`SharedArrowStore.reserve` gives a job a :class:`SegmentSlot` -- a small value that pickles
  over whatever channel hands the job to its worker -- the worker writes its result tables into it
  (:meth:`SegmentSlot.write`), and the serving process maps them (:meth:`SharedArrowStore.claim`): no row is pickled
  and none copied. The serving process owns every slot from reservation to release: it claims it, or discards it
  when the job failed, was abandoned or its worker died.

**Nothing leaks, judged by state, never by time.** A reader of a version -- the serving process's lease
(:meth:`SharedArrowStore.lease`) handed with a job, and the worker while it maps the files -- holds a shared
``flock`` on the version's lock file, and a worker writing a slot holds an exclusive one on its staging lock. The
operating system drops a lock when its holder closes it or dies. A sweep, after every publish, release, claim and
discard, deletes every version that is not current and that no reader holds, every result whose slot nobody owns, and
every staging directory whose writer is gone. A new store removes whatever the directory held that no live process
holds (a container restarted in the same pod keeps its ``emptyDir``). There is no timer and no TTL.

**Bounded.** A store is given the bytes it may hold, every version and result together (``max_bytes``, set below the
volume's own limit, since a pod whose ``emptyDir`` passes its ``sizeLimit`` is evicted). An open slot counts at the
most it may take (its ``max_bytes``); a publish or a reservation that would pass the bound, after a sweep, is refused
with :class:`SharedArrowFullError` and leaves nothing behind, and a worker's write that would pass its slot's bytes
is refused in the worker.

**Worker processes.** :class:`SharedArrowPool` runs jobs ``job(context, *args)`` in spawned, long-lived workers
(spawned, never forked: a serving process runs threads, whose locks a fork copies mid-hold). Each job may read a
version, leased for the job's whole run, and may write a result, claimed back as a :class:`MappedSegment`. A worker
keeps the version it read last mapped between jobs and lets it go after any job that finds a newer one current. A
worker whose private memory (its anonymous pages: the mapped files are shared and not its own) passes the pool's
bound after a job has its pool replaced, so what it allocated goes back to the system.

Guide: ``docs/shared-arrow.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import multiprocessing
import os
import pickle
import resource
import shutil
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Final

from threetears.observe import contain, get_logger

from threetears.core.utils.atomic_write import atomic_write_sync

try:
    import duckdb
    import pyarrow as pa
    import pyarrow.ipc as ipc

    _SNAPSHOT_EXTRA: ImportError | None = None
except ImportError as exc:  # NOSILENT: the snapshot extra is optional; using shared Arrow says so
    _SNAPSHOT_EXTRA = exc

__all__ = [
    "JobContext",
    "JobResult",
    "MappedSegment",
    "MappedTables",
    "MappedVersion",
    "SegmentSlot",
    "SharedArrowFullError",
    "SharedArrowPool",
    "SharedArrowStore",
    "SharedLease",
    "SharedVersion",
    "epochs_version",
    "read_version",
]

log = get_logger(__name__)

#: the store's two halves: published versions, and workers' results
_VERSIONS_DIR: Final = "versions"
_SEGMENTS_DIR: Final = "segments"

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
    if not name or len(name) > _NAME_MAX or name.startswith(".") or not set(name) <= _NAME_CHARS:
        raise ValueError(f"a shared Arrow {what} name must be 1-{_NAME_MAX} of [A-Za-z0-9._-], not hidden: {name!r}")
    if name == _CURRENT_FILE:
        raise ValueError(f"{name!r} is reserved")
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
    """a publish, a reservation or a worker's write would pass its bytes; nothing of it was kept."""


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


def _removed(path: Path) -> None:
    """remove a directory: renamed out of its name first, so nothing opens it while its files go.

    :param path: the directory
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    gone = path.with_name(f"{_RETIRING_PREFIX}{path.name}-{uuid.uuid7().hex[-8:]}")
    try:
        os.rename(path, gone)
    except FileNotFoundError:
        # NOSILENT: already gone, which is what was wanted
        return
    shutil.rmtree(gone, ignore_errors=True)


def _directory_bytes(path: Path) -> int:
    """the bytes the files directly in a directory hold.

    :param path: the directory
    :ptype path: Path
    :return: bytes; 0 when it is gone
    :rtype: int
    """
    total = 0
    with contextlib.suppress(FileNotFoundError):  # NOSILENT: a directory removed meanwhile holds nothing
        for entry in path.iterdir():
            with contextlib.suppress(FileNotFoundError):  # NOSILENT: likewise a file
                total += entry.stat().st_size
    return total


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


def _write_tables(directory: Path, tables: Mapping[str, Any], room: int) -> int:
    """write each table as an uncompressed Arrow IPC file in ``directory``, refusing a batch that would pass ``room``.

    :param directory: where
    :ptype directory: Path
    :param tables: name -> a table or a reader of its batches
    :ptype tables: Mapping[str, pa.Table | pa.RecordBatchReader]
    :param room: the bytes every table together may take
    :ptype room: int
    :return: the bytes written
    :rtype: int
    :raises SharedArrowFullError: when they would take more than ``room``
    """
    options = ipc.IpcWriteOptions(compression=None)
    written = 0
    for name, source in tables.items():
        path = directory / f"{_valid_name(name, 'table')}{_TABLE_SUFFIX}"
        schema, batches = _batches(source)
        with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, schema, options=options) as writer:
            for batch in batches:
                if written + sink.tell() + batch.nbytes > room:
                    raise SharedArrowFullError(f"table {name!r} would pass its {room} bytes")
                writer.write_batch(batch)
        written += path.stat().st_size
    return written


class MappedTables:
    """Arrow tables memory-mapped read-only from a directory: every process mapping it shares its pages, and no row
    is copied into this one. Close it when done; a table kept past :meth:`close` keeps its pages mapped, never its
    lock or its files' place in the store.

    :ivar name: the directory's name (a version's, a result's slot)
    :ivar tables: table name -> the mapped table
    """

    def __init__(self, name: str, path: Path, *, lock: bool) -> None:
        self.name = name
        self._descriptor: int | None = None
        if lock:
            self._descriptor = _locked(path, fcntl.LOCK_SH)
            if self._descriptor is None:
                raise LookupError(f"shared Arrow {name!r} is gone")
        self._maps: list[Any] = []
        tables: dict[str, Any] = {}
        try:
            for entry in sorted(path.iterdir()):
                if entry.name.endswith(_TABLE_SUFFIX):
                    mapped = pa.memory_map(str(entry), "r")
                    self._maps.append(mapped)
                    tables[entry.name[: -len(_TABLE_SUFFIX)]] = ipc.open_file(mapped).read_all()
        except BaseException:
            self._unmap()
            raise
        self.tables: Mapping[str, Any] = MappingProxyType(tables)

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
    :raises LookupError: when the version is gone
    """

    def __init__(self, root: Path | str, version: str) -> None:
        name = _valid_name(version, "version")
        super().__init__(name, Path(root) / _VERSIONS_DIR / name, lock=True)


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
            kept.close()
            kept = None
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
            written = _write_tables(staging, tables, self.max_bytes)
            os.rename(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return written


@dataclass(frozen=True)
class SharedVersion:
    """one version the store holds.

    :ivar name: the version's name
    :ivar path: its directory
    :ivar tables: its tables' names
    :ivar nbytes: the bytes its files hold
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
    files = sorted(entry for entry in path.iterdir() if entry.name.endswith(_TABLE_SUFFIX))
    return SharedVersion(
        name=path.name,
        path=path,
        tables=tuple(entry.name[: -len(_TABLE_SUFFIX)] for entry in files),
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
    """

    def __init__(self, slot_id: str, path: Path, on_close: Callable[[str], object]) -> None:
        super().__init__(slot_id, path, lock=False)
        self._on_close: Callable[[str], object] | None = on_close
        self.nbytes = _directory_bytes(path)

    def close(self) -> None:
        """unmap the result and remove its files. Harmless twice.

        :return: nothing
        :rtype: None
        """
        super().close()
        on_close, self._on_close = self._on_close, None
        if on_close is not None:
            on_close(self.name)


def _streamed(cursor: Any, sql: str, batch_rows: int) -> Callable[[], Any]:
    """what opens a query's answer over ``cursor`` as a reader of record batches, when the write reaches it.

    :param cursor: the DuckDB cursor
    :ptype cursor: duckdb.DuckDBPyConnection
    :param sql: the query
    :ptype sql: str
    :param batch_rows: rows per batch
    :ptype batch_rows: int
    :return: the opener
    :rtype: Callable[[], pa.RecordBatchReader]
    """

    def opened() -> Any:
        answer = cursor.execute(sql)
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
        self._versions.mkdir(parents=True, exist_ok=True)
        self._segments.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        #: held while the current version, the slots or the claims are read or moved, and while the directory is
        #: swept: briefly, never while a table is written
        self._state = threading.Lock()
        #: one publish at a time, so the room each is given is exact
        self._publishing = threading.Lock()
        self._current: SharedVersion | None = None
        #: the version staging directory being written now, which a sweep leaves alone
        self._staging: str | None = None
        #: slots reserved and not yet claimed or discarded, with the bytes each may take
        self._slots: dict[str, int] = {}
        #: results claimed and not yet closed
        self._claimed: set[str] = set()
        self.published = 0
        self.retired = 0
        self.refused = 0
        with contextlib.suppress(FileNotFoundError):  # NOSILENT: nothing current is the start's state either way
            (self._versions / _CURRENT_FILE).unlink()
        self.sweep()

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

    def held_bytes(self) -> int:
        """the bytes the directory's files hold now, staged ones included.

        :return: bytes
        :rtype: int
        """
        return sum(
            _directory_bytes(entry)
            for half in (self._versions, self._segments)
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
        return {
            "current": current,
            "versions": list(self.versions()),
            "slots_open": slots,
            "segments_claimed": claimed,
            "held_bytes": self.held_bytes(),
            "max_bytes": self._max_bytes,
            "published": self.published,
            "retired": self.retired,
            "refused": self.refused,
        }

    def publish(self, version: str, tables: Mapping[str, Any]) -> SharedVersion:
        """write ``tables`` as ``version`` and make it current; a version already present is made current as it is.

        :param version: the version's name (:func:`epochs_version`)
        :ptype version: str
        :param tables: name -> a ``pyarrow.Table``, a ``pyarrow.RecordBatchReader`` of its rows, or a callable taking no
            argument that returns either, called only as the write reaches the table (never when the version is
            present already)
        :ptype tables: Mapping[str, pa.Table | pa.RecordBatchReader | Callable[[], pa.Table | pa.RecordBatchReader]]
        :return: the version
        :rtype: SharedVersion
        :raises SharedArrowFullError: when it would pass ``max_bytes``; nothing of it is kept
        :raises ValueError: for a version or table name that is not one plain path segment
        """
        name = _valid_name(version, "version")
        for table in tables:
            _valid_name(table, "table")
        target = self._versions / name
        with self._publishing:
            if not (target / _LOCK_FILE).is_file():
                self.sweep()
                self._write_version(name, tables)
            with self._state:
                published = _described(target)
                atomic_write_sync(self._versions / _CURRENT_FILE, name, mode=0o644)
                self._current = published
                self.published += 1
        self.sweep()
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
        return self.publish(version, {table: _streamed(cursor, sql, batch_rows) for table, sql in queries.items()})

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
        with self._state:
            if self._committed_bytes() + max_bytes > self._max_bytes:
                self.refused += 1
                raise SharedArrowFullError(f"no room for a {max_bytes}-byte result in {self._max_bytes} bytes")
            slot_id = f"s-{uuid.uuid7().hex}"
            self._slots[slot_id] = max_bytes
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

    def sweep(self) -> list[str]:
        """remove every version neither current nor held, every result no open slot or claim owns, and every staging
        directory whose writer is gone.

        :return: the versions retired
        :rtype: list[str]
        """
        with self._state:
            retired = self._sweep_versions()
            self._sweep_segments()
            self.retired += len(retired)
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

    def _committed_bytes(self) -> int:
        """the bytes the directory may be holding: what its files hold, except an open slot's, which counts at its
        ``max_bytes`` however much it has written. Called under ``_state``.

        :return: bytes
        :rtype: int
        """
        held = sum(_directory_bytes(entry) for entry in self._versions.iterdir() if entry.is_dir())
        held += sum(
            _directory_bytes(entry)
            for entry in self._segments.iterdir()
            if entry.is_dir() and _slot_of(entry.name) not in self._slots
        )
        return held + sum(self._slots.values())

    def _sweep_versions(self) -> list[str]:
        """remove every version neither current nor held, and every leftover staging directory. Called under
        ``_state``.

        :return: the versions retired
        :rtype: list[str]
        """
        retired: list[str] = []
        current = None if self._current is None else self._current.name
        for entry in list(self._versions.iterdir()):
            name = entry.name
            if not entry.is_dir() or name in (current, self._staging):
                continue
            if name.startswith(_RETIRING_PREFIX):
                shutil.rmtree(entry, ignore_errors=True)
                continue
            descriptor = _locked(entry, fcntl.LOCK_EX)
            if descriptor is None and (entry / _LOCK_FILE).is_file():
                continue  # a reader holds it
            try:
                _removed(entry)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            if not name.startswith(_STAGING_PREFIX):
                retired.append(name)
        return retired

    def _sweep_segments(self) -> None:
        """remove every result no open slot or claim owns, and every staging directory whose writer is gone. Called
        under ``_state``.

        :return: nothing
        :rtype: None
        """
        for entry in list(self._segments.iterdir()):
            name = entry.name
            if name.startswith(_RETIRING_PREFIX):
                shutil.rmtree(entry, ignore_errors=True)
            elif name.startswith(_STAGING_PREFIX):
                # a writer still writing holds its lock; one that died, or whose write failed, does not
                descriptor = _locked(entry, fcntl.LOCK_EX)
                if descriptor is not None:
                    try:
                        _removed(entry)
                    finally:
                        os.close(descriptor)
                elif not (entry / _LOCK_FILE).exists():
                    _removed(entry)
            elif name not in self._slots and name not in self._claimed:
                _removed(entry)

    def _write_version(self, name: str, tables: Mapping[str, Any]) -> None:
        """write a version under a staging name and rename it into place whole; on any failure keep nothing.

        :param name: the version
        :ptype name: str
        :param tables: its tables
        :ptype tables: Mapping[str, Any]
        :return: nothing
        :rtype: None
        :raises SharedArrowFullError: when it would pass ``max_bytes``
        """
        staging_name = f"{_STAGING_PREFIX}{name}-{uuid.uuid7().hex[-8:]}"
        staging = self._versions / staging_name
        with self._state:
            self._staging = staging_name
            room = self._max_bytes - self._committed_bytes()
        try:
            staging.mkdir()
            _write_tables(staging, tables, room)
            (staging / _LOCK_FILE).touch()
            os.rename(staging, self._versions / name)
        except SharedArrowFullError:
            self.refused += 1
            log.warning(
                "shared Arrow version refused: it would pass the store's bytes",
                extra={"extra_data": {"version": name, "max_bytes": self._max_bytes}},
            )
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            with self._state:
                self._staging = None


def _slot_of(name: str) -> str:
    """the slot a result directory belongs to: its name, or its staging name's slot.

    :param name: the directory's name
    :ptype name: str
    :return: the slot
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
    """

    version: MappedVersion | None
    result: SegmentSlot | None


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
    root: str, version: str | None, slot: SegmentSlot | None, job: Callable[..., Any], args: tuple[Any, ...]
) -> tuple[Any, BaseException | None, int, int]:
    """a worker's job: ``job(context, *args)``, over the version leased for it and with the slot reserved for it.

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
    :return: its value; what it raised (None for nothing); this worker's private bytes after; its process id
    :rtype: tuple[Any, BaseException | None, int, int]
    """
    value: Any = None
    error: BaseException | None = None
    try:
        if version is None:
            value = job(JobContext(version=None, result=slot), *args)
        else:
            with read_version(root, version) as mapped:
                value = job(JobContext(version=mapped, result=slot), *args)
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


class _Handoff:
    """one run's lease and slot, each released exactly once whichever way the run ends: the lease when the worker is
    done with the version, the slot claimed by the caller or discarded once the worker is done with it (the job
    failed, the caller stopped waiting, the worker died).

    :param store: the store
    :ptype store: SharedArrowStore
    :param lease: the version's lease; None for none
    :ptype lease: SharedLease | None
    """

    def __init__(self, store: SharedArrowStore, lease: SharedLease | None) -> None:
        self._store = store
        self._lease = lease
        self.slot: SegmentSlot | None = None

    def worker_done(self, _: Future[Any]) -> None:
        """the worker is done with the version: the lease goes.

        :return: nothing
        :rtype: None
        """
        if self._lease is not None:
            self._lease.release()

    def abandon(self, future: Future[Any]) -> None:
        """nobody will claim the slot: discarded once the worker is done with it (now, when it already is).

        :param future: the job's future
        :ptype future: Future[Any]
        :return: nothing
        :rtype: None
        """
        slot = self.slot
        if slot is not None:
            future.add_done_callback(lambda _: self._store.discard(slot))

    def release_all(self) -> None:
        """release what was taken, the job never submitted.

        :return: nothing
        :rtype: None
        """
        if self._lease is not None:
            self._lease.release()
        if self.slot is not None:
            self._store.discard(self.slot)


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
        self.jobs = 0
        self.replaced = 0

    async def run(
        self, job: Callable[..., Any], *args: Any, read: bool | str = True, result_bytes: int | None = None
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
        :return: its value, and its result tables mapped here (the caller closes them)
        :rtype: JobResult
        :raises LookupError: when nothing is published, or the version named is gone
        :raises SharedArrowFullError: when the store has no room for ``result_bytes``
        :raises Exception: what the job raised; ``BrokenProcessPool`` when its worker died (the pool is replaced)
        """
        # leased, reserved and submitted on a thread: a fresh pool spawns its workers there, and a lease waits on a
        # sweep
        handoff, future = await asyncio.to_thread(self._submit, read, result_bytes, job, args)
        claimed = False
        try:
            try:
                value, error, private, worker = await asyncio.wrap_future(future)
            except BrokenProcessPool:
                self._stale = True
                raise
            self.jobs += 1
            if private > self._private_bytes:
                log.info(
                    "shared Arrow worker past its private memory; its pool is replaced",
                    extra={"extra_data": {"pool": self._name, "worker": worker, "private_bytes": private}},
                )
                self._stale = True
            if error is not None:
                raise error
            segment = None
            if handoff.slot is not None:
                segment = await asyncio.to_thread(self._store.claim, handoff.slot)
            claimed = True
            return JobResult(value=value, segment=segment)
        finally:
            if not claimed:
                handoff.abandon(future)

    def _submit(
        self, read: bool | str, result_bytes: int | None, job: Callable[..., Any], args: tuple[Any, ...]
    ) -> tuple[_Handoff, Future[tuple[Any, BaseException | None, int, int]]]:
        """lease the version, reserve the slot and submit the job to the current pool, or a fresh one; on a thread.

        :param read: as :meth:`run`
        :ptype read: bool | str
        :param result_bytes: as :meth:`run`
        :ptype result_bytes: int | None
        :param job: the job
        :ptype job: Callable[..., Any]
        :param args: its arguments
        :ptype args: tuple[Any, ...]
        :return: the run's hand-off and the job's future (the lease is released when it is done)
        :rtype: tuple[_Handoff, Future[tuple[Any, BaseException | None, int, int]]]
        """
        lease = None if read is False else self._store.lease(None if read is True else read)
        handoff = _Handoff(self._store, lease)
        try:
            if result_bytes is not None:
                handoff.slot = self._store.reserve(result_bytes)
            with self._lock:
                if self._pool is not None and self._stale:
                    self._pool.shutdown(wait=False)
                    self._pool = None
                    self.replaced += 1
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
                    handoff.slot,
                    job,
                    args,
                )
        except BaseException:
            handoff.release_all()
            raise
        future.add_done_callback(handoff.worker_done)
        return handoff, future

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
