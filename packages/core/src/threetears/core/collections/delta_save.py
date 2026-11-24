"""a write's delta files built and saved in a worker process: the serving process hands Arrow and waits.

**Why.** A commit's delta files are saved before the commit (``design-pod-delta-files.md``, decision 11), and the
save is CPU: each part typed as the table's columns, joined, compressed with zstd and hashed (sha256), on top of the
write's own work. It ran on a thread of the serving process; the 2026-11-23 audit moved it out. Measured
(2026-11-23, macOS arm64, Python 3.14, a load of 337,000 rows of 179 columns, 527 MiB of Arrow): saved on a thread it
cost the serving process 1,431 ms of CPU; see :func:`save_in_worker` for what it costs now.

**What crosses, and how.** Plain data (:mod:`~threetears.core.process_workers`): each table's name, key, types (an
empty Arrow schema's IPC bytes), defaults, prefixes, and its parts in order (:meth:`TableDelta.segments`): a small
part as Arrow IPC bytes, a part past :data:`_INLINE_BYTES` through a shared-memory hand-off
(:meth:`~threetears.core.shared_arrow.SharedArrowStore.hand`: written as Arrow IPC with the interpreter lock released,
mapped read-only by the worker, never pickled), and a load's parquet file by its path (the worker reads it). Back
come the files' names. Rows a write handed as dicts are typed into Arrow first, where the dicts are, by Arrow's own
conversion (:func:`~threetears.core.collections.delta_files._rows_table`): the one step that cannot cross, since a
dict crosses only pickled.

**Where the hand-off lives.** ``THREETEARS_DELTA_HANDOFF_DIR``: a directory in memory (a pod's ``emptyDir`` with
``medium: Memory``, never a disk), bounded by ``THREETEARS_DELTA_HANDOFF_BYTES``. Unset, or refusing a part (full), a
save with a large part runs on a thread of the serving process as before, counted
(``delta_file_saves_total{where="thread"}``) and logged once, so a pod without the directory saves as it did rather
than refusing its writes. A save with only small parts never needs it.

**The commit still waits for its files** (decision 11): the worker's answer comes before the commit is sent. A worker
that dies fails the save (``BrokenProcessPool``), and the write is refused, never committed without its file; a job
that fails part way removes the files it saved.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from threetears.observe import get_logger
from threetears.observe.metrics import counter, histogram

from threetears.core.collections.delta_files import (
    TableDelta,
    _delta_file_name,
    _partial_name,
    build_file_table,
    write_delta_file,
    write_table_file,
)
from threetears.core.process_workers import ProcessWorkers
from threetears.core.shared_arrow import HandedTables, SharedArrowFullError, SharedArrowStore

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a delta file refuses without it, saying so
    _PYARROW = False

__all__ = ["save_in_worker"]

log = get_logger(__name__)

#: the memory-backed directory large parts cross to the worker through (module docstring)
_HANDOFF_DIR_ENV: Final = "THREETEARS_DELTA_HANDOFF_DIR"
#: the most bytes the hand-off's directory may hold
_HANDOFF_BYTES_ENV: Final = "THREETEARS_DELTA_HANDOFF_BYTES"
#: the hand-off's bound when the environment names none: one large load's parts at a time
_DEFAULT_HANDOFF_BYTES: Final = 2 << 30

#: a part crosses inline, as IPC bytes, up to this: pickled on the pool's feeder thread with the interpreter lock held,
#: about 1 ms a MiB at most (a 527 MiB part held the loop 15-48 ms, measured 2026-11-23), so a part past it is handed
#: through shared memory instead
_INLINE_BYTES: Final = 8 << 20

#: worker processes for saves: a load's requests side by side (``connections``) save side by side
_SAVE_WORKERS: Final = 2

#: saves by where they ran: ``worker`` (a worker process) or ``thread`` (the serving process, no hand-off directory)
_SAVES = counter(
    "delta_file_saves_total",
    description="writes' delta file saves by where they ran: worker (a worker process) or thread (a thread of the "
    "serving process: a large part and no shared-memory hand-off directory, THREETEARS_DELTA_HANDOFF_DIR)",
    label_names=("where",),
)

#: each save's seconds, from the write handing its deltas to its files saved (the commit waits on this), by where it
#: ran (:data:`_SAVES`): observed where the save completes, a histogram observe on the loop and nothing more
_SAVE_SECONDS = histogram(
    "delta_file_save_seconds",
    description="seconds of a write's delta file save, from its deltas handed to its files saved (its commit waits on "
    "it), by where it ran: worker (a worker process) or thread (a thread of the serving process)",
    label_names=("where",),
)

#: the pool every save of this process runs in
_WORKERS: Final = ProcessWorkers(workers=_SAVE_WORKERS, name="delta file saves", process_name="3t-delta-save")


class _Handoffs:
    """the hand-off store of the directory the environment names, made at its first large part (on a thread: a store's
    start sweeps its directory), and whether the thread fallback was said. One store, replaced whole when the
    directory named changes (a test's): never a mapping another call reaches."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: tuple[str, SharedArrowStore] | None = None
        self.warned = False

    def store(self) -> SharedArrowStore | None:
        """the store of the directory the environment names, or None when it names none. Blocking.

        :return: the store, or None
        :rtype: SharedArrowStore | None
        """
        root = os.environ.get(_HANDOFF_DIR_ENV, "").strip()
        if not root:
            return None
        with self._lock:
            if self._held is None or self._held[0] != root:
                max_bytes = int(os.environ.get(_HANDOFF_BYTES_ENV, "").strip() or _DEFAULT_HANDOFF_BYTES)
                Path(root).mkdir(parents=True, exist_ok=True)
                self._held = (root, SharedArrowStore(Path(root), max_bytes=max_bytes, name="delta file saves"))
            return self._held[1]


_HANDOFFS: Final = _Handoffs()


def _ipc_bytes(table: Any) -> bytes:
    """a table as Arrow IPC stream bytes. Blocking.

    :param table: the table
    :ptype table: pyarrow.Table
    :return: the bytes
    :rtype: bytes
    """
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue().to_pybytes())


def _ipc_table(data: bytes) -> Any:
    """IPC stream bytes as their table. Blocking.

    :param data: the bytes
    :ptype data: bytes
    :return: the table
    :rtype: pyarrow.Table
    """
    return pa_ipc.open_stream(pa.BufferReader(data)).read_all()


def _planned(deltas: Mapping[str, TableDelta]) -> tuple[tuple[tuple[Any, ...], ...], Mapping[str, Any]]:
    """every table's plain plan, and the tables too large to cross inline, by their names in the hand-off. Blocking: on
    a thread.

    :param deltas: table -> its delta
    :ptype deltas: Mapping[str, TableDelta]
    :return: each table's ``(name, key, schema IPC, defaults, prefixes, segments)``; the large tables, read-only
    :rtype: tuple[tuple, Mapping[str, pyarrow.Table]]
    """
    large: dict[str, Any] = {}

    def source(table: Any) -> tuple[str, Any]:
        # inline as IPC bytes, or by its name in the hand-off
        if table.nbytes <= _INLINE_BYTES:
            return ("ipc", _ipc_bytes(table))
        name = f"p{len(large)}"
        large[name] = table
        return ("handed", name)

    plans: list[tuple[Any, ...]] = []
    for delta in deltas.values():
        schema = pa.schema([pa.field(name, kind) for name, kind in delta.types.items()])
        segments: list[tuple[Any, ...]] = []
        for kind, part, *rest in delta.segments():
            if kind == "load":
                given = tuple(
                    source(rows if isinstance(rows, pa.Table) else pa.Table.from_batches([rows]))
                    if isinstance(rows, (pa.Table, pa.RecordBatch))
                    else ("path", os.fspath(rows))
                    for rows in part
                )
                segments.append(("load", given, tuple(rest[0]), _constants_bytes(rest[1])))
            elif kind == "deleted":
                segments.append(("deleted", source(part)))
            else:
                written = rest[0]
                plain = tuple(written) if not written or isinstance(written[0], str) else tuple(map(tuple, written))
                segments.append((kind, source(part), plain))
        plans.append(
            (
                delta.name,
                tuple(delta.key),
                bytes(schema.serialize().to_pybytes()),
                tuple(delta.defaults.items()),
                tuple(tuple(prefix.items()) for prefix in delta.prefixes),
                tuple(segments),
            )
        )
    return tuple(plans), MappingProxyType(large)


def _save_all_here(root: Path, deltas: Mapping[str, TableDelta], write_id: str) -> dict[str, str]:
    """every file saved on this thread, a failure removing those already saved. Blocking.

    :param root: the delta files' directory
    :ptype root: Path
    :param deltas: table -> its delta
    :ptype deltas: Mapping[str, TableDelta]
    :param write_id: the write's unique id
    :ptype write_id: str
    :return: table -> its file's name
    :rtype: dict[str, str]
    """
    saved: dict[str, str] = {}
    try:
        for table, delta in deltas.items():
            saved[table] = write_delta_file(root, delta, write_id)
    except BaseException:
        _remove(root, list(saved.values()))
        raise
    return saved


def _remove(root: Path, names: Sequence[str]) -> None:
    """remove files a save made before it failed, and their temporary names. Blocking.

    :param root: the delta files' directory
    :ptype root: Path
    :param names: the files, under the root
    :ptype names: Sequence[str]
    :return: nothing
    :rtype: None
    """
    for name in names:
        (root / name).unlink(missing_ok=True)
        (root / _partial_name(name)).unlink(missing_ok=True)


async def save_in_worker(root: Path, deltas: Mapping[str, TableDelta], write_id: str) -> dict[str, str]:
    """save every table's delta file in a worker process (module docstring), one write id for them all; the loop only
    waits. Measured (2026-11-23, macOS arm64, Python 3.14): see the commit that made it.

    :param root: the delta files' directory
    :ptype root: Path
    :param deltas: table -> what the write wrote to it
    :ptype deltas: Mapping[str, TableDelta]
    :param write_id: the write's unique id
    :ptype write_id: str
    :return: table -> its file's name under the root
    :rtype: dict[str, str]
    :raises OSError: when a file cannot be written
    :raises ValueError: as :func:`~threetears.core.collections.delta_files.build_file_table`
    :raises BrokenProcessPool: when the worker died
    """
    began = time.monotonic()
    plans, large = await asyncio.to_thread(_planned, deltas)
    handed: HandedTables | None = None
    store: SharedArrowStore | None = None
    if large:
        store = await asyncio.to_thread(_HANDOFFS.store)
        if store is not None:
            try:
                handed = await asyncio.to_thread(store.hand, large)
            except SharedArrowFullError:  # NOSILENT: counted and logged below, saved on a thread as before
                store = None
        if store is None:
            _SAVES.labels(where="thread").inc()
            if not _HANDOFFS.warned:
                _HANDOFFS.warned = True
                log.warning(
                    "a delta file save with a large part runs on a thread of the serving process: no shared-memory "
                    "hand-off directory has room for it (THREETEARS_DELTA_HANDOFF_DIR)",
                    extra={"extra_data": {"bytes": sum(table.nbytes for table in large.values())}},
                )
            saved = await asyncio.to_thread(_save_all_here, root, deltas, write_id)
            _SAVE_SECONDS.labels(where="thread").observe(time.monotonic() - began)
            return saved
    handoff = None if handed is None else (handed.root, handed.handoff_id, tuple(handed.tables), int(handed.nbytes))
    try:
        answer = await _WORKERS.run(save_delta_job, str(root), write_id, plans, handoff)
    finally:
        if handed is not None and store is not None:
            await asyncio.to_thread(store.release, handed)
    _SAVES.labels(where="worker").inc()
    _SAVE_SECONDS.labels(where="worker").observe(time.monotonic() - began)
    return dict(answer)


def _constants_bytes(constants: Mapping[str, Any]) -> bytes:
    """a load's constants as plain data a worker takes: one Arrow row, as IPC bytes (:func:`_constants_of` reads them
    back). Blocking.

    Why: a constant is any value a column takes (an LSF run's timestamp is a ``datetime``), and a worker is handed
    plain data alone (:func:`~threetears.core.process_workers.require_plain`): handed as ``(column, value)`` pairs, a
    load with a ``datetime`` constant was refused before its file was saved, so every LSF load from files failed
    (2026-11-24, test_refresh_live's rollout tests). Arrow types each value as ``pa.repeat`` would where the file is made.

    :param constants: column -> the value every row takes for it
    :ptype constants: Mapping[str, Any]
    :return: the bytes; empty for no constant
    :rtype: bytes
    """
    if not constants:
        return b""
    return _ipc_bytes(pa.table({column: [value] for column, value in constants.items()}))


def _constants_of(data: bytes) -> dict[str, Any]:
    """a load's constants as :func:`_constants_bytes` handed them: column -> its value, an Arrow scalar.

    :param data: the bytes
    :ptype data: bytes
    :return: the constants
    :rtype: dict[str, Any]
    """
    if not data:
        return {}
    row = _ipc_table(data)
    return {column: row.column(column)[0] for column in row.column_names}


def _resolved(source: tuple[str, Any], handed: Mapping[str, Any]) -> Any:
    """a part's table in the worker.

    :param source: ``("ipc", bytes)``, ``("handed", name)`` or ``("path", parquet path)``
    :ptype source: tuple[str, Any]
    :param handed: the hand-off's tables, mapped
    :ptype handed: Mapping[str, pyarrow.Table]
    :return: the table, or the path as is
    :rtype: Any
    """
    how, value = source
    if how == "ipc":
        return _ipc_table(value)
    if how == "handed":
        return handed[value]
    return value


def save_delta_job(
    root: str,
    write_id: str,
    plans: tuple[tuple[Any, ...], ...],
    handoff: tuple[str, str, tuple[str, ...], int] | None,
) -> tuple[tuple[str, str], ...]:
    """build and save each table's delta file, in a worker process (:func:`save_in_worker`); a failure removes the
    files this job saved and raises.

    :param root: the delta files' directory
    :ptype root: str
    :param write_id: the write's unique id
    :ptype write_id: str
    :param plans: each table's plan (:func:`_planned`)
    :ptype plans: tuple
    :param handoff: the shared-memory hand-off's ``(root, id, tables, bytes)``; None when every part crossed inline
    :ptype handoff: tuple | None
    :return: each table and its file's name
    :rtype: tuple[tuple[str, str], ...]
    """
    base = Path(root)
    saved: list[tuple[str, str]] = []
    mapped = None if handoff is None else HandedTables(*handoff).mapped()
    try:
        handed = {} if mapped is None else mapped.tables
        for name, key, schema_bytes, defaults, prefixes, segments in plans:
            schema = pa_ipc.read_schema(pa.py_buffer(schema_bytes))
            types = {field.name: field.type for field in schema}
            resolved: list[tuple[Any, ...]] = []
            for kind, source, *rest in segments:
                if kind == "load":
                    given = [_resolved(entry, handed) for entry in source]
                    resolved.append(("load", given, list(rest[0]), _constants_of(rest[1])))
                elif kind == "deleted":
                    resolved.append(("deleted", _resolved(source, handed)))
                else:
                    written = rest[0]
                    plain = list(written) if not written or isinstance(written[0], str) else [list(w) for w in written]
                    resolved.append((kind, _resolved(source, handed), plain))
            table = build_file_table(
                name, types, key, dict(defaults), [dict(prefix) for prefix in prefixes], resolved, write_id
            )
            saved.append((name, write_table_file(base, name, write_id, table)))
    except BaseException:
        _remove(base, [_delta_file_name(name, write_id) for name, *_ in plans])
        raise
    finally:
        if mapped is not None:
            mapped.close()
    return tuple(saved)
