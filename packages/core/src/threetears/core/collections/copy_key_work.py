"""a copy follower's key work, the CPU of a pass that scales with its keys, run in its worker process.

**Why.** CPU in a serving process belongs in a worker process, never on the event loop and never on a thread
(:mod:`~threetears.core.process_workers`): a thread's Arrow kernels release the interpreter lock, but they still take
the pod's cores from page serving. A pass's key work held the loop 12.2 s in one page (``distinct_keys`` under
``_stage_keyed``) and 24.3 s more hearing keys (``key_text_table`` under ``_drained``) in the local burst replay of
2026-11-25, and a waited-for file's 200,000 keys held one turn 11.1 ms on the pool host brooks-14th-mbp. Each piece
runs here instead, in the copy's own worker process (:class:`~threetears.core.collections.copy_follower.CopyFollower`'s
one :class:`~threetears.core.process_workers.ProcessWorkers`, the one its drift checks run in).

**What crosses.** Plain data (:mod:`~threetears.core.process_workers`): the key's columns, its timestamp columns, a
scope's prefix, the held scopes, epochs as text, every table as Arrow IPC stream bytes, and back the tables as IPC bytes
and the flags as numpy ``bool`` bytes. Each ``*_job`` is the worker's entry; its pure core (the same name without
``_job``) runs on the caller's thread for input under :data:`INLINE_KEYS`, where the round trip costs more than the work.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, Final

from threetears.core.collections.delta_files import joined_key_texts, key_text_table
from threetears.core.utils.cpu_quota import bound_native_threads

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as pa_ipc

    # Arrow's own key kernels: a module of the Arrow extra, so imported with it
    from threetears.core.arrow_keys import distinct_keys
except ImportError:  # NOSILENT: the snapshot extra is optional; a copy follower says so when built
    pa = None

__all__ = [
    "INLINE_KEYS",
    "flags_in",
    "from_ipc",
    "heard_job",
    "heard_of",
    "held_scopes_answer",
    "held_scopes_job",
    "held_scopes_of",
    "ipc_bytes",
    "page_scopes_answer",
    "page_scopes_job",
    "page_scopes_of",
    "reached_job",
    "reached_of",
    "scope_texts",
    "texts_of",
]

#: a heard batch's epoch column
_EPOCH: Final = "__epoch"

#: a key's scope column, beside its key columns
_SCOPE: Final = "__scope"

#: below this many keys the work runs on the caller's thread: the worker's round trip (the IPC written, pickled through
#: the pool's pipe and read back) costs more than the work. Measured 2026-11-25 (macOS arm64, Python 3.14, a loaded
#: host): a 200,000-key page's heard batch took 3-8 ms on a thread and 25-144 ms through the worker (12-30 ms of it the
#: serving process's CPU); its page scopes 196-290 ms on a thread (all of it the serving process's CPU) and 342-656 ms
#: through the worker (43-74 ms of it the serving process's CPU)
INLINE_KEYS: Final = 2_000


def _one_cpu() -> bool:
    """bound this worker's Arrow pools to one CPU (each job: setting a pool to its own size changes nothing): the copy's
    worker is one process doing one job at a time, and pyarrow sizes its pools from the machine's cores (64 compute threads a worker on a 64-core host, each
    counted against the pod's thread limit: "can't start new thread" on the pool host monad-1, 2026-11-25).

    :return: whether pyarrow was there to bound
    :rtype: bool
    """
    return bound_native_threads(1)


def ipc_bytes(table: Any) -> bytes:
    """a table as Arrow IPC stream bytes.

    :param table: the table
    :ptype table: pyarrow.Table
    :return: the bytes
    :rtype: bytes
    """
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue())


def from_ipc(data: bytes) -> Any:
    """a table from Arrow IPC stream bytes (its buffers the bytes', not copied).

    :param data: the bytes
    :ptype data: bytes
    :return: the table
    :rtype: pyarrow.Table
    """
    return pa_ipc.open_stream(pa.py_buffer(data)).read_all()


def _flags(flags: Any) -> bytes:
    return np.asarray(flags, dtype=bool).tobytes()


def _unflags(data: bytes) -> Any:
    return np.frombuffer(data, dtype=bool) if data else None


def flags_in(values: Any, among: Any) -> Any:
    """whether each value is among ``among``, as numpy flags (a null is not).

    :param values: the values
    :ptype values: pyarrow.Array | pyarrow.ChunkedArray
    :param among: the values looked for
    :ptype among: pyarrow.Array
    :return: one flag a value
    :rtype: numpy.ndarray
    """
    found = pc.fill_null(pc.is_in(values, value_set=among), False)
    return np.asarray(found.to_numpy(zero_copy_only=False), dtype=bool)


def texts_of(column: Any) -> Any:
    """a column's values as text, a null as ``None`` (``str(value)``'s spelling of each, as a scope is named).

    :param column: the column
    :ptype column: pyarrow.Array | pyarrow.ChunkedArray
    :return: the texts
    :rtype: pyarrow.Array
    """
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    texts = column if pa.types.is_string(column.type) else column.cast(pa.string())
    return pc.fill_null(texts, "None")


def scope_texts(prefix: str, column: Any) -> Any:
    """each value of a scope column as the scope a row holding it belongs to: ``prefix`` and the value's text.

    :param prefix: the scope's prefix (``state:``)
    :ptype prefix: str
    :param column: the scope column's values
    :ptype column: pyarrow.Array | pyarrow.ChunkedArray
    :return: the scopes
    :rtype: pyarrow.Array
    """
    return pc.binary_join_element_wise(pa.scalar(prefix), texts_of(column), "")


def heard_of(key: Sequence[str], instants: Collection[str], keys: Any, epoch: str | None) -> Any:
    """keys heard at one epoch, as one immutable batch: each key column as text (a timestamp key column's in its
    canonical text), and the epoch.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param keys: one row a key, its key columns' values, in key order (by position)
    :ptype keys: pyarrow.Table
    :param epoch: the epoch they were heard at; None when heard with none
    :ptype epoch: str | None
    :return: the batch
    :rtype: pyarrow.Table
    """
    count = keys.num_rows
    named = pa.table({column: keys.column(at) for at, column in enumerate(key)})
    texts = key_text_table(named, key, instants)
    columns = [texts.column(column).combine_chunks() for column in key]
    epochs = pa.nulls(count, pa.string()) if epoch is None else pa.repeat(pa.scalar(epoch, pa.string()), count)
    schema = pa.schema([(column, pa.string()) for column in key] + [(_EPOCH, pa.string())])
    return pa.Table.from_arrays([*columns, epochs], schema=schema)


def heard_job(
    key: tuple[str, ...], instants: frozenset[str], keys: tuple[bytes, ...], epochs: tuple[str | None, ...]
) -> tuple[bytes, ...]:
    """:func:`heard_of` of each batch of keys at its epoch: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param keys: each batch's keys, IPC
    :ptype keys: tuple[bytes, ...]
    :param epochs: each batch's epoch
    :ptype epochs: tuple[str | None, ...]
    :return: each heard batch, IPC
    :rtype: tuple[bytes, ...]
    """
    _one_cpu()
    return tuple(
        ipc_bytes(heard_of(key, instants, from_ipc(batch), epoch)) for batch, epoch in zip(keys, epochs, strict=True)
    )


def reached_of(key: Sequence[str], instants: Collection[str], held: Sequence[Any], known: Any, epoch: str) -> Any:
    """the keys a cleared prefix reaches, each once, as the batch heard at the clearing epoch: the keys of the copy's
    rows under it, and those heard or staged under it.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param held: the copy's rows under the prefix (its key columns, as the copy holds them), in batches
    :ptype held: Sequence[pyarrow.Table]
    :param known: the keys heard or staged under it, key columns as text
    :ptype known: pyarrow.Table
    :param epoch: the clearing epoch
    :ptype epoch: str
    :return: the batch
    :rtype: pyarrow.Table
    """
    reached = [key_text_table(rows, key, instants) for rows in held] + [known.select(list(key))]
    keys = distinct_keys(pa.concat_tables(reached), list(key))
    return heard_of(key, instants, keys, epoch)


def reached_job(
    key: tuple[str, ...], instants: frozenset[str], held: tuple[bytes, ...], known: bytes, epoch: str
) -> bytes:
    """:func:`reached_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param held: the copy's rows under the prefix, IPC
    :ptype held: tuple[bytes, ...]
    :param known: the keys heard or staged under it, IPC
    :ptype known: bytes
    :param epoch: the clearing epoch
    :ptype epoch: str
    :return: the batch, IPC
    :rtype: bytes
    """
    _one_cpu()
    return ipc_bytes(reached_of(key, instants, [from_ipc(rows) for rows in held], from_ipc(known), epoch))


def held_scopes_of(
    key: Sequence[str], instants: Collection[str], scope: tuple[str, str], page: Any, held: Any, scopes: Any
) -> tuple[Any, Any]:
    """the rows the copy holds of a page's keys as each key and its scope, and which keys of the page are staged now (a
    key whose held row is of a scope a write holds is left out).

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param scope: the scope column and its prefix
    :ptype scope: tuple[str, str]
    :param page: the page's keys, key columns as text
    :ptype page: pyarrow.Table
    :param held: the copy's rows of them: key and scope columns as the copy holds them
    :ptype held: pyarrow.Table
    :param scopes: the scopes a write in progress holds; None for none
    :ptype scopes: pyarrow.Array | None
    :return: the held rows' keys as text and ``__scope``, and per key of the page whether it is staged now (None:
        every one is)
    :rtype: tuple[pyarrow.Table, numpy.ndarray | None]
    """
    mine = key_text_table(held, key, instants).append_column(_SCOPE, scope_texts(scope[1], held.column(scope[0])))
    if scopes is None or not mine.num_rows:
        return mine, None
    late = mine.filter(pa.array(flags_in(mine.column(_SCOPE), scopes)))
    if not late.num_rows:
        return mine, None
    return mine, ~flags_in(joined_key_texts(page, list(key)), joined_key_texts(late, list(key)))


def held_scopes_job(
    key: tuple[str, ...],
    instants: frozenset[str],
    scope: tuple[str, str],
    page: bytes,
    held: bytes,
    scopes: tuple[str, ...] | None,
) -> tuple[bytes, bytes]:
    """:func:`held_scopes_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param scope: the scope column and its prefix
    :ptype scope: tuple[str, str]
    :param page: the page's keys, IPC
    :ptype page: bytes
    :param held: the copy's rows of them, IPC
    :ptype held: bytes
    :param scopes: the scopes a write in progress holds; None for none
    :ptype scopes: tuple[str, ...] | None
    :return: the held rows' keys and scopes, IPC; the flags, or empty when every key is staged
    :rtype: tuple[bytes, bytes]
    """
    _one_cpu()
    among = None if scopes is None else pa.array(scopes, pa.string())
    mine, keep = held_scopes_of(key, instants, scope, from_ipc(page), from_ipc(held), among)
    return ipc_bytes(mine), b"" if keep is None else _flags(keep)


def held_scopes_answer(answer: tuple[bytes, bytes]) -> tuple[Any, Any]:
    """:func:`held_scopes_job`'s answer read back.

    :param answer: the answer
    :ptype answer: tuple[bytes, bytes]
    :return: as :func:`held_scopes_of`
    :rtype: tuple[pyarrow.Table, numpy.ndarray | None]
    """
    return from_ipc(answer[0]), _unflags(answer[1])


def page_scopes_of(
    key: Sequence[str],
    instants: Collection[str],
    scope: tuple[str, str],
    chunk: Any,
    found: Any,
    mine: Any,
    scopes: Any,
) -> tuple[Any, Any, Any]:
    """a page's keys' scopes, the row the copy holds and the row read now (a delete's, a move's) each, and which keys
    and rows stay staged: a key whose new row is of a scope a write holds waits.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param scope: the scope column and its prefix
    :ptype scope: tuple[str, str]
    :param chunk: the page's keys staged now, key columns as text
    :ptype chunk: pyarrow.Table
    :param found: the rows read now: key and scope columns, as read
    :ptype found: pyarrow.Table
    :param mine: the rows the copy holds of the page (:func:`held_scopes_of`)
    :ptype mine: pyarrow.Table
    :param scopes: the scopes a write in progress holds; None for none
    :ptype scopes: pyarrow.Array | None
    :return: the keys with each row's scope (``__scope``), and per key of the chunk and per row read whether it stays
        (None: every one does)
    :rtype: tuple[pyarrow.Table, numpy.ndarray | None, numpy.ndarray | None]
    """
    columns = list(key)
    found_texts = key_text_table(found, columns, instants)
    found_scoped = found_texts.append_column(_SCOPE, scope_texts(scope[1], found.column(scope[0])))
    both = distinct_keys(pa.concat_tables([mine, found_scoped]), [*columns, _SCOPE])
    chunk_texts = joined_key_texts(chunk, columns)
    both = both.filter(pa.array(flags_in(joined_key_texts(both, columns), chunk_texts)))
    if scopes is None:
        return both, None, None
    left = both.filter(pa.array(flags_in(both.column(_SCOPE), scopes)))
    if not left.num_rows:
        return both, None, None
    left_texts = joined_key_texts(left, columns)
    stays = ~flags_in(chunk_texts, left_texts)
    rows_stay = ~flags_in(joined_key_texts(found_texts, columns), left_texts)
    return both.filter(pa.array(~flags_in(joined_key_texts(both, columns), left_texts))), stays, rows_stay


def page_scopes_job(
    key: tuple[str, ...],
    instants: frozenset[str],
    scope: tuple[str, str],
    chunk: bytes,
    found: bytes,
    mine: bytes,
    scopes: tuple[str, ...] | None,
) -> tuple[bytes, bytes, bytes]:
    """:func:`page_scopes_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param scope: the scope column and its prefix
    :ptype scope: tuple[str, str]
    :param chunk: the page's keys staged now, IPC
    :ptype chunk: bytes
    :param found: the rows read now (key and scope columns), IPC
    :ptype found: bytes
    :param mine: the rows the copy holds of the page, IPC
    :ptype mine: bytes
    :param scopes: the scopes a write in progress holds; None for none
    :ptype scopes: tuple[str, ...] | None
    :return: the keys with their scopes, IPC; the chunk's and the rows' flags, each empty when every one stays
    :rtype: tuple[bytes, bytes, bytes]
    """
    _one_cpu()
    among = None if scopes is None else pa.array(scopes, pa.string())
    both, stays, rows_stay = page_scopes_of(
        key, instants, scope, from_ipc(chunk), from_ipc(found), from_ipc(mine), among
    )
    return ipc_bytes(both), b"" if stays is None else _flags(stays), b"" if rows_stay is None else _flags(rows_stay)


def page_scopes_answer(answer: tuple[bytes, bytes, bytes]) -> tuple[Any, Any, Any]:
    """:func:`page_scopes_job`'s answer read back.

    :param answer: the answer
    :ptype answer: tuple[bytes, bytes, bytes]
    :return: as :func:`page_scopes_of`
    :rtype: tuple[pyarrow.Table, numpy.ndarray | None, numpy.ndarray | None]
    """
    return from_ipc(answer[0]), _unflags(answer[1]), _unflags(answer[2])
