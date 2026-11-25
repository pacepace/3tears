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
from threetears.core.collections.generation import split_generation_token
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
    "HEARD_INLINE_TEXT_KEYS",
    "INLINE_KEYS",
    "cleared_reach_job",
    "cleared_reach_of",
    "flags_in",
    "from_ipc",
    "heard_job",
    "heard_of",
    "heard_schema",
    "held_scopes_answer",
    "held_scopes_job",
    "held_scopes_of",
    "ipc_bytes",
    "joined",
    "joined_ipc",
    "later_than",
    "page_scopes_answer",
    "page_scopes_job",
    "page_scopes_of",
    "partition_mine_job",
    "partition_mine_of",
    "partition_page_job",
    "partition_page_of",
    "reached_of",
    "scope_texts",
    "taken_answer",
    "taken_job",
    "taken_of",
    "texts_of",
    "touched_job",
    "touched_of",
    "whole_page_answer",
    "whole_page_job",
    "whole_page_of",
]

#: a heard batch's epoch column
_EPOCH: Final = "__epoch"

#: a key's scope column, beside its key columns
_SCOPE: Final = "__scope"

#: a pair's key's position among a pass's keys
_AT: Final = "__at"

#: below this many keys the work runs on the caller's thread: the worker's round trip (the IPC written, pickled through
#: the pool's pipe and read back) costs more than the work. Measured 2026-11-25 (macOS arm64, Python 3.14, a loaded
#: host): a 200,000-key page's heard batch took 3-8 ms on a thread and 25-144 ms through the worker (12-30 ms of it the
#: serving process's CPU); its page scopes 196-290 ms on a thread (all of it the serving process's CPU) and 342-656 ms
#: through the worker (43-74 ms of it the serving process's CPU);
#: the jobs added since keep the bound, each measured against it the same day: at 2,000 keys the work here took
#: 0.43 ms (a no-scope table's touched keys made distinct), 0.33 ms (a cleared prefix's reach), 0.28 ms (a whole read's
#: page with a scope held), 0.05 ms (a partition's page) of the loop's CPU, while the worker's round trip cost the
#: serving process 0.6-0.8 ms of CPU however small the job; past it the work grows with the keys (15 ms, 2.3 ms, 8.5 ms
#: at 100,000) and the round trip barely does (1.7 ms, 1.7 ms, 0.9 ms)
INLINE_KEYS: Final = 2_000

#: a heard batch of text keys (no timestamp key column) is spelled inline on the loop up to this many keys, in the worker
#: past it. Why: its work is a cast that keeps text as it is and one column of the epoch, about 6 ns a key of the loop's
#: CPU (2026-11-25, macOS arm64, Python 3.14, 4 text key columns: 0.33 ms at 50,000 keys, 0.79 ms at 100,000, 1.24 ms
#: at 200,000), while the worker's round trip cost the serving process 12-30 ms of CPU at 200,000 (the IPC written and
#: the answer read back). Past 100,000 keys (under 1 ms) the turn would grow toward the tests' 10 ms bar on slower pool
#: hosts, so a larger batch still goes to the worker. A timestamp key column is spelled a distinct value at a time in
#: Python (key_text), so such a batch goes to the worker past :data:`INLINE_KEYS`
HEARD_INLINE_TEXT_KEYS: Final = 100_000


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
    return pa.Table.from_arrays([*columns, epochs], schema=heard_schema(key))


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


def heard_schema(key: Sequence[str]) -> Any:
    """the schema of a batch of keys heard: each key column as text, named as the key, and the epoch.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :return: the schema
    :rtype: pyarrow.Schema
    """
    return pa.schema([(column, pa.string()) for column in key] + [(_EPOCH, pa.string())])


def joined(batches: Sequence[Any], schema: Any) -> Any:
    """some batches as one table (their chunks, not copied); the schema's empty table when there are none.

    :param batches: the batches, each of ``schema``'s columns at least
    :ptype batches: Sequence[pyarrow.Table]
    :param schema: what each holds
    :ptype schema: pyarrow.Schema
    :return: the table
    :rtype: pyarrow.Table
    """
    if not batches:
        return schema.empty_table()
    return pa.concat_tables([batch.select(schema.names).cast(schema) for batch in batches])


def joined_ipc(batches: Sequence[Any], schema: Any) -> bytes:
    """some batches as one Arrow IPC stream, for the copy's worker process: their chunks joined (nothing copied) and
    written whole, one record batch a chunk. A copy of their buffers into the stream: the hand-off, so the caller runs
    it on a thread (IO, as :func:`ipc_bytes`).

    :param batches: the batches
    :ptype batches: Sequence[pyarrow.Table]
    :param schema: what each holds
    :ptype schema: pyarrow.Schema
    :return: the bytes
    :rtype: bytes
    """
    return ipc_bytes(joined(batches, schema))


def later_than(rows: Any, key: Sequence[str], bar: str | None) -> Any:
    """per row of keys heard at their epochs, whether its key was heard at an epoch past ``bar`` (the table's generation
    as read before the pass's stamp): every row of such a key, so a key waits whole. One judgment a distinct epoch (a
    commit), never one a key; an epoch of another incarnation, or none, is never past it (nothing orders them).

    :param rows: the keys at their epochs (key columns as text, ``__epoch``)
    :ptype rows: pyarrow.Table
    :param key: the key's columns
    :ptype key: Sequence[str]
    :param bar: the generation read before the stamp; None for none
    :ptype bar: str | None
    :return: the flags, or None when no row is past it
    :rtype: numpy.ndarray | None
    """
    split = split_generation_token(bar) if bar is not None else None
    if split is None or not rows.num_rows:
        return None
    epochs = rows.column(_EPOCH)
    later = []
    for epoch in pc.unique(pc.drop_null(epochs)).to_pylist():  # one a distinct epoch (a commit), never one a key
        parsed = split_generation_token(epoch)
        if parsed is not None and parsed[0] == split[0] and parsed[1] > split[1]:
            later.append(epoch)
    if not later:
        return None
    newer = rows.filter(pa.array(flags_in(epochs, pa.array(later, pa.string()))))
    return flags_in(joined_key_texts(rows, list(key)), joined_key_texts(newer, list(key)))


def taken_of(
    key: Sequence[str],
    scope_at: tuple[str, int] | None,
    delta_files: bool,
    heard: Any,
    staged: Any,
    held: Any,
    bar: str | None,
) -> tuple[Any, Any, Any, int, bool]:
    """a pass's keys of one table from its batches heard (joined): each key at each epoch once, those of a scope a
    write holds left pending (when the key names its scope), and for a table that opts in to delta files, every epoch
    a key was staged from by an earlier pass beside the new ones, so its row is merged again from every change not yet
    applied.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param scope_at: the scope's prefix and the key column naming it, when the key names it; None otherwise
    :ptype scope_at: tuple[str, int] | None
    :param delta_files: whether the table opts in to delta files
    :ptype delta_files: bool
    :param heard: the batches heard, joined (key columns as text, ``__epoch``)
    :ptype heard: pyarrow.Table
    :param staged: the keys at their epochs earlier passes staged and no apply has taken yet, joined; None for none
    :ptype staged: pyarrow.Table | None
    :param held: the scopes a write in progress holds; None for none
    :ptype held: pyarrow.Array | None
    :param bar: the table's generation read before the pass's stamp; None for none
    :ptype bar: str | None
    :return: the keys (each once, in key order), the pairs (each key at each epoch, ``__at`` its key's position, in
        that order), the pairs left pending (None for none), how many keys they hold, and whether some were heard past
        ``bar``
    :rtype: tuple[pyarrow.Table, pyarrow.Table, pyarrow.Table | None, int, bool]
    """
    key = list(key)
    rows = distinct_keys(heard, [*key, _EPOCH])
    left = None
    left_keys = 0
    if held is not None and scope_at is not None:
        hold = pa.array(flags_in(scope_texts(scope_at[0], rows.column(key[scope_at[1]])), held))
        left = rows.filter(hold)
        rows = rows.filter(pc.invert(hold))
        left_keys = distinct_keys(left, key).num_rows
        left = left if left.num_rows else None
    later = later_than(rows, key, bar)
    held_later = later is not None and bool(later.any())
    if held_later:
        # a key heard at an epoch past the generation read before the pass's stamp: its commit may be newer than the
        # stamp, so every epoch of it waits for the next pass, whose stamp holds it (one read, never read and redone)
        late = rows.filter(pa.array(later))
        rows = rows.filter(pa.array(~later))
        left_keys += distinct_keys(late, key).num_rows
        left = late if left is None else pa.concat_tables([left, late])
    if staged is not None and staged.num_rows and rows.num_rows:
        # a key heard again at an epoch an earlier pass staged it at (its row broadcast and its epoch's record, one
        # before the pass and one after) is staged already: read again, it was read twice for one change (the live
        # refresh test, 2 runs in 6, 2026-11-24). Each record's keys are read once; a key heard with no epoch has none
        # to match and is read as before
        again = pc.and_(
            pc.is_valid(rows.column(_EPOCH)),
            pa.array(flags_in(joined_key_texts(rows, [*key, _EPOCH]), joined_key_texts(staged, [*key, _EPOCH]))),
        )
        rows = rows.filter(pc.invert(pc.fill_null(again, False)))
        if delta_files and rows.num_rows:
            bring = staged.filter(pa.array(flags_in(joined_key_texts(staged, key), joined_key_texts(rows, key))))
            if bring.num_rows:
                rows = distinct_keys(pa.concat_tables([rows, bring]), [*key, _EPOCH])
    keys = distinct_keys(rows, key).sort_by([(column, "ascending") for column in key])
    positions = pc.index_in(joined_key_texts(rows, key), value_set=joined_key_texts(keys, key)).cast(pa.int64())
    pairs = rows.append_column(_AT, positions).sort_by([(_AT, "ascending")])
    return keys, pairs, left, left_keys, held_later


def taken_job(
    key: tuple[str, ...],
    scope_at: tuple[str, int] | None,
    delta_files: bool,
    heard: bytes,
    staged: bytes,
    held: tuple[str, ...] | None,
    bar: str | None,
) -> tuple[bytes, bytes, bytes, int, bool]:
    """:func:`taken_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param scope_at: the scope's prefix and the key column naming it; None when the key names none
    :ptype scope_at: tuple[str, int] | None
    :param delta_files: whether the table opts in to delta files
    :ptype delta_files: bool
    :param heard: the batches heard, joined, IPC
    :ptype heard: bytes
    :param staged: the keys earlier passes staged, joined, IPC; empty for none
    :ptype staged: bytes
    :param held: the scopes a write in progress holds; None for none
    :ptype held: tuple[str, ...] | None
    :param bar: the table's generation read before the pass's stamp; None for none
    :ptype bar: str | None
    :return: the keys and the pairs, IPC; the pairs left pending, IPC (empty for none); their keys' count; whether
        some were heard past ``bar``
    :rtype: tuple[bytes, bytes, bytes, int, bool]
    """
    _one_cpu()
    among = None if held is None else pa.array(held, pa.string())
    keys, pairs, left, left_keys, later = taken_of(
        key, scope_at, delta_files, from_ipc(heard), from_ipc(staged) if staged else None, among, bar
    )
    return ipc_bytes(keys), ipc_bytes(pairs), b"" if left is None else ipc_bytes(left), left_keys, later


def taken_answer(answer: tuple[bytes, bytes, bytes, int, bool]) -> tuple[Any, Any, Any, int, bool]:
    """:func:`taken_job`'s answer read back.

    :param answer: the answer
    :ptype answer: tuple[bytes, bytes, bytes, int, bool]
    :return: as :func:`taken_of`
    :rtype: tuple[pyarrow.Table, pyarrow.Table, pyarrow.Table | None, int, bool]
    """
    keys, pairs, left, left_keys, later = answer
    return from_ipc(keys), from_ipc(pairs), from_ipc(left) if left else None, left_keys, later


def under_of(key: Sequence[str], rows: Any, prefixes: Sequence[Sequence[tuple[str, str]]]) -> Any:
    """the keys of ``rows`` (keys at epochs, as text) under one of ``prefixes``, each once (compared as text).

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param rows: the keys heard or staged, joined
    :ptype rows: pyarrow.Table
    :param prefixes: the prefixes, each its columns' values as ``(column, value)`` pairs
    :ptype prefixes: Sequence[Sequence[tuple[str, str]]]
    :return: the keys, key columns as text
    :rtype: pyarrow.Table
    """
    under = np.zeros(rows.num_rows, dtype=bool)
    for prefix in prefixes:
        within = np.ones(rows.num_rows, dtype=bool)
        for column, value in prefix:
            equal = pc.fill_null(pc.equal(rows.column(column), pa.scalar(value, pa.string())), False)
            within &= np.asarray(equal.to_numpy(zero_copy_only=False), dtype=bool)
        under |= within
    return distinct_keys(rows.filter(pa.array(under)), list(key))


def cleared_reach_of(
    key: Sequence[str],
    instants: Collection[str],
    held: Sequence[Any],
    known: Any,
    prefixes: Sequence[Sequence[tuple[str, str]]],
    epoch: str,
) -> Any:
    """the keys cleared prefixes reach, each once, as the batch heard at the clearing epoch: the keys of the copy's rows
    under them, and those heard or staged under them (:func:`under_of`, :func:`reached_of`).

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param held: the copy's rows under the prefixes (its key columns, as the copy holds them), in batches
    :ptype held: Sequence[pyarrow.Table]
    :param known: the keys heard or staged, joined (key columns as text, ``__epoch``)
    :ptype known: pyarrow.Table
    :param prefixes: the prefixes, each its columns' values as ``(column, value)`` pairs
    :ptype prefixes: Sequence[Sequence[tuple[str, str]]]
    :param epoch: the clearing epoch
    :ptype epoch: str
    :return: the batch
    :rtype: pyarrow.Table
    """
    return reached_of(key, instants, held, under_of(key, known, prefixes), epoch)


def cleared_reach_job(
    key: tuple[str, ...],
    instants: frozenset[str],
    held: tuple[bytes, ...],
    known: bytes,
    prefixes: tuple[tuple[tuple[str, str], ...], ...],
    epoch: str,
) -> bytes:
    """:func:`cleared_reach_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param held: the copy's rows under the prefixes, IPC
    :ptype held: tuple[bytes, ...]
    :param known: the keys heard or staged, joined, IPC
    :ptype known: bytes
    :param prefixes: the prefixes
    :ptype prefixes: tuple[tuple[tuple[str, str], ...], ...]
    :param epoch: the clearing epoch
    :ptype epoch: str
    :return: the batch, IPC
    :rtype: bytes
    """
    _one_cpu()
    return ipc_bytes(
        cleared_reach_of(key, instants, [from_ipc(rows) for rows in held], from_ipc(known), prefixes, epoch)
    )


def touched_of(key: Sequence[str], keys: Any) -> Any:
    """the keys an apply replaced in a table of no scope, each once: what a checkpoint patches its one part by.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param keys: the keys staged, joined (key columns as text)
    :ptype keys: pyarrow.Table
    :return: the keys
    :rtype: pyarrow.Table
    """
    return distinct_keys(keys, list(key))


def touched_job(key: tuple[str, ...], keys: bytes) -> bytes:
    """:func:`touched_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param keys: the keys staged, joined, IPC
    :ptype keys: bytes
    :return: the keys, IPC
    :rtype: bytes
    """
    _one_cpu()
    return ipc_bytes(touched_of(key, from_ipc(keys)))


def _scopes_of(prefix: str | None, column: Any, count: int) -> Any:
    """each row's scope: ``prefix`` and its scope column's text; nulls for a table of no scope.

    :param prefix: the scope's prefix; None for a table of no scope
    :ptype prefix: str | None
    :param column: the scope column's values; None for a table of no scope
    :ptype column: pyarrow.ChunkedArray | None
    :param count: the rows
    :ptype count: int
    :return: the scopes
    :rtype: pyarrow.Array
    """
    if prefix is None or column is None:
        return pa.nulls(count, pa.string())
    return scope_texts(prefix, column)


def _distinct_scopes(scopes: Any) -> tuple[str | None, ...]:
    """the distinct scopes of a column of them, sorted (a null first): one a state, never one a row.

    :param scopes: the scopes
    :ptype scopes: pyarrow.Array
    :return: the scopes
    :rtype: tuple[str | None, ...]
    """
    found = pc.unique(scopes).to_pylist()  # one a distinct scope (a state), never one a row
    return tuple(sorted(found, key=lambda scope: (scope is not None, scope or "")))


def whole_page_of(
    scope: tuple[str, str] | None, page: Any, held: Any
) -> tuple[Any, tuple[str | None, ...], tuple[tuple[str, int], ...]]:
    """a page of a whole read judged by its scope column: which rows stay (a row of a scope a write holds is left out;
    None: every row stays), the distinct scopes of the rows that stay, and how many rows of each held scope were left
    out. The page itself is filtered by the caller: a copy of the rows kept, memory, not a pass of judgment.

    :param scope: the scope column and its prefix; None for a table of no scope
    :ptype scope: tuple[str, str] | None
    :param page: the page (its scope column at least)
    :ptype page: pyarrow.Table
    :param held: the scopes a write in progress holds; None for none
    :ptype held: pyarrow.Array | None
    :return: the flags or None, the scopes, the held scopes' counts
    :rtype: tuple[numpy.ndarray | None, tuple[str | None, ...], tuple[tuple[str, int], ...]]
    """
    if scope is None:
        return None, (None,) if page.num_rows else (), ()
    of = scope_texts(scope[1], page.column(scope[0]))
    if held is None:
        return None, _distinct_scopes(of) if page.num_rows else (), ()
    hit = flags_in(of, held)
    if not hit.any():
        return None, _distinct_scopes(of) if page.num_rows else (), ()
    # one count a held scope (a state), never one a row
    counts = tuple(
        (found["values"], found["counts"]) for found in pc.value_counts(of.filter(pa.array(hit))).to_pylist()
    )
    stays = ~hit
    kept = of.filter(pa.array(stays))
    return stays, _distinct_scopes(kept) if len(kept) else (), counts


def whole_page_job(
    scope: tuple[str, str], page: bytes, held: tuple[str, ...] | None
) -> tuple[bytes, tuple[str | None, ...], tuple[tuple[str, int], ...]]:
    """:func:`whole_page_of`: the worker's job.

    :param scope: the scope column and its prefix
    :ptype scope: tuple[str, str]
    :param page: the page's scope column, IPC
    :ptype page: bytes
    :param held: the scopes a write in progress holds; None for none
    :ptype held: tuple[str, ...] | None
    :return: the flags (empty: every row stays), the scopes, the held scopes' counts
    :rtype: tuple[bytes, tuple[str | None, ...], tuple[tuple[str, int], ...]]
    """
    _one_cpu()
    among = None if held is None else pa.array(held, pa.string())
    stays, scopes, counts = whole_page_of(scope, from_ipc(page), among)
    return b"" if stays is None else _flags(stays), scopes, counts


def whole_page_answer(
    answer: tuple[bytes, tuple[str | None, ...], tuple[tuple[str, int], ...]],
) -> tuple[Any, tuple[str | None, ...], tuple[tuple[str, int], ...]]:
    """:func:`whole_page_job`'s answer read back.

    :param answer: the answer
    :ptype answer: tuple[bytes, tuple[str | None, ...], tuple[tuple[str, int], ...]]
    :return: as :func:`whole_page_of`
    :rtype: tuple[numpy.ndarray | None, tuple[str | None, ...], tuple[tuple[str, int], ...]]
    """
    return _unflags(answer[0]), answer[1], answer[2]


def partition_page_of(
    key: Sequence[str], instants: Collection[str], scope: tuple[str, str] | None, page: Any, held: Any
) -> tuple[bool, tuple[str | None, ...], Any]:
    """a page of a partition's L3 rows judged: whether a row is of a scope a write holds (the partition waits), the
    distinct scopes of its rows, and for a table of no scope the keys it replaces, as text.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param scope: the scope column and its prefix; None for a table of no scope
    :ptype scope: tuple[str, str] | None
    :param page: the page's key and scope columns
    :ptype page: pyarrow.Table
    :param held: the scopes a write in progress holds; None for none
    :ptype held: pyarrow.Array | None
    :return: whether it clashed, the scopes, the keys replaced (None for a scoped table or a clash)
    :rtype: tuple[bool, tuple[str | None, ...], pyarrow.Table | None]
    """
    of = _scopes_of(scope[1] if scope else None, page.column(scope[0]) if scope else None, page.num_rows)
    if held is not None and flags_in(of, held).any():
        return True, (), None
    replaced = key_text_table(page, key, instants) if scope is None else None
    return False, _distinct_scopes(of) if page.num_rows else (), replaced


def partition_page_job(
    key: tuple[str, ...],
    instants: frozenset[str],
    scope: tuple[str, str] | None,
    page: bytes,
    held: tuple[str, ...] | None,
) -> tuple[bool, tuple[str | None, ...], bytes]:
    """:func:`partition_page_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param scope: the scope column and its prefix; None for a table of no scope
    :ptype scope: tuple[str, str] | None
    :param page: the page's key and scope columns, IPC
    :ptype page: bytes
    :param held: the scopes a write in progress holds; None for none
    :ptype held: tuple[str, ...] | None
    :return: whether it clashed, the scopes, the keys replaced, IPC (empty for none)
    :rtype: tuple[bool, tuple[str | None, ...], bytes]
    """
    _one_cpu()
    among = None if held is None else pa.array(held, pa.string())
    clashed, scopes, replaced = partition_page_of(key, instants, scope, from_ipc(page), among)
    return clashed, scopes, b"" if replaced is None else ipc_bytes(replaced)


def partition_mine_of(
    key: Sequence[str],
    instants: Collection[str],
    scope: tuple[str, str] | None,
    last: str,
    mine: Any,
    held: Any,
) -> tuple[tuple[str, ...], Any, tuple[str | None, ...], Any]:
    """the copy's own rows of some partitions (one prefix shape) judged: the values of the prefix's last column a row
    of a scope a write holds sits under (those partitions wait for the write), the rows kept (their key columns, as
    the copy holds them), their distinct scopes, and for a table of no scope their keys as text.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param scope: the scope column and its prefix; None for a table of no scope
    :ptype scope: tuple[str, str] | None
    :param last: the prefix's last column
    :ptype last: str
    :param mine: the copy's rows under the prefixes: key and scope columns, as the copy holds them
    :ptype mine: pyarrow.Table
    :param held: the scopes a write in progress holds; None for none
    :ptype held: pyarrow.Array | None
    :return: the late values (sorted), the rows kept (key columns), the scopes, the keys replaced (None for a scoped
        table)
    :rtype: tuple[tuple[str, ...], pyarrow.Table, tuple[str | None, ...], pyarrow.Table | None]
    """
    of = _scopes_of(scope[1] if scope else None, mine.column(scope[0]) if scope else None, mine.num_rows)
    late: tuple[str, ...] = ()
    if held is not None and mine.num_rows:
        hit = flags_in(of, held)
        if hit.any():
            values = key_text_table(mine, [last], instants).column(last)
            # the prefix values a held scope's rows sit under: one a prefix of the batch, never one a row
            late = tuple(sorted(pc.unique(values.filter(pa.array(hit))).to_pylist()))
            stays = pa.array(~flags_in(values, pa.array(late, pa.string())))
            mine = mine.filter(stays)
            of = of.filter(stays)
    replaced = key_text_table(mine, key, instants) if scope is None and mine.num_rows else None
    return late, mine.select(list(key)), _distinct_scopes(of) if mine.num_rows else (), replaced


def partition_mine_job(
    key: tuple[str, ...],
    instants: frozenset[str],
    scope: tuple[str, str] | None,
    last: str,
    mine: bytes,
    held: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], bytes, tuple[str | None, ...], bytes]:
    """:func:`partition_mine_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param scope: the scope column and its prefix; None for a table of no scope
    :ptype scope: tuple[str, str] | None
    :param last: the prefix's last column
    :ptype last: str
    :param mine: the copy's rows under the prefixes, IPC
    :ptype mine: bytes
    :param held: the scopes a write in progress holds; None for none
    :ptype held: tuple[str, ...] | None
    :return: the late values, the rows kept (IPC), the scopes, the keys replaced (IPC, empty for none)
    :rtype: tuple[tuple[str, ...], bytes, tuple[str | None, ...], bytes]
    """
    _one_cpu()
    among = None if held is None else pa.array(held, pa.string())
    late, kept, scopes, replaced = partition_mine_of(key, instants, scope, last, from_ipc(mine), among)
    return late, ipc_bytes(kept), scopes, b"" if replaced is None else ipc_bytes(replaced)
