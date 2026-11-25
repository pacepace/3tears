"""``SqlL3Backend`` — the default L3 backend over an asyncpg-shaped pool.

Implements **both** L3-tier protocols:

- :class:`~threetears.core.backends.protocol.L3Backend` — the raw-SQL transport, by
  delegating to the wrapped pool (``fetch`` / ``fetchrow`` / ``execute`` / ``acquire`` /
  ``transaction``; ``execute_batch`` runs the batch in one pool transaction).
- :class:`~threetears.core.backends.protocol.DurableStore` — the structured ops, by
  *generating* parameterized SQL (``fetch_one`` → SELECT, ``upsert`` →
  INSERT…ON CONFLICT, ``delete`` → DELETE, ``scan`` → SELECT…WHERE). This is the
  reference SQL implementation of the same structured contract a ``GitL3Backend``
  satisfies with file read/write/delete + commit.

**Schema awareness.** A :class:`~threetears.core.collections.schema_backed.SchemaBackedCollection`
registers its :class:`TableSchema` via :meth:`SqlL3Backend.register_schema`. Once a schema
is registered for a table, the structured ops generate **schema-aware** SQL (declared-
columns-only SELECT with ``VECTOR``/``TSVECTOR`` ``::text`` projection casts, ``::jsonb`` /
``::vector`` write casts, composite-PK WHERE, server-default column dropping, CAS fencing,
``on_conflict`` modes) + read/write codec — byte-identical to the hand-rolled collection
path it replaces. When **no** schema is registered for a table, the generic ``SELECT *`` /
INSERT path is used (the contract a non-collection caller of :class:`DurableStore` relies on).

The wrapped pool is the previously-untyped ``l3_pool`` (a bare asyncpg ``Pool`` or the
``NatsProxyL3Backend``); ``SqlL3Backend`` is the named, protocol-conformant wrapper the
``L3Backend``-retyped registry/base resolve to.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, TYPE_CHECKING
from urllib.parse import urlparse, urlsplit
from urllib.request import url2pathname

from threetears.interfaces.keyed.rows import DeclaredKey, KeyValues
from threetears.observe import get_logger

import threetears.core.backends.schema_sql as schema_sql
from threetears.core.backends.arrow_copy import arrow_copy_available, planned_write_table, write_type
from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.backends.dialect import known_on_yugabyte
from threetears.core.backends.protocol import (
    L3_RAIL_ROW_CAP,
    ArrowWritePlan,
    L3AnswerCutError,
    L3Page,
    FileLoad,
    LoadFile,
    LoadingStore,
    LoadResult,
    TruncatingStore,
    parse_rowcount,
)
from threetears.core.declared import whole_read
from threetears.core.loop_turns import STEP_ITEMS, LoopTurns, emptied, stepped
from threetears.core.ddl.change_notes import (
    NOTE_REACH_SETTING,
    NOTE_WHOLE_SETTING,
    WHOLE_COPY_ROWS,
    note_reach_text,
    prefix_write_mark_sql,
    prefix_write_restore_sql,
)
from threetears.core.ddl.ddl_lock import database_ddl_lock
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, UnstampedWriteError
from threetears.core.ddl.session import ConnectionSession
from threetears.core.keyed._filters import binds_native_arrays
from threetears.core.keyed.point import read_row
from threetears.core.keyed.writes import delete_keyed
from threetears.core.keyset import read_keyset_pages
from threetears.core.sql_fragments import as_written

try:
    import httpx

    _HTTPX = True
except ImportError:  # NOSILENT: the http extra is optional; a load naming an https file says so without it
    _HTTPX = False

try:
    import pyarrow.parquet as pq

    _PARQUET = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a load of parquet files says so without it
    _PARQUET = False

if TYPE_CHECKING:
    from threetears.core.collections.schema_backed import TableSchema

__all__ = ["SqlL3Backend", "bound_request_connection"]

_logger = get_logger(__name__)

_ON_CONFLICT_VALUES = frozenset({"update", "ignore", "raise"})

#: the asyncpg module whose release path raises the masking AttributeError.
_ASYNCPG_POOL_FILE = os.path.join("asyncpg", "pool.py")


def _is_asyncpg_release_race(exc: AttributeError) -> bool:
    """Whether ``exc`` is asyncpg's pool-release race masking the query's own error.

    asyncpg 0.31.0's ``PoolConnectionHolder.release`` waits for an in-flight cancellation
    after a query fails. If the server closes the connection during that wait, the holder's
    connection is cleared, and release then calls ``reset`` on ``None``. The resulting
    ``AttributeError`` escapes ``pool.acquire()``'s exit and replaces the error the query
    actually raised, which survives only as ``__context__``. Recognised by all three: raised
    inside asyncpg's ``pool.py``, on a ``None`` object, with an exception already in flight.

    :param exc: the AttributeError a pool call raised
    :ptype exc: AttributeError
    :return: whether it is the release race rather than a real attribute error
    :rtype: bool
    """
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    raised_in_pool = tb is not None and tb.tb_frame.f_code.co_filename.endswith(_ASYNCPG_POOL_FILE)
    return raised_in_pool and exc.obj is None and exc.__context__ is not None


@contextmanager
def _unmask_asyncpg_release_race() -> Iterator[None]:
    """Re-raise the query's own error when asyncpg's release race has masked it.

    Any other ``AttributeError`` propagates unchanged.

    :return: context manager wrapping one pool call
    :rtype: Iterator[None]
    :raises BaseException: the exception the query raised, in place of the masking one
    """
    try:
        yield
    except AttributeError as exc:
        original = exc.__context__
        if not _is_asyncpg_release_race(exc) or original is None:
            raise
        _logger.warning(
            "asyncpg pool release raced a server-side close and masked the query's own "
            "error; raising the original %s instead of the release AttributeError",
            type(original).__name__,
        )
        raise original from None


# ── request-scoped L3 connection (per-request transaction; RLS session-GUC support) ─────────
# When a connection is bound here, every ``SqlL3Backend`` op in the async context runs ITS
# statements on that one connection -- so a hub request can open ONE transaction, set a
# txn-local session GUC (e.g. RLS's ``app.customer_ids``) on it, and have every collection
# read/write in the request share that transaction + GUC, instead of each statement taking a
# fresh autocommit connection from the pool. UNBOUND (the default) the behaviour is unchanged:
# ops go straight to the pool exactly as before. The binder owns the connection lifecycle +
# the transaction; ``SqlL3Backend`` only READS this var.
_request_l3_conn: ContextVar[Any | None] = ContextVar("threetears_request_l3_conn", default=None)


@asynccontextmanager
async def bound_request_connection(conn: Any) -> AsyncIterator[Any]:
    """Bind ``conn`` as the request-scoped L3 connection for the enclosing async context.

    Within the context, every :class:`SqlL3Backend` ``fetch`` / ``execute`` / ``transaction`` /
    ``acquire`` runs on ``conn`` (one transaction). The CALLER owns ``conn`` -- acquire it, open
    its transaction, set any session/txn GUCs, then enter this context; this helper only sets +
    resets the context variable. Nested binds restore the previous binding on exit.

    :param conn: an asyncpg-shaped connection already inside a transaction.
    :ptype conn: Any
    :return: async iterator yielding ``conn``.
    :rtype: AsyncIterator[Any]
    """
    token = _request_l3_conn.set(conn)
    try:
        yield conn
    finally:
        _request_l3_conn.reset(token)


class _BoundConnCM:
    """Async-CM yielding an already-bound request connection WITHOUT acquiring/releasing it.

    Returned by :meth:`SqlL3Backend.acquire` / :meth:`SqlL3Backend.transaction` when a request
    connection is bound, so explicit-connection / transactional code reuses the one request
    transaction instead of escaping to a fresh pooled connection (which would miss the request's
    session GUC and, under RLS ``FORCE``, see zero rows). Enter/exit are no-ops: the binder owns
    the connection's lifecycle and transaction.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *exc: object) -> None:
        return None


def _unstamped(table: str) -> UnstampedWriteError:
    """the refusal of a write to a table that declares a row epoch, of a row that carries none.

    :param table: the table
    :ptype table: str
    :return: the error
    :rtype: UnstampedWriteError
    """
    return UnstampedWriteError(
        f"{table} declares a row epoch: every row written carries the epoch of its write ({EPOCH_COLUMN!r}), and a row "
        f"of this write has none. Write it through its collection on a transaction that took its write's turn "
        f"(ScopeEpochs.touch), which stamps it; a load from files names its write's version (write_version)"
    )


def _refuse_unstamped_rows(schema: Any, table: str, rows: Iterable[Mapping[str, Any]]) -> None:
    """refuse rows of a table that declares a row epoch when any carries none: before anything is sent, so a write
    that would leave the newest version of a key unknowable never reaches L3.

    :param schema: the table's registered schema
    :ptype schema: TableSchema
    :param table: the table
    :ptype table: str
    :param rows: the rows
    :ptype rows: Iterable[Mapping[str, Any]]
    :return: nothing
    :rtype: None
    :raises UnstampedWriteError: when a row carries no epoch
    """
    if getattr(schema, "row_epoch", None) is not None and any(row.get(EPOCH_COLUMN) is None for row in rows):
        raise _unstamped(table)


def _refuse_unstamped_arrow(schema: Any, table: str, rows: Any) -> None:
    """:func:`_refuse_unstamped_rows` for rows in an Arrow table: its epoch column, read by Arrow's null count, never a
    Python value a row.

    :param schema: the table's registered schema
    :ptype schema: TableSchema
    :param table: the table
    :ptype table: str
    :param rows: the rows
    :ptype rows: pyarrow.Table
    :return: nothing
    :rtype: None
    :raises UnstampedWriteError: when the rows have no epoch column, or a row's is null
    """
    if getattr(schema, "row_epoch", None) is None:
        return
    if EPOCH_COLUMN not in rows.column_names or rows.column(EPOCH_COLUMN).null_count:
        raise _unstamped(table)


def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier, rejecting embedded quotes (no injection via column names).

    :param name: a table or column identifier.
    :ptype name: str
    :return: the double-quoted identifier.
    :rtype: str
    :raises ValueError: if ``name`` contains a double-quote.
    """
    if '"' in name:
        raise ValueError(f"illegal SQL identifier {name!r}")
    return f'"{name}"'


def _plain_identifier(name: str) -> str:
    """``name`` when it is a plain, optionally schema-qualified SQL identifier; refused otherwise.

    For a statement that interpolates a table name it cannot bind, as the generated SQL elsewhere
    does (unquoted, so it folds case as they do).

    :param name: a table name, ``table`` or ``schema.table``
    :ptype name: str
    :return: the name unchanged
    :rtype: str
    :raises ValueError: when any part is not a plain identifier
    """
    if not all(part.isidentifier() and part.isascii() for part in name.split(".")) or name.count(".") > 1:
        raise ValueError(f"illegal table name {name!r}")
    return name


#: the hosts a direct load may fetch plain ``http`` from: a bucket served on this machine (a local
#: MinIO, a test's server), never one across a network
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _check_load_url(url: str) -> None:
    """refuse a URL a direct load would not fetch.

    A direct load runs in a process that already holds the database's own credentials, so it gains
    no reach by fetching; still it fetches only what its callers legitimately hand it: a presigned
    ``https`` GET (S3), a presigned ``http`` GET of a bucket on this machine (a local MinIO, whose
    presign is ``http://localhost:9000/...``), or a ``file://`` URL, which is how a
    :class:`~threetears.object_store.FilesystemObjectStore` presigns. The broker, which fetches with
    the hub's network position for a pod, admits far less: an ``https`` URL of the caller's own
    recorded export only (``l3.load_files``, :meth:`NatsProxyL3Backend.load_files
    <threetears.core.backends.nats_proxy.NatsProxyL3Backend.load_files>`).

    :param url: the file's URL
    :ptype url: str
    :return: nothing
    :rtype: None
    :raises ValueError: for any other scheme, or plain ``http`` to a host not on this machine
    """
    parts = urlsplit(url)
    if parts.scheme in ("https", "file"):
        return
    if parts.scheme == "http":
        if (parts.hostname or "").lower() not in _LOOPBACK_HOSTS:
            raise ValueError(f"a load fetches plain http only from this machine, not from {parts.hostname!r}")
        return
    raise ValueError(f"a load fetches https, local http and file URLs, not {parts.scheme!r}")


async def _fetch_load_file(url: str) -> bytes:
    """the bytes of one file a load names, once :func:`_check_load_url` admitted its URL.

    :param url: the file's URL
    :ptype url: str
    :return: its bytes
    :rtype: bytes
    :raises ValueError: for a URL :func:`_check_load_url` refuses
    """
    _check_load_url(url)
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = url2pathname(parsed.path)

        def read() -> bytes:
            with open(path, "rb") as handle:
                return handle.read()

        return await asyncio.to_thread(read)
    if not _HTTPX:
        raise ImportError("fetching a load's file over https needs httpx: install 3tears[http]")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), follow_redirects=False) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


def _parquet_columns(data: bytes, columns: Sequence[str]) -> tuple[int, list[list[Any]]]:
    """one parquet file's row count and the values of ``columns``, each column's in file order.

    :param data: the file
    :ptype data: bytes
    :param columns: the columns to read
    :ptype columns: Sequence[str]
    :return: the file's rows, and each column's values
    :rtype: tuple[int, list[list[Any]]]
    :raises ValueError: when the file lacks one of ``columns``
    """
    if not _PARQUET:
        raise ImportError("reading a load's parquet file needs pyarrow: install 3tears[snapshot]")
    table = pq.read_table(io.BytesIO(data))
    missing = [column for column in columns if column not in table.column_names]
    if missing:
        raise ValueError(f"the file holds no column {missing[0]!r}; it holds {table.column_names}")
    return table.num_rows, [table.column(column).to_pylist() for column in columns]


def _request_measure(transport: Any) -> Callable[[str, list[Any]], int | None] | None:
    """the transport's measure of the message it sends for a statement, or None when it has none.

    ``NatsProxyL3Backend`` and its connections measure their own requests
    (``statement_request_bytes``); a direct asyncpg pool sends no message and has none. An answer
    that is not a count (a mock's auto-attribute) is no measure, and the statement is sized by its
    text and parameters alone.

    :param transport: the pool or connection the statements go to
    :ptype transport: Any
    :return: the measure, or None
    :rtype: Callable[[str, list[Any]], int | None] | None
    """
    stated = getattr(transport, "statement_request_bytes", None)
    if not callable(stated):
        return None

    def measure(query: str, params: list[Any]) -> int | None:
        size = stated(query, params)
        return size if isinstance(size, int) and not isinstance(size, bool) else None

    return measure


def _bulk_params(
    schema: TableSchema, table: str, rows: Sequence[Mapping[str, Any]], form: str, turns: LoopTurns | None = None
) -> tuple[list[Any], list[list[Any]]]:
    """the columns a bulk write names, and every row's parameters for them, in the rows' order.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param table: the table, for the refusal
    :ptype table: str
    :param rows: the rows, each keyed by column; at least one
    :ptype rows: Sequence[Mapping[str, Any]]
    :param form: ``upsert`` or ``copy``, for the refusal
    :ptype form: str
    :param turns: the building thread's turns with the caller's loop, taken a row at a time; None to take none
    :ptype turns: LoopTurns | None
    :return: the columns, and each row's parameters
    :rtype: tuple[list[Any], list[list[Any]]]
    :raises ValueError: when the rows disagree on which columns they supply
    :raises UnstampedWriteError: when the table declares a row epoch and a row carries none
    """
    columns = schema_sql.insert_columns_for_data(schema, dict(rows[0]))
    names = [c.name for c in columns]
    params: list[list[Any]] = []
    # checked here, a row at a time on the building thread, never in a pass of its own on the loop
    stamped = getattr(schema, "row_epoch", None) is not None
    for row in rows:
        if turns is not None:
            turns.take()
        data = dict(row)
        if stamped and data.get(EPOCH_COLUMN) is None:
            raise _unstamped(table)
        if [c.name for c in schema_sql.insert_columns_for_data(schema, data)] != names:
            raise ValueError(
                f"{table}: rows of one bulk {form} must supply the same columns; "
                f"a server-default column is supplied by some rows and not others"
            )
        params.append(schema_sql.build_insert_params(schema, data))
    return list(columns), params


def _upsert_batches(
    schema: TableSchema,
    table: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int,
    max_bytes: int,
    measure: Callable[[str, list[Any]], int | None] | None,
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple[list[Any], list[list[list[Any]]]]:
    """:meth:`SqlBackend.upsert_many`'s columns and statement batches: the CPU half of the write, run off the loop.

    Every row's parameters are built and each candidate statement is measured as the transport sends it,
    which for a page of a thousand rows is tens of milliseconds of Python: on the event loop it held every
    call the process answers for that long, a page after a page.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param table: the table
    :ptype table: str
    :param rows: the rows; at least one
    :ptype rows: Sequence[Mapping[str, Any]]
    :param max_rows: the most rows one statement writes
    :ptype max_rows: int
    :param max_bytes: the most bytes one statement's request comes to
    :ptype max_bytes: int
    :param measure: the transport's measure of a statement's request, or None
    :ptype measure: Callable[[str, list[Any]], int | None] | None
    :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`): every row's parameters
        built and measured a row at a time, 33,786 rows of 179 columns a second of Python that held the loop 15 ms at
        a time under the default switch interval (the WD2 bench, 2026-11-16); None off any loop
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: the columns, and the batches of rows' parameters
    :rtype: tuple[list[Any], list[list[list[Any]]]]
    :raises ValueError: when the rows disagree on which columns they supply
    """
    turns = LoopTurns(loop)
    columns, params = _bulk_params(schema, table, rows, "upsert", turns)
    batches = schema_sql.bulk_insert_batches(
        schema, columns, params, max_rows=max_rows, max_bytes=max_bytes, request_bytes=measure, turns=turns
    )
    return columns, list(batches)


#: a bulk statement binding more parameters than this has its text and parameters laid out on a thread
#: (:func:`_bulk_statement`), as the rail's connection builds and encodes its request (``_OFFLOAD_PARAMS``)
_STATEMENT_ON_THREAD_PARAMS: Final = 256


def _bulk_statement(
    schema: TableSchema,
    columns: list[Any],
    batch: Sequence[Sequence[Any]],
    texts: dict[int, str],
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple[str, tuple[Any, ...]]:
    """one batch's bulk INSERT (:func:`schema_sql.build_bulk_insert_sql`) and its parameters, flattened in row order.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param columns: the columns every row supplies
    :ptype columns: list[Any]
    :param batch: each row's parameters, in the order of ``columns``
    :ptype batch: Sequence[Sequence[Any]]
    :param texts: the statements' texts made so far, by their row count; this one's added when new
    :ptype texts: dict[int, str]
    :param loop: the caller's event loop, handed the thread's turns a row at a time; None on the loop itself
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: the statement's text and its parameters
    :rtype: tuple[str, tuple[Any, ...]]
    """
    sql = texts.get(len(batch))
    if sql is None:
        sql = texts[len(batch)] = schema_sql.build_bulk_insert_sql(schema, rows=len(batch), columns=columns)
    rows = batch if loop is None else stepped(batch, LoopTurns(loop), every=1)
    return sql, tuple(value for row in rows for value in row)


def _copy_batches(
    schema: TableSchema,
    table: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int,
    max_bytes: int,
    conn: Any,
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple[list[str], list[list[tuple[Any, ...]]]]:
    """:meth:`SqlBackend.copy_many`'s column names and copy batches: the CPU half of the copy, run off the loop.

    As :func:`_upsert_batches`: every row's values built, ordered by key and each sized as the request
    sends it, which on the event loop held the process's every call a page at a time.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param table: the table
    :ptype table: str
    :param rows: the rows; at least one
    :ptype rows: Sequence[Mapping[str, Any]]
    :param max_rows: the most rows one copy carries
    :ptype max_rows: int
    :param max_bytes: the most bytes one copy's request comes to
    :ptype max_bytes: int
    :param conn: the connection the copies go to, whose ``copy_request_bytes`` measures a request
    :ptype conn: Any
    :param loop: the caller's event loop, handed the thread's turns as :func:`_upsert_batches` hands them; None off
        any loop
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: the column names, and the batches of records
    :rtype: tuple[list[str], list[list[tuple[Any, ...]]]]
    :raises ValueError: when the rows disagree on which columns they supply, or a column is one a
        copy cannot write
    """
    turns = LoopTurns(loop)
    columns, params = _bulk_params(schema, table, rows, "copy", turns)
    names = [c.name for c in columns]
    unwritable = [c for c in columns if not schema_sql.copy_writes(c)]
    if unwritable:
        raise ValueError(
            f"{table}: a copy cannot write the {unwritable[0].column_type} column {unwritable[0].name!r}; "
            f"save the rows with an upsert"
        )
    key_positions = [names.index(name) for name in schema.pk_columns]
    params.sort(key=lambda values: tuple(values[position] for position in key_positions))
    measure = getattr(conn, "copy_request_bytes", None)
    if not callable(measure):
        # a direct connection streams a copy, so nothing bounds its bytes: rows alone cut it, and no row is
        # encoded only to be measured (a third of a 50,000-row copy's preparation)
        return names, [
            [tuple(values) for values in params[start : start + max_rows]] for start in range(0, len(params), max_rows)
        ]
    measured = measure(table, names, [])
    base = measured if isinstance(measured, int) and not isinstance(measured, bool) else 0
    batches = schema_sql.bulk_batches(
        params,
        max_rows=max_rows,
        max_bytes=max_bytes,
        # a copy binds no parameters, so only the row and byte bounds cut it
        max_params=max_rows * len(names),
        base_bytes=base,
        # each row after the first follows a ", " in the request's list of rows
        position_bytes=lambda position: 2 if position else 0,
        turns=turns,
    )
    return names, [[tuple(values) for values in batch] for batch in batches]


def _key_led_deletes(
    schema: TableSchema, named: list[tuple[Any, ...]], *, max_rows: int, turns: LoopTurns | None = None
) -> list[tuple[str, list[Any]]]:
    """:meth:`SqlBackend.delete_many`'s statements for ``named`` keys, each with its arguments.

    Two key-led forms, whichever needs fewer statements (the second on a tie): whole keys, ``max_rows`` a
    statement, one array a key column; or the keys grouped by all but one key column, ``max_rows`` values of
    the varying one a statement.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param named: the keys, as a statement binds them
    :ptype named: list[tuple[Any, ...]]
    :param max_rows: the most keys one statement names
    :ptype max_rows: int
    :param turns: the building thread's turns with the caller's loop, taken a key at a time; None to take none
    :ptype turns: LoopTurns | None
    :return: each statement and its arguments
    :rtype: list[tuple[str, list[Any]]]
    """
    if not named:
        return []
    varying, groups = _key_led_groups(schema, named, turns)
    grouped = sum(-(-len(values) // max_rows) for _, values in groups)
    whole = -(-len(named) // max_rows)
    if len(schema.pk_columns) > 1 and whole < grouped and _every_key_column_has_an_array(schema):
        sql = schema_sql.build_whole_key_delete_sql(schema)
        statements: list[tuple[str, list[Any]]] = []
        for start in range(0, len(named), max_rows):
            statements.append((sql, [list(column) for column in zip(*named[start : start + max_rows], strict=True)]))
            if turns is not None:
                turns.take()
        return statements
    sql = schema_sql.build_key_led_delete_sql(schema, varying=schema.pk_columns[varying])
    return [
        (sql, [values[start : start + max_rows], *fixed])
        for fixed, values in (groups if turns is None else stepped(groups, turns))
        for start in range(0, len(values), max_rows)
    ]


def _callers_by_bound(
    named: Sequence[tuple[Any, ...]], keys: Sequence[Sequence[Any]], loop: asyncio.AbstractEventLoop | None = None
) -> dict[tuple[Any, ...], tuple[Any, ...]]:
    """each key as the caller named it, under the key as the delete bound it (:meth:`SqlL3Backend._keys_to_delete`);
    pure, for a thread.

    :param named: each key as bound, in ``keys``' order
    :ptype named: Sequence[tuple[Any, ...]]
    :param keys: each key as the caller named it
    :ptype keys: Sequence[Sequence[Any]]
    :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`); None on the loop itself
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: the caller's keys by their bound keys
    :rtype: dict[tuple[Any, ...], tuple[Any, ...]]
    """
    pairs = zip(named, keys, strict=True)
    return {bound: tuple(key) for bound, key in (pairs if loop is None else stepped(pairs, LoopTurns(loop)))}


def _returned_rows(
    schema: TableSchema,
    rows: Sequence[Mapping[str, Any]],
    callers: Mapping[tuple[Any, ...], tuple[Any, ...]],
    found: dict[tuple[Any, ...], dict[str, Any]],
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """read the rows a ``DELETE ... RETURNING`` returned into ``found``, each under its key as the caller named it and
    read as :meth:`SqlL3Backend.fetch_many` reads it; a row whose key the caller did not name left out. Pure but for
    ``found``, for a thread.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param rows: the returned rows, in the order returned
    :ptype rows: Sequence[Mapping[str, Any]]
    :param callers: each key as the caller named it, under the key as bound (:func:`_callers_by_bound`)
    :ptype callers: Mapping[tuple[Any, ...], tuple[Any, ...]]
    :param found: the rows read so far, by the caller's key; added to in order
    :ptype found: dict[tuple[Any, ...], dict[str, Any]]
    :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`); None on the loop itself
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: nothing
    :rtype: None
    """
    key_columns = [schema.column(name) for name in schema.pk_columns]
    # a turn looked at every STEP_ITEMS values, not rows: a row of 49 columns read in 10 us here, 128 of them past
    # the half millisecond a turn allows
    every = max(1, STEP_ITEMS // max(1, len(schema.columns)))
    for row in rows if loop is None else stepped(rows, LoopTurns(loop), every):
        read = dict(row)
        bound = tuple(schema_sql.normalize_write_value(column, read[column.name]) for column in key_columns)
        if bound in callers:
            found[callers[bound]] = schema_sql.coerce_row(schema, read)


def _key_led_groups(
    schema: TableSchema, keys: list[tuple[Any, ...]], turns: LoopTurns | None = None
) -> tuple[int, list[tuple[tuple[Any, ...], list[Any]]]]:
    """group whole keys for key-led deletes: one key column varies in an array, the rest are fixed.

    The varying column is the one leaving the fewest groups, the leading column on a tie, among the
    key columns with an array form. Each group is the fixed columns' values, in key order, and the
    varying column's values, in the order the keys came. Keys are told apart by their values' JSON
    form, so a fixed jsonb key column (a dict, which cannot key a dict) groups like any other.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param keys: whole keys, write-coerced, in the schema's key order
    :ptype keys: list[tuple[Any, ...]]
    :param turns: the building thread's turns with the caller's loop, taken a key at a time: a JSON text a key a
        key column, 33,786 keys of seven columns, held the loop a quarter of a second (2026-11-16); None to take none
    :ptype turns: LoopTurns | None
    :return: the varying column's index in the key, and the groups
    :rtype: tuple[int, list[tuple[tuple[Any, ...], list[Any]]]]
    :raises ValueError: when no key column has an array form
    """
    best: tuple[int, list[tuple[tuple[Any, ...], list[Any]]]] | None = None
    for index, name in enumerate(schema.pk_columns):
        if schema_sql.key_array_type(schema, name) is None:
            continue
        groups: dict[str, tuple[tuple[Any, ...], list[Any]]] = {}
        for key in keys if turns is None else stepped(keys, turns):
            fixed = key[:index] + key[index + 1 :]
            token = json.dumps(list(fixed), default=schema_sql.json_default, sort_keys=True)
            groups.setdefault(token, (fixed, []))[1].append(key[index])
        if best is None or len(groups) < len(best[1]):
            best = (index, list(groups.values()))
    if best is None:
        raise ValueError(f"{schema.name}: no key column of {schema.pk_columns} has an array form to delete by")
    return best


def _every_key_column_has_an_array(schema: TableSchema) -> bool:
    """whether every key column can be named as an array, so whole keys can go one array a column.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :return: True when each has an array form
    :rtype: bool
    """
    return all(schema_sql.key_array_type(schema, name) is not None for name in schema.pk_columns)


@dataclass
class _LedRead:
    """one key-led read: its statement, the transport's cap, and a tally of what it took.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param sql: the batch statement (:func:`~threetears.core.backends.schema_sql.build_led_by_select_sql`)
    :ptype sql: str
    :param columns: the columns read, the key's among them
    :ptype columns: tuple[str, ...]
    :param row_cap: the most rows the transport answers a statement, or None when it never cuts
    :ptype row_cap: int | None
    :param reader: what the statements run on: the caller's connection, or the backend
    :ptype reader: Any
    """

    schema: TableSchema
    sql: str
    columns: tuple[str, ...]
    row_cap: int | None
    reader: Any
    statements: int = 0
    splits: int = 0
    paged_values: int = 0

    async def batch(self, values: list[Any]) -> list[dict[str, Any]]:
        """the rows a batch of leading-key values holds: in halves while an answer reaches the cap, then by pages.

        :param values: the batch's leading-key values, write-coerced
        :ptype values: list[Any]
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        self.statements += 1
        try:
            rows = [dict(row) for row in await self.reader.fetch(self.sql, values)]
        except L3AnswerCutError:
            # the rail said it cut the answer: read as an answer that reached the cap is, in halves then by pages
            rows = None
        if rows is not None and (self.row_cap is None or len(rows) < self.row_cap):
            return rows
        if self.row_cap is None:
            raise L3AnswerCutError(f"{self.schema.name}: a transport that states no cap cut a key-led answer", rows=0)
        if len(values) > 1:
            self.splits += 1
            half = len(values) // 2
            return await self.batch(values[:half]) + await self.batch(values[half:])
        return await self._paged(values[0], self.row_cap)

    async def _paged(self, value: Any, row_cap: int) -> list[dict[str, Any]]:
        """every row one leading-key value holds, paged by the rest of its key under the cap.

        :param value: the leading-key value, write-coerced
        :ptype value: Any
        :param row_cap: the transport's cap
        :ptype row_cap: int
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        lead, *rest = self.schema.pk_columns
        self.paged_values += 1
        select_from = f"SELECT {schema_sql.build_select_column_list(self.schema, self.columns)} FROM {self.schema.name}"
        read = await read_keyset_pages(
            self.reader,
            select_from,
            rest,
            where={lead: value},
            page_size=row_cap - 1,
            quote=as_written,
            mark=lambda column, index: schema_sql.render_param(self.schema.column(column), index),
        )
        self.statements += read.statements
        return read.rows


class SqlL3Backend:
    """An :class:`L3Backend` + :class:`DurableStore` over an asyncpg-shaped pool.

    :param pool: the underlying durable handle — an asyncpg ``Pool`` (or anything with
        the same ``fetch`` / ``fetchrow`` / ``execute`` / ``acquire`` surface, e.g. the
        ``NatsProxyL3Backend``). Lifecycle (open/close) is owned by the caller.
    :ptype pool: Any
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        # table name -> TableSchema. when a schema is registered for a table the
        # structured DurableStore ops generate schema-aware SQL + run the per-column
        # read/write codec; otherwise the generic SELECT */INSERT path is used.
        self._schemas: Mapping[str, TableSchema] = MappingProxyType({})
        # capability sniff (mirrors the ``hasattr(self._pool, "transaction")`` check in
        # ``transaction`` below): a namespace-aware transport such as
        # ``NatsProxyL3Backend`` carries ``accepts_scoped_reads = True`` and routes
        # ``namespace`` + ``customer_scope`` to the broker, so the raw-SQL transport
        # methods forward those kwargs instead of dropping them. NOT an isinstance gate
        # -- NatsProxyL3Backend omits ``fetchval`` so it fails the L3Backend protocol
        # check, which would silently leave forwarding off. a bare asyncpg pool lacks
        # the marker, so its behaviour is unchanged (kwargs dropped).
        #
        # identity (``is True``), not truthiness: the marker is the literal ``True`` on the
        # transport class. ``bool(getattr(...))`` would be fooled by a ``MagicMock`` pool,
        # whose auto-created attribute is a truthy mock object, flipping forwarding on and
        # pushing ``namespace=`` into a mock signature that rejects it.
        self._scope_aware: bool = getattr(pool, "accepts_scoped_reads", False) is True

    def __getattr__(self, name: str) -> Any:
        """Delegate any unlisted attribute to the wrapped pool (the raw-SQL escape hatch).

        ``L3Backend`` formalizes the common transport surface (``fetch`` / ``fetchrow`` /
        ``execute`` / ``execute_batch`` / ``acquire`` / ``transaction``), but the
        ``l3_pool`` ivar is also the documented ad-hoc-SQL escape hatch — product code
        reaches for the full asyncpg pool surface through it (e.g. ``fetchval``,
        ``copy_records_to_table``). Forwarding the long tail keeps the wrapper a
        transparent superset of the pool it wraps, so wrapping never removes a method a
        caller already relied on. Invoked only for attributes not found normally
        (``_pool`` / ``_schemas`` and every defined method resolve before this).

        :param name: the missing attribute name.
        :ptype name: str
        :return: the corresponding attribute on the wrapped pool.
        :rtype: Any
        :raises AttributeError: if the wrapped pool has no such attribute.
        """
        return getattr(self._pool, name)

    async def answers_from_yugabyte(self) -> bool:
        """whether the reader behind this backend answers from YugabyteDB, asked of that reader
        (:func:`~threetears.core.backends.dialect.known_on_yugabyte`): this wrapper sends no ``SELECT version()``.

        WHY: a collection's ``l3_pool`` is this wrapper around a bare pool or the rail's proxy. Asked as a pool, it sent
        ``version()`` through the rail, which the hub's broker refuses (a pod's keyed read with an index hint failed),
        and over a bare pool it was a second question beside the pool's own (test_known_dialect).

        :return: whether the database behind it is YugabyteDB
        :rtype: bool
        """
        return await known_on_yugabyte(self._pool)

    def register_schema(self, table: str, schema: TableSchema) -> None:
        """Register a :class:`TableSchema` so the structured ops emit schema-aware SQL for ``table``.

        Idempotent — re-registering the same table overwrites the prior schema (a
        Collection registers its schema on construction; reconstruction is harmless).

        :param table: target table name (matches ``schema.name``).
        :ptype table: str
        :param schema: the table schema driving SQL generation + value coercion.
        :ptype schema: TableSchema
        :return: nothing
        :rtype: None
        """
        self._schemas = MappingProxyType({**self._schemas, table: schema})

    # ── L3Backend: raw-SQL transport (delegate to the pool) ─────────────────────────
    # ``namespace`` + extra ``**kwargs`` (e.g. NatsProxy's ``customer_scope``) are
    # forwarded to the pool ONLY when it declared ``accepts_scoped_reads`` (see
    # __init__): a namespace-aware transport routes them to the broker; a bare asyncpg
    # pool would TypeError on them, so they are dropped for it (the historical path).
    # generic by design -- the wrapper forwards opaque kwargs and stays ignorant of
    # NATS-specific concepts like ``customer_scope``.
    def _target(self) -> tuple[Any, bool]:
        """The handle to run a statement on, plus whether it forwards scoped-read kwargs.

        A request-scoped bound connection (a raw conn whose ``search_path`` + session GUCs the
        binder already set) takes precedence over the pool and is NOT scope-aware (``namespace``
        + ``customer_scope`` kwargs dropped -- routing is already pinned on the connection).
        UNBOUND, the pool is used with its declared scope-awareness -- the historical path,
        byte-for-byte unchanged.
        """
        bound = _request_l3_conn.get()
        if bound is not None:
            return bound, False
        return self._pool, self._scope_aware

    @property
    def binds_native_arrays(self) -> bool:
        """whether its pool binds typed arrays natively (asyncpg's own); the L3 rail's carries JSON and does not.

        :return: whether it does
        :rtype: bool
        """
        return binds_native_arrays(self._pool)

    async def fetch(
        self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Run a SELECT and return all rows as dicts (bound request conn, else the pool)."""
        target, scope_aware = self._target()
        with _unmask_asyncpg_release_race():
            if scope_aware:
                rows = await target.fetch(query, *params, namespace=namespace, **kwargs)
            else:
                rows = await target.fetch(query, *params)
        return [dict(r) for r in rows]

    async def fetchrow(
        self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Run a SELECT and return the first row dict, or ``None`` (bound request conn, else pool)."""
        target, scope_aware = self._target()
        with _unmask_asyncpg_release_race():
            if scope_aware:
                row = await target.fetchrow(query, *params, namespace=namespace, **kwargs)
            else:
                row = await target.fetchrow(query, *params)
        return dict(row) if row is not None else None

    async def fetchval(self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any) -> Any:
        """Run a SELECT and return the first column of the first row (bound request conn, else pool)."""
        target, scope_aware = self._target()
        with _unmask_asyncpg_release_race():
            if scope_aware:
                value = await target.fetchval(query, *params, namespace=namespace, **kwargs)
            else:
                value = await target.fetchval(query, *params)
        return value

    async def execute(self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any) -> str:
        """Run an INSERT/UPDATE/DELETE; return the command-tag (bound request conn, else pool)."""
        target, scope_aware = self._target()
        with _unmask_asyncpg_release_race():
            if scope_aware:
                result = await target.execute(query, *params, namespace=namespace, **kwargs)
            else:
                result = await target.execute(query, *params)
        return result if isinstance(result, str) else ""

    async def execute_batch(
        self, queries: list[dict[str, Any]], *, namespace: str | None = None, transaction: bool = True
    ) -> list[Any]:
        """Run ``{query, params}`` dicts atomically in one transaction when ``transaction``."""
        bound = _request_l3_conn.get()
        if bound is not None:
            # already inside the request transaction; run every statement on the bound conn so
            # the batch shares the request's transaction + session GUC (no nested transaction).
            on_conn: list[Any] = []
            for q in queries:
                on_conn.append(await bound.execute(q["query"], *q.get("params", [])))
            return on_conn
        out: list[Any] = []
        with _unmask_asyncpg_release_race():
            if not transaction:
                for q in queries:
                    out.append(await self._pool.execute(q["query"], *q.get("params", [])))
            else:
                async with self._pool.acquire() as conn, conn.transaction():
                    for q in queries:
                        out.append(await conn.execute(q["query"], *q.get("params", [])))
        return out

    def acquire(self) -> Any:
        """Return an ``acquire()`` async-CM: the bound request conn when set, else a pooled one."""
        bound = _request_l3_conn.get()
        if bound is not None:
            return _BoundConnCM(bound)
        return self._pool.acquire()

    def transaction(self, namespace: str | None = None) -> Any:
        """Return a transaction async-CM: the bound request conn when set (reuse the request

        transaction, no nesting), else a fresh pool-level transaction.
        """
        bound = _request_l3_conn.get()
        if bound is not None:
            return _BoundConnCM(bound)
        if hasattr(self._pool, "transaction"):
            return self._pool.transaction(namespace=namespace)
        return _PoolTransactionCM(self._pool)

    # ── DurableStore: structured ops (generate SQL) ─────────────────────────────────
    async def fetch_one(self, table: str, pk: KeyValues, *, conn: Any = None) -> dict[str, Any] | None:
        """Fetch the row whose primary key equals ``pk`` via a generated SELECT, or ``None``.

        When a schema is registered for ``table`` the SELECT projects the schema's
        declared columns (``VECTOR``/``TSVECTOR`` cast ``::text``), the pk values are
        write-coerced, and the row is read-coerced — byte-identical to the collection's
        old ``fetch_from_store``. Otherwise a generic ``SELECT *`` is issued.
        """
        schema = self._schemas.get(table)
        if schema is None:
            _quote_ident(table)  # a name holding a quote is refused, as it always was
            return await read_row(self if conn is None else conn, table, by=dict(pk))
        coerced: list[Any] = [
            schema_sql.normalize_write_value(schema.column(name), pk[name]) for name in schema.pk_columns
        ]
        row = await self._fetchrow(schema_sql.build_fetch_sql(schema), *coerced, conn=conn)
        if row is None:
            return None
        return schema_sql.coerce_row(schema, dict(row))

    async def upsert(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Annotated[Sequence[str], DeclaredKey] | None = None,
        on_conflict: str = "update",
        cas: datetime | None = None,
        conn: Any = None,
    ) -> int:
        """Insert-or-update ``row`` via a generated INSERT…ON CONFLICT; return rows affected.

        When a schema is registered for ``table`` the INSERT / CAS-UPDATE SQL + params are
        generated from the schema (``on_conflict`` modes, server-default dropping, CAS
        fencing, ``::jsonb`` / ``::vector`` casts) — byte-identical to the collection's old
        ``save_to_store``; ``pk`` / ``on_conflict`` arguments are taken from the schema.
        Otherwise the generic INSERT path runs, honoring the ``pk`` / ``on_conflict`` / ``cas``
        arguments directly.

        :param conn: optional caller-supplied connection (a transaction handle) the write
            binds to instead of the pool, so it commits atomically with the caller's
            other operations. ``None`` uses the wrapped pool.
        :ptype conn: Any
        """
        schema = self._schemas.get(table)
        if schema is not None:
            _refuse_unstamped_rows(schema, table, (row,))
            return await self._upsert_schema(schema, dict(row), cas=cas, conn=conn)

        if pk is None:
            raise ValueError("upsert without a registered schema requires the pk argument")
        if on_conflict not in _ON_CONFLICT_VALUES:
            raise ValueError(f"on_conflict must be one of {sorted(_ON_CONFLICT_VALUES)!r}, got {on_conflict!r}")
        cols = list(row.keys())
        placeholders = ", ".join(f"${i}" for i in range(1, len(cols) + 1))
        col_sql = ", ".join(_quote_ident(c) for c in cols)
        pk_sql = ", ".join(_quote_ident(c) for c in pk)
        insert = f"INSERT INTO {_quote_ident(table)} ({col_sql}) VALUES ({placeholders})"
        params: list[Any] = [row[c] for c in cols]

        if on_conflict == "raise":
            return parse_rowcount(await self._execute(insert, *params, conn=conn))
        if on_conflict == "ignore":
            return parse_rowcount(
                await self._execute(f"{insert} ON CONFLICT ({pk_sql}) DO NOTHING", *params, conn=conn)
            )

        # "update": upsert the mutable (non-pk) columns.
        mutable = [c for c in cols if c not in set(pk)]
        if cas is not None and "date_updated" in row:
            # a fence value means the caller read the row as EXISTING, so this is an
            # update and never an insert. an upsert fenced in its DO UPDATE branch would
            # re-create a row deleted since that read, because a missing row reaches the
            # INSERT and the fence never runs.
            return await self._update_fenced(table, row, pk=pk, mutable=mutable, cas=cas, conn=conn)
        set_sql = (
            ", ".join(f"{_quote_ident(c)} = EXCLUDED.{_quote_ident(c)}" for c in mutable) or pk_sql + " = " + pk_sql
        )
        sql = f"{insert} ON CONFLICT ({pk_sql}) DO UPDATE SET {set_sql}"
        return parse_rowcount(await self._execute(sql, *params, conn=conn))

    async def upsert_many(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        max_rows: int,
        max_bytes: int,
        conn: Any = None,
    ) -> int:
        """Insert-or-update many rows of a schema-registered table in multi-row statements.

        The column list is derived once, from the rows (:func:`schema_sql.insert_columns_for_data`,
        which leaves out a server-default column a row omits), and both each statement's SQL and
        every row's parameters are built from it, so the two cannot disagree. Rows that disagree on
        which server-default columns they supply are refused: one statement names one column list.
        Batches split by :func:`schema_sql.bulk_insert_batches`, each statement's whole request
        (its text, its parameters and what the transport sends around them, measured by the
        transport that sends it) within ``max_bytes``. On ``conn`` a failing statement fails the
        caller's transaction, so every batch rolls back with it.

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param rows: the rows, each keyed by column
        :ptype rows: Sequence[Mapping[str, Any]]
        :param max_rows: the most rows one statement writes
        :ptype max_rows: int
        :param max_bytes: the most bytes one statement's request comes to on the L3 rail
        :ptype max_bytes: int
        :param conn: the caller's transaction handle; ``None`` uses the pool
        :ptype conn: Any
        :return: rows written
        :rtype: int
        :raises ValueError: when no schema is registered for ``table``, or the rows disagree on
            which columns they supply
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"upsert_many needs the schema of {table!r} registered")
        written = 0
        if rows:
            # the rows' parameters built and each statement measured on a thread: tens of milliseconds of
            # Python a page, which on the loop held every call the process answers (_upsert_batches)
            columns, batches = await asyncio.to_thread(
                _upsert_batches,
                schema,
                table,
                rows,
                max_rows=max_rows,
                max_bytes=max_bytes,
                measure=_request_measure(conn if conn is not None else self._pool),
                loop=asyncio.get_running_loop(),
            )
            # each statement's text and parameters laid out on a thread too: a page's text names every parameter
            # (32,000 of them at 179 columns) and its parameters are flattened a value at a time, 7-8 ms a statement
            # on the loop, 187 statements a 33,786-row write (the WD2 bench, 2026-11-16). One text a statement size
            texts: dict[int, str] = {}
            for batch in batches:
                sql, params = (
                    await asyncio.to_thread(_bulk_statement, schema, columns, batch, texts, asyncio.get_running_loop())
                    if len(batch) * len(columns) > _STATEMENT_ON_THREAD_PARAMS
                    else _bulk_statement(schema, columns, batch, texts)
                )
                await self._execute(sql, *params, conn=conn)
                written += len(batch)
            # the rows' parameters are this call's alone: freed as it returned, 33,786 rows of 179 held the loop
            # 10-15 ms (the WD2 bench, 2026-11-16), so they are freed on a thread, taking turns with the loop
            await asyncio.to_thread(emptied, [*batches, batches], asyncio.get_running_loop())
        return written

    async def copy_many(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        max_rows: int,
        max_bytes: int,
        conn: Any,
    ) -> int:
        """Insert many new rows of a schema-registered table with ``COPY``, on the caller's transaction.

        The bulk load's other form beside :meth:`upsert_many`: no conflict clause, so a key already
        held fails the copy and with it the caller's transaction. The rows are ordered by primary
        key first, the order a bulk load into YugabyteDB wants, and each copy goes to the
        connection's ``copy_records_to_table`` -- asyncpg's own on a direct connection, the
        ``l3.tx.copy`` request on the rail's. Columns and values are the ones :meth:`upsert_many`
        would bind (:func:`schema_sql.insert_columns_for_data`, :func:`schema_sql.build_insert_params`),
        so the two forms write the same row. Batches split by ``max_rows`` and, where the connection
        measures its request (``copy_request_bytes``: the rail's), by ``max_bytes`` of the whole request; a
        direct connection streams the copy, so its batches are cut by rows alone.

        A copy of :data:`~threetears.core.ddl.change_notes.WHOLE_COPY_ROWS` rows or more on a direct
        connection marks its transaction whole (:data:`~threetears.core.ddl.change_notes.NOTE_WHOLE_SETTING`):
        one change note for the table, not a note (a write) a row that names nothing past that many. On
        the rail's connection the broker marks it, counting the session's copied rows; a client may not
        set it there.

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param rows: the rows, each keyed by column
        :ptype rows: Sequence[Mapping[str, Any]]
        :param max_rows: the most rows one copy carries; at least one
        :ptype max_rows: int
        :param max_bytes: the most bytes one copy's request comes to on the L3 rail
        :ptype max_bytes: int
        :param conn: the caller's connection, its transaction open
        :ptype conn: Any
        :return: rows written
        :rtype: int
        :raises ValueError: when no schema is registered for ``table``, ``max_rows`` is under one,
            the rows disagree on which columns they supply, or a column is one a copy cannot write
            (:func:`schema_sql.copy_writes`: a jsonb or a vector)
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"copy_many needs the schema of {table!r} registered")
        if max_rows < 1:
            raise ValueError(f"{table}: max_rows must be at least 1, got {max_rows}")
        if not rows:
            return 0
        # the rows' values built, ordered and sized on a thread, as an upsert's are (_copy_batches)
        names, batches = await asyncio.to_thread(
            _copy_batches,
            schema,
            table,
            rows,
            max_rows=max_rows,
            max_bytes=max_bytes,
            conn=conn,
            loop=asyncio.get_running_loop(),
        )
        if len(rows) >= WHOLE_COPY_ROWS and getattr(conn, "copy_request_bytes", None) is None:
            await conn.execute("SELECT set_config($1, '1', true)", NOTE_WHOLE_SETTING)
        written = 0
        for batch in batches:
            await conn.copy_records_to_table(table, records=batch, columns=names)
            written += len(batch)
        # the records are this call's alone, freed on a thread as an upsert's parameters are (upsert_many)
        await asyncio.to_thread(emptied, [*batches, batches], asyncio.get_running_loop())
        _logger.debug("bulk copy", extra={"extra_data": {"table": table, "rows": written}})
        return written

    def arrow_write_plan(
        self,
        table: str,
        columns: Sequence[str],
        types: Sequence[Any],
        *,
        kind: Literal["upsert", "copy", "delete"],
        conn: Any,
    ) -> ArrowWritePlan | None:
        """how an Arrow table's rows are written to ``table`` through the hub's Arrow write door on ``conn``; None
        when they cannot be, and are written as Python values instead (:meth:`upsert_many`, :meth:`copy_many`,
        :meth:`delete_many`).

        The rows go as :meth:`upsert_many` or :meth:`copy_many` would send them: the columns
        :func:`schema_sql.insert_columns_for_data` emits for them (a server-default column they leave out is left
        out; a nullable one is sent all null; a required one is refused), an upsert's ``SET`` list the bulk
        statement's (:func:`schema_sql.bulk_update_columns`), a copy ordered by key. A delete sends the key's columns
        alone. None, the rows written as Python values: on a connection without the door (a direct connection, or a
        hub that announced no ``write_arrow``), without pyarrow, for an upsert into a table whose conflict clause is
        ``raise``, when a column sent is one a copy cannot write (jsonb, vector) or the rows' type for it is one the
        door does not carry (:func:`~threetears.core.backends.arrow_copy.write_type`).

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param columns: the rows' columns
        :ptype columns: Sequence[str]
        :param types: each column's Arrow type, in the order of ``columns``
        :ptype types: Sequence[pyarrow.DataType]
        :param kind: an upsert, a copy of new rows, or a delete of keys
        :ptype kind: Literal["upsert", "copy", "delete"]
        :param conn: the caller's connection, its transaction open
        :ptype conn: Any
        :return: the plan, or None
        :rtype: ArrowWritePlan | None
        :raises ValueError: when no schema is registered for ``table``
        :raises KeyError: when the rows lack a key column, or a required column of an upsert or a copy
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"an Arrow write needs the schema of {table!r} registered")
        given = dict(zip(columns, types, strict=True))
        missing_key = [name for name in schema.pk_columns if name not in given]
        if missing_key:
            raise KeyError(f"{table}: the rows have no key column {missing_key[0]!r}")
        if getattr(conn, "write_arrow", False) is not True or not arrow_copy_available():
            return None
        emitted = (
            [column for name in schema.pk_columns if (column := schema.get_column(name)) is not None]
            if kind == "delete"
            else schema_sql.insert_columns_for_data(schema, given)
        )
        update = None if kind != "upsert" else schema_sql.bulk_update_columns(schema, emitted)
        if (kind == "upsert" and update is None) or not all(schema_sql.copy_writes(column) for column in emitted):
            return None
        sent: list[tuple[str, str | None, Any]] = []
        for column in emitted:
            if column.name in given:
                carried = write_type(given[column.name], as_utc=column.column_type == "datetimetz")
                if carried is None:
                    return None
                sent.append((column.name, column.name, carried))
            elif column.nullable:
                sent.append((column.name, None, None))
            else:
                raise KeyError(f"{table}.save_to_store: required column {column.name!r} missing from data")
        return ArrowWritePlan(
            kind=kind,
            columns=tuple(sent),
            # the stored key: a versioned table's conflict is a version of the key (its epoch), never the key alone
            conflict_columns=tuple(schema.stored_key) if kind == "upsert" else (),
            update_columns=tuple(update or ()),
            sort_by=tuple(schema.pk_columns) if kind == "copy" else (),
        )

    async def write_arrow(
        self, table: str, rows: Any, plan: ArrowWritePlan, *, conn: Any, max_rows: int | None = None
    ) -> int:
        """write an Arrow table's rows to ``table`` as :meth:`arrow_write_plan` planned, on the caller's transaction:
        the request's table built by Arrow on a thread (each column cast to the type the door carries, a copy ordered
        by key), then sent through the connection's write door (``write_arrow_rows``), cut by Arrow slices against the
        request budget and, given, ``max_rows``. A failing request fails the caller's transaction.

        :param table: the table
        :ptype table: str
        :param rows: the rows, of the columns and types the plan was made for
        :ptype rows: pyarrow.Table
        :param plan: the plan
        :ptype plan: ArrowWritePlan
        :param conn: the caller's connection, its transaction open and its hub's door announced
        :ptype conn: Any
        :param max_rows: the most rows one request (one statement at the hub) carries; None for the budget's bytes alone
        :ptype max_rows: int | None
        :return: the rows the hub's statements counted
        :rtype: int
        """
        if not rows.num_rows:
            return 0
        if plan.kind != "delete":
            _refuse_unstamped_arrow(self._schemas.get(table), table, rows)
        planned = await asyncio.to_thread(planned_write_table, rows, plan.columns, sort_by=plan.sort_by)
        written: int = await conn.write_arrow_rows(
            table,
            planned,
            conflict_columns=plan.conflict_columns,
            update_columns=plan.update_columns,
            delete=plan.kind == "delete",
            max_rows=max_rows,
        )
        _logger.debug(
            "Arrow write", extra={"extra_data": {"table": table, "kind": plan.kind, "rows": int(rows.num_rows)}}
        )
        return written

    async def truncate(self, table: str, *, reason: str | None = None) -> None:
        """Empty ``table`` with ``TRUNCATE``, alone and outside any transaction. On a direct pool the collection
        records the epoch itself, ``reason`` with it, and the truncate's change note names it too
        (:data:`~threetears.core.ddl.change_notes.NOTE_REACH_SETTING`, set for the statement's session and cleared
        after), so a log restored from the notes puts the reset back as a reset.

        On the rail the broker runs it (``NatsProxyL3Backend.truncate``, the ``l3.truncate``
        request), under the database's DDL lock and only for a table of the caller's own data
        space. On a direct pool it runs here, on one connection holding that same lock
        (:func:`~threetears.core.data.migrations.ddl_lock.database_ddl_lock`), so it never meets a
        migration's DDL halfway. ``TRUNCATE`` is not transactional, so it is refused inside a
        request's bound transaction (:func:`bound_request_connection`) rather than run beside it.

        :param table: the table, named as the store's other statements name it
        :ptype table: str
        :param reason: ``"reset"`` for an operator's reset; on the rail the request names it for the broker
        :ptype reason: str | None
        :return: nothing
        :rtype: None
        :raises ValueError: when a request-scoped connection is bound, or the name is not a plain
            identifier
        """
        if _request_l3_conn.get() is not None:
            raise ValueError(
                f"TRUNCATE of {table!r} runs alone, outside any transaction; a request's bound connection is open"
            )
        if isinstance(self._pool, TruncatingStore):
            await whole_read(
                self._pool.truncate(table, **({"reason": reason} if reason is not None else {})),
                kind="operator_reset",
                reason="an operator's reset empties the table with TRUNCATE; SchemaBackedCollection.reset is its one caller",
                ruling="Pace 2026-11-12: operator_reset declared in code; coordinator 2026-11-12: plain truncate() retired, reset() the one caller",
                proof="packages/core/tests/unit/collections/test_truncate.py::test_the_store_s_truncate_is_reached_only_from_an_operator_s_reset",
            )
            return
        name = as_written(_plain_identifier(table))
        reach = note_reach_text(reason=reason)
        with _unmask_asyncpg_release_race():
            async with self._pool.acquire() as conn, database_ddl_lock(ConnectionSession(conn)):
                # TRUNCATE runs alone, so its reach is the session's for the one statement, never left behind
                if reach:
                    await conn.execute("SELECT set_config($1, $2, false)", NOTE_REACH_SETTING, reach)
                try:
                    await whole_read(
                        conn.execute(f"TRUNCATE TABLE {name}"),
                        kind="operator_reset",
                        reason="an operator's reset empties the table with TRUNCATE; SchemaBackedCollection.reset is its one caller",
                        ruling="Pace 2026-11-12: operator_reset declared in code; coordinator 2026-11-12: plain truncate() retired, reset() the one caller",
                        proof="packages/core/tests/unit/collections/test_truncate.py::test_the_store_s_truncate_is_reached_only_from_an_operator_s_reset",
                    )
                finally:
                    if reach:
                        await conn.execute("SELECT set_config($1, $2, false)", NOTE_REACH_SETTING, "")
        _logger.info("table truncated", extra={"extra_data": {"table": table}})

    async def analyze(self, table: str) -> None:
        """refresh ``table``'s planner statistics: ``ANALYZE`` on a direct connection; nothing on the L3 rail,
        whose broker analyses a loaded table itself once its loads settle (and refuses the statement from a pod).

        :param table: the table
        :ptype table: str
        :return: nothing
        :rtype: None
        """
        if getattr(type(self._pool), "accepts_scoped_reads", False):
            return
        await whole_read(
            self.execute(f"ANALYZE {_quote_ident(table)}"),
            kind="cold_start",
            reason="a table a reload emptied and filled again is analysed once, so its first reads plan by its indexes",
            ruling="Pace 2026-11-12: a true cold start's fill reads whole; Pace 2026-11-15: a reload's one ANALYZE is part of its cold start",
            proof="packages/core/tests/unit/collections/test_load_files.py::test_the_store_s_analyze_is_reached_only_from_a_reload",
            speed="packages/core/tests/integration/test_reload_analyzed_live.py::test_a_reload_s_analyze_speed_at_real_size",
        )

    async def load_files(
        self,
        table: str,
        files: Sequence[LoadFile],
        *,
        columns: Sequence[str],
        constants: Mapping[str, Any],
        rows_per_transaction: int,
        timeout_ms: int,
        key_prefixes: Sequence[Mapping[str, Any]] | None = None,
    ) -> LoadResult:
        """Insert every row of parquet ``files`` into ``table`` with ``COPY``, committing as it goes. On a direct pool
        the collection records the epoch itself, ``key_prefixes`` with it, and each transaction's whole-table change
        note names them too (:data:`~threetears.core.ddl.change_notes.NOTE_REACH_SETTING`), so a log restored from
        the notes puts the load back as its prefixes. On the rail the request names them for the broker.

        The load's columns are checked against the table's registered schema first, on either path,
        so a caller's local run refuses what the rail would. On the rail the broker then does it
        (``NatsProxyL3Backend.load_files``, the ``l3.load_files`` request), fetching each file itself
        and admitting only an ``https`` URL of the caller's own export. On a direct pool -- a local
        run, a test -- it runs here the way the broker runs it: each URL admitted
        (:func:`_check_load_url`: ``https``, ``http`` on this machine, or ``file``), each file
        fetched (:func:`_fetch_load_file`) and read (pyarrow,
        off the loop), held to the row count its :class:`LoadFile` declares, its rows given the
        ``constants`` and ordered by primary key, and copied in transactions of
        ``rows_per_transaction`` rows (``copy_records_to_table``), each bounded by ``timeout_ms``.
        Each transaction is marked whole (:data:`~threetears.core.ddl.change_notes.NOTE_WHOLE_SETTING`), so
        a noted table's trigger writes one whole-table change note for it rather than one per row.
        Values are written as an insert binds them (:func:`schema_sql.normalize_write_value`).
        Insert-only: a key already held fails that transaction, and the load stops there; the
        transactions before it stay committed. The load's own checks run before anything is
        written, and each file's before any of its rows are.

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param files: the parquet files, in order
        :ptype files: Sequence[LoadFile]
        :param columns: the table's columns each file supplies
        :ptype columns: Sequence[str]
        :param constants: column -> the value every row takes for it
        :ptype constants: Mapping[str, Any]
        :param rows_per_transaction: the most rows one committed transaction writes; at least one
        :ptype rows_per_transaction: int
        :param timeout_ms: the most time one transaction may take, in milliseconds
        :ptype timeout_ms: int
        :param key_prefixes: the key prefixes the load's rows fall in, each the leading key columns and their values
        :ptype key_prefixes: Sequence[Mapping[str, Any]] | None
        :return: the rows written, every file's and each file's
        :rtype: LoadResult
        :raises ValueError: when no schema is registered for ``table``, no column is named, a column
            is named twice, is not the table's or is one a copy cannot write, the key is not wholly supplied,
            ``rows_per_transaction`` or ``timeout_ms`` is under one, a request's bound connection is
            open, or a file lacks a column or holds a different number of rows than declared
        :raises asyncpg.UniqueViolationError: when a row names a key the table already holds
        """
        if _request_l3_conn.get() is not None:
            raise ValueError(f"a load of {table!r} commits its own transactions; a request's bound connection is open")
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"load_files needs the schema of {table!r} registered")
        if rows_per_transaction < 1:
            raise ValueError(f"{table}: rows_per_transaction must be at least 1, got {rows_per_transaction}")
        if timeout_ms < 1:
            raise ValueError(f"{table}: timeout_ms must be at least 1, got {timeout_ms}")
        if not columns:
            raise ValueError(f"{table}: a load names the columns its files supply; none were named")
        names = [*columns, *constants]
        if len(set(names)) != len(names):
            raise ValueError(f"{table}: a column is named twice among the files' columns and the constants: {names}")
        declared = {column.name: column for column in schema.columns}
        unknown = [name for name in names if name not in declared]
        if unknown:
            raise ValueError(f"{table} has no column {unknown[0]!r}")
        unkeyed = [name for name in schema.pk_columns if name not in names]
        if unkeyed:
            raise ValueError(
                f"{table}: a load must supply every key column; {unkeyed[0]!r} is in no file and no constant"
            )
        if schema.row_epoch is not None and EPOCH_COLUMN not in columns and constants.get(EPOCH_COLUMN) is None:
            raise _unstamped(table)
        if isinstance(self._pool, LoadingStore):
            # the checks above are the schema's, so both paths refuse alike; what follows is this
            # path's own: the broker copies parquet with its own codecs (so a column a binary copy
            # of bound values cannot write is for it to judge) and admits a URL by its own rule
            return await self._pool.load_files(
                table,
                files,
                columns=columns,
                constants=constants,
                rows_per_transaction=rows_per_transaction,
                timeout_ms=timeout_ms,
                **({"key_prefixes": key_prefixes} if key_prefixes else {}),
            )
        unwritable = [declared[name] for name in names if not schema_sql.copy_writes(declared[name])]
        if unwritable:
            raise ValueError(
                f"{table}: a copy cannot write the {unwritable[0].column_type} column {unwritable[0].name!r}"
            )
        for file in files:
            _check_load_url(file.url)
        fixed = [schema_sql.normalize_write_value(declared[name], value) for name, value in constants.items()]
        key_positions = [names.index(name) for name in schema.pk_columns]
        reach = note_reach_text(prefixes=key_prefixes)
        loaded: list[FileLoad] = []
        for index, file in enumerate(files):
            data = await _fetch_load_file(file.url)
            try:
                held, values = await asyncio.to_thread(_parquet_columns, data, list(columns))
            except ValueError as exc:
                raise ValueError(f"{table}: file {index}: {exc}") from exc
            if held != file.rows:
                raise ValueError(f"{table}: file {index} holds {held} rows; the load declares {file.rows}")
            records = [
                (
                    *(schema_sql.normalize_write_value(declared[name], value) for name, value in zip(columns, row)),
                    *fixed,
                )
                for row in zip(*values)
            ]
            records.sort(key=lambda record: tuple(record[position] for position in key_positions))
            for start in range(0, len(records), rows_per_transaction):
                batch = records[start : start + rows_per_transaction]
                with _unmask_asyncpg_release_race():
                    async with asyncio.timeout(timeout_ms / 1000), self._pool.acquire() as conn, conn.transaction():
                        # the load's epoch names the whole table, so its change note does too: one per
                        # transaction, never one per copied row (threetears.core.collections.change_notes)
                        if reach:
                            # and names the load's prefixes, as its epoch does
                            await conn.execute(
                                "SELECT set_config($1, '1', true), set_config($2, $3, true)",
                                NOTE_WHOLE_SETTING,
                                NOTE_REACH_SETTING,
                                reach,
                            )
                        else:
                            await conn.execute("SELECT set_config($1, '1', true)", NOTE_WHOLE_SETTING)
                        await conn.copy_records_to_table(table, records=batch, columns=names)
            loaded.append(FileLoad(index=index, rows_loaded=len(records)))
        result = LoadResult(row_count=sum(file.rows_loaded for file in loaded), files=tuple(loaded))
        _logger.info(
            "table loaded from files",
            extra={"extra_data": {"table": table, "files": len(loaded), "rows": result.row_count}},
        )
        return result

    async def fetch_page(self, query: str, *params: Any) -> L3Page:
        """a keyset page's SELECT through the transport that may cut it: its rows, whether it was cut, its bytes.

        On the rail, its own :meth:`NatsProxyL3Backend.fetch_page`; on a bound or direct connection, which cuts
        nothing, :meth:`fetch`'s rows, never cut, at no stated size.

        :param query: the SELECT
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :return: the page
        :rtype: L3Page
        """
        target, _ = self._target()
        # looked up on the class: a mock pool's instance answers any name it is asked
        paged = getattr(target, "fetch_page") if target is self._pool and hasattr(type(target), "fetch_page") else None
        if paged is None:
            return L3Page(rows=await self.fetch(query, *params), truncated=False, reply_bytes=0)
        page: L3Page = await paged(query, *params)
        return page

    async def fetch_arrow_page(self, query: str, *params: Any, types: Mapping[str, Any]) -> L3Page:
        """:meth:`fetch_page`, its rows one Arrow table (:func:`~threetears.core.backends.arrow_reply.l3_fetch_arrow`).

        :param query: the SELECT
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :param types: column name -> its Arrow type, every column selected, in order
        :ptype types: Mapping[str, pyarrow.DataType]
        :return: the page, its rows a ``pyarrow.Table``
        :rtype: L3Page
        """
        target, _ = self._target()
        paged = (
            getattr(target, "fetch_arrow_page")
            if target is self._pool and hasattr(type(target), "fetch_arrow_page")
            else None
        )
        if paged is None:
            rows = await l3_fetch_arrow(self, query, *params, types=types)
            return L3Page(rows=rows, truncated=False, reply_bytes=0)
        page: L3Page = await paged(query, *params, types=types)
        return page

    @property
    def rows_per_statement(self) -> int | None:
        """the most rows the wrapped transport answers one statement, or None when it never cuts.

        The transport says so as its own ``rows_per_statement`` (``NatsProxyL3Backend``: the rail's
        :data:`~threetears.core.backends.protocol.L3_RAIL_ROW_CAP`). One that says nothing, or
        says it in a form that is not a count (a mock's auto-attribute), is taken to be the rail:
        a needless split at worst, never a row lost to a cut nobody looked for.

        :return: the cap, or None
        :rtype: int | None
        """
        stated = getattr(self._pool, "rows_per_statement", L3_RAIL_ROW_CAP)
        if stated is None:
            return None
        if isinstance(stated, int) and not isinstance(stated, bool):
            return stated
        return L3_RAIL_ROW_CAP

    async def delete_many(
        self,
        table: str,
        keys: Sequence[Sequence[Any]],
        *,
        max_rows: int,
        conn: Any = None,
    ) -> int:
        """Delete the rows of a schema-registered table named by ``keys``, every statement led by the key.

        Two key-led forms, whichever needs fewer statements (the first on a tie):

        - the keys grouped by all but one key column, the varying one, chosen to make the fewest
          groups (the leading column on a tie), each group ``max_rows`` keys a statement as
          ``DELETE ... WHERE <varying> = ANY($1) AND <rest of the key> = ...``
          (:func:`~threetears.core.backends.schema_sql.build_key_led_delete_sql`): one statement for
          a generation retired, or one race's geographies;
        - whole keys, ``max_rows`` a statement, one array a key column
          (:func:`~threetears.core.backends.schema_sql.build_whole_key_delete_sql`): for keys that
          share no value, which grouped would take a statement each.

        Either way a statement names whole keys, so a hash-sharded table is looked up at each one
        and never read whole: at most ``ceil(len(keys) / max_rows)`` statements when every key
        column has an array form. On ``conn`` a failing statement fails the caller's transaction,
        so every batch rolls back with it.

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param keys: each row's key values, in the schema's key order
        :ptype keys: Sequence[Sequence[Any]]
        :param max_rows: the most keys one statement names; at least one
        :ptype max_rows: int
        :param conn: the caller's transaction handle; ``None`` uses the pool
        :ptype conn: Any
        :return: keys named
        :rtype: int
        :raises ValueError: when no schema is registered for ``table``, ``max_rows`` is under one,
            a key is not as wide as the table's key, or no key column has an array form
        """
        _, named, statements = await self._planned_delete(table, keys, max_rows=max_rows)
        for sql, args in statements:
            await self._execute(sql, *args, conn=conn)
        _logger.debug(
            "key-led delete",
            extra={"extra_data": {"table": table, "keys": len(named), "statements": len(statements)}},
        )
        return len(named)

    async def delete_many_returning(
        self,
        table: str,
        keys: Sequence[Sequence[Any]],
        *,
        max_rows: int,
        conn: Any = None,
    ) -> dict[tuple[Any, ...], dict[str, Any]]:
        """:meth:`delete_many`, each statement returning the rows it deleted (``DELETE ... RETURNING``).

        One statement a batch reads and deletes the rows, where :meth:`fetch_many` then :meth:`delete_many`
        took two: a 5,000-row delete read for its messages cost 2.7 times one keyed delete on YugabyteDB that
        way. The statements are :meth:`delete_many`'s, key-led, and the rows are read as :meth:`fetch_many`
        reads them.

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param keys: each row's key values, in the schema's key order
        :ptype keys: Sequence[Sequence[Any]]
        :param max_rows: the most keys one statement names; at least one
        :ptype max_rows: int
        :param conn: the caller's connection; ``None`` uses the backend's own
        :ptype conn: Any
        :return: each deleted row under its key as ``keys`` names it; a key no row held left out
        :rtype: dict[tuple[Any, ...], dict[str, Any]]
        :raises ValueError: when no schema is registered for ``table``, ``max_rows`` is under one, a key is
            not as wide as the table's key, or no key column has an array form
        """
        schema, named, statements = await self._planned_delete(table, keys, max_rows=max_rows)
        # each key as the caller named it under the key bound, and each statement's returned rows read, on a worker
        # thread taking turns with the loop past _STATEMENT_ON_THREAD_PARAMS values: on the loop a statement's 1,000
        # rows of 49 columns read in one stretch, 33,786 keys' rows held it 10 ms a statement (2026-11-16)
        loop = asyncio.get_running_loop() if len(named) * len(schema.pk_columns) > _STATEMENT_ON_THREAD_PARAMS else None
        callers = (
            await asyncio.to_thread(_callers_by_bound, named, keys, loop)
            if loop is not None
            else _callers_by_bound(named, keys)
        )
        found: dict[tuple[Any, ...], dict[str, Any]] = {}
        for sql, args in statements:
            rows = await self._fetch(schema_sql.returning_rows_sql(schema, sql), *args, conn=conn)
            if len(rows) * len(schema.columns) > _STATEMENT_ON_THREAD_PARAMS:
                await asyncio.to_thread(_returned_rows, schema, rows, callers, found, asyncio.get_running_loop())
            else:
                _returned_rows(schema, rows, callers, found)
        _logger.debug(
            "key-led delete returning its rows",
            extra={"extra_data": {"table": table, "keys": len(named), "statements": len(statements)}},
        )
        return found

    async def delete_prefixes(self, table: str, prefixes: Sequence[Mapping[str, Any]], *, conn: Any) -> int:
        """Delete every row of a schema-registered table whose key leads with one of ``prefixes``, on ``conn``'s
        transaction, in statements whose count is the prefixes' and never the rows': nothing is read back.

        The prefixes are grouped by their columns and by every value but the last column's, and each group is one
        key-led ``DELETE`` (:func:`~threetears.core.backends.schema_sql.build_prefix_delete_sql`), its last column an
        array when that column has an array form, else one statement a prefix. A state's gone runs, ``(state,
        run_id)``, are one statement however many rows they hold.

        The delete is noted as ONE whole-table change note of the table naming the prefixes (the note a load from
        files leaves, :mod:`threetears.core.ddl.change_notes`), never a note a row: the statements are marked whole
        before (:func:`~threetears.core.ddl.change_notes.prefix_write_mark_sql`) and the marks put back after
        (:func:`~threetears.core.ddl.change_notes.prefix_write_restore_sql`), so the transaction's other writes are
        noted as they would have been. Two statements more, whatever the rows.

        On the L3 rail a pod may not set those marks (the broker refuses ``set_config``), so the broker runs the
        delete itself: the connection's ``delete_prefixes`` (the ``l3.tx.delete_prefixes`` request), whose commit
        the broker then records by the prefixes.

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param prefixes: each prefix, the key's leading columns and their values
        :ptype prefixes: Sequence[Mapping[str, Any]]
        :param conn: the caller's connection, its transaction open
        :ptype conn: Any
        :return: the rows deleted
        :rtype: int
        :raises ValueError: when no schema is registered for ``table``, or a prefix is not a leading part of the key
        """
        if self._scope_aware:
            deleted = await conn.delete_prefixes(table, [dict(prefix) for prefix in prefixes])
            return int(deleted)
        statements = self._prefix_deletes(table, prefixes)
        if not statements:
            return 0
        reach = note_reach_text(prefixes=prefixes)
        prior = await self._fetchrow(
            prefix_write_mark_sql(), NOTE_WHOLE_SETTING, NOTE_REACH_SETTING, reach, table, conn=conn
        )
        assert prior is not None  # narrow: a SELECT of no FROM answers one row
        deleted = 0
        for sql, args in statements:
            deleted += parse_rowcount(await self._execute(sql, *args, conn=conn))
        await self._execute(
            prefix_write_restore_sql(),
            NOTE_WHOLE_SETTING,
            prior["whole"] or "",
            NOTE_REACH_SETTING,
            prior["reach"] or "",
            prior["guard"],
            prior["noted"] or "",
            conn=conn,
        )
        _logger.debug(
            "key-prefix delete",
            extra={
                "extra_data": {
                    "table": table,
                    "prefixes": len(prefixes),
                    "statements": len(statements),
                    "rows": deleted,
                }
            },
        )
        return deleted

    def _prefix_deletes(self, table: str, prefixes: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[Any]]]:
        """:meth:`delete_prefixes`' statements, each with its arguments: the prefixes grouped by their columns and
        every value but the last column's, each value bound with its column's write form.

        :param table: the table
        :ptype table: str
        :param prefixes: each prefix, the key's leading columns and their values
        :ptype prefixes: Sequence[Mapping[str, Any]]
        :return: each statement and its arguments, in the prefixes' first-seen order
        :rtype: list[tuple[str, list[Any]]]
        :raises ValueError: when no schema is registered, or a prefix is not a leading part of the key
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"delete_prefixes needs the schema of {table!r} registered")
        groups: dict[tuple[tuple[str, ...], tuple[Any, ...]], list[Any]] = {}
        for prefix in prefixes:
            columns = tuple(schema.pk_columns[: len(prefix)])
            if not prefix or set(columns) != set(prefix):
                raise ValueError(
                    f"{table}: {dict(prefix)} is not a key prefix (the leading columns of {list(schema.pk_columns)})"
                )
            values = [schema_sql.normalize_write_value(schema.column(name), prefix[name]) for name in columns]
            last = groups.setdefault((columns, tuple(values[:-1])), [])
            if values[-1] not in last:
                last.append(values[-1])
        statements: list[tuple[str, list[Any]]] = []
        for (columns, lead), lasts in groups.items():
            if schema_sql.key_array_type(schema, columns[-1]) is not None:
                sql = schema_sql.build_prefix_delete_sql(schema, columns, last_as_array=True)
                statements.append((sql, [*lead, lasts]))
            else:
                sql = schema_sql.build_prefix_delete_sql(schema, columns, last_as_array=False)
                statements += [(sql, [*lead, value]) for value in lasts]
        return statements

    async def _planned_delete(
        self, table: str, keys: Sequence[Sequence[Any]], *, max_rows: int
    ) -> tuple[TableSchema, list[tuple[Any, ...]], list[tuple[str, list[Any]]]]:
        """:meth:`_delete_plan`, on a worker thread taking turns with the loop for a delete binding more than
        :data:`_STATEMENT_ON_THREAD_PARAMS` key values: on the loop, 33,786 keys of seven columns, each value bound
        and the keys grouped by a JSON text a key a key column, held it 250 ms (2026-11-16).

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param keys: each row's key values, in the schema's key order
        :ptype keys: Sequence[Sequence[Any]]
        :param max_rows: the most keys one statement names; at least one
        :ptype max_rows: int
        :return: the schema, each key's bound values in ``keys``' order, and each statement with its arguments
        :rtype: tuple[TableSchema, list[tuple[Any, ...]], list[tuple[str, list[Any]]]]
        :raises ValueError: as :meth:`_keys_to_delete`, and when no key column has an array form
        """
        schema = self._schemas.get(table)
        if schema is not None and len(keys) * len(schema.pk_columns) > _STATEMENT_ON_THREAD_PARAMS:
            return await asyncio.to_thread(
                self._delete_plan, table, keys, max_rows=max_rows, loop=asyncio.get_running_loop()
            )
        return self._delete_plan(table, keys, max_rows=max_rows)

    def _delete_plan(
        self,
        table: str,
        keys: Sequence[Sequence[Any]],
        *,
        max_rows: int,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> tuple[TableSchema, list[tuple[Any, ...]], list[tuple[str, list[Any]]]]:
        """the delete's keys checked and bound (:meth:`_keys_to_delete`) and its key-led statements
        (:func:`_key_led_deletes`); pure, for a thread (:meth:`_planned_delete`).

        :param table: the table
        :ptype table: str
        :param keys: each row's key values
        :ptype keys: Sequence[Sequence[Any]]
        :param max_rows: the most keys one statement names
        :ptype max_rows: int
        :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`); None on the loop itself
        :ptype loop: asyncio.AbstractEventLoop | None
        :return: the schema, the bound keys, and the statements with their arguments
        :rtype: tuple[TableSchema, list[tuple[Any, ...]], list[tuple[str, list[Any]]]]
        """
        turns = LoopTurns(loop) if loop is not None else None
        schema, named = self._keys_to_delete(table, keys, max_rows=max_rows, turns=turns)
        return schema, named, _key_led_deletes(schema, named, max_rows=max_rows, turns=turns)

    def _keys_to_delete(
        self, table: str, keys: Sequence[Sequence[Any]], *, max_rows: int, turns: LoopTurns | None = None
    ) -> tuple[TableSchema, list[tuple[Any, ...]]]:
        """the table's schema and ``keys`` as a delete binds them, checked.

        :param table: the table
        :ptype table: str
        :param keys: each row's key values
        :ptype keys: Sequence[Sequence[Any]]
        :param max_rows: the most keys one statement names
        :ptype max_rows: int
        :param turns: the binding thread's turns with the caller's loop, taken a key at a time; None to take none
        :ptype turns: LoopTurns | None
        :return: the schema, and each key's bound values in ``keys``' order
        :rtype: tuple[TableSchema, list[tuple[Any, ...]]]
        :raises ValueError: when no schema is registered, ``max_rows`` is under one, or a key is not as wide
            as the table's key
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"delete_many needs the schema of {table!r} registered")
        if max_rows < 1:
            raise ValueError(f"{table}: max_rows must be at least 1, got {max_rows}")
        key_columns = [schema.column(name) for name in schema.pk_columns]
        named: list[tuple[Any, ...]] = []
        for key in keys if turns is None else stepped(keys, turns):
            if len(key) != len(key_columns):
                raise ValueError(
                    f"{table}: a key to delete has {len(key)} values where the table's key has {len(key_columns)}"
                )
            named.append(
                tuple(
                    schema_sql.normalize_write_value(column, value)
                    for column, value in zip(key_columns, key, strict=True)
                )
            )
        return schema, named

    async def fetch_many(
        self,
        table: str,
        keys: Sequence[Sequence[Any]],
        *,
        max_rows: int,
        conn: Any = None,
    ) -> dict[tuple[Any, ...], dict[str, Any]]:
        """the rows of a schema-registered table named by whole ``keys``, each read as :meth:`fetch_one` reads it.

        ``max_rows`` keys a statement (:func:`~threetears.core.backends.schema_sql.build_whole_key_select_sql`),
        led by the key: not a statement a key. A key no row holds is left out.

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param keys: each row's key values, in the schema's key order
        :ptype keys: Sequence[Sequence[Any]]
        :param max_rows: the most keys one statement names; at least one
        :ptype max_rows: int
        :param conn: the caller's transaction handle; ``None`` uses the pool
        :ptype conn: Any
        :return: each held row, under its key as ``keys`` names it
        :rtype: dict[tuple[Any, ...], dict[str, Any]]
        :raises ValueError: when no schema is registered for ``table``, ``max_rows`` is under one, a key is not
            as wide as the table's key, or a key column has no array form
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"fetch_many needs the schema of {table!r} registered")
        if max_rows < 1:
            raise ValueError(f"{table}: max_rows must be at least 1, got {max_rows}")
        key_columns = [schema.column(name) for name in schema.pk_columns]
        # each key as the statement binds it, and as the caller named it
        named: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        for key in keys:
            if len(key) != len(key_columns):
                raise ValueError(
                    f"{table}: a key to read has {len(key)} values where the table's key has {len(key_columns)}"
                )
            bound = tuple(
                schema_sql.normalize_write_value(column, value) for column, value in zip(key_columns, key, strict=True)
            )
            named[bound] = tuple(key)
        found: dict[tuple[Any, ...], dict[str, Any]] = {}
        sql = schema_sql.build_whole_key_select_sql(schema)
        bounds = list(named)
        for start in range(0, len(bounds), max_rows):
            batch = bounds[start : start + max_rows]
            for row in await self._fetch(sql, *(list(column) for column in zip(*batch, strict=True)), conn=conn):
                read = dict(row)
                bound = tuple(schema_sql.normalize_write_value(column, read[column.name]) for column in key_columns)
                if bound in named:
                    found[named[bound]] = schema_sql.coerce_row(schema, read)
        return found

    async def fetch_led_by(
        self,
        table: str,
        values: Sequence[Any],
        *,
        columns: Sequence[str],
        max_values: int,
        conn: Any = None,
    ) -> list[dict[str, Any]]:
        """Read ``columns`` of the rows whose leading key column is one of ``values``, every statement led by the key.

        ``max_values`` values a statement, ``WHERE <lead> = ANY($1)``. A transport that cuts answers
        (:attr:`rows_per_statement`) may cut one without saying so, so an answer that reaches the
        cap is read again in halves, and a single value that alone reaches it is paged by the rest
        of its key through :func:`~threetears.core.keyset.read_keyset_pages`
        (``WHERE <lead> = $1 AND (<rest>) > (...) ORDER BY <rest> LIMIT``), which is the key's own
        order inside one hash bucket. No statement reads or sorts the whole table. A read that had
        to split or page says so once at INFO, with its statement count: the sign that
        ``max_values`` is too many for the table.

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param values: the leading key column's values, each named once
        :ptype values: Sequence[Any]
        :param columns: the columns to read, the key's among them
        :ptype columns: Sequence[str]
        :param max_values: the most values one statement names; at least one
        :ptype max_values: int
        :param conn: the caller's connection; ``None`` uses the pool
        :ptype conn: Any
        :return: the rows, read-coerced
        :rtype: list[dict[str, Any]]
        :raises ValueError: when no schema is registered for ``table``, ``max_values`` is under
            one, or the transport's row cap is under two
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"fetch_led_by needs the schema of {table!r} registered")
        if max_values < 1:
            raise ValueError(f"{table}: max_values must be at least 1, got {max_values}")
        row_cap = self.rows_per_statement
        if row_cap is not None and row_cap < 2:
            raise ValueError(f"{table}: a row cap of {row_cap} leaves no room to tell a full answer from a cut one")
        lead = schema.column(schema.pk_columns[0])
        normalized = [schema_sql.normalize_write_value(lead, value) for value in values]
        sql = schema_sql.build_led_by_select_sql(schema, columns)
        read = _LedRead(
            schema=schema, sql=sql, columns=tuple(columns), row_cap=row_cap, reader=conn if conn is not None else self
        )
        rows: list[dict[str, Any]] = []
        for start in range(0, len(normalized), max_values):
            rows += await read.batch(normalized[start : start + max_values])
        if read.splits or read.paged_values:
            _logger.info(
                "key-led read split or paged to stay under the transport's row cap",
                extra={
                    "extra_data": {
                        "table": table,
                        "values": len(normalized),
                        "rows": len(rows),
                        "statements": read.statements,
                        "splits": read.splits,
                        "paged_values": read.paged_values,
                        "row_cap": row_cap,
                    }
                },
            )
        return [schema_sql.coerce_row(schema, row) for row in rows]

    async def _update_fenced(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Annotated[Sequence[str], DeclaredKey],
        mutable: Sequence[str],
        cas: datetime,
        conn: Any,
    ) -> int:
        """Update one existing row only while its ``date_updated`` still equals ``cas``.

        The generic (schema-less) half of the update-only rule: 0 rows when the row
        changed OR no longer exists, never an insert.

        :param table: table name
        :ptype table: str
        :param row: the row, pk columns included
        :ptype row: Mapping[str, Any]
        :param pk: the pk column names
        :ptype pk: Sequence[str]
        :param mutable: the non-pk columns to write
        :ptype mutable: Sequence[str]
        :param cas: the ``date_updated`` value the caller read
        :ptype cas: datetime
        :param conn: optional caller-supplied connection
        :ptype conn: Any
        :return: rows affected, 0 on a fence miss or a vanished row
        :rtype: int
        """
        params: list[Any] = [row[c] for c in mutable]
        set_sql = ", ".join(f"{_quote_ident(c)} = ${i + 1}" for i, c in enumerate(mutable))
        pk_where: list[str] = []
        for c in pk:
            params.append(row[c])
            pk_where.append(f"{_quote_ident(c)} = ${len(params)}")
        params.append(cas)
        pk_where.append(f"{_quote_ident('date_updated')} = ${len(params)}")
        sql = f"UPDATE {_quote_ident(table)} SET {set_sql} WHERE {' AND '.join(pk_where)}"
        return parse_rowcount(await self._execute(sql, *params, conn=conn))

    async def _upsert_schema(
        self,
        schema: TableSchema,
        data: dict[str, Any],
        *,
        cas: datetime | None,
        conn: Any,
    ) -> int:
        """Schema-aware upsert: byte-identical to the collection's old ``save_to_store``.

        Three generated shapes, selected by the schema and by whether the caller
        read a version (a non-``None`` ``cas``):

        * ``cas_column`` set, ``on_conflict='update'`` and a non-``None`` ``cas``
          -- the fenced ``UPDATE ... WHERE pk AND <cas> = $N``, on EVERY such
          schema, ``cas_null_safe`` included. a fence value means the caller read
          the row as existing, so the write is update-only: a row deleted since
          that read (a respondent erasure, a relocation retire) affects 0 rows and
          the caller gets ``ConcurrentModificationError``. the NULL-safe upsert
          used to serve this case too, and because a missing row reaches its
          INSERT with the fence never evaluated, a save racing a delete
          re-created the row the delete had just removed.
        * ``cas_null_safe=True`` and ``cas=None`` -- ONE statement:
          ``INSERT ... ON CONFLICT (pk) DO UPDATE SET ... WHERE t.<cas> IS NOT
          DISTINCT FROM $N``. ``cas=None`` is FENCE-ELIGIBLE here, which is the
          whole point: a derived (non-random) primary key means two concurrent
          FIRST writers compute the same id, and NULL-safe equality is what lets
          the loser affect 0 rows instead of silently overwriting the winner.
        * everything else -- the unfenced ``INSERT ... ON CONFLICT`` from
          ``build_insert_sql``. Unchanged, and still what ``cas=None`` selects on
          every schema that has not opted in.

        ``cas_null_safe`` is read through ``getattr`` because this module reads the
        schema duck-typed (see the ``schema_sql`` module docstring); a non-
        ``TableSchema`` descriptor without the attribute keeps the old behaviour.
        """
        cas_declared = schema.cas_column is not None and schema.on_conflict == "update"
        if cas_declared and cas is not None:
            sql = schema_sql.build_cas_update_sql(schema, data)
            params = schema_sql.build_cas_params(schema, data, cas)
        elif cas_declared and getattr(schema, "cas_null_safe", False):
            sql = schema_sql.build_cas_upsert_sql(schema, data)
            params = schema_sql.build_cas_upsert_params(schema, data, cas)
        else:
            sql = schema_sql.build_insert_sql(schema, data)
            params = schema_sql.build_insert_params(schema, data)
        return parse_rowcount(await self._execute(sql, *params, conn=conn))

    async def upsert_ordered(self, table: str, row: Mapping[str, Any], *, conn: Any = None) -> int:
        """Insert ``row``, or update the stored row only when its compare-and-swap order is older.

        :class:`~threetears.core.backends.protocol.OrderedDurableStore`'s one operation. The SQL
        is generated from the registered schema by
        :func:`~threetears.core.backends.schema_sql.build_ordered_upsert_sql`, so the table must
        have one: without it neither the key nor the mutable columns are known.

        :param table: the table, whose schema is registered
        :ptype table: str
        :param row: the row, ``l2_epoch`` / ``l2_revision`` included
        :ptype row: Mapping[str, Any]
        :param conn: optional caller-supplied connection (a transaction handle) the write binds to
        :ptype conn: Any
        :return: ``1`` when written, ``0`` when the stored order is newer or equal
        :rtype: int
        :raises ValueError: when no schema is registered for ``table``
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"upsert_ordered on {table!r} needs its schema registered; none is")
        data = dict(row)
        sql = schema_sql.build_ordered_upsert_sql(schema, data)
        params = schema_sql.build_insert_params(schema, data)
        return parse_rowcount(await self._execute(sql, *params, conn=conn))

    async def delete(self, table: str, pk: KeyValues, *, conn: Any = None) -> None:
        """Delete the row whose primary key equals ``pk`` via a generated DELETE (missing is not an error).

        When a schema is registered for ``table`` the pk values are write-coerced and the
        DELETE projects the declared composite pk — byte-identical to the collection's old
        ``delete_from_store``. Otherwise a generic DELETE keyed on the ``pk`` mapping runs.
        """
        schema = self._schemas.get(table)
        if schema is None:
            _quote_ident(table)  # a name holding a quote is refused, as it always was
            await delete_keyed(self if conn is None else conn, table, by=dict(pk))
            return
        coerced: list[Any] = [
            schema_sql.normalize_write_value(schema.column(name), pk[name]) for name in schema.pk_columns
        ]
        await self._execute(schema_sql.build_delete_sql(schema), *coerced, conn=conn)

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return rows matching the equality ``filters`` (all rows when empty), read in pages of a declared key.

        The table's schema must be registered (:meth:`register_schema`; a collection registers its own on
        construction): the pages are in the order of the declared index the filters lead
        (:meth:`TableSchema.keyset_order <threetears.core.collections.schema_backed.TableSchema.keyset_order>`), so
        each reads about its own rows, under the transport's row cap (:attr:`rows_per_statement`), so a scan of more
        rows than the L3 rail answers a statement reads every one. A transport that never cuts is paged at the rail's
        cap all the same: one page a statement is the same read, and never one gulp of the table.

        Why no unregistered form (2026-11-14, no-whole-reads drive): it was one ``SELECT *`` (with any ``WHERE``) over
        a table whose key the store did not know, so nothing bounded it; no caller in 3tears, the hub or reports scans
        an unregistered table through this store (the git stores answer :class:`DurableStoreCollection`'s scans).

        :raises ValueError: when ``table``'s schema is not registered, or a filter names a column it does not hold
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(
                f"SqlL3Backend.scan: {table!r} has no registered schema, so there is no key to read it by; "
                "register its TableSchema first (register_schema)"
            )
        unknown = [name for name in filters or {} if schema.get_column(name) is None]
        if unknown:
            raise ValueError(f"SqlL3Backend.scan: {table!r} has no column {unknown[0]!r} to filter on")
        cap = self.rows_per_statement
        order = schema.keyset_order(filters or {})
        read = await read_keyset_pages(
            self,
            f"SELECT * FROM {_quote_ident(table)}",  # noqa: S608 - a registered schema's table
            list(order),
            where=dict(filters or {}),
            # a page asks one row past its size to see the next page (keyset.py), so it stays under the cap
            page_size=max((L3_RAIL_ROW_CAP if cap is None else cap) - 1, 1),
            lead_array=schema.hashed_lead_array(order),
        )
        return read.rows

    # ── connection-aware execution helpers ──────────────────────────────────────────
    async def _execute(self, query: str, *params: Any, conn: Any = None) -> str:
        """Run a write on ``conn`` when supplied (transactional override), else the pool."""
        if conn is not None:
            result = await conn.execute(query, *params)
            return result if isinstance(result, str) else ""
        return await self.execute(query, *params)

    async def _fetch(self, query: str, *params: Any, conn: Any = None) -> list[dict[str, Any]]:
        """Fetch rows on ``conn`` when supplied (transactional override), else the pool."""
        if conn is not None:
            return [dict(row) for row in await conn.fetch(query, *params)]
        return await self.fetch(query, *params)

    async def _fetchrow(self, query: str, *params: Any, conn: Any = None) -> dict[str, Any] | None:
        """Fetch one row on ``conn`` when supplied (transactional override), else the pool."""
        if conn is not None:
            row = await conn.fetchrow(query, *params)
            return dict(row) if row is not None else None
        return await self.fetchrow(query, *params)

    @staticmethod
    def _where(cols: Mapping[str, Any], *, start: int) -> tuple[str, list[Any]]:
        """Build a ``col = $n AND …`` clause + ordered params from a column→value mapping."""
        parts: list[str] = []
        params: list[Any] = []
        for i, (col, val) in enumerate(cols.items(), start=start):
            parts.append(f"{_quote_ident(col)} = ${i}")
            params.append(val)
        return " AND ".join(parts), params


class _PoolTransactionCM:
    """``acquire()`` + ``conn.transaction()`` for a bare pool without a ``transaction()`` helper."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._acquire_cm: Any = None
        self._conn: Any = None
        self._tx: Any = None

    async def __aenter__(self) -> Any:
        self._acquire_cm = self._pool.acquire()
        self._conn = await self._acquire_cm.__aenter__()
        self._tx = self._conn.transaction()
        await self._tx.__aenter__()
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        await self._tx.__aexit__(*exc)
        await self._acquire_cm.__aexit__(*exc)
