"""``SqliteL3Backend``: an L3 durable tier in one SQLite file, for a deployment of one process on one machine.

**What it is for.** An application that runs on one machine with no Postgres, no NATS and no hub -- aidc's local
mode, whose send queues, delivery ledgers, watches and dead letters must survive a restart -- keeps its rows in
3tears collections over this store, built with ``NO_L2``. The L1 in front of it is the ordinary SQLite cache
(:mod:`~threetears.core.cache.sqlite`), evictions and LRU included: an evicted row is read back from here, so no
eviction loses one. Moving such a deployment onto Postgres (and adding L2 on a mesh) is configuration: the same
collections over :class:`~threetears.core.backends.sql.SqlL3Backend`.

**The same statements as Postgres.** A schema-registered table's reads and writes are the statements
:mod:`~threetears.core.backends.schema_sql` generates for :class:`~threetears.core.backends.sql.SqlL3Backend` --
the conflict modes, the compare-and-swap ``UPDATE`` on ``cas_column``, the NULL-safe CAS upsert, the ordered
upsert -- with the Postgres spellings SQLite does not take rewritten: ``$N`` placeholders become ``?N`` and
``::type`` casts are dropped (SQLite 3.39 or later reads the rest, ``IS NOT DISTINCT FROM`` included). The rows
affected are SQLite's own count, so a lost CAS answers 0 exactly as it does on Postgres. The behaviour both stores
must share is held by one contract suite (``packages/core/tests/l3_contract.py``) run against each.

**The file.** One SQLite database at the path the application names, in write-ahead-log mode, every commit synced
(``synchronous=FULL``): a write that returned survives the process being killed. One process holds it, by an
exclusive ``flock`` on ``<path>.lock``; a second is refused (:class:`SqliteStoreLockedError`). The file carries
:data:`_SQLITE_STORE_FORMAT` as its ``user_version`` (how values are stored, below); a file of another format is
refused (:class:`SqliteStoreFormatError`).

**Tables.** :meth:`SqliteL3Backend.register_schema` (which a schema-backed collection calls when it is built)
creates its table when the file has none, with the declared columns, the primary key and each ``server_default``
the store can spell (``now()``, a quoted literal, a number, ``true`` / ``false``). A table already in the file must
hold the declared columns under the same storage types; a declared column it lacks is added when it is nullable or
has a default (an additive migration, the one kind applied here), and anything else -- a column dropped, a type
changed -- is refused with the difference named, rather than misreading stored rows. Indexes, foreign keys and
unique constraints beyond the primary key are not created.

**Values.** A UUID, a ``NUMERIC`` (its exact decimal text), a JSONB value (JSON text), a vector (its bracketed
text) and a ``TIMESTAMPTZ`` (its UTC instant to the microsecond, :func:`~threetears.core.cache.base.stored_instant`,
which sorts and compares as the instant) are stored as text under a declared type the reader converts back from;
a boolean as ``0`` / ``1``; ints and bytes as themselves. Reads answer the Python types the Postgres store answers.

**Raw SQL.** :class:`~threetears.core.backends.protocol.L3Backend`'s escape hatch (``fetch``, ``execute``,
``transaction`` ...) runs a statement with the same two rewrites. A statement written for Postgres alone
(``= ANY($1::uuid[])``, ``jsonb`` operators, ``COPY``, a Postgres function) fails as SQLite reports it; the
collection features built on such statements (change notes, scope epochs, delta files, Arrow copies, partition
loads) are not served by this store.

**Concurrency.** Writes take one lock, in this process, for their statement or for a whole transaction; reads go
to a second connection and see the last commit (WAL), never waiting for a writer. Each statement runs on a worker
thread, never on the event loop. A write outside an open transaction, from the task that holds it, is refused
(:class:`RuntimeError`) rather than left to wait on itself for ever.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import re
import sqlite3
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypeVar

from threetears.core.backends import schema_sql
from threetears.core.backends.schema_sql import json_default
from threetears.core.cache.base import stored_instant
from threetears.core.declared import whole_read
from threetears.core.keyset import read_keyset_pages
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.core.collections.schema_backed import Column, TableSchema

__all__ = [
    "SqliteL3Backend",
    "SqliteStoreFormatError",
    "SqliteStoreLockedError",
]

log = get_logger(__name__)

#: how this store keeps values (the storage types below and the text of each), as the file's ``user_version``. Raised
#: when that changes in a way a file written before would be misread under
_SQLITE_STORE_FORMAT: Final = 1

#: the oldest SQLite that reads every statement the Postgres store's generator writes (``IS NOT DISTINCT FROM``)
_MIN_SQLITE: Final = (3, 39, 0)

#: rows a scan reads a statement: a local file has no rail cap, and a page this size is a few milliseconds
_SCAN_PAGE_ROWS: Final = 1000

_T = TypeVar("_T")


class SqliteStoreLockedError(RuntimeError):
    """Another process holds the store's file: one process writes it."""


class SqliteStoreFormatError(RuntimeError):
    """The store's file, or a table in it, was written in a shape this code would misread."""


# ── values ─────────────────────────────────────────────────────────────────────────────────────────────────────────
# A column's declared type in the file: the first word names the reader's converter (sqlite3 PARSE_DECLTYPES), and
# the second gives SQLite the storage affinity, TEXT for every value kept as text so SQLite never turns '1.50' into
# a number. The words are this store's own, so no other code's converter answers for them.
_UUID_DECL: Final = "tt3_uuid TEXT"
_INSTANT_DECL: Final = "tt3_instant TEXT"
_JSON_DECL: Final = "tt3_json TEXT"
_DECIMAL_DECL: Final = "tt3_decimal TEXT"
_BOOL_DECL: Final = "tt3_bool INTEGER"
_VECTOR_DECL: Final = "tt3_vector TEXT"


def _decl_for(column: Column) -> str:
    """the declared type ``column`` is stored under.

    :param column: the column
    :ptype column: Column
    :return: its declaration
    :rtype: str
    """
    return {
        "uuid": _UUID_DECL,
        "datetimetz": _INSTANT_DECL,
        "jsonb": _JSON_DECL,
        "numeric": _DECIMAL_DECL,
        "bool": _BOOL_DECL,
        "vector": _VECTOR_DECL,
        "int": "INTEGER",
        "bigint": "INTEGER",
        "bytes": "BLOB",
    }.get(column.column_type, "TEXT")


def _bound(value: Any) -> Any:
    """a parameter as this store binds it: what SQLite takes, in the text the file keeps.

    :param value: the parameter, as the Postgres store's generator built it
    :ptype value: Any
    :return: the value SQLite binds
    :rtype: Any
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return stored_instant(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=json_default)
    if value is not None and type(value).__name__ == "UUID" and hasattr(value, "hex"):
        # uuid_utils' UUID, which is not a uuid.UUID
        return str(value)
    return value


def _instant(raw: bytes) -> datetime:
    return datetime.fromisoformat(raw.decode())


sqlite3.register_converter("tt3_uuid", lambda raw: uuid.UUID(raw.decode()))
sqlite3.register_converter("tt3_instant", _instant)
sqlite3.register_converter("tt3_json", lambda raw: json.loads(raw.decode()))
sqlite3.register_converter("tt3_decimal", lambda raw: Decimal(raw.decode()))
sqlite3.register_converter("tt3_bool", lambda raw: bool(int(raw)))
sqlite3.register_converter("tt3_vector", lambda raw: [float(v) for v in json.loads(raw.decode())])


# ── statements ─────────────────────────────────────────────────────────────────────────────────────────────────────
_PLACEHOLDER: Final = re.compile(r"\$(\d+)")
#: a Postgres cast (``::jsonb``, ``::text::public.vector``, ``::uuid[]``); SQLite keeps no such types
_CAST: Final = re.compile(r"::\s*[A-Za-z_][A-Za-z0-9_.]*(?:\s*\[\])?")
_VERB: Final = re.compile(r"^\s*(\w+)")
_NOW_TEXT: Final = "(strftime('%Y-%m-%dT%H:%M:%f', 'now') || '000+00:00')"
_LITERAL_DEFAULT: Final = re.compile(r"^\s*('(?:[^']|'')*'|-?\d+(?:\.\d+)?)\s*(?:::\s*[A-Za-z_][A-Za-z0-9_.]*)?\s*$")


def _sqlite_statement(sql: str) -> str:
    """``sql`` as SQLite reads it: ``$N`` placeholders as ``?N``, Postgres casts dropped.

    :param sql: a statement written for Postgres
    :ptype sql: str
    :return: the statement for SQLite
    :rtype: str
    """
    return _PLACEHOLDER.sub(r"?\1", _CAST.sub("", sql))


def _default_sql(column: Column) -> str | None:
    """the SQLite ``DEFAULT`` clause for ``column``'s ``server_default``, or ``None`` when it declares none.

    :param column: the column
    :ptype column: Column
    :return: the expression, or ``None``
    :rtype: str | None
    :raises SqliteStoreFormatError: when the default is one this store cannot spell
    """
    expr = column.server_default
    if expr is None:
        return None
    lowered = expr.strip().lower()
    if lowered in ("now()", "current_timestamp", "clock_timestamp()", "transaction_timestamp()"):
        return _NOW_TEXT
    if lowered in ("true", "false"):
        return "1" if lowered == "true" else "0"
    matched = _LITERAL_DEFAULT.match(expr)
    if matched is not None:
        return matched.group(1)
    raise SqliteStoreFormatError(
        f"column {column.name!r}: the SQLite store cannot spell its server_default {expr!r}; supply the value",
    )


def _tag(sql: str, rows: int) -> str:
    """the Postgres command tag for ``rows`` rows of ``sql`` (``"UPDATE 1"``, ``"INSERT 0 1"``).

    :param sql: the statement
    :ptype sql: str
    :param rows: the rows it affected
    :ptype rows: int
    :return: the tag
    :rtype: str
    """
    matched = _VERB.match(sql)
    verb = matched.group(1).upper() if matched is not None else "EXECUTE"
    return f"INSERT 0 {rows}" if verb == "INSERT" else f"{verb} {max(rows, 0)}"


class _SqliteStoreConnection:
    """the store's writing connection, as a transaction or an acquired connection hands it out.

    Passed as ``conn=`` to the store's structured operations, it runs them inside its transaction; it also takes
    raw statements, as an asyncpg connection does. Its life is the ``async with`` that made it.

    :param store: the store
    :ptype store: SqliteL3Backend
    """

    def __init__(self, store: SqliteL3Backend) -> None:
        self._store = store

    @property
    def store(self) -> SqliteL3Backend:
        """the store this connection writes.

        :return: it
        :rtype: SqliteL3Backend
        """
        return self._store

    async def fetch(self, query: str, *params: Any, **_scope: Any) -> list[dict[str, Any]]:
        """run a SELECT on this connection; every row as a dict."""
        return await self._store.run_on_writer(lambda conn: _rows(conn, query, params))

    async def fetchrow(self, query: str, *params: Any, **_scope: Any) -> dict[str, Any] | None:
        """run a SELECT on this connection; its first row, or ``None``."""
        rows = await self.fetch(query, *params)
        return rows[0] if rows else None

    async def fetchval(self, query: str, *params: Any, **_scope: Any) -> Any:
        """run a SELECT on this connection; the first column of its first row, or ``None``."""
        row = await self.fetchrow(query, *params)
        return next(iter(row.values())) if row else None

    async def execute(self, query: str, *params: Any, **_scope: Any) -> str:
        """run a statement on this connection; its Postgres command tag."""
        rows = await self._store.run_on_writer(lambda conn: _changed(conn, query, params))
        return _tag(query, rows)

    def transaction(self, **_options: Any) -> contextlib.AbstractAsyncContextManager[_SqliteStoreConnection]:
        """a transaction on this connection (a savepoint inside one already open), as asyncpg's ``transaction()``.

        asyncpg's options (``isolation=`` ...) are taken and need nothing here: SQLite's one writer at a time makes
        every transaction serializable.
        """
        return self._store.savepoint(self)


def _rows(conn: sqlite3.Connection, query: str, params: Sequence[Any]) -> list[dict[str, Any]]:
    cursor = conn.execute(_sqlite_statement(query), [_bound(p) for p in params])
    names = [d[0] for d in cursor.description or ()]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _changed(conn: sqlite3.Connection, query: str, params: Sequence[Any]) -> int:
    return conn.execute(_sqlite_statement(query), [_bound(p) for p in params]).rowcount


class SqliteL3Backend:
    """An L3 :class:`~threetears.core.backends.protocol.DurableStore` (and raw-SQL ``L3Backend``) in one SQLite file.

    Built with its path; nothing is opened until :meth:`open` (or ``async with``). Hand it to a registry as its L3
    (``registry.configure(l3_pool=store)``) and build the collections with ``nats_client=NO_L2``::

        store = SqliteL3Backend(state_dir / "state.sqlite")
        store.open()
        registry.configure(l1_backend=SQLiteBackend("aidc"), l3_pool=store)
        queue = collection_for_schema(QUEUE_SCHEMA)(registry, config, NO_L2)

    :param path: the database file; its directory is made when missing
    :ptype path: Path
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._schemas: Mapping[str, TableSchema] = MappingProxyType({})
        self._lock_fd: int | None = None
        self._writer: sqlite3.Connection | None = None
        self._reader: sqlite3.Connection | None = None
        #: one write at a time, from this process; held for a whole transaction
        self._write_lock = asyncio.Lock()
        #: the reading connection is used by one thread at a time
        self._read_lock = threading.Lock()
        #: the task holding an open transaction, so a write it makes outside that transaction is refused
        self._transaction_task: asyncio.Task[Any] | None = None
        self._savepoints = 0

    @property
    def path(self) -> Path:
        """the database file.

        :return: it
        :rtype: Path
        """
        return self._path

    @property
    def is_open(self) -> bool:
        """whether :meth:`open` has run and :meth:`close` has not.

        :return: whether it is open
        :rtype: bool
        """
        return self._writer is not None

    # ── lifecycle ──────────────────────────────────────────────────────────────────────────────────────────────────
    def open(self) -> None:
        """hold the file, open it in WAL mode with every commit synced, and check its format.

        :return: nothing
        :rtype: None
        :raises SqliteStoreLockedError: when another process holds the file
        :raises SqliteStoreFormatError: when the file was written under another format
        :raises RuntimeError: when this Python's SQLite is older than 3.39, or the store is already open
        """
        if self._writer is not None:
            raise RuntimeError(f"the SQLite store {self._path} is already open")
        if sqlite3.sqlite_version_info < _MIN_SQLITE:
            raise RuntimeError(
                f"the SQLite store needs SQLite {'.'.join(map(str, _MIN_SQLITE))} or later; this Python has "
                f"{sqlite3.sqlite_version}",
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(f"{self._path}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise SqliteStoreLockedError(
                f"the SQLite store {self._path} is held by another process; one process writes it",
            ) from exc
        writer: sqlite3.Connection | None = None
        try:
            writer = self._connect()
            writer.execute("PRAGMA journal_mode=WAL")
            self._check_format(writer)
            reader = self._connect()
        except BaseException:  # prawduct:allow prawduct/broad-except -- gives the file back, then re-raises unchanged
            if writer is not None:
                writer.close()
            os.close(fd)
            raise
        self._lock_fd, self._writer, self._reader = fd, writer, reader
        log.info("SQLite L3 store opened", extra={"extra_data": {"path": str(self._path)}})

    def close(self) -> None:
        """close both connections and give the file back; the file stays.

        :return: nothing
        :rtype: None
        """
        for conn in (self._reader, self._writer):
            if conn is not None:
                conn.close()
        self._reader = self._writer = None
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    async def __aenter__(self) -> SqliteL3Backend:
        self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()

    async def answers_from_yugabyte(self) -> bool:
        """``False``: the statements a keyset read spells per database are spelled for Postgres, which SQLite reads
        (:func:`~threetears.core.backends.dialect.known_on_yugabyte` asks this rather than send ``version()``).

        :return: ``False``
        :rtype: bool
        """
        return False

    async def health(self) -> None:
        """answer when the file can be read, raise when it cannot: a readiness probe's question.

        :return: nothing
        :rtype: None
        :raises RuntimeError: when the store is not open
        :raises sqlite3.Error: when the file cannot be read
        """
        await self._read(lambda conn: conn.execute("SELECT 1").fetchone())

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,
            detect_types=sqlite3.PARSE_DECLTYPES,
            cached_statements=256,
        )
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _check_format(self, conn: sqlite3.Connection) -> None:
        found = conn.execute("PRAGMA user_version").fetchone()[0]
        if found == _SQLITE_STORE_FORMAT:
            return
        tables = conn.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
        if found == 0 and tables == 0:
            conn.execute(f"PRAGMA user_version = {_SQLITE_STORE_FORMAT}")
            return
        raise SqliteStoreFormatError(
            f"the SQLite store {self._path} was written under format {found}; this code reads format "
            f"{_SQLITE_STORE_FORMAT}. Migrate the file, or move it aside to start empty",
        )

    # ── running statements ─────────────────────────────────────────────────────────────────────────────────────────
    async def _read(self, read: Callable[[sqlite3.Connection], _T]) -> _T:
        reader = self._reader
        if reader is None:
            raise RuntimeError(f"the SQLite store {self._path} is not open")

        def run() -> _T:
            with self._read_lock:
                return read(reader)

        return await asyncio.to_thread(run)

    async def run_on_writer(self, work: Callable[[sqlite3.Connection], _T]) -> _T:
        """run ``work`` on the writing connection, for a caller already holding the write lock (a transaction).

        :param work: what to run
        :ptype work: Callable[[sqlite3.Connection], _T]
        :return: what it answers
        :rtype: _T
        """
        writer = self._writer
        if writer is None:
            raise RuntimeError(f"the SQLite store {self._path} is not open")
        return await asyncio.to_thread(work, writer)

    async def _write(self, work: Callable[[sqlite3.Connection], _T], conn: Any) -> _T:
        """run ``work`` in its own transaction, or in the caller's when ``conn`` is one of this store's.

        :param work: the statements
        :ptype work: Callable[[sqlite3.Connection], _T]
        :param conn: a :class:`_SqliteStoreConnection` from :meth:`transaction` or :meth:`acquire`, or ``None``
        :ptype conn: Any
        :return: what ``work`` answers
        :rtype: _T
        :raises RuntimeError: when ``conn`` is another store's, or this task holds a transaction and passed none
        """
        if conn is not None:
            if not isinstance(conn, _SqliteStoreConnection) or conn.store is not self:
                raise RuntimeError(f"a connection of another store was passed to the SQLite store {self._path}")
            return await self.run_on_writer(work)
        if self._transaction_task is not None and self._transaction_task is asyncio.current_task():
            raise RuntimeError(
                f"a write to the SQLite store {self._path} outside the transaction this task holds would wait for "
                "that transaction for ever: pass conn= to write inside it",
            )
        async with self._write_lock:

            def in_transaction(writer: sqlite3.Connection) -> _T:
                writer.execute("BEGIN IMMEDIATE")
                try:
                    result = work(writer)
                    writer.execute("COMMIT")
                except BaseException:  # prawduct:allow prawduct/broad-except -- rolls back, then re-raises unchanged
                    if writer.in_transaction:
                        writer.execute("ROLLBACK")
                    raise
                return result

            return await self.run_on_writer(in_transaction)

    async def _fetch_on(self, conn: Any, query: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        if conn is not None:
            if not isinstance(conn, _SqliteStoreConnection) or conn.store is not self:
                raise RuntimeError(f"a connection of another store was passed to the SQLite store {self._path}")
            return await self.run_on_writer(lambda writer: _rows(writer, query, params))
        return await self._read(lambda reader: _rows(reader, query, params))

    # ── transactions ───────────────────────────────────────────────────────────────────────────────────────────────
    @contextlib.asynccontextmanager
    async def _held(self, *, transaction: bool) -> AsyncIterator[_SqliteStoreConnection]:
        if self._transaction_task is not None and self._transaction_task is asyncio.current_task():
            raise RuntimeError(f"this task already holds the SQLite store {self._path}: nest with conn.transaction()")
        async with self._write_lock:
            self._transaction_task = asyncio.current_task()
            handle = _SqliteStoreConnection(self)
            try:
                if transaction:
                    async with self.savepoint(handle):
                        yield handle
                else:
                    yield handle
            finally:
                self._transaction_task = None

    def transaction(self, namespace: str | None = None) -> contextlib.AbstractAsyncContextManager[Any]:
        """a transaction over the store: every write given its connection commits together, or none does.

        :param namespace: unused (no schemas to route between); taken for the ``L3Backend`` shape
        :ptype namespace: str | None
        :return: an async context manager yielding a :class:`_SqliteStoreConnection`
        :rtype: contextlib.AbstractAsyncContextManager[Any]
        """
        return self._held(transaction=True)

    def acquire(self) -> contextlib.AbstractAsyncContextManager[Any]:
        """the writing connection, held for the block, each statement committing on its own unless the block opens
        ``conn.transaction()``, as an asyncpg pool's ``acquire()``.

        :return: an async context manager yielding a :class:`_SqliteStoreConnection`
        :rtype: contextlib.AbstractAsyncContextManager[Any]
        """
        return self._held(transaction=False)

    @contextlib.asynccontextmanager
    async def savepoint(self, handle: _SqliteStoreConnection) -> AsyncIterator[_SqliteStoreConnection]:
        """a transaction on ``handle`` (the outermost) or a savepoint inside it, committed when the block ends cleanly.

        :param handle: the held connection
        :ptype handle: _SqliteStoreConnection
        :return: the same connection
        :rtype: AsyncIterator[_SqliteStoreConnection]
        """
        depth = self._savepoints
        name = f"tt3_sp_{depth}"
        begin, commit, rollback = (
            ("BEGIN IMMEDIATE", "COMMIT", ("ROLLBACK",))
            if depth == 0
            else (f"SAVEPOINT {name}", f"RELEASE {name}", (f"ROLLBACK TO {name}", f"RELEASE {name}"))
        )
        await self.run_on_writer(lambda writer: writer.execute(begin))
        self._savepoints += 1
        try:
            yield handle
        except BaseException:  # prawduct:allow prawduct/broad-except -- rolls back, then re-raises unchanged
            self._savepoints -= 1
            await asyncio.shield(self.run_on_writer(lambda writer: _run_each(writer, rollback)))
            raise
        self._savepoints -= 1
        await self.run_on_writer(lambda writer: writer.execute(commit))

    # ── tables ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def register_schema(self, table: str, schema: TableSchema) -> None:
        """take ``schema`` as ``table``'s, creating the table or checking (and adding to) the one the file holds.

        A collection over this store calls it when it is built, as it does for the Postgres store.

        :param table: the table's name (``schema.name``)
        :ptype table: str
        :param schema: its schema
        :ptype schema: TableSchema
        :return: nothing
        :rtype: None
        :raises SqliteStoreFormatError: when the stored table cannot hold the declared columns, or a default cannot
            be spelled
        :raises ValueError: when the table's name is schema-qualified
        :raises RuntimeError: when the store is not open
        """
        if "." in table:
            raise ValueError(f"the SQLite store has no schemas: {table!r} is qualified")
        if self._writer is None:
            raise RuntimeError(f"the SQLite store {self._path} is not open; open() it before building collections")
        # a connection of its own: the writing one may be inside another task's transaction on a worker thread, and
        # a table change waits for that transaction (busy_timeout) rather than landing inside it
        ddl = self._connect()
        try:
            stored = {row[1]: row[2] for row in ddl.execute(f'PRAGMA table_info("{table}")').fetchall()}
            if not stored:
                ddl.execute(self._create_table_sql(table, schema))
            else:
                for statement in self._additions(table, schema, stored):
                    ddl.execute(statement)
        finally:
            ddl.close()
        self._schemas = MappingProxyType({**self._schemas, table: schema})

    def _create_table_sql(self, table: str, schema: TableSchema) -> str:
        parts = []
        for column in schema.columns:
            default = _default_sql(column)
            null = " NOT NULL" if column.name in schema.pk_columns else ""
            parts.append(
                f'"{column.name}" {_decl_for(column)}{null}' + ("" if default is None else f" DEFAULT {default}")
            )
        keys = ", ".join(f'"{name}"' for name in schema.pk_columns)
        parts.append(f"PRIMARY KEY ({keys})")
        return f'CREATE TABLE "{table}" ({", ".join(parts)})'

    def _additions(self, table: str, schema: TableSchema, stored: Mapping[str, str]) -> list[str]:
        declared = {column.name: column for column in schema.columns}
        dropped = sorted(set(stored) - set(declared))
        changed = sorted(
            name for name, column in declared.items() if name in stored and stored[name] != _decl_for(column)
        )
        added = [column for name, column in declared.items() if name not in stored]
        unaddable = [c.name for c in added if not (c.nullable or c.server_default is not None)]
        if dropped or changed or unaddable:
            raise SqliteStoreFormatError(
                f"the SQLite store {self._path} holds table {table!r} in another shape than declared: "
                f"columns it holds and the schema does not: {dropped}; columns whose type changed: {changed}; "
                f"new columns that are neither nullable nor defaulted: {unaddable}. Migrate the file",
            )
        statements = []
        for column in added:
            default = _default_sql(column)
            statements.append(
                f'ALTER TABLE "{table}" ADD COLUMN "{column.name}" {_decl_for(column)}'
                + ("" if default is None else f" DEFAULT {default}")
            )
            log.info(
                "SQLite L3 store: added a declared column",
                extra={"extra_data": {"path": str(self._path), "table": table, "column": column.name}},
            )
        return statements

    def _schema(self, table: str) -> TableSchema:
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"the SQLite store has no schema registered for {table!r} (register_schema)")
        return schema

    # ── DurableStore ───────────────────────────────────────────────────────────────────────────────────────────────
    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        """the row whose primary key equals ``pk``, read-coerced as the Postgres store answers it, or ``None``."""
        schema = self._schema(table)
        params = [schema_sql.normalize_write_value(schema.column(name), pk[name]) for name in schema.pk_columns]
        rows = await self._fetch_on(conn, schema_sql.build_fetch_sql(schema), params)
        return schema_sql.coerce_row(schema, rows[0]) if rows else None

    async def upsert(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Sequence[str] | None = None,
        on_conflict: str = "update",
        cas: datetime | None = None,
        conn: Any = None,
    ) -> int:
        """insert or update ``row`` by the schema's conflict mode and fence; rows affected (0 on a lost CAS).

        ``pk`` and ``on_conflict`` are the registered schema's, as on the Postgres store.
        """
        schema = self._schema(table)
        sql, params = _upsert_statement(schema, dict(row), cas)
        return await self._write(lambda writer: _changed(writer, sql, params), conn)

    async def upsert_ordered(self, table: str, row: Mapping[str, Any], *, conn: Any = None) -> int:
        """insert ``row``, or update the stored row only when its compare-and-swap order is older; rows affected."""
        schema = self._schema(table)
        data = dict(row)
        sql = schema_sql.build_ordered_upsert_sql(schema, data)
        params = schema_sql.build_insert_params(schema, data)
        return await self._write(lambda writer: _changed(writer, sql, params), conn)

    async def upsert_many(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        max_rows: int,
        max_bytes: int,
        conn: Any = None,
    ) -> int:
        """insert or update every row in one transaction (the caller's when ``conn``); rows written.

        ``max_rows`` and ``max_bytes`` bound a statement on the L3 rail; a local file has no such bound, and each
        row is its own statement inside the one transaction.
        """
        schema = self._schema(table)
        statements = [_upsert_statement(schema, dict(row), None) for row in rows]

        def write(writer: sqlite3.Connection) -> int:
            for sql, params in statements:
                _changed(writer, sql, params)
            return len(statements)

        return await self._write(write, conn) if statements else 0

    async def delete(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> None:
        """delete the row whose primary key equals ``pk``; a missing row is not an error."""
        schema = self._schema(table)
        params = [schema_sql.normalize_write_value(schema.column(name), pk[name]) for name in schema.pk_columns]
        sql = schema_sql.build_delete_sql(schema)
        await self._write(lambda writer: _changed(writer, sql, params), conn)

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """every row matching the equality ``filters`` (all rows when empty), read-coerced.

        Read in pages of the declared key the filters lead (:meth:`TableSchema.keyset_order`), as the Postgres
        store reads them, so a scan of a large table is many bounded statements, never one.

        :raises ValueError: when ``table``'s schema is not registered, or a filter names a column it does not hold
        """
        schema = self._schema(table)
        unknown = [name for name in filters or {} if schema.get_column(name) is None]
        if unknown:
            raise ValueError(f"SqliteL3Backend.scan: {table!r} has no column {unknown[0]!r} to filter on")
        order = schema.keyset_order(filters or {})
        read = await read_keyset_pages(
            self,
            f'SELECT {schema_sql.build_select_column_list(schema)} FROM "{table}"',  # noqa: S608 - registered names
            list(order),
            where=dict(filters or {}),
            page_size=_SCAN_PAGE_ROWS,
            lead_array=schema.hashed_lead_array(order),
        )
        return [schema_sql.coerce_row(schema, row) for row in read.rows]

    async def truncate(self, table: str, *, reason: str | None = None) -> None:
        """remove every row of ``table``; an empty table is not an error.

        :param table: the table
        :ptype table: str
        :param reason: what the Postgres store records with an operator's reset; nothing here records epochs
        :ptype reason: str | None
        :return: nothing
        :rtype: None
        """
        self._schema(table)
        await whole_read(
            self._write(lambda writer: writer.execute(f'DELETE FROM "{table}"').rowcount, None),  # noqa: S608
            kind="operator_reset",
            reason="an operator's reset empties the table; SchemaBackedCollection.reset is its one caller",
            ruling="Pace 2026-11-12: operator_reset declared in code; coordinator 2026-11-12: plain truncate() retired, reset() the one caller",
            proof="packages/core/tests/unit/collections/test_truncate.py::test_the_store_s_truncate_is_reached_only_from_an_operator_s_reset",
        )

    # ── L3Backend: raw SQL ─────────────────────────────────────────────────────────────────────────────────────────
    async def fetch(
        self, query: str, *params: Any, namespace: str | None = None, **_scope: Any
    ) -> list[dict[str, Any]]:
        """run a SELECT on the reading connection (the last commit); every row as a dict."""
        return await self._read(lambda reader: _rows(reader, query, params))

    async def fetchrow(
        self, query: str, *params: Any, namespace: str | None = None, **_scope: Any
    ) -> dict[str, Any] | None:
        """run a SELECT; its first row, or ``None``."""
        rows = await self.fetch(query, *params)
        return rows[0] if rows else None

    async def fetchval(self, query: str, *params: Any, namespace: str | None = None, **_scope: Any) -> Any:
        """run a SELECT; the first column of its first row, or ``None``."""
        row = await self.fetchrow(query, *params)
        return next(iter(row.values())) if row else None

    async def execute(self, query: str, *params: Any, namespace: str | None = None, **_scope: Any) -> str:
        """run a statement in its own transaction; its Postgres command tag (``"UPDATE 1"``)."""
        rows = await self._write(lambda writer: _changed(writer, query, params), None)
        return _tag(query, rows)

    async def execute_batch(
        self, queries: list[dict[str, Any]], *, namespace: str | None = None, transaction: bool = True
    ) -> list[Any]:
        """run ``{query, params}`` dicts, in one transaction when ``transaction``; their command tags."""

        def run(writer: sqlite3.Connection) -> list[Any]:
            return [_tag(q["query"], _changed(writer, q["query"], q.get("params", []))) for q in queries]

        if transaction:
            return await self._write(run, None)
        return [await self.execute(q["query"], *q.get("params", [])) for q in queries]


def _upsert_statement(schema: TableSchema, data: dict[str, Any], cas: datetime | None) -> tuple[str, list[Any]]:
    """the statement and parameters the Postgres store writes ``data`` with (``SqlL3Backend._upsert_schema``'s rule).

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param data: the row
    :ptype data: dict[str, Any]
    :param cas: the fence value, or ``None``
    :ptype cas: datetime | None
    :return: the statement and its parameters
    :rtype: tuple[str, list[Any]]
    """
    cas_declared = schema.cas_column is not None and schema.on_conflict == "update"
    if cas_declared and cas is not None:
        return schema_sql.build_cas_update_sql(schema, data), schema_sql.build_cas_params(schema, data, cas)
    if cas_declared and getattr(schema, "cas_null_safe", False):
        return schema_sql.build_cas_upsert_sql(schema, data), schema_sql.build_cas_upsert_params(schema, data, cas)
    return schema_sql.build_insert_sql(schema, data), schema_sql.build_insert_params(schema, data)


def _run_each(conn: sqlite3.Connection, statements: Sequence[str]) -> None:
    """run each statement in turn (never ``executescript``, which commits a transaction left open first).

    :param conn: the connection
    :ptype conn: sqlite3.Connection
    :param statements: the statements
    :ptype statements: Sequence[str]
    :return: nothing
    :rtype: None
    """
    for statement in statements:
        conn.execute(statement)
