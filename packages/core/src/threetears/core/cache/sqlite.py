"""SQLite L1 cache backend with thread-local connections and type-aware serialization.

Uses a named in-memory database (memdb VFS) so multiple connections can access
the same data. The anchor connection keeps the database alive. Schema is derived
from SQLAlchemy metadata, with type-aware serialization/deserialization for
UUID, datetime, date, time, JSON, boolean, vector, bytea and exact NUMERIC columns.

An exact ``NUMERIC(p, s)`` column is kept as an ``INTEGER``: its value scaled by ``10**s``
(``9.50`` in ``NUMERIC(12, 2)`` is ``950``), and read back as the ``Decimal`` at scale ``s`` that L3
answers (type code ``INTEGER_DECIMAL_<s>``). Being an integer it is exact, and compares, sorts, takes
``MIN`` / ``MAX`` and sums correctly in SQL run against L1 directly; a ``SUM`` is the scaled total,
exact, which :meth:`SQLiteBackend.deserialize_field` turns back into the ``Decimal``. A ``SUM`` holds
only while the scaled total stays within a 64-bit integer (``2**63 - 1``, so ``92233720368547.75807``
at scale 5); past it :meth:`SQLiteBackend.execute_query` raises ``OverflowError`` naming the column.
Compare such a column with an operand through :meth:`SQLiteBackend.numeric_comparison`, which bounds
the scaled column by the operand exactly, never rounded to the column's scale: ``price > 9.995``
against ``10.00`` is true, as in L3. :meth:`SQLiteBackend.serialize_value` rounds to the scale, as a
stored value is, and is not for operands. A ``NUMERIC`` wider than 18 digits
cannot fit a 64-bit integer, and one with no declared precision and scale has no scale to keep, so
either is refused when its table is created (:class:`~threetears.core.cache.base.UnsupportedL1ColumnError`).
SQLite's own ``NUMERIC`` and ``REAL`` are binary floats: money read back from them is a ``float``,
and ``Decimal * float`` raises.
"""

from __future__ import annotations

import enum
import json
import sqlite3
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time
from decimal import MAX_EMAX, MIN_EMIN, ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from types import MappingProxyType
from typing import Any

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, LargeBinary, String, Text, Time
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, TSVECTOR, UUID
from sqlalchemy.sql.sqltypes import ARRAY
from sqlalchemy.sql.sqltypes import UUID as UuidType  # noqa: N811

from threetears.core.backends.schema_sql import PGVECTOR_TYPES, json_default
from threetears.core.cache.base import (
    build_select_clause,
    bulk_columns,
    exact_decimal_type,
    stored_decimal,
    stored_time,
)
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import contain, get_logger

__all__ = [
    "SQLiteBackend",
]

try:
    from uuid_utils import UUID as _UuidUtilsUUID

    _UUID_TYPES: tuple[type, ...] = (uuid.UUID, _UuidUtilsUUID)
except ImportError:
    _UUID_TYPES = (uuid.UUID,)

try:
    from uuid_utils import UUID as UuidUtilsUUID

    #: uuid_utils' UUID, when the uuid-utils extra is installed; core works without it
    _UUID_UTILS_TYPES: tuple[type, ...] = (UuidUtilsUUID,)
except ImportError:  # NOSILENT: the uuid-utils extra is optional; its values never reach L1 without it
    _UUID_UTILS_TYPES = ()


log = get_logger(__name__)

#: the type code of an exact ``NUMERIC`` column, before its ``_<scale>`` suffix
_DECIMAL_CODE = "INTEGER_DECIMAL"
#: the widest exact ``NUMERIC`` whose scaled value always fits SQLite's 64-bit ``INTEGER``
_SQLITE_MAX_DECIMAL_PRECISION = 18
#: SQLite's largest ``INTEGER``; every stored scaled value is under ``10**18``, well inside it
_SQLITE_MAX_INTEGER = 2**63 - 1
#: what an operator in :meth:`SQLiteBackend.numeric_comparison` answers for an operand above
#: every finite value (``NaN``, which Postgres sorts above all, or ``Infinity``); below every one,
#: ``-Infinity``, it answers the opposite of each but ``=`` and ``!=``
_ABOVE_EVERY_VALUE: Mapping[str, bool] = MappingProxyType(
    {"=": False, "!=": True, "<>": True, "<": True, "<=": True, ">": False, ">=": False}
)
_BELOW_EVERY_VALUE: Mapping[str, bool] = MappingProxyType(
    {"=": False, "!=": True, "<>": True, "<": False, "<=": False, ">": True, ">=": True}
)


def _decimal_code_scale(col_type: str) -> int | None:
    """the scale an exact ``NUMERIC`` column's type code carries, or ``None`` for any other code.

    :param col_type: a type code (:meth:`SQLiteBackend.column_types`)
    :ptype col_type: str
    :return: the column's scale, or ``None`` when the code is not an exact ``NUMERIC``'s
    :rtype: int | None
    """
    prefix = f"{_DECIMAL_CODE}_"
    return int(col_type[len(prefix) :]) if col_type.startswith(prefix) else None


class _PooledConnection:
    """Proxy that prevents callers from closing thread-local connections.

    Delegates all attribute access to the underlying sqlite3.Connection
    but makes close() a no-op. The real connection is managed by
    SQLiteBackend and closed only during reset().
    """

    __slots__ = ("_conn",)

    def __init__(self, conn: sqlite3.Connection) -> None:
        object.__setattr__(self, "_conn", conn)

    def close(self) -> None:
        """No-op. Connection is pooled and reused across calls."""

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
        """Execute SQL statement."""
        cursor: sqlite3.Cursor = self._conn.execute(sql, parameters)
        return cursor

    def executemany(self, sql: str, parameters: Any) -> sqlite3.Cursor:
        """Execute SQL with multiple parameter sets."""
        cursor: sqlite3.Cursor = self._conn.executemany(sql, parameters)
        return cursor

    def commit(self) -> None:
        """Commit the current transaction."""
        self._conn.commit()

    @property
    def row_factory(self) -> Any:
        """Get the row factory."""
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        """Set the row factory."""
        self._conn.row_factory = value

    def __getattr__(self, name: str) -> Any:
        """Delegate remaining attribute access to the underlying connection."""
        return getattr(self._conn, name)


class SQLiteBackend:
    """L1 cache backend using SQLite named in-memory database.

    Instance-based (not classmethod-based) so multiple backends can coexist.
    Each instance manages its own named in-memory database, anchor connection,
    thread-local connections, and schema registry.
    """

    def __init__(self, db_name: str = "threetears_cache") -> None:
        self._db_name = db_name
        self._anchor_conn: sqlite3.Connection | None = None
        self._initialized: bool = False
        self._schema_info: dict[str, dict[str, str]] = {}
        self._local: threading.local = threading.local()
        self._pool_lock: threading.Lock = threading.Lock()
        self._pooled_connections: list[sqlite3.Connection] = []

    def _setup_connection(self, conn: sqlite3.Connection) -> None:
        """Configure connection with busy timeout."""
        conn.execute("PRAGMA busy_timeout=5000")

    def _make_connection(self) -> sqlite3.Connection:
        """Create new connection to the named in-memory database."""
        conn = sqlite3.connect(
            f"file:/{self._db_name}?vfs=memdb",
            uri=True,
            check_same_thread=False,
            cached_statements=0,
            isolation_level=None,
        )
        self._setup_connection(conn)
        return conn

    def initialize(self, sa_metadata: Any) -> None:
        """Initialize SQLite with schema from SQLAlchemy metadata.

        Creates a named in-memory database and initializes all tables.
        The anchor connection is kept open to prevent garbage collection.
        Builds a schema registry for type-aware serialization.

        Additive across calls: every call registers the tables in
        ``sa_metadata`` that this backend has not seen yet and leaves
        already-registered tables untouched, so several single-table
        initializations (the dynamic-collection path) compose on one
        shared backend exactly like a single all-tables metadata.

        :raises UnsupportedL1ColumnError: when a column is one this L1 cannot hold exactly
        """
        # a column this L1 cannot hold exactly refuses the whole call before anything is created
        for table in sa_metadata.tables.values():
            for column in table.columns:
                self._column_code(table.name, column)
        if self._anchor_conn is None:
            self._anchor_conn = self._make_connection()

        new_tables = 0
        for table in sa_metadata.tables.values():
            if table.name in self._schema_info:
                log.debug(f"SQLite table already registered, skipping: {table.name}")
                continue
            ddl = self._generate_create_table(table)
            self._anchor_conn.execute(ddl)
            self._schema_info[table.name] = {col.name: self._column_code(table.name, col) for col in table.columns}
            new_tables += 1
            log.debug(f"Created SQLite table: {table.name}")

        if not self._initialized:
            # Register type adapters so UUID/datetime values are automatically
            # serialized at the SQLite boundary.
            sqlite3.register_adapter(uuid.UUID, str)
            sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
            for uuid_type in _UUID_UTILS_TYPES:
                sqlite3.register_adapter(uuid_type, str)
            self._initialized = True

        log.debug(
            "SQLite L1 cache initialized",
            extra={"extra_data": {"table_count": len(self._schema_info), "new_tables": new_tables}},
        )

    def get_connection(self) -> _PooledConnection:
        """Get a thread-local connection to the named in-memory database.

        Returns a cached connection for the current thread, created once
        per thread and reused. The returned proxy makes close() a no-op.
        """
        if not self._initialized:
            raise RuntimeError("SQLite not initialized - call initialize() first")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._make_connection()
            self._local.conn = conn
            with self._pool_lock:
                self._pooled_connections.append(conn)
        return _PooledConnection(conn)

    @staticmethod
    def _pk_columns(primary_key: str | tuple[str, ...]) -> tuple[str, ...]:
        """normalize ``primary_key`` argument to a tuple of column names.

        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :return: tuple of pk column names (length 1 for single-PK)
        :rtype: tuple[str, ...]
        """
        if isinstance(primary_key, tuple):
            return primary_key
        return (primary_key,)

    @staticmethod
    def _pk_values(entity_id: Any, pk_cols: tuple[str, ...]) -> tuple[Any, ...]:
        """normalize ``entity_id`` argument to tuple of pk values.

        length MUST match ``pk_cols`` when composite; single-value
        inputs are wrapped in a 1-tuple for single-PK tables.

        :param entity_id: pk value or tuple of pk values
        :ptype entity_id: Any
        :param pk_cols: normalized tuple of pk column names
        :ptype pk_cols: tuple[str, ...]
        :return: tuple of pk values matching ``pk_cols`` length
        :rtype: tuple[Any, ...]
        :raises ValueError: if tuple length does not match ``pk_cols``
        """
        if isinstance(entity_id, tuple):
            values = entity_id
        else:
            values = (entity_id,)
        if len(values) != len(pk_cols):
            raise ValueError(
                f"primary key arity mismatch: got {len(values)} value(s) for {len(pk_cols)} column(s) {pk_cols}"
            )
        return values

    def _serialize_pk_values(
        self,
        table: str,
        pk_cols: tuple[str, ...],
        pk_vals: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """serialize pk values for SQLite parameter binding.

        mirrors the :meth:`upsert` write path, which routes every column
        value through :meth:`serialize_value`. the read/delete paths used
        to bind pk values raw and rely on sqlite3's adapter registry --
        but adapter lookup is exact-type, so asyncpg's
        ``pgproto.pgproto.UUID`` (a :class:`uuid.UUID` subclass returned
        for every L3 ``uuid`` column) raised ``ProgrammingError`` instead
        of matching the ``uuid.UUID`` adapter. serializing at this
        boundary keys on ``isinstance``, which covers subclasses.

        :param table: target table name (resolves the schema registry)
        :ptype table: str
        :param pk_cols: normalized tuple of pk column names
        :ptype pk_cols: tuple[str, ...]
        :param pk_vals: tuple of pk values matching ``pk_cols``
        :ptype pk_vals: tuple[Any, ...]
        :return: tuple of serialized pk values safe to bind
        :rtype: tuple[Any, ...]
        """
        schema = self._schema_info.get(table, {})
        return tuple(
            self.serialize_value(value, schema.get(col, "TEXT")) for col, value in zip(pk_cols, pk_vals, strict=True)
        )

    def upsert(self, table: str, data: dict[str, Any], primary_key: str | tuple[str, ...] = "id") -> None:
        """insert or update row atomically.

        :param table: destination table name
        :ptype table: str
        :param data: row data keyed by column name. every pk column
            named in ``primary_key`` MUST be present.
        :ptype data: dict[str, Any]
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK). composite
            emits an ``ON CONFLICT (col_a, col_b, ...) DO UPDATE``
            clause; single-PK keeps the one-column shape.
        :ptype primary_key: str | tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        self.upsert_many(table, [data], primary_key)

    def upsert_many(
        self, table: str, rows: Sequence[Mapping[str, Any]], primary_key: str | tuple[str, ...] = "id"
    ) -> int:
        """insert or update many rows in one transaction, as ``upsert`` would one by one.

        :param table: destination table name
        :ptype table: str
        :param rows: the rows, each keyed by column name, every pk column present
        :ptype rows: Sequence[Mapping[str, Any]]
        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :return: how many rows were written
        :rtype: int
        :raises ValueError: when the rows do not all name the same columns
        """
        pk_cols = self._pk_columns(primary_key)
        schema = self._schema_info.get(table, {})
        # Only the table's own columns are written. ``BaseCollection.save_entity``
        # injects ``date_created`` / ``date_updated`` for new entities, and not every
        # table carries them (``agent_skill_invocations`` has ``invoked_at`` and
        # neither); writing an unknown column raises. The L3 path projects to declared
        # columns too. An unregistered table (no schema) writes every key named.
        columns = bulk_columns(rows, schema)
        if rows:
            # A table whose every column is its key has nothing to update.
            update_cols = [c for c in columns if c not in pk_cols]
            on_conflict = (
                "DO UPDATE SET "
                + ", ".join(f"{quote_identifier(c)} = EXCLUDED.{quote_identifier(c)}" for c in update_cols)
                if update_cols
                else "DO NOTHING"
            )
            sql = (
                f"INSERT INTO {quote_identifier(table)} ({', '.join(quote_identifier(c) for c in columns)}) VALUES ({', '.join('?' for _ in columns)}) "
                f"ON CONFLICT ({', '.join(quote_identifier(c) for c in pk_cols)}) {on_conflict}"
            )
            values = [tuple(self.serialize_value(row[c], schema.get(c, "TEXT")) for c in columns) for row in rows]
            self._in_write_transaction(lambda conn: conn.executemany(sql, values))
        return len(rows)

    def column_types(self, table: str) -> Mapping[str, str]:
        """the type codes this backend reads and writes a table's columns by, by column name.

        codes, not SQL types: ``TEXT_UUID`` is a TEXT column holding UUIDs, for one.

        :param table: the table
        :ptype table: str
        :return: each column's type code (empty for a table it does not know)
        :rtype: Mapping[str, str]
        """
        schema = self._schema_info.get(table, {})
        return MappingProxyType(dict(schema))

    def select_by_id(
        self,
        table: str,
        entity_id: Any,
        primary_key: str | tuple[str, ...] = "id",
        columns: Sequence[str] | None = None,
    ) -> dict[str, Any] | None:
        """select single row by primary key with type deserialization.

        :param table: target table name
        :ptype table: str
        :param entity_id: pk value (single-PK) or tuple of pk values in
            declared column order (composite-PK)
        :ptype entity_id: Any
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK)
        :ptype primary_key: str | tuple[str, ...]
        :param columns: columns to select and deserialize; ``None``
            selects every column. Exactly the named columns come back --
            pk columns are NOT implicitly added.
        :ptype columns: Sequence[str] | None
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        :raises ValueError: if ``columns`` is empty, or names a column
            the table's registered schema does not have
        """
        pk_cols = self._pk_columns(primary_key)
        pk_vals = self._serialize_pk_values(table, pk_cols, self._pk_values(entity_id, pk_cols))
        select_clause = build_select_clause(self._schema_info.get(table), table, columns)
        where_clause = " AND ".join(f"{quote_identifier(c)} = ?" for c in pk_cols)
        sql = f"SELECT {select_clause} FROM {quote_identifier(table)} WHERE {where_clause}"
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, pk_vals)
        row = cursor.fetchone()
        if row is None:
            return None
        return self._deserialize_row(table, dict(row))

    def select_batch(
        self,
        table: str,
        entity_ids: list[Any],
        primary_key: str | tuple[str, ...] = "id",
        columns: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """select multiple rows by primary key with type deserialization.

        :param table: target table name
        :ptype table: str
        :param entity_ids: list of pk values (single-PK) or list of
            tuples of pk values (composite-PK). composite emits a
            disjunction of ``(col_a = ? AND col_b = ?)`` predicates;
            single-PK emits ``col IN (?, ?, ...)``.
        :ptype entity_ids: list[Any]
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK)
        :ptype primary_key: str | tuple[str, ...]
        :param columns: columns to select and deserialize; ``None``
            selects every column. Exactly the named columns come back --
            pk columns are NOT implicitly added.
        :ptype columns: Sequence[str] | None
        :return: list of row dicts; empty list when ``entity_ids`` is empty
        :rtype: list[dict[str, Any]]
        :raises ValueError: if ``columns`` is empty, or names a column
            the table's registered schema does not have
        """
        if not entity_ids:
            return []
        select_clause = build_select_clause(self._schema_info.get(table), table, columns)
        pk_cols = self._pk_columns(primary_key)
        if len(pk_cols) == 1:
            placeholders = ", ".join(["?" for _ in entity_ids])
            sql = f"SELECT {select_clause} FROM {quote_identifier(table)} WHERE {quote_identifier(pk_cols[0])} IN ({placeholders})"
            params: tuple[Any, ...] = tuple(
                self._serialize_pk_values(table, pk_cols, self._pk_values(eid, pk_cols))[0] for eid in entity_ids
            )
        else:
            per_key = " AND ".join(f"{quote_identifier(c)} = ?" for c in pk_cols)
            disjunct = " OR ".join([f"({per_key})" for _ in entity_ids])
            sql = f"SELECT {select_clause} FROM {quote_identifier(table)} WHERE {disjunct}"
            flat: list[Any] = []
            for eid in entity_ids:
                flat.extend(self._serialize_pk_values(table, pk_cols, self._pk_values(eid, pk_cols)))
            params = tuple(flat)
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params)
        return [self._deserialize_row(table, dict(row)) for row in cursor.fetchall()]

    def delete_by_id(
        self,
        table: str,
        entity_id: Any,
        primary_key: str | tuple[str, ...] = "id",
    ) -> None:
        """delete single row by primary key.

        :param table: target table name
        :ptype table: str
        :param entity_id: pk value (single-PK) or tuple of pk values in
            declared column order (composite-PK)
        :ptype entity_id: Any
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK)
        :ptype primary_key: str | tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        pk_cols = self._pk_columns(primary_key)
        pk_vals = self._serialize_pk_values(table, pk_cols, self._pk_values(entity_id, pk_cols))
        where_clause = " AND ".join(f"{quote_identifier(c)} = ?" for c in pk_cols)
        sql = f"DELETE FROM {quote_identifier(table)} WHERE {where_clause}"
        self._in_write_transaction(lambda conn: conn.execute(sql, pk_vals))

    def _in_write_transaction(self, write: Callable[[Any], object]) -> None:
        """run one write in an immediate transaction on this thread's connection.

        Any failure inside rolls the transaction back before it propagates, so a bad
        row or a constraint never leaves ``BEGIN IMMEDIATE`` open holding the write
        lock for the thread's next write. ``BEGIN`` itself sits outside: a busy lock
        there opened nothing to roll back, and its ``OperationalError`` propagates as
        it is (callers such as the max-age drop rely on that).

        :param write: issues the write's statements on the connection
        :ptype write: Callable[[Any], object]
        :return: nothing
        :rtype: None
        """
        conn = self.get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            write(conn)
            conn.execute("COMMIT")
        except (
            BaseException
        ):  # prawduct:allow prawduct/broad-except -- rolls back whatever failed, then re-raises it unchanged
            # SQLite rolls back by itself on some failures (a full disk, an I/O error, an
            # interrupt), and a ROLLBACK then raises "no transaction is active" in place of
            # the error that caused it
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def execute_query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a generic SELECT query, returning list of row dicts.

        :param sql: SELECT statement with ``?`` placeholders
        :ptype sql: str
        :param params: positional parameter values
        :ptype params: tuple[Any, ...]
        :return: row dicts
        :rtype: list[dict[str, Any]]
        """
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            columns = self._numeric_columns_in(sql) if str(exc) == "integer overflow" else ()
            if not columns:
                raise
            raise OverflowError(
                f"a total of the exact NUMERIC column(s) {', '.join(columns)} passed SQLite's 64-bit integer "
                f"(2**63 - 1, scaled by 10**scale): this L1 cannot sum them; sum them in L3"
            ) from exc
        if not rows:
            return []
        return [dict(row) for row in rows]

    def _numeric_columns_in(self, sql: str) -> tuple[str, ...]:
        """every exact ``NUMERIC`` column, as ``table.column``, whose table and column ``sql`` names.

        :param sql: a statement
        :ptype sql: str
        :return: the columns, in table then declared order
        :rtype: tuple[str, ...]
        """
        return tuple(
            f"{table}.{column}"
            for table, columns in self._schema_info.items()
            if table in sql
            for column, code in columns.items()
            if column in sql and _decimal_code_scale(code) is not None
        )

    def numeric_comparison(
        self, column_sql: str, operator: str, value: Any, col_type: str
    ) -> tuple[str, tuple[Any, ...]]:
        """a SQL condition comparing an exact ``NUMERIC`` column with ``value`` as L3 compares them.

        The column holds its value times ``10**s`` as an integer. The operand is scaled exactly
        and bounded by the operator, never rounded to the column's scale: ``> v`` is ``> floor(v')``,
        ``>= v`` is ``>= ceil(v')``, ``< v`` is ``< ceil(v')`` and ``<= v`` is ``<= floor(v')``, where
        ``v'`` is ``v * 10**s``. ``=`` and ``!=`` with an operand finer than the scale, which no
        stored value equals, are ``column <> column`` and ``column = column``: false and true for
        every value, ``NULL`` for ``NULL``, as in L3. ``NaN`` and ``Infinity`` compare above every
        value and ``-Infinity`` below it.

        :param column_sql: the column as the statement names it, already quoted
        :ptype column_sql: str
        :param operator: one of ``=``, ``!=``, ``<>``, ``<``, ``<=``, ``>``, ``>=``
        :ptype operator: str
        :param value: the operand: a ``Decimal``, ``int``, ``float`` or decimal string
        :ptype value: Any
        :param col_type: the column's type code (:meth:`column_types`)
        :ptype col_type: str
        :return: the condition, with ``?`` placeholders, and its parameters
        :rtype: tuple[str, tuple[Any, ...]]
        :raises ValueError: when ``operator`` is not one of those, or ``col_type`` is not an exact
            ``NUMERIC``'s
        :raises decimal.InvalidOperation: when ``value`` is not a number
        """
        scale = _decimal_code_scale(col_type)
        if scale is None or operator not in _ABOVE_EVERY_VALUE:
            raise ValueError(f"no exact NUMERIC comparison {operator!r} for a column of type {col_type!r}")
        number = (
            value if isinstance(value, Decimal) else Decimal(repr(value) if isinstance(value, float) else str(value))
        )
        params: tuple[Any, ...] = ()
        if not number.is_finite():
            below = number.is_infinite() and number < 0
            always = (_BELOW_EVERY_VALUE if below else _ABOVE_EVERY_VALUE)[operator]
            condition = f"{column_sql} {'=' if always else '<>'} {column_sql}"
        else:
            context = Context(prec=max(1, len(number.as_tuple().digits)), Emax=MAX_EMAX, Emin=MIN_EMIN)
            exact = number.scaleb(scale, context)
            if operator in ("=", "!=", "<>") and exact != exact.to_integral_value():
                condition = f"{column_sql} {'<>' if operator == '=' else '='} {column_sql}"
            else:
                rounding = ROUND_CEILING if operator in (">=", "<") else ROUND_FLOOR
                bound = exact.to_integral_value(rounding=rounding)
                # every stored value is under 10**18, so a bound clamped to SQLite's integer range
                # answers each operator as the bound itself would
                params = (int(max(Decimal(-_SQLITE_MAX_INTEGER), min(Decimal(_SQLITE_MAX_INTEGER), bound))),)
                condition = f"{column_sql} {operator} ?"
        return condition, params

    def serialize_value(self, value: Any, col_type: str) -> Any:
        """Serialize a Python value for SQLite storage based on column type.

        An exact ``NUMERIC(p, s)`` column stores the value as L3 stores it, rounded to scale ``s``
        (:func:`~threetears.core.cache.base.stored_decimal`), times ``10**s`` as an integer. That
        rounding is right for a stored value and wrong for an operand: compare the column with one
        through :meth:`numeric_comparison`.
        """
        if value is None:
            return None

        result: Any = value
        scale = _decimal_code_scale(col_type)
        if scale is not None and not isinstance(value, bool):
            return int(stored_decimal(value, scale).scaleb(scale))

        # JSON values encode with the platform default L2 uses, so a UUID, datetime or
        # Decimal nested in one caches as its string, as the other tiers store it. a bare
        # json.dumps raised on them, AFTER the L3 commit: the caller was told a committed
        # write failed, and the invalidation broadcast after the cache step never ran.
        if isinstance(value, enum.Enum):
            result = value.value
        elif isinstance(value, dict):
            result = json.dumps(value, default=json_default)
        elif isinstance(value, _UUID_TYPES):
            result = str(value)
        elif isinstance(value, datetime):
            result = value.isoformat()
        elif isinstance(value, time):
            result = stored_time(value, zoned=col_type.endswith("_TIMETZ"))
        elif isinstance(value, date):
            result = value.isoformat()
        elif isinstance(value, bool):
            result = int(value)
        elif isinstance(value, Decimal):
            result = float(value)
        elif isinstance(value, (tuple, list)):
            result = json.dumps(list(value), default=json_default)
        elif isinstance(value, bytes):
            result = value.hex()

        return result

    def deserialize_field(self, value: Any, col_type: str) -> Any:
        """Deserialize a single SQLite value back to the correct Python type."""
        if value is None:
            return None

        result: Any = value
        scale = _decimal_code_scale(col_type)
        if scale is not None:
            result = Decimal(value).scaleb(-scale)
        elif col_type == "TEXT_UUID":
            result = uuid.UUID(value) if value else None
        elif col_type in ("TEXT_JSON", "TEXT_ARRAY", "TEXT_VECTOR"):
            if value and isinstance(value, str):
                try:
                    result = json.loads(value)
                except json.JSONDecodeError, ValueError:
                    result = value if col_type == "TEXT_JSON" else (value or [])
            elif col_type in ("TEXT_ARRAY", "TEXT_VECTOR"):
                result = value or []
        elif col_type == "INTEGER_BOOL":
            result = bool(value) if value is not None else None
        elif col_type == "TEXT_DATETIME":
            result = datetime.fromisoformat(value) if value else None
        elif col_type == "TEXT_DATE":
            result = date.fromisoformat(value) if value else None
        elif col_type in ("TEXT_TIME", "TEXT_TIMETZ"):
            result = time.fromisoformat(value) if value else None
        elif col_type == "TEXT_BYTEA":
            result = bytes.fromhex(value) if value else None
        return result

    def _deserialize_row(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        """Deserialize a SQLite row back to Python types using schema registry."""
        schema = self._schema_info.get(table, {})
        return {
            col_name: self.deserialize_field(value, schema.get(col_name, "TEXT")) for col_name, value in row.items()
        }

    def reset(self) -> None:
        """Close all connections and clear state."""
        with self._pool_lock:
            for conn in self._pooled_connections:
                try:
                    conn.close()
                # NOSILENT: teardown best-effort; a connection that will not close is already gone
                except Exception as exc:  # noqa: BLE001
                    contain(exc, log=log, what="sqlite L1 backend close: a connection did not close")
            self._pooled_connections = []
        self._local = threading.local()
        if self._anchor_conn is not None:
            self._anchor_conn.close()
        self._anchor_conn = None
        self._initialized = False
        self._schema_info = {}

    def is_initialized(self) -> bool:
        """Return True if the backend has been initialized."""
        return self._initialized

    def has_table(self, table: str) -> bool:
        """Return True if ``table`` was registered via ``initialize()`` (see
        :meth:`~threetears.core.cache.base.L1Backend.has_table`)."""
        return table in self._schema_info

    def _generate_create_table(self, table: Any) -> str:
        """Generate SQLite CREATE TABLE from SQLAlchemy table."""
        pk_cols = [col.name for col in table.columns if col.primary_key]
        is_composite_pk = len(pk_cols) > 1

        columns = []
        for column in table.columns:
            col_type = self._column_code(table.name, column)
            # Strip serialization hint suffix for DDL (TEXT_UUID -> TEXT, etc.)
            ddl_type = col_type.split("_")[0] if "_" in col_type else col_type
            nullable = ""
            primary = ""
            if column.primary_key:
                nullable = " NOT NULL"
                if not is_composite_pk:
                    primary = " PRIMARY KEY"
            columns.append(f"{quote_identifier(column.name)} {ddl_type}{nullable}{primary}")

        if is_composite_pk:
            pk_clause = ", ".join(quote_identifier(c) for c in pk_cols)
            columns.append(f"PRIMARY KEY ({pk_clause})")

        columns_sql = ", ".join(columns)
        return f"CREATE TABLE IF NOT EXISTS {quote_identifier(table.name)} ({columns_sql})"

    def _column_code(self, table: str, column: Any) -> str:
        """the type code a column of ``table`` is kept by, refusing one this L1 cannot hold exactly.

        :param table: the table
        :ptype table: str
        :param column: the SQLAlchemy column
        :ptype column: Any
        :return: the type code
        :rtype: str
        :raises UnsupportedL1ColumnError: for an exact ``NUMERIC`` with no declared precision and
            scale, or wider than 18 digits
        """
        exact = exact_decimal_type(
            column.type,
            table=table,
            column=column.name,
            max_precision=_SQLITE_MAX_DECIMAL_PRECISION,
            backend="SQLite",
        )
        return self._map_sqlalchemy_type(column.type) if exact is None else f"{_DECIMAL_CODE}_{exact[1]}"

    @staticmethod
    def _map_sqlalchemy_type(sa_type: Any) -> str:
        """Map SQLAlchemy type to SQLite equivalent with serialization hints."""
        if isinstance(sa_type, PGVECTOR_TYPES):
            return "TEXT_VECTOR"

        if isinstance(sa_type, (UUID, UuidType)):
            return "TEXT_UUID"
        if isinstance(sa_type, JSONB):
            return "TEXT_JSON"
        if isinstance(sa_type, Boolean):
            return "INTEGER_BOOL"
        if isinstance(sa_type, Float):
            return "REAL"
        if isinstance(sa_type, Integer):
            return "INTEGER"
        if isinstance(sa_type, (DateTime, TIMESTAMP)):
            return "TEXT_DATETIME"
        # a Date or Time column is its ISO text, as a DateTime column is
        # (:func:`~threetears.core.cache.base.stored_time`)
        if isinstance(sa_type, Date):
            return "TEXT_DATE"
        if isinstance(sa_type, Time):
            return "TEXT_TIMETZ" if sa_type.timezone else "TEXT_TIME"
        if isinstance(sa_type, (String, Text)):
            return "TEXT"
        # Generic LargeBinary covers both sqlalchemy.LargeBinary and the
        # postgresql BYTEA dialect type (PgBYTEA subclasses LargeBinary).
        if isinstance(sa_type, LargeBinary):
            return "TEXT_BYTEA"

        # PostgreSQL-only types that map cleanly to TEXT in SQLite

        if isinstance(sa_type, TSVECTOR):
            return "TEXT"
        if isinstance(sa_type, ARRAY):
            return "TEXT_ARRAY"

        log.warning(f"Unknown SQLAlchemy type {type(sa_type)}, defaulting to TEXT")
        return "TEXT"
