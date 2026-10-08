"""DuckDB L1 cache backend with thread-local connections and type-aware serialization.

Uses an in-memory DuckDB database. Schema is derived from SQLAlchemy metadata,
with type-aware serialization/deserialization. DuckDB is an optional dependency.
"""

from __future__ import annotations

import enum
import hashlib
import json
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

from threetears.core.backends.schema_sql import json_default
from threetears.core.cache.base import build_select_clause, bulk_columns, quote_identifier
from threetears.observe import get_logger

__all__ = [
    "DuckDBBackend",
    "PartitionReplacement",
]

try:
    import duckdb

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

try:
    from uuid_utils import UUID as _UuidUtilsUUID

    _UUID_TYPES: tuple[type, ...] = (uuid.UUID, _UuidUtilsUUID)
except ImportError:
    _UUID_TYPES = (uuid.UUID,)

log = get_logger(__name__)


@dataclass(frozen=True)
class PartitionReplacement:
    """everything one scope of a table is to hold: its rows, or an Arrow table of them.

    The scope is the rows whose ``column`` equals ``value`` (``IS NULL`` for ``None``). Give
    ``arrow`` (an Arrow table, as :meth:`DuckDBBackend.export_partition` writes one) or ``rows``
    (mappings, written as :meth:`DuckDBBackend.upsert_many` writes them, which needs the
    ``primary_key``); neither, or an empty one, leaves the scope empty.

    :ivar table: a table the backend created
    :ivar column: the scope column
    :ivar value: the scope's value
    :ivar rows: the scope's rows as mappings
    :ivar arrow: the scope's rows as an Arrow table
    :ivar primary_key: the table's key, for ``rows``
    """

    table: str
    column: str
    value: Any
    rows: Sequence[Mapping[str, Any]] | None = None
    arrow: Any = None
    primary_key: str | tuple[str, ...] = "id"


def _insert_through_arrow(connection: Any, target: str, columns: Sequence[str], lists: Sequence[list[Any]]) -> bool:
    """insert or replace rows given as one list per column, through one Arrow table; False when it cannot.

    Binding Python lists as parameters converts every value through DuckDB's Python layer, which for
    each one tries to import ``pandas`` (to recognise its types); where pandas is not installed that
    is a search of the whole import path per value -- seconds for a few thousand rows, and far longer
    on a starved host. One Arrow table is one conversion. Without pyarrow, or for a column whose values
    Arrow cannot type as one (mixed kinds), the caller binds the lists instead.

    :param connection: the connection to write on, its lock held by the caller
    :ptype connection: Any
    :param target: the quoted table and column list
    :ptype target: str
    :param columns: the columns, in order
    :ptype columns: Sequence[str]
    :param lists: each column's values, serialized to its storage form
    :ptype lists: Sequence[list[Any]]
    :return: whether the rows were inserted
    :rtype: bool
    """
    try:
        import pyarrow as pa  # noqa: PLC0415 -- optional: the snapshot extra
    except ImportError:
        return False
    try:
        arrow = pa.table({f"c{i}": values for i, values in enumerate(lists)})
    except pa.ArrowInvalid, pa.ArrowTypeError, OverflowError:
        return False
    names = ", ".join(f"c{i}" for i in range(len(columns)))
    connection.register("_bulk_rows", arrow)
    try:
        connection.execute(f"INSERT OR REPLACE INTO {target} SELECT {names} FROM _bulk_rows")  # noqa: S608
    finally:
        connection.unregister("_bulk_rows")
    return True


class DuckDBBackend:
    """L1 cache backend using DuckDB in-memory database.

    Instance-based so multiple backends can coexist. Each instance manages
    its own in-memory database, thread-local connections, and schema registry.

    DuckDB is an optional dependency. If not installed, instantiation raises
    ImportError with installation instructions.
    """

    def __init__(self) -> None:
        if not _HAS_DUCKDB:
            raise ImportError("DuckDB backend requires the 'duckdb' package. Install with: pip install 3tears[duckdb]")
        self._db: Any = None  # duckdb.DuckDBPyConnection
        self._initialized: bool = False
        self._schema_info: dict[str, dict[str, str]] = {}
        self._local: threading.local = threading.local()
        self._pool_lock: threading.Lock = threading.Lock()
        self._pooled_connections: list[Any] = []
        self._db_lock: threading.Lock = threading.Lock()

    def _make_connection(self) -> Any:
        """Create a new cursor/connection from the shared database."""
        conn = self._db.cursor()
        with self._pool_lock:
            self._pooled_connections.append(conn)
        return conn

    def initialize(self, sa_metadata: Any) -> None:
        """Initialize DuckDB with schema from SQLAlchemy metadata."""
        if self._initialized:
            log.debug("DuckDB already initialized, skipping")
            return

        self._db = duckdb.connect(":memory:")

        for table in sa_metadata.tables.values():
            ddl = self._generate_create_table(table)
            self._db.execute(ddl)
            self._schema_info[table.name] = {col.name: self._map_sqlalchemy_type(col.type) for col in table.columns}
            log.debug(f"Created DuckDB table: {table.name}")

        self._initialized = True
        log.debug(
            "DuckDB L1 cache initialized",
            extra={"extra_data": {"table_count": len(sa_metadata.tables)}},
        )

    def get_connection(self) -> Any:
        """Get a thread-local connection (cursor) to the DuckDB database."""
        if not self._initialized:
            raise RuntimeError("DuckDB not initialized - call initialize() first")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._make_connection()
            self._local.conn = conn
        return conn

    @staticmethod
    def _pk_columns(primary_key: str | tuple[str, ...]) -> tuple[str, ...]:
        """normalize pk argument to tuple of column names."""
        if isinstance(primary_key, tuple):
            return primary_key
        return (primary_key,)

    @staticmethod
    def _pk_values(entity_id: Any, pk_cols: tuple[str, ...]) -> tuple[Any, ...]:
        """normalize entity_id argument to tuple of pk values.

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

    def upsert(self, table: str, data: dict[str, Any], primary_key: str | tuple[str, ...] = "id") -> None:
        """insert or update row.

        duckdb supports ``INSERT OR REPLACE INTO`` for tables with
        primary keys; the statement honours whatever primary key (single
        or composite) was declared on the table, so no conflict column
        list is needed in the SQL.

        :param table: destination table name
        :ptype table: str
        :param data: row data keyed by column name
        :ptype data: dict[str, Any]
        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        self.upsert_many(table, [data], primary_key)

    def upsert_many(
        self, table: str, rows: Sequence[Mapping[str, Any]], primary_key: str | tuple[str, ...] = "id"
    ) -> int:
        """insert or update many rows in one columnar statement, as ``upsert`` would one by one.

        each column travels as one list and is unnested in the insert, which is
        roughly twenty times faster than DuckDB's ``executemany`` (measured on a
        168-column table: 2.6 s against 54 s for 20,000 rows).

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
        with self._db_lock:
            self._insert_rows(self._db, table, rows, primary_key)
        return len(rows)

    def _insert_rows(
        self, connection: Any, table: str, rows: Sequence[Mapping[str, Any]], primary_key: str | tuple[str, ...]
    ) -> None:
        """insert or replace ``rows`` in one columnar statement on ``connection``; the caller holds the lock.

        :param connection: the connection to write on
        :ptype connection: Any
        :param table: destination table name
        :ptype table: str
        :param rows: the rows, each keyed by column name, every pk column present
        :ptype rows: Sequence[Mapping[str, Any]]
        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :return: nothing
        :rtype: None
        :raises ValueError: when the rows do not all name the same columns
        """
        pk_cols = self._pk_columns(primary_key)
        schema = self._schema_info.get(table, {})
        # Only the table's own columns are written: the collection's pull-through
        # injects keys (the L1 cache-age stamp among them) this backend declares no
        # column for. An unregistered table (no schema) writes every key named.
        columns = bulk_columns(rows, schema)
        # DuckDB refuses to touch one key twice in a statement; one by one, the last
        # write of a key is the one that stays, so the batch keeps only that. keys are
        # compared as stored, so a UUID and its text are the same key, as they are in L1.
        latest = list(
            {
                tuple(self.serialize_value(row[c], schema.get(c, "VARCHAR")) for c in pk_cols): row for row in rows
            }.values()
        )
        if latest:
            lists = [[self.serialize_value(row[c], schema.get(c, "VARCHAR")) for row in latest] for c in columns]
            # values are already serialized to each column's storage form, so the insert's own
            # conversion types them; the registry's names are logical (VARCHAR_UUID), not SQL
            target = f"{quote_identifier(table)} ({', '.join(quote_identifier(c) for c in columns)})"
            if not _insert_through_arrow(connection, target, columns, lists):
                select = ", ".join(f"unnest(${i}) AS {quote_identifier(c)}" for i, c in enumerate(columns, 1))
                connection.execute(f"INSERT OR REPLACE INTO {target} SELECT {select}", lists)  # noqa: S608

    def replace_all(self, table: str, rows: Sequence[Mapping[str, Any]], primary_key: str | tuple[str, ...]) -> int:
        """make a table hold exactly ``rows``, in one transaction: what it held before is gone.

        For an L1 that is a whole copy of its table (:mod:`threetears.core.collections.complete_copy`):
        a row the source no longer holds must leave the copy, which an upsert never does. A
        failure rolls back, so the table keeps what it held. Rows repeating a key keep the last.

        :param table: a table this backend created
        :ptype table: str
        :param rows: every row the table is to hold, each keyed by column name
        :ptype rows: Sequence[Mapping[str, Any]]
        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :return: how many rows were given
        :rtype: int
        :raises ValueError: when the table is not one this backend created
        """
        if table not in self._schema_info:
            raise ValueError(f"unknown table {table!r}: replace_all fills tables this backend created")
        with self._db_lock:
            committed = False
            self._db.execute("BEGIN TRANSACTION")
            try:
                self._db.execute(f"DELETE FROM {quote_identifier(table)}")
                self._insert_rows(self._db, table, rows, primary_key)
                self._db.execute("COMMIT")
                committed = True
            finally:
                if not committed:
                    self._db.execute("ROLLBACK")
        log.info("replaced an L1 table whole", extra={"extra_data": {"table": table, "rows": len(rows)}})
        return len(rows)

    def stored_keys(self, table: str, key: Sequence[str]) -> list[tuple[Any, ...]]:
        """every row's key as this backend stores it: the form :meth:`serialize_value` writes.

        :param table: the table
        :ptype table: str
        :param key: the key's columns, in order
        :ptype key: Sequence[str]
        :return: one tuple of stored values per row, in no particular order
        :rtype: list[tuple[Any, ...]]
        """
        columns = ", ".join(quote_identifier(column) for column in key)
        with self._db_lock:
            rows = self._db.execute(f"SELECT {columns} FROM {quote_identifier(table)}").fetchall()
        return [tuple(row) for row in rows]

    def load_parquet(self, table: str, path: str | Path, *, row_number_column: str | None = None) -> int:
        """load a Parquet file into a table this backend created, in one statement.

        DuckDB reads Parquet natively, so this is the fast path for an analytic
        table filled from a file: the file's columns that the table declares are
        loaded and the rest are ignored; table columns the file lacks stay null. A file
        repeating a key is refused by DuckDB (one statement cannot write a key twice),
        unlike ``upsert_many``, which keeps a batch's last row per key.

        :param table: the destination table
        :ptype table: str
        :param path: the Parquet file
        :ptype path: str | Path
        :param row_number_column: a table column to fill with each row's position
            (from 0), as a key for files that carry none
        :ptype row_number_column: str | None
        :return: how many rows the file held
        :rtype: int
        :raises ValueError: when the table is not one this backend created
        """
        schema = self._schema_info.get(table)
        if schema is None:
            raise ValueError(f"unknown table {table!r}: load_parquet fills tables this backend created")
        with self._db_lock:
            in_file = {
                row[0] for row in self._db.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
            }
            columns = [c for c in schema if c in in_file and c != row_number_column]
            selects = [quote_identifier(c) for c in columns]
            if row_number_column is not None:
                columns.insert(0, row_number_column)
                selects.insert(0, "row_number() OVER () - 1")
            self._db.execute(
                f"INSERT OR REPLACE INTO {quote_identifier(table)} ({', '.join(quote_identifier(c) for c in columns)}) "
                f"SELECT {', '.join(selects)} FROM read_parquet(?)",
                [str(path)],
            )
            (count,) = self._db.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()
        log.info(
            "loaded a Parquet file into an L1 table",
            extra={"extra_data": {"table": table, "rows": count, "columns": len(columns)}},
        )
        return int(count)

    def _partition_filter(self, table: str, column: str, value: Any) -> tuple[str, list[Any]]:
        """the condition naming one scope of a table, and its parameter.

        :param table: the table
        :ptype table: str
        :param column: the scope column
        :ptype column: str
        :param value: the scope's value; ``None`` is the rows where the column is null
        :ptype value: Any
        :return: the condition and its parameters
        :rtype: tuple[str, list[Any]]
        :raises ValueError: when the table is not one this backend created, or has no such column
        """
        schema = self._schema_info.get(table)
        if schema is None or column not in schema:
            raise ValueError(
                f"unknown table or column {table!r}.{column!r}: partitions are of tables this backend created"
            )
        if value is None:
            return f"{quote_identifier(column)} IS NULL", []
        return f"{quote_identifier(column)} = ?", [self.serialize_value(value, schema[column])]

    def export_partition(self, table: str, column: str, value: Any, *, order_by: Sequence[str]) -> Any:
        """one scope of a table as an Arrow table, its rows in ``order_by`` order.

        In key order, so the same rows always encode to the same bytes. Needs ``pyarrow``
        (``3tears[snapshot]``).

        :param table: a table this backend created
        :ptype table: str
        :param column: the scope column
        :ptype column: str
        :param value: the scope's value; ``None`` for the rows where the column is null
        :ptype value: Any
        :param order_by: the columns ordering the rows, the table's key
        :ptype order_by: Sequence[str]
        :return: the rows
        :rtype: pyarrow.Table
        :raises ValueError: when the table or column is unknown
        """
        condition, params = self._partition_filter(table, column, value)
        order = ", ".join(quote_identifier(c) for c in order_by)
        sql = f"SELECT * FROM {quote_identifier(table)} WHERE {condition}" + (f" ORDER BY {order}" if order else "")  # noqa: S608
        with self._db_lock:
            result = self._db.execute(sql, params)
            # ``to_arrow_table`` replaced ``fetch_arrow_table`` (deprecated in 1.4); older releases have only the latter
            fetch = getattr(result, "to_arrow_table", None) or result.fetch_arrow_table
            return fetch()

    def export_rows(
        self,
        table: str,
        column: str,
        value: Any,
        rows: Sequence[Mapping[str, Any]],
        *,
        primary_key: str | tuple[str, ...],
        order_by: Sequence[str],
    ) -> Any:
        """the Arrow table one scope WOULD hold with ``rows``, as :meth:`export_partition` would
        write it after a replacement, changing nothing.

        For a writer that encodes a scope's next content before it may show it: the rows are typed
        and ordered by this backend's own columns, so the export equals the one taken after
        :meth:`replace_partitions` with the same rows. It runs in a transaction that is rolled back,
        so no reader, on :meth:`read_snapshot` or otherwise, ever sees the rows. Needs ``pyarrow``.

        :param table: a table this backend created
        :ptype table: str
        :param column: the scope column
        :ptype column: str
        :param value: the scope's value; ``None`` for the rows where the column is null
        :ptype value: Any
        :param rows: every row the scope would hold, each keyed by column name
        :ptype rows: Sequence[Mapping[str, Any]]
        :param primary_key: the table's key, as :meth:`upsert_many` takes it
        :ptype primary_key: str | tuple[str, ...]
        :param order_by: the columns ordering the rows, the table's key
        :ptype order_by: Sequence[str]
        :return: the rows
        :rtype: pyarrow.Table
        :raises ValueError: when the table or column is unknown
        """
        condition, params = self._partition_filter(table, column, value)
        quoted = quote_identifier(table)
        order = ", ".join(quote_identifier(c) for c in order_by)
        sql = f"SELECT * FROM {quoted} WHERE {condition}" + (f" ORDER BY {order}" if order else "")  # noqa: S608
        with self._db_lock:
            self._db.execute("BEGIN TRANSACTION")
            try:
                self._db.execute(f"DELETE FROM {quoted} WHERE {condition}", params)  # noqa: S608
                if rows:
                    self._insert_rows(self._db, table, rows, primary_key)
                result = self._db.execute(sql, params)
                fetch = getattr(result, "to_arrow_table", None) or result.fetch_arrow_table
                exported = fetch()
            finally:
                self._db.execute("ROLLBACK")
        return exported

    def replace_partitions(self, replacements: Sequence[PartitionReplacement]) -> int:
        """make each named scope hold exactly what its replacement gives, all in ONE transaction.

        A reader on another cursor (:meth:`read_snapshot`) sees every scope as it was until the
        commit and every one as it is after, never part of the change. A failure rolls back, so
        every table keeps what it held.

        :param replacements: the scopes and what each is to hold
        :ptype replacements: Sequence[PartitionReplacement]
        :return: how many rows were written
        :rtype: int
        :raises ValueError: when a table or column is unknown; nothing changes
        """
        filters = [self._partition_filter(r.table, r.column, r.value) for r in replacements]
        written = 0
        with self._db_lock:
            committed = False
            self._db.execute("BEGIN TRANSACTION")
            try:
                for replacement, (condition, params) in zip(replacements, filters, strict=True):
                    table = quote_identifier(replacement.table)
                    self._db.execute(f"DELETE FROM {table} WHERE {condition}", params)  # noqa: S608
                    if replacement.arrow is not None and replacement.arrow.num_rows:
                        self._db.register("_partition_chunk", replacement.arrow)
                        try:
                            self._db.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _partition_chunk")  # noqa: S608
                        finally:
                            self._db.unregister("_partition_chunk")
                        written += replacement.arrow.num_rows
                    elif replacement.rows:
                        self._insert_rows(self._db, replacement.table, replacement.rows, replacement.primary_key)
                        written += len(replacement.rows)
                self._db.execute("COMMIT")
                committed = True
            finally:
                if not committed:
                    self._db.execute("ROLLBACK")
        return written

    @contextmanager
    def read_snapshot(self) -> Iterator[Any]:
        """a cursor reading one state of every table, held still until the block ends.

        Every query on it reads the database as it stood when the block began, whatever
        :meth:`replace_partitions` commits meanwhile, so several queries answering one request
        agree with each other. Read-only by use; it holds no lock, so writers are not kept waiting.

        :return: the cursor, inside a read transaction
        :rtype: Iterator[duckdb.DuckDBPyConnection]
        :raises RuntimeError: before :meth:`initialize`
        """
        if not self._initialized:
            raise RuntimeError("DuckDB not initialized - call initialize() first")
        with self._db_lock:
            cursor = self._db.cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")
            try:
                yield cursor
            finally:
                cursor.execute("ROLLBACK")
        finally:
            cursor.close()

    def schema_digest(self, table: str) -> str:
        """a digest of a table's columns and their type codes, in declaration order.

        Two backends whose tables give the same digest read and write the same columns the same
        way, so an Arrow export of one loads into the other.

        :param table: the table
        :ptype table: str
        :return: the hex digest
        :rtype: str
        :raises ValueError: when the table is not one this backend created
        """
        schema = self._schema_info.get(table)
        if schema is None:
            raise ValueError(f"unknown table {table!r}")
        text = "\n".join(f"{name}:{code}" for name, code in schema.items())
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def column_types(self, table: str) -> Mapping[str, str]:
        """the type codes this backend reads and writes a table's columns by, by column name.

        codes, not SQL types: ``VARCHAR_JSON`` is a VARCHAR column holding JSON, for one.

        :param table: the table
        :ptype table: str
        :return: each column's type code (empty for a table it does not know)
        :rtype: Mapping[str, str]
        """
        return MappingProxyType(dict(self._schema_info.get(table, {})))

    def select_by_id(
        self,
        table: str,
        entity_id: Any,
        primary_key: str | tuple[str, ...] = "id",
        columns: Sequence[str] | None = None,
        *,
        max_age_seconds: float | None = None,
        now_monotonic: float | None = None,  # noqa: ARG002 - protocol parity; unused without expiry
    ) -> dict[str, Any] | None:
        """select single row by primary key with type deserialization.

        Expiry is NOT supported here; passing ``max_age_seconds`` raises.

        :param table: target table name
        :ptype table: str
        :param entity_id: pk value or tuple of pk values in declared order
        :ptype entity_id: Any
        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :param columns: columns to select and deserialize; ``None``
            selects every column. Exactly the named columns come back --
            pk columns are NOT implicitly added.
        :ptype columns: Sequence[str] | None
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        :raises ValueError: if ``columns`` is empty, or names a column
            the table's registered schema does not have
        :raises NotImplementedError: if ``max_age_seconds`` is given; this
            backend injects no cache-age stamp, so it cannot honour a bound
        """
        if max_age_seconds is not None:
            raise NotImplementedError(
                "DuckDBBackend does not implement L1 max-age expiry: it never injects the "
                "cache-age stamp, so the request would silently never fire and the staleness "
                "bound the caller asked for would not exist. Use SQLiteBackend."
            )
        pk_cols = self._pk_columns(primary_key)
        pk_vals = self._pk_values(entity_id, pk_cols)
        select_clause = build_select_clause(self._schema_info.get(table), table, columns)
        where_clause = " AND ".join(f"{quote_identifier(c)} = ?" for c in pk_cols)
        sql = f"SELECT {select_clause} FROM {quote_identifier(table)} WHERE {where_clause}"
        with self._db_lock:
            result = self._db.execute(sql, list(pk_vals))
            result_columns = [desc[0] for desc in result.description]
            row = result.fetchone()
        if row is not None:
            row_dict = dict(zip(result_columns, row))
            return self._deserialize_row(table, row_dict)
        return None

    def select_batch(
        self,
        table: str,
        entity_ids: list[Any],
        primary_key: str | tuple[str, ...] = "id",
        columns: Sequence[str] | None = None,
        *,
        max_age_seconds: float | None = None,
        now_monotonic: float | None = None,  # noqa: ARG002 - protocol parity; unused without expiry
    ) -> list[dict[str, Any]]:
        """select multiple rows by primary key with type deserialization.

        Expiry is NOT supported here; passing ``max_age_seconds`` raises.

        :param table: target table name
        :ptype table: str
        :param entity_ids: list of pk values or list of tuples
        :ptype entity_ids: list[Any]
        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :param columns: columns to select and deserialize; ``None``
            selects every column. Exactly the named columns come back --
            pk columns are NOT implicitly added.
        :ptype columns: Sequence[str] | None
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        :raises ValueError: if ``columns`` is empty, or names a column
            the table's registered schema does not have
        :raises NotImplementedError: if ``max_age_seconds`` is given; this
            backend injects no cache-age stamp, so it cannot honour a bound
        """
        if max_age_seconds is not None:
            raise NotImplementedError(
                "DuckDBBackend does not implement L1 max-age expiry: it never injects the "
                "cache-age stamp, so the request would silently never fire and the staleness "
                "bound the caller asked for would not exist. Use SQLiteBackend."
            )
        if not entity_ids:
            return []
        select_clause = build_select_clause(self._schema_info.get(table), table, columns)
        pk_cols = self._pk_columns(primary_key)
        if len(pk_cols) == 1:
            placeholders = ", ".join(["?" for _ in entity_ids])
            sql = f"SELECT {select_clause} FROM {quote_identifier(table)} WHERE {quote_identifier(pk_cols[0])} IN ({placeholders})"
            params: list[Any] = list(entity_ids)
        else:
            per_key = " AND ".join(f"{quote_identifier(c)} = ?" for c in pk_cols)
            disjunct = " OR ".join([f"({per_key})" for _ in entity_ids])
            sql = f"SELECT {select_clause} FROM {quote_identifier(table)} WHERE {disjunct}"
            params = []
            for eid in entity_ids:
                params.extend(self._pk_values(eid, pk_cols))
        with self._db_lock:
            result = self._db.execute(sql, params)
            result_columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
        return [self._deserialize_row(table, dict(zip(result_columns, row))) for row in rows]

    def delete_by_id(
        self,
        table: str,
        entity_id: Any,
        primary_key: str | tuple[str, ...] = "id",
    ) -> None:
        """delete single row by primary key.

        :param table: target table name
        :ptype table: str
        :param entity_id: pk value or tuple of pk values in declared order
        :ptype entity_id: Any
        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        pk_cols = self._pk_columns(primary_key)
        pk_vals = self._pk_values(entity_id, pk_cols)
        where_clause = " AND ".join(f"{quote_identifier(c)} = ?" for c in pk_cols)
        sql = f"DELETE FROM {quote_identifier(table)} WHERE {where_clause}"
        with self._db_lock:
            self._db.execute(sql, list(pk_vals))

    def execute_query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a generic SELECT query, returning list of row dicts."""
        with self._db_lock:
            result = self._db.execute(sql, list(params))
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def serialize_value(self, value: Any, col_type: str) -> Any:
        """Serialize a Python value for DuckDB storage based on column type.

        JSON values encode with the storage handler every tier shares
        (:func:`~threetears.core.backends.schema_sql.json_default`), as the SQLite L1 does: a
        nested UUID, Decimal or datetime is cached as the string the other tiers store, where a
        bare ``json.dumps`` raised on it.
        """
        if value is None:
            return None

        result: Any = value

        if isinstance(value, enum.Enum):
            result = value.value
        elif isinstance(value, dict):
            result = json.dumps(value, default=json_default)
        elif isinstance(value, _UUID_TYPES):
            result = str(value)
        elif isinstance(value, datetime):
            result = value.isoformat()
        elif isinstance(value, bool):
            result = value  # DuckDB has native BOOLEAN
        elif isinstance(value, Decimal):
            result = float(value)
        elif isinstance(value, (tuple, list)):
            result = json.dumps(list(value), default=json_default)
        elif isinstance(value, bytes):
            result = value.hex()

        return result

    def deserialize_field(self, value: Any, col_type: str) -> Any:
        """Deserialize a single DuckDB value back to the correct Python type."""
        if value is None:
            return None

        result: Any = value
        if col_type == "VARCHAR_UUID":
            result = uuid.UUID(value) if value else None
        elif col_type in ("VARCHAR_JSON", "VARCHAR_ARRAY", "VARCHAR_VECTOR"):
            if value and isinstance(value, str):
                try:
                    result = json.loads(value)
                except json.JSONDecodeError, ValueError:
                    result = value if col_type == "VARCHAR_JSON" else (value or [])
            elif col_type in ("VARCHAR_ARRAY", "VARCHAR_VECTOR"):
                result = value or []
        elif col_type == "BOOLEAN":
            result = bool(value) if value is not None else None
        elif col_type == "VARCHAR_DATETIME":
            if isinstance(value, str):
                result = datetime.fromisoformat(value)
            elif isinstance(value, datetime):
                result = value
            else:
                result = datetime.fromisoformat(str(value)) if value else None
        elif col_type == "VARCHAR_BYTEA":
            result = bytes.fromhex(value) if value else None
        return result

    def _deserialize_row(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        """Deserialize a DuckDB row back to Python types using schema registry."""
        schema = self._schema_info.get(table, {})
        return {
            col_name: self.deserialize_field(value, schema.get(col_name, "VARCHAR")) for col_name, value in row.items()
        }

    def reset(self) -> None:
        """Close all connections and clear state."""
        with self._pool_lock:
            for conn in self._pooled_connections:
                try:
                    conn.close()
                # NOSILENT: teardown best-effort; a connection that will not close is already gone
                except Exception:  # noqa: BLE001
                    pass
            self._pooled_connections = []
        self._local = threading.local()
        if self._db is not None:
            try:
                self._db.close()
            # NOSILENT: teardown best-effort; a connection that will not close is already gone
            except Exception:  # noqa: BLE001
                pass
        self._db = None
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
        """Generate DuckDB CREATE TABLE from SQLAlchemy table."""
        pk_cols = [col.name for col in table.columns if col.primary_key]
        is_composite_pk = len(pk_cols) > 1

        columns = []
        for column in table.columns:
            col_type = self._map_sqlalchemy_type(column.type)
            # Strip serialization hint suffix for DDL (VARCHAR_UUID -> VARCHAR, etc.)
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

    @staticmethod
    def _map_sqlalchemy_type(sa_type: Any) -> str:
        """Map SQLAlchemy type to DuckDB equivalent with serialization hints."""
        from sqlalchemy import (
            Boolean,
            DateTime,
            Float,
            Integer,
            LargeBinary,
            Numeric,
            String,
            Text,
        )
        from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
        from sqlalchemy.sql.sqltypes import UUID as UuidType  # noqa: N811

        # Check for pgvector Vector type
        try:
            from pgvector.sqlalchemy import Vector

            if isinstance(sa_type, Vector):
                return "VARCHAR_VECTOR"
        # NOSILENT: optional dependency probe; absence is a supported configuration
        except ImportError:
            pass

        if isinstance(sa_type, (UUID, UuidType)):
            return "VARCHAR_UUID"
        if isinstance(sa_type, JSONB):
            return "VARCHAR_JSON"
        if isinstance(sa_type, Boolean):
            return "BOOLEAN"
        if isinstance(sa_type, (Float, Numeric)):
            return "DOUBLE"
        if isinstance(sa_type, Integer):
            return "BIGINT"
        if isinstance(sa_type, (DateTime, TIMESTAMP)):
            return "VARCHAR_DATETIME"
        if isinstance(sa_type, (String, Text)):
            return "VARCHAR"
        # Generic LargeBinary covers both sqlalchemy.LargeBinary and the
        # postgresql BYTEA dialect type (PgBYTEA subclasses LargeBinary).
        if isinstance(sa_type, LargeBinary):
            return "VARCHAR_BYTEA"

        # PostgreSQL-only types
        from sqlalchemy.dialects.postgresql import TSVECTOR
        from sqlalchemy.sql.sqltypes import ARRAY

        if isinstance(sa_type, TSVECTOR):
            return "VARCHAR"
        if isinstance(sa_type, ARRAY):
            return "VARCHAR_ARRAY"

        log.warning(f"Unknown SQLAlchemy type {type(sa_type)}, defaulting to VARCHAR")
        return "VARCHAR"
