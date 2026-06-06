"""SQLite L1 cache backend with thread-local connections and type-aware serialization.

Uses a named in-memory database (memdb VFS) so multiple connections can access
the same data. The anchor connection keeps the database alive. Schema is derived
from SQLAlchemy metadata, with type-aware serialization/deserialization for
UUID, datetime, JSON, boolean, vector, and bytea columns.
"""

from __future__ import annotations

import enum
import json
import sqlite3
import threading
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from threetears.observe import get_logger

__all__ = [
    "SQLiteBackend",
]

try:
    from uuid_utils import UUID as _UuidUtilsUUID

    _UUID_TYPES: tuple[type, ...] = (uuid.UUID, _UuidUtilsUUID)
except ImportError:
    _UUID_TYPES = (uuid.UUID,)

log = get_logger(__name__)


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
        """
        if self._anchor_conn is None:
            self._anchor_conn = self._make_connection()

        new_tables = 0
        for table in sa_metadata.tables.values():
            if table.name in self._schema_info:
                log.debug(f"SQLite table already registered, skipping: {table.name}")
                continue
            ddl = self._generate_create_table(table)
            self._anchor_conn.execute(ddl)
            self._schema_info[table.name] = {col.name: self._map_sqlalchemy_type(col.type) for col in table.columns}
            new_tables += 1
            log.debug(f"Created SQLite table: {table.name}")

        if not self._initialized:
            # Register type adapters so UUID/datetime values are automatically
            # serialized at the SQLite boundary.
            sqlite3.register_adapter(uuid.UUID, str)
            sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
            try:
                from uuid_utils import UUID as UuidUtilsUUID

                sqlite3.register_adapter(UuidUtilsUUID, str)
            except ImportError:
                pass
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
        pk_cols = self._pk_columns(primary_key)
        schema = self._schema_info.get(table, {})
        # Filter ``data`` to columns the L1 table actually has. The
        # framework's ``BaseCollection.save_entity`` unconditionally
        # injects ``date_created`` / ``date_updated`` for new entities,
        # but not every entity's table carries those columns -- e.g.
        # ``agent_skill_invocations`` uses ``invoked_at`` and has neither
        # timestamp column. Writing an unknown column to SQLite raises
        # ``OperationalError: table X has no column named date_created``.
        # The L3 path already projects to declared columns
        # (``save_to_postgres``); mirror that here so the L1 write never
        # diverges from the table shape. When the schema is unknown
        # (table not registered via ``_generate_create_table``), fall
        # back to writing every key so existing behaviour is preserved.
        if schema:
            columns = [c for c in data if c in schema]
        else:
            columns = list(data.keys())
        placeholders = ", ".join(["?" for _ in columns])
        column_names = ", ".join(columns)

        values = []
        for col_name in columns:
            value = data[col_name]
            col_type = schema.get(col_name, "TEXT")
            values.append(self.serialize_value(value, col_type))

        update_cols = [c for c in columns if c not in pk_cols]
        update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
        conflict_clause = ", ".join(pk_cols)

        sql = f"""
            INSERT INTO {table} ({column_names})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_clause}) DO UPDATE SET {update_clause}
        """
        values_tuple = tuple(values)

        conn = self.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(sql, values_tuple)
            conn.execute("COMMIT")
        except sqlite3.OperationalError:
            conn.execute("ROLLBACK")
            raise

    def select_by_id(
        self,
        table: str,
        entity_id: Any,
        primary_key: str | tuple[str, ...] = "id",
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
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        pk_cols = self._pk_columns(primary_key)
        pk_vals = self._serialize_pk_values(table, pk_cols, self._pk_values(entity_id, pk_cols))
        where_clause = " AND ".join(f"{c} = ?" for c in pk_cols)
        sql = f"SELECT * FROM {table} WHERE {where_clause}"
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, pk_vals)
        row = cursor.fetchone()
        if row is not None:
            return self._deserialize_row(table, dict(row))
        return None

    def select_batch(
        self,
        table: str,
        entity_ids: list[Any],
        primary_key: str | tuple[str, ...] = "id",
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
        :return: list of row dicts; empty list when ``entity_ids`` is empty
        :rtype: list[dict[str, Any]]
        """
        if not entity_ids:
            return []
        pk_cols = self._pk_columns(primary_key)
        if len(pk_cols) == 1:
            placeholders = ", ".join(["?" for _ in entity_ids])
            sql = f"SELECT * FROM {table} WHERE {pk_cols[0]} IN ({placeholders})"
            params: tuple[Any, ...] = tuple(
                self._serialize_pk_values(table, pk_cols, self._pk_values(eid, pk_cols))[0] for eid in entity_ids
            )
        else:
            per_key = " AND ".join(f"{c} = ?" for c in pk_cols)
            disjunct = " OR ".join([f"({per_key})" for _ in entity_ids])
            sql = f"SELECT * FROM {table} WHERE {disjunct}"
            flat: list[Any] = []
            for eid in entity_ids:
                flat.extend(self._serialize_pk_values(table, pk_cols, self._pk_values(eid, pk_cols)))
            params = tuple(flat)
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        return [self._deserialize_row(table, dict(row)) for row in rows]

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
        where_clause = " AND ".join(f"{c} = ?" for c in pk_cols)
        sql = f"DELETE FROM {table} WHERE {where_clause}"
        conn = self.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(sql, pk_vals)
            conn.execute("COMMIT")
        except sqlite3.OperationalError:
            conn.execute("ROLLBACK")
            raise

    def execute_query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a generic SELECT query, returning list of row dicts."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def serialize_value(self, value: Any, col_type: str) -> Any:
        """Serialize a Python value for SQLite storage based on column type."""
        if value is None:
            return None

        result: Any = value

        if isinstance(value, enum.Enum):
            result = value.value
        elif isinstance(value, dict):
            result = json.dumps(value)
        elif isinstance(value, _UUID_TYPES):
            result = str(value)
        elif isinstance(value, datetime):
            result = value.isoformat()
        elif isinstance(value, bool):
            result = int(value)
        elif isinstance(value, Decimal):
            result = float(value)
        elif isinstance(value, (tuple, list)):
            result = json.dumps(list(value))
        elif isinstance(value, bytes):
            result = value.hex()

        return result

    def deserialize_field(self, value: Any, col_type: str) -> Any:
        """Deserialize a single SQLite value back to the correct Python type."""
        if value is None:
            return None

        result: Any = value
        if col_type == "TEXT_UUID":
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
                except Exception:  # noqa: BLE001
                    pass
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

    def _generate_create_table(self, table: Any) -> str:
        """Generate SQLite CREATE TABLE from SQLAlchemy table."""
        pk_cols = [col.name for col in table.columns if col.primary_key]
        is_composite_pk = len(pk_cols) > 1

        columns = []
        for column in table.columns:
            col_type = self._map_sqlalchemy_type(column.type)
            # Strip serialization hint suffix for DDL (TEXT_UUID -> TEXT, etc.)
            ddl_type = col_type.split("_")[0] if "_" in col_type else col_type
            nullable = ""
            primary = ""
            if column.primary_key:
                nullable = " NOT NULL"
                if not is_composite_pk:
                    primary = " PRIMARY KEY"
            columns.append(f'"{column.name}" {ddl_type}{nullable}{primary}')

        if is_composite_pk:
            pk_clause = ", ".join(f'"{c}"' for c in pk_cols)
            columns.append(f"PRIMARY KEY ({pk_clause})")

        columns_sql = ", ".join(columns)
        return f"CREATE TABLE IF NOT EXISTS {table.name} ({columns_sql})"

    @staticmethod
    def _map_sqlalchemy_type(sa_type: Any) -> str:
        """Map SQLAlchemy type to SQLite equivalent with serialization hints."""
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
                return "TEXT_VECTOR"
        except ImportError:
            pass

        if isinstance(sa_type, (UUID, UuidType)):
            return "TEXT_UUID"
        if isinstance(sa_type, JSONB):
            return "TEXT_JSON"
        if isinstance(sa_type, Boolean):
            return "INTEGER_BOOL"
        if isinstance(sa_type, (Float, Numeric)):
            return "REAL"
        if isinstance(sa_type, Integer):
            return "INTEGER"
        if isinstance(sa_type, (DateTime, TIMESTAMP)):
            return "TEXT_DATETIME"
        if isinstance(sa_type, (String, Text)):
            return "TEXT"
        # Generic LargeBinary covers both sqlalchemy.LargeBinary and the
        # postgresql BYTEA dialect type (PgBYTEA subclasses LargeBinary).
        if isinstance(sa_type, LargeBinary):
            return "TEXT_BYTEA"

        # PostgreSQL-only types that map cleanly to TEXT in SQLite
        from sqlalchemy.dialects.postgresql import TSVECTOR
        from sqlalchemy.sql.sqltypes import ARRAY

        if isinstance(sa_type, TSVECTOR):
            return "TEXT"
        if isinstance(sa_type, ARRAY):
            return "TEXT_ARRAY"

        log.warning(f"Unknown SQLAlchemy type {type(sa_type)}, defaulting to TEXT")
        return "TEXT"
