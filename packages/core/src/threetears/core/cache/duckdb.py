"""DuckDB L1 cache backend with thread-local connections and type-aware serialization.

Uses an in-memory DuckDB database. Schema is derived from SQLAlchemy metadata,
with type-aware serialization/deserialization. DuckDB is an optional dependency.
"""

from __future__ import annotations

import enum
import json
import threading
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from threetears.observe import get_logger

__all__ = [
    "DuckDBBackend",
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
        _ = self._pk_columns(primary_key)  # validate shape, unused in SQL
        columns = list(data.keys())
        schema = self._schema_info.get(table, {})

        values = []
        for col_name in columns:
            value = data[col_name]
            col_type = schema.get(col_name, "VARCHAR")
            values.append(self.serialize_value(value, col_type))

        column_names = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in columns])

        sql = f"INSERT OR REPLACE INTO {table} ({column_names}) VALUES ({placeholders})"

        with self._db_lock:
            self._db.execute(sql, values)

    def select_by_id(
        self,
        table: str,
        entity_id: Any,
        primary_key: str | tuple[str, ...] = "id",
    ) -> dict[str, Any] | None:
        """select single row by primary key with type deserialization.

        :param table: target table name
        :ptype table: str
        :param entity_id: pk value or tuple of pk values in declared order
        :ptype entity_id: Any
        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        pk_cols = self._pk_columns(primary_key)
        pk_vals = self._pk_values(entity_id, pk_cols)
        where_clause = " AND ".join(f"{c} = ?" for c in pk_cols)
        sql = f"SELECT * FROM {table} WHERE {where_clause}"
        with self._db_lock:
            result = self._db.execute(sql, list(pk_vals))
            columns = [desc[0] for desc in result.description]
            row = result.fetchone()
        if row is not None:
            row_dict = dict(zip(columns, row))
            return self._deserialize_row(table, row_dict)
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
        :param entity_ids: list of pk values or list of tuples
        :ptype entity_ids: list[Any]
        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if not entity_ids:
            return []
        pk_cols = self._pk_columns(primary_key)
        if len(pk_cols) == 1:
            placeholders = ", ".join(["?" for _ in entity_ids])
            sql = f"SELECT * FROM {table} WHERE {pk_cols[0]} IN ({placeholders})"
            params: list[Any] = list(entity_ids)
        else:
            per_key = " AND ".join(f"{c} = ?" for c in pk_cols)
            disjunct = " OR ".join([f"({per_key})" for _ in entity_ids])
            sql = f"SELECT * FROM {table} WHERE {disjunct}"
            params = []
            for eid in entity_ids:
                params.extend(self._pk_values(eid, pk_cols))
        with self._db_lock:
            result = self._db.execute(sql, params)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
        return [self._deserialize_row(table, dict(zip(columns, row))) for row in rows]

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
        where_clause = " AND ".join(f"{c} = ?" for c in pk_cols)
        sql = f"DELETE FROM {table} WHERE {where_clause}"
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
        """Serialize a Python value for DuckDB storage based on column type."""
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
            result = value  # DuckDB has native BOOLEAN
        elif isinstance(value, Decimal):
            result = float(value)
        elif isinstance(value, (tuple, list)):
            result = json.dumps(list(value))
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
                except Exception:  # noqa: BLE001
                    pass
            self._pooled_connections = []
        self._local = threading.local()
        if self._db is not None:
            try:
                self._db.close()
            except Exception:  # noqa: BLE001
                pass
        self._db = None
        self._initialized = False
        self._schema_info = {}

    def is_initialized(self) -> bool:
        """Return True if the backend has been initialized."""
        return self._initialized

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
            columns.append(f'"{column.name}" {ddl_type}{nullable}{primary}')

        if is_composite_pk:
            pk_clause = ", ".join(f'"{c}"' for c in pk_cols)
            columns.append(f"PRIMARY KEY ({pk_clause})")

        columns_sql = ", ".join(columns)
        return f"CREATE TABLE IF NOT EXISTS {table.name} ({columns_sql})"

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
