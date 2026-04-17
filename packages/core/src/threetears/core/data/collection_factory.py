"""dynamic collection factory for creating BaseCollection subclasses from TableDef."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.serialization import deserialize_from_json, serialize_to_json
from threetears.observe import get_logger

from threetears.core.data.schema import TableDef

log = get_logger(__name__)

_COLUMN_TYPE_TO_PYTHON: dict[str, type] = {
    "text": str,
    "integer": int,
    "bigint": int,
    "boolean": bool,
    "timestamp": datetime,
    "uuid": UUID,
    "jsonb": dict,
    "decimal": Decimal,
    "bytea": bytes,
}


def _build_field_types(table_def: TableDef) -> dict[str, type]:
    """build field_types mapping from TableDef columns for deserialization.

    maps each column name to its corresponding Python type using
    the column_type string from the TableDef definition.

    :param table_def: table definition with column metadata
    :ptype table_def: TableDef
    :return: mapping of column name to Python type
    :rtype: dict[str, type]
    """
    result: dict[str, type] = {}
    for col in table_def.columns:
        python_type = _COLUMN_TYPE_TO_PYTHON.get(col.column_type, str)
        result[col.name] = python_type
    return result


def _build_fetch_sql(table_name: str, pk_column: str) -> str:
    """build SELECT SQL for fetching single entity by primary key.

    :param table_name: name of database table
    :ptype table_name: str
    :param pk_column: primary key column name
    :ptype pk_column: str
    :return: parameterized SELECT SQL string
    :rtype: str
    """
    result = f"SELECT * FROM {table_name} WHERE {pk_column} = $1"
    return result


def _build_upsert_sql(table_name: str, columns: list[str], pk_column: str) -> str:
    """build INSERT ON CONFLICT UPDATE SQL for persisting entity data.

    generates parameterized INSERT with ON CONFLICT DO UPDATE for all
    non-primary-key columns. parameter placeholders use $N style.

    :param table_name: name of database table
    :ptype table_name: str
    :param columns: list of all column names
    :ptype columns: list[str]
    :param pk_column: primary key column name
    :ptype pk_column: str
    :return: parameterized upsert SQL string
    :rtype: str
    """
    placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
    cols_str = ", ".join(columns)
    update_cols = [c for c in columns if c != pk_column]
    set_parts = [f"{c} = EXCLUDED.{c}" for c in update_cols]
    set_clause = ", ".join(set_parts)
    result = (
        f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders}) "
        f"ON CONFLICT ({pk_column}) DO UPDATE SET {set_clause}"
    )
    return result


def _build_delete_sql(table_name: str, pk_column: str) -> str:
    """build DELETE SQL for removing entity by primary key.

    :param table_name: name of database table
    :ptype table_name: str
    :param pk_column: primary key column name
    :ptype pk_column: str
    :return: parameterized DELETE SQL string
    :rtype: str
    """
    result = f"DELETE FROM {table_name} WHERE {pk_column} = $1"
    return result


def _build_sa_metadata(table_def: TableDef) -> Any:
    """build SQLAlchemy MetaData for L1 cache initialization.

    maps TableDef column types to SQLAlchemy column types and creates
    a SQLAlchemy Table definition suitable for SQLiteBackend.initialize().

    :param table_def: table definition with column metadata
    :ptype table_def: TableDef
    :return: SQLAlchemy MetaData containing table definition
    :rtype: Any
    """
    from sqlalchemy import (
        Boolean,
        Column,
        DateTime,
        Integer,
        MetaData,
        Numeric,
        String,
        Table,
    )
    from sqlalchemy.dialects.postgresql import BYTEA, JSONB
    from sqlalchemy.sql.sqltypes import UUID as SAUuid

    sa_type_map: dict[str, Any] = {
        "text": String(255),
        "integer": Integer(),
        "bigint": Integer(),
        "boolean": Boolean(),
        "timestamp": DateTime(),
        "uuid": SAUuid(),
        "jsonb": JSONB(),
        "decimal": Numeric(),
        "bytea": BYTEA(),
    }

    metadata = MetaData()
    sa_columns: list[Column[Any]] = []
    for col in table_def.columns:
        sa_type = sa_type_map.get(col.column_type, String(255))
        sa_columns.append(Column(col.name, sa_type, primary_key=col.primary_key, nullable=col.nullable))
    Table(table_def.name, metadata, *sa_columns)
    return metadata


def _find_pk_column(table_def: TableDef) -> str:
    """find primary key column name from table definition.

    :param table_def: table definition with column metadata
    :ptype table_def: TableDef
    :return: primary key column name, defaults to "id" if none marked
    :rtype: str
    """
    for col in table_def.columns:
        if col.primary_key:
            return col.name
    result = "id"
    return result


def create_dynamic_collection(
    table_def: TableDef,
    registry: CollectionRegistry,
    config: CoreConfig,
    nats_client: Any = None,
) -> BaseCollection[Any]:
    """create a BaseCollection subclass dynamically from a TableDef.

    generates the required abstract method implementations:
    - _fetch_from_postgres: SELECT * WHERE pk = $1
    - _save_to_postgres: INSERT ... ON CONFLICT DO UPDATE
    - _delete_from_postgres: DELETE WHERE pk = $1
    - _serialize: serialize_to_json
    - _deserialize: deserialize_from_json with field_types from TableDef

    initializes the L1 backend with SQLAlchemy metadata derived from
    the TableDef if an L1 backend is configured in the registry.

    :param table_def: complete table definition for collection
    :ptype table_def: TableDef
    :param registry: collection registry for dependency resolution
    :ptype registry: CollectionRegistry
    :param config: core configuration for flush strategy
    :ptype config: CoreConfig
    :param nats_client: optional NATS client for L2 caching
    :ptype nats_client: Any
    :return: instantiated BaseCollection for the given table; parameterized
        over the dynamically generated entity class so the concrete type is
        only known at runtime (hence ``BaseCollection[Any]``)
    :rtype: BaseCollection[Any]
    """
    tbl_name = table_def.name
    pk_column = _find_pk_column(table_def)
    field_types = _build_field_types(table_def)
    column_names = [col.name for col in table_def.columns]
    fetch_sql = _build_fetch_sql(tbl_name, pk_column)
    upsert_sql = _build_upsert_sql(tbl_name, column_names, pk_column)
    delete_sql = _build_delete_sql(tbl_name, pk_column)
    sa_metadata = _build_sa_metadata(table_def)

    # initialize L1 backend with table schema if available
    l1_backend = registry.get_l1_backend(tbl_name)
    if l1_backend is not None and hasattr(l1_backend, "initialize"):
        l1_backend.initialize(sa_metadata)

    class DynamicEntity(BaseEntity):
        """dynamically generated entity for table."""

        _primary_key_field: str = pk_column

    class DynamicCollection(BaseCollection[DynamicEntity]):
        """dynamically generated collection for table."""

        _primary_key_column: str = pk_column

        @property
        def table_name(self) -> str:
            """return database table name."""
            return tbl_name

        @property
        def entity_class(self) -> type[DynamicEntity]:
            """return entity class for this collection."""
            return DynamicEntity

        async def _fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
            """fetch single entity from L3 by primary key.

            :param entity_id: primary key value
            :ptype entity_id: Any
            :return: entity data as dict, or None if not found
            :rtype: dict[str, Any] | None
            """
            pool = self._l3_pool
            if pool is None:
                return None
            rows = await pool.fetch(fetch_sql, entity_id)
            result = rows[0] if rows else None
            return result

        async def _save_to_postgres(self, data: dict[str, Any], original_timestamp: datetime | None = None) -> int:
            """persist entity data to L3 via upsert.

            when original_timestamp is provided, performs optimistic
            concurrency check before writing.

            :param data: entity data to persist
            :ptype data: dict[str, Any]
            :param original_timestamp: expected date_updated for concurrency check
            :ptype original_timestamp: datetime | None
            :return: number of rows affected
            :rtype: int
            """
            pool = self._l3_pool
            if pool is None:
                return 0
            values = [data.get(col, None) for col in column_names]
            result_str = await pool.execute(upsert_sql, *values)
            result = 1 if result_str else 0
            return result

        async def _delete_from_postgres(self, entity_id: Any) -> None:
            """delete entity from L3 by primary key.

            :param entity_id: primary key value
            :ptype entity_id: Any
            """
            pool = self._l3_pool
            if pool is None:
                return
            await pool.execute(delete_sql, entity_id)

        def _serialize(self, data: dict[str, Any]) -> bytes:
            """serialize entity data to JSON bytes for L2 cache.

            :param data: entity data dictionary
            :ptype data: dict[str, Any]
            :return: JSON-encoded bytes
            :rtype: bytes
            """
            result = serialize_to_json(data)
            return result

        def _deserialize(self, data: bytes) -> dict[str, Any]:
            """deserialize JSON bytes from L2 cache to entity data.

            :param data: JSON-encoded bytes
            :ptype data: bytes
            :return: entity data dictionary with typed values
            :rtype: dict[str, Any]
            """
            result = deserialize_from_json(data, field_types)
            return result

    DynamicEntity.__name__ = f"{tbl_name.title().replace('_', '')}Entity"
    DynamicEntity.__qualname__ = DynamicEntity.__name__
    DynamicCollection.__name__ = f"{tbl_name.title().replace('_', '')}Collection"
    DynamicCollection.__qualname__ = DynamicCollection.__name__

    collection = DynamicCollection(
        registry=registry,
        config=config,
        nats_client=nats_client,
    )

    log.info(
        "created dynamic collection",
        extra={"extra_data": {"table": tbl_name, "pk_column": pk_column, "columns": column_names}},
    )

    return collection
