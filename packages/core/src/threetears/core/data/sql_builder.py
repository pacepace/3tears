"""SQL builder for CREATE TABLE and CREATE INDEX statements from schema definitions."""

from __future__ import annotations

import re

from threetears.core.data.schema import IndexDef, TableDef

from threetears.observe import get_logger

__all__ = [
    "build_create_index_sql",
    "build_create_table_sql",
]

log = get_logger(__name__)

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

_COLUMN_TYPE_MAP: dict[str, str] = {
    "text": "TEXT",
    "integer": "INTEGER",
    "bigint": "BIGINT",
    "boolean": "BOOLEAN",
    "timestamp": "TIMESTAMP",
    "uuid": "UUID",
    "jsonb": "JSONB",
    "decimal": "DECIMAL",
    "bytea": "BYTEA",
}


def _validate_identifier(value: str, label: str) -> None:
    """validate SQL identifier against safe pattern.

    :param value: identifier string to validate
    :ptype value: str
    :param label: human-readable label for error messages
    :ptype label: str
    :raises ValueError: if identifier does not match allowed pattern
    """
    if not _IDENTIFIER_PATTERN.match(value):
        msg = f"{label} must match ^[a-z][a-z0-9_]*$, got: {value!r}"
        raise ValueError(msg)


def build_create_table_sql(table: TableDef) -> str:
    """build CREATE TABLE IF NOT EXISTS SQL from table definition.

    validates all identifiers against allowlist patterns. maps column_type
    strings to PostgreSQL types. adds PRIMARY KEY, NOT NULL, DEFAULT, and
    foreign key constraints.

    :param table: complete table definition with columns, indexes, and foreign keys
    :ptype table: TableDef
    :return: complete CREATE TABLE IF NOT EXISTS SQL statement
    :rtype: str
    :raises ValueError: if any identifier fails validation
    """
    _validate_identifier(table.name, "table name")

    parts: list[str] = []

    for col in table.columns:
        _validate_identifier(col.name, "column name")
        pg_type = _COLUMN_TYPE_MAP[col.column_type]
        col_parts = [f"    {col.name} {pg_type}"]
        if not col.nullable:
            col_parts.append("NOT NULL")
        if col.default is not None:
            col_parts.append(f"DEFAULT {col.default}")
        parts.append(" ".join(col_parts))

    pk_columns = [col.name for col in table.columns if col.primary_key]
    if pk_columns:
        pk_cols_str = ", ".join(pk_columns)
        parts.append(f"    PRIMARY KEY ({pk_cols_str})")

    for fk in table.foreign_keys:
        _validate_identifier(fk.name, "foreign key name")
        _validate_identifier(fk.references_table, "references table name")
        for fk_col in fk.columns:
            _validate_identifier(fk_col, "foreign key column name")
        for ref_col in fk.references_columns:
            _validate_identifier(ref_col, "references column name")
        fk_cols_str = ", ".join(fk.columns)
        ref_cols_str = ", ".join(fk.references_columns)
        fk_line = (
            f"    CONSTRAINT {fk.name} "
            f"FOREIGN KEY ({fk_cols_str}) "
            f"REFERENCES {fk.references_table} ({ref_cols_str}) "
            f"ON DELETE {fk.on_delete} "
            f"ON UPDATE {fk.on_update}"
        )
        parts.append(fk_line)

    body = ",\n".join(parts)
    sql = f"CREATE TABLE IF NOT EXISTS {table.name} (\n{body}\n)"
    return sql


def build_create_index_sql(table_name: str, index: IndexDef) -> str:
    """build CREATE INDEX IF NOT EXISTS SQL from index definition.

    :param table_name: name of table to create index on
    :ptype table_name: str
    :param index: index definition with name, columns, and uniqueness flag
    :ptype index: IndexDef
    :return: complete CREATE INDEX IF NOT EXISTS SQL statement
    :rtype: str
    :raises ValueError: if table name or any identifier fails validation
    """
    _validate_identifier(table_name, "table name")
    _validate_identifier(index.name, "index name")
    for col in index.columns:
        _validate_identifier(col, "index column name")

    unique_prefix = "UNIQUE " if index.unique else ""
    cols_str = ", ".join(index.columns)
    sql = f"CREATE {unique_prefix}INDEX IF NOT EXISTS {index.name} ON {table_name} ({cols_str})"
    return sql
