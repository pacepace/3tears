"""dynamic data layer for creating tables and collections at runtime."""

from __future__ import annotations

from threetears.core.data.collection_factory import create_dynamic_collection
from threetears.core.data.migrations import MigrationRunner
from threetears.core.data.schema import ColumnDef, ForeignKeyDef, IndexDef, TableDef
from threetears.core.data.sql_builder import build_create_index_sql, build_create_table_sql
from threetears.core.data.store import DataStore

__all__ = [
    "ColumnDef",
    "DataStore",
    "ForeignKeyDef",
    "IndexDef",
    "MigrationRunner",
    "TableDef",
    "build_create_index_sql",
    "build_create_table_sql",
    "create_dynamic_collection",
]
