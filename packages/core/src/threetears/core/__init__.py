"""threetears.core — three-tier caching framework.

Public API re-exports for convenient top-level imports.
"""

__version__ = "0.5.0"

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig, DefaultCoreConfig
from threetears.core.data.collection_factory import create_dynamic_collection
from threetears.core.data.migrations import MigrationRunner
from threetears.core.data.schema import ColumnDef, ForeignKeyDef, IndexDef, TableDef
from threetears.core.data.sql_builder import build_create_index_sql, build_create_table_sql
from threetears.core.data.store import DataStore
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError, DataLayerUnavailableError
from threetears.core.serialization import deserialize_from_json, serialize_to_json

__all__ = [
    "BaseCollection",
    "BaseEntity",
    "CollectionRegistry",
    "ColumnDef",
    "ConcurrentModificationError",
    "CoreConfig",
    "DataLayerUnavailableError",
    "DataStore",
    "DefaultCoreConfig",
    "ForeignKeyDef",
    "IndexDef",
    "MigrationRunner",
    "TableDef",
    "build_create_index_sql",
    "build_create_table_sql",
    "create_dynamic_collection",
    "deserialize_from_json",
    "serialize_to_json",
]
