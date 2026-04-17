"""threetears.core — three-tier caching framework.

Public API re-exports for convenient top-level imports.
"""

__version__ = "0.5.0"

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig, DefaultCoreConfig
from threetears.core.coordination import (
    KVLease,
    LeaseHandle,
    LeaseLost,
    LeaseTimeout,
    LeaseUnavailable,
)
from threetears.core.data.collection_factory import create_dynamic_collection
from threetears.core.data.migrations import MigrationRunner
from threetears.core.data.schema import ColumnDef, ForeignKeyDef, IndexDef, TableDef
from threetears.core.data.sql_builder import build_create_index_sql, build_create_table_sql
from threetears.core.data.store import DataStore
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError, DataLayerUnavailableError
from threetears.core.security import (
    PathSandbox,
    Sandbox,
    SandboxDecision,
    SandboxDenied,
)
from threetears.core.serialization import (
    FormatHandler,
    UnknownFormatError,
    deserialize_from_json,
    handler_for,
    register_handler,
    serialize_to_json,
)
from threetears.core.utils.atomic_write import atomic_write

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
    "FormatHandler",
    "IndexDef",
    "KVLease",
    "LeaseHandle",
    "LeaseLost",
    "LeaseTimeout",
    "LeaseUnavailable",
    "MigrationRunner",
    "PathSandbox",
    "Sandbox",
    "SandboxDecision",
    "SandboxDenied",
    "TableDef",
    "UnknownFormatError",
    "atomic_write",
    "build_create_index_sql",
    "build_create_table_sql",
    "create_dynamic_collection",
    "deserialize_from_json",
    "handler_for",
    "register_handler",
    "serialize_to_json",
]
