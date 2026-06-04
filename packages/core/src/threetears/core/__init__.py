"""threetears.core — three-tier caching framework.

Public API re-exports for convenient top-level imports.
"""

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

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
from threetears.core.namespaces import (
    PLURAL_PREFIX_BY_NAMESPACE_TYPE,
    build_namespace_name,
    sanitize_segment,
)
from threetears.core.pagination import (
    CursorError,
    Keyset,
    Page,
    decode_cursor,
    encode_cursor,
)
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
    "CursorError",
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
    "PLURAL_PREFIX_BY_NAMESPACE_TYPE",
    "PathSandbox",
    "Sandbox",
    "SandboxDecision",
    "SandboxDenied",
    "TableDef",
    "UnknownFormatError",
    "atomic_write",
    "build_create_index_sql",
    "build_create_table_sql",
    "build_namespace_name",
    "create_dynamic_collection",
    "deserialize_from_json",
    "handler_for",
    "register_handler",
    "sanitize_segment",
    "serialize_to_json",
]
