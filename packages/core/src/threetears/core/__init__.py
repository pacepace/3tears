"""threetears.core — three-tier caching framework.

Public API re-exports for convenient top-level imports.
"""

from __future__ import annotations

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING

try:
    __version__ = _version("3tears")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

# lazy public API (PEP 562). the package namespace no longer imports its
# implementation modules eagerly: importing this package (or any of its
# submodules) costs only this file, and each public attribute resolves
# its defining module on first access. the TYPE_CHECKING block carries
# the real imports so mypy and IDEs see the full statically-typed API;
# the _LAZY map is the runtime equivalent. the three-way agreement
# between __all__, _LAZY, and the TYPE_CHECKING block is pinned by the
# package's lazy-surface consistency test.
# decision record: docs/separate-concerns-decisions.md (hand-rolled
# PEP 562 over lazy_loader -- zero added runtime deps, no stub drift).
if TYPE_CHECKING:
    from threetears.core.collections.base import BaseCollection
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import CoreConfig, DefaultCoreConfig
    from threetears.core.coordination import KVLease, LeaseHandle, LeaseLost, LeaseTimeout, LeaseUnavailable
    from threetears.core.data.collection_factory import create_dynamic_collection
    from threetears.core.data.migrations import MigrationRunner
    from threetears.core.data.schema import ColumnDef, ForeignKeyDef, IndexDef, TableDef
    from threetears.core.data.sql_builder import build_create_index_sql, build_create_table_sql
    from threetears.core.data.store import DataStore
    from threetears.core.entities.base import BaseEntity
    from threetears.core.exceptions import ConcurrentModificationError, DataLayerUnavailableError
    from threetears.core.http_client import TracedHttpClient, UpstreamHttpError
    from threetears.core.namespaces import PLURAL_PREFIX_BY_NAMESPACE_TYPE, build_namespace_name, sanitize_segment
    from threetears.core.pagination import CursorError, Keyset, Page, decode_cursor, encode_cursor
    from threetears.core.security import PathSandbox, Sandbox, SandboxDecision, SandboxDenied
    from threetears.core.task_registry import KeyedTaskRegistry
    from threetears.core.serialization import (
        FormatHandler,
        UnknownFormatError,
        deserialize_from_json,
        handler_for,
        register_handler,
        serialize_to_json,
    )
    from threetears.core.utils.atomic_write import atomic_write

# public attribute -> (defining module, attribute name in that module)
_LAZY: dict[str, tuple[str, str]] = {
    "BaseCollection": ("threetears.core.collections.base", "BaseCollection"),
    "BaseEntity": ("threetears.core.entities.base", "BaseEntity"),
    "CollectionRegistry": ("threetears.core.collections.registry", "CollectionRegistry"),
    "ColumnDef": ("threetears.core.data.schema", "ColumnDef"),
    "ConcurrentModificationError": ("threetears.core.exceptions", "ConcurrentModificationError"),
    "CoreConfig": ("threetears.core.config", "CoreConfig"),
    "CursorError": ("threetears.core.pagination", "CursorError"),
    "DataLayerUnavailableError": ("threetears.core.exceptions", "DataLayerUnavailableError"),
    "DataStore": ("threetears.core.data.store", "DataStore"),
    "DefaultCoreConfig": ("threetears.core.config", "DefaultCoreConfig"),
    "ForeignKeyDef": ("threetears.core.data.schema", "ForeignKeyDef"),
    "FormatHandler": ("threetears.core.serialization", "FormatHandler"),
    "IndexDef": ("threetears.core.data.schema", "IndexDef"),
    "KVLease": ("threetears.core.coordination", "KVLease"),
    "KeyedTaskRegistry": ("threetears.core.task_registry", "KeyedTaskRegistry"),
    "Keyset": ("threetears.core.pagination", "Keyset"),
    "LeaseHandle": ("threetears.core.coordination", "LeaseHandle"),
    "LeaseLost": ("threetears.core.coordination", "LeaseLost"),
    "LeaseTimeout": ("threetears.core.coordination", "LeaseTimeout"),
    "LeaseUnavailable": ("threetears.core.coordination", "LeaseUnavailable"),
    "MigrationRunner": ("threetears.core.data.migrations", "MigrationRunner"),
    "PLURAL_PREFIX_BY_NAMESPACE_TYPE": ("threetears.core.namespaces", "PLURAL_PREFIX_BY_NAMESPACE_TYPE"),
    "Page": ("threetears.core.pagination", "Page"),
    "PathSandbox": ("threetears.core.security", "PathSandbox"),
    "Sandbox": ("threetears.core.security", "Sandbox"),
    "SandboxDecision": ("threetears.core.security", "SandboxDecision"),
    "SandboxDenied": ("threetears.core.security", "SandboxDenied"),
    "TableDef": ("threetears.core.data.schema", "TableDef"),
    "TracedHttpClient": ("threetears.core.http_client", "TracedHttpClient"),
    "UnknownFormatError": ("threetears.core.serialization", "UnknownFormatError"),
    "UpstreamHttpError": ("threetears.core.http_client", "UpstreamHttpError"),
    "atomic_write": ("threetears.core.utils.atomic_write", "atomic_write"),
    "build_create_index_sql": ("threetears.core.data.sql_builder", "build_create_index_sql"),
    "build_create_table_sql": ("threetears.core.data.sql_builder", "build_create_table_sql"),
    "build_namespace_name": ("threetears.core.namespaces", "build_namespace_name"),
    "create_dynamic_collection": ("threetears.core.data.collection_factory", "create_dynamic_collection"),
    "decode_cursor": ("threetears.core.pagination", "decode_cursor"),
    "deserialize_from_json": ("threetears.core.serialization", "deserialize_from_json"),
    "encode_cursor": ("threetears.core.pagination", "encode_cursor"),
    "handler_for": ("threetears.core.serialization", "handler_for"),
    "register_handler": ("threetears.core.serialization", "register_handler"),
    "sanitize_segment": ("threetears.core.namespaces", "sanitize_segment"),
    "serialize_to_json": ("threetears.core.serialization", "serialize_to_json"),
}

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
    "KeyedTaskRegistry",
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
    "TracedHttpClient",
    "UnknownFormatError",
    "UpstreamHttpError",
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


def __getattr__(name: str) -> object:
    """resolve a public attribute from its defining module on first access.

    :param name: attribute name being resolved
    :ptype name: str
    :return: the resolved attribute (also cached in module globals so
        ``__getattr__`` fires at most once per name)
    :rtype: object
    :raises AttributeError: when ``name`` is not part of the public API
    """
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr = entry
    value: object = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """include lazy attributes in ``dir()`` output.

    :return: sorted union of materialized globals and lazy names
    :rtype: list[str]
    """
    return sorted(set(globals()) | set(_LAZY))
