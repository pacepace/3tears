"""Collection primitives.

``DerivedCollection`` is resolved lazily: it is the one collection type that
takes a NATS distributed lock at runtime, so importing it eagerly would pull
the NATS client into every consumer of ``BaseCollection`` — including L1-only
ones that never leave SQLite. Everything else here is eager as before.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from threetears.core.collections.asyncpg_init import (
    init_connection,
    register_jsonb_text_codec,
)
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.bucket import (
    COLLECTIONS_BUCKET_SUFFIX,
    bind_collections_bucket,
)
from threetears.core.collections.durable_store import DurableStoreCollection
from threetears.core.collections.flush import FlushStrategy, WriteBuffer, flush_pending
from threetears.core.collections.merge import repoint_user_rows
from threetears.core.collections.registry import (
    CacheInvalidationMessage,
    CollectionRegistry,
)
from threetears.core.collections.salience import apply_salience_decay
from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    VECTOR_TYPE,
    Column,
    PartitionEnforcementError,
    SchemaBackedCollection,
    TableSchema,
    encode_jsonb,
    spans_partitions,
)
from threetears.core.serialization import deserialize_from_json, serialize_to_json

if TYPE_CHECKING:
    from threetears.core.collections.derived import DerivedCollection


def __getattr__(name: str) -> object:
    """Resolve ``DerivedCollection`` on first access (PEP 562).

    It holds the only runtime NATS dependency in this package; deferring it is
    what keeps ``from threetears.core.collections import BaseCollection`` free
    of the client.
    """
    if name != "DerivedCollection":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module("threetears.core.collections.derived"), name)
    globals()[name] = value  # cache: __getattr__ will not fire again
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), "DerivedCollection"})


__all__ = [
    "BOOL_TYPE",
    "BYTES_TYPE",
    "COLLECTIONS_BUCKET_SUFFIX",
    "BaseCollection",
    "CacheInvalidationMessage",
    "CollectionRegistry",
    "Column",
    "DATETIMETZ_TYPE",
    "DerivedCollection",
    "DurableStoreCollection",
    "FlushStrategy",
    "INT_TYPE",
    "JSONB_TYPE",
    "PartitionEnforcementError",
    "STRING_TYPE",
    "SchemaBackedCollection",
    "TableSchema",
    "UUID_TYPE",
    "VECTOR_TYPE",
    "WriteBuffer",
    "apply_salience_decay",
    "bind_collections_bucket",
    "deserialize_from_json",
    "encode_jsonb",
    "flush_pending",
    "init_connection",
    "register_jsonb_text_codec",
    "repoint_user_rows",
    "serialize_to_json",
    "spans_partitions",
]
