# Import Mapping: MetaLLM -> threetears.core

This document maps MetaLLM internal imports to their `threetears.core` equivalents for migration.

## Top-level imports

| MetaLLM import | threetears.core import |
|---|---|
| `from src.data.entities.base import BaseEntity` | `from threetears.core import BaseEntity` |
| `from src.data.collections.base import BaseCollection` | `from threetears.core import BaseCollection` |
| `from src.data.collections.registry import CollectionRegistry` | `from threetears.core import CollectionRegistry` |
| `from src.data.exceptions import ConcurrentModificationError` | `from threetears.core import ConcurrentModificationError` |
| `from src.data.exceptions import DataLayerUnavailableError` | `from threetears.core import DataLayerUnavailableError` |
| `from src.config import Settings` (flush config subset) | `from threetears.core import CoreConfig, DefaultCoreConfig` |

## Cache layer

| MetaLLM import | threetears.core import |
|---|---|
| `from src.data.cache.sqlite import SQLiteCache` | `from threetears.core.cache.sqlite import SQLiteBackend` |
| `from src.data.cache import MISSING` | `from threetears.core.cache import MISSING` |
| `from src.data.cache import L1Backend` | `from threetears.core.cache import L1Backend` |

## Collections

| MetaLLM import | threetears.core import |
|---|---|
| `from src.data.collections.flush import FlushStrategy` | `from threetears.core.collections.flush import FlushStrategy` |
| `from src.data.collections.flush import WriteBuffer` | `from threetears.core.collections.flush import WriteBuffer` |
| `from src.data.collections.flush import flush_pending` | `from threetears.core.collections.flush import flush_pending` |

## Model mixins (new)

MetaLLM uses inline column definitions on every model. threetears.core provides optional mixins:

| MetaLLM pattern | threetears.core mixin |
|---|---|
| `user_id = mapped_column(UUID(as_uuid=True), primary_key=True)` | `UUIDPrimaryKeyMixin` (provides `id` column; override `_pk_column_name` for custom names) |
| `date_created = mapped_column(DateTime(timezone=True), ...)` + `date_updated = ...` | `TimestampMixin` (provides both with server defaults) |
| `is_deleted = mapped_column(Boolean, ...)` + `date_deleted = ...` | `SoftDeleteMixin` (provides both columns) |

Import: `from threetears.core.models import UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin`

## Configuration mapping

MetaLLM `Settings` fields that map to `CoreConfig` protocol:

| MetaLLM `Settings` field | `CoreConfig` attribute |
|---|---|
| `COLLECTION_FLUSH_STRATEGY` | `collection_flush` |
| `COLLECTION_FLUSH_INTERVAL` | `collection_flush_interval` |
| `COLLECTION_FLUSH_TABLES` | `collection_flush_tables` |

## Migration notes

1. `CoreConfig` is a `Protocol` -- existing Settings classes satisfy it without inheritance if they have the right attributes.
2. `BaseEntity` and `BaseCollection` are functionally identical to their MetaLLM counterparts; only the import path changes.
3. The model mixins are optional. Existing MetaLLM models can continue using inline column definitions.
4. `SQLiteBackend` replaces `SQLiteCache` with an instance-based API (no classmethods).
