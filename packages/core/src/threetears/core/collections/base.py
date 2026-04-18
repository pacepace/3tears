"""Base collection with three-tier caching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from threetears.core._bridge import fire_and_forget, sync_await
from threetears.core.cache import MISSING
from threetears.core.collections.flush import FlushStrategy, WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError
from threetears.observe import get_logger, traced

__all__ = ["BaseCollection", "EntityT"]

log = get_logger(__name__)

EntityT = TypeVar("EntityT", bound=BaseEntity)


class BaseCollection(ABC, Generic[EntityT]):
    """abstract base collection with three-tier caching (L1 -> L2 -> L3).

    :cvar primary_key_column: name of the primary-key column; part of
        the collection-entity contract (siblings read it during CAS
        and cache writes). subclasses override with their table's PK.
    :ivar l3_pool: asyncpg-compatible connection pool bound to the
        agent's L3 schema (or ``None`` when the collection is configured
        without L3, e.g. unit tests using only L1+L2). this is the
        public extension seam for ad-hoc SQL: subclasses and external
        callers (hub endpoints implementing keyset pagination, JOINs,
        bulk queries) may invoke ``await self.l3_pool.fetch(...)`` /
        ``execute(...)`` / ``fetchrow(...)`` directly when the query
        cannot be expressed through the Collection API. prefer the
        collection methods (``get``, ``save_entity``, ``delete``,
        ``__getitem__``, ``__setitem__``) for standard CRUD; drop to
        raw SQL only when no Collection method fits. the pool is
        shared across every collection bound to the same agent schema
        (resolved through :class:`CollectionRegistry`); callers MUST
        NOT call ``close()`` on it from a collection method or in any
        per-request flow -- the pool's lifecycle is owned by the
        process that constructed the registry. ``None`` is a valid
        value: callers that need to operate without an L3 pool must
        guard with ``if self.l3_pool is not None`` rather than
        assuming presence.
    """

    primary_key_column: str = "id"

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        self._registry = registry
        self._config = config
        self._nats_client = nats_client
        self._write_buffer = write_buffer
        self._flush_strategy = FlushStrategy(config.collection_flush)
        self._flush_tables = frozenset(t.strip() for t in config.collection_flush_tables.split(",") if t.strip())
        # Resolve L1 and L3 from registry
        self._l1 = registry.get_l1_backend(self.table_name)
        self.l3_pool = registry.get_l3_pool(self.table_name)
        # Auto-register
        registry.register(self)

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Return the database table name for this collection."""
        ...

    @property
    @abstractmethod
    def entity_class(self) -> type[EntityT]:
        """Return the entity class for this collection."""
        ...

    @abstractmethod
    async def _fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None: ...

    @abstractmethod
    async def _save_to_postgres(self, data: dict[str, Any], original_timestamp: datetime | None = None) -> int: ...

    @abstractmethod
    async def _delete_from_postgres(self, entity_id: Any) -> None: ...

    @abstractmethod
    def _serialize(self, data: dict[str, Any]) -> bytes: ...

    @abstractmethod
    def _deserialize(self, data: bytes) -> dict[str, Any]: ...

    # --- L1 cache (sync, for BaseEntity) ---
    #
    # these five methods are the synchronous cache-access API shared
    # with BaseEntity (and, transitively, the subclasses' __getitem__
    # and __setitem__ paths). they are public because BaseEntity and
    # some subclass-level collections (agent-tools ContextItems,
    # agent-memory MemoriesCollection) call them across the class
    # boundary -- the contract is "if you hold a collection reference
    # you may read/write its L1 through these five methods". mutations
    # always return a bool so the entity can fall back to in-memory
    # ``_changes`` when L1 is absent.

    def get_field_sync(self, entity_id: Any, field: str) -> Any:
        """read one column synchronously from the L1 cache.

        :param entity_id: primary-key value identifying the row
        :ptype entity_id: Any
        :param field: column name to read
        :ptype field: str
        :return: column value, or ``MISSING`` sentinel when L1 is
            absent, the row is not cached, or the column is absent
            from the cached row
        :rtype: Any
        """
        if self._l1 is None:
            return MISSING
        row = self._l1.select_by_id(self.table_name, str(entity_id), self.primary_key_column)
        if row is None:
            return MISSING
        return row.get(field, MISSING)

    def set_field_sync(self, entity_id: Any, field: str, value: Any) -> bool:
        """write one column synchronously into the L1 cache.

        :param entity_id: primary-key value identifying the row
        :ptype entity_id: Any
        :param field: column name to write
        :ptype field: str
        :param value: new value for the column
        :ptype value: Any
        :return: true on successful write, false when L1 is absent or
            the row is not yet cached (caller must fall back to the
            in-memory change buffer)
        :rtype: bool
        """
        if self._l1 is None:
            return False
        row = self._l1.select_by_id(self.table_name, str(entity_id), self.primary_key_column)
        if row is None:
            return False
        row[field] = value
        self._l1.upsert(self.table_name, row, self.primary_key_column)
        return True

    def get_row_sync(self, entity_id: Any) -> dict[str, Any] | None:
        """read the full cached row for an entity, synchronously.

        :param entity_id: primary-key value identifying the row
        :ptype entity_id: Any
        :return: row dict, or ``None`` when L1 is absent or the row
            is not cached
        :rtype: dict[str, Any] | None
        """
        if self._l1 is None:
            return None
        return self._l1.select_by_id(self.table_name, str(entity_id), self.primary_key_column)  # type: ignore[no-any-return]

    def write_to_cache_sync(self, data: dict[str, Any], primary_key: str | None = None) -> bool:
        """upsert a full row into the L1 cache, synchronously.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param primary_key: override the collection's PK column for
            this write; defaults to the collection's
            ``primary_key_column``
        :ptype primary_key: str | None
        :return: true on successful write, false when L1 is absent
        :rtype: bool
        """
        if self._l1 is None:
            return False
        pk = primary_key or self.primary_key_column
        self._l1.upsert(self.table_name, data, pk)
        return True

    def exists_in_cache_sync(self, entity_id: Any) -> bool:
        """true iff the given entity is present in the L1 cache.

        :param entity_id: primary-key value identifying the row
        :ptype entity_id: Any
        :return: presence flag; false when L1 is absent
        :rtype: bool
        """
        if self._l1 is None:
            return False
        row = self._l1.select_by_id(self.table_name, str(entity_id), self.primary_key_column)
        return row is not None

    # --- L2 cache (NATS KV, async) ---

    def _l2_bucket(self) -> str:
        return self._nats_client.bucket_name("collections")  # type: ignore[no-any-return]

    def _l2_key(self, entity_id: Any) -> str:
        return f"{self.table_name}.{entity_id}"

    async def _get_from_l2(self, entity_id: Any) -> dict[str, Any] | None:
        if self._nats_client is None:
            return None
        try:
            raw = await self._nats_client.get(self._l2_bucket(), self._l2_key(entity_id))
            if raw is None:
                return None
            return self._deserialize(raw)
        except Exception as exc:
            log.warning(
                "L2 cache read failed",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "error": str(exc),
                    }
                },
            )
            return None

    async def _save_to_l2(self, entity_id: Any, data: dict[str, Any]) -> bool:
        if self._nats_client is None:
            return False
        try:
            return await self._nats_client.put(self._l2_bucket(), self._l2_key(entity_id), self._serialize(data))  # type: ignore[no-any-return]
        except Exception as exc:
            log.warning(
                "L2 cache write failed",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "error": str(exc),
                    }
                },
            )
            return False

    async def _delete_from_l2(self, entity_id: Any) -> bool:
        if self._nats_client is None:
            return False
        try:
            return await self._nats_client.delete(self._l2_bucket(), self._l2_key(entity_id))  # type: ignore[no-any-return]
        except Exception as exc:
            log.warning(
                "L2 cache delete failed",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "error": str(exc),
                    }
                },
            )
            return False

    # --- Subscript access (sync, transparent pull-through) ---

    def _ensure_in_l1(self, entity_id: Any) -> dict[str, Any] | None:
        """Pull entity into L1 via L2/L3 if not already cached. Sync.

        Returns the row data if found, None if not found in any tier.
        """
        if self._l1 is not None:
            row = self._l1.select_by_id(self.table_name, str(entity_id), self.primary_key_column)
            if row is not None:
                return row  # type: ignore[no-any-return]
        return sync_await(self._pull_through(entity_id))

    async def _pull_through(self, entity_id: Any) -> dict[str, Any] | None:
        """Async pull-through: L2 -> L1, then L3 -> L1+L2. Returns the data or None."""
        l2_data = await self._get_from_l2(entity_id)
        if l2_data is not None:
            if self._l1 is not None:
                self._l1.upsert(self.table_name, l2_data, self.primary_key_column)
            return l2_data
        pg_data = await self._fetch_from_postgres(entity_id)
        if pg_data is not None:
            if self._l1 is not None:
                self._l1.upsert(self.table_name, pg_data, self.primary_key_column)
            await self._save_to_l2(entity_id, pg_data)
            return pg_data
        return None

    def _resolve_row(self, entity_id: Any) -> dict[str, Any]:
        """Get row from L1, pulling through L2/L3 on miss. Raises KeyError if not found."""
        row = self.get_row_sync(entity_id)
        if row is not None:
            return row
        data = self._ensure_in_l1(entity_id)
        if data is None:
            raise KeyError(f"{self.table_name}[{entity_id!r}]: entity not found")
        # If L1 exists, re-read from it (ensure_in_l1 populated it)
        if self._l1 is not None:
            row = self.get_row_sync(entity_id)
            if row is not None:
                return row
        # No L1 — return the data directly from pull-through
        return data

    def __getitem__(self, key: Any) -> Any:
        """Subscript read with transparent three-tier pull-through.

        collection[entity_id]          -> EntityT
        collection[entity_id, "field"] -> field value

        On L1 miss, transparently pulls data through L2/L3 into L1
        via a background event loop. Raises KeyError only if the entity
        doesn't exist in any tier.
        """
        if isinstance(key, tuple):
            entity_id, field = key
            result = self.get_field_sync(entity_id, field)
            if result is MISSING:
                row = self._resolve_row(entity_id)
                result = row.get(field, MISSING)
                if result is MISSING:
                    raise KeyError(f"{self.table_name}[{entity_id!r}, {field!r}]: field not found")
            return result
        entity_id = key
        row = self._resolve_row(entity_id)
        entity = self.entity_class(row, is_new=False, collection=self)
        entity.original_date_updated = row.get("date_updated")
        return entity

    def __setitem__(self, key: Any, value: Any) -> None:
        """Subscript write with three-tier propagation.

        collection[entity_id] = data_dict       -> write full entity
        collection[entity_id, "field"] = value   -> write single field

        Writes to L1 synchronously. L2 and L3 writes are non-blocking
        (fire-and-forget on the background event loop). L3 writes only
        happen if flush strategy is ALWAYS; otherwise the change is
        buffered for later flush.
        """
        if isinstance(key, tuple):
            entity_id, field = key
            self.set_field_sync(entity_id, field, value)
            row = self.get_row_sync(entity_id)
            if row is not None:
                self._propagate_write(entity_id, row)
        else:
            entity_id = key
            if not isinstance(value, dict):
                raise TypeError(f"collection[id] = value requires a dict, got {type(value).__name__}")
            self.write_to_cache_sync(value)
            self._propagate_write(entity_id, value)

    def _propagate_write(self, entity_id: Any, data: dict[str, Any]) -> None:
        """Non-blocking propagation of a write to L2 and optionally L3."""
        fire_and_forget(self._async_propagate_write(entity_id, dict(data)))

    async def _async_propagate_write(self, entity_id: Any, data: dict[str, Any]) -> None:
        """Async write propagation: always L2, conditionally L3, always signal."""
        now = datetime.now(UTC)
        data["date_updated"] = now

        # Always update L1 with the new timestamp
        if self._l1 is not None:
            self._l1.upsert(self.table_name, data, self.primary_key_column)

        # Always propagate to L2
        await self._save_to_l2(entity_id, data)

        # Signal other pods to evict stale L1
        await self._publish_invalidation(entity_id)

        # L3: immediate or deferred based on flush strategy
        should_defer = (
            self._flush_strategy != FlushStrategy.ALWAYS
            and self.table_name in self._flush_tables
            and self._write_buffer is not None
        )
        if should_defer:
            assert self._write_buffer is not None
            await self._write_buffer.add(self.table_name, entity_id, data)
        else:
            try:
                await self._save_to_postgres(data)
            except Exception as exc:
                log.error(
                    "Background L3 write failed",
                    extra={
                        "extra_data": {
                            "entity_id": str(entity_id),
                            "table": self.table_name,
                            "error": str(exc),
                        }
                    },
                )

    def __contains__(self, entity_id: Any) -> bool:
        """Check if entity is in L1 cache."""
        return self.exists_in_cache_sync(entity_id)

    # --- Cache coherence signaling ---

    async def _publish_invalidation(self, entity_id: Any) -> None:
        """Signal other pods to evict this entity from their L1 caches."""
        await self._registry.publish_invalidation(self._nats_client, self.table_name, entity_id)

    # --- Span attribute helpers (no-op when OTel unavailable) ---

    @staticmethod
    def _set_span_attr(key: str, value: Any) -> None:
        """Set an attribute on the current OTel span, if available."""
        try:
            from opentelemetry import trace as _trace

            span = _trace.get_current_span()
            span.set_attribute(key, value)
        except ImportError:
            pass

    def _set_span_table(self) -> None:
        """Set ``cache.table`` on the current span."""
        self._set_span_attr("cache.table", self.table_name)

    # --- Three-tier operations ---

    async def ensure(self, entity_id: Any) -> dict[str, Any] | None:
        """Pull entity into L1 cache through L2/L3 if not already present.

        Returns entity data dict if found (in any tier), None if not found
        anywhere. After ensure() returns data, subscript access is guaranteed
        to hit L1 (when L1 is available).
        """
        if self._l1 is not None:
            row: dict[str, Any] | None = self._l1.select_by_id(
                self.table_name,
                str(entity_id),
                self.primary_key_column,
            )
            if row is not None:
                return row
        data = await self._pull_through(entity_id)
        return data

    @traced()
    async def get(self, entity_id: Any) -> EntityT | None:
        """Three-tier read: L1 -> L2 -> L3, promote on miss."""
        self._set_span_table()
        data = await self.ensure(entity_id)
        if data is None:
            self._set_span_attr("cache.hit_tier", "miss")
            return None
        self._set_span_attr("cache.hit_tier", "L1+")
        result: EntityT = self.entity_class(data, is_new=False, collection=self)
        return result

    @traced()
    async def save_entity(self, entity: BaseEntity) -> None:
        """Save entity through three-tier write path."""
        self._set_span_table()
        entity_id = entity.id
        data = entity.to_dict()
        original_timestamp = getattr(entity, "original_date_updated", None)

        now = datetime.now(UTC)
        if entity.is_new:
            data["date_created"] = now
        if "date_updated" in data or not entity.is_new:
            data["date_updated"] = now

        # Convert aware -> naive for TIMESTAMP columns at database border
        for key, val in data.items():
            if isinstance(val, datetime) and val.tzinfo is not None:
                data[key] = val.replace(tzinfo=None)

        defer = (
            self._flush_strategy != FlushStrategy.ALWAYS
            and self.table_name in self._flush_tables
            and self._write_buffer is not None
        )

        if defer:
            if self._l1 is not None:
                self._l1.upsert(self.table_name, data, self.primary_key_column)
            await self._save_to_l2(entity_id, data)
            assert self._write_buffer is not None
            await self._write_buffer.add(self.table_name, entity_id, data)
            entity.mark_clean()
            entity.original_date_updated = data.get("date_updated")
        else:
            rows_affected = await self._save_to_postgres(data, original_timestamp)
            if rows_affected == 0:
                if entity.is_new:
                    raise RuntimeError(f"INSERT failed for {self.table_name} entity {entity_id}: 0 rows affected")
                raise ConcurrentModificationError(self.table_name, entity_id, original_timestamp or datetime.min)
            entity.mark_clean()
            entity.original_date_updated = data.get("date_updated")
            if self._l1 is not None:
                self._l1.upsert(self.table_name, data, self.primary_key_column)
            else:
                # No L1 backend: repopulate _changes so entity fields remain accessible
                object.__setattr__(entity, "_changes", dict(data))
            await self._save_to_l2(entity_id, data)

        await self._publish_invalidation(entity_id)

    async def persist_to_postgres(self, data: dict[str, Any]) -> int:
        """Used by flush_pending."""
        return await self._save_to_postgres(data)

    @traced()
    async def reload_entity(self, entity: BaseEntity) -> None:
        """Reload entity from L3."""
        self._set_span_table()
        entity_id = entity.id
        if self._write_buffer is not None:
            await self._write_buffer.remove(self.table_name, entity_id)
        data = await self._fetch_from_postgres(entity_id)
        if data is None:
            raise ValueError(f"Entity {entity_id} not found in storage")
        entity.set_data(data)
        entity.original_date_updated = data.get("date_updated")
        if self._l1 is not None:
            self._l1.upsert(self.table_name, data, self.primary_key_column)
        await self._save_to_l2(entity_id, data)
        await self._publish_invalidation(entity_id)

    @traced()
    async def delete(self, entity_id: Any) -> bool:
        """Delete entity from all tiers."""
        self._set_span_table()
        if self._write_buffer is not None:
            await self._write_buffer.remove(self.table_name, entity_id)
        await self._delete_from_postgres(entity_id)
        if self._l1 is not None:
            self._l1.delete_by_id(self.table_name, str(entity_id), self.primary_key_column)
        await self._delete_from_l2(entity_id)
        await self._publish_invalidation(entity_id)
        return True

    @traced()
    async def invalidate_cache(self, entity_id: Any) -> None:
        """Delete from L1 and L2, signal other pods."""
        self._set_span_table()
        if self._l1 is not None:
            self._l1.delete_by_id(self.table_name, str(entity_id), self.primary_key_column)
        await self._delete_from_l2(entity_id)
        await self._publish_invalidation(entity_id)

    def create(self, data: dict[str, Any]) -> EntityT:
        """Create new entity (not persisted until save)."""
        return self.entity_class(data, is_new=True, collection=self)
