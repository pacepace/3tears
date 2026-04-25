"""base collection with three-tier caching.

composite primary keys are first-class. a subclass declares
``primary_key_column = "memory_id"`` for single-pk tables (the default
shape) or ``primary_key_column = ("conversation_id", "item_id")`` for
composite-pk tables. internally, every cache-keying path normalizes
the declared pk and caller-supplied id into a tuple via
:meth:`BaseCollection.normalize_pk`; the L1 (SQLite / DuckDB), L2
(NATS KV), and L3 (postgres) tiers all accept the tuple uniformly.
the invalidation wire envelope carries ``ids`` (plural, always an
array) matching the pk column order, so single-pk emits a length-1
array and composite-pk emits a length-N array.
"""

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

    :cvar primary_key_column: name of primary-key column (single-pk
        shape, ``str``) or tuple of column names in declared order
        (composite-pk shape, ``tuple[str, ...]``). part of the
        collection-entity contract (siblings read it during CAS and
        cache writes). subclasses override with their table's pk.
        :attr:`primary_key_columns` is the internal-use normalized
        tuple form; callers iterating pk columns MUST read that
        property rather than inspecting the attribute directly.
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

    primary_key_column: str | tuple[str, ...] = "id"

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

    @property
    def primary_key_columns(self) -> tuple[str, ...]:
        """normalize :attr:`primary_key_column` to tuple form.

        single-pk subclasses declare ``primary_key_column = "foo"`` and
        read ``("foo",)`` here; composite-pk subclasses declare
        ``primary_key_column = ("a", "b")`` and read the same tuple.
        every internal caller iterating pk columns uses this property.

        :return: tuple of pk column names in declared order
        :rtype: tuple[str, ...]
        """
        if isinstance(self.primary_key_column, tuple):
            return self.primary_key_column
        return (self.primary_key_column,)

    def normalize_pk(self, entity_id: Any) -> tuple[Any, ...]:
        """normalize caller-supplied id to tuple of pk values.

        single-pk collections accept either ``value`` or ``(value,)``
        and return ``(value,)``; composite-pk collections MUST receive
        a tuple of length matching :attr:`primary_key_columns`. a
        non-tuple input is wrapped in a 1-tuple.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk)
        :ptype entity_id: Any
        :return: tuple of pk values matching
            :attr:`primary_key_columns` length
        :rtype: tuple[Any, ...]
        :raises ValueError: if tuple length does not match
            :attr:`primary_key_columns`
        """
        if isinstance(entity_id, tuple):
            values = entity_id
        else:
            values = (entity_id,)
        pk_cols = self.primary_key_columns
        if len(values) != len(pk_cols):
            raise ValueError(
                f"{self.table_name}: primary key arity mismatch: "
                f"got {len(values)} value(s) for {len(pk_cols)} column(s) {pk_cols}"
            )
        return values

    @abstractmethod
    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch row from L3 keyed by pk.

        public extension point. subclasses override to emit their
        own SELECT. framework invokes on L1+L2 miss via
        :meth:`_pull_through` and on :meth:`reload_entity`. callers
        that need a direct-to-L3 read without cache side-effects may
        invoke this method from outside the collection; prefer
        :meth:`ensure` or :meth:`get` for the normal three-tier path.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk). subclass implementations that hand-roll
            SQL for a composite-pk table MUST accept the tuple shape;
            existing single-pk subclasses accept the scalar shape
            unchanged.
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        ...

    @abstractmethod
    async def save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """persist row to L3.

        public extension point. subclasses override to emit their
        own INSERT ... ON CONFLICT DO UPDATE. framework invokes on
        every non-deferred :meth:`save_entity` and from
        :meth:`persist_to_postgres` during write-buffer flush.

        :param data: row data keyed by column name; pk columns named in
            :attr:`primary_key_columns` MUST be present
        :ptype data: dict[str, Any]
        :param original_timestamp: pre-modification ``date_updated``
            for optimistic-lock validation, ``None`` for inserts
        :ptype original_timestamp: datetime | None
        :param conn: optional asyncpg-compatible connection that
            overrides :attr:`l3_pool` for this single write. when
            supplied, the INSERT/UPDATE binds to the caller's
            transaction so the write commits atomically with whatever
            other operations the caller already issued on the same
            connection. ``None`` defers to the collection's own pool
            (legacy behaviour). subclasses MUST honor this parameter
            so the framework's transactional save_entity path stays
            atomic
        :ptype conn: Any
        :return: rows affected (0 on optimistic-lock failure, 1 on success)
        :rtype: int
        """
        ...

    @abstractmethod
    async def delete_from_postgres(self, entity_id: Any) -> None:
        """delete row from L3 keyed by pk.

        public extension point. subclasses override to emit their
        own DELETE. framework invokes from :meth:`delete`.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk)
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        ...

    @abstractmethod
    def serialize(self, data: dict[str, Any]) -> bytes:
        """encode row dict to bytes for the L2 (NATS KV) tier.

        public extension point. subclasses override to apply their
        JSON codec (typically :func:`threetears.core.serialization.serialize_to_json`)
        plus any domain-specific pre-encoding.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :return: serialized bytes ready for L2 write
        :rtype: bytes
        """
        ...

    @abstractmethod
    def deserialize(self, data: bytes) -> dict[str, Any]:
        """decode bytes from the L2 tier back to a row dict.

        public extension point. subclasses override to reverse
        :meth:`serialize`, rehydrating typed fields (UUID, Decimal,
        datetime) from their JSON representations.

        :param data: serialized bytes previously produced by
            :meth:`serialize`
        :ptype data: bytes
        :return: row dict keyed by column name
        :rtype: dict[str, Any]
        """
        ...

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
        row = self._l1.select_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)
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
        row = self._l1.select_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)
        if row is None:
            return False
        row[field] = value
        self._l1.upsert(self.table_name, row, self.primary_key_columns)
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
        return self._l1.select_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)  # type: ignore[no-any-return]

    def write_to_cache_sync(
        self,
        data: dict[str, Any],
        primary_key: str | tuple[str, ...] | None = None,
    ) -> bool:
        """upsert full row into L1 cache, synchronously.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param primary_key: override collection's pk column(s) for this
            write; ``None`` defaults to :attr:`primary_key_columns`.
            accepts either single column name (str) or tuple of column
            names (composite-pk override).
        :ptype primary_key: str | tuple[str, ...] | None
        :return: ``True`` on successful write, ``False`` when L1 is absent
        :rtype: bool
        """
        if self._l1 is None:
            return False
        pk: str | tuple[str, ...] = primary_key if primary_key is not None else self.primary_key_columns
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
        row = self._l1.select_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)
        return row is not None

    def evict_from_cache_sync(self, entity_id: Any) -> bool:
        """remove a row from the L1 cache only, synchronously.

        narrower than :meth:`invalidate_cache`: drops the L1 slot for
        this pod without touching L2, without publishing a cross-pod
        invalidation, and without awaiting anything. intended for test
        harnesses that want to simulate an L1 eviction and exercise
        the L2 / L3 fall-through path, and for single-pod cache-
        management flows where L2 coherence is driven separately.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk) identifying the row
        :ptype entity_id: Any
        :return: ``True`` when L1 was present and the row (if any) was
            deleted; ``False`` when L1 is absent
        :rtype: bool
        """
        if self._l1 is None:
            return False
        self._l1.delete_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)
        return True

    # --- L2 cache (NATS KV, async) ---

    def _l2_bucket(self) -> str:
        return self._nats_client.bucket_name("collections")  # type: ignore[no-any-return]

    def l2_key(self, entity_id: Any) -> str:
        """build NATS KV key for given pk.

        single-pk shape: ``{table_name}.{value}``. composite-pk shape:
        ``{table_name}.{v1}:{v2}:...`` -- pk values stringified at the
        NATS boundary (per CLAUDE.md UUID/datetime border-conversion
        rule) and joined with ``":"``.

        **edge case**: if a pk value naturally contains ``":"`` the
        composite form is ambiguous with a pk value that contains the
        resulting joined substring. this is a theoretical concern only
        for the realistic pk types in use (UUIDs, integers, slug
        strings, postgres oids) -- none of which legitimately contain
        ``":"``. callers that introduce colon-bearing pk values MUST
        either escape them before passing or override this method.

        :param entity_id: pk value (single-pk) or tuple of pk values
            in declared order (composite-pk)
        :ptype entity_id: Any
        :return: nats KV key, scoped by table name
        :rtype: str
        """
        pk_values = self.normalize_pk(entity_id)
        joined = ":".join(str(v) for v in pk_values)
        return f"{self.table_name}.{joined}"

    async def _get_from_l2(self, entity_id: Any) -> dict[str, Any] | None:
        if self._nats_client is None:
            return None
        try:
            raw = await self._nats_client.get(self._l2_bucket(), self.l2_key(entity_id))
            if raw is None:
                return None
            return self.deserialize(raw)
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
            return await self._nats_client.put(self._l2_bucket(), self.l2_key(entity_id), self.serialize(data))  # type: ignore[no-any-return]
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
            return await self._nats_client.delete(self._l2_bucket(), self.l2_key(entity_id))  # type: ignore[no-any-return]
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
            row = self._l1.select_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)
            if row is not None:
                return row  # type: ignore[no-any-return]
        return sync_await(self._pull_through(entity_id))

    async def _pull_through(self, entity_id: Any) -> dict[str, Any] | None:
        """Async pull-through: L2 -> L1, then L3 -> L1+L2. Returns the data or None."""
        l2_data = await self._get_from_l2(entity_id)
        if l2_data is not None:
            if self._l1 is not None:
                self._l1.upsert(self.table_name, l2_data, self.primary_key_columns)
            return l2_data
        pg_data = await self.fetch_from_postgres(entity_id)
        if pg_data is not None:
            if self._l1 is not None:
                self._l1.upsert(self.table_name, pg_data, self.primary_key_columns)
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
            self._l1.upsert(self.table_name, data, self.primary_key_columns)

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
                await self.save_to_postgres(data)
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
        """pull entity into L1 cache through L2/L3 if not already present.

        :param entity_id: pk value (single-pk) or tuple of pk values in
            declared column order (composite-pk)
        :ptype entity_id: Any
        :return: entity data dict if found in any tier, ``None`` if not
            found anywhere. after ``ensure()`` returns data, subscript
            access is guaranteed to hit L1 (when L1 is available).
        :rtype: dict[str, Any] | None
        """
        if self._l1 is not None:
            row: dict[str, Any] | None = self._l1.select_by_id(
                self.table_name,
                self.normalize_pk(entity_id),
                self.primary_key_columns,
            )
            if row is not None:
                return row
        data = await self._pull_through(entity_id)
        return data

    @traced()
    async def get(self, entity_id: Any) -> EntityT | None:
        """three-tier read: L1 -> L2 -> L3, promote on miss.

        :param entity_id: pk value (single-pk) or tuple of pk values in
            declared column order (composite-pk)
        :ptype entity_id: Any
        :return: entity instance on hit in any tier, ``None`` on
            total-miss
        :rtype: EntityT | None
        """
        self._set_span_table()
        data = await self.ensure(entity_id)
        if data is None:
            self._set_span_attr("cache.hit_tier", "miss")
            return None
        self._set_span_attr("cache.hit_tier", "L1+")
        result: EntityT = self.entity_class(data, is_new=False, collection=self)
        return result

    @traced()
    async def save_entity(
        self, entity: BaseEntity, *, conn: Any = None,
    ) -> None:
        """save entity through the three-tier write path.

        :param entity: entity instance to persist
        :ptype entity: BaseEntity
        :param conn: optional asyncpg-compatible connection that
            overrides :attr:`l3_pool` for the L3 write only. when
            supplied, the L3 INSERT/UPDATE binds to the caller's
            transaction (the caller is responsible for COMMIT /
            ROLLBACK), which makes the write atomic with whatever
            other DDL/DML the caller already issued on the same
            connection. L1 / L2 / invalidation publish run unchanged.
            ``None`` keeps the legacy behaviour: the collection's
            own pool services the write
        :ptype conn: Any
        :return: nothing
        :rtype: None
        :raises ConcurrentModificationError: on optimistic-lock fence
            mismatch when the entity carries an
            ``original_date_updated`` value
        """
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
                self._l1.upsert(self.table_name, data, self.primary_key_columns)
            await self._save_to_l2(entity_id, data)
            assert self._write_buffer is not None
            await self._write_buffer.add(self.table_name, entity_id, data)
            entity.mark_clean()
            entity.original_date_updated = data.get("date_updated")
        else:
            if conn is not None:
                rows_affected = await self.save_to_postgres(
                    data, original_timestamp, conn=conn,
                )
            else:
                rows_affected = await self.save_to_postgres(
                    data, original_timestamp,
                )
            if rows_affected == 0:
                if entity.is_new:
                    raise RuntimeError(f"INSERT failed for {self.table_name} entity {entity_id}: 0 rows affected")
                raise ConcurrentModificationError(self.table_name, entity_id, original_timestamp or datetime.min)
            entity.mark_clean()
            entity.original_date_updated = data.get("date_updated")
            if self._l1 is not None:
                self._l1.upsert(self.table_name, data, self.primary_key_columns)
            else:
                # No L1 backend: repopulate _changes so entity fields remain accessible
                object.__setattr__(entity, "_changes", dict(data))
            await self._save_to_l2(entity_id, data)

        await self._publish_invalidation(entity_id)

    async def persist_to_postgres(self, data: dict[str, Any]) -> int:
        """Used by flush_pending."""
        return await self.save_to_postgres(data)

    @traced()
    async def reload_entity(self, entity: BaseEntity) -> None:
        """Reload entity from L3."""
        self._set_span_table()
        entity_id = entity.id
        if self._write_buffer is not None:
            await self._write_buffer.remove(self.table_name, entity_id)
        data = await self.fetch_from_postgres(entity_id)
        if data is None:
            raise ValueError(f"Entity {entity_id} not found in storage")
        entity.set_data(data)
        entity.original_date_updated = data.get("date_updated")
        if self._l1 is not None:
            self._l1.upsert(self.table_name, data, self.primary_key_columns)
        await self._save_to_l2(entity_id, data)
        await self._publish_invalidation(entity_id)

    @traced()
    async def delete(self, entity_id: Any) -> bool:
        """delete entity from all tiers.

        :param entity_id: pk value (single-pk) or tuple of pk values in
            declared column order (composite-pk)
        :ptype entity_id: Any
        :return: always ``True`` (delete is idempotent across tiers)
        :rtype: bool
        """
        self._set_span_table()
        if self._write_buffer is not None:
            await self._write_buffer.remove(self.table_name, entity_id)
        await self.delete_from_postgres(entity_id)
        if self._l1 is not None:
            self._l1.delete_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)
        await self._delete_from_l2(entity_id)
        await self._publish_invalidation(entity_id)
        return True

    @traced()
    async def invalidate_cache(self, entity_id: Any) -> None:
        """delete from L1 and L2, signal other pods.

        :param entity_id: pk value (single-pk) or tuple of pk values in
            declared column order (composite-pk)
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        self._set_span_table()
        if self._l1 is not None:
            self._l1.delete_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)
        await self._delete_from_l2(entity_id)
        await self._publish_invalidation(entity_id)

    def create(self, data: dict[str, Any]) -> EntityT:
        """Create new entity (not persisted until save)."""
        return self.entity_class(data, is_new=True, collection=self)
