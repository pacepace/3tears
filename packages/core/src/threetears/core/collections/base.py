"""base collection with three-tier caching.

composite primary keys are first-class. a subclass declares
``primary_key_column = "memory_id"`` for single-pk tables (the default
shape) or ``primary_key_column = ("conversation_id", "item_id")`` for
composite-pk tables. internally, every cache-keying path normalizes
the declared pk and caller-supplied id into a tuple via
:meth:`BaseCollection.normalize_pk`; the L1 (SQLite / DuckDB), L2
(NATS KV), and L3 (pluggable durable backend — SQL, git, …) tiers all
accept the tuple uniformly.
the invalidation wire envelope carries ``ids`` (plural, always an
array) matching the pk column order, so single-pk emits a length-1
array and composite-pk emits a length-N array.
"""

from __future__ import annotations

import hashlib
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Final, Generic, Literal, TypeVar

from threetears.core._bridge import fire_and_forget, sync_await
from threetears.core.backends.protocol import L3Backend
from threetears.core.cache import MISSING
from threetears.core.cache.base import _CACHED_AT_COLUMN
from threetears.core.collections.flush import FlushStrategy, WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError, CorruptCacheEntry
from threetears.nats.errors import KvError
from threetears.observe import get_logger, traced

if TYPE_CHECKING:
    # annotation-only: importing these eagerly would load the NATS client into
    # every L1-only consumer of this module. the `isinstance` below tests the
    # local `_NatsClientFromRegistry` sentinel, not `NatsClient`.
    from threetears.nats import NatsClient, NatsKvBucket

__all__ = ["NATS_CLIENT_FROM_REGISTRY", "BaseCollection", "EntityT"]

log = get_logger(__name__)

EntityT = TypeVar("EntityT", bound=BaseEntity)

#: the JetStream KV key grammar enforced by ``nats-server`` (``kv.go``).
#: a key (or, here, a key *body*) that falls outside this character set
#: is rejected with ``nats: JetStream.InvalidKeyError`` at runtime, so a
#: pk value carrying a colon / space / other out-of-grammar character
#: cannot be interpolated raw into a KV key. matched as a whole string.
_KV_KEY_GRAMMAR: Final = re.compile(r"^[-/_=.a-zA-Z0-9]+$")


class _NatsClientFromRegistry:
    """sentinel type for :data:`NATS_CLIENT_FROM_REGISTRY`."""

    __slots__ = ()


# default for the ``nats_client`` constructor argument: resolve the L2
# client from the registry (``CollectionRegistry.get_l2_client``), so
# ``registry.configure(l2_client=...)`` / ``bind_table(..., l2_client=...)``
# are effective wiring paths. distinguishes "argument omitted" from an
# explicit ``None`` (which keeps its historical meaning: L2 disabled for
# this collection regardless of registry state).
NATS_CLIENT_FROM_REGISTRY: Final = _NatsClientFromRegistry()


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
    :ivar l3_pool: the L3 durable-store handle for this collection. its
        concrete type depends on the configured backend (an **asyncpg pool**
        for the SQL backend; a git store for a git backend; ``None`` when the
        collection is configured without L3, e.g. unit tests using only L1+L2).
        for the **SQL backend** this is the public extension seam for ad-hoc
        SQL: subclasses and external callers (hub endpoints implementing keyset
        pagination, JOINs, bulk queries) may invoke ``await self.l3_pool.fetch(...)``
        / ``execute(...)`` / ``fetchrow(...)`` directly when the query cannot be
        expressed through the Collection API. prefer the collection methods
        (``get``, ``save_entity``, ``delete``, ``__getitem__``, ``__setitem__``)
        for standard CRUD; drop to a backend-specific handle only when no
        Collection method fits. the handle is shared across every collection
        bound to the same backend (resolved through :class:`CollectionRegistry`);
        callers MUST NOT call ``close()`` on it from a collection method or in
        any per-request flow -- its lifecycle is owned by the process that
        constructed the registry. ``None`` is a valid value: callers that need
        to operate without L3 must guard with ``if self.l3_pool is not None``
        rather than assuming presence.
    """

    primary_key_column: str | tuple[str, ...] = "id"

    #: Columns holding timestamps, declared rather than hand-rehydrated.
    #:
    #: The L2 JSON codec renders a ``datetime`` as an ISO string, so a row read through L2
    #: differs in TYPE from the identical row read through L1 or L3 unless something restores
    #: it. That asymmetry is cosmetic while the row is only read; it becomes a fault the moment
    #: one is written BACK, because an update fences on ``date_updated`` as an optimistic lock
    #: and a string bound against ``TIMESTAMPTZ`` fails at the asyncpg border.
    #:
    #: Declaring the columns rather than overriding :meth:`deserialize` is what stops this being
    #: solved once per collection. It had been solved three times, in three packages, with three
    #: different answers to what "rehydrate" means -- tuple versus frozenset, coerce versus pass
    #: through, raise versus preserve the string. Every collection now gets one answer, and gets
    #: it by naming its columns.
    #:
    #: A ``frozenset`` so a subclass can extend its parent's set with ``|`` rather than
    #: restating it, which is how a column gets silently dropped.
    datetime_columns: ClassVar[frozenset[str]] = frozenset()

    # datasource-task-06 DS-06-04: per-concrete-class memo of table
    # names that have already emitted the "nats_client missing"
    # warning, so a busy write path logs the wiring gap once rather
    # than per-write. each subclass gets its own set via the
    # ``type(self)._missing_nats_warned_tables = ...`` assignment in
    # :meth:`_warn_missing_nats_client_once`; declared here so the
    # attribute is typed + present on the base.
    _missing_nats_warned_tables: ClassVar[set[str]] = set()

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: NatsClient | _NatsClientFromRegistry | None = NATS_CLIENT_FROM_REGISTRY,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        self._registry = registry
        self._config = config
        # L2 resolution mirrors L1/L3: when the argument is omitted, the
        # registry is the wiring path (``configure(l2_client=...)`` /
        # ``bind_table``). an explicit client always wins; an explicit
        # ``None`` disables L2 for this collection.
        if isinstance(nats_client, _NatsClientFromRegistry):
            self._nats_client: NatsClient | None = registry.get_l2_client(self.table_name)
        else:
            self._nats_client = nats_client
        self._kv: NatsKvBucket | None = None
        self._write_buffer = write_buffer
        self._flush_strategy = FlushStrategy(config.collection_flush)
        self._flush_tables = frozenset(t.strip() for t in config.collection_flush_tables.split(",") if t.strip())
        # Resolve L1 and L3 from registry
        self._l1 = registry.get_l1_backend(self.table_name)
        self.l3_pool = registry.get_l3_pool(self.table_name)
        # Auto-register
        registry.register(self)

    @property
    def required_l3_pool(self) -> L3Backend:
        """:attr:`l3_pool`, or a clear failure saying why it had to be there.

        For the raw-SQL escape hatch. :attr:`l3_pool` is legitimately ``None`` -- a
        collection can be configured on L1+L2 alone -- so every ad-hoc query has to
        establish that it is not, and the documented instruction is to guard rather than
        assume. In practice the guard was routinely skipped, because ``await
        self.l3_pool.fetch(...)`` reads fine and only fails when someone actually
        constructs the collection without L3. What they then get is
        ``AttributeError: 'NoneType' object has no attribute 'fetch'`` from inside a query
        method, which says nothing about the real mistake.

        A query written in SQL cannot degrade to "no L3" in any meaningful way, so the
        honest behaviour is to fail immediately and say what is missing. Use this wherever
        the query genuinely requires the backend; keep ``if self.l3_pool is not None`` for
        the callers that have a real fallback.

        :return: the L3 backend handle, guaranteed non-``None``
        :rtype: L3Backend
        :raises RuntimeError: when this collection has no L3 backend configured
        """
        if self.l3_pool is None:
            raise RuntimeError(
                f"{type(self).__name__} (table {self.table_name!r}) needs an L3 backend for this "
                "query, but none is configured. Raw SQL has no meaningful L1/L2-only fallback -- "
                "either configure an L3 pool on the CollectionRegistry, or call a Collection API "
                "method that can serve from cache instead."
            )
        return self.l3_pool

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
    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch one entity's row from the L3 durable tier, keyed by pk.

        public extension point — the L3 backend is **pluggable** (storage-
        agnostic): the SQL backend issues a SELECT, a git backend reads the
        entity's file, an in-memory backend reads a dict — the framework only
        requires a row dict on hit / ``None`` on miss. invoked on an L1+L2 miss
        via :meth:`_pull_through` and on :meth:`reload_entity`. callers needing
        a direct-to-L3 read without cache side-effects may invoke it directly;
        prefer :meth:`ensure` or :meth:`get` for the normal three-tier path.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk). composite-pk backends MUST accept the tuple shape;
            single-pk backends accept the scalar shape.
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        ...

    @abstractmethod
    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """persist one entity's row to the L3 durable tier.

        public extension point — the L3 backend is **pluggable** (storage-
        agnostic): the SQL backend issues an upsert, a git backend writes +
        stages the entity's file, an in-memory backend writes a dict. invoked
        on every non-deferred :meth:`save_entity` and from :meth:`persist_to_store`
        during write-buffer flush.

        :param data: row data keyed by column name; pk columns named in
            :attr:`primary_key_columns` MUST be present
        :ptype data: dict[str, Any]
        :param original_timestamp: pre-modification ``date_updated``
            for optimistic-lock validation, ``None`` for inserts
        :ptype original_timestamp: datetime | None
        :param conn: optional **backend-specific transaction handle** (e.g. an
            asyncpg connection for the SQL backend) that overrides
            :attr:`l3_pool` for this single write, so it commits atomically
            with whatever other operations the caller already issued on the
            same transaction. ``None`` uses the collection's own L3 store.
            backends MUST honor this so the framework's transactional
            save_entity path stays atomic.
        :ptype conn: Any
        :return: rows affected (0 on optimistic-lock failure, 1 on success)
        :rtype: int
        """
        ...

    @abstractmethod
    async def delete_from_store(self, entity_id: Any) -> None:
        """delete one entity's row from the L3 durable tier, keyed by pk.

        public extension point — the L3 backend is **pluggable** (storage-
        agnostic): the SQL backend issues a DELETE, a git backend removes the
        entity's file, an in-memory backend drops the dict entry. invoked from
        :meth:`delete`.

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

    # --- timestamp discipline across the L2 boundary ---
    #
    # One rule, enforced at both ends: every datetime in this system is timezone-aware UTC.
    # Write-side normalisation is the half that matters, because it is the only one that
    # PREVENTS anything. Read-side coercion is a legacy tail: it repairs values written before
    # the rule existed, says so in the log, and can be deleted once those age out.
    #
    # Coercing a naive value to UTC on read, alone, is the mistake rather than the fix. If the
    # value was local time, coercion shifts it silently by hours and stamps the result
    # authoritative -- worse than leaving a string, because it now looks correct. Normalising on
    # write means the guess has a shrinking, observable lifetime instead of a permanent one.

    def _normalise_datetimes_for_write(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return *data* with every :attr:`datetime_columns` value aware-UTC.

        A naive value is assumed UTC and stamped, because by the time a row reaches here the
        information needed to interpret it any other way is gone. That assumption is recorded
        at WARNING: it is a guess, and a caller producing naive timestamps has a bug upstream
        that will keep producing them until somebody sees this line.

        Copies rather than mutating: the caller's dict is often an entity's live row, and a
        serialization step that edits its input is a surprise nobody reads the code to find.
        """
        if not self.datetime_columns:
            return data
        out = dict(data)
        for column in self.datetime_columns:
            value = out.get(column)
            if isinstance(value, datetime) and value.tzinfo is None:
                log.warning(
                    "naive datetime normalised to UTC on write; the value's real offset is "
                    "unknowable here, so this is an assumption and the producer should be fixed",
                    extra={"extra_data": {"table": self.table_name, "column": column}},
                )
                out[column] = value.replace(tzinfo=UTC)
        return out

    def _rehydrate_datetimes(self, row: dict[str, Any]) -> dict[str, Any]:
        """Restore :attr:`datetime_columns` from the ISO strings the L2 codec produced.

        :raises CorruptCacheEntry: when a value will not parse. The caller treats that as a
            cache miss and falls through to L3 rather than failing the read -- see the
            exception's own docstring for why that beats propagating or papering over it.
        """
        if not self.datetime_columns:
            return row
        for column in self.datetime_columns:
            value = row.get(column)
            if not isinstance(value, str):
                continue
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise CorruptCacheEntry(self.table_name, column, value) from exc
            if parsed.tzinfo is None:
                log.warning(
                    "naive datetime read from L2 and assumed UTC; written before write-side "
                    "normalisation, or by something that bypasses it",
                    extra={"extra_data": {"table": self.table_name, "column": column}},
                )
                parsed = parsed.replace(tzinfo=UTC)
            row[column] = parsed
        return row

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
        row = self._select_from_l1(entity_id)
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
        row = self._select_from_l1(entity_id)
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
        return self._select_from_l1(entity_id)

    @property
    def l1_max_age_seconds(self) -> float | None:
        """How long an L1 row cached from a lower tier may be served, or ``None``.

        **``None`` unless a collection opts in, and structurally ``None`` when
        there is no L3.** The second half is the load-bearing one. A collection
        with no L3 pool does not fall back to a slower tier on a miss: the
        L1+L2-only collections override :meth:`get` to return ``None`` on a
        total miss, so an expired row does not become a pull-through, it becomes
        "this row does not exist". Downstream, a compare-and-set that reads
        absence writes a fresh row over a live one -- a presence room with ten
        members replaced by a room with one, and no error anywhere. Expiry is a
        cache mechanism, and a tier that is the source of truth is not a cache.

        :return: the configured bound, or ``None`` when expiry is off
        :rtype: float | None
        """
        if self.l3_pool is None:
            return None
        return self._registry.get_l1_max_age(self.table_name)

    def _select_from_l1(self, entity_id: Any, *, expiring: bool = False) -> dict[str, Any] | None:
        """The one L1 read, with expiry applied only where a miss is repairable.

        Every reader **in this class** routes through here rather than calling
        the backend itself, so the policy has one home. But the callers do not
        share a contract, and that is why ``expiring`` is a parameter rather
        than always-on:

        - **Repairing callers** (:meth:`ensure`, :meth:`_ensure_in_l1`) treat a
          miss as "go to the lower tier", so expiring a row makes it reload.
          That is the whole mechanism, and they pass ``expiring=True``.
        - **Reporting callers** (:meth:`get_row_sync`, :meth:`get_field_sync`,
          :meth:`has_field_sync`, :meth:`exists_in_cache_sync`) treat a miss as
          "not cached" and return it to a caller that will not fall back.
          Expiring for them turns a stale row into a *deleted* one and reports
          absence, which is worse than the staleness it was bounding:
          ``__setitem__`` reads a field write back through
          :meth:`get_row_sync` and skips propagation when it sees ``None``, so
          the write is silently dropped, and an entity handle held across the
          bound starts answering ``None`` for fields it has.

        The distinction is a miss's *meaning*, not its value. Expiry converts a
        stale hit into a miss, which is only an improvement where a miss is
        cheap and self-correcting.

        The class scoping is deliberate. A subclass in another package can
        still reach ``self._l1`` directly, and one does --
        ``ContextItemCollection.touch`` reads L1 to stamp ``date_accessed``.
        That read is outside the bound, which is harmless there because it
        neither serves the row to a caller nor clears the stamp on write-back,
        but it is not covered by this funnel and should not be assumed to be.

        :param entity_id: primary-key value identifying the row
        :ptype entity_id: Any
        :param expiring: whether to apply the collection's max-age bound; only
            a caller that repairs a miss by pulling through may pass ``True``
        :ptype expiring: bool
        :return: row dict, or ``None`` when L1 is absent, the row is not
            cached, or (when ``expiring``) the row was past its max age
        :rtype: dict[str, Any] | None
        """
        if self._l1 is None:
            return None
        row: dict[str, Any] | None = self._l1.select_by_id(
            self.table_name,
            self.normalize_pk(entity_id),
            self.primary_key_columns,
            max_age_seconds=self.l1_max_age_seconds if expiring else None,
        )
        return row

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
        row = self._select_from_l1(entity_id)
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

    L2_BUCKET_SUFFIX = "collections"

    async def _ensure_kv(self) -> NatsKvBucket | None:
        """lazily resolve and cache the JetStream KV bucket for L2.

        the new wrapper KV API lives on :class:`NatsKvBucket`, not on
        the bare client, so the collection holds a bucket handle
        rather than a raw client. resolution is lazy so collections
        constructed without a connected client (unit tests, L1+L3
        configurations) never touch JetStream.

        :return: ready bucket handle, or ``None`` when no NATS client
            was supplied
        :rtype: NatsKvBucket | None
        :raises KvError: when bucket binding fails (does NOT swallow
            -- API drift / config errors must surface, not warn)
        """
        if self._nats_client is None:
            return None
        if self._kv is None:
            self._kv = await self._nats_client.kv_bucket(name=self.L2_BUCKET_SUFFIX)
        return self._kv

    def l2_key(self, entity_id: Any) -> str:
        """build a grammar-safe NATS KV key for given pk.

        single-pk shape: ``{table_name}.{value}``. composite-pk shape:
        ``{table_name}.{v1}_{v2}_...`` -- pk values stringified at the
        NATS boundary (per CLAUDE.md UUID/datetime border-conversion
        rule) and joined with ``"_"`` to form the **key body**.

        the body is constrained by the JetStream KV grammar
        (``^[-/_=.a-zA-Z0-9]+$`` per ``nats-server`` ``kv.go``): a body
        carrying a colon ``:``, a space, or any other out-of-grammar
        character is rejected with ``nats: JetStream.InvalidKeyError``.
        so the body is checked against the grammar:

        - **grammar-safe** (the common case -- UUIDs use ``-``,
          integers/postgres oids are pure digits, slugs stay in
          ``[-_a-zA-Z0-9]``): the readable body is kept verbatim, so
          every existing pk in the codebase keeps its current,
          human-readable key (backward-compatible).
        - **out-of-grammar**: the body is replaced by a **SHA-256 hex
          digest** of it -- always grammar-valid (hex is in-grammar) and
          collision-resistant (distinct bodies map to distinct digests,
          unlike a naive ``:``->``=`` replace which silently collides).

        the ``{table_name}.`` prefix is always grammar-safe (table names
        are ``[a-z_]`` identifiers) and is never hashed, so it stays a
        readable namespace. only the body is conditionally hashed. the
        raw pk continues to round-trip through L1, the stored value, and
        the invalidation envelope, so reversibility is not needed.

        deterministic: the same pk always yields the same key (the
        grammar check and the digest are both pure functions of the
        stringified pk values).

        **edge case**: if a *grammar-safe* pk value naturally contains
        ``"_"`` the composite form is ambiguous with a pk value that
        contains the resulting joined substring. callers that introduce
        underscore-bearing grammar-safe pk values MUST either escape
        them before passing or override this method.

        :param entity_id: pk value (single-pk) or tuple of pk values
            in declared order (composite-pk)
        :ptype entity_id: Any
        :return: grammar-safe nats KV key, scoped by table name
        :rtype: str
        """
        pk_values = self.normalize_pk(entity_id)
        body = "_".join(str(v) for v in pk_values)
        if not _KV_KEY_GRAMMAR.match(body):
            body = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return f"{self.table_name}.{body}"

    async def _get_from_l2(self, entity_id: Any) -> dict[str, Any] | None:
        """read entity payload from the L2 NATS KV bucket.

        narrow exception scope: only :class:`KvError` (real transport
        failure) degrades to ``None``. programming errors
        (``AttributeError``, ``TypeError``, etc.) propagate so wrapper
        API drift surfaces loudly instead of silently warning. bucket
        *resolution* (:meth:`_ensure_kv`) is covered by this same catch,
        not just the subsequent op -- an L2 outage during first-open
        (bucket never resolved yet, e.g. right after NATS drops) is
        exactly as much a transport failure as one during an already-open
        bucket's get/put/delete, and must degrade the same way.
        """
        try:
            kv = await self._ensure_kv()
            if kv is None:
                return None
            raw = await kv.get(key=self.l2_key(entity_id))
        except KvError as exc:
            log.warning(
                "L2 cache read failed",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "error": str(exc),
                    },
                },
            )
            return None
        if raw is None:
            return None
        try:
            return self._rehydrate_datetimes(self.deserialize(raw))
        except CorruptCacheEntry as exc:
            # A cache miss, not a failure. Returning None sends the caller to L3, which is
            # authoritative -- the same path a cold key takes. Failing the read instead would
            # let one poisoned key break a lookup that L3 could have answered, and returning
            # the row undecoded would hand back a string where the caller declared a datetime.
            log.warning(
                "L2 entry could not be decoded; falling through to L3",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "column": exc.column,
                    },
                },
            )
            return None

    async def _save_to_l2(self, entity_id: Any, data: dict[str, Any]) -> bool:
        """write entity payload to the L2 NATS KV bucket.

        narrow exception scope: only :class:`KvError` degrades to
        ``False``. programming errors propagate. bucket resolution
        (:meth:`_ensure_kv`) is covered by this same catch -- see
        :meth:`_get_from_l2`'s docstring for why.
        """
        try:
            kv = await self._ensure_kv()
            if kv is None:
                return False
            await kv.put(key=self.l2_key(entity_id), value=self.serialize(self._normalise_datetimes_for_write(data)))
        except KvError as exc:
            log.warning(
                "L2 cache write failed",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "error": str(exc),
                    },
                },
            )
            return False
        return True

    async def _delete_from_l2(self, entity_id: Any) -> bool:
        """delete entity payload from the L2 NATS KV bucket.

        narrow exception scope: only :class:`KvError` degrades to
        ``False``. programming errors propagate. bucket resolution
        (:meth:`_ensure_kv`) is covered by this same catch -- see
        :meth:`_get_from_l2`'s docstring for why.
        """
        try:
            kv = await self._ensure_kv()
            if kv is None:
                return False
            return await kv.delete(key=self.l2_key(entity_id))
        except KvError as exc:
            log.warning(
                "L2 cache delete failed",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "error": str(exc),
                    },
                },
            )
            return False

    # --- Subscript access (sync, transparent pull-through) ---

    def _ensure_in_l1(self, entity_id: Any) -> dict[str, Any] | None:
        """Pull entity into L1 via L2/L3 if not already cached. Sync.

        Returns the row data if found, None if not found in any tier.
        """
        row = self._select_from_l1(entity_id, expiring=True)
        if row is not None:
            return row
        return sync_await(self._pull_through(entity_id))

    async def _pull_through(self, entity_id: Any) -> dict[str, Any] | None:
        """Async pull-through: L2 -> L1, then L3 -> L1+L2. Returns the data or None."""
        l2_data = await self._get_from_l2(entity_id)
        if l2_data is not None:
            if self._l1 is not None:
                self._l1.upsert(self.table_name, self._stamped(l2_data), self.primary_key_columns)
            return l2_data
        pg_data = await self.fetch_from_store(entity_id)
        if pg_data is not None:
            if self._l1 is not None:
                self._l1.upsert(self.table_name, self._stamped(pg_data), self.primary_key_columns)
            await self._save_to_l2(entity_id, pg_data)
            return pg_data
        return None

    @staticmethod
    def _stamped(data: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of ``data`` carrying the cache-age stamp for this instant.

        Stamped wherever a row arrives from a LOWER tier -- both pull-through
        sites and :meth:`reload_entity` -- and nowhere else. The stamp records
        provenance, not local activity: stamping on every write would let
        ``set_field_sync`` renew a stale row's lifetime by editing one field,
        which makes exactly the rows this bounds immortal.

        A copy, because the caller's dict is returned to the caller and becomes
        entity data. The stamp is storage bookkeeping and must not ride along.
        """
        return {**data, _CACHED_AT_COLUMN: time.monotonic()}

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
                await self.save_to_store(data)
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
        """Signal other pods to evict this entity from their L1 caches.

        datasource-task-06 DS-06-04: when ``nats_client`` is missing
        the publish is a no-op -- consumer pods serve stale L1
        entries indefinitely. log a one-shot WARNING the first
        time this happens for a given table so the wiring gap is
        visible in the operator's log. the warning is per-table to
        keep the log scannable across a fleet with many collections.

        design choice: WARNING vs hard-fail-at-construction.
        the spec (DS-06-04) accepted either. WARNING is the
        chosen shape because:

        - the underlying wiring gap (Hub's datasource collections
          constructed without ``nats_client``) is FIXED in
          ``3tears.hub.app.lifespan``; the WARNING is the
          regression net, not the primary fix.
        - several legitimate test scaffolds construct collections
          without a NATS client (SQLite-only L1 tests in
          ``threetears.core``). hard-failing at construction would
          force every such test to stand up a NATS mock for a
          feature it does not exercise.
        - the WARNING fires at the first attempted publish, NOT at
          construction, so collections that never publish (read-
          only utility flows) stay quiet.
        - one-shot dedup means a busy write path doesn't drown
          the log.
        """
        if self._nats_client is None:
            self._warn_missing_nats_client_once()
        await self._registry.publish_invalidation(
            self._nats_client,
            self.table_name,
            entity_id,
        )

    def _warn_missing_nats_client_once(self) -> None:
        """emit a one-shot WARNING when invalidation cannot publish.

        guards against log spam under workloads that issue many
        writes per second -- the wiring gap is process-wide and
        worth surfacing once. uses a class-level set keyed on
        table_name so collections of distinct types each get one
        warning.

        :return: nothing
        :rtype: None
        """
        cls = type(self)
        warned: set[str] = getattr(cls, "_missing_nats_warned_tables", None) or set()
        if self.table_name in warned:
            return
        warned.add(self.table_name)
        cls._missing_nats_warned_tables = warned
        log.warning(
            "collection invalidation is silently disabled: table=%s -- "
            "consumer pods will serve stale L1 entries until the "
            "collection is reconstructed with a non-None nats_client. "
            "wiring gap: datasource-task-06 DS-06-04.",
            self.table_name,
        )

    # --- Span attribute helpers (no-op when OTel unavailable) ---

    @staticmethod
    def _set_span_attr(key: str, value: Any) -> None:
        """Set an attribute on the current OTel span, if available."""
        try:
            from opentelemetry import trace as _trace

            span = _trace.get_current_span()
            span.set_attribute(key, value)
        # NOSILENT: optional dependency probe; absence is a supported configuration
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
        row = self._select_from_l1(entity_id, expiring=True)
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
        self,
        entity: BaseEntity,
        *,
        conn: Any = None,
    ) -> None:
        """save entity through the three-tier write path.

        :param entity: entity instance to persist
        :ptype entity: BaseEntity
        :param conn: optional **backend-specific transaction handle** (e.g. an
            asyncpg connection for the SQL backend) that overrides
            :attr:`l3_pool` for the L3 write only. when supplied, the L3 write
            binds to the caller's transaction (the caller owns commit /
            rollback), making the write atomic with whatever other operations
            the caller already issued on the same transaction. L1 / L2 /
            invalidation publish run unchanged. ``None`` lets the collection's
            own L3 store service the write
        :ptype conn: Any
        :return: nothing
        :rtype: None
        :raises ConcurrentModificationError: on optimistic-lock fence
            mismatch when the entity carries an
            ``original_date_updated`` value
        """
        self._set_span_table()
        data = entity.to_dict()
        # composite-PK collections (eg. ``datasources`` keyed on
        # ``(customer_id, id)`` after the v054 row_scope partition
        # split) need a tuple key for ``_save_to_l2`` /
        # ``_publish_invalidation`` -- ``BaseEntity.id`` only carries
        # the singular ``primary_key_field`` value, so the framework
        # rebuilds the composite from the on-the-wire payload here.
        # single-PK collections short-circuit to ``entity.id`` to
        # match the behaviour every existing caller already relies on.
        pk_cols = self.primary_key_columns
        if len(pk_cols) > 1:
            entity_id: Any = tuple(data[col] for col in pk_cols)
        else:
            entity_id = entity.id
        original_timestamp = getattr(entity, "original_date_updated", None)

        now = datetime.now(UTC)
        if entity.is_new:
            data["date_created"] = now
        if "date_updated" in data or not entity.is_new:
            data["date_updated"] = now

        # Datetime values flow through aware-UTC end to end. Per-column
        # coercion at the L3 backend border is the responsibility of
        # subclasses (SchemaBackedCollection drives this from declared
        # column types). collections-task-05 eliminated DATETIME_TYPE /
        # TIMESTAMP from the platform; the unconditional tzinfo strip
        # that used to live here was load-bearing only while TIMESTAMP
        # columns existed and is gone with them.

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
                rows_affected = await self.save_to_store(
                    data,
                    original_timestamp,
                    conn=conn,
                )
            else:
                rows_affected = await self.save_to_store(
                    data,
                    original_timestamp,
                )
            if rows_affected == 0:
                if entity.is_new:
                    raise RuntimeError(f"L3 insert failed for {self.table_name} entity {entity_id}: 0 rows affected")
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

    async def persist_to_store(self, data: dict[str, Any], *, conn: Any = None) -> int:
        """Persist a write-buffer entry to L3. Used by ``flush_pending``.

        :param data: row payload keyed by column name
        :ptype data: dict[str, Any]
        :param conn: optional **backend-specific transaction handle** that overrides
            :attr:`l3_pool` for this write, so the flush can persist a toposorted batch
            inside ONE backend transaction (``flush_pending`` atomic-batch path).
            ``None`` uses the collection's own L3 store (the per-entity fallback).
        :ptype conn: Any
        :return: rows affected reported by the backend
        :rtype: int
        """
        return await self.save_to_store(data, conn=conn)

    @traced()
    async def reload_entity(self, entity: BaseEntity) -> None:
        """Reload entity from L3."""
        self._set_span_table()
        entity_id = entity.id
        if self._write_buffer is not None:
            await self._write_buffer.remove(self.table_name, entity_id)
        data = await self.fetch_from_store(entity_id)
        if data is None:
            raise ValueError(f"Entity {entity_id} not found in storage")
        entity.set_data(data)
        entity.original_date_updated = data.get("date_updated")
        if self._l1 is not None:
            # Stamped: this row came from L3, so its provenance is a lower tier
            # even though no pull-through ran. Leaving it unstamped would make a
            # freshly-reloaded row read as locally authored, and locally
            # authored rows never expire.
            self._l1.upsert(self.table_name, self._stamped(data), self.primary_key_columns)
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
        await self.delete_from_store(entity_id)
        if self._l1 is not None:
            self._l1.delete_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)
        await self._delete_from_l2(entity_id)
        await self._publish_invalidation(entity_id)
        return True

    @traced()
    async def l2_cas_mutate(
        self,
        entity_id: Any,
        mutate: Callable[[dict[str, Any] | None], tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]],
        *,
        max_retries: int = 8,
    ) -> None:
        """atomically read-modify-write one entity's L2 value under revision CAS.

        the framework's optimistic-lock fence lives in the L3 write
        path (:meth:`save_to_store` returning ``0`` rows), so an
        L1+L2-only collection (no L3 pool) has no ``save_entity``-level
        CAS for atomically mutating a single value. this primitive
        fills that gap: it compare-and-swaps the entity's single L2
        (NATS-KV) value directly, using the wrapper's documented
        revision primitives, then reconciles L1 and fires the cross-pod
        invalidation. it operates on exactly one pk-keyed value (not a
        key-listing), so it stays within the pk-keyed collection
        contract.

        the ``mutate`` callback receives the current row dict (or
        ``None`` when the value is absent) and returns
        ``(action, new_row)``:

        - ``("noop", None)`` -- nothing to do; returns without writing.
        - ``("delete", *)`` -- CAS-delete the value (e.g. the last
          member left). ``new_row`` is ignored. deleting an
          already-absent value is a successful idempotent no-op, so a
          callback may return ``"delete"`` on a ``None`` row without
          special-casing.
        - ``("upsert", new_row)`` -- create-if-absent (so a racing
          creator loses the row) when the value was absent, else
          CAS-update against the read revision. ``new_row`` MUST be
          non-``None``.

        ``date_created`` / ``date_updated`` are stamped on an upsert
        consistently with :meth:`save_entity` (``date_created`` only
        when the value is newly created; ``date_updated`` on every
        upsert). the callback MUST be **idempotent and side-effect
        free** -- it is re-invoked from the freshly-read state on every
        retry.

        **L1-only fallback**: when no NATS client is wired
        (:meth:`_ensure_kv` is ``None`` -- unit / single-pod), there is
        no cross-pod contention to CAS against, so the method degrades
        to a single uncontended L1 read-modify-write via :meth:`get` /
        :meth:`save_entity` / :meth:`delete`.

        **deliberately NOT degrade-on-KvError, unlike its L1+L2+L3
        siblings**: :meth:`_get_from_l2`/:meth:`_save_to_l2`/
        :meth:`_delete_from_l2` catch :class:`KvError` from
        :meth:`_ensure_kv` (bucket resolution) the same as from the
        op itself, because L3 remains the durable source of truth
        behind them -- an L2 outage there is a cache miss, not data
        loss. this method has no such backstop: for the L1+L2-only
        collections it targets (e.g. ``channels/presence``'s room
        membership), L2 **is** the source of truth, so a bucket-open
        failure here is a real write/read failure, not best-effort
        cache noise, and MUST propagate rather than silently no-op a
        membership change that never actually happened.

        :param entity_id: pk value (single-pk) or tuple of pk values in
            declared column order (composite-pk)
        :ptype entity_id: Any
        :param mutate: pure callback computing the next value from the
            current row (or ``None`` when absent)
        :ptype mutate: Callable[[dict[str, Any] | None], tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]]
        :param max_retries: how many times to retry on a CAS conflict
            before surfacing the error
        :ptype max_retries: int
        :return: nothing
        :rtype: None
        :raises ConcurrentModificationError: when the retry budget is
            exhausted (a genuine livelock, never the common case)
        :raises KvError: on any L2 transport failure, including
            during bucket resolution -- intentionally not swallowed,
            see the note above
        """
        self._set_span_table()
        kv = await self._ensure_kv()
        if kv is None:
            await self._l1_only_cas_mutate(entity_id, mutate)
            return

        key = self.l2_key(entity_id)
        for attempt in range(max_retries):
            entry = await kv.get_entry(key=key)
            if entry is None:
                row: dict[str, Any] | None = None
                revision: int | None = None
            else:
                raw_bytes, revision = entry
                try:
                    row = self._rehydrate_datetimes(self.deserialize(raw_bytes))
                except CorruptCacheEntry as exc:
                    # Treated as no prior row, but the REVISION is kept deliberately. The write
                    # below then takes the `update` branch and compare-and-swaps the corrupt
                    # entry away at the revision that held it, which self-heals the key.
                    # Dropping the revision too would take the `create` branch against a key
                    # that exists, fail, and retry until the attempt budget ran out.
                    log.warning(
                        "L2 entry could not be decoded; replacing it in this CAS round",
                        extra={
                            "extra_data": {
                                "entity_id": str(entity_id),
                                "table": self.table_name,
                                "column": exc.column,
                            },
                        },
                    )
                    row = None

            action, new_row = mutate(row)
            if action == "noop":
                return

            now = datetime.now(UTC)
            ok: bool
            if action == "delete":
                ok = await kv.delete(key=key, revision=revision)
            else:
                assert new_row is not None, "'upsert' action must carry a non-None new_row"
                new_row["date_updated"] = now
                if revision is None:
                    # value absent: create-if-absent so a racing creator loses.
                    new_row.setdefault("date_created", now)
                    ok = (
                        await kv.create(key=key, value=self.serialize(self._normalise_datetimes_for_write(new_row)))
                        is not None
                    )
                else:
                    ok = (
                        await kv.update(
                            key=key,
                            value=self.serialize(self._normalise_datetimes_for_write(new_row)),
                            revision=revision,
                        )
                        is not None
                    )

            if not ok:
                if attempt == max_retries - 1:
                    raise ConcurrentModificationError(self.table_name, entity_id, datetime.min)
                log.info(
                    "L2 CAS conflict; retrying",
                    extra={
                        "extra_data": {
                            "entity_id": str(entity_id),
                            "table": self.table_name,
                            "attempt": attempt + 1,
                            "action": action,
                        }
                    },
                )
                continue

            # CAS won: reconcile L1 and notify peers.
            if action == "delete":
                if self._l1 is not None:
                    self._l1.delete_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)
            else:
                assert new_row is not None  # narrow: "upsert" always carries a row
                if self._l1 is not None:
                    self._l1.upsert(self.table_name, new_row, self.primary_key_columns)
            await self._publish_invalidation(entity_id)
            return

    async def _l1_only_cas_mutate(
        self,
        entity_id: Any,
        mutate: Callable[[dict[str, Any] | None], tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]],
    ) -> None:
        """uncontended L1 read-modify-write fallback for :meth:`l2_cas_mutate`.

        used when no NATS client is wired (L1-only unit / single-pod
        mode): there is no cross-pod contention to CAS against, so a
        plain read-modify-write via :meth:`get` + :meth:`save_entity` /
        :meth:`delete` is correct. ``date_created`` / ``date_updated``
        stamping is owned by :meth:`save_entity`, so this path does not
        re-stamp.

        :param entity_id: pk value (single-pk) or tuple of pk values
        :ptype entity_id: Any
        :param mutate: same callback contract as :meth:`l2_cas_mutate`
        :ptype mutate: Callable[[dict[str, Any] | None], tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]]
        :return: nothing
        :rtype: None
        """
        entity = await self.get(entity_id)
        row = entity.to_dict() if entity is not None else None
        action, new_row = mutate(row)
        if action == "noop":
            return
        if action == "delete":
            await self.delete(entity_id)
            return
        assert new_row is not None, "'upsert' action must carry a non-None new_row"
        if entity is None:
            await self.save_entity(self.create(new_row))
        else:
            entity.set_data(new_row)
            await self.save_entity(entity)

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
