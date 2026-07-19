"""Collection registry — DI container + table_name lookup + cache coherence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ValidationError
from threetears.nats import NatsClient, Subjects
from threetears.nats.errors import PublishError, SubscribeError
from threetears.observe import get_logger
from uuid_utils import uuid7

if TYPE_CHECKING:
    from threetears.core.backends import L3Backend

__all__ = [
    "CacheInvalidationMessage",
    "CollectionRegistry",
]

log = get_logger(__name__)


def _as_l3_backend(l3_pool: Any) -> Any:
    """normalize a raw L3 transport to a :class:`DurableStore`-capable backend.

    The collection CRUD lifecycle routes through the structured
    :class:`~threetears.core.backends.protocol.DurableStore` ops, so the resolved
    ``l3_pool`` must expose them. A backend that already satisfies ``DurableStore``
    (the named :class:`~threetears.core.backends.sql.SqlL3Backend`, or a non-SQL
    backend such as scriob's ``GitL3Backend``) is returned **unchanged**. A raw
    transport that only speaks raw SQL (a bare asyncpg ``Pool`` or the
    ``NatsProxyL3Backend``) is wrapped in a ``SqlL3Backend`` so it gains the
    structured ops (which generate SQL). ``None`` passes through.

    Imported lazily to keep this module free of an import-time dependency on the
    backends package.

    :param l3_pool: a raw transport, a ``DurableStore`` backend, or ``None``.
    :ptype l3_pool: Any
    :return: a ``DurableStore``-capable backend, or ``None``.
    :rtype: Any
    """
    if l3_pool is None:
        return None
    from threetears.core.backends.protocol import DurableStore
    from threetears.core.backends.sql import SqlL3Backend

    if isinstance(l3_pool, DurableStore):
        return l3_pool
    return SqlL3Backend(l3_pool)


class CacheInvalidationMessage(BaseModel):
    """typed wire envelope for cross-pod cache invalidation broadcasts.

    publishers emit one message per write op; every other pod
    subscribed to :func:`Subjects.cache_invalidate` evicts the named
    entity from its local L1 cache. ``ids`` is always an array whose
    length matches the target collection's
    :attr:`BaseCollection.primary_key_columns` declaration -- single-pk
    collections emit length-1 arrays; composite-pk collections emit
    length-N arrays in declared column order.

    typed Pydantic envelope replaces the previous raw-JSON-bytes
    publish; the typed wrapper enforces wire-format consistency and
    surfaces drift at parse time rather than via opaque key errors
    deep in the listener.

    :ivar table: target table name (matches :attr:`BaseCollection.table_name`)
    :ivar ids: stringified pk values in declared column order
    :ivar origin: opaque per-registry id of the publisher. a receiving
        registry skips any message carrying its OWN origin -- a single
        pod both publishes and subscribes on this subject, and acting on
        its own invalidation would evict the L1 row it just wrote (the
        fresh value is already local; the broadcast is for OTHER pods).
        ``None`` for messages from a pre-origin publisher -> the receiver
        cannot prove self-origin and evicts (the historical behaviour),
        which is safe: a redundant local eviction only forces a
        pull-through, never stale data.
    """

    table: str
    ids: list[str]
    origin: str | None = None


class CollectionRegistry:
    """Registry for collection instances with dependency injection.

    Holds default L1/L2/L3 dependencies and per-collection overrides.
    Collections register themselves and resolve their dependencies through
    the registry.
    """

    def __init__(self) -> None:
        self._l1_backend: Any = None
        self._l2_client: Any = None
        self._l3_pool: L3Backend | None = None
        self._collections: dict[str, Any] = {}  # table_name -> collection instance
        self._overrides: dict[str, dict[str, Any]] = {}  # table_name -> {l1_backend, l2_client, l3_pool}
        # Per-registry (effectively per-pod) identity stamped on every
        # invalidation this registry publishes, so its own listener can
        # skip self-published messages and avoid evicting rows it just
        # wrote. An opaque token, never used as a UUID.
        self._origin_id: str = str(uuid7())  # convert at border: invalidation wire-envelope origin token

    def configure(
        self,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: L3Backend | None = None,
    ) -> None:
        """Set default dependencies for all collections."""
        if l1_backend is not None:
            self._l1_backend = l1_backend
        if l2_client is not None:
            self._l2_client = l2_client
        if l3_pool is not None:
            self._l3_pool = _as_l3_backend(l3_pool)

    def register(
        self,
        collection: Any,
        *,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: L3Backend | None = None,
    ) -> None:
        """Register a collection instance with optional per-collection overrides."""
        table = collection.table_name
        self._collections[table] = collection
        if l1_backend or l2_client or l3_pool:
            self._overrides[table] = {}
            if l1_backend:
                self._overrides[table]["l1_backend"] = l1_backend
            if l2_client:
                self._overrides[table]["l2_client"] = l2_client
            if l3_pool:
                self._overrides[table]["l3_pool"] = _as_l3_backend(l3_pool)

    def bind_table(
        self,
        table_name: str,
        *,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: L3Backend | None = None,
    ) -> None:
        """pin per-table backend overrides BEFORE the Collection is constructed.

        :class:`BaseCollection.__init__` reads ``l3_pool`` (and friends)
        from the registry via :meth:`get_l3_pool` immediately and then
        auto-registers. :meth:`register`'s ``l3_pool=`` kwarg records
        an override but fires too late -- the Collection has already
        snapped its pool from the registry default. ``bind_table``
        records the override under the table name so the subsequent
        Collection construction reads the intended backend on its
        first :meth:`get_l3_pool` call.

        used by multi-pool agent-side bootstraps (three-tier-task-01
        Phase C2: the rbac metadata Collections need a separate
        :class:`NatsProxyL3Backend` pool bound to ``system.platform.rbac``
        because the broker route different namespaces to different
        schemas). every key left as ``None`` is ignored so callers can
        layer l1 / l2 / l3 bindings independently.

        :param table_name: target table name (matches
            :attr:`BaseCollection.table_name` on the Collection that
            will later be constructed)
        :ptype table_name: str
        :param l1_backend: L1 backend override for this table, or
            ``None`` to leave any existing binding untouched
        :ptype l1_backend: Any
        :param l2_client: L2 client override for this table
        :ptype l2_client: Any
        :param l3_pool: L3 pool override for this table
        :ptype l3_pool: Any
        :return: nothing
        :rtype: None
        """
        if l1_backend is None and l2_client is None and l3_pool is None:
            return
        existing = self._overrides.setdefault(table_name, {})
        if l1_backend is not None:
            existing["l1_backend"] = l1_backend
        if l2_client is not None:
            existing["l2_client"] = l2_client
        if l3_pool is not None:
            existing["l3_pool"] = _as_l3_backend(l3_pool)

    def get_collection(self, table_name: str) -> Any | None:
        """Look up a registered collection by table name."""
        return self._collections.get(table_name)

    def get_l1_backend(self, table_name: str) -> Any:
        """Get L1 backend for a collection (override or default)."""
        overrides = self._overrides.get(table_name, {})
        return overrides.get("l1_backend", self._l1_backend)

    def get_l2_client(self, table_name: str) -> Any:
        """Get L2 client for a collection (override or default)."""
        overrides = self._overrides.get(table_name, {})
        return overrides.get("l2_client", self._l2_client)

    def get_l3_pool(self, table_name: str) -> L3Backend | None:
        """Get L3 pool for a collection (override or default)."""
        overrides = self._overrides.get(table_name, {})
        return cast("L3Backend | None", overrides.get("l3_pool", self._l3_pool))

    # ------------------------------------------------------------------
    # Cache coherence -- cross-pod L1 invalidation via typed NATS pub/sub
    # ------------------------------------------------------------------

    async def start_invalidation_listener(self, nats_client: NatsClient) -> None:
        """subscribe to cache invalidation signals from other pods.

        wire envelope is :class:`CacheInvalidationMessage` -- typed
        Pydantic, serialized via ``model_dump_json()``. on receipt the
        listener evicts the named entity from local L1 cache for the
        matching collection. callers must invoke after every
        Collection has registered.

        narrow exception scope: malformed payloads (Pydantic
        :class:`ValidationError`) and unknown-table receipts log and
        skip; programming errors (``AttributeError`` / ``TypeError``)
        propagate so wrapper drift / collection-misregistration
        surfaces immediately rather than as silent
        invalidation-skips.

        :param nats_client: connected typed NATS wrapper client
        :ptype nats_client: NatsClient
        :return: nothing
        :rtype: None
        :raises SubscribeError: if the underlying subscribe fails to
            register (transport / config error)
        """
        self._nats_client = nats_client

        async def _on_invalidation(message: CacheInvalidationMessage) -> None:
            # Skip invalidations this registry published itself. A single
            # pod both publishes and subscribes on this subject; acting on
            # our own broadcast would evict the L1 row save_entity just
            # wrote, so a subsequent L1-only read of that entity (e.g. an
            # entity field accessor) would miss. The broadcast exists to
            # evict OTHER pods, which carry a different origin.
            if message.origin is not None and message.origin == self._origin_id:
                return

            collection = self._collections.get(message.table)
            if collection is None:
                # unknown-table receipts are expected during partial
                # rollouts (sender has a Collection the receiver does
                # not). log + skip without warning.
                return

            l1 = self.get_l1_backend(message.table)
            if l1 is None:
                return
            if hasattr(l1, "has_table") and not l1.has_table(message.table):
                # this pod's L1 backend was never initialize()'d with this
                # table's schema -- its OWN collections never touch it
                # locally, so there is nothing to evict. Same "unknown
                # receipts are expected during partial rollouts" treatment
                # as the `collection is None` case above: without this
                # check, `l1.delete_by_id` below raises straight through
                # (sqlite3.OperationalError: no such table / DuckDB's
                # equivalent CatalogException) on every single broadcast
                # for a table this pod doesn't cache, which is the common
                # case for any agent that doesn't use every framework
                # feature (e.g. a security-scanning agent hearing knowledge-
                # subsystem invalidations for `concepts`/`playbook_entries`
                # it never reads). `hasattr` guards backends (or test
                # doubles) that predate this method.
                return

            pk_cols = collection.primary_key_columns
            if len(message.ids) != len(pk_cols):
                log.warning(
                    "Invalidation pk arity mismatch",
                    extra={
                        "extra_data": {
                            "table": message.table,
                            "expected_columns": list(pk_cols),
                            "received_values": len(message.ids),
                        },
                    },
                )
                return
            entity_id = tuple(message.ids)
            l1.delete_by_id(message.table, entity_id, pk_cols)

        try:
            await nats_client.subscribe_typed(
                subject=Subjects.cache_invalidate(),
                message_type=CacheInvalidationMessage,
                cb=_on_invalidation,
            )
        except SubscribeError:
            # surface subscribe failure; cache coherence is not optional
            raise

    async def publish_invalidation(
        self,
        nats_client: NatsClient | None,
        table_name: str,
        entity_id: Any,
    ) -> None:
        """publish cache invalidation signal for an entity.

        called by :class:`BaseCollection` after any write operation.
        emits :class:`CacheInvalidationMessage` via the typed wrapper.
        ``ids`` is the stringified pk-value tuple in declared column
        order.

        narrow exception scope: only :class:`PublishError` is logged
        and swallowed (the write has already succeeded; the
        invalidation broadcast is best-effort because the next pod
        read will pull a stale-but-still-correct row from L3 if it
        misses the eviction). programming errors propagate.

        :param nats_client: connected typed NATS wrapper client;
            ``None`` short-circuits (no-op)
        :ptype nats_client: NatsClient | None
        :param table_name: target table name
        :ptype table_name: str
        :param entity_id: pk value (single-pk) or tuple of pk values
            in declared order (composite-pk)
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if nats_client is None:
            return
        if isinstance(entity_id, tuple):
            values = entity_id
        else:
            values = (entity_id,)
        ids = [str(v) for v in values]  # convert at border: invalidation wire-envelope pk values
        message = CacheInvalidationMessage(table=table_name, ids=ids, origin=self._origin_id)
        try:
            await nats_client.publish(
                subject=Subjects.cache_invalidate(),
                message=message,
            )
        except PublishError as exc:
            log.warning(
                "Failed to publish invalidation signal",
                extra={
                    "extra_data": {
                        "table": table_name,
                        "entity_id": str(
                            entity_id
                        ),  # convert at border: invalidation-publish-failed log extra_data field
                        "error": str(exc),
                    },
                },
            )
        except ValidationError as exc:
            # wire envelope failed validation -- programming error,
            # but propagating would mask the calling write op. log
            # loud and continue; surfaces as a real failure in CI
            # because tests assert on this counter.
            log.error(
                "Invalidation envelope validation failed",
                extra={
                    "extra_data": {
                        "table": table_name,
                        "entity_id": str(
                            entity_id
                        ),  # convert at border: invalidation-envelope-invalid log extra_data field
                        "error": str(exc),
                    },
                },
            )

    def clear(self) -> None:
        """Remove all registered collections and overrides (for tests)."""
        self._collections.clear()
        self._overrides.clear()
