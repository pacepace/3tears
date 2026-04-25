"""Collection registry — DI container + table_name lookup + cache coherence."""

from __future__ import annotations

import json
from typing import Any

from threetears.observe import get_logger

__all__ = [
    "CollectionRegistry",
    "INVALIDATION_SUBJECT",
]

log = get_logger(__name__)

INVALIDATION_SUBJECT = "threetears.cache.invalidate"


class CollectionRegistry:
    """Registry for collection instances with dependency injection.

    Holds default L1/L2/L3 dependencies and per-collection overrides.
    Collections register themselves and resolve their dependencies through
    the registry.
    """

    def __init__(self) -> None:
        self._l1_backend: Any = None
        self._l2_client: Any = None
        self._l3_pool: Any = None
        self._collections: dict[str, Any] = {}  # table_name -> collection instance
        self._overrides: dict[str, dict[str, Any]] = {}  # table_name -> {l1_backend, l2_client, l3_pool}

    def configure(
        self,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: Any = None,
    ) -> None:
        """Set default dependencies for all collections."""
        if l1_backend is not None:
            self._l1_backend = l1_backend
        if l2_client is not None:
            self._l2_client = l2_client
        if l3_pool is not None:
            self._l3_pool = l3_pool

    def register(
        self,
        collection: Any,
        *,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: Any = None,
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
                self._overrides[table]["l3_pool"] = l3_pool

    def bind_table(
        self,
        table_name: str,
        *,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: Any = None,
    ) -> None:
        """pin per-table backend overrides BEFORE the Collection is constructed.

        :class:`BaseCollection.__init__` reads ``l3_pool`` (and friends)
        from the registry via :meth:`get_l3_pool` immediately and then
        auto-registers. :meth:`register`'s ``l3_pool=`` kwarg records
        an override but fires too late — the Collection has already
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
            existing["l3_pool"] = l3_pool

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

    def get_l3_pool(self, table_name: str) -> Any:
        """Get L3 pool for a collection (override or default)."""
        overrides = self._overrides.get(table_name, {})
        return overrides.get("l3_pool", self._l3_pool)

    # ------------------------------------------------------------------
    # Cache coherence — cross-pod L1 invalidation via NATS pub/sub
    # ------------------------------------------------------------------

    async def start_invalidation_listener(self, nats_client: Any) -> None:
        """subscribe to cache invalidation signals from other pods.

        wire envelope (stable contract, no shims):
        ``{"table": <table_name>, "ids": [<v1>, <v2>, ...]}`` -- the
        ``ids`` field is ALWAYS an array, length matches the target
        collection's :attr:`BaseCollection.primary_key_columns`
        declaration. single-pk collections publish length-1 arrays;
        composite-pk collections publish length-N arrays in declared
        column order. the previous ``entity_id`` string field is
        retired; every publisher in the platform emits ``ids``.

        when a signal is received, evicts the entity from local L1
        cache of the matching collection. this ensures cross-pod
        cache coherence. must be called at application startup after
        all collections are registered.

        :param nats_client: nats client with ``subscribe`` method
        :ptype nats_client: Any
        :return: nothing
        :rtype: None
        """
        self._nats_client = nats_client

        async def _on_invalidation(data: bytes) -> None:
            try:
                payload = json.loads(data)
                table = payload.get("table")
                ids = payload.get("ids")
                if not table or not isinstance(ids, list) or not ids:
                    log.warning(
                        "Malformed invalidation signal",
                        extra={"extra_data": {"raw": data[:200].decode(errors="replace")}},
                    )
                    return

                collection = self._collections.get(table)
                if collection is None:
                    return  # unknown table — ignore

                l1 = self.get_l1_backend(table)
                if l1 is not None:
                    pk_cols = collection.primary_key_columns
                    if len(ids) != len(pk_cols):
                        log.warning(
                            "Invalidation pk arity mismatch",
                            extra={
                                "extra_data": {
                                    "table": table,
                                    "expected_columns": list(pk_cols),
                                    "received_values": len(ids),
                                }
                            },
                        )
                        return
                    entity_id = tuple(ids)
                    l1.delete_by_id(table, entity_id, pk_cols)
            except Exception as exc:
                log.warning(
                    "Error processing invalidation signal",
                    extra={"extra_data": {"error": str(exc)}},
                )

        await nats_client.subscribe(INVALIDATION_SUBJECT, cb=_on_invalidation)

    async def publish_invalidation(self, nats_client: Any, table_name: str, entity_id: Any) -> None:
        """publish cache invalidation signal for an entity.

        called by :class:`BaseCollection` after any write operation.
        other pods subscribed to the invalidation subject will evict
        this entity from their L1 cache. wire envelope carries
        ``ids`` as an array of stringified pk values (length matches
        the collection's pk column count).

        :param nats_client: nats client with ``publish`` method;
            ``None`` short-circuits (no-op)
        :ptype nats_client: Any
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
        try:
            if isinstance(entity_id, tuple):
                values = entity_id
            else:
                values = (entity_id,)
            ids = [str(v) for v in values]
            payload = json.dumps({"table": table_name, "ids": ids}).encode()
            await nats_client.publish(INVALIDATION_SUBJECT, payload)
        except Exception as exc:
            log.warning(
                "Failed to publish invalidation signal",
                extra={
                    "extra_data": {
                        "table": table_name,
                        "entity_id": str(entity_id),
                        "error": str(exc),
                    }
                },
            )

    def clear(self) -> None:
        """Remove all registered collections and overrides (for tests)."""
        self._collections.clear()
        self._overrides.clear()
