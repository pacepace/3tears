"""Collection registry — DI container + table_name lookup + cache coherence."""

from __future__ import annotations

import json
from typing import Any

from threetears.core.logging import get_logger

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
        """Subscribe to cache invalidation signals from other pods.

        When a signal is received, evicts the entity from the local L1 cache
        of the matching collection. This ensures cross-pod cache coherence.

        Must be called at application startup after all collections are registered.
        """
        self._nats_client = nats_client

        async def _on_invalidation(data: bytes) -> None:
            try:
                payload = json.loads(data)
                table = payload.get("table")
                entity_id = payload.get("entity_id")
                if not table or not entity_id:
                    log.warning(
                        "Malformed invalidation signal",
                        extra={"extra_data": {"raw": data[:200].decode(errors="replace")}},
                    )
                    return

                collection = self._collections.get(table)
                if collection is None:
                    return  # Unknown table — ignore

                l1 = self.get_l1_backend(table)
                if l1 is not None:
                    l1.delete_by_id(table, str(entity_id), collection._primary_key_column)
            except Exception as exc:
                log.warning(
                    "Error processing invalidation signal",
                    extra={"extra_data": {"error": str(exc)}},
                )

        await nats_client.subscribe(INVALIDATION_SUBJECT, _on_invalidation)

    async def publish_invalidation(self, nats_client: Any, table_name: str, entity_id: Any) -> None:
        """Publish a cache invalidation signal for an entity.

        Called by BaseCollection after any write operation. Other pods
        subscribed to the invalidation subject will evict this entity
        from their L1 cache.
        """
        if nats_client is None:
            return
        try:
            payload = json.dumps({"table": table_name, "entity_id": str(entity_id)}).encode()
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
