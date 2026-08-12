"""``DurableStoreCollection`` — a collection whose L3 tier is a structured ``DurableStore``.

The standard CRUD lifecycle (``get`` / ``save_entity`` / ``delete`` over the three tiers)
reaches L3 through the seam ``fetch_from_store`` / ``save_to_store`` / ``delete_from_store``.
``SchemaBackedCollection`` implements that seam by generating SQL against ``l3_pool``;
``DurableStoreCollection`` implements it by delegating to an injected
:class:`~threetears.core.backends.protocol.DurableStore` — the structured,
**SQL-free** contract (``fetch_one`` / ``upsert`` / ``delete``).

This is the base a deploying app subclasses to back a collection with a non-SQL durable
tier: scriob's git content store (a ``GitL3Backend`` implementing ``DurableStore`` over a
working tree) drops in here, and the same L1/L2 cache + cross-pod invalidation machinery
runs unchanged. ``l3_pool`` (the raw-SQL transport) is unused and stays ``None``.

Subclasses supply :attr:`table_name`, :attr:`entity_class`, :attr:`primary_key_column`,
and the L2 codec (:meth:`serialize` / :meth:`deserialize`); a JSON codec is provided by
default and may be overridden.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Generic

from threetears.core.backends.protocol import DurableStore
from threetears.core.collections.base import NATS_CLIENT_FROM_REGISTRY, BaseCollection, EntityT
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.serialization import serialize_to_json

__all__ = ["DurableStoreCollection"]


class DurableStoreCollection(BaseCollection[EntityT], Generic[EntityT]):
    """A :class:`BaseCollection` whose L3 tier is a structured :class:`DurableStore`.

    :param durable_store: the structured durable backend (e.g. a ``SqlL3Backend`` or a
        ``GitL3Backend``). The collection's L3 CRUD routes through its
        ``fetch_one`` / ``upsert`` / ``delete``; no SQL string is ever constructed here.
    :ptype durable_store: DurableStore
    """

    #: Conflict policy for the L3 upsert; subclasses may override (``"update"`` / ``"ignore"`` / ``"raise"``).
    on_conflict: str = "update"

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        durable_store: DurableStore,
        nats_client: Any = NATS_CLIENT_FROM_REGISTRY,
        write_buffer: Any = None,
    ) -> None:
        self._durable_store = durable_store
        super().__init__(registry, config, nats_client, write_buffer)

    def _pk_mapping(self, entity_id: Any) -> dict[str, Any]:
        """The pk as a column→value mapping, normalised for single- or composite-pk."""
        return dict(zip(self.primary_key_columns, self.normalize_pk(entity_id), strict=True))

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """Read the entity from L3 via the structured ``DurableStore.fetch_one`` (no SQL)."""
        return await self._durable_store.fetch_one(self.table_name, self._pk_mapping(entity_id))

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        """Persist the entity to L3 via the structured ``DurableStore.upsert`` (CAS rides ``original_timestamp``).

        ``conn`` is forwarded rather than dropped. ``flush_pending``'s atomic-batch path opens
        ``async with backend.transaction() as conn`` and threads that handle down through
        ``persist_to_store`` so a whole toposorted batch commits together; a store that
        implements ``transaction()`` needs the handle to know which batch a write belongs to.

        This used to be discarded on the reasoning that a structured ``DurableStore`` owns its
        own atomicity and a git backend has no connection to bind to. That holds for git, which
        does not implement ``transaction()`` and is served by the per-entity degrade. It does
        not hold for a backend whose ``transaction()`` is a batch accumulator — dropping the
        handle left the accumulator empty, every write executed on its own, and the batch path
        silently produced per-entity durable writes. ``DurableStore.upsert`` declares ``conn``
        precisely so the structured layer can participate; a backend that has no use for it
        ignores it, which is what the protocol already asks of a non-transactional store.
        """
        return await self._durable_store.upsert(
            self.table_name,
            data,
            pk=self.primary_key_columns,
            on_conflict=self.on_conflict,
            cas=original_timestamp,
            conn=conn,
        )

    async def delete_from_store(self, entity_id: Any) -> None:
        """Delete the entity from L3 via the structured ``DurableStore.delete`` (no SQL)."""
        await self._durable_store.delete(self.table_name, self._pk_mapping(entity_id))

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Default L2 codec: canonical JSON bytes. Override for a domain codec."""
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Default L2 codec: decode canonical JSON bytes to a row dict. Override for a domain codec."""
        result: dict[str, Any] = json.loads(data)
        return result
