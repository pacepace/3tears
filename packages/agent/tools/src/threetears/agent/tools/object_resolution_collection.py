"""The tool-pod runtime's own two-tier collection: resolved object keys, shared by replicas.

:class:`~threetears.agent.tools.object_resolver.HubObjectResolver` turns an object id
into the key its bytes are stored under by asking the hub, and it used to remember the
answer in a plain ``dict`` with FIFO eviction. That dict is pod-LOCAL, so every replica
of one pod paid its own round trip for a mapping a sibling had already resolved, and a
restart threw the lot away. It is also a hand-rolled cache in a codebase whose one state
primitive is :class:`~threetears.core.collections.base.BaseCollection`.

This is that cache as a collection: **L1 + L2 and no L3**, the shape
``HeartbeatCollection`` and the presence collections already ship. The two tiers map
exactly onto what the data is:

- **L1** (the pod's in-process SQLite) is the fast path the dict used to be.
- **L2** (the shared ``{ns}-collections`` KV bucket, under this pod's own key scope) is
  what the dict could never be: ``tool_pods.id`` is configured once per DEPLOYMENT, so
  every replica resolves to one scope and reads one key. A resolution one replica paid
  for serves all of them.
- **L3 is absent and its methods RAISE.** A tool pod cannot reach L3 at all today -- the
  broker reads the principal off a hub-minted identity token, and a tool pod holds none
  until the handshake reaches it. A collection that quietly accepted a durable write
  would report success for a row nobody stored. Refusing loudly is the only honest
  behaviour until that lands, and it costs nothing here: the hub is the system of
  record for the mapping, so a total cache miss is one request, not lost data.

**Why the mapping is safe to share and safe to lose.** A committed object's id -> key
mapping is immutable, so a cached value can go stale only by being deleted upstream, and
the row is rebuildable from the hub on any miss. The pk carries the VERIFIED
``customer_id`` alongside the object id, so a mapping resolved for one tenant is never
served to another -- the same guarantee the dict's ``(customer_id, object_id)`` key gave,
kept rather than relaxed.

What the invalidation broadcast does here is worth stating, because it is the opposite of
what it does for a three-tier collection. Replicas of one pod share a key scope and
therefore share ONE L2 key, so a peer that hears the broadcast drops its L1 row and pulls
the writer's fresh value back through L2. It does NOT evict L2:
:meth:`~threetears.core.collections.base.BaseCollection.delete_l2_entry` is structurally a
no-op without an L3 pool, because with nothing beneath it an eviction is a deletion.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.media.contracts import ObjectHandle
from threetears.observe import get_logger

__all__ = [
    "OBJECT_RESOLUTIONS_TABLE",
    "ObjectResolutionCollection",
    "ObjectResolutionEntity",
]

log = get_logger(__name__)

#: L1 table (and L2 key segment) holding one resolved object mapping per row. Declared
#: here rather than in :mod:`threetears.agent.tools.l1_cache` so the collection owns its
#: own name and the cache module imports it; the SQLAlchemy table itself is the cache
#: module's, because that is where a tool pod's L1 schema is assembled.
OBJECT_RESOLUTIONS_TABLE = "object_resolutions"


class ObjectResolutionEntity(BaseEntity):
    """row in the ``object_resolutions`` L1 table.

    fields: ``customer_id`` / ``object_id`` (the composite primary key, both stringified
    UUIDs) / ``s3_key`` / ``mime_type`` / ``size_bytes`` / ``summary`` / ``category`` /
    ``date_created`` / ``date_updated``.

    :cvar primary_key_field: the bare row id; the partition half of the key reaches the
        framework through the collection's declared ``primary_key_column``
    """

    primary_key_field: str = "object_id"


class ObjectResolutionCollection(BaseCollection[ObjectResolutionEntity]):
    """L1+L2 collection mapping a verified ``(customer, object)`` to its stored key.

    Constructed once per pod process by :class:`ToolServerBootstrap` and handed to the
    pod's :class:`~threetears.agent.tools.object_resolver.HubObjectResolver`.

    The composite primary key is ``(customer_id, object_id)`` rather than one derived
    string, so the framework builds the L2 key body itself (``{v1}_{v2}``) and the
    invalidation envelope carries both values in declared order. Both are stringified
    UUIDs: they are grammar-safe for the JetStream KV key charset and contain no
    underscore, so the composite body is unambiguous.
    """

    primary_key_column: str | tuple[str, ...] = ("customer_id", "object_id")

    #: rehydrated to aware-UTC by the framework's L2 codec on every read.
    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """wire the collection against the registry and force L3 off.

        :param registry: the pod's collection registry, carrying the L1 backend, the L2
            client and this pod's key scope
        :ptype registry: CollectionRegistry
        :param config: core configuration; the write path never consults the flush
            strategy here, so the value is DI symmetry with sibling collections
        :ptype config: CoreConfig
        :param nats_client: NATS client for the L2 tier and the invalidation broadcast;
            ``None`` runs L1-only, which is a single-replica pod or a unit test
        :ptype nats_client: Any
        :param write_buffer: unused; there is no deferred L3 flush to buffer
        :ptype write_buffer: WriteBuffer | None
        :return: nothing
        :rtype: None
        """
        super().__init__(registry, config, nats_client, write_buffer)
        # discard whatever pool the registry offered: this collection is two-tier by
        # design, and inheriting one would turn the deliberate refusals below into
        # writes against a store the mapping does not belong in.
        self.l3_pool = None

    @property
    def table_name(self) -> str:
        """return the L1 table name holding resolved object mappings.

        :return: table name
        :rtype: str
        """
        return OBJECT_RESOLUTIONS_TABLE

    @property
    def entity_class(self) -> type[ObjectResolutionEntity]:
        """return the entity class for this collection.

        :return: :class:`ObjectResolutionEntity`
        :rtype: type[ObjectResolutionEntity]
        """
        return ObjectResolutionEntity

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize a row dict to JSON bytes for L2 storage.

        :param data: row data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize JSON bytes from L2 back into a row dict.

        :attr:`datetime_columns` rehydration is the framework's, applied around this
        call, so this method does not repeat it.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row dict
        :rtype: dict[str, Any]
        """
        row: dict[str, Any] = json.loads(data.decode("utf-8"))
        return row

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """unreachable -- this collection is L1+L2 only.

        :param entity_id: ignored; kept for signature symmetry
        :ptype entity_id: Any
        :return: never returns
        :rtype: dict[str, Any] | None
        :raises RuntimeError: always; a tool pod has no L3 to read
        """
        raise RuntimeError(
            f"{type(self).__name__} is L1+L2 only; fetch_from_store must never be "
            f"reached (no L3 pool bound for '{self.table_name}')",
        )

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """unreachable -- this collection is L1+L2 only.

        Raising rather than returning ``0`` is deliberate: a silent no-op here reports a
        durable write that never happened.

        :param data: ignored; kept for signature symmetry
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored
        :ptype original_timestamp: datetime | None
        :param conn: ignored; kept for LSP parity with the base class
        :ptype conn: Any
        :return: never returns
        :rtype: int
        :raises RuntimeError: always; a tool pod has no L3 to write
        """
        raise RuntimeError(
            f"{type(self).__name__} is L1+L2 only; save_to_store must never be reached "
            f"(remember() writes L1 and L2 directly)",
        )

    async def delete_from_store(self, entity_id: Any) -> None:
        """unreachable -- this collection is L1+L2 only.

        :param entity_id: ignored; kept for signature symmetry
        :ptype entity_id: Any
        :return: never returns
        :rtype: None
        :raises RuntimeError: always; a tool pod has no L3 to delete from
        """
        raise RuntimeError(
            f"{type(self).__name__} is L1+L2 only; delete_from_store must never be "
            f"reached (forget() clears L1 and L2 directly)",
        )

    async def lookup(self, customer_id: UUID, object_id: UUID) -> ObjectHandle | None:
        """return the stored handle for a verified ``(customer, object)``, or ``None``.

        Reads L1 first, then pulls through L2 on a miss and fills L1 so the next read on
        this replica is local. A total miss is ``None`` -- the caller re-asks the hub,
        which is the source of truth.

        :param customer_id: the VERIFIED owning customer
        :ptype customer_id: UUID
        :param object_id: the object whose stored key is wanted
        :ptype object_id: UUID
        :return: the handle, or ``None`` when neither tier holds it
        :rtype: ObjectHandle | None
        """
        entity_id = self._pk(customer_id, object_id)
        row: dict[str, Any] | None = None
        if self._l1 is not None:
            row = self._l1.select_by_id(self.table_name, entity_id, self.primary_key_columns)
        if row is None:
            row = await self._get_from_l2(entity_id)
            if row is not None and self._l1 is not None:
                self._l1.upsert(self.table_name, row, self.primary_key_columns)
        if row is None:
            return None
        return _handle_from_row(row)

    async def remember(self, customer_id: UUID, handle: ObjectHandle) -> None:
        """record a resolved handle in L1 and L2, and tell the other replicas.

        L2 is written on the foreground path rather than fire-and-forget: the point of
        the tier is that a sibling replica sees the mapping without its own round trip,
        and a write that has not landed yet is a round trip.

        :param customer_id: the VERIFIED owning customer the hub resolved against
        :ptype customer_id: UUID
        :param handle: the resolved handle, carrying the object id it belongs to
        :ptype handle: ObjectHandle
        :return: nothing
        :rtype: None
        """
        entity_id = self._pk(customer_id, handle.object_id)
        now = datetime.now(UTC)
        data: dict[str, Any] = {
            # convert at border: L1 column / L2 key body
            "customer_id": str(customer_id),
            "object_id": str(handle.object_id),
            "s3_key": handle.s3_key,
            "mime_type": handle.mime_type,
            "size_bytes": handle.size_bytes,
            "summary": handle.summary,
            "category": handle.category,
            "date_created": now,
            "date_updated": now,
        }
        if self._l1 is not None:
            self._l1.upsert(self.table_name, data, self.primary_key_columns)
        await self._save_to_l2(entity_id, data)
        await self._publish_invalidation(entity_id)

    async def forget(self, customer_id: UUID, object_id: UUID) -> None:
        """drop a mapping from L1 and L2 and tell the other replicas.

        There is no upstream event that requires this today -- a committed object's key
        never changes -- so it exists for the case that does arise: an object deleted
        upstream, whose mapping must stop being served without waiting for a restart.

        :param customer_id: the VERIFIED owning customer
        :ptype customer_id: UUID
        :param object_id: the object whose mapping is being dropped
        :ptype object_id: UUID
        :return: nothing
        :rtype: None
        """
        entity_id = self._pk(customer_id, object_id)
        if self._l1 is not None:
            self._l1.delete_by_id(self.table_name, entity_id, self.primary_key_columns)
        await self._delete_from_l2(entity_id)
        await self._publish_invalidation(entity_id)

    @staticmethod
    def _pk(customer_id: UUID, object_id: UUID) -> tuple[str, str]:
        """build the declared-order addressing key for one mapping.

        :param customer_id: the VERIFIED owning customer
        :ptype customer_id: UUID
        :param object_id: the object id
        :ptype object_id: UUID
        :return: the composite pk in declared column order
        :rtype: tuple[str, str]
        """
        # convert at border: L1 pk values / L2 key body / invalidation envelope ids
        return (str(customer_id), str(object_id))


def _handle_from_row(row: dict[str, Any]) -> ObjectHandle:
    """rebuild an :class:`ObjectHandle` from a cached row.

    :param row: row dict as stored in L1 / L2
    :ptype row: dict[str, Any]
    :return: the handle the row describes
    :rtype: ObjectHandle
    """
    return ObjectHandle(
        object_id=UUID(str(row["object_id"])),
        s3_key=str(row["s3_key"]),
        mime_type=str(row["mime_type"]),
        size_bytes=int(row["size_bytes"]),
        summary=row.get("summary"),
        category=row.get("category"),
    )
