"""L1+L2-only presence collections for cross-pod room membership.

channels-task-01 replaces the racy single-pod dict ``ConnectionRegistry``
with a concurrency-safe, cross-pod, tenancy-ready state layer. presence
state lives in :class:`~threetears.core.collections.base.BaseCollection`
instances (L1 SQLite + L2 NATS KV) — **never dicts** — modelled on
:class:`~threetears.registry.heartbeat_collection.HeartbeatCollection`.

``BaseCollection`` is pk-keyed only (no secondary-field query; the KV
wrapper exposes no key-listing; ``SchemaBackedCollection``'s queries hit
L3/postgres, wrong for ephemeral presence). so "who is in room X" is a
**pk-get**, never a scan. the state is modelled as two pk-keyed entry
types, each its own Collection over its own L1 table:

- :class:`PresenceConnectionCollection` (pk ``connection_id``) — one row
  per live connection; heartbeats refresh THIS row only.
- :class:`RoomIndexCollection` (pk ``room_id``) — one row per room
  carrying the member ``connection_id`` set; updated only on join/leave
  under optimistic-concurrency CAS.

:class:`PresenceCollection` is the thin facade binding the two together
and is what callers (the registry / sweeper) construct and consume.

both Collections are L1+L2 only: ``l3_pool`` is forced to ``None`` and
the three L3 methods raise loudly, exactly like ``HeartbeatCollection``.
cross-pod coherence rides the collection invalidation envelope
(:func:`~threetears.nats.Subjects.cache_invalidate`) — free with the L2
write path: a join on pod A invalidates pod B's L1 copy of the
room-index, and B refills from L2 on its next ``members`` read.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.observe import get_logger

from threetears.channels.presence.entities import (
    PresenceConnectionEntity,
    RoomIndexEntity,
)

__all__ = [
    "PresenceCollection",
    "PresenceConnectionCollection",
    "RoomIndexCollection",
]

log = get_logger(__name__)

#: how many times :meth:`RoomIndexCollection.add_member` /
#: :meth:`RoomIndexCollection.remove_member` retry on an optimistic-lock
#: conflict before surfacing the error. join/leave is low-churn, so a
#: handful of retries comfortably absorbs the rare concurrent-write race
#: on the same room without masking a genuine livelock.
_CAS_MAX_RETRIES = 8

#: outcome of computing a room-index member change: write the row,
#: delete it (last member left), or do nothing (idempotent add / absent
#: remove). distinguishing "delete" from "noop" keeps the empty-room
#: cleanup explicit.
_MemberAction = Literal["upsert", "delete", "noop"]


def _coerce_datetimes(raw: dict[str, Any], fields: tuple[str, ...]) -> None:
    """rehydrate ISO datetime strings to aware-UTC in place.

    mirrors :meth:`HeartbeatCollection.deserialize`: the L2 JSON codec
    renders datetimes as ISO strings; on the way back any naive value
    (legacy / hand-written payload) is coerced to aware-UTC so the rest
    of the pipeline can rely on aware comparisons (the sweep does
    ``datetime.now(UTC) - entity.date_last_heartbeat``).

    :param raw: row dict mutated in place
    :ptype raw: dict[str, Any]
    :param fields: datetime column names to coerce
    :ptype fields: tuple[str, ...]
    :return: nothing
    :rtype: None
    """
    for field_name in fields:
        value = raw.get(field_name)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            raw[field_name] = parsed


class _L1L2OnlyCollection(BaseCollection[Any]):
    """shared L1+L2-only base for the two presence Collections.

    forces ``l3_pool = None`` and raises on every L3 method so an
    accidental framework invocation surfaces loudly rather than silently
    no-opping. ``get`` is overridden so an L1+L2 total-miss resolves to
    ``None`` instead of falling through to ``fetch_from_postgres`` (which
    raises by design). exactly the ``HeartbeatCollection`` shape.

    subclasses supply :attr:`table_name`, :attr:`entity_class`,
    :attr:`primary_key_column`, and the datetime columns to rehydrate.
    """

    #: datetime columns the L2 codec must rehydrate to aware-UTC.
    _datetime_columns: tuple[str, ...] = ("date_created", "date_updated")

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """wire the Collection against the registry and force L3 off.

        :param registry: shared collection registry; ``l1_backend`` and
            ``l2_client`` are resolved through it identically to every
            other pod-side Collection. any registry ``l3_pool`` default
            is deliberately discarded — presence never round-trips L3.
        :ptype registry: CollectionRegistry
        :param config: core configuration; the write path short-circuits
            before the flush strategy is consulted, so the value is
            effectively ignored beyond DI symmetry with sibling
            Collections.
        :ptype config: CoreConfig
        :param nats_client: NATS client for L2 cache coherence. may be
            ``None`` for L1-only operation (unit tests, single-pod
            deployments); every L2 hop guards on ``None``.
        :ptype nats_client: Any
        :param write_buffer: unused; presence writes bypass the
            deferred-flush path entirely (no L3).
        :ptype write_buffer: WriteBuffer | None
        """
        super().__init__(registry, config, nats_client, write_buffer)
        # override whatever L3 pool the registry snapped on: this
        # Collection is L1+L2 only by design. the base-class L3 guards
        # (``if self.l3_pool is None: return``) then fire before the
        # method-level overrides can raise.
        self.l3_pool = None

    async def get(self, entity_id: Any) -> Any:
        """read from L1 first, pull-through L2 on miss; ``None`` on total-miss.

        overrides the three-tier read so an L1+L2 miss resolves to
        ``None`` rather than tripping ``fetch_from_postgres`` (which
        raises by design). mirrors :meth:`HeartbeatCollection.get`.

        :param entity_id: primary key value
        :ptype entity_id: Any
        :return: hydrated entity, or ``None`` on miss
        :rtype: Any
        """
        row: dict[str, Any] | None = self.get_row_sync(entity_id)
        if row is None:
            l2_data = await self._get_from_l2(entity_id)
            if l2_data is not None:
                if self._l1 is not None:
                    self._l1.upsert(self.table_name, l2_data, self.primary_key_columns)
                row = l2_data
        if row is None:
            return None
        result = self.entity_class(row, is_new=False, collection=self)
        result.original_date_updated = row.get("date_updated")
        return result

    async def save_entity(self, entity: Any, *, conn: Any = None) -> None:
        """persist entity to L1 + L2 (no L3), then publish invalidation.

        overrides the three-tier write: the base path starts with
        ``save_to_postgres`` (which raises here). lays down L1 first,
        then L2, then the cross-pod invalidation. mirrors
        :meth:`HeartbeatCollection.save_entity`.

        :param entity: entity to persist
        :ptype entity: Any
        :param conn: ignored; kept for LSP parity with the base class
        :ptype conn: Any
        :return: nothing
        :rtype: None
        """
        del conn  # never threaded through the L1+L2-only path
        data = entity.to_dict()
        now = datetime.now(UTC)
        if entity.is_new:
            data.setdefault("date_created", now)
        data["date_updated"] = now
        if self._l1 is not None:
            self._l1.upsert(self.table_name, data, self.primary_key_columns)
        await self._save_to_l2(entity.id, data)
        await self._publish_invalidation(entity.id)
        entity.mark_clean()
        entity.original_date_updated = data.get("date_updated")

    async def delete(self, entity_id: Any) -> bool:
        """remove the row from L1 and L2 and notify peers.

        :param entity_id: primary key value
        :ptype entity_id: Any
        :return: ``True`` unconditionally — deletes are idempotent
        :rtype: bool
        """
        if self._l1 is not None:
            self._l1.delete_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)
        await self._delete_from_l2(entity_id)
        await self._publish_invalidation(entity_id)
        return True

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """unreachable — presence is L1+L2 only.

        :param entity_id: ignored; kept for signature symmetry
        :ptype entity_id: Any
        :return: never returns
        :rtype: dict[str, Any] | None
        :raises RuntimeError: always; no L3 pool is bound for presence
        """
        raise RuntimeError(
            f"{type(self).__name__} is L1+L2 only; fetch_from_postgres must "
            f"never be reached (no L3 pool bound for '{self.table_name}')",
        )

    async def save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """unreachable — presence is L1+L2 only.

        :param data: ignored; kept for signature symmetry
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored
        :ptype original_timestamp: datetime | None
        :param conn: ignored; kept for LSP parity with the base class
        :ptype conn: Any
        :return: never returns
        :rtype: int
        :raises RuntimeError: always; no L3 pool is bound for presence
        """
        raise RuntimeError(
            f"{type(self).__name__} is L1+L2 only; save_to_postgres must "
            f"never be reached (save_entity is overridden to skip L3)",
        )

    async def delete_from_postgres(self, entity_id: Any) -> None:
        """unreachable — presence is L1+L2 only.

        :param entity_id: ignored; kept for signature symmetry
        :ptype entity_id: Any
        :return: never returns
        :rtype: None
        :raises RuntimeError: always; no L3 pool is bound for presence
        """
        raise RuntimeError(
            f"{type(self).__name__} is L1+L2 only; delete_from_postgres must "
            f"never be reached (delete is overridden to skip L3)",
        )

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

        rehydrates the subclass's declared datetime columns to aware-UTC
        at the boundary.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row dict with aware-UTC datetime columns
        :rtype: dict[str, Any]
        """
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        _coerce_datetimes(raw, self._datetime_columns)
        return raw


class PresenceConnectionCollection(_L1L2OnlyCollection):
    """L1+L2 Collection of per-connection presence rows (pk ``connection_id``).

    one row per live websocket connection. heartbeats refresh the
    ``date_last_heartbeat`` of THIS row only — no room-index contention.
    """

    primary_key_column: str = "connection_id"
    _datetime_columns = ("date_last_heartbeat", "date_created", "date_updated")

    @property
    def table_name(self) -> str:
        """return the L1 table name holding per-connection rows.

        :return: table name
        :rtype: str
        """
        return "presence_connections"

    @property
    def entity_class(self) -> type[PresenceConnectionEntity]:
        """return the entity class for this Collection.

        :return: :class:`PresenceConnectionEntity`
        :rtype: type[PresenceConnectionEntity]
        """
        return PresenceConnectionEntity


class RoomIndexCollection(_L1L2OnlyCollection):
    """L1+L2 Collection of room-index rows (pk ``room_id``).

    one row per ``{customer}:{story}:{branch}:{file}`` room carrying the
    member ``connection_id`` set. updated only on join/leave under
    optimistic-concurrency CAS (retry on revision conflict) via the
    framework primitive :meth:`~threetears.core.collections.base.BaseCollection.l2_cas_mutate`.

    the room id (``{customer}:{story}:{branch}:{file}``) carries ``:``
    separators AND app-supplied segments (branch/file) that may hold any
    character — neither fits the JetStream KV key grammar
    (``^[-/_=.a-zA-Z0-9]+$``). the framework
    :meth:`~threetears.core.collections.base.BaseCollection.l2_key` now
    detects the out-of-grammar body and keys off a SHA-256 hex digest of
    it (always grammar-valid, collision-resistant), so this collection
    needs no ``l2_key`` override. the raw room id round-trips unchanged
    through L1 and the invalidation envelope, so cross-pod coherence is
    unaffected.
    """

    primary_key_column: str = "room_id"

    @property
    def table_name(self) -> str:
        """return the L1 table name holding room-index rows.

        :return: table name
        :rtype: str
        """
        return "presence_rooms"

    @property
    def entity_class(self) -> type[RoomIndexEntity]:
        """return the entity class for this Collection.

        :return: :class:`RoomIndexEntity`
        :rtype: type[RoomIndexEntity]
        """
        return RoomIndexEntity

    async def members(self, room_id: str) -> list[str]:
        """return the current member connection-id set for a room.

        a pk-get of the room-index row (cross-pod-complete: the row is
        pk-keyed and L2-coherent, so a join on a peer pod has already
        invalidated this pod's stale L1 copy). a missing room is empty.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :return: member connection ids (a fresh list copy)
        :rtype: list[str]
        """
        entity = await self.get(room_id)
        if entity is None:
            return []
        current: list[str] = list(entity.members)
        return current

    async def add_member(self, room_id: str, customer_id: str, connection_id: str) -> None:
        """add a connection to a room's member set under L2 CAS.

        reads the room-index row + its L2 revision, appends
        ``connection_id`` if absent, and writes back under a
        compare-and-swap so a concurrent join/leave on the SAME room
        (even from another pod) cannot lose a write. creates the
        room-index row on first join. idempotent: a connection already
        present is a no-op.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :param customer_id: tenant id owning the room (denormalised onto
            the index row)
        :ptype customer_id: str
        :param connection_id: connection to add
        :ptype connection_id: str
        :return: nothing
        :rtype: None
        :raises ConcurrentModificationError: if the CAS retry budget is
            exhausted (a genuine livelock, never the common case)
        """
        await self.l2_cas_mutate(
            room_id,
            lambda row: self._apply_member_change(row, room_id, customer_id, connection_id, add=True),
            max_retries=_CAS_MAX_RETRIES,
        )

    async def remove_member(self, room_id: str, connection_id: str) -> None:
        """remove a connection from a room's member set under L2 CAS.

        reads the room-index row + its L2 revision, drops
        ``connection_id`` if present, and writes back under CAS. when
        the last member leaves, the room-index row is deleted so an
        empty room leaves no lingering state. a missing room or absent
        member is a no-op.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :param connection_id: connection to remove
        :ptype connection_id: str
        :return: nothing
        :rtype: None
        :raises ConcurrentModificationError: if the CAS retry budget is
            exhausted
        """
        await self.l2_cas_mutate(
            room_id,
            lambda row: self._apply_member_change(row, room_id, None, connection_id, add=False),
            max_retries=_CAS_MAX_RETRIES,
        )

    def _apply_member_change(
        self,
        row: dict[str, Any] | None,
        room_id: str,
        customer_id: str | None,
        connection_id: str,
        *,
        add: bool,
    ) -> tuple[_MemberAction, dict[str, Any] | None]:
        """compute the next room-index row for an add/remove.

        returns ``("upsert", new_row)`` to write the row,
        ``("delete", None)`` when the last member left, or
        ``("noop", None)`` for an idempotent add / absent remove.

        :param row: current row dict, or ``None`` when absent
        :ptype row: dict[str, Any] | None
        :param room_id: room key
        :ptype room_id: str
        :param customer_id: tenant id (required to create on add)
        :ptype customer_id: str | None
        :param connection_id: connection to add or remove
        :ptype connection_id: str
        :param add: ``True`` to add, ``False`` to remove
        :ptype add: bool
        :return: ``(action, new_row)`` — ``new_row`` is non-``None`` iff
            ``action == "upsert"``
        :rtype: tuple[_MemberAction, dict[str, Any] | None]
        """
        if add:
            if row is None:
                if customer_id is None:
                    raise ValueError("customer_id is required to create a room index on join")
                return "upsert", {"room_id": room_id, "customer_id": customer_id, "members": [connection_id]}
            members = list(row.get("members", []))
            if connection_id in members:
                return "noop", None
            members.append(connection_id)
            return "upsert", {**row, "members": members}
        if row is None:
            return "noop", None
        members = list(row.get("members", []))
        if connection_id not in members:
            return "noop", None
        members.remove(connection_id)
        if not members:
            return "delete", None
        return "upsert", {**row, "members": members}


class PresenceCollection:
    """facade binding the per-connection and room-index Collections.

    the public surface the registry + sweeper construct and consume.
    holds the two pk-keyed Collections — :attr:`connections` (pk
    ``connection_id``) and :attr:`rooms` (pk ``room_id``) — both L1+L2
    only and wired against the same registry / NATS client, so a write
    on either propagates the invalidation envelope cross-pod.

    keeping the two Collections behind one facade keeps each strictly
    pk-keyed (no secondary scan) while giving callers a single object to
    inject. there is no shared dict here — all state lives in the
    Collections' L1+L2 tiers.
    """

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """construct both presence Collections against one registry.

        :param registry: shared collection registry (provides L1 backend
            + L2 client)
        :ptype registry: CollectionRegistry
        :param config: core configuration
        :ptype config: CoreConfig
        :param nats_client: NATS client for L2 coherence; ``None`` for
            L1-only operation
        :ptype nats_client: Any
        :param write_buffer: unused (no L3); kept for DI symmetry
        :ptype write_buffer: WriteBuffer | None
        """
        self.connections = PresenceConnectionCollection(registry, config, nats_client, write_buffer)
        self.rooms = RoomIndexCollection(registry, config, nats_client, write_buffer)
