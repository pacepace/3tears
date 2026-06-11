"""RoomState — the reshaped, cross-pod-safe connection/room registry.

channels-task-01 replaces the racy single-pod dict ``ConnectionRegistry``
(two in-process dicts mutated while iterated across ``await``) with this
state layer. membership/presence lives in the
:class:`~threetears.channels.presence.collection.PresenceCollection`
(L1 SQLite + L2 NATS, cross-pod-coherent), **never dicts**. the ONLY
in-process structure here is the synchronized ``connection_id → live
socket`` map — the unavoidable map of non-serializable live handles that
genuinely cannot leave the pod.

concurrency discipline for the socket map (this stack is async AND
threaded):

- the map is guarded by a single :class:`asyncio.Lock`.
- mutations (register/unregister) take the lock briefly.
- reads that fan out (``local_sockets``) take the lock only to copy a
  snapshot list, release it, and return the snapshot — the caller
  iterates the snapshot and ``await``\\ s sends with the lock already
  released. the lock is NEVER held across an ``await`` of a socket send,
  and the live map is NEVER iterated directly.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from threetears.observe import get_logger

from threetears.channels.presence.collection import PresenceCollection

__all__ = ["RoomMember", "RoomState"]

log = get_logger(__name__)


class RoomMember:
    """a single room member resolved from the presence collection.

    a lightweight value object returned by :meth:`RoomState.members`,
    carrying the cross-pod-visible identity of one connection in a room.
    holds NO live socket handle — those live only in the pod-local map
    and are resolved (for the local subset) via
    :meth:`RoomState.local_sockets`.

    :ivar connection_id: opaque per-socket id
    :ivar user_id: authenticated principal
    :ivar pod_id: id of the pod hosting the live socket
    :ivar customer_id: tenant id
    """

    __slots__ = ("connection_id", "customer_id", "pod_id", "user_id")

    def __init__(self, connection_id: str, user_id: str, pod_id: str, customer_id: str) -> None:
        self.connection_id = connection_id
        self.user_id = user_id
        self.pod_id = pod_id
        self.customer_id = customer_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RoomMember):
            return NotImplemented
        return (
            self.connection_id == other.connection_id
            and self.user_id == other.user_id
            and self.pod_id == other.pod_id
            and self.customer_id == other.customer_id
        )

    def __hash__(self) -> int:
        return hash((self.connection_id, self.user_id, self.pod_id, self.customer_id))

    def __repr__(self) -> str:
        return (
            f"RoomMember(connection_id={self.connection_id!r}, user_id={self.user_id!r}, "
            f"pod_id={self.pod_id!r}, customer_id={self.customer_id!r})"
        )


class RoomState:
    """cross-pod connection/room registry over the presence collection.

    one instance per pod. membership + presence go through the injected
    :class:`PresenceCollection` (cross-pod via L2 + invalidation); the
    only pod-local structure is the synchronized ``connection_id → live
    socket`` map.

    :param collection: the presence state surface (per-connection +
        room-index Collections)
    :ptype collection: PresenceCollection
    :param pod_id: id of the pod this instance runs on; stamped on every
        per-connection presence row so the fanout can resolve which pod
        holds a member's live handle
    :ptype pod_id: str
    """

    def __init__(self, collection: PresenceCollection, pod_id: str) -> None:
        self._collection = collection
        self._pod_id = pod_id
        # The ONE in-process structure: live, non-serializable socket
        # handles keyed by connection_id. Guarded by ``_lock``; never
        # iterated directly, never mutated across an ``await`` of a send.
        self._sockets: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    @property
    def pod_id(self) -> str:
        """return the id of the pod this state instance runs on.

        :return: pod id
        :rtype: str
        """
        return self._pod_id

    async def register(self, connection_id: str, socket: Any) -> None:
        """record a live socket handle for a connection (pod-local).

        :param connection_id: opaque per-socket id
        :ptype connection_id: str
        :param socket: live socket handle (non-serializable)
        :ptype socket: Any
        :return: nothing
        :rtype: None
        """
        async with self._lock:
            self._sockets[connection_id] = socket

    async def unregister(self, connection_id: str, socket: Any | None = None) -> None:
        """drop a connection's live socket handle (pod-local).

        when ``socket`` is supplied the handle is removed only if it
        still matches (guards against a reconnect having replaced it);
        when ``None`` the handle is removed unconditionally. safe to call
        for an unknown connection.

        :param connection_id: opaque per-socket id
        :ptype connection_id: str
        :param socket: the specific handle to remove, or ``None`` to
            remove whatever is mapped
        :ptype socket: Any | None
        :return: nothing
        :rtype: None
        """
        async with self._lock:
            current = self._sockets.get(connection_id)
            if current is None:
                return
            if socket is None or current is socket:
                del self._sockets[connection_id]

    async def join(
        self,
        room_id: str,
        connection_id: str,
        user_id: str,
        customer_id: str,
        pod_id: str | None = None,
    ) -> None:
        """add a connection to a room and write its presence row.

        writes the per-connection presence row (identity + heartbeat)
        then CAS-adds the connection to the room-index member set. both
        propagate cross-pod via the collection invalidation envelope, so
        a peer pod's :meth:`members` sees the new member on its next
        read.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :param connection_id: opaque per-socket id
        :ptype connection_id: str
        :param user_id: authenticated principal
        :ptype user_id: str
        :param customer_id: tenant id (also embedded in ``room_id``)
        :ptype customer_id: str
        :param pod_id: hosting pod id; defaults to this instance's pod
        :ptype pod_id: str | None
        :return: nothing
        :rtype: None
        """
        now = datetime.now(UTC)
        entity = self._collection.connections.create(
            {
                "connection_id": connection_id,
                "room_id": room_id,
                "user_id": user_id,
                "pod_id": pod_id if pod_id is not None else self._pod_id,
                "customer_id": customer_id,
                "date_last_heartbeat": now,
            }
        )
        await self._collection.connections.save_entity(entity)
        await self._collection.rooms.add_member(room_id, customer_id, connection_id)
        log.info(
            "connection joined room",
            extra={
                "extra_data": {
                    "room_id": room_id,
                    "connection_id": connection_id,
                    "user_id": user_id,
                    "customer_id": customer_id,
                }
            },
        )

    async def leave(self, room_id: str, connection_id: str) -> None:
        """remove a connection from a room and drop its presence row.

        CAS-removes the connection from the room-index member set and
        deletes its per-connection presence row. cross-pod-visible via
        the invalidation envelope.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :param connection_id: opaque per-socket id
        :ptype connection_id: str
        :return: nothing
        :rtype: None
        """
        await self._collection.rooms.remove_member(room_id, connection_id)
        await self._collection.connections.delete(connection_id)
        log.info(
            "connection left room",
            extra={"extra_data": {"room_id": room_id, "connection_id": connection_id}},
        )

    async def heartbeat(self, connection_id: str) -> bool:
        """refresh a connection's ``date_last_heartbeat`` (no room CAS).

        touches the per-connection presence row ONLY — the busy heartbeat
        path never contends on the room-index. a heartbeat for a
        connection with no presence row (already swept / never joined) is
        a no-op.

        :param connection_id: opaque per-socket id
        :ptype connection_id: str
        :return: ``True`` if a row was refreshed, ``False`` if none
            existed
        :rtype: bool
        """
        entity = await self._collection.connections.get(connection_id)
        if entity is None:
            return False
        entity.date_last_heartbeat = datetime.now(UTC)
        await self._collection.connections.save_entity(entity)
        return True

    async def members(self, room_id: str) -> list[RoomMember]:
        """return every member of a room across all pods.

        a pk-get of the room-index row (cross-pod-coherent via L2 +
        invalidation), then a pk-get of each member's per-connection
        presence row to resolve identity. a member whose presence row is
        already gone (mid-sweep) is skipped. never a secondary scan.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :return: members across all pods, in room-index order
        :rtype: list[RoomMember]
        """
        member_ids = await self._collection.rooms.members(room_id)
        result: list[RoomMember] = []
        for connection_id in member_ids:
            conn = await self._collection.connections.get(connection_id)
            if conn is None:
                continue
            result.append(
                RoomMember(
                    connection_id=connection_id,
                    user_id=conn.user_id,
                    pod_id=conn.pod_id,
                    customer_id=conn.customer_id,
                )
            )
        return result

    async def local_sockets(self, room_id: str) -> list[Any]:
        """return live socket handles for this pod's members of a room.

        resolves the room's member ids (cross-pod), then snapshots — under
        the lock — the live handles this pod holds for those ids. the
        snapshot is built and the lock released BEFORE the caller fans
        out, so the fanout (task-02) never holds the lock across a send
        and never iterates the live map.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :return: live socket handles for the local members (a snapshot
            list)
        :rtype: list[Any]
        """
        member_ids = await self._collection.rooms.members(room_id)
        async with self._lock:
            # snapshot under the lock; iterate it (not the live map) after release
            return [self._sockets[cid] for cid in member_ids if cid in self._sockets]

    async def local_socket(self, connection_id: str) -> Any | None:
        """return the live socket handle for one connection, if local.

        :param connection_id: opaque per-socket id
        :ptype connection_id: str
        :return: the live handle, or ``None`` if this pod does not hold it
        :rtype: Any | None
        """
        async with self._lock:
            return self._sockets.get(connection_id)

    async def local_connection_count(self) -> int:
        """return how many live sockets this pod currently holds.

        :return: count of pod-local live socket handles
        :rtype: int
        """
        async with self._lock:
            return len(self._sockets)
