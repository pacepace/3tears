"""RoomFanout — the cross-pod room message backplane (channels-task-02).

the one genuinely net-new mechanism of the cross-pod design
(``docs/channels-cross-pod-design.md`` D1 / shard B): live room message
delivery across pods. no 3tears primitive provides it — the collections
give cross-pod *state* (task-01), this gives cross-pod *fanout*.

it composes :class:`~threetears.channels.presence.room_state.RoomState`
(task-01) for membership + the pod-local live-socket handles, and the
``NatsClient`` typed pub/sub for the broker hop. it mirrors the ``epoch``
module's publish→per-pod-local-act shape:

- :meth:`broadcast` **publishes one** :class:`RoomFrame` to
  :meth:`Subjects.room` and returns. it does **NOT** fan locally at
  publish time — every pod (including the sender's own, via the NATS
  echo path) fans on *receive*. a local fan at publish would
  double-deliver to the sender pod's members.
- :meth:`_deliver` is the per-pod subscription callback: it resolves
  this pod's local member sockets for the frame's room and sends the
  payload to each, skipping the excluded connection-id.

the fanout holds **no** room membership of its own (that is task-01's
``RoomState`` / ``PresenceCollection``). its only own state is the
per-room subscription **ref-count + handle**: transport bookkeeping, not
domain state — a pod subscribes to a room's subject only while it holds
≥1 local member (ref ``0→1`` on join), and unsubscribes when the last
local member leaves (ref ``1→0``). the ref-count + handle map is guarded
by an :class:`asyncio.Lock` because it is touched from both the WS loop
(join/leave) and the NATS callback path.

NO per-process ``seq``/ordering lives here (design D4): this is transient
fast-notify; durable ordering + replay is the op-log's job (scriob /
task-03). a dropped frame is not recovered here.
"""

from __future__ import annotations

import asyncio

from threetears.nats import NatsClient, Subjects
from threetears.nats.client import Subscription
from threetears.observe import get_logger

from threetears.channels.presence.room_state import RoomState
from threetears.channels.presence.wire import RoomFrame

__all__ = [
    "RoomFanout",
]

log = get_logger(__name__)


class RoomFanout:
    """thin cross-pod room-message backplane over :class:`RoomState` + NATS.

    one instance per pod, paired with that pod's :class:`RoomState`.
    composes membership/presence (task-01) for delivery targets and the
    ``NatsClient`` typed pub/sub for the cross-pod hop.

    :param room_state: this pod's presence/room state surface (membership
        + the pod-local live-socket handle map)
    :ptype room_state: RoomState
    :param nats_client: connected typed NATS wrapper for publish/subscribe
    :ptype nats_client: NatsClient
    """

    def __init__(self, room_state: RoomState, nats_client: NatsClient) -> None:
        """capture the state surface + nats client; no I/O.

        :param room_state: this pod's :class:`RoomState`
        :ptype room_state: RoomState
        :param nats_client: connected :class:`NatsClient`
        :ptype nats_client: NatsClient
        :return: nothing
        :rtype: None
        """
        self._room_state = room_state
        self._nats = nats_client
        # per-room subscription bookkeeping (transport, NOT domain state):
        # connection ref-count + the retained Subscription handle so the
        # last leave can tear the subscription down. guarded by ``_lock``
        # because join/leave (WS loop) and the NATS callback both touch it.
        self._ref_counts: dict[str, int] = {}
        self._subscriptions: dict[str, Subscription] = {}
        self._lock = asyncio.Lock()

    async def join_room(
        self,
        room_id: str,
        connection_id: str,
        user_id: str,
        customer_id: str,
    ) -> None:
        """record membership (task-01) and subscribe on the first local member.

        delegates membership/presence to
        :meth:`RoomState.join` (the cross-pod-coherent collection write),
        then increments this room's local ref-count; on the ``0→1``
        transition the pod subscribes to the room's NATS subject so it is
        listening before any broadcast arrives.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :param connection_id: opaque per-socket id of the joining socket
        :ptype connection_id: str
        :param user_id: authenticated principal
        :ptype user_id: str
        :param customer_id: tenant id (also embedded in ``room_id``)
        :ptype customer_id: str
        :return: nothing
        :rtype: None
        """
        await self._room_state.join(room_id, connection_id, user_id, customer_id)
        async with self._lock:
            count = self._ref_counts.get(room_id, 0)
            self._ref_counts[room_id] = count + 1
            if count == 0:
                # subscribe UNDER the lock so the ref bump and the stored
                # subscription handle move as ONE atomic step. were the
                # subscribe awaited outside the lock, a concurrent leave
                # could run in the window before the handle is stored, find
                # nothing to tear down, and orphan the subscription (the pod
                # stays subscribed to a room it holds no one in, forever).
                # join/leave are connection-lifecycle events, not the hot
                # path, so serializing the cheap subscribe call is fine; the
                # broadcast/_deliver path takes no lock across IO.
                await self._subscribe_locked(room_id)

    async def leave_room(self, room_id: str, connection_id: str) -> None:
        """drop membership (task-01) and unsubscribe on the last local member.

        delegates membership removal to :meth:`RoomState.leave`, then
        decrements this room's local ref-count; on the ``1→0`` transition
        the pod unsubscribes from the room's NATS subject — pods never
        stay subscribed to a room they hold no one in.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :param connection_id: opaque per-socket id of the leaving socket
        :ptype connection_id: str
        :return: nothing
        :rtype: None
        """
        await self._room_state.leave(room_id, connection_id)
        async with self._lock:
            count = self._ref_counts.get(room_id, 0)
            if count > 1:
                self._ref_counts[room_id] = count - 1
                return
            # last local member (or a defensive count <= 0): drop the ref and
            # tear the subscription down atomically with it, under the lock —
            # the symmetric counterpart to the atomic subscribe in join_room.
            self._ref_counts.pop(room_id, None)
            await self._unsubscribe_locked(room_id)

    async def broadcast(
        self,
        room_id: str,
        payload: str,
        *,
        exclude: str | None = None,
    ) -> None:
        """publish ONE room frame; every pod fans to its own sockets on receive.

        **publish-only.** this method does NOT fan to local sockets — the
        sender pod receives its own published frame back through its NATS
        subscription (the echo path) and fans in :meth:`_deliver`, exactly
        like every other pod. fanning here too would double-deliver to the
        sender pod's local members.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :param payload: opaque message body delivered verbatim to each
            local socket
        :ptype payload: str
        :param exclude: connection-id to omit from delivery on every pod
            (typically the originating socket), or ``None`` for everyone
        :ptype exclude: str | None
        :return: nothing
        :rtype: None
        """
        frame = RoomFrame(
            room_id=room_id,
            payload=payload,
            exclude=exclude,
            origin_pod=self._room_state.pod_id,
        )
        await self._nats.publish(subject=Subjects.room(room_id), message=frame)

    async def _deliver(self, frame: RoomFrame) -> None:
        """subscription callback: fan a received frame to local sockets.

        resolves THIS pod's local member ``(connection_id, socket)`` pairs
        for the frame's room (a task-01 snapshot, taken under the state
        lock and released before the sends), then sends the payload to
        each socket — skipping the connection whose id equals
        ``frame.exclude``. delivery is best-effort PER socket: a send that
        raises (a dead/closing handle) is logged and skipped so it cannot
        starve the room's other members or deadletter the whole frame.

        :param frame: the received room frame
        :ptype frame: RoomFrame
        :return: nothing
        :rtype: None
        """
        pairs = await self._room_state.local_member_sockets(frame.room_id)
        for connection_id, socket in pairs:
            if frame.exclude is not None and connection_id == frame.exclude:
                continue
            try:
                await socket.send_text(frame.payload)
            except Exception:  # prawduct:allow prawduct/broad-except -- best-effort transient fanout: one dead socket must not starve the room's other members or deadletter the frame
                log.debug(
                    "room fanout delivery to a socket failed; skipping it",
                    extra={
                        "extra_data": {
                            "room_id": frame.room_id,
                            "connection_id": connection_id,
                            "pod_id": self._room_state.pod_id,
                        }
                    },
                )

    async def _subscribe_locked(self, room_id: str) -> None:
        """subscribe this pod to a room's subject. **Caller holds ``_lock``.**

        registers the typed subscription and stores its handle in the same
        critical section as the ``0→1`` ref bump, so the ref count and the
        handle are never observable out of step (the orphaned-subscription
        race). exactly one subscription exists per room with a local member.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :return: nothing
        :rtype: None
        """
        self._subscriptions[room_id] = await self._nats.subscribe_typed(
            subject=Subjects.room(room_id),
            message_type=RoomFrame,
            cb=self._deliver,
        )
        log.debug(
            "room fanout subscribed",
            extra={"extra_data": {"room_id": room_id, "pod_id": self._room_state.pod_id}},
        )

    async def _unsubscribe_locked(self, room_id: str) -> None:
        """unsubscribe this pod from a room's subject. **Caller holds ``_lock``.**

        pops + tears down the retained subscription handle in the same
        critical section as the ``1→0`` ref drop. idempotent: a room with
        no live handle (already torn down) is a no-op.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :return: nothing
        :rtype: None
        """
        subscription = self._subscriptions.pop(room_id, None)
        if subscription is None:
            return
        await subscription.unsubscribe()
        log.debug(
            "room fanout unsubscribed",
            extra={"extra_data": {"room_id": room_id, "pod_id": self._room_state.pod_id}},
        )
