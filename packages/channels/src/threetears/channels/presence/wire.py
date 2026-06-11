"""typed wire envelope for cross-pod room fanout (channels-task-02).

:class:`RoomFanout` publishes exactly one :class:`RoomFrame` per
:meth:`~threetears.channels.presence.fanout.RoomFanout.broadcast` call to
:meth:`threetears.nats.subjects.Subjects.room`. every pod holding a local
member of the room subscribes via
:meth:`threetears.nats.NatsClient.subscribe_typed` with
``message_type=RoomFrame`` and delivers ``payload`` to its own live
sockets — validation failures deadletter via the standard typed-NATS path
and never reach the delivery callback.

this is **transient fast-notify** transport (design D4): the frame carries
NO sequence / ordering field. durable ordering + replay is the op-log's
job (scriob), never this layer — adding a ``seq`` here would create a
second, racing source of truth. an enforcement test
(``tests/unit/channels/presence/test_fanout.py``) freezes the field set
against an accidental ordering field.

mirrors :class:`threetears.epoch.wire.EpochBumpMessage` discipline: typed
Pydantic envelope, ``model_dump_json`` serialization, opaque to the
framework after parsing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = [
    "RoomFrame",
]


class RoomFrame(BaseModel):
    """typed wire envelope for one cross-pod room message.

    :ivar room_id: raw ``{customer}:{story}:{branch}:{file}`` room key the
        frame targets; rides in the envelope because the NATS subject
        token is a one-way SHA-256 digest of it (see
        :meth:`threetears.nats.subjects.Subjects.room`)
    :ivar payload: the opaque message body delivered verbatim to each
        local socket via ``send_text``; the framework never inspects it
    :ivar exclude: connection-id to omit from delivery on every pod
        (typically the originating socket), or ``None`` to deliver to
        every member; honoured after the cross-pod hop, by connection-id
    :ivar origin_pod: id of the pod that published the frame; diagnostics
        only — delivery never depends on it
    """

    # ``extra="ignore"`` keeps the wire forward-compatible (a newer
    # publisher may carry fields an older subscriber does not understand;
    # silently dropping them keeps the migration additive). ``frozen=True``
    # forbids mutation after construction so a dispatched frame cannot be
    # mutated mid-delivery. NOTE: deliberately NO ``seq``/ordering field —
    # ordering is the durable op-log's job (design D4).
    model_config = ConfigDict(frozen=True, extra="ignore")

    room_id: str
    payload: str
    exclude: str | None = None
    origin_pod: str
