"""typed frame envelope + injected-seam protocols (channels-task-03).

this module is the typed boundary the cross-pod WebSocket handler routes
on (``docs/channels-task-03-authz-typed-frames-resume.md``):

- :class:`Frame` — the typed inbound/outbound envelope every raw
  websocket message is parsed into (design T3-D2): a ``type``
  discriminator plus the optional ``room`` / ``payload`` / ``seq`` fields
  the room-aware frame types carry. ``extra="ignore"`` keeps the wire
  forward-compatible (an older subscriber tolerates a newer publisher's
  unknown keys). it carries **no** sequence counter of its own — the only
  ``seq`` it ever holds is the durable op-log sequence echoed in (on a
  ``resume`` cursor) or out (on a broadcast op frame); ordering is the
  op-log's job (design T3-D4), never this envelope's.
- the injected-seam **protocols** (:class:`NsResolver` / :class:`OpHandler`
  / :class:`ReplaySource`) + the supporting :class:`OpResult` and the
  structural :class:`NsEntity` — the scriob→channels boundary seams
  (T3-D1/D3/D4). channels depends only on these shapes, so it composes the
  policy (the ACL namespace resolver), the durable append (the op-log
  handler), and the replay tail (the op-log replay) scriob injects without
  importing any scriob type.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

__all__ = [
    "Frame",
    "NsEntity",
    "NsResolver",
    "OpHandler",
    "OpResult",
    "ReplaySource",
]


class Frame(BaseModel):
    """typed inbound/outbound websocket frame envelope.

    a minimal, forward-compatible envelope the typed-frame router parses
    every raw message into and dispatches by :attr:`type`. only ``type``
    is required; the room-aware frame types add ``room`` (the
    ``{customer}:{story}:{branch}:{file}`` key), an opaque ``payload``,
    and — for the durable op / resume cursor path — a ``seq`` (the op-log
    sequence, **never** an in-process counter; design T3-D4).

    :ivar type: frame discriminator (``message`` / ``join`` / ``leave`` /
        ``editor.op`` / ``cursor`` / ``typing`` / ``presence`` /
        ``resume`` / ``error`` …)
    :ivar room: target room key, or ``None`` for non-room frames (chat)
    :ivar payload: opaque body broadcast verbatim to room members, or
        ``None``
    :ivar seq: the durable op-log sequence — inbound on a ``resume``
        cursor (resume from this seq) and outbound on a broadcast op frame
        (the seq the injected ``op_handler`` assigned); ``None`` for
        transient frames
    """

    # ``extra="ignore"`` keeps the wire forward-compatible: a newer client
    # may send fields an older handler does not understand; dropping them
    # silently keeps the migration additive (mirrors ``RoomFrame``).
    # deliberately NO mint-your-own ``seq`` counter — the only sequence is
    # the durable op-log's, echoed through this field (design T3-D4).
    model_config = ConfigDict(extra="ignore")

    type: str
    room: str | None = None
    payload: str | None = None
    seq: int | None = None

    @classmethod
    def from_raw(cls, raw: str) -> Frame:
        """parse raw JSON text into a :class:`Frame`.

        :param raw: raw websocket text message
        :ptype raw: str
        :return: the parsed, typed frame
        :rtype: Frame
        :raises json.JSONDecodeError: when ``raw`` is not valid JSON (the
            caller emits an ``error`` frame)
        """
        return cls.model_validate(json.loads(raw))

    @staticmethod
    def error(message: str) -> str:
        """build a serialized outbound ``error`` frame.

        returns wire-ready JSON text (not a :class:`Frame`) so the caller
        sends it straight through ``send_text`` — matching the existing
        ``{"type": "error", "message": …}`` shape the chat path already
        emits, so an unknown / denied / malformed frame surfaces to the
        client identically.

        :param message: human-readable error description
        :ptype message: str
        :return: JSON text of the error frame
        :rtype: str
        """
        return json.dumps({"type": "error", "message": message})


@dataclass(frozen=True)
class OpResult:
    """result of an injected durable op append (design T3-D3).

    the injected :class:`OpHandler` appends an ``editor.op`` to scriob's
    op-log and returns this carrying the **authoritative** op-log sequence
    the handler then broadcasts on the op frame. channels keeps no
    sequence of its own (design T3-D4); this ``seq`` is the op-log's.

    :ivar seq: the op-log sequence assigned to the appended op
    """

    seq: int


@runtime_checkable
class NsEntity(Protocol):
    """structural namespace entity ``authorize_on_entity`` reads (T3-D1).

    the four attributes ``threetears.agent.acl.authorize_on_entity`` reads
    off its ``ns_entity``. declared structurally so channels resolves a
    room to its policy namespace without importing a scriob (or hub)
    entity type — the injected :class:`NsResolver` returns anything that
    exposes these.
    """

    @property
    def id(self) -> object: ...

    @property
    def customer_id(self) -> object: ...

    @property
    def namespace_type(self) -> object: ...

    @property
    def owner_agent_id(self) -> object: ...


@runtime_checkable
class NsResolver(Protocol):
    """injected seam: resolve a room id to its ACL namespace entity (T3-D1).

    scriob injects this (room → the story/branch namespace = its policy)
    so channels can call ``authorize_on_entity`` on the room without
    knowing how scriob maps rooms to namespaces.
    """

    async def __call__(self, room_id: str) -> NsEntity:
        """resolve ``room_id`` to the namespace entity authz evaluates against.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :return: the namespace entity for the room
        :rtype: NsEntity
        """
        ...


@runtime_checkable
class OpHandler(Protocol):
    """injected seam: append an ``editor.op`` durably, return its seq (T3-D3).

    scriob injects this (the op-log append). channels routes ``editor.op``
    frames here, then broadcasts the op carrying the returned
    :attr:`OpResult.seq`. channels never appends to any op-log itself.
    """

    async def __call__(self, room_id: str, user_id: str, frame: Frame) -> OpResult:
        """append the op carried by ``frame`` and return the op-log result.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :param user_id: authenticated principal authoring the op
        :ptype user_id: str
        :param frame: the inbound ``editor.op`` frame
        :ptype frame: Frame
        :return: the op-log result carrying the authoritative seq
        :rtype: OpResult
        """
        ...


@runtime_checkable
class ReplaySource(Protocol):
    """injected seam: stream the durable op-log tail from a seq (T3-D4).

    scriob injects this (``OpLog.replay(from_seq)``). on reconnect the
    handler streams this source's frames to the socket before going live,
    so a reconnect to any pod loses nothing.
    """

    def __call__(self, room_id: str, from_seq: int) -> AsyncIterator[str]:
        """yield the durable frames after ``from_seq`` for ``room_id``, in order.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :param from_seq: the last seq the client already has; replay
            everything after it
        :ptype from_seq: int
        :return: async iterator of wire-ready frame payloads, in seq order
        :rtype: AsyncIterator[str]
        """
        ...
