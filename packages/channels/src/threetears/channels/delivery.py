"""durable channel-delivery wire model.

the agent publishes a finished answer to the durable JetStream delivery subject
(:func:`threetears.nats.Subjects.channels_deliver`) with the channel routing
lifted off the inbound message; the channel adapter durable-consumes it and
posts it to the destination thread. this is the typed wire envelope for that
hand-off -- a Pydantic model because it crosses the NATS boundary (the channel
``protocol`` dataclasses are internal working objects, never serialized).
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, Field

__all__ = ["ChannelDeliveryMessage"]


class ChannelDeliveryMessage(BaseModel):
    """a finished agent answer awaiting durable delivery to a channel thread.

    carries the answer content plus exactly the routing the adapter needs to
    post out-of-band: the channel family, the workspace / channel / thread refs
    (lifted off the inbound message's metadata), and the dedup coordinates.

    :param correlation_id: per-turn correlation id of the originating request;
        with ``conversation_id`` forms the dedup key so a JetStream redelivery
        (or a second adapter replica) never double-posts
    :ptype correlation_id: UUID
    :param conversation_id: derived conversation id the answer belongs to
    :ptype conversation_id: UUID
    :param agent_id: agent that produced the answer
    :ptype agent_id: UUID
    :param channel_type: channel family (``slack``, ``discord``, ...)
    :ptype channel_type: str
    :param workspace_ref: platform workspace / team reference (slack ``team``);
        the adapter maps this to the right bot token
    :ptype workspace_ref: str | None
    :param channel_ref: platform channel reference to post into
    :ptype channel_ref: str | None
    :param thread_ref: platform thread reference to reply in (slack ``thread_ts``)
    :ptype thread_ref: str | None
    :param content: answer text in markdown
    :ptype content: str
    :param date_created: answer-ready timestamp (timezone-aware UTC)
    :ptype date_created: datetime
    :param turn_id: the answered turn's CSPRNG id (signed in its turn assertion); keys the dedup
        key, so a forger cannot pre-claim it. ``None`` from an agent that predates it
    :ptype turn_id: str | None
    :param turn_assertion: the hub's turn assertion the answered turn carried
        (:mod:`threetears.core.security.turn_assertion`): the delivery worker posts only an answer to
        a turn the hub signed, for this agent and conversation. ``None`` from an agent that predates it
    :ptype turn_assertion: str | None
    """

    correlation_id: UUID
    conversation_id: UUID
    agent_id: UUID
    channel_type: str
    workspace_ref: str | None = None
    channel_ref: str | None = None
    thread_ref: str | None = None
    content: str
    date_created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # 32 hex characters (``new_turn_id``), so it is always a valid KV key segment
    turn_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    turn_assertion: str | None = None

    @property
    def dedup_key(self) -> str:
        """return deterministic dedup key for at-least-once delivery.

        keyed on ``(conversation, correlation)`` so the same answer redelivered
        by JetStream -- or drained by a second adapter replica -- is posted at
        most once. the separator is ``-`` (not ``:``) because this string is a
        NATS KV key, and KV keys admit only ``[-/_=.a-zA-Z0-9]`` -- a colon
        raises ``InvalidKeyError``. both halves are 32-char hex, so ``-`` is
        unambiguous.

        keyed on the turn's CSPRNG id when the answer carries one: a correlation id is known to
        every tool pod in the turn, so a forged answer sent first under it would claim the key and
        suppress the real one; a turn id is not.

        :return: dedup key ``{conversation_hex}-{turn_id}``, or ``{conversation_hex}-{correlation_hex}``
            for an answer from an agent that predates the turn id
        :rtype: str
        """
        return f"{self.conversation_id.hex}-{self.turn_id or self.correlation_id.hex}"
